#include "scheduler.hpp"
#include "demangle.hpp"
#include "primitive/performance_probe.hpp"
#include "primitive/spin.hpp"
#include "primitive/thread_affinity.hpp"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace talos::scheduler {

// ============================================================================
// Internal Helpers
// ============================================================================

/// Helper: Print wake chains for a given set of systems
/// @param title Section title to print
/// @param source_systems Vector of source systems to iterate
/// @param affects_mask Vector of bitmasks indicating which target systems are affected
/// @param target_systems Vector of target systems for name lookup
template <typename SourceSystemEntry>
static void print_wake_chains(
    const char* title, const std::vector<SourceSystemEntry>& source_systems,
    const std::vector<std::uint64_t>& affects_mask,
    const std::vector<SystemEntry>& target_systems) {
    printf("%s\n", title);
    for (std::size_t i = 0; i < source_systems.size(); ++i) {
        const auto& src_name = source_systems[i].system->meta().name;
        const auto affects   = i < affects_mask.size() ? affects_mask[i] : 0U;

        printf("  [%s] -> ", src_name.c_str());
        if (affects == 0) {
            printf("(none)");
        } else {
            bool first = true;
            for (std::size_t j = 0; j < target_systems.size(); ++j) {
                if (affects & (1UL << j)) {
                    if (!first)
                        printf(", ");
                    printf("[%s]", target_systems[j].system->meta().name.c_str());
                    first = false;
                }
            }
        }
        printf("\n");
    }
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

Scheduler::Scheduler(World& world, const SchedulerConfig config)
    : world_(world)
    , config_(config)
    , stats_start_time_(Clock::now()) {}

Scheduler::~Scheduler() {
    stop();

    for (auto& t : fixed_rate_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

auto Scheduler::add_system(std::unique_ptr<SystemBase> system)
    -> std::expected<uint64_t, SchedulerError> {
    if (!system) {
        std::abort();
    }
    ensure_not_running();

    if (built_.load(std::memory_order_acquire)) {
        built_.store(false, std::memory_order_release);
    }

    const auto policy = system->meta().policy;
    SystemEntry entry{
        .system = std::move(system),
        .policy = policy,
    };

    if (policy.is_fixed_rate()) {
        fixed_rate_systems_.push_back(std::move(entry));
        return static_cast<uint64_t>(fixed_rate_systems_.size() - 1);
    }
    if (policy.is_pool() || policy.is_visualization()) {
        compute_systems_.push_back(std::move(entry));
        return static_cast<uint64_t>(compute_systems_.size() - 1);
    }
    std::unreachable();
}

auto Scheduler::build() -> BuildResult {
    // Compute execution levels for pool_compute systems
    // This re-validates all channels and rebuilds wake-up chains on each call
    if (auto result = compute_levels(); !result) {
        return result;
    }

    // Create TBB arena only on first call
    if (!compute_arena_) {
        if (config_.compute_concurrency > 0) {
            compute_arena_ =
                std::make_unique<tbb::task_arena>(static_cast<int>(config_.compute_concurrency));
        } else {
            compute_arena_ = std::make_unique<tbb::task_arena>();
        }
    }

    // Ensure stats capacity matches system count
    fixed_rate_stats_.resize(fixed_rate_systems_.size());
    compute_stats_.resize(compute_systems_.size());

    // Bind all systems (pre-create channel readers/writers, single-threaded)
    auto bind_all_systems = [&](auto& systems) {
        for (auto& entry : systems) {
            entry.system->bind(world_);
        }
    };
    bind_all_systems(fixed_rate_systems_);
    bind_all_systems(compute_systems_);

    built_.store(true, std::memory_order_release);
    return {};
}

auto Scheduler::run() -> std::expected<void, SchedulerError> {
    if (!built_.load(std::memory_order_acquire)) {
        return std::unexpected(SchedulerError::NotBuilt);
    }

    if (running_.exchange(true, std::memory_order_acq_rel)) {
        SPDLOG_CRITICAL("run(): scheduler is already running!");
        std::abort();
    }

    // Start fixed_rate threads
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        auto& entry = fixed_rate_systems_[i];
        FixedRateContext ctx{
            .system       = entry.system.get(),
            .policy       = entry.policy,
            .running      = &running_,
            .scheduler    = this,
            .world        = &world_,
            .system_index = i,
        };

        fixed_rate_threads_.emplace_back([ctx]() mutable { run_fixed_rate_thread(ctx); });
    }

    // Run compute loop on main thread (blocking)
    run_compute_loop();

    // Wait for fixed_rate threads
    for (auto& t : fixed_rate_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    fixed_rate_threads_.clear();

    return {};
}

void Scheduler::stop() noexcept {
    running_.store(false, std::memory_order_release);

    // Wake up compute loop if it's paused
    {
        std::lock_guard lock(pause_mutex_);
        pause_requested_.store(false, std::memory_order_release);
    }
    pause_cv_.notify_all();
}

auto Scheduler::hot_add_system(std::unique_ptr<SystemBase> system) -> BuildResult {
    if (!system) {
        std::abort();
    }

    // If not running, just do a regular add + build
    if (!running_.load(std::memory_order_acquire)) {
        const auto policy = system->meta().policy;
        SystemEntry entry{
            .system = std::move(system),
            .policy = policy,
        };

        if (policy.is_fixed_rate()) {
            fixed_rate_systems_.push_back(std::move(entry));
        } else if (policy.is_pool() || policy.is_visualization()) {
            compute_systems_.push_back(std::move(entry));
        } else {
            std::unreachable();
        }

        return build();
    }

    // Hot-add: request pause
    request_compute_loop_pause();

    // Now safely add the system
    const auto policy = system->meta().policy;
    SystemEntry entry{
        .system = std::move(system),
        .policy = policy,
    };

    const bool is_fixed_rate = policy.is_fixed_rate();
    if (is_fixed_rate) {
        fixed_rate_systems_.push_back(std::move(entry));
    } else if (policy.is_pool() || policy.is_visualization()) {
        compute_systems_.push_back(std::move(entry));
    } else {
        std::unreachable();
    }

    // Rebuild wake chains
    auto result = compute_levels();
    if (!result) {
        // Rollback: remove the just-added system
        if (is_fixed_rate) {
            fixed_rate_systems_.pop_back();
        } else {
            compute_systems_.pop_back();
        }

        // Resume compute loop
        resume_compute_loop();

        return result;
    }

    // Update stats capacity
    fixed_rate_stats_.resize(fixed_rate_systems_.size());
    compute_stats_.resize(compute_systems_.size());

    // Bind the new system
    if (is_fixed_rate) {
        auto& new_entry = fixed_rate_systems_.back();
        new_entry.system->bind(world_);

        // Launch new fixed_rate thread
        const auto idx = fixed_rate_systems_.size() - 1;
        FixedRateContext ctx{
            .system       = new_entry.system.get(),
            .policy       = new_entry.policy,
            .running      = &running_,
            .scheduler    = this,
            .world        = &world_,
            .system_index = idx,
        };
        fixed_rate_threads_.emplace_back([ctx]() mutable { run_fixed_rate_thread(ctx); });
    } else {
        compute_systems_.back().system->bind(world_);
    }

    // Resume compute loop
    resume_compute_loop();

    return {};
}

auto Scheduler::is_running() const noexcept -> bool {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// Pause/Resume Helpers (for hot-add)
// ============================================================================

void Scheduler::request_compute_loop_pause() noexcept {
    // Request pause
    {
        std::lock_guard lock(pause_mutex_);
        pause_requested_.store(true, std::memory_order_release);
    }
    // Wait for compute loop to pause
    {
        std::unique_lock lock(pause_mutex_);
        pause_cv_.wait(lock, [this] { return paused_.load(std::memory_order_acquire); });
    }
}

void Scheduler::resume_compute_loop() noexcept {
    {
        std::lock_guard lock(pause_mutex_);
        pause_requested_.store(false, std::memory_order_release);
    }
    pause_cv_.notify_all();
}

std::uint64_t Scheduler::sum_notify_counts() const noexcept {
    std::uint64_t total = 0;
    for (const auto& s : fixed_rate_stats_) {
        total += s.notify_count.load(std::memory_order_relaxed);
    }
    return total;
}

// ============================================================================
// Notification
// ============================================================================

void Scheduler::notify(const std::size_t system_index) noexcept {
    // Selective wake: only mark affected compute systems as ready
    if (system_index < fixed_rate_affects_.size()) {
        ready_systems_.fetch_or(fixed_rate_affects_[system_index], std::memory_order_release);
    }
    if (system_index < fixed_rate_stats_.size()) {
        fixed_rate_stats_[system_index].notify_count.fetch_add(1, std::memory_order_relaxed);
    }
}

// ============================================================================
// Stats
// ============================================================================

auto Scheduler::stats() const noexcept -> Stats {
    return Stats{
        .notify_count        = sum_notify_counts(),
        .compute_cycle_count = compute_cycles_.load(std::memory_order_relaxed),
    };
}

void Scheduler::print_systems() const {
    // Unified printer for systems with metadata (eliminates duplicate loop structure)
    auto print_systems = [](const auto& systems, const char* type_name, auto print_fn) {
        printf("%s systems (%zu):\n", type_name, systems.size());
        for (const auto& entry : systems) {
            const auto& m = entry.system->meta();
            print_fn(entry, m);
        }
    };

    print_systems(fixed_rate_systems_, "FixedRate", [](const auto& entry, const auto& m) {
        printf(
            "  [%s] freq:%uHz affinity:%d priority:%d notify:%s spsc:%zu spmc:%zu\n",
            m.name.c_str(), entry.policy.frequency_hz, entry.policy.cpu_affinity,
            entry.policy.thread_priority, entry.policy.notifies ? "yes" : "no",
            m.spsc_channels.size(), m.spmc_channels.size());
    });

    print_systems(
        compute_systems_, "Compute", [](const auto& entry [[maybe_unused]], const auto& m) {
            printf(
                "  [%s] spsc:%zu spmc:%zu res:%zu res_mut:%zu\n", m.name.c_str(),
                m.spsc_channels.size(), m.spmc_channels.size(), m.reads.size(), m.writes.size());
        });

    printf("Execution levels: %zu\n", levels_.size());
    for (std::size_t i = 0; i < levels_.size(); ++i) {
        printf("  Level %zu: ", i);
        for (const auto idx : levels_[i]) {
            printf("[%s] ", compute_systems_[idx].system->meta().name.c_str());
        }
        printf("\n");
    }

    // Print wake chains using unified helper
    print_wake_chains(
        "Wake chains (fixed_rate -> compute):", fixed_rate_systems_, fixed_rate_affects_,
        compute_systems_);
    print_wake_chains(
        "Wake chains (compute -> compute):", compute_systems_, compute_affects_, compute_systems_);
}

void Scheduler::print_mermaid_wake_chains() const {
    printf("```mermaid\n");
    printf("flowchart LR\n\n");

    // Build in_edges and out_edges for each compute system
    std::vector<std::size_t> in_edges(compute_systems_.size(), 0);
    std::vector<std::size_t> out_edges(compute_systems_.size(), 0);

    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        out_edges[i] = (i < compute_affects_.size()) ? std::popcount(compute_affects_[i]) : 0;
    }

    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if (i != j && (compute_affects_[i] & (1UL << j))) {
                in_edges[j]++;
            }
        }
    }

    // Also count edges from fixed_rate systems
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        auto affects = i < fixed_rate_affects_.size() ? fixed_rate_affects_[i] : 0U;
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if (affects & (1UL << j)) {
                in_edges[j]++;
            }
        }
    }

    // Helper: classify system type based on graph edges
    auto classify_system = [&](std::size_t idx) -> const char* {
        const bool has_out = out_edges[idx] > 0;
        const bool has_in  = in_edges[idx] > 0;

        if (has_out && !has_in)
            return "input";    // (1, 0) 只出不进
        if (has_out && has_in)
            return "pipeline"; // (1, 1) 有进有出
        return "output";       // (0, 1) 或 (0, 0) 只进不出或孤立
    };

    // Define subgraphs for system types
    printf("    subgraph External[FixedRate Systems]\n");
    printf("        direction TB\n");
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        const auto& name = fixed_rate_systems_[i].system->meta().name;
        const auto freq  = fixed_rate_systems_[i].policy.frequency_hz;
        printf("        E%d[\"%s<br/>%dHz\"]:::fixedStyle\n", (int)i, name.c_str(), freq);
    }
    printf("    end\n\n");

    // Group compute systems by type
    std::vector<std::size_t> input_systems, pipeline_systems, output_systems;
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        const auto* type = classify_system(i);
        if (strcmp(type, "input") == 0)
            input_systems.push_back(i);
        else if (strcmp(type, "output") == 0)
            output_systems.push_back(i);
        else
            pipeline_systems.push_back(i);
    }

    if (!input_systems.empty()) {
        printf("    subgraph Input[Input Systems]\n");
        printf("        direction TB\n");
        for (auto i : input_systems) {
            const auto& name = compute_systems_[i].system->meta().name;
            printf("        C%d[\"%s\"]:::inputStyle\n", (int)i, name.c_str());
        }
        printf("    end\n\n");
    }

    if (!pipeline_systems.empty()) {
        printf("    subgraph Pipeline[Pipeline Systems]\n");
        printf("        direction TB\n");
        for (auto i : pipeline_systems) {
            const auto& name = compute_systems_[i].system->meta().name;
            printf("        C%d[\"%s\"]:::pipelineStyle\n", (int)i, name.c_str());
        }
        printf("    end\n\n");
    }

    if (!output_systems.empty()) {
        printf("    subgraph Output[Output Systems]\n");
        printf("        direction TB\n");
        for (auto i : output_systems) {
            const auto& name = compute_systems_[i].system->meta().name;
            printf("        C%d[\"%s\"]:::outputStyle\n", (int)i, name.c_str());
        }
        printf("    end\n\n");
    }

    // Helper: collect channel topics that connect writer to reader
    auto get_channel_topics = [&](const SystemBase* writer,
                                  const SystemBase* reader) -> std::string {
        std::vector<std::string> topics;
        const auto& writer_meta = writer->meta();
        const auto& reader_meta = reader->meta();

        // Helper to check channel connections
        auto check_channels = [&](const auto& writer_channels, const auto& reader_channels) {
            for (const auto& wc : writer_channels) {
                if (wc.kind != channel_kind::spsc_writer && wc.kind != channel_kind::spmc_writer) {
                    continue;
                }
                for (const auto& rc : reader_channels) {
                    if (rc.kind != channel_kind::spsc_reader
                        && rc.kind != channel_kind::spmc_reader) {
                        continue;
                    }
                    if (wc.type == rc.type && wc.topic == rc.topic) {
                        // Demangle topic name for readability
                        topics.push_back(talos::scheduler::detail::demangle(wc.topic.name()));
                    }
                }
            }
        };

        check_channels(writer_meta.spsc_channels, reader_meta.spsc_channels);
        check_channels(writer_meta.spsc_channels, reader_meta.spmc_channels);
        check_channels(writer_meta.spmc_channels, reader_meta.spmc_channels);

        if (topics.empty()) {
            return "";
        }
        if (topics.size() == 1) {
            return topics[0];
        }
        // Multiple channels: join with commas
        std::string result;
        for (const auto& t : topics) {
            if (!result.empty())
                result += ", ";
            result += t;
        }
        return result;
    };

    // Helper: get resource topic between writer and reader
    auto get_resource_topic = [&](const SystemBase* writer,
                                  const SystemBase* reader) -> std::string {
        const auto& writer_meta = writer->meta();
        const auto& reader_meta = reader->meta();

        // Find common resource type (writer has res_mut, reader has res)
        for (const auto& writer_type : writer_meta.writes) {
            for (const auto& reader_type : reader_meta.reads) {
                if (writer_type == reader_type) {
                    return talos::scheduler::detail::demangle(writer_type.name());
                }
            }
        }
        return "";
    };

    // FixedRate → Compute
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        auto affects = i < fixed_rate_affects_.size() ? fixed_rate_affects_[i] : 0U;
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if (affects & (1UL << j)) {
                const auto* writer = fixed_rate_systems_[i].system.get();
                const auto* reader = compute_systems_[j].system.get();
                const auto topic   = get_channel_topics(writer, reader);
                if (topic.empty()) {
                    printf("    E%d --> C%d\n", (int)i, (int)j);
                } else {
                    printf("    E%d -->|\"%s\"| C%d\n", (int)i, topic.c_str(), (int)j);
                }
            }
        }
    }

    // Compute → Compute (channels + pipeline resources)
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        auto affects = i < compute_affects_.size() ? compute_affects_[i] : 0U;
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if ((affects & (1UL << j)) && i != j) {
                const auto* writer = compute_systems_[i].system.get();
                const auto* reader = compute_systems_[j].system.get();

                // Check for channel dependency
                const auto channel_topic = get_channel_topics(writer, reader);
                // Check for resource dependency
                const auto resource_topic = get_resource_topic(writer, reader);

                std::string label;
                if (!channel_topic.empty() && !resource_topic.empty()) {
                    label = channel_topic + " | res:" + resource_topic;
                } else if (!channel_topic.empty()) {
                    label = channel_topic;
                } else if (!resource_topic.empty()) {
                    label = "res:" + resource_topic;
                }

                if (label.empty()) {
                    printf("    C%d --> C%d\n", (int)i, (int)j);
                } else {
                    printf("    C%d -->|\"%s\"| C%d\n", (int)i, label.c_str(), (int)j);
                }
            }
        }
    }

    printf("```\n");
}

void Scheduler::print_mermaid_data_flow() const {
    printf("```mermaid\n");
    printf("flowchart LR\n\n");

    // Define distinct path colors for different data flows
    const char* path_colors[] = {
        "#e53935", // Red
        "#1e88e5", // Blue
        "#43a047", // Green
        "#fb8c00", // Orange
        "#8e24aa", // Purple
        "#00acc1", // Cyan
        "#fdd835", // Yellow
        "#6d4c41", // Brown
        "#546e7a", // Blue Grey
        "#d81b60", // Pink
    };
    constexpr std::size_t num_colors = sizeof(path_colors) / sizeof(path_colors[0]);

    // ========================================================================
    // Build data flow graphs: channels AND resources
    // ========================================================================

    struct ChannelEndpoint {
        std::size_t system_idx;
        bool is_fixed_rate;
    };

    // Channel connection: (type, topic) -> (writer, readers)
    struct ChannelConnection {
        std::optional<ChannelEndpoint> writer;
        std::vector<ChannelEndpoint> readers;
        std::type_index type;
        std::type_index topic;
        channel_kind kind;
    };

    // Resource connection: type -> (writers, readers)
    struct ResourceConnection {
        std::vector<ChannelEndpoint> writers; // res_mut<T>
        std::vector<ChannelEndpoint> readers; // res<T>
        std::type_index type;
    };

    std::unordered_map<ChannelKey, ChannelConnection, ChannelKeyHash> channel_graph;
    std::unordered_map<std::type_index, ResourceConnection> resource_graph;

    const std::size_t fixed_count   = fixed_rate_systems_.size();
    const std::size_t compute_count = compute_systems_.size();

    // Helper: process a system's channels
    auto process_system_channels = [&](std::size_t idx, bool is_fixed, const SystemMeta& meta) {
        ChannelEndpoint endpoint{idx, is_fixed};

        // Process SPSC channels
        for (const auto& ch : meta.spsc_channels) {
            ChannelKey key{ch.type, ch.topic};
            auto it = channel_graph.find(key);
            if (it == channel_graph.end()) {
                it = channel_graph
                         .emplace(
                             key,
                             ChannelConnection{
                                 .writer  = std::nullopt,
                                 .readers = {},
                                 .type    = ch.type,
                                 .topic   = ch.topic,
                                 .kind    = ch.kind})
                         .first;
            }

            if (ch.kind == channel_kind::spsc_writer) {
                it->second.writer = endpoint;
            } else if (ch.kind == channel_kind::spsc_reader) {
                it->second.readers.push_back(endpoint);
            }
        }

        // Process SPMC channels
        for (const auto& ch : meta.spmc_channels) {
            ChannelKey key{ch.type, ch.topic};
            auto it = channel_graph.find(key);
            if (it == channel_graph.end()) {
                it = channel_graph
                         .emplace(
                             key,
                             ChannelConnection{
                                 .writer  = std::nullopt,
                                 .readers = {},
                                 .type    = ch.type,
                                 .topic   = ch.topic,
                                 .kind    = ch.kind})
                         .first;
            }

            if (ch.kind == channel_kind::spmc_writer) {
                it->second.writer = endpoint;
            } else if (ch.kind == channel_kind::spmc_reader) {
                it->second.readers.push_back(endpoint);
            }
        }
    };

    // Helper: process a system's resources
    auto process_system_resources = [&](std::size_t idx, bool is_fixed, const SystemMeta& meta) {
        ChannelEndpoint endpoint{idx, is_fixed};

        // Process res_mut<T> (writers)
        for (const auto& res_type : meta.writes) {
            auto it = resource_graph.find(res_type);
            if (it == resource_graph.end()) {
                it = resource_graph
                         .emplace(
                             res_type,
                             ResourceConnection{.writers = {}, .readers = {}, .type = res_type})
                         .first;
            }
            it->second.writers.push_back(endpoint);
        }

        // Process res<T> (readers)
        for (const auto& res_type : meta.reads) {
            auto it = resource_graph.find(res_type);
            if (it == resource_graph.end()) {
                it = resource_graph
                         .emplace(
                             res_type,
                             ResourceConnection{.writers = {}, .readers = {}, .type = res_type})
                         .first;
            }
            it->second.readers.push_back(endpoint);
        }
    };

    // Build graphs from all systems
    for (std::size_t i = 0; i < fixed_count; ++i) {
        const auto& meta = fixed_rate_systems_[i].system->meta();
        process_system_channels(i, true, meta);
        process_system_resources(i, true, meta);
    }
    for (std::size_t i = 0; i < compute_count; ++i) {
        const auto& meta = compute_systems_[i].system->meta();
        process_system_channels(i, false, meta);
        process_system_resources(i, false, meta);
    }

    // Helper: classify system by channels AND resources
    auto classify_by_channels = [&](std::size_t idx, bool is_fixed) -> const char* {
        const auto& meta = is_fixed ? fixed_rate_systems_[idx].system->meta()
                                    : compute_systems_[idx].system->meta();

        // Check for writer channels (pub)
        const bool has_pub =
            std::any_of(
                meta.spsc_channels.begin(), meta.spsc_channels.end(),
                [](const auto& c) { return c.kind == channel_kind::spsc_writer; })
            || std::any_of(meta.spmc_channels.begin(), meta.spmc_channels.end(), [](const auto& c) {
                   return c.kind == channel_kind::spmc_writer;
               });

        // Check for reader channels (sub)
        const bool has_sub =
            std::any_of(
                meta.spsc_channels.begin(), meta.spsc_channels.end(),
                [](const auto& c) { return c.kind == channel_kind::spsc_reader; })
            || std::any_of(meta.spmc_channels.begin(), meta.spmc_channels.end(), [](const auto& c) {
                   return c.kind == channel_kind::spmc_reader;
               });

        // Check for resources (res/res_mut)
        const bool has_res_mut =
            !meta.writes.empty();                 // res_mut<T> - writes resource (input source)
        const bool has_res = !meta.reads.empty(); // res<T> - reads resource (output sink)

        // Classification considering BOTH channels and resources
        if (has_pub && has_sub)
            return "pipeline"; // Pipeline: has both pub and sub channels

        // Input: has res_mut (writes to resource) OR only publishes (no sub)
        if (has_res_mut || (has_pub && !has_sub))
            return "input";

        // Output: has res (reads from resource, no writes) OR only subscribes (no pub)
        if ((has_res && !has_res_mut) || (!has_pub && has_sub))
            return "output";

        return "orphan"; // No channels, no resources
    };

    // Group systems by channel-based classification
    std::vector<std::pair<std::size_t, bool>> input_systems, pipeline_systems, output_systems,
        orphan_systems;

    for (std::size_t i = 0; i < fixed_count; ++i) {
        const auto* type = classify_by_channels(i, true);
        if (strcmp(type, "input") == 0)
            input_systems.push_back({i, true});
        else if (strcmp(type, "pipeline") == 0)
            pipeline_systems.push_back({i, true});
        else if (strcmp(type, "output") == 0)
            output_systems.push_back({i, true});
        else
            orphan_systems.push_back({i, true});
    }

    for (std::size_t i = 0; i < compute_count; ++i) {
        const auto* type = classify_by_channels(i, false);
        if (strcmp(type, "input") == 0)
            input_systems.push_back({i, false});
        else if (strcmp(type, "pipeline") == 0)
            pipeline_systems.push_back({i, false});
        else if (strcmp(type, "output") == 0)
            output_systems.push_back({i, false});
        else
            orphan_systems.push_back({i, false});
    }

    // Helper: print system node
    auto print_system_node = [&](std::size_t idx, bool is_fixed, const char* style) {
        const auto& meta   = is_fixed ? fixed_rate_systems_[idx].system->meta()
                                      : compute_systems_[idx].system->meta();
        const char* prefix = is_fixed ? "E" : "C";
        if (is_fixed) {
            const auto freq = fixed_rate_systems_[idx].policy.frequency_hz;
            printf(
                "        %s%d[\"%s<br/>%dHz\"]:::%s\n", prefix, (int)idx, meta.name.c_str(), freq,
                style);
        } else {
            printf("        %s%d[\"%s\"]:::%s\n", prefix, (int)idx, meta.name.c_str(), style);
        }
    };

    // Print Input systems
    if (!input_systems.empty()) {
        printf("    subgraph Input[Input Systems]\n");
        for (auto [idx, is_fixed] : input_systems) {
            print_system_node(idx, is_fixed, "inputStyle");
        }
        printf("    end\n\n");
    }

    // Print Pipeline systems
    if (!pipeline_systems.empty()) {
        printf("    subgraph Pipeline[Pipeline Systems]\n");
        for (auto [idx, is_fixed] : pipeline_systems) {
            print_system_node(idx, is_fixed, "pipelineStyle");
        }
        printf("    end\n\n");
    }

    // Print Output systems
    if (!output_systems.empty()) {
        printf("    subgraph Output[Output Systems]\n");
        for (auto [idx, is_fixed] : output_systems) {
            print_system_node(idx, is_fixed, "outputStyle");
        }
        printf("    end\n\n");
    }

    // Print Orphan systems
    if (!orphan_systems.empty()) {
        printf("    subgraph Orphan[Orphan Systems]\n");
        for (auto [idx, is_fixed] : orphan_systems) {
            print_system_node(idx, is_fixed, "orphanStyle");
        }
        printf("    end\n\n");
    }

    // ========================================================================
    // Draw edges: Channel connections AND Resource dependencies
    // ========================================================================

    // Map to store color index for each channel (same channel = same color)
    std::unordered_map<std::string, std::size_t> channel_color_map;
    std::size_t next_color_idx = 0;

    // Draw channel edges (pub/sub) - collect edges first
    std::vector<std::tuple<std::string, std::string, std::string>> channel_edges;
    std::vector<std::string> edge_colors;

    for (const auto& [key, conn] : channel_graph) {
        if (!conn.writer.has_value() || conn.readers.empty())
            continue;

        const char* writer_prefix = conn.writer->is_fixed_rate ? "E" : "C";
        const auto& topic_name    = talos::scheduler::detail::demangle(conn.topic.name());

        // Assign color to this channel (same topic = same color)
        if (!channel_color_map.contains(topic_name)) {
            channel_color_map[topic_name] = next_color_idx % num_colors;
            next_color_idx++;
        }
        const std::size_t color_idx = channel_color_map[topic_name];

        for (const auto& reader : conn.readers) {
            const char* reader_prefix = reader.is_fixed_rate ? "E" : "C";
            char edge_buf[256];
            snprintf(
                edge_buf, sizeof(edge_buf), "    %s%d -->|\"%s\"| %s%d", writer_prefix,
                (int)conn.writer->system_idx, topic_name.c_str(), reader_prefix,
                (int)reader.system_idx);
            channel_edges.emplace_back(edge_buf, std::string(path_colors[color_idx]), topic_name);
        }
    }

    // Print all channel edges
    for (const auto& [edge_def, color, _] : channel_edges) {
        printf("%s\n", edge_def.c_str());
    }

    // Print linkStyle for channel edges (apply colors)
    for (std::size_t i = 0; i < channel_edges.size(); ++i) {
        const auto& color = std::get<1>(channel_edges[i]);
        printf("    linkStyle %zu stroke:%s,stroke-width:2px\n", i, color.c_str());
    }

    // Draw resource edges (res_mut → res) - use grey for all resource edges
    const std::size_t resource_edge_start = channel_edges.size();
    std::size_t resource_edge_count       = 0;

    for (const auto& [type, conn] : resource_graph) {
        if (conn.writers.empty() || conn.readers.empty())
            continue;

        const auto type_name = talos::scheduler::detail::demangle(type.name());

        for (const auto& writer : conn.writers) {
            const char* writer_prefix = writer.is_fixed_rate ? "E" : "C";
            for (const auto& reader : conn.readers) {
                const char* reader_prefix = reader.is_fixed_rate ? "E" : "C";
                printf(
                    "    %s%d -.->|\"res:%s\"| %s%d\n", writer_prefix, (int)writer.system_idx,
                    type_name.c_str(), reader_prefix, (int)reader.system_idx);
                resource_edge_count++;
            }
        }
    }

    // Print linkStyle for resource edges (all grey, dashed)
    for (std::size_t i = 0; i < resource_edge_count; ++i) {
        printf(
            "    linkStyle %zu stroke:#9e9e9e,stroke-width:2px,stroke-dasharray: 5 5\n",
            resource_edge_start + i);
    }

    printf("```\n");
}

void Scheduler::print_mermaid_execution_levels() const {
    printf("```mermaid\n");
    printf("flowchart LR\n\n"); // Changed from TB to LR for horizontal layout

    // Print FixedRate systems (source wakers)
    if (!fixed_rate_systems_.empty()) {
        printf("    subgraph Wakers[FixedRate Wakers]\n");
        printf("        direction TB\n"); // Vertical within subgraph for horizontal main flow
        for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
            const auto& name = fixed_rate_systems_[i].system->meta().name;
            const auto freq  = fixed_rate_systems_[i].policy.frequency_hz;
            printf(
                "        E%d[\"E%d: %s<br/>%dHz\"]:::fixedStyle\n", (int)i, (int)i, name.c_str(),
                freq);
        }
        printf("    end\n\n");
    }

    if (levels_.empty()) {
        printf("    NoLevels[\"No execution levels computed\"]\n");
        printf("```\n");
        return;
    }

    // Build in_edges and out_edges for each compute system
    std::vector<std::size_t> in_edges(compute_systems_.size(), 0);
    std::vector<std::size_t> out_edges(compute_systems_.size(), 0);

    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        out_edges[i] = (i < compute_affects_.size()) ? std::popcount(compute_affects_[i]) : 0;
    }

    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if (i != j && (compute_affects_[i] & (1UL << j))) {
                in_edges[j]++;
            }
        }
    }

    // Also count edges from fixed_rate systems
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        auto affects = i < fixed_rate_affects_.size() ? fixed_rate_affects_[i] : 0U;
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if (affects & (1UL << j)) {
                in_edges[j]++;
            }
        }
    }

    // Helper: classify system type based on graph edges
    auto classify_by_edges = [&](std::size_t idx) -> const char* {
        const bool has_out = out_edges[idx] > 0;
        const bool has_in  = in_edges[idx] > 0;

        if (has_out && !has_in)
            return "input";    // (1, 0) 只出不进
        if (has_out && has_in)
            return "pipeline"; // (1, 1) 有进有出
        return "output";       // (0, 1) 或 (0, 0) 只进不出或孤立
    };

    // Print each level as a subgraph
    for (std::size_t level_idx = 0; level_idx < levels_.size(); ++level_idx) {
        const auto& level = levels_[level_idx];
        printf(
            "    subgraph L%d[\"Level %d: %zu systems\"]\n", (int)level_idx, (int)level_idx,
            level.size());
        printf("        direction TB\n"); // Vertical within subgraph for horizontal main flow

        for (std::size_t i = 0; i < level.size(); ++i) {
            const auto sys_idx = level[i];
            const auto& name   = compute_systems_[sys_idx].system->meta().name;
            const auto& meta   = compute_systems_[sys_idx].system->meta();
            const auto readers = meta.spsc_channels.size() + meta.spmc_channels.size();
            const auto writers = std::count_if(
                meta.spsc_channels.begin(), meta.spsc_channels.end(), [](const auto& c) {
                    return c.kind == channel_kind::spsc_writer
                        || c.kind == channel_kind::spmc_writer;
                });

            const auto* sys_type    = classify_by_edges(sys_idx);
            const char* style_class = (strcmp(sys_type, "input") == 0)  ? "inputStyle"
                                    : (strcmp(sys_type, "output") == 0) ? "outputStyle"
                                                                        : "pipelineStyle";

            printf(
                "        C%d[\"C%d: %s\\n(%d in, %d out)\"]:::%s\n", (int)sys_idx, (int)sys_idx,
                name.c_str(), (int)readers, (int)writers, style_class);
        }

        printf("    end\n\n");
    }

    // ========================================================================
    // Draw edges: FixedRate -> compute systems, then between consecutive levels
    // ========================================================================

    // FixedRate wakers -> compute systems (direct wake targets)
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        auto affects = i < fixed_rate_affects_.size() ? fixed_rate_affects_[i] : 0U;
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if (affects & (1UL << j)) {
                printf("    E%d --> C%d\n", (int)i, (int)j);
            }
        }
    }

    // Draw edges between consecutive levels
    for (std::size_t level_idx = 0; level_idx + 1 < levels_.size(); ++level_idx) {
        const auto& current_level = levels_[level_idx];
        const auto& next_level    = levels_[level_idx + 1];

        for (std::size_t i = 0; i < current_level.size(); ++i) {
            const auto src_idx = current_level[i];
            auto src_affects   = src_idx < compute_affects_.size() ? compute_affects_[src_idx] : 0U;

            for (std::size_t j = 0; j < next_level.size(); ++j) {
                const auto dst_idx = next_level[j];
                if (src_affects & (1UL << dst_idx)) {
                    printf("    C%d --> C%d\n", (int)src_idx, (int)dst_idx);
                }
            }
        }
    }

    printf("```\n");
    printf("\n%% Pipeline Analysis:\n");

    // Count system types
    std::size_t input_count = 0, pipeline_count = 0, output_count = 0;
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        const auto* sys_type = classify_by_edges(i);
        if (strcmp(sys_type, "input") == 0)
            input_count++;
        else if (strcmp(sys_type, "output") == 0)
            output_count++;
        else
            pipeline_count++;
    }

    printf("%% - FixedRate wakers: %zu\n", (std::size_t)fixed_rate_systems_.size());
    printf("%% - Total depth: %zu levels\n", (std::size_t)levels_.size());
    printf(
        "%% - Compute system types: %zu input, %zu pipeline, %zu output\n", input_count,
        pipeline_count, output_count);
    printf("%% - Max parallelism: ");
    std::size_t max_parallel = 0;
    std::size_t max_level    = 0;
    for (std::size_t i = 0; i < levels_.size(); ++i) {
        if (levels_[i].size() > max_parallel) {
            max_parallel = levels_[i].size();
            max_level    = i;
        }
    }
    printf("%zu systems at Level %zu\n", max_parallel, max_level);
}

void Scheduler::print_stats() const {
    const auto now            = Clock::now();
    const auto elapsed_sec    = std::chrono::duration<double>(now - stats_start_time_).count();
    const auto compute_cycles = compute_cycles_.load(std::memory_order_relaxed);
    const auto compute_total  = compute_total_time_ns_.load(std::memory_order_relaxed);
    const auto compute_last   = compute_last_time_ns_.load(std::memory_order_relaxed);
    const auto notify_total   = sum_notify_counts();

    printf("\n");
    printf("=== Scheduler Performance (%.1fs) ===\n", elapsed_sec);

    // Compute loop stats - more compact
    printf("Loop: %-4" PRIu64 " cycles (%.1f Hz)", compute_cycles, compute_cycles / elapsed_sec);
    if (compute_cycles > 0) {
        printf(
            " | avg: %.2f ms | last: %.2f ms", compute_total / 1e6 / compute_cycles,
            compute_last / 1e6);
    }
    printf(" | notifies: %-4" PRIu64 " (%.1f Hz)\n", notify_total, notify_total / elapsed_sec);

    // Helper lambda: print latency statistics
    auto print_latency_stats = [](const auto& latency) {
        if (latency.count > 0) {
            printf(
                "   min:%.2f p50:%.2f p95:%.2f max:%.2f ms", latency.min_ns / 1e6,
                latency.p50_ns / 1e6, latency.p95_ns / 1e6, latency.max_ns / 1e6);
            printf("\n");
        }
    };

    // Unified stats printer (eliminates duplicate loop structure)
    auto print_system_stats = [&](const auto& systems, const auto& stats, auto print_fn) {
        for (std::size_t i = 0; i < systems.size(); ++i) {
            const auto& entry     = systems[i];
            const auto& sys_stats = stats[i];
            print_fn(entry, sys_stats, elapsed_sec);
        }
    };

    // FixedRate systems - system name on one line, stats on the next
    print_system_stats(
        fixed_rate_systems_, fixed_rate_stats_,
        [&](const auto& entry, const auto& sys_stats, double) {
            const auto notify_cnt = sys_stats.notify_count.load(std::memory_order_relaxed);
            const auto latency    = sys_stats.latency_hist.compute();
            const auto actual_hz  = latency.count / elapsed_sec;

            printf(
                "  %-30s@%.1fHz(%uHz) runs:%zu", entry.system->meta().name.c_str(), actual_hz,
                entry.policy.frequency_hz, latency.count);
            if (notify_cnt > 0) {
                printf(" notif:%" PRIu64, notify_cnt);
            }
            printf("\n");

            print_latency_stats(latency);
        });

    // Compute systems - system name on one line, stats on the next
    print_system_stats(
        compute_systems_, compute_stats_, [&](const auto& entry, const auto& sys_stats, double) {
            const auto latency   = sys_stats.latency_hist.compute();
            const auto actual_hz = latency.count / elapsed_sec;

            printf(
                "  %-30s@%.1fHz runs:%zu", entry.system->meta().name.c_str(), actual_hz,
                latency.count);
            printf("\n");

            print_latency_stats(latency);
        });
    printf("======================================\n");
    printf("\n");
}

auto Scheduler::get_stats_json() const -> std::string {
    using json = nlohmann::json;

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - stats_start_time_);
    const double elapsed_sec = elapsed.count();

    // Compute loop stats
    const auto cycles       = compute_cycles_.load(std::memory_order_relaxed);
    const auto total_time   = compute_total_time_ns_.load(std::memory_order_relaxed);
    const auto last_time    = compute_last_time_ns_.load(std::memory_order_relaxed);
    const auto notify_total = sum_notify_counts();

    json root;
    root["runtime_seconds"] = elapsed_sec;

    // Compute loop
    json& compute_json             = root["compute_loop"];
    compute_json["cycles"]         = cycles;
    compute_json["frequency_hz"]   = cycles / elapsed_sec;
    compute_json["avg_time_ms"]    = cycles > 0 ? (total_time / 1e6 / cycles) : 0.0;
    compute_json["last_time_ms"]   = last_time / 1e6;
    compute_json["total_notifies"] = notify_total;

    // Helper lambda: add latency statistics to JSON object
    auto add_latency_stats = [](json& sys, const auto& latency, double elapsed_sec) {
        sys["runs"]      = latency.count;
        sys["actual_hz"] = latency.count / elapsed_sec;
        sys["min_ms"]    = latency.min_ns / 1e6;
        sys["p50_ms"]    = latency.p50_ns / 1e6;
        sys["p95_ms"]    = latency.p95_ns / 1e6;
        sys["p99_ms"]    = latency.p99_ns / 1e6;
        sys["p999_ms"]   = latency.p999_ns / 1e6;
        sys["mean_ms"]   = latency.mean_ns / 1e6;
        sys["max_ms"]    = latency.max_ns / 1e6;
        sys["stddev_ms"] = latency.stddev_ns / 1e6;
    };

    // Unified JSON stats generator (eliminates duplicate loop structure)
    auto generate_system_stats = [&](const auto& systems, const auto& stats, json& arr,
                                     auto add_extra_fn) {
        for (std::size_t i = 0; i < systems.size(); ++i) {
            const auto& entry     = systems[i];
            const auto& sys_stats = stats[i];
            const auto latency    = sys_stats.latency_hist.compute();

            json sys;
            add_latency_stats(sys, latency, elapsed_sec);
            add_extra_fn(sys, entry, sys_stats);
            arr[entry.system->meta().name] = sys;
        }
    };

    // FixedRate systems
    json& fixed_rate_arr = root["fixed_rate_systems"] = json::object();
    generate_system_stats(
        fixed_rate_systems_, fixed_rate_stats_, fixed_rate_arr,
        [](json& sys, const auto& entry, const auto& stats) {
            const auto notify_cnt = stats.notify_count.load(std::memory_order_relaxed);
            sys["target_hz"]      = entry.policy.frequency_hz;
            sys["notifies"]       = notify_cnt;
        });

    // Compute systems
    json& compute_arr = root["compute_systems"] = json::object();
    generate_system_stats(
        compute_systems_, compute_stats_, compute_arr, [](json&, const auto&, const auto&) {
            // Compute systems have no extra fields
        });

    return root.dump();
}

// ============================================================================
// FixedRate Thread
// ============================================================================

void Scheduler::run_fixed_rate_thread(FixedRateContext ctx) {
    // Set CPU affinity
    if (ctx.policy.cpu_affinity >= 0) {
        auto result = primitive::ThreadAffinity::pin_to_core(
            static_cast<std::uint32_t>(ctx.policy.cpu_affinity));
        if (!result) {
            // Log warning but continue
            SPDLOG_WARN(
                "set CPU affinity for {}: {}", ctx.system->meta().name,
                primitive::error_message(result.error()));
        }
    }

    // Set thread priority
    if (ctx.policy.thread_priority > 0) {
        primitive::ThreadAffinity::RealtimeConfig rt_config{};
        rt_config.priority = static_cast<std::uint8_t>(ctx.policy.thread_priority);
        if (auto result = primitive::ThreadAffinity::set_realtime_priority(rt_config); !result) {
            SPDLOG_WARN(
                "set thread priority for {}: {}", ctx.system->meta().name,
                primitive::error_message(result.error()));
        }
    }

    const auto frequency = ctx.policy.frequency_hz;
    auto& stats          = ctx.scheduler->fixed_rate_stats_[ctx.system_index];

    if (frequency == 0) {
        // Unlimited frequency mode: run as fast as possible
        const auto writer = ctx.system->meta().writes.size() > 0;
        while (ctx.running->load(std::memory_order_acquire)) {
            primitive::LatencyProbe probe;
            const bool written = ctx.system->run(*ctx.world);
            const auto elapsed = probe.elapsed_ns();

            if (!writer || (writer && written)) [[likely]] {
                stats.latency_hist.record(elapsed);
                if (written && ctx.policy.notifies) [[likely]] {
                    ctx.scheduler->notify(ctx.system_index);
                }
            }
        }
        return;
    }

    // Fixed frequency mode
    // Assume frequency > 0 to help optimizer eliminate division-by-zero checks
    [[assume(frequency > 0)]];
    const auto period =
        std::chrono::nanoseconds(1'000'000'000ULL / static_cast<std::uint64_t>(frequency));
    auto next_tick = Clock::now();

    const auto writer = ctx.system->meta().writes.size() > 0;
    while (ctx.running->load(std::memory_order_acquire)) [[likely]] {
        next_tick += period;

        // Execute system
        primitive::LatencyProbe probe;
        const bool written = ctx.system->run(*ctx.world);
        const auto elapsed = probe.elapsed_ns();

        if (!writer || (writer && written)) [[likely]] {
            stats.latency_hist.record(elapsed);
            if (written && ctx.policy.notifies) [[likely]] {
                ctx.scheduler->notify(ctx.system_index);
            }
        }

        // Busy wait until next tick
        std::this_thread::sleep_until(next_tick);
    }
}

// ============================================================================
// Compute Loop
// ============================================================================

void Scheduler::run_compute_loop() {
    // Adaptive backoff: spin -> yield -> short sleep
    // This reduces CPU usage during idle while maintaining low latency
    constexpr std::size_t SPIN_LIMIT  = 100;  // Spin iterations before yield
    constexpr std::size_t YIELD_LIMIT = 1000; // Yield iterations before sleep
    constexpr auto SLEEP_DURATION     = std::chrono::microseconds(10); // Short sleep
    std::size_t idle_count            = 0;
    const auto print_interval         = std::chrono::seconds(5);
    auto last_print                   = Clock::now();

    while (running_.load(std::memory_order_acquire)) [[likely]] {
        // Check for pause request (hot-add in progress)
        if (pause_requested_.load(std::memory_order_acquire)) [[unlikely]] {
            // Reset idle counter on resume
            idle_count = 0;

            // Signal that we've paused
            {
                std::lock_guard lock(pause_mutex_);
                paused_.store(true, std::memory_order_release);
            }
            pause_cv_.notify_all();

            // Wait for resume signal
            {
                std::unique_lock lock(pause_mutex_);
                pause_cv_.wait(lock, [this] {
                    return !pause_requested_.load(std::memory_order_acquire)
                        || !running_.load(std::memory_order_acquire);
                });
                paused_.store(false, std::memory_order_release);
            }

            // Check if we should exit after resume
            if (!running_.load(std::memory_order_acquire)) {
                break;
            }
            continue;
        }

        // Atomically exchange ready_systems with 0, getting the set of systems to run
        const auto ready = ready_systems_.exchange(0, std::memory_order_acq_rel);

        if (ready != 0) [[likely]] {
            // Reset idle counter when work is available
            idle_count = 0;

            primitive::LatencyProbe probe;
            const auto written = run_compute_selective(ready);
            const auto elapsed = probe.elapsed_ns();

            compute_cycles_.fetch_add(1, std::memory_order_relaxed);
            compute_total_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
            compute_last_time_ns_.store(elapsed, std::memory_order_relaxed);

            // Cascade wake: mark dependents of systems that actually wrote
            std::uint64_t cascade = 0;
            for (std::size_t i = 0; i < compute_affects_.size(); ++i) {
                if (written & (1UL << i)) {
                    cascade |= compute_affects_[i];
                }
            }
            if (cascade != 0) {
                ready_systems_.fetch_or(cascade, std::memory_order_release);
            }

            // Check stats after compute (amortize cost)
            // /*
            const auto now = Clock::now();
            if (now - last_print >= print_interval) [[unlikely]] {
                print_stats();
                last_print = now;
            }
            // */
        } else {
            // Adaptive backoff strategy:
            // - Spin first (lowest latency, ~10 cycles per SPIN_HINT)
            // - Then yield (medium latency, ~1-10μs)
            // - Finally short sleep (higher latency but saves CPU)
            ++idle_count;

            if (idle_count <= SPIN_LIMIT) {
                // Phase 1: Spin (lowest latency)
                SPIN_HINT();
            } else if (idle_count <= YIELD_LIMIT) {
                // Phase 2: Yield (let other threads run)
                std::this_thread::yield();
            } else {
                // Phase 3: Short sleep (save CPU during extended idle)
                std::this_thread::sleep_for(SLEEP_DURATION);
            }
        }
    }
}

auto Scheduler::run_compute_selective(const std::uint64_t ready_mask) -> std::uint64_t {
    if (compute_systems_.empty()) [[unlikely]] {
        return 0;
    }

    std::atomic<std::uint64_t> written_mask{0};

    compute_arena_->execute([this, ready_mask, &written_mask] {
        // Reusable buffer for system indices to avoid per-call allocation
        // Max 64 systems, so 64 elements is sufficient
        std::size_t to_run_buf[64];
        std::size_t to_run_count = 0;

        for (const auto& level : levels_) {
            // Filter systems in this level that are ready (inline, no temporary vector)
            to_run_count = 0;
            for (const auto idx : level) {
                if (ready_mask & (1UL << idx)) {
                    to_run_buf[to_run_count++] = idx;
                }
            }

            if (to_run_count == 0) [[unlikely]] {
                continue;
            }

            if (to_run_count == 1) {
                // Single system: execute directly, avoid task_group overhead
                const auto idx = to_run_buf[0];
                primitive::LatencyProbe probe;
                const bool written = compute_systems_[idx].system->run(world_);
                const auto elapsed = probe.elapsed_ns();
                const auto writer  = compute_systems_[idx].system->meta().writes.size() > 0;

                if (!writer || (writer && written)) [[likely]] {
                    compute_stats_[idx].latency_hist.record(elapsed);
                    if (written) [[likely]] {
                        written_mask.fetch_or(1UL << idx, std::memory_order_relaxed);
                    }
                }
            } else {
                // Multiple systems: execute in parallel
                tbb::task_group tg;
                for (std::size_t i = 0; i < to_run_count; ++i) {
                    const auto idx = to_run_buf[i];
                    tg.run([this, idx, &written_mask] {
                        primitive::LatencyProbe probe;
                        const bool written = compute_systems_[idx].system->run(world_);
                        const auto elapsed = probe.elapsed_ns();
                        const auto writer  = compute_systems_[idx].system->meta().writes.size() > 0;

                        if (!writer || (writer && written)) [[likely]] {
                            compute_stats_[idx].latency_hist.record(elapsed);
                            if (written) [[likely]] {
                                written_mask.fetch_or(1UL << idx, std::memory_order_relaxed);
                            }
                        }
                    });
                }
                tg.wait();
            }
        }
    });

    return written_mask.load(std::memory_order_relaxed);
}

// ============================================================================
// Dependency Graph
// ============================================================================

auto Scheduler::compute_levels() -> BuildResult {
    if (compute_systems_.empty()) {
        return {};
    }

    const auto n = compute_systems_.size();

    // Enforce max 64 compute systems for bitmask scheduling
    if (n > TooManyComputeSystemsError::max_count) {
        return std::unexpected(TooManyComputeSystemsError{.count = n});
    }

    // ========================================================================
    // Phase 1: Collect channel usage information
    // ========================================================================

    // Helper: check if channel kind is a writer
    const auto is_writer = [](channel_kind k) {
        return k == channel_kind::spsc_writer || k == channel_kind::spmc_writer;
    };

    // Helper: check if channel kind is SPSC
    const auto is_spsc = [](channel_kind k) {
        return k == channel_kind::spsc_reader || k == channel_kind::spsc_writer;
    };

    struct ChannelUsage {
        channel_kind kind = channel_kind::local; // Will be set on first use
        std::vector<std::size_t> writers;        // Compute system indices
        std::vector<std::size_t> readers;        // Compute system indices
        bool has_fixed_rate_writer = false;      // Written by fixed_rate system
        std::string first_system;                // For kind conflict reporting
    };

    std::map<ChannelKey, ChannelUsage> channels;

    // First, collect fixed_rate systems' outputs as external sources
    for (const auto& entry : fixed_rate_systems_) {
        const auto& meta = entry.system->meta();

        // Helper lambda to process writer channels (unified for spsc/spmc)
        auto collect_writer = [&](const ChannelMeta& ch) {
            if (is_writer(ch.kind)) {
                ChannelKey key{ch.type, ch.topic};
                auto& usage                 = channels[key];
                usage.kind                  = ch.kind;
                usage.has_fixed_rate_writer = true;
                usage.first_system          = meta.name;
            }
        };

        for (const auto& ch : meta.spsc_channels) {
            collect_writer(ch);
        }
        for (const auto& ch : meta.spmc_channels) {
            collect_writer(ch);
        }
    }

    // Then collect compute systems' channels
    for (std::size_t i = 0; i < n; ++i) {
        const auto& meta = compute_systems_[i].system->meta();

        // Process channels using lambda to avoid duplication
        auto process_channel = [&](const ChannelMeta& ch, bool expect_spc) -> BuildResult {
            ChannelKey key{ch.type, ch.topic};
            auto& usage = channels[key];

            // Check for kind conflict (same key used as both SPSC and SPMC)
            if (!usage.writers.empty() || !usage.readers.empty()) {
                const auto is_expected_kind =
                    expect_spc ? is_spsc(usage.kind) : !is_spsc(usage.kind);
                if (!is_expected_kind) {
                    return std::unexpected(
                        ChannelKindConflict{
                            .key           = ChannelKeyInfo(ch.type, ch.topic, expect_spc),
                            .first_system  = usage.first_system,
                            .second_system = meta.name,
                        });
                }
            } else {
                usage.kind         = ch.kind;
                usage.first_system = meta.name;
            }

            if (is_writer(ch.kind)) {
                usage.writers.push_back(i);
            } else {
                usage.readers.push_back(i);
            }
            return {};
        };

        for (const auto& ch : meta.spsc_channels) {
            if (auto result = process_channel(ch, true); !result)
                return result;
        }

        for (const auto& ch : meta.spmc_channels) {
            if (auto result = process_channel(ch, false); !result)
                return result;
        }
    }

    // ========================================================================
    // Phase 2: Validate channel constraints
    // ========================================================================

    // Helper: collect reader names from indices
    auto collect_reader_names = [&](const std::vector<std::size_t>& reader_indices) {
        std::vector<std::string> reader_names;
        reader_names.reserve(reader_indices.size());
        for (auto idx : reader_indices) {
            reader_names.push_back(compute_systems_[idx].system->meta().name);
        }
        return reader_names;
    };

    for (const auto& [key, usage] : channels) {
        // Count total writers (fixed_rate + compute)
        std::size_t total_writers = usage.writers.size() + (usage.has_fixed_rate_writer ? 1 : 0);

        // Multiple writers is always an error
        if (total_writers > 1) {
            std::vector<std::string> writer_names;
            writer_names.reserve(usage.writers.size() + (usage.has_fixed_rate_writer ? 1 : 0));
            if (usage.has_fixed_rate_writer) {
                writer_names.emplace_back(fmt::format("{} (fixed_rate)", usage.first_system));
            }
            for (auto idx : usage.writers) {
                writer_names.emplace_back(compute_systems_[idx].system->meta().name);
            }
            return std::unexpected(
                MultipleWritersError{
                    .key     = ChannelKeyInfo(key.type, key.topic, is_spsc(usage.kind)),
                    .writers = std::move(writer_names),
                });
        }

        // SPSC with multiple readers is an error
        if (is_spsc(usage.kind) && usage.readers.size() > 1) {
            return std::unexpected(
                MultipleReadersError{
                    .key     = ChannelKeyInfo(key.type, key.topic, true),
                    .readers = collect_reader_names(usage.readers),
                });
        }

        // Reader without writer is an error (neither fixed_rate nor compute writer)
        if (total_writers == 0 && !usage.readers.empty()) {
            return std::unexpected(
                OrphanedReaderError{
                    .key     = ChannelKeyInfo(key.type, key.topic, is_spsc(usage.kind)),
                    .readers = collect_reader_names(usage.readers),
                });
        }

        // Orphaned compute writer is a warning (printed but not an error)
        // Note: fixed_rate writers with no readers are fine (they notify the scheduler)
        if (!usage.writers.empty() && usage.readers.empty() && !usage.has_fixed_rate_writer) {
            SPDLOG_WARN(
                "[WARN] Channel has writer '{}' but no readers\n",
                compute_systems_[usage.writers[0]].system->meta().name.c_str());
        }
    }

    // ========================================================================
    // Phase 3: Build adjacency list and in-degree (channels only)
    // ========================================================================

    // Use bitmask for adjacency instead of std::set (O(1) vs O(log n))
    // adj_mask[i] = bitmask of nodes that node i points to
    std::vector<std::uint64_t> adj_mask(n, 0);
    std::vector<std::size_t> in_degree(n, 0);

    for (const auto& [key, usage] : channels) {
        if (usage.writers.empty() || usage.readers.empty()) {
            continue;
        }

        std::size_t writer = usage.writers[0];
        for (std::size_t reader : usage.readers) {
            if (writer != reader) {
                const std::uint64_t bit = 1UL << reader;
                if ((adj_mask[writer] & bit) == 0) {
                    // New edge
                    adj_mask[writer] |= bit;
                    ++in_degree[reader];
                }
            }
        }
    }

    // ========================================================================
    // Phase 3.5: Build resource dependencies
    // ========================================================================
    // Resource dependency semantics:
    // - Only readers → output systems (sink nodes, no downstream deps from this resource)
    // - Only writer → input system (source node, reads external data)
    // - Writer + readers → pipeline dependency (writer → readers)

    struct ResourceUsage {
        std::vector<std::size_t> readers; // res<T> readers
        std::vector<std::size_t> writers; // res_mut<T> writers
    };

    std::map<std::type_index, ResourceUsage> resources;

    // Collect resource usage from all compute systems
    for (std::size_t i = 0; i < n; ++i) {
        const auto& meta = compute_systems_[i].system->meta();

        for (const auto& type : meta.reads) {
            resources[type].readers.push_back(i);
        }
        for (const auto& type : meta.writes) {
            resources[type].writers.push_back(i);
        }
    }

    // Validate and build edges for resource dependencies
    for (const auto& [type, usage] : resources) {
        // Rule: Only one res_mut writer per resource type
        if (usage.writers.size() > 1) {
            std::vector<std::string> writer_names;
            writer_names.reserve(usage.writers.size());
            for (auto idx : usage.writers) {
                writer_names.push_back(compute_systems_[idx].system->meta().name);
            }
            std::string names_str;
            for (size_t i = 0; i < writer_names.size(); ++i) {
                if (i > 0)
                    names_str += ", ";
                names_str += writer_names[i];
            }
            SPDLOG_CRITICAL(
                "Resource type '{}' has multiple res_mut writers: {}",
                talos::scheduler::detail::demangle(type.name()), names_str);
            std::abort();
        }

        // Only readers: output systems (sink nodes, no dependency edges created)
        // Only writer: input system (source node, no dependency edges created)
        // Writer + readers: create dependency edges (writer → readers)
        if (!usage.writers.empty() && !usage.readers.empty()) {
            // Pipeline dependency: writer must execute before all readers
            const std::size_t writer = usage.writers[0];
            for (std::size_t reader : usage.readers) {
                if (writer != reader) {
                    const std::uint64_t bit = 1UL << reader;
                    if ((adj_mask[writer] & bit) == 0) {
                        adj_mask[writer] |= bit;
                        ++in_degree[reader];
                    }
                }
            }
        }
    }

    // ========================================================================
    // Phase 4: Kahn's algorithm for topological sort (by levels)
    // ========================================================================

    levels_.clear();
    std::vector<std::size_t> current_level;

    // Start with nodes that have no dependencies
    for (std::size_t i = 0; i < n; ++i) {
        if (in_degree[i] == 0) {
            current_level.push_back(i);
        }
    }

    std::size_t processed = 0;
    while (!current_level.empty()) {
        levels_.push_back(current_level);
        processed += current_level.size();

        std::vector<std::size_t> next_level;
        next_level.reserve(compute_systems_.size()); // Pre-allocate max possible
        for (std::size_t node : current_level) {
            std::uint64_t neighbors = adj_mask[node];
            while (neighbors) {
                const std::size_t neighbor = std::countr_zero(neighbors);
                neighbors &= neighbors - 1;          // Clear lowest set bit
                if (--in_degree[neighbor] == 0) {
                    next_level.push_back(neighbor);
                }
            }
        }
        current_level = std::move(next_level);
    }

    // ========================================================================
    // Phase 5: Cycle detection
    // ========================================================================

    if (processed < n) {
        // There's a cycle - find it using DFS
        std::vector<std::string> cycle;

        // Simple cycle detection: follow edges until we revisit a node
        std::vector visited(n, false);
        std::vector in_stack(n, false);
        std::vector<std::size_t> path;

        std::function<bool(std::size_t)> find_cycle = [&](const std::size_t node) -> bool {
            visited[node]  = true;
            in_stack[node] = true;
            path.push_back(node);

            std::uint64_t neighbors = adj_mask[node];
            while (neighbors) {
                const std::size_t neighbor = std::countr_zero(neighbors);
                neighbors &= neighbors - 1; // Clear lowest set bit

                if (!visited[neighbor]) {
                    if (find_cycle(neighbor)) {
                        return true;
                    }
                } else if (in_stack[neighbor]) {
                    // Found cycle - extract it
                    auto it = std::find(path.begin(), path.end(), neighbor);
                    for (; it != path.end(); ++it) {
                        cycle.push_back(compute_systems_[*it].system->meta().name);
                    }
                    cycle.push_back(compute_systems_[neighbor].system->meta().name);
                    return true;
                }
            }

            path.pop_back();
            in_stack[node] = false;
            return false;
        };

        // Start DFS from nodes with remaining in-degree
        for (std::size_t i = 0; i < n && cycle.empty(); ++i) {
            if (in_degree[i] > 0 && !visited[i]) {
                find_cycle(i);
            }
        }

        return std::unexpected(DependencyCycleError{.cycle = std::move(cycle)});
    }

    // ========================================================================
    // Phase 6: Build wake chains (waker -> direct wakee only)
    // ========================================================================
    // Wake chains connect systems that directly communicate via channels OR resources.
    // Transitive dependencies are handled by cascade wake in compute loop.

    // Wake chains for fixed_rate systems:
    // fixed_rate_affects_[i] = bitmask of compute systems that directly read this fixed_rate's
    // channels
    fixed_rate_affects_.resize(fixed_rate_systems_.size(), 0);

    // Helper lambda: process writer channels and collect direct readers only
    auto collect_direct_readers = [&](const auto& meta_channels, std::uint64_t& affects) {
        for (const auto& ch : meta_channels) {
            if (is_writer(ch.kind)) {
                ChannelKey key{ch.type, ch.topic};
                if (auto it = channels.find(key); it != channels.end()) {
                    for (auto reader : it->second.readers) {
                        affects |= (1UL << reader); // Direct reader only
                    }
                }
            }
        }
    };

    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        std::uint64_t affects = 0;
        const auto& meta      = fixed_rate_systems_[i].system->meta();

        // Collect direct readers of this fixed_rate's channels
        collect_direct_readers(meta.spsc_channels, affects);
        collect_direct_readers(meta.spmc_channels, affects);

        fixed_rate_affects_[i] = affects;
    }

    // Wake chains for compute systems:
    // compute_affects_[i] = bitmask of compute systems that directly depend on system i
    // via channels OR pipeline resources (writer + readers)
    // This enables cascade wake: when compute A finishes, wake compute B if B depends on A
    compute_affects_.resize(n, 0);

    // Channel dependencies
    for (const auto& [key, usage] : channels) {
        if (usage.writers.empty() || usage.readers.empty()) {
            continue;
        }

        const std::size_t writer = usage.writers[0];
        for (std::size_t reader : usage.readers) {
            if (writer != reader) {
                compute_affects_[writer] |= (1UL << reader);
            }
        }
    }

    // Resource dependencies: only include pipeline resources (writer + readers)
    // - Only readers: output systems (no wake edges)
    // - Only writer: input systems (no wake edges)
    // - Writer + readers: pipeline dependency (writer → readers)
    for (const auto& [type, usage] : resources) {
        if (!usage.writers.empty() && !usage.readers.empty()) {
            // Pipeline dependency: include in wake chains
            const std::size_t writer = usage.writers[0];
            for (std::size_t reader : usage.readers) {
                if (writer != reader) {
                    compute_affects_[writer] |= (1UL << reader);
                }
            }
        }
    }

    return {};
}

} // namespace talos::scheduler
