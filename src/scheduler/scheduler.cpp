#include "scheduler/scheduler/scheduler.hpp"
#include "performance_probe.hpp"
#include "scheduler/primitives/thread_affinity.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <unordered_map>

namespace talos {

// ============================================================================
// Constructor / Destructor
// ============================================================================

Scheduler::Scheduler(World& world, const SchedulerConfig config)
    : world_(world)
    , config_(config)
    , stats_start_time_(Clock::now()) {}

Scheduler::~Scheduler() {
    stop();

    for (auto& t : exclusive_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

auto Scheduler::build() -> BuildResult {
    if (built_.exchange(true, std::memory_order_acq_rel)) {
        return std::unexpected(SchedulerError::AlreadyBuilt);
    }

    // Compute execution levels for pool_compute systems
    if (auto result = compute_levels(); !result) {
        built_.store(false, std::memory_order_release);
        return result;
    }

    // Create TBB arena
    if (config_.compute_concurrency > 0) {
        compute_arena_ =
            std::make_unique<tbb::task_arena>(static_cast<int>(config_.compute_concurrency));
    } else {
        compute_arena_ = std::make_unique<tbb::task_arena>();
    }

    // Initialize stats for exclusive systems
    exclusive_stats_.reserve(exclusive_systems_.size());
    for (std::size_t i = 0; i < exclusive_systems_.size(); ++i) {
        exclusive_stats_.emplace_back();
    }

    // Initialize stats for compute systems
    compute_stats_.reserve(compute_systems_.size());
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        compute_stats_.emplace_back();
    }

    return {};
}

auto Scheduler::run() -> std::expected<void, SchedulerError> {
    if (!built_.load(std::memory_order_acquire)) {
        return std::unexpected(SchedulerError::NotBuilt);
    }

    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return std::unexpected(SchedulerError::AlreadyRunning);
    }

    // Start exclusive threads
    for (std::size_t i = 0; i < exclusive_systems_.size(); ++i) {
        auto& entry = exclusive_systems_[i];
        ExclusiveContext ctx{
            .system       = entry.system.get(),
            .policy       = entry.policy,
            .running      = &running_,
            .scheduler    = this,
            .world        = &world_,
            .system_index = i,
        };

        exclusive_threads_.emplace_back([ctx]() mutable { run_exclusive_thread(ctx); });
    }

    // Run compute loop on main thread (blocking)
    run_compute_loop();

    // Wait for exclusive threads
    for (auto& t : exclusive_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    exclusive_threads_.clear();

    return {};
}

void Scheduler::stop() noexcept { running_.store(false, std::memory_order_release); }

auto Scheduler::is_running() const noexcept -> bool {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// Notification
// ============================================================================

void Scheduler::notify(const std::size_t system_index) noexcept {
    work_version_.fetch_add(1, std::memory_order_release);
    if (system_index < exclusive_stats_.size()) {
        exclusive_stats_[system_index].notify_count.fetch_add(1, std::memory_order_relaxed);
    }
}

// ============================================================================
// Stats
// ============================================================================

auto Scheduler::stats() const noexcept -> Stats {
    return Stats{
        .notify_count        = work_version_.load(std::memory_order_relaxed),
        .compute_cycle_count = compute_cycles_.load(std::memory_order_relaxed),
    };
}

void Scheduler::print_systems() const {
    printf("Exclusive systems (%zu):\n", exclusive_systems_.size());
    for (const auto& entry : exclusive_systems_) {
        const auto& m = entry.system->meta();
        printf(
            "  [%s] freq:%uHz affinity:%d priority:%d notify:%s\n", m.name.c_str(),
            entry.policy.frequency_hz, entry.policy.cpu_affinity, entry.policy.thread_priority,
            entry.policy.notifies ? "yes" : "no");
    }

    printf("Compute systems (%zu):\n", compute_systems_.size());
    for (const auto& entry : compute_systems_) {
        const auto& m = entry.system->meta();
        printf(
            "  [%s] spsc:%zu spmc:%zu res:%zu res_mut:%zu\n", m.name.c_str(),
            m.spsc_channels.size(), m.spmc_channels.size(), m.reads.size(), m.writes.size());
    }

    printf("Execution levels: %zu\n", levels_.size());
    for (std::size_t i = 0; i < levels_.size(); ++i) {
        printf("  Level %zu: ", i);
        for (const auto idx : levels_[i]) {
            printf("[%s] ", compute_systems_[idx].system->meta().name.c_str());
        }
        printf("\n");
    }
}

void Scheduler::print_stats() const {
    const auto now            = Clock::now();
    const auto elapsed_sec    = std::chrono::duration<double>(now - stats_start_time_).count();
    const auto compute_cycles = compute_cycles_.load(std::memory_order_relaxed);
    const auto compute_total  = compute_total_time_ns_.load(std::memory_order_relaxed);
    const auto compute_last   = compute_last_time_ns_.load(std::memory_order_relaxed);
    const auto notify_total   = work_version_.load(std::memory_order_relaxed);

    printf("\n");
    printf("========== Scheduler Performance Stats ==========\n");
    printf("Runtime: %.1fs\n", elapsed_sec);
    printf("\n");

    // Compute loop stats
    printf("Compute Loop:\n");
    printf("  Cycles:     %" PRIu64 " (%.1f Hz)\n", compute_cycles, compute_cycles / elapsed_sec);
    if (compute_cycles > 0) {
        printf("  Avg time:   %.2f ms\n", compute_total / 1e6 / compute_cycles);
        printf("  Last time:  %.2f ms\n", compute_last / 1e6);
    }
    printf("  Total notifies: %" PRIu64 " (%.1f Hz)\n", notify_total, notify_total / elapsed_sec);
    printf("\n");

    // Exclusive systems stats
    printf("Exclusive Systems:\n");
    for (std::size_t i = 0; i < exclusive_systems_.size(); ++i) {
        const auto& entry     = exclusive_systems_[i];
        const auto& sys_stats = exclusive_stats_[i];
        const auto notify_cnt = sys_stats.notify_count.load(std::memory_order_relaxed);
        const auto latency    = sys_stats.latency_hist.compute();

        printf(
            "  [%s] (target: %uHz)\n", entry.system->meta().name.c_str(),
            entry.policy.frequency_hz);
        printf("    Runs:     %zu (%.1f Hz)\n", latency.count, latency.count / elapsed_sec);
        printf("    Notifies: %" PRIu64 "\n", notify_cnt);
        if (latency.count > 0) {
            printf("    Min:      %.3f ms\n", latency.min_ns / 1e6);
            printf("    P50:      %.3f ms\n", latency.p50_ns / 1e6);
            printf("    P95:      %.3f ms\n", latency.p95_ns / 1e6);
            printf("    P99:      %.3f ms\n", latency.p99_ns / 1e6);
            printf("    P99.9:    %.3f ms\n", latency.p999_ns / 1e6);
            printf("    Mean:     %.3f ms\n", latency.mean_ns / 1e6);
            printf("    Max:      %.3f ms\n", latency.max_ns / 1e6);
            printf("    StdDev:   %.3f ms\n", latency.stddev_ns / 1e6);
        }
    }

    // Compute systems stats
    printf("\nCompute Systems:\n");
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        const auto& entry     = compute_systems_[i];
        const auto& sys_stats = compute_stats_[i];
        const auto latency    = sys_stats.latency_hist.compute();

        printf("  [%s]\n", entry.system->meta().name.c_str());
        printf("    Runs:   %zu (%.1f Hz)\n", latency.count, latency.count / elapsed_sec);
        if (latency.count > 0) {
            printf("    Min:    %.3f ms\n", latency.min_ns / 1e6);
            printf("    P50:    %.3f ms\n", latency.p50_ns / 1e6);
            printf("    P95:    %.3f ms\n", latency.p95_ns / 1e6);
            printf("    P99:    %.3f ms\n", latency.p99_ns / 1e6);
            printf("    P99.9:  %.3f ms\n", latency.p999_ns / 1e6);
            printf("    Mean:   %.3f ms\n", latency.mean_ns / 1e6);
            printf("    Max:    %.3f ms\n", latency.max_ns / 1e6);
            printf("    StdDev: %.3f ms\n", latency.stddev_ns / 1e6);
        }
    }
    printf("==================================================\n");
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
    const auto notify_total = work_version_.load(std::memory_order_relaxed);

    json root;
    root["runtime_seconds"] = elapsed_sec;

    // Compute loop
    json& compute_json             = root["compute_loop"];
    compute_json["cycles"]         = cycles;
    compute_json["frequency_hz"]   = cycles / elapsed_sec;
    compute_json["avg_time_ms"]    = cycles > 0 ? (total_time / 1e6 / cycles) : 0.0;
    compute_json["last_time_ms"]   = last_time / 1e6;
    compute_json["total_notifies"] = notify_total;

    // Exclusive systems
    json& exclusive_arr = root["exclusive_systems"] = json::array();
    for (std::size_t i = 0; i < exclusive_systems_.size(); ++i) {
        const auto& entry     = exclusive_systems_[i];
        const auto& sys_stats = exclusive_stats_[i];
        const auto notify_cnt = sys_stats.notify_count.load(std::memory_order_relaxed);
        const auto latency    = sys_stats.latency_hist.compute();

        json sys;
        sys["name"]      = entry.system->meta().name;
        sys["target_hz"] = entry.policy.frequency_hz;
        sys["runs"]      = latency.count;
        sys["actual_hz"] = latency.count / elapsed_sec;
        sys["notifies"]  = notify_cnt;
        sys["min_ms"]    = latency.min_ns / 1e6;
        sys["p50_ms"]    = latency.p50_ns / 1e6;
        sys["p95_ms"]    = latency.p95_ns / 1e6;
        sys["p99_ms"]    = latency.p99_ns / 1e6;
        sys["p999_ms"]   = latency.p999_ns / 1e6;
        sys["mean_ms"]   = latency.mean_ns / 1e6;
        sys["max_ms"]    = latency.max_ns / 1e6;
        sys["stddev_ms"] = latency.stddev_ns / 1e6;

        exclusive_arr.push_back(sys);
    }

    // Compute systems
    json& compute_arr = root["compute_systems"] = json::array();
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        const auto& entry     = compute_systems_[i];
        const auto& sys_stats = compute_stats_[i];
        const auto latency    = sys_stats.latency_hist.compute();

        json sys;
        sys["name"]      = entry.system->meta().name;
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

        compute_arr.push_back(sys);
    }

    return root.dump();
}

// ============================================================================
// Exclusive Thread
// ============================================================================

void Scheduler::run_exclusive_thread(ExclusiveContext ctx) {
    // Set CPU affinity
    if (ctx.policy.cpu_affinity >= 0) {
        auto result = primitive::ThreadAffinity::pin_to_core(
            static_cast<std::uint32_t>(ctx.policy.cpu_affinity));
        if (!result) {
            // Log warning but continue
            printf(
                "[WARN] Failed to set CPU affinity for %s: %s\n", ctx.system->meta().name.c_str(),
                primitive::error_message(result.error()).data());
        }
    }

    // Set thread priority
    if (ctx.policy.thread_priority > 0) {
        primitive::ThreadAffinity::RealtimeConfig rt_config{};
        rt_config.priority = static_cast<std::uint8_t>(ctx.policy.thread_priority);
        if (auto result = primitive::ThreadAffinity::set_realtime_priority(rt_config); !result) {
            printf(
                "[WARN] Failed to set thread priority for %s: %s\n",
                ctx.system->meta().name.c_str(), primitive::error_message(result.error()).data());
        }
    }

    const auto frequency     = ctx.policy.frequency_hz;
    const bool should_notify = ctx.policy.notifies;
    auto& stats              = ctx.scheduler->exclusive_stats_[ctx.system_index];

    if (frequency == 0) {
        // Unlimited frequency mode: run as fast as possible
        while (ctx.running->load(std::memory_order_acquire)) {
            LatencyProbe probe;
            ctx.system->run(*ctx.world);
            const auto elapsed = probe.elapsed_ns();

            stats.latency_hist.record(elapsed);

            if (should_notify) {
                ctx.scheduler->notify(ctx.system_index);
            }
        }
        return;
    }

    // Fixed frequency mode
    const auto period =
        std::chrono::nanoseconds(1'000'000'000ULL / static_cast<std::uint64_t>(frequency));
    auto next_tick = Clock::now();

    while (ctx.running->load(std::memory_order_acquire)) {
        next_tick += period;

        // Execute system
        LatencyProbe probe;
        ctx.system->run(*ctx.world);
        const auto elapsed = probe.elapsed_ns();

        stats.latency_hist.record(elapsed);

        // Notify scheduler if configured
        if (should_notify) {
            ctx.scheduler->notify(ctx.system_index);
        }

        // Busy wait until next tick
        std::this_thread::sleep_until(next_tick);
    }
}

// ============================================================================
// Compute Loop
// ============================================================================

void Scheduler::run_compute_loop() {
    const auto print_interval = std::chrono::seconds(5);
    auto last_print           = Clock::now();

    while (running_.load(std::memory_order_acquire)) {
        if (const auto current_version = work_version_.load(std::memory_order_acquire);
            current_version != last_seen_version_) {
            last_seen_version_ = current_version;

            LatencyProbe probe;
            run_compute_once();
            const auto elapsed = probe.elapsed_ns();

            compute_cycles_.fetch_add(1, std::memory_order_relaxed);
            compute_total_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
            compute_last_time_ns_.store(elapsed, std::memory_order_relaxed);
        }

        // Print stats every 5 seconds
        const auto now = Clock::now();
        if (now - last_print >= print_interval) {
            print_stats();
            last_print = now;
        }

        std::this_thread::yield();
    }
}

void Scheduler::run_compute_once() {
    if (compute_systems_.empty()) {
        return;
    }

    compute_arena_->execute([this] {
        for (const auto& level : levels_) {
            if (level.empty()) {
                continue;
            }

            if (level.size() == 1) {
                // Single system: execute directly, avoid task_group overhead
                const auto idx = level[0];
                LatencyProbe probe;
                compute_systems_[idx].system->run(world_);
                const auto elapsed = probe.elapsed_ns();

                compute_stats_[idx].latency_hist.record(elapsed);
            } else {
                // Multiple systems: execute in parallel
                tbb::task_group tg;
                for (const auto idx : level) {
                    tg.run([this, idx] {
                        LatencyProbe probe;
                        compute_systems_[idx].system->run(world_);
                        const auto elapsed = probe.elapsed_ns();

                        compute_stats_[idx].latency_hist.record(elapsed);
                    });
                }
                tg.wait();
            }
        }
    });
}

// ============================================================================
// Dependency Graph
// ============================================================================

auto Scheduler::compute_levels() -> BuildResult {
    if (compute_systems_.empty()) {
        return {};
    }

    const auto n = compute_systems_.size();

    // ========================================================================
    // Phase 1: Collect channel usage information
    // ========================================================================

    struct ChannelUsage {
        bool is_spsc = false;
        std::vector<std::size_t> writers;  // Compute system indices
        std::vector<std::size_t> readers;  // Compute system indices
        bool has_exclusive_writer = false; // Written by exclusive system
        std::string first_system;          // For kind conflict reporting
    };

    std::unordered_map<ChannelKey, ChannelUsage, ChannelKeyHash> channels;

    // First, collect exclusive systems' outputs as external sources
    for (const auto& entry : exclusive_systems_) {
        const auto& meta = entry.system->meta();

        for (const auto& ch : meta.spsc_channels) {
            if (!ch.is_reader) {
                ChannelKey key{ch.type, ch.tag};
                auto& usage                = channels[key];
                usage.is_spsc              = true;
                usage.has_exclusive_writer = true;
                usage.first_system         = meta.name;
            }
        }

        for (const auto& ch : meta.spmc_channels) {
            if (!ch.is_reader) {
                ChannelKey key{ch.type, ch.tag};
                auto& usage                = channels[key];
                usage.is_spsc              = false;
                usage.has_exclusive_writer = true;
                usage.first_system         = meta.name;
            }
        }
    }

    // Then collect compute systems' channels
    for (std::size_t i = 0; i < n; ++i) {
        const auto& meta = compute_systems_[i].system->meta();

        // Process SPSC channels
        for (const auto& ch : meta.spsc_channels) {
            ChannelKey key{ch.type, ch.tag};
            auto& usage = channels[key];

            // Check for kind conflict (same key used as both SPSC and SPMC)
            if (!usage.writers.empty() || !usage.readers.empty()) {
                if (!usage.is_spsc) {
                    return std::unexpected(
                        ChannelKindConflict{
                            .key           = ChannelKeyInfo(ch.type, ch.tag, true),
                            .first_system  = usage.first_system,
                            .second_system = meta.name,
                        });
                }
            } else {
                usage.is_spsc      = true;
                usage.first_system = meta.name;
            }

            if (ch.is_reader) {
                usage.readers.push_back(i);
            } else {
                usage.writers.push_back(i);
            }
        }

        // Process SPMC channels
        for (const auto& ch : meta.spmc_channels) {
            ChannelKey key{ch.type, ch.tag};
            auto& usage = channels[key];

            // Check for kind conflict
            if (!usage.writers.empty() || !usage.readers.empty()) {
                if (usage.is_spsc) {
                    return std::unexpected(
                        ChannelKindConflict{
                            .key           = ChannelKeyInfo(ch.type, ch.tag, false),
                            .first_system  = usage.first_system,
                            .second_system = meta.name,
                        });
                }
            } else {
                usage.is_spsc      = false;
                usage.first_system = meta.name;
            }

            if (ch.is_reader) {
                usage.readers.push_back(i);
            } else {
                usage.writers.push_back(i);
            }
        }
    }

    // ========================================================================
    // Phase 2: Validate channel constraints
    // ========================================================================

    for (const auto& [key, usage] : channels) {
        // Count total writers (exclusive + compute)
        std::size_t total_writers = usage.writers.size() + (usage.has_exclusive_writer ? 1 : 0);

        // Multiple writers is always an error
        if (total_writers > 1) {
            std::vector<std::string> writer_names;
            if (usage.has_exclusive_writer) {
                writer_names.push_back(usage.first_system + " (exclusive)");
            }
            for (auto idx : usage.writers) {
                writer_names.push_back(compute_systems_[idx].system->meta().name);
            }
            return std::unexpected(
                MultipleWritersError{
                    .key     = ChannelKeyInfo(key.type, key.tag, usage.is_spsc),
                    .writers = std::move(writer_names),
                });
        }

        // SPSC with multiple readers is an error
        if (usage.is_spsc && usage.readers.size() > 1) {
            std::vector<std::string> reader_names;
            reader_names.reserve(usage.readers.size());
            for (auto idx : usage.readers) {
                reader_names.push_back(compute_systems_[idx].system->meta().name);
            }
            return std::unexpected(
                MultipleReadersError{
                    .key     = ChannelKeyInfo(key.type, key.tag, true),
                    .readers = std::move(reader_names),
                });
        }

        // Reader without writer is an error (neither exclusive nor compute writer)
        if (total_writers == 0 && !usage.readers.empty()) {
            std::vector<std::string> reader_names;
            reader_names.reserve(usage.readers.size());
            for (auto idx : usage.readers) {
                reader_names.push_back(compute_systems_[idx].system->meta().name);
            }
            return std::unexpected(
                OrphanedReaderError{
                    .key     = ChannelKeyInfo(key.type, key.tag, usage.is_spsc),
                    .readers = std::move(reader_names),
                });
        }

        // Orphaned compute writer is a warning (printed but not an error)
        // Note: exclusive writers with no readers are fine (they notify the scheduler)
        if (!usage.writers.empty() && usage.readers.empty() && !usage.has_exclusive_writer) {
            printf(
                "[WARN] Channel has writer '%s' but no readers\n",
                compute_systems_[usage.writers[0]].system->meta().name.c_str());
        }
    }

    // ========================================================================
    // Phase 3: Build adjacency list and in-degree
    // ========================================================================

    std::vector<std::set<std::size_t>> adj(n); // Use set to deduplicate edges
    std::vector<std::size_t> in_degree(n, 0);

    for (const auto& [key, usage] : channels) {
        if (usage.writers.empty() || usage.readers.empty()) {
            continue;
        }

        std::size_t writer = usage.writers[0];
        for (std::size_t reader : usage.readers) {
            if (writer != reader && adj[writer].insert(reader).second) {
                // Only increment in_degree if this is a new edge
                ++in_degree[reader];
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
        for (std::size_t node : current_level) {
            for (std::size_t neighbor : adj[node]) {
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

        // Find a node still in the cycle (in_degree > 0)
        std::size_t start = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (in_degree[i] > 0) {
                start = i;
                break;
            }
        }

        // Simple cycle detection: follow edges until we revisit a node
        std::vector visited(n, false);
        std::vector in_stack(n, false);
        std::vector<std::size_t> path;

        std::function<bool(std::size_t)> find_cycle = [&](const std::size_t node) -> bool {
            visited[node]  = true;
            in_stack[node] = true;
            path.push_back(node);

            for (const std::size_t neighbor : adj[node]) {
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

    return {};
}

} // namespace talos
