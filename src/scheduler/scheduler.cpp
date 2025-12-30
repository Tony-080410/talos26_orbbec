#include "scheduler/scheduler/scheduler.hpp"
#include "scheduler/primitives/thread_affinity.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <set>
#include <unordered_map>

#ifdef __x86_64__
# include <immintrin.h>
# define SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__)
# define SPIN_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
# define SPIN_PAUSE() ((void)0)
#endif

namespace talos {

// ============================================================================
// Constructor / Destructor
// ============================================================================

Scheduler::Scheduler(World& world, const SchedulerConfig config)
    : world_(world)
    , config_(config) {}

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
    for (auto& entry : exclusive_systems_) {
        ExclusiveContext ctx{
            .system    = entry.system.get(),
            .policy    = entry.policy,
            .running   = &running_,
            .scheduler = this,
            .world     = &world_,
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

void Scheduler::notify() noexcept { work_version_.fetch_add(1, std::memory_order_release); }

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

    if (frequency == 0) {
        // Unlimited frequency mode: run as fast as possible
        while (ctx.running->load(std::memory_order_acquire)) {
            ctx.system->run(*ctx.world);
            if (should_notify) {
                ctx.scheduler->notify();
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
        ctx.system->run(*ctx.world);

        // Notify scheduler if configured
        if (should_notify) {
            ctx.scheduler->notify();
        }

        // Busy wait until next tick
        while (Clock::now() < next_tick) {
            if (!ctx.running->load(std::memory_order_relaxed)) {
                return;
            }
            SPIN_PAUSE();
        }
    }
}

// ============================================================================
// Compute Loop
// ============================================================================

void Scheduler::run_compute_loop() {
    while (running_.load(std::memory_order_acquire)) {
        if (const auto current_version = work_version_.load(std::memory_order_acquire);
            current_version != last_seen_version_) {
            last_seen_version_ = current_version;
            run_compute_once();
            compute_cycles_.fetch_add(1, std::memory_order_relaxed);
        }

        SPIN_PAUSE();
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
                compute_systems_[level[0]].system->run(world_);
            } else {
                // Multiple systems: execute in parallel
                tbb::task_group tg;
                for (const auto idx : level) {
                    tg.run([this, idx] { compute_systems_[idx].system->run(world_); });
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
