#pragma once

/**
 * @file
 * @brief Scheduler for data-driven task execution
 *
 * This module provides a scheduler that manages exclusive (single-threaded)
 * and compute (thread-pool) systems with dependency-aware execution.
 *
 * ## Architecture
 *
 * - **Exclusive systems**: Run on dedicated threads, can notify the scheduler
 * - **Compute systems**: Run on TBB thread pool, triggered by notifications
 * - **Dependency graph**: Built from channel connections, enables selective wakeup
 *
 * ## Execution model
 *
 * 1. Exclusive systems run on their own threads
 * 2. When an exclusive system notifies, it triggers dependent compute systems
 * 3. Compute systems run in topological order (levels)
 * 4. TBB arena manages thread pool concurrency
 *
 * ## Build errors
 *
 * The scheduler validates the system graph during `build()`:
 * - Single writer per channel (SPSC) / multiple writers allowed (SPMC)
 * - No cycles in dependency graph
 * - Orphaned readers detected
 * - Max 32 compute systems for bitmask scheduling
 */

#include "ecs/execution_policy.hpp"
#include "ecs/system.hpp"
#include "ecs/world.hpp"
#include "performance_probe.hpp"

#include <cstdint>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <expected>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace talos {

// ============================================================================
// Runtime Errors (simple enum for lifecycle operations)
// ============================================================================

/**
 * @brief Scheduler lifecycle errors
 */
enum class SchedulerError {
    AlreadyBuilt,   ///< Scheduler was already built
    AlreadyRunning, ///< Scheduler is already running
    NotBuilt,       ///< Scheduler was not built before run
};

/**
 * @brief Convert SchedulerError to human-readable string
 *
 * ## Returns
 *
 * C-string describing the error
 */
[[nodiscard]] constexpr std::string_view to_string(const SchedulerError e) noexcept {
    switch (e) {
    case SchedulerError::AlreadyBuilt: return "Scheduler already built";
    case SchedulerError::AlreadyRunning: return "Scheduler already running";
    case SchedulerError::NotBuilt: return "Scheduler not built";
    }
    return "Unknown error";
}

// ============================================================================
// Build Errors (detailed variant for dependency analysis)
// ============================================================================

/**
 * @brief Channel identifier for error reporting
 *
 * Combines data type and tag to uniquely identify a channel.
 */
struct ChannelKeyInfo {
    std::type_index type;
    std::type_index tag;
    bool is_spsc;

    /**
     * @brief Construct a ChannelKeyInfo
     *
     * ## Parameters
     *
     * - `t`: data type index
     * - `tg`: tag type index
     * - `spsc`: true if SPSC channel, false if SPMC
     */
    ChannelKeyInfo(const std::type_index t, const std::type_index tg, const bool spsc)
        : type(t)
        , tag(tg)
        , is_spsc(spsc) {}
};

/**
 * @brief Same channel key used for both SPSC and SPMC
 */
struct ChannelKindConflict {
    ChannelKeyInfo key;
    std::string first_system;
    std::string second_system;
};

/**
 * @brief SPSC or SPMC channel has multiple writers
 */
struct MultipleWritersError {
    ChannelKeyInfo key;
    std::vector<std::string> writers;
};

/**
 * @brief SPSC channel has multiple readers
 */
struct MultipleReadersError {
    ChannelKeyInfo key;
    std::vector<std::string> readers;
};

/**
 * @brief Reader without a corresponding writer
 */
struct OrphanedReaderError {
    ChannelKeyInfo key;
    std::vector<std::string> readers;
};

/**
 * @brief Dependency cycle detected in the graph
 */
struct DependencyCycleError {
    std::vector<std::string> cycle;
};

/**
 * @brief Too many compute systems (max 32 for efficient bitmask scheduling)
 */
struct TooManyComputeSystemsError {
    std::size_t count;
    static constexpr std::size_t max_count = 32;
};

/**
 * @brief All possible build errors
 */
using BuildError = std::variant<
    SchedulerError, // Runtime/lifecycle errors (AlreadyBuilt, etc.)
    ChannelKindConflict, MultipleWritersError, MultipleReadersError, OrphanedReaderError,
    DependencyCycleError, TooManyComputeSystemsError>;

/**
 * @brief Build result type
 */
using BuildResult = std::expected<void, BuildError>;

/**
 * @brief Format BuildError to string for debugging
 *
 * ## Parameters
 *
 * - `err`: the build error to format
 *
 * ## Returns
 *
 * Human-readable error description
 */
[[nodiscard]] inline auto format_build_error(const BuildError& err) -> std::string {
    return std::visit(
        [](const auto& e) -> std::string {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, SchedulerError>) {
                return std::string(to_string(e));
            } else if constexpr (std::is_same_v<T, ChannelKindConflict>) {
                return "ChannelKindConflict: channel used as both SPSC and SPMC by '"
                     + e.first_system + "' and '" + e.second_system + "'";
            } else if constexpr (std::is_same_v<T, MultipleWritersError>) {
                std::string s = "MultipleWriters: channel has multiple writers: ";
                for (std::size_t i = 0; i < e.writers.size(); ++i) {
                    if (i > 0)
                        s += ", ";
                    s += "'" + e.writers[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, MultipleReadersError>) {
                std::string s = "MultipleReaders: SPSC channel has multiple readers: ";
                for (std::size_t i = 0; i < e.readers.size(); ++i) {
                    if (i > 0)
                        s += ", ";
                    s += "'" + e.readers[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, OrphanedReaderError>) {
                std::string s = "OrphanedReader: reader(s) without writer: ";
                for (std::size_t i = 0; i < e.readers.size(); ++i) {
                    if (i > 0)
                        s += ", ";
                    s += "'" + e.readers[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, DependencyCycleError>) {
                std::string s = "DependencyCycle: ";
                for (std::size_t i = 0; i < e.cycle.size(); ++i) {
                    if (i > 0)
                        s += " -> ";
                    s += "'" + e.cycle[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, TooManyComputeSystemsError>) {
                return "TooManyComputeSystems: " + std::to_string(e.count) + " (max "
                     + std::to_string(e.max_count) + ")";
            }
            return "Unknown error";
        },
        err);
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief Scheduler configuration
 */
struct SchedulerConfig {
    /**
     * @brief TBB arena concurrency
     *
     * Number of threads in the compute pool. 0 = hardware concurrency.
     *
     * ## Default
     *
     * 0 (use all available cores)
     */
    std::size_t compute_concurrency = 0;
};

// ============================================================================
// System Entry
// ============================================================================

/**
 * @brief Internal storage for a registered system
 */
struct SystemEntry {
    std::unique_ptr<SystemBase> system;
    PolicyInfo policy;
};

// ============================================================================
// Scheduler
// ============================================================================

/**
 * @brief Main scheduler for data-driven task execution
 *
 * The scheduler manages exclusive systems (single-threaded, dedicated threads)
 * and compute systems (thread-pool based) with dependency-aware execution.
 *
 * ## Usage
 *
 * ```cpp
 * World world;
 * Scheduler scheduler(world);
 *
 * scheduler.add_system<exclusive<30>>("camera", [](talos::spmc_mut<fcs::AtLegacyImageFrame,
 * fcs::AtLegacyImageChannelTag> cam_out) {
 *     // Process camera input, write to channel
 *     cam_out.write(...);
 * });
 *
 * auto result = scheduler.build();
 * if (!result) {
 *     // Handle build error
 * }
 *
 * scheduler.run();  // Blocks until stop() is called
 * ```
 *
 * ## Thread safety
 *
 * The scheduler is not thread-safe after construction. Configure and build
 * in a single-threaded context before calling `run()`.
 */
class Scheduler {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration  = Clock::duration;

    /**
     * @brief Construct a scheduler
     *
     * ## Parameters
     *
     * - `world`: the ECS world containing resources and channels
     * - `config`: optional scheduler configuration
     */
    explicit Scheduler(World& world, SchedulerConfig config = {});
    ~Scheduler();

    Scheduler(const Scheduler&)            = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&)                 = delete;
    Scheduler& operator=(Scheduler&&)      = delete;

    // ========================================================================
    // System Registration
    // ========================================================================

    /**
     * @brief Add a system to the scheduler
     *
     * Systems are analyzed at build time to extract channel dependencies
     * and determine the execution policy.
     *
     * ## Template parameters
     *
     * - `Policy`: execution policy (exclusive, exclusive_silent, pool_compute)
     * - `F`: callable type for the system function
     *
     * ## Parameters
     *
     * - `name`: human-readable system name (for debugging)
     * - `func`: the system function to execute
     */
    template <typename Policy = default_policy, typename F>
    auto add_system(std::string name, F&& func) -> void;

    /**
     * @brief Add a pre-built custom system to the scheduler
     *
     * This overload allows advanced users to provide their own SystemBase implementation,
     * including custom bind/run behavior and runtime-constructed SystemMeta.
     *
     * The system's execution policy is taken from `system->meta().policy`.
     *
     * ## Notes
     *
     * - Registration is only valid before `build()`.
     * - The scheduler's dependency analysis and validation still uses SystemMeta.
     */
    auto add_system(std::unique_ptr<SystemBase> system) -> uint32_t;

    /**
     * @brief Get the ECS world
     *
     * ## Returns
     *
     * Reference to the World containing resources and channels
     */
    [[nodiscard]] World& world() noexcept { return world_; }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * @brief Build dependency graph and prepare for execution
     *
     * Validates the system graph:
     * - No dependency cycles
     * - Channel constraints (SPSC single writer, etc.)
     * - Orphaned readers detected
     *
     * Can be called multiple times for dynamic system addition (idempotent).
     * Each call re-validates all channels and rebuilds wake-up chains.
     *
     * ## Returns
     *
     * `std::expected<void, BuildError>` - success or detailed error
     *
     * ## Thread safety
     *
     * Must be called from a single thread before `run()`.
     */
    [[nodiscard]] auto build() -> BuildResult;

    /**
     * @brief Start scheduler, blocks until stop() is called
     *
     * Launches exclusive system threads and enters the compute loop.
     *
     * ## Returns
     *
     * `std::expected<void, SchedulerError>` - success or lifecycle error
     *
     * ## Panics
     *
     * Panics if called without a prior successful `build()`.
     */
    [[nodiscard]] auto run() -> std::expected<void, SchedulerError>;

    /**
     * @brief Request graceful shutdown
     *
     * Sets the running flag to false. Exclusive threads will exit
     * after their current tick completes.
     */
    void stop() noexcept;

    /**
     * @brief Hot-add a system while the scheduler is running
     *
     * Pauses the compute loop, adds the system, rebuilds wake chains,
     * and resumes execution. Safe to call from any thread.
     *
     * ## Parameters
     *
     * - `system`: the system to add (moved)
     *
     * ## Returns
     *
     * BuildResult indicating success or failure
     *
     * ## Thread safety
     *
     * Thread-safe. Uses pause/resume protocol to safely modify the graph.
     *
     * ## Notes
     *
     * If the scheduler is not running, this behaves like add_system() + build().
     */
    [[nodiscard]] auto hot_add_system(std::unique_ptr<SystemBase> system) -> BuildResult;

    /**
     * @brief Check if running
     *
     * ## Returns
     *
     * `true` if the scheduler is currently running
     */
    [[nodiscard]] auto is_running() const noexcept -> bool;

    // ========================================================================
    // Notification (called by exclusive threads)
    // ========================================================================

    /**
     * @brief Notify scheduler that a system has produced data
     *
     * Called by exclusive systems to trigger dependent compute systems.
     * The notification uses a bitmask for efficient selective wakeup.
     *
     * ## Parameters
     *
     * - `system_index`: index of the exclusive system in exclusive_systems_
     *
     * ## Thread safety
     *
     * Safe to call from any exclusive system thread.
     */
    void notify(std::size_t system_index) noexcept;

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /**
     * @brief Runtime statistics
     */
    struct Stats {
        std::uint64_t notify_count        = 0; ///< Total notifications received
        std::uint64_t compute_cycle_count = 0; ///< Total compute cycles executed
    };

    /**
     * @brief Get runtime statistics
     *
     * ## Returns
     *
     * Current scheduler statistics
     */
    [[nodiscard]] auto stats() const noexcept -> Stats;

    /**
     * @brief Print registered systems info
     *
     * Outputs system names, policies, and channel counts to stdout.
     */
    void print_systems() const;

    /*
     * @brief Print wake chains as Mermaid diagram
     *
     * Outputs the dependency graph showing which systems trigger other systems
     * in Mermaid flowchart syntax for visualization.
     *
     */
    void print_mermaid_wake_chains() const;

    /**
     * @brief Print performance statistics
     *
     * Outputs latency histograms and throughput metrics to stdout.
     */
    void print_stats() const;

    /**
     * @brief Get performance statistics as JSON string
     *
     * ## Returns
     *
     * JSON-formatted statistics string
     */
    [[nodiscard]] std::string get_stats_json() const;

private:
    // ========================================================================
    // Internal Types
    // ========================================================================

    struct ExclusiveContext {
        SystemBase* system;
        PolicyInfo policy;
        std::atomic<bool>* running;
        Scheduler* scheduler;
        World* world;
        std::size_t system_index;
    };

    /**
     * @brief Performance statistics for exclusive systems
     */
    struct ExclusiveStats {
        std::atomic<std::uint64_t> notify_count{0};
        LatencyHistogram latency_hist;

        ExclusiveStats() = default;

        // Move constructor: atomics are not movable, so we load and store
        ExclusiveStats(ExclusiveStats&& other) noexcept
            : notify_count(other.notify_count.load(std::memory_order_relaxed))
            , latency_hist(other.latency_hist) {}

        ExclusiveStats(const ExclusiveStats&)            = delete;
        ExclusiveStats& operator=(const ExclusiveStats&) = delete;
        ExclusiveStats& operator=(ExclusiveStats&&)      = delete;
    };

    /**
     * @brief Performance statistics for compute systems
     */
    struct ComputeStats {
        LatencyHistogram latency_hist;

        ComputeStats()                               = default;
        ComputeStats(ComputeStats&&)                 = default;
        ComputeStats(const ComputeStats&)            = delete;
        ComputeStats& operator=(const ComputeStats&) = delete;
        ComputeStats& operator=(ComputeStats&&)      = delete;
    };

    // ========================================================================
    // Internal Methods
    // ========================================================================

    static void run_exclusive_thread(ExclusiveContext ctx);
    void run_compute_loop();
    [[nodiscard]] std::uint64_t
        run_compute_selective(std::uint64_t ready_mask); // Returns mask of systems that wrote

    [[nodiscard]] auto compute_levels() -> BuildResult;

    // ========================================================================
    // Members
    // ========================================================================

    World& world_;
    SchedulerConfig config_;

    // Systems
    std::vector<SystemEntry> exclusive_systems_;
    std::vector<SystemEntry> compute_systems_;

    // Execution levels for compute systems (topologically sorted)
    std::vector<std::vector<std::size_t>> levels_;

    // Threads
    std::vector<std::jthread> exclusive_threads_;

    // TBB
    std::unique_ptr<tbb::task_arena> compute_arena_;

    // State
    std::atomic<bool> built_{false};
    std::atomic<bool> running_{false};

    // Selective notification: wake chains for all systems
    std::vector<std::uint64_t> exclusive_affects_; // [exclusive_idx] -> bitmask of compute systems
    std::vector<std::uint64_t>
        compute_affects_; // [compute_idx] -> bitmask of dependent compute systems
    std::atomic<std::uint64_t> ready_systems_{0}; // Bitmask of compute systems pending execution

    // Stats
    std::atomic<std::uint64_t> compute_cycles_{0};
    std::atomic<std::uint64_t> compute_total_time_ns_{0};
    std::atomic<std::uint64_t> compute_last_time_ns_{0};
    std::vector<ExclusiveStats> exclusive_stats_;
    std::vector<ComputeStats> compute_stats_;
    TimePoint stats_start_time_;

    // Hot-add pause state
    std::atomic<bool> pause_requested_{false};
    std::atomic<bool> paused_{false};
    std::condition_variable pause_cv_;
    std::mutex pause_mutex_;
};

// ============================================================================
// Template Implementation
// ============================================================================

template <typename Policy, typename F>
auto Scheduler::add_system(std::string name, F&& func) -> void {
    // If already running, must use hot_add_system instead
    if (running_.load(std::memory_order_acquire)) {
        std::abort();
    }

    // Allow adding systems after build (will require re-build)
    if (built_.load(std::memory_order_acquire)) {
        built_.store(false, std::memory_order_release);
    }

    auto system = make_system<F, Policy>(std::move(name), std::forward<F>(func));
    auto policy = make_policy_info<Policy>();

    SystemEntry entry{
        .system = std::move(system),
        .policy = policy,
    };

    if constexpr (is_exclusive_policy_v<Policy>) {
        exclusive_systems_.push_back(std::move(entry));
        return;
    }
    if constexpr (is_pool_policy_v<Policy> || is_visualization_policy_v<Policy>) {
        compute_systems_.push_back(std::move(entry));
        return;
    }
    std::unreachable();
}

} // namespace talos
