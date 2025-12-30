#pragma once

#include "../ecs/execution_policy.hpp"
#include "../ecs/system.hpp"
#include "../ecs/system_meta.hpp"
#include "../ecs/world.hpp"

#include <tbb/task_arena.h>
#include <tbb/task_group.h>

#include <atomic>
#include <chrono>
#include <expected>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace talos {

// ============================================================================
// Runtime Errors (simple enum for lifecycle operations)
// ============================================================================

enum class SchedulerError {
    AlreadyBuilt,
    AlreadyRunning,
    NotBuilt,
};

[[nodiscard]] constexpr auto to_string(const SchedulerError e) noexcept -> const char* {
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

/// Channel identifier for error reporting
struct ChannelKeyInfo {
    std::type_index type;
    std::type_index tag;
    bool is_spsc;

    ChannelKeyInfo(const std::type_index t, const std::type_index tg, const bool spsc)
        : type(t)
        , tag(tg)
        , is_spsc(spsc) {}
};

/// Same channel key used for both SPSC and SPMC
struct ChannelKindConflict {
    ChannelKeyInfo key;
    std::string first_system;
    std::string second_system;
};

/// SPSC or SPMC channel has multiple writers
struct MultipleWritersError {
    ChannelKeyInfo key;
    std::vector<std::string> writers;
};

/// SPSC channel has multiple readers
struct MultipleReadersError {
    ChannelKeyInfo key;
    std::vector<std::string> readers;
};

/// Reader without a corresponding writer
struct OrphanedReaderError {
    ChannelKeyInfo key;
    std::vector<std::string> readers;
};

/// Dependency cycle detected in the graph
struct DependencyCycleError {
    std::vector<std::string> cycle;
};

/// All possible build errors
using BuildError = std::variant<
    SchedulerError, // Runtime/lifecycle errors (AlreadyBuilt, etc.)
    ChannelKindConflict, MultipleWritersError, MultipleReadersError, OrphanedReaderError,
    DependencyCycleError>;

/// Build result type
using BuildResult = std::expected<void, BuildError>;

/// Format BuildError to string for debugging
[[nodiscard]] inline auto format_build_error(const BuildError& err) -> std::string {
    return std::visit(
        [](const auto& e) -> std::string {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, SchedulerError>) {
                return to_string(e);
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
            }
            return "Unknown error";
        },
        err);
}

// ============================================================================
// Configuration
// ============================================================================

struct SchedulerConfig {
    /// TBB arena concurrency, 0 = hardware concurrency
    std::size_t compute_concurrency = 0;
};

// ============================================================================
// System Entry
// ============================================================================

struct SystemEntry {
    std::unique_ptr<SystemBase> system;
    PolicyInfo policy;
};

// ============================================================================
// Scheduler
// ============================================================================

class Scheduler {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration  = Clock::duration;

    explicit Scheduler(World& world, SchedulerConfig config = {});
    ~Scheduler();

    Scheduler(const Scheduler&)            = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&)                 = delete;
    Scheduler& operator=(Scheduler&&)      = delete;

    // ========================================================================
    // System Registration
    // ========================================================================

    template <typename Policy = default_policy, typename F>
    auto add_system(std::string name, F&& func) -> std::expected<void, SchedulerError>;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Build dependency graph and prepare for execution
    [[nodiscard]] auto build() -> BuildResult;

    /// Start scheduler, blocks until stop() is called
    [[nodiscard]] auto run() -> std::expected<void, SchedulerError>;

    /// Request graceful shutdown
    void stop() noexcept;

    /// Check if running
    [[nodiscard]] auto is_running() const noexcept -> bool;

    // ========================================================================
    // Notification (called by exclusive threads)
    // ========================================================================

    void notify() noexcept;

    // ========================================================================
    // Diagnostics
    // ========================================================================

    struct Stats {
        std::uint64_t notify_count        = 0;
        std::uint64_t compute_cycle_count = 0;
    };

    [[nodiscard]] auto stats() const noexcept -> Stats;

    /// Print registered systems info
    void print_systems() const;

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
    };

    // ========================================================================
    // Internal Methods
    // ========================================================================

    static void run_exclusive_thread(ExclusiveContext ctx);
    void run_compute_loop();
    void run_compute_once();

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

    // Synchronization
    std::atomic<std::uint64_t> work_version_{0};
    std::uint64_t last_seen_version_{0}; // Only accessed by compute loop

    // Stats
    std::atomic<std::uint64_t> compute_cycles_{0};
};

// ============================================================================
// Template Implementation
// ============================================================================

template <typename Policy, typename F>
auto Scheduler::add_system(std::string name, F&& func) -> std::expected<void, SchedulerError> {
    if (built_.load(std::memory_order_acquire)) {
        return std::unexpected(SchedulerError::AlreadyBuilt);
    }

    auto system = make_system<F, Policy>(std::move(name), std::forward<F>(func));
    auto policy = make_policy_info<Policy>();

    SystemEntry entry{
        .system = std::move(system),
        .policy = policy,
    };

    if constexpr (is_exclusive_policy_v<Policy>) {
        exclusive_systems_.push_back(std::move(entry));
    } else {
        compute_systems_.push_back(std::move(entry));
    }

    return {};
}

} // namespace talos
