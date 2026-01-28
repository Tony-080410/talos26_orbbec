#include "scheduler/scheduler.hpp"
#include "primitives/performance_probe.hpp"
#include "primitives/thread_affinity.hpp"
#include "scheduler/error_formatter.hpp"

#include "scheduler/spin.hpp"
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <flat_map>
#include <functional>
#include <nlohmann/json.hpp>
#include <set>
#include <spdlog/spdlog.h>

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

    for (auto& t : fixed_rate_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

uint32_t Scheduler::add_system(std::unique_ptr<SystemBase> system) {
    if (!system) {
        std::abort();
    }
    // If already running, must use hot_add_system instead
    if (running_.load(std::memory_order_acquire)) {
        SPDLOG_CRITICAL("Cannot call add_system while scheduler is running. Use hot_add_system.");
        std::abort();
    }

    // Allow adding systems after build (will require re-build)
    // Mark as not-built so build() can be called again
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
        return fixed_rate_systems_.size() - 1;
    }
    if (policy.is_pool() || policy.is_visualization()) {
        compute_systems_.push_back(std::move(entry));
        return compute_systems_.size() - 1;
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
    if (fixed_rate_stats_.size() < fixed_rate_systems_.size()) {
        fixed_rate_stats_.reserve(fixed_rate_systems_.size());
        for (std::size_t i = fixed_rate_stats_.size(); i < fixed_rate_systems_.size(); ++i) {
            fixed_rate_stats_.emplace_back();
        }
    }

    if (compute_stats_.size() < compute_systems_.size()) {
        compute_stats_.reserve(compute_systems_.size());
        for (std::size_t i = compute_stats_.size(); i < compute_systems_.size(); ++i) {
            compute_stats_.emplace_back();
        }
    }

    // Bind all systems (pre-create channel readers/writers, single-threaded)
    for (auto& entry : fixed_rate_systems_) {
        entry.system->bind(world_);
    }
    for (auto& entry : compute_systems_) {
        entry.system->bind(world_);
    }

    built_.store(true, std::memory_order_release);
    return {};
}

auto Scheduler::run() -> std::expected<void, SchedulerError> {
    if (!built_.load(std::memory_order_acquire)) {
        return std::unexpected(SchedulerError::NotBuilt);
    }

    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return std::unexpected(SchedulerError::AlreadyRunning);
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
    {
        std::lock_guard lock(pause_mutex_);
        pause_requested_.store(true, std::memory_order_release);
    }

    // Wait for compute loop to pause
    {
        std::unique_lock lock(pause_mutex_);
        pause_cv_.wait(lock, [this] { return paused_.load(std::memory_order_acquire); });
    }

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
        {
            std::lock_guard lock(pause_mutex_);
            pause_requested_.store(false, std::memory_order_release);
        }
        pause_cv_.notify_all();

        return result;
    }

    // Update stats capacity
    if (fixed_rate_stats_.size() < fixed_rate_systems_.size()) {
        fixed_rate_stats_.reserve(fixed_rate_systems_.size());
        for (std::size_t i = fixed_rate_stats_.size(); i < fixed_rate_systems_.size(); ++i) {
            fixed_rate_stats_.emplace_back();
        }
    }

    if (compute_stats_.size() < compute_systems_.size()) {
        compute_stats_.reserve(compute_systems_.size());
        for (std::size_t i = compute_stats_.size(); i < compute_systems_.size(); ++i) {
            compute_stats_.emplace_back();
        }
    }

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
    {
        std::lock_guard lock(pause_mutex_);
        pause_requested_.store(false, std::memory_order_release);
    }
    pause_cv_.notify_all();

    return {};
}

auto Scheduler::is_running() const noexcept -> bool {
    return running_.load(std::memory_order_acquire);
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
    // Sum notify counts from all fixed_rate systems
    std::uint64_t total_notifies = 0;
    for (const auto& s : fixed_rate_stats_) {
        total_notifies += s.notify_count.load(std::memory_order_relaxed);
    }
    return Stats{
        .notify_count        = total_notifies,
        .compute_cycle_count = compute_cycles_.load(std::memory_order_relaxed),
    };
}

void Scheduler::print_systems() const {
    printf("FixedRate systems (%zu):\n", fixed_rate_systems_.size());
    for (const auto& entry : fixed_rate_systems_) {
        const auto& m = entry.system->meta();
        printf(
            "  [%s] freq:%uHz affinity:%d priority:%d notify:%s spsc:%zu spmc:%zu\n",
            m.name.c_str(), entry.policy.frequency_hz, entry.policy.cpu_affinity,
            entry.policy.thread_priority, entry.policy.notifies ? "yes" : "no",
            m.spsc_channels.size(), m.spmc_channels.size());
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

    // Print wake chains (which systems trigger which compute systems)
    printf("Wake chains (fixed_rate -> compute):\n");
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        const auto& excl_name = fixed_rate_systems_[i].system->meta().name;
        const auto affects    = i < fixed_rate_affects_.size() ? fixed_rate_affects_[i] : 0U;

        printf("  [%s] -> ", excl_name.c_str());
        if (affects == 0) {
            printf("(none)");
        } else {
            bool first = true;
            for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
                if (affects & (1U << j)) {
                    if (!first)
                        printf(", ");
                    printf("[%s]", compute_systems_[j].system->meta().name.c_str());
                    first = false;
                }
            }
        }
        printf("\n");
    }

    printf("Wake chains (compute -> compute):\n");
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        const auto& comp_name = compute_systems_[i].system->meta().name;
        const auto affects    = i < compute_affects_.size() ? compute_affects_[i] : 0U;

        printf("  [%s] -> ", comp_name.c_str());
        if (affects == 0) {
            printf("(none)");
        } else {
            bool first = true;
            for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
                if (affects & (1U << j)) {
                    if (!first)
                        printf(", ");
                    printf("[%s]", compute_systems_[j].system->meta().name.c_str());
                    first = false;
                }
            }
        }
        printf("\n");
    }
}

void Scheduler::print_mermaid_wake_chains() const {
    printf("```mermaid\n");
    printf("flowchart LR\n\n");

    // FixedRate nodes
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        const auto& name = fixed_rate_systems_[i].system->meta().name;
        printf("    E%d[%s]\n", (int)i, name.c_str());
    }

    // Compute nodes
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        const auto& name = compute_systems_[i].system->meta().name;
        printf("    C%d[%s]\n", (int)i, name.c_str());
    }

    printf("\n");

    // FixedRate → Compute
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        auto affects = i < fixed_rate_affects_.size() ? fixed_rate_affects_[i] : 0U;
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if (affects & (1U << j)) {
                printf("    E%d --> C%d\n", (int)i, (int)j);
            }
        }
    }

    // Compute → Compute
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        auto affects = i < compute_affects_.size() ? compute_affects_[i] : 0U;
        for (std::size_t j = 0; j < compute_systems_.size(); ++j) {
            if ((affects & (1U << j)) && i != j) {
                printf("    C%d --> C%d\n", (int)i, (int)j);
            }
        }
    }

    printf("```\n");
}

void Scheduler::print_stats() const {
    const auto now            = Clock::now();
    const auto elapsed_sec    = std::chrono::duration<double>(now - stats_start_time_).count();
    const auto compute_cycles = compute_cycles_.load(std::memory_order_relaxed);
    const auto compute_total  = compute_total_time_ns_.load(std::memory_order_relaxed);
    const auto compute_last   = compute_last_time_ns_.load(std::memory_order_relaxed);

    std::uint64_t notify_total = 0;
    for (const auto& s : fixed_rate_stats_) {
        notify_total += s.notify_count.load(std::memory_order_relaxed);
    }

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
    // FixedRate systems - system name on one line, stats on the next
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        const auto& entry     = fixed_rate_systems_[i];
        const auto& sys_stats = fixed_rate_stats_[i];
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

        if (latency.count > 0) {
            printf(
                "   min:%.2f p50:%.2f p95:%.2f max:%.2f ms", latency.min_ns / 1e6,
                latency.p50_ns / 1e6, latency.p95_ns / 1e6, latency.max_ns / 1e6);
            printf("\n");
        }
    }

    // Compute systems - system name on one line, stats on the next
    for (std::size_t i = 0; i < compute_systems_.size(); ++i) {
        const auto& entry     = compute_systems_[i];
        const auto& sys_stats = compute_stats_[i];
        const auto latency    = sys_stats.latency_hist.compute();
        const auto actual_hz  = latency.count / elapsed_sec;

        printf(
            "  %-30s@%.1fHz runs:%zu", entry.system->meta().name.c_str(), actual_hz, latency.count);
        printf("\n");

        if (latency.count > 0) {
            printf(
                "   min:%.2f p50:%.2f p95:%.2f max:%.2f ms", latency.min_ns / 1e6,
                latency.p50_ns / 1e6, latency.p95_ns / 1e6, latency.max_ns / 1e6);
            printf("\n");
        }
    }
    printf("======================================\n");
    printf("\n");
}

auto Scheduler::get_stats_json() const -> std::string {
    using json = nlohmann::json;

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - stats_start_time_);
    const double elapsed_sec = elapsed.count();

    // Compute loop stats
    const auto cycles     = compute_cycles_.load(std::memory_order_relaxed);
    const auto total_time = compute_total_time_ns_.load(std::memory_order_relaxed);
    const auto last_time  = compute_last_time_ns_.load(std::memory_order_relaxed);

    // Sum notify counts from all fixed_rate systems
    std::uint64_t notify_total = 0;
    for (const auto& s : fixed_rate_stats_) {
        notify_total += s.notify_count.load(std::memory_order_relaxed);
    }

    json root;
    root["runtime_seconds"] = elapsed_sec;

    // Compute loop
    json& compute_json             = root["compute_loop"];
    compute_json["cycles"]         = cycles;
    compute_json["frequency_hz"]   = cycles / elapsed_sec;
    compute_json["avg_time_ms"]    = cycles > 0 ? (total_time / 1e6 / cycles) : 0.0;
    compute_json["last_time_ms"]   = last_time / 1e6;
    compute_json["total_notifies"] = notify_total;

    // FixedRate systems
    json& fixed_rate_arr = root["fixed_rate_systems"] = json::array();
    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        const auto& entry     = fixed_rate_systems_[i];
        const auto& sys_stats = fixed_rate_stats_[i];
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

        fixed_rate_arr.push_back(sys);
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
            LatencyProbe probe;
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
        LatencyProbe probe;
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
    const auto print_interval = std::chrono::seconds(5);
    auto last_print           = Clock::now();

    while (running_.load(std::memory_order_acquire)) [[likely]] {
        // Check for pause request (hot-add in progress)
        if (pause_requested_.load(std::memory_order_acquire)) [[unlikely]] {
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
            LatencyProbe probe;
            const auto written = run_compute_selective(ready);
            const auto elapsed = probe.elapsed_ns();

            compute_cycles_.fetch_add(1, std::memory_order_relaxed);
            compute_total_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
            compute_last_time_ns_.store(elapsed, std::memory_order_relaxed);

            // Cascade wake: mark dependents of systems that actually wrote
            std::uint64_t cascade = 0;
            for (std::size_t i = 0; i < compute_affects_.size(); ++i) {
                if (written & (1U << i)) {
                    cascade |= compute_affects_[i];
                }
            }
            if (cascade != 0) {
                ready_systems_.fetch_or(cascade, std::memory_order_release);
            }

            // Check stats after compute (amortize cost)
            /*
            const auto now = Clock::now();
            if (now - last_print >= print_interval) [[unlikely]] {
                print_stats();
                last_print = now;
                }
            */
        } else {
            // Low-latency spin: use CPU hint instead of yield()
            // - _mm_pause (x86): ~10 cycles, no context switch
            // - yield (ARM): hint to switch to another hyperthread
            // Much faster than std::this_thread::yield() which causes ~100μs delay
            SPIN_HINT();
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
        // Max 32 systems, so 32 elements is sufficient
        std::size_t to_run_buf[32];
        std::size_t to_run_count = 0;

        for (const auto& level : levels_) {
            // Filter systems in this level that are ready (inline, no temporary vector)
            to_run_count = 0;
            for (const auto idx : level) {
                if (ready_mask & (1U << idx)) {
                    to_run_buf[to_run_count++] = idx;
                }
            }

            if (to_run_count == 0) [[unlikely]] {
                continue;
            }

            if (to_run_count == 1) {
                // Single system: execute directly, avoid task_group overhead
                const auto idx = to_run_buf[0];
                LatencyProbe probe;
                const bool written = compute_systems_[idx].system->run(world_);
                const auto elapsed = probe.elapsed_ns();
                const auto writer  = compute_systems_[idx].system->meta().writes.size() > 0;

                if (!writer || (writer && written)) [[likely]] {
                    compute_stats_[idx].latency_hist.record(elapsed);
                    if (written) [[likely]] {
                        written_mask.fetch_or(1U << idx, std::memory_order_relaxed);
                    }
                }
            } else {
                // Multiple systems: execute in parallel
                tbb::task_group tg;
                for (std::size_t i = 0; i < to_run_count; ++i) {
                    const auto idx = to_run_buf[i];
                    tg.run([this, idx, &written_mask] {
                        LatencyProbe probe;
                        const bool written = compute_systems_[idx].system->run(world_);
                        const auto elapsed = probe.elapsed_ns();
                        const auto writer  = compute_systems_[idx].system->meta().writes.size() > 0;

                        if (!writer || (writer && written)) [[likely]] {
                            compute_stats_[idx].latency_hist.record(elapsed);
                            if (written) [[likely]] {
                                written_mask.fetch_or(1U << idx, std::memory_order_relaxed);
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

    // Enforce max 32 compute systems for bitmask scheduling
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

    std::flat_map<ChannelKey, ChannelUsage> channels;

    // First, collect fixed_rate systems' outputs as external sources
    for (const auto& entry : fixed_rate_systems_) {
        const auto& meta = entry.system->meta();

        for (const auto& ch : meta.spsc_channels) {
            if (is_writer(ch.kind)) {
                ChannelKey key{ch.type, ch.topic};
                auto& usage                 = channels[key];
                usage.kind                  = ch.kind;
                usage.has_fixed_rate_writer = true;
                usage.first_system          = meta.name;
            }
        }

        for (const auto& ch : meta.spmc_channels) {
            if (is_writer(ch.kind)) {
                ChannelKey key{ch.type, ch.topic};
                auto& usage                 = channels[key];
                usage.kind                  = ch.kind;
                usage.has_fixed_rate_writer = true;
                usage.first_system          = meta.name;
            }
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

    for (const auto& [key, usage] : channels) {
        // Count total writers (fixed_rate + compute)
        std::size_t total_writers = usage.writers.size() + (usage.has_fixed_rate_writer ? 1 : 0);

        // Multiple writers is always an error
        if (total_writers > 1) {
            std::vector<std::string> writer_names;
            if (usage.has_fixed_rate_writer) {
                writer_names.push_back(usage.first_system + " (fixed_rate)");
            }
            for (auto idx : usage.writers) {
                writer_names.push_back(compute_systems_[idx].system->meta().name);
            }
            return std::unexpected(
                MultipleWritersError{
                    .key     = ChannelKeyInfo(key.type, key.topic, is_spsc(usage.kind)),
                    .writers = std::move(writer_names),
                });
        }

        // SPSC with multiple readers is an error
        if (is_spsc(usage.kind) && usage.readers.size() > 1) {
            std::vector<std::string> reader_names;
            reader_names.reserve(usage.readers.size());
            for (auto idx : usage.readers) {
                reader_names.push_back(compute_systems_[idx].system->meta().name);
            }
            return std::unexpected(
                MultipleReadersError{
                    .key     = ChannelKeyInfo(key.type, key.topic, true),
                    .readers = std::move(reader_names),
                });
        }

        // Reader without writer is an error (neither fixed_rate nor compute writer)
        if (total_writers == 0 && !usage.readers.empty()) {
            std::vector<std::string> reader_names;
            reader_names.reserve(usage.readers.size());
            for (auto idx : usage.readers) {
                reader_names.push_back(compute_systems_[idx].system->meta().name);
            }
            return std::unexpected(
                OrphanedReaderError{
                    .key     = ChannelKeyInfo(key.type, key.topic, is_spsc(usage.kind)),
                    .readers = std::move(reader_names),
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
    // Phase 3: Build adjacency list and in-degree
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
                const std::uint64_t bit = 1U << reader;
                if ((adj_mask[writer] & bit) == 0) {
                    // New edge
                    adj_mask[writer] |= bit;
                    ++in_degree[reader];
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
        for (std::size_t node : current_level) {
            std::uint64_t neighbors = adj_mask[node];
            while (neighbors) {
                const std::size_t neighbor = __builtin_ctzll(neighbors);
                neighbors &= neighbors - 1; // Clear lowest set bit
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

            std::uint64_t neighbors = adj_mask[node];
            while (neighbors) {
                const std::size_t neighbor = __builtin_ctzll(neighbors);
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
    // Phase 6: Compute notify groups (which fixed_rate systems wake which compute systems)
    // ========================================================================
    // Build transitive closure of compute system dependencies
    // dependents[i] = bitmask of all systems that depend on system i (directly or transitively)
    std::vector<std::uint64_t> dependents = adj_mask; // Direct dependencies are already in adj_mask

    // Compute transitive closure (Floyd-Warshall style)
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            if (dependents[i] & (1U << k)) {
                dependents[i] |= dependents[k];
            }
        }
    }

    // For each fixed_rate system, compute which compute systems it affects
    fixed_rate_affects_.resize(fixed_rate_systems_.size(), 0);

    for (std::size_t i = 0; i < fixed_rate_systems_.size(); ++i) {
        std::uint64_t affects = 0;
        const auto& meta      = fixed_rate_systems_[i].system->meta();
        // Find compute systems that directly read from this fixed_rate's channels
        for (const auto& ch : meta.spsc_channels) {
            if (is_writer(ch.kind)) {
                ChannelKey key{ch.type, ch.topic};
                if (auto it = channels.find(key); it != channels.end()) {
                    for (auto reader : it->second.readers) {
                        affects |= (1U << reader);     // Direct reader
                        affects |= dependents[reader]; // Transitive dependents
                    }
                }
            }
        }

        for (const auto& ch : meta.spmc_channels) {
            if (is_writer(ch.kind)) {
                ChannelKey key{ch.type, ch.topic};
                if (auto it = channels.find(key); it != channels.end()) {
                    for (auto reader : it->second.readers) {
                        affects |= (1U << reader);
                        affects |= dependents[reader];
                    }
                }
            }
        }

        fixed_rate_affects_[i] = affects;
    }

    // For each compute system, compute which other compute systems it affects
    // This enables cascade wake: when compute A finishes, wake compute B if B depends on A
    compute_affects_.resize(n, 0);

    for (std::size_t i = 0; i < n; ++i) {
        // dependents[i] already contains all systems that depend on system i (transitively)
        compute_affects_[i] = dependents[i];
    }

    return {};
}

} // namespace talos
