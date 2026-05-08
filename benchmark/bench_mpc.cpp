// Rigorous dual-axis MPC benchmark: 2x TinyMPC (double/dynamic) vs
// DualSmallMpcSolver (float/fixed/batched).
//
// Build: cmake --build build --target bench_mpc
// Run:   ./build/bin/bench_mpc

#include "dual_small_mpc_solver.hpp"
#include "tiny_api.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
}

[[nodiscard]] static RefTrajectory ref_chirp(int N, double amp) {
    Eigen::MatrixXd ref(2, N);
    const double f0 = 0.5;
    const double f1 = 3.0;
    const double T  = static_cast<double>(N) * kDt;
    for (int i = 0; i < N; ++i) {
        const double t      = static_cast<double>(i - kHorizonBack) * kDt;
        const double t_norm = (t + static_cast<double>(kHorizonBack) * kDt) / T;
        const double freq   = f0 + (f1 - f0) * t_norm;
        const double phase =
            2.0 * std::numbers::pi_v<double> * (f0 * t + 0.5 * (f1 - f0) / T * t * t);
        ref(0, i) = amp * std::sin(phase);
        ref(1, i) = amp * std::cos(phase) * (2.0 * std::numbers::pi_v<double> * freq);
    }
    return ref;
}
}

struct TinyPairSetup {
    TinySolverPtr yaw;
    TinySolverPtr pitch;
    tiny_setup(&raw, A, B, f, Q, R, kRho, 2, 1, kHorizon, 0);
    TinySolverPtr solver(raw);
    auto* s                  = solver.get();
    s->settings->abs_pri_tol = kAbsTol;
    s->settings->abs_dua_tol = kAbsTol;
    Eigen::MatrixXd x_min    = Eigen::MatrixXd::Constant(2, kHorizon, -1e17);
    Eigen::MatrixXd x_max    = Eigen::MatrixXd::Constant(2, kHorizon, 1e17);
    if (is_pitch) {
        x_min.row(0).setConstant(-0.698);
        x_max.row(0).setConstant(0.698);
    }
    tiny_set_bound_constraints(
        s, x_min, x_max, Eigen::MatrixXd::Constant(1, kHorizon - 1, -p.max_acc),
        Eigen::MatrixXd::Constant(1, kHorizon - 1, p.max_acc));
    return solver;
}
}
;

struct DualSetup {
    DualSmallMpcSolver solver;

            static_cast<float>(kDt), kHorizon, static_cast<float>(kRho), yaw_cfg, pitch_cfg);
            assert(result.has_value());
            solver = std::move(*result);
            solver.set_settings(static_cast<float>(kAbsTol), 1000);
} [[nodiscard]] static Stats compute_stats(std::vector<double> samples, int converged) {
    assert(!samples.empty());
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (double s : samples)
        sum += s;
    const double mean = sum / static_cast<double>(samples.size());
    double sq_sum     = 0.0;
    for (double s : samples) {
        const double d = s - mean;
        sq_sum += d * d;
    }
    return {
        samples.front(), samples[samples.size() / 2],
            samples[static_cast<size_t>(static_cast<double>(samples.size()) * 0.95)],
            samples.back(), mean, std::sqrt(sq_sum / static_cast<double>(samples.size())),
            std::vector<double> times;
        times.reserve(trials);
        int converged  = 0;
        int iterations = 0;

        for (int i = 0; i < trials; ++i) {
            tiny_set_x0(yaw, yaw_ref.col(0));
            const auto t0 = Clock::now();
            compiler_barrier();
            const int yaw_status   = tiny_solve(yaw);
            const int pitch_status = tiny_solve(pitch);
            compiler_barrier();
            const auto t1 = Clock::now();
            compiler_barrier();
            do_not_optimize(yaw_status);
            do_not_optimize(pitch_status);
            if (yaw_status == 0 && pitch_status == 0 && yaw->solution->solved == 1
                && pitch->solution->solved == 1) {
                ++converged;
            }
            iterations = std::max(yaw->solution->iter, pitch->solution->iter);
            times.push_back(elapsed_us(t0, t1));
        }

        return {compute_stats(std::move(times), converged), iterations};
    }

    [[nodiscard]] static TimingResult bench_dual_solve(int max_iter, int trials) {
        setup.solver.set_settings(static_cast<float>(kAbsTol), max_iter);
        const Eigen::MatrixXf yaw_ref   = yaw_ref_d.cast<float>();
        const Eigen::MatrixXf pitch_ref = pitch_ref_d.cast<float>();

        std::vector<double> times;
        times.reserve(trials);
        int converged  = 0;
        int iterations = 0;

        for (int i = 0; i < trials; ++i) {
            setup.solver.set_x0(
                yaw_ref_d(0, 0), yaw_ref_d(1, 0), pitch_ref_d(0, 0), pitch_ref_d(1, 0));
            const auto t0 = Clock::now();
            compiler_barrier();
            const bool ok = setup.solver.solve();
            compiler_barrier();
            const auto t1 = Clock::now();
            compiler_barrier();

            do_not_optimize(ok);
            if (ok)
                ++converged;
            iterations = setup.solver.iterations();
            times.push_back(elapsed_us(t0, t1));
        }

        return {compute_stats(std::move(times), converged), iterations};
    }

[[nodiscard]] static Stats bench_tiny_pair_pipeline(
    std::vector<double> times;
    times.reserve(trials);
    int converged = 0;

    for (int i = 0; i < trials; ++i) {
        ++converged;
        }
        times.push_back(elapsed_us(t0, t1));
}

    int max_iter, int trials) {
        setup.solver.set_settings(static_cast<float>(kAbsTol), max_iter);
        const Eigen::MatrixXf yaw_ref   = yaw_ref_d.cast<float>();
        const Eigen::MatrixXf pitch_ref = pitch_ref_d.cast<float>();

        std::vector<double> times;
        times.reserve(trials);
        int converged = 0;

        for (int i = 0; i < trials; ++i) {
            if (ok)
                ++converged;
            times.push_back(elapsed_us(t0, t1));
        }
    }

    struct AccuracyReport {
        double max_abs_state_err;
        double rmse_state;
        double max_abs_input_err;
        double center_yaw_pos_err;
        double center_pitch_pos_err;
        bool tiny_converged;
        bool dual_converged;
    };

    [[nodiscard]] static AccuracyReport compare_accuracy(
        const Eigen::MatrixXd& yaw_ref, const Eigen::MatrixXd& pitch_ref, int max_iter) {
        TinyPairSetup tiny;
        DualSetup dual;
        double sum_sq    = 0.0;
        double max_input = 0.0;
        for (int k = 0; k < kHorizon; ++k) {
            const double yaw_pos_err = std::abs(
                yaw->solution->x(0, k) pitch->solution->x(1, k)
                - static_cast<double>(dual.solver.state(DualSmallMpcSolver::kPitchAxis, 1, k)));
            max_state =
                std::max({max_state, yaw_pos_err, yaw_vel_err, pitch_pos_err, pitch_vel_err});
            sum_sq += yaw_pos_err * yaw_pos_err + yaw_vel_err * yaw_vel_err;
            sum_sq += pitch_pos_err * pitch_pos_err + pitch_vel_err * pitch_vel_err;
        }
        for (int k = 0; k < kHorizon - 1; ++k) {
            max_input = std::max(
                max_input,
                std::max(
                    std::abs(
                        yaw->solution->u(0, k)
                        - static_cast<double>(dual.solver.input(DualSmallMpcSolver::kYawAxis, k))),
                    std::abs(
                        pitch->solution->u(0, k)
                        - static_cast<double>(
                            dual.solver.input(DualSmallMpcSolver::kPitchAxis, k)))));
        }

        return {
            std::sqrt(sum_sq / static_cast<double>(4 * kHorizon)), max_input,
                std::abs(
                    yaw->solution->x(0, kHorizonBack)
                    - static_cast<double>(
                        dual.solver.state(DualSmallMpcSolver::kYawAxis, 0, kHorizonBack))),
                std::abs(
                    pitch->solution->x(0, kHorizonBack)
                    - static_cast<double>(
                        dual.solver.state(DualSmallMpcSolver::kPitchAxis, 0, kHorizonBack))),
                yaw_status == 0 && pitch_status == 0 && yaw->solution->solved == 1
                    && pitch->solution->solved == 1,
                dual_ok, const auto yaw_ref_chirp = ref_chirp(kHorizon, 0.3);
            const auto pitch_ref_noisy = ref_noisy_sinusoid(kHorizon, 42, 0.2);

            print_section("Dual-Axis MPC Benchmark");
            std::cout << "  Horizon: " << kHorizon << " (ahead=" << kHorizonAhead {
                print_section("Solve-Only Timing");
                TinyPairSetup tiny;
                DualSetup dual;
                const auto tiny_t =
                    bench_tiny_pair_solve(tiny, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
                const auto dual_t = bench_dual_solve(dual, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
                print_stats("2x TinyMPC", tiny_t.stats, kTrials);
                print_stats("DualSmallMpcSolver", dual_t.stats, kTrials);
                std::cout << "  speedup=" << std::fixed << std::setprecision(2)
                          << tiny_t.stats.p50 / std::max(dual_t.stats.p50, 0.001) << "x\n";
            }
            {
                print_section("Full Pipeline Timing");
                TinyPairSetup tiny;
                DualSetup dual;
                const auto tiny_t =
                    bench_tiny_pair_pipeline(tiny, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
                const auto dual_t =
                    bench_dual_pipeline(dual, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
                print_stats("2x TinyMPC", tiny_t, kTrials);
                print_stats("DualSmallMpcSolver", dual_t, kTrials);
                std::cout << "  speedup=" << std::fixed << std::setprecision(2)
                          << tiny_t.p50 / std::max(dual_t.p50, 0.001) << "x\n";
            }

            {
                print_section("Scenario Sweep");
                struct Scenario {
                    const char* name;
                    const Eigen::MatrixXd& yaw_ref;
                    const Eigen::MatrixXd& pitch_ref;
                };
                const Scenario scenarios[] = {
                    {    "sin/sin",   yaw_ref_sin,   pitch_ref_sin},
                    {"chirp/noisy", yaw_ref_chirp, pitch_ref_noisy},
                };

                std::cout << "  " << std::left << std::setw(16) << "Scenario" << std::setw(14)
                          << "Tiny p50" << std::setw(14) << "Dual p50" << std::setw(10) << "Speedup"
                          << "\n";
                std::cout << "  " << std::string(54, '-') << "\n";
                for (const auto& scenario : scenarios) {
                    TinyPairSetup tiny;
                    DualSetup dual;
                    const auto tiny_t = bench_tiny_pair_solve(
                        tiny, scenario.yaw_ref, scenario.pitch_ref, 10, kTrials);
                    const auto dual_t =
                        bench_dual_solve(dual, scenario.yaw_ref, scenario.pitch_ref, 10, kTrials);
                    std::cout << "  " << std::left << std::setw(16) << scenario.name << std::fixed
                              << std::setprecision(2) << std::setw(14) << tiny_t.stats.p50
                              << std::setw(14) << dual_t.stats.p50
                              << std::setw(10)

                    {
                        print_section("Numerical Accuracy");
                        const auto acc = compare_accuracy(yaw_ref_sin, pitch_ref_noisy, 100);
                        std::cout << "  tiny_ok=" << (acc.tiny_converged ? "Y" : "N")
                                  << " dual_ok=" << (acc.dual_converged ? "Y" : "N") << "\n";
                        std::cout << "  max_state_err=" << std::scientific << std::setprecision(4)
                                  << acc.max_abs_state_err << " rmse_state=" << acc.rmse_state
                                  << " max_input_err=" << acc.max_abs_input_err << "\n";
                        std::cout << "  center_yaw_err=" << acc.center_yaw_pos_err
                                  << " center_pitch_err=" << acc.center_pitch_pos_err << "\n";
                    }

                    return 0;
                    sigma_r0 = 1e-5 sigma_r1 = 1e-5 sigma_h =
                        1e-2 #Aligned with at_vision(was 0.5) meas_yaw_var_floor =
                            4e-3 meas_pitch_var_floor = 4e-3 meas_dist_var_floor = 0.3
#pragma once

                    /**
                     * @file
                     * @brief Lock-free single-producer multiple-consumer triple buffer
                     *
                     * This module provides a lock-free SPMC (Single Producer Multiple Consumers)
                     * triple buffer implementation optimized for low-latency data transfer
                     * between threads. It also keeps a bounded backlog of recent writes so active
                     * readers can catch up on short bursts instead of immediately jumping to the
                     * latest generation.
                     *
                     * ## Design
                     *
                     * - **RWSpinLock**: Readers don't block each other, only writer blocks
                     * - **Readers**: Acquire shared lock (atomic increment)
                     * - **Writer**: Acquires exclusive lock (waits for readers to drain)
                     *
                     * ## Performance characteristics
                     *
                     * - **Write**: ~50-100ns (wait for readers, then atomic swap)
                     * - **Read**: ~20-40ns (atomic increment/decrement + copy)
                     * - **Readers NEVER block each other** (unlike naive spinlock)
                     *
                     * ## Thread safety
                     *
                     * All operations are thread-safe. Multiple readers can read concurrently
                     * without blocking each other.
                     */

#include "spin.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <utility>

                        namespace talos::primitive {

                        /**
                         * @brief Lock-free single-producer multiple-consumer triple buffer
                         *
                         * A triple buffer provides thread-safe data transfer from one writer
                         * to multiple readers. Each reader operates independently and sees
                         * a consistent view of the data. Active readers can drain a small backlog
                         * of retained generations; readers that fall farther behind skip to the
                         * oldest retained generation.
                         *
                         * ## Memory model
                         *
                         * The buffer stores `shared_ptr<const T>`, allowing zero-copy transfers
                         * when the producer and consumers share ownership of the underlying data.
                         *
                         * ## Usage
                         *
                         * ```cpp
                         * auto [writer, reader] = SpmcTripleBuffer<Frame>::create();
                         *
                         * // Producer thread:
                         * writer.write(Frame{...});
                         *
                         * // Consumer threads (each has their own reader):
                         * auto reader2 = reader.clone();
                         * if (auto frame = reader2.read()) {
                         *     // process frame
                         * }
                         * ```
                         *
                         * ## Thread safety
                         *
                         * - Single writer, multiple readers
                         * - Writers block until all readers release their locks
                         * - Readers never block each other
                         *
                         * ## Performance
                         *
                         * - Write: O(1) atomic operations
                         * - Read: O(backlog_capacity) bounded scan
                         * - Memory: backlog_capacity * sizeof(shared_ptr) + atomic overhead
                         */
                        template <typename T>
                        requires(std::movable<T> && std::copyable<T>) class SpmcTripleBuffer {
                        public:
                            static constexpr std::size_t backlog_capacity = 64;

                        private:
                            /**
                             * @brief Reader-writer spinlock optimized for many readers
                             *
                             * ## Encoding
                             *
                             * - Bit 31: writer waiting/active flag
                             * - Bits 0-30: current reader count
                             *
                             * ## Algorithm
                             *
                             * - Readers increment the counter (shared lock)
                             * - Writers set the writer bit and wait for counter to reach 0
                             */
                            class RWSpinLock {
                                static constexpr std::uint32_t WRITER_BIT  = 0x80000000u;
                                static constexpr std::uint32_t READER_MASK = 0x7FFFFFFFu;
                                std::atomic<std::uint32_t> state_{0};

                            public:
                                /**
                                 * @brief Acquire shared lock for reading
                                 *
                                 * Spins until the writer bit is clear, then atomically
                                 * increments the reader count.
                                 *
                                 * ## Thread safety
                                 *
                                 * Safe to call from multiple reader threads concurrently.
                                 */
                                void read_lock() noexcept {
                                    while (true) {
                                        auto s = state_.load(std::memory_order_relaxed);
                                        // If writer is waiting/active, spin
                                        if (s & WRITER_BIT) {
                                            SPIN_HINT();
                                            continue;
                                        }
                                        // Try to increment reader count
                                        if (state_.compare_exchange_weak(
                                                s, s + 1, std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
                                            return;
                                        }
                                    }
                                }

                                void read_unlock() noexcept {
                                    state_.fetch_sub(1, std::memory_order_release);
                                }

                                void write_lock() noexcept {
                                    // Set writer bit
                                    while (true) {
                                        auto s = state_.load(std::memory_order_relaxed);
                                        if (s & WRITER_BIT) {
                                            // Another writer waiting
                                            SPIN_HINT();
                                            continue;
                                        }
                                        if (state_.compare_exchange_weak(
                                                s, s | WRITER_BIT, std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
                                            break;
                                        }
                                    }
                                    // Wait for readers to drain
                                    while ((state_.load(std::memory_order_acquire) & READER_MASK)
                                           != 0) {
                                        SPIN_HINT();
                                    }
                                }

                                void write_unlock() noexcept {
                                    state_.fetch_and(~WRITER_BIT, std::memory_order_release);
                                }
                            };

                            struct State {
                                struct Slot {
                                    std::shared_ptr<const T> value{nullptr};
                                    std::uint64_t generation{0};
                                };

                                alignas(std::hardware_destructive_interference_size)
                                    RWSpinLock lock{};
                                std::shared_ptr<const T> current{nullptr};
                                std::array<Slot, backlog_capacity> backlog{};
                                std::size_t write_index{0};
                                alignas(std::hardware_destructive_interference_size)
                                    std::atomic<std::uint64_t> generation{0};

                                [[nodiscard]] static std::uint64_t oldest_retained_generation(
                                    const std::uint64_t latest_generation) noexcept {
                                    if (latest_generation == 0) {
                                        return 0;
                                    }

                                    const auto retained = std::min<std::uint64_t>(
                                        latest_generation, backlog_capacity);
                                    return latest_generation - retained + 1;
                                }
                            };

                        public:
                            template <typename U>
                            class Write;
                            template <typename U>
                            class Read;

                            /**
                             * @brief Write handle for the producer
                             *
                             * Owned by the single producer thread. Provides thread-safe
                             * write access to the buffer.
                             */
                            template <typename U>
                            class Write {
                            public:
                                Write(const Write&)                = delete;
                                Write& operator=(const Write&)     = delete;
                                Write(Write&&) noexcept            = default;
                                Write& operator=(Write&&) noexcept = default;

                                /**
                                 * @brief Write data to the buffer
                                 *
                                 * Atomically publishes new data for all readers.
                                 *
                                 * ## Performance
                                 *
                                 * May spin briefly waiting for readers to release their locks.
                                 * The write operation itself is O(1).
                                 *
                                 * ## Thread safety
                                 *
                                 * Must only be called from the single producer thread.
                                 *
                                 * ## Panics
                                 *
                                 * Never panics. The operation is guaranteed to complete.
                                 */
                                void write(U data) noexcept {
                                    auto ptr = std::make_shared<const U>(std::move(data));

                                    state_->lock.write_lock();
                                    const auto next_generation =
                                        state_->generation.load(std::memory_order_relaxed) + 1;
                                    auto& slot      = state_->backlog[state_->write_index];
                                    slot.value      = ptr;
                                    slot.generation = next_generation;
                                    state_->current = std::move(ptr);
                                    state_->write_index =
                                        (state_->write_index + 1) % backlog_capacity;
                                    state_->generation.store(
                                        next_generation, std::memory_order_release);
                                    state_->lock.write_unlock();
                                }

                                [[nodiscard]] std::uint64_t generation() const noexcept {
                                    return state_->generation.load(std::memory_order_acquire);
                                }

                                [[nodiscard]] bool valid() const noexcept {
                                    return state_ != nullptr;
                                }
                                explicit operator bool() const noexcept { return valid(); }

                            private:
                                friend class SpmcTripleBuffer;
                                explicit Write(std::shared_ptr<State> s)
                                    : state_(std::move(s)) {}

                                std::shared_ptr<State> state_;
                            };

                            /**
                             * @brief Read handle for consumers
                             *
                             * Copyable and movable. Each reader maintains its own generation
                             * counter to track which data it has already seen.
                             *
                             * ## Thread safety
                             *
                             * Safe to use from any thread. Multiple readers can read
                             * concurrently without blocking each other.
                             */
                            template <typename U>
                            class Read {
                            public:
                                Read(const Read& other)
                                    : state_(other.state_)
                                    , last_gen_(other.last_gen_) {}

                                Read& operator=(const Read& other) {
                                    if (this != &other) {
                                        state_    = other.state_;
                                        last_gen_ = other.last_gen_;
                                    }
                                    return *this;
                                }

                                Read(Read&&) noexcept            = default;
                                Read& operator=(Read&&) noexcept = default;

                                /**
                                 * @brief Clone this reader
                                 *
                                 * Creates a new reader that starts with the same generation
                                 * counter as this reader.
                                 *
                                 * ## Performance
                                 *
                                 * O(1) - just copies the shared state pointer.
                                 */
                                [[nodiscard]] Read clone() const { return *this; }

                                /**
                                 * @brief Read new data if available
                                 *
                                 * Returns new data if it hasn't been seen by this reader,
                                 * otherwise returns `std::nullopt`.
                                 *
                                 * ## Algorithm
                                 *
                                 * 1. Fast path: check generation counter without locking
                                 * 2. Slow path: acquire read lock and find the next retained
                                 * generation
                                 * 3. Copy data out if new
                                 *
                                 * ## Thread safety
                                 *
                                 * Safe to call from any thread.
                                 *
                                 * ## Returns
                                 *
                                 * - `std::optional<U>` containing new data if available
                                 * - `std::nullopt` if no new data since last read
                                 */
                                [[nodiscard]] std::optional<U> read() noexcept {
                                    // Fast path: check generation without lock
                                    if (const std::uint64_t gen =
                                            state_->generation.load(std::memory_order_acquire);
                                        gen <= last_gen_) {
                                        return std::nullopt;
                                    }

                                    // Acquire read lock (doesn't block other readers)
                                    state_->lock.read_lock();

                                    const std::uint64_t locked_gen =
                                        state_->generation.load(std::memory_order_acquire);
                                    std::uint64_t target_gen = locked_gen;

                                    if (last_gen_ != 0 && locked_gen > last_gen_) {
                                        const auto oldest_gen =
                                            State::oldest_retained_generation(locked_gen);
                                        target_gen = std::max(last_gen_ + 1, oldest_gen);
                                    }

                                    std::shared_ptr<const U> ptr{nullptr};
                                    for (const auto& slot : state_->backlog) {
                                        if (slot.generation == target_gen) {
                                            ptr = slot.value;
                                            break;
                                        }
                                    }
                                    if (!ptr) [[unlikely]] {
                                        ptr        = state_->current;
                                        target_gen = locked_gen;
                                    }

                                    state_->lock.read_unlock();

                                    if (!ptr) {
                                        return std::nullopt;
                                    }

                                    // Check generation again (might have been stale in fast path)
                                    if (target_gen <= last_gen_) {
                                        return std::nullopt;
                                    }

                                    last_gen_ = target_gen;
                                    return *ptr; // copy out
                                }

                                [[nodiscard]] std::optional<U> read_current() noexcept {
                                    state_->lock.read_lock();

                                    std::shared_ptr<const U> ptr = state_->current;
                                    std::uint64_t locked_gen =
                                        state_->generation.load(std::memory_order_acquire);

                                    state_->lock.read_unlock();

                                    if (!ptr) {
                                        return std::nullopt;
                                    }

                                    last_gen_ = locked_gen;
                                    return *ptr;
                                }

                                [[nodiscard]] bool has_new() const noexcept {
                                    return state_->generation.load(std::memory_order_acquire)
                                         > last_gen_;
                                }

                                [[nodiscard]] std::uint64_t last_generation() const noexcept {
                                    return last_gen_;
                                }

                                [[nodiscard]] bool valid() const noexcept {
                                    return state_ != nullptr;
                                }
                                explicit operator bool() const noexcept { return valid(); }

                            private:
                                friend class SpmcTripleBuffer;
                                explicit Read(std::shared_ptr<State> s)
                                    : state_(std::move(s)) {}

                                std::shared_ptr<State> state_;
                                std::uint64_t last_gen_{0};
                            };

                            SpmcTripleBuffer()                                   = delete;
                            SpmcTripleBuffer(const SpmcTripleBuffer&)            = delete;
                            SpmcTripleBuffer& operator=(const SpmcTripleBuffer&) = delete;

                            [[nodiscard]] static std::pair<Write<T>, Read<T>> create() {
                                auto state = std::make_shared<State>();
                                return {Write<T>(state), Read<T>(state)};
                            }
                        };

                    } // namespace talos::primitive

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>

                    /**
                     * @file
                     * @brief Lock-free single-producer single-consumer triple buffer
                     *
                     * This module provides a lock-free SPSC (Single Producer Single Consumer)
                     * triple buffer optimized for minimal latency in a two-thread pipeline.
                     *
                     * ## Design
                     *
                     * - **Three-slot design**: write slot, read slot, and intermediate slot
                     * - **State machine**: uses bit flags to coordinate handoff
                     * - **Move semantics**: data is moved in and out for zero-copy transfer
                     *
                     * ## Memory ordering
                     *
                     * Uses `memory_order_acq_rel` for publish operations to ensure
                     * visibility of the written data before the consumer sees the flag.
                     *
                     * ## Thread safety
                     *
                     * - **Single writer, single reader only**
                     * - No locking required
                     * - Data race free when used as designed
                     */
                    namespace talos::primitive {

                    /**
                     * @brief Lock-free single-producer single-consumer triple buffer
                     *
                     * A two-thread communication channel with minimal latency.
                     * Uses three slots and a state machine to coordinate handoff.
                     *
                     * ## Usage
                     *
                     * ```cpp
                     * auto [writer, reader] = SpscTripleBuffer<Frame>::create();
                     *
                     * // Producer:
                     * writer.write(Frame{...});
                     *
                     * // Consumer:
                     * if (auto frame = reader.read()) {
                     *     // process frame
                     * }
                     * ```
                     *
                     * ## Performance
                     *
                     * - **Write**: O(1) atomic operations (~20-30ns)
                     * - **Read**: O(1) atomic operations (~20-30ns)
                     * - **Zero-copy**: data is moved, not copied
                     */
                    template <typename T>
                    requires(std::movable<T>) class SpscTripleBuffer {
                        static constexpr uint8_t FLAG_NEW   = 0x80;
                        static constexpr uint8_t INDEX_MASK = 0x03;

                        struct State {
                            alignas(std::hardware_destructive_interference_size)
                                std::array<T, 3> slots{};
                            alignas(std::hardware_destructive_interference_size) uint8_t write_idx{
                                0};
                            alignas(std::hardware_destructive_interference_size)
                                std::atomic<uint8_t> shared{1};
                            alignas(std::hardware_destructive_interference_size) uint8_t read_idx{
                                2};
                        };

                    public:
                        class Read;

                        /**
                         * @brief Write handle for the producer
                         *
                         * Move-only. Owned by the single producer thread.
                         */
                        template <typename U>
                        class Write {
                        public:
                            Write& operator=(Write&&) noexcept = default;

                            /**
                             * @brief Write data by moving it into the buffer
                             *
                             * The data is moved into the write slot, then published
                             * to make it visible to the consumer.
                             *
                             * ## Thread safety
                             *
                             * Must only be called from the single producer thread.
                             */
                            void write(U data) noexcept {
                                state_->slots[state_->write_idx] = std::move(data);
                                publish();
                            }

                            /**
                             * @brief Borrow mutable reference to write slot
                             *
                             * Allows direct construction of data in the buffer,
                             * avoiding a move operation.
                             *
                             * ## Thread safety
                             *
                             * Must only be called from the producer thread.
                             *
                             * ## Returns
                             *
                             * Reference to the write slot. The caller is responsible
                             * for ensuring the data is properly initialized before
                             * calling `publish()`.
                             */
                            [[nodiscard]] U& borrow_mut() noexcept {
                                return state_->slots[state_->write_idx];
                            }

                            /**
                             * @brief Publish the current write slot
                             *
                             * Atomically signals to the consumer that new data is available.
                             *
                             * ## Thread safety
                             *
                             * Must only be called from the producer thread.
                             */
                            void publish() noexcept {
                                const auto old = state_->shared.exchange(
                                    state_->write_idx | FLAG_NEW, std::memory_order_acq_rel);
                                state_->write_idx = old & INDEX_MASK;
                            }

                            [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
                            explicit Write(std::shared_ptr<State> s)
                                : state_(std::move(s)) {}
                            std::shared_ptr<State> state_;
                        };

                        /**
                         * @brief Read handle for the consumer
                         *
                         * Move-only. Owned by the single consumer thread.
                         */
                        template <typename U>
                        class Read {
                        public:
                            Read& operator=(Read&&) noexcept = default;

                            /**
                             * @brief Try to read new data
                             *
                             * Attempts to read data from the buffer. If new data is
                             * available (since the last read), it is moved out and
                             * returned.
                             *
                             * ## Thread safety
                             *
                             * Must only be called from the single consumer thread.
                             *
                             * ## Returns
                             *
                             * - `std::optional<U>` containing the data if new data is available
                             * - `std::nullopt` if no new data since last read
                             */
                            [[nodiscard]] std::optional<U> read() noexcept {
                                auto expected = state_->shared.load(std::memory_order_acquire);
                                if (!(expected & FLAG_NEW)) {
                                    return std::nullopt;
                                }

                                auto ready_idx = expected & INDEX_MASK;
                                if (!state_->shared.compare_exchange_strong(
                                        expected, state_->read_idx, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
                                    if (!(expected & FLAG_NEW)) {
                                        return std::nullopt;
                                    }
                                    ready_idx = expected & INDEX_MASK;
                                    if (!state_->shared.compare_exchange_strong(
                                            expected, state_->read_idx, std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
                                        return std::nullopt;
                                    }
                                }

                                state_->read_idx = ready_idx;
                                return std::move(state_->slots[state_->read_idx]);
                            }

                            // has_new() removed - TOCTOU hazard makes it fundamentally unsafe.
                            // Use read() and check the return value instead.

                            // current() removed - UAB hazard (reference may point to slot being
                            // written). Use read() for ownership transfer instead.

                            [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
                            explicit operator bool() const noexcept { return valid(); }

                        private:
                            friend class SpscTripleBuffer;
                            explicit Read(std::shared_ptr<State> s)
                                : state_(std::move(s)) {}
                            std::shared_ptr<State> state_;
                        };

                        SpscTripleBuffer()                                   = delete;
                        SpscTripleBuffer& operator=(const SpscTripleBuffer&) = delete;

                        /**
                         * @brief Create buffer, returns {Write<T>, Read<T>}
                         */
                        [[nodiscard]] static std::pair<Write<T>, Read<T>> create() {
                            auto state = std::make_shared<State>();
                            return {Write<T>(state), Read<T>(state)};
                        }

                        TEST(SpscTest, BorrowMut) { auto [w, r] = SpscTripleBuffer<int>::create(); }

                        TEST(SpmcTest, IndependentGeneration) {
                            auto [w, r] = SpmcTripleBuffer<int>::create();

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <limits>
#include <string>

                            namespace fcs::L5 {

                            /// Fixed-capacity two-axis MPC solver for the yaw/pitch production
                            /// path.
                            ///
                            /// The hot path packs both axes into fixed-size Eigen lanes while
                            /// preserving the external interface and per-axis numerical behavior.
                            class DualSmallMpcSolver {
                            public:
                                struct AxisConfig {
                                    float q_pos{9e6f};
                                    float q_vel{0.0f};
                                    float r{1.0f};
                                    float max_acc{50.0f};
                                    float state_min{-std::numeric_limits<float>::infinity()};
                                    float state_max{std::numeric_limits<float>::infinity()};
                                    bool enable_state_bound{false};
                                };

                                static constexpr int kAxisCount       = 2;
                                static constexpr int kYawAxis         = 0;
                                static constexpr int kPitchAxis       = 1;
                                static constexpr int kMaxHorizon      = 101;
                                static constexpr int kMaxInputHorizon = kMaxHorizon - 1;

                                DualSmallMpcSolver() = default;

                                DualSmallMpcSolver(const DualSmallMpcSolver&)            = delete;
                                DualSmallMpcSolver& operator=(const DualSmallMpcSolver&) = delete;
                                DualSmallMpcSolver(DualSmallMpcSolver&&) noexcept        = default;
                                DualSmallMpcSolver&
                                    operator=(DualSmallMpcSolver&&) noexcept = default;
                                ~DualSmallMpcSolver()                        = default;

                                [[nodiscard]] static std::expected<DualSmallMpcSolver, std::string>
                                    create(
                                        float dt, int horizon, float rho, const AxisConfig& yaw,
                                        const AxisConfig& pitch) {
                                    if (dt <= 0.0f) {
                                        return std::unexpected("dt must be positive");
                                    }
                                    if (horizon < 2) {
                                        return std::unexpected("horizon must be >= 2");
                                    }
                                    if (horizon > kMaxHorizon) {
                                        return std::unexpected(
                                            "horizon exceeds fixed-capacity limit of "
                                            + std::to_string(kMaxHorizon));
                                    }

                                    DualSmallMpcSolver s;
                                    s.N_                = horizon;
                                    s.last_state_index_ = horizon - 1;
                                    s.last_input_index_ = horizon - 2;
                                    s.rho_              = rho;

                                    s.a00_   = 1.0f;
                                    s.a01_   = dt;
                                    s.a10_   = 0.0f;
                                    s.a11_   = 1.0f;
                                    s.b0_    = 0.0f;
                                    s.b1_    = dt;
                                    s.f_pos_ = 0.0f;
                                    s.f_vel_ = 0.0f;

                                    auto init = [&](int axis, const AxisConfig& cfg)
                                        -> std::expected<void, std::string> {
                                        if (!s.init_axis(axis, cfg)) {
                                            return std::unexpected(
                                                "failed to initialize batched MPC axis");
                                        }
                                        return {};
                                    };
                                    if (!init(kYawAxis, yaw)) {
                                        return std::unexpected("failed to initialize yaw axis");
                                    }
                                    if (!init(kPitchAxis, pitch)) {
                                        return std::unexpected("failed to initialize pitch axis");
                                    }

                                    s.reset_workspace();
                                    return s;
                                }

                                void set_settings(float abs_tol, int max_iter) noexcept {
                                    abs_tol_  = abs_tol;
                                    max_iter_ = max_iter;
                                }

                                void set_x0(int axis, double pos, double vel) noexcept {
                                    x_pos_(axis, 0) = static_cast<float>(pos);
                                    x_vel_(axis, 0) = static_cast<float>(vel);
                                }

                                void set_x0(
                                    double yaw_pos, double yaw_vel, double pitch_pos,
                                    double pitch_vel) noexcept {
                                    set_x0(kYawAxis, yaw_pos, yaw_vel);
                                    set_x0(kPitchAxis, pitch_pos, pitch_vel);
                                }

                                void set_ref_col(int axis, int k, float pos, float vel) noexcept {
                                    qref_pos_(axis, k) = -(q_pos_(axis) + rho_) * pos;
                                    qref_vel_(axis, k) = -(q_vel_(axis) + rho_) * vel;
                                    if (k == last_state_index_) {
                                        update_terminal_cache(axis, pos, vel);
                                    }
                                }

                                void set_ref_col(
                                    int k, float yaw_pos, float yaw_vel, float pitch_pos,
                                    float pitch_vel) noexcept {
                                    qref_pos_.col(k) << -(q_pos_(kYawAxis) + rho_) * yaw_pos,
                                        -(q_pos_(kPitchAxis) + rho_) * pitch_pos;
                                    qref_vel_.col(k) << -(q_vel_(kYawAxis) + rho_) * yaw_vel,
                                        -(q_vel_(kPitchAxis) + rho_) * pitch_vel;
                                    if (k == last_state_index_) {
                                        update_terminal_cache(kYawAxis, yaw_pos, yaw_vel);
                                        update_terminal_cache(kPitchAxis, pitch_pos, pitch_vel);
                                    }
                                }

                                void set_reference(
                                    const Eigen::Ref<const Eigen::MatrixXf>& yaw_ref,
                                    const Eigen::Ref<const Eigen::MatrixXf>& pitch_ref) noexcept {
                                    qref_pos_.row(kYawAxis).head(N_) =
                                        (-(q_pos_(kYawAxis) + rho_) * yaw_ref.row(0).array())
                                            .matrix();
                                    qref_vel_.row(kYawAxis).head(N_) =
                                        (-(q_vel_(kYawAxis) + rho_) * yaw_ref.row(1).array())
                                            .matrix();
                                    qref_pos_.row(kPitchAxis).head(N_) =
                                        (-(q_pos_(kPitchAxis) + rho_) * pitch_ref.row(0).array())
                                            .matrix();
                                    qref_vel_.row(kPitchAxis).head(N_) =
                                        (-(q_vel_(kPitchAxis) + rho_) * pitch_ref.row(1).array())
                                            .matrix();

                                    update_terminal_cache(
                                        kYawAxis, yaw_ref(0, last_state_index_),
                                        yaw_ref(1, last_state_index_));
                                    update_terminal_cache(
                                        kPitchAxis, pitch_ref(0, last_state_index_),
                                        pitch_ref(1, last_state_index_));
                                }

                                bool solve() noexcept {
                                    converged_ = false;
                                    converged_axis_.fill(false);
                                    iters_ = 0;

                                    for (int iter = 0; iter < max_iter_; ++iter) {
                                        const LaneMask active = active_mask();
                                        if (!active.any()) {
                                            converged_ = true;
                                            return true;
                                        }

                                        backward_terminal(active);
                                        for (int i = last_input_index_; i >= 0; --i) {
                                            backward_step(i, active);
                                        }

                                        Lane pri_s = Lane::Zero();
                                        Lane pri_u = Lane::Zero();
                                        Lane dua_s = Lane::Zero();
                                        Lane dua_u = Lane::Zero();
                                        Lane x_pos = x_pos_.col(0);
                                        Lane x_vel = x_vel_.col(0);

                                        for (int i = 0; i <= last_input_index_; ++i) {
                                            forward_step(
                                                i, active, x_pos, x_vel, pri_s, pri_u, dua_s,
                                                dua_u);
                                        }
                                        terminal_step(active, x_pos, x_vel, pri_s, dua_s);

                                        ++iters_;

                                        if ((iters_ % check_term_) == 0) {
                                            const Lane dua_s_scaled = Lane::Constant(rho_) * dua_s;
                                            const Lane dua_u_scaled = Lane::Constant(rho_) * dua_u;
                                            for (int axis = 0; axis < kAxisCount; ++axis) {
                                                if (converged_axis_[axis]) {
                                                    continue;
                                                }
                                                converged_axis_[axis] =
                                                    pri_s(axis) < abs_tol_ && pri_u(axis) < abs_tol_
                                                    && dua_s_scaled(axis) < abs_tol_
                                                    && dua_u_scaled(axis) < abs_tol_;
                                            }
                                            if (converged_axis_[kYawAxis]
                                                && converged_axis_[kPitchAxis]) {
                                                converged_ = true;
                                                return true;
                                            }
                                        }
                                    }
                                    return false;
                                }

                                [[nodiscard]] float state(int axis, int dim, int k) const noexcept {
                                    return dim == 0 ? x_pos_(axis, k) : x_vel_(axis, k);
                                }

                                [[nodiscard]] float input(int axis, int k) const noexcept {
                                    return u_(axis, k);
                                }
                                [[nodiscard]] bool converged() const noexcept { return converged_; }
                                [[nodiscard]] bool converged(int axis) const noexcept {
                                    return converged_axis_[axis];
                                }
                                [[nodiscard]] int iterations() const noexcept { return iters_; }
                                [[nodiscard]] int horizon() const noexcept { return N_; }

                            private:
                                using Lane     = Eigen::Array<float, kAxisCount, 1>;
                                using LaneMask = Eigen::Array<bool, kAxisCount, 1>;
                                using StateMat = Eigen::Array<float, kAxisCount, kMaxHorizon>;
                                using InputMat = Eigen::Array<float, kAxisCount, kMaxInputHorizon>;

                                float a00_{1.0f};
                                float a01_{0.0f};
                                float a10_{0.0f};
                                float a11_{1.0f};
                                float b0_{0.0f};
                                float b1_{0.0f};
                                float f_pos_{0.0f};
                                float f_vel_{0.0f};
                                float rho_{1.0f};

                                Lane q_pos_{Lane::Zero()};
                                Lane q_vel_{Lane::Zero()};
                                Lane r_val_{Lane::Zero()};
                                Lane k0_{Lane::Zero()};
                                Lane k1_{Lane::Zero()};
                                Lane pinf00_{Lane::Zero()};
                                Lane pinf01_{Lane::Zero()};
                                Lane pinf10_{Lane::Zero()};
                                Lane pinf11_{Lane::Zero()};
                                Lane quu_inv_{Lane::Zero()};
                                Lane am00_{Lane::Zero()};
                                Lane am01_{Lane::Zero()};
                                Lane am10_{Lane::Zero()};
                                Lane am11_{Lane::Zero()};
                                Lane apf_pos_{Lane::Zero()};
                                Lane apf_vel_{Lane::Zero()};
                                Lane bpf_{Lane::Zero()};
                                Lane terminal_ref_pos_{Lane::Zero()};
                                Lane terminal_ref_vel_{Lane::Zero()};
                                Lane u_min_{Lane::Zero()};
                                Lane u_max_{Lane::Zero()};
                                Lane state_min_{
                                    Lane::Constant(-std::numeric_limits<float>::infinity())};
                                Lane state_max_{
                                    Lane::Constant(std::numeric_limits<float>::infinity())};

                                int N_{0};
                                int last_state_index_{0};
                                int last_input_index_{0};

                                StateMat x_pos_{StateMat::Zero()};
                                StateMat x_vel_{StateMat::Zero()};
                                StateMat p_pos_{StateMat::Zero()};
                                StateMat p_vel_{StateMat::Zero()};
                                StateMat v_cost_pos_{StateMat::Zero()};
                                StateMat v_cost_vel_{StateMat::Zero()};
                                StateMat v_pos_{StateMat::Zero()};
                                StateMat v_vel_{StateMat::Zero()};
                                StateMat g_pos_{StateMat::Zero()};
                                StateMat g_vel_{StateMat::Zero()};
                                StateMat qref_pos_{StateMat::Zero()};
                                StateMat qref_vel_{StateMat::Zero()};

                                InputMat u_{InputMat::Zero()};
                                InputMat d_{InputMat::Zero()};
                                InputMat z_cost_{InputMat::Zero()};
                                InputMat z_{InputMat::Zero()};
                                InputMat y_{InputMat::Zero()};

                                float abs_tol_{1e-3f};
                                int max_iter_{10};
                                int check_term_{1};

                                bool converged_{false};
                                std::array<bool, kAxisCount> converged_axis_{};
                                int iters_{0};

                                [[nodiscard]] static Lane clamp_lane(
                                    const Lane& v, const Lane& lo, const Lane& hi) noexcept {
                                    return v.cwiseMin(hi).cwiseMax(lo);
                                }

                                [[nodiscard]] LaneMask active_mask() const noexcept {
                                    LaneMask mask;
                                    mask << !converged_axis_[kYawAxis],
                                        !converged_axis_[kPitchAxis];
                                    return mask;
                                }

                                bool init_axis(int axis, const AxisConfig& cfg) noexcept {
                                    const Eigen::Vector2f Q_aug{cfg.q_pos + rho_, cfg.q_vel + rho_};
                                    const float R_aug = cfg.r + rho_;

                                    using Mat2 = Eigen::Matrix2f;
                                    using Row2 = Eigen::Matrix<float, 1, 2>;

                                    const Eigen::Matrix2f A =
                                        (Eigen::Matrix2f{} << a00_, a01_, a10_, a11_).finished();
                                    const Eigen::Matrix<float, 2, 1> B =
                                        (Eigen::Matrix<float, 2, 1>{} << b0_, b1_).finished();

                                    Mat2 P      = rho_ * Mat2::Identity();
                                    Row2 K_prev = Row2::Zero();
                                    Row2 K      = Row2::Zero();
                                    Mat2 Pinf   = Mat2::Zero();

                                    for (int i = 0; i < 1000; ++i) {
                                        const float S     = R_aug + (B.transpose() * P * B)(0);
                                        const float S_inv = 1.0f / S;
                                        K                 = S_inv * (B.transpose() * P) * A;
                                        Pinf              = Q_aug.asDiagonal().toDenseMatrix()
                                             + A.transpose() * P * (A - B * K);
                                        if ((K - K_prev).cwiseAbs().maxCoeff() < 1e-5f) {
                                            break;
                                        }
                                        K_prev = K;
                                        P      = Pinf;
                                    }

                                    k0_(axis) = K(0, 0);
                                    k1_(axis) = K(0, 1);

                                    pinf00_(axis) = Pinf(0, 0);
                                    pinf01_(axis) = Pinf(0, 1);
                                    pinf10_(axis) = Pinf(1, 0);
                                    pinf11_(axis) = Pinf(1, 1);

                                    const float Quu = R_aug + (B.transpose() * Pinf * B)(0);
                                    quu_inv_(axis)  = 1.0f / Quu;

                                    const Eigen::Matrix2f AmBKt = (A - B * K).transpose();
                                    am00_(axis)                 = AmBKt(0, 0);
                                    am01_(axis)                 = AmBKt(0, 1);
                                    am10_(axis)                 = AmBKt(1, 0);
                                    am11_(axis)                 = AmBKt(1, 1);

                                    const Eigen::Vector2f f   = Eigen::Vector2f::Zero();
                                    const Eigen::Vector2f APf = AmBKt * Pinf * f;
                                    apf_pos_(axis)            = APf(0);
                                    apf_vel_(axis)            = APf(1);
                                    bpf_(axis)                = (B.transpose() * Pinf * f)(0);

                                    q_pos_(axis)     = cfg.q_pos;
                                    q_vel_(axis)     = cfg.q_vel;
                                    r_val_(axis)     = cfg.r;
                                    u_min_(axis)     = -cfg.max_acc;
                                    u_max_(axis)     = cfg.max_acc;
                                    state_min_(axis) = cfg.enable_state_bound
                                                         ? cfg.state_min
                                                         : -std::numeric_limits<float>::infinity();
                                    state_max_(axis) = cfg.enable_state_bound
                                                         ? cfg.state_max
                                                         : std::numeric_limits<float>::infinity();
                                    return true;
                                }

                                void reset_workspace() noexcept {
                                    x_pos_.setZero();
                                    x_vel_.setZero();
                                    p_pos_.setZero();
                                    p_vel_.setZero();
                                    v_cost_pos_.setZero();
                                    v_cost_vel_.setZero();
                                    v_pos_.setZero();
                                    v_vel_.setZero();
                                    g_pos_.setZero();
                                    g_vel_.setZero();
                                    qref_pos_.setZero();
                                    qref_vel_.setZero();
                                    u_.setZero();
                                    d_.setZero();
                                    z_cost_.setZero();
                                    z_.setZero();
                                    y_.setZero();
                                    terminal_ref_pos_.setZero();
                                    terminal_ref_vel_.setZero();
                                    converged_axis_.fill(false);
                                }

                                void
                                    update_terminal_cache(int axis, float pos, float vel) noexcept {
                                    terminal_ref_pos_(axis) =
                                        -(pinf00_(axis) * pos + pinf01_(axis) * vel);
                                    terminal_ref_vel_(axis) =
                                        -(pinf10_(axis) * pos + pinf11_(axis) * vel);
                                }

                                void backward_terminal(const LaneMask& active) noexcept {
                                    const Lane p_pos = terminal_ref_pos_
                                                     - Lane::Constant(rho_)
                                                           * (v_cost_pos_.col(last_state_index_)
                                                              - g_pos_.col(last_state_index_));
                                    const Lane p_vel = terminal_ref_vel_
                                                     - Lane::Constant(rho_)
                                                           * (v_cost_vel_.col(last_state_index_)
                                                              - g_vel_.col(last_state_index_));
                                    p_pos_.col(last_state_index_) =
                                        active.select(p_pos, p_pos_.col(last_state_index_));
                                    p_vel_.col(last_state_index_) =
                                        active.select(p_vel, p_vel_.col(last_state_index_));
                                }

                                void backward_step(int i, const LaneMask& active) noexcept {
                                    const Lane r =
                                        Lane::Constant(-rho_) * (z_cost_.col(i) - y_.col(i));
                                    const Lane q_pos = qref_pos_.col(i)
                                                     - Lane::Constant(rho_)
                                                           * (v_cost_pos_.col(i) - g_pos_.col(i));
                                    const Lane q_vel = qref_vel_.col(i)
                                                     - Lane::Constant(rho_)
                                                           * (v_cost_vel_.col(i) - g_vel_.col(i));
                                    const Lane p1_pos = p_pos_.col(i + 1);
                                    const Lane p1_vel = p_vel_.col(i + 1);

                                    const Lane d = quu_inv_
                                                 * (Lane::Constant(b0_) * p1_pos
                                                    + Lane::Constant(b1_) * p1_vel + r + bpf_);
                                    const Lane p_pos = q_pos + am00_ * p1_pos + am01_ * p1_vel
                                                     - k0_ * r + apf_pos_;
                                    const Lane p_vel = q_vel + am10_ * p1_pos + am11_ * p1_vel
                                                     - k1_ * r + apf_vel_;

                                    d_.col(i)     = active.select(d, d_.col(i));
                                    p_pos_.col(i) = active.select(p_pos, p_pos_.col(i));
                                    p_vel_.col(i) = active.select(p_vel, p_vel_.col(i));
                                }

                                void forward_step(
                                    int i, const LaneMask& active, Lane& x_pos, Lane& x_vel,
                                    Lane& pri_s, Lane& pri_u, Lane& dua_s, Lane& dua_u) noexcept {
                                    const Lane u = -(k0_ * x_pos + k1_ * x_vel) - d_.col(i);
                                    u_.col(i)    = active.select(u, u_.col(i));

                                    const Lane z_prev = z_.col(i);
                                    const Lane z_new  = clamp_lane(u + y_.col(i), u_min_, u_max_);
                                    z_cost_.col(i)    = active.select(z_new, z_cost_.col(i));
                                    z_.col(i)         = active.select(z_new, z_.col(i));
                                    y_.col(i) = active.select(y_.col(i) + (u - z_new), y_.col(i));

                                    const Lane abs_u_res = (u - z_new).abs();
                                    const Lane abs_z_res = (z_prev - z_new).abs();
                                    pri_u = pri_u.max(active.select(abs_u_res, Lane::Zero()));
                                    dua_u = dua_u.max(active.select(abs_z_res, Lane::Zero()));

                                    const Lane v_prev_pos = v_pos_.col(i);
                                    const Lane v_prev_vel = v_vel_.col(i);
                                    const Lane v_new_pos =
                                        clamp_lane(x_pos + g_pos_.col(i), state_min_, state_max_);
                                    const Lane v_new_vel = x_vel + g_vel_.col(i);

                                    v_cost_pos_.col(i) =
                                        active.select(v_new_pos, v_cost_pos_.col(i));
                                    v_cost_vel_.col(i) =
                                        active.select(v_new_vel, v_cost_vel_.col(i));
                                    v_pos_.col(i) = active.select(v_new_pos, v_pos_.col(i));
                                    v_vel_.col(i) = active.select(v_new_vel, v_vel_.col(i));

                                    const Lane x_minus_v_pos =
                                        active.select(x_pos - v_new_pos, Lane::Zero());
                                    const Lane x_minus_v_vel =
                                        active.select(x_vel - v_new_vel, Lane::Zero());

                                    g_pos_.col(i) =
                                        active.select(g_pos_.col(i) + x_minus_v_pos, g_pos_.col(i));
                                    g_vel_.col(i) =
                                        active.select(g_vel_.col(i) + x_minus_v_vel, g_vel_.col(i));

                                    pri_s = pri_s.max(x_minus_v_pos.abs().max(x_minus_v_vel.abs()));
                                    dua_s = dua_s.max(active.select(
                                        (v_prev_pos - v_new_pos)
                                            .abs()
                                            .max((v_prev_vel - v_new_vel).abs()),
                                        Lane::Zero()));

                                    const Lane next_pos =
                                        Lane::Constant(a00_) * x_pos + Lane::Constant(a01_) * x_vel
                                        + Lane::Constant(b0_) * u + Lane::Constant(f_pos_);
                                    const Lane next_vel =
                                        Lane::Constant(a10_) * x_pos + Lane::Constant(a11_) * x_vel
                                        + Lane::Constant(b1_) * u + Lane::Constant(f_vel_);
                                    x_pos_.col(i + 1) = active.select(next_pos, x_pos_.col(i + 1));
                                    x_vel_.col(i + 1) = active.select(next_vel, x_vel_.col(i + 1));
                                    x_pos             = active.select(next_pos, x_pos_.col(i + 1));
                                    x_vel             = active.select(next_vel, x_vel_.col(i + 1));
                                }

                                void terminal_step(
                                    const LaneMask& active, const Lane& x_pos, const Lane& x_vel,
                                    Lane& pri_s, Lane& dua_s) noexcept {
                                    const Lane v_prev_pos = v_pos_.col(last_state_index_);
                                    const Lane v_prev_vel = v_vel_.col(last_state_index_);
                                    const Lane v_new_pos  = clamp_lane(
                                        x_pos + g_pos_.col(last_state_index_), state_min_,
                                        state_max_);
                                    const Lane v_new_vel = x_vel + g_vel_.col(last_state_index_);

                                    v_cost_pos_.col(last_state_index_) = active.select(
                                        v_new_pos, v_cost_pos_.col(last_state_index_));
                                    v_cost_vel_.col(last_state_index_) = active.select(
                                        v_new_vel, v_cost_vel_.col(last_state_index_));
                                    v_pos_.col(last_state_index_) =
                                        active.select(v_new_pos, v_pos_.col(last_state_index_));
                                    v_vel_.col(last_state_index_) =
                                        active.select(v_new_vel, v_vel_.col(last_state_index_));

                                    const Lane x_minus_v_pos =
                                        active.select(x_pos - v_new_pos, Lane::Zero());
                                    const Lane x_minus_v_vel =
                                        active.select(x_vel - v_new_vel, Lane::Zero());

                                    g_pos_.col(last_state_index_) = active.select(
                                        g_pos_.col(last_state_index_) + x_minus_v_pos,
                                        g_pos_.col(last_state_index_));
                                    g_vel_.col(last_state_index_) = active.select(
                                        g_vel_.col(last_state_index_) + x_minus_v_vel,
                                        g_vel_.col(last_state_index_));

                                    pri_s = pri_s.max(x_minus_v_pos.abs().max(x_minus_v_vel.abs()));
                                    dua_s = dua_s.max(active.select(
                                        (v_prev_pos - v_new_pos)
                                            .abs()
                                            .max((v_prev_vel - v_new_vel).abs()),
                                        Lane::Zero()));
                                }
                            };

                            } // namespace fcs::L5

                            for (int k = 0; k < reference_horizon; ++k) {
                                const int ref_k = reference_start_index + k;
                                debug.reference_plan.push_back(
                                    TrajectoryPlanSample{
                                        .yaw = normalize_angle(
                                            trajectory.state(0, ref_k) + trajectory.yaw_origin),
                                        .selected_armor_id       = 0,
                                        .rough_selected_armor_id = 0,
                                        .world_position          = Eigen::Vector3d::Zero(),
                                    });

                                const double optimized_yaw = normalize_angle(
                                    static_cast<double>(yaw_solver.state(0, k))
                                    + trajectory.yaw_origin);
                                const double optimized_pitch =
                                    static_cast<double>(pitch_solver.state(0, k));
                                const Eigen::Vector3d reconstructed_optimized =
                                    spherical_to_cartesian(
                                        trajectory.distances[ref_k], optimized_yaw,
                                        optimized_pitch);
        debug.optimized_plan.push_back(
                            }

                            /// @brief Publish scene entities if non-empty
                            template <typename SceneMessageType>
                            inline void publish_scene_if_nonempty(
                                const std::shared_ptr<FoxgloveServer>& server,
                                std::vector<::foxglove::schemas::SceneEntity> entities) noexcept {
                                if (!entities.empty()) {
                                    SceneMessageType msg;
                                    msg.payload.entities = std::move(entities);
                                    server->enqueue_message(std::move(msg));
                                }
                            }
                            entities.push_back(
                                viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                                    reference.world_position, i - cmd->viz_debug->center_index,
                                    true, cmd->timestamp_ns));
                        }

                        for (int i = 0; i < static_cast<int>(cmd->viz_debug->optimized_plan.size());
                             ++i) {
                            entities.push_back(
                                viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                                    optimized.world_position, i - cmd->viz_debug->center_index,
                                    false, cmd->timestamp_ns));
                        }

                        // ── Reference trajectory connecting line ─────────────────────
                    }
                    }
                    publish_scene_if_nonempty<MpcPredictionSceneMessage>(
                        *server, std::move(entities));
                });
            }
        } // namespace fcs::visualization::foxglove::systems

        // Correctness target:
        //   Only compare the six center outputs actually consumed by production:
        //
        //     yaw
        //     pitch
        //     yaw_vel
        //     pitch_vel
        //     yaw_accel
        //     pitch_accel
        //
        // Build: cmake --build build --target bench_mpc
        // Run:   ./build/bin/bench_mpc

#include "dual_mpc_osqp_solver.hpp"
#include "tiny_api.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

        using namespace fcs::L5;

        struct TinySolverDeleter {
            void operator()(TinySolver* s) const noexcept {
                if (!s)
                    return;
                auto d = [](auto* p) { delete p; };
                d(s->work);
                d(s->settings);
                d(s->cache);
                d(s->solution);
                d(s);
            }
        };
        using TinySolverPtr = std::unique_ptr<TinySolver, TinySolverDeleter>;

        [[maybe_unused]] static inline void compiler_barrier() noexcept {
            asm volatile("" ::: "memory");
        }

        template <typename T>
        [[maybe_unused]] static inline void do_not_optimize(const T& val) noexcept {
            asm volatile("" : : "r"(&val) : "memory");
        }

        using Clock     = std::chrono::steady_clock;
        using TimePoint = std::chrono::time_point<Clock>;

        [[nodiscard]] static inline double elapsed_us(TimePoint a, TimePoint b) noexcept {
            return std::chrono::duration<double, std::micro>(b - a).count();
        }

        static constexpr int kHorizonAhead = 50;
        static constexpr int kHorizonBack  = 50;
        static constexpr int kHorizon      = kHorizonAhead + kHorizonBack + 1;
        static constexpr double kDt        = 0.01;
        static constexpr double kRho       = 1.0;
        static constexpr double kAbsTol    = 1e-3;
        static constexpr int kWarmup       = 200;
        static constexpr int kTrials       = 1000;

        struct AxisParams {
            double q_pos, q_vel, r, max_acc;
        };

        static constexpr AxisParams kYaw   = {9e6, 0.0, 1.0, 50.0};
        static constexpr AxisParams kPitch = {9e6, 0.0, 1.0, 100.0};

        using RefTrajectory = Eigen::MatrixXd;

        [[nodiscard]] static RefTrajectory ref_sinusoid(int N, double freq, double amp) {
            Eigen::MatrixXd ref(2, N);
            const double omega = 2.0 * std::numbers::pi_v<double> * freq;
            for (int i = 0; i < N; ++i) {
                const double t = static_cast<double>(i - kHorizonBack) * kDt;
                ref(0, i)      = amp * std::sin(omega * t);
                ref(1, i)      = amp * omega * std::cos(omega * t);
            }
            return ref;
            // Clean chirp: position and velocity are derived from the same shifted phase.
            // This avoids testing against an internally inconsistent reference.
            [[nodiscard]] static RefTrajectory ref_chirp(int N, double amp) {
                Eigen::MatrixXd ref(2, N);
                const double f0 = 0.5;
                const double f1 = 3.0;
                const double T  = static_cast<double>(std::max(N - 1, 1)) * kDt;

                for (int i = 0; i < N; ++i) {
                    const double tau   = static_cast<double>(i) * kDt;
                    const double freq  = f0 + (f1 - f0) * tau / T;
                    const double phase = 2.0 * std::numbers::pi_v<double>
                                       * (f0 * tau + 0.5 * (f1 - f0) / T * tau * tau);

                    ref(0, i) = amp * std::sin(phase);
                    ref(1, i) = amp * std::cos(phase) * 2.0 * std::numbers::pi_v<double> * freq;
                }

                return ref;
            }

            [[nodiscard]] static RefTrajectory ref_noisy_sinusoid(
                int N, unsigned seed, double amp) {
                std::mt19937 rng(seed);
                std::normal_distribution<float> noise(0.0f, 0.01f);
                auto ref = ref_sinusoid(N, 1.0, amp);
                for (int i = 0; i < N; ++i) {
                    ref(0, i) += static_cast<double>(noise(rng));
                    ref(1, i) += static_cast<double>(noise(rng)) * 10.0;
                }
                return ref;
                [[nodiscard]] static RefTrajectory ref_zero(int N) {
                    return Eigen::MatrixXd::Zero(2, N);
                }

                [[nodiscard]] static RefTrajectory ref_step(int N, double pos, double vel = 0.0) {
                    Eigen::MatrixXd ref(2, N);
                    ref.row(0).setConstant(pos);
                    ref.row(1).setConstant(vel);
                    return ref;
                }

                [[nodiscard]] static RefTrajectory ref_ramp(int N, double start, double slope) {
                    Eigen::MatrixXd ref(2, N);
                    for (int i = 0; i < N; ++i) {
                        const double t = static_cast<double>(i - kHorizonBack) * kDt;
                        ref(0, i)      = start + slope * t;
                        ref(1, i)      = slope;
                    }
                    return ref;
                }

                struct TinyPairSetup {
                    TinySolverPtr yaw;
                    TinySolverPtr pitch;

                    TinyPairSetup() {
                        yaw   = make(false);
                        pitch = make(true);
                    }

                private:
                    [[nodiscard]] static TinySolverPtr make(bool is_pitch) {
                        const AxisParams& p = is_pitch ? kPitch : kYaw;
                        const Eigen::Matrix2d A =
                            (Eigen::Matrix2d{} << 1.0, kDt, 0.0, 1.0).finished();
                        const Eigen::Matrix<double, 2, 1> B =
                            (Eigen::Matrix<double, 2, 1>{} << 0.0, kDt).finished();
                        const Eigen::Vector2d f = Eigen::Vector2d::Zero();
                        const Eigen::Matrix2d Q = Eigen::Vector2d{p.q_pos, p.q_vel}.asDiagonal();
                        const Eigen::Matrix<double, 1, 1> R =
                            Eigen::Matrix<double, 1, 1>::Constant(p.r);

                        TinySolver* raw = nullptr;

                        auto* s                        = solver.get();
                        s->settings->abs_pri_tol       = kAbsTol;
                        s->settings->abs_dua_tol       = kAbsTol;
                        s->settings->check_termination = 1;
                        s->settings->en_input_bound    = 1;
                        s->settings->en_state_bound    = is_pitch ? 1 : 0;

                        if (is_pitch) {
                            x_min.row(0).setConstant(-0.698);
                            x_max.row(0).setConstant(0.698);
                        }

                        tiny_set_bound_constraints(
                            s, x_min, x_max, Eigen::MatrixXd::Constant(1, kHorizon - 1, -p.max_acc),
                            Eigen::MatrixXd::Constant(1, kHorizon - 1, p.max_acc));

                        return solver;
                    }
                };
                using DualSmallMpcSolver = DualMpcOsqpSolver;
                struct DualSetup {
                    DualSmallMpcSolver solver;

                    DualSetup() {
                        DualSmallMpcSolver::AxisConfig yaw_cfg;
                        yaw_cfg.q_pos   = static_cast<float>(kYaw.q_pos);
                        yaw_cfg.q_vel   = static_cast<float>(kYaw.q_vel);
                        yaw_cfg.r       = static_cast<float>(kYaw.r);
                        yaw_cfg.max_acc = static_cast<float>(kYaw.max_acc);

                        DualSmallMpcSolver::AxisConfig pitch_cfg;
                        pitch_cfg.q_pos              = static_cast<float>(kPitch.q_pos);
                        pitch_cfg.q_vel              = static_cast<float>(kPitch.q_vel);
                        pitch_cfg.r                  = static_cast<float>(kPitch.r);
                        pitch_cfg.max_acc            = static_cast<float>(kPitch.max_acc);
                        pitch_cfg.enable_state_bound = true;
                        pitch_cfg.state_min          = -0.698f;
                        pitch_cfg.state_max          = 0.698f;

        auto result = DualSmallMpcSolver::create(

        solver = std::move(*result);
        solver.set_settings(static_cast<float>(kAbsTol), 1000);
                    }
                };

                struct Stats {
                    double min, p50, p95, max;
                    double mean, stddev;
                    int converged_count;
                };

                std::sort(samples.begin(), samples.end());

                double sum = 0.0;
                for (double s : samples)
                    sum += s;

                const double mean = sum / static_cast<double>(samples.size());

                double sq_sum = 0.0;
                for (double s : samples) {
                    const double d = s - mean;
                    sq_sum += d * d;
                }

                const size_t p95_index = std::min(
                    samples.size() - 1,
                    static_cast<size_t>(static_cast<double>(samples.size()) * 0.95));

                return {
                    samples.front(),
                    samples[samples.size() / 2],
                    samples[p95_index],
                    samples.back(),
                    mean,
                    std::sqrt(sq_sum / static_cast<double>(samples.size())),
                    converged,
                };
            }

            static void print_stats(const char* label, const Stats& s, int total) {
                std::cout << "  " << std::left << std::setw(24) << label << " ";
                std::cout << std::fixed << std::setprecision(2) << "p50=" << std::setw(8) << s.p50
                          << "us"
                          << " mean=" << std::setw(8) << s.mean << "us"
                          << " p95=" << std::setw(8) << s.p95 << "us"
                          << " min=" << std::setw(8) << s.min << "us"
                          << " max=" << std::setw(8) << s.max << "us"
                          << " conv=" << s.converged_count << "/" << total << "\n";
            }

            struct TimingResult {
                Stats stats;
                int iterations;
            };

            [[nodiscard]] static TimingResult bench_tiny_pair_solve(
                TinyPairSetup & setup, const Eigen::MatrixXd& yaw_ref,
                const Eigen::MatrixXd& pitch_ref, int max_iter, int trials) {
                auto* yaw                 = setup.yaw.get();
                auto* pitch               = setup.pitch.get();
                yaw->settings->max_iter   = max_iter;
                pitch->settings->max_iter = max_iter;

                for (int w = 0; w < kWarmup; ++w) {
                    tiny_set_x0(yaw, yaw_ref.col(0));
                    tiny_set_x0(pitch, pitch_ref.col(0));
                    yaw->work->Xref   = yaw_ref;
                    pitch->work->Xref = pitch_ref;
                    tiny_solve(yaw);
                    tiny_solve(pitch);
                }

                int converged      = 0;
                int max_iterations = 0;

                for (int i = 0; i < trials; ++i) {
                    tiny_set_x0(yaw, yaw_ref.col(0));
                    tiny_set_x0(pitch, pitch_ref.col(0));
                    yaw->work->Xref   = yaw_ref;
                    pitch->work->Xref = pitch_ref;

                    compiler_barrier();

                    const int yaw_status   = tiny_solve(yaw);
                    const int pitch_status = tiny_solve(pitch);

                    compiler_barrier();
                    const auto t1 = Clock::now();
                    compiler_barrier();

                    if (yaw_status == 0 && pitch_status == 0 && yaw->solution->solved == 1
                        && pitch->solution->solved == 1) {
                        ++converged;
                    }

                    max_iterations = std::max(
                        max_iterations, std::max(yaw->solution->iter, pitch->solution->iter));
                    times.push_back(elapsed_us(t0, t1));
                }

                return {compute_stats(std::move(times), converged), max_iterations};
            }

[[nodiscard]] static TimingResult bench_dual_solve(
    DualSetup& setup, const Eigen::MatrixXd& yaw_ref_d, const Eigen::MatrixXd& pitch_ref_d,

    const Eigen::MatrixXf yaw_ref   = yaw_ref_d.cast<float>();
    const Eigen::MatrixXf pitch_ref = pitch_ref_d.cast<float>();

    for (int w = 0; w < kWarmup; ++w) {
            setup.solver.set_x0(
                yaw_ref_d(0, 0), yaw_ref_d(1, 0), pitch_ref_d(0, 0), pitch_ref_d(1, 0));
            setup.solver.set_reference(yaw_ref, pitch_ref);
            setup.solver.solve();
    }


    int converged      = 0;
    int max_iterations = 0;

    for (int i = 0; i < trials; ++i) {
            setup.solver.set_x0(
                yaw_ref_d(0, 0), yaw_ref_d(1, 0), pitch_ref_d(0, 0), pitch_ref_d(1, 0));
            setup.solver.set_reference(yaw_ref, pitch_ref);

            compiler_barrier();

            const bool ok = setup.solver.solve();

            compiler_barrier();
            const auto t1 = Clock::now();
            compiler_barrier();

            do_not_optimize(ok);

            if (ok)
                ++converged;

            max_iterations = std::max(max_iterations, setup.solver.iterations());
            times.push_back(elapsed_us(t0, t1));
    }

    return {compute_stats(std::move(times), converged), max_iterations};
        }

        [[nodiscard]] static Stats bench_tiny_pair_pipeline(
            TinyPairSetup & setup, const Eigen::MatrixXd& yaw_ref, const Eigen::MatrixXd& pitch_ref,
            int max_iter, int trials) {
            auto* yaw                 = setup.yaw.get();
            auto* pitch               = setup.pitch.get();
            yaw->settings->max_iter   = max_iter;
            pitch->settings->max_iter = max_iter;

            int converged = 0;

            for (int i = 0; i < trials; ++i) {
                compiler_barrier();
                const auto t0 = Clock::now();
                compiler_barrier();

                tiny_set_x0(yaw, yaw_ref.col(0));
                tiny_set_x0(pitch, pitch_ref.col(0));
                yaw->work->Xref        = yaw_ref;
                pitch->work->Xref      = pitch_ref;
                const int yaw_status   = tiny_solve(yaw);
                const int pitch_status = tiny_solve(pitch);

                compiler_barrier();
                const auto t1 = Clock::now();
                compiler_barrier();

                if (yaw_status == 0 && pitch_status == 0 && yaw->solution->solved == 1
                    && pitch->solution->solved == 1) {

                    times.push_back(elapsed_us(t0, t1));
                }

                return compute_stats(std::move(times), converged);
            }

[[nodiscard]] static Stats bench_dual_pipeline(
    DualSetup& setup, const Eigen::MatrixXd& yaw_ref_d, const Eigen::MatrixXd& pitch_ref_d,

    const Eigen::MatrixXf yaw_ref   = yaw_ref_d.cast<float>();
    const Eigen::MatrixXf pitch_ref = pitch_ref_d.cast<float>();

    std::vector<double> times;
    times.reserve(trials);

    int converged = 0;

    for (int i = 0; i < trials; ++i) {
            compiler_barrier();
            const auto t0 = Clock::now();
            compiler_barrier();

            setup.solver.set_x0(
                yaw_ref_d(0, 0), yaw_ref_d(1, 0), pitch_ref_d(0, 0), pitch_ref_d(1, 0));
            setup.solver.set_reference(yaw_ref, pitch_ref);
            const bool ok = setup.solver.solve();

            compiler_barrier();
            const auto t1 = Clock::now();
            compiler_barrier();

            times.push_back(elapsed_us(t0, t1));
    }

    return compute_stats(std::move(times), converged);
    // ================================================================================================
// Center output consistency: this is the correctness metric that matches production usage.
// ================================================================================================

struct CenterOutput {
            double yaw;
            double pitch;
            double yaw_vel;
            double pitch_vel;
            double yaw_accel;
            double pitch_accel;
};

struct CenterOutputReport {
            CenterOutput tiny;
            CenterOutput dual;
            CenterOutput abs_err;
            CenterOutput norm_err;

            double max_abs_err;
            double max_norm_err;

            int center_index;
            int control_index;

            bool tiny_converged;
            bool dual_converged;
            bool pass;
};

[[nodiscard]] static double max_center_output_component(const CenterOutput& v) {
            return std::max({
                v.yaw,
                v.pitch,
                v.yaw_vel,
                v.pitch_vel,
                v.yaw_accel,
                v.pitch_accel,
            });
}

[[nodiscard]] static CenterOutput abs_diff(const CenterOutput& a, const CenterOutput& b) {
            return {
                std::abs(a.yaw - b.yaw),
                std::abs(a.pitch - b.pitch),
                std::abs(a.yaw_vel - b.yaw_vel),
                std::abs(a.pitch_vel - b.pitch_vel),
                std::abs(a.yaw_accel - b.yaw_accel),
                std::abs(a.pitch_accel - b.pitch_accel),
            };
}

[[nodiscard]] static CenterOutput normalized_error(const CenterOutput& e) {
            // These are engineering scales, not solver scales.
            // Tune them if your actuator/sensor output resolution is tighter or looser.
            constexpr double kYawScale        = 1.0;
            constexpr double kPitchScale      = 1.0;
            constexpr double kYawVelScale     = 10.0;
            constexpr double kPitchVelScale   = 10.0;
            constexpr double kYawAccelScale   = kYaw.max_acc;
            constexpr double kPitchAccelScale = kPitch.max_acc;

            return {
                e.yaw / kYawScale,
                e.pitch / kPitchScale,
                e.yaw_vel / kYawVelScale,
                e.pitch_vel / kPitchVelScale,
                e.yaw_accel / kYawAccelScale,
                e.pitch_accel / kPitchAccelScale,
            };
}

[[nodiscard]] static CenterOutputReport compare_center_output(
    const Eigen::MatrixXd& yaw_ref, const Eigen::MatrixXd& pitch_ref, int max_iter) {
            TinyPairSetup tiny;
            DualSetup dual;

            auto* yaw                 = tiny.yaw.get();
            auto* pitch               = tiny.pitch.get();
            yaw->settings->max_iter   = max_iter;
            pitch->settings->max_iter = max_iter;
            dual.solver.set_settings(static_cast<float>(kAbsTol), max_iter);

            tiny_set_x0(yaw, yaw_ref.col(0));
            tiny_set_x0(pitch, pitch_ref.col(0));
            yaw->work->Xref        = yaw_ref;
            pitch->work->Xref      = pitch_ref;
            const int yaw_status   = tiny_solve(yaw);
            const int pitch_status = tiny_solve(pitch);

            dual.solver.set_x0(yaw_ref(0, 0), yaw_ref(1, 0), pitch_ref(0, 0), pitch_ref(1, 0));
            dual.solver.set_reference(yaw_ref.cast<float>(), pitch_ref.cast<float>());
            const bool dual_ok = dual.solver.solve();

            // Mirror production optimize() semantics:
            //   center_index  = clamp(horizon_back, 0, effective_horizon - 1)
            //   control_index = clamp(center_index, 0, effective_horizon - 2)
            //
            // In this benchmark, effective_horizon == kHorizon.
            const int effective_horizon = kHorizon;
            const int center_index      = std::clamp(kHorizonBack, 0, effective_horizon - 1);
            const int control_index     = std::clamp(center_index, 0, effective_horizon - 2);

            const CenterOutput tiny_out{
                yaw->solution->x(0, center_index),  pitch->solution->x(0, center_index),
                yaw->solution->x(1, center_index),  pitch->solution->x(1, center_index),
                yaw->solution->u(0, control_index), pitch->solution->u(0, control_index),
            };

            const CenterOutput dual_out{
                static_cast<double>(
                    dual.solver.state(DualSmallMpcSolver::kYawAxis, 0, center_index)),
                static_cast<double>(
                    dual.solver.state(DualSmallMpcSolver::kPitchAxis, 0, center_index)),
                static_cast<double>(
                    dual.solver.state(DualSmallMpcSolver::kYawAxis, 1, center_index)),
                static_cast<double>(
                    dual.solver.state(DualSmallMpcSolver::kPitchAxis, 1, center_index)),
                static_cast<double>(dual.solver.input(DualSmallMpcSolver::kYawAxis, control_index)),
                static_cast<double>(
                    dual.solver.input(DualSmallMpcSolver::kPitchAxis, control_index)),
            };

            const CenterOutput err      = abs_diff(tiny_out, dual_out);
            const CenterOutput norm_err = normalized_error(err);

            const double max_abs  = max_center_output_component(err);
            const double max_norm = max_center_output_component(norm_err);

            const bool tiny_ok = yaw_status == 0 && pitch_status == 0 && yaw->solution->solved == 1
                              && pitch->solution->solved == 1;

            // Suggested correctness tolerances.
            //
            // These are intentionally focused on the exported command values, not the full horizon.
            // Tune accel tolerance to actuator command resolution.
            constexpr double kPosTol   = 1e-3;
            constexpr double kVelTol   = 1e-2;
            constexpr double kAccelTol = 1e-1;
            constexpr double kNormTol  = 1e-3;

            const bool pass = tiny_ok && dual_ok && err.yaw <= kPosTol && err.pitch <= kPosTol
                           && err.yaw_vel <= kVelTol && err.pitch_vel <= kVelTol
                           && err.yaw_accel <= kAccelTol && err.pitch_accel <= kAccelTol
                           && max_norm <= kNormTol;

            return {
                tiny_out,     dual_out,      err,     norm_err, max_abs, max_norm,
                center_index, control_index, tiny_ok, dual_ok,  pass,
            };
}

static void print_center_output_summary_header() {
            std::cout << "  " << std::left << std::setw(16) << "Scenario" << std::setw(12) << "yaw"
                      << std::setw(12) << "pitch" << std::setw(12) << "yaw_vel" << std::setw(12)
                      << "pit_vel" << std::setw(12) << "yaw_acc" << std::setw(12) << "pit_acc"
                      << std::setw(12) << "max_norm" << std::setw(8) << "Pass"
                      << "\n";
            std::cout << "  " << std::string(108, '-') << "\n";
}

static void print_center_output_summary_row(const char* name, const CenterOutputReport& r) {
            std::cout << "  " << std::left << std::setw(16) << name << std::scientific
                      << std::setprecision(2) << std::setw(12) << r.abs_err.yaw << std::setw(12)
                      << r.abs_err.pitch << std::setw(12) << r.abs_err.yaw_vel << std::setw(12)
                      << r.abs_err.pitch_vel << std::setw(12) << r.abs_err.yaw_accel
                      << std::setw(12) << r.abs_err.pitch_accel << std::setw(12) << r.max_norm_err
                      << std::setw(8) << (r.pass ? "Y" : "N") << "\n";
}

static void print_center_output_report(const char* label, const CenterOutputReport& r) {
            std::cout << "  " << label << "\n";
            std::cout << "  center_index=" << r.center_index << " control_index=" << r.control_index
                      << " tiny_ok=" << (r.tiny_converged ? "Y" : "N")
                      << " dual_ok=" << (r.dual_converged ? "Y" : "N")
                      << " pass=" << (r.pass ? "Y" : "N") << "\n";

            auto print_row = [](const char* name, double tiny, double dual, double err,
                                double norm) {
                std::cout << "    " << std::left << std::setw(12) << name
                          << " tiny=" << std::scientific << std::setprecision(6) << std::setw(14)
                          << tiny << " dual=" << std::setw(14) << dual
                          << " abs_err=" << std::setw(14) << err << " norm_err=" << norm << "\n";
            };

            print_row("yaw", r.tiny.yaw, r.dual.yaw, r.abs_err.yaw, r.norm_err.yaw);
            print_row("pitch", r.tiny.pitch, r.dual.pitch, r.abs_err.pitch, r.norm_err.pitch);
            print_row(
                "yaw_vel", r.tiny.yaw_vel, r.dual.yaw_vel, r.abs_err.yaw_vel, r.norm_err.yaw_vel);
            print_row(
                "pitch_vel", r.tiny.pitch_vel, r.dual.pitch_vel, r.abs_err.pitch_vel,
                r.norm_err.pitch_vel);
            print_row(
                "yaw_accel", r.tiny.yaw_accel, r.dual.yaw_accel, r.abs_err.yaw_accel,
                r.norm_err.yaw_accel);
            print_row(
                "pitch_accel", r.tiny.pitch_accel, r.dual.pitch_accel, r.abs_err.pitch_accel,
                r.norm_err.pitch_accel);

            std::cout << "    max_abs_err=" << std::scientific << std::setprecision(6)
                      << r.max_abs_err << " max_norm_err=" << r.max_norm_err << "\n";
}

// ================================================================================================
// Full horizon debug accuracy. This is intentionally not the main correctness metric.
// ================================================================================================

struct FullHorizonDebugReport {
            double max_abs_state_err;
            double rmse_state;
            double max_abs_input_err;
            bool tiny_converged;
            bool dual_converged;
};

[[nodiscard]] static FullHorizonDebugReport compare_full_horizon_debug(
    const Eigen::MatrixXd& yaw_ref, const Eigen::MatrixXd& pitch_ref, int max_iter) {
            TinyPairSetup tiny;
            DualSetup dual;

            auto* yaw                 = tiny.yaw.get();
            auto* pitch               = tiny.pitch.get();
            yaw->settings->max_iter   = max_iter;
            pitch->settings->max_iter = max_iter;
            dual.solver.set_settings(static_cast<float>(kAbsTol), max_iter);

            tiny_set_x0(yaw, yaw_ref.col(0));
            tiny_set_x0(pitch, pitch_ref.col(0));
            yaw->work->Xref        = yaw_ref;
            pitch->work->Xref      = pitch_ref;
            const int yaw_status   = tiny_solve(yaw);
            const int pitch_status = tiny_solve(pitch);

            dual.solver.set_x0(yaw_ref(0, 0), yaw_ref(1, 0), pitch_ref(0, 0), pitch_ref(1, 0));
            dual.solver.set_reference(yaw_ref.cast<float>(), pitch_ref.cast<float>());
            const bool dual_ok = dual.solver.solve();

            double max_state = 0.0;

            for (int k = 0; k < kHorizon; ++k) {
                const double yaw_pos_err = std::abs(
                    yaw->solution->x(0, k)
                    - static_cast<double>(dual.solver.state(DualSmallMpcSolver::kYawAxis, 0, k)));
                const double yaw_vel_err = std::abs(
                    yaw->solution->x(1, k)
                    - static_cast<double>(dual.solver.state(DualSmallMpcSolver::kYawAxis, 1, k)));
                const double pitch_pos_err = std::abs(
                    pitch->solution->x(0, k)
                    - static_cast<double>(dual.solver.state(DualSmallMpcSolver::kPitchAxis, 0, k)));
        const double pitch_vel_err = std::abs(

        max_state = std::max({max_state, yaw_pos_err, yaw_vel_err, pitch_pos_err, pitch_vel_err});
        sum_sq += yaw_pos_err * yaw_pos_err + yaw_vel_err * yaw_vel_err;
        sum_sq += pitch_pos_err * pitch_pos_err + pitch_vel_err * pitch_vel_err;
            }

            for (int k = 0; k < kHorizon - 1; ++k) {
                const double yaw_input_err = std::abs(
                    yaw->solution->u(0, k)
                    - static_cast<double>(dual.solver.input(DualSmallMpcSolver::kYawAxis, k)));
                const double pitch_input_err = std::abs(
                    pitch->solution->u(0, k)
                    - static_cast<double>(dual.solver.input(DualSmallMpcSolver::kPitchAxis, k)));

                max_input = std::max({max_input, yaw_input_err, pitch_input_err});
            }

            return {
                max_state,

            };
}

static void print_section(const char* title) {
            std::cout << "\n" << std::string(78, '=') << "\n";
            std::cout << "  " << title << "\n";
            std::cout << std::string(78, '=') << "\n";
}

int main() {
            std::cout << std::unitbuf;

            const auto yaw_ref_sin     = ref_sinusoid(kHorizon, 1.0, 0.3);
            const auto pitch_ref_sin   = ref_sinusoid(kHorizon, 0.8, 0.2);
            const auto yaw_ref_zero    = ref_zero(kHorizon);
            const auto pitch_ref_zero  = ref_zero(kHorizon);
            const auto yaw_ref_step    = ref_step(kHorizon, 0.3);
            const auto pitch_ref_step  = ref_step(kHorizon, 0.6);
            const auto yaw_ref_ramp    = ref_ramp(kHorizon, 0.0, 0.8);
            const auto pitch_ref_bound = ref_step(kHorizon, 0.68);

            print_section("Dual-Axis MPC Benchmark");
            std::cout << "  Horizon: " << kHorizon << " (ahead=" << kHorizonAhead
                      << " back=" << kHorizonBack << ")\n";
            std::cout << "  dt=" << kDt << " rho=" << kRho << " tol=" << kAbsTol << "\n";
            std::cout << "  Trials=" << kTrials << " Warmup=" << kWarmup << "\n";

            TinyPairSetup tiny;
            DualSetup dual;

            const auto tiny_t =
                bench_tiny_pair_solve(tiny, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
            const auto dual_t = bench_dual_solve(dual, yaw_ref_sin, pitch_ref_sin, 10, kTrials);

            print_stats("2x TinyMPC", tiny_t.stats, kTrials);
            print_stats("DualSmallMpcSolver", dual_t.stats, kTrials);

            std::cout << "  tiny_max_iter_seen=" << tiny_t.iterations
                      << " dual_max_iter_seen=" << dual_t.iterations << "\n";

            std::cout << "  speedup=" << std::fixed << std::setprecision(2)
                      << tiny_t.stats.p50 / std::max(dual_t.stats.p50, 0.001) << "x\n";
    }


        TinyPairSetup tiny;
        DualSetup dual;

        const auto tiny_t = bench_tiny_pair_pipeline(tiny, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
        const auto dual_t = bench_dual_pipeline(dual, yaw_ref_sin, pitch_ref_sin, 10, kTrials);

        print_stats("2x TinyMPC", tiny_t, kTrials);
        print_stats("DualSmallMpcSolver", dual_t, kTrials);

        std::cout << "  speedup=" << std::fixed << std::setprecision(2)
                  << tiny_t.p50 / std::max(dual_t.p50, 0.001) << "x\n";
        }

        struct Scenario {
            const char* name;
            const Eigen::MatrixXd& yaw_ref;
            const Eigen::MatrixXd& pitch_ref;
        };

        const Scenario scenarios[] = {
            {    "sin/sin",   yaw_ref_sin,   pitch_ref_sin},
            {"chirp/noisy", yaw_ref_chirp, pitch_ref_noisy},
            {  "zero/zero",  yaw_ref_zero,  pitch_ref_zero},
            { "step/bound",  yaw_ref_step, pitch_ref_bound},
            {  "ramp/step",  yaw_ref_ramp,  pitch_ref_step},
        };

        {
            print_section("Scenario Timing Sweep");

            std::cout << "  " << std::left << std::setw(16) << "Scenario" << std::setw(14)
                      << "Tiny p50" << std::setw(14) << "Dual p50" << std::setw(10) << "Speedup"
                      << "\n";
            std::cout << "  " << std::string(54, '-') << "\n";

            for (const auto& scenario : scenarios) {
                TinyPairSetup tiny;
                DualSetup dual;

                const auto tiny_t =
                    bench_tiny_pair_solve(tiny, scenario.yaw_ref, scenario.pitch_ref, 10, kTrials);
                const auto dual_t =
                    bench_dual_solve(dual, scenario.yaw_ref, scenario.pitch_ref, 10, kTrials);

                std::cout << "  " << std::left << std::setw(16) << scenario.name << std::fixed
                          << std::setprecision(2) << std::setw(14) << tiny_t.stats.p50
                          << std::setw(14) << dual_t.stats.p50 << std::setw(10)
                          << tiny_t.stats.p50 / std::max(dual_t.stats.p50, 0.001) << "\n";
            }
        }
        print_section("Center Output Consistency");

        bool all_pass = true;

        print_center_output_summary_header();

        for (const auto& scenario : scenarios) {
            const auto report = compare_center_output(scenario.yaw_ref, scenario.pitch_ref, 100);

            print_center_output_summary_row(scenario.name, report);
            all_pass = all_pass && report.pass;
        }

        std::cout << "  overall_center_output_consistency=" << (all_pass ? "PASS" : "FAIL") << "\n";

        std::cout << "\n";
        const auto detailed = compare_center_output(yaw_ref_sin, pitch_ref_noisy, 100);
        print_center_output_report("detailed sin/noisy", detailed);
    }

    {
        print_section("Full Horizon Debug Accuracy");

        const auto debug = compare_full_horizon_debug(yaw_ref_sin, pitch_ref_noisy, 100);

        std::cout << "  tiny_ok=" << (debug.tiny_converged ? "Y" : "N")
                  << " dual_ok=" << (debug.dual_converged ? "Y" : "N") << "\n";
        std::cout << "  max_state_err=" << std::scientific << std::setprecision(4)
                  << debug.max_abs_state_err << " rmse_state=" << debug.rmse_state
                  << " max_input_err=" << debug.max_abs_input_err << "\n";

        std::cout << "  note: full horizon errors are debug-only. "
                  << "The pass/fail correctness metric is Center Output Consistency.\n";
    }

    return 0;
    }
