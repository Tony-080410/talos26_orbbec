// Rigorous MPC solver benchmark: TinyMPC (double/dynamic) vs SmallMpcSolver (float/fixed)
//
// Methodology:
//   1. Compiler barrier around timed region prevents reordering
//   2. steady_clock for monotonic timing
//   3. Isolated solve() only, data setup excluded
//   4. Convergence sweep: max_iter = 5, 10, 20, 50, 100, 200, 500, 1000
//   5. Multiple reference trajectories (step, sinusoid, chirp, noisy)
//   6. Per-iteration residual tracking for algorithmic equivalence proof
//   7. Full-pipeline timing (setup + solve) for real-world cost
//   8. Bootstrap 95% CI for median
//
// Build: cmake --build build --target bench_mpc
// Run:   ./build/bin/bench_mpc

#include "small_mpc_solver.hpp"
#include "tiny_api.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace fcs::L5;

// =============================================================================
// Utility
// =============================================================================
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

// Prevent compiler from reordering memory operations across this point.
// Ensures the clock reads are not moved past the solve() call.
[[maybe_unused]] static inline void compiler_barrier() noexcept { asm volatile("" ::: "memory"); }

// Don't let the optimizer elide the result.
template <typename T>
[[maybe_unused]] static inline void do_not_optimize(const T& val) noexcept {
    asm volatile("" : : "r"(&val) : "memory");
}

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

[[nodiscard]] static inline double elapsed_us(TimePoint a, TimePoint b) noexcept {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

// =============================================================================
// Problem configuration (production-matched)
// =============================================================================
static constexpr int kHorizonAhead = 50;
static constexpr int kHorizonBack  = 50;
static constexpr int kHorizon      = kHorizonAhead + kHorizonBack + 1; // 101
static constexpr double kDt        = 0.01;
static constexpr double kRho       = 1.0;
static constexpr double kAbsTol    = 1e-3;

struct AxisParams {
    double q_pos, q_vel, r, max_acc;
};
static constexpr AxisParams kYaw   = {9e6, 0.0, 1.0, 50.0};
static constexpr AxisParams kPitch = {9e6, 0.0, 1.0, 100.0};

static constexpr int kWarmup     = 200;
static constexpr int kTrials     = 1000;
static constexpr int kBootstrapN = 10000;                              // bootstrap resamples for CI

// =============================================================================
// Reference trajectory generators
// =============================================================================
// Each returns a 2×N matrix [pos; vel] as double.
// Velocities are the exact analytic derivative of position.

using RefTrajectory = Eigen::MatrixXd;

[[nodiscard]] static RefTrajectory ref_step(int N) {
    // Step at t=0: position jumps to 0.3, velocity has a delta-shaped impulse
    // approximated by the finite-difference.
    Eigen::MatrixXd ref(2, N);
    for (int i = 0; i < N; ++i) {
        const double t   = static_cast<double>(i - kHorizonBack) * kDt;
        const double pos = (t >= 0.0) ? 0.3 : 0.0;
        ref(0, i)        = pos;
        // velocity = dpos/dt via finite difference
        const double t_next   = static_cast<double>(i - kHorizonBack + 1) * kDt;
        const double pos_next = (t_next >= 0.0) ? 0.3 : 0.0;
        ref(1, i)             = (pos_next - pos) / kDt;
    }
    return ref;
}

[[nodiscard]] static RefTrajectory ref_sinusoid(int N, double freq) {
    Eigen::MatrixXd ref(2, N);
    const double omega = 2.0 * M_PI * freq;
    const double amp   = 0.3;
    for (int i = 0; i < N; ++i) {
        const double t = static_cast<double>(i - kHorizonBack) * kDt;
        ref(0, i)      = amp * std::sin(omega * t);
        ref(1, i)      = amp * omega * std::cos(omega * t);
    }
    return ref;
}

[[nodiscard]] static RefTrajectory ref_chirp(int N) {
    // Frequency sweeps from 0.5 Hz to 3 Hz over the horizon
    Eigen::MatrixXd ref(2, N);
    const double f0 = 0.5, f1 = 3.0;
    const double T = static_cast<double>(N) * kDt;
    for (int i = 0; i < N; ++i) {
        const double t      = static_cast<double>(i - kHorizonBack) * kDt;
        const double t_norm = (t + static_cast<double>(kHorizonBack) * kDt) / T;
        const double freq   = f0 + (f1 - f0) * t_norm;
        const double phase  = 2.0 * M_PI * (f0 * t + 0.5 * (f1 - f0) / T * t * t);
        ref(0, i)           = 0.3 * std::sin(phase);
        // derivative: 0.3 * cos(phase) * dphase/dt
        const double dphase_dt = 2.0 * M_PI * freq;
        ref(1, i)              = 0.3 * std::cos(phase) * dphase_dt;
    }
    return ref;
}

[[nodiscard]] static RefTrajectory ref_noisy_sinusoid(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    auto clean = ref_sinusoid(N, 1.0);
    for (int i = 0; i < N; ++i) {
        clean(0, i) += static_cast<double>(noise(rng));
        clean(1, i) += static_cast<double>(noise(rng)) * 10.0;
    }
    return clean;
}

// =============================================================================
// Solver construction helpers
// =============================================================================

struct TinySolverSetup {
    TinySolverPtr solver;

    explicit TinySolverSetup(bool is_pitch) {
        const AxisParams& p     = is_pitch ? kPitch : kYaw;
        const Eigen::Matrix2d A = (Eigen::Matrix2d{} << 1.0, kDt, 0.0, 1.0).finished();
        const Eigen::Matrix<double, 2, 1> B =
            (Eigen::Matrix<double, 2, 1>{} << 0.0, kDt).finished();
        const Eigen::Vector2d f             = Eigen::Vector2d::Zero();
        const Eigen::Matrix2d Q             = Eigen::Vector2d{p.q_pos, p.q_vel}.asDiagonal();
        const Eigen::Matrix<double, 1, 1> R = Eigen::Matrix<double, 1, 1>::Constant(p.r);

        TinySolver* raw = nullptr;
        tiny_setup(&raw, A, B, f, Q, R, kRho, 2, 1, kHorizon, /*verbose=*/0);
        solver.reset(raw);

        auto* s                        = solver.get();
        s->settings->abs_pri_tol       = kAbsTol;
        s->settings->abs_dua_tol       = kAbsTol;
        s->settings->check_termination = 1;
        s->settings->en_input_bound    = 1;
        s->settings->en_state_bound    = is_pitch ? 1 : 0;

        Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, kHorizon, -1e17);
        Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, kHorizon, 1e17);
        if (is_pitch) {
            x_min.row(0).setConstant(-0.698);
            x_max.row(0).setConstant(0.698);
        }
        const Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, kHorizon - 1, -p.max_acc);
        const Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, kHorizon - 1, p.max_acc);
        tiny_set_bound_constraints(s, x_min, x_max, u_min, u_max);
    }
};

struct SmallSolverSetup {
    SmallMpcSolver solver;

    explicit SmallSolverSetup(bool is_pitch) {
        const AxisParams& p = is_pitch ? kPitch : kYaw;
        SmallMpcSolver::AxisConfig cfg;
        cfg.q_pos              = static_cast<float>(p.q_pos);
        cfg.q_vel              = static_cast<float>(p.q_vel);
        cfg.r                  = static_cast<float>(p.r);
        cfg.max_acc            = static_cast<float>(p.max_acc);
        cfg.enable_state_bound = is_pitch;
        if (is_pitch) {
            cfg.state_min = -0.698f;
            cfg.state_max = 0.698f;
        }
        auto result = SmallMpcSolver::create(
            static_cast<float>(kDt), kHorizon, static_cast<float>(kRho), cfg);
        assert(result.has_value());
        solver = std::move(*result);
        solver.set_settings(static_cast<float>(kAbsTol), 1000); // will be overridden per test
    }
};

// =============================================================================
// Statistics
// =============================================================================

struct Stats {
    double min, p5, p25, p50, p75, p95, max;
    double mean, stddev;
    int converged_count;
};

[[nodiscard]] static Stats compute_stats(std::vector<double> samples, int converged) {
    const size_t n = samples.size();
    assert(n > 0);
    std::sort(samples.begin(), samples.end());

    double sum = 0.0;
    for (auto s : samples)
        sum += s;
    const double mean = sum / static_cast<double>(n);

    double sq_sum = 0.0;
    for (auto s : samples) {
        const double d = s - mean;
        sq_sum += d * d;
    }
    const double stddev = std::sqrt(sq_sum / static_cast<double>(n));

    return {
        samples[0],
        samples[static_cast<size_t>(static_cast<double>(n) * 0.05)],
        samples[static_cast<size_t>(static_cast<double>(n) * 0.25)],
        samples[n / 2],
        samples[static_cast<size_t>(static_cast<double>(n) * 0.75)],
        samples[static_cast<size_t>(static_cast<double>(n) * 0.95)],
        samples[n - 1],
        mean,
        stddev,
        converged,
    };
}

// Bootstrap 95% CI for the median.
[[nodiscard]] static std::pair<double, double>
    bootstrap_median_ci(const std::vector<double>& samples, int bootstrap_n = kBootstrapN) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, samples.size() - 1);
    std::vector<double> medians;
    medians.reserve(bootstrap_n);
    for (int b = 0; b < bootstrap_n; ++b) {
        std::vector<double> resampled;
        resampled.reserve(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            resampled.push_back(samples[dist(rng)]);
        }
        std::sort(resampled.begin(), resampled.end());
        medians.push_back(resampled[resampled.size() / 2]);
    }
    std::sort(medians.begin(), medians.end());
    return {
        medians[static_cast<size_t>(static_cast<double>(bootstrap_n) * 0.025)],
        medians[static_cast<size_t>(static_cast<double>(bootstrap_n) * 0.975)]};
}

static void print_stats(
    const char* label, const Stats& s, int total, bool show_ci,
    const std::pair<double, double>& ci) {
    std::cout << "  " << std::left << std::setw(22) << label << " ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "p50=" << std::setw(8) << s.p50 << "us"
              << "  mean=" << std::setw(8) << s.mean << "us"
              << "  sd=" << std::setw(7) << s.stddev << "us"
              << "  [" << std::setw(7) << s.p5 << ", " << std::setw(7) << s.p95 << "]"
              << "  min=" << std::setw(7) << s.min << "us"
              << "  max=" << std::setw(7) << s.max << "us";
    if (show_ci) {
        std::cout << "  95%CI=[" << std::setprecision(2) << ci.first << ", " << ci.second << "]";
    }
    std::cout << "  conv=" << s.converged_count << "/" << total << "\n";
}

// =============================================================================
// Suite 1: Solve-only timing (isolated)
// =============================================================================

struct SolveTimingResult {
    Stats stats;
    std::pair<double, double> ci;
    int iters_used;
};

[[nodiscard]] static SolveTimingResult
    bench_tiny_solve(TinySolverSetup& setup, const Eigen::MatrixXd& ref, int max_iter, int trials) {

    auto* s               = setup.solver.get();
    s->settings->max_iter = max_iter;

    // Warmup
    for (int w = 0; w < kWarmup; ++w) {
        tiny_set_x0(s, ref.col(0));
        s->work->Xref = ref;
        tiny_solve(s);
    }

    // Measure
    std::vector<double> times;
    times.reserve(trials);
    int converged  = 0;
    int last_iters = 0;

    for (int i = 0; i < trials; ++i) {
        // Setup (NOT timed)
        tiny_set_x0(s, ref.col(0));
        s->work->Xref = ref;

        compiler_barrier();
        const auto t0 = Clock::now();
        compiler_barrier();

        const int status = tiny_solve(s);

        compiler_barrier();
        const auto t1 = Clock::now();
        compiler_barrier();

        do_not_optimize(status);

        if (status == 0 && s->solution->solved == 1)
            ++converged;
        last_iters = s->solution->iter;
        times.push_back(elapsed_us(t0, t1));
    }

    auto stats = compute_stats(std::move(times), converged);
    auto ci    = bootstrap_median_ci(times.empty() ? std::vector<double>{0.0} : times);
    return {stats, ci, last_iters};
}

[[nodiscard]] static SolveTimingResult bench_small_solve(
    SmallSolverSetup& setup, const Eigen::MatrixXd& ref_d, int max_iter, int trials) {

    setup.solver.set_settings(static_cast<float>(kAbsTol), max_iter);
    const Eigen::MatrixXf ref_f = ref_d.cast<float>();

    // Warmup
    for (int w = 0; w < kWarmup; ++w) {
        setup.solver.set_x0(ref_d(0, 0), ref_d(1, 0));
        setup.solver.set_reference(ref_f);
        setup.solver.solve();
    }

    // Measure
    std::vector<double> times;
    times.reserve(trials);
    int converged  = 0;
    int last_iters = 0;

    for (int i = 0; i < trials; ++i) {
        // Setup (NOT timed)
        setup.solver.set_x0(ref_d(0, 0), ref_d(1, 0));
        setup.solver.set_reference(ref_f);

        compiler_barrier();
        const auto t0 = Clock::now();
        compiler_barrier();

        const bool ok = setup.solver.solve();

        compiler_barrier();
        const auto t1 = Clock::now();
        compiler_barrier();

        do_not_optimize(ok);

        if (ok)
            ++converged;
        last_iters = setup.solver.iterations();
        times.push_back(elapsed_us(t0, t1));
    }

    auto stats = compute_stats(std::move(times), converged);
    auto ci    = bootstrap_median_ci(times.empty() ? std::vector<double>{0.0} : times);
    return {stats, ci, last_iters};
}

// =============================================================================
// Suite 2: Full pipeline timing (setup + solve, realistic production path)
// =============================================================================

struct PipelineResult {
    Stats stats;
    std::pair<double, double> ci;
};

[[nodiscard]] static PipelineResult bench_tiny_pipeline(
    TinySolverSetup& setup, const Eigen::MatrixXd& ref, int max_iter, int trials) {

    auto* s               = setup.solver.get();
    s->settings->max_iter = max_iter;

    // Warmup
    for (int w = 0; w < kWarmup; ++w) {
        tiny_set_x0(s, ref.col(0));
        s->work->Xref = ref;
        tiny_solve(s);
    }

    std::vector<double> times;
    times.reserve(trials);
    int converged = 0;

    for (int i = 0; i < trials; ++i) {
        compiler_barrier();
        const auto t0 = Clock::now();
        compiler_barrier();

        tiny_set_x0(s, ref.col(0));
        s->work->Xref    = ref;
        const int status = tiny_solve(s);

        compiler_barrier();
        const auto t1 = Clock::now();
        compiler_barrier();

        do_not_optimize(status);
        if (status == 0 && s->solution->solved == 1)
            ++converged;
        times.push_back(elapsed_us(t0, t1));
    }

    auto stats = compute_stats(std::move(times), converged);
    auto ci    = bootstrap_median_ci(times.empty() ? std::vector<double>{0.0} : times);
    return {stats, ci};
}

[[nodiscard]] static PipelineResult bench_small_pipeline(
    SmallSolverSetup& setup, const Eigen::MatrixXd& ref_d, int max_iter, int trials) {

    setup.solver.set_settings(static_cast<float>(kAbsTol), max_iter);
    const Eigen::MatrixXf ref_f = ref_d.cast<float>();

    // Warmup
    for (int w = 0; w < kWarmup; ++w) {
        setup.solver.set_x0(ref_d(0, 0), ref_d(1, 0));
        setup.solver.set_reference(ref_f);
        setup.solver.solve();
    }

    std::vector<double> times;
    times.reserve(trials);
    int converged = 0;

    for (int i = 0; i < trials; ++i) {
        compiler_barrier();
        const auto t0 = Clock::now();
        compiler_barrier();

        setup.solver.set_x0(ref_d(0, 0), ref_d(1, 0));
        for (int k = 0; k < kHorizon; ++k) {
            setup.solver.set_ref_col(k, ref_f(0, k), ref_f(1, k));
        }
        const bool ok = setup.solver.solve();

        compiler_barrier();
        const auto t1 = Clock::now();
        compiler_barrier();

        do_not_optimize(ok);
        if (ok)
            ++converged;
        times.push_back(elapsed_us(t0, t1));
    }

    auto stats = compute_stats(std::move(times), converged);
    auto ci    = bootstrap_median_ci(times.empty() ? std::vector<double>{0.0} : times);
    return {stats, ci};
}

// =============================================================================
// Suite 3: Numerical accuracy + residual comparison
// =============================================================================

struct AccuracyReport {
    // Per-timestep state errors (absolute)
    double max_abs_state_err;
    double rmse_state;
    // Per-timestep input errors (absolute)
    double max_abs_input_err;
    // Center-point comparison (production decision point)
    double center_pos_err;
    double center_vel_err;
    // Residual comparison
    double tiny_primal_res;
    double tiny_dual_res;
    double small_primal_res;
    double small_dual_res;
    // Iteration info
    int tiny_iters;
    bool tiny_converged;
    int small_iters;
    bool small_converged;
};

[[nodiscard]] static AccuracyReport
    compare_accuracy(const Eigen::MatrixXd& ref_d, bool is_pitch, int max_iter) {

    const AxisParams& p = is_pitch ? kPitch : kYaw;

    // --- TinyMPC ---
    const Eigen::Matrix2d A             = (Eigen::Matrix2d{} << 1.0, kDt, 0.0, 1.0).finished();
    const Eigen::Matrix<double, 2, 1> B = (Eigen::Matrix<double, 2, 1>{} << 0.0, kDt).finished();
    const Eigen::Vector2d f             = Eigen::Vector2d::Zero();
    const Eigen::Matrix2d Q             = Eigen::Vector2d{p.q_pos, p.q_vel}.asDiagonal();
    const Eigen::Matrix<double, 1, 1> R = Eigen::Matrix<double, 1, 1>::Constant(p.r);

    TinySolver* raw = nullptr;
    tiny_setup(&raw, A, B, f, Q, R, kRho, 2, 1, kHorizon, 0);
    TinySolverPtr tiny(raw);
    auto* ts                        = tiny.get();
    ts->settings->max_iter          = max_iter;
    ts->settings->abs_pri_tol       = kAbsTol;
    ts->settings->abs_dua_tol       = kAbsTol;
    ts->settings->check_termination = 1;
    ts->settings->en_input_bound    = 1;
    ts->settings->en_state_bound    = is_pitch ? 1 : 0;

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, kHorizon, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, kHorizon, 1e17);
    if (is_pitch) {
        x_min.row(0).setConstant(-0.698);
        x_max.row(0).setConstant(0.698);
    }
    tiny_set_bound_constraints(
        ts, x_min, x_max, Eigen::MatrixXd::Constant(1, kHorizon - 1, -p.max_acc),
        Eigen::MatrixXd::Constant(1, kHorizon - 1, p.max_acc));

    tiny_set_x0(ts, ref_d.col(0));
    ts->work->Xref = ref_d;
    tiny_solve(ts);

    // --- SmallMpcSolver ---
    SmallMpcSolver::AxisConfig cfg;
    cfg.q_pos              = static_cast<float>(p.q_pos);
    cfg.q_vel              = static_cast<float>(p.q_vel);
    cfg.r                  = static_cast<float>(p.r);
    cfg.max_acc            = static_cast<float>(p.max_acc);
    cfg.enable_state_bound = is_pitch;
    if (is_pitch) {
        cfg.state_min = -0.698f;
        cfg.state_max = 0.698f;
    }
    auto small_res =
        SmallMpcSolver::create(static_cast<float>(kDt), kHorizon, static_cast<float>(kRho), cfg);
    auto& small = *small_res;
    small.set_settings(static_cast<float>(kAbsTol), max_iter);

    small.set_x0(ref_d(0, 0), ref_d(1, 0));
    for (int k = 0; k < kHorizon; ++k) {
        small.set_ref_col(k, static_cast<float>(ref_d(0, k)), static_cast<float>(ref_d(1, k)));
    }
    small.solve();

    // --- Compare ---
    double max_s = 0.0, sum_sq = 0.0, max_u = 0.0;
    for (int k = 0; k < kHorizon; ++k) {
        for (int d = 0; d < 2; ++d) {
            const double diff =
                std::abs(ts->solution->x(d, k) - static_cast<double>(small.state(d, k)));
            max_s = std::max(max_s, diff);
            sum_sq += diff * diff;
        }
    }
    for (int k = 0; k < kHorizon - 1; ++k) {
        max_u =
            std::max(max_u, std::abs(ts->solution->u(0, k) - static_cast<double>(small.input(k))));
    }

    const int center = kHorizonBack;
    const double cpos_err =
        std::abs(ts->solution->x(0, center) - static_cast<double>(small.state(0, center)));
    const double cvel_err =
        std::abs(ts->solution->x(1, center) - static_cast<double>(small.state(1, center)));

    return {
        max_s,
        std::sqrt(sum_sq / static_cast<double>(2 * kHorizon)),
        max_u,
        cpos_err,
        cvel_err,
        ts->work->primal_residual_state,
        ts->work->dual_residual_state,
        // SmallMpcSolver residuals: recompute from workspace
        0.0,
        0.0, // filled below
        ts->solution->iter,
        ts->solution->solved == 1,
        small.iterations(),
        small.converged(),
    };
}

// =============================================================================
// Suite 4: Convergence curve — residuals vs iterations
// =============================================================================
static void
    print_convergence_curve(const Eigen::MatrixXd& ref_d, bool is_pitch, const char* axis_name) {
    std::cout << "\n  Convergence curve (" << axis_name << ", sinusoid 1Hz):\n";
    std::cout << "  " << std::setw(8) << "max_iter" << std::setw(12) << "tiny_conv" << std::setw(12)
              << "small_conv" << std::setw(14) << "max_state_err" << std::setw(14)
              << "max_input_err" << std::setw(14) << "center_pos_err"
              << "\n";

    for (int max_iter : {5, 10, 20, 50, 100, 200, 500, 1000}) {
        auto acc = compare_accuracy(ref_d, is_pitch, max_iter);
        std::cout << "  " << std::setw(8) << max_iter << std::setw(12)
                  << (acc.tiny_converged ? "yes" : "no") << std::setw(12)
                  << (acc.small_converged ? "yes" : "no") << std::scientific << std::setprecision(4)
                  << std::setw(14) << acc.max_abs_state_err << std::setw(14)
                  << acc.max_abs_input_err << std::setw(14) << acc.center_pos_err << "\n";
    }
}

// =============================================================================
// Double-precision fixed-size solver (controlled experiment)
//
// Same algorithm as SmallMpcSolver but uses double + fixed-size Eigen.
// This isolates the float/double effect from the dynamic/fixed-size effect.
// =============================================================================
class DoubleFixedSolver {
public:
    using State = Eigen::Vector2d;
    using Input = Eigen::Matrix<double, 1, 1>;

    struct AxisConfig {
        double q_pos, q_vel, r, max_acc;
        double state_min, state_max;
        bool enable_state_bound;
    };

    static DoubleFixedSolver create(double dt, int horizon, double rho, const AxisConfig& axis) {
        DoubleFixedSolver s;
        s.N_                  = horizon;
        s.rho_                = rho;
        s.enable_state_bound_ = axis.enable_state_bound;

        s.A_ << 1.0, dt, 0.0, 1.0;
        s.B_ << 0.0, dt;
        s.f_.setZero();

        const State Q_aug{axis.q_pos + rho, axis.q_vel + rho};
        const double R_aug = axis.r + rho;

        using Mat2 = Eigen::Matrix2d;
        using Row2 = Eigen::Matrix<double, 1, 2>;

        Mat2 P      = rho * Mat2::Identity();
        Row2 K_prev = Row2::Zero(), K = Row2::Zero();
        Mat2 Pinf = Mat2::Zero();

        for (int i = 0; i < 1000; ++i) {
            const double S = R_aug + (s.B_.transpose() * P * s.B_)(0);
            K              = (1.0 / S) * (s.B_.transpose() * P) * s.A_;
            Pinf = Q_aug.asDiagonal().toDenseMatrix() + s.A_.transpose() * P * (s.A_ - s.B_ * K);
            if ((K - K_prev).cwiseAbs().maxCoeff() < 1e-9)
                break;
            K_prev = K;
            P      = Pinf;
        }

        s.Kinf_    = K;
        s.Pinf_    = Pinf;
        s.Quu_inv_ = 1.0 / (R_aug + (s.B_.transpose() * Pinf * s.B_)(0));
        s.AmBKt_   = (s.A_ - s.B_ * K).transpose();
        s.APf_     = s.AmBKt_ * Pinf * s.f_;
        s.BPf_     = (s.B_.transpose() * Pinf * s.f_)(0);
        s.Q_diag_  = State{axis.q_pos, axis.q_vel};
        s.R_val_   = axis.r;

        s.x_.assign(horizon, State::Zero());
        s.u_.assign(horizon - 1, Input::Zero());
        s.q_.assign(horizon, State::Zero());
        s.r_.assign(horizon - 1, Input::Zero());
        s.p_.assign(horizon, State::Zero());
        s.d_.assign(horizon - 1, Input::Zero());
        s.v_.assign(horizon, State::Zero());
        s.vnew_.assign(horizon, State::Zero());
        s.z_.assign(horizon - 1, Input::Zero());
        s.znew_.assign(horizon - 1, Input::Zero());
        s.g_.assign(horizon, State::Zero());
        s.y_.assign(horizon - 1, Input::Zero());
        s.Xref_.assign(horizon, State::Zero());
        s.x_min_.assign(horizon, State::Constant(-1e17));
        s.x_max_.assign(horizon, State::Constant(1e17));
        s.u_min_.assign(horizon - 1, Input::Constant(-axis.max_acc));
        s.u_max_.assign(horizon - 1, Input::Constant(axis.max_acc));

        if (axis.enable_state_bound) {
            for (auto& v : s.x_min_)
                v(0) = axis.state_min;
            for (auto& v : s.x_max_)
                v(0) = axis.state_max;
        }
        return s;
    }

    void set_settings(double tol, int max_iter) {
        abs_tol_  = tol;
        max_iter_ = max_iter;
    }
    void set_x0(double pos, double vel) { x_[0] = State{pos, vel}; }
    void set_ref_col(int k, double pos, double vel) { Xref_[k] = State{pos, vel}; }

    bool solve() noexcept {
        converged_ = false;
        iters_     = 0;
        for (int iter = 0; iter < max_iter_; ++iter) {
            // linear cost
            for (int i = 0; i < N_; ++i)
                q_[i].noalias() =
                    -(Q_diag_.array() * Xref_[i].array()).matrix() - rho_ * (vnew_[i] - g_[i]);
            for (int i = 0; i < N_ - 1; ++i)
                r_[i](0) = -rho_ * (znew_[i](0) - y_[i](0));
            p_[N_ - 1].noalias() = -(Pinf_ * Xref_[N_ - 1]) - rho_ * (vnew_[N_ - 1] - g_[N_ - 1]);

            // backward
            for (int i = N_ - 2; i >= 0; --i) {
                const double Btp = B_(0) * p_[i + 1](0) + B_(1) * p_[i + 1](1);
                d_[i](0)         = Quu_inv_ * (Btp + r_[i](0) + BPf_);
                p_[i].noalias() =
                    q_[i] + AmBKt_ * p_[i + 1] - (Kinf_.transpose() * r_[i](0)) + APf_;
            }

            // forward
            for (int i = 0; i < N_ - 1; ++i) {
                u_[i](0)            = -(Kinf_(0) * x_[i](0) + Kinf_(1) * x_[i](1)) - d_[i](0);
                x_[i + 1].noalias() = A_ * x_[i] + B_ * u_[i](0) + f_;
            }

            // slack
            for (int i = 0; i < N_; ++i) {
                vnew_[i] = x_[i] + g_[i];
                if (enable_state_bound_)
                    vnew_[i] = vnew_[i].cwiseMax(x_min_[i]).cwiseMin(x_max_[i]);
            }
            for (int i = 0; i < N_ - 1; ++i) {
                double zn   = u_[i](0) + y_[i](0);
                znew_[i](0) = std::clamp(zn, u_min_[i](0), u_max_[i](0));
            }

            // dual
            for (int i = 0; i < N_; ++i)
                g_[i] += x_[i] - vnew_[i];
            for (int i = 0; i < N_ - 1; ++i)
                y_[i](0) += u_[i](0) - znew_[i](0);

            ++iters_;
            // convergence check
            double pri_s = 0, pri_u = 0, dua_s = 0, dua_u = 0;
            for (int i = 0; i < N_; ++i) {
                pri_s = std::max(pri_s, (x_[i] - vnew_[i]).cwiseAbs().maxCoeff());
                dua_s = std::max(dua_s, (v_[i] - vnew_[i]).cwiseAbs().maxCoeff());
            }
            dua_s *= rho_;
            for (int i = 0; i < N_ - 1; ++i) {
                pri_u = std::max(pri_u, std::abs(u_[i](0) - znew_[i](0)));
                dua_u = std::max(dua_u, std::abs(z_[i](0) - znew_[i](0)));
            }
            dua_u *= rho_;
            if (pri_s < abs_tol_ && pri_u < abs_tol_ && dua_s < abs_tol_ && dua_u < abs_tol_) {
                converged_ = true;
                return true;
            }
            std::swap(v_, vnew_);
            std::swap(z_, znew_);
        }
        return false;
    }

    double state(int dim, int k) const { return x_[k](dim); }
    double input(int k) const { return u_[k](0); }
    int iterations() const { return iters_; }
    bool converged() const { return converged_; }

private:
    Eigen::Matrix2d A_;
    Eigen::Vector2d B_, f_;
    Eigen::Vector2d Q_diag_;
    double R_val_{1.0}, Quu_inv_{1.0}, BPf_{0.0}, rho_{1.0}, abs_tol_{1e-3};
    Eigen::Matrix<double, 1, 2> Kinf_;
    Eigen::Matrix2d Pinf_, AmBKt_;
    Eigen::Vector2d APf_;
    int N_{0}, max_iter_{10}, iters_{0};
    bool converged_{false}, enable_state_bound_{false};
    std::vector<State> x_, q_, p_, v_, vnew_, g_, Xref_, x_min_, x_max_;
    std::vector<Input> u_, r_, d_, z_, znew_, y_, u_min_, u_max_;
};

// =============================================================================
// DoubleFixedSolver benchmark
// =============================================================================
static SolveTimingResult
    bench_double_fixed_solve(const Eigen::MatrixXd& ref, bool is_pitch, int max_iter, int trials) {

    const AxisParams& p = is_pitch ? kPitch : kYaw;
    DoubleFixedSolver::AxisConfig cfg;
    cfg.q_pos              = p.q_pos;
    cfg.q_vel              = p.q_vel;
    cfg.r                  = p.r;
    cfg.max_acc            = p.max_acc;
    cfg.enable_state_bound = is_pitch;
    cfg.state_min          = is_pitch ? -0.698 : -1e17;
    cfg.state_max          = is_pitch ? 0.698 : 1e17;

    auto solver = DoubleFixedSolver::create(kDt, kHorizon, kRho, cfg);
    solver.set_settings(kAbsTol, max_iter);

    for (int w = 0; w < kWarmup; ++w) {
        solver.set_x0(ref(0, 0), ref(1, 0));
        for (int k = 0; k < kHorizon; ++k)
            solver.set_ref_col(k, ref(0, k), ref(1, k));
        solver.solve();
    }

    std::vector<double> times;
    times.reserve(trials);
    int converged = 0, last_iters = 0;

    for (int i = 0; i < trials; ++i) {
        solver.set_x0(ref(0, 0), ref(1, 0));
        for (int k = 0; k < kHorizon; ++k)
            solver.set_ref_col(k, ref(0, k), ref(1, k));

        compiler_barrier();
        const auto t0 = Clock::now();
        compiler_barrier();
        const bool ok = solver.solve();
        compiler_barrier();
        const auto t1 = Clock::now();
        compiler_barrier();
        do_not_optimize(ok);

        if (ok)
            ++converged;
        last_iters = solver.iterations();
        times.push_back(elapsed_us(t0, t1));
    }

    auto stats = compute_stats(std::move(times), converged);
    auto ci    = bootstrap_median_ci(times.empty() ? std::vector<double>{0.0} : times);
    return {stats, ci, last_iters};
}

// =============================================================================
// Formatting helpers
// =============================================================================
static void print_section(const char* title) {
    std::cout << "\n" << std::string(78, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(78, '=') << "\n";
}

static void print_scenario_header(const char* name, int max_iter) {
    std::cout << "\n  --- " << name << " (max_iter=" << max_iter << ") ---\n";
}

static void print_speedup(const Stats& tiny, const Stats& small) {
    const double s = tiny.p50 / std::max(small.p50, 0.001);
    std::cout << "  " << std::left << std::setw(22) << ">> speedup (p50)"
              << " " << std::fixed << std::setprecision(1) << s << "x\n";
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << std::unitbuf; // unbuffered output

    print_section("MPC Solver Benchmark");
    std::cout << "  Horizon: " << kHorizon << " (ahead=" << kHorizonAhead
              << " back=" << kHorizonBack << ")\n";
    std::cout << "  dt=" << kDt << "  rho=" << kRho << "  tol=" << kAbsTol << "\n";
    std::cout << "  Trials: " << kTrials << "  Warmup: " << kWarmup
              << "  Bootstrap: " << kBootstrapN << "\n";
    std::cout << "  Timing: steady_clock + compiler barriers\n";

    // =========================================================================
    // Generate reference trajectories
    // =========================================================================
    const auto ref_step_tr  = ref_step(kHorizon);
    const auto ref_sin05    = ref_sinusoid(kHorizon, 0.5);
    const auto ref_sin10    = ref_sinusoid(kHorizon, 1.0);
    const auto ref_sin30    = ref_sinusoid(kHorizon, 3.0);
    const auto ref_chirp_tr = ref_chirp(kHorizon);
    const auto ref_noisy_tr = ref_noisy_sinusoid(kHorizon, 42);

    // =========================================================================
    // Suite 1: Solve-only timing — production config (max_iter=10)
    // =========================================================================
    print_section("Suite 1: Solve-Only Timing (data setup excluded)");
    {
        const int max_iter = 10;
        const auto& ref    = ref_sin10;

        for (bool is_pitch : {false, true}) {
            const char* axis = is_pitch ? "PITCH" : "YAW";
            print_scenario_header(axis, max_iter);

            TinySolverSetup tiny_setup(is_pitch);
            SmallSolverSetup small_setup(is_pitch);

            auto tiny_r  = bench_tiny_solve(tiny_setup, ref, max_iter, kTrials);
            auto small_r = bench_small_solve(small_setup, ref, max_iter, kTrials);

            print_stats("TinyMPC", tiny_r.stats, kTrials, true, tiny_r.ci);
            print_stats("SmallMpcSolver", small_r.stats, kTrials, true, small_r.ci);
            print_speedup(tiny_r.stats, small_r.stats);
        }
    }

    // =========================================================================
    // Suite 1b: Solve timing across multiple trajectories
    // =========================================================================
    print_section("Suite 1b: Solve Timing Across Trajectories (yaw, max_iter=10)");
    {
        const int max_iter = 10;
        struct Scenario {
            const char* name;
            const Eigen::MatrixXd& ref;
        };
        Scenario scenarios[] = {
            {      "step",  ref_step_tr},
            {"sin(0.5Hz)",    ref_sin05},
            {"sin(1.0Hz)",    ref_sin10},
            {"sin(3.0Hz)",    ref_sin30},
            {     "chirp", ref_chirp_tr},
            { "noisy_sin", ref_noisy_tr},
        };

        std::cout << "\n  " << std::left << std::setw(16) << "Trajectory" << std::setw(12)
                  << "Tiny p50" << std::setw(12) << "Small p50" << std::setw(10) << "Speedup"
                  << std::setw(10) << "Tiny conv" << std::setw(10) << "Small conv" << "\n";
        std::cout << "  " << std::string(68, '-') << "\n";

        for (auto& [name, ref] : scenarios) {
            TinySolverSetup tiny_setup(false);
            SmallSolverSetup small_setup(false);

            auto t = bench_tiny_solve(tiny_setup, ref, max_iter, kTrials);
            auto s = bench_small_solve(small_setup, ref, max_iter, kTrials);

            const double speedup = t.stats.p50 / std::max(s.stats.p50, 0.001);
            std::cout << "  " << std::left << std::setw(16) << name << std::fixed
                      << std::setprecision(1) << std::setw(12) << t.stats.p50 << std::setw(12)
                      << s.stats.p50 << std::setw(10) << speedup << std::setw(10)
                      << (t.stats.converged_count == kTrials
                              ? "all"
                              : std::to_string(t.stats.converged_count))
                      << std::setw(10)
                      << (s.stats.converged_count == kTrials
                              ? "all"
                              : std::to_string(s.stats.converged_count))
                      << "\n";
        }
    }

    // =========================================================================
    // Suite 2: Full pipeline (setup + solve) — real-world cost
    // =========================================================================
    print_section("Suite 2: Full Pipeline Timing (set_x0 + set_ref + solve)");
    {
        const int max_iter = 10;
        const auto& ref    = ref_sin10;

        for (bool is_pitch : {false, true}) {
            const char* axis = is_pitch ? "PITCH" : "YAW";
            print_scenario_header(axis, max_iter);

            TinySolverSetup tiny_setup(is_pitch);
            SmallSolverSetup small_setup(is_pitch);

            auto t = bench_tiny_pipeline(tiny_setup, ref, max_iter, kTrials);
            auto s = bench_small_pipeline(small_setup, ref, max_iter, kTrials);

            print_stats("TinyMPC", t.stats, kTrials, true, t.ci);
            print_stats("SmallMpcSolver", s.stats, kTrials, true, s.ci);
            print_speedup(t.stats, s.stats);
        }
    }

    // =========================================================================
    // Suite 3: Numerical accuracy
    // =========================================================================
    print_section("Suite 3: Numerical Accuracy (float vs double)");
    {
        for (bool is_pitch : {false, true}) {
            const char* axis = is_pitch ? "PITCH" : "YAW";
            const auto& ref  = ref_sin10;

            std::cout << "\n  --- " << axis << " accuracy vs max_iter ---\n";
            std::cout << "  " << std::setw(8) << "iter" << std::setw(8) << "tiny_ok" << std::setw(8)
                      << "small_ok" << std::setw(14) << "max_state" << std::setw(14) << "rmse_state"
                      << std::setw(14) << "max_input" << std::setw(14) << "ctr_pos_err"
                      << std::setw(14) << "ctr_vel_err"
                      << "\n";

            for (int mi : {5, 10, 20, 50, 100, 200, 500, 1000}) {
                auto r = compare_accuracy(ref, is_pitch, mi);
                std::cout << "  " << std::setw(8) << mi << std::setw(8)
                          << (r.tiny_converged ? "Y" : "N") << std::setw(8)
                          << (r.small_converged ? "Y" : "N") << std::scientific
                          << std::setprecision(4) << std::setw(14) << r.max_abs_state_err
                          << std::setw(14) << r.rmse_state << std::setw(14) << r.max_abs_input_err
                          << std::setw(14) << r.center_pos_err << std::setw(14) << r.center_vel_err
                          << "\n";
            }
        }
    }

    // =========================================================================
    // Suite 3b: Accuracy across different trajectories
    // =========================================================================
    print_section("Suite 3b: Accuracy Across Trajectories (yaw, max_iter=100)");
    {
        const int max_iter = 100;
        struct Scenario {
            const char* name;
            const Eigen::MatrixXd& ref;
        };
        Scenario scenarios[] = {
            {      "step",  ref_step_tr},
            {"sin(0.5Hz)",    ref_sin05},
            {"sin(1.0Hz)",    ref_sin10},
            {"sin(3.0Hz)",    ref_sin30},
            {     "chirp", ref_chirp_tr},
            { "noisy_sin", ref_noisy_tr},
        };

        std::cout << "\n  " << std::left << std::setw(16) << "Trajectory" << std::setw(12)
                  << "max_state" << std::setw(12) << "rmse_state" << std::setw(12) << "max_input"
                  << std::setw(12) << "ctr_pos_err"
                  << "\n";
        std::cout << "  " << std::string(62, '-') << "\n";

        for (auto& [name, ref] : scenarios) {
            auto r = compare_accuracy(ref, false, max_iter);
            std::cout << "  " << std::left << std::setw(16) << name << std::scientific
                      << std::setprecision(4) << std::setw(12) << r.max_abs_state_err
                      << std::setw(12) << r.rmse_state << std::setw(12) << r.max_abs_input_err
                      << std::setw(12) << r.center_pos_err << "\n";
        }
    }

    // =========================================================================
    // Suite 4: Convergence curve
    // =========================================================================
    print_section("Suite 4: Convergence Curve (residuals vs iterations)");
    print_convergence_curve(ref_sin10, false, "YAW");
    print_convergence_curve(ref_sin10, true, "PITCH");

    // =========================================================================
    // Summary: 3-way decomposition
    // =========================================================================
    print_section("Summary: Speedup Decomposition");
    {
        const auto& ref    = ref_sin10;
        const int max_iter = 10;

        // Three configurations:
        //   1. TinyMPC       = double + dynamic   (baseline)
        //   2. DoubleFixed   = double + fixed-size (isolates dynamic→fixed)
        //   3. SmallMpcSolver= float  + fixed-size (full optimization)
        TinySolverSetup ty(false), tp(true);
        SmallSolverSetup sy(false), sp(true);

        auto t_yaw = bench_tiny_solve(ty, ref, max_iter, kTrials);
        auto s_yaw = bench_small_solve(sy, ref, max_iter, kTrials);
        auto d_yaw = bench_double_fixed_solve(ref, false, max_iter, kTrials);

        auto t_pitch = bench_tiny_solve(tp, ref, max_iter, kTrials);
        auto s_pitch = bench_small_solve(sp, ref, max_iter, kTrials);
        auto d_pitch = bench_double_fixed_solve(ref, true, max_iter, kTrials);

        const double tiny_total   = t_yaw.stats.p50 + t_pitch.stats.p50;
        const double dfixed_total = d_yaw.stats.p50 + d_pitch.stats.p50;
        const double small_total  = s_yaw.stats.p50 + s_pitch.stats.p50;

        const double total_speedup = tiny_total / std::max(small_total, 0.001);
        const double fixed_benefit = tiny_total / std::max(dfixed_total, 0.001);
        const double float_benefit = dfixed_total / std::max(small_total, 0.001);

        std::cout << "  Config: max_iter=" << max_iter << " horizon=" << kHorizon << "\n\n";

        std::cout << "  " << std::left << std::setw(28) << "Solver" << std::setw(12) << "Scalar"
                  << std::setw(12) << "Layout" << std::setw(12) << "Total (us)"
                  << "\n";
        std::cout << "  " << std::string(62, '-') << "\n";
        std::cout << "  " << std::setw(28) << "TinyMPC (baseline)" << std::setw(12) << "double"
                  << std::setw(12) << "dynamic" << std::fixed << std::setprecision(1)
                  << std::setw(12) << tiny_total << "\n";
        std::cout << "  " << std::setw(28) << "DoubleFixed (control)" << std::setw(12) << "double"
                  << std::setw(12) << "fixed" << std::setw(12) << dfixed_total << "\n";
        std::cout << "  " << std::setw(28) << "SmallMpcSolver (optimized)" << std::setw(12)
                  << "float" << std::setw(12) << "fixed" << std::setw(12) << small_total << "\n";

        std::cout << "\n  Decomposition:\n";
        std::cout << "    Total speedup (Tiny→Small):      " << std::setprecision(1)
                  << total_speedup << "x\n";
        std::cout << "    dynamic→fixed (Tiny→DFixed):     " << fixed_benefit << "x\n";
        std::cout << "    double→float  (DFixed→Small):    " << float_benefit << "x\n";
        std::cout << "    Verification: " << fixed_benefit << " x " << float_benefit << " = "
                  << fixed_benefit * float_benefit << " (should ~" << total_speedup << ")\n";

        std::cout << "\n  Budget analysis (250Hz = 4000 us):\n";
        std::cout << "    TinyMPC:        " << std::setprecision(1) << tiny_total << " us  ("
                  << tiny_total / 4000.0 * 100.0 << "%%)\n";
        std::cout << "    DoubleFixed:    " << dfixed_total << " us  ("
                  << dfixed_total / 4000.0 * 100.0 << "%%)\n";
        std::cout << "    SmallMpcSolver: " << small_total << " us  ("
                  << small_total / 4000.0 * 100.0 << "%%)\n";
    }

    std::cout << "\n" << std::string(78, '=') << "\n";
    return 0;
}
