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

[[maybe_unused]] static inline void compiler_barrier() noexcept { asm volatile("" ::: "memory"); }

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

[[nodiscard]] static RefTrajectory ref_noisy_sinusoid(int N, unsigned seed, double amp) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    auto ref = ref_sinusoid(N, 1.0, amp);
    for (int i = 0; i < N; ++i) {
        ref(0, i) += static_cast<double>(noise(rng));
        ref(1, i) += static_cast<double>(noise(rng)) * 10.0;
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
        const AxisParams& p     = is_pitch ? kPitch : kYaw;
        const Eigen::Matrix2d A = (Eigen::Matrix2d{} << 1.0, kDt, 0.0, 1.0).finished();
        const Eigen::Matrix<double, 2, 1> B =
            (Eigen::Matrix<double, 2, 1>{} << 0.0, kDt).finished();
        const Eigen::Vector2d f             = Eigen::Vector2d::Zero();
        const Eigen::Matrix2d Q             = Eigen::Vector2d{p.q_pos, p.q_vel}.asDiagonal();
        const Eigen::Matrix<double, 1, 1> R = Eigen::Matrix<double, 1, 1>::Constant(p.r);

        TinySolver* raw = nullptr;
        tiny_setup(&raw, A, B, f, Q, R, kRho, 2, 1, kHorizon, 0);
        TinySolverPtr solver(raw);
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
        tiny_set_bound_constraints(
            s, x_min, x_max, Eigen::MatrixXd::Constant(1, kHorizon - 1, -p.max_acc),
            Eigen::MatrixXd::Constant(1, kHorizon - 1, p.max_acc));
        return solver;
    }
};

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
            static_cast<float>(kDt), kHorizon, static_cast<float>(kRho), yaw_cfg, pitch_cfg);
        assert(result.has_value());
        solver = std::move(*result);
        solver.set_settings(static_cast<float>(kAbsTol), 1000);
    }
};

struct Stats {
    double min, p50, p95, max;
    double mean, stddev;
    int converged_count;
};

[[nodiscard]] static Stats compute_stats(std::vector<double> samples, int converged) {
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
        samples.front(),
        samples[samples.size() / 2],
        samples[static_cast<size_t>(static_cast<double>(samples.size()) * 0.95)],
        samples.back(),
        mean,
        std::sqrt(sq_sum / static_cast<double>(samples.size())),
        converged,
    };
}

static void print_stats(const char* label, const Stats& s, int total) {
    std::cout << "  " << std::left << std::setw(24) << label << " ";
    std::cout << std::fixed << std::setprecision(2) << "p50=" << std::setw(8) << s.p50 << "us"
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
    TinyPairSetup& setup, const Eigen::MatrixXd& yaw_ref, const Eigen::MatrixXd& pitch_ref,
    int max_iter, int trials) {
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

    std::vector<double> times;
    times.reserve(trials);
    int converged  = 0;
    int iterations = 0;

    for (int i = 0; i < trials; ++i) {
        tiny_set_x0(yaw, yaw_ref.col(0));
        tiny_set_x0(pitch, pitch_ref.col(0));
        yaw->work->Xref   = yaw_ref;
        pitch->work->Xref = pitch_ref;

        compiler_barrier();
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

[[nodiscard]] static TimingResult bench_dual_solve(
    DualSetup& setup, const Eigen::MatrixXd& yaw_ref_d, const Eigen::MatrixXd& pitch_ref_d,
    int max_iter, int trials) {
    setup.solver.set_settings(static_cast<float>(kAbsTol), max_iter);
    const Eigen::MatrixXf yaw_ref   = yaw_ref_d.cast<float>();
    const Eigen::MatrixXf pitch_ref = pitch_ref_d.cast<float>();

    for (int w = 0; w < kWarmup; ++w) {
        setup.solver.set_x0(yaw_ref_d(0, 0), yaw_ref_d(1, 0), pitch_ref_d(0, 0), pitch_ref_d(1, 0));
        setup.solver.set_reference(yaw_ref, pitch_ref);
        setup.solver.solve();
    }

    std::vector<double> times;
    times.reserve(trials);
    int converged  = 0;
    int iterations = 0;

    for (int i = 0; i < trials; ++i) {
        setup.solver.set_x0(yaw_ref_d(0, 0), yaw_ref_d(1, 0), pitch_ref_d(0, 0), pitch_ref_d(1, 0));
        setup.solver.set_reference(yaw_ref, pitch_ref);

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
        iterations = setup.solver.iterations();
        times.push_back(elapsed_us(t0, t1));
    }

    return {compute_stats(std::move(times), converged), iterations};
}

[[nodiscard]] static Stats bench_tiny_pair_pipeline(
    TinyPairSetup& setup, const Eigen::MatrixXd& yaw_ref, const Eigen::MatrixXd& pitch_ref,
    int max_iter, int trials) {
    auto* yaw                 = setup.yaw.get();
    auto* pitch               = setup.pitch.get();
    yaw->settings->max_iter   = max_iter;
    pitch->settings->max_iter = max_iter;

    std::vector<double> times;
    times.reserve(trials);
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
            ++converged;
        }
        times.push_back(elapsed_us(t0, t1));
    }

    return compute_stats(std::move(times), converged);
}

[[nodiscard]] static Stats bench_dual_pipeline(
    DualSetup& setup, const Eigen::MatrixXd& yaw_ref_d, const Eigen::MatrixXd& pitch_ref_d,
    int max_iter, int trials) {
    setup.solver.set_settings(static_cast<float>(kAbsTol), max_iter);
    const Eigen::MatrixXf yaw_ref   = yaw_ref_d.cast<float>();
    const Eigen::MatrixXf pitch_ref = pitch_ref_d.cast<float>();

    std::vector<double> times;
    times.reserve(trials);
    int converged = 0;

    for (int i = 0; i < trials; ++i) {
        compiler_barrier();
        const auto t0 = Clock::now();
        compiler_barrier();

        setup.solver.set_x0(yaw_ref_d(0, 0), yaw_ref_d(1, 0), pitch_ref_d(0, 0), pitch_ref_d(1, 0));
        setup.solver.set_reference(yaw_ref, pitch_ref);
        const bool ok = setup.solver.solve();

        compiler_barrier();
        const auto t1 = Clock::now();
        compiler_barrier();

        if (ok)
            ++converged;
        times.push_back(elapsed_us(t0, t1));
    }

    return compute_stats(std::move(times), converged);
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
    double sum_sq    = 0.0;
    double max_input = 0.0;
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
            pitch->solution->x(1, k)
            - static_cast<double>(dual.solver.state(DualSmallMpcSolver::kPitchAxis, 1, k)));
        max_state = std::max({max_state, yaw_pos_err, yaw_vel_err, pitch_pos_err, pitch_vel_err});
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
                    - static_cast<double>(dual.solver.input(DualSmallMpcSolver::kPitchAxis, k)))));
    }

    return {
        max_state,
        std::sqrt(sum_sq / static_cast<double>(4 * kHorizon)),
        max_input,
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
        dual_ok,
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
    const auto yaw_ref_chirp   = ref_chirp(kHorizon, 0.3);
    const auto pitch_ref_noisy = ref_noisy_sinusoid(kHorizon, 42, 0.2);

    print_section("Dual-Axis MPC Benchmark");
    std::cout << "  Horizon: " << kHorizon << " (ahead=" << kHorizonAhead
              << " back=" << kHorizonBack << ")\n";
    std::cout << "  dt=" << kDt << " rho=" << kRho << " tol=" << kAbsTol << "\n";
    std::cout << "  Trials=" << kTrials << " Warmup=" << kWarmup << "\n";

    {
        print_section("Solve-Only Timing");
        TinyPairSetup tiny;
        DualSetup dual;
        const auto tiny_t = bench_tiny_pair_solve(tiny, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
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
        const auto tiny_t = bench_tiny_pair_pipeline(tiny, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
        const auto dual_t = bench_dual_pipeline(dual, yaw_ref_sin, pitch_ref_sin, 10, kTrials);
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

        std::cout << "  " << std::left << std::setw(16) << "Scenario" << std::setw(14) << "Tiny p50"
                  << std::setw(14) << "Dual p50" << std::setw(10) << "Speedup"
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
                      << std::setprecision(2) << std::setw(14) << tiny_t.stats.p50 << std::setw(14)
                      << dual_t.stats.p50 << std::setw(10)
                      << tiny_t.stats.p50 / std::max(dual_t.stats.p50, 0.001) << "\n";
        }
    }

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
}
