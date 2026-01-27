#include "uni_kf/redact/filter.hpp"
#include "uni_kf/redact/spaces/euclidean.hpp"
#include "uni_kf/redact/spaces/lie_group.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>

namespace {

using namespace uni_kf::redact;
using namespace uni_kf::redact::spaces;

struct Options {
    int mc_runs        = 200;
    double duration_s  = 10.0;
    double pred_hz     = 100.0;
    double meas_hz     = 50.0;
    std::uint32_t seed = 0xC0FFEEu;
    bool csv_header    = true;
};

[[nodiscard]] Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto next = [&]() -> std::string_view {
            if (i + 1 < argc) {
                return std::string_view(argv[++i]);
            }
            return {};
        };

        if (a == "--runs") {
            opt.mc_runs = std::atoi(std::string(next()).c_str());
        } else if (a == "--duration") {
            opt.duration_s = std::atof(std::string(next()).c_str());
        } else if (a == "--pred-hz") {
            opt.pred_hz = std::atof(std::string(next()).c_str());
        } else if (a == "--meas-hz") {
            opt.meas_hz = std::atof(std::string(next()).c_str());
        } else if (a == "--seed") {
            opt.seed =
                static_cast<std::uint32_t>(std::strtoul(std::string(next()).c_str(), nullptr, 0));
        } else if (a == "--no-header") {
            opt.csv_header = false;
        } else {
            // ignore unknown args
        }
    }
    return opt;
}

[[nodiscard]] Eigen::VectorXd sample_gaussian(std::mt19937& rng, const Eigen::MatrixXd& cov) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(cov.rows());
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x(i) = nd(rng);
    }

    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    if (llt.info() == Eigen::Success) {
        return llt.matrixL() * x;
    }

    // If user provided non-SPD cov, return zero; benchmark assumes SPD cov.
    return Eigen::VectorXd::Zero(cov.rows());
}

template <typename VecDerived, typename MatDerived>
[[nodiscard]] double
    compute_nees(const Eigen::MatrixBase<VecDerived>& err, const Eigen::MatrixBase<MatDerived>& P) {
    const Eigen::VectorXd e  = err;
    const Eigen::MatrixXd Pm = P;
    const auto ldlt          = Pm.ldlt();
    if (ldlt.info() == Eigen::Success) {
        const Eigen::VectorXd y = ldlt.solve(e);
        return e.dot(y);
    }
    const Eigen::VectorXd y = Pm.completeOrthogonalDecomposition().solve(e);
    return e.dot(y);
}

struct Aggregate {
    double rmse    = std::numeric_limits<double>::quiet_NaN();
    double nees    = std::numeric_limits<double>::quiet_NaN();
    double time_us = 0.0;
    int state_dim  = 0;
    int steps      = 0;
    int mc_runs    = 0;
    int failures   = 0;
};

// ============================================================================
// Models used by this benchmark
// ============================================================================

using CVBelief = Euclidean<4>;
using CVMeas   = Euclidean<2>;

struct CVPerception {
    using Belief      = CVBelief;
    using Measurement = CVMeas;

    Eigen::Matrix2d R;

    Measurement::Flux assume(const Belief::Flux& x) const {
        return Measurement::Flux{x.data.head<2>()};
    }
    Eigen::MatrixXd measurement_noise_covariance() const { return R; }

    Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const {
        Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
        H(0, 0)                       = 1.0;
        H(1, 1)                       = 1.0;
        return H;
    }

    template <typename Cov>
    Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
        const Eigen::MatrixXd K =
            cov.compute_kalman_gain(measurement_jacobian(Belief::zero_flux()), R);
        return Belief::Drift{(K * innovation.data).eval()};
    }
};

struct CVAssumption {
    using Perception = CVPerception;
    using Belief     = typename Perception::Belief;

    Eigen::Matrix4d Q;

    Belief::Flux maneuver(auto&, const Belief::Flux& x, double dt) const {
        Eigen::Vector4d y = x.data;
        y(0) += y(2) * dt;
        y(1) += y(3) * dt;
        return Belief::Flux{y};
    }
    Eigen::MatrixXd process_noise_covariance() const { return Q; }
};

using SO3Belief = SO3Right;
using SO3Meas   = Euclidean<3>;

struct SO3Perception {
    using Belief      = SO3Belief;
    using Measurement = SO3Meas;

    Eigen::Matrix3d R;

    // For right-invariant error, we generate measurements as z = Log(X_est⁻¹ * X_true) + noise
    // For sigma-point filters (UKF/CKF), assume() returns the state's log for sigma-point transform
    // For EKF-style filters, the measurement Jacobian H=I is correct due to right-invariant error
    Measurement::Flux assume(const Belief::Flux& x) const {
        return Measurement::Flux{group::SO3<double>::log(x)};
    }
    Eigen::MatrixXd measurement_noise_covariance() const { return R; }
    Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const {
        return Eigen::Matrix3d::Identity();
    }

    template <typename Cov>
    Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
        const Eigen::MatrixXd K = cov.compute_kalman_gain(Eigen::Matrix3d::Identity(), R);
        return K * innovation.data;
    }
};

struct SO3Assumption {
    using Perception = SO3Perception;
    using Belief     = typename Perception::Belief;

    Eigen::Matrix3d Q;
    Eigen::Vector3d omega;

    Belief::Flux maneuver(auto&, const Belief::Flux& x, double dt) const {
        return x * group::SO3<double>::exp(omega * dt);
    }
    Eigen::MatrixXd process_noise_covariance() const { return Q; }
};

using SE3Belief = SE3Right;
using SE3Meas   = Euclidean<6>;

// Standard perception for UKF/CKF (uses log measurements)
struct SE3Perception {
    using Belief      = SE3Belief;
    using Measurement = SE3Meas;

    Eigen::Matrix<double, 6, 6> R;

    Measurement::Flux assume(const Belief::Flux& x) const {
        return Measurement::Flux{group::SEn3<double, 1>::log(x)};
    }
    Eigen::MatrixXd measurement_noise_covariance() const { return R; }
    Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const {
        return Eigen::Matrix<double, 6, 6>::Identity();
    }

    template <typename Cov>
    Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
        const Eigen::MatrixXd K =
            cov.compute_kalman_gain(Eigen::Matrix<double, 6, 6>::Identity(), R);
        return K * innovation.data;
    }
};

// Right-invariant perception for InKF/InEKF (uses error measurements)
struct SE3PerceptionRightInvariant {
    using Belief      = SE3Belief;
    using Measurement = SE3Meas;

    Eigen::Matrix<double, 6, 6> R;

    // For right-invariant error, the measurement IS the error: z = ξ = Log(X_est⁻¹ * X_true)
    // The predicted measurement is zero (we expect no error)
    Measurement::Flux assume(const Belief::Flux& x) const {
        // Predicted measurement is zero (no expected error)
        return Measurement::Flux{Eigen::Matrix<double, 6, 1>::Zero()};
    }
    Eigen::MatrixXd measurement_noise_covariance() const { return R; }
    Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const {
        return Eigen::Matrix<double, 6, 6>::Identity();
    }

    template <typename Cov>
    Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
        const Eigen::MatrixXd K =
            cov.compute_kalman_gain(Eigen::Matrix<double, 6, 6>::Identity(), R);
        return K * innovation.data;
    }
};

struct SE3Assumption {
    using Perception = SE3Perception;
    using Belief     = typename Perception::Belief;

    Eigen::Matrix<double, 6, 6> Q;
    Eigen::Matrix<double, 6, 1> twist;

    Belief::Flux maneuver(auto&, const Belief::Flux& x, double dt) const {
        return x * group::SEn3<double, 1>::exp(twist * dt);
    }
    Eigen::MatrixXd process_noise_covariance() const { return Q; }
};

// ============================================================================
// Helper to detect perception type and generate measurements
// ============================================================================

// Compile-time detection: is this a right-invariant perception?
template <typename T>
struct is_right_invariant_perception : std::false_type {};

template <>
struct is_right_invariant_perception<SE3PerceptionRightInvariant> : std::true_type {};

// Helper to extract perception type from filter type
// This works because FilterT = KalmanFilter<Cov, Perception, Assumption>
template <typename FilterT>
struct get_filter_perception;

// Specialization: extract from KalmanFilter
template <typename Cov, typename Per, typename Ass>
struct get_filter_perception<uni_kf::redact::KalmanFilter<Cov, Per, Ass>> {
    using type = Per;
};

template <typename FilterT>
using get_filter_perception_t = typename get_filter_perception<FilterT>::type;

using SE23Belief = SE23Right;
using SE23Meas   = Euclidean<9>;

struct SE23Perception {
    using Belief      = SE23Belief;
    using Measurement = SE23Meas;

    Eigen::Matrix<double, 9, 9> R;

    // For right-invariant error, we generate measurements as z = Log(X_est⁻¹ * X_true) + noise
    // For sigma-point filters (UKF/CKF), assume() returns the state's log for sigma-point transform
    // For EKF-style filters, the measurement Jacobian H=I is correct due to right-invariant error
    Measurement::Flux assume(const Belief::Flux& x) const {
        return Measurement::Flux{SE23Belief::log(x)};
    }
    Eigen::MatrixXd measurement_noise_covariance() const { return R; }
    Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const {
        return Eigen::Matrix<double, 9, 9>::Identity();
    }

    template <typename Cov>
    Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
        const Eigen::MatrixXd K =
            cov.compute_kalman_gain(Eigen::Matrix<double, 9, 9>::Identity(), R);
        return K * innovation.data;
    }
};

struct SE23Assumption {
    using Perception = SE23Perception;
    using Belief     = typename Perception::Belief;

    Eigen::Matrix<double, 9, 9> Q;
    Eigen::Matrix<double, 9, 1> xi;

    Belief::Flux maneuver(auto&, const Belief::Flux& x, double dt) const {
        return x * SE23Belief::exp(xi * dt);
    }
    Eigen::MatrixXd process_noise_covariance() const { return Q; }
};

template <typename FilterT>
[[nodiscard]] Aggregate run_cv_benchmark(const Options& opt) {

    const double dt_pred = 1.0 / opt.pred_hz;
    const double dt_meas = 1.0 / opt.meas_hz;
    const int steps      = static_cast<int>(opt.duration_s * opt.pred_hz);
    const int meas_int   = std::max(1, static_cast<int>(std::lround(dt_meas / dt_pred)));

    // Noise parameters
    Eigen::Matrix4d Q  = Eigen::Matrix4d::Zero();
    Q(2, 2)            = 0.05 * 0.05;
    Q(3, 3)            = 0.05 * 0.05;
    Eigen::Matrix2d R  = Eigen::Matrix2d::Identity() * (0.5 * 0.5);
    Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity() * 0.5;

    double rmse_sum = 0.0;
    double nees_sum = 0.0;
    double time_sum = 0.0;

    for (int run = 0; run < opt.mc_runs; ++run) {
        std::mt19937 rng(opt.seed + static_cast<std::uint32_t>(run));

        Eigen::Vector4d x_true;
        x_true << 0.0, 0.0, 1.0, 0.5;

        const Eigen::VectorXd init_err = sample_gaussian(rng, P0);
        Eigen::Vector4d x0             = x_true + init_err.head<4>();

        FilterT filter(CVBelief::Flux{x0}, P0, CVPerception{R}, CVAssumption{Q});

        double err_sq_sum    = 0.0;
        double nees_run_sum  = 0.0;
        std::uint64_t cycles = 0;
        bool run_failed      = false;

        for (int k = 0; k < steps; ++k) {
            // Ground truth propagate with process noise
            x_true(0) += x_true(2) * dt_pred;
            x_true(1) += x_true(3) * dt_pred;
            x_true += sample_gaussian(rng, Q).head<4>();

            auto t0 = std::chrono::high_resolution_clock::now();
            (void)filter.predict(dt_pred);
            if ((k % meas_int) == 0) {
                const Eigen::Vector2d z = x_true.head<2>() + sample_gaussian(rng, R).head<2>();
                if (!filter.percept(CVMeas::Flux{z}, dt_pred).has_value()) {
                    run_failed = true;
                    break;
                }
            }
            auto t1         = std::chrono::high_resolution_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            time_sum += us;
            ++cycles;

            const Eigen::Vector4d x_est = filter.state().data;
            const Eigen::Vector4d e     = x_est - x_true;
            err_sq_sum += e.squaredNorm();
            nees_run_sum += compute_nees(e, filter.covariance());
        }

        if (run_failed) {
            break;
        }

        const double rmse_run = std::sqrt(err_sq_sum / static_cast<double>(steps));
        const double nees_run = nees_run_sum / static_cast<double>(steps);
        rmse_sum += rmse_run;
        nees_sum += nees_run;
    }

    Aggregate out;
    if (rmse_sum > 0.0) {
        out.rmse      = rmse_sum / static_cast<double>(opt.mc_runs);
        out.nees      = nees_sum / static_cast<double>(opt.mc_runs);
        out.time_us   = time_sum / static_cast<double>(opt.mc_runs * steps);
        out.state_dim = 4;
        out.steps     = steps;
        out.mc_runs   = opt.mc_runs;
    } else {
        out.failures = opt.mc_runs;
    }

    return out;
}

template <typename FilterT>
[[nodiscard]] Aggregate run_so3_benchmark(const Options& opt) {
    const double dt_pred = 1.0 / opt.pred_hz;
    const double dt_meas = 1.0 / opt.meas_hz;
    const int steps      = static_cast<int>(opt.duration_s * opt.pred_hz);
    const int meas_int   = std::max(1, static_cast<int>(std::lround(dt_meas / dt_pred)));

    const Eigen::Vector3d omega(0.2, -0.1, 0.05);

    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * (0.02 * 0.02);
    Eigen::Matrix3d R  = Eigen::Matrix3d::Identity() * (0.05 * 0.05);
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.1;

    double rmse_sum = 0.0;
    double nees_sum = 0.0;
    double time_sum = 0.0;

    for (int run = 0; run < opt.mc_runs; ++run) {
        std::mt19937 rng(opt.seed + 0x9E3779B9u + static_cast<std::uint32_t>(run));

        group::SO3<double> R_true;

        const Eigen::VectorXd init_err = sample_gaussian(rng, P0);
        const group::SO3<double> R0    = R_true * group::SO3<double>::exp(init_err.head<3>());

        FilterT filter(SO3Belief::Flux{R0}, P0, SO3Perception{R}, SO3Assumption{Q, omega});

        double err_sq_sum   = 0.0;
        double nees_run_sum = 0.0;
        bool run_failed     = false;

        for (int k = 0; k < steps; ++k) {
            R_true = R_true * group::SO3<double>::exp(omega * dt_pred);
            R_true = R_true * group::SO3<double>::exp(sample_gaussian(rng, Q).head<3>());

            auto t0 = std::chrono::high_resolution_clock::now();
            (void)filter.predict(dt_pred);
            if ((k % meas_int) == 0) {
                const Eigen::Vector3d z =
                    group::SO3<double>::log(R_true) + sample_gaussian(rng, R).head<3>();
                if (!filter.percept(SO3Meas::Flux{z}, dt_pred).has_value()) {
                    run_failed = true;
                    break;
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration<double, std::micro>(t1 - t0).count();

            const group::SO3<double> R_est = filter.state();
            const Eigen::Vector3d e        = group::SO3<double>::log(R_est.inv() * R_true);
            err_sq_sum += e.squaredNorm();
            nees_run_sum += compute_nees(e, filter.covariance());
        }

        if (run_failed) {
            break;
        }

        rmse_sum += std::sqrt(err_sq_sum / static_cast<double>(steps));
        nees_sum += nees_run_sum / static_cast<double>(steps);
    }

    Aggregate out;
    if (rmse_sum > 0.0) {
        out.rmse      = rmse_sum / static_cast<double>(opt.mc_runs);
        out.nees      = nees_sum / static_cast<double>(opt.mc_runs);
        out.time_us   = time_sum / static_cast<double>(opt.mc_runs * steps);
        out.state_dim = 3;
        out.steps     = steps;
        out.mc_runs   = opt.mc_runs;
    } else {
        out.failures = opt.mc_runs;
    }

    return out;
}

template <typename FilterT>
[[nodiscard]] Aggregate run_se3_benchmark(const Options& opt) {
    const double dt_pred = 1.0 / opt.pred_hz;
    const double dt_meas = 1.0 / opt.meas_hz;
    const int steps      = static_cast<int>(opt.duration_s * opt.pred_hz);
    const int meas_int   = std::max(1, static_cast<int>(std::lround(dt_meas / dt_pred)));

    Eigen::Matrix<double, 6, 1> twist;
    twist << 0.2, -0.1, 0.05, 0.5, 0.0, 0.0;

    Eigen::Matrix<double, 6, 6> Q  = Eigen::Matrix<double, 6, 6>::Identity() * (0.02 * 0.02);
    Eigen::Matrix<double, 6, 6> R  = Eigen::Matrix<double, 6, 6>::Identity() * (0.05 * 0.05);
    Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;

    double rmse_sum = 0.0;
    double nees_sum = 0.0;
    double time_sum = 0.0;

    for (int run = 0; run < opt.mc_runs; ++run) {
        std::mt19937 rng(opt.seed + 0x7F4A7C15u + static_cast<std::uint32_t>(run));

        group::SEn3<double, 1> T_true;

        const Eigen::VectorXd init_err  = sample_gaussian(rng, P0);
        const group::SEn3<double, 1> T0 = T_true * group::SEn3<double, 1>::exp(init_err.head<6>());

        // Filter construction: extract perception type from FilterT at compile time
        using PerceptionT = get_filter_perception_t<FilterT>;
        FilterT filter(SE3Belief::Flux{T0}, P0, PerceptionT{R}, SE3Assumption{Q, twist});

        double err_sq_sum   = 0.0;
        double nees_run_sum = 0.0;
        bool run_failed     = false;

        for (int k = 0; k < steps; ++k) {
            T_true = T_true * group::SEn3<double, 1>::exp(twist * dt_pred);
            T_true = T_true * group::SEn3<double, 1>::exp(sample_gaussian(rng, Q).head<6>());

            auto t0 = std::chrono::high_resolution_clock::now();
            (void)filter.predict(dt_pred);
            if ((k % meas_int) == 0) {
                // Generate measurement based on perception type:
                // - Standard (UKF/CKF): z = log(T_true) + noise
                // - Right-invariant (InKF/InEKF): z = Log(T_est⁻¹ * T_true) + noise
                using PerceptionT = get_filter_perception_t<FilterT>;
                Eigen::Matrix<double, 6, 1> z;
                if constexpr (is_right_invariant_perception<PerceptionT>::value) {
                    // Right-invariant: measurement is the error between estimate and truth
                    const group::SEn3<double, 1> T_est = filter.state();
                    z = SE3Belief::diff(T_est, T_true) + sample_gaussian(rng, R).head<6>();
                } else {
                    // Standard: measurement is log of true state
                    z = group::SEn3<double, 1>::log(T_true) + sample_gaussian(rng, R).head<6>();
                }
                if (!filter.percept(SE3Meas::Flux{z}, dt_pred).has_value()) {
                    run_failed = true;
                    break;
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration<double, std::micro>(t1 - t0).count();

            const group::SEn3<double, 1> T_est  = filter.state();
            const Eigen::Matrix<double, 6, 1> e = SE3Belief::diff(T_est, T_true);
            err_sq_sum += e.squaredNorm();
            nees_run_sum += compute_nees(e, filter.covariance());
        }

        if (run_failed) {
            break;
        }

        rmse_sum += std::sqrt(err_sq_sum / static_cast<double>(steps));
        nees_sum += nees_run_sum / static_cast<double>(steps);
    }

    Aggregate out;
    if (rmse_sum > 0.0) {
        out.rmse      = rmse_sum / static_cast<double>(opt.mc_runs);
        out.nees      = nees_sum / static_cast<double>(opt.mc_runs);
        out.time_us   = time_sum / static_cast<double>(opt.mc_runs * steps);
        out.state_dim = 6;
        out.steps     = steps;
        out.mc_runs   = opt.mc_runs;
    } else {
        out.failures = opt.mc_runs;
    }

    return out;
}

template <typename FilterT>
[[nodiscard]] Aggregate run_se23_benchmark(const Options& opt) {
    const double dt_pred = 1.0 / opt.pred_hz;
    const double dt_meas = 1.0 / opt.meas_hz;
    const int steps      = static_cast<int>(opt.duration_s * opt.pred_hz);
    const int meas_int   = std::max(1, static_cast<int>(std::lround(dt_meas / dt_pred)));

    Eigen::Matrix<double, 9, 1> xi;
    xi << 0.2, -0.1, 0.05, 0.5, 0.0, 0.0, 0.1, 0.2, -0.1;

    Eigen::Matrix<double, 9, 9> Q  = Eigen::Matrix<double, 9, 9>::Identity() * (0.02 * 0.02);
    Eigen::Matrix<double, 9, 9> R  = Eigen::Matrix<double, 9, 9>::Identity() * (0.05 * 0.05);
    Eigen::Matrix<double, 9, 9> P0 = Eigen::Matrix<double, 9, 9>::Identity() * 0.1;

    double rmse_sum = 0.0;
    double nees_sum = 0.0;
    double time_sum = 0.0;

    for (int run = 0; run < opt.mc_runs; ++run) {
        std::mt19937 rng(opt.seed + 0x51ED270Bu + static_cast<std::uint32_t>(run));

        group::SEn3<double, 2> X_true;

        const Eigen::VectorXd init_err  = sample_gaussian(rng, P0);
        const group::SEn3<double, 2> X0 = X_true * SE23Belief::exp(init_err.head<9>());

        FilterT filter(SE23Belief::Flux{X0}, P0, SE23Perception{R}, SE23Assumption{Q, xi});

        double err_sq_sum   = 0.0;
        double nees_run_sum = 0.0;
        bool run_failed     = false;

        for (int k = 0; k < steps; ++k) {
            X_true = X_true * SE23Belief::exp(xi * dt_pred);
            X_true = X_true * SE23Belief::exp(sample_gaussian(rng, Q).head<9>());

            auto t0 = std::chrono::high_resolution_clock::now();
            (void)filter.predict(dt_pred);
            if ((k % meas_int) == 0) {
                const Eigen::Matrix<double, 9, 1> z =
                    SE23Belief::log(X_true) + sample_gaussian(rng, R).head<9>();
                if (!filter.percept(SE23Meas::Flux{z}, dt_pred).has_value()) {
                    run_failed = true;
                    break;
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration<double, std::micro>(t1 - t0).count();

            const group::SEn3<double, 2> X_est  = filter.state();
            const Eigen::Matrix<double, 9, 1> e = SE23Belief::diff(X_est, X_true);
            err_sq_sum += e.squaredNorm();
            nees_run_sum += compute_nees(e, filter.covariance());
        }

        if (run_failed) {
            break;
        }

        rmse_sum += std::sqrt(err_sq_sum / static_cast<double>(steps));
        nees_sum += nees_run_sum / static_cast<double>(steps);
    }

    Aggregate out;
    if (rmse_sum > 0.0) {
        out.rmse      = rmse_sum / static_cast<double>(opt.mc_runs);
        out.nees      = nees_sum / static_cast<double>(opt.mc_runs);
        out.time_us   = time_sum / static_cast<double>(opt.mc_runs * steps);
        out.state_dim = 9;
        out.steps     = steps;
        out.mc_runs   = opt.mc_runs;
    } else {
        out.failures = opt.mc_runs;
    }

    return out;
}

void print_row(
    std::string_view scenario, std::string_view filter, std::string_view repr,
    std::string_view updater, const Aggregate& a) {
    std::cout << scenario << ',' << filter << ',' << repr << ',' << updater << ','
              << std::setprecision(8) << a.rmse << ',' << std::setprecision(8) << a.nees << ','
              << std::setprecision(8) << a.time_us << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);

    if (opt.csv_header) {
        std::cout << "scenario,filter,repr,updater,rmse,nees,avg_cycle_us\n";
    }

    // Scenario A: Euclidean constant-velocity (R^4)
    {
        using B = CVBelief;

        using KF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, LinearStrategy>, CVPerception, CVAssumption>;
        using EKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, LinearizedStrategy>, CVPerception, CVAssumption>;
        using UKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, UnscentedStrategy>, CVPerception, CVAssumption>;
        using CKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, CubatureStrategy>, CVPerception, CVAssumption>;

        using KF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, LinearStrategy>, CVPerception, CVAssumption>;
        using EKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, LinearizedStrategy>, CVPerception, CVAssumption>;
        using UKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, UnscentedStrategy>, CVPerception, CVAssumption>;
        using CKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, CubatureStrategy>, CVPerception, CVAssumption>;

        using KF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, LinearStrategy>, CVPerception, CVAssumption>;
        using EKF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, LinearizedStrategy>, CVPerception, CVAssumption>;
        using KF_J = KalmanFilter<
            Covariance<B, Representation::Direct, LinearStrategy, DirectJosephUpdater>,
            CVPerception, CVAssumption>;
        using EKF_J = KalmanFilter<
            Covariance<B, Representation::Direct, LinearizedStrategy, DirectJosephUpdater>,
            CVPerception, CVAssumption>;

        print_row("cv", "KF", "direct", "simple", run_cv_benchmark<KF_Direct>(opt));
        print_row("cv", "EKF", "direct", "simple", run_cv_benchmark<EKF_Direct>(opt));
        print_row("cv", "UKF", "direct", "simple", run_cv_benchmark<UKF_Direct>(opt));
        print_row("cv", "CKF", "direct", "simple", run_cv_benchmark<CKF_Direct>(opt));

        print_row("cv", "KF", "sr", "sr", run_cv_benchmark<KF_SR>(opt));
        print_row("cv", "EKF", "sr", "sr", run_cv_benchmark<EKF_SR>(opt));
        print_row("cv", "UKF", "sr", "sr", run_cv_benchmark<UKF_SR>(opt));
        print_row("cv", "CKF", "sr", "sr", run_cv_benchmark<CKF_SR>(opt));

        print_row("cv", "KF", "udu", "udu", run_cv_benchmark<KF_UDU>(opt));
        print_row("cv", "EKF", "udu", "udu", run_cv_benchmark<EKF_UDU>(opt));

        print_row("cv", "KF", "direct", "joseph", run_cv_benchmark<KF_J>(opt));
        print_row("cv", "EKF", "direct", "joseph", run_cv_benchmark<EKF_J>(opt));
    }

    // Scenario B: SO(3) rotation
    {
        using B = SO3Belief;

        using InKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearStrategy>, SO3Perception,
            SO3Assumption>;
        using InEKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearizedStrategy>, SO3Perception,
            SO3Assumption>;
        using UKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, UnscentedStrategy>, SO3Perception, SO3Assumption>;
        using CKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, CubatureStrategy>, SO3Perception, SO3Assumption>;

        using InKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, InvariantLinearStrategy>, SO3Perception,
            SO3Assumption>;
        using InEKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, InvariantLinearizedStrategy>, SO3Perception,
            SO3Assumption>;
        using UKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, UnscentedStrategy>, SO3Perception, SO3Assumption>;
        using CKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, CubatureStrategy>, SO3Perception, SO3Assumption>;

        using InKF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, InvariantLinearStrategy>, SO3Perception,
            SO3Assumption>;
        using InEKF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, InvariantLinearizedStrategy>, SO3Perception,
            SO3Assumption>;
        using InKF_J = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearStrategy, DirectJosephUpdater>,
            SO3Perception, SO3Assumption>;
        using InEKF_J = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearizedStrategy, DirectJosephUpdater>,
            SO3Perception, SO3Assumption>;

        print_row("so3", "InKF", "direct", "simple", run_so3_benchmark<InKF_Direct>(opt));
        print_row("so3", "InEKF", "direct", "simple", run_so3_benchmark<InEKF_Direct>(opt));
        print_row("so3", "UKF", "direct", "simple", run_so3_benchmark<UKF_Direct>(opt));
        print_row("so3", "CKF", "direct", "simple", run_so3_benchmark<CKF_Direct>(opt));

        print_row("so3", "InKF", "sr", "sr", run_so3_benchmark<InKF_SR>(opt));
        print_row("so3", "InEKF", "sr", "sr", run_so3_benchmark<InEKF_SR>(opt));
        print_row("so3", "UKF", "sr", "sr", run_so3_benchmark<UKF_SR>(opt));
        print_row("so3", "CKF", "sr", "sr", run_so3_benchmark<CKF_SR>(opt));

        print_row("so3", "InKF", "udu", "udu", run_so3_benchmark<InKF_UDU>(opt));
        print_row("so3", "InEKF", "udu", "udu", run_so3_benchmark<InEKF_UDU>(opt));

        print_row("so3", "InKF", "direct", "joseph", run_so3_benchmark<InKF_J>(opt));
        print_row("so3", "InEKF", "direct", "joseph", run_so3_benchmark<InEKF_J>(opt));
    }

    // Scenario C: SE(3) pose (rotation + translation)
    {
        using B = SE3Belief;

        // InKF/InEKF use right-invariant perception (H=I, measurement is error)
        using InKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearStrategy>,
            SE3PerceptionRightInvariant, SE3Assumption>;
        using InEKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearizedStrategy>,
            SE3PerceptionRightInvariant, SE3Assumption>;
        // UKF/CKF use standard perception (H=I, measurement is log state)
        using UKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, UnscentedStrategy>, SE3Perception, SE3Assumption>;
        using CKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, CubatureStrategy>, SE3Perception, SE3Assumption>;

        using InKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, InvariantLinearStrategy>, SE3PerceptionRightInvariant,
            SE3Assumption>;
        using InEKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, InvariantLinearizedStrategy>,
            SE3PerceptionRightInvariant, SE3Assumption>;
        using UKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, UnscentedStrategy>, SE3Perception, SE3Assumption>;
        using CKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, CubatureStrategy>, SE3Perception, SE3Assumption>;

        using InKF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, InvariantLinearStrategy>,
            SE3PerceptionRightInvariant, SE3Assumption>;
        using InEKF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, InvariantLinearizedStrategy>,
            SE3PerceptionRightInvariant, SE3Assumption>;

        using InKF_J = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearStrategy, DirectJosephUpdater>,
            SE3PerceptionRightInvariant, SE3Assumption>;
        using InEKF_J = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearizedStrategy, DirectJosephUpdater>,
            SE3PerceptionRightInvariant, SE3Assumption>;

        print_row("se3", "InKF", "direct", "simple", run_se3_benchmark<InKF_Direct>(opt));
        print_row("se3", "InEKF", "direct", "simple", run_se3_benchmark<InEKF_Direct>(opt));
        print_row("se3", "UKF", "direct", "simple", run_se3_benchmark<UKF_Direct>(opt));
        print_row("se3", "CKF", "direct", "simple", run_se3_benchmark<CKF_Direct>(opt));

        print_row("se3", "InKF", "sr", "sr", run_se3_benchmark<InKF_SR>(opt));
        print_row("se3", "InEKF", "sr", "sr", run_se3_benchmark<InEKF_SR>(opt));
        print_row("se3", "UKF", "sr", "sr", run_se3_benchmark<UKF_SR>(opt));
        print_row("se3", "CKF", "sr", "sr", run_se3_benchmark<CKF_SR>(opt));

        print_row("se3", "InKF", "udu", "udu", run_se3_benchmark<InKF_UDU>(opt));
        print_row("se3", "InEKF", "udu", "udu", run_se3_benchmark<InEKF_UDU>(opt));

        print_row("se3", "InKF", "direct", "joseph", run_se3_benchmark<InKF_J>(opt));
        print_row("se3", "InEKF", "direct", "joseph", run_se3_benchmark<InEKF_J>(opt));
    }

    // Scenario D: SE2(3) pose (rotation + translation + scale)
    {
        using B = SE23Belief;

        using InKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearStrategy>, SE23Perception,
            SE23Assumption>;
        using InEKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearizedStrategy>, SE23Perception,
            SE23Assumption>;
        using UKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, UnscentedStrategy>, SE23Perception,
            SE23Assumption>;
        using CKF_Direct = KalmanFilter<
            Covariance<B, Representation::Direct, CubatureStrategy>, SE23Perception,
            SE23Assumption>;

        using InKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, InvariantLinearStrategy>, SE23Perception,
            SE23Assumption>;
        using InEKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, InvariantLinearizedStrategy>, SE23Perception,
            SE23Assumption>;
        using UKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, UnscentedStrategy>, SE23Perception, SE23Assumption>;
        using CKF_SR = KalmanFilter<
            Covariance<B, Representation::Sr, CubatureStrategy>, SE23Perception, SE23Assumption>;

        using InKF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, InvariantLinearStrategy>, SE23Perception,
            SE23Assumption>;
        using InEKF_UDU = KalmanFilter<
            Covariance<B, Representation::Udu, InvariantLinearizedStrategy>, SE23Perception,
            SE23Assumption>;

        using InKF_J = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearStrategy, DirectJosephUpdater>,
            SE23Perception, SE23Assumption>;
        using InEKF_J = KalmanFilter<
            Covariance<B, Representation::Direct, InvariantLinearizedStrategy, DirectJosephUpdater>,
            SE23Perception, SE23Assumption>;

        print_row("se23", "InKF", "direct", "simple", run_se23_benchmark<InKF_Direct>(opt));
        print_row("se23", "InEKF", "direct", "simple", run_se23_benchmark<InEKF_Direct>(opt));
        print_row("se23", "UKF", "direct", "simple", run_se23_benchmark<UKF_Direct>(opt));
        print_row("se23", "CKF", "direct", "simple", run_se23_benchmark<CKF_Direct>(opt));

        print_row("se23", "InKF", "sr", "sr", run_se23_benchmark<InKF_SR>(opt));
        print_row("se23", "InEKF", "sr", "sr", run_se23_benchmark<InEKF_SR>(opt));
        print_row("se23", "UKF", "sr", "sr", run_se23_benchmark<UKF_SR>(opt));
        print_row("se23", "CKF", "sr", "sr", run_se23_benchmark<CKF_SR>(opt));

        print_row("se23", "InKF", "udu", "udu", run_se23_benchmark<InKF_UDU>(opt));
        print_row("se23", "InEKF", "udu", "udu", run_se23_benchmark<InEKF_UDU>(opt));

        print_row("se23", "InKF", "direct", "joseph", run_se23_benchmark<InKF_J>(opt));
        print_row("se23", "InEKF", "direct", "joseph", run_se23_benchmark<InEKF_J>(opt));
    }

    return 0;
}
