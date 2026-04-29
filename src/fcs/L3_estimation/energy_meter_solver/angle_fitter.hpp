#pragma once

#include "L3_estimation/energy_meter_solver/types.hpp"

#include <ceres/ceres.h>

#include <array>
#include <random>
#include <shared_mutex>
#include <vector>

namespace energy_meter {

// ============================================================================
// AngleFitterConfig (defined outside AngleFitter to avoid clang parse issue)
// ============================================================================

struct AngleFitterConfig {
    /// Minimum data points before first fit attempt.
    int min_fit_data_size{20};
    /// Maximum stored data points (excess is trimmed).
    int max_fit_data_size{1200};

    /// Initial parameters [a, ω, t0, b, c] for big-rune fitting.
    /// Note: a ∈ [0.780, 1.045], ω ∈ [1.884, 2.000], b ∈ [1.045, 1.310]
    std::array<double, 5> initial_params{0.912, 1.942, 0.0, 1.178, 0.0};

    /// Small rune constant rotation speed (rad/s).
    double small_rune_rotation_speed{1.04719};

    /// RANSAC iterations when data < few_data_threshold.
    int ransac_iterations_few{200};
    /// RANSAC iterations when data >= few_data_threshold.
    int ransac_iterations_many{20};
    /// Threshold between "few" and "many" RANSAC iterations.
    int few_data_threshold{400};
    /// Threshold to enable inlier/outlier exchange.
    int large_data_threshold{800};
    /// Sample size drawn from inliers during RANSAC.
    int inlier_sample_size{200};
    /// Tail kept at end of inlier vector (never shuffled out).
    int outlier_keep_tail{100};
    /// Percentile used for inlier/outlier threshold.
    double inlier_percentile{0.95};

    /// Fraction of data required to declare convexity.
    double convexity_threshold{0.75};
    /// Convexity-dependent parameter bounds for t0.
    double convex_concave_t0_upper{-2.8};
    double convex_concave_t0_lower{-4.0};
    double convex_convex_t0_upper{-1.1};
    double convex_convex_t0_lower{-2.3};

    /// Physical parameter bounds for big-rune model (applied to all data sizes).
    double amplitude_lower{0.780}; // a: 振幅下限
    double amplitude_upper{1.045}; // a: 振幅上限
    double omega_lower{1.884};     // ω: 角频率下限 (rad/s)
    double omega_upper{2.000};     // ω: 角频率上限 (rad/s)
    /// Equality constraint: a + b = 2.090
    double amplitude_offset_sum{2.090};
    /// Weight for a+b constraint (should be much larger than data residuals)
    double equality_constraint_weight{1e5};

    /// Regularisation weights [a, ω, b] for small datasets.
    std::array<double, 3> regularization_few{10.0, 1.0, 1.0};
    /// Regularisation weights [a, ω, b] for large datasets.
    std::array<double, 3> regularization_many{60.0, 50.0, 50.0};

    /// Frames needed before direction is resolved.
    int direction_threshold{5};
    /// Angle-difference threshold for direction voting (rad).
    double direction_angle_threshold{1.5e-2};
};

// ============================================================================
// Ceres Cost Functors (from rm_power_rune)
// ============================================================================

/// Parameter regularisation: penalises deviation from a preset value.
class CostFunctor1 : public ceres::SizedCostFunction<1, 5> {
public:
    CostFunctor1(double truth, int id)
        : truth_(truth)
        , id_(id) {}
    ~CostFunctor1() override = default;

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        double pre   = parameters[0][id_];
        residuals[0] = pre - truth_;
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            for (int i = 0; i < 5; ++i) {
                jacobians[0][i] = (i == id_) ? 1.0 : 0.0;
            }
        }
        return true;
    }

private:
    double truth_;
    int id_;
};

/// Parameter regularisation for 4-parameter constrained model [a, w, t0, c]
class CostFunctor1Constrained : public ceres::SizedCostFunction<1, 4> {
public:
    CostFunctor1Constrained(double truth, int id)
        : truth_(truth)
        , id_(id) {}
    ~CostFunctor1Constrained() override = default;

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        double pre   = parameters[0][id_];
        residuals[0] = pre - truth_;
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            for (int i = 0; i < 4; ++i) {
                jacobians[0][i] = (i == id_) ? 1.0 : 0.0;
            }
        }
        return true;
    }

private:
    double truth_;
    int id_;
};

/// Equality constraint: a + b = 2.090 (enforced via strong penalty)
class EqualityConstraint : public ceres::SizedCostFunction<1, 5> {
public:
    explicit EqualityConstraint(double target_sum)
        : target_sum_(target_sum) {}
    ~EqualityConstraint() override = default;

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        const double a = parameters[0][0]; // amplitude
        const double b = parameters[0][3]; // offset
        residuals[0]   = a + b - target_sum_;

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            std::fill_n(jacobians[0], 5, 0.0);
            jacobians[0][0] = 1.0;         // ∂/∂a
            jacobians[0][3] = 1.0;         // ∂/∂b
        }
        return true;
    }

private:
    double target_sum_;
};

/// Fitting residual:  θ(t) = -a·cos(ω·(t+t0)) + b·t + c
class CostFunctor2 : public ceres::SizedCostFunction<1, 5> {
public:
    CostFunctor2(double t, double y)
        : t_(t)
        , y_(y) {}
    ~CostFunctor2() override = default;

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        const double a  = parameters[0][0];
        const double w  = parameters[0][1];
        const double t0 = parameters[0][2];
        const double b  = parameters[0][3];
        const double c  = parameters[0][4];

        const double cs = std::cos(w * (t_ + t0));
        const double sn = std::sin(w * (t_ + t0));

        // Model: θ(t) = -(a/w) * cos(w * (t + t0)) + b*t + c
        residuals[0] = -(a / w) * cs + b * t_ + c - y_;

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -(1.0 / w) * cs;
            jacobians[0][1] = (a / (w * w)) * cs + (a / w) * (t_ + t0) * sn;
            jacobians[0][2] = a * sn;
            jacobians[0][3] = t_;
            jacobians[0][4] = 1.0;
        }
        return true;
    }

private:
    double t_;
    double y_;
};

/// Fitting residual with hard constraint a + b = 2.090
/// Model: θ(t) = -(a/w) * cos(w * (t + t0)) + b*t + c, with b = 2.090 - a
/// Parameters: [a, w, t0, c] (4 parameters instead of 5)
class CostFunctorConstrained : public ceres::SizedCostFunction<1, 4> {
public:
    CostFunctorConstrained(double t, double y, double amplitude_offset_sum)
        : t_(t)
        , y_(y)
        , sum_(amplitude_offset_sum) {}
    ~CostFunctorConstrained() override = default;

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override {
        const double a  = parameters[0][0];
        const double w  = parameters[0][1];
        const double t0 = parameters[0][2];
        const double c  = parameters[0][3];

        // b = sum - a (hard constraint)
        const double b = sum_ - a;

        const double cs = std::cos(w * (t_ + t0));
        const double sn = std::sin(w * (t_ + t0));

        // Model: θ(t) = -(a/w) * cos(w * (t + t0)) + b*t + c
        residuals[0] = -(a / w) * cs + b * t_ + c - y_;

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            // ∂residual/∂a = -(1/w) * cs - t
            jacobians[0][0] = -(1.0 / w) * cs - t_;
            // ∂residual/∂w = (a/w²) * cs + (a/w) * (t+t0) * sn
            jacobians[0][1] = (a / (w * w)) * cs + (a / w) * (t_ + t0) * sn;
            // ∂residual/∂t0 = a * sn
            jacobians[0][2] = a * sn;
            // ∂residual/∂c = 1
            jacobians[0][3] = 1.0;
        }
        return true;
    }

private:
    double t_;
    double y_;
    double sum_; // amplitude_offset_sum (a + b = sum)
};

// ============================================================================
// AngleFitter
// ============================================================================

class AngleFitter {
public:
    using Config = AngleFitterConfig;

    explicit AngleFitter(Config config = {});
    ~AngleFitter();

    AngleFitter(const AngleFitter&)            = delete;
    AngleFitter& operator=(const AngleFitter&) = delete;

    // ── observation interface ──

    /// Add a (time, |angle|) observation.  time is relative to start, in seconds.
    void addObservation(double time, double abs_angle);

    // ── fitting ──

    /// Fit big-rune model using RANSAC + Ceres LSQ.
    /// Returns true if params are valid after fitting.
    bool fitBigRune();

    /// Set small-rune constant rotation speed (no LSQ needed).
    bool fitSmallRune();

    // ── query ──

    auto getModel() const -> AngleModel;
    auto getDirection() const -> Direction;
    auto getDataPointCount() const -> int;
    bool isModelValid() const;

    /// Reset all internal state.
    void reset();

    // ── prediction ──

    // /// Compute δ-angle from current_time to future_time using fitted big-rune model.
    // double predictDeltaAngle(double current_time, double future_time) const;

    // ── direction ──

    /// Push one continuous-angle sample for direction voting.
    void updateDirection(double continuous_angle);

private:
    Config config_;

    // ── fitting data (guarded by data_mutex_) ──
    mutable std::shared_mutex data_mutex_;
    std::vector<std::pair<double, double>> fit_data_; // (time, |angle|)

    // ── model state (guarded by model_mutex_) ──
    mutable std::shared_mutex model_mutex_;
    std::array<double, 5> params_;
    bool params_valid_{false};
    Convexity convexity_{Convexity::UNKNOWN};
    double fit_error_{0.0}; // RMSE of last fit

    // ── direction voting ──
    Direction direction_{Direction::UNKNOWN};
    std::vector<double> direction_data_;

    // ── rng ──
    std::mt19937 random_engine_{std::random_device{}()};

    // ── internal fitting ──
    std::array<double, 5> ransacFitting(Convexity convexity);
    std::array<double, 5> leastSquareEstimate(
        const std::vector<std::pair<double, double>>& points,
        const std::array<double, 5>& initial_params, Convexity convexity);
    Convexity detectConvexity(const std::vector<std::pair<double, double>>& data) const;
};

// ============================================================================
// Inline helpers (rm_power_rune semantics)
// ============================================================================

/// Evaluate the big-rune model at time t.
inline double getAngleBig(double time, const std::array<double, 5>& params) {
    return -(params[0] / params[1]) * std::cos(params[1] * (time + params[2])) + params[3] * time
         + params[4];
}

// /// Rotation angle from current_time to (current_time + compensate + distance/bullet_speed).
// inline double getRotationAngleBig(
//     double distance, double bullet_speed, const std::array<double, 5>& params,
//     double compensate_time_s, double current_time_s) {
//     double predict_time = distance / bullet_speed + current_time_s + compensate_time_s;
//     return getAngleBig(predict_time, params) - getAngleBig(current_time_s, params);
// }

// /// Small-rune rotation angle.
// inline double getRotationAngleSmall(
//     double distance, double bullet_speed, double rotation_speed, double compensate_time_s) {
//     return rotation_speed * (distance / bullet_speed + compensate_time_s);
// }

} // namespace energy_meter
