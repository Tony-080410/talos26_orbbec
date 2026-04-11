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
    std::array<double, 5> initial_params{0.470, 1.942, 0.0, 1.178, 0.0};

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

    /// Regularisation weights [a, ω, b] for small datasets.
    std::array<double, 3> regularization_few{10.0, 1.0, 1.0};
    /// Regularisation weights [a, ω, b] for large datasets.
    std::array<double, 3> regularization_many{60.0, 50.0, 50.0};

    /// Frames needed before direction is resolved.
    int direction_threshold{3};
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

        residuals[0] = -a * cs + b * t_ + c - y_;

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -cs;
            jacobians[0][1] = a * (t_ + t0) * sn;
            jacobians[0][2] = a * w * sn;
            jacobians[0][3] = t_;
            jacobians[0][4] = 1.0;
        }
        return true;
    }

private:
    double t_;
    double y_;
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

    /// Compute δ-angle from current_time to future_time using fitted big-rune model.
    double predictDeltaAngle(double current_time, double future_time) const;

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
    return -params[0] * std::cos(params[1] * (time + params[2])) + params[3] * time + params[4];
}

/// Rotation angle from current_time to (current_time + compensate + distance/bullet_speed).
inline double getRotationAngleBig(
    double distance, double bullet_speed, const std::array<double, 5>& params,
    double compensate_time_s, double current_time_s) {
    double predict_time = distance / bullet_speed + current_time_s + compensate_time_s;
    return getAngleBig(predict_time, params) - getAngleBig(current_time_s, params);
}

/// Small-rune rotation angle.
inline double getRotationAngleSmall(
    double distance, double bullet_speed, double rotation_speed, double compensate_time_s) {
    return rotation_speed * (distance / bullet_speed + compensate_time_s);
}

} // namespace energy_meter
