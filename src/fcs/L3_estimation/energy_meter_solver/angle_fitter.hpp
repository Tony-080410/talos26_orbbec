#pragma once

#include "L3_estimation/energy_meter_solver/types.hpp"

#include <deque>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <vector>

namespace energy_meter {

class AngleFitter {
public:
    // Talos-style constructor: accepts individual parameters
    // smooth_alpha: velocity filter exponential smoothing factor
    // min_data_points: minimum observations before fitting
    explicit AngleFitter(double smooth_alpha = 0.8, int min_data_points = 15);
    ~AngleFitter();

    AngleFitter(const AngleFitter&)            = delete;
    AngleFitter& operator=(const AngleFitter&) = delete;

    bool addObservation(const AngleObservation& observation);

    auto getModel() const -> AngleModel {
        std::shared_lock<std::shared_mutex> lock(model_mutex_);
        return current_model_;
    }

    void reset();

    auto getDataPointCount() const -> int {
        std::shared_lock<std::shared_mutex> lock(observations_mutex_);
        return static_cast<int>(observations_.size());
    }

    void setFilteredVelocity(double velocity) {
        std::unique_lock<std::shared_mutex> lock(model_mutex_);
        current_model_.filtered_velocity     = velocity;
        current_model_.velocity_filter_valid = true;
    }

    // Fit sinusoidal model (big rune) using velocity-based LSQ
    // Returns true if fitting succeeded
    bool fitAngleModelLSQ();

    // Apply posterior constraints to the current model (a, ω, b bounds)
    // Should be called after fitAngleModelLSQ() for big runes
    void applyPosteriorConstraints();

    // Fit linear model (small rune): θ(t) = ω·t + b
    // Returns true if fitting succeeded
    bool fitLinearModelPublic();

private:
    int min_data_points_{};
    size_t max_observations_{};

    mutable std::shared_mutex observations_mutex_;
    std::deque<AngleObservation> observations_;

    mutable std::shared_mutex model_mutex_;
    AngleModel current_model_{};

    double last_prediction_error_{};
    double start_time_{};

    std::mt19937 random_engine_{std::random_device{}()};

    static constexpr double AMPLITUDE_MIN  = 0.780;
    static constexpr double AMPLITUDE_MAX  = 1.045;
    static constexpr double AMPLITUDE_INIT = (AMPLITUDE_MIN + AMPLITUDE_MAX) / 2.0;

    static constexpr double OMEGA_MIN  = 1.884;
    static constexpr double OMEGA_MAX  = 2.000;
    static constexpr double OMEGA_INIT = (OMEGA_MIN + OMEGA_MAX) / 2.0;

    static constexpr double OFFSET_BASE = 2.090;

    double last_angle_{};
    double last_timestamp_{};
    bool has_previous_observation_{};

    double last_real_observation_time_{};
    double last_real_observation_angle_{};
    std::deque<double> frame_interval_history_;
    static constexpr size_t MAX_FRAME_INTERVALS = 20;

    bool fitLinearModel(AngleModel& model);

    auto getRelativeTimeNoLock(double timestamp) const -> double;

    static bool linearRegression(
        const std::vector<double>& x_values, const std::vector<double>& y_values, double& slope,
        double& intercept);

    auto calculateFitError(const AngleModel& model) -> double;

    auto getAverageFrameInterval() const -> double;

    void generateSyntheticObservations(
        double loss_start_time, double loss_end_time, double avg_frame_interval);
};

} // namespace energy_meter
