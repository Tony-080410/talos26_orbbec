#include "L3_estimation/energy_meter_solver/angle_fitter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace energy_meter {

AngleFitter::AngleFitter(double smooth_alpha, int min_data_points)
    : min_data_points_(min_data_points)
    , max_observations_(100)
    , last_prediction_error_(0)
    , start_time_(0)
    , smooth_alpha_(smooth_alpha)
    , fit_interval_ms_(50)
    , last_angle_(0.0)
    , last_timestamp_(0.0)
    , has_previous_observation_(false)
    , last_real_observation_time_(0)
    , last_real_observation_angle_(0) {
    current_model_.is_valid    = false;
    current_model_.is_big_rune = false;     // Will be set by fit methods
}

AngleFitter::~AngleFitter() = default;

void AngleFitter::reset() {
    {
        std::unique_lock<std::shared_mutex> lock(observations_mutex_);
        observations_.clear();
        frame_interval_history_.clear();
    }

    {
        std::unique_lock<std::shared_mutex> lock(model_mutex_);
        current_model_             = AngleModel();
        current_model_.is_valid    = false;
        current_model_.is_big_rune = false; // Will be set by fit methods
    }

    start_time_                  = 0.0;
    last_prediction_error_       = 0.0;
    last_angle_                  = 0.0;
    last_timestamp_              = 0.0;
    has_previous_observation_    = false;
    last_real_observation_time_  = 0.0;
    last_real_observation_angle_ = 0.0;
}

bool AngleFitter::addObservation(const AngleObservation& observation) {
    std::unique_lock<std::shared_mutex> lock(observations_mutex_);

    if (observations_.empty()) {
        start_time_ = observation.timestamp;
    } else {
        if (observation.timestamp < observations_.back().timestamp) {
            return false;
        }

        const double frame_interval = observation.timestamp - observations_.back().timestamp;

        if (frame_interval > 0.1 && frame_interval < 2.0) {
            bool has_valid_model = false;
            {
                lock.unlock();
                std::shared_lock<std::shared_mutex> model_lock(model_mutex_);
                has_valid_model = current_model_.is_valid;
            }
            lock.lock();

            if (has_valid_model && last_real_observation_time_ > 0) {
                const double avg_interval = getAverageFrameInterval();
                if (avg_interval > 1e-6) {
                    lock.unlock();
                    generateSyntheticObservations(
                        last_real_observation_time_, observation.timestamp, avg_interval);
                    lock.lock();
                }
            }
        }

        if (frame_interval > 0 && frame_interval < 0.1) {
            frame_interval_history_.push_back(frame_interval);
            if (frame_interval_history_.size() > MAX_FRAME_INTERVALS) {
                frame_interval_history_.pop_front();
            }
        }
    }

    last_real_observation_time_  = observation.timestamp;
    last_real_observation_angle_ = observation.continuous_angle;

    observations_.push_back(observation);
    if (observations_.size() > max_observations_) {
        observations_.pop_front();
    }

    lock.unlock();

    {
        std::unique_lock<std::shared_mutex> model_lock(model_mutex_);
        current_model_.last_continuous_angle = observation.continuous_angle;
        current_model_.last_observation_time = observation.timestamp;
        current_model_.data_points_used      = static_cast<int>(observations_.size());
    }

    return false;
}

bool AngleFitter::fitAngleModelLSQ() {
    std::shared_lock<std::shared_mutex> lock(observations_mutex_);

    if (observations_.size() < static_cast<size_t>(min_data_points_)) {
        return false;
    }

    std::vector<double> t_values;
    std::vector<double> v_values;
    t_values.reserve(observations_.size());
    v_values.reserve(observations_.size());

    for (const auto& obs : observations_) {
        t_values.push_back(getRelativeTimeNoLock(obs.timestamp));
        v_values.push_back(obs.velocity);
    }

    lock.unlock();

    const int n = static_cast<int>(t_values.size());
    if (n < 3) {
        return false;
    }

    double best_error = std::numeric_limits<double>::max();
    double best_a     = AMPLITUDE_INIT;
    double best_omega = OMEGA_INIT;
    double best_phi   = 0.0;
    double best_b     = OFFSET_BASE - best_a;

    for (double omega = OMEGA_MIN; omega <= OMEGA_MAX + 1e-6; omega += 0.005) {
        Eigen::MatrixXd H(n, 3);
        Eigen::VectorXd V(n);

        for (int i = 0; i < n; ++i) {
            const double t = t_values[i];
            H(i, 0)        = std::cos(omega * t);
            H(i, 1)        = std::sin(omega * t);
            H(i, 2)        = 1.0;
            V(i)           = v_values[i];
        }

        const Eigen::Matrix3d HtH = H.transpose() * H;
        const Eigen::Vector3d HtV = H.transpose() * V;
        const Eigen::Vector3d X   = HtH.ldlt().solve(HtV);

        const double A = X(0);
        const double B = X(1);
        const double b = X(2);

        const double a   = std::sqrt(A * A + B * B);
        const double phi = std::atan2(A, B);

        const Eigen::VectorXd residual = V - H * X;
        const double error             = residual.squaredNorm() / n;

        if (error < best_error) {
            best_error = error;
            best_a     = a;
            best_omega = omega;
            best_phi   = phi;
            best_b     = b;
        }
    }

    const double t_last = t_values.back();
    double best_t0      = best_phi - best_omega * t_last;
    best_t0             = std::atan2(std::sin(best_t0), std::cos(best_t0));

    lock.lock();
    const double theta_last = observations_.back().continuous_angle;
    lock.unlock();

    const double best_c =
        theta_last
        - (-best_a / best_omega * std::cos(best_omega * t_last + best_t0) + best_b * t_last);

    {
        std::unique_lock<std::shared_mutex> model_lock(model_mutex_);
        current_model_.sin_amplitude    = best_a;
        current_model_.sin_omega        = best_omega;
        current_model_.sin_phase        = best_t0;
        current_model_.sin_offset       = best_b;
        current_model_.sin_const_term   = best_c;
        current_model_.fit_error        = std::sqrt(best_error);
        current_model_.is_big_rune      = true;
        current_model_.is_valid         = true;
        current_model_.data_points_used = n;
        current_model_.last_fit_time    = t_last + start_time_;
    }

    return true;
}

void AngleFitter::applyPosteriorConstraints() {
    std::unique_lock<std::shared_mutex> lock(model_mutex_);

    if (!current_model_.is_valid || !current_model_.is_big_rune) {
        return;
    }

    current_model_.sin_amplitude =
        std::clamp(current_model_.sin_amplitude, AMPLITUDE_MIN, AMPLITUDE_MAX);

    current_model_.sin_omega = std::clamp(current_model_.sin_omega, OMEGA_MIN, OMEGA_MAX);

    current_model_.sin_offset = OFFSET_BASE - current_model_.sin_amplitude;
}

bool AngleFitter::fitLinearModelPublic() {
    AngleModel linear_model;
    if (fitLinearModel(linear_model)) {
        linear_model.is_big_rune = false;

        std::unique_lock<std::shared_mutex> lock(model_mutex_);
        linear_model.last_continuous_angle = current_model_.last_continuous_angle;
        linear_model.last_observation_time = current_model_.last_observation_time;
        linear_model.filtered_velocity     = current_model_.filtered_velocity;
        linear_model.velocity_filter_valid = current_model_.velocity_filter_valid;
        linear_model.is_valid              = true;
        linear_model.data_points_used      = current_model_.data_points_used;

        current_model_ = linear_model;
        return true;
    }
    return false;
}

bool AngleFitter::fitLinearModel(AngleModel& model) {
    std::shared_lock<std::shared_mutex> lock(observations_mutex_);

    if (observations_.size() < static_cast<size_t>(min_data_points_)) {
        return false;
    }

    std::vector<double> all_t_values;
    std::vector<double> all_angle_values;
    all_t_values.reserve(observations_.size());
    all_angle_values.reserve(observations_.size());

    for (size_t i = 0; i < observations_.size(); ++i) {
        all_t_values.push_back(getRelativeTimeNoLock(observations_[i].timestamp));
        all_angle_values.push_back(observations_[i].continuous_angle);
    }

    lock.unlock();

    double slope     = 0.0;
    double intercept = 0.0;
    if (!linearRegression(all_t_values, all_angle_values, slope, intercept)) {
        return false;
    }

    model.lin_omega   = slope;
    model.lin_offset  = intercept;
    model.is_big_rune = false;
    model.fit_error   = calculateFitError(model);

    return true;
}

bool AngleFitter::linearRegression(
    const std::vector<double>& x_values, const std::vector<double>& y_values, double& slope,
    double& intercept) {
    if (x_values.size() != y_values.size() || x_values.size() < 2) {
        return false;
    }

    const int n = static_cast<int>(x_values.size());

    double sum_x  = 0.0;
    double sum_y  = 0.0;
    double sum_xy = 0.0;
    double sum_x2 = 0.0;

    for (int i = 0; i < n; ++i) {
        sum_x += x_values[i];
        sum_y += y_values[i];
        sum_xy += x_values[i] * y_values[i];
        sum_x2 += x_values[i] * x_values[i];
    }

    const double denominator = n * sum_x2 - sum_x * sum_x;
    if (std::fabs(denominator) < 1e-10) {
        return false;
    }

    slope     = (n * sum_xy - sum_x * sum_y) / denominator;
    intercept = (sum_y - slope * sum_x) / n;

    return true;
}

auto AngleFitter::getAverageFrameInterval() const -> double {
    if (frame_interval_history_.empty()) {
        return 1.0 / 30.0;
    }

    double sum = 0.0;
    for (double interval : frame_interval_history_) {
        sum += interval;
    }

    return sum / static_cast<double>(frame_interval_history_.size());
}

void AngleFitter::generateSyntheticObservations(
    double loss_start_time, double loss_end_time, double avg_frame_interval) {
    std::unique_lock<std::shared_mutex> obs_lock(observations_mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_mutex_);

    if (!current_model_.is_valid || loss_end_time <= loss_start_time) {
        return;
    }

    const double loss_duration = loss_end_time - loss_start_time;
    int num_synthetic_frames   = static_cast<int>(loss_duration / avg_frame_interval);
    if (num_synthetic_frames > 30) {
        num_synthetic_frames = 30;
    }

    if (num_synthetic_frames < 1) {
        return;
    }

    const double window_start_time =
        observations_.empty() ? start_time_ : observations_.front().timestamp;

    for (int i = 1; i <= num_synthetic_frames; ++i) {
        const double synthetic_time = loss_start_time + i * avg_frame_interval;

        double predicted_angle = 0.0;
        const double t         = synthetic_time - window_start_time;
        if (current_model_.is_big_rune) {
            const double a     = current_model_.sin_amplitude;
            const double omega = current_model_.sin_omega;
            const double t0    = current_model_.sin_phase;
            const double b     = current_model_.sin_offset;
            const double c     = current_model_.sin_const_term;
            predicted_angle    = -a / omega * std::cos(omega * t + t0) + b * t + c;
        } else {
            predicted_angle = current_model_.lin_omega * t + current_model_.lin_offset;
        }

        AngleObservation synthetic_obs;
        synthetic_obs.timestamp        = synthetic_time;
        synthetic_obs.continuous_angle = predicted_angle;
        if (!observations_.empty()) {
            synthetic_obs.blade_offset   = observations_.back().blade_offset;
            synthetic_obs.absolute_angle = observations_.back().absolute_angle;
        }

        observations_.push_back(synthetic_obs);
        if (observations_.size() > max_observations_) {
            observations_.pop_front();
        }
    }
}

auto AngleFitter::getRelativeTimeNoLock(double timestamp) const -> double {
    if (!observations_.empty()) {
        return timestamp - observations_.front().timestamp;
    }
    return timestamp - start_time_;
}

auto AngleFitter::calculateFitError(const AngleModel& model) -> double {
    std::shared_lock<std::shared_mutex> lock(observations_mutex_);

    if (observations_.size() < 2) {
        return 0.0;
    }

    double sum_squared_error = 0.0;
    int count                = 0;

    for (size_t i = 1; i < observations_.size(); ++i) {
        const double dt = observations_[i].timestamp - observations_[i - 1].timestamp;
        const double actual_delta =
            observations_[i].continuous_angle - observations_[i - 1].continuous_angle;
        double predicted_delta = 0.0;

        if (dt < 1e-6) {
            continue;
        }

        if (model.is_big_rune) {
            double v_prev = 0.0;
            if (i >= 2) {
                const double dt_prev =
                    observations_[i - 1].timestamp - observations_[i - 2].timestamp;
                if (dt_prev > 1e-6) {
                    v_prev = (observations_[i - 1].continuous_angle
                              - observations_[i - 2].continuous_angle)
                           / dt_prev;
                }
            } else {
                v_prev = actual_delta / dt;
            }

            const double a     = model.sin_amplitude;
            const double b     = model.sin_offset;
            const double omega = model.sin_omega;

            const double sin_phi = std::clamp((v_prev - b) / a, -1.0, 1.0);
            const double phi_now = std::asin(sin_phi);

            const double phi_future = phi_now + omega * dt;
            predicted_delta =
                -a / omega * std::cos(phi_future) + a / omega * std::cos(phi_now) + b * dt;
        } else {
            predicted_delta = model.lin_omega * dt;
        }

        const double error = actual_delta - predicted_delta;
        sum_squared_error += error * error;
        ++count;
    }

    if (count == 0) {
        return 0.0;
    }

    return std::sqrt(sum_squared_error / count);
}

} // namespace energy_meter
