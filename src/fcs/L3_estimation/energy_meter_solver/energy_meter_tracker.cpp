#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"

#include "L3_estimation/energy_meter_solver/types.hpp"
#include "euler.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

namespace energy_meter {

namespace {

constexpr double kBladeAngle             = 2.0 * M_PI / static_cast<double>(ARMORS_NUM);
constexpr double kMinDirectionDelta      = 0.003;
constexpr double kStableBladeTieSlackRad = 0.08;

[[nodiscard]] auto positive_mod(int value, int mod) noexcept -> int {
    const int x = value % mod;
    return x < 0 ? x + mod : x;
}

[[nodiscard]] auto circular_blade_distance(int lhs, int rhs) noexcept -> int {
    const int delta = std::abs(positive_mod(lhs - rhs, ARMORS_NUM));
    return std::min(delta, ARMORS_NUM - delta);
}

template <typename ModelT>
[[nodiscard]] constexpr auto is_big_tracker() noexcept -> bool {
    return std::is_same_v<ModelT, BigRuneModel>;
}

template <typename ModelT>
[[nodiscard]] auto default_speed() noexcept -> double {
    if constexpr (is_big_tracker<ModelT>()) {
        return BigRuneModel::AMPLITUDE_SUM / 2.0;
    } else {
        return FIXED_RUNE_SPEED;
    }
}

[[nodiscard]] auto model_angle(const BigFitResult& fit, double t) noexcept -> double {
    return -(fit.a / fit.omega) * std::cos(fit.omega * t + fit.phase) + fit.b * t + fit.c;
}

[[nodiscard]] auto robust_loss(double residual, double threshold) noexcept -> double {
    const double delta = std::max(1e-6, threshold);
    const double abs_r = std::abs(residual);
    if (abs_r <= delta) {
        return residual * residual;
    }
    return 2.0 * delta * abs_r - delta * delta;
}

[[nodiscard]] auto outside_range(double value, double lower, double upper) noexcept -> double {
    if (value < lower) {
        return lower - value;
    }
    if (value > upper) {
        return value - upper;
    }
    return 0.0;
}

template <typename ModelT>
struct LinearOmegaFit {
    BigFitResult fit{};
    double score{std::numeric_limits<double>::infinity()};
    bool valid{false};
};

template <typename ModelT>
[[nodiscard]] auto fit_linear_at_omega(
    const std::vector<typename Tracker<ModelT>::AngleSample>& samples,
    const std::vector<std::size_t>& indices, const typename Tracker<ModelT>::Params& params,
    double omega, double sign) -> LinearOmegaFit<ModelT> {
    static_assert(is_big_tracker<ModelT>());

    LinearOmegaFit<ModelT> result{};
    if (indices.size() < 4 || omega < BigRuneModel::W_LOWER || omega > BigRuneModel::W_UPPER) {
        return result;
    }

    Eigen::MatrixXd x(static_cast<Eigen::Index>(indices.size()), 4);
    Eigen::VectorXd y(static_cast<Eigen::Index>(indices.size()));
    for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(indices.size()); ++row) {
        const auto& sample = samples[indices[static_cast<std::size_t>(row)]];
        x(row, 0)          = std::sin(omega * sample.t);
        x(row, 1)          = std::cos(omega * sample.t);
        x(row, 2)          = sample.t;
        x(row, 3)          = 1.0;
        y(row)             = sign * sample.angle;
    }

    const Eigen::Vector4d theta = x.colPivHouseholderQr().solve(y);
    if (!theta.allFinite()) {
        return result;
    }

    const double b_sin     = theta[0];
    const double c_cos     = theta[1];
    const double b_lsq     = theta[2];
    const double amplitude = std::hypot(b_sin, c_cos);
    if (amplitude < 1e-9) {
        return result;
    }

    const double raw_a = omega * amplitude;
    const double a     = std::clamp(raw_a, BigRuneModel::A_LOWER, BigRuneModel::A_UPPER);
    const double phase = normalize_rad(std::atan2(b_sin, -c_cos));
    const double b     = BigRuneModel::AMPLITUDE_SUM - a;

    double c_sum = 0.0;
    for (const std::size_t idx : indices) {
        const auto& sample = samples[idx];
        const double base  = -(a / omega) * std::cos(omega * sample.t + phase) + b * sample.t;
        c_sum += sign * sample.angle - base;
    }
    const double c = c_sum / static_cast<double>(indices.size());

    BigFitResult fit{
        .a     = a,
        .omega = omega,
        .phase = phase,
        .c     = c,
        .t     = samples.back().t,
        .b     = b,
        .valid = true,
    };

    const double threshold = std::max(1e-6, params.fit_inlier_threshold_rad);
    double score           = 0.0;
    for (const std::size_t idx : indices) {
        const auto& sample = samples[idx];
        score += robust_loss(model_angle(fit, sample.t) - sign * sample.angle, threshold);
    }
    score /= static_cast<double>(indices.size());

    const double constraint_error = raw_a + b_lsq - BigRuneModel::AMPLITUDE_SUM;
    const double a_bound_error = outside_range(raw_a, BigRuneModel::A_LOWER, BigRuneModel::A_UPPER);
    score += std::max(0.0, params.fit_penalty_a)
           * (constraint_error * constraint_error + 4.0 * a_bound_error * a_bound_error);

    result.fit   = fit;
    result.score = score;
    result.valid = std::isfinite(score) && std::isfinite(fit.a) && std::isfinite(fit.omega)
                && std::isfinite(fit.phase) && std::isfinite(fit.c);
    return result;
}

template <typename ModelT>
[[nodiscard]] auto optimize_omega_fit(
    const std::vector<typename Tracker<ModelT>::AngleSample>& samples,
    const std::vector<std::size_t>& indices, const typename Tracker<ModelT>::Params& params,
    const BigFitResult& prev_fit, double sign) -> LinearOmegaFit<ModelT> {
    static_assert(is_big_tracker<ModelT>());

    constexpr double rule_lower = BigRuneModel::W_LOWER;
    constexpr double rule_upper = BigRuneModel::W_UPPER;
    const bool use_local_search =
        prev_fit.valid && prev_fit.omega >= rule_lower && prev_fit.omega <= rule_upper;
    const double lower =
        use_local_search ? std::max(rule_lower, prev_fit.omega - 0.015) : rule_lower;
    const double upper =
        use_local_search ? std::min(rule_upper, prev_fit.omega + 0.015) : rule_upper;
    const int grid_count   = use_local_search ? std::max(9, params.fit_iterations * 4)
                                              : std::max(24, params.fit_iterations * 12);
    const double grid_step = (upper - lower) / static_cast<double>(grid_count - 1);

    LinearOmegaFit<ModelT> best{};
    auto consider = [&](double omega) {
        const auto candidate = fit_linear_at_omega<ModelT>(
            samples, indices, params, std::clamp(omega, rule_lower, rule_upper), sign);
        if (candidate.valid && candidate.score < best.score) {
            best = candidate;
        }
    };

    for (int i = 0; i < grid_count; ++i) {
        consider(lower + grid_step * static_cast<double>(i));
    }
    if (prev_fit.valid) {
        consider(prev_fit.omega);
    }

    if (!best.valid) {
        return best;
    }

    double lo = std::max(rule_lower, best.fit.omega - grid_step);
    double hi = std::min(rule_upper, best.fit.omega + grid_step);
    if (hi - lo < 1e-8) {
        return best;
    }

    constexpr double inv_phi = 0.6180339887498948482;
    double x1                = hi - inv_phi * (hi - lo);
    double x2                = lo + inv_phi * (hi - lo);
    auto f1                  = fit_linear_at_omega<ModelT>(samples, indices, params, x1, sign);
    auto f2                  = fit_linear_at_omega<ModelT>(samples, indices, params, x2, sign);

    const int refine_iterations = std::max(12, params.fit_iterations * 8);
    for (int iter = 0; iter < refine_iterations; ++iter) {
        if (!f1.valid || (f2.valid && f2.score < f1.score)) {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + inv_phi * (hi - lo);
            f2 = fit_linear_at_omega<ModelT>(samples, indices, params, x2, sign);
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - inv_phi * (hi - lo);
            f1 = fit_linear_at_omega<ModelT>(samples, indices, params, x1, sign);
        }
    }

    consider(0.5 * (lo + hi));
    return best;
}

template <typename ModelT>
[[nodiscard]] auto fit_big_model(
    const std::vector<typename Tracker<ModelT>::AngleSample>& samples,
    const typename Tracker<ModelT>::Params& params, const BigFitResult& prev_fit, int direction)
    -> BigFitResult {
    static_assert(is_big_tracker<ModelT>());

    BigFitResult result{};
    if (static_cast<int>(samples.size()) < params.min_fit_samples) {
        return result;
    }

    std::vector<std::size_t> candidate_indices;
    candidate_indices.reserve(samples.size());
    const double newest_t = samples.back().t;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (newest_t - samples[i].t <= params.fit_window_sec) {
            candidate_indices.push_back(i);
        }
    }

    if (static_cast<int>(candidate_indices.size()) < params.min_fit_samples) {
        return result;
    }

    if (static_cast<int>(candidate_indices.size()) > params.max_fit_samples) {
        std::vector<std::size_t> downsampled;
        downsampled.reserve(params.max_fit_samples);
        const double step = static_cast<double>(candidate_indices.size() - 1)
                          / static_cast<double>(params.max_fit_samples - 1);
        for (int i = 0; i < params.max_fit_samples; ++i) {
            downsampled.push_back(
                candidate_indices[static_cast<std::size_t>(std::llround(step * i))]);
        }
        candidate_indices = std::move(downsampled);
    }

    const double sign = direction < 0 ? -1.0 : 1.0;

    std::vector<std::size_t> inliers = candidate_indices;
    LinearOmegaFit<ModelT> best_fit{};
    for (int iter = 0; iter < params.fit_iterations; ++iter) {
        if (static_cast<int>(inliers.size()) < params.min_fit_samples) {
            break;
        }

        best_fit = optimize_omega_fit<ModelT>(samples, inliers, params, prev_fit, sign);
        if (!best_fit.valid) {
            return {};
        }

        std::vector<std::size_t> next_inliers;
        next_inliers.reserve(candidate_indices.size());
        for (const std::size_t idx : candidate_indices) {
            const auto& sample = samples[idx];
            if (std::abs(model_angle(best_fit.fit, sample.t) - sign * sample.angle)
                <= params.fit_inlier_threshold_rad) {
                next_inliers.push_back(idx);
            }
        }

        if (next_inliers == inliers) {
            break;
        }
        inliers = std::move(next_inliers);
    }

    if (!best_fit.valid) {
        return {};
    }
    best_fit.fit.t = newest_t;
    return best_fit.fit;
}

template <typename ModelT>
[[nodiscard]] auto extract_raw_angle(
    const Eigen::Matrix3d& reference_rotation, const Eigen::Quaterniond& target_quat) noexcept
    -> double {
    const Eigen::Matrix3d relative =
        reference_rotation.transpose() * target_quat.toRotationMatrix();
    return std::atan2(relative(2, 1), relative(2, 2));
}

} // namespace

template <typename ModelT>
void Tracker<ModelT>::reset_runtime_state() noexcept {
    radius_ = 0.7;
    center_.setZero();
    yaw_                    = 0.0;
    pitch_                  = 0.0;
    dt_                     = 0.0;
    initialized_            = false;
    reference_timestamp_ns_ = 0;
    reference_rotation_.setIdentity();
    current_time_sec_    = 0.0;
    raw_angle_           = 0.0;
    reference_roll_      = 0.0;
    continuous_angle_    = 0.0;
    cumulative_switches_ = 0;
    blade_id_            = 0;
    dir_                 = 0;
    dir_votes_pos_       = 0;
    dir_votes_neg_       = 0;
    dir_detect_count_    = 0;
    detecting_count_     = 0;
    lost_count_          = 0;
    samples_.clear();
    fit_         = {};
    model_valid_ = false;
}

template <typename ModelT>
void Tracker<ModelT>::reset_filter() noexcept {
    reset_runtime_state();
    state = IDLE;
}

template <typename ModelT>
void Tracker<ModelT>::initialize_reference(
    const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
    const Eigen::Quaterniond& target_quat) noexcept {
    center_              = r_center;
    radius_              = (target_pos - r_center).norm();
    reference_rotation_  = target_quat.toRotationMatrix();
    raw_angle_           = 0.0;
    continuous_angle_    = 0.0;
    cumulative_switches_ = 0;
    blade_id_            = 0;
    current_time_sec_    = 0.0;
    initialized_         = true;

    const auto [roll, pitch, yaw] = math_fuxk::rpy(target_quat).rpy();
    reference_roll_               = roll;
    pitch_                        = pitch;
    yaw_                          = yaw;

    samples_.clear();
    samples_.push_back(AngleSample{0.0, 0.0});
    fit_         = {};
    model_valid_ = !is_big_tracker<ModelT>();
}

template <typename ModelT>
bool Tracker<ModelT>::first_meet_u(
    const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
    const Eigen::Quaterniond& target_quat) {
    reset_runtime_state();
    reference_timestamp_ns_ = 0;
    initialize_reference(r_center, target_pos, target_quat);
    state = DETECTING;
    return true;
}

template <typename ModelT>
void Tracker<ModelT>::predict(double dt) noexcept {
    dt_ = std::max(0.0, dt);
}

template <typename ModelT>
auto Tracker<ModelT>::instantaneous_speed(double t) const noexcept -> double {
    if constexpr (is_big_tracker<ModelT>()) {
        if (fit_.valid) {
            return fit_.a * std::sin(fit_.omega * t + fit_.phase) + fit_.b;
        }
    }
    return default_speed<ModelT>();
}

template <typename ModelT>
auto Tracker<ModelT>::select_target(
    const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) const -> std::optional<std::size_t> {
    if (target_positions.empty() || target_quats.empty()
        || target_positions.size() != target_quats.size()) {
        return std::nullopt;
    }
    if (!initialized_) {
        return std::size_t{0};
    }

    const double predicted_angle =
        continuous_angle_
        + static_cast<double>(dir_) * instantaneous_speed(current_time_sec_) * dt_;

    double best_cost        = std::numeric_limits<double>::max();
    double best_residual    = std::numeric_limits<double>::max();
    int best_blade_distance = ARMORS_NUM;
    std::optional<std::size_t> best_index;
    for (std::size_t i = 0; i < target_quats.size(); ++i) {
        const double raw       = extract_raw_angle<ModelT>(reference_rotation_, target_quats[i]);
        const double delta_raw = normalize_rad(raw - raw_angle_);
        const int switch_delta = static_cast<int>(std::llround(delta_raw / kBladeAngle));
        const int candidate_blade_id =
            positive_mod(cumulative_switches_ + switch_delta, ARMORS_NUM);
        const double candidate =
            continuous_angle_ + delta_raw - static_cast<double>(switch_delta) * kBladeAngle;
        const double residual        = std::abs(candidate - predicted_angle);
        const double radius_residual = std::abs((target_positions[i] - center_).norm() - radius_);
        const double cost            = residual + 0.15 * radius_residual;
        const int blade_distance     = circular_blade_distance(candidate_blade_id, blade_id_);
        const bool better_cost       = cost + 1e-9 < best_cost;
        const bool stable_tie =
            cost <= best_cost + kStableBladeTieSlackRad && blade_distance < best_blade_distance;
        if (better_cost || stable_tie) {
            best_cost           = cost;
            best_residual       = residual;
            best_blade_distance = blade_distance;
            best_index          = i;
        }
    }

    const double dynamic_gate = params.angle_match_gate_rad + 2.1 * std::max(0.0, dt_);
    if (!best_index || best_residual > dynamic_gate) {
        return std::nullopt;
    }
    return best_index;
}

template <typename ModelT>
void Tracker<ModelT>::update_direction(double new_angle) noexcept {
    if (dir_detect_count_ > 0) {
        const double delta = new_angle - continuous_angle_;
        if (delta > kMinDirectionDelta) {
            ++dir_votes_pos_;
        } else if (delta < -kMinDirectionDelta) {
            ++dir_votes_neg_;
        }
        if (dir_votes_pos_ != dir_votes_neg_) {
            dir_ = dir_votes_pos_ > dir_votes_neg_ ? 1 : -1;
        }
    }
    ++dir_detect_count_;
    if (dir_detect_count_ >= params.direction_lock_frames && dir_ == 0) {
        dir_ = dir_votes_pos_ >= dir_votes_neg_ ? 1 : -1;
    }
}

template <typename ModelT>
bool Tracker<ModelT>::update(
    const Eigen::Vector3d& rcenter_position, const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {
    if (target_positions.empty() || target_quats.empty()
        || target_positions.size() != target_quats.size()) {
        state_machine(false);
        return false;
    }

    const auto selected = select_target(target_positions, target_quats);
    if (!selected.has_value()) {
        state_machine(false);
        return false;
    }

    const std::size_t idx = *selected;
    if (!initialized_) {
        initialize_reference(rcenter_position, target_positions[idx], target_quats[idx]);
    }

    center_ = rcenter_position;
    radius_ = (target_positions[idx] - rcenter_position).norm();

    const auto [roll, pitch, yaw] = math_fuxk::rpy(target_quats[idx]).rpy();
    (void)roll;
    pitch_ = pitch;
    yaw_   = yaw;

    const double raw       = extract_raw_angle<ModelT>(reference_rotation_, target_quats[idx]);
    const double delta_raw = normalize_rad(raw - raw_angle_);
    const int switch_delta = static_cast<int>(std::llround(delta_raw / kBladeAngle));
    const double new_angle =
        continuous_angle_ + delta_raw - static_cast<double>(switch_delta) * kBladeAngle;

    update_direction(new_angle);

    raw_angle_ = raw;
    cumulative_switches_ += switch_delta;
    continuous_angle_ = new_angle;
    blade_id_         = positive_mod(cumulative_switches_, ARMORS_NUM);

    state_machine(true);
    if (state == TRACKING || state == TEMP_LOST) {
        samples_.push_back(AngleSample{current_time_sec_, continuous_angle_});
        if constexpr (is_big_tracker<ModelT>()) {
            update_fit();
        } else {
            model_valid_ = true;
        }
    }

    return true;
}

template <typename ModelT>
void Tracker<ModelT>::step(
    double dt, const Eigen::Vector3d& rcenter_position,
    const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {
    predict(dt);
    current_time_sec_ += std::max(0.0, dt_);
    update(rcenter_position, target_positions, target_quats);
}

template <typename ModelT>
void Tracker<ModelT>::update_fit() {
    if constexpr (is_big_tracker<ModelT>()) {
        const auto raw_fit = fit_big_model<ModelT>(samples_, params, fit_, dir_);
        if (!raw_fit.valid) {
            model_valid_ = false;
            return;
        }

        if (!fit_.valid) {
            fit_ = raw_fit;
        } else {
            const double alpha = std::clamp(params.fit_filter_alpha, 0.0, 1.0);
            fit_.a             = fit_.a + alpha * (raw_fit.a - fit_.a);
            fit_.omega         = fit_.omega + alpha * (raw_fit.omega - fit_.omega);
            fit_.phase         = fit_.phase + alpha * normalize_rad(raw_fit.phase - fit_.phase);
            fit_.c             = fit_.c + alpha * (raw_fit.c - fit_.c);
            fit_.t             = raw_fit.t;
            fit_.b             = BigRuneModel::AMPLITUDE_SUM - fit_.a;
            fit_.valid         = true;
        }
        fit_.t       = current_time_sec_;
        fit_.b       = BigRuneModel::AMPLITUDE_SUM - fit_.a;
        model_valid_ = static_cast<int>(samples_.size()) >= params.min_fit_samples;
    }
}

template <typename ModelT>
void Tracker<ModelT>::state_machine(bool found) noexcept {
    switch (state) {
    case IDLE:
        if (found) {
            state            = DETECTING;
            detecting_count_ = 1;
        }
        break;

    case DETECTING:
        if (found) {
            ++detecting_count_;
            if (detecting_count_ > params.tracking_thres) {
                state            = TRACKING;
                detecting_count_ = 0;
            }
        } else {
            state            = IDLE;
            detecting_count_ = 0;
            model_valid_     = false;
        }
        break;

    case TRACKING:
        if (!found) {
            state       = TEMP_LOST;
            lost_count_ = 1;
        }
        break;

    case TEMP_LOST:
        if (found) {
            state       = TRACKING;
            lost_count_ = 0;
        } else {
            ++lost_count_;
            if (lost_count_ > params.lost_thres) {
                reset_filter();
            }
        }
        break;
    }
}

template struct Tracker<BigRuneModel>;
template struct Tracker<SmallRuneModel>;

} // namespace energy_meter
