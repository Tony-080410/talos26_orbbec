#include "L4_planning/aimer/aimer.hpp"
#include "core/math/normalize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

namespace fcs::L4 {
namespace {

using fcs::core::math::normalize_angle;

[[nodiscard]] uint64_t offset_timestamp_ns(uint64_t base_ns, double offset_s) noexcept {
    if (!std::isfinite(offset_s)) {
        return base_ns;
    }

    const long double shifted_ns =
        static_cast<long double>(base_ns) + static_cast<long double>(offset_s) * 1.0e9L;
    if (shifted_ns <= 0.0L) {
        return 0;
    }

    const long double max_ns = static_cast<long double>(std::numeric_limits<uint64_t>::max());
    if (shifted_ns >= max_ns) {
        return std::numeric_limits<uint64_t>::max();
    }

    return static_cast<uint64_t>(std::llround(shifted_ns));
}

[[nodiscard]] double center_yaw_from_xy(double x, double y) noexcept { return std::atan2(y, x); }

[[nodiscard]] double aim_ray_yaw_from_position(
    const Eigen::Vector3d& position, const Eigen::Vector3d& muzzle_pos) noexcept {
    return std::atan2(position.y() - muzzle_pos.y(), position.x() - muzzle_pos.x());
}

[[nodiscard]] std::optional<int>
    sanitize_preferred(const std::optional<int>& preferred, int armors_num) noexcept {
    if (!preferred.has_value()) {
        return std::nullopt;
    }
    if (*preferred < 0 || *preferred >= armors_num) {
        return std::nullopt;
    }
    return preferred;
}

[[nodiscard]] int pick_best_by_min_delta(
    const std::vector<int>& candidates, const std::vector<double>& delta_angles) noexcept {
    if (candidates.empty()) {
        return 0;
    }

    int best        = candidates.front();
    double best_abs = std::abs(delta_angles[static_cast<size_t>(best)]);
    for (const int id : candidates) {
        const double value = std::abs(delta_angles[static_cast<size_t>(id)]);
        if (value < best_abs) {
            best     = id;
            best_abs = value;
        }
    }
    return best;
}

[[nodiscard]] int pick_locked_or_min_delta(
    const std::vector<int>& candidates, const std::vector<double>& delta_angles,
    const std::optional<int>& preferred) noexcept {
    if (preferred.has_value()
        && std::find(candidates.begin(), candidates.end(), *preferred) != candidates.end()) {
        return *preferred;
    }
    return pick_best_by_min_delta(candidates, delta_angles);
}

template <size_t N>
[[nodiscard]] std::vector<double>
    build_delta_angles(const std::array<Eigen::Vector4d, N>& armor_poses, double center_yaw) {
    std::vector<double> deltas;
    deltas.reserve(N);
    for (const auto& pose : armor_poses) {
        deltas.push_back(normalize_angle(pose[3] - center_yaw));
    }
    return deltas;
}

template <size_t N>
[[nodiscard]] int select_armor_id(
    const std::array<Eigen::Vector4d, N>& armor_poses, double center_yaw,
    const ArmorAimContext& context, const AimerConfig& config, double target_v_yaw,
    bool is_outpost) noexcept {
    const int armors_num = static_cast<int>(N);
    const auto preferred = sanitize_preferred(context.preferred_armor_id, armors_num);

    if (!context.target_jumped) {
        return 0;
    }

    const std::vector<double> delta_angles = build_delta_angles(armor_poses, center_yaw);

    std::vector<int> all_ids;
    all_ids.reserve(armors_num);
    for (int i = 0; i < armors_num; ++i) {
        all_ids.push_back(i);
    }

    const auto fallback_best = [&]() noexcept {
        return pick_best_by_min_delta(all_ids, delta_angles);
    };

    if (context.phase == ArmorAimPhase::SingleArmor && !is_outpost) {
        const double front_window = config.front_window_deg * std::numbers::pi / 180.0;
        std::vector<int> candidates;
        candidates.reserve(armors_num);
        for (int i = 0; i < armors_num; ++i) {
            if (std::abs(delta_angles[static_cast<size_t>(i)]) <= front_window) {
                candidates.push_back(i);
            }
        }
        if (candidates.empty()) {
            return 0;
        }
        if (candidates.size() == 1U) {
            return candidates.front();
        }
        return pick_locked_or_min_delta(candidates, delta_angles, preferred);
    }

    if (context.phase == ArmorAimPhase::WholeCarPair && !is_outpost) {
        std::vector<int> pair_ids;
        pair_ids.reserve(2);
        const bool use_high_layer = armor_poses[1][2] > armor_poses[0][2];
        if (use_high_layer) {
            if (armors_num > 1) {
                pair_ids.push_back(1);
            }
            if (armors_num > 3) {
                pair_ids.push_back(3);
            }
        } else {
            pair_ids.push_back(0);
            if (armors_num > 2) {
                pair_ids.push_back(2);
            }
        }
        if (!pair_ids.empty()) {
            return pick_best_by_min_delta(pair_ids, delta_angles);
        }
    }

    (void)config;
    (void)target_v_yaw;
    return fallback_best();
}

[[nodiscard]] Eigen::Vector3d project_center_aim(
    const Eigen::Vector3d& center, const Eigen::Vector3d& muzzle_pos, double radius,
    double armor_z) noexcept {
    Eigen::Vector2d ray   = center.head<2>() - muzzle_pos.head<2>();
    const double ray_norm = ray.norm();
    if (ray_norm > 1e-6) {
        ray /= ray_norm;
    } else {
        const double yaw = center_yaw_from_xy(center.x(), center.y());
        ray              = Eigen::Vector2d{std::cos(yaw), std::sin(yaw)};
    }

    const Eigen::Vector2d projected_xy = center.head<2>() - radius * ray;
    return Eigen::Vector3d{projected_xy.x(), projected_xy.y(), armor_z};
}

} // namespace

// ============================================================================
// predict_position - compatibility helpers
// ============================================================================

Eigen::Vector3d
    Aimer::predict_position(const L3::RobotTargetState& target, double dt) const noexcept {
    return predict_aim_point(target, dt, {}, Eigen::Vector3d::Zero()).position;
}

Eigen::Vector3d
    Aimer::predict_position(const L3::OutpostTargetState& target, double dt) const noexcept {
    return predict_aim_point(target, dt, {}, Eigen::Vector3d::Zero()).position;
}

Aimer::PredictedAimPoint Aimer::predict_aim_point(
    const L3::RobotTargetState& target, double dt, const ArmorAimContext& context,
    const Eigen::Vector3d& muzzle_pos, std::optional<int> forced_armor_id) const noexcept {
    L3::RobotTargetState predicted = target;
    predicted.position += predicted.velocity * dt;
    predicted.yaw += predicted.v_yaw * dt;

    const auto armor_poses  = predicted.armor_poses();
    const double center_yaw = center_yaw_from_xy(predicted.position.x(), predicted.position.y());
    const int armor_id =
        sanitize_preferred(forced_armor_id, predicted.armors_num)
            .value_or(
                select_armor_id(armor_poses, center_yaw, context, config_, predicted.v_yaw, false));
    const Eigen::Vector4d& armor_pose = armor_poses[static_cast<size_t>(armor_id)];
    const double radius               = (armor_id % 2 == 0) ? predicted.radius0 : predicted.radius1;

    PredictedAimPoint result;
    result.armor_id = armor_id;
    result.diff_yaw = normalize_angle(armor_pose[3] - center_yaw);
    result.position = armor_pose.head<3>();

    if (context.phase == ArmorAimPhase::WholeCarCenter) {
        result.position = project_center_aim(predicted.position, muzzle_pos, radius, armor_pose[2]);
    }

    result.fire_yaw_diff =
        normalize_angle(armor_pose[3] - aim_ray_yaw_from_position(result.position, muzzle_pos));

    return result;
}

Aimer::PredictedAimPoint Aimer::predict_aim_point(
    const L3::OutpostTargetState& target, double dt, const ArmorAimContext& context,
    const Eigen::Vector3d& muzzle_pos, std::optional<int> forced_armor_id) const noexcept {
    L3::OutpostTargetState predicted = target;
    predicted.position += predicted.velocity.head<2>() * dt;
    predicted.yaw += predicted.v_yaw * dt;

    const auto armor_poses  = predicted.armor_poses();
    const double center_yaw = center_yaw_from_xy(predicted.position.x(), predicted.position.y());
    const int armor_id = sanitize_preferred(forced_armor_id, L3::OutpostTargetState::armors_num)
                             .value_or(select_armor_id(
                                 armor_poses, center_yaw, context, config_, predicted.v_yaw, true));
    const Eigen::Vector4d& armor_pose = armor_poses[static_cast<size_t>(armor_id)];

    PredictedAimPoint result;
    result.armor_id = armor_id;
    result.diff_yaw = normalize_angle(armor_pose[3] - center_yaw);
    result.position = armor_pose.head<3>();

    if (context.phase == ArmorAimPhase::WholeCarCenter) {
        result.position = project_center_aim(
            Eigen::Vector3d{predicted.position.x(), predicted.position.y(), 0.0}, muzzle_pos,
            L3::OutpostTargetState::radius, armor_pose[2]);
    }

    result.fire_yaw_diff =
        normalize_angle(armor_pose[3] - aim_ray_yaw_from_position(result.position, muzzle_pos));

    return result;
}

// ============================================================================
// aim_generic - Template method for targets with prediction
// ============================================================================

template <class TargetType>
std::expected<TargetPrediction, std::string> Aimer::aim_generic(
    const TargetType& target, const ArmorAimContext& context, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {

    TargetPrediction output;
    output.target_type     = get_target_type<TargetType>();
    output.timestamp_ns    = current_ns;
    output.target_velocity = target.velocity;
    output.target_v_yaw    = target.v_yaw;
    output.aim_phase       = context.phase;

    const double age =
        static_cast<double>(static_cast<int64_t>(current_ns) - static_cast<int64_t>(measurement_ns))
        * 1e-9;
    const double total_delay = age + prediction_delay;

    const Eigen::Vector3d muzzle_pos  = muzzle.translation();
    const auto rough_predicted        = predict_aim_point(target, total_delay, context, muzzle_pos);
    const int rough_selected_armor_id = rough_predicted.armor_id;
    const auto refine_total_prediction_time = [&](int forced_armor_id) noexcept {
        return refine_flying_time(
            solver,
            [&](double dt) -> Eigen::Vector3d {
                return predict_aim_point(target, dt, context, muzzle_pos, forced_armor_id).position
                     - muzzle_pos;
            },
            total_delay, bullet_speed);
    };

    int refined_armor_id         = rough_selected_armor_id;
    double total_prediction_time = total_delay;
    PredictedAimPoint predicted{};
    bool stabilized = false;

    // Stabilize ballistic refinement across armor handoffs so time-of-flight and aim point
    // describe the same armor when the final selection changes during iteration.
    for (int pass = 0; pass < 3; ++pass) {
        total_prediction_time = refine_total_prediction_time(refined_armor_id);
        predicted = predict_aim_point(target, total_prediction_time, context, muzzle_pos);
        if (predicted.armor_id == refined_armor_id) {
            stabilized = true;
            break;
        }
        refined_armor_id = predicted.armor_id;
    }

    if (!stabilized) {
        total_prediction_time = refine_total_prediction_time(refined_armor_id);
        predicted =
            predict_aim_point(target, total_prediction_time, context, muzzle_pos, refined_armor_id);
    }

    output.predicted_position      = predicted.position;
    output.predicted_future_ns     = offset_timestamp_ns(measurement_ns, total_prediction_time);
    output.flying_time             = total_prediction_time - total_delay;
    output.selected_armor_id       = predicted.armor_id;
    output.rough_selected_armor_id = rough_selected_armor_id;
    output.diff_yaw                = predicted.diff_yaw;
    output.fire_yaw_diff           = predicted.fire_yaw_diff;

    auto aim_angles =
        compute_gimbal_aim_angles(output.predicted_position, gimbal, muzzle, solver, bullet_speed);
    if (!aim_angles) {
        return std::unexpected(aim_angles.error());
    }

    output.aim_yaw   = aim_angles->first;
    output.aim_pitch = aim_angles->second;
    output.distance  = (output.predicted_position - muzzle.translation()).norm();

    return output;
}

// ============================================================================
// aim_generic_no_prediction - Template method for targets without prediction
// ============================================================================

template <class TargetType>
std::expected<TargetPrediction, std::string> Aimer::aim_generic_no_prediction(
    const TargetType& target, const GimbalTransform& gimbal, const MuzzleTransform& muzzle,
    uint64_t current_ns, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {

    TargetPrediction output;
    output.target_type     = get_target_type<TargetType>();
    output.timestamp_ns    = current_ns;
    output.target_velocity = Eigen::Vector3d::Zero();
    output.target_v_yaw    = 0.0;

    output.predicted_position  = target.predicted_position_odom;
    output.predicted_future_ns = target.predicted_future_ns;
    const auto solution =
        solver.solve(output.predicted_position - muzzle.translation(), bullet_speed);
    if (!solution.is_valid()) {
        return std::unexpected("ballistic solver failed to produce a valid aim solution");
    }
    output.flying_time = solution.time_of_flight;

    const auto [yaw, pitch] = compose_gimbal_aim_angles(solution, gimbal, muzzle);

    output.aim_yaw   = yaw;
    output.aim_pitch = pitch;
    output.distance  = (output.predicted_position - muzzle.translation()).norm();

    return output;
}

// ============================================================================
// aim - Robot
// ============================================================================

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::RobotTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double bullet_speed, const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim(
        target, ArmorAimContext{}, gimbal, muzzle, measurement_ns, current_ns, 0.0, bullet_speed,
        solver);
}

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::RobotTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double extra_prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim(
        target, ArmorAimContext{}, gimbal, muzzle, measurement_ns, current_ns,
        extra_prediction_delay, bullet_speed, solver);
}

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::RobotTargetState& target, const ArmorAimContext& context,
    const GimbalTransform& gimbal, const MuzzleTransform& muzzle, uint64_t measurement_ns,
    uint64_t current_ns, double extra_prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(
        target, context, gimbal, muzzle, measurement_ns, current_ns,
        config_.delay + extra_prediction_delay, bullet_speed, solver);
}

// ============================================================================
// aim - Outpost
// ============================================================================

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::OutpostTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double bullet_speed, const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim(
        target, ArmorAimContext{}, gimbal, muzzle, measurement_ns, current_ns, 0.0, bullet_speed,
        solver);
}

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::OutpostTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double extra_prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim(
        target, ArmorAimContext{}, gimbal, muzzle, measurement_ns, current_ns,
        extra_prediction_delay, bullet_speed, solver);
}

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::OutpostTargetState& target, const ArmorAimContext& context,
    const GimbalTransform& gimbal, const MuzzleTransform& muzzle, uint64_t measurement_ns,
    uint64_t current_ns, double extra_prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(
        target, context, gimbal, muzzle, measurement_ns, current_ns,
        config_.delay + extra_prediction_delay, bullet_speed, solver);
}

// ============================================================================
// aim - Rune (EnergyMeterState)
// ============================================================================

std::expected<TargetPrediction, std::string> Aimer::aim(
    const energy_meter::EnergyMeterState& state, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t current_ns, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic_no_prediction(state, gimbal, muzzle, current_ns, bullet_speed, solver);
}

} // namespace fcs::L4
