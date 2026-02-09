#include "fcs/L4_planning/aimer/aimer.hpp"

#include <numbers>

namespace fcs::L4 {

// ============================================================================
// predict_position - Robot
// ============================================================================

Eigen::Vector3d Aimer::predict_position(const L3::RobotTargetState& target, double dt) noexcept {
    const Eigen::Vector3d center   = target.position;
    const Eigen::Vector3d velocity = target.velocity;
    const double yaw               = target.yaw;
    const double v_yaw             = target.v_yaw;
    const double h                 = target.z1 - center.z();

    const Eigen::Vector3d center_pred = center + velocity * dt;
    const double yaw_pred             = yaw + v_yaw * dt;

    const int n                   = target.armors_num;
    double min_dist               = 1e10;
    Eigen::Vector3d closest_armor = center_pred;

    for (int i = 0; i < n; ++i) {
        const double armor_yaw = yaw_pred + i * (2.0 * std::numbers::pi / n);
        const double r         = (i % 2 == 0) ? target.radius0 : target.radius1;

        const double armor_x = center_pred.x() - r * std::cos(armor_yaw);
        const double armor_y = center_pred.y() - r * std::sin(armor_yaw);
        const double armor_z = center_pred.z() + ((i % 2 == 0) ? 0.0 : h);

        const Eigen::Vector3d armor_pos{armor_x, armor_y, armor_z};
        const double dist = armor_pos.norm();
        if (dist < min_dist) {
            min_dist      = dist;
            closest_armor = armor_pos;
        }
    }

    return closest_armor;
}

// ============================================================================
// predict_position - Outpost
// ============================================================================

Eigen::Vector3d Aimer::predict_position(const L3::OutpostTargetState& target, double dt) noexcept {
    const Eigen::Vector2d center_2d = target.position;
    const Eigen::Vector3d velocity  = target.velocity;
    const double yaw                = target.yaw;
    const double v_yaw              = target.v_yaw;
    constexpr double r              = L3::OutpostTargetState::radius;

    const Eigen::Vector2d center_pred = center_2d + velocity.head<2>() * dt;

    const double center_x_pred = center_pred.x();
    const double center_y_pred = center_pred.y();
    const double yaw_pred      = yaw + v_yaw * dt;

    double min_dist = 1e10;
    Eigen::Vector3d closest_armor{center_x_pred, center_y_pred, target.z[0]};

    for (int i = 0; i < 3; ++i) {
        const double armor_yaw = yaw_pred + i * (2.0 * std::numbers::pi / 3.0);
        const double armor_x   = center_x_pred - r * std::cos(armor_yaw);
        const double armor_y   = center_y_pred - r * std::sin(armor_yaw);
        const Eigen::Vector3d armor_pos{armor_x, armor_y, target.z[i]};

        const double dist = armor_pos.norm();
        if (dist < min_dist) {
            min_dist      = dist;
            closest_armor = armor_pos;
        }
    }

    return closest_armor;
}

// ============================================================================
// aim_generic - Template method for targets with prediction
// ============================================================================

template <class TargetType>
std::expected<TargetPrediction, std::string> Aimer::aim_generic(
    const TargetType& target, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
    uint64_t measurement_ns, uint64_t current_ns, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {

    TargetPrediction output;
    output.target_type     = get_target_type<TargetType>();
    output.timestamp_ns    = current_ns;
    output.target_velocity = target.velocity;
    output.target_v_yaw    = target.v_yaw;

    const double age         = static_cast<double>(current_ns - measurement_ns) * 1e-9;
    const double total_delay = age + config_.delay;

    const Eigen::Vector3d t            = muzzle->translation();
    const double total_prediction_time = refine_flying_time(
        solver, [&](double dt) -> Eigen::Vector3d { return predict_position(target, dt) - t; },
        total_delay, bullet_speed);

    output.predicted_position = predict_position(target, total_prediction_time);
    output.predicted_future_ns =
        measurement_ns + static_cast<uint64_t>(total_prediction_time * 1e9);
    output.flying_time = total_prediction_time - total_delay;

    const auto [yaw, pitch] =
        compute_gimbal_aim_angles(output.predicted_position, gimbal, muzzle, solver, bullet_speed);

    output.aim_yaw   = yaw;
    output.aim_pitch = pitch;
    output.distance  = output.predicted_position.norm();

    return output;
}

// ============================================================================
// aim_generic_no_prediction - Template method for targets without prediction
// ============================================================================

template <class TargetType>
std::expected<TargetPrediction, std::string> Aimer::aim_generic_no_prediction(
    const TargetType& target, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
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
        solver.solve(output.predicted_position - muzzle->translation(), bullet_speed);
    output.flying_time = solution.time_of_flight;

    const auto [yaw, pitch] =
        compute_gimbal_aim_angles(output.predicted_position, gimbal, muzzle, solver, bullet_speed);

    output.aim_yaw   = yaw;
    output.aim_pitch = pitch;
    output.distance  = output.predicted_position.norm();

    return output;
}

// ============================================================================
// aim - Robot
// ============================================================================

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::RobotTargetState& target, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
    uint64_t measurement_ns, uint64_t current_ns, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(target, gimbal, muzzle, measurement_ns, current_ns, bullet_speed, solver);
}

// ============================================================================
// aim - Outpost
// ============================================================================

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::OutpostTargetState& target, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
    uint64_t measurement_ns, uint64_t current_ns, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(target, gimbal, muzzle, measurement_ns, current_ns, bullet_speed, solver);
}

// ============================================================================
// aim - Rune (EnergyMeterState)
// ============================================================================

std::expected<TargetPrediction, std::string> Aimer::aim(
    const energy_meter::EnergyMeterState& state, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
    uint64_t current_ns, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic_no_prediction(state, gimbal, muzzle, current_ns, bullet_speed, solver);
}

} // namespace fcs::L4
