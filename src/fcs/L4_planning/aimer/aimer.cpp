#include "L4_planning/aimer/aimer.hpp"

#include <cmath>
#include <limits>
#include <numbers>

namespace fcs::L4 {
namespace {

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

} // namespace

// ============================================================================
// select_best_armor_index - Angle-based armor selection
// ============================================================================

int Aimer::select_best_armor_index(
    const Eigen::Vector3d& target_center, double target_yaw, double target_v_yaw,
    int armors_num) const noexcept {

    // Angle between the robot's center and the X-axis (odom → center)
    const double alpha = std::atan2(target_center.y(), target_center.x());

    // Angle between the front of observed armor and the X-axis (odom → armor front)
    const double beta = target_yaw;

    // clang-format off
    Eigen::Matrix2d R_odom2center;
    Eigen::Matrix2d R_odom2armor;
    R_odom2center << std::cos(alpha), std::sin(alpha),
                     -std::sin(alpha), std::cos(alpha);
    R_odom2armor  << std::cos(beta), std::sin(beta),
                     -std::sin(beta), std::cos(beta);
    // clang-format on

    // Compute relative rotation: center → armor
    Eigen::Matrix2d R_center2armor = R_odom2center.transpose() * R_odom2armor;

    // Use full angle to avoid aliasing when |alpha - beta| > 90° (outpost has 3 armors)
    const double decision_angle = std::atan2(R_center2armor(1, 0), R_center2armor(0, 0));

    // Angle threshold for armor switching (hysteresis)
    double theta = (target_v_yaw > 0 ? config_.side_angle : -config_.side_angle) * M_PI / 180.0;

    // Avoid frequent switching between two armors when rotation is slow
    if (std::abs(target_v_yaw) < config_.min_switching_v_yaw) {
        theta = 0.0;
    }

    // Quantize to armor slot
    const double step = 2.0 * M_PI / static_cast<double>(armors_num);
    double temp_angle = decision_angle + step / 2.0 - theta;
    temp_angle        = std::fmod(temp_angle + 2.0 * M_PI, 2.0 * M_PI); // Wrap to [0, 2π)
    int selected_id   = static_cast<int>(temp_angle / step);

    return selected_id;
}

// ============================================================================
// predict_position - Robot
// ============================================================================

Eigen::Vector3d
    Aimer::predict_position(const L3::RobotTargetState& target, double dt) const noexcept {
    const Eigen::Vector3d center   = target.position;
    const Eigen::Vector3d velocity = target.velocity;
    const double yaw               = target.yaw;
    const double v_yaw             = target.v_yaw;
    const double h                 = target.z1 - center.z();

    // Predict center and yaw
    const Eigen::Vector3d center_pred = center + velocity * dt;
    const double yaw_pred             = yaw + v_yaw * dt;

    // Select best armor index using angle-based algorithm
    const int armor_idx = select_best_armor_index(center_pred, yaw_pred, v_yaw, target.armors_num);

    // Compute selected armor position
    const double armor_yaw = yaw_pred + armor_idx * (2.0 * std::numbers::pi / target.armors_num);
    const double r         = (armor_idx % 2 == 0) ? target.radius0 : target.radius1;

    const double armor_x = center_pred.x() - r * std::cos(armor_yaw);
    const double armor_y = center_pred.y() - r * std::sin(armor_yaw);
    const double armor_z = center_pred.z() + ((armor_idx % 2 == 0) ? 0.0 : h);

    return Eigen::Vector3d{armor_x, armor_y, armor_z};
}

// ============================================================================
// predict_position - Outpost
// ============================================================================

Eigen::Vector3d
    Aimer::predict_position(const L3::OutpostTargetState& target, double dt) const noexcept {
    const Eigen::Vector2d center_2d = target.position;
    const Eigen::Vector3d velocity  = target.velocity;
    const double yaw                = target.yaw;
    const double v_yaw              = target.v_yaw;
    constexpr double r              = L3::OutpostTargetState::radius;

    // Predict center and yaw
    const Eigen::Vector2d center_pred = center_2d + velocity.head<2>() * dt;
    const double center_x_pred        = center_pred.x();
    const double center_y_pred        = center_pred.y();
    const double yaw_pred             = yaw + v_yaw * dt;

    // Select best armor index using angle-based algorithm
    const int armor_idx = select_best_armor_index(
        Eigen::Vector3d{center_x_pred, center_y_pred, target.z[0]}, yaw_pred, v_yaw, 3);

    // Compute selected armor position
    const double armor_yaw = yaw_pred + armor_idx * (2.0 * std::numbers::pi / 3.0);
    const double armor_x   = center_x_pred - r * std::cos(armor_yaw);
    const double armor_y   = center_y_pred - r * std::sin(armor_yaw);
    const double armor_z   = target.z[armor_idx];

    return Eigen::Vector3d{armor_x, armor_y, armor_z};
}

// ============================================================================
// aim_generic - Template method for targets with prediction
// ============================================================================

template <class TargetType>
std::expected<TargetPrediction, std::string> Aimer::aim_generic(
    const TargetType& target, const GimbalTransform& gimbal, const MuzzleTransform& muzzle,
    uint64_t measurement_ns, uint64_t current_ns, double prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {

    TargetPrediction output;
    output.target_type     = get_target_type<TargetType>();
    output.timestamp_ns    = current_ns;
    output.target_velocity = target.velocity;
    output.target_v_yaw    = target.v_yaw;

    const double age =
        static_cast<double>(static_cast<int64_t>(current_ns) - static_cast<int64_t>(measurement_ns))
        * 1e-9;
    const double total_delay = age + prediction_delay;

    const Eigen::Vector3d t            = muzzle.translation();
    const double total_prediction_time = refine_flying_time(
        solver, [&](double dt) -> Eigen::Vector3d { return predict_position(target, dt) - t; },
        total_delay, bullet_speed);

    output.predicted_position  = predict_position(target, total_prediction_time);
    output.predicted_future_ns = offset_timestamp_ns(measurement_ns, total_prediction_time);
    output.flying_time         = total_prediction_time - total_delay;

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
    const L3::RobotTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double bullet_speed, const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(
        target, gimbal, muzzle, measurement_ns, current_ns, config_.delay, bullet_speed, solver);
}

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::RobotTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double extra_prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(
        target, gimbal, muzzle, measurement_ns, current_ns, config_.delay + extra_prediction_delay,
        bullet_speed, solver);
}

// ============================================================================
// aim - Outpost
// ============================================================================

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::OutpostTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double bullet_speed, const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(
        target, gimbal, muzzle, measurement_ns, current_ns, config_.delay, bullet_speed, solver);
}

std::expected<TargetPrediction, std::string> Aimer::aim(
    const L3::OutpostTargetState& target, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
    double extra_prediction_delay, double bullet_speed,
    const core::trajectory::solver::TrajectorySolver& solver) const noexcept {
    return aim_generic(
        target, gimbal, muzzle, measurement_ns, current_ns, config_.delay + extra_prediction_delay,
        bullet_speed, solver);
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
