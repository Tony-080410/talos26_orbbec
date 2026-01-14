#pragma once

#include <Eigen/Core>
#include <numbers>

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"

namespace fcs::L4 {

/// Reference trajectory generator for MPC planner
///
/// Generates a reference trajectory for the gimbal to follow over the MPC
/// horizon. The trajectory is computed by predicting the target position at
/// each timestep and computing the required gimbal angles (with ballistic
/// compensation).
///
/// The trajectory spans from t = -half_horizon*dt to t = +half_horizon*dt
/// relative to the current time, so that the "current" control is at the
/// middle of the horizon (index = half_horizon).
class TrajectoryGenerator {
public:
    /// Reference trajectory matrix: [yaw, yaw_vel, pitch, pitch_vel] × horizon
    using Trajectory = Eigen::Matrix<double, 4, Eigen::Dynamic>;

    /// Constructor
    ///
    /// @param config MPC planner configuration
    explicit TrajectoryGenerator(const MpcPlannerConfig& config) noexcept
        : dt_(config.dt)
        , horizon_(config.horizon)
        , half_horizon_(config.half_horizon) {}

    /// Update configuration
    void set_config(const MpcPlannerConfig& config) noexcept {
        dt_           = config.dt;
        horizon_      = config.horizon;
        half_horizon_ = config.half_horizon;
    }

    /// Generate reference trajectory for robot target
    ///
    /// @param target Tracker output with robot state
    /// @param measurement_ts Measurement timestamp (ns)
    /// @param current_time_ns Current time (ns)
    /// @param plan_to_muzzle_delay Additional delay (seconds)
    /// @param compensator Ballistic compensator for pitch calculation
    /// @param bullet_speed Bullet velocity (m/s)
    /// @param yaw0 Output: reference yaw at current time (for offset)
    /// @return Reference trajectory [yaw-yaw0, yaw_vel, pitch, pitch_vel] × horizon
    [[nodiscard]] Trajectory generate_robot(
        const L3::RobotTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
        double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
        double& yaw0) const noexcept;

    /// Generate reference trajectory for outpost target
    [[nodiscard]] Trajectory generate_outpost(
        const L3::OutpostTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
        double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
        double& yaw0) const noexcept;

private:
    /// Normalize angle to [-pi, pi] in O(1)
    [[nodiscard]] static double normalize_rad(double angle) noexcept {
        return std::remainder(angle, 2.0 * std::numbers::pi);
    }

    /// Predict robot target position at future time dt
    ///
    /// @param target Current target state
    /// @param dt Time delta from current state (seconds)
    /// @return Predicted armor position (x, y, z)
    [[nodiscard]] Eigen::Vector3d
        predict_robot_position(const L3::RobotTargetState& target, double dt) const noexcept;

    /// Predict outpost target position at future time dt
    [[nodiscard]] Eigen::Vector3d
        predict_outpost_position(const L3::OutpostTargetState& target, double dt) const noexcept;

    /// Compute aim point (yaw, pitch) for given target position
    ///
    /// @param target_pos Target position in odom frame
    /// @param compensator Ballistic compensator
    /// @param bullet_speed Bullet velocity (m/s)
    /// @return (yaw, pitch) in radians
    [[nodiscard]] Eigen::Vector2d compute_aim_point(
        const Eigen::Vector3d& target_pos, const TrajectoryCompensator& compensator,
        double bullet_speed) const noexcept;

    double dt_{0.01};
    int horizon_{100};
    int half_horizon_{50};
};

// ============================================================================
// Implementation (header-only for simplicity)
// ============================================================================

inline TrajectoryGenerator::Trajectory TrajectoryGenerator::generate_robot(
    const L3::RobotTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
    double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
    double& yaw0) const noexcept {

    Trajectory traj(4, horizon_);

    // Time since measurement
    const double time_since_meas = static_cast<double>(current_time_ns - measurement_ts) * 1e-9;

    // Iteratively estimate flying time (target moves during bullet flight)
    // 2-3 iterations typically converge for most target speeds
    double flying_time       = 0.0;
    Eigen::Vector3d pred_pos = predict_robot_position(target, time_since_meas);
    for (int iter = 0; iter < 3; ++iter) {
        flying_time = compensator.get_flying_time(pred_pos);
        pred_pos    = predict_robot_position(target, time_since_meas + flying_time);
    }
    const double estimated_flying_time = flying_time;

    // Base time offset: time since measurement + flying time + muzzle delay
    const double base_dt = time_since_meas + estimated_flying_time + plan_to_muzzle_delay;

    // Start from -half_horizon steps before "now"
    const double start_dt = base_dt - half_horizon_ * dt_;

    // Compute initial aim point for yaw0 reference
    const auto pos_now = predict_robot_position(target, base_dt);
    const auto aim_now = compute_aim_point(pos_now, compensator, bullet_speed);
    yaw0               = aim_now.x();

    // Generate trajectory with central differencing for velocities
    Eigen::Vector2d aim_prev, aim_curr, aim_next;

    // Initialize prev and curr
    aim_prev = compute_aim_point(
        predict_robot_position(target, start_dt - dt_), compensator, bullet_speed);
    aim_curr =
        compute_aim_point(predict_robot_position(target, start_dt), compensator, bullet_speed);

    for (int i = 0; i < horizon_; ++i) {
        const double t = start_dt + (i + 1) * dt_;
        aim_next = compute_aim_point(predict_robot_position(target, t), compensator, bullet_speed);

        // Central difference for velocity (with wrap-around handling)
        const double yaw_vel   = normalize_rad(aim_next.x() - aim_prev.x()) / (2.0 * dt_);
        const double pitch_vel = (aim_next.y() - aim_prev.y()) / (2.0 * dt_);

        // Store trajectory point (yaw relative to yaw0, unwrapped for continuity)
        double yaw_rel = aim_curr.x() - yaw0;
        // Unwrap: if previous point exists, ensure continuity
        if (i > 0) {
            double prev_yaw = traj(0, i - 1);
            double delta    = normalize_rad(yaw_rel - prev_yaw);
            yaw_rel         = prev_yaw + delta;
        } else {
            yaw_rel = normalize_rad(yaw_rel);
        }
        traj(0, i) = yaw_rel;
        traj(1, i) = yaw_vel;
        traj(2, i) = aim_curr.y();
        traj(3, i) = pitch_vel;

        // Shift for next iteration
        aim_prev = aim_curr;
        aim_curr = aim_next;
    }

    return traj;
}

inline TrajectoryGenerator::Trajectory TrajectoryGenerator::generate_outpost(
    const L3::OutpostTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
    double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
    double& yaw0) const noexcept {

    Trajectory traj(4, horizon_);

    // Time since measurement
    const double time_since_meas = static_cast<double>(current_time_ns - measurement_ts) * 1e-9;

    // Iteratively estimate flying time (target moves during bullet flight)
    // 2-3 iterations typically converge for most target speeds
    double flying_time       = 0.0;
    Eigen::Vector3d pred_pos = predict_outpost_position(target, time_since_meas);
    for (int iter = 0; iter < 3; ++iter) {
        flying_time = compensator.get_flying_time(pred_pos);
        pred_pos    = predict_outpost_position(target, time_since_meas + flying_time);
    }
    const double estimated_flying_time = flying_time;

    // Base time offset: time since measurement + flying time + muzzle delay
    const double base_dt  = time_since_meas + estimated_flying_time + plan_to_muzzle_delay;
    const double start_dt = base_dt - half_horizon_ * dt_;

    const auto pos_now = predict_outpost_position(target, base_dt);
    const auto aim_now = compute_aim_point(pos_now, compensator, bullet_speed);
    yaw0               = aim_now.x();

    Eigen::Vector2d aim_prev, aim_curr, aim_next;
    aim_prev = compute_aim_point(
        predict_outpost_position(target, start_dt - dt_), compensator, bullet_speed);
    aim_curr =
        compute_aim_point(predict_outpost_position(target, start_dt), compensator, bullet_speed);

    for (int i = 0; i < horizon_; ++i) {
        const double t = start_dt + (i + 1) * dt_;
        aim_next =
            compute_aim_point(predict_outpost_position(target, t), compensator, bullet_speed);

        const double yaw_vel   = normalize_rad(aim_next.x() - aim_prev.x()) / (2.0 * dt_);
        const double pitch_vel = (aim_next.y() - aim_prev.y()) / (2.0 * dt_);

        // Unwrap yaw for continuity
        double yaw_rel = aim_curr.x() - yaw0;
        if (i > 0) {
            double prev_yaw = traj(0, i - 1);
            double delta    = normalize_rad(yaw_rel - prev_yaw);
            yaw_rel         = prev_yaw + delta;
        } else {
            yaw_rel = normalize_rad(yaw_rel);
        }
        traj(0, i) = yaw_rel;
        traj(1, i) = yaw_vel;
        traj(2, i) = aim_curr.y();
        traj(3, i) = pitch_vel;

        aim_prev = aim_curr;
        aim_curr = aim_next;
    }

    return traj;
}

inline Eigen::Vector3d TrajectoryGenerator::predict_robot_position(
    const L3::RobotTargetState& target, double dt) const noexcept {
    // Robot model: rotating armors around center
    // Predict center position and yaw, then find the closest armor

    const Eigen::Vector3d center   = target.position;
    const Eigen::Vector3d velocity = target.velocity;
    const double yaw               = target.yaw;
    const double v_yaw             = target.v_yaw;

    // Height offset h = z1 - z0 (constant geometry parameter)
    const double h = target.z1 - center.z();

    // Predicted center position
    const Eigen::Vector3d center_pred = center + velocity * dt;

    // Predicted yaw angle
    const double yaw_pred = yaw + v_yaw * dt;

    // Compute all armor positions and select closest one (matching sp_vision_25 behavior)
    const int n                   = target.armors_num;
    double min_dist               = 1e10;
    Eigen::Vector3d closest_armor = center_pred;

    for (int i = 0; i < n; ++i) {
        const double angle_offset = i * (2.0 * std::numbers::pi / n);
        const double armor_yaw    = yaw_pred + angle_offset;

        // Select radius based on armor index (0,2 use radius0; 1,3 use radius1)
        const double r = (i % 2 == 0) ? target.radius0 : target.radius1;

        // Armor position in odom frame (front-facing convention: -r*cos, -r*sin)
        const double armor_x = center_pred.x() - r * std::cos(armor_yaw);
        const double armor_y = center_pred.y() - r * std::sin(armor_yaw);
        // Height: armors 0,2 at predicted z0, armors 1,3 at predicted z0 + h
        const double armor_z = center_pred.z() + ((i % 2 == 0) ? 0.0 : h);

        const Eigen::Vector3d armor_pos{armor_x, armor_y, armor_z};

        // Use horizontal distance for selection (matching sp_vision_25)
        const double dist = armor_pos.head<2>().norm();
        if (dist < min_dist) {
            min_dist      = dist;
            closest_armor = armor_pos;
        }
    }

    return closest_armor;
}

inline Eigen::Vector3d TrajectoryGenerator::predict_outpost_position(
    const L3::OutpostTargetState& target, double dt) const noexcept {
    // Outpost model: 3 armors rotating around center
    const Eigen::Vector2d center_2d = target.position;
    const Eigen::Vector3d velocity  = target.velocity;
    const double yaw                = target.yaw;
    const double v_yaw              = target.v_yaw;
    constexpr double r              = L3::OutpostTargetState::radius;

    const double center_x_pred = center_2d.x() + velocity.x() * dt;
    const double center_y_pred = center_2d.y() + velocity.y() * dt;
    const double yaw_pred      = yaw + v_yaw * dt;

    // Find closest armor among 3 armors
    double min_dist = 1e10;
    Eigen::Vector3d closest_armor{center_x_pred, center_y_pred, target.z[0]};

    for (int i = 0; i < 3; ++i) {
        const double angle_offset = i * (2.0 * std::numbers::pi / 3.0);
        const double armor_yaw    = yaw_pred + angle_offset;

        const double armor_x = center_x_pred - r * std::cos(armor_yaw);
        const double armor_y = center_y_pred - r * std::sin(armor_yaw);
        const Eigen::Vector3d armor_pos{armor_x, armor_y, target.z[i]};

        const double dist = armor_pos.head<2>().norm();
        if (dist < min_dist) {
            min_dist      = dist;
            closest_armor = armor_pos;
        }
    }

    return closest_armor;
}

inline Eigen::Vector2d TrajectoryGenerator::compute_aim_point(
    const Eigen::Vector3d& target_pos, const TrajectoryCompensator& compensator,
    double bullet_speed) const noexcept {
    // Compute yaw (azimuth angle)
    const double yaw = std::atan2(target_pos.y(), target_pos.x());

    // Compute pitch with ballistic compensation (const-safe with bullet_speed param)
    double pitch             = 0.0;
    double yaw_out           = yaw;
    [[maybe_unused]] bool ok = compensator.compensate(target_pos, yaw_out, pitch, bullet_speed);

    return {yaw, pitch};
}

} // namespace fcs::L4
