#pragma once

#include <Eigen/Core>
#include <numbers>
#include <utility>

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/angle_utils.hpp"
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
    using TrajectoryMat = Eigen::Matrix<double, 4, Eigen::Dynamic>;
    using Trajectory    = std::pair<std::pair<double, Eigen::Vector3d>, TrajectoryMat>;

    /// Constructor
    ///
    /// @param config MPC planner configuration
    explicit TrajectoryGenerator(const MpcPlannerConfig& config) noexcept
        : ponder_dt_(config.ponder_dt)
        , ponder_total(config.ponder_ahead + config.ponder_back)
        , ponder_back(config.ponder_back) {}

    /// Update configuration
    void set_config(const MpcPlannerConfig& config) noexcept {
        ponder_dt_   = config.ponder_dt;
        ponder_total = config.ponder_ahead + config.ponder_back;
        ponder_back  = config.ponder_back;
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

    /// Generate reference trajectory for robot target with custom muzzle origin
    [[nodiscard]] Trajectory generate_robot(
        const L3::RobotTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
        double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
        const Eigen::Vector3d& origin_odom, double& yaw0) const noexcept;

    /// Generate reference trajectory for outpost target
    ///
    /// @param target Tracker output with outpost state
    /// @param measurement_ts Measurement timestamp (ns)
    /// @param current_time_ns Current time (ns)
    /// @param plan_to_muzzle_delay Additional delay (seconds)
    /// @param compensator Ballistic compensator for pitch calculation
    /// @param bullet_speed Bullet velocity (m/s)
    /// @param yaw0 Output: reference yaw at current time (for offset)
    /// @return Reference trajectory [yaw-yaw0, yaw_vel, pitch, pitch_vel] × horizon
    [[nodiscard]] Trajectory generate_outpost(
        const L3::OutpostTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
        double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
        double& yaw0) const noexcept;

    /// Generate reference trajectory for outpost target with custom muzzle origin
    [[nodiscard]] Trajectory generate_outpost(
        const L3::OutpostTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
        double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
        const Eigen::Vector3d& origin_odom, double& yaw0) const noexcept;

    /// Predict robot target position at future time dt
    ///
    /// @param target Current target state
    /// @param dt Time delta from current state (seconds)
    /// @return Predicted armor position (x, y, z)
    [[nodiscard]] static Eigen::Vector3d
        predict_robot_position(const L3::RobotTargetState& target, double dt) noexcept;

    /// Predict outpost target position at future time dt
    [[nodiscard]] static Eigen::Vector3d
        predict_outpost_position(const L3::OutpostTargetState& target, double dt) noexcept;

    /// Compute aim point (yaw, pitch) for given target position
    ///
    /// @param target_pos Target position in odom frame
    /// @param compensator Ballistic compensator
    /// @param bullet_speed Bullet velocity (m/s)
    /// @return (yaw, pitch) in radians
    [[nodiscard]] static Eigen::Vector2d compute_aim_point(
        const Eigen::Vector3d& target_pos, const TrajectoryCompensator& compensator,
        double bullet_speed) noexcept;

private:
    /// Large initial distance for closest armor selection
    static constexpr double kInitialMinDist = 1e10;

    /// Generate trajectory from prediction function (zero-cost abstraction)
    template <class PredictFn>
    [[nodiscard]] Trajectory generate_trajectory(
        PredictFn&& predict, uint64_t measurement_ts, uint64_t current_time_ns,
        double action_overhead, const TrajectoryCompensator& compensator, double bullet_speed,
        const Eigen::Vector3d& origin_odom, double& yaw0) const noexcept;

    double ponder_dt_{0.01};
    int ponder_total{100};
    int ponder_back{50};
};

// ============================================================================
// Implementation (header-only for simplicity)
// ============================================================================

template <class PredictFn>
inline TrajectoryGenerator::Trajectory TrajectoryGenerator::generate_trajectory(
    PredictFn&& predict, uint64_t measurement_ts, uint64_t current_time_ns, double action_overhead,
    const TrajectoryCompensator& compensator, double bullet_speed,
    const Eigen::Vector3d& origin_odom, double& yaw0) const noexcept {
    TrajectoryMat traj(4, ponder_total);

    static constexpr int refine_max_iter     = 10;
    static constexpr double refine_tolerance = 1e-3;

    action_overhead += static_cast<double>(current_time_ns - measurement_ts) * 1e-9;

    const auto predict_muzzle = [&](double dt) -> Eigen::Vector3d {
        return predict(dt) - origin_odom;
    };

    double flying_time = 0.0;
    for (auto i = 0; i < refine_max_iter; ++i) {
        auto predicted_target = predict_muzzle(action_overhead + flying_time);
        auto t_curr           = compensator.get_flying_time(predicted_target);

        if (std::abs(t_curr - flying_time) < refine_tolerance) {
            break;
        }
        flying_time = t_curr;
    }

    const double ideal_time = action_overhead + flying_time;

    const double start_dt = ideal_time - ponder_back * ponder_dt_;

    const auto pos_now = predict_muzzle(ideal_time);
    const auto aim_now = compute_aim_point(pos_now, compensator, bullet_speed);
    yaw0               = aim_now.x();

    Eigen::Vector2d aim_prev, aim_curr, aim_next;
    aim_prev =
        compute_aim_point(predict(start_dt - ponder_dt_) - origin_odom, compensator, bullet_speed);
    aim_curr = compute_aim_point(predict(start_dt) - origin_odom, compensator, bullet_speed);

    for (int i = 0; i < ponder_total; ++i) {
        const auto t = start_dt + (i + 1) * ponder_dt_;
        aim_next     = compute_aim_point(predict_muzzle(t), compensator, bullet_speed);

        const double yaw_vel   = normalize_rad(aim_next.x() - aim_prev.x()) / (2.0 * ponder_dt_);
        const double pitch_vel = (aim_next.y() - aim_prev.y()) / (2.0 * ponder_dt_);

        double yaw_rel = aim_curr.x() - yaw0;
        if (i > 0) {
            const auto prev_yaw = traj(0, i - 1);
            const auto delta    = normalize_rad(yaw_rel - prev_yaw);
            yaw_rel             = prev_yaw + delta;
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

    return std::make_pair(std::make_pair(action_overhead, predict(action_overhead)), traj);
}

inline TrajectoryGenerator::Trajectory TrajectoryGenerator::generate_robot(
    const L3::RobotTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
    double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
    double& yaw0) const noexcept {
    return generate_robot(
        target, measurement_ts, current_time_ns, plan_to_muzzle_delay, compensator, bullet_speed,
        Eigen::Vector3d::Zero(), yaw0);
}

inline TrajectoryGenerator::Trajectory TrajectoryGenerator::generate_robot(
    const L3::RobotTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
    double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
    const Eigen::Vector3d& origin_odom, double& yaw0) const noexcept {
    return generate_trajectory(
        [&](double dt) noexcept { return predict_robot_position(target, dt); }, measurement_ts,
        current_time_ns, plan_to_muzzle_delay, compensator, bullet_speed, origin_odom, yaw0);
}

inline TrajectoryGenerator::Trajectory TrajectoryGenerator::generate_outpost(
    const L3::OutpostTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
    double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
    double& yaw0) const noexcept {
    return generate_outpost(
        target, measurement_ts, current_time_ns, plan_to_muzzle_delay, compensator, bullet_speed,
        Eigen::Vector3d::Zero(), yaw0);
}

inline TrajectoryGenerator::Trajectory TrajectoryGenerator::generate_outpost(
    const L3::OutpostTargetState& target, uint64_t measurement_ts, uint64_t current_time_ns,
    double plan_to_muzzle_delay, const TrajectoryCompensator& compensator, double bullet_speed,
    const Eigen::Vector3d& origin_odom, double& yaw0) const noexcept {
    return generate_trajectory(
        [&](double dt) noexcept { return predict_outpost_position(target, dt); }, measurement_ts,
        current_time_ns, plan_to_muzzle_delay, compensator, bullet_speed, origin_odom, yaw0);
}

inline Eigen::Vector3d TrajectoryGenerator::predict_robot_position(
    const L3::RobotTargetState& target, double dt) noexcept {
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
    double min_dist               = kInitialMinDist;
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
    const L3::OutpostTargetState& target, double dt) noexcept {
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
    double min_dist = kInitialMinDist;
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
    double bullet_speed) noexcept {
    // Compute yaw (azimuth angle)
    const double yaw = std::atan2(target_pos.y(), target_pos.x());

    // Compute pitch with ballistic compensation (const-safe with bullet_speed param)
    double pitch             = 0.0;
    double yaw_out           = yaw;
    [[maybe_unused]] bool ok = compensator.compensate(target_pos, yaw_out, pitch, bullet_speed);

    return {yaw, pitch};
}

} // namespace fcs::L4
