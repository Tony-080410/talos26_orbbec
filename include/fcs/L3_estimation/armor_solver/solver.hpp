#pragma once

#include "config.hpp"
#include "trajectory_compensator.hpp"
#include "types.hpp"
#include "util.hpp"

#include "math/fast_tf/matrix.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <expected>
#include <memory>
#include <numbers>
#include <string>

namespace fcs::L3 {

/// Solver class for computing gimbal commands from tracker output
/// Handles ballistic compensation and armor selection
class Solver {
public:
    enum class State {
        TrackingArmor  = 0, // Tracking individual armor
        TrackingCenter = 1, // Tracking target center (fast rotation)
    };

    Solver() noexcept
        : compensator_(create_compensator(config_.compensator_type)) {}

    explicit Solver(const SolverConfig& config) noexcept
        : config_(config)
        , compensator_(create_compensator(config.compensator_type)) {
        update_compensator_params();
    }

    /// Update configuration
    void set_config(const SolverConfig& config) noexcept {
        config_      = config;
        compensator_ = create_compensator(config.compensator_type);
        update_compensator_params();
    }

    /// Solve gimbal command from tracker output
    /// @param target Tracker output
    /// @param current_time_ns Current timestamp
    /// @param muzzle_transform Transform from odom to muzzle_link
    /// @return GimbalCommand or error message
    [[nodiscard]] std::expected<GimbalCommand, std::string> solve(
        const TrackerOutput& target, uint64_t current_time_ns,
        const fast_tf::TransformMatrix4d& muzzle_transform) const noexcept {

        if (!target.is_tracking()) {
            return std::unexpected("Not tracking");
        }

        // Extract muzzle position and orientation from transform
        Eigen::Vector3d muzzle_pos = muzzle_transform.translation();

        // Extract current gimbal angles (roll, pitch, yaw)
        auto [roll_, pitch_, yaw_] = muzzle_transform.euler_rot().rpy();
        pitch_                     = -pitch_; // Negate pitch convention

        // Calculate target center and parameters based on target type
        Eigen::Vector3d target_position;
        Eigen::Vector3d target_velocity{0, 0, 0};
        double target_yaw   = 0;
        double target_v_yaw = 0;
        double radius0      = 0.22;
        double radius1      = 0.22;
        double z1           = 0;
        double z2           = 0;
        int armors_num      = 4;

        if (target.is_robot()) {
            const auto& state = *target.robot_state();
            target_position   = {
                state.position.x(), state.position.y(),
                (state.position.z() + state.z1) / 2.0 // Average z
            };
            target_velocity = state.velocity;
            target_yaw      = state.yaw;
            target_v_yaw    = state.v_yaw;
            radius0         = state.radius0;
            radius1         = state.radius1;
            z1              = state.z1;
            z2              = state.z1;                 // Same for robot
            armors_num      = 4;
        } else if (target.is_outpost()) {
            const auto& state = *target.outpost_state();
            target_position   = {
                state.position.x(), state.position.y(),
                (state.z[0] + state.z[1] + state.z[2]) / 3.0};
            target_yaw   = state.yaw;
            target_v_yaw = state.v_yaw;
            radius0      = OutpostTargetState::radius;
            radius1      = OutpostTargetState::radius;
            z1           = state.z[1];
            z2           = state.z[2];
            armors_num   = 3;
        } else {
            return std::unexpected("Unknown target type");
        }

        // Step 1: Initial flying time estimate
        double flying_time = compensator_->get_flying_time(target_position - muzzle_pos);

        // Calculate dt including prediction delay
        double dt = static_cast<double>(current_time_ns - target.timestamp_ns) / 1e9 + flying_time
                  + config_.prediction_delay;

        // Step 2: Predict target state
        target_position.x() += dt * target_velocity.x();
        target_position.y() += dt * target_velocity.y();
        target_position.z() += dt * target_velocity.z();
        target_yaw += dt * target_v_yaw;

        // Step 3: Get armor positions
        std::vector<Eigen::Vector4d> armor_poses;
        if (armors_num == 4) {
            armor_poses = get_robo_armor_poses(
                target_position, target_yaw, radius0, radius1, target.robot_state()->position.z(),
                z1, armors_num);
        } else {
            armor_poses = get_outpost_armor_poses(
                {target_position.x(), target_position.y()}, target_yaw, radius0,
                target.outpost_state()->z[0], target.outpost_state()->z[1],
                target.outpost_state()->z[2]);
        }

        // Step 4: Select best armor
        int idx =
            select_best_armor(armor_poses, target_position, target_yaw, target_v_yaw, armors_num);

        // Step 5: Get chosen armor position relative to muzzle
        auto chosen_armor_world = armor_poses[idx];
        Eigen::Vector3d armor_position{
            chosen_armor_world.x() - muzzle_pos.x(), chosen_armor_world.y() - muzzle_pos.y(),
            chosen_armor_world.z() - muzzle_pos.z()};
        double distance = armor_position.norm();

        // Step 6: Refine flying time
        double refined_flying_time = compensator_->get_flying_time(armor_position);
        if (std::abs(refined_flying_time - flying_time) > 0.001) {
            double dt_diff = refined_flying_time - flying_time;
            target_position.x() += dt_diff * target_velocity.x();
            target_position.y() += dt_diff * target_velocity.y();
            target_position.z() += dt_diff * target_velocity.z();
            target_yaw += dt_diff * target_v_yaw;

            // Recalculate armor position
            if (armors_num == 4) {
                armor_poses = get_robo_armor_poses(
                    target_position, target_yaw, radius0, radius1,
                    target.robot_state()->position.z(), z1, armors_num);
            } else {
                armor_poses = get_outpost_armor_poses(
                    {target_position.x(), target_position.y()}, target_yaw, radius0,
                    target.outpost_state()->z[0], target.outpost_state()->z[1],
                    target.outpost_state()->z[2]);
            }
            chosen_armor_world = armor_poses[idx];
            armor_position     = {
                chosen_armor_world.x() - muzzle_pos.x(), chosen_armor_world.y() - muzzle_pos.y(),
                chosen_armor_world.z() - muzzle_pos.z()};
            distance = armor_position.norm();
        }

        // Step 7: Calculate yaw and pitch with ballistic compensation
        double yaw, pitch;
        calc_yaw_pitch(armor_position, yaw, pitch);

        // Step 8: Initialize gimbal command
        GimbalCommand cmd;
        cmd.timestamp_ns = current_time_ns;
        cmd.distance     = distance;
        cmd.fire_advice  = is_on_target(yaw_, pitch_, yaw, pitch, distance);

        // State machine for tracking mode
        switch (state_) {
        case State::TrackingArmor:
            if (std::abs(target_v_yaw) > config_.max_tracking_v_yaw) {
                overflow_count_++;
            } else {
                overflow_count_ = 0;
            }
            if (overflow_count_ > config_.transfer_thresh) {
                transition_to(State::TrackingCenter);
            }
            break;

        case State::TrackingCenter:
            if (std::abs(target_v_yaw) < config_.max_tracking_v_yaw) {
                overflow_count_++;
            } else {
                overflow_count_ = 0;
            }
            if (overflow_count_ > config_.transfer_thresh) {
                transition_to(State::TrackingArmor);
            }
            // Track center instead of armor
            cmd.fire_advice            = true;
            Eigen::Vector3d center_rel = target_position - muzzle_pos;
            calc_yaw_pitch(center_rel, yaw, pitch);
            break;
        }

        cmd.yaw        = yaw;
        cmd.pitch      = pitch;
        cmd.yaw_diff   = normalize_rad(yaw - yaw_);
        cmd.pitch_diff = pitch - pitch_;
        cmd.predicted_position =
            Eigen::Vector3d(chosen_armor_world.x(), chosen_armor_world.y(), chosen_armor_world.z());
        cmd.trajectory = compensator_->get_trajectory(15, pitch);
        // Rotate
        for (auto& p : cmd.trajectory) {
            double x = p.first;
            double y = p.second;
            p.first  = x * cos(pitch) + y * sin(pitch);
            p.second = -x * sin(pitch) + y * cos(pitch);
        }
        return cmd;
    }

    /// Get current tracking state
    [[nodiscard]] State state() const noexcept { return state_; }

private:
    SolverConfig config_;
    mutable std::unique_ptr<TrajectoryCompensator> compensator_;
    mutable State state_{State::TrackingArmor};
    mutable int overflow_count_{0};

    void update_compensator_params() noexcept {
        if (compensator_) {
            compensator_->velocity        = config_.trajectory.bullet_speed;
            compensator_->iteration_times = config_.trajectory.iteration_times;
            compensator_->gravity         = config_.trajectory.gravity;
            compensator_->resistance      = config_.trajectory.resistance;
        }
    }

    void transition_to(State new_state) const noexcept {
        if (state_ != new_state) {
            overflow_count_ = 0;
            state_          = new_state;
        }
    }

    [[nodiscard]] int select_best_armor(
        const std::vector<Eigen::Vector4d>& armor_positions, const Eigen::Vector3d& target_center,
        double target_yaw, double target_v_yaw, int armors_num) const noexcept {

        // Angle between target center and X-axis
        double alpha = std::atan2(target_center.y(), target_center.x());
        // Angle of front armor
        double beta = target_yaw;

        // Rotation matrices
        Eigen::Matrix2d R_odom2center;
        Eigen::Matrix2d R_odom2armor;
        R_odom2center << std::cos(alpha), std::sin(alpha), -std::sin(alpha), std::cos(alpha);
        R_odom2armor << std::cos(beta), std::sin(beta), -std::sin(beta), std::cos(beta);

        Eigen::Matrix2d R_center2armor = R_odom2center.transpose() * R_odom2armor;
        double decision_angle          = std::atan2(R_center2armor(1, 0), R_center2armor(0, 0));

        // Angle threshold for armor switching
        double theta = (target_v_yaw > 0 ? config_.side_angle : -config_.side_angle)
                     * std::numbers::pi / 180.0;

        // Avoid frequent switching
        if (std::abs(target_v_yaw) < config_.min_switching_v_yaw) {
            theta = 0;
        }

        const double step = 2 * std::numbers::pi / static_cast<double>(armors_num);
        double temp_angle = decision_angle + step / 2 - theta;
        temp_angle        = std::fmod(temp_angle + 2 * std::numbers::pi, 2 * std::numbers::pi);
        int selected_id   = static_cast<int>(temp_angle / step);

        return std::clamp(selected_id, 0, armors_num - 1);
    }

    void calc_yaw_pitch(const Eigen::Vector3d& p, double& yaw, double& pitch) const noexcept {
        yaw   = std::atan2(p.y(), p.x());
        pitch = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));

        // Apply ballistic compensation
        double compensated_pitch = pitch;
        if (compensator_->compensate(p, compensated_pitch)) {
            pitch = compensated_pitch;
        }
    }

    [[nodiscard]] bool is_on_target(
        double cur_yaw, double cur_pitch, double target_yaw, double target_pitch,
        double distance) const noexcept {

        double shooting_range_yaw = std::abs(std::atan2(config_.shooting_range_w / 2.0, distance));
        double shooting_range_pitch =
            std::abs(std::atan2(config_.shooting_range_h / 2.0, distance));

        // Minimum 1 degree range
        constexpr double min_range = std::numbers::pi / 180.0;
        shooting_range_yaw         = std::max(shooting_range_yaw, min_range);
        shooting_range_pitch       = std::max(shooting_range_pitch, min_range);

        return std::abs(normalize_rad(cur_yaw - target_yaw)) < shooting_range_yaw
            && std::abs(cur_pitch - target_pitch) < shooting_range_pitch;
    }
};

} // namespace fcs::L3
