#pragma once

#include "config.hpp"
#include "fcs/L3_estimation/tracker/util.hpp"
#include "trajectory_compensator.hpp"
#include "types.hpp"

#include "math/fast_tf/matrix.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <expected>
#include <memory>
#include <numbers>
#include <string>

namespace fcs::L4 {

/// GimbalPlanner: L3 TrackerOutput -> yaw/pitch/fire_advice
///
/// Problem:
///   Convert an estimated target state (L3) into a gimbal aim command (L4) that accounts for
///   delays and bullet flight, and provide a conservative "fire now?" suggestion.
///
/// Intuition:
///   1) Time-align / predict: extrapolate the target to the bullet-arrival time.
///   2) Choose which armor to aim at (rotation-aware).
///   3) Apply ballistic compensation to obtain yaw/pitch, then compute fire_advice.
///
/// Analogy:
///   Like throwing a ball at a moving target: lead by (time since you saw it) + (your reaction /
///   actuation delay) + (ball travel time).
///
/// Precise (implementation details that matter for tuning):
///   - dt = (now - measurement_ts) + flying_time + plan_to_muzzle_delay
///   - `now` is the passed `current_time_ns` (real-time), consistent with using the latest TF for
///     the muzzle pose.
///   - `flying_time` is first estimated cheaply using LOS-to-center and `bullet_speed`, then
///     refined using the configured trajectory compensator.
///   - Naming pitfalls:
///       * `max_tracking_v_yaw` is a mode-switch threshold (armor-tracking <-> center-tracking),
///         not a "max allowed yaw rate".
///       * `shooting_range_w/h` define the physical fire window size, not a distance/range.
class GimbalPlanner {
public:
    enum class State {
        TrackingArmor  = 0, // Tracking individual armor
        TrackingCenter = 1, // Tracking target center (fast rotation)
    };

    GimbalPlanner() noexcept
        : compensator_(create_compensator(config_.compensator_type)) {}

    explicit GimbalPlanner(const GimbalPlannerConfig& config) noexcept
        : config_(config)
        , compensator_(create_compensator(config.compensator_type)) {
        update_compensator_params();
    }

    /// Update configuration
    void set_config(const GimbalPlannerConfig& config) noexcept {
        config_      = config;
        compensator_ = create_compensator(config.compensator_type);
        update_compensator_params();
    }

    /// Plan gimbal command from tracker output
    /// @param target Tracker output from L3
    /// @param current_time_ns Current timestamp
    /// @param muzzle_transform Transform from odom to muzzle_link
    /// @return GimbalPlan or error message
    [[nodiscard]] std::expected<GimbalPlan, std::string> plan(
        const fcs::L3::TrackerOutput& target, uint64_t current_time_ns,
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
            radius0      = fcs::L3::OutpostTargetState::radius;
            radius1      = fcs::L3::OutpostTargetState::radius;
            z1           = state.z[1];
            z2           = state.z[2];
            armors_num   = 3;
        } else {
            return std::unexpected("Unknown target type");
        }

        // Step 1: Initial flying time estimate (cheap LOS estimate to avoid heavy simulation here)
        // The refined flying time (after armor selection) still uses the configured compensator.
        const Eigen::Vector3d rel_center = target_position - muzzle_pos;
        const double rel_center_xy       = std::hypot(rel_center.x(), rel_center.y());
        const double rel_center_pitch    = std::atan2(rel_center.z(), rel_center_xy);
        const double v0                  = std::max(1e-3, config_.trajectory.bullet_speed);
        const double flying_time         = rel_center_xy / (v0 * std::cos(rel_center_pitch));

        // Calculate dt including observation delay + actuation delay + flight time.
        //
        // Problem:
        //   The target state is estimated at `target.timestamp_ns`. We must predict forward to the
        //   time the bullet reaches the target.
        // Intuition:
        //   dt = (time since measurement) + (time until bullet leaves muzzle) + (time in flight).
        // Analogy:
        //   "What you saw" + "your reaction time" + "ball travel time".
        // Precise:
        //   dt = (current_time_ns - target.timestamp_ns)/1e9 + flying_time + plan_to_muzzle_delay.
        double dt = static_cast<double>(current_time_ns - target.timestamp_ns) / 1e9 + flying_time
                  + config_.plan_to_muzzle_delay;
        uint64_t predicted_future =
            current_time_ns + uint64_t(config_.plan_to_muzzle_delay * 1e9 + flying_time * 1e9);

        // Step 2: Predict target state
        target_position.x() += dt * target_velocity.x();
        target_position.y() += dt * target_velocity.y();
        target_position.z() += dt * target_velocity.z();
        target_yaw += dt * target_v_yaw;

        // Step 3: Get armor positions
        std::vector<Eigen::Vector4d> armor_poses;
        if (armors_num == 4) {
            armor_poses = fcs::L3::get_robo_armor_poses(
                target_position, target_yaw, radius0, radius1, target_position.z(), z1, armors_num);
        } else {
            armor_poses = fcs::L3::get_outpost_armor_poses(
                {target_position.x(), target_position.y()}, target_yaw, radius0,
                target.outpost_state()->z[0], target.outpost_state()->z[1],
                target.outpost_state()->z[2]);
        }

        // Step 4: Select best armor
        int idx =
            select_best_armor(armor_poses, target_position, target_yaw, target_v_yaw, armors_num);

        // Step 5: Get chosen armor position relative to muzzle
        auto chosen_armor_world           = armor_poses[idx];
        Eigen::Vector4d chosen_armor_pose = chosen_armor_world;
        chosen_armor_pose.x() -= muzzle_pos.x();
        chosen_armor_pose.y() -= muzzle_pos.y();
        chosen_armor_pose.z() -= muzzle_pos.z();

        Eigen::Vector3d armor_position{
            chosen_armor_pose.x(), chosen_armor_pose.y(), chosen_armor_pose.z()};
        Eigen::Vector3d predicted_position{
            chosen_armor_world.x(), chosen_armor_world.y(), chosen_armor_world.z()};
        double distance = chosen_armor_pose.norm();

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
                armor_poses = fcs::L3::get_robo_armor_poses(
                    target_position, target_yaw, radius0, radius1, target_position.z(), z1,
                    armors_num);
            } else {
                armor_poses = fcs::L3::get_outpost_armor_poses(
                    {target_position.x(), target_position.y()}, target_yaw, radius0,
                    target.outpost_state()->z[0], target.outpost_state()->z[1],
                    target.outpost_state()->z[2]);
            }
            chosen_armor_world = armor_poses[idx];
            chosen_armor_pose  = chosen_armor_world;
            chosen_armor_pose.x() -= muzzle_pos.x();
            chosen_armor_pose.y() -= muzzle_pos.y();
            chosen_armor_pose.z() -= muzzle_pos.z();

            armor_position = {chosen_armor_pose.x(), chosen_armor_pose.y(), chosen_armor_pose.z()};
            predicted_position = {
                chosen_armor_world.x(), chosen_armor_world.y(), chosen_armor_world.z()};
            distance = chosen_armor_pose.norm();
        }

        // Step 7: Calculate yaw and pitch with ballistic compensation
        double yaw, pitch;
        calc_yaw_pitch(armor_position, yaw, pitch);

        // Step 8: Initialize gimbal plan
        GimbalPlan plan;
        plan.timestamp_ns = current_time_ns;
        plan.distance     = distance;
        plan.fire_advice  = is_on_target(yaw_, pitch_, yaw, pitch, distance);

        // State machine for tracking mode.
        //
        // Problem:
        //   At high yaw-rate, individual armors swap/occlude quickly and "tracking an armor" can
        //   jitter. A center-tracking mode is more stable.
        // Intuition:
        //   Switch to TrackingCenter if |v_yaw| stays above the threshold; switch back if it stays
        //   below the threshold.
        // Analogy:
        //   Track the spinning wheel hub instead of a wheel spoke.
        // Precise:
        //   The condition is debounced with a counter using `overflow_count_ > transfer_thresh`
        //   (strict '>'), so it requires `transfer_thresh + 1` consecutive frames.
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
            plan.fire_advice = true;
            calc_yaw_pitch(target_position - muzzle_pos, yaw, pitch);
            predicted_position = target_position;
            break;
        }

        const double cmd_yaw = fcs::L3::normalize_rad(yaw);
        plan.yaw             = cmd_yaw;
        plan.pitch           = pitch;
        plan.yaw_diff        = cmd_yaw - yaw_;
        plan.pitch_diff      = pitch - pitch_;
        // SPDLOG_INFO("pitch_diff: {}", plan.pitch_diff);
        plan.predicted_future   = predicted_future;
        plan.predicted_position = {armor_position.x(), armor_position.y(), armor_position.z()};
        plan.track_center       = state_ == State::TrackingCenter;
        plan.trajectory         = compensator_->get_trajectory(15, pitch_);
        // Rotate
        for (auto& p : plan.trajectory) {
            double x = p.first;
            double y = p.second;
            p.first  = x * cos(pitch_) + y * sin(pitch_);
            p.second = -x * sin(pitch_) + y * cos(pitch_);
        }
        return plan;
    }

    /// Get current tracking state
    [[nodiscard]] State state() const noexcept { return state_; }

private:
    GimbalPlannerConfig config_;
    mutable std::unique_ptr<TrajectoryCompensator> compensator_;
    mutable State state_{State::TrackingArmor};
    mutable int overflow_count_{0};

    void update_compensator_params() noexcept {
        if (compensator_) {
            compensator_->velocity        = config_.trajectory.bullet_speed;
            compensator_->iteration_times = config_.trajectory.iteration_times;
            compensator_->gravity         = config_.trajectory.gravity;
            compensator_->resistance      = config_.trajectory.resistance;
            compensator_->cd              = config_.trajectory.cd;
            compensator_->rho             = config_.trajectory.rho;
            compensator_->diameter        = config_.trajectory.diameter;
            compensator_->mass            = config_.trajectory.mass;
            compensator_->rk4_dt          = config_.trajectory.rk4_dt;
            compensator_->rk4_min_steps   = config_.trajectory.rk4_min_steps;
            compensator_->rk4_max_steps   = config_.trajectory.rk4_max_steps;
            compensator_->on_params_updated();
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

        // Angle bias for armor selection.
        //
        // Problem:
        //   On a rotating target, the "best" armor right now may not be the best armor at
        //   bullet-arrival time.
        // Intuition:
        //   Bias the decision boundary in the direction of rotation so we pick the armor that is
        //   more likely to remain visible/centered when the bullet arrives.
        // Analogy:
        //   Step ahead in the direction a carousel is turning.
        // Precise:
        //   theta = sign(v_yaw) * side_angle (deg→rad), and it is disabled when
        //   |v_yaw| < min_switching_v_yaw to avoid jitter near boundaries.
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
        yaw                   = std::atan2(p.y(), p.x());
        const auto horizontal = std::hypot(p.x(), p.y()); // More accurate and faster
        pitch                 = std::atan2(p.z(), horizontal);

        // Apply ballistic compensation
        double compensated_yaw   = yaw;
        double compensated_pitch = pitch;
        if (compensator_->compensate(p, compensated_yaw, compensated_pitch)) {
            yaw   = compensated_yaw;
            pitch = compensated_pitch;
        }
    }

    [[nodiscard]] bool is_on_target(
        double cur_yaw, double cur_pitch, double target_yaw, double target_pitch,
        double distance) const noexcept {

        // Convert physical window (meters) to angular tolerance (radians).
        double shooting_range_yaw = std::abs(std::atan2(config_.shooting_range_w / 2.0, distance));
        double shooting_range_pitch =
            std::abs(std::atan2(config_.shooting_range_h / 2.0, distance));

        // Minimum 1 degree range
        constexpr double min_range = (std::numbers::pi / 180.0);
        shooting_range_yaw         = std::max(shooting_range_yaw, min_range);
        shooting_range_pitch       = std::max(shooting_range_pitch, min_range);

        // NOTE: This uses a raw subtraction; near +/-pi wrap-around, the apparent difference can be
        // large even if angles are close. If this becomes an issue, consider angle-normalized diff.
        return std::abs(cur_yaw - target_yaw) < shooting_range_yaw
            && std::abs(cur_pitch - target_pitch) < shooting_range_pitch;
    }
};

} // namespace fcs::L4
