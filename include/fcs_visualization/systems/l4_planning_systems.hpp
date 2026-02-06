#pragma once

#include "base.hpp"
#include "fcs_visualization/foxglove_types.hpp"
#include "fcs_visualization/scene_builder.hpp"
#include "fcs_visualization/system_helpers.hpp"

#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/types.hpp"
#include "fcs/L5_weapon/fire_control.hpp"
#include "fcs/core/channel_topics.hpp"
#include "fcs/core/types.hpp"
#include "math/fast_tf/frame.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace fcs::foxglove::systems {

/// @brief Register L4 planning layer systems (gimbal, MPC)
///
/// This includes:
/// - foxglove_l4_gimbal_cmd_pub: Publishes gimbal command JSON
/// - foxglove_l4_gimbal_scene: Publishes gimbal scene (selection, prediction, trajectory)
/// - foxglove_l4_mpc_trajectory_pub: Publishes MPC trajectory JSON
/// - foxglove_l4_mpc_prediction_scene: Publishes full MPC horizon visualization
template <typename App>
void register_l4_planning_systems(App& app) {

    // =========================================================================
    // GIMBAL COMMAND PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l4_gimbal_cmd_pub",
        [](talos::spmc<::fcs::L5::WeaponCommand, WeaponCommandChannelTopic> cmd_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, cmd_in)) {
                return;
            }

            auto cmd = cmd_in.read();
            if (!cmd) {
                return;
            }

            nlohmann::json cmd_json;
            cmd_json["timestamp_ns"] = cmd->timestamp_ns;
            cmd_json["yaw"]          = cmd->yaw;
            cmd_json["pitch"]        = cmd->pitch;
            cmd_json["ref_yaw"]      = cmd->ref_yaw;
            cmd_json["ref_pitch"]    = cmd->ref_pitch;
            cmd_json["v_yaw"]        = cmd->yaw_vel;
            cmd_json["v_pitch"]      = cmd->pitch_vel;
            cmd_json["a_yaw"]        = cmd->yaw_accel;
            cmd_json["a_pitch"]      = cmd->pitch_accel;
            cmd_json["distance"]     = cmd->distance;
            cmd_json["fire_advice"]  = cmd->fire;

            detail::publish_json_message<GimbalCmdMessage>(*server, cmd_json);
        });

    // =========================================================================
    // GIMBAL SCENE VISUALIZATION (L4) - Selection + Prediction + Trajectory
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l4_gimbal_scene",
        [](talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in))
                return;
            auto plan = plan_in.read();
            if (!plan || plan->distance <= 0)
                return;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Selection marker (fire decision aware)
            {
                ::foxglove::schemas::SceneEntity entity =
                    viz::EntityBuilder::create<fast_tf::gimbal>("l4", "selection")
                        .timestamp(plan->timestamp_ns)
                        .build();

                const auto position =
                    detail::spherical_to_cartesian(plan->distance, plan->yaw, plan->pitch);

                const auto color =
                    plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L4::GIMBAL_AIM_HOLD;

                ::foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = viz::make_pose(position);
                sphere.size = {
                    tac::L4::SELECTION_SIZE, tac::L4::SELECTION_SIZE, tac::L4::SELECTION_SIZE};
                sphere.color = color;
                entity.spheres.push_back(sphere);

                entities.push_back(entity);
            }

            // Predicted position marker (at bullet impact time for verification)
            if (plan->predicted_position) {
                // Use predicted_future timestamp to show where target will be when bullet hits
                // This allows verification by comparing with camera images at impact time
                ::foxglove::schemas::SceneEntity entity =
                    viz::EntityBuilder::create<fast_tf::odom>("l3", "predicted")
                        .timestamp(plan->predicted_future)
                        .build();

                const auto position = viz::make_vector3(
                    plan->predicted_position->x(), plan->predicted_position->y(),
                    plan->predicted_position->z());

                ::foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = viz::make_pose(position);
                sphere.size = {
                    tac::L4::PREDICTION_SIZE, tac::L4::PREDICTION_SIZE, tac::L4::PREDICTION_SIZE};
                sphere.color = tac::L3::PREDICTION_AMBER;
                entity.spheres.push_back(sphere);

                entities.push_back(entity);
            }

            // Bullet trajectory
            if (plan->viz_debug && !plan->viz_debug->trajectory.empty()) {
                std::vector<Eigen::Vector3d> traj_points;
                for (const auto& [x, z] : plan->viz_debug->trajectory) {
                    traj_points.emplace_back(x, 0, z);
                }
                auto traj_entities = viz::patterns::bullet_trajectory(
                    traj_points, plan->fire_advice, plan->timestamp_ns);
                entities.insert(
                    entities.end(), std::make_move_iterator(traj_entities.begin()),
                    std::make_move_iterator(traj_entities.end()));
            }

            viz::publish_scene_if_nonempty<GimbalSceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // MPC TRAJECTORY JSON PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l4_mpc_trajectory_pub",
        [](talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || !plan->mpc_debug || plan->mpc_debug->ref_yaw.empty()) {
                return; // No MPC data available
            }

            const auto& debug = *plan->mpc_debug;
            const int idx     = debug.mpc_current_idx;

            // Build JSON with current values only
            nlohmann::json mpc_json;
            mpc_json["timestamp_ns"]   = plan->timestamp_ns;
            mpc_json["tracking_error"] = plan->tracking_error;
            mpc_json["fire_advice"]    = plan->fire_advice;

            // Reference (input to MPC) at current time
            mpc_json["ref_yaw"]       = debug.ref_yaw[idx];
            mpc_json["ref_pitch"]     = debug.ref_pitch[idx];
            mpc_json["ref_yaw_vel"]   = debug.ref_yaw_vel[idx];
            mpc_json["ref_pitch_vel"] = debug.ref_pitch_vel[idx];

            // Optimized (output from MPC) at current time
            mpc_json["opt_yaw"]       = plan->yaw;
            mpc_json["opt_pitch"]     = plan->pitch;
            mpc_json["opt_yaw_vel"]   = plan->yaw_vel;
            mpc_json["opt_pitch_vel"] = plan->pitch_vel;

            // Feedforward accelerations at current time
            mpc_json["yaw_acc"]   = plan->yaw_acc;
            mpc_json["pitch_acc"] = plan->pitch_acc;

            detail::publish_json_message<MpcTrajectoryMessage>(*server, mpc_json);
        });

    // =========================================================================
    // MPC PREDICTION SCENE (L4) - Full Horizon Visualization
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l4_mpc_prediction_scene",
        [](talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in))
                return;

            auto plan = plan_in.read();
            if (!plan || !plan->mpc_debug || plan->mpc_debug->ref_yaw.empty())
                return;

            auto tracker_out = tracker_in.read_current();
            if (!tracker_out || !tracker_out->is_tracking())
                return;

            const auto& debug = *plan->mpc_debug;
            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Reference trajectory
            for (size_t k = 0; k < debug.ref_yaw.size(); ++k) {
                const double t    = (static_cast<int>(k) - debug.mpc_current_idx) * debug.mpc_dt;
                const double dist = plan->distance + t * tac::Temporal::MPC_EXTRAPOLATION_VEL;
                const double x = dist * std::cos(debug.ref_pitch[k]) * std::cos(debug.ref_yaw[k]);
                const double y = dist * std::cos(debug.ref_pitch[k]) * std::sin(debug.ref_yaw[k]);
                const double z = dist * std::sin(debug.ref_pitch[k]);

                int temporal_dist = std::abs(static_cast<int>(k) - debug.mpc_current_idx);
                entities.push_back(
                    viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                        Eigen::Vector3d(x, y, z), temporal_dist,
                        true, // is_reference
                        plan->timestamp_ns));
            }

            // Optimized trajectory
            for (size_t k = 0; k < debug.opt_yaw.size(); ++k) {
                const double t    = (static_cast<int>(k) - debug.mpc_current_idx) * debug.mpc_dt;
                const double dist = plan->distance + t * tac::Temporal::MPC_EXTRAPOLATION_VEL;
                const double x = dist * std::cos(debug.opt_pitch[k]) * std::cos(debug.opt_yaw[k]);
                const double y = dist * std::cos(debug.opt_pitch[k]) * std::sin(debug.opt_yaw[k]);
                const double z = dist * std::sin(debug.opt_pitch[k]);

                int temporal_dist = std::abs(static_cast<int>(k) - debug.mpc_current_idx);
                entities.push_back(
                    viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                        Eigen::Vector3d(x, y, z), temporal_dist,
                        false, // is_optimized
                        plan->timestamp_ns));
            }

            viz::publish_scene_if_nonempty<MpcPredictionSceneMessage>(*server, std::move(entities));
        });
}

} // namespace fcs::foxglove::systems
