#pragma once

#include "L5_weapon/fire_control.hpp"
#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "system_helpers.hpp"

#include "L4_planning/gimbal_planner/types.hpp"
#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tactical_palette.hpp>

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
            if (!cmd || cmd->distance < 0) {
                return;
            }

            nlohmann::json cmd_json;
            cmd_json["timestamp_ns"]      = cmd->timestamp_ns;
            cmd_json["plan_timestamp_ns"] = cmd->plan_timestamp_ns;
            cmd_json["timestamp"]         = nlohmann::json::object();
            cmd_json["timestamp"]["sec"]  = cmd->timestamp_ns / 1000000000L;
            cmd_json["timestamp"]["nsec"] = cmd->timestamp_ns % 1000000000L;
            cmd_json["pre_plan"]          = {
                {        "yaw",   cmd->plan_yaw},
                {      "pitch", cmd->plan_pitch},
                {"fire_advice",  cmd->plan_fire},
            };
            cmd_json["post_plan"] = {
                {        "yaw",   cmd->yaw},
                {      "pitch", cmd->pitch},
                {"fire_advice",  cmd->fire},
            };
            cmd_json["reference_center"] = {
                {  "yaw",   cmd->ref_yaw},
                {"pitch", cmd->ref_pitch},
            };
            cmd_json["yaw"]         = cmd->yaw;
            cmd_json["pitch"]       = cmd->pitch;
            cmd_json["v_yaw"]       = cmd->yaw_vel;
            cmd_json["v_pitch"]     = cmd->pitch_vel;
            cmd_json["a_yaw"]       = cmd->yaw_accel;
            cmd_json["a_pitch"]     = cmd->pitch_accel;
            cmd_json["distance"]    = cmd->distance;
            cmd_json["fire_advice"] = cmd->fire;

            detail::publish_json_message<GimbalCmdMessage>(*server, cmd_json);
        });

    app.template add_system<talos::pool_compute>(
        "foxglove_l4_mpc_trajectory_pub",
        [](talos::spmc<::fcs::L5::WeaponCommand, WeaponCommandChannelTopic> cmd_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, cmd_in)) {
                return;
            }

            auto cmd = cmd_in.read();
            if (!cmd || !cmd->viz_debug) {
                return;
            }

            nlohmann::json traj_json;
            traj_json["timestamp_ns"]      = cmd->timestamp_ns;
            traj_json["plan_timestamp_ns"] = cmd->plan_timestamp_ns;
            traj_json["center_index"]      = cmd->viz_debug->center_index;
            traj_json["lookahead_index"]   = cmd->viz_debug->lookahead_index;
            traj_json["reference"]         = nlohmann::json::array();
            traj_json["optimized"]         = nlohmann::json::array();

            for (int i = 0; i < static_cast<int>(cmd->viz_debug->reference_plan.size()); ++i) {
                const auto& reference = cmd->viz_debug->reference_plan[i];
                traj_json["reference"].push_back({
                    {          "index",                                i},
                    {"temporal_offset", i - cmd->viz_debug->center_index},
                    {            "yaw",                    reference.yaw},
                    {          "pitch",                  reference.pitch},
                    {       "distance",               reference.distance},
                    {            "tof",                    reference.tof},
                });
            }

            for (int i = 0; i < static_cast<int>(cmd->viz_debug->optimized_plan.size()); ++i) {
                const auto& optimized = cmd->viz_debug->optimized_plan[i];
                traj_json["optimized"].push_back({
                    {          "index",                                i},
                    {"temporal_offset", i - cmd->viz_debug->center_index},
                    {            "yaw",                    optimized.yaw},
                    {          "pitch",                  optimized.pitch},
                    {       "distance",               optimized.distance},
                    {            "tof",                    optimized.tof},
                });
            }

            detail::publish_json_message<MpcTrajectoryMessage>(*server, traj_json);
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
                sphere.color =
                    plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L3::PREDICTION_AMBER;

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

    app.template add_system<talos::pool_compute>(
        "foxglove_l4_mpc_prediction_scene",
        [](talos::spmc<::fcs::L5::WeaponCommand, WeaponCommandChannelTopic> cmd_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, cmd_in)) {
                return;
            }

            auto cmd = cmd_in.read();
            if (!cmd || !cmd->viz_debug) {
                return;
            }

            std::vector<::foxglove::schemas::SceneEntity> entities;
            entities.reserve(
                cmd->viz_debug->reference_plan.size() + cmd->viz_debug->optimized_plan.size());

            for (int i = 0; i < static_cast<int>(cmd->viz_debug->reference_plan.size()); ++i) {
                const auto& reference = cmd->viz_debug->reference_plan[i];
                entities.push_back(
                    viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                        reference.world_position, i - cmd->viz_debug->center_index, true,
                        cmd->timestamp_ns));
            }

            for (int i = 0; i < static_cast<int>(cmd->viz_debug->optimized_plan.size()); ++i) {
                const auto& optimized = cmd->viz_debug->optimized_plan[i];
                entities.push_back(
                    viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                        optimized.world_position, i - cmd->viz_debug->center_index, false,
                        cmd->timestamp_ns));
            }

            viz::publish_scene_if_nonempty<MpcPredictionSceneMessage>(*server, std::move(entities));
        });
}
} // namespace fcs::foxglove::systems
