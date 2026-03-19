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
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <tactical_palette.hpp>

#include <algorithm>
#include <array>
#include <vector>

namespace fcs::foxglove::systems {

namespace l4_detail {

struct FutureArmorSnapshot {
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    Eigen::Vector3d linear_velocity{Eigen::Vector3d::Zero()};
    double angular_velocity{0.0};
    std::vector<Eigen::Vector4d> armors;
};

[[nodiscard]] inline std::string yes_no(bool value) { return value ? "YES" : "NO"; }

[[nodiscard]] inline std::string planner_seed_tracker_last_text(const ::fcs::L4::GimbalPlan& plan) {
    if (!plan.planner_seed || !plan.planner_seed->tracker_last_armor_id.has_value()) {
        return "none";
    }
    return std::to_string(*plan.planner_seed->tracker_last_armor_id);
}

[[nodiscard]] inline std::string short_plan_label(const ::fcs::L4::GimbalPlan& plan) {
    return fmt::format(
        "{} {}\nSelected {}\nRough {}\n{}", magic_enum::enum_name(plan.source),
        magic_enum::enum_name(plan.aim_phase), plan.selected_armor_id, plan.rough_selected_armor_id,
        plan.fire_advice ? "Fire" : "Hold");
}

[[nodiscard]] inline double prediction_dt_s(const ::fcs::L4::GimbalPlan& plan) noexcept {
    if (!plan.planner_seed || plan.predicted_future <= plan.planner_seed->state_timestamp_ns) {
        return 0.0;
    }
    return static_cast<double>(plan.predicted_future - plan.planner_seed->state_timestamp_ns)
         * 1e-9;
}

[[nodiscard]] inline std::optional<FutureArmorSnapshot>
    future_armor_snapshot(const ::fcs::L4::GimbalPlan& plan) {
    if (!plan.planner_seed) {
        return std::nullopt;
    }

    const double dt = prediction_dt_s(plan);

    if (const auto* robot = plan.planner_seed->robot_state()) {
        ::fcs::L3::RobotTargetState predicted = *robot;
        predicted.position += predicted.velocity * dt;
        predicted.yaw += predicted.v_yaw * dt;

        FutureArmorSnapshot snapshot;
        snapshot.center           = predicted.position;
        snapshot.linear_velocity  = predicted.velocity;
        snapshot.angular_velocity = predicted.v_yaw;

        const auto armors = predicted.armor_poses();
        snapshot.armors.assign(armors.begin(), armors.end());
        return snapshot;
    }

    if (const auto* outpost = plan.planner_seed->outpost_state()) {
        ::fcs::L3::OutpostTargetState predicted = *outpost;
        predicted.position += predicted.velocity.head<2>() * dt;
        predicted.yaw += predicted.v_yaw * dt;

        FutureArmorSnapshot snapshot;
        snapshot.center = Eigen::Vector3d{
            predicted.position.x(),
            predicted.position.y(),
            (predicted.z[0] + predicted.z[1] + predicted.z[2]) / 3.0,
        };
        snapshot.linear_velocity  = predicted.velocity;
        snapshot.angular_velocity = predicted.v_yaw;

        const auto armors = predicted.armor_poses();
        snapshot.armors.assign(armors.begin(), armors.end());
        return snapshot;
    }

    return std::nullopt;
}

[[nodiscard]] inline std::string armor_id_label(int armor_id) {
    return fmt::format("Armor {}", armor_id);
}

[[nodiscard]] inline Eigen::Quaterniond armor_orientation(double yaw) {
    return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(tac::Geometry::ARMOR_TILT_ANGLE, Eigen::Vector3d::UnitY());
}

[[nodiscard]] inline std::string
    fire_gate_board(const ::fcs::L5::WeaponVisualizationDebugData& dbg) {
    return fmt::format(
        "{}\nForward Delay {}\nBackward Delay {}\nPhase {}",
        dbg.final_fire_gate_ok ? "Fire" : "Hold", dbg.delay_gate_forward_ok ? "Pass" : "Fail",
        dbg.delay_gate_backward_ok ? "Pass" : "Fail", dbg.phase_gate_ok ? "Pass" : "Fail");
}

[[nodiscard]] inline std::string
    key_point_label(const std::string& name, const ::fcs::L5::TrajectoryPlanSample& sample) {
    return fmt::format(
        "{}\nSelected {}\nRough {}", name,
        sample.selected_armor_id ? std::to_string(*sample.selected_armor_id) : "-",
        sample.rough_selected_armor_id ? std::to_string(*sample.rough_selected_armor_id) : "-");
}

[[nodiscard]] inline nlohmann::json json_vec3(const Eigen::Vector3d& value) {
    return {value.x(), value.y(), value.z()};
}

[[nodiscard]] inline nlohmann::json json_robot_state(const ::fcs::L3::RobotTargetState& state) {
    nlohmann::json state_json = {
        {       "type",                        "robot"},
        {   "position",      json_vec3(state.position)},
        {   "velocity",      json_vec3(state.velocity)},
        {        "yaw",                      state.yaw},
        {      "v_yaw",                    state.v_yaw},
        {     "radius", {state.radius0, state.radius1}},
        {         "z1",                       state.z1},
        { "armors_num",               state.armors_num},
        {"armor_poses",        nlohmann::json::array()},
    };

    const auto armors = state.armor_poses();
    for (size_t i = 0; i < armors.size(); ++i) {
        state_json["armor_poses"].push_back({
            {      "id",                        static_cast<int>(i)},
            {"position", {armors[i][0], armors[i][1], armors[i][2]}},
            {     "yaw",                               armors[i][3]},
        });
    }

    return state_json;
}

[[nodiscard]] inline nlohmann::json json_outpost_state(const ::fcs::L3::OutpostTargetState& state) {
    nlohmann::json state_json = {
        {       "type",                                 "outpost"},
        {   "position",  {state.position.x(), state.position.y()}},
        {   "velocity",                 json_vec3(state.velocity)},
        {        "yaw",                                 state.yaw},
        {      "v_yaw",                               state.v_yaw},
        {          "z",      {state.z[0], state.z[1], state.z[2]}},
        {     "radius",     ::fcs::L3::OutpostTargetState::radius},
        { "armors_num", ::fcs::L3::OutpostTargetState::armors_num},
        {"armor_poses",                   nlohmann::json::array()},
    };

    const auto armors = state.armor_poses();
    for (size_t i = 0; i < armors.size(); ++i) {
        state_json["armor_poses"].push_back({
            {      "id",                        static_cast<int>(i)},
            {"position", {armors[i][0], armors[i][1], armors[i][2]}},
            {     "yaw",                               armors[i][3]},
        });
    }

    return state_json;
}

[[nodiscard]] inline nlohmann::json json_planner_seed(const ::fcs::L4::PlannerSeed& seed) {
    nlohmann::json seed_json = {
        {   "state_timestamp_ns",                                        seed.state_timestamp_ns},
        {        "target_jumped",                                             seed.target_jumped},
        {"tracker_last_armor_id", seed.tracker_last_armor_id
 ? nlohmann::json(*seed.tracker_last_armor_id)
 : nlohmann::json(nullptr)                                     },
        {            "aim_phase",             std::string(magic_enum::enum_name(seed.aim_phase))},
        {      "aim_phase_value",                               static_cast<int>(seed.aim_phase)},
        {    "selected_armor_id",                                         seed.selected_armor_id},
        {                "state",                                                        nullptr},
    };

    if (const auto* robot = seed.robot_state()) {
        seed_json["state"] = json_robot_state(*robot);
    } else if (const auto* outpost = seed.outpost_state()) {
        seed_json["state"] = json_outpost_state(*outpost);
    }

    return seed_json;
}

[[nodiscard]] inline nlohmann::json json_ballistic_trajectory(const ::fcs::L4::GimbalPlan& plan) {
    nlohmann::json trajectory = nlohmann::json::array();
    if (!plan.viz_debug) {
        return trajectory;
    }

    for (const auto& [x, z] : plan.viz_debug->trajectory) {
        trajectory.push_back({
            {"x", x},
            {"z", z},
        });
    }

    return trajectory;
}

[[nodiscard]] inline nlohmann::json json_gimbal_plan(const ::fcs::L4::GimbalPlan& plan) {
    nlohmann::json plan_json = {
        {           "timestamp_ns",         plan.timestamp_ns                                   },
        {               "distance",                                                plan.distance},
        {    "predicted_future_ns",                                        plan.predicted_future},
        {                 "source",              std::string(magic_enum::enum_name(plan.source))},
        {           "source_value",                                static_cast<int>(plan.source)},
        {                    "yaw",                                                     plan.yaw},
        {                  "pitch",                                                   plan.pitch},
        {               "diff_yaw",                                                plan.diff_yaw},
        {              "aim_phase",           std::string(magic_enum::enum_name(plan.aim_phase))},
        {        "aim_phase_value",                             static_cast<int>(plan.aim_phase)},
        {      "selected_armor_id",                                       plan.selected_armor_id},
        {"rough_selected_armor_id",                                 plan.rough_selected_armor_id},
        {                "yaw_vel",                                                 plan.yaw_vel},
        {              "pitch_vel",                                               plan.pitch_vel},
        {                "yaw_acc",                                                 plan.yaw_acc},
        {              "pitch_acc",                                               plan.pitch_acc},
        {            "fire_advice",                                             plan.fire_advice},
        {         "tracking_error",                                          plan.tracking_error},
        {      "aimer_fire_advice",                                       plan.aimer_fire_advice},
        {                "aim_yaw",                                                 plan.aim_yaw},
        {              "aim_pitch",                                               plan.aim_pitch},
        {     "predicted_position",
         plan.predicted_position ? json_vec3(*plan.predicted_position) : nlohmann::json(nullptr)},
        {           "planner_seed",
         plan.planner_seed ? json_planner_seed(*plan.planner_seed) : nlohmann::json(nullptr)    },
        {   "ballistic_trajectory",                              json_ballistic_trajectory(plan)},
    };
    return plan_json;
}

template <typename PlanPtr>
[[nodiscard]] inline const ::fcs::L4::GimbalPlan*
    matched_plan(const PlanPtr& plan, uint64_t plan_timestamp_ns) {
    if (!plan || plan->timestamp_ns != plan_timestamp_ns) {
        return nullptr;
    }
    return std::addressof(*plan);
}

inline void append_plan_metadata(viz::EntityBuilder& builder, const ::fcs::L4::GimbalPlan& plan) {
    builder.metadata("source", std::string(magic_enum::enum_name(plan.source)))
        .metadata("aim_phase", std::string(magic_enum::enum_name(plan.aim_phase)))
        .metadata("selected_armor_id", std::to_string(plan.selected_armor_id))
        .metadata("rough_selected_armor_id", std::to_string(plan.rough_selected_armor_id))
        .metadata("fire_advice", yes_no(plan.fire_advice))
        .metadata("aimer_fire_advice", yes_no(plan.aimer_fire_advice))
        .metadata("predicted_future_ns", std::to_string(plan.predicted_future));

    if (plan.planner_seed) {
        builder.metadata("planner_seed.target_jumped", yes_no(plan.planner_seed->target_jumped))
            .metadata("planner_seed.tracker_last_armor_id", planner_seed_tracker_last_text(plan));
    }
}

} // namespace l4_detail

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
           talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, cmd_in)) {
                return;
            }

            auto cmd = cmd_in.read();
            if (!cmd || cmd->distance < 0) {
                return;
            }
            const auto plan =
                l4_detail::matched_plan(plan_in.read_current(), cmd->plan_timestamp_ns);

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
            cmd_json["tof"]         = cmd->tof;
            cmd_json["fire_advice"] = cmd->fire;
            if (cmd->viz_debug) {
                cmd_json["mpc"] = {
                    {             "center_index",              cmd->viz_debug->center_index},
                    {          "lookahead_index",           cmd->viz_debug->lookahead_index},
                    {           "reference_size",     cmd->viz_debug->reference_plan.size()},
                    {           "optimized_size",     cmd->viz_debug->optimized_plan.size()},
                    {              "delay_steps",               cmd->viz_debug->delay_steps},
                    {     "delay_gate_threshold",      cmd->viz_debug->delay_gate_threshold},
                    { "delay_gate_forward_error",  cmd->viz_debug->delay_gate_forward_error},
                    {"delay_gate_backward_error", cmd->viz_debug->delay_gate_backward_error},
                    {    "delay_gate_forward_ok",     cmd->viz_debug->delay_gate_forward_ok},
                    {   "delay_gate_backward_ok",    cmd->viz_debug->delay_gate_backward_ok},
                    {            "phase_gate_ok",             cmd->viz_debug->phase_gate_ok},
                    {       "final_fire_gate_ok",        cmd->viz_debug->final_fire_gate_ok},
                };
            }
            if (plan) {
                cmd_json["l4_plan"] = l4_detail::json_gimbal_plan(*plan);
            }

            detail::publish_json_message<GimbalCmdMessage>(*server, cmd_json);
        });

    app.template add_system<talos::pool_compute>(
        "foxglove_l4_mpc_trajectory_pub",
        [](talos::spmc<::fcs::L5::WeaponCommand, WeaponCommandChannelTopic> cmd_in,
           talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, cmd_in)) {
                return;
            }

            auto cmd = cmd_in.read();
            if (!cmd || !cmd->viz_debug) {
                return;
            }
            const auto plan =
                l4_detail::matched_plan(plan_in.read_current(), cmd->plan_timestamp_ns);

            nlohmann::json traj_json;
            traj_json["timestamp_ns"]      = cmd->timestamp_ns;
            traj_json["plan_timestamp_ns"] = cmd->plan_timestamp_ns;
            traj_json["center_index"]      = cmd->viz_debug->center_index;
            traj_json["lookahead_index"]   = cmd->viz_debug->lookahead_index;
            traj_json["fire_gate"]         = {
                {              "delay_steps",               cmd->viz_debug->delay_steps},
                {     "delay_gate_threshold",      cmd->viz_debug->delay_gate_threshold},
                { "delay_gate_forward_error",  cmd->viz_debug->delay_gate_forward_error},
                {"delay_gate_backward_error", cmd->viz_debug->delay_gate_backward_error},
                {    "delay_gate_forward_ok",     cmd->viz_debug->delay_gate_forward_ok},
                {   "delay_gate_backward_ok",    cmd->viz_debug->delay_gate_backward_ok},
                {            "phase_gate_ok",             cmd->viz_debug->phase_gate_ok},
                {       "final_fire_gate_ok",        cmd->viz_debug->final_fire_gate_ok},
            };
            traj_json["reference"] = nlohmann::json::array();
            traj_json["optimized"] = nlohmann::json::array();
            if (plan) {
                traj_json["l4_plan"] = l4_detail::json_gimbal_plan(*plan);
            }

            for (int i = 0; i < static_cast<int>(cmd->viz_debug->reference_plan.size()); ++i) {
                const auto& reference = cmd->viz_debug->reference_plan[i];
                traj_json["reference"].push_back({
                    {                  "index",                       i                                               },
                    {        "temporal_offset",                                       i - cmd->viz_debug->center_index},
                    {                    "yaw",                                                          reference.yaw},
                    {                  "pitch",                                                        reference.pitch},
                    {               "distance",                                                     reference.distance},
                    {                    "tof",                                                          reference.tof},
                    {               "diff_yaw",                                                     reference.diff_yaw},
                    {      "selected_armor_id", reference.selected_armor_id
 ? nlohmann::json(*reference.selected_armor_id)
 : nlohmann::json(nullptr)                                             },
                    {"rough_selected_armor_id",
                     reference.rough_selected_armor_id
                     ? nlohmann::json(*reference.rough_selected_armor_id)
                     : nlohmann::json(nullptr)                                                                        },
                    {         "world_position",                         l4_detail::json_vec3(reference.world_position)},
                });
            }

            for (int i = 0; i < static_cast<int>(cmd->viz_debug->optimized_plan.size()); ++i) {
                const auto& optimized = cmd->viz_debug->optimized_plan[i];
                traj_json["optimized"].push_back({
                    {                  "index",                       i                                               },
                    {        "temporal_offset",                                       i - cmd->viz_debug->center_index},
                    {                    "yaw",                                                          optimized.yaw},
                    {                  "pitch",                                                        optimized.pitch},
                    {               "distance",                                                     optimized.distance},
                    {                    "tof",                                                          optimized.tof},
                    {               "diff_yaw",                                                     optimized.diff_yaw},
                    {      "selected_armor_id", optimized.selected_armor_id
 ? nlohmann::json(*optimized.selected_armor_id)
 : nlohmann::json(nullptr)                                             },
                    {"rough_selected_armor_id",
                     optimized.rough_selected_armor_id
                     ? nlohmann::json(*optimized.rough_selected_armor_id)
                     : nlohmann::json(nullptr)                                                                        },
                    {         "world_position",                         l4_detail::json_vec3(optimized.world_position)},
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

            if (const auto snapshot = l4_detail::future_armor_snapshot(*plan)) {
                entities.push_back(
                    viz::EntityBuilder::create<fast_tf::odom>("l4", "target_center")
                        .timestamp(plan->predicted_future)
                        .position(snapshot->center)
                        .size(tac::L3::ROBOT_CENTER_SIZE * 0.75)
                        .color(tac::L4::MPC_PRESENT)
                        .alpha(0.6)
                        .sphere()
                        .text_with_offset(
                            "Target Center", 0.0, 0.0, tac::L3::LABEL_OFFSET_Z,
                            tac::Text::SIZE_SMALL)
                        .build());

                entities.push_back(
                    viz::patterns::velocity_arrows<fast_tf::odom>(
                        snapshot->center, snapshot->linear_velocity, snapshot->angular_velocity,
                        plan->predicted_future));

                for (int i = 0; i < static_cast<int>(snapshot->armors.size()); ++i) {
                    const auto& armor_pose          = snapshot->armors[static_cast<size_t>(i)];
                    const Eigen::Vector3d armor_pos = armor_pose.head<3>();
                    entities.push_back(
                        viz::EntityBuilder::create<fast_tf::odom>(
                            "l4", fmt::format("future_armor_{}", i))
                            .timestamp(plan->predicted_future)
                            .position(armor_pos)
                            .orientation(l4_detail::armor_orientation(armor_pose[3]))
                            .color(tac::L4::GIMBAL_AIM_HOLD)
                            .alpha(0.18)
                            .cube(
                                tac::Geometry::ARMOR_THICKNESS, tac::Geometry::ARMOR_HEIGHT_SMALL,
                                tac::Geometry::ARMOR_WIDTH)
                            .text_with_offset(
                                std::to_string(i), 0.0, 0.0, tac::L3::LABEL_OFFSET_Z,
                                tac::Text::SIZE_SMALL)
                            .metadata("armor_id", std::to_string(i))
                            .build());
                }

                const auto add_overlay =
                    [&](int armor_id, const std::string& entity_id, const std::string& label,
                        const ::foxglove::schemas::Color& color, double size_scale) {
                        if (armor_id < 0 || armor_id >= static_cast<int>(snapshot->armors.size())) {
                            return;
                        }
                        const Eigen::Vector3d armor_pos =
                            snapshot->armors[static_cast<size_t>(armor_id)].head<3>();
                        entities.push_back(
                            viz::EntityBuilder::create<fast_tf::odom>("l4", entity_id)
                                .timestamp(plan->predicted_future)
                                .position(armor_pos)
                                .size(tac::L4::SELECTION_SIZE * size_scale)
                                .color(color)
                                .sphere()
                                .text_with_offset(
                                    label, 0.0, 0.0, tac::L3::LABEL_OFFSET_Z * 1.6,
                                    tac::Text::SIZE_SMALL)
                                .metadata("armor_id", std::to_string(armor_id))
                                .build());
                    };

                if (plan->selected_armor_id == plan->rough_selected_armor_id) {
                    add_overlay(
                        plan->selected_armor_id, "selected_rough_armor",
                        fmt::format("Selected and Rough {}", plan->selected_armor_id),
                        plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L4::GIMBAL_AIM_HOLD,
                        1.25);
                } else {
                    add_overlay(
                        plan->rough_selected_armor_id, "rough_armor",
                        fmt::format("Rough {}", plan->rough_selected_armor_id),
                        tac::L4::MPC_REFERENCE, 1.0);
                    add_overlay(
                        plan->selected_armor_id, "selected_armor",
                        fmt::format("Selected {}", plan->selected_armor_id),
                        plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L4::GIMBAL_AIM_HOLD,
                        1.25);
                }
            }

            // Predicted position marker (at bullet impact time for verification)
            if (plan->predicted_position) {
                auto builder = viz::EntityBuilder::create<fast_tf::odom>("l4", "predicted")
                                   .timestamp(plan->predicted_future)
                                   .position(*plan->predicted_position)
                                   .size(tac::L4::PREDICTION_SIZE)
                                   .color(
                                       plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE
                                                         : tac::L3::PREDICTION_AMBER)
                                   .sphere();

                l4_detail::append_plan_metadata(builder, *plan);

                builder.color(tac::Text::PRIMARY)
                    .text_with_offset(
                        "Aim Point", 0.0, 0.0, tac::L3::LABEL_OFFSET_Z + tac::L4::PREDICTION_SIZE,
                        tac::Text::SIZE_SMALL)
                    .text_with_offset(
                        l4_detail::short_plan_label(*plan), tac::L4::SELECTION_SIZE * 1.8, 0.0,
                        tac::L3::LABEL_OFFSET_Z + tac::L4::PREDICTION_SIZE, tac::Text::SIZE_SMALL);

                entities.push_back(builder.build());
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

            const int center_index = std::clamp(
                cmd->viz_debug->center_index, 0,
                static_cast<int>(cmd->viz_debug->optimized_plan.size()) - 1);
            const int forward_index = std::clamp(
                center_index + cmd->viz_debug->delay_steps, 0,
                static_cast<int>(cmd->viz_debug->optimized_plan.size()) - 1);
            const int backward_index = std::clamp(
                center_index - cmd->viz_debug->delay_steps, 0,
                static_cast<int>(cmd->viz_debug->optimized_plan.size()) - 1);

            const auto add_key_marker = [&](const std::string& entity_id,
                                            const ::fcs::L5::TrajectoryPlanSample& sample,
                                            const std::string& label,
                                            const ::foxglove::schemas::Color& color, double size) {
                entities.push_back(
                    viz::EntityBuilder::create<fast_tf::odom>("l4", entity_id)
                        .timestamp(cmd->timestamp_ns)
                        .position(sample.world_position)
                        .size(size)
                        .color(color)
                        .sphere()
                        .text_with_offset(
                            label, 0.0, 0.0, tac::L3::LABEL_OFFSET_Z, tac::Text::SIZE_SMALL)
                        .build());
            };

            add_key_marker(
                "mpc_ref_center", cmd->viz_debug->reference_plan[static_cast<size_t>(center_index)],
                l4_detail::key_point_label(
                    "Reference Center",
                    cmd->viz_debug->reference_plan[static_cast<size_t>(center_index)]),
                tac::L4::MPC_REFERENCE, tac::L4::SELECTION_SIZE * 1.1);

            add_key_marker(
                "mpc_opt_center", cmd->viz_debug->optimized_plan[static_cast<size_t>(center_index)],
                l4_detail::key_point_label(
                    "Optimized Center",
                    cmd->viz_debug->optimized_plan[static_cast<size_t>(center_index)]),
                tac::L4::MPC_PRESENT, tac::L4::SELECTION_SIZE * 1.15);

            entities.push_back(
                viz::EntityBuilder::create<fast_tf::odom>("l4", "mpc_gate_board")
                    .timestamp(cmd->timestamp_ns)
                    .position(cmd->viz_debug->optimized_plan[static_cast<size_t>(center_index)]
                                  .world_position)
                    .color(tac::Text::PRIMARY)
                    .text_with_offset(
                        l4_detail::fire_gate_board(*cmd->viz_debug), tac::L4::SELECTION_SIZE * 1.8,
                        0.0, tac::L3::LABEL_OFFSET_Z, tac::Text::SIZE_SMALL)
                    .build());

            if (forward_index != center_index) {
                add_key_marker(
                    "mpc_plus_delay",
                    cmd->viz_debug->optimized_plan[static_cast<size_t>(forward_index)],
                    fmt::format(
                        "Forward Delay {}\n{:.4f}",
                        cmd->viz_debug->delay_gate_forward_ok ? "Pass" : "Fail",
                        cmd->viz_debug->delay_gate_forward_error),
                    cmd->viz_debug->delay_gate_forward_ok ? tac::L5::FIRE_READY
                                                          : tac::L5::FIRE_ABORTED,
                    tac::L4::SELECTION_SIZE);
            }

            if (backward_index != center_index && backward_index != forward_index) {
                add_key_marker(
                    "mpc_minus_delay",
                    cmd->viz_debug->optimized_plan[static_cast<size_t>(backward_index)],
                    fmt::format(
                        "Backward Delay {}\n{:.4f}",
                        cmd->viz_debug->delay_gate_backward_ok ? "Pass" : "Fail",
                        cmd->viz_debug->delay_gate_backward_error),
                    cmd->viz_debug->delay_gate_backward_ok ? tac::L5::FIRE_READY
                                                           : tac::L5::FIRE_ABORTED,
                    tac::L4::SELECTION_SIZE);
            }

            viz::publish_scene_if_nonempty<MpcPredictionSceneMessage>(*server, std::move(entities));
        });
}
} // namespace fcs::foxglove::systems
