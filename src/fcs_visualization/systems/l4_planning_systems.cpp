#include "L4_planning/selected_target_snapshot.hpp"
#include "L4_planning/target_selection_trace.hpp"
#include "L5_weapon/fire_control.hpp"
#include "base.hpp"
#include "core/time.hpp"
#include "core/trajectory/resource.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"

#include "L4_planning/gimbal_planner/types.hpp"
#include "core/channel_topics.hpp"
#include "frame.hpp"

#include <fmt/format.h>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <tactical_palette.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace fcs::visualization::foxglove::systems {

namespace l4_detail {

struct FutureArmorSnapshot {
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    Eigen::Vector3d linear_velocity{Eigen::Vector3d::Zero()};
    double angular_velocity{0.0};
    std::vector<Eigen::Vector4d> armors;
};

[[nodiscard]] inline std::string yes_no(bool value) { return value ? "YES" : "NO"; }

[[nodiscard]] inline std::string tracker_identity(const ::fcs::L3::TrackerOutput& target) {
    return fmt::format(
        "{}/{}", magic_enum::enum_name(target.target_color),
        magic_enum::enum_name(target.target_name));
}

[[nodiscard]] inline std::string
    selection_identity(const ::fcs::L4::TargetSelectionTraceEntry& entry) {
    return fmt::format(
        "{}/{}", magic_enum::enum_name(entry.target_color),
        magic_enum::enum_name(entry.target_name));
}

[[nodiscard]] inline std::string format_score(double value) { return fmt::format("{:.3f}", value); }

[[nodiscard]] inline std::string format_image_center_px(double value) {
    return std::isfinite(value) ? fmt::format("{:.1f}", value) : "n/a";
}

[[nodiscard]] inline std::string planner_seed_tracker_last_text(const ::fcs::L4::GimbalPlan& plan) {
    if (!plan.planner_seed || !plan.planner_seed->tracker_last_armor_id.has_value()) {
        return "none";
    }
    return std::to_string(*plan.planner_seed->tracker_last_armor_id);
}

[[nodiscard]] inline std::string
    selection_trace_entity_id(const ::fcs::L4::TargetSelectionTraceEntry& entry) {
    return fmt::format(
        "selection_candidate_{}_{}", magic_enum::enum_name(entry.target_color),
        magic_enum::enum_name(entry.target_name));
}

inline void append_selection_trace_metadata(
    viz::EntityBuilder& builder, const ::fcs::L4::TargetSelectionTrace& trace,
    const ::fcs::L4::TargetSelectionTraceEntry& entry) {
    builder.metadata("target_selection.identity", selection_identity(entry))
        .metadata("target_selection.rank", std::to_string(entry.rank))
        .metadata("target_selection.selected", yes_no(entry.selected))
        .metadata("target_selection.runner_up", yes_no(entry.runner_up))
        .metadata("target_selection.was_previously_selected", yes_no(entry.was_previously_selected))
        .metadata("target_selection.aim_valid", yes_no(entry.aim_valid))
        .metadata(
            "target_selection.track_status", std::string(magic_enum::enum_name(entry.track_status)))
        .metadata("target_selection.total_score", format_score(entry.total_score))
        .metadata("target_selection.image_center_score", format_score(entry.image_center_score))
        .metadata("target_selection.track_state_score", format_score(entry.track_state_score))
        .metadata("target_selection.tof_score", format_score(entry.tof_score))
        .metadata("target_selection.distance_score", format_score(entry.distance_score))
        .metadata("target_selection.gimbal_effort_score", format_score(entry.gimbal_effort_score))
        .metadata("target_selection.armor_name_score", format_score(entry.armor_name_score))
        .metadata(
            "target_selection.image_center_weighted", format_score(entry.image_center_weighted))
        .metadata("target_selection.track_state_weighted", format_score(entry.track_state_weighted))
        .metadata("target_selection.tof_weighted", format_score(entry.tof_weighted))
        .metadata("target_selection.distance_weighted", format_score(entry.distance_weighted))
        .metadata(
            "target_selection.gimbal_effort_weighted", format_score(entry.gimbal_effort_weighted))
        .metadata("target_selection.armor_name_weighted", format_score(entry.armor_name_weighted))
        .metadata("target_selection.weighted_sum", format_score(entry.weighted_sum))
        .metadata("target_selection.total_weight", format_score(entry.total_weight))
        .metadata(
            "target_selection.image_center_distance_px",
            format_image_center_px(entry.image_center_distance_px))
        .metadata("target_selection.optical_age_s", format_score(entry.optical_age_s))
        .metadata(
            "target_selection.tof_s",
            std::isfinite(entry.tof_s) ? format_score(entry.tof_s) : "n/a")
        .metadata(
            "target_selection.distance_m",
            std::isfinite(entry.distance_m) ? format_score(entry.distance_m) : "n/a")
        .metadata(
            "target_selection.yaw_effort_deg",
            std::isfinite(entry.yaw_effort_deg) ? format_score(entry.yaw_effort_deg) : "n/a")
        .metadata(
            "target_selection.pitch_effort_deg",
            std::isfinite(entry.pitch_effort_deg) ? format_score(entry.pitch_effort_deg) : "n/a")
        .metadata("target_selection.kept_current_target", yes_no(trace.kept_current_target))
        .metadata("target_selection.switch_margin", format_score(trace.switch_margin))
        .metadata(
            "target_selection.previous_target",
            trace.had_previous_target
                ? fmt::format(
                      "{}/{}", magic_enum::enum_name(trace.previous_target_color),
                      magic_enum::enum_name(trace.previous_target_name))
                : "none")
        .metadata("target_selection.aim_error", entry.aim_error.empty() ? "none" : entry.aim_error);

    if (entry.target_center) {
        builder.metadata(
            "target_selection.target_center",
            fmt::format(
                "{:.3f}, {:.3f}, {:.3f}", entry.target_center->x(), entry.target_center->y(),
                entry.target_center->z()));
    }
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

[[nodiscard]] inline nlohmann::json json_tracker_output(const ::fcs::L3::TrackerOutput& output) {
    nlohmann::json target_json = {
        {                 "timestamp_ns",           output.timestamp_ns                                         },
        {                       "status",                                   magic_enum::enum_name(output.status)},
        {                  "target_name",                              magic_enum::enum_name(output.target_name)},
        {                 "target_color",                             magic_enum::enum_name(output.target_color)},
        {                "target_jumped",                                                   output.target_jumped},
        {                "last_armor_id",
         output.last_armor_id ? nlohmann::json(*output.last_armor_id) : nlohmann::json(nullptr)                 },
        {"last_image_center_distance_px", std::isfinite(output.last_image_center_distance_px)
 ? nlohmann::json(output.last_image_center_distance_px)
 : nlohmann::json(nullptr)                                             },
        {"last_observation_timestamp_ns",                                   output.last_observation_timestamp_ns},
        {                   "state_kind",                                                                "empty"},
    };

    if (const auto* robot = output.robot_state()) {
        target_json["state_kind"] = "robot";
        target_json["state"]      = json_robot_state(*robot);
    } else if (const auto* outpost = output.outpost_state()) {
        target_json["state_kind"] = "outpost";
        target_json["state"]      = json_outpost_state(*outpost);
    } else {
        target_json["state"] = nlohmann::json::object();
    }

    return target_json;
}

[[nodiscard]] inline nlohmann::json
    json_target_selection_trace_entry(const ::fcs::L4::TargetSelectionTraceEntry& entry) {
    return nlohmann::json{
        {                    "rank",                                  entry.rank                                    },
        {             "target_name",                           std::string(magic_enum::enum_name(entry.target_name))},
        {            "target_color",                          std::string(magic_enum::enum_name(entry.target_color))},
        {            "track_status",                          std::string(magic_enum::enum_name(entry.track_status))},
        {               "aim_valid",                                                                 entry.aim_valid},
        { "was_previously_selected",                                                   entry.was_previously_selected},
        {                "selected",                                                                  entry.selected},
        {               "runner_up",                                                                 entry.runner_up},
        {               "aim_error",                                                                 entry.aim_error},
        {           "target_center",          entry.target_center ? nlohmann::json(json_vec3(*entry.target_center))
          : nlohmann::json(nullptr)                                             },
        {"image_center_distance_px",               std::isfinite(entry.image_center_distance_px)
               ? nlohmann::json(entry.image_center_distance_px)
               : nlohmann::json(nullptr)                                        },
        {           "optical_age_s",                                                             entry.optical_age_s},
        {                   "tof_s",
         std::isfinite(entry.tof_s) ? nlohmann::json(entry.tof_s) : nlohmann::json(nullptr)                         },
        {              "distance_m", std::isfinite(entry.distance_m) ? nlohmann::json(entry.distance_m)
 : nlohmann::json(nullptr)                                                      },
        {          "yaw_effort_deg",                         std::isfinite(entry.yaw_effort_deg)
                         ? nlohmann::json(entry.yaw_effort_deg)
                         : nlohmann::json(nullptr)                              },
        {        "pitch_effort_deg",                       std::isfinite(entry.pitch_effort_deg)
                       ? nlohmann::json(entry.pitch_effort_deg)
                       : nlohmann::json(nullptr)                                },
        {      "image_center_score",                                                        entry.image_center_score},
        {       "track_state_score",                                                         entry.track_state_score},
        {               "tof_score",                                                                 entry.tof_score},
        {          "distance_score",                                                            entry.distance_score},
        {     "gimbal_effort_score",                                                       entry.gimbal_effort_score},
        {        "armor_name_score",                                                          entry.armor_name_score},
        {   "image_center_weighted",                                                     entry.image_center_weighted},
        {    "track_state_weighted",                                                      entry.track_state_weighted},
        {            "tof_weighted",                                                              entry.tof_weighted},
        {       "distance_weighted",                                                         entry.distance_weighted},
        {  "gimbal_effort_weighted",                                                    entry.gimbal_effort_weighted},
        {     "armor_name_weighted",                                                       entry.armor_name_weighted},
        {            "weighted_sum",                                                              entry.weighted_sum},
        {            "total_weight",                                                              entry.total_weight},
        {             "total_score",                                                               entry.total_score},
    };
}

[[nodiscard]] inline nlohmann::json
    json_target_selection_trace(const ::fcs::L4::TargetSelectionTrace& trace) {
    nlohmann::json candidates = nlohmann::json::array();
    for (const auto& candidate : trace.candidates) {
        candidates.push_back(json_target_selection_trace_entry(candidate));
    }

    return nlohmann::json{
        {         "timestamp_ns",                                              trace.timestamp_ns},
        {  "had_previous_target",                                       trace.had_previous_target},
        { "previous_target_name",  std::string(magic_enum::enum_name(trace.previous_target_name))},
        {"previous_target_color", std::string(magic_enum::enum_name(trace.previous_target_color))},
        {  "kept_current_target",                                       trace.kept_current_target},
        {        "switch_margin",                                             trace.switch_margin},
        {           "candidates",                                           std::move(candidates)},
    };
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

[[nodiscard]] inline nlohmann::json json_gimbal_plan(const ::fcs::L4::GimbalPlan& plan) {
    nlohmann::json plan_json = {
        {           "timestamp_ns",         plan.timestamp_ns                                   },
        {               "distance",                                                plan.distance},
        {    "predicted_future_ns",                                        plan.predicted_future},
        {                 "source",              std::string(magic_enum::enum_name(plan.source))},
        {           "source_value",                                static_cast<int>(plan.source)},
        {                    "yaw",                                                     plan.yaw},
        {                  "pitch",                                                   plan.pitch},
        {              "aim_phase",           std::string(magic_enum::enum_name(plan.aim_phase))},
        {        "aim_phase_value",                             static_cast<int>(plan.aim_phase)},
        {      "selected_armor_id",                                       plan.selected_armor_id},
        {"rough_selected_armor_id",                                 plan.rough_selected_armor_id},
        {                "yaw_vel",                                                 plan.yaw_vel},
        {              "pitch_vel",                                               plan.pitch_vel},
        {                "yaw_acc",                                                 plan.yaw_acc},
        {              "pitch_acc",                                               plan.pitch_acc},
        {            "fire_advice",                                             plan.fire_advice},
        {                "aim_yaw",                                                 plan.aim_yaw},
        {              "aim_pitch",                                               plan.aim_pitch},
        {     "predicted_position",
         plan.predicted_position ? json_vec3(*plan.predicted_position) : nlohmann::json(nullptr)},
        {           "planner_seed",
         plan.planner_seed ? json_planner_seed(*plan.planner_seed) : nlohmann::json(nullptr)    },
    };

    return plan_json;
}

[[nodiscard]] inline nlohmann::json
    json_solver_target(const ::fcs::L4::SelectedTargetSnapshot& snapshot) {
    nlohmann::json target_json             = json_tracker_output(snapshot.tracker);
    target_json["timestamp_ns"]            = snapshot.timestamp_ns;
    target_json["timestamp"]               = nlohmann::json::object();
    target_json["timestamp"]["sec"]        = snapshot.timestamp_ns / 1000000000L;
    target_json["timestamp"]["nsec"]       = snapshot.timestamp_ns % 1000000000L;
    target_json["tracking"]                = snapshot.tracker.is_tracking();
    target_json["valid"]                   = snapshot.has_target();
    target_json["optimal_target"]          = target_json["valid"];
    target_json["source"]                  = std::string(magic_enum::enum_name(snapshot.source));
    target_json["plan_distance"]           = snapshot.distance;
    target_json["predicted_future_ns"]     = snapshot.predicted_future_ns;
    target_json["aim_phase"]               = std::string(magic_enum::enum_name(snapshot.aim_phase));
    target_json["selected_armor_id"]       = snapshot.selected_armor_id;
    target_json["rough_selected_armor_id"] = snapshot.rough_selected_armor_id;

    return target_json;
}

inline void append_selected_target_metadata(
    viz::EntityBuilder& builder, const ::fcs::L4::SelectedTargetSnapshot& snapshot) {
    builder.metadata("target.identity", tracker_identity(snapshot.tracker))
        .metadata("target.status", std::string(magic_enum::enum_name(snapshot.tracker.status)))
        .metadata(
            "target.last_image_center_distance_px",
            format_image_center_px(snapshot.tracker.last_image_center_distance_px))
        .metadata(
            "target.last_observation_timestamp_ns",
            std::to_string(snapshot.tracker.last_observation_timestamp_ns));
}

inline void append_plan_metadata(
    viz::EntityBuilder& builder, const ::fcs::L4::GimbalPlan& plan,
    const ::fcs::L4::SelectedTargetSnapshot* selected_target = nullptr) {
    builder.metadata("source", std::string(magic_enum::enum_name(plan.source)))
        .metadata("aim_phase", std::string(magic_enum::enum_name(plan.aim_phase)))
        .metadata("selected_armor_id", std::to_string(plan.selected_armor_id))
        .metadata("rough_selected_armor_id", std::to_string(plan.rough_selected_armor_id))
        .metadata("fire_advice", yes_no(plan.fire_advice))
        .metadata("predicted_future_ns", std::to_string(plan.predicted_future));

    if (selected_target) {
        append_selected_target_metadata(builder, *selected_target);
    }

    if (plan.planner_seed) {
        builder.metadata("planner_seed.target_jumped", yes_no(plan.planner_seed->target_jumped))
            .metadata("planner_seed.tracker_last_armor_id", planner_seed_tracker_last_text(plan));
    }
}

} // namespace l4_detail

/// @brief Register L4 planning layer systems (gimbal, MPC)
///
/// This includes:
/// - foxglove_solver_target_pub: Publishes the current optimal target JSON
/// - foxglove_target_selection_pub: Publishes full choose-optimal trace JSON
/// - foxglove_l4_gimbal_cmd_pub: Publishes gimbal command JSON
/// - foxglove_l4_gimbal_scene: Publishes gimbal scene (selection, prediction, trajectory)
/// - foxglove_l4_mpc_trajectory_pub: Publishes MPC trajectory JSON
/// - foxglove_l4_mpc_prediction_scene: Publishes full MPC horizon visualization
void register_l4_planning_systems(talos::scheduler::Scheduler& app) {
    app.add_system<talos::pool_compute>(
        "foxglove_solver_target_pub",
        [](talos::spmc<::fcs::L4::SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic>
               selected_target_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, selected_target_in)) {
                return;
            }

            auto selected_target = selected_target_in.read();
            if (!selected_target) {
                return;
            }

            detail::publish_json_message<TargetMessage>(
                *server, l4_detail::json_solver_target(*selected_target));
        });

    app.add_system<talos::pool_compute>(
        "foxglove_target_selection_pub",
        [](talos::spmc<::fcs::L4::TargetSelectionTrace, TargetSelectionTraceChannelTopic> trace_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, trace_in)) {
                return;
            }

            auto trace = trace_in.read();
            if (!trace) {
                return;
            }

            detail::publish_json_message<TargetSelectionTraceMessage>(
                *server, l4_detail::json_target_selection_trace(*trace));
        });

    // =========================================================================
    // GIMBAL COMMAND PUBLISHER
    // =========================================================================

    app.add_system<talos::pool_compute>(
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
            const auto plan_snapshot = plan_in.read_current();
            const auto* plan =
                (plan_snapshot && plan_snapshot->timestamp_ns == cmd->plan_timestamp_ns)
                    ? std::addressof(*plan_snapshot)
                    : nullptr;

            nlohmann::json cmd_json;
            cmd_json["timestamp_ns"]      = cmd->timestamp_ns;
            cmd_json["plan_timestamp_ns"] = cmd->plan_timestamp_ns;
            cmd_json["timestamp"]         = nlohmann::json::object();
            cmd_json["timestamp"]["sec"]  = cmd->timestamp_ns / 1000000000L;
            cmd_json["timestamp"]["nsec"] = cmd->timestamp_ns % 1000000000L;
            cmd_json["pre_plan"]          = {
                {     "yaw",      cmd->plan_yaw},
                {   "pitch",    cmd->plan_pitch},
                {"distance", cmd->plan_distance},
            };
            cmd_json["post_plan"] = {
                {        "yaw",   cmd->yaw},
                {      "pitch", cmd->pitch},
                {"fire_advice",  cmd->fire},
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
                    {   "center_index",          cmd->viz_debug->center_index},
                    {"lookahead_index",       cmd->viz_debug->lookahead_index},
                    { "reference_size", cmd->viz_debug->reference_plan.size()},
                    { "optimized_size", cmd->viz_debug->optimized_plan.size()},
                };
            }
            if (plan) {
                cmd_json["l4_plan"] = l4_detail::json_gimbal_plan(*plan);
            }

            detail::publish_json_message<GimbalCmdMessage>(*server, cmd_json);
        });

    app.add_system<talos::pool_compute>(
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
            const auto plan_snapshot = plan_in.read_current();
            const auto* plan =
                (plan_snapshot && plan_snapshot->timestamp_ns == cmd->plan_timestamp_ns)
                    ? std::addressof(*plan_snapshot)
                    : nullptr;

            nlohmann::json traj_json;
            traj_json["timestamp_ns"]      = cmd->timestamp_ns;
            traj_json["plan_timestamp_ns"] = cmd->plan_timestamp_ns;
            traj_json["center_index"]      = cmd->viz_debug->center_index;
            traj_json["lookahead_index"]   = cmd->viz_debug->lookahead_index;
            traj_json["reference"]         = nlohmann::json::array();
            traj_json["optimized"]         = nlohmann::json::array();
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

    app.add_system<talos::pool_compute>(
        "foxglove_l4_gimbal_scene",
        [](talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::spmc<::fcs::L4::SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic>
               selected_target_in,
           talos::spmc<::fcs::L4::TargetSelectionTrace, TargetSelectionTraceChannelTopic> trace_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in))
                return;
            auto plan = plan_in.read();
            if (!plan || plan->distance <= 0)
                return;
            const auto selected_target_snapshot = selected_target_in.read_current();
            const auto* selected_target =
                (selected_target_snapshot
                 && selected_target_snapshot->timestamp_ns == plan->timestamp_ns)
                    ? std::addressof(*selected_target_snapshot)
                    : nullptr;
            const auto trace_snapshot = trace_in.read_current();
            const auto* trace =
                (trace_snapshot && trace_snapshot->timestamp_ns == plan->timestamp_ns)
                    ? std::addressof(*trace_snapshot)
                    : nullptr;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // if (const auto snapshot = l4_detail::future_armor_snapshot(*plan)) {
            //     // ── Target center ──────────────────────────────────────────
            //     entities.push_back(
            //         viz::EntityBuilder::create<fast_tf::odom>("l4", "target_center")
            //             .timestamp(plan->timestamp_ns)
            //             .position(snapshot->center)
            //             .size(tac::L3::ROBOT_CENTER_SIZE * 0.75)
            //             .color(tac::L4::MPC_PRESENT)
            //             .alpha(0.6)
            //             .sphere()
            //             .text_with_offset(
            //                 "Target Center", 0.0, 0.0, tac::L3::LABEL_OFFSET_Z,
            //                 tac::Text::SIZE_SMALL)
            //             .build());

            //     // ── Velocity arrows ────────────────────────────────────────
            //     entities.push_back(
            //         viz::patterns::velocity_arrows<fast_tf::odom>(
            //             snapshot->center, snapshot->linear_velocity, snapshot->angular_velocity,
            //             plan->timestamp_ns));

            //     // ── Future armor plates (clearly visible) ──────────────────
            //     std::vector<::foxglove::schemas::Point3> armor_link_points;
            //     const auto center_pt = viz::make_point3(snapshot->center);
            //     for (int i = 0; i < static_cast<int>(snapshot->armors.size()); ++i) {
            //         const auto& armor_pose          = snapshot->armors[static_cast<size_t>(i)];
            //         const Eigen::Vector3d armor_pos = armor_pose.head<3>();
            //         entities.push_back(
            //             viz::EntityBuilder::create<fast_tf::odom>(
            //                 "l4", fmt::format("future_armor_{}", i))
            //                 .timestamp(plan->timestamp_ns)
            //                 .position(armor_pos)
            //                 .orientation(l4_detail::armor_orientation(armor_pose[3]))
            //                 .color(tac::L4::FUTURE_ARMOR)
            //                 .cube(
            //                     tac::Geometry::ARMOR_THICKNESS,
            //                     tac::Geometry::ARMOR_HEIGHT_SMALL, tac::Geometry::ARMOR_WIDTH)
            //                 .text_with_offset(
            //                     std::to_string(i), 0.0, 0.0, tac::L3::LABEL_OFFSET_Z,
            //                     tac::Text::SIZE_SMALL)
            //                 .metadata("armor_id", std::to_string(i))
            //                 .build());
            //         // Collect points for connecting lines
            //         armor_link_points.push_back(center_pt);
            //         armor_link_points.push_back(viz::make_point3(armor_pos));
            //     }

            //     // ── Target center → armors connecting lines ────────────────
            //     if (armor_link_points.size() >= 4) {
            //         entities.push_back(
            //             viz::EntityBuilder::create<fast_tf::odom>("l4", "armor_links")
            //                 .timestamp(plan->timestamp_ns)
            //                 .line_list(
            //                     std::move(armor_link_points), tac::L4::SPATIAL_LINK,
            //                     tac::L4::SPATIAL_LINK_THICKNESS)
            //                 .build());
            //     }

            //     // ── Selection overlay (primary visual) ─────────────────────
            //     const auto add_overlay =
            //         [&](int armor_id, const std::string& entity_id, const std::string& label,
            //             const ::foxglove::schemas::Color& color, double size_scale) {
            //             if (armor_id < 0 || armor_id >=
            //             static_cast<int>(snapshot->armors.size())) {
            //                 return;
            //             }
            //             const Eigen::Vector3d armor_pos =
            //                 snapshot->armors[static_cast<size_t>(armor_id)].head<3>();
            //             entities.push_back(
            //                 viz::EntityBuilder::create<fast_tf::odom>("l4", entity_id)
            //                     .timestamp(plan->timestamp_ns)
            //                     .position(armor_pos)
            //                     .size(tac::L4::SELECTION_SIZE * size_scale)
            //                     .color(color)
            //                     .sphere()
            //                     .text_with_offset(
            //                         label, 0.0, 0.0, tac::L3::LABEL_OFFSET_Z * 1.6,
            //                         tac::Text::SIZE_SMALL)
            //                     .metadata("armor_id", std::to_string(armor_id))
            //                     .build());
            //         };

            //     if (plan->selected_armor_id == plan->rough_selected_armor_id) {
            //         add_overlay(
            //             plan->selected_armor_id, "selected_rough_armor",
            //             fmt::format("Selected {}", plan->selected_armor_id),
            //             plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L4::GIMBAL_AIM_HOLD,
            //             1.25);
            //     } else {
            //         add_overlay(
            //             plan->rough_selected_armor_id, "rough_armor",
            //             fmt::format("Rough {}", plan->rough_selected_armor_id),
            //             tac::L4::MPC_REFERENCE, 1.0);
            //         add_overlay(
            //             plan->selected_armor_id, "selected_armor",
            //             fmt::format("Selected {}", plan->selected_armor_id),
            //             plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L4::GIMBAL_AIM_HOLD,
            //             1.25);
            //     }
            // }

            // Predicted position marker (at bullet impact time for verification)
            if (plan->predicted_position) {
                const auto aim_color =
                    plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L4::GIMBAL_AIM_HOLD;

                auto builder = viz::EntityBuilder::create<fast_tf::odom>("l4", "predicted")
                                   .timestamp(plan->predicted_future)
                                   .position(*plan->predicted_position)
                                   .size(tac::L4::PREDICTION_SIZE)
                                   .color(aim_color)
                                   .sphere();

                l4_detail::append_plan_metadata(builder, *plan, selected_target);

                // builder.color(tac::Text::PRIMARY)
                //     .text_with_offset(
                //         plan->fire_advice ? "● FIRE" : "○ HOLD", 0.0, 0.0,
                //         tac::L3::LABEL_OFFSET_Z + tac::L4::PREDICTION_SIZE,
                //         tac::Text::SIZE_SMALL);

                entities.push_back(builder.build());
            }

            // ── Selection trace — visual hierarchy ─────────────────────────
            if (trace) {
                for (const auto& candidate : trace->candidates) {
                    if (!candidate.target_center) {
                        continue;
                    }

                    // Visual hierarchy: selected > runner_up > eliminated
                    const bool is_selected  = candidate.selected;
                    const bool is_runner_up = candidate.runner_up;
                    const bool show_label   = is_selected || is_runner_up;

                    // Size & alpha by tier
                    const double size_scale = is_selected ? 1.25 : is_runner_up ? 1.0 : 0.7;
                    const double alpha_val  = is_selected ? 0.95 : is_runner_up ? 0.75 : 0.25;

                    // Color by tier
                    const auto tier_color = is_selected
                                              ? (plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE
                                                                   : tac::L4::GIMBAL_AIM_HOLD)
                                          : is_runner_up ? tac::L4::CANDIDATE_RUNNER_UP
                                                         : tac::L4::CANDIDATE_ELIMINATED;

                    auto builder = viz::EntityBuilder::create<fast_tf::odom>(
                                       "l4", l4_detail::selection_trace_entity_id(candidate))
                                       .timestamp(trace->timestamp_ns)
                                       .position(*candidate.target_center)
                                       .size(tac::L4::SELECTION_SIZE * size_scale)
                                       .color(tier_color)
                                       .alpha(alpha_val)
                                       .sphere();

                    // Simplified label — identity + rank only, scores in metadata
                    if (show_label) {
                        builder.text_with_offset(
                            fmt::format(
                                "#{}{} {}", candidate.rank, is_selected ? "*" : "",
                                l4_detail::selection_identity(candidate)),
                            tac::L4::SELECTION_SIZE * 1.5, 0.0, tac::L3::LABEL_OFFSET_Z,
                            tac::Text::SIZE_SMALL);
                    }

                    l4_detail::append_selection_trace_metadata(builder, *trace, candidate);
                    entities.push_back(builder.build());
                }
            }
            publish_scene_if_nonempty<GimbalSceneMessage>(*server, std::move(entities));
        });

    app.add_system<talos::fixed_rate<250>>(
        "foxglove_l4_mpc_prediction_scene",
        [](talos::spmc<::fcs::L5::WeaponCommand, WeaponCommandChannelTopic> cmd_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server,
           core::trajectory::trajectory_solver solver, talos::res<fast_tf::CoordinateSystem> coord,
           core::trajectory::bullet_speed bullet_speed) {
            if (!detail::foxglove_ready(*server, cmd_in)) {
                return;
            }

            std::vector<::foxglove::schemas::SceneEntity> entities;
            auto cmd = cmd_in.read();
            if (cmd && cmd->viz_debug) {
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

                // ── Reference trajectory connecting line ─────────────────────
                if (cmd->viz_debug->reference_plan.size() >= 2) {
                    std::vector<::foxglove::schemas::Point3> ref_line_points;
                    ref_line_points.reserve(cmd->viz_debug->reference_plan.size());
                    for (const auto& sample : cmd->viz_debug->reference_plan) {
                        ref_line_points.push_back(viz::make_point3(sample.world_position));
                    }
                    entities.push_back(
                        viz::EntityBuilder::create<fast_tf::odom>("l4", "mpc_ref_line")
                            .timestamp(cmd->timestamp_ns)
                            .line_strip(
                                std::move(ref_line_points), tac::L4::MPC_REFERENCE,
                                tac::L4::TRAJECTORY_LINE_THICKNESS)
                            .build());
                }

                // ── Optimized trajectory connecting line ─────────────────────
                if (cmd->viz_debug->optimized_plan.size() >= 2) {
                    std::vector<::foxglove::schemas::Point3> opt_line_points;
                    opt_line_points.reserve(cmd->viz_debug->optimized_plan.size());
                    for (const auto& sample : cmd->viz_debug->optimized_plan) {
                        opt_line_points.push_back(viz::make_point3(sample.world_position));
                    }
                    entities.push_back(
                        viz::EntityBuilder::create<fast_tf::odom>("l4", "mpc_opt_line")
                            .timestamp(cmd->timestamp_ns)
                            .line_strip(
                                std::move(opt_line_points), tac::L4::MPC_PRESENT,
                                tac::L4::TRAJECTORY_LINE_THICKNESS)
                            .build());
                }

                const int center_index = std::clamp(
                    cmd->viz_debug->center_index, 0,
                    static_cast<int>(cmd->viz_debug->optimized_plan.size()) - 1);

                const auto add_key_marker =
                    [&](const std::string& entity_id, const ::fcs::L5::TrajectoryPlanSample& sample,
                        const std::string& label, const ::foxglove::schemas::Color& color,
                        double size) {
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
                    "mpc_ref_center",
                    cmd->viz_debug->reference_plan[static_cast<size_t>(center_index)],
                    l4_detail::key_point_label(
                        "Ref", cmd->viz_debug->reference_plan[static_cast<size_t>(center_index)]),
                    tac::L4::MPC_REFERENCE, tac::L4::SELECTION_SIZE);

                add_key_marker(
                    "mpc_opt_center",
                    cmd->viz_debug->optimized_plan[static_cast<size_t>(center_index)],
                    l4_detail::key_point_label(
                        "Opt", cmd->viz_debug->optimized_plan[static_cast<size_t>(center_index)]),
                    tac::L4::MPC_PRESENT, tac::L4::SELECTION_SIZE);
                auto trajectory = solver->get()->generate_trajectory(
                    cmd->pitch, bullet_speed->bullet_speed, cmd->distance);
                std::vector<Eigen::Vector3d> traj_points;
                traj_points.reserve(trajectory.size());
                for (const auto& [x, z] : trajectory) {
                    traj_points.emplace_back(
                        x * std::cos(cmd->pitch) + z * std::sin(cmd->pitch), 0,
                        -x * std::sin(cmd->pitch) + z * std::cos(cmd->pitch));
                }
                auto traj_entities =
                    viz::patterns::bullet_trajectory(traj_points, cmd->fire, cmd->timestamp_ns);
                entities.insert(
                    entities.end(), std::make_move_iterator(traj_entities.begin()),
                    std::make_move_iterator(traj_entities.end()));
            } else {

                auto f = fast_tf::lookup_clamped<fast_tf::odom, fast_tf::gimbal>(
                    *coord, clock::now_ns());
                if (f) {
                    auto [r, p, y] = f->euler_rot().rpy();
                    auto trajectory =
                        solver->get()->generate_trajectory(p, bullet_speed->bullet_speed, 15.0);
                    std::vector<Eigen::Vector3d> traj_points;
                    traj_points.reserve(trajectory.size());
                    for (const auto& [x, z] : trajectory) {
                        traj_points.emplace_back(
                            x * std::cos(p) + z * std::sin(p), 0,
                            -x * std::sin(p) + z * std::cos(p));
                    }
                    auto traj_entities =
                        viz::patterns::bullet_trajectory(traj_points, false, clock::now_ns());
                    entities.insert(
                        entities.end(), std::make_move_iterator(traj_entities.begin()),
                        std::make_move_iterator(traj_entities.end()));
                }
            }
            publish_scene_if_nonempty<MpcPredictionSceneMessage>(*server, std::move(entities));
        });
}
} // namespace fcs::visualization::foxglove::systems
