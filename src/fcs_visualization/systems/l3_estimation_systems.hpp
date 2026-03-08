#pragma once

#include "L3_estimation/tracker/vis_helpers.hpp"
#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "system_helpers.hpp"

#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"

#include <Eigen/Core>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <optional>

namespace fcs::foxglove::systems {

/// @brief Register L3 estimation layer systems (tracking, association)
///
/// This includes:
/// - foxglove_l3_tracker_scene: Publishes tracker scene visualization
/// - foxglove_l3_tracker_pub: Publishes tracker JSON data
/// - foxglove_l3_association_scene: Publishes association scene (predicted vs measured)
template <typename App>
void register_l3_estimation_systems(App& app) {

    using namespace L3::vis;

    // =========================================================================
    // TRACKER VISUALIZATION (L3)
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l3_tracker_scene",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;
            auto output = tracker_in.read();
            if (!output || !output->is_tracking())
                return;

            const auto pos   = get_tracker_position(*output);
            const auto vel   = get_tracker_velocity(*output);
            const auto v_yaw = get_tracker_v_yaw(*output);

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Main target sphere (status-colored)
            entities.push_back(
                viz::patterns::tracker_target<fast_tf::odom>(
                    Eigen::Vector3d(pos.x, pos.y, pos.z), static_cast<int>(output->status),
                    output->timestamp_ns));

            // Velocity arrows
            entities.push_back(
                viz::patterns::velocity_arrows<fast_tf::odom>(
                    Eigen::Vector3d(pos.x, pos.y, pos.z), vel, v_yaw, output->timestamp_ns));

            // Armor cubes
            if (output->is_robot()) {
                ::foxglove::schemas::SceneEntity armor_entity =
                    viz::EntityBuilder::create<fast_tf::odom>("l3", "robot_armors")
                        .timestamp(output->timestamp_ns)
                        .build();

                add_robot_armor_cubes(armor_entity, *output->robot_state(), output->target_name);
                entities.push_back(std::move(armor_entity));
            } else if (output->is_outpost()) {
                ::foxglove::schemas::SceneEntity armor_entity =
                    viz::EntityBuilder::create<fast_tf::odom>("l3", "outpost_armors")
                        .timestamp(output->timestamp_ns)
                        .build();

                add_outpost_armor_cubes(armor_entity, *output->outpost_state());
                entities.push_back(std::move(armor_entity));
            }

            viz::publish_scene_if_nonempty<TrackSceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // TRACKER JSON PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l3_tracker_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output) {
                return;
            }

            // Publish target state JSON
            nlohmann::json target_json;
            target_json["timestamp_ns"]      = output->timestamp_ns;
            target_json["timestamp"]         = nlohmann::json::object();
            target_json["timestamp"]["sec"]  = output->timestamp_ns / 1000000000L;
            target_json["timestamp"]["nsec"] = output->timestamp_ns % 1000000000L;
            target_json["status"]            = magic_enum::enum_name(output->status);
            target_json["tracking"]          = output->is_tracking();

            std::string type                        = "unknown";
            std::optional<Eigen::Vector3d> position = std::nullopt, velocity = std::nullopt;
            std::optional<double> yaw = std::nullopt, v_yaw = std::nullopt;

            if (output->is_robot()) {
                const auto& state     = *output->robot_state();
                type                  = "robot";
                position              = state.position;
                velocity              = state.velocity;
                yaw                   = state.yaw;
                v_yaw                 = state.v_yaw;
                target_json["radius"] = {state.radius0, state.radius1};
                target_json["z"]      = state.z1;
            } else if (output->is_outpost()) {
                const auto& state = *output->outpost_state();
                type              = "outpost";
                position          = {
                    state.position.x(), state.position.y(),
                    (state.z[0] + state.z[1] + state.z[2]) / 3.0};
                velocity         = state.velocity;
                yaw              = state.yaw;
                v_yaw            = state.v_yaw;
                target_json["z"] = {state.z[0], state.z[1], state.z[2]};
            }
            target_json["type"] = type;
            if (velocity) {
                target_json["velocity"]      = {velocity->x(), velocity->y(), velocity->z()};
                target_json["velocity_norm"] = velocity->norm();
            }
            if (position) {
                target_json["position"] = {position->x(), position->y(), position->z()};
                target_json["distance"] = position->norm();
            }
            if (yaw) {
                target_json["yaw"] = yaw;
            }
            if (v_yaw) {
                target_json["v_yaw"] = v_yaw;
            }

            detail::publish_json_message<TargetMessage>(*server, target_json);
        });

    // =========================================================================
    // ASSOCIATION VISUALIZATION (L3) - Predicted vs Measured
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l3_association_scene",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;

            auto output = tracker_in.read();
            if (!output || !output->is_tracking())
                return;

            auto batch = meas_in.read_current();
            if (!batch)
                return;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Predicted armors (present) - neon cyan
            if (output->is_robot()) {
                const auto& state = *output->robot_state();
                const auto armors = state.armor_poses();
                for (int i = 0; i < 4; i++) {
                    entities.push_back(
                        viz::patterns::predicted_armor<fast_tf::odom>(
                            Eigen::Vector3d(armors[i][0], armors[i][1], armors[i][2]), i,
                            output->timestamp_ns));
                }
            } else if (output->is_outpost()) {
                const auto& state           = *output->outpost_state();
                constexpr double kAngleStep = 2.0 * std::numbers::pi / 3.0;
                for (int i = 0; i < 3; i++) {
                    const double armor_yaw = state.yaw + i * kAngleStep;
                    const auto& pos        = state.position;
                    constexpr double r     = L3::OutpostTargetState::radius;
                    const Eigen::Vector3d armor_pos{
                        pos.x() + r * std::cos(armor_yaw), pos.y() + r * std::sin(armor_yaw),
                        state.z[i]};

                    entities.push_back(
                        viz::EntityBuilder::create<fast_tf::odom>("l3", fmt::format("pred_{}", i))
                            .position(armor_pos)
                            .size(tac::L3::ARMOR_PLATE_SIZE)
                            .color(tac::L3::TRACKING_LOCKED)
                            .sphere()
                            .text_with_offset(
                                fmt::format("pred_{}", i), 0, 0, tac::L3::LABEL_OFFSET_Z)
                            .timestamp(output->timestamp_ns)
                            .build());
                }
            }

            // Measurements (past) - ghost silver
            for (size_t i = 0; i < batch->measurements.size(); ++i) {
                const auto& m = batch->measurements[i];
                const auto t  = m.transform->translation();
                entities.push_back(
                    viz::patterns::measurement_sphere<fast_tf::odom>(
                        Eigen::Vector3d(t.x(), t.y(), t.z()), m.confidence, m.timestamp_ns));
            }

            viz::publish_scene_if_nonempty<AssociationSceneMessage>(*server, std::move(entities));
        });
}

} // namespace fcs::foxglove::systems
