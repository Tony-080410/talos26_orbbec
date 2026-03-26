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
#include <cmath>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

namespace fcs::foxglove::systems {

[[nodiscard]] inline std::string tracker_image_center_text(double value) {
    return std::isfinite(value) ? fmt::format("{:.1f}", value) : "n/a";
}

[[nodiscard]] inline std::string tracker_label(const ::fcs::L3::TrackerOutput& output) {
    return fmt::format(
        "{}/{}\n{}\nOpt {} px", magic_enum::enum_name(output.target_color),
        magic_enum::enum_name(output.target_name), magic_enum::enum_name(output.status),
        tracker_image_center_text(output.last_image_center_distance_px));
}

/// @brief Register L3 estimation layer systems (tracking, association)
///
/// This includes:
/// - foxglove_l3_tracker_scene: Publishes tracker scene visualization
/// - foxglove_l3_association_scene: Publishes association scene (predicted vs measured)
template <typename App>
void register_l3_estimation_systems(App& app) {

    using namespace L3::vis;

    // =========================================================================
    // TRACKER VISUALIZATION (L3)
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l3_tracker_scene",
        [](talos::spmc<std::vector<::fcs::L3::TrackerOutput>, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;
            auto outputs = tracker_in.read();
            if (!outputs || outputs->empty())
                return;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Visualize all tracking targets
            for (const auto& output : *outputs) {
                if (!output.is_tracking())
                    continue;

                const std::string target_key =
                    fmt::format("{}_{}", output.target_color, output.target_name);
                const auto pos   = get_tracker_position(output);
                const auto vel   = get_tracker_velocity(output);
                const auto v_yaw = get_tracker_v_yaw(output);

                // Main target sphere (status-colored)
                entities.push_back(
                    viz::patterns::tracker_target<fast_tf::odom>(
                        Eigen::Vector3d(pos.x, pos.y, pos.z), static_cast<int>(output.status),
                        output.timestamp_ns, fmt::format("target_{}", target_key)));

                entities.push_back(
                    viz::EntityBuilder::create<fast_tf::odom>(
                        "l3",
                        fmt::format("tracker_info_{}_{}", output.target_color, output.target_name))
                        .timestamp(output.timestamp_ns)
                        .position(pos)
                        .color(tac::Text::PRIMARY)
                        .text_with_offset(
                            tracker_label(output), tac::L3::TARGET_SIZE * 1.4, 0.0,
                            tac::L3::LABEL_OFFSET_Z, tac::Text::SIZE_SMALL)
                        .metadata(
                            "target_name", std::string(magic_enum::enum_name(output.target_name)))
                        .metadata(
                            "target_color", std::string(magic_enum::enum_name(output.target_color)))
                        .metadata("status", std::string(magic_enum::enum_name(output.status)))
                        .metadata(
                            "last_image_center_distance_px",
                            tracker_image_center_text(output.last_image_center_distance_px))
                        .metadata(
                            "last_observation_timestamp_ns",
                            std::to_string(output.last_observation_timestamp_ns))
                        .build());

                // Velocity arrows
                entities.push_back(
                    viz::patterns::velocity_arrows<fast_tf::odom>(
                        Eigen::Vector3d(pos.x, pos.y, pos.z), vel, v_yaw, output.timestamp_ns,
                        fmt::format("velocity_{}", target_key)));

                // Armor cubes
                if (output.is_robot()) {
                    ::foxglove::schemas::SceneEntity armor_entity =
                        viz::EntityBuilder::create<fast_tf::odom>(
                            "l3",
                            fmt::format(
                                "robot_armors_{}_{}", output.target_color, output.target_name))
                            .timestamp(output.timestamp_ns)
                            .build();

                    add_robot_armor_cubes(armor_entity, *output.robot_state(), output.target_name);
                    entities.push_back(std::move(armor_entity));
                } else if (output.is_outpost()) {
                    ::foxglove::schemas::SceneEntity armor_entity =
                        viz::EntityBuilder::create<fast_tf::odom>(
                            "l3", fmt::format("outpost_armors_{}", output.target_color))
                            .timestamp(output.timestamp_ns)
                            .build();

                    add_outpost_armor_cubes(armor_entity, *output.outpost_state());
                    entities.push_back(std::move(armor_entity));
                }
            }

            viz::publish_scene_if_nonempty<TrackSceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // ASSOCIATION VISUALIZATION (L3) - Predicted vs Measured
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l3_association_scene",
        [](talos::spmc<std::vector<::fcs::L3::TrackerOutput>, TrackerOutputChannelTopic> tracker_in,
           talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;

            auto outputs = tracker_in.read();
            if (!outputs || outputs->empty())
                return;

            auto batch = meas_in.read_current();
            if (!batch)
                return;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Predicted armors (present) - neon cyan for all tracking targets
            for (const auto& output : *outputs) {
                if (!output.is_tracking())
                    continue;

                const std::string target_key =
                    fmt::format("{}_{}", output.target_color, output.target_name);
                if (output.is_robot()) {
                    const auto& state = *output.robot_state();
                    const auto armors = state.armor_poses();
                    for (int i = 0; i < 4; i++) {
                        entities.push_back(
                            viz::patterns::predicted_armor<fast_tf::odom>(
                                Eigen::Vector3d(armors[i][0], armors[i][1], armors[i][2]), i,
                                output.timestamp_ns, fmt::format("pred_{}_{}", target_key, i)));
                    }
                } else if (output.is_outpost()) {
                    const auto& state           = *output.outpost_state();
                    constexpr double kAngleStep = 2.0 * std::numbers::pi / 3.0;
                    for (int i = 0; i < 3; i++) {
                        const double armor_yaw = state.yaw + i * kAngleStep;
                        const auto& pos        = state.position;
                        constexpr double r     = L3::OutpostTargetState::radius;
                        const Eigen::Vector3d armor_pos{
                            pos.x() + r * std::cos(armor_yaw), pos.y() + r * std::sin(armor_yaw),
                            state.z[i]};

                        entities.push_back(
                            viz::EntityBuilder::create<fast_tf::odom>(
                                "l3", fmt::format("pred_{}_{}", target_key, i))
                                .position(armor_pos)
                                .size(tac::L3::ARMOR_PLATE_SIZE)
                                .color(tac::L3::TRACKING_LOCKED)
                                .sphere()
                                .text_with_offset(
                                    fmt::format("{}:{}", target_key, i), 0, 0,
                                    tac::L3::LABEL_OFFSET_Z)
                                .timestamp(output.timestamp_ns)
                                .build());
                    }
                }
            }

            // Measurements (past) - ghost silver
            for (const auto& m : batch->measurements) {
                const auto t = m.transform.translation();
                entities.push_back(
                    viz::patterns::measurement_sphere<fast_tf::odom>(
                        Eigen::Vector3d(t.x(), t.y(), t.z()), m.confidence, m.timestamp_ns));
            }

            viz::publish_scene_if_nonempty<AssociationSceneMessage>(*server, std::move(entities));
        });
}

} // namespace fcs::foxglove::systems
