#pragma once

#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "system_helpers.hpp"

#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"

#include <magic_enum.hpp>
#include <nlohmann/json.hpp>

namespace fcs::foxglove::systems {

/// @brief Register L2 perception layer systems (detection, measurement)
///
/// This includes:
/// - foxglove_l2_detection_pub: Publishes armor detection JSON data
/// - foxglove_l2_measurement_scene: Publishes measurement scene visualization
/// - foxglove_l2_measurement_pub: Publishes measurement JSON data
template <typename App>
void register_l2_perception_systems(App& app) {

    // =========================================================================
    // DETECTION PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l2_detection_pub",
        [](talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTopic> det_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, det_in)) {
                return;
            }

            auto batch = det_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json armors_json;
            armors_json["timestamp_ns"] = batch->timestamp_ns;
            armors_json["armors"]       = nlohmann::json::array();

            for (size_t i = 0; i < batch->detections.size(); ++i) {
                const auto& det = batch->detections[i];
                nlohmann::json obj;
                obj["center"]["x"] = det.center().x;
                obj["center"]["y"] = det.center().y;

                obj["corners"] = nlohmann::json::array();
                for (const auto& pt : det.corners) {
                    obj["corners"].push_back({pt.x, pt.y});
                }
                obj["confidence"] = det.confidence;
                obj["color"]      = magic_enum::enum_name(det.color);
                obj["type"]       = magic_enum::enum_name(det.type);
                obj["name"]       = magic_enum::enum_name(det.name);
                armors_json["armors"].push_back(obj);
            }

            detail::publish_json_message<DebugArmorsMessage>(*server, armors_json);
        });

    // =========================================================================
    // MEASUREMENT VISUALIZATION (L2)
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l2_measurement_scene",
        [](talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, meas_in))
                return;

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            std::vector<::foxglove::schemas::SceneEntity> entities;

            for (size_t i = 0; i < batch->measurements.size(); ++i) {
                const auto& m = batch->measurements[i];
                const auto t  = m.transform->translation();
                const auto q  = m.transform->quaternion();

                // Use convenience builder - semantic layer organization
                auto entity =
                    viz::EntityBuilder::create<fast_tf::odom>("l2", fmt::format("armor_{}", i))
                        .position(t.x(), t.y(), t.z())
                        .orientation(q.x(), q.y(), q.z(), q.w())
                        .cube(
                            tac::Geometry::ARMOR_THICKNESS, tac::Geometry::ARMOR_HEIGHT_SMALL,
                            tac::Geometry::ARMOR_WIDTH)
                        .color(tac::L2::MEASUREMENT_GHOST)
                        .text_with_offset(
                            fmt::format(
                                "{} {}", magic_enum::enum_name(m.name),
                                magic_enum::enum_name(m.type)),
                            0, -tac::Text::SIZE_MEDIUM, 0, tac::L2::LABEL_FONT_SIZE)
                        .metadata("confidence", fmt::format("{:.3f}", m.confidence))
                        .metadata("color", std::string(magic_enum::enum_name(m.color)))
                        .timestamp(m.timestamp_ns)
                        .build();

                // Also add a sphere at same position
                auto sphere =
                    viz::EntityBuilder::create<fast_tf::odom>("l2", fmt::format("sphere_{}", i))
                        .position(t.x(), t.y(), t.z())
                        .size(tac::L2::ARMOR_SIZE)
                        .color(tac::L2::MEASUREMENT_GHOST)
                        .sphere()
                        .timestamp(m.timestamp_ns)
                        .build();

                entities.push_back(std::move(entity));
                entities.push_back(std::move(sphere));
            }

            // Publish to semantic channel
            viz::publish_scene_if_nonempty<SceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // MEASUREMENT JSON PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l2_measurement_pub",
        [](talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, meas_in)) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json j;
            j["timestamp_ns"] = batch->timestamp_ns;
            j["measurements"] = nlohmann::json::array();

            for (const auto& m : batch->measurements) {
                const auto t = m.transform->translation();
                const auto q = m.transform->quaternion();
                nlohmann::json mj;
                mj["timestamp_ns"] = m.timestamp_ns;
                mj["name"]         = std::string(magic_enum::enum_name(m.name));
                mj["type"]         = std::string(magic_enum::enum_name(m.type));
                mj["color"]        = std::string(magic_enum::enum_name(m.color));
                mj["confidence"]   = m.confidence;
                mj["position"]     = {t.x(), t.y(), t.z()};
                mj["orientation"]  = {q.x(), q.y(), q.z(), q.w()};
                j["measurements"].push_back(mj);
            }

            detail::publish_json_message<MeasurementMessage>(*server, j);
        });
}

} // namespace fcs::foxglove::systems
