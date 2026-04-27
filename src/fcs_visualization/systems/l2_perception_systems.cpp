#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"

#include "L2_perception/ldm/ldm_config.hpp"
#include "L2_perception/ldm/ldm_geometry.hpp"
#include "L2_perception/ldm/types.hpp"
#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"

#include <magic_enum.hpp>
#include <nlohmann/json.hpp>

namespace fcs::visualization::foxglove::systems {

namespace {

constexpr uint64_t kLdmSceneLifetimeNs = 5'000'000;

[[nodiscard]] ::foxglove::schemas::Color
    ldm_scene_color(const fcs::L2::ldm::LdmMeasurement& measurement) noexcept {
    switch (measurement.depth_quality) {
    case fcs::L2::ldm::LdmDepthQuality::Stable: return tac::Semantic::STATUS_ACTIVE;
    case fcs::L2::ldm::LdmDepthQuality::Constrained: return tac::Semantic::ALERT;
    case fcs::L2::ldm::LdmDepthQuality::BearingOnly: return tac::L2::MEASUREMENT_CONFIDENCE;
    case fcs::L2::ldm::LdmDepthQuality::None: return tac::Semantic::INERT;
    }
    return tac::Semantic::INERT;
}

[[nodiscard]] std::string ldm_label(const fcs::L2::ldm::LdmMeasurement& measurement) {
    return fmt::format(
        "LDM {}/{} {}", measurement.selected_pair_count, measurement.pair_count_total,
        measurement.depth_quality);
}

[[nodiscard]] std::vector<::foxglove::schemas::Point3>
    ldm_top_outline_points(const fcs::L2::ldm::LdmDetectorConfig& config) {
    const auto outline = fcs::L2::ldm::volume_outline_points(config.geometry);
    std::vector<::foxglove::schemas::Point3> points;
    points.reserve(8);
    for (size_t i = 0; i < 8; ++i) {
        points.push_back(viz::make_point3(outline[i].x, outline[i].y, outline[i].z));
    }
    return points;
}

[[nodiscard]] std::vector<::foxglove::schemas::Point3>
    ldm_bottom_outline_points(const fcs::L2::ldm::LdmDetectorConfig& config) {
    const auto outline = fcs::L2::ldm::volume_outline_points(config.geometry);
    std::vector<::foxglove::schemas::Point3> points;
    points.reserve(8);
    for (size_t i = 8; i < 16; ++i) {
        points.push_back(viz::make_point3(outline[i].x, outline[i].y, outline[i].z));
    }
    return points;
}

[[nodiscard]] std::vector<::foxglove::schemas::Point3>
    ldm_vertical_outline_points(const fcs::L2::ldm::LdmDetectorConfig& config) {
    const auto outline = fcs::L2::ldm::volume_outline_points(config.geometry);
    std::vector<::foxglove::schemas::Point3> points;
    points.reserve(16);
    for (size_t i = 0; i < 8; ++i) {
        points.push_back(viz::make_point3(outline[i].x, outline[i].y, outline[i].z));
        points.push_back(viz::make_point3(outline[i + 8].x, outline[i + 8].y, outline[i + 8].z));
    }
    return points;
}

[[nodiscard]] ::foxglove::schemas::SceneEntity make_ldm_volume_entity_odom(
    const fcs::L2::ldm::LdmMeasurement& measurement,
    const fcs::L2::ldm::LdmDetectorConfig& config) {
    const auto color = ldm_scene_color(measurement);
    const auto& pose = *measurement.transform_odom;

    auto builder = viz::EntityBuilder::create<fast_tf::odom>("l2", "ldm");
    builder.timestamp(measurement.timestamp_ns)
        .lifetime(kLdmSceneLifetimeNs)
        .position(pose.translation())
        .orientation(pose.quaternion());
    builder.color(color)
        .line_loop(ldm_top_outline_points(config), color, 0.01)
        .line_loop(ldm_bottom_outline_points(config), color, 0.01)
        .line_list(ldm_vertical_outline_points(config), color, 0.01)
        .size(0.04)
        .sphere()
        .text_with_offset(
            ldm_label(measurement), 0.0, 0.0, config.geometry.volume_height_m * 0.8,
            tac::L2::LABEL_FONT_SIZE)
        .metadata("confidence", fmt::format("{:.3f}", measurement.confidence))
        .metadata("depth_quality", std::string(magic_enum::enum_name(measurement.depth_quality)))
        .metadata("pair_count_total", std::to_string(measurement.pair_count_total))
        .metadata("selected_pair_count", std::to_string(measurement.selected_pair_count));
    return builder.build();
}

[[nodiscard]] ::foxglove::schemas::SceneEntity make_ldm_volume_entity_camera(
    const fcs::L2::ldm::LdmMeasurement& measurement,
    const fcs::L2::ldm::LdmDetectorConfig& config) {
    const auto color = ldm_scene_color(measurement);
    const auto& pose = *measurement.transform_cam;

    auto builder = viz::EntityBuilder::create<fast_tf::camera_optical>("l2", "ldm_cam");
    builder.timestamp(measurement.timestamp_ns)
        .lifetime(kLdmSceneLifetimeNs)
        .position(pose.translation())
        .orientation(pose.quaternion());
    builder.color(color)
        .line_loop(ldm_top_outline_points(config), color, 0.01)
        .line_loop(ldm_bottom_outline_points(config), color, 0.01)
        .line_list(ldm_vertical_outline_points(config), color, 0.01)
        .size(0.04)
        .sphere()
        .text_with_offset(
            ldm_label(measurement), 0.0, 0.0, config.geometry.volume_height_m * 0.8,
            tac::L2::LABEL_FONT_SIZE)
        .metadata("confidence", fmt::format("{:.3f}", measurement.confidence))
        .metadata("depth_quality", std::string(magic_enum::enum_name(measurement.depth_quality)))
        .metadata("pair_count_total", std::to_string(measurement.pair_count_total))
        .metadata("selected_pair_count", std::to_string(measurement.selected_pair_count));
    return builder.build();
}

[[nodiscard]] ::foxglove::schemas::SceneEntity
    make_ldm_bearing_entity(const fcs::L2::ldm::LdmMeasurement& measurement) {
    const auto color          = ldm_scene_color(measurement);
    const Eigen::Vector3d end = measurement.bearing_cam.normalized() * 2.0;

    return viz::EntityBuilder::create<fast_tf::camera_optical>("l2", "ldm_bearing")
        .timestamp(measurement.timestamp_ns)
        .lifetime(kLdmSceneLifetimeNs)
        .line_strip({viz::make_point3(0.0, 0.0, 0.0), viz::make_point3(end)}, color, 0.01)
        .position(end)
        .color(color)
        .size(0.04)
        .sphere()
        .text_with_offset(ldm_label(measurement), 0.0, 0.0, 0.08, tac::L2::LABEL_FONT_SIZE)
        .metadata("confidence", fmt::format("{:.3f}", measurement.confidence))
        .metadata("depth_quality", std::string(magic_enum::enum_name(measurement.depth_quality)))
        .metadata("pair_count_total", std::to_string(measurement.pair_count_total))
        .metadata("selected_pair_count", std::to_string(measurement.selected_pair_count))
        .build();
}

} // namespace

/// @brief Register L2 perception layer systems (detection, measurement)
///
/// This includes:
/// - foxglove_l2_detection_pub: Publishes armor detection JSON data
/// - foxglove_l2_measurement_scene: Publishes measurement scene visualization
/// - foxglove_l2_measurement_pub: Publishes measurement JSON data
void register_l2_perception_systems(talos::scheduler::Scheduler& app) {

    // =========================================================================
    // DETECTION PUBLISHER
    // =========================================================================

    app.add_system<talos::pool_compute>(
        "foxglove_l2_detection_pub",
        [](talos::spmc<ArmorDetectionBatch, DetectionChannelTopic> det_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!foxglove_ready(*server, det_in)) {
                return;
            }

            auto batch = det_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json armors_json;
            armors_json["timestamp_ns"]      = batch->timestamp_ns;
            armors_json["timestamp"]         = nlohmann::json::object();
            armors_json["timestamp"]["sec"]  = batch->timestamp_ns / 1000000000L;
            armors_json["timestamp"]["nsec"] = batch->timestamp_ns % 1000000000L;
            armors_json["armors"]            = nlohmann::json::array();

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

    app.add_system<talos::pool_compute>(
        "foxglove_l2_measurement_scene",
        [](talos::spmc<ArmorMeasurementBatch, MeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!foxglove_ready(*server, meas_in))
                return;

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            std::vector<::foxglove::schemas::SceneEntity> entities;

            for (size_t i = 0; i < batch->measurements.size(); ++i) {
                const auto& m = batch->measurements[i];
                const auto t  = m.transform.translation();
                const auto q  = m.transform.quaternion();

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

                entities.push_back(std::move(entity));
            }

            // Publish to semantic channel
            publish_scene_if_nonempty<SceneMessage>(*server, std::move(entities));
        });

    app.add_system<talos::pool_compute>(
        "foxglove_l2_ldm_scene",
        [](talos::spmc<fcs::L2::ldm::LdmMeasurement, LdmMeasurementChannelTopic> ldm_meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server,
           talos::res<fcs::L2::ldm::LdmDetectorConfig> ldm_config) {
            if (!foxglove_ready(*server, ldm_meas_in)) {
                return;
            }

            auto measurement = ldm_meas_in.read();
            if (!measurement) {
                return;
            }

            std::vector<::foxglove::schemas::SceneEntity> entities;
            if (measurement->transform_odom.has_value()) {
                entities.push_back(make_ldm_volume_entity_odom(*measurement, *ldm_config));
            } else if (measurement->transform_cam.has_value()) {
                entities.push_back(make_ldm_volume_entity_camera(*measurement, *ldm_config));
            } else {
                entities.push_back(make_ldm_bearing_entity(*measurement));
            }

            publish_scene_if_nonempty<SceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // MEASUREMENT JSON PUBLISHER
    // =========================================================================

    app.add_system<talos::pool_compute>(
        "foxglove_l2_measurement_pub",
        [](talos::spmc<ArmorMeasurementBatch, MeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, meas_in)) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json j;
            j["timestamp_ns"]      = batch->timestamp_ns;
            j["timestamp"]         = nlohmann::json::object();
            j["timestamp"]["sec"]  = batch->timestamp_ns / 1000000000L;
            j["timestamp"]["nsec"] = batch->timestamp_ns % 1000000000L;
            j["measurements"]      = nlohmann::json::array();

            for (const auto& m : batch->measurements) {
                const auto t = m.transform.translation();
                const auto q = m.transform.quaternion();
                nlohmann::json mj;
                mj["timestamp_ns"]      = m.timestamp_ns;
                mj["timestamp"]         = nlohmann::json::object();
                mj["timestamp"]["sec"]  = m.timestamp_ns / 1000000000L;
                mj["timestamp"]["nsec"] = m.timestamp_ns % 1000000000L;
                mj["name"]              = std::string(magic_enum::enum_name(m.name));
                mj["type"]              = std::string(magic_enum::enum_name(m.type));
                mj["color"]             = std::string(magic_enum::enum_name(m.color));
                mj["confidence"]        = m.confidence;
                mj["position"]          = {t.x(), t.y(), t.z()};
                mj["orientation"]       = {q.x(), q.y(), q.z(), q.w()};
                j["measurements"].push_back(mj);
            }

            detail::publish_json_message<MeasurementMessage>(*server, j);
        });
}

} // namespace fcs::visualization::foxglove::systems
