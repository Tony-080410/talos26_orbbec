#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"

#include "L2_perception/ldm/ldm_config.hpp"
#include "L2_perception/ldm/ldm_geometry.hpp"
#include "L2_perception/ldm/types.hpp"
#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"

#include <Eigen/Geometry>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <numbers>

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

// ============================================================================
// Face visibility from camera (same method as 2D overlay)
// ============================================================================

/// @brief Determine which of the 8 octagon faces are visible from the camera.
///
/// Uses the same face-outward-normal-dot-product logic as the 2D projected
/// overlay in l1_sensor_systems.cpp. `cam_to_ldm` is the camera‑to‑LDM
/// transform (i.e. `transform_cam` from the measurement).
template <typename Transform>
[[nodiscard]] std::array<bool, 8> ldm_volume_visible_faces(
    const Transform& cam_to_ldm, const fcs::L2::ldm::LdmGeometryConfig& geometry) noexcept {
    // Camera position in LDM local frame
    const Eigen::Vector3d cam_local =
        cam_to_ldm.rotation().transpose() * (-cam_to_ldm.translation());

    const double R                = geometry.octagon_circumradius_m;
    constexpr double kAngleOffset = std::numbers::pi_v<double> / 8.0;
    constexpr double kStep        = std::numbers::pi_v<double> / 4.0;
    const double inradius         = R * std::cos(kAngleOffset);

    std::array<bool, 8> visible{};
    for (size_t i = 0; i < 8; ++i) {
        const double angle = kAngleOffset + static_cast<double>(i) * kStep;
        const Eigen::Vector3d local_normal(std::sin(angle), 0.0, -std::cos(angle));
        const Eigen::Vector3d local_face_center = inradius * local_normal;

        const Eigen::Vector3d dir_to_cam = cam_local - local_face_center;
        const double dist                = dir_to_cam.norm();
        if (!std::isfinite(dist) || dist <= 1e-9) {
            visible[i] = false;
            continue;
        }
        visible[i] = local_normal.dot(dir_to_cam / dist) > 1e-6;
    }
    return visible;
}

// ============================================================================
// Dashed 3D line simulation (Foxglove has no native dashed line support)
// ============================================================================

/// @brief Add a dashed edge from p1 to p2 as individual line_strip primitives.
inline void add_ldm_dashed_edge(
    viz::EntityBuilder& builder, const ::foxglove::schemas::Point3& p1,
    const ::foxglove::schemas::Point3& p2, const ::foxglove::schemas::Color& color,
    double thickness, double dash_len, double gap_len) {
    const double dx  = p2.x - p1.x;
    const double dy  = p2.y - p1.y;
    const double dz  = p2.z - p1.z;
    const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!std::isfinite(len) || len <= 1e-6) {
        return;
    }

    const double step  = dash_len + gap_len;
    const int num_segs = std::max(1, static_cast<int>(len / step));

    for (int s = 0; s < num_segs; ++s) {
        const double t0 = static_cast<double>(s) * step / len;
        const double t1 = std::min(1.0, (static_cast<double>(s) * step + dash_len) / len);
        if (t0 >= 1.0) {
            break;
        }
        builder.line_strip(
            {viz::make_point3(p1.x + dx * t0, p1.y + dy * t0, p1.z + dz * t0),
             viz::make_point3(p1.x + dx * t1, p1.y + dy * t1, p1.z + dz * t1)},
            color, thickness);
    }
}

// ============================================================================
// Add LDM volume outline edges to a builder with visibility-based styling
// ============================================================================

/// @brief Add all LDM volume edges (top ring, bottom ring, vertical) to the
///        given entity builder. Visible edges are drawn as solid line_strip
///        primitives; invisible (back‑facing) edges are drawn as dashed
///        line_strip primitives so the user can distinguish them at a glance.
inline void add_ldm_volume_edges(
    viz::EntityBuilder& builder, const fcs::L2::ldm::LdmDetectorConfig& config,
    const std::array<bool, 8>& visible_faces, const ::foxglove::schemas::Color& solid_color,
    const ::foxglove::schemas::Color& dashed_color, double thickness) {
    const auto outline = fcs::L2::ldm::volume_outline_points(config.geometry);

    // Convert cv::Point3f → foxglove::Point3
    std::array<::foxglove::schemas::Point3, 16> pts;
    for (size_t i = 0; i < 16; ++i) {
        pts[i] = viz::make_point3(
            static_cast<double>(outline[i].x), static_cast<double>(outline[i].y),
            static_cast<double>(outline[i].z));
    }

    constexpr double kDashLen = 0.020;
    constexpr double kGapLen  = 0.012;

    for (size_t i = 0; i < 8; ++i) {
        const size_t next = (i + 1) % 8;

        // --- Top ring edge: i → next (visible if face `next` is visible) ---
        if (visible_faces[next]) {
            builder.line_strip({pts[i], pts[next]}, solid_color, thickness);
        } else {
            add_ldm_dashed_edge(
                builder, pts[i], pts[next], dashed_color, thickness, kDashLen, kGapLen);
        }

        // --- Bottom ring edge: i+8 → next+8 (same visibility) ---
        if (visible_faces[next]) {
            builder.line_strip({pts[i + 8], pts[next + 8]}, solid_color, thickness);
        } else {
            add_ldm_dashed_edge(
                builder, pts[i + 8], pts[next + 8], dashed_color, thickness, kDashLen, kGapLen);
        }

        // --- Vertical edge: i ↔ i+8 (visible if either adjacent face is visible) ---
        const bool vert_visible = visible_faces[i] || visible_faces[next];
        if (vert_visible) {
            builder.line_strip({pts[i], pts[i + 8]}, solid_color, thickness);
        } else {
            add_ldm_dashed_edge(
                builder, pts[i], pts[i + 8], dashed_color, thickness, kDashLen, kGapLen);
        }
    }
}

// ============================================================================
// Entity builders
// ============================================================================

[[nodiscard]] ::foxglove::schemas::SceneEntity make_ldm_volume_entity_odom(
    const fcs::L2::ldm::LdmMeasurement& measurement,
    const fcs::L2::ldm::LdmDetectorConfig& config) {
    const auto color = ldm_scene_color(measurement);
    const auto& pose = *measurement.transform_odom;

    // Determine face visibility from the camera-to-LDM transform
    const auto visible_faces =
        measurement.transform_cam.has_value()
            ? ldm_volume_visible_faces(*measurement.transform_cam, config.geometry)
            : std::array<bool, 8>{true, true, true, true, true, true, true, true};

    auto builder = viz::EntityBuilder::create<fast_tf::odom>("l2", "ldm");
    builder.timestamp(measurement.timestamp_ns)
        .lifetime(kLdmSceneLifetimeNs)
        .position(pose.translation())
        .orientation(pose.quaternion())
        .color(color);

    // Dimmed colour for invisible (dashed) edges
    auto dimmed_color = color;
    dimmed_color.r *= 0.4f;
    dimmed_color.g *= 0.4f;
    dimmed_color.b *= 0.4f;

    // Draw edges: solid for visible, dashed for invisible
    add_ldm_volume_edges(builder, config, visible_faces, color, dimmed_color, 0.0025);

    builder.size(0.025)
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

    const auto visible_faces =
        ldm_volume_visible_faces(*measurement.transform_cam, config.geometry);
    auto builder = viz::EntityBuilder::create<fast_tf::camera_optical>("l2", "ldm_cam");
    builder.timestamp(measurement.timestamp_ns)
        .lifetime(kLdmSceneLifetimeNs)
        .position(pose.translation())
        .orientation(pose.quaternion())
        .color(color);

    // Dimmed colour for invisible (dashed) edges
    auto dimmed_color = color;
    dimmed_color.r *= 0.4f;
    dimmed_color.g *= 0.4f;
    dimmed_color.b *= 0.4f;

    // Draw edges: solid for visible, dashed for invisible
    add_ldm_volume_edges(builder, config, visible_faces, color, dimmed_color, 0.0025);

    builder.size(0.0025)
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
        .line_strip({viz::make_point3(0.0, 0.0, 0.0), viz::make_point3(end)}, color, 0.0025)
        .position(end)
        .color(color)
        .size(0.025)
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
