#pragma once

#include "L4_planning/gimbal_planner/types.hpp"
#include "L5_weapon/fire_control.hpp"
#include "camera_config.hpp"
#include "ecs/execution_policy.hpp"
#include "foxglove_server.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "tactical_palette.hpp"

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/types.hpp"
#include "math/fast_tf/frame.hpp"
#include "math/fast_tf/matrix.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/scheduler.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>

namespace fcs {

// Shorthand for commonly used namespaces
namespace viz = visualization::builder;
namespace tac = visualization::tactical;

// ============================================================================
// SYSTEM HELPERS
// ============================================================================

namespace detail {

/// Check if Foxglove server is ready and channel has new data
[[nodiscard]] inline bool
    foxglove_ready(const std::shared_ptr<FoxgloveServer>& server, const auto& channel) noexcept {
    return server && server->is_initialized() && channel.has_new();
}

/// Publish JSON message to Foxglove server
template <typename MessageType>
inline void publish_json_message(
    const std::shared_ptr<FoxgloveServer>& server, const nlohmann::json& json_obj) noexcept {
    MessageType msg;
    msg.data = json_to_bytes(json_obj.dump());
    server->enqueue_message(std::move(msg));
}

/// Convert spherical coordinates to Cartesian Vector3
[[nodiscard]] inline foxglove::schemas::Vector3
    spherical_to_cartesian(double distance, double yaw, double pitch) noexcept {
    foxglove::schemas::Vector3 v;
    v.x = distance * std::cos(pitch) * std::cos(yaw);
    v.y = distance * std::cos(pitch) * std::sin(yaw);
    v.z = distance * std::sin(pitch);
    return v;
}

/// Get center position from tracker output as Vector3
[[nodiscard]] inline foxglove::schemas::Vector3
    get_tracker_position(const L3::TrackerOutput& output) noexcept {
    if (output.is_robot()) {
        const auto& state = *output.robot_state();
        return viz::make_vector3(state.position.x(), state.position.y(), state.position.z());
    }
    if (output.is_outpost()) {
        const auto& state  = *output.outpost_state();
        const double avg_z = (state.z[0] + state.z[1] + state.z[2]) / 3.0;
        return viz::make_vector3(state.position.x(), state.position.y(), avg_z);
    }
    return viz::make_vector3(0, 0, 0);
}

/// Get velocity from tracker output
[[nodiscard]] inline Eigen::Vector3d
    get_tracker_velocity(const L3::TrackerOutput& output) noexcept {
    if (output.is_robot()) {
        return output.robot_state()->velocity;
    }
    if (output.is_outpost()) {
        return output.outpost_state()->velocity;
    }
    return Eigen::Vector3d::Zero();
}

/// Get angular velocity from tracker output
[[nodiscard]] inline double get_tracker_v_yaw(const L3::TrackerOutput& output) noexcept {
    if (output.is_robot()) {
        return output.robot_state()->v_yaw;
    }
    if (output.is_outpost()) {
        return output.outpost_state()->v_yaw;
    }
    return 0.0;
}

/// Add armor cubes to scene entity for outpost target
inline void add_outpost_armor_cubes(
    foxglove::schemas::SceneEntity& entity, const L3::OutpostTargetState& state) noexcept {
    const auto armor_poses = state.armor_poses();
    static_assert(armor_poses.size() == 3);

    for (const auto& armor_pose : armor_poses) {
        const Eigen::Quaterniond q =
            Eigen::AngleAxisd(armor_pose[3], Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxisd(-tac::Geometry::ARMOR_TILT_ANGLE, Eigen::Vector3d::UnitY());

        entity.cubes.push_back(
            viz::EntityBuilder::create<fast_tf::odom>("l3", "outpost_armor")
                .position(armor_pose[0], armor_pose[1], armor_pose[2])
                .orientation(q)
                .cube(
                    tac::Geometry::ARMOR_THICKNESS, tac::Geometry::ARMOR_HEIGHT_SMALL,
                    tac::Geometry::ARMOR_WIDTH)
                .build()
                .cubes.back());
    }
}

/// Add armor cubes to scene entity for robot target
inline void add_robot_armor_cubes(
    foxglove::schemas::SceneEntity& entity, const L3::RobotTargetState& state,
    ArmorName target_name) noexcept {
    const bool is_big_armor = target_name == ArmorName::One || target_name == ArmorName::Base;
    const auto armor_poses  = state.armor_poses();
    static_assert(armor_poses.size() == 4);

    for (int i = 0; i < 4; i++) {
        const Eigen::Quaterniond q =
            Eigen::AngleAxisd(armor_poses[i][3], Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxisd(tac::Geometry::ARMOR_TILT_ANGLE, Eigen::Vector3d::UnitY());

        const auto armor_entity =
            viz::EntityBuilder::create<fast_tf::odom>("l3", fmt::format("armor_{}", i))
                .position(armor_poses[i][0], armor_poses[i][1], armor_poses[i][2])
                .orientation(q)
                .cube(
                    tac::Geometry::ARMOR_THICKNESS,
                    is_big_armor ? tac::Geometry::ARMOR_HEIGHT_BIG
                                 : tac::Geometry::ARMOR_HEIGHT_SMALL,
                    tac::Geometry::ARMOR_WIDTH)
                .text_with_offset(
                    fmt::format("{}", i), 0, 0, tac::L3::LABEL_OFFSET_Z, tac::Text::SIZE_SMALL)
                .build();

        entity.cubes.push_back(armor_entity.cubes.back());
        entity.texts.push_back(armor_entity.texts.back());
    }
}

} // namespace detail

// ============================================================================
// FOXGLOVE PUBLISHER SYSTEMS - TACTICAL NEURAL INTERFACE V2.0
// ============================================================================

/// Register all Foxglove publishing systems
template <typename App>
void register_foxglove_systems(App& app, talos::Scheduler* scheduler_ptr_ = nullptr) {

    // =========================================================================
    // IMAGE PUBLISHER (Camera calibration + TF)
    // =========================================================================

    app.template add_system<talos::exclusive<30>>(
        "foxglove_image_pub",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTag> img_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server, talos::res<fcs::CameraConfig> cam,
           talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> det_in,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer) {
            if (!detail::foxglove_ready(*server, img_in)) {
                return;
            }

            auto frame = img_in.read();
            if (!frame || frame->image.empty()) {
                return;
            }

            // Publish TF
            (*server)->publish_tf(**tf_buffer, frame->timestamp_ns);

            // Clone image and draw detection boxes
            cv::Mat img_bgr = frame->image.clone();
            auto batch      = det_in.read_current();
            if (batch) {
                for (size_t i = 0; i < batch->detections.size(); ++i) {
                    const auto& det = batch->detections[i];
                    cv::circle(img_bgr, det.center(), 5, cv::Scalar(0, 0, 255), 5);
                    for (auto i = 0; i < det.corners.size(); i++) {
                        auto pp1 = det.corners[i];
                        auto pp2 = det.corners[(i + 1) % 4];
                        cv::line(img_bgr, pp1, pp2, cv::Scalar(0, 255, 0), 4);
                    }
                }
            }

            // Compress to JPEG
            std::vector<uint8_t> compressed;
            std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 85};
            if (!cv::imencode(".jpg", img_bgr, compressed, params)) {
                return;
            }

            // Construct image message
            ImageMessage msg;
            msg.img.timestamp = timestamp_from_ns(frame->timestamp_ns);
            msg.img.frame_id  = "camera_optical_frame";
            msg.img.format    = "jpeg";
            msg.img.data      = std::vector<std::byte>(
                reinterpret_cast<const std::byte*>(compressed.data()),
                reinterpret_cast<const std::byte*>(compressed.data() + compressed.size()));

            (*server)->enqueue_message(std::move(msg));

            // Publish camera calibration
            std::array<double, 9> camera_matrix_arr;
            std::copy_n(cam->camera_matrix.data(), 9, camera_matrix_arr.begin());
            std::vector<double> dist_coeffs(
                cam->distort_coefficient.data(),
                cam->distort_coefficient.data() + cam->distort_coefficient.size());

            (*server)->publish_camera_calibration(
                cam->width, cam->height, camera_matrix_arr, dist_coeffs, frame->timestamp_ns);
        });

    // =========================================================================
    // DETECTION PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_detection_pub",
        [](talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> det_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, det_in)) {
                return;
            }

            auto batch = det_in.read();
            if (!batch) {
                return;
            }

            // Publish armor detection JSON
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
                obj["color"]      = to_string(det.color);
                obj["type"]       = to_string(det.type);
                obj["name"]       = to_string(det.name);
                armors_json["armors"].push_back(obj);
            }

            detail::publish_json_message<DebugArmorsMessage>(*server, armors_json);
        });

    // =========================================================================
    // MEASUREMENT VISUALIZATION (L2)
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l2_measurement_scene",
        [](talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, meas_in))
                return;

            auto batch = meas_in.read();
            if (!batch)
                return;

            std::vector<foxglove::schemas::SceneEntity> entities;

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
                            fmt::format("{} {}", to_string(m.name), to_string(m.type)), 0,
                            -tac::Text::SIZE_MEDIUM, 0, tac::L2::LABEL_FONT_SIZE)
                        .metadata("confidence", fmt::format("{:.3f}", m.confidence))
                        .metadata("color", std::string(to_string(m.color)))
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
        "foxglove_measurement_pub",
        [](talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, meas_in)) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            // Publish measurement JSON
            nlohmann::json meas_json;
            meas_json["timestamp_ns"] = batch->timestamp_ns;
            meas_json["measurements"] = nlohmann::json::array();

            for (size_t i = 0; i < batch->measurements.size(); ++i) {
                const auto& m = batch->measurements[i];

                const auto translation = m.transform->translation();
                nlohmann::json obj;
                obj["position"] = {translation.x(), translation.y(), translation.z()};
                obj["rpy"]      = m.transform->euler_rot().rpy();
                obj["color"]    = to_string(m.color);
                obj["type"]     = to_string(m.type);
                obj["name"]     = to_string(m.name);
                meas_json["measurements"].push_back(obj);
            }

            detail::publish_json_message<MeasurementMessage>(*server, meas_json);
        });

    // =========================================================================
    // TRACKER VISUALIZATION (L3)
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l3_tracker_scene",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;
            auto output = tracker_in.read();
            if (!output || !output->is_tracking())
                return;

            const auto pos   = detail::get_tracker_position(*output);
            const auto vel   = detail::get_tracker_velocity(*output);
            const auto v_yaw = detail::get_tracker_v_yaw(*output);

            std::vector<foxglove::schemas::SceneEntity> entities;

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
                foxglove::schemas::SceneEntity armor_entity =
                    viz::EntityBuilder::create<fast_tf::odom>("l3", "robot_armors")
                        .timestamp(output->timestamp_ns)
                        .build();

                detail::add_robot_armor_cubes(
                    armor_entity, *output->robot_state(), output->target_name);
                entities.push_back(std::move(armor_entity));
            } else if (output->is_outpost()) {
                foxglove::schemas::SceneEntity armor_entity =
                    viz::EntityBuilder::create<fast_tf::odom>("l3", "outpost_armors")
                        .timestamp(output->timestamp_ns)
                        .build();

                detail::add_outpost_armor_cubes(armor_entity, *output->outpost_state());
                entities.push_back(std::move(armor_entity));
            }

            viz::publish_scene_if_nonempty<TrackSceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // TRACKER JSON PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_tracker_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
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
            target_json["timestamp_ns"] = output->timestamp_ns;
            target_json["status"]       = to_string(output->status);
            target_json["tracking"]     = output->is_tracking();

            if (output->is_robot()) {
                const auto& state       = *output->robot_state();
                target_json["type"]     = "robot";
                target_json["position"] = {
                    state.position.x(), state.position.y(), state.position.z()};
                target_json["velocity"] = {
                    state.velocity.x(), state.velocity.y(), state.velocity.z()};
                target_json["yaw"]    = state.yaw;
                target_json["v_yaw"]  = state.v_yaw;
                target_json["radius"] = {state.radius0, state.radius1};
                target_json["z1"]     = state.z1;
            } else if (output->is_outpost()) {
                const auto& state       = *output->outpost_state();
                target_json["type"]     = "outpost";
                target_json["position"] = {
                    state.position.x(), state.position.y(),
                    (state.z[0] + state.z[1] + state.z[2]) / 3.0};
                auto x = Eigen::Vector3d{
                    state.position.x(), state.position.y(),
                    (state.z[0] + state.z[1] + state.z[2]) / 3.0};
                target_json["dist"]     = x.norm();
                target_json["velocity"] = {
                    state.velocity.x(), state.velocity.y(), state.velocity.z()};
                target_json["yaw"]   = state.yaw;
                target_json["v_yaw"] = state.v_yaw;
                target_json["z"]     = {state.z[0], state.z[1], state.z[2]};
            }

            detail::publish_json_message<TargetMessage>(*server, target_json);
        });

    // =========================================================================
    // ASSOCIATION VISUALIZATION (L3) - Predicted vs Measured
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l3_association_scene",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;

            auto output = tracker_in.read();
            if (!output || !output->is_tracking())
                return;

            auto batch = meas_in.read_current();
            if (!batch)
                return;

            std::vector<foxglove::schemas::SceneEntity> entities;

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

    // =========================================================================
    // GIMBAL COMMAND PUBLISHER
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_gimbal_cmd_pub",
        [](talos::spmc<L5::WeaponCommand, WeaponCommandChannelTag> cmd_in,
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
        "foxglove_l4_gimbal_scene", [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
                                       talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in))
                return;
            auto plan = plan_in.read();
            if (!plan || plan->distance <= 0)
                return;

            std::vector<foxglove::schemas::SceneEntity> entities;

            // Selection marker (fire decision aware)
            {
                foxglove::schemas::SceneEntity entity =
                    viz::EntityBuilder::create<fast_tf::gimbal>("l4", "selection")
                        .timestamp(plan->timestamp_ns)
                        .build();

                const auto position =
                    detail::spherical_to_cartesian(plan->distance, plan->yaw, plan->pitch);

                const auto color =
                    plan->fire_advice ? tac::L4::GIMBAL_AIM_FIRE : tac::L4::GIMBAL_AIM_HOLD;

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = viz::make_pose(position);
                sphere.size = {
                    tac::L4::SELECTION_SIZE, tac::L4::SELECTION_SIZE, tac::L4::SELECTION_SIZE};
                sphere.color = color;
                entity.spheres.push_back(sphere);

                entities.push_back(entity);
            }

            // Predicted position marker
            if (plan->predicted_position) {
                foxglove::schemas::SceneEntity entity =
                    viz::EntityBuilder::create<fast_tf::odom>("l3", "predicted")
                        .timestamp(plan->predicted_future)
                        .build();

                const auto position = viz::make_vector3(
                    plan->predicted_position->x(), plan->predicted_position->y(),
                    plan->predicted_position->z());

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = viz::make_pose(position);
                sphere.size = {
                    tac::L4::PREDICTION_SIZE, tac::L4::PREDICTION_SIZE, tac::L4::PREDICTION_SIZE};
                sphere.color = tac::L3::PREDICTION_AMBER;
                entity.spheres.push_back(sphere);

                entities.push_back(entity);
            }

            // Bullet trajectory
            if (!plan->trajectory.empty()) {
                std::vector<Eigen::Vector3d> traj_points;
                for (const auto& [x, z] : plan->trajectory) {
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
        "foxglove_mpc_trajectory_pub", [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
                                          talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return; // No MPC data available
            }

            const int idx = plan->mpc_current_idx;

            // Build JSON with current values only
            nlohmann::json mpc_json;
            mpc_json["timestamp_ns"]   = plan->timestamp_ns;
            mpc_json["tracking_error"] = plan->tracking_error;
            mpc_json["fire_advice"]    = plan->fire_advice;

            // Reference (input to MPC) at current time
            mpc_json["ref_yaw"]       = plan->ref_yaw[idx];
            mpc_json["ref_pitch"]     = plan->ref_pitch[idx];
            mpc_json["ref_yaw_vel"]   = plan->ref_yaw_vel[idx];
            mpc_json["ref_pitch_vel"] = plan->ref_pitch_vel[idx];

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
        [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
           talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in))
                return;

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty())
                return;

            auto tracker_out = tracker_in.read_current();
            if (!tracker_out || !tracker_out->is_tracking())
                return;

            std::vector<foxglove::schemas::SceneEntity> entities;

            // Reference trajectory
            for (size_t k = 0; k < plan->ref_yaw.size(); ++k) {
                const double t    = (static_cast<int>(k) - plan->mpc_current_idx) * plan->mpc_dt;
                const double dist = plan->distance + t * tac::Temporal::MPC_EXTRAPOLATION_VEL;
                const double x = dist * std::cos(plan->ref_pitch[k]) * std::cos(plan->ref_yaw[k]);
                const double y = dist * std::cos(plan->ref_pitch[k]) * std::sin(plan->ref_yaw[k]);
                const double z = dist * std::sin(plan->ref_pitch[k]);

                int temporal_dist = std::abs(static_cast<int>(k) - plan->mpc_current_idx);
                entities.push_back(
                    viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                        Eigen::Vector3d(x, y, z), temporal_dist,
                        true, // is_reference
                        plan->timestamp_ns));
            }

            // Optimized trajectory
            for (size_t k = 0; k < plan->opt_yaw.size(); ++k) {
                const double t    = (static_cast<int>(k) - plan->mpc_current_idx) * plan->mpc_dt;
                const double dist = plan->distance + t * tac::Temporal::MPC_EXTRAPOLATION_VEL;
                const double x = dist * std::cos(plan->opt_pitch[k]) * std::cos(plan->opt_yaw[k]);
                const double y = dist * std::cos(plan->opt_pitch[k]) * std::sin(plan->opt_yaw[k]);
                const double z = dist * std::sin(plan->opt_pitch[k]);

                int temporal_dist = std::abs(static_cast<int>(k) - plan->mpc_current_idx);
                entities.push_back(
                    viz::patterns::mpc_trajectory_point<fast_tf::odom>(
                        Eigen::Vector3d(x, y, z), temporal_dist,
                        false, // is_optimized
                        plan->timestamp_ns));
            }

            viz::publish_scene_if_nonempty<MpcPredictionSceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // L2 DEBUG: PnP Solver Metrics
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_pnp_debug_pub",
        [](talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, meas_in)) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json pnp_json;
            pnp_json["timestamp_ns"] = batch->timestamp_ns;
            pnp_json["measurements"] = nlohmann::json::array();

            for (size_t i = 0; i < batch->measurements.size(); ++i) {
                const auto& m = batch->measurements[i];
                nlohmann::json obj;
                obj["index"]      = i;
                obj["name"]       = to_string(m.name);
                const auto t      = m.transform->translation();
                obj["position"]   = {t.x(), t.y(), t.z()};
                const auto rpy    = m.transform->euler_rot().rpy();
                obj["rpy"]        = {std::get<0>(rpy), std::get<1>(rpy), std::get<2>(rpy)};
                obj["confidence"] = m.confidence;
                pnp_json["measurements"].push_back(obj);
            }

            detail::publish_json_message<PnPSolverMessage>(*server, pnp_json);
        });

    // =========================================================================
    // L2 DEBUG: NN Confidence Scores
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_nn_confidence_pub",
        [](talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> det_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, det_in)) {
                return;
            }

            auto batch = det_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json nn_json;
            nn_json["timestamp_ns"] = batch->timestamp_ns;
            nn_json["detections"]   = nlohmann::json::array();

            for (size_t i = 0; i < batch->detections.size(); ++i) {
                const auto& det = batch->detections[i];
                nlohmann::json obj;
                obj["index"]      = i;
                obj["confidence"] = det.confidence;
                obj["color"]      = to_string(det.color);
                obj["type"]       = to_string(det.type);
                obj["name"]       = to_string(det.name);
                nn_json["detections"].push_back(obj);
            }

            detail::publish_json_message<NNConfidenceMessage>(*server, nn_json);
        });

    // =========================================================================
    // L3 DEBUG: Association Phase Probabilities
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_association_phase_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->association_debug) {
                return;
            }

            const auto& debug = *output->association_debug;
            nlohmann::json phase_json;
            phase_json["timestamp_ns"] = output->timestamp_ns;
            phase_json["target_name"]  = to_string(output->target_name);
            phase_json["phase_probs"]  = debug.phase_probs;
            phase_json["map_phase"]    = debug.map_phase;
            phase_json["is_ambiguous"] = debug.is_ambiguous;
            phase_json["n_phases"]     = debug.phase_probs.size();

            detail::publish_json_message<AssociationPhaseMessage>(*server, phase_json);
        });

    // =========================================================================
    // L3 DEBUG: Association Permutation Rankings
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_association_permutation_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->association_debug) {
                return;
            }

            const auto& debug = *output->association_debug;
            nlohmann::json perm_json;
            perm_json["timestamp_ns"] = output->timestamp_ns;
            perm_json["permutations"] = nlohmann::json::array();

            for (const auto& perm_info : debug.top_permutations) {
                nlohmann::json p;
                p["perm"]           = perm_info.perm;
                p["log_prob"]       = perm_info.log_prob;
                p["gate_passed"]    = perm_info.gate_passed;
                p["mahalanobis_sq"] = perm_info.mahalanobis_sq;
                perm_json["permutations"].push_back(p);
            }

            detail::publish_json_message<AssociationPermutationMessage>(*server, perm_json);
        });

    // =========================================================================
    // L3 DEBUG: Filter Innovation Vectors
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_filter_innovation_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->filter_debug) {
                return;
            }

            const auto& debug = *output->filter_debug;
            nlohmann::json innov_json;
            innov_json["timestamp_ns"] = output->timestamp_ns;

            if (output->is_robot()) {
                innov_json["state_type"] = "robot";
                const auto& state        = *output->robot_state();
                innov_json["position"]   = {
                    state.position.x(), state.position.y(), state.position.z()};
                innov_json["velocity"] = {
                    state.velocity.x(), state.velocity.y(), state.velocity.z()};
                innov_json["yaw"] = state.yaw;
            } else if (output->is_outpost()) {
                innov_json["state_type"] = "outpost";
                const auto& state        = *output->outpost_state();
                innov_json["position"]   = {state.position.x(), state.position.y()};
                innov_json["z"]          = {state.z[0], state.z[1], state.z[2]};
                innov_json["yaw"]        = state.yaw;
            }

            if (debug.innovation.size() > 0) {
                std::vector<double> innov_vec(
                    debug.innovation.data(), debug.innovation.data() + debug.innovation.size());
                innov_json["innovation"] = innov_vec;
            }
            innov_json["covariance_trace"] = debug.covariance_trace;

            detail::publish_json_message<FilterInnovationMessage>(*server, innov_json);
        });

    // =========================================================================
    // L3 DEBUG: Filter Covariance Trace
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_filter_covariance_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->filter_debug) {
                return;
            }

            const auto& debug = *output->filter_debug;
            nlohmann::json cov_json;
            cov_json["timestamp_ns"]    = output->timestamp_ns;
            cov_json["cov_trace"]       = debug.covariance_trace;
            cov_json["yaw_uncertainty"] = debug.yaw_uncertainty;

            detail::publish_json_message<FilterCovarianceMessage>(*server, cov_json);
        });

    // =========================================================================
    // L3 DEBUG: Filter Covariance 3D Scene
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_filter_covariance_scene_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;

            auto output = tracker_in.read();
            if (!output || !output->is_tracking())
                return;

            std::vector<foxglove::schemas::SceneEntity> entities;

            const auto pos_vec = detail::get_tracker_position(*output);
            const Eigen::Vector3d pos(pos_vec.x, pos_vec.y, pos_vec.z);

            entities.push_back(
                viz::patterns::uncertainty_sphere<fast_tf::odom>(pos, 1.0, output->timestamp_ns));

            viz::publish_scene_if_nonempty<FilterCovarianceSceneMessage>(
                *server, std::move(entities));
        });

    // =========================================================================
    // L3 DEBUG: Viterbi Smaller Trellis
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_viterbi_trellis_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->association_debug) {
                return;
            }

            const auto& debug = *output->association_debug;
            nlohmann::json viterbi_json;
            viterbi_json["timestamp_ns"] = output->timestamp_ns;
            viterbi_json["n_phases"]     = debug.phase_probs.size();
            viterbi_json["phase_probs"]  = debug.phase_probs;
            viterbi_json["map_phase"]    = debug.map_phase;
            viterbi_json["is_ambiguous"] = debug.is_ambiguous;

            detail::publish_json_message<ViterbiTrellisMessage>(*server, viterbi_json);
        });

    // =========================================================================
    // L3 DEBUG: State Machine Status
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_state_machine_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->state_machine_debug) {
                return;
            }

            const auto& debug = *output->state_machine_debug;
            nlohmann::json sm_json;
            sm_json["timestamp_ns"]        = output->timestamp_ns;
            sm_json["status"]              = to_string(output->status);
            sm_json["is_tracking"]         = output->is_tracking();
            sm_json["target_name"]         = to_string(output->target_name);
            sm_json["time_in_state"]       = debug.time_in_state;
            sm_json["detecting_count"]     = debug.detecting_count;
            sm_json["lost_time"]           = debug.lost_time;
            sm_json["consecutive_updates"] = debug.consecutive_updates;

            detail::publish_json_message<StateMachineMessage>(*server, sm_json);
        });

    // =========================================================================
    // L4 DEBUG: Full MPC Prediction Horizon
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_mpc_horizon_pub", [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
                                       talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return;
            }

            nlohmann::json horizon_json;
            horizon_json["timestamp_ns"] = plan->timestamp_ns;
            horizon_json["mpc_dt"]       = plan->mpc_dt;
            horizon_json["current_idx"]  = plan->mpc_current_idx;
            horizon_json["horizon"]      = plan->ref_yaw.size();
            horizon_json["time"]         = nlohmann::json::array();

            for (size_t k = 0; k < plan->ref_yaw.size(); ++k) {
                const double t = (static_cast<int>(k) - plan->mpc_current_idx) * plan->mpc_dt;
                horizon_json["time"].push_back(t);
            }
            horizon_json["ref_yaw"]       = plan->ref_yaw;
            horizon_json["ref_pitch"]     = plan->ref_pitch;
            horizon_json["opt_yaw"]       = plan->opt_yaw;
            horizon_json["opt_pitch"]     = plan->opt_pitch;
            horizon_json["ref_yaw_vel"]   = plan->ref_yaw_vel;
            horizon_json["ref_pitch_vel"] = plan->ref_pitch_vel;
            horizon_json["opt_yaw_vel"]   = plan->opt_yaw_vel;
            horizon_json["opt_pitch_vel"] = plan->opt_pitch_vel;

            detail::publish_json_message<MpcHorizonMessage>(*server, horizon_json);
        });

    // =========================================================================
    // L4 DEBUG: MPC Cost Evolution
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_mpc_cost_pub", [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
                                    talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return;
            }

            nlohmann::json cost_json;
            cost_json["timestamp_ns"]         = plan->timestamp_ns;
            cost_json["total_tracking_error"] = plan->tracking_error;
            cost_json["total_cost"]           = plan->total_cost;
            cost_json["mpc_iterations"]       = plan->mpc_iterations;
            cost_json["mpc_solve_time_us"]    = plan->mpc_solve_time_us;
            cost_json["mpc_converged"]        = plan->mpc_converged;
            cost_json["cost_yaw_pos"]         = plan->cost_yaw_pos;
            cost_json["cost_yaw_vel"]         = plan->cost_yaw_vel;
            cost_json["cost_pitch_pos"]       = plan->cost_pitch_pos;
            cost_json["cost_pitch_vel"]       = plan->cost_pitch_vel;

            detail::publish_json_message<MpcCostMessage>(*server, cost_json);
        });

    // =========================================================================
    // L4 DEBUG: MPC Constraint Margins
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_mpc_constraint_pub", [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
                                          talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return;
            }

            nlohmann::json constraint_json;
            constraint_json["timestamp_ns"]     = plan->timestamp_ns;
            constraint_json["yaw_acc_margin"]   = plan->constraint_yaw_acc_margin;
            constraint_json["pitch_acc_margin"] = plan->constraint_pitch_acc_margin;

            detail::publish_json_message<MpcConstraintMessage>(*server, constraint_json);
        });

    // =========================================================================
    // PERFORMANCE STATISTICS PUBLISHER (1 Hz, exclusive_silent)
    // =========================================================================

    if (scheduler_ptr_) {
        app.template add_system<talos::exclusive_silent<100>>(
            "foxglove_perf_stats_pub",
            [scheduler_ptr_](talos::res<std::shared_ptr<FoxgloveServer>> server) {
                auto& server_ptr = *server; // 引用绑定
                if (!server_ptr->is_initialized()) {
                    return;
                }

                // Get scheduler performance statistics JSON
                std::string json_str = scheduler_ptr_->get_stats_json();

                // Send to foxglove
                PerfStatsMessage msg;
                msg.data = json_to_bytes(json_str);
                server_ptr->enqueue_message(std::move(msg));
            });
    }
}

} // namespace fcs
