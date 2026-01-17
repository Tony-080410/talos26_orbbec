#pragma once

// Implementation header; prefer including `fcs/visualization/foxglove_systems.hpp`.

#include "L4_planning/gimbal_planner/types.hpp"
#include "L5_weapon/fire_control.hpp"
#include "foxglove_server.hpp"
#include "foxglove_types.hpp"

#include "auto_aim/config.hpp"
#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/types.hpp"
#include "math/fast_tf/frame.hpp"
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

// ============================================================================
// Foxglove System Helpers
// ============================================================================

namespace detail {

// Constants for visualization
constexpr double ARROW_SHAFT_DIAMETER       = 0.02;
constexpr double ARROW_HEAD_DIAMETER        = 0.04;
constexpr double ARROW_MIN_LENGTH           = 0.0001;
constexpr double ARROW_SHAFT_LENGTH_RATIO   = 0.8;
constexpr double ARROW_HEAD_LENGTH_RATIO    = 0.2;
constexpr double TEXT_DEFAULT_FONT_SIZE     = 0.05;
constexpr uint64_t SCENE_ENTITY_LIFETIME_NS = 100000000;
constexpr double ARMOR_CUBE_THICKNESS       = 0.03;
constexpr double ARMOR_CUBE_HEIGHT_SMALL    = 0.135;
constexpr double ARMOR_CUBE_HEIGHT_BIG      = 0.23;
constexpr double ARMOR_CUBE_WIDTH           = 0.125;
constexpr double ARMOR_TILT_ANGLE           = 0.2618; // 15 degrees in radians

// Colors
constexpr foxglove::schemas::Color COLOR_YELLOW = {1.0, 1.0, 0.0, 0.8};
constexpr foxglove::schemas::Color COLOR_CYAN   = {0.0, 1.0, 1.0, 0.8};
constexpr foxglove::schemas::Color COLOR_GREEN  = {0.0, 1.0, 0.0, 0.6};
constexpr foxglove::schemas::Color COLOR_WHITE  = {1.0, 1.0, 1.0, 0.3};
constexpr foxglove::schemas::Color COLOR_RED    = {1.0, 0.0, 0.0, 0.6};
constexpr foxglove::schemas::Color COLOR_ORANGE = {1.0, 0.5, 0.0, 0.3};

// Sizes
constexpr double SPHERE_SIZE_DEFAULT  = 0.1;
constexpr double SPHERE_SIZE_SMALL    = 0.08;
constexpr double SPHERE_SIZE_TINY     = 0.01;
constexpr double ARMOR_LABEL_OFFSET_Z = 0.1;
constexpr double TEXT_OFFSET_Y        = 0.1;

// Quaternions
constexpr double QUAT_IDENTITY_X = 0.0;
constexpr double QUAT_IDENTITY_Y = 0.0;
constexpr double QUAT_IDENTITY_Z = 0.0;
constexpr double QUAT_IDENTITY_W = 1.0;
constexpr double QUAT_90_DEG_X   = 0.0;
constexpr double QUAT_90_DEG_Y   = 0.7071;
constexpr double QUAT_90_DEG_Z   = 0.0;
constexpr double QUAT_90_DEG_W   = 0.7071;

// Image compression
constexpr int JPEG_COMPRESSION_QUALITY     = 85;
constexpr double VECTOR_PARALLEL_THRESHOLD = 1e-6;

/// Check if Foxglove server is ready and channel has new data
[[nodiscard]] inline bool
    foxglove_ready(const std::shared_ptr<FoxgloveServer>& server, const auto& channel) noexcept {
    return server && server->is_initialized() && channel.has_new();
}

/// Create a SceneEntity with common fields
[[nodiscard]] inline foxglove::schemas::SceneEntity
    create_scene_entity(std::string id, std::string frame_id, uint64_t timestamp_ns) noexcept {
    foxglove::schemas::SceneEntity entity;
    entity.id        = std::move(id);
    entity.frame_id  = std::move(frame_id);
    entity.lifetime  = foxglove::schemas::Duration{.nsec = SCENE_ENTITY_LIFETIME_NS};
    entity.timestamp = timestamp_from_ns(timestamp_ns);
    return entity;
}

/// Create a Pose from position (optional) and orientation (optional)
[[nodiscard]] inline foxglove::schemas::Pose create_pose(
    std::optional<foxglove::schemas::Vector3> position       = std::nullopt,
    std::optional<foxglove::schemas::Quaternion> orientation = std::nullopt) noexcept {
    foxglove::schemas::Pose pose;
    pose.position    = std::move(position);
    pose.orientation = std::move(orientation);
    return pose;
}

/// Create a Vector3
[[nodiscard]] inline foxglove::schemas::Vector3
    create_vector3(double x, double y, double z) noexcept {
    foxglove::schemas::Vector3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

/// Create a Quaternion from components
[[nodiscard]] inline foxglove::schemas::Quaternion
    create_quaternion(double x, double y, double z, double w) noexcept {
    foxglove::schemas::Quaternion q;
    q.x = x;
    q.y = y;
    q.z = z;
    q.w = w;
    return q;
}

/// Create a Color
[[nodiscard]] inline foxglove::schemas::Color
    create_color(double r, double g, double b, double a) noexcept {
    foxglove::schemas::Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    c.a = a;
    return c;
}

/// Create a standardized text primitive
[[nodiscard]] inline foxglove::schemas::TextPrimitive create_text_primitive(
    std::string text, foxglove::schemas::Pose pose, double font_size = TEXT_DEFAULT_FONT_SIZE,
    foxglove::schemas::Color color = create_color(1.0, 1.0, 1.0, 0.9)) noexcept {
    foxglove::schemas::TextPrimitive text_prim;
    text_prim.text      = std::move(text);
    text_prim.pose      = std::move(pose);
    text_prim.font_size = font_size;
    text_prim.color     = std::move(color);
    text_prim.billboard = true;
    return text_prim;
}

/// Create arrow primitive for angular velocity (along Z axis)
[[nodiscard]] inline foxglove::schemas::ArrowPrimitive create_angular_velocity_arrow(
    const foxglove::schemas::Vector3& position, double v_yaw) noexcept {
    foxglove::schemas::ArrowPrimitive arrow;
    const double length = std::abs(v_yaw) / M_PI;
    if (length <= ARROW_MIN_LENGTH) {
        return arrow;
    }

    arrow.pose           = create_pose();
    arrow.pose->position = position;
    const double sign    = v_yaw >= 0 ? -1.0 : 1.0;
    arrow.pose->orientation =
        create_quaternion(QUAT_90_DEG_X, sign * QUAT_90_DEG_Y, QUAT_90_DEG_Z, QUAT_90_DEG_W);
    arrow.shaft_length   = length * ARROW_SHAFT_LENGTH_RATIO;
    arrow.shaft_diameter = ARROW_SHAFT_DIAMETER;
    arrow.head_length    = length * ARROW_HEAD_LENGTH_RATIO;
    arrow.head_diameter  = ARROW_HEAD_DIAMETER;
    arrow.color          = COLOR_YELLOW;
    return arrow;
}

/// Create arrow primitive for linear velocity
[[nodiscard]] inline foxglove::schemas::ArrowPrimitive create_linear_velocity_arrow(
    const foxglove::schemas::Vector3& position, const Eigen::Vector3d& velocity) noexcept {
    foxglove::schemas::ArrowPrimitive arrow;
    const double length = velocity.norm();
    if (length <= ARROW_MIN_LENGTH) {
        return arrow;
    }

    arrow.pose           = create_pose();
    arrow.pose->position = position;

    const Eigen::Vector3d vel_dir = velocity.normalized();
    const Eigen::Vector3d default_dir(1, 0, 0);
    Eigen::Quaterniond q;
    if ((vel_dir + default_dir).norm() < VECTOR_PARALLEL_THRESHOLD) {
        q = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
    } else {
        q = Eigen::Quaterniond::FromTwoVectors(default_dir, vel_dir);
    }
    arrow.pose->orientation = create_quaternion(q.x(), q.y(), q.z(), q.w());
    arrow.shaft_length      = length * ARROW_SHAFT_LENGTH_RATIO;
    arrow.shaft_diameter    = ARROW_SHAFT_DIAMETER;
    arrow.head_length       = length * ARROW_HEAD_LENGTH_RATIO;
    arrow.head_diameter     = ARROW_HEAD_DIAMETER;
    arrow.color             = COLOR_CYAN;
    return arrow;
}

/// Add velocity arrows to scene entity for robot/outpost states
inline void add_velocity_arrows(
    foxglove::schemas::SceneEntity& entity, const foxglove::schemas::Vector3& position,
    double v_yaw, const Eigen::Vector3d& linear_velocity) noexcept {
    if (auto angular_arrow = create_angular_velocity_arrow(position, v_yaw);
        angular_arrow.pose.has_value()) {
        entity.arrows.push_back(std::move(angular_arrow));
    }
    if (auto linear_arrow = create_linear_velocity_arrow(position, linear_velocity);
        linear_arrow.pose.has_value()) {
        entity.arrows.push_back(std::move(linear_arrow));
    }
}

/// Create armor cube primitive with standardized sizing
[[nodiscard]] inline foxglove::schemas::CubePrimitive create_armor_cube(
    const foxglove::schemas::Pose& pose, bool is_big_armor, double alpha = 0.3) noexcept {
    foxglove::schemas::CubePrimitive cube;
    cube.pose = pose;
    cube.size = {
        ARMOR_CUBE_THICKNESS, is_big_armor ? ARMOR_CUBE_HEIGHT_BIG : ARMOR_CUBE_HEIGHT_SMALL,
        ARMOR_CUBE_WIDTH};
    cube.color = create_color(0.0, 1.0, 0.0, alpha);
    return cube;
}

/// Create quaternion from armor pose (yaw, pitch, roll)
[[nodiscard]] inline Eigen::Quaterniond
    armor_pose_to_quaternion(double yaw, double pitch, double roll) noexcept {
    return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
}

/// Add armor cubes to scene entity for outpost target
inline void add_outpost_armor_cubes(
    foxglove::schemas::SceneEntity& entity, const L3::OutpostTargetState& state) noexcept {
    const auto armor_poses = state.armor_poses();
    static_assert(armor_poses.size() == 3);

    for (const auto& armor_pose : armor_poses) {
        const Eigen::Quaterniond q =
            armor_pose_to_quaternion(armor_pose[3], -ARMOR_TILT_ANGLE, 0.0);

        const auto pose = create_pose(
            create_vector3(armor_pose[0], armor_pose[1], armor_pose[2]),
            create_quaternion(q.x(), q.y(), q.z(), q.w()));

        entity.cubes.push_back(create_armor_cube(pose, false));
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
            armor_pose_to_quaternion(armor_poses[i][3], ARMOR_TILT_ANGLE, 0.0);

        const auto pose = create_pose(
            create_vector3(armor_poses[i][0], armor_poses[i][1], armor_poses[i][2]),
            create_quaternion(q.x(), q.y(), q.z(), q.w()));

        entity.cubes.push_back(create_armor_cube(pose, is_big_armor));

        auto label_pose = pose;
        label_pose.position->z += ARMOR_LABEL_OFFSET_Z;
        entity.texts.push_back(create_text_primitive(fmt::format("{}", i), label_pose));
    }
}

/// Get center position from tracker output as Vector3
[[nodiscard]] inline foxglove::schemas::Vector3
    get_tracker_position(const L3::TrackerOutput& output) noexcept {
    if (output.is_robot()) {
        const auto& state = *output.robot_state();
        return create_vector3(state.position.x(), state.position.y(), state.position.z());
    }
    if (output.is_outpost()) {
        const auto& state  = *output.outpost_state();
        const double avg_z = (state.z[0] + state.z[1] + state.z[2]) / 3.0;
        return create_vector3(state.position.x(), state.position.y(), avg_z);
    }
    return create_vector3(0, 0, 0);
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

/// Add metadata key-value pair to scene entity
inline void add_metadata(
    foxglove::schemas::SceneEntity& entity, std::string key, std::string value) noexcept {
    foxglove::schemas::KeyValuePair kv;
    kv.key   = std::move(key);
    kv.value = std::move(value);
    entity.metadata.push_back(std::move(kv));
}

/// Create armor color based on armor color enum
[[nodiscard]] inline foxglove::schemas::Color
    get_armor_color(ArmorColor color, bool alternate = false) noexcept {
    if (color == ArmorColor::Red) {
        return alternate ? create_color(1.0, 0.5, 0.0, 0.5) : create_color(1.0, 0.0, 0.0, 0.5);
    }
    return alternate ? create_color(0.0, 0.5, 1.0, 0.5) : create_color(0.0, 0.0, 1.0, 0.5);
}

/// Create a sphere primitive with standardized sizing
[[nodiscard]] inline foxglove::schemas::SpherePrimitive create_sphere(
    const foxglove::schemas::Pose& pose, double size = SPHERE_SIZE_DEFAULT,
    foxglove::schemas::Color color = COLOR_GREEN) noexcept {
    foxglove::schemas::SpherePrimitive sphere;
    sphere.pose  = pose;
    sphere.size  = {size, size, size};
    sphere.color = color;
    return sphere;
}

/// Create identity pose (optional position with identity orientation)
[[nodiscard]] inline foxglove::schemas::Pose create_identity_pose(
    std::optional<foxglove::schemas::Vector3> position = std::nullopt) noexcept {
    return create_pose(
        std::move(position),
        create_quaternion(QUAT_IDENTITY_X, QUAT_IDENTITY_Y, QUAT_IDENTITY_Z, QUAT_IDENTITY_W));
}

} // namespace detail

// ============================================================================
// Foxglove Publisher Systems
// ============================================================================

/// 注册所有 Foxglove 发布系统
template <typename App>
void register_foxglove_systems(App& app, talos::Scheduler* scheduler_ptr_ = nullptr) {
    // 图像发布系统 (同时发布 camera_calibration 和 TF)
    app.template add_system<talos::pool_compute>(
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

            // 发布 TF
            (*server)->publish_tf(**tf_buffer, frame->timestamp_ns);

            // 克隆图像避免数据竞争
            cv::Mat img_bgr = frame->image.clone();
            auto batch      = det_in.read_current();
            if (batch) {
                for (size_t i = 0; i < batch->detections.size(); ++i) {
                    const auto& det = batch->detections[i];
                    nlohmann::json obj;
                    obj["center"]["x"] = det.center().x;
                    obj["center"]["y"] = det.center().y;
                    cv::circle(img_bgr, det.center(), 5, cv::Scalar(0, 0, 255), 5);
                    for (auto i = 0; i < det.corners.size(); i++) {
                        auto pp1 = det.corners[i];
                        auto pp2 = det.corners[(i + 1) % 4];
                        cv::line(img_bgr, pp1, pp2, cv::Scalar(0, 255, 0), 4);
                    }
                }
            }
            // 压缩为 JPEG
            std::vector<uint8_t> compressed;
            std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, detail::JPEG_COMPRESSION_QUALITY};
            if (!cv::imencode(".jpg", img_bgr, compressed, params)) {
                return;
            }

            // 构造消息
            ImageMessage msg;
            msg.img.timestamp = timestamp_from_ns(frame->timestamp_ns);
            msg.img.frame_id  = "camera_optical_frame";
            msg.img.format    = "jpeg";
            msg.img.data      = std::vector<std::byte>(
                reinterpret_cast<const std::byte*>(compressed.data()),
                reinterpret_cast<const std::byte*>(compressed.data() + compressed.size()));

            (*server)->enqueue_message(std::move(msg));

            // 发布相机标定信息
            std::array<double, 9> camera_matrix_arr;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    camera_matrix_arr[i * 3 + j] = cam->camera_matrix(i, j);
                }
            }
            std::vector<double> dist_coeffs(
                cam->distort_coefficient.data(),
                cam->distort_coefficient.data() + cam->distort_coefficient.size());

            (*server)->publish_camera_calibration(
                cam->width, cam->height, camera_matrix_arr, dist_coeffs, frame->timestamp_ns);
        });

    // 检测结果发布系统
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

            // 发布装甲板检测 JSON
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

            DebugArmorsMessage msg;
            msg.data = json_to_bytes(armors_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // 测量值发布系统 (PnP 结果)
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

            // 发布测量值 JSON
            nlohmann::json meas_json;
            meas_json["timestamp_ns"] = batch->timestamp_ns;
            meas_json["measurements"] = nlohmann::json::array();

            // 构造 3D 场景
            std::vector<foxglove::schemas::SceneEntity> entities;

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

                foxglove::schemas::SceneEntity entity = detail::create_scene_entity(
                    "armor_" + std::to_string(i), "odom", m.timestamp_ns);

                const auto position =
                    detail::create_vector3(translation.x(), translation.y(), translation.z());
                const auto q    = m.transform->quaternion();
                const auto pose = detail::create_pose(
                    position, detail::create_quaternion(q.x(), q.y(), q.z(), q.w()));

                foxglove::schemas::CubePrimitive cube;
                cube.pose = pose;
                cube.size = {
                    detail::ARMOR_CUBE_THICKNESS, detail::ARMOR_CUBE_HEIGHT_SMALL,
                    detail::ARMOR_CUBE_WIDTH};
                cube.color = detail::get_armor_color(m.color, false);
                entity.cubes.push_back(cube);

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = pose;
                sphere.size = {
                    detail::ARMOR_CUBE_THICKNESS, detail::ARMOR_CUBE_HEIGHT_SMALL,
                    detail::ARMOR_CUBE_WIDTH};
                sphere.color = detail::get_armor_color(m.color, true);
                entity.spheres.push_back(sphere);

                auto text_pose = pose;
                text_pose.position->y -= detail::TEXT_OFFSET_Y;
                entity.texts.push_back(
                    detail::create_text_primitive(
                        fmt::format("{} {}", to_string(m.name), to_string(m.type)), text_pose));

                detail::add_metadata(entity, "confidence", std::to_string(m.confidence));
                detail::add_metadata(entity, "color", std::string(to_string(m.color)));

                entities.push_back(entity);
            }

            // 发送测量值 JSON
            MeasurementMessage meas_msg;
            meas_msg.data = json_to_bytes(meas_json.dump());
            (*server)->enqueue_message(std::move(meas_msg));

            // 发送 3D 场景
            if (!entities.empty()) {
                SceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });

    // Tracker 输出发布系统
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

            // 发布目标状态 JSON
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

            TargetMessage msg;
            msg.data = json_to_bytes(target_json.dump());
            (*server)->enqueue_message(std::move(msg));

            // 发布跟踪场景 (预测装甲板位置)
            if (output->is_tracking()) {
                std::vector<foxglove::schemas::SceneEntity> entities;

                foxglove::schemas::SceneEntity entity =
                    detail::create_scene_entity("target", "odom", output->timestamp_ns);

                const auto position = detail::get_tracker_position(*output);
                const auto velocity = detail::get_tracker_velocity(*output);
                const auto v_yaw    = detail::get_tracker_v_yaw(*output);

                const auto pose = detail::create_identity_pose(position);

                entity.spheres.push_back(
                    detail::create_sphere(pose, detail::SPHERE_SIZE_DEFAULT, detail::COLOR_GREEN));

                // 绘制速度箭头
                detail::add_velocity_arrows(entity, position, v_yaw, velocity);

                // 添加装甲板
                if (output->is_outpost()) {
                    detail::add_outpost_armor_cubes(entity, *output->outpost_state());
                } else if (output->is_robot()) {
                    detail::add_robot_armor_cubes(
                        entity, *output->robot_state(), output->target_name);
                }

                entities.push_back(entity);
                TrackSceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });

    // 云台指令发布系统
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

            GimbalCmdMessage msg;
            msg.data = json_to_bytes(cmd_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // 云台场景发布系统 (弹道/预测位置/选择点可视化)
    app.template add_system<talos::pool_compute>(
        "foxglove_gimbal_scene_pub", [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> cmd_in,
                                        talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, cmd_in)) {
                return;
            }

            auto cmd = cmd_in.read();
            if (!cmd || cmd->distance <= 0) {
                return;
            }

            std::vector<foxglove::schemas::SceneEntity> entities;

            // Selection marker - 瞄准点 (黄绿色球体)
            {
                foxglove::schemas::SceneEntity entity =
                    detail::create_scene_entity("selection", "gimbal_link", cmd->timestamp_ns);

                // 根据 yaw, pitch, distance 计算瞄准点位置
                const double x = cmd->distance * std::cos(cmd->pitch) * std::cos(cmd->yaw);
                const double y = cmd->distance * std::cos(cmd->pitch) * std::sin(cmd->yaw);
                const double z = cmd->distance * std::sin(cmd->pitch);

                entity.spheres.push_back(
                    detail::create_sphere(
                        detail::create_identity_pose(detail::create_vector3(x, y, z)),
                        detail::SPHERE_SIZE_DEFAULT, detail::COLOR_ORANGE));
                entities.push_back(entity);
            }

            // Predicted position marker - 预测位置 (橙色球体)
            if (cmd->predicted_position) {
                foxglove::schemas::SceneEntity entity =
                    detail::create_scene_entity("predicted", "odom", cmd->predicted_future);

                entity.spheres.push_back(
                    detail::create_sphere(
                        detail::create_identity_pose(
                            detail::create_vector3(
                                cmd->predicted_position->x(), cmd->predicted_position->y(),
                                cmd->predicted_position->z())),
                        detail::SPHERE_SIZE_SMALL, detail::COLOR_RED));
                entities.push_back(entity);
            }

            // Trajectory marker - 弹道轨迹 (点列)
            if (!cmd->trajectory.empty()) {
                foxglove::schemas::SceneEntity entity =
                    detail::create_scene_entity("trajectory", "muzzle_link", cmd->timestamp_ns);

                // 根据 fire_advice 改变颜色
                const auto traj_color =
                    cmd->fire_advice ? detail::COLOR_GREEN : detail::COLOR_WHITE;

                for (const auto& [x, z] : cmd->trajectory) {
                    entity.spheres.push_back(
                        detail::create_sphere(
                            detail::create_identity_pose(detail::create_vector3(x, 0, z)),
                            detail::SPHERE_SIZE_TINY, traj_color));
                }
                entities.push_back(entity);
            }

            if (!entities.empty()) {
                GimbalSceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });

    // MPC 轨迹发布系统 (参考轨迹、优化轨迹、前馈加速度)
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

            MpcTrajectoryMessage msg;
            msg.data = json_to_bytes(mpc_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // ============================================================================
    // L2 Debug Visualization Systems
    // ============================================================================

    // L2: PnP solver metrics (reprojection errors, solve time)
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

            PnPSolverMessage msg;
            msg.data = json_to_bytes(pnp_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L2: NN confidence scores
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

            NNConfidenceMessage msg;
            msg.data = json_to_bytes(nn_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // ============================================================================
    // L3 Debug Visualization Systems
    // ============================================================================

    // L3: Association phase probabilities
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

            AssociationPhaseMessage msg;
            msg.data = json_to_bytes(phase_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L3: Association permutation rankings
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

            AssociationPermutationMessage msg;
            msg.data = json_to_bytes(perm_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L3: Association 3D scene (predicted vs matched armors)
    app.template add_system<talos::pool_compute>(
        "foxglove_association_scene_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->is_tracking()) {
                return;
            }

            auto batch = meas_in.read_current();
            if (!batch) {
                return;
            }

            std::vector<foxglove::schemas::SceneEntity> entities;

            // Create predicted armors (from tracker state)
            foxglove::schemas::SceneEntity pred_entity =
                detail::create_scene_entity("predicted_armors", "odom", output->timestamp_ns);

            if (output->is_robot()) {
                const auto& state      = *output->robot_state();
                const auto armor_poses = state.armor_poses();
                for (int i = 0; i < 4; i++) {
                    const auto pose = detail::create_identity_pose(
                        detail::create_vector3(
                            armor_poses[i][0], armor_poses[i][1], armor_poses[i][2]));

                    pred_entity.spheres.push_back(
                        detail::create_sphere(
                            pose, detail::SPHERE_SIZE_SMALL, detail::COLOR_GREEN));

                    auto label_pose = pose;
                    label_pose.position->z += detail::TEXT_OFFSET_Y;
                    pred_entity.texts.push_back(
                        detail::create_text_primitive(fmt::format("pred_{}", i), label_pose));
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

                    const auto pose = detail::create_identity_pose(
                        detail::create_vector3(armor_pos.x(), armor_pos.y(), armor_pos.z()));

                    pred_entity.spheres.push_back(
                        detail::create_sphere(
                            pose, detail::SPHERE_SIZE_SMALL, detail::COLOR_GREEN));

                    auto label_pose = pose;
                    label_pose.position->z += detail::TEXT_OFFSET_Y;
                    pred_entity.texts.push_back(
                        detail::create_text_primitive(fmt::format("pred_{}", i), label_pose));
                }
            }
            entities.push_back(pred_entity);

            // Create measurements (from PnP solver)
            foxglove::schemas::SceneEntity meas_entity =
                detail::create_scene_entity("measurements", "odom", batch->timestamp_ns);

            for (size_t i = 0; i < batch->measurements.size(); ++i) {
                const auto& m = batch->measurements[i];
                const auto t  = m.transform->translation();
                const auto pose =
                    detail::create_identity_pose(detail::create_vector3(t.x(), t.y(), t.z()));

                meas_entity.spheres.push_back(
                    detail::create_sphere(pose, detail::SPHERE_SIZE_SMALL, detail::COLOR_YELLOW));

                auto label_pose = pose;
                label_pose.position->z += detail::TEXT_OFFSET_Y;
                meas_entity.texts.push_back(
                    detail::create_text_primitive(fmt::format("meas_{}", i), label_pose));
            }
            entities.push_back(meas_entity);

            if (!entities.empty()) {
                AssociationSceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });

    // L3: Filter innovation vectors
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

            FilterInnovationMessage msg;
            msg.data = json_to_bytes(innov_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L3: Filter covariance trace
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

            FilterCovarianceMessage msg;
            msg.data = json_to_bytes(cov_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L3: Filter covariance 3D scene (uncertainty visualization)
    app.template add_system<talos::pool_compute>(
        "foxglove_filter_covariance_scene_pub",
        [](talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->is_tracking()) {
                return;
            }

            std::vector<foxglove::schemas::SceneEntity> entities;

            foxglove::schemas::SceneEntity entity =
                detail::create_scene_entity("covariance", "odom", output->timestamp_ns);

            const auto position = detail::get_tracker_position(*output);

            // Sphere size indicates uncertainty (placeholder)
            const double uncertainty_scale = 0.2;
            entity.spheres.push_back(
                detail::create_sphere(
                    detail::create_identity_pose(position),
                    detail::SPHERE_SIZE_DEFAULT * uncertainty_scale,
                    detail::create_color(1.0, 0.5, 0.0, 0.2)));

            entities.push_back(entity);

            if (!entities.empty()) {
                FilterCovarianceSceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });

    // L3: Viterbi smoother trellis
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

            ViterbiTrellisMessage msg;
            msg.data = json_to_bytes(viterbi_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L3: State machine status
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

            StateMachineMessage msg;
            msg.data = json_to_bytes(sm_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // ============================================================================
    // L4 Debug Visualization Systems
    // ============================================================================

    // L4: Full MPC prediction horizon
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

            MpcHorizonMessage msg;
            msg.data = json_to_bytes(horizon_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L4: MPC cost evolution
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

            MpcCostMessage msg;
            msg.data = json_to_bytes(cost_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L4: MPC constraint margins
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

            MpcConstraintMessage msg;
            msg.data = json_to_bytes(constraint_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // L4: MPC prediction 3D scene (entire horizon trajectory)
    app.template add_system<talos::pool_compute>(
        "foxglove_mpc_prediction_scene_pub",
        [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
           talos::spmc<L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return;
            }

            auto tracker_out = tracker_in.read_current();
            if (!tracker_out || !tracker_out->is_tracking()) {
                return;
            }

            std::vector<foxglove::schemas::SceneEntity> entities;

            // Reference trajectory (what MPC tries to track)
            foxglove::schemas::SceneEntity ref_entity =
                detail::create_scene_entity("mpc_reference", "odom", plan->timestamp_ns);

            for (size_t k = 0; k < plan->ref_yaw.size(); ++k) {
                const double t = (static_cast<int>(k) - plan->mpc_current_idx) * plan->mpc_dt;
                const double distance = plan->distance + t * 5.0;
                const double x =
                    distance * std::cos(plan->ref_pitch[k]) * std::cos(plan->ref_yaw[k]);
                const double y =
                    distance * std::cos(plan->ref_pitch[k]) * std::sin(plan->ref_yaw[k]);
                const double z = distance * std::sin(plan->ref_pitch[k]);

                const auto pose = detail::create_identity_pose(detail::create_vector3(x, y, z));

                const double alpha = (k == static_cast<size_t>(plan->mpc_current_idx)) ? 1.0 : 0.3;
                ref_entity.spheres.push_back(
                    detail::create_sphere(
                        pose, detail::SPHERE_SIZE_TINY,
                        detail::create_color(0.0, 0.5, 1.0, alpha)));
            }
            entities.push_back(ref_entity);

            // Optimized trajectory (MPC output)
            foxglove::schemas::SceneEntity opt_entity =
                detail::create_scene_entity("mpc_optimized", "odom", plan->timestamp_ns);

            for (size_t k = 0; k < plan->opt_yaw.size(); ++k) {
                const double t = (static_cast<int>(k) - plan->mpc_current_idx) * plan->mpc_dt;
                const double distance = plan->distance + t * 5.0;
                const double x =
                    distance * std::cos(plan->opt_pitch[k]) * std::cos(plan->opt_yaw[k]);
                const double y =
                    distance * std::cos(plan->opt_pitch[k]) * std::sin(plan->opt_yaw[k]);
                const double z = distance * std::sin(plan->opt_pitch[k]);

                const auto pose = detail::create_identity_pose(detail::create_vector3(x, y, z));

                const double alpha = (k == static_cast<size_t>(plan->mpc_current_idx)) ? 1.0 : 0.5;
                opt_entity.spheres.push_back(
                    detail::create_sphere(
                        pose, detail::SPHERE_SIZE_TINY,
                        detail::create_color(0.0, 1.0, 0.0, alpha)));
            }
            entities.push_back(opt_entity);

            // Current target with highlighted marker
            foxglove::schemas::SceneEntity current_entity =
                detail::create_scene_entity("mpc_current", "odom", plan->timestamp_ns);

            const double distance = plan->distance;
            const double x        = distance * std::cos(plan->pitch) * std::cos(plan->yaw);
            const double y        = distance * std::cos(plan->pitch) * std::sin(plan->yaw);
            const double z        = distance * std::sin(plan->pitch);

            current_entity.spheres.push_back(
                detail::create_sphere(
                    detail::create_identity_pose(detail::create_vector3(x, y, z)),
                    detail::SPHERE_SIZE_DEFAULT, detail::COLOR_GREEN));
            entities.push_back(current_entity);

            if (!entities.empty()) {
                MpcPredictionSceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });

    // 性能统计发布系统 (1 Hz, exclusive_silent)
    if (scheduler_ptr_) {
        app.template add_system<talos::exclusive_silent<100>>(
            "foxglove_perf_stats_pub",
            [scheduler_ptr_](talos::res<std::shared_ptr<FoxgloveServer>> server) {
                auto& server_ptr = *server; // 引用绑定
                if (!server_ptr->is_initialized()) {
                    return;
                }

                // 获取 scheduler 的性能统计 JSON
                std::string json_str = scheduler_ptr_->get_stats_json();

                // 发送到 foxglove
                PerfStatsMessage msg;
                msg.data = json_to_bytes(json_str);
                server_ptr->enqueue_message(std::move(msg));
            });
    }
}

} // namespace fcs
