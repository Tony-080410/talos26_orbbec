#pragma once

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

#include <Eigen/Geometry>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>

namespace fcs {

// ============================================================================
// Foxglove Publisher Systems
// ============================================================================

/// 注册所有 Foxglove 发布系统
template <typename App>
void register_foxglove_systems(App& app, talos::Scheduler* scheduler_ptr = nullptr) {
    // 图像发布系统 (同时发布 camera_calibration 和 TF)
    app.template add_system<talos::pool_compute>(
        "foxglove_image_pub",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTag> img_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server, talos::res<fcs::CameraConfig> cam,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer) {
            if (!(*server)->is_initialized() || !img_in.has_new()) {
                return;
            }

            auto frame = img_in.read();
            if (!frame || frame->image.empty()) {
                return;
            }

            // 克隆图像避免数据竞争
            cv::Mat img_bgr = frame->image.clone();

            // 压缩为 JPEG
            std::vector<uint8_t> compressed;
            std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 85};
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

            // 发布 TF
            (*server)->publish_tf(**tf_buffer, frame->timestamp_ns);
        });

    // 检测结果发布系统
    app.template add_system<talos::pool_compute>(
        "foxglove_detection_pub",
        [](talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> det_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!(*server)->is_initialized() || !det_in.has_new()) {
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
            if (!(*server)->is_initialized() || !meas_in.has_new()) {
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

                // 创建 3D 实体
                foxglove::schemas::SceneEntity entity;
                entity.id        = "armor_" + std::to_string(i);
                entity.frame_id  = "odom";
                entity.lifetime  = foxglove::schemas::Duration{.nsec = 100000000};
                entity.timestamp = timestamp_from_ns(batch->timestamp_ns);

                foxglove::schemas::Vector3 position;
                position.x = translation.x();
                position.y = translation.y();
                position.z = translation.z();

                const auto q = m.transform->quaternion();
                foxglove::schemas::Quaternion orientation;
                orientation.x = q.x();
                orientation.y = q.y();
                orientation.z = q.z();
                orientation.w = q.w();

                foxglove::schemas::Pose pose;
                pose.position    = position;
                pose.orientation = orientation;

                foxglove::schemas::CubePrimitive cube;
                cube.pose = pose;
                if (m.type == ArmorType::Large) {
                    cube.size = {0.03, 0.15, 0.125};   // 装甲板尺寸
                } else {
                    cube.size = {0.03, 0.15, 0.125};
                }
                cube.color = m.color == ArmorColor::Red
                               ? foxglove::schemas::Color{1.0, 0.0, 0.0, 0.5}
                               : foxglove::schemas::Color{0.0, 0.0, 1.0, 0.5};
                entity.cubes.push_back(cube);

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = pose;
                if (m.type == ArmorType::Large) {
                    sphere.size = {0.03, 0.15, 0.125}; // 装甲板尺寸
                } else {
                    sphere.size = {0.03, 0.15, 0.125};
                }
                sphere.color = m.color == ArmorColor::Red
                                 ? foxglove::schemas::Color{1.0, 0.5, 0.0, 0.5}
                                 : foxglove::schemas::Color{0.0, 0.5, 1.0, 0.5};
                entity.spheres.push_back(sphere);

                // 创建文本标签
                foxglove::schemas::TextPrimitive text;
                text.text = std::string(to_string(m.name)) + " " + std::string(to_string(m.type));
                text.pose = pose;
                if (text.pose.has_value() && text.pose->position.has_value()) {
                    text.pose->position->y -= 0.1;
                }
                text.font_size = 0.05;
                foxglove::schemas::Color text_color;
                text_color.r   = 1.0; // 白色
                text_color.g   = 1.0;
                text_color.b   = 1.0;
                text_color.a   = 0.9;
                text.color     = text_color;
                text.billboard = true;
                entity.texts.push_back(text);

                // 添加元数据
                {
                    foxglove::schemas::KeyValuePair kv;
                    kv.key   = "confidence";
                    kv.value = std::to_string(m.confidence);
                    entity.metadata.push_back(kv);
                }
                {
                    foxglove::schemas::KeyValuePair kv;
                    kv.key   = "color";
                    kv.value = std::string(to_string(m.color));
                    entity.metadata.push_back(kv);
                }

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
            if (!(*server)->is_initialized() || !tracker_in.has_new()) {
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
                target_json["position"] = {state.position.x(), state.position.y()};
                target_json["yaw"]      = state.yaw;
                target_json["v_yaw"]    = state.v_yaw;
                target_json["z"]        = {state.z[0], state.z[1], state.z[2]};
            }

            TargetMessage msg;
            msg.data = json_to_bytes(target_json.dump());
            (*server)->enqueue_message(std::move(msg));

            // 发布跟踪场景 (预测装甲板位置)
            if (output->is_tracking()) {
                std::vector<foxglove::schemas::SceneEntity> entities;

                foxglove::schemas::SceneEntity entity;
                entity.id        = "target";
                entity.frame_id  = "odom";
                entity.lifetime  = foxglove::schemas::Duration{.nsec = 100000000};
                entity.timestamp = timestamp_from_ns(output->timestamp_ns);

                foxglove::schemas::Vector3 position;
                if (output->is_robot()) {
                    const auto& state = *output->robot_state();
                    position.x        = state.position.x();
                    position.y        = state.position.y();
                    position.z        = state.position.z();
                } else if (output->is_outpost()) {
                    const auto& state = *output->outpost_state();
                    position.x        = state.position.x();
                    position.y        = state.position.y();
                    position.z        = (state.z[0] + state.z[1] + state.z[2]) / 3;
                }

                foxglove::schemas::Pose pose;
                pose.position    = position;
                pose.orientation = foxglove::schemas::Quaternion{0, 0, 0, 1};

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose  = pose;
                sphere.size  = {0.1, 0.1, 0.1};
                sphere.color = foxglove::schemas::Color{0.0, 1.0, 0.0, 0.6};
                entity.spheres.push_back(sphere);

                // 绘制速度箭头
                if (output->is_robot()) {
                    const auto& state = *output->robot_state();

                    // 角速度箭头 (沿z轴, 长度为 v_yaw / π)
                    double angular_vel_length = std::abs(state.v_yaw) / M_PI;
                    if (angular_vel_length > 0.01) { // 最小阈值避免过小箭头
                        foxglove::schemas::ArrowPrimitive angular_arrow;
                        angular_arrow.pose = foxglove::schemas::Pose();
                        angular_arrow.pose->position.emplace();
                        angular_arrow.pose->position->x = position.x;
                        angular_arrow.pose->position->y = position.y;
                        angular_arrow.pose->position->z = position.z;
                        // 箭头沿z轴向上 (或向下如果角速度为负)
                        if (state.v_yaw >= 0) {
                            angular_arrow.pose->orientation = {
                                0, -0.7071, 0, 0.7071}; // 绕Y轴旋转-90度，指向+Z
                        } else {
                            angular_arrow.pose->orientation = {
                                0, 0.7071, 0, 0.7071};  // 绕Y轴旋转90度，指向-Z
                        }
                        angular_arrow.shaft_length   = angular_vel_length * 0.8;
                        angular_arrow.shaft_diameter = 0.02;
                        angular_arrow.head_length    = angular_vel_length * 0.2;
                        angular_arrow.head_diameter  = 0.04;
                        angular_arrow.color = foxglove::schemas::Color{1.0, 1.0, 0.0, 0.8}; // 黄色
                        entity.arrows.push_back(angular_arrow);
                    }

                    // 线速度箭头 (方向为速度方向, 长度为速度模)
                    double linear_vel_length = state.velocity.norm();
                    if (linear_vel_length > 0.0001) { // 最小阈值
                        foxglove::schemas::ArrowPrimitive linear_arrow;
                        linear_arrow.pose = foxglove::schemas::Pose();
                        linear_arrow.pose->position.emplace();
                        linear_arrow.pose->position->x = position.x;
                        linear_arrow.pose->position->y = position.y;
                        linear_arrow.pose->position->z = position.z;
                        // 计算速度方向的四元数
                        Eigen::Vector3d vel_dir = state.velocity.normalized();
                        Eigen::Vector3d default_dir(1, 0, 0); // 箭头默认指向 +X
                        Eigen::Quaterniond q;
                        if ((vel_dir + default_dir).norm() < 1e-6) {
                            // 速度方向与默认方向相反
                            q = Eigen::Quaterniond(
                                Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
                        } else {
                            q = Eigen::Quaterniond::FromTwoVectors(default_dir, vel_dir);
                        }
                        linear_arrow.pose->orientation = {q.x(), q.y(), q.z(), q.w()};
                        linear_arrow.shaft_length      = linear_vel_length * 0.8;
                        linear_arrow.shaft_diameter    = 0.02;
                        linear_arrow.head_length       = linear_vel_length * 0.2;
                        linear_arrow.head_diameter     = 0.04;
                        linear_arrow.color = foxglove::schemas::Color{0.0, 1.0, 1.0, 0.8}; // 青色
                        entity.arrows.push_back(linear_arrow);
                    }
                } else if (output->is_outpost()) {
                    const auto& state = *output->outpost_state();

                    // 前哨站角速度箭头 (沿z轴)
                    double angular_vel_length = std::abs(state.v_yaw) / M_PI;
                    if (angular_vel_length > 0.0001) {
                        foxglove::schemas::ArrowPrimitive angular_arrow;
                        angular_arrow.pose = foxglove::schemas::Pose();
                        angular_arrow.pose->position.emplace();
                        angular_arrow.pose->position->x = position.x;
                        angular_arrow.pose->position->y = position.y;
                        angular_arrow.pose->position->z = position.z;
                        if (state.v_yaw >= 0) {
                            angular_arrow.pose->orientation = {0, -0.7071, 0, 0.7071}; // 指向+Z
                        } else {
                            angular_arrow.pose->orientation = {0, 0.7071, 0, 0.7071};  // 指向-Z
                        }
                        angular_arrow.shaft_length   = angular_vel_length * 0.8;
                        angular_arrow.shaft_diameter = 0.02;
                        angular_arrow.head_length    = angular_vel_length * 0.2;
                        angular_arrow.head_diameter  = 0.04;
                        angular_arrow.color = foxglove::schemas::Color{1.0, 1.0, 0.0, 0.8}; // 黄色
                        entity.arrows.push_back(angular_arrow);
                    }
                }

                if (output->is_outpost()) {
                    const auto& state = output->outpost_state();
                    auto armor_poses  = state->armor_poses();
                    static_assert(armor_poses.size() == 3);
                    for (int i = 0; i < 3; i++) {
                        foxglove::schemas::CubePrimitive cube;
                        cube.pose = foxglove::schemas::Pose();
                        cube.pose->position.emplace();
                        cube.pose->orientation.emplace();
                        cube.size.emplace(0.03, 0.135, 0.125);
                        cube.color.emplace(0.0, 1.0, 0.0, 0.3);
                        cube.pose->position->x = armor_poses[i][0];
                        cube.pose->position->y = armor_poses[i][1];
                        cube.pose->position->z = armor_poses[i][2];
                        Eigen::Quaterniond q =
                            Eigen::AngleAxisd(armor_poses[i][3], Eigen::Vector3d::UnitZ())
                            * Eigen::AngleAxisd(-0.2618, Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
                        cube.pose->orientation = {q.x(), q.y(), q.z(), q.w()};
                        entity.cubes.emplace_back(cube);
                    }
                } else if (output->is_robot()) {
                    const auto& state = output->robot_state();
                    bool is_big_armor = output->target_name == ArmorName::One
                                     || output->target_name == ArmorName::Base;
                    auto armor_poses = state->armor_poses();
                    static_assert(armor_poses.size() == 4);
                    for (int i = 0; i < 4; i++) {
                        foxglove::schemas::Vector3 position;
                        position.x = armor_poses[i][0];
                        position.y = armor_poses[i][1];
                        position.z = armor_poses[i][2];
                        Eigen::Quaterniond q =
                            Eigen::AngleAxisd(armor_poses[i][3], Eigen::Vector3d::UnitZ())
                            * Eigen::AngleAxisd(0.2618, Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
                        foxglove::schemas::Quaternion orientation = {q.x(), q.y(), q.z(), q.w()};

                        foxglove::schemas::Pose pose;
                        pose.position    = position;
                        pose.orientation = orientation;
                        foxglove::schemas::CubePrimitive cube;
                        cube.pose = pose;
                        double y  = is_big_armor ? 0.23 : 0.135;
                        cube.size.emplace(0.03, y, 0.125);
                        cube.color.emplace(0.0, 1.0, 0.0, 0.3);
                        cube.pose->position->x = armor_poses[i][0];
                        cube.pose->position->y = armor_poses[i][1];
                        cube.pose->position->z = armor_poses[i][2];
                        entity.cubes.emplace_back(cube);

                        foxglove::schemas::TextPrimitive text;
                        text.text = fmt::format("{}", i);
                        text.pose = pose;
                        text.pose->position->z += 0.1;
                        text.font_size = 0.05;
                        foxglove::schemas::Color text_color;
                        text_color.r   = 1.0;                                               // 白色
                        text_color.g   = 1.0;
                        text_color.b   = 1.0;
                        text_color.a   = 0.9;
                        text.color     = text_color;
                        text.billboard = true;
                        entity.texts.push_back(text);
                    }
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
            if (!(*server)->is_initialized() || !cmd_in.has_new()) {
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
            cmd_json["distance"]     = cmd->distance;
            cmd_json["fire_advice"]  = cmd->fire;
            cmd_json["track_center"] = cmd->track_center;

            GimbalCmdMessage msg;
            msg.data = json_to_bytes(cmd_json.dump());
            (*server)->enqueue_message(std::move(msg));
        });

    // 云台场景发布系统 (弹道/预测位置/选择点可视化)
    app.template add_system<talos::pool_compute>(
        "foxglove_gimbal_scene_pub", [](talos::spmc<L4::GimbalPlan, GimbalPlanChannelTag> cmd_in,
                                        talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!(*server)->is_initialized() || !cmd_in.has_new()) {
                return;
            }

            auto cmd = cmd_in.read();
            if (!cmd || cmd->distance <= 0) {
                return;
            }

            std::vector<foxglove::schemas::SceneEntity> entities;

            // Selection marker - 瞄准点 (黄绿色球体)
            {
                foxglove::schemas::SceneEntity entity;
                entity.id        = "selection";
                entity.frame_id  = "gimbal_link";
                entity.lifetime  = foxglove::schemas::Duration{.nsec = 100000000};
                entity.timestamp = timestamp_from_ns(cmd->timestamp_ns);

                // 根据 yaw, pitch, distance 计算瞄准点位置
                double x = cmd->distance * std::cos(cmd->pitch) * std::cos(cmd->yaw);
                double y = cmd->distance * std::cos(cmd->pitch) * std::sin(cmd->yaw);
                double z = cmd->distance * std::sin(cmd->pitch);

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = foxglove::schemas::Pose();
                sphere.pose->position.emplace();
                sphere.pose->position->x = x;
                sphere.pose->position->y = y;
                sphere.pose->position->z = z;
                sphere.pose->orientation = {0, 0, 0, 1};
                sphere.size              = {0.1, 0.1, 0.1};
                sphere.color             = foxglove::schemas::Color{1.0, 1.0, 0.0, 0.3}; // 黄绿色
                entity.spheres.push_back(sphere);
                entities.push_back(entity);
            }

            // Predicted position marker - 预测位置 (橙色球体)
            if (cmd->predicted_position) {
                foxglove::schemas::SceneEntity entity;
                entity.id        = "predicted";
                entity.frame_id  = "odom";
                entity.lifetime  = foxglove::schemas::Duration{.nsec = 100000000};
                entity.timestamp = timestamp_from_ns(cmd->predicted_future);

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose = foxglove::schemas::Pose();
                sphere.pose->position.emplace();
                sphere.pose->position->x = cmd->predicted_position->x();
                sphere.pose->position->y = cmd->predicted_position->y();
                sphere.pose->position->z = cmd->predicted_position->z();
                sphere.pose->orientation = {0, 0, 0, 1};
                sphere.size              = {0.08, 0.08, 0.08};
                sphere.color             = foxglove::schemas::Color{1.0, 0.0, 0.0, 0.6}; // 橙色
                entity.spheres.push_back(sphere);
                entities.push_back(entity);
            }

            // Trajectory marker - 弹道轨迹 (点列)
            if (!cmd->trajectory.empty()) {
                foxglove::schemas::SceneEntity entity;
                entity.id        = "trajectory";
                entity.frame_id  = "muzzle_link";
                entity.lifetime  = foxglove::schemas::Duration{.nsec = 100000000};
                entity.timestamp = timestamp_from_ns(cmd->timestamp_ns);

                // 根据 fire_advice 改变颜色
                foxglove::schemas::Color traj_color = cmd->fire_advice
                                                        ? foxglove::schemas::Color{0, 1, 0, 0.6}
                                                        : foxglove::schemas::Color{1, 1, 1, 0.3};

                for (const auto& [x, z] : cmd->trajectory) {
                    foxglove::schemas::SpherePrimitive point;
                    point.pose = foxglove::schemas::Pose();
                    point.pose->position.emplace();
                    point.pose->position->x = x;
                    point.pose->position->y = 0;
                    point.pose->position->z = z;
                    point.pose->orientation = {0, 0, 0, 1};
                    point.size              = {0.01, 0.01, 0.01};
                    point.color             = traj_color;
                    entity.spheres.push_back(point);
                }
                entities.push_back(entity);
            }

            if (!entities.empty()) {
                GimbalSceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });

    // 性能统计发布系统 (1 Hz, exclusive_silent)
    if (scheduler_ptr) {
        app.template add_system<talos::exclusive_silent<1>>(
            "foxglove_perf_stats_pub",
            [scheduler_ptr, server_ptr = std::shared_ptr<FoxgloveServer>{}](
                talos::res<std::shared_ptr<FoxgloveServer>> server) mutable {
                // 第一次调用时保存 server 指针
                if (!server_ptr) {
                    server_ptr = *server;
                }

                if (!server_ptr->is_initialized()) {
                    return;
                }

                // 获取 scheduler 的性能统计 JSON
                std::string json_str = scheduler_ptr->get_stats_json();

                // 发送到 foxglove
                PerfStatsMessage msg;
                msg.data = json_to_bytes(json_str);
                server_ptr->enqueue_message(std::move(msg));
            });
    }
}

} // namespace fcs
