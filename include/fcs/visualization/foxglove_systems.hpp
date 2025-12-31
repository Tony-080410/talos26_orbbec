#pragma once

#include "foxglove_server.hpp"
#include "foxglove_types.hpp"

#include "auto_aim/config.hpp"
#include "fcs/L3_estimation/armor_solver/types.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/types.hpp"
#include "math/fast_tf/infantry.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/scheduler.hpp"

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
void register_foxglove_systems(App& app) {
    // 图像发布系统 (同时发布 camera_calibration 和 TF)
    app.template add_system<talos::pool_compute>(
        "foxglove_image_pub",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTag> img_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server,
           talos::res<std::shared_ptr<TalosConfig>> config,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<64>>> tf_buffer) {
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
            const auto& cam = (*config)->camera;
            std::array<double, 9> camera_matrix_arr;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    camera_matrix_arr[i * 3 + j] = cam.camera_matrix(i, j);
                }
            }
            std::vector<double> dist_coeffs(
                cam.distort_coefficient.data(),
                cam.distort_coefficient.data() + cam.distort_coefficient.size());

            (*server)->publish_camera_calibration(
                cam.camera_info.width, cam.camera_info.height, camera_matrix_arr, dist_coeffs,
                frame->timestamp_ns);

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
                obj["center"]  = {det.center().x, det.center().y};
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
                entity.frame_id  = "camera_optical_frame";
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
                entity.id        = "target_center";
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
                    position.z        = state.z[0];
                }

                foxglove::schemas::Pose pose;
                pose.position    = position;
                pose.orientation = foxglove::schemas::Quaternion{0, 0, 0, 1};

                foxglove::schemas::SpherePrimitive sphere;
                sphere.pose  = pose;
                sphere.size  = {0.1, 0.1, 0.1};
                sphere.color = foxglove::schemas::Color{0.0, 1.0, 0.0, 0.6};
                entity.spheres.push_back(sphere);

                entities.push_back(entity);

                TrackSceneMessage scene_msg;
                scene_msg.scene.entities = std::move(entities);
                (*server)->enqueue_message(std::move(scene_msg));
            }
        });
    /*
        // 云台指令发布系统
        app.template add_system<talos::pool_compute>(
            "foxglove_gimbal_cmd_pub",
            [](talos::spmc<L3::GimbalCommand, SolverOutputChannelTag> cmd_in,
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
                cmd_json["yaw_diff"]     = cmd->yaw_diff;
                cmd_json["pitch_diff"]   = cmd->pitch_diff;
                cmd_json["distance"]     = cmd->distance;
                cmd_json["fire_advice"]  = cmd->fire_advice;

                if (cmd->predicted_position) {
                    cmd_json["predicted_position"] = {
                        cmd->predicted_position->x(),
                        cmd->predicted_position->y(),
                        cmd->predicted_position->z()};
                }

                GimbalCmdMessage msg;
                msg.data = json_to_bytes(cmd_json.dump());
                (*server)->enqueue_message(std::move(msg));
            });*/
}

} // namespace fcs
