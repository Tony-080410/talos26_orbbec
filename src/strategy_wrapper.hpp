#pragma once

#include "strategy.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <variant>

#include <nlohmann/json.hpp>

// 很难绷得住
#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Weverything"
# pragma clang diagnostic ignored "-Werror"
# pragma clang diagnostic ignored "-Wdocumentation"
#endif

#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

#include <fast_tf/infantry_tf.hpp>
#include <spdlog/spdlog.h>

// 消息队列消息类型定义
struct ImageMessage {
    foxglove::schemas::CompressedImage img;
};

struct SceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

struct TfMessage {
    foxglove::schemas::FrameTransforms tf;
};

struct ArmorDataMessage {
    std::vector<uint8_t> data;  // JSON序列化数据
};

// 新增: 调试图像消息
struct BinaryImageMessage {
    foxglove::schemas::CompressedImage img;
};

struct PatternImageMessage {
    foxglove::schemas::CompressedImage img;
};

// 新增: 调试数据消息 (JSON)
struct DebugLightsMessage {
    std::vector<uint8_t> data;
};

struct DebugArmorsMessage {
    std::vector<uint8_t> data;
};

// 新增: 解算/追踪消息 (JSON)
struct MeasurementMessage {
    std::vector<uint8_t> data;
};

struct TargetMessage {
    std::vector<uint8_t> data;
};

struct GimbalCmdMessage {
    std::vector<uint8_t> data;
};

using FoxgloveMessage = std::variant<
    ImageMessage,
    SceneMessage,
    TfMessage,
    ArmorDataMessage,
    BinaryImageMessage,
    PatternImageMessage,
    DebugLightsMessage,
    DebugArmorsMessage,
    MeasurementMessage,
    TargetMessage,
    GimbalCmdMessage
>;

// FoxgloveServer 类声明
class FoxgloveServer {
public:
    explicit FoxgloveServer(uint16_t port = 8765, std::string host = "0.0.0.0") noexcept;
    ~FoxgloveServer() noexcept;

    // 禁止拷贝和移动
    FoxgloveServer(const FoxgloveServer&)            = delete;
    FoxgloveServer& operator=(const FoxgloveServer&) = delete;
    FoxgloveServer(FoxgloveServer&&)                 = delete;
    FoxgloveServer& operator=(FoxgloveServer&&)      = delete;

    bool is_initialized() const noexcept;

    // 线程安全的消息入队方法 - 供其他调用者使用
    void enqueue_message(FoxgloveMessage msg) noexcept;

    // TF 更新方法 (使用统一时间戳发布完整 TF 树)
    void publish_tf(double gimbal_roll, double gimbal_pitch, double gimbal_yaw, uint64_t timestamp_ns) noexcept;

    // 相机标定信息发布
    void publish_camera_calibration(
        uint32_t width, uint32_t height,
        const std::array<double, 9>& camera_matrix,  // row-major [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        const std::vector<double>& distortion,       // [k1, k2, p1, p2, k3]
        uint64_t timestamp_ns = 0) noexcept;

private:
    // 消息发送线程的主函数
    void message_sender_thread();

    bool server_initialized_{false};
    std::atomic<bool> should_stop_{false};

    std::optional<foxglove::WebSocketServer> server_;
    std::optional<foxglove::schemas::FrameTransformsChannel> tf_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> scene_ch_;
    std::optional<foxglove::schemas::CompressedImageChannel> img_ch_;
    std::optional<foxglove::RawChannel> armor_ch_;  // 用于发送装甲板JSON数据

    // 新增: 调试 channels
    std::optional<foxglove::schemas::CompressedImageChannel> binary_img_ch_;   // /debug/binary_img
    std::optional<foxglove::schemas::CompressedImageChannel> pattern_img_ch_;  // /debug/number_img
    std::optional<foxglove::RawChannel> debug_lights_ch_;   // /debug/lights (JSON)
    std::optional<foxglove::RawChannel> debug_armors_ch_;   // /debug/armors (JSON)

    // 新增: 解算/追踪 channels
    std::optional<foxglove::RawChannel> measurement_ch_;    // /solver/measurement (JSON)
    std::optional<foxglove::RawChannel> target_ch_;         // /solver/target (JSON)
    std::optional<foxglove::RawChannel> cmd_gimbal_ch_;     // /solver/cmd_gimbal (JSON)

    // 相机标定 channel
    std::optional<foxglove::schemas::CameraCalibrationChannel> camera_calib_ch_;  // /camera_info

    // TF Manager
    std::unique_ptr<fast_tf::infantry::InfantryTFManager> tf_manager_;

    // 消息队列
    alignas(std::hardware_destructive_interference_size)
        tbb::concurrent_queue<FoxgloveMessage> message_queue_;
    std::thread sender_thread_;
};

template <scheduler::SchedulerStrategy T>
struct FoxgloveWrapper {
    std::reference_wrapper<T> inner;
    std::reference_wrapper<FoxgloveServer> server;

    // 构造函数：需要传入策略和FoxgloveServer的引用
    FoxgloveWrapper(T& inner_, FoxgloveServer& server_) noexcept
        : inner(inner_)
        , server(server_) {}

    auto output() noexcept -> void {
        inner.get().output();

        // 发布云台控制命令 (JSON)
        const auto& cmd = inner.get().get_last_gimbal_cmd();
        {
            nlohmann::json cmd_json;
            cmd_json["valid"] = cmd.valid;
            cmd_json["fire_advice"] = cmd.fire_advice;
            cmd_json["yaw"] = cmd.yaw;
            cmd_json["pitch"] = cmd.pitch;
            cmd_json["distance"] = cmd.distance;

            std::string json_str = cmd_json.dump();
            GimbalCmdMessage cmd_msg;
            cmd_msg.data = std::vector(
                reinterpret_cast<const uint8_t*>(json_str.data()),
                reinterpret_cast<const uint8_t*>(json_str.data() + json_str.size()));
            server.get().enqueue_message(std::move(cmd_msg));
        }

        // 更新 TF (使用实际 IMU 姿态，统一时间戳)
        uint64_t tf_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        server.get().publish_tf(
            inner.get().get_gimbal_roll(),
            inner.get().get_gimbal_pitch(),
            inner.get().get_gimbal_yaw(),
            tf_ts);
    }

    auto detect(const scheduler::Captured& feat, talos::ArmorSlots& armor) noexcept -> void {
        inner.get().detect(feat, armor);

        // 首次发布相机标定信息

        server.get().publish_camera_calibration(
            inner.get().get_image_width(), inner.get().get_image_height(),
            inner.get().get_camera_matrix(), inner.get().get_distortion(), feat.timestamp_ns);


        // 第一步：完整克隆图像数据(防止数据竞争)
        auto img_bgr = feat.image->mat().clone();

        // 绘制装甲板边框和标签
        for (const auto& a : armor.armors) {
            if (!armor.is_valid(static_cast<int>(&a - armor.armors.data()))) continue;
            auto tl = cv::Point2f(a.pts[0], a.pts[1]);
            auto tr = cv::Point2f(a.pts[2], a.pts[3]);
            auto br = cv::Point2f(a.pts[4], a.pts[5]);
            auto bl = cv::Point2f(a.pts[6], a.pts[7]);
            cv::line(img_bgr, tl, tr, cv::Scalar(0, 255, 0), 2);
            cv::line(img_bgr, tr, br, cv::Scalar(0, 255, 0), 2);
            cv::line(img_bgr, br, bl, cv::Scalar(0, 255, 0), 2);
            cv::line(img_bgr, bl, tl, cv::Scalar(0, 255, 0), 2);

            const auto type  = a.type == talos::ArmorType::big ? "big" : "small";
            const auto name  = talos::armor_name(a.name);
            const auto color = talos::color_name(a.color);
            cv::putText(
                img_bgr, fmt::format("{} {} {}", name, type, color), tl, cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 0, 0), 1);
        }

        // 第二步：使用OpenCV的imencode方法压缩为JPEG格式
        std::vector<uint8_t> compressed_data;
        std::vector encode_params{cv::IMWRITE_JPEG_QUALITY, 85};

        bool encode_success = cv::imencode(".jpg", img_bgr, compressed_data, encode_params);
        if (!encode_success) {
            spdlog::warn("Failed to encode image to JPEG");
            return;
        }

        // 第三步：构造图像消息
        ImageMessage img_msg;
        img_msg.img.timestamp = fxglv_ts_from_ns(feat.timestamp_ns);
        img_msg.img.frame_id  = "camera_optical_frame";
        img_msg.img.format    = "jpeg";
        img_msg.img.data      = std::vector(
            reinterpret_cast<const std::byte*>(compressed_data.data()),
            reinterpret_cast<const std::byte*>(compressed_data.data() + compressed_data.size()));
        server.get().enqueue_message(std::move(img_msg));

        // ========== 调试数据发布 ==========

        // 发布二值化图像 (preprocessed)
        if (feat.preprocessed && feat.preprocessed->rows() > 0) {
            std::vector<uint8_t> binary_compressed;
            cv::Mat binary_bgr;
            cv::cvtColor(feat.preprocessed->mat(), binary_bgr, cv::COLOR_GRAY2BGR);
            if (cv::imencode(".jpg", binary_bgr, binary_compressed, encode_params)) {
                BinaryImageMessage binary_msg;
                binary_msg.img.timestamp = fxglv_ts_from_ns(feat.timestamp_ns);
                binary_msg.img.frame_id  = "camera_optical_frame";
                binary_msg.img.format    = "jpeg";
                binary_msg.img.data      = std::vector(
                    reinterpret_cast<const std::byte*>(binary_compressed.data()),
                    reinterpret_cast<const std::byte*>(binary_compressed.data() + binary_compressed.size()));
                server.get().enqueue_message(std::move(binary_msg));
            }
        }

        // 获取检测器工作缓冲区 (用于 pattern 和详细调试信息)
        const auto& buffers = inner.get().get_detector_buffers();

        // 发布数字识别图案 (第一个装甲板的 pattern)
        if (!buffers.armors.empty()) {
            const auto& first_armor = buffers.armors.front();
            if (!first_armor.pattern.empty()) {
                std::vector<uint8_t> pattern_compressed;
                cv::Mat pattern_bgr;
                if (first_armor.pattern.channels() == 1) {
                    cv::cvtColor(first_armor.pattern, pattern_bgr, cv::COLOR_GRAY2BGR);
                } else {
                    pattern_bgr = first_armor.pattern;
                }
                if (cv::imencode(".jpg", pattern_bgr, pattern_compressed, encode_params)) {
                    PatternImageMessage pattern_msg;
                    pattern_msg.img.timestamp = fxglv_ts_from_ns(feat.timestamp_ns);
                    pattern_msg.img.frame_id  = "camera_optical_frame";
                    pattern_msg.img.format    = "jpeg";
                    pattern_msg.img.data      = std::vector(
                        reinterpret_cast<const std::byte*>(pattern_compressed.data()),
                        reinterpret_cast<const std::byte*>(pattern_compressed.data() + pattern_compressed.size()));
                    server.get().enqueue_message(std::move(pattern_msg));
                }
            }
        }

        // 发布灯条调试信息 (JSON)
        {
            nlohmann::json lights_json;
            lights_json["timestamp_ns"] = feat.timestamp_ns;
            lights_json["lights"] = nlohmann::json::array();
            for (const auto& light : buffers.lightbars) {
                nlohmann::json light_obj;
                light_obj["x"] = light.center.x;
                light_obj["y"] = light.center.y;
                light_obj["angle"] = light.angle;
                light_obj["angle_error"] = light.angle_error;
                light_obj["length"] = light.length;
                light_obj["width"] = light.width;
                light_obj["ratio"] = light.ratio;
                light_obj["color"] = std::string(talos::color_name(light.color));
                lights_json["lights"].push_back(light_obj);
            }
            std::string json_str = lights_json.dump();
            DebugLightsMessage lights_msg;
            lights_msg.data = std::vector(
                reinterpret_cast<const uint8_t*>(json_str.data()),
                reinterpret_cast<const uint8_t*>(json_str.data() + json_str.size()));
            server.get().enqueue_message(std::move(lights_msg));
        }

        // 发布装甲板详细调试信息 (JSON)
        {
            nlohmann::json armors_json;
            armors_json["timestamp_ns"] = feat.timestamp_ns;
            armors_json["armors"] = nlohmann::json::array();
            for (const auto& a : buffers.armors) {
                nlohmann::json armor_obj;
                armor_obj["center"]["x"] = a.center.x;
                armor_obj["center"]["y"] = a.center.y;
                armor_obj["pts"] = nlohmann::json::array();
                for (const auto& pt : a.points) {
                    armor_obj["pts"].push_back({pt.x, pt.y});
                }
                armor_obj["ratio"] = a.ratio;
                armor_obj["side_ratio"] = a.side_ratio;
                armor_obj["rectangular_error"] = a.rectangular_error;
                armor_obj["confidence"] = a.confidence;
                armor_obj["color"] = std::string(talos::color_name(a.color));
                armor_obj["type"] = a.type == talos::ArmorType::big ? "big" : "small";
                armor_obj["name"] = std::string(talos::armor_name(a.name));
                armor_obj["duplicated"] = a.duplicated;
                armors_json["armors"].push_back(armor_obj);
            }
            std::string json_str = armors_json.dump();
            DebugArmorsMessage armors_msg;
            armors_msg.data = std::vector(
                reinterpret_cast<const uint8_t*>(json_str.data()),
                reinterpret_cast<const uint8_t*>(json_str.data() + json_str.size()));
            server.get().enqueue_message(std::move(armors_msg));
        }
    }

    auto track(
        talos::TrackerSoA& tracker_soa, const talos::ArmorSlots& armor,
        const auto_aim::Poses3D& poses3d) noexcept -> void {
        inner.get().track(tracker_soa, armor, poses3d);

        // 发布跟踪目标信息 (JSON)
        {
            auto& target = inner.get().get_target();
            nlohmann::json target_json;
            target_json["timestamp_ns"] = armor.timestamp_ns;
            target_json["state"] = std::string(talos::tracker_state(tracker_soa.state.load()));
            target_json["position"]["x"] = tracker_soa.pos[0];
            target_json["position"]["y"] = tracker_soa.pos[1];
            target_json["position"]["z"] = tracker_soa.pos[2];
            target_json["velocity"]["x"] = tracker_soa.vel[0];
            target_json["velocity"]["y"] = tracker_soa.vel[1];
            target_json["velocity"]["z"] = tracker_soa.vel[2];
            target_json["yaw"] = tracker_soa.yaw;
            target_json["v_yaw"] = tracker_soa.v_yaw;
            target_json["radius"] = {tracker_soa.radius[0], tracker_soa.radius[1]};
            target_json["detecting_count"] = tracker_soa.detecting_count;
            target_json["lost_count"] = tracker_soa.lost_count;

            // 装甲板列表 (从 Target 获取)
            target_json["armor_list"] = nlohmann::json::array();
            if (target.converged()) {
                const auto armor_list = target.armor_xyza_list();
                for (const auto& xyza : armor_list) {
                    target_json["armor_list"].push_back({
                        {"x", xyza[0]},
                        {"y", xyza[1]},
                        {"z", xyza[2]},
                        {"angle", xyza[3]}
                    });
                }
            }

            std::string json_str = target_json.dump();
            TargetMessage target_msg;
            target_msg.data = std::vector(
                reinterpret_cast<const uint8_t*>(json_str.data()),
                reinterpret_cast<const uint8_t*>(json_str.data() + json_str.size()));
            server.get().enqueue_message(std::move(target_msg));
        }
    }

    auto solve(const talos::ArmorSlots& armor, auto_aim::Poses3D& poses3d) noexcept -> void {
        inner.get().solve(armor, poses3d);

        std::vector<foxglove::schemas::SceneEntity> scene_entities;
        for (int i = 0; i < 10; i++) {
            if (!armor.is_valid(i)) continue;

            const auto& a = armor.armors[i];
            const auto& pose = poses3d[i];

            // 创建SceneEntity
            foxglove::schemas::SceneEntity entity;
            entity.id = "armor_" + std::to_string(i);
            entity.frame_id = "gimbal_link";  // 使用云台坐标系
            entity.timestamp = fxglv_ts_from_ns(armor.timestamp_ns);

            // 构造位置和方向 (使用云台坐标系)
            foxglove::schemas::Vector3 position;
            position.x = pose.xyz_in_gimbal[0];
            position.y = pose.xyz_in_gimbal[1];
            position.z = pose.xyz_in_gimbal[2];

            // yaw pitch roll -> qx qy qz qw (云台坐标系下的姿态)
            // eulers(R, 2, 1, 0) 提取的是 ZYX 内旋欧拉角
            // 重建: R = Rz(yaw) * Ry(pitch) * Rx(roll)
            Eigen::AngleAxisd yawz(pose.ypr_in_gimbal[0], Eigen::Vector3d::UnitZ());
            Eigen::AngleAxisd pitchy(pose.ypr_in_gimbal[1], Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd rollx(pose.ypr_in_gimbal[2], Eigen::Vector3d::UnitX());

            Eigen::Quaterniond q = yawz * pitchy * rollx;
            foxglove::schemas::Quaternion orientation;
            orientation.x = q.x();
            orientation.y = q.y();
            orientation.z = q.z();
            orientation.w = q.w();

            foxglove::schemas::Pose p;
            p.position = position;
            p.orientation = orientation;

            // 创建立方体 (装甲板)
            foxglove::schemas::Vector3 cube_size;
            if (a.type == talos::ArmorType::big) {
                cube_size.z = 0.225;
                cube_size.y = 0.27;
                cube_size.x = 0.02;
            } else {
                cube_size.z = 0.115;
                cube_size.y = 0.27;
                cube_size.x = 0.02;
            }

            foxglove::schemas::CubePrimitive cube;
            cube.pose = p;
            cube.size = cube_size;
            cube.color = fxgl_get_color(a.color);
            entity.cubes.push_back(cube);

            // 创建文本标签
            foxglove::schemas::TextPrimitive text;
            text.text = std::string(talos::armor_name(a.name)) + " " +
                       std::string(a.type == talos::ArmorType::big ? "big" : "small");
            text.pose = p;
            text.font_size = 0.05;
            foxglove::schemas::Color text_color;
            text_color.r = 1.0;  // 白色
            text_color.g = 1.0;
            text_color.b = 1.0;
            text_color.a = 1.0;
            text.color = text_color;
            text.billboard = true;
            entity.texts.push_back(text);

            // 添加元数据
            {
                foxglove::schemas::KeyValuePair kv;
                kv.key = "confidence";
                kv.value = std::to_string(a.confidence);
                entity.metadata.push_back(kv);
            }
            {
                foxglove::schemas::KeyValuePair kv;
                kv.key = "color";
                kv.value = std::string(talos::color_name(a.color));
                entity.metadata.push_back(kv);
            }

            scene_entities.push_back(entity);
        }

        // 发送场景消息
        if (!scene_entities.empty()) {
            SceneMessage scene_msg;
            scene_msg.scene.entities = std::move(scene_entities);
            server.get().enqueue_message(std::move(scene_msg));
        }

        // 第五步：构造和发送JSON装甲板数据
        nlohmann::json armor_json = nlohmann::json::array();
        for (int i = 0; i < 10; i++) {
            if (!armor.is_valid(i)) continue;

            const auto& a = armor.armors[i];
            const auto& pose = poses3d[i];

            nlohmann::json armor_obj;
            armor_obj["id"] = i;
            armor_obj["name"] = std::string(talos::armor_name(a.name));
            armor_obj["type"] = a.type == talos::ArmorType::big ? "big" : "small";
            armor_obj["color"] = std::string(talos::color_name(a.color));
            armor_obj["confidence"] = a.confidence;
            armor_obj["pose"]["position"]["x"] = pose.xyz_in_world[0];
            armor_obj["pose"]["position"]["y"] = pose.xyz_in_world[1];
            armor_obj["pose"]["position"]["z"] = pose.xyz_in_world[2];
            armor_obj["pose"]["euler"]["yaw"] = pose.ypr_in_world[0];
            armor_obj["pose"]["euler"]["pitch"] = pose.ypr_in_world[1];
            armor_obj["pose"]["euler"]["roll"] = pose.ypr_in_world[2];

            armor_json.push_back(armor_obj);
        }

        // 发送JSON数据
        if (!armor_json.empty()) {
            std::string json_str = armor_json.dump();
            ArmorDataMessage armor_msg;
            armor_msg.data = std::vector(
                reinterpret_cast<const uint8_t*>(json_str.data()),
                reinterpret_cast<const uint8_t*>(json_str.data() + json_str.size()));
            server.get().enqueue_message(std::move(armor_msg));
        }

        // 发布测量值 (measurement) - 每个有效装甲板的3D位姿
        {
            nlohmann::json measurement_json;
            measurement_json["timestamp_ns"] = armor.timestamp_ns;
            measurement_json["measurements"] = nlohmann::json::array();
            for (int i = 0; i < 10; i++) {
                if (!armor.is_valid(i)) continue;
                const auto& pose = poses3d[i];
                nlohmann::json m;
                m["id"] = i;
                m["xyz_in_gimbal"] = {pose.xyz_in_gimbal[0], pose.xyz_in_gimbal[1], pose.xyz_in_gimbal[2]};
                m["xyz_in_world"] = {pose.xyz_in_world[0], pose.xyz_in_world[1], pose.xyz_in_world[2]};
                m["ypr_in_gimbal"] = {pose.ypr_in_gimbal[0], pose.ypr_in_gimbal[1], pose.ypr_in_gimbal[2]};
                m["ypr_in_world"] = {pose.ypr_in_world[0], pose.ypr_in_world[1], pose.ypr_in_world[2]};
                m["ypd_in_world"] = {pose.ypd_in_world[0], pose.ypd_in_world[1], pose.ypd_in_world[2]};
                m["yaw_raw"] = pose.yaw_raw;
                measurement_json["measurements"].push_back(m);
            }
            std::string json_str = measurement_json.dump();
            MeasurementMessage measurement_msg;
            measurement_msg.data = std::vector(
                reinterpret_cast<const uint8_t*>(json_str.data()),
                reinterpret_cast<const uint8_t*>(json_str.data() + json_str.size()));
            server.get().enqueue_message(std::move(measurement_msg));
        }
    }

    auto preprocess(scheduler::Captured& sf) -> void { inner.get().preprocess(sf); }

    // 颜色映射函数
    static foxglove::schemas::Color fxgl_get_color(const talos::Color c) noexcept {
        foxglove::schemas::Color color;
        color.a = 0.3;
        switch(c) {
            case talos::Color::red:
                color.r = 1.0; color.g = 0.0; color.b = 0.0;
                break;
            case talos::Color::blue:
                color.r = 0.0; color.g = 0.0; color.b = 1.0;
                break;
            case talos::Color::extinguish:
                color.r = 1.0; color.g = 1.0; color.b = 0.0;  // 黄色
                break;
            case talos::Color::purple:
                color.r = 1.0; color.g = 0.0; color.b = 1.0;  // 紫色
                break;
            default:
                color.r = 0.5; color.g = 0.5; color.b = 0.5;  // 灰色
        }
        return color;
    }

    static foxglove::schemas::Timestamp fxglv_ts_from_ns(const uint64_t ns) {
        using namespace std::chrono;
        foxglove::schemas::Timestamp t;
        t.nsec = static_cast<uint32_t>(ns);
        t.sec  = static_cast<int32_t>(ns / 1'000'000'000);
        return t;
    }
};

static_assert(scheduler::SchedulerStrategy<FoxgloveWrapper<TalosStrategy>>);
