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

struct ArmorMessage {};

using FoxgloveMessage = std::variant<ImageMessage, SceneMessage, TfMessage, ArmorDataMessage>;

// FoxgloveServer 类声明
class FoxgloveServer {
public:
    explicit FoxgloveServer(uint16_t port = 8765, std::string host = "127.0.0.1") noexcept;
    ~FoxgloveServer() noexcept;

    // 禁止拷贝和移动
    FoxgloveServer(const FoxgloveServer&)            = delete;
    FoxgloveServer& operator=(const FoxgloveServer&) = delete;
    FoxgloveServer(FoxgloveServer&&)                 = delete;
    FoxgloveServer& operator=(FoxgloveServer&&)      = delete;

    bool is_initialized() const noexcept;

    // 线程安全的消息入队方法 - 供其他调用者使用
    void enqueue_message(FoxgloveMessage msg) noexcept;

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

    auto output() noexcept -> void { inner.get().output(); }

    auto detect(const scheduler::SharedFeature& feat, talos::ArmorSlots& armor) noexcept -> void {
        inner.get().detect(feat, armor);

        // 第一步：完整克隆图像数据(防止数据竞争)
        auto img_bgr = feat.image->mat().clone();
            spdlog::info("use {}",feat.frame_seq);

        // 绘制装甲板边框和标签
        for (const auto& a : armor.armors) {
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
    }

    auto track(
        talos::TrackerSoA& tracker_soa, const talos::ArmorSlots& armor,
        const auto_aim::Poses3D& poses3d) noexcept -> void {
        inner.get().track(tracker_soa, armor, poses3d);

        // 发布目标位置信息
        // TODO: 通过SceneMessage发送tracker信息
        // server.get().enqueue_message(std::move(armor_msg));
        // spdlog::debug("Target tracking completed with {} poses", poses3d.size());
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
            entity.frame_id = "world";
            entity.timestamp = fxglv_ts_from_ns(armor.timestamp_ns);

            // 构造位置和方向
            foxglove::schemas::Vector3 position;
            position.x = pose.xyz_in_world[0];
            position.y = pose.xyz_in_world[1];
            position.z = pose.xyz_in_world[2];

            // yaw pitch roll -> qx qy qz qw
            Eigen::AngleAxisd yawz(pose.ypr_in_world[0], Eigen::Vector3d::UnitZ());
            Eigen::AngleAxisd pitchy(pose.ypr_in_world[1], Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd rollx(pose.ypr_in_world[2], Eigen::Vector3d::UnitX());

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
                cube_size.x = 0.225;
                cube_size.y = 0.27;
                cube_size.z = 0.02;
            } else {
                cube_size.x = 0.115;
                cube_size.y = 0.27;
                cube_size.z = 0.02;
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
    }

    auto preprocess(scheduler::SharedFeature& sf) -> void { inner.get().preprocess(sf); }

    // 颜色映射函数
    static foxglove::schemas::Color fxgl_get_color(const talos::Color c) noexcept {
        foxglove::schemas::Color color;
        color.a = 1.0;
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
