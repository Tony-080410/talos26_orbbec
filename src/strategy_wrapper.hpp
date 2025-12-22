#pragma once

#include "strategy.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <variant>

// 很难绷得住
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Weverything"
#  pragma clang diagnostic ignored "-Werror"
#  pragma clang diagnostic ignored "-Wdocumentation"
#endif

#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>

#if defined(__clang__)
#  pragma clang diagnostic pop
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

struct ArmorMessage {

};

using FoxgloveMessage = std::variant<ImageMessage, SceneMessage, TfMessage>;

// FoxgloveServer 类声明
class FoxgloveServer {
public:
    explicit FoxgloveServer(uint16_t port = 8765, std::string_view host = "127.0.0.1");
    ~FoxgloveServer();

    // 禁止拷贝和移动
    FoxgloveServer(const FoxgloveServer&)            = delete;
    FoxgloveServer& operator=(const FoxgloveServer&) = delete;
    FoxgloveServer(FoxgloveServer&&)                 = delete;
    FoxgloveServer& operator=(FoxgloveServer&&)      = delete;

    bool is_initialized() const;

    // 线程安全的消息入队方法 - 供装饰节点使用
    void enqueue_message(FoxgloveMessage msg);

private:
    // 消息发送线程的主函数
    void message_sender_thread();

private:
    bool server_initialized_{false};
    bool should_stop_{false};

    std::optional<foxglove::WebSocketServer> server_;
    std::optional<foxglove::schemas::FrameTransformsChannel> tf_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> scene_ch_;
    std::optional<foxglove::schemas::CompressedImageChannel> img_ch_;

    // 消息队列及同步原语
    std::queue<FoxgloveMessage> message_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
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

        // 压缩图像数据并转换为消息 - 装饰节点不持有数据所有权
        auto img = feat.image->mat().clone();

        // 使用OpenCV的imencode方法压缩为JPEG格式
        std::vector<uint8_t> compressed_data;
        std::vector<int> encode_params{cv::IMWRITE_JPEG_QUALITY, 85};  // 85质量
        cv::imencode(".jpg", img, compressed_data, encode_params);

        ImageMessage img_msg;
        img_msg.img.timestamp = fxglv_ts_from_ns(feat.timestamp_ns);
        img_msg.img.frame_id  = "camera";
        img_msg.img.format    = "jpeg";
        img_msg.img.data      = std::vector<std::byte>(
            reinterpret_cast<const std::byte*>(compressed_data.data()),
            reinterpret_cast<const std::byte*>(compressed_data.data() + compressed_data.size()));

        // 将消息加入FoxgloveServer队列 - 线程安全
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

        // 发布装甲板位置信息
        // TODO: 实现通过SceneMessage发送Marker显示装甲板位置
        // size_t valid_count = 0;
        for (const auto& pose : poses3d) {
            if (pose.valid) {
                // valid_count++;
            }
        }
        // spdlog::debug("Solved {} valid armor poses", valid_count);
    }

    auto preprocess(scheduler::SharedFeature& sf) -> void { inner.get().preprocess(sf); }

    static foxglove::schemas::Timestamp fxglv_ts_from_ns(uint64_t ns) {
        using namespace std::chrono;
        foxglove::schemas::Timestamp t;
        t.nsec = static_cast<uint32_t>(ns);
        t.sec  = static_cast<int32_t>(ns / 1'000'000'000);
        return t;
    }
};

static_assert(scheduler::SchedulerStrategy<FoxgloveWrapper<TalosStrategy>>);
