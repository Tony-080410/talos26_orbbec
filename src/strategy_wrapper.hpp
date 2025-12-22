#pragma once

#include "strategy.hpp"

#include <opencv2/imgproc.hpp>

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
    foxglove::schemas::RawImage img;
};

struct SceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

struct TfMessage {
    foxglove::schemas::FrameTransforms tf;
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
    std::optional<foxglove::schemas::RawImageChannel> img_ch_;

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

        // 拷贝图像数据到消息队列 - 装饰节点不持有数据所有权
        auto img = feat.image->mat().clone();
        ImageMessage img_msg;
        img_msg.img.timestamp = fxglv_ts_from_ns(feat.timestamp_ns);
        img_msg.img.width     = static_cast<uint32_t>(img.cols);
        img_msg.img.height    = static_cast<uint32_t>(img.rows);
        img_msg.img.encoding  = "rgb8";
        img_msg.img.step      = static_cast<uint32_t>(img.step);
        img_msg.img.data      = std::vector<std::byte>(
            reinterpret_cast<const std::byte*>(img.data),
            reinterpret_cast<const std::byte*>(img.data + img.total() * img.elemSize()));

        // 将消息加入FoxgloveServer队列 - 线程安全
        server.get().enqueue_message(std::move(img_msg));
    }

    auto track(
        talos::TrackerSoA& tracker_soa, const talos::ArmorSlots& armor,
        const auto_aim::Poses3D& poses3d) noexcept -> void {
        inner.get().track(tracker_soa, armor, poses3d);

        // 发布目标位置信息
        // TODO: 通过SceneMessage发送tracker信息
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
