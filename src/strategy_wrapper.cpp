#include "strategy_wrapper.hpp"
#include <spdlog/spdlog.h>

// FoxgloveServer 的构造函数实现
FoxgloveServer::FoxgloveServer(uint16_t port, std::string_view host) {
    foxglove::setLogLevel(foxglove::LogLevel::Info);

    foxglove::WebSocketServerOptions ws;
    ws.host = std::string(host);
    ws.port = port;

    auto server_r = foxglove::WebSocketServer::create(std::move(ws));
    if (!server_r.has_value()) {
        spdlog::error(
            "WebSocketServer create failed: {}", foxglove::strerror(server_r.error()));
        return;
    }
    server_.emplace(std::move(server_r.value()));
    spdlog::info("Foxglove server listening on ws://{}:{}", host, server_->port());

    auto tf_r = foxglove::schemas::FrameTransformsChannel::create("/tf");
    if (!tf_r.has_value()) {
        spdlog::error(
            "FrameTransformsChannel create failed: {}", foxglove::strerror(tf_r.error()));
        return;
    }
    tf_ch_.emplace(std::move(tf_r.value()));

    auto scene_r = foxglove::schemas::SceneUpdateChannel::create("/scene");
    if (!scene_r.has_value()) {
        spdlog::error(
            "SceneUpdateChannel create failed: {}", foxglove::strerror(scene_r.error()));
        return;
    }
    scene_ch_.emplace(std::move(scene_r.value()));

    auto img_r = foxglove::schemas::CompressedImageChannel::create("/image");
    if (!img_r.has_value()) {
        spdlog::error("CompressedImageChannel create failed: {}", foxglove::strerror(img_r.error()));
        return;
    }
    img_ch_.emplace(std::move(img_r.value()));

    server_initialized_ = true;

    // 启动消息发送线程
    sender_thread_ = std::thread(&FoxgloveServer::message_sender_thread, this);
}

FoxgloveServer::~FoxgloveServer() {
    // 关闭发送线程
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        should_stop_ = true;
    }
    queue_cv_.notify_one();
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
}

void FoxgloveServer::message_sender_thread() {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // 等待消息或停止信号
        queue_cv_.wait(lock, [this] { return !message_queue_.empty() || should_stop_; });

        if (should_stop_ && message_queue_.empty()) {
            break;
        }

        if (message_queue_.empty()) {
            continue;
        }

        auto msg = std::move(message_queue_.front());
        message_queue_.pop();

        lock.unlock();

        // 处理消息 - 在锁外执行以避免阻塞
        std::visit(
            [this](auto& m) {
                if constexpr (std::is_same_v<decltype(m), ImageMessage&>) {
                    if (img_ch_) {
                        img_ch_->log(m.img);
                    }
                } else if constexpr (std::is_same_v<decltype(m), SceneMessage&>) {
                    if (scene_ch_) {
                        scene_ch_->log(m.scene);
                    }
                } else if constexpr (std::is_same_v<decltype(m), TfMessage&>) {
                    if (tf_ch_) {
                        tf_ch_->log(m.tf);
                    }
                }
            },
            msg);
    }
    spdlog::info("Foxglove message sender thread stopped");
}

bool FoxgloveServer::is_initialized() const { return server_initialized_; }

void FoxgloveServer::enqueue_message(FoxgloveMessage msg) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!should_stop_) {
            message_queue_.push(std::move(msg));
        }
    }
    queue_cv_.notify_one();
}
