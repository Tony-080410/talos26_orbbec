#include "strategy_wrapper.hpp"
#include <spdlog/spdlog.h>

// FoxgloveServer 的构造函数实现
FoxgloveServer::FoxgloveServer(const uint16_t port, std::string host) noexcept {
    foxglove::setLogLevel(foxglove::LogLevel::Info);

    foxglove::WebSocketServerOptions ws;
    ws.host = host;
    ws.port = port;

    auto server_r = foxglove::WebSocketServer::create(std::move(ws));
    if (!server_r.has_value()) {
        spdlog::error("WebSocketServer create failed: {}", foxglove::strerror(server_r.error()));
        return;
    }
    server_.emplace(std::move(server_r.value()));
    spdlog::info("Foxglove server listening on ws://{}:{}", host, server_->port());

    auto tf_r = foxglove::schemas::FrameTransformsChannel::create("/tf");
    if (!tf_r.has_value()) {
        spdlog::error("FrameTransformsChannel create failed: {}", foxglove::strerror(tf_r.error()));
        return;
    }
    tf_ch_.emplace(std::move(tf_r.value()));

    auto scene_r = foxglove::schemas::SceneUpdateChannel::create("/scene");
    if (!scene_r.has_value()) {
        spdlog::error("SceneUpdateChannel create failed: {}", foxglove::strerror(scene_r.error()));
        return;
    }
    scene_ch_.emplace(std::move(scene_r.value()));

    auto img_r = foxglove::schemas::CompressedImageChannel::create("/image");
    if (!img_r.has_value()) {
        spdlog::error(
            "CompressedImageChannel create failed: {}", foxglove::strerror(img_r.error()));
        return;
    }
    img_ch_.emplace(std::move(img_r.value()));

    auto armor_r = foxglove::RawChannel::create("/armor_data", "json");
    if (!armor_r.has_value()) {
        spdlog::error("RawChannel create failed: {}", foxglove::strerror(armor_r.error()));
        return;
    }
    armor_ch_.emplace(std::move(armor_r.value()));

    server_initialized_ = true;

    // 启动消息发送线程
    sender_thread_ = std::thread(&FoxgloveServer::message_sender_thread, this);
}

FoxgloveServer::~FoxgloveServer() noexcept {
    // 关闭发送线程
    { should_stop_ = true; }
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
}

void FoxgloveServer::message_sender_thread() {
    while (true) {
        if (should_stop_) [[unlikely]] {
            break;
        }
        FoxgloveMessage msg;
        if (const auto success = message_queue_.try_pop(msg); !success) {
            continue;
        }

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
                } else if constexpr (std::is_same_v<decltype(m), ArmorDataMessage&>) {
                    if (armor_ch_) {
                        armor_ch_->log(
                            reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    }
                }
            },
            msg);
    }
    spdlog::info("Foxglove message sender thread stopped");
}

bool FoxgloveServer::is_initialized() const noexcept { return server_initialized_; }

void FoxgloveServer::enqueue_message(FoxgloveMessage msg) noexcept {
    message_queue_.push(std::move(msg));
}
