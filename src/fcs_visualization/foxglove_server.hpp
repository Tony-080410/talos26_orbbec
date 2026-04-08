#pragma once

#include "foxglove_types.hpp"

#include "foxglove_export.hpp"
#include "frame.hpp"

#include <atomic>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/mcap.hpp>
#include <foxglove/server.hpp>

namespace fcs {

class FoxgloveServer;
class FoxgloveServerFactory;

namespace detail {

// ============================================================================
// Channel Creation Helpers — driven by descriptors
// ============================================================================

template <typename Def>
[[nodiscard]] inline std::expected<void, std::string> init_typed_channel(
    std::optional<typename Def::channel_type>& member, const foxglove::Context& context) {
    auto result = Def::channel_type::create(std::string(Def::topic), context);
    if (result.has_value()) {
        member = std::move(*result);
        return {};
    }
    return std::unexpected(fmt::format("{}: {}", Def::topic, foxglove::strerror(result.error())));
}

template <typename Def>
[[nodiscard]] inline std::expected<void, std::string> init_raw_channel(
    std::optional<foxglove::RawChannel>& member, const foxglove::Context& context) {
    auto result = foxglove::RawChannel::create(
        std::string(Def::topic), std::string(Def::encoding), {}, context);
    if (result.has_value()) {
        member = std::move(*result);
        return {};
    }
    return std::unexpected(fmt::format("{}: {}", Def::topic, foxglove::strerror(result.error())));
}

template <typename Def>
[[nodiscard]] inline std::expected<void, std::string>
    init_channel_for(FoxgloveChannels& channels, const foxglove::Context& context) {
    if constexpr (requires { Def::transport; }) {
        // Skip channels that don't match the current transport.
        // Caller must set transport on the descriptor or check separately.
    }
    if constexpr (Def::is_raw) {
        return init_raw_channel<Def>(channels.*(Def::member), context);
    } else {
        return init_typed_channel<Def>(channels.*(Def::member), context);
    }
}

// Transport-aware initialization: skip if descriptor has a transport constraint
template <typename Def>
[[nodiscard]] inline std::expected<void, std::string> init_channel_for(
    FoxgloveChannels& channels, const foxglove::Context& context,
    FoxgloveTransport active_transport) {
    if constexpr (requires { Def::transport; }) {
        if (Def::transport != active_transport)
            return {};
    }
    return init_channel_for<Def>(channels, context);
}

// Fold-expression: initialize all channels in a descriptor tuple
template <typename... Defs>
[[nodiscard]] inline std::expected<void, std::string> init_all_channels(
    FoxgloveChannels& channels, const foxglove::Context& context, FoxgloveTransport transport,
    std::tuple<Defs...>) {
    std::expected<void, std::string> result;
    bool ok =
        ((result = init_channel_for<Defs>(channels, context, transport), result.has_value())
         && ...);
    if (!ok)
        return result;
    return {};
}

// ============================================================================
// Sink Creation
// ============================================================================

[[nodiscard]] inline std::expected<foxglove::WebSocketServer, std::string>
    create_websocket_sink(const FoxgloveConfig& config, const foxglove::Context& context) {
    foxglove::WebSocketServerOptions ws;
    ws.context    = context;
    ws.host       = config.host;
    ws.port       = config.port;
    ws.session_id = "talos";

    auto server = foxglove::WebSocketServer::create(std::move(ws));
    if (server.has_value())
        return std::move(server.value());
    return std::unexpected(fmt::format("WebSocketServer: {}", foxglove::strerror(server.error())));
}

[[nodiscard]] inline std::expected<foxglove::McapWriter, std::string>
    create_mcap_sink(const FoxgloveConfig& config, const foxglove::Context& context) {
    if (config.mcap_path.empty()) {
        return std::unexpected("McapWriter: mcap_path must not be empty");
    }

    const auto path = std::filesystem::path(config.mcap_path);
    if (const auto parent = path.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return std::unexpected(
                fmt::format(
                    "create MCAP parent directory '{}': {}", parent.string(), ec.message()));
        }
    }

    foxglove::McapWriterOptions options;
    options.context             = context;
    options.path                = config.mcap_path;
    options.truncate            = true;
    options.sink_channel_filter = [](const foxglove::ChannelDescriptor& channel) {
        return channel.topic() != "/debug/ekf_heatmap";
    };

    auto writer = foxglove::McapWriter::create(options);
    if (writer.has_value())
        return std::move(writer.value());
    return std::unexpected(fmt::format("McapWriter: {}", foxglove::strerror(writer.error())));
}

} // namespace detail

// ============================================================================
// Factory & Free Functions
// ============================================================================

class FoxgloveServerFactory {
public:
    [[nodiscard]] static std::expected<std::unique_ptr<FoxgloveServer>, std::string>
        create(FoxgloveConfig config);
};

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(FoxgloveConfig config);

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(uint16_t port, std::string host);

// ============================================================================
// FoxgloveServer
// ============================================================================

class FoxgloveServer {
private:
    struct ConstructorToken {
        explicit ConstructorToken() = default;
    };
    friend class FoxgloveServerFactory;

public:
    // Public constructor requiring a private token — only FoxgloveServerFactory can invoke
    explicit FoxgloveServer(
        ConstructorToken, foxglove::Context context, FoxgloveChannels channels,
        std::optional<foxglove::WebSocketServer> websocket_server,
        std::optional<foxglove::McapWriter> mcap_writer)
        : context_(std::move(context))
        , channels_(std::move(channels))
        , websocket_server_(std::move(websocket_server))
        , mcap_writer_(std::move(mcap_writer)) {
        foxglove::setLogLevel(foxglove::LogLevel::Info);
        if (channels_.tf_ch) {
            tf_exporter_.emplace(std::move(*channels_.tf_ch));
            channels_.tf_ch.reset();
        }
        server_initialized_ = true;
        sender_thread_      = std::thread(&FoxgloveServer::message_sender_thread, this);
    }

    ~FoxgloveServer() {
        bool expected = false;
        if (!shutdown_done_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
            return;

        sink_alive_->store(false, std::memory_order_release);
        should_stop_.store(true, std::memory_order_release);

        if (sender_thread_.joinable())
            sender_thread_.join();

        // Drain remaining messages before teardown
        FoxgloveMessage msg;
        while (message_queue_.try_pop(msg)) {
            dispatch_message(msg);
        }

        if (websocket_server_) {
            if (auto r = websocket_server_->stop(); r != foxglove::FoxgloveError::Ok) {
                SPDLOG_WARN("Foxglove WebSocket stop failed: {}", foxglove::strerror(r));
            }
        }
        if (mcap_writer_) {
            if (auto r = mcap_writer_->close(); r != foxglove::FoxgloveError::Ok) {
                SPDLOG_WARN("Foxglove MCAP close failed: {}", foxglove::strerror(r));
            }
        }
    }

    FoxgloveServer(const FoxgloveServer&)            = delete;
    FoxgloveServer& operator=(const FoxgloveServer&) = delete;
    FoxgloveServer(FoxgloveServer&&)                 = delete;
    FoxgloveServer& operator=(FoxgloveServer&&)      = delete;

    [[nodiscard]] bool is_initialized() const noexcept { return server_initialized_; }
    void enqueue_message(FoxgloveMessage msg) noexcept { message_queue_.push(std::move(msg)); }
    [[nodiscard]] std::shared_ptr<std::atomic<bool>> sink_alive() const noexcept {
        return sink_alive_;
    }

    void publish_camera_calibration(
        uint32_t width, uint32_t height, const std::array<double, 9>& camera_matrix,
        const std::vector<double>& distortion, uint64_t timestamp_ns = 0) {
        if (!channels_.camera_calib_ch)
            return;

        foxglove::schemas::CameraCalibration calib;
        calib.timestamp        = timestamp_from_ns(timestamp_ns);
        calib.frame_id         = "camera_optical_frame";
        calib.width            = width;
        calib.height           = height;
        calib.distortion_model = "plumb_bob";
        calib.d.assign(distortion.begin(), distortion.end());
        calib.k = camera_matrix;
        calib.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        calib.p = {
            camera_matrix[0],
            0.0,
            camera_matrix[2],
            0.0,
            0.0,
            camera_matrix[4],
            camera_matrix[5],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0};
        channels_.camera_calib_ch->log(calib);
    }

    void publish_tf(const fast_tf::CoordinateSystem& tf_buffer, uint64_t timestamp_ns) noexcept {
        if (tf_exporter_) {
            tf_exporter_->publish_all(tf_buffer, timestamp_ns);
        }
    }

private:
    // Descriptor-driven dispatch — no ChannelTraits needed
    template <typename Def>
    void dispatch_one(const FoxgloveMsg<Def>& m) noexcept {
        auto& opt_ch = channels_.*(Def::member);
        if (!opt_ch)
            return;
        using Payload = typename Def::payload_type;
        detail::PayloadLogger<Payload>::log(*opt_ch, m.payload);
    }

    void dispatch_message(const FoxgloveMessage& msg) noexcept {
        std::visit([this](const auto& m) { dispatch_one(m); }, msg);
    }

    void message_sender_thread() {
        while (true) {
            FoxgloveMessage msg;
            if (message_queue_.try_pop(msg)) {
                dispatch_message(msg);
                continue;
            }

            if (should_stop_.load(std::memory_order_acquire)) {
                break;
            }

            std::this_thread::yield();
        }
    }

    bool server_initialized_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> shutdown_done_{false};
    std::shared_ptr<std::atomic<bool>> sink_alive_{std::make_shared<std::atomic<bool>>(true)};
    foxglove::Context context_{};
    FoxgloveChannels channels_;
    std::optional<foxglove::WebSocketServer> websocket_server_;
    std::optional<foxglove::McapWriter> mcap_writer_;
    std::optional<fast_tf::FoxgloveExporter> tf_exporter_;
    alignas(64) tbb::concurrent_queue<FoxgloveMessage> message_queue_;
    std::thread sender_thread_;
};

// ============================================================================
// Factory Implementation — descriptor-driven
// ============================================================================

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    FoxgloveServerFactory::create(FoxgloveConfig config) {
    foxglove::Context context = foxglove::Context::create();

    std::optional<foxglove::WebSocketServer> websocket_server;
    std::optional<foxglove::McapWriter> mcap_writer;

    switch (config.transport) {
    case FoxgloveTransport::WebSocket: {
        auto server = detail::create_websocket_sink(config, context);
        if (!server)
            return std::unexpected(server.error());
        websocket_server = std::move(*server);
        SPDLOG_INFO("Foxglove WebSocket listening on {}:{}", config.host, config.port);
        break;
    }
    case FoxgloveTransport::Mcap: {
        auto writer = detail::create_mcap_sink(config, context);
        if (!writer)
            return std::unexpected(writer.error());
        mcap_writer = std::move(*writer);
        SPDLOG_INFO("Foxglove MCAP writing to {}", config.mcap_path);
        break;
    }
    }

    // Initialize all channels from the descriptor registry
    FoxgloveChannels channels;
    if (auto r = detail::init_all_channels(channels, context, config.transport, ChannelRegistry{});
        !r) {
        return std::unexpected(std::move(r.error()));
    }

    return std::make_unique<FoxgloveServer>(
        FoxgloveServer::ConstructorToken{}, std::move(context), std::move(channels),
        std::move(websocket_server), std::move(mcap_writer));
}

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(FoxgloveConfig config) {
    return FoxgloveServerFactory::create(std::move(config));
}

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(uint16_t port, std::string host) {
    return create_foxglove_server(
        FoxgloveConfig{
            .transport = FoxgloveTransport::WebSocket,
            .host      = std::move(host),
            .port      = port,
        });
}

} // namespace fcs
