#pragma once

#include "foxglove_config.hpp"
#include "foxglove_types.hpp"

#include "foxglove_export.hpp"
#include "frame.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <variant>

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/mcap.hpp>
#include <foxglove/server.hpp>

namespace fcs {

#define FOXGLOVE_SCENE_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::SceneUpdateChannel> MemberName;
#define FOXGLOVE_IMAGE_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::CompressedImageChannel> MemberName;
#define FOXGLOVE_JSON_CHANNEL(MemberName) std::optional<foxglove::RawChannel> MemberName;
#define FOXGLOVE_TF_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::FrameTransformsChannel> MemberName;
#define FOXGLOVE_CALIB_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::CameraCalibrationChannel> MemberName;
#define FOXGLOVE_LOG_CHANNEL(MemberName) std::optional<foxglove::schemas::LogChannel> MemberName;

struct FoxgloveChannels {
    FOXGLOVE_SCENE_CHANNEL(scene_ch)
    FOXGLOVE_SCENE_CHANNEL(track_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(gimbal_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(association_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(mpc_prediction_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(rune_scene_ch)

    FOXGLOVE_IMAGE_CHANNEL(img_ch)
    FOXGLOVE_IMAGE_CHANNEL(binary_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(pattern_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(rune_arrow_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(rune_target_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(rune_center_img_ch)

    FOXGLOVE_JSON_CHANNEL(debug_lights_ch)
    FOXGLOVE_JSON_CHANNEL(debug_armors_ch)
    FOXGLOVE_JSON_CHANNEL(measurement_ch)
    FOXGLOVE_JSON_CHANNEL(target_ch)
    FOXGLOVE_JSON_CHANNEL(target_selection_trace_ch)
    FOXGLOVE_JSON_CHANNEL(cmd_gimbal_ch)
    FOXGLOVE_JSON_CHANNEL(perf_stats_ch)
    FOXGLOVE_JSON_CHANNEL(mpc_traj_ch)
    FOXGLOVE_JSON_CHANNEL(rune_debug_ch)

    FOXGLOVE_JSON_CHANNEL(pnp_solver_ch)
    FOXGLOVE_JSON_CHANNEL(nn_confidence_ch)
    FOXGLOVE_JSON_CHANNEL(energy_meter_ch)

    FOXGLOVE_IMAGE_CHANNEL(ekf_heatmap_ch)

    FOXGLOVE_TF_CHANNEL(tf_ch)
    FOXGLOVE_CALIB_CHANNEL(camera_calib_ch)
    FOXGLOVE_LOG_CHANNEL(log_ch)
};

#undef FOXGLOVE_SCENE_CHANNEL
#undef FOXGLOVE_IMAGE_CHANNEL
#undef FOXGLOVE_JSON_CHANNEL
#undef FOXGLOVE_TF_CHANNEL
#undef FOXGLOVE_CALIB_CHANNEL
#undef FOXGLOVE_LOG_CHANNEL

class FoxgloveServer;
class FoxgloveServerFactory;

struct FoxgloveServerConfig {
    FoxgloveTransport transport{FoxgloveTransport::WebSocket};
    std::string host{"0.0.0.0"};
    uint16_t port{8765};
    std::string mcap_path{};
};

namespace detail {

template <typename Msg>
struct ChannelTraits;

#define FOXGLOVE_JSON_TRAIT(MsgType, ChMember)                                                   \
    template <>                                                                                  \
    struct ChannelTraits<MsgType> {                                                              \
        static auto get(FoxgloveChannels& ch) noexcept -> std::optional<foxglove::RawChannel>* { \
            return std::addressof(ch.ChMember);                                                  \
        }                                                                                        \
    };

FOXGLOVE_JSON_TRAIT(DebugLightsMessage, debug_lights_ch)
FOXGLOVE_JSON_TRAIT(DebugArmorsMessage, debug_armors_ch)
FOXGLOVE_JSON_TRAIT(RuneDebugMessage, rune_debug_ch)
FOXGLOVE_JSON_TRAIT(MeasurementMessage, measurement_ch)
FOXGLOVE_JSON_TRAIT(TargetMessage, target_ch)
FOXGLOVE_JSON_TRAIT(TargetSelectionTraceMessage, target_selection_trace_ch)
FOXGLOVE_JSON_TRAIT(GimbalCmdMessage, cmd_gimbal_ch)
FOXGLOVE_JSON_TRAIT(PerfStatsMessage, perf_stats_ch)
FOXGLOVE_JSON_TRAIT(MpcTrajectoryMessage, mpc_traj_ch)
FOXGLOVE_JSON_TRAIT(PnPSolverMessage, pnp_solver_ch)
FOXGLOVE_JSON_TRAIT(NNConfidenceMessage, nn_confidence_ch)
FOXGLOVE_JSON_TRAIT(EnergyMeterMessage, energy_meter_ch)
#undef FOXGLOVE_JSON_TRAIT

#define FOXGLOVE_SCENE_TRAIT(MsgType, ChMember)                        \
    template <>                                                        \
    struct ChannelTraits<MsgType> {                                    \
        static auto get(FoxgloveChannels& ch) noexcept                 \
            -> std::optional<foxglove::schemas::SceneUpdateChannel>* { \
            return std::addressof(ch.ChMember);                        \
        }                                                              \
    };

FOXGLOVE_SCENE_TRAIT(SceneMessage, scene_ch)
FOXGLOVE_SCENE_TRAIT(TrackSceneMessage, track_scene_ch)
FOXGLOVE_SCENE_TRAIT(GimbalSceneMessage, gimbal_scene_ch)
FOXGLOVE_SCENE_TRAIT(RuneSceneMessage, rune_scene_ch)
FOXGLOVE_SCENE_TRAIT(AssociationSceneMessage, association_scene_ch)
FOXGLOVE_SCENE_TRAIT(MpcPredictionSceneMessage, mpc_prediction_scene_ch)
#undef FOXGLOVE_SCENE_TRAIT

#define FOXGLOVE_IMAGE_TRAIT(MsgType, ChMember)                            \
    template <>                                                            \
    struct ChannelTraits<MsgType> {                                        \
        static auto get(FoxgloveChannels& ch) noexcept                     \
            -> std::optional<foxglove::schemas::CompressedImageChannel>* { \
            return std::addressof(ch.ChMember);                            \
        }                                                                  \
    };

FOXGLOVE_IMAGE_TRAIT(ImageMessage, img_ch)
FOXGLOVE_IMAGE_TRAIT(BinaryImageMessage, binary_img_ch)
FOXGLOVE_IMAGE_TRAIT(PatternImageMessage, pattern_img_ch)
FOXGLOVE_IMAGE_TRAIT(RuneArrowImageMessage, rune_arrow_img_ch)
FOXGLOVE_IMAGE_TRAIT(RuneTargetImageMessage, rune_target_img_ch)
FOXGLOVE_IMAGE_TRAIT(RuneCenterImageMessage, rune_center_img_ch)
FOXGLOVE_IMAGE_TRAIT(EkfHeatmapMessage, ekf_heatmap_ch)
#undef FOXGLOVE_IMAGE_TRAIT

#define FOXGLOVE_LOG_TRAIT(MsgType, ChMember)                  \
    template <>                                                \
    struct ChannelTraits<MsgType> {                            \
        static auto get(FoxgloveChannels& ch) noexcept         \
            -> std::optional<foxglove::schemas::LogChannel>* { \
            return std::addressof(ch.ChMember);                \
        }                                                      \
    };

FOXGLOVE_LOG_TRAIT(LogMessage, log_ch)
#undef FOXGLOVE_LOG_TRAIT

template <>
struct ChannelTraits<TfMessage> {
    static auto get(FoxgloveChannels& ch) noexcept
        -> std::optional<foxglove::schemas::FrameTransformsChannel>* {
        return std::addressof(ch.tf_ch);
    }
};

template <typename ChannelType>
[[nodiscard]] inline std::expected<ChannelType, std::string>
    try_create_channel(std::string_view topic, const foxglove::Context& context) noexcept {
    auto result = ChannelType::create(std::string(topic), context);
    if (result.has_value())
        return std::move(result.value());
    return std::unexpected(fmt::format("{}: {}", topic, foxglove::strerror(result.error())));
}

[[nodiscard]] inline std::expected<foxglove::RawChannel, std::string> try_create_raw_channel(
    std::string_view topic, std::string_view encoding, const foxglove::Context& context) noexcept {
    auto result =
        foxglove::RawChannel::create(std::string(topic), std::string(encoding), {}, context);
    if (result.has_value())
        return std::move(result.value());
    return std::unexpected(fmt::format("{}: {}", topic, foxglove::strerror(result.error())));
}

[[nodiscard]] inline std::expected<foxglove::WebSocketServer, std::string>
    try_create_websocket_sink(
        const FoxgloveServerConfig& config, const foxglove::Context& context) noexcept {
    foxglove::WebSocketServerOptions ws;
    ws.context    = context;
    ws.host       = config.host;
    ws.port       = config.port;
    ws.session_id = "talos";

    assert(static_cast<int>(ws.capabilities) == 0);
    auto server = foxglove::WebSocketServer::create(std::move(ws));
    if (server.has_value())
        return std::move(server.value());
    return std::unexpected(fmt::format("WebSocketServer: {}", foxglove::strerror(server.error())));
}

[[nodiscard]] inline std::expected<foxglove::McapWriter, std::string> try_create_mcap_sink(
    const FoxgloveServerConfig& config, const foxglove::Context& context) noexcept {
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
    options.context  = context;
    options.path     = config.mcap_path;
    options.truncate = true;

    auto writer = foxglove::McapWriter::create(options);
    if (writer.has_value())
        return std::move(writer.value());
    return std::unexpected(fmt::format("McapWriter: {}", foxglove::strerror(writer.error())));
}

} // namespace detail

class FoxgloveServerFactory {
public:
    [[nodiscard]] static std::expected<std::unique_ptr<FoxgloveServer>, std::string>
        create(FoxgloveServerConfig config) noexcept;
};

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(FoxgloveServerConfig config) noexcept;

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(uint16_t port = 8765, std::string host = "0.0.0.0") noexcept;

class FoxgloveServer {
public:
    ~FoxgloveServer() noexcept {
        bool expected = false;
        if (!shutdown_done_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
            return;

        sink_alive_->store(false, std::memory_order_release);
        should_stop_.store(true, std::memory_order_release);

        if (sender_thread_.joinable())
            sender_thread_.join();

        if (mcap_writer_) {
            const auto result = mcap_writer_->close();
            if (result != foxglove::FoxgloveError::Ok) {
                SPDLOG_WARN("Foxglove MCAP close failed: {}", foxglove::strerror(result));
            }
        }

        // Intentionally leak websocket server for fast foxglove startup
        // if (websocket_server_) {
        //     const auto result = websocket_server_->stop();
        //     if (result != foxglove::FoxgloveError::Ok) {
        //         SPDLOG_WARN("Foxglove WebSocket stop failed: {}", foxglove::strerror(result));
        //     }
        // }
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
        const std::vector<double>& distortion, uint64_t timestamp_ns = 0) noexcept {
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
    friend class FoxgloveServerFactory;

    explicit FoxgloveServer(
        foxglove::Context context, FoxgloveChannels channels,
        std::optional<foxglove::WebSocketServer> websocket_server,
        std::optional<foxglove::McapWriter> mcap_writer) noexcept
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

    template <typename Msg>
    void dispatch_one(const Msg& m) noexcept {
        using Traits = detail::ChannelTraits<Msg>;
        dispatch_to_channel(m, *Traits::get(channels_));
    }

    template <typename Msg, typename Ch>
    void dispatch_to_channel(const Msg& m, std::optional<Ch>& ch) noexcept {
        if (!ch)
            return;

        if constexpr (requires { m.data; }) {
            ch->log(reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
        } else if constexpr (requires { m.scene; }) {
            ch->log(m.scene);
        } else if constexpr (requires { m.img; }) {
            ch->log(m.img);
        } else if constexpr (requires { m.log_msg; }) {
            ch->log(m.log_msg);
        }
    }

    static void dispatch_to_channel(
        const TfMessage& m, std::optional<foxglove::schemas::FrameTransformsChannel>& ch) noexcept {
        if (ch)
            ch->log(m.tf);
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

            if (should_stop_.load(std::memory_order_acquire))
                break;

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

#define TRY_CREATE_SCENE_CHANNEL(ChName, Topic)                                                \
    do {                                                                                       \
        auto ch =                                                                              \
            detail::try_create_channel<foxglove::schemas::SceneUpdateChannel>(Topic, context); \
        if (!ch)                                                                               \
            return std::unexpected(ch.error());                                                \
        channels.ChName = std::move(*ch);                                                      \
    } while (0)

#define TRY_CREATE_IMAGE_CHANNEL(ChName, Topic)                                                    \
    do {                                                                                           \
        auto ch =                                                                                  \
            detail::try_create_channel<foxglove::schemas::CompressedImageChannel>(Topic, context); \
        if (!ch)                                                                                   \
            return std::unexpected(ch.error());                                                    \
        channels.ChName = std::move(*ch);                                                          \
    } while (0)

#define TRY_CREATE_JSON_CHANNEL(ChName, Topic)                            \
    do {                                                                  \
        auto ch = detail::try_create_raw_channel(Topic, "json", context); \
        if (!ch)                                                          \
            return std::unexpected(ch.error());                           \
        channels.ChName = std::move(*ch);                                 \
    } while (0)

#define TRY_CREATE_SPECIAL_CHANNEL(ChName, Topic, ChType)             \
    do {                                                              \
        auto ch = detail::try_create_channel<ChType>(Topic, context); \
        if (!ch)                                                      \
            return std::unexpected(ch.error());                       \
        channels.ChName = std::move(*ch);                             \
    } while (0)

#define TRY_CREATE_LOG_CHANNEL(ChName, Topic)                                                \
    do {                                                                                     \
        auto ch = detail::try_create_channel<foxglove::schemas::LogChannel>(Topic, context); \
        if (!ch)                                                                             \
            return std::unexpected(ch.error());                                              \
        channels.ChName = std::move(*ch);                                                    \
    } while (0)

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    FoxgloveServerFactory::create(FoxgloveServerConfig config) noexcept {
    foxglove::Context context = foxglove::Context::create();

    std::optional<foxglove::WebSocketServer> websocket_server;
    std::optional<foxglove::McapWriter> mcap_writer;

    switch (config.transport) {
    case FoxgloveTransport::WebSocket: {
        auto server = detail::try_create_websocket_sink(config, context);
        if (!server)
            return std::unexpected(server.error());
        websocket_server = std::move(*server);
        SPDLOG_INFO("Foxglove WebSocket listening on {}:{}", config.host, config.port);
        break;
    }
    case FoxgloveTransport::Mcap: {
        auto writer = detail::try_create_mcap_sink(config, context);
        if (!writer)
            return std::unexpected(writer.error());
        mcap_writer = std::move(*writer);
        SPDLOG_INFO("Foxglove MCAP writing to {}", config.mcap_path);
        break;
    }
    }

    FoxgloveChannels channels;

    TRY_CREATE_SCENE_CHANNEL(scene_ch, "/scene");
    TRY_CREATE_SCENE_CHANNEL(track_scene_ch, "/track/scene");
    TRY_CREATE_SCENE_CHANNEL(gimbal_scene_ch, "/solver/scene");
    TRY_CREATE_SCENE_CHANNEL(association_scene_ch, "/debug/association_scene");
    TRY_CREATE_SCENE_CHANNEL(mpc_prediction_scene_ch, "/debug/mpc_prediction_scene");
    TRY_CREATE_SCENE_CHANNEL(rune_scene_ch, "/rune/scene");

    TRY_CREATE_IMAGE_CHANNEL(img_ch, "/image");
    TRY_CREATE_IMAGE_CHANNEL(binary_img_ch, "/debug/binary_img");
    TRY_CREATE_IMAGE_CHANNEL(pattern_img_ch, "/debug/number_img");
    TRY_CREATE_IMAGE_CHANNEL(rune_arrow_img_ch, "/rune/arrow_img");
    TRY_CREATE_IMAGE_CHANNEL(rune_target_img_ch, "/rune/target_img");
    TRY_CREATE_IMAGE_CHANNEL(rune_center_img_ch, "/rune/center_img");
    TRY_CREATE_IMAGE_CHANNEL(ekf_heatmap_ch, "/debug/ekf_heatmap");

    TRY_CREATE_JSON_CHANNEL(debug_lights_ch, "/debug/lights");
    TRY_CREATE_JSON_CHANNEL(debug_armors_ch, "/debug/armors");
    TRY_CREATE_JSON_CHANNEL(measurement_ch, "/solver/measurement");
    TRY_CREATE_JSON_CHANNEL(target_ch, "/solver/target");
    TRY_CREATE_JSON_CHANNEL(target_selection_trace_ch, "/solver/target_selection");
    TRY_CREATE_JSON_CHANNEL(cmd_gimbal_ch, "/solver/cmd_gimbal");
    TRY_CREATE_JSON_CHANNEL(perf_stats_ch, "/perf_stats");
    TRY_CREATE_JSON_CHANNEL(mpc_traj_ch, "/mpc/trajectory");
    TRY_CREATE_JSON_CHANNEL(rune_debug_ch, "/rune/debug");

    TRY_CREATE_JSON_CHANNEL(pnp_solver_ch, "/debug/pnp_solver");
    TRY_CREATE_JSON_CHANNEL(nn_confidence_ch, "/debug/nn_confidence");
    TRY_CREATE_JSON_CHANNEL(energy_meter_ch, "/energy_meter/state");

    TRY_CREATE_SPECIAL_CHANNEL(tf_ch, "/tf", foxglove::schemas::FrameTransformsChannel);
    TRY_CREATE_SPECIAL_CHANNEL(
        camera_calib_ch, "/camera_info", foxglove::schemas::CameraCalibrationChannel);
    TRY_CREATE_LOG_CHANNEL(log_ch, "/log");

    return std::unique_ptr<FoxgloveServer>(new FoxgloveServer(
        std::move(context), std::move(channels), std::move(websocket_server),
        std::move(mcap_writer)));
}

#undef TRY_CREATE_SCENE_CHANNEL
#undef TRY_CREATE_IMAGE_CHANNEL
#undef TRY_CREATE_JSON_CHANNEL
#undef TRY_CREATE_SPECIAL_CHANNEL
#undef TRY_CREATE_LOG_CHANNEL

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(FoxgloveServerConfig config) noexcept {
    return FoxgloveServerFactory::create(std::move(config));
}

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(uint16_t port, std::string host) noexcept {
    return create_foxglove_server(
        FoxgloveServerConfig{
            .transport = FoxgloveTransport::WebSocket,
            .host      = std::move(host),
            .port      = port,
        });
}

} // namespace fcs
