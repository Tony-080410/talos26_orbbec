#pragma once

#include "foxglove_types.hpp"

#include "foxglove_export.hpp"
#include "frame.hpp"

#include <array>
#include <atomic>
#include <expected>
#include <memory>
#include <string>
#include <thread>
#include <variant>

#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/server.hpp>

namespace fcs {

// ============================================================================
// Channel Member Macros - 生成 channel 成员变量
// ============================================================================

#define FOXGLOVE_SCENE_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::SceneUpdateChannel> MemberName;
#define FOXGLOVE_IMAGE_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::CompressedImageChannel> MemberName;
#define FOXGLOVE_JSON_CHANNEL(MemberName) std::optional<foxglove::RawChannel> MemberName;
#define FOXGLOVE_TF_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::FrameTransformsChannel> MemberName;
#define FOXGLOVE_CALIB_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::CameraCalibrationChannel> MemberName;

// ============================================================================
// Channel Registry - 所有 channel 的统一管理
// ============================================================================

struct FoxgloveChannels {
    FOXGLOVE_SCENE_CHANNEL(scene_ch)
    FOXGLOVE_SCENE_CHANNEL(track_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(gimbal_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(association_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(filter_covariance_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(mpc_prediction_scene_ch)
    FOXGLOVE_SCENE_CHANNEL(rune_scene_ch)

    FOXGLOVE_IMAGE_CHANNEL(img_ch)
    FOXGLOVE_IMAGE_CHANNEL(binary_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(pattern_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(rune_arrow_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(rune_target_img_ch)
    FOXGLOVE_IMAGE_CHANNEL(rune_center_img_ch)

    FOXGLOVE_JSON_CHANNEL(armor_ch)
    FOXGLOVE_JSON_CHANNEL(debug_lights_ch)
    FOXGLOVE_JSON_CHANNEL(debug_armors_ch)
    FOXGLOVE_JSON_CHANNEL(measurement_ch)
    FOXGLOVE_JSON_CHANNEL(target_ch)
    FOXGLOVE_JSON_CHANNEL(cmd_gimbal_ch)
    FOXGLOVE_JSON_CHANNEL(perf_stats_ch)
    FOXGLOVE_JSON_CHANNEL(mpc_traj_ch)
    FOXGLOVE_JSON_CHANNEL(rune_debug_ch)

    FOXGLOVE_JSON_CHANNEL(pnp_solver_ch)
    FOXGLOVE_JSON_CHANNEL(nn_confidence_ch)

    FOXGLOVE_JSON_CHANNEL(mpc_horizon_ch)
    FOXGLOVE_JSON_CHANNEL(mpc_cost_ch)
    FOXGLOVE_JSON_CHANNEL(mpc_constraint_ch)

    FOXGLOVE_JSON_CHANNEL(energy_meter_ch)

    FOXGLOVE_TF_CHANNEL(tf_ch)
    FOXGLOVE_CALIB_CHANNEL(camera_calib_ch)
};

#undef FOXGLOVE_SCENE_CHANNEL
#undef FOXGLOVE_IMAGE_CHANNEL
#undef FOXGLOVE_JSON_CHANNEL
#undef FOXGLOVE_TF_CHANNEL
#undef FOXGLOVE_CALIB_CHANNEL

// ============================================================================
// Forward declarations
// ============================================================================

class FoxgloveServer;
class FoxgloveServerFactory;

// ============================================================================
// Channel Traits - 编译期映射 message type -> channel pointer
// ============================================================================

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

FOXGLOVE_JSON_TRAIT(ArmorDataMessage, armor_ch)
FOXGLOVE_JSON_TRAIT(DebugLightsMessage, debug_lights_ch)
FOXGLOVE_JSON_TRAIT(DebugArmorsMessage, debug_armors_ch)
FOXGLOVE_JSON_TRAIT(RuneDebugMessage, rune_debug_ch)
FOXGLOVE_JSON_TRAIT(MeasurementMessage, measurement_ch)
FOXGLOVE_JSON_TRAIT(TargetMessage, target_ch)
FOXGLOVE_JSON_TRAIT(GimbalCmdMessage, cmd_gimbal_ch)
FOXGLOVE_JSON_TRAIT(PerfStatsMessage, perf_stats_ch)
FOXGLOVE_JSON_TRAIT(MpcTrajectoryMessage, mpc_traj_ch)
FOXGLOVE_JSON_TRAIT(PnPSolverMessage, pnp_solver_ch)
FOXGLOVE_JSON_TRAIT(NNConfidenceMessage, nn_confidence_ch)
FOXGLOVE_JSON_TRAIT(MpcHorizonMessage, mpc_horizon_ch)
FOXGLOVE_JSON_TRAIT(MpcCostMessage, mpc_cost_ch)
FOXGLOVE_JSON_TRAIT(MpcConstraintMessage, mpc_constraint_ch)
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
#undef FOXGLOVE_IMAGE_TRAIT

template <>
struct ChannelTraits<TfMessage> {
    static auto get(FoxgloveChannels& ch) noexcept
        -> std::optional<foxglove::schemas::FrameTransformsChannel>* {
        return std::addressof(ch.tf_ch);
    }
};

} // namespace detail

// ============================================================================
// Factory
// ============================================================================

class FoxgloveServerFactory {
public:
    [[nodiscard]] static std::expected<std::unique_ptr<FoxgloveServer>, std::string>
        create(uint16_t port, std::string host) noexcept;
};

[[nodiscard]] std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(uint16_t port = 8765, std::string host = "0.0.0.0") noexcept;

// ============================================================================
// Foxglove WebSocket Server
// ============================================================================

class FoxgloveServer {
public:
    ~FoxgloveServer() noexcept {
        should_stop_ = true;
        if (sender_thread_.joinable())
            sender_thread_.join();
        server_->stop();
    }

    FoxgloveServer(const FoxgloveServer&)            = delete;
    FoxgloveServer& operator=(const FoxgloveServer&) = delete;
    FoxgloveServer(FoxgloveServer&&)                 = delete;
    FoxgloveServer& operator=(FoxgloveServer&&)      = delete;

    [[nodiscard]] bool is_initialized() const noexcept { return server_initialized_; }
    void enqueue_message(FoxgloveMessage msg) noexcept { message_queue_.push(std::move(msg)); }

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
        FoxgloveChannels channels, std::shared_ptr<foxglove::WebSocketServer> server) noexcept
        : server_(std::move(server))
        , channels_(std::move(channels)) {
        foxglove::setLogLevel(foxglove::LogLevel::Info);
        if (channels_.tf_ch) {
            tf_exporter_.emplace(std::move(*channels_.tf_ch));
            channels_.tf_ch.reset();
        }
        server_initialized_ = true;
        sender_thread_      = std::thread(&FoxgloveServer::message_sender_thread, this);
    }

    // Dispatch helpers
    template <typename Msg>
    void dispatch_one(const Msg& m) noexcept {
        using Traits = detail::ChannelTraits<Msg>;
        auto* ch     = Traits::get(channels_);
        if (ch)
            dispatch_to_channel(m, *ch);
    }

    template <typename Msg, typename Ch>
    void dispatch_to_channel(const Msg& m, std::optional<Ch>& ch) noexcept {
        if constexpr (requires { m.data; }) {
            ch->log(reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
        } else if constexpr (requires { m.scene; }) {
            ch->log(m.scene);
        } else if constexpr (requires { m.img; }) {
            ch->log(m.img);
        }
    }

    static void dispatch_to_channel(
        const TfMessage& m, std::optional<foxglove::schemas::FrameTransformsChannel>& ch) noexcept {
        if (ch)
            ch->log(m.tf);
    }

    void dispatch_message(const FoxgloveMessage& msg) noexcept {
        std::visit([this](auto&& m) { dispatch_one(m); }, msg);
    }

    void message_sender_thread() {
        while (!should_stop_) {
            FoxgloveMessage msg;
            if (!message_queue_.try_pop(msg)) {
                std::this_thread::yield();
                continue;
            }
            dispatch_message(msg);
        }
        SPDLOG_INFO("Foxglove message sender thread stopped");
    }

    bool server_initialized_{false};
    std::atomic<bool> should_stop_{false};
    std::shared_ptr<foxglove::WebSocketServer> server_;
    FoxgloveChannels channels_;
    std::optional<fast_tf::FoxgloveExporter> tf_exporter_;
    alignas(64) tbb::concurrent_queue<FoxgloveMessage> message_queue_;
    std::thread sender_thread_;
};

// ============================================================================
// Factory Implementation
// ============================================================================

#define TRY_CREATE_SCENE_CHANNEL(ChName, Topic)                                             \
    do {                                                                                    \
        auto ch = detail::try_create_channel<foxglove::schemas::SceneUpdateChannel>(Topic); \
        if (!ch)                                                                            \
            return std::unexpected(fmt::format("{} channel create failed", Topic));         \
        channels.ChName = std::move(*ch);                                                   \
    } while (0)

#define TRY_CREATE_IMAGE_CHANNEL(ChName, Topic)                                                 \
    do {                                                                                        \
        auto ch = detail::try_create_channel<foxglove::schemas::CompressedImageChannel>(Topic); \
        if (!ch)                                                                                \
            return std::unexpected(fmt::format("{} channel create failed", Topic));             \
        channels.ChName = std::move(*ch);                                                       \
    } while (0)

#define TRY_CREATE_JSON_CHANNEL(ChName, Topic)                                      \
    do {                                                                            \
        auto ch = detail::try_create_raw_channel(Topic, "json");                    \
        if (!ch)                                                                    \
            return std::unexpected(fmt::format("{} channel create failed", Topic)); \
        channels.ChName = std::move(*ch);                                           \
    } while (0)

#define TRY_CREATE_SPECIAL_CHANNEL(ChName, Topic, ChType)                           \
    do {                                                                            \
        auto ch = detail::try_create_channel<ChType>(Topic);                        \
        if (!ch)                                                                    \
            return std::unexpected(fmt::format("{} channel create failed", Topic)); \
        channels.ChName = std::move(*ch);                                           \
    } while (0)

namespace detail {

template <typename ChannelType>
[[nodiscard]] std::expected<std::optional<ChannelType>, std::string>
    try_create_channel(std::string_view topic) noexcept {
    auto result = ChannelType::create(std::string(topic));
    if (result.has_value())
        return std::make_optional(std::move(result.value()));
    return std::unexpected(fmt::format("{}: {}", topic, foxglove::strerror(result.error())));
}

[[nodiscard]] std::expected<std::optional<foxglove::RawChannel>, std::string>
    try_create_raw_channel(std::string_view topic, std::string_view encoding) noexcept {
    auto result = foxglove::RawChannel::create(std::string(topic), std::string(encoding));
    if (result.has_value())
        return std::make_optional(std::move(result.value()));
    return std::unexpected(fmt::format("{}: {}", topic, foxglove::strerror(result.error())));
}

} // namespace detail

std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    FoxgloveServerFactory::create(uint16_t port, std::string host) noexcept {
    foxglove::WebSocketServerOptions ws;
    ws.host       = std::move(host);
    ws.port       = port;
    ws.session_id = "talos";

    assert(static_cast<int>(ws.capabilities) == 0);
    auto server_r = foxglove::WebSocketServer::create(std::move(ws));
    if (!server_r.has_value())
        return std::unexpected(
            fmt::format("WebSocketServer: {}", foxglove::strerror(server_r.error())));

    FoxgloveChannels channels;

    TRY_CREATE_SCENE_CHANNEL(scene_ch, "/scene");
    TRY_CREATE_SCENE_CHANNEL(track_scene_ch, "/track/scene");
    TRY_CREATE_SCENE_CHANNEL(gimbal_scene_ch, "/solver/scene");
    TRY_CREATE_SCENE_CHANNEL(association_scene_ch, "/debug/association_scene");
    TRY_CREATE_SCENE_CHANNEL(filter_covariance_scene_ch, "/debug/filter_covariance_scene");
    TRY_CREATE_SCENE_CHANNEL(mpc_prediction_scene_ch, "/debug/mpc_prediction_scene");
    TRY_CREATE_SCENE_CHANNEL(rune_scene_ch, "/rune/scene");

    TRY_CREATE_IMAGE_CHANNEL(img_ch, "/image");
    TRY_CREATE_IMAGE_CHANNEL(binary_img_ch, "/debug/binary_img");
    TRY_CREATE_IMAGE_CHANNEL(pattern_img_ch, "/debug/number_img");
    TRY_CREATE_IMAGE_CHANNEL(rune_arrow_img_ch, "/rune/arrow_img");
    TRY_CREATE_IMAGE_CHANNEL(rune_target_img_ch, "/rune/target_img");
    TRY_CREATE_IMAGE_CHANNEL(rune_center_img_ch, "/rune/center_img");

    TRY_CREATE_JSON_CHANNEL(armor_ch, "/armor_data");
    TRY_CREATE_JSON_CHANNEL(debug_lights_ch, "/debug/lights");
    TRY_CREATE_JSON_CHANNEL(debug_armors_ch, "/debug/armors");
    TRY_CREATE_JSON_CHANNEL(measurement_ch, "/solver/measurement");
    TRY_CREATE_JSON_CHANNEL(target_ch, "/solver/target");
    TRY_CREATE_JSON_CHANNEL(cmd_gimbal_ch, "/solver/cmd_gimbal");
    TRY_CREATE_JSON_CHANNEL(perf_stats_ch, "/perf_stats");
    TRY_CREATE_JSON_CHANNEL(mpc_traj_ch, "/mpc/trajectory");
    TRY_CREATE_JSON_CHANNEL(rune_debug_ch, "/rune/debug");

    TRY_CREATE_JSON_CHANNEL(pnp_solver_ch, "/debug/pnp_solver");
    TRY_CREATE_JSON_CHANNEL(nn_confidence_ch, "/debug/nn_confidence");

    TRY_CREATE_JSON_CHANNEL(mpc_horizon_ch, "/debug/mpc_horizon");
    TRY_CREATE_JSON_CHANNEL(mpc_cost_ch, "/debug/mpc_cost");
    TRY_CREATE_JSON_CHANNEL(mpc_constraint_ch, "/debug/mpc_constraint");

    TRY_CREATE_JSON_CHANNEL(energy_meter_ch, "/energy_meter/state");

    TRY_CREATE_SPECIAL_CHANNEL(tf_ch, "/tf", foxglove::schemas::FrameTransformsChannel);
    TRY_CREATE_SPECIAL_CHANNEL(
        camera_calib_ch, "/camera_info", foxglove::schemas::CameraCalibrationChannel);

    auto server = std::make_shared<foxglove::WebSocketServer>(std::move(server_r.value()));
    SPDLOG_INFO("Foxglove server listening on port {}", port);

    return std::unique_ptr<FoxgloveServer>(
        new FoxgloveServer(std::move(channels), std::move(server)));
}

#undef TRY_CREATE_SCENE_CHANNEL
#undef TRY_CREATE_IMAGE_CHANNEL
#undef TRY_CREATE_JSON_CHANNEL
#undef TRY_CREATE_SPECIAL_CHANNEL

[[nodiscard]] inline std::expected<std::unique_ptr<FoxgloveServer>, std::string>
    create_foxglove_server(uint16_t port, std::string host) noexcept {
    return FoxgloveServerFactory::create(port, std::move(host));
}

} // namespace fcs
