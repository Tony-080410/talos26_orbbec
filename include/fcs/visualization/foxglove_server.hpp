#pragma once

#include "foxglove_types.hpp"

#include "math/fast_tf/foxglove_export.hpp"
#include "math/fast_tf/frame.hpp"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Weverything"
#endif

#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/server.hpp>

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

namespace fcs {

/// Foxglove WebSocket 服务器
/// 管理所有 channel 并异步发送消息
class FoxgloveServer {
public:
    explicit FoxgloveServer(uint16_t port = 8765, std::string host = "0.0.0.0") noexcept {
        foxglove::setLogLevel(foxglove::LogLevel::Info);

        foxglove::WebSocketServerOptions ws;
        ws.host = std::move(host);
        ws.port = port;

        auto server_r = foxglove::WebSocketServer::create(std::move(ws));
        if (!server_r.has_value()) {
            SPDLOG_ERROR("WebSocketServer create failed: {}", foxglove::strerror(server_r.error()));
            return;
        }
        server_.emplace(std::move(server_r.value()));
        SPDLOG_INFO("Foxglove server listening on ws://{}:{}", ws.host, server_->port());

        // 创建 channels
        init_channels();

        if (!init_success_) {
            return;
        }

        // 初始化 TF exporter (不再创建内部 tf_buffer，使用外部共享的 CoordinateSystem)
        if (tf_ch_) {
            tf_exporter_.emplace(std::move(*tf_ch_));
            tf_ch_.reset();
        }

        server_initialized_ = true;

        // 启动消息发送线程
        sender_thread_ = std::thread(&FoxgloveServer::message_sender_thread, this);
    }

    ~FoxgloveServer() noexcept {
        should_stop_ = true;
        if (sender_thread_.joinable()) {
            sender_thread_.join();
        }
        server_->stop();
    }

    // 禁止拷贝和移动
    FoxgloveServer(const FoxgloveServer&)            = delete;
    FoxgloveServer& operator=(const FoxgloveServer&) = delete;
    FoxgloveServer(FoxgloveServer&&)                 = delete;
    FoxgloveServer& operator=(FoxgloveServer&&)      = delete;

    [[nodiscard]] bool is_initialized() const noexcept { return server_initialized_; }

    /// 线程安全的消息入队
    void enqueue_message(FoxgloveMessage msg) noexcept { message_queue_.push(std::move(msg)); }

    /// 发布相机标定信息
    void publish_camera_calibration(
        uint32_t width, uint32_t height, const std::array<double, 9>& camera_matrix,
        const std::vector<double>& distortion, uint64_t timestamp_ns = 0) noexcept {
        if (!camera_calib_ch_) {
            return;
        }

        foxglove::schemas::CameraCalibration calib;
        calib.timestamp        = timestamp_from_ns(timestamp_ns);
        calib.frame_id         = "camera_optical_frame";
        calib.width            = width;
        calib.height           = height;
        calib.distortion_model = "plumb_bob";

        calib.d.reserve(distortion.size());
        for (const auto& coeff : distortion) {
            calib.d.push_back(coeff);
        }

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

        camera_calib_ch_->log(calib);
    }

    /// 发布 TF 树 (使用外部共享的 CoordinateSystem)
    void publish_tf(
        const fast_tf::CoordinateSystem<1024>& tf_buffer, uint64_t timestamp_ns) noexcept {
        if (tf_exporter_) {
            tf_exporter_->publish_entire<fast_tf::camera_optical_fuxk_frame>(
                tf_buffer, timestamp_ns);
            tf_exporter_->publish_entire<fast_tf::muzzle_link_fuxk_frame>(tf_buffer, timestamp_ns);
        }
    }

private:
    void init_channels() noexcept {
        init_success_ = true;

        // TF channel
        if (auto r = foxglove::schemas::FrameTransformsChannel::create("/tf"); r.has_value()) {
            tf_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("FrameTransformsChannel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // Scene channels
        if (auto r = foxglove::schemas::SceneUpdateChannel::create("/scene"); r.has_value()) {
            scene_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("SceneUpdateChannel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::schemas::SceneUpdateChannel::create("/track/scene"); r.has_value()) {
            track_scene_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("track_scene_ch create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // Image channels
        if (auto r = foxglove::schemas::CompressedImageChannel::create("/image"); r.has_value()) {
            img_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("CompressedImageChannel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::schemas::CompressedImageChannel::create("/debug/binary_img");
            r.has_value()) {
            binary_img_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("binary_img channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::schemas::CompressedImageChannel::create("/debug/number_img");
            r.has_value()) {
            pattern_img_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("pattern_img channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // Raw JSON channels
        if (auto r = foxglove::RawChannel::create("/armor_data", "json"); r.has_value()) {
            armor_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("armor_ch create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/lights", "json"); r.has_value()) {
            debug_lights_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("debug_lights channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/armors", "json"); r.has_value()) {
            debug_armors_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("debug_armors channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/solver/measurement", "json"); r.has_value()) {
            measurement_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("measurement channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/solver/target", "json"); r.has_value()) {
            target_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("target channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/solver/cmd_gimbal", "json"); r.has_value()) {
            cmd_gimbal_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("cmd_gimbal channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // Gimbal scene channel (弹道/预测位置可视化)
        if (auto r = foxglove::schemas::SceneUpdateChannel::create("/solver/scene");
            r.has_value()) {
            gimbal_scene_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("gimbal_scene channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // Camera calibration channel
        if (auto r = foxglove::schemas::CameraCalibrationChannel::create("/camera_info");
            r.has_value()) {
            camera_calib_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("camera_calib channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // Performance statistics channel
        if (auto r = foxglove::RawChannel::create("/perf_stats", "json"); r.has_value()) {
            perf_stats_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("perf_stats channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // MPC trajectory channel
        if (auto r = foxglove::RawChannel::create("/mpc/trajectory", "json"); r.has_value()) {
            mpc_traj_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("mpc_traj channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // L2 Debug: PnP solver metrics
        if (auto r = foxglove::RawChannel::create("/debug/pnp_solver", "json"); r.has_value()) {
            pnp_solver_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("pnp_solver channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/nn_confidence", "json"); r.has_value()) {
            nn_confidence_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("nn_confidence channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // L3 Debug: Association
        if (auto r = foxglove::RawChannel::create("/debug/association_phase", "json");
            r.has_value()) {
            association_phase_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "association_phase channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/association_permutation", "json");
            r.has_value()) {
            association_permutation_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "association_permutation channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // L3 Debug: Filter
        if (auto r = foxglove::RawChannel::create("/debug/filter_innovation", "json");
            r.has_value()) {
            filter_innovation_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "filter_innovation channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/filter_covariance", "json");
            r.has_value()) {
            filter_covariance_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "filter_covariance channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/viterbi_trellis", "json");
            r.has_value()) {
            viterbi_trellis_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "viterbi_trellis channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/state_machine", "json"); r.has_value()) {
            state_machine_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("state_machine channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // L3 Debug: 3D scenes
        if (auto r = foxglove::schemas::SceneUpdateChannel::create("/debug/association_scene");
            r.has_value()) {
            association_scene_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "association_scene channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r =
                foxglove::schemas::SceneUpdateChannel::create("/debug/filter_covariance_scene");
            r.has_value()) {
            filter_covariance_scene_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "filter_covariance_scene channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        // L4 Debug: MPC
        if (auto r = foxglove::RawChannel::create("/debug/mpc_horizon", "json"); r.has_value()) {
            mpc_horizon_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("mpc_horizon channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/mpc_cost", "json"); r.has_value()) {
            mpc_cost_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("mpc_cost channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::RawChannel::create("/debug/mpc_constraint", "json"); r.has_value()) {
            mpc_constraint_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR("mpc_constraint channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }

        if (auto r = foxglove::schemas::SceneUpdateChannel::create("/debug/mpc_prediction_scene");
            r.has_value()) {
            mpc_prediction_scene_ch_.emplace(std::move(r.value()));
        } else {
            SPDLOG_ERROR(
                "mpc_prediction_scene channel create failed: {}", foxglove::strerror(r.error()));
            init_success_ = false;
        }
    }

    void message_sender_thread() {
        while (!should_stop_) {
            FoxgloveMessage msg;
            if (!message_queue_.try_pop(msg)) {
                std::this_thread::yield();
                continue;
            }

            std::visit(
                [this](auto& m) {
                    using T = std::decay_t<decltype(m)>;
                    if constexpr (std::is_same_v<T, ImageMessage>) {
                        if (img_ch_)
                            img_ch_->log(m.img);
                    } else if constexpr (std::is_same_v<T, SceneMessage>) {
                        if (scene_ch_)
                            scene_ch_->log(m.scene);
                    } else if constexpr (std::is_same_v<T, TrackSceneMessage>) {
                        if (track_scene_ch_)
                            track_scene_ch_->log(m.scene);
                    } else if constexpr (std::is_same_v<T, TfMessage>) {
                        if (tf_ch_)
                            tf_ch_->log(m.tf);
                    } else if constexpr (std::is_same_v<T, ArmorDataMessage>) {
                        if (armor_ch_)
                            armor_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, BinaryImageMessage>) {
                        if (binary_img_ch_)
                            binary_img_ch_->log(m.img);
                    } else if constexpr (std::is_same_v<T, PatternImageMessage>) {
                        if (pattern_img_ch_)
                            pattern_img_ch_->log(m.img);
                    } else if constexpr (std::is_same_v<T, DebugLightsMessage>) {
                        if (debug_lights_ch_)
                            debug_lights_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, DebugArmorsMessage>) {
                        if (debug_armors_ch_)
                            debug_armors_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, MeasurementMessage>) {
                        if (measurement_ch_)
                            measurement_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, TargetMessage>) {
                        if (target_ch_)
                            target_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, GimbalCmdMessage>) {
                        if (cmd_gimbal_ch_)
                            cmd_gimbal_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, GimbalSceneMessage>) {
                        if (gimbal_scene_ch_)
                            gimbal_scene_ch_->log(m.scene);
                    } else if constexpr (std::is_same_v<T, PerfStatsMessage>) {
                        if (perf_stats_ch_)
                            perf_stats_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, MpcTrajectoryMessage>) {
                        if (mpc_traj_ch_)
                            mpc_traj_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, PnPSolverMessage>) {
                        if (pnp_solver_ch_)
                            pnp_solver_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, NNConfidenceMessage>) {
                        if (nn_confidence_ch_)
                            nn_confidence_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, AssociationPhaseMessage>) {
                        if (association_phase_ch_)
                            association_phase_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, AssociationPermutationMessage>) {
                        if (association_permutation_ch_)
                            association_permutation_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, FilterInnovationMessage>) {
                        if (filter_innovation_ch_)
                            filter_innovation_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, FilterCovarianceMessage>) {
                        if (filter_covariance_ch_)
                            filter_covariance_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, ViterbiTrellisMessage>) {
                        if (viterbi_trellis_ch_)
                            viterbi_trellis_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, StateMachineMessage>) {
                        if (state_machine_ch_)
                            state_machine_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, AssociationSceneMessage>) {
                        if (association_scene_ch_)
                            association_scene_ch_->log(m.scene);
                    } else if constexpr (std::is_same_v<T, FilterCovarianceSceneMessage>) {
                        if (filter_covariance_scene_ch_)
                            filter_covariance_scene_ch_->log(m.scene);
                    } else if constexpr (std::is_same_v<T, MpcHorizonMessage>) {
                        if (mpc_horizon_ch_)
                            mpc_horizon_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, MpcCostMessage>) {
                        if (mpc_cost_ch_)
                            mpc_cost_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, MpcConstraintMessage>) {
                        if (mpc_constraint_ch_)
                            mpc_constraint_ch_->log(
                                reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    } else if constexpr (std::is_same_v<T, MpcPredictionSceneMessage>) {
                        if (mpc_prediction_scene_ch_)
                            mpc_prediction_scene_ch_->log(m.scene);
                    }
                },
                msg);
        }
        SPDLOG_INFO("Foxglove message sender thread stopped");
    }

    bool server_initialized_{false};
    bool init_success_{false};
    std::atomic<bool> should_stop_{false};

    std::optional<foxglove::WebSocketServer> server_;

    // Channels
    std::optional<foxglove::schemas::FrameTransformsChannel> tf_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> scene_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> track_scene_ch_;
    std::optional<foxglove::schemas::CompressedImageChannel> img_ch_;
    std::optional<foxglove::schemas::CompressedImageChannel> binary_img_ch_;
    std::optional<foxglove::schemas::CompressedImageChannel> pattern_img_ch_;
    std::optional<foxglove::RawChannel> armor_ch_;
    std::optional<foxglove::RawChannel> debug_lights_ch_;
    std::optional<foxglove::RawChannel> debug_armors_ch_;
    std::optional<foxglove::RawChannel> measurement_ch_;
    std::optional<foxglove::RawChannel> target_ch_;
    std::optional<foxglove::RawChannel> cmd_gimbal_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> gimbal_scene_ch_;
    std::optional<foxglove::schemas::CameraCalibrationChannel> camera_calib_ch_;
    std::optional<foxglove::RawChannel> perf_stats_ch_;
    std::optional<foxglove::RawChannel> mpc_traj_ch_;

    // Debug channels (L2/L3/L4)
    std::optional<foxglove::RawChannel> pnp_solver_ch_;
    std::optional<foxglove::RawChannel> nn_confidence_ch_;
    std::optional<foxglove::RawChannel> association_phase_ch_;
    std::optional<foxglove::RawChannel> association_permutation_ch_;
    std::optional<foxglove::RawChannel> filter_innovation_ch_;
    std::optional<foxglove::RawChannel> filter_covariance_ch_;
    std::optional<foxglove::RawChannel> viterbi_trellis_ch_;
    std::optional<foxglove::RawChannel> state_machine_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> association_scene_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> filter_covariance_scene_ch_;
    std::optional<foxglove::RawChannel> mpc_horizon_ch_;
    std::optional<foxglove::RawChannel> mpc_cost_ch_;
    std::optional<foxglove::RawChannel> mpc_constraint_ch_;
    std::optional<foxglove::schemas::SceneUpdateChannel> mpc_prediction_scene_ch_;

    // TF exporter (使用外部共享的 CoordinateSystem)
    std::optional<fast_tf::FoxgloveExporter> tf_exporter_;

    // Message queue
    alignas(64) tbb::concurrent_queue<FoxgloveMessage> message_queue_;
    std::thread sender_thread_;
};

} // namespace fcs
