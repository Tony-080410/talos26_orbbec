#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Weverything"
#endif

#include <foxglove/schemas.hpp>

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

namespace fcs {

// ============================================================================
// Foxglove Message Types
// ============================================================================

/// 压缩图像消息
struct ImageMessage {
    foxglove::schemas::CompressedImage img;
};

/// 场景更新消息 (PnP 解算结果)
struct SceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

/// 跟踪场景更新消息 (Tracker 预测结果)
struct TrackSceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

/// TF 变换消息
struct TfMessage {
    foxglove::schemas::FrameTransforms tf;
};

/// 装甲板数据消息 (JSON)
struct ArmorDataMessage {
    std::vector<uint8_t> data;
};

/// 二值化图像消息
struct BinaryImageMessage {
    foxglove::schemas::CompressedImage img;
};

/// 数字识别图像消息
struct PatternImageMessage {
    foxglove::schemas::CompressedImage img;
};

/// 灯条调试消息 (JSON)
struct DebugLightsMessage {
    std::vector<uint8_t> data;
};

/// 装甲板调试消息 (JSON)
struct DebugArmorsMessage {
    std::vector<uint8_t> data;
};

/// 测量值消息 (JSON)
struct MeasurementMessage {
    std::vector<uint8_t> data;
};

/// 跟踪目标消息 (JSON)
struct TargetMessage {
    std::vector<uint8_t> data;
};

/// 云台指令消息 (JSON)
struct GimbalCmdMessage {
    std::vector<uint8_t> data;
};

/// 云台场景消息 (弹道/预测位置可视化)
struct GimbalSceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

/// 性能统计消息 (JSON)
struct PerfStatsMessage {
    std::vector<uint8_t> data;
};

/// MPC 轨迹消息 (JSON)
/// 包含参考轨迹、优化轨迹和控制序列，用于 Foxglove Plot 面板可视化
struct MpcTrajectoryMessage {
    std::vector<uint8_t> data;
};

// ============================================================================
// Debug Message Types (L2/L3/L4 intermediate data)
// ============================================================================

// L2 Debug: PnP solver metrics
struct PnPSolverMessage {
    std::vector<uint8_t> data;
};

// L2 Debug: NN confidence scores
struct NNConfidenceMessage {
    std::vector<uint8_t> data;
};

// L3 Debug: Association phase probabilities
struct AssociationPhaseMessage {
    std::vector<uint8_t> data;
};

// L3 Debug: Association permutation rankings
struct AssociationPermutationMessage {
    std::vector<uint8_t> data;
};

// L3 Debug: Filter innovation vectors
struct FilterInnovationMessage {
    std::vector<uint8_t> data;
};

// L3 Debug: Filter covariance trace
struct FilterCovarianceMessage {
    std::vector<uint8_t> data;
};

// L3 Debug: Viterbi smoother trellis
struct ViterbiTrellisMessage {
    std::vector<uint8_t> data;
};

// L3 Debug: State machine status
struct StateMachineMessage {
    std::vector<uint8_t> data;
};

// L3 Debug: 3D scene for association visualization
struct AssociationSceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

// L3 Debug: 3D scene for filter covariance visualization
struct FilterCovarianceSceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

// L4 Debug: Full MPC prediction horizon
struct MpcHorizonMessage {
    std::vector<uint8_t> data;
};

// L4 Debug: MPC cost evolution
struct MpcCostMessage {
    std::vector<uint8_t> data;
};

// L4 Debug: MPC constraint margins
struct MpcConstraintMessage {
    std::vector<uint8_t> data;
};

// L4 Debug: 3D scene for MPC prediction visualization
struct MpcPredictionSceneMessage {
    foxglove::schemas::SceneUpdate scene;
};

/// Foxglove 消息联合类型
using FoxgloveMessage = std::variant<
    ImageMessage, SceneMessage, TrackSceneMessage, TfMessage, ArmorDataMessage, BinaryImageMessage,
    PatternImageMessage, DebugLightsMessage, DebugArmorsMessage, MeasurementMessage, TargetMessage,
    GimbalCmdMessage, GimbalSceneMessage, PerfStatsMessage, MpcTrajectoryMessage,
    // L2 Debug
    PnPSolverMessage, NNConfidenceMessage,
    // L3 Debug
    AssociationPhaseMessage, AssociationPermutationMessage, FilterInnovationMessage,
    FilterCovarianceMessage, ViterbiTrellisMessage, StateMachineMessage, AssociationSceneMessage,
    FilterCovarianceSceneMessage,
    // L4 Debug
    MpcHorizonMessage, MpcCostMessage, MpcConstraintMessage, MpcPredictionSceneMessage>;

// ============================================================================
// Utility Functions
// ============================================================================

/// 从纳秒时间戳创建 Foxglove 时间戳
[[nodiscard]] inline foxglove::schemas::Timestamp timestamp_from_ns(uint64_t ns) noexcept {
    foxglove::schemas::Timestamp t;
    t.sec  = static_cast<uint32_t>(ns / 1'000'000'000ULL);
    t.nsec = static_cast<uint32_t>(ns % 1'000'000'000ULL);
    return t;
}

/// 从 JSON 字符串创建数据向量
[[nodiscard]] inline std::vector<uint8_t> json_to_bytes(const std::string& json_str) noexcept {
    return {
        reinterpret_cast<const uint8_t*>(json_str.data()),
        reinterpret_cast<const uint8_t*>(json_str.data() + json_str.size())};
}

} // namespace fcs
