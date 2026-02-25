#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include <foxglove/schemas.hpp>

namespace fcs {

// ============================================================================
// Message Type Macros
// ============================================================================

/// Macro to generate a JSON message type (has std::vector<uint8_t> data member)
#define FOXGLOVE_JSON_MESSAGE(MessageName) \
    struct MessageName {                   \
        std::vector<uint8_t> data;         \
    };

/// Macro to generate a scene message type (has foxglove::schemas::SceneUpdate scene member)
#define FOXGLOVE_SCENE_MESSAGE(MessageName)   \
    struct MessageName {                      \
        foxglove::schemas::SceneUpdate scene; \
    };

/// Macro to generate an image message type (has foxglove::schemas::CompressedImage img member)
#define FOXGLOVE_IMAGE_MESSAGE(MessageName)     \
    struct MessageName {                        \
        foxglove::schemas::CompressedImage img; \
    };

// ============================================================================
// Foxglove Message Types
// ============================================================================

/// 压缩图像消息
FOXGLOVE_IMAGE_MESSAGE(ImageMessage)

/// 场景更新消息 (PnP 解算结果)
FOXGLOVE_SCENE_MESSAGE(SceneMessage)

/// 跟踪场景更新消息 (Tracker 预测结果)
FOXGLOVE_SCENE_MESSAGE(TrackSceneMessage)

/// TF 变换消息
struct TfMessage {
    foxglove::schemas::FrameTransforms tf;
};

/// 装甲板数据消息 (JSON)
FOXGLOVE_JSON_MESSAGE(ArmorDataMessage)

/// 二值化图像消息
FOXGLOVE_IMAGE_MESSAGE(BinaryImageMessage)

/// 数字识别图像消息
FOXGLOVE_IMAGE_MESSAGE(PatternImageMessage)

/// 灯条调试消息 (JSON)
FOXGLOVE_JSON_MESSAGE(DebugLightsMessage)

/// 装甲板调试消息 (JSON)
FOXGLOVE_JSON_MESSAGE(DebugArmorsMessage)

FOXGLOVE_IMAGE_MESSAGE(RuneArrowImageMessage)
FOXGLOVE_IMAGE_MESSAGE(RuneTargetImageMessage)
FOXGLOVE_IMAGE_MESSAGE(RuneCenterImageMessage)
FOXGLOVE_JSON_MESSAGE(RuneDebugMessage)
FOXGLOVE_SCENE_MESSAGE(RuneSceneMessage)

/// 测量值消息 (JSON)
FOXGLOVE_JSON_MESSAGE(MeasurementMessage)

/// 跟踪目标消息 (JSON)
FOXGLOVE_JSON_MESSAGE(TargetMessage)

/// 云台指令消息 (JSON)
FOXGLOVE_JSON_MESSAGE(GimbalCmdMessage)

/// 云台场景消息 (弹道/预测位置可视化)
FOXGLOVE_SCENE_MESSAGE(GimbalSceneMessage)

/// 性能统计消息 (JSON)
FOXGLOVE_JSON_MESSAGE(PerfStatsMessage)

/// MPC 轨迹消息 (JSON)
FOXGLOVE_JSON_MESSAGE(MpcTrajectoryMessage)

// ============================================================================
// Debug Message Types (L2/L3/L4 intermediate data)
// ============================================================================

// L2 Debug: PnP solver metrics
FOXGLOVE_JSON_MESSAGE(PnPSolverMessage)

// L2 Debug: NN confidence scores
FOXGLOVE_JSON_MESSAGE(NNConfidenceMessage)

// L3 Debug: Association phase probabilities
FOXGLOVE_JSON_MESSAGE(AssociationPhaseMessage)

// L3 Debug: Association permutation rankings
FOXGLOVE_JSON_MESSAGE(AssociationPermutationMessage)

// L3 Debug: Filter innovation vectors
FOXGLOVE_JSON_MESSAGE(FilterInnovationMessage)

// L3 Debug: Filter covariance trace
FOXGLOVE_JSON_MESSAGE(FilterCovarianceMessage)

// L3 Debug: State machine status
FOXGLOVE_JSON_MESSAGE(StateMachineMessage)

// L3 Debug: 3D scene for association visualization
FOXGLOVE_SCENE_MESSAGE(AssociationSceneMessage)

// L4 Debug: Full MPC prediction horizon
FOXGLOVE_JSON_MESSAGE(MpcHorizonMessage)

// L4 Debug: MPC cost evolution
FOXGLOVE_JSON_MESSAGE(MpcCostMessage)

// L4 Debug: MPC constraint margins
FOXGLOVE_JSON_MESSAGE(MpcConstraintMessage)

// L4 Debug: 3D scene for MPC prediction visualization
FOXGLOVE_SCENE_MESSAGE(MpcPredictionSceneMessage)

/// Energy meter state message (JSON)
FOXGLOVE_JSON_MESSAGE(EnergyMeterMessage)

// ============================================================================
// Foxglove Message Variant
// ============================================================================

/// Foxglove 消息联合类型
using FoxgloveMessage = std::variant<
    ImageMessage, SceneMessage, TrackSceneMessage, TfMessage, ArmorDataMessage, BinaryImageMessage,
    PatternImageMessage, DebugLightsMessage, DebugArmorsMessage, RuneArrowImageMessage,
    RuneTargetImageMessage, RuneCenterImageMessage, RuneDebugMessage, RuneSceneMessage,
    MeasurementMessage, TargetMessage, GimbalCmdMessage, GimbalSceneMessage, PerfStatsMessage,
    MpcTrajectoryMessage,
    // L2 Debug
    PnPSolverMessage, NNConfidenceMessage,
    // L3 Debug
    AssociationSceneMessage,
    // L4 Debug
    MpcHorizonMessage, MpcCostMessage, MpcConstraintMessage, MpcPredictionSceneMessage,
    // Energy Meter
    EnergyMeterMessage>;

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
