#pragma once

#include <variant>

#include "concepts.hpp"
#include "pipeline_types.hpp"

namespace auto_aim {

// ============ 前向声明 ============

// 检测器实现
class TraditionalDetector; // 传统灯条配对检测器
class NNDetector;          // NN 检测器（Phase 6 移植）

// 解算器实现
class Solver;   // PnP 解算器
class BASolver; // BA 解算器（未来扩展）

// 追踪器实现
class Tracker;       // EKF 追踪器
class ISRCKFTracker; // ISRCKF 追踪器
class SRUKFTracker;  // SRUKF 追踪器（Phase 5 移植）

// ============ Variant 类型定义 ============

/**
 * @brief 检测器 Variant
 */
using DetectorVariant = std::variant<TraditionalDetector, NNDetector>;

/**
 * @brief 解算器 Variant
 */
using SolverVariant = std::variant<Solver, BASolver>;

/**
 * @brief 追踪器 Variant
 */
using TrackerVariant = std::variant<Tracker, ISRCKFTracker, SRUKFTracker>;

// ============ Variant 访问器 ============

/**
 * @brief 对 DetectorVariant 执行检测
 */
inline void detect(
    DetectorVariant& detector, const PreprocessedFrame& frame, DetectionResult& out,
    DebugData& debug) noexcept {
    std::visit([&](auto& det) { det.detect(frame, out, debug); }, detector);
}

/**
 * @brief 对 SolverVariant 执行解算
 */
inline void solve(
    SolverVariant& solver, const DetectionResult& det, SolveResult& out,
    DebugData& debug) noexcept {
    std::visit([&](auto& s) { s.solve(det, out, debug); }, solver);
}

/**
 * @brief 对 TrackerVariant 执行追踪
 */
inline void track(
    TrackerVariant& tracker, const DetectionResult& det, const SolveResult& solve,
    TrackerOutput& out, DebugData& debug) noexcept {
    std::visit([&](auto& trk) { trk.track(det, solve, out, debug); }, tracker);
}

/**
 * @brief 重置 TrackerVariant
 */
inline void reset_tracker(TrackerVariant& tracker) noexcept {
    std::visit([](auto& trk) { trk.reset(); }, tracker);
}

/**
 * @brief 获取 DetectorVariant 需要的预处理类型
 */
[[nodiscard]] inline PreprocessType get_preprocess_type(const DetectorVariant& detector) noexcept {
    return std::visit(
        [](const auto& det) -> PreprocessType {
            using T = std::decay_t<decltype(det)>;
            return T::preprocess_type();
        },
        detector);
}

} // namespace auto_aim
