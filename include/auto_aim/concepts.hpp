#pragma once

#include <concepts>
#include <type_traits>

#include "pipeline_types.hpp"

namespace auto_aim {

// ============ overloaded helper ============

/**
 * @brief std::visit 的 overloaded helper
 *
 * 使用示例:
 * ```cpp
 * std::visit(overloaded{
 *     [](const talos::ArmorSlots& s) { ... },
 *     [](const NNDetectionExtra& nn) { ... },
 *     [](std::monostate) { ... }
 * }, detection);
 * ```
 */
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// ============ 检测器 Concept ============

/**
 * @brief 检测器 Concept
 *
 * 要求:
 * - detect(frame, out, debug) noexcept -> void
 * - T::preprocess_type() -> PreprocessType (静态方法)
 *
 * 所有检测器实现必须满足此 Concept。
 */
template <typename T>
concept DetectorConcept =
    requires(T& det, const PreprocessedFrame& frame, DetectionResult& out, DebugData& debug) {
        { det.detect(frame, out, debug) } noexcept -> std::same_as<void>;
        { T::preprocess_type() } -> std::same_as<PreprocessType>;
    };

// ============ 解算器 Concept ============

/**
 * @brief 解算器 Concept
 *
 * 要求:
 * - solve(det, out, debug) noexcept -> void
 *
 * 所有解算器实现必须满足此 Concept。
 */
template <typename T>
concept SolverConcept =
    requires(T& solver, const DetectionResult& det, SolveResult& out, DebugData& debug) {
        { solver.solve(det, out, debug) } noexcept -> std::same_as<void>;
    };

// ============ 追踪器 Concept ============

/**
 * @brief 追踪器 Concept
 *
 * 要求:
 * - track(det, solve, out, debug) noexcept -> void
 * - reset() noexcept -> void
 *
 * 所有追踪器实现必须满足此 Concept。
 * 追踪器自己持有内部状态（状态向量、协方差矩阵等）。
 */
template <typename T>
concept TrackerConcept = requires(
    T& trk, const DetectionResult& det, const SolveResult& solve, TrackerOutput& out,
    DebugData& debug) {
    { trk.track(det, solve, out, debug) } noexcept -> std::same_as<void>;
    { trk.reset() } noexcept -> std::same_as<void>;
};

// ============ Variant 类型萃取 ============

/**
 * @brief 检查类型是否为 DetectionResult 的有效变体
 */
template <typename T>
concept IsDetectionVariant = std::same_as<T, std::monostate> || std::same_as<T, talos::ArmorSlots>
                          || std::same_as<T, NNDetectionExtra>;

/**
 * @brief 检查类型是否为 SolveResult 的有效变体
 */
template <typename T>
concept IsSolveVariant =
    std::same_as<T, std::monostate> || std::same_as<T, Poses3D> || std::same_as<T, BAPoses3D>;

} // namespace auto_aim
