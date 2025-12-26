#pragma once

#include <chrono>
#include <string>

#include "detector_state.hpp"
#include "fast_tf/frame.hpp"
#include "pipeline_types.hpp"
#include "target.hpp"
#include "tracker_soa.hpp"
#include "types.hpp"

namespace auto_aim {

/**
 * @brief 装甲板跟踪器 (无状态)
 *
 * 移植自 sp_vision_25/tasks/auto_aim/tracker
 * 功能：
 * 1. 状态机管理（IDLE/DETECTING/TRACKING/TEMP_LOST）
 * 2. 目标初始化和更新
 * 3. 发散检测和收敛检测
 * 4. 坐标变换（通过fast_tf）
 *
 * 所有运行时状态通过参数传递:
 * - TrackerSoA: 状态机 + 时间戳 (64字节缓存行对齐)
 * - Target: EKF跟踪目标 (由调用方持有)
 */
class Tracker {
public:
    /**
     * @brief 构造跟踪器
     * @param config 跟踪器配置
     * @param tf_buffer TF缓冲区引用（用于坐标变换）
     */
    explicit Tracker(
        const SpVisionTrackerConfig& config, fast_tf::CoordinateSystem<100>& tf_buffer);

    // ============ TrackerConcept 方法 ============

    /**
     * @brief Concept 兼容的追踪方法
     * @param det 检测结果 (variant)
     * @param solve 解算结果 (variant)
     * @param out 追踪输出 (TrackerSoA)
     * @param debug 调试数据
     */
    void track(
        const DetectionResult& det, const SolveResult& solve, TrackerOutput& out,
        DebugData& debug) noexcept;

    /**
     * @brief 重置追踪器状态
     */
    void reset() noexcept;

    // ============ 原有方法 ============

    /**
     * @brief 跟踪装甲板 (原有接口)
     * @param state 状态机 (输入/输出)
     * @param target 跟踪目标 (输入/输出)
     * @param slots 检测结果
     * @param poses3d 解算得到的3D位姿
     * @param t 时间戳
     * @return 是否有有效目标
     */
    bool track(
        talos::TrackerSoA& state, Target& target, const talos::ArmorSlots& slots,
        const Poses3D& poses3d, std::chrono::steady_clock::time_point t) const;

    /**
     * @brief 获取目标装甲板列表 (预测位置)
     */
    [[nodiscard]] static std::vector<Eigen::Vector4d>
        armor_xyza_list(const talos::TrackerSoA& state, const Target& target);

private:
    fast_tf::CoordinateSystem<100>& tf_buffer_;
    talos::Color enemy_color_;
    int min_detect_count_;
    int normal_temp_lost_count_;
    int outpost_max_temp_lost_count_;

    /**
     * @brief 状态机更新
     */
    void state_machine(talos::TrackerSoA& state, const Target& target, bool found) const;

    /**
     * @brief 设置新目标
     */
    bool set_target(
        talos::TrackerSoA& state, Target& target, const talos::ArmorSlots& slots,
        const Poses3D& poses3d, std::chrono::steady_clock::time_point t) const;

    /**
     * @brief 更新现有目标
     */
    bool update_target(
        talos::TrackerSoA& state, Target& target, const talos::ArmorSlots& slots,
        const Poses3D& poses3d, std::chrono::steady_clock::time_point t) const;
};

/**
 * @brief ISRCKF 追踪器占位符
 *
 * Phase 4 实现。
 */
class ISRCKFTracker {
public:
    void track(
        const DetectionResult& /*det*/, const SolveResult& /*solve*/, TrackerOutput& /*out*/,
        DebugData& /*debug*/) noexcept {}

    void reset() noexcept {}
};

/**
 * @brief SRUKF 追踪器占位符
 *
 * Phase 5 实现。
 */
class SRUKFTracker {
public:
    void track(
        const DetectionResult& /*det*/, const SolveResult& /*solve*/, TrackerOutput& /*out*/,
        DebugData& /*debug*/) noexcept {}

    void reset() noexcept {}
};

} // namespace auto_aim
