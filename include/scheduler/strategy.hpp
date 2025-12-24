#pragma once

#include <concepts>
#include <cstdint>

#include "auto_aim/types.hpp"
#include "detector_state.hpp"
#include "tracker_soa.hpp"
#include "types.hpp"

namespace scheduler {

/**
 * @brief C++20 Concept: SchedulerStrategy
 *
 * 定义调度器策略的接口。任何实现此concept的类都可以作为Scheduler的模板参数。
 *
 * 数据所有权:
 * - Scheduler 持有: TrackerSoA, poses3d
 * - Strategy 持有: Target, DetectorWorkBuffers (内部状态)
 *
 * 时间戳说明:
 * - timestamp_ns: 帧采集时间戳 (纳秒)，用于fast_tf时间插值查询
 */
template <typename T>
concept SchedulerStrategy = requires(
    T& strategy, const Captured& feat, talos::ArmorSlots& armor_mut,
    const talos::ArmorSlots& armor_const, talos::TrackerSoA& tracker_mut,
    auto_aim::Poses3D& poses3d_mut, const auto_aim::Poses3D& poses3d_const, uint64_t timestamp_ns) {
    { strategy.detect(feat, armor_mut) } noexcept -> std::convertible_to<void>;
    {
        strategy.solve(armor_const, poses3d_mut, timestamp_ns)
    } noexcept -> std::convertible_to<void>;
    {
        strategy.track(tracker_mut, armor_const, poses3d_const, timestamp_ns)
    } noexcept -> std::convertible_to<void>;
    { strategy.output() } noexcept -> std::convertible_to<void>;
};

} // namespace scheduler
