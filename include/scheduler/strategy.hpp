#pragma once

#include <concepts>

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
     */
    template<typename T>
    concept SchedulerStrategy = requires(T &strategy, const Captured &feat, talos::ArmorSlots &armor_mut,
                                         const talos::ArmorSlots &armor_const, talos::TrackerSoA &tracker_mut,
                                         auto_aim::Poses3D &poses3d_mut,
                                         const auto_aim::Poses3D &poses3d_const) {
        { strategy.detect(feat, armor_mut) } noexcept -> std::convertible_to<void>;
        { strategy.solve(armor_const, poses3d_mut) } noexcept -> std::convertible_to<void>;
        { strategy.track(tracker_mut, armor_const, poses3d_const) } noexcept -> std::convertible_to<void>;
        { strategy.output() } noexcept -> std::convertible_to<void>;
    };

} // namespace scheduler
