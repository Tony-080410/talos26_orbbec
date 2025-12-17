#pragma once

#include <concepts>
#include "scheduler/types.hpp"

namespace scheduler {

    /**
     * @brief C++20 Concept: SchedulerStrategy
     *
     * 定义调度器策略的接口。任何实现此concept的类都可以作为Scheduler的模板参数。
     *
     * 可选方法（通过if constexpr检测）:
     * - preprocess(SharedFeature&): 预处理回调
     *
     * 必需方法:
     * - detect(const SharedFeature&) -> Detection: 检测
     * - track(const Detection&, const SharedFeature&) -> TrackResult: 跟踪
     * - solve(const TrackResult&) -> SolverOutput: 求解
     * - output(const SolverOutput&): 输出
     */
    template<typename T>
    concept SchedulerStrategy =
            requires(T &t, const SharedFeature &f, const Detection &d, const TrackResult &tr, const SolverOutput &out) {
                // 必需：检测方法
                { t.detect(f) } -> std::convertible_to<Detection>;

                // 必需：跟踪方法
                { t.track(d, f) } -> std::convertible_to<TrackResult>;

                // 必需：求解方法
                { t.solve(tr) } -> std::convertible_to<SolverOutput>;

                // 必需：输出方法
                { t.output(out) };
            };

} // namespace scheduler
