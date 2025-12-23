#pragma once

#include <functional>
#include "auto_aim/solver.hpp"
#include "detector_state.hpp"
#include "scheduler/strategy.hpp"
#include "scheduler/types.hpp"
#include "tracker_soa.hpp"

namespace scheduler::test {

    /**
     * @brief 测试用Strategy - 支持设置lambda回调
     *
     * 用于灵活的单元测试，允许动态注册回调
     */
    class TestStrategy {
    public:
        using DetectCb = std::function<void(const Captured &, talos::ArmorSlots &)>;
        using SolveCb = std::function<void(const talos::ArmorSlots &, std::array<auto_aim::ArmorPose3D, 10> &)>;
        using TrackCb = std::function<void(talos::TrackerSoA &, const talos::ArmorSlots &,
                                           const std::array<auto_aim::ArmorPose3D, 10> &)>;
        using PreprocessCb = std::function<void(Captured &)>;
        using OutputCb = std::function<void()>;

        // 设置回调
        void set_detect(const DetectCb &cb) { detect_cb_ = cb; }
        void set_solve(const SolveCb &cb) { solve_cb_ = cb; }
        void set_track(const TrackCb &cb) { track_cb_ = cb; }
        void set_preprocess(const PreprocessCb &cb) { preprocess_cb_ = cb; }
        void set_output(const OutputCb &cb) { output_cb_ = cb; }

        // SchedulerStrategy concept实现
        void preprocess(Captured &f) const {
            if (preprocess_cb_)
                preprocess_cb_(f);
        }

        void detect(const Captured &f, talos::ArmorSlots &armor) const noexcept {
            if (detect_cb_)
                detect_cb_(f, armor);
        }

        void solve(const talos::ArmorSlots &armor,
                   std::array<auto_aim::ArmorPose3D, 10> &poses3d) const noexcept {
            if (solve_cb_)
                solve_cb_(armor, poses3d);
        }

        void track(talos::TrackerSoA &tracker_soa, const talos::ArmorSlots &armor,
                   const std::array<auto_aim::ArmorPose3D, 10> &poses3d) const noexcept {
            if (track_cb_)
                track_cb_(tracker_soa, armor, poses3d);
        }

        void output() const noexcept {
            if (output_cb_) {
                output_cb_();
            }
        }

    private:
        DetectCb detect_cb_;
        SolveCb solve_cb_;
        TrackCb track_cb_;
        PreprocessCb preprocess_cb_;
        OutputCb output_cb_;
    };

    static_assert(SchedulerStrategy<TestStrategy>);

} // namespace scheduler::test
