#pragma once

#include <functional>
#include "scheduler/types.hpp"

namespace scheduler::test {

    /**
     * @brief 测试用Strategy - 支持设置lambda回调
     *
     * 用于灵活的单元测试，允许动态注册回调
     */
    class TestStrategy {
    public:
        using DetectCb = std::function<Detection(const SharedFeature &)>;
        using TrackCb = std::function<TrackResult(const Detection &, const SharedFeature &)>;
        using SolveCb = std::function<SolverOutput(const TrackResult &)>;
        using PreprocessCb = std::function<void(SharedFeature &)>;
        using OutputCb = std::function<void(const SolverOutput &)>;

        // 设置回调
        void set_detect(DetectCb cb) { detect_cb_ = cb; }
        void set_track(TrackCb cb) { track_cb_ = cb; }
        void set_solve(SolveCb cb) { solve_cb_ = cb; }
        void set_preprocess(PreprocessCb cb) { preprocess_cb_ = cb; }
        void set_output(OutputCb cb) { output_cb_ = cb; }

        // SchedulerStrategy concept实现
        inline void preprocess(SharedFeature &f) const {
            if (preprocess_cb_)
                preprocess_cb_(f);
        }

        inline Detection detect(const SharedFeature &f) const {
            if (detect_cb_)
                return detect_cb_(f);
            return Detection{};
        }

        inline TrackResult track(const Detection &d, const SharedFeature &f) const {
            if (track_cb_)
                return track_cb_(d, f);
            return TrackResult{};
        }

        inline SolverOutput solve(const TrackResult &tr) const {
            if (solve_cb_)
                return solve_cb_(tr);
            return SolverOutput{};
        }

        inline void output(const SolverOutput &out) const {
            if (output_cb_)
                output_cb_(out);
        }

    private:
        DetectCb detect_cb_;
        TrackCb track_cb_;
        SolveCb solve_cb_;
        PreprocessCb preprocess_cb_;
        OutputCb output_cb_;
    };

} // namespace scheduler::test
