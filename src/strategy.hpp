#pragma once
#include "detector.hpp"
#include "primitives/buffer_pool.hpp"
#include "primitives/latency_stats.hpp"
#include "scheduler/strategy.hpp"
#include "target.hpp"
#include "tracker.hpp"
#include "types.hpp"

class TalosStrategy {
public:
    // SchedulerStrategy concept methods
    void detect(const scheduler::SharedFeature &feat, talos::ArmorSlots &armor) noexcept;
    void solve(const talos::ArmorSlots &armor, auto_aim::Poses3D &poses3d) const noexcept;
    void track(talos::TrackerSoA &tracker_soa, const talos::ArmorSlots &armor,
               const auto_aim::Poses3D &poses3d) noexcept;
    void output() noexcept;

    // Additional methods (not part of concept)
    void preprocess(scheduler::SharedFeature &sf);

    TalosStrategy(const bool debug, const SpVisionConfig &config, const size_t buffer_count, const int rows,
                  const int cols) noexcept :
        detector(config, debug), solver(config.solver), tracker(config.tracker),
        frame_pool(buffer_count, rows, cols, CV_8UC3), gray_pool(buffer_count, rows, cols, CV_8UC1),
        small_pool(buffer_count, rows, cols, CV_8UC1) {}


    // Initialize pipeline components (stateless)
    auto_aim::TraditionalDetector detector;
    auto_aim::Solver solver;
    auto_aim::Tracker tracker;

    // Buffer pool (复用 buffer)
    scheduler::BufferPool frame_pool;
    scheduler::BufferPool gray_pool;
    scheduler::BufferPool small_pool;

    // 统计
    scheduler::LatencyStats capture_latency;
    scheduler::LatencyStats preprocess_latency;
    scheduler::LatencyStats detect_latency;
    scheduler::LatencyStats track_latency;
    scheduler::LatencyStats e2e_latency;

    std::atomic<uint64_t> frame_count{0};
    std::atomic<uint64_t> processed_count{0};
    std::atomic<uint64_t> output_count{0};

private:
    // Internal state owned by strategy
    auto_aim::Target target_{};                          // EKF 跟踪目标
    auto_aim::DetectorWorkBuffers detector_buffers_{};   // 检测器工作缓冲
};
static_assert(scheduler::SchedulerStrategy<TalosStrategy>);
