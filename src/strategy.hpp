#pragma once
#include "detector.hpp"
#include "primitives/buffer_pool.hpp"
#include "primitives/latency_stats.hpp"
#include "tracker.hpp"
#include "types.hpp"

class TalosStrategy {
public:
    void preprocess(scheduler::SharedFeature &sf);
    scheduler::Detection detect(const scheduler::SharedFeature &sf);
    scheduler::TrackResult track(const scheduler::Detection &det, const scheduler::SharedFeature &sf);
    scheduler::SolverOutput solve(const scheduler::TrackResult &tr);
    void output(const scheduler::SolverOutput &out);

    TalosStrategy(const bool debug, const SpVisionConfig &config, const size_t buffer_count, const int rows,
                  const int cols) noexcept :
        detector(config, debug), solver(config.solver), tracker(config.tracker), frame_pool(buffer_count, rows, cols, CV_8UC3),
        gray_pool(buffer_count, rows, cols, CV_8UC1), small_pool(buffer_count, rows, cols, CV_8UC1) {}


    // Initialize pipeline components
    auto_aim::TraditionalDetector detector;
    auto_aim::Solver solver;
    auto_aim::Tracker tracker;

    talos::ArmorSlots slots{};
    std::array<auto_aim::ArmorPose3D, 10> poses3d{};

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
};
static_assert(scheduler::SchedulerStrategy<TalosStrategy>);
