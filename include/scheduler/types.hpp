#pragma once

#include <cstdint>
#include <atomic>
#include <vector>
#include <opencv2/core.hpp>
#include "scheduler/primitives/latency_histogram.hpp"

namespace scheduler {

// ============ 数据类型 ============

struct Frame {
    uint64_t seq{0};
    uint64_t timestamp_ns{0};
    cv::Mat image;
};

struct SharedFeature {
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    cv::Mat image;
    cv::Mat preprocessed;
};

struct Detection {
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
};

struct TrackResult {
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    // 跟踪状态扩展...
};

struct SolverOutput {
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    // 解算结果: pitch, yaw, 预测位置等
};

// ============ 运行时配置 ============

struct RuntimeConfig {
    // Detector
    int detector_mode{0};        // 0=传统, 1=NN吞吐, 2=NN延迟
    int detector_threads{4};
    int max_frame_skip{2};

    // Tracker
    float smooth_alpha{0.8f};

    // Solver
    int solver_max_iter{10};
};

// ============ 统计信息 ============

struct Stats {
    // 计数器
    std::atomic<uint64_t> frame_in{0};
    std::atomic<uint64_t> frame_drop{0};
    std::atomic<uint64_t> detect_count{0};
    std::atomic<uint64_t> track_count{0};
    std::atomic<uint64_t> solve_count{0};

    // 最新延迟 (ns) - 用于实时监控
    std::atomic<uint64_t> detect_latency{0};
    std::atomic<uint64_t> track_latency{0};
    std::atomic<uint64_t> solve_latency{0};
    std::atomic<uint64_t> e2e_latency{0};

    // 延迟直方图 - 用于百分位数统计
    LatencyHistogram detect_hist;
    LatencyHistogram track_hist;
    LatencyHistogram solve_hist;
    LatencyHistogram e2e_hist;

    void reset() noexcept {
        frame_in = frame_drop = 0;
        detect_count = track_count = solve_count = 0;
        detect_latency = track_latency = solve_latency = e2e_latency = 0;
        detect_hist.reset();
        track_hist.reset();
        solve_hist.reset();
        e2e_hist.reset();
    }
};

} // namespace scheduler
