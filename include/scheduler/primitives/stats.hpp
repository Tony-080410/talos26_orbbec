#pragma once

#include "latency_histogram.hpp"
#include <atomic>
#include <expected>
#include <memory>
#include <new>
#include <opencv2/core.hpp>

namespace primitive {

struct Stats {
    // Line 1: 计数器（低频，但分离避免与latency false sharing）
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> frame_in{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> frame_drop{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> detect_count{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> track_count{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> output_count{0};

    // Line 2: 最新延迟（实时监控，高频读写）
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> detect_latency{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> solve_latency{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> track_latency{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> output_latency{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> e2e_latency{0};

    // Line 3: 延迟直方图（用于百分位数统计，低频更新）
    alignas(std::hardware_destructive_interference_size) LatencyHistogram detect_hist;
    alignas(std::hardware_destructive_interference_size) LatencyHistogram solve_hist;
    alignas(std::hardware_destructive_interference_size) LatencyHistogram track_hist;
    alignas(std::hardware_destructive_interference_size) LatencyHistogram output_hist;
    alignas(std::hardware_destructive_interference_size) LatencyHistogram e2e_hist;

    void reset() noexcept {
        frame_in = frame_drop = 0;
        detect_count = track_count = output_count = 0;
        detect_latency = solve_latency = track_latency = output_latency = e2e_latency = 0;
        detect_hist.reset();
        solve_hist.reset();
        track_hist.reset();
        output_hist.reset();
        e2e_hist.reset();
    }
};

} // namespace primitive
