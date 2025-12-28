#pragma once

#include "primitives/frame.hpp"
#include "primitives/latency_histogram.hpp"
#include <atomic>
#include <expected>
#include <memory>
#include <new>
#include <opencv2/core.hpp>
#include <vector>

namespace scheduler {

using primitive::LatencyHistogram;

// Forward declarations for backward compatibility
} // namespace scheduler

// Re-export primitives in scheduler namespace (backward compatibility)
#include "primitives/buffer_pool.hpp"
#include "primitives/latency_stats.hpp"

namespace scheduler {
using primitive::BufferPool;
using primitive::LatencyStats;

// ============ 缓存行配置 ============

/**
 * @brief L3缓存行大小 (x86/ARM标准)
 *
 * 目的：避免false sharing - 当多个线程访问同一缓存行中的不同变量时
 * 会导致缓存一致性协议的开销。通过alignas分离高频变量到不同的缓存行，
 * 消除缓存行竞争（cache line bouncing）。
 */
static constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;

// ============ 0-Copy 图像缓冲 ============

// Use primitive::ImageBuffer for backward compatibility
using ImageBuffer = primitive::ImageBuffer;
using ImagePtr    = primitive::ImagePtr;

// ============ 数据类型 ============

struct Frame {
    uint64_t seq{0};
    uint64_t timestamp_ns{0};
    ImagePtr image; // 0-copy: shared_ptr 引用计数

    // 便捷构造: 从 cv::Mat
    static Frame from_mat(const uint64_t seq, const uint64_t ts, cv::Mat mat) {
        Frame f;
        f.seq          = seq;
        f.timestamp_ns = ts;
        f.image        = ImageBuffer::from_mat(std::move(mat));
        return f;
    }

    // 便捷构造: 从外部缓冲 (0-copy)
    static Frame wrap(
        const uint64_t seq, const uint64_t ts, void* data, const int rows, const int cols,
        const int type) {
        Frame f;
        f.seq          = seq;
        f.timestamp_ns = ts;
        f.image        = ImageBuffer::wrap(data, rows, cols, type);
        return f;
    }
};

struct Captured {
    // Line 1: 帧元数据（detector写，tracker读）
    alignas(CACHE_LINE_SIZE) uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    uint32_t padding1[6]; // 填充到64字节

    // Line 2: 图像数据（detector写，tracker读，solver读）
    alignas(CACHE_LINE_SIZE) ImagePtr image;

    // Line 3: 预处理后图像（可选）
    alignas(CACHE_LINE_SIZE) ImagePtr preprocessed;

    // Line 4: 调试数据 - 二值化图像
    alignas(CACHE_LINE_SIZE) ImagePtr binary_img;

    // Line 5: 调试数据 - 数字识别图案 (pattern)
    alignas(CACHE_LINE_SIZE) ImagePtr pattern_img;
};

// ============ 运行时配置 ============
struct RuntimeConfig {
    // Detector
    int detector_mode{0}; // 0=传统, 1=NN吞吐, 2=NN延迟
};

// ============ 统计信息 ============

struct Stats {
    // Line 1: 计数器（低频，但分离避免与latency false sharing）
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> frame_in{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> frame_drop{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> detect_count{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> track_count{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> output_count{0};

    // Line 2: 最新延迟（实时监控，高频读写）
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> detect_latency{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> solve_latency{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> track_latency{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> output_latency{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> e2e_latency{0};

    // Line 3: 延迟直方图（用于百分位数统计，低频更新）
    alignas(CACHE_LINE_SIZE) LatencyHistogram detect_hist;
    alignas(CACHE_LINE_SIZE) LatencyHistogram solve_hist;
    alignas(CACHE_LINE_SIZE) LatencyHistogram track_hist;
    alignas(CACHE_LINE_SIZE) LatencyHistogram output_hist;
    alignas(CACHE_LINE_SIZE) LatencyHistogram e2e_hist;

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

} // namespace scheduler
