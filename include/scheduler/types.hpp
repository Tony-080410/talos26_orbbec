#pragma once

#include <cstdint>
#include <atomic>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "scheduler/primitives/latency_histogram.hpp"

namespace scheduler {

// ============ 0-Copy 图像缓冲 ============

/**
 * @brief 0-copy 图像缓冲包装器
 *
 * 支持两种模式:
 * 1. 外部缓冲 (相机直接提供): 持有指针，不负责释放
 * 2. 内部缓冲 (cv::Mat): 引用计数管理
 */
class ImageBuffer {
public:
    // 从外部指针构造 (0-copy, 调用者负责生命周期)
    static std::shared_ptr<ImageBuffer> wrap(void* data, int rows, int cols, int type) {
        auto buf = std::make_shared<ImageBuffer>();
        buf->mat_ = cv::Mat(rows, cols, type, data);
        buf->external_ = true;
        return buf;
    }

    // 从 cv::Mat 构造 (引用计数，可能 0-copy)
    static std::shared_ptr<ImageBuffer> from_mat(cv::Mat mat) {
        auto buf = std::make_shared<ImageBuffer>();
        buf->mat_ = std::move(mat);
        buf->external_ = false;
        return buf;
    }

    // 创建空缓冲
    static std::shared_ptr<ImageBuffer> create(int rows, int cols, int type) {
        auto buf = std::make_shared<ImageBuffer>();
        buf->mat_ = cv::Mat(rows, cols, type);
        buf->external_ = false;
        return buf;
    }

    [[nodiscard]] cv::Mat& mat() noexcept { return mat_; }
    [[nodiscard]] const cv::Mat& mat() const noexcept { return mat_; }

    [[nodiscard]] void* data() noexcept { return mat_.data; }
    [[nodiscard]] const void* data() const noexcept { return mat_.data; }

    [[nodiscard]] int rows() const noexcept { return mat_.rows; }
    [[nodiscard]] int cols() const noexcept { return mat_.cols; }
    [[nodiscard]] int type() const noexcept { return mat_.type(); }

    [[nodiscard]] bool is_external() const noexcept { return external_; }

    ImageBuffer() = default;

private:
    cv::Mat mat_;
    bool external_{false};
};

using ImagePtr = std::shared_ptr<ImageBuffer>;

// ============ 数据类型 ============

struct Frame {
    uint64_t seq{0};
    uint64_t timestamp_ns{0};
    ImagePtr image;  // 0-copy: shared_ptr 引用计数

    // 便捷构造: 从 cv::Mat
    static Frame from_mat(uint64_t seq, uint64_t ts, cv::Mat mat) {
        Frame f;
        f.seq = seq;
        f.timestamp_ns = ts;
        f.image = ImageBuffer::from_mat(std::move(mat));
        return f;
    }

    // 便捷构造: 从外部缓冲 (0-copy)
    static Frame wrap(uint64_t seq, uint64_t ts, void* data, int rows, int cols, int type) {
        Frame f;
        f.seq = seq;
        f.timestamp_ns = ts;
        f.image = ImageBuffer::wrap(data, rows, cols, type);
        return f;
    }
};

struct SharedFeature {
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    ImagePtr image;        // 0-copy: 引用 Frame 的图像
    ImagePtr preprocessed; // 预处理后的图像 (可选)
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
