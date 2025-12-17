#pragma once

#include <atomic>
#include <expected>
#include <memory>
#include <opencv2/core.hpp>
#include <toml++/toml.hpp>
#include <vector>

#include "scheduler/primitives/latency_histogram.hpp"
#include "toml_helper.hpp"

namespace scheduler {

    // ============ 缓存行配置 ============

    /**
     * @brief L3缓存行大小 (x86/ARM标准)
     *
     * 目的：避免false sharing - 当多个线程访问同一缓存行中的不同变量时
     * 会导致缓存一致性协议的开销。通过alignas分离高频变量到不同的缓存行，
     * 消除缓存行竞争（cache line bouncing）。
     */
    static constexpr size_t CACHE_LINE_SIZE = 64;

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
        static std::shared_ptr<ImageBuffer> wrap(void *data, int rows, int cols, int type) {
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

        [[nodiscard]] cv::Mat &mat() noexcept { return mat_; }
        [[nodiscard]] const cv::Mat &mat() const noexcept { return mat_; }

        [[nodiscard]] void *data() noexcept { return mat_.data; }
        [[nodiscard]] const void *data() const noexcept { return mat_.data; }

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
        ImagePtr image; // 0-copy: shared_ptr 引用计数

        // 便捷构造: 从 cv::Mat
        static Frame from_mat(uint64_t seq, uint64_t ts, cv::Mat mat) {
            Frame f;
            f.seq = seq;
            f.timestamp_ns = ts;
            f.image = ImageBuffer::from_mat(std::move(mat));
            return f;
        }

        // 便捷构造: 从外部缓冲 (0-copy)
        static Frame wrap(uint64_t seq, uint64_t ts, void *data, int rows, int cols, int type) {
            Frame f;
            f.seq = seq;
            f.timestamp_ns = ts;
            f.image = ImageBuffer::wrap(data, rows, cols, type);
            return f;
        }
    };

    struct SharedFeature {
        // Line 1: 帧元数据（detector写，tracker读）
        alignas(CACHE_LINE_SIZE) uint64_t frame_seq{0};
        uint64_t timestamp_ns{0};
        uint32_t padding1[6]; // 填充到64字节

        // Line 2: 图像数据（detector写，tracker读，solver读）
        alignas(CACHE_LINE_SIZE) ImagePtr image;

        // Line 3: 预处理后图像（可选）
        alignas(CACHE_LINE_SIZE) ImagePtr preprocessed;
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
        int detector_mode{0}; // 0=传统, 1=NN吞吐, 2=NN延迟
        int detector_threads{4};
        int max_frame_skip{2};

        std::expected<void, std::string> read_from(const toml::table &tbl) {
            if (const auto _t = tbl["runtime"].as_table()) {
                auto t = *_t;
                READ(t, "detector_mode", detector_mode);
                READ(t, "detector_threads", detector_threads);
                READ(t, "max_frame_skip", max_frame_skip);
                return {};
            }
            return std::unexpected("runtime is not a table");
        }
    };

    // ============ 统计信息 ============

    struct Stats {
        // Line 1: 计数器（低频，但分离避免与latency false sharing）
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> frame_in{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> frame_drop{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> detect_count{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> track_count{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> solve_count{0};

        // Line 2: 最新延迟（实时监控，高频读写）
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> detect_latency{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> track_latency{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> solve_latency{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> e2e_latency{0};

        // Line 3: 延迟直方图（用于百分位数统计，低频更新）
        alignas(CACHE_LINE_SIZE) LatencyHistogram detect_hist;
        alignas(CACHE_LINE_SIZE) LatencyHistogram track_hist;
        alignas(CACHE_LINE_SIZE) LatencyHistogram solve_hist;
        alignas(CACHE_LINE_SIZE) LatencyHistogram e2e_hist;

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
