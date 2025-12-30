#pragma once

#include "../L2_perception/detector.hpp"
#include "../L2_perception/solver.hpp"
#include "../L3_estimation/estimator.hpp"
#include "../L5_weapon/fire_control.hpp"
#include "../core/types.hpp"

#include <array>
#include <cstdint>
#include <new>
#include <opencv2/core.hpp>

namespace fcs::L1 {

// ============================================================================
// ImageBuffer - 图像数据所有权管理
// ============================================================================

struct ImageBuffer {
    cv::Mat data;

    static std::shared_ptr<ImageBuffer> from_mat(cv::Mat mat) {
        auto buf  = std::make_shared<ImageBuffer>();
        buf->data = std::move(mat);
        return buf;
    }

    static std::shared_ptr<ImageBuffer>
        wrap(void* ptr, const int rows, const int cols, const int type) {
        auto buf  = std::make_shared<ImageBuffer>();
        buf->data = cv::Mat(rows, cols, type, ptr);
        return buf;
    }
};

using ImagePtr = std::shared_ptr<ImageBuffer>;

// ============================================================================
// CaptureParcel - 采集帧 (Camera → Perception)
// ============================================================================

struct alignas(std::hardware_constructive_interference_size) CaptureParcel {
    uint64_t seq{0};
    uint64_t timestamp_ns{0};
    ImagePtr image; // 实际图像数据所有权
    cv::Mat gray;   // Mat header，数据指向 image
    cv::Mat binary; // Mat header，数据指向 image 或独立分配

    void clear() noexcept {
        seq          = 0;
        timestamp_ns = 0;
        image.reset();
        gray   = cv::Mat();
        binary = cv::Mat();
    }
};

// ============================================================================
// PerceptionParcel - 感知结果 (Perception → Estimation)
// ============================================================================

struct alignas(std::hardware_constructive_interference_size) PerceptionParcel {
    uint64_t timestamp_ns{0};
    uint8_t count{0};
    std::array<L2::Detection, 10> detections{};
    std::array<L2::Measurement, 10> measurements;

    void clear() noexcept {
        timestamp_ns = 0;
        count        = 0;
        // 不清空 array 内容，只重置 count
    }

    [[nodiscard]] auto begin() noexcept { return detections.begin(); }
    [[nodiscard]] auto end() noexcept { return detections.begin() + count; }
    [[nodiscard]] auto begin() const noexcept { return detections.begin(); }
    [[nodiscard]] auto end() const noexcept { return detections.begin() + count; }
};

// ============================================================================
// TrackParcel - 航迹结果 (Estimation → Output)
// ============================================================================

struct alignas(std::hardware_constructive_interference_size) TrackParcel {
    uint64_t timestamp_ns{0};
    bool has_track{false};
    L3::TrackState track{};
    L5::WeaponCommand command{};

    void clear() noexcept {
        timestamp_ns = 0;
        has_track    = false;
        command      = L5::WeaponCommand{};
    }
};

// ============================================================================
// Frame - 输入帧 (外部 → MissionComputer)
// ============================================================================

struct Frame {
    uint64_t seq{0};
    uint64_t timestamp_ns{0};
    cv::Mat image;

    static Frame from_mat(const uint64_t seq, const uint64_t ts, cv::Mat mat) {
        return Frame{seq, ts, std::move(mat)};
    }
};

} // namespace fcs::L1
