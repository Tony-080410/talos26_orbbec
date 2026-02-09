#pragma once

#include <cstdint>
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
