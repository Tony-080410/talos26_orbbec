#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <utility>

namespace primitive {

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
    static std::shared_ptr<ImageBuffer>
        wrap(void* data, const int rows, const int cols, const int type) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type, data, cv::Mat::AUTO_STEP);
        buf->external_ = true;
        return buf;
    }

    // 从外部指针构造 (0-copy, 允许指定 step)
    static std::shared_ptr<ImageBuffer>
        wrap(void* data, const int rows, const int cols, const int type, const std::size_t step) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type, data, step);
        buf->external_ = true;
        return buf;
    }

    // 从外部指针构造 (0-copy, 通过 owner 保活/释放外部资源)
    static std::shared_ptr<ImageBuffer> wrap_owned(
        void* data, const int rows, const int cols, const int type, const std::size_t step,
        std::shared_ptr<void> owner) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type, data, step);
        buf->owner_    = std::move(owner);
        buf->external_ = true;
        return buf;
    }

    // 从 cv::Mat 构造 (引用计数，可能 0-copy)
    static std::shared_ptr<ImageBuffer> from_mat(cv::Mat mat) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = std::move(mat);
        buf->external_ = false;
        return buf;
    }

    // 创建空缓冲
    static std::shared_ptr<ImageBuffer> create(const int rows, const int cols, const int type) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type);
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
    std::shared_ptr<void> owner_;
    bool external_{false};
};

using ImagePtr = std::shared_ptr<ImageBuffer>;
} // namespace primitive
