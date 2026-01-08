#pragma once

/**
 * @file
 * @brief Zero-copy image buffer wrapper
 *
 * This module provides a reference-counted image buffer that supports
 * both external (zero-copy) and internal (managed) memory modes.
 *
 * ## Memory modes
 *
 * - **External**: Camera provides raw pointer, buffer does not free memory
 * - **Internal**: Buffer owns cv::Mat, reference counting manages lifetime
 *
 * ## Thread safety
 *
 * The reference count is thread-safe (shared_ptr semantics).
 * Image data must not be modified while shared.
 */

#include <memory>
#include <opencv2/core.hpp>
#include <utility>

namespace primitive {

/**
 * @brief Zero-copy image buffer wrapper
 *
 * A reference-counted wrapper around cv::Mat that supports both
 * external (borrowed) and internal (owned) memory.
 *
 * ## Usage
 *
 * ```cpp
 * // External buffer (from camera):
 * auto buf = ImageBuffer::wrap(raw_data, rows, cols, CV_8UC3);
 *
 * // Internal buffer (managed):
 * auto buf = ImageBuffer::from_mat(cv::Mat(rows, cols, CV_8UC3));
 * ```
 *
 * ## Thread safety
 *
 * Reference counting is thread-safe. The underlying image data
 * should not be modified while the buffer is shared across threads.
 *
 * ## Memory model
 *
 * Uses `shared_ptr` for reference counting, allowing zero-copy
 * transfer between pipeline stages.
 */
class ImageBuffer {
public:
    /**
     * @brief Wrap external data (zero-copy)
     *
     * Creates a buffer that references external memory. The caller
     * is responsible for ensuring the external data remains valid
     * as long as the buffer exists.
     *
     * ## Performance
     *
     * Zero-copy - no data is copied.
     *
     * ## Thread safety
     *
     * Caller must ensure external data outlives the buffer.
     *
     * ## Parameters
     *
     * - `data`: pointer to external image data
     * - `rows`: number of rows in the image
     * - `cols`: number of columns in the image
     * - `type`: OpenCV image type (e.g., CV_8UC3)
     */
    static std::shared_ptr<ImageBuffer>
        wrap(void* data, const int rows, const int cols, const int type) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type, data, cv::Mat::AUTO_STEP);
        buf->external_ = true;
        return buf;
    }

    /**
     * @brief Wrap external data with explicit stride
     *
     * ## Parameters
     *
     * - `data`: pointer to external image data
     * - `rows`: number of rows
     * - `cols`: number of columns
     * - `type`: OpenCV image type
     * - `step`: number of bytes per row (stride)
     */
    static std::shared_ptr<ImageBuffer>
        wrap(void* data, const int rows, const int cols, const int type, const std::size_t step) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type, data, step);
        buf->external_ = true;
        return buf;
    }

    /**
     * @brief Wrap external data with ownership semantics
     *
     * The provided `owner` shared_ptr keeps the external data alive
     * as long as the buffer exists. When all buffers referencing
     * the data are destroyed, the owner is released.
     *
     * ## Use case
     *
     * Wrapping DMA buffers or hardware-managed memory where the
     * buffer must not be released until all consumers are done.
     *
     * ## Parameters
     *
     * - `data`: pointer to external image data
     * - `rows`: number of rows
     * - `cols`: number of columns
     * - `type`: OpenCV image type
     * - `step`: number of bytes per row (stride)
     * - `owner`: shared_ptr that controls lifetime of external data
     */
    static std::shared_ptr<ImageBuffer> wrap_owned(
        void* data, const int rows, const int cols, const int type, const std::size_t step,
        std::shared_ptr<void> owner) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type, data, step);
        buf->owner_    = std::move(owner);
        buf->external_ = true;
        return buf;
    }

    /**
     * @brief Create from cv::Mat (managed)
     *
     * The matrix is copied into the buffer. If the matrix uses
     * reference counting (e.g., from another ImageBuffer), this
     * may be a shallow copy.
     *
     * ## Performance
     *
     * May be zero-copy if the input matrix is already ref-counted.
     *
     * ## Parameters
     *
     * - `mat`: source matrix (moved from)
     */
    static std::shared_ptr<ImageBuffer> from_mat(cv::Mat mat) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = std::move(mat);
        buf->external_ = false;
        return buf;
    }

    /**
     * @brief Create empty buffer
     *
     * Allocates new memory for the specified dimensions and type.
     *
     * ## Parameters
     *
     * - `rows`: number of rows
     * - `cols`: number of columns
     * - `type`: OpenCV image type
     */
    static std::shared_ptr<ImageBuffer> create(const int rows, const int cols, const int type) {
        auto buf       = std::make_shared<ImageBuffer>();
        buf->mat_      = cv::Mat(rows, cols, type);
        buf->external_ = false;
        return buf;
    }

    /**
     * @brief Get mutable reference to underlying matrix
     *
     * ## Returns
     *
     * Reference to the cv::Mat
     */
    [[nodiscard]] cv::Mat& mat() noexcept { return mat_; }

    /**
     * @brief Get const reference to underlying matrix
     *
     * ## Returns
     *
     * Const reference to the cv::Mat
     */
    [[nodiscard]] const cv::Mat& mat() const noexcept { return mat_; }

    /**
     * @brief Get raw data pointer
     *
     * ## Returns
     *
     * Pointer to the first pixel
     */
    [[nodiscard]] void* data() noexcept { return mat_.data; }

    /**
     * @brief Get const data pointer
     *
     * ## Returns
     *
     * Const pointer to the first pixel
     */
    [[nodiscard]] const void* data() const noexcept { return mat_.data; }

    /**
     * @brief Get number of rows
     *
     * ## Returns
     *
     * Image height in pixels
     */
    [[nodiscard]] int rows() const noexcept { return mat_.rows; }

    /**
     * @brief Get number of columns
     *
     * ## Returns
     *
     * Image width in pixels
     */
    [[nodiscard]] int cols() const noexcept { return mat_.cols; }

    /**
     * @brief Get image type
     *
     * ## Returns
     *
     * OpenCV image type (e.g., CV_8UC3)
     */
    [[nodiscard]] int type() const noexcept { return mat_.type(); }

    /**
     * @brief Check if memory is external
     *
     * Returns true if the buffer wraps external (non-owned) memory.
     *
     * ## Returns
     *
     * `true` if external, `false` if owned
     */
    [[nodiscard]] bool is_external() const noexcept { return external_; }

    ImageBuffer() = default;

private:
    cv::Mat mat_;
    std::shared_ptr<void> owner_;
    bool external_{false};
};

using ImagePtr = std::shared_ptr<ImageBuffer>;
} // namespace primitive
