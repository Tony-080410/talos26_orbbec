/**
 * @brief Hikcamera stub implementation (SDK not available)
 */
#include "hik_camera.hpp"

namespace hikcamera {

// SDK 全局初始化/清理（stub 版本）
std::expected<void, CameraError> initialize_sdk() {
    return std::unexpected(CameraError{"HIK SDK not available (stub build)", 0});
}

void finalize_sdk() {
    // Stub 版本无需清理
}

class ImageCapturer::Impl {};

ImageCapturer::ImageCapturer()
    : impl_(nullptr) {}

ImageCapturer::~ImageCapturer() = default;

std::expected<std::unique_ptr<ImageCapturer>, CameraError> ImageCapturer::create(
    const CameraProfile& /*profile*/, const char* /*user_defined_name*/,
    const SyncMode& /*sync_mode*/) {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<void, CameraError> ImageCapturer::init(
    const CameraProfile& /*profile*/, const char* /*user_defined_name*/,
    const SyncMode& /*sync_mode*/) {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<cv::Mat, CameraError>
    ImageCapturer::read(std::chrono::duration<unsigned int, std::milli> /*timeout*/) {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<std::tuple<int, int>, CameraError> ImageCapturer::get_width_height() const {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<void, CameraError> ImageCapturer::software_trigger_on() {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<void, CameraError>
    ImageCapturer::set_frame_rate_inner_trigger_mode(float /*frame_rate*/) {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

} // namespace hikcamera
