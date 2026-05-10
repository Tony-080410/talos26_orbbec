/**
 * @brief Hikcamera stub implementation (SDK not available)
 */
#include "hik_camera.hpp"

namespace hikcamera {

std::string error_code_to_message(unsigned int code) noexcept { return "unsupported"; }

class ImageCapturer::Impl {};

ImageCapturer::ImageCapturer() noexcept
    : impl_(nullptr) {}

ImageCapturer::~ImageCapturer() noexcept = default;

std::expected<std::unique_ptr<ImageCapturer>, CameraError> ImageCapturer::create(
    const CameraProfile& /*profile*/, const char* /*user_defined_name*/,
    const SyncMode& /*sync_mode*/) noexcept {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<void, CameraError> ImageCapturer::init(
    const CameraProfile& /*profile*/, const char* /*user_defined_name*/,
    const SyncMode& /*sync_mode*/) noexcept {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<cv::Mat, CameraError>
    ImageCapturer::read(std::chrono::duration<unsigned int, std::milli> /*timeout*/) noexcept {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<std::tuple<int, int>, CameraError> ImageCapturer::get_width_height() const noexcept {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<void, CameraError> ImageCapturer::software_trigger_on() noexcept {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<void, CameraError>
    ImageCapturer::set_frame_rate_inner_trigger_mode(float /*frame_rate*/) noexcept {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

std::expected<void, CameraError> ImageCapturer::stop_grabbing() noexcept {
    return std::unexpected(
        CameraError{.message = "HIK SDK not available (stub build)", .error_code = 0});
}

} // namespace hikcamera
