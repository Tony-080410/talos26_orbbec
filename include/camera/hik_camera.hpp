/**
 * @author Qzh (zihanqin2048@gmail.com)
 * @brief Hikcamera
 * @copyright Copyright (c) 2024 by Alliance, All Rights Reserved.
 */
#pragma once

#include "fcs/camera_config.hpp"
#include <chrono>
#include <expected>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <ratio>
#include <string>
#include <tuple>

namespace hikcamera {

enum class SyncMode {
    NONE,

    SOFTWARE // use soft trigger and multi thread timer to get image
};

struct CameraError {
    std::string message;
    unsigned int error_code = 0;

    std::string what() const { return message + " (code: " + std::to_string(error_code) + ")"; }
};

class ImageCapturer final {
public:
    struct CameraProfile {
        CameraProfile() noexcept {
            using namespace std::chrono_literals;
            trigger_mode  = false;
            invert_image  = false;
            exposure_time = 10ms;
            gain          = 16.7;
            rotate_type   = fcs::RotateType::None;
        }

        bool trigger_mode;
        bool invert_image;

        std::chrono::duration<float, std::micro> exposure_time;
        float gain;
        fcs::RotateType rotate_type;
    };

    // Factory method for creating ImageCapturer with std::expected
    static std::expected<std::unique_ptr<ImageCapturer>, CameraError> create(
        const CameraProfile& profile = CameraProfile{}, const char* user_defined_name = nullptr,
        const SyncMode& sync_mode = SyncMode::NONE);

    ImageCapturer(const ImageCapturer&)            = delete;
    ImageCapturer& operator=(const ImageCapturer&) = delete;

    ~ImageCapturer();

    [[nodiscard]] std::expected<cv::Mat, CameraError>
        read(std::chrono::duration<unsigned int, std::milli> timeout = std::chrono::seconds(5));

    [[nodiscard]] std::expected<std::tuple<int, int>, CameraError> get_width_height() const;

    [[nodiscard]] std::expected<void, CameraError> software_trigger_on();
    [[nodiscard]] std::expected<void, CameraError>
        set_frame_rate_inner_trigger_mode(float frame_rate);

private:
    // Default constructor for internal use by create()
    ImageCapturer();

    [[nodiscard]] std::expected<void, CameraError> init(
        const CameraProfile& profile, const char* user_defined_name, const SyncMode& sync_mode);

    friend std::expected<std::unique_ptr<ImageCapturer>, CameraError>
        ImageCapturer::create(const CameraProfile&, const char*, const SyncMode&);

    class Impl;
    Impl* impl_;
};

} // namespace hikcamera
