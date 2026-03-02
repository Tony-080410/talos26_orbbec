#pragma once

#include <expected>
#include <string_view>
#include <vector>

#include "core/armor_types.hpp"
#include "core/types.hpp"

namespace cv {
class Mat;
}

namespace fcs::L2 {

enum class DetectorError {
    NotInitialized,
    InvalidImage,
    InferenceError,
    InferenceFailed,
    ModelLoadError,
    InitFailed,
    PreprocessFailed,
};

[[nodiscard]] constexpr std::string_view detector_error_str(DetectorError e) noexcept {
    switch (e) {
    case DetectorError::NotInitialized: return "not_initialized";
    case DetectorError::InvalidImage: return "invalid_image";
    case DetectorError::InferenceError: return "inference_error";
    case DetectorError::InferenceFailed: return "inference_failed";
    case DetectorError::ModelLoadError: return "model_load_error";
    case DetectorError::InitFailed: return "init_failed";
    case DetectorError::PreprocessFailed: return "preprocess_failed";
    }
    return "unknown";
}

class IDetectorBackend {
public:
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;

    virtual ~IDetectorBackend() = default;

    [[nodiscard]] virtual DetectionResult detect(const cv::Mat& image) = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept       = 0;

    virtual void set_detect_color(ArmorColor) noexcept {}
    [[nodiscard]] virtual ArmorColor detect_color() const noexcept { return ArmorColor::Blue; }
};

} // namespace fcs::L2
