#pragma once

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Forward declare OpenCV types to avoid heavy header pollution
namespace cv {
class Mat;
}

namespace fcs {
struct AtLegacyArmorDetection;
}

namespace fcs::L2 {

// ============================================================================
// Detector Backend Error Types
// ============================================================================

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
    case DetectorError::NotInitialized: return "Not initialized";
    case DetectorError::InvalidImage: return "Invalid image";
    case DetectorError::InferenceError: return "Inference error";
    case DetectorError::InferenceFailed: return "Inference failed";
    case DetectorError::ModelLoadError: return "Model load error";
    case DetectorError::InitFailed: return "Initialization failed";
    case DetectorError::PreprocessFailed: return "Preprocess failed";
    }
    return "Unknown error";
}

// ============================================================================
// Detector Backend Interface (CRTP Base)
// ============================================================================

/// CRTP base for detector backends (zero-cost abstraction when type is known)
template <typename Derived>
class AtLegacyDetectorBackendBase {
public:
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;

    /// Detect armors in image
    [[nodiscard]] DetectionResult detect(const cv::Mat& image) {
        return static_cast<Derived*>(this)->detect_impl(image);
    }

    /// Check if backend is initialized
    [[nodiscard]] bool is_initialized() const noexcept {
        return static_cast<const Derived*>(this)->is_initialized_impl();
    }

    /// Get backend name
    [[nodiscard]] std::string_view name() const noexcept {
        return static_cast<const Derived*>(this)->name_impl();
    }

protected:
    AtLegacyDetectorBackendBase()  = default;
    ~AtLegacyDetectorBackendBase() = default;
};

} // namespace fcs::L2