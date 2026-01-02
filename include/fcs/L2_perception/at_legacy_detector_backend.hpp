#pragma once

#include "../core/types.hpp"

#include <expected>
#include <memory>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>

namespace fcs {

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

// ============================================================================
// Forward Declarations for Concrete Backends
// ============================================================================

class AtLegacyTraditionalBackend;
class AtLegacyNNBackend;
class AtLegacyNNTupBackend;

// ============================================================================
// Runtime Polymorphic Backend Wrapper (using std::variant)
// ============================================================================

/// Backend variant type for runtime switching
using AtLegacyDetectorBackendVariant = std::variant<
    std::unique_ptr<AtLegacyTraditionalBackend>, std::unique_ptr<AtLegacyNNBackend>,
    std::unique_ptr<AtLegacyNNTupBackend>>;

/// Runtime-configurable detector backend wrapper
/// Implementation methods are in at_legacy_detector_backend.cpp to avoid
/// incomplete type issues with std::visit
class AtLegacyDetectorBackend {
public:
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;

    AtLegacyDetectorBackend();
    ~AtLegacyDetectorBackend();

    // Move operations
    AtLegacyDetectorBackend(AtLegacyDetectorBackend&&) noexcept;
    AtLegacyDetectorBackend& operator=(AtLegacyDetectorBackend&&) noexcept;

    // No copy
    AtLegacyDetectorBackend(const AtLegacyDetectorBackend&)            = delete;
    AtLegacyDetectorBackend& operator=(const AtLegacyDetectorBackend&) = delete;

    /// Construct with a specific backend
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyTraditionalBackend> backend);
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNBackend> backend);
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNTupBackend> backend);

    /// Detect armors in image (dispatches to active backend)
    [[nodiscard]] DetectionResult detect(const cv::Mat& image);

    /// Check if any backend is active and initialized
    [[nodiscard]] bool is_initialized() const noexcept;

    /// Get active backend name
    [[nodiscard]] std::string_view name() const noexcept;

    /// Switch to a different backend
    void set_backend(std::unique_ptr<AtLegacyTraditionalBackend> backend);
    void set_backend(std::unique_ptr<AtLegacyNNBackend> backend);
    void set_backend(std::unique_ptr<AtLegacyNNTupBackend> backend);

    /// Check if backend variant holds a specific type
    template <typename Backend>
    [[nodiscard]] bool holds() const noexcept {
        return std::holds_alternative<std::unique_ptr<Backend>>(backend_);
    }

    /// Get reference to specific backend (throws if wrong type)
    template <typename Backend>
    [[nodiscard]] Backend& get() {
        return *std::get<std::unique_ptr<Backend>>(backend_);
    }

    template <typename Backend>
    [[nodiscard]] const Backend& get() const {
        return *std::get<std::unique_ptr<Backend>>(backend_);
    }

private:
    AtLegacyDetectorBackendVariant backend_;
};

// ============================================================================
// Backend Factory (forward declarations)
// ============================================================================

// Forward declare config types (defined in at_legacy_config.hpp)
struct AtLegacyConfig;
struct AtLegacyNNConfig;
struct AtLegacyNNTupConfig;
struct AtLegacyTraditionalConfig;

/// Create detector backend from unified config
/// Implementation in at_legacy_detector_backend.cpp
[[nodiscard]] std::expected<AtLegacyDetectorBackend, DetectorError>
    create_detector_backend(const AtLegacyConfig& config);

} // namespace fcs
