#pragma once

#include "L2_perception/armor/at_legacy_readback_roi.hpp"
#include "core/types.hpp"

#include <expected>
#include <memory>
#include <variant>
#include <vector>

// Forward declare OpenCV types to avoid heavy header pollution
namespace cv {
class Mat;
}

#include "backends/at_legacy_detector_backend_base.hpp"

namespace fcs::L2 {

// ============================================================================
// Forward Declarations for Concrete Backends
// ============================================================================

class AtLegacyTraditionalBackend;
#if TALOS_HAS_OPENVINO
class AtLegacyNNBackend;
class AtLegacyNNTupBackend;
#endif
#if TALOS_HAS_TENSORRT
class AtLegacyTrtBackend;
#endif
#if TALOS_HAS_AXERA
class AtLegacyAxeraBackend;
#endif

// Forward declare config types (defined in at_legacy_config.hpp)
struct AtLegacyConfig;
#if TALOS_HAS_OPENVINO
struct AtLegacyNNConfig;
struct AtLegacyNNTupConfig;
#endif
#if TALOS_HAS_TENSORRT
struct AtLegacyTrtConfig;
#endif
#if TALOS_HAS_AXERA
struct AtLegacyAxeraConfig;
#endif
struct AtLegacyTraditionalConfig;

// ============================================================================
// Runtime Polymorphic Backend Wrapper (using std::variant)
// ============================================================================

/// Backend variant type for runtime switching
#if TALOS_HAS_AXERA
using AtLegacyDetectorBackendVariant = std::variant<
    std::unique_ptr<AtLegacyTraditionalBackend>,
# if TALOS_HAS_OPENVINO
    std::unique_ptr<AtLegacyNNBackend>, std::unique_ptr<AtLegacyNNTupBackend>,
# endif
# if TALOS_HAS_TENSORRT
    std::unique_ptr<AtLegacyTrtBackend>,
# endif
    std::unique_ptr<AtLegacyAxeraBackend>>;
#elif TALOS_HAS_TENSORRT
using AtLegacyDetectorBackendVariant = std::variant<
    std::unique_ptr<AtLegacyTraditionalBackend>,
# if TALOS_HAS_OPENVINO
    std::unique_ptr<AtLegacyNNBackend>, std::unique_ptr<AtLegacyNNTupBackend>,
# endif
    std::unique_ptr<AtLegacyTrtBackend>>;
#elif TALOS_HAS_OPENVINO
using AtLegacyDetectorBackendVariant = std::variant<
    std::unique_ptr<AtLegacyTraditionalBackend>, std::unique_ptr<AtLegacyNNBackend>,
    std::unique_ptr<AtLegacyNNTupBackend>>;
#else
using AtLegacyDetectorBackendVariant = std::variant<std::unique_ptr<AtLegacyTraditionalBackend>>;
#endif

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
#if TALOS_HAS_OPENVINO
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNBackend> backend);
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNTupBackend> backend);
#endif
#if TALOS_HAS_TENSORRT
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyTrtBackend> backend);
#endif
#if TALOS_HAS_AXERA
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyAxeraBackend> backend);
#endif

    /// Detect armors in image (dispatches to active backend)
    [[nodiscard]] DetectionResult detect(const cv::Mat& image);

    /// Check if any backend is active and initialized
    [[nodiscard]] bool is_initialized() const noexcept;

    /// Get active backend name
    [[nodiscard]] std::string_view name() const noexcept;

    /// Set detection color for all backends
    void set_detect_color(ArmorColor color) noexcept;

    /// Whether the active backend supports readback ROI crop.
    [[nodiscard]] bool supports_readback_roi() const noexcept;

    /// Input resolution used to derive ROI aspect ratio and minimum crop size.
    [[nodiscard]] AtLegacyBackendInputResolution input_resolution() const noexcept;

    /// Switch to a different backend
    void set_backend(std::unique_ptr<AtLegacyTraditionalBackend> backend);
#if TALOS_HAS_OPENVINO
    void set_backend(std::unique_ptr<AtLegacyNNBackend> backend);
    void set_backend(std::unique_ptr<AtLegacyNNTupBackend> backend);
#endif
#if TALOS_HAS_TENSORRT
    void set_backend(std::unique_ptr<AtLegacyTrtBackend> backend);
#endif
#if TALOS_HAS_AXERA
    void set_backend(std::unique_ptr<AtLegacyAxeraBackend> backend);
#endif

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

/// Create detector backend from unified config
/// Implementation in at_legacy_detector_backend.cpp
[[nodiscard]] std::expected<AtLegacyDetectorBackend, DetectorError>
    create_detector_backend(const AtLegacyConfig& config);

} // namespace fcs::L2
