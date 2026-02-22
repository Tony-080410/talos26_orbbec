#pragma once

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
class AtLegacyNNBackend;
class AtLegacyNNTupBackend;
#if TALOS_HAS_TENSORRT
class AtLegacyTrtBackend;
#endif

// Forward declare config types (defined in at_legacy_config.hpp)
struct AtLegacyConfig;
struct AtLegacyNNConfig;
struct AtLegacyNNTupConfig;
#if TALOS_HAS_TENSORRT
struct AtLegacyTrtConfig;
#endif
struct AtLegacyTraditionalConfig;

// ============================================================================
// Runtime Polymorphic Backend Wrapper (using std::variant)
// ============================================================================

/// Backend variant type for runtime switching
#if TALOS_HAS_TENSORRT
using AtLegacyDetectorBackendVariant = std::variant<
    std::unique_ptr<AtLegacyTraditionalBackend>, std::unique_ptr<AtLegacyNNBackend>,
    std::unique_ptr<AtLegacyNNTupBackend>, std::unique_ptr<AtLegacyTrtBackend>>;
#else
using AtLegacyDetectorBackendVariant = std::variant<
    std::unique_ptr<AtLegacyTraditionalBackend>, std::unique_ptr<AtLegacyNNBackend>,
    std::unique_ptr<AtLegacyNNTupBackend>>;
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
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNBackend> backend);
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNTupBackend> backend);
#if TALOS_HAS_TENSORRT
    explicit AtLegacyDetectorBackend(std::unique_ptr<AtLegacyTrtBackend> backend);
#endif

    /// Detect armors in image (dispatches to active backend)
    [[nodiscard]] DetectionResult detect(const cv::Mat& image);

    /// Check if any backend is active and initialized
    [[nodiscard]] bool is_initialized() const noexcept;

    /// Get active backend name
    [[nodiscard]] std::string_view name() const noexcept;

    /// Set detection color for all backends
    void set_detect_color(ArmorColor color) noexcept;

    /// Switch to a different backend
    void set_backend(std::unique_ptr<AtLegacyTraditionalBackend> backend);
    void set_backend(std::unique_ptr<AtLegacyNNBackend> backend);
    void set_backend(std::unique_ptr<AtLegacyNNTupBackend> backend);
#if TALOS_HAS_TENSORRT
    void set_backend(std::unique_ptr<AtLegacyTrtBackend> backend);
#endif

    /// Update config for the active backend (forward declaration, implemented in .cpp)
    void update_config(const AtLegacyConfig& config) noexcept;

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
