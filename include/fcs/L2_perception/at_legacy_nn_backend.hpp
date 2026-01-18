#pragma once

#include "at_legacy_detector_backend.hpp"
#include "core/types.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Forward declare OpenVINO types to avoid header pollution
namespace ov {
class Core;
class CompiledModel;
class InferRequest;
} // namespace ov

namespace fcs {

// Forward declare config (defined in at_legacy_config.hpp)
struct AtLegacyNNConfig;

// ============================================================================
// AT Legacy NN Backend Implementation
// ============================================================================

/// Neural network detection backend using OpenVINO
/// Uses atomic shared_ptr for config hot-reload
class AtLegacyNNBackend : public AtLegacyDetectorBackendBase<AtLegacyNNBackend> {
public:
    static constexpr int INPUT_W     = 416;
    static constexpr int INPUT_H     = 416;
    static constexpr int NUM_COLORS  = 4;
    static constexpr int NUM_SIZES   = 2;
    static constexpr int NUM_CLASSES = 8; // G, 1, 2, 3, 4, 5, O, B
    static constexpr int NUM_KPTS    = 4; // 4 corner keypoints (x, y)
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;
    using ConfigPtr       = std::shared_ptr<const AtLegacyNNConfig>;

    explicit AtLegacyNNBackend(ConfigPtr config);
    ~AtLegacyNNBackend();

    // Move-only (OpenVINO resources)
    AtLegacyNNBackend(AtLegacyNNBackend&&) noexcept;
    AtLegacyNNBackend& operator=(AtLegacyNNBackend&&) noexcept;
    AtLegacyNNBackend(const AtLegacyNNBackend&)            = delete;
    AtLegacyNNBackend& operator=(const AtLegacyNNBackend&) = delete;

    /// Initialize the backend (load model)
    [[nodiscard]] std::expected<void, DetectorError> init();

    /// Detect armors in image (synchronous)
    [[nodiscard]] DetectionResult detect_impl(const cv::Mat& image);

    [[nodiscard]] bool is_initialized_impl() const noexcept { return initialized_; }
    [[nodiscard]] std::string_view name_impl() const noexcept { return "nn"; }

    void set_detect_color(ArmorColor color) noexcept { detect_color_ = color; }
    [[nodiscard]] ArmorColor detect_color() const noexcept { return detect_color_; }

    /// Get execution devices
    [[nodiscard]] std::vector<std::string> execution_devices() const;

    /// Hot-reload config (thread-safe)
    void update_config(ConfigPtr new_config) noexcept {
        std::atomic_store(&config_, std::move(new_config));
    }

    /// Get current config (thread-safe)
    [[nodiscard]] ConfigPtr get_config() const noexcept { return std::atomic_load(&config_); }

private:
    /// Preprocess image for inference (letterbox + normalize)
    struct PreprocContext {
        cv::Mat preprocessed; // Preprocessed input tensor
        float scale_x = 1.0f; // Scale factor for x coordinates
        float scale_y = 1.0f; // Scale factor for y coordinates
        float pad_x   = 0.0f; // Padding offset x
        float pad_y   = 0.0f; // Padding offset y
        int orig_w    = 0;
        int orig_h    = 0;
    };

    [[nodiscard]] PreprocContext
        preprocess(const cv::Mat& image, const AtLegacyNNConfig& cfg) const;

    /// Run inference
    [[nodiscard]] bool infer(const PreprocContext& ctx);

    /// Postprocess inference output
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        postprocess(const PreprocContext& ctx, const AtLegacyNNConfig& cfg) const;

    /// Transform coordinates from model output to original image
    void transform_coordinates(
        std::vector<AtLegacyArmorDetection>& detections, const PreprocContext& ctx) const;

    /// Non-maximum suppression
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        nms(std::vector<AtLegacyArmorDetection>& detections, const AtLegacyNNConfig& cfg) const;

private:
    ConfigPtr config_; // Atomic shared_ptr for thread-safe config access
    ArmorColor detect_color_ = ArmorColor::Blue;
    bool initialized_        = false;

    // OpenVINO resources (PIMPL to avoid header pollution)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fcs
