#pragma once

#include "at_legacy_detector_backend.hpp"

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

// ============================================================================
// AT Legacy NN Backend Configuration
// ============================================================================

struct AtLegacyNNBackendConfig {
    std::string model_path;
    std::string device            = "AUTO";
    std::string device_priorities = "GPU,CPU";
    bool enable_profiling         = false;
    bool enable_multi_thread      = false;

    // Model parameters
    int input_width  = 416;
    int input_height = 416;
    int num_classes  = 8;
    int num_colors   = 4;

    // Detection thresholds
    float confidence_threshold = 0.75f;
    float nms_threshold        = 0.30f;
    int top_k                  = 128;
};

// ============================================================================
// AT Legacy NN Backend Implementation
// ============================================================================

/// Neural network detection backend using OpenVINO
class AtLegacyNNBackend : public AtLegacyDetectorBackendBase<AtLegacyNNBackend> {
public:
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;

    explicit AtLegacyNNBackend(const AtLegacyNNBackendConfig& config);
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

    /// Set detection color (0=BLUE, 1=RED)
    void set_detect_color(int color) noexcept { detect_color_ = color; }
    [[nodiscard]] int detect_color() const noexcept { return detect_color_; }

    /// Get execution devices
    [[nodiscard]] std::vector<std::string> execution_devices() const;

    /// Update detection thresholds
    void set_thresholds(float confidence, float nms) {
        config_.confidence_threshold = confidence;
        config_.nms_threshold        = nms;
    }

private:
    /// Preprocess image for inference (letterbox + normalize)
    struct PreprocContext {
        cv::Mat preprocessed; // Preprocessed input tensor
        float scale_x = 1.0f; // Scale factor for x coordinates
        float scale_y = 1.0f; // Scale factor for y coordinates
        float pad_x   = 0.0f; // Padding offset x
        float pad_y   = 0.0f; // Padding offset y
    };

    [[nodiscard]] PreprocContext preprocess(const cv::Mat& image) const;

    /// Run inference
    [[nodiscard]] bool infer(const PreprocContext& ctx);

    /// Postprocess inference output
    [[nodiscard]] std::vector<AtLegacyArmorDetection> postprocess(const PreprocContext& ctx) const;

    /// Transform coordinates from model output to original image
    void transform_coordinates(
        std::vector<AtLegacyArmorDetection>& detections, const PreprocContext& ctx) const;

    /// Non-maximum suppression
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        nms(std::vector<AtLegacyArmorDetection>& detections) const;

private:
    AtLegacyNNBackendConfig config_;
    int detect_color_ = 1; // 0=BLUE, 1=RED
    bool initialized_ = false;

    // OpenVINO resources (PIMPL to avoid header pollution)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fcs
