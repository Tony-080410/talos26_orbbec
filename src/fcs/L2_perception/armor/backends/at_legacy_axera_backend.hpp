#pragma once

#include "../at_legacy_config.hpp"
#include "at_legacy_detector_backend_base.hpp"
#include "core/armor_types.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// AX650 NPU API
#include "ax_engine_api.h"
#include "ax_sys_api.h"

namespace fcs::L2 {

// ============================================================================
// AT Legacy Axera Backend Implementation
// ============================================================================

/// Neural network detection backend using AX650 NPU
/// Implements RAII for AX engine resources
class AtLegacyAxeraBackend : public AtLegacyDetectorBackendBase<AtLegacyAxeraBackend> {
public:
    static constexpr int INPUT_W     = 640;
    static constexpr int INPUT_H     = 640; // Square input for letterbox
    static constexpr int NUM_COLORS  = 4;
    static constexpr int NUM_CLASSES = 9;   // Class2 from AXCL
    static constexpr int NUM_KPTS    = 4;   // 4 corner keypoints (x, y)
    static constexpr int NUM_PAIRS   = 12;  // Armor type classes

    // Feature flag: support both 4-output and single-output models formats
    static constexpr bool SUPPORTS_FOUR_HEAD_OUTPUT = true;

    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;
    using Config          = AtLegacyAxeraConfig;

    explicit AtLegacyAxeraBackend(Config config);
    ~AtLegacyAxeraBackend();

    // Move-only (AX engine resources are non-copyable)
    AtLegacyAxeraBackend(AtLegacyAxeraBackend&&) noexcept;
    AtLegacyAxeraBackend& operator=(AtLegacyAxeraBackend&&) noexcept;
    AtLegacyAxeraBackend(const AtLegacyAxeraBackend&)            = delete;
    AtLegacyAxeraBackend& operator=(const AtLegacyAxeraBackend&) = delete;

    /// Initialize the backend (load model)
    [[nodiscard]] std::expected<void, DetectorError> init();

    /// Detect armors in image (synchronous)
    [[nodiscard]] DetectionResult detect_impl(const cv::Mat& image);

    [[nodiscard]] bool is_initialized_impl() const noexcept { return initialized_; }
    [[nodiscard]] std::string_view name_impl() const noexcept { return "axera"; }

    void set_detect_color(ArmorColor color) noexcept { detect_color_ = color; }
    [[nodiscard]] ArmorColor detect_color() const noexcept { return detect_color_; }

    /// Get current config
    [[nodiscard]] const Config& get_config() const noexcept { return config_; }

private:
    /// Preprocess context for letterbox
    struct PreprocContext {
        float ratio = 1.0f; // Scale factor (same for both axes)
        float pad_w = 0.0f; // Horizontal padding (pixels)
        float pad_h = 0.0f; // Vertical padding (pixels)
        int orig_w  = 0;
        int orig_h  = 0;
    };

    /// Preprocess image using letterbox (maintain aspect ratio) - compute params only
    [[nodiscard]] PreprocContext
        preprocess_letterbox(const cv::Mat& image, const AtLegacyAxeraConfig& cfg) const;

    /// Preprocess image into NPU buffer (resize + BGR2RGB + cache flush)
    void preprocess_image(const cv::Mat& image, const PreprocContext& ctx);

    /// Run pure NPU inference (assumes input buffer is ready)
    [[nodiscard]] bool infer();

    /// Decoded box from model output (shared between 4-head and legacy)
    struct DecodedBox {
        float kpts[8];    // 4 keypoints x,y
        float conf;       // confidence
        float class1[4];  // color class
        float class2[12]; // armor pair class (12 pairs)
    };

    /// Decode grid anchor centers (for 4-head predecode outputs)
    struct DecodeGrid {
        std::vector<std::pair<float, float>> grid_xy; // (x, y) anchor centers
        std::vector<float> strides;                   // stride per anchor
    };

    /// Build decode grid for anchor-based decoding (called once at init)
    [[nodiscard]] DecodeGrid build_decode_grid() const;

    /// Cached decode grid (built once at initialization)
    DecodeGrid cached_grid_;

    /// Decode 4-head predecode outputs into unified predictions
    [[nodiscard]] std::vector<DecodedBox> decode_four_head_outputs(
        const float* raw_boxes,    // Shape: (1, 4, num_anchors)
        const float* color_scores, // Shape: (1, 4, num_anchors)
        const float* obj_scores,   // Shape: (1, num_pairs, num_anchors)
        const float* raw_kpts,     // Shape: (1, 4, 2, num_anchors)
        int num_anchors, const DecodeGrid& grid) const;

    /// Scale coordinates back to original image space (letterbox inverse)
    [[nodiscard]] static cv::Point2f scale_point(const cv::Point2f& pt, const PreprocContext& ctx);

    /// Postprocess inference output (4-head network)
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        postprocess_four_head(const PreprocContext& ctx, const AtLegacyAxeraConfig& cfg) const;

    /// Non-maximum suppression with keypoints
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        nms(std::vector<AtLegacyArmorDetection>& detections, const AtLegacyAxeraConfig& cfg) const;

    /// Check if two boxes overlap (based on keypoints)
    [[nodiscard]] static bool
        is_overlap(const AtLegacyArmorDetection& box1, const AtLegacyArmorDetection& box2);

private:
    Config config_;
    ArmorColor detect_color_ = ArmorColor::Blue;
    bool initialized_        = false;

    // AX650 NPU resources
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_T io_data_{};
    std::vector<char> model_buffer_;
};

} // namespace fcs::L2
