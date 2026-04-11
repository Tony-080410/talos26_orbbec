#pragma once

#include "../at_legacy_config.hpp"
#include "at_legacy_detector_backend_base.hpp"
#include "core/armor_types.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// AX650 NPU API
#include "ax_engine_api.h"
#include "ax_sys_api.h"

// AX650 IVPS (Image Video Processing System) API for hardware-accelerated preprocessing
#include "ax_ivps_api.h"
#include "ax_ivps_type.h"

namespace fcs::L2 {

// ============================================================================
// AT Legacy Axera Backend Implementation
// ============================================================================

/// Neural network detection backend using AX650 NPU
/// Implements RAII for AX engine resources
/// Construction IS initialization — no separate init() needed.
class AtLegacyAxeraBackend : public AtLegacyDetectorBackendBase<AtLegacyAxeraBackend> {
public:
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, std::string>;
    using Config          = AtLegacyAxeraConfig;

    /// Factory: construct a fully-initialized backend (load model onto NPU).
    /// Construction IS initialization — no separate init() needed.
    [[nodiscard]] static std::expected<AtLegacyAxeraBackend, std::string>
        create(Config config) noexcept;

    ~AtLegacyAxeraBackend();

    // Move-only (AX engine resources are non-copyable)
    AtLegacyAxeraBackend(AtLegacyAxeraBackend&&) noexcept;
    AtLegacyAxeraBackend& operator=(AtLegacyAxeraBackend&&) noexcept;
    AtLegacyAxeraBackend(const AtLegacyAxeraBackend&)            = delete;
    AtLegacyAxeraBackend& operator=(const AtLegacyAxeraBackend&) = delete;

    /// Detect armors in image (synchronous)
    [[nodiscard]] DetectionResult detect_impl(const cv::Mat& image) noexcept;

    void set_detect_color(ArmorColor color) noexcept { detect_color_ = color; }
    [[nodiscard]] ArmorColor detect_color() const noexcept { return detect_color_; }

    /// Get current config
    [[nodiscard]] const Config& get_config() const noexcept { return config_; }

private:
    /// Private constructor — use create() factory
    explicit AtLegacyAxeraBackend(Config config);

    /// Direct-resize preprocessing context.
    struct PreprocContext {
        float scale_x      = 1.0f;
        float scale_y      = 1.0f;
        int orig_w         = 0;
        int orig_h         = 0;
        bool is_four_three = false;
    };

    /// Build direct-resize transform for current frame.
    [[nodiscard]] PreprocContext preprocess(const cv::Mat& image) const;

    /// Preprocess image into NPU buffer (resize + BGR2RGB + cache flush).
    void preprocess_image(const cv::Mat& image);

    /// Run pure NPU inference (assumes input buffer is ready)
    [[nodiscard]] bool infer();

    /// Cached anchor grid used to decode the predecode-concat output on CPU.
    struct DecodeGrid {
        std::vector<float> grid_x;
        std::vector<float> grid_y;
        std::vector<float> strides;
    };

    struct DecodedCandidate {
        std::array<cv::Point2f, 4> corners{};
        ArmorName name   = ArmorName::Invalid;
        ArmorColor color = ArmorColor::Neutral;
        float confidence = 0.0f;
        int pair_id      = -1;
    };

    struct RankedAnchor {
        float best_obj_logit = -std::numeric_limits<float>::infinity();
        int pair_id          = -1;
        int anchor_idx       = -1;
    };

    static constexpr int TILE_GRID_SIZE = 2;

    /// Build decode grid for anchor-based decoding (called once at init).
    [[nodiscard]] DecodeGrid build_decode_grid() const;

    /// Cached decode grid (built once at initialization).
    DecodeGrid cached_grid_;

    // Reusable postprocess/NMS buffers (mutable: internal detail, logical const-ness unaffected)
    mutable std::vector<RankedAnchor> postproc_ranked_buf_;
    mutable std::vector<DecodedCandidate> postproc_candidates_buf_;
    mutable std::vector<uint8_t> nms_removed_buf_;
    mutable std::array<std::vector<int>, TILE_GRID_SIZE * TILE_GRID_SIZE> nms_tile_bins_buf_;

    /// Decode and postprocess single-output concat-predecode tensor.
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        postprocess_concat_predecode(const PreprocContext& ctx) const;

    /// Tile-based NMS on pts corner points.
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        nms(std::vector<DecodedCandidate>& detections, int orig_w, int orig_h) const;

private:
    Config config_;
    ArmorColor detect_color_          = ArmorColor::Blue;
    bool warned_non_four_three_input_ = false;

    // AX650 NPU resources
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_T io_data_{};
    std::vector<char> model_buffer_;

    // IVPS (Image Video Processing System) resources for hardware-accelerated preprocessing
    bool ivps_initialized_       = false;
    AX_U64 ivps_src_phy_addr_    = 0;            // Physical address for source buffer
    AX_VOID* ivps_src_vir_addr_  = nullptr;      // Virtual address for source buffer
    AX_U32 ivps_src_buffer_size_ = 0;            // Source buffer size (bytes)
    AX_U32 ivps_src_stride_      = 0;            // Source stride (aligned)
    int ivps_max_src_width_      = 0;
    int ivps_max_src_height_     = 0;
    AX_IVPS_ASPECT_RATIO_T ivps_aspect_ratio_{}; // Cached aspect ratio config

    // IVPS helper methods
    [[nodiscard]] std::expected<void, std::string> init_ivps();
    void deinit_ivps();
    [[nodiscard]] std::expected<void, std::string>
        allocate_ivps_buffers(int max_src_width, int max_src_height);
    void free_ivps_buffers();
    void preprocess_image_ivps(const cv::Mat& image);
};

} // namespace fcs::L2
