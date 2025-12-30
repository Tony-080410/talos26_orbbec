#pragma once

#include "at_legacy_detector_backend.hpp"
#include "at_legacy_traditional_types.hpp"

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace fcs {

// Forward declare config (defined in at_legacy_config.hpp)
struct AtLegacyTraditionalConfig;

// ============================================================================
// AT Legacy Traditional Backend Implementation
// ============================================================================

class AtLegacyTraditionalBackend : public AtLegacyDetectorBackendBase<AtLegacyTraditionalBackend> {
public:
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;
    using ConfigPtr       = std::shared_ptr<const AtLegacyTraditionalConfig>;

    explicit AtLegacyTraditionalBackend(ConfigPtr config);

    /// Detect armors in image
    [[nodiscard]] DetectionResult detect_impl(const cv::Mat& input);

    [[nodiscard]] bool is_initialized_impl() const noexcept { return config_ != nullptr; }
    [[nodiscard]] std::string_view name_impl() const noexcept { return "traditional"; }

    void set_detect_color(ArmorColor color) noexcept { detect_color_ = color; }
    [[nodiscard]] ArmorColor detect_color() const noexcept { return detect_color_; }

    /// Hot-reload config (thread-safe)
    void update_config(ConfigPtr new_config) noexcept {
        std::atomic_store(&config_, std::move(new_config));
    }

    /// Get current config (thread-safe)
    [[nodiscard]] ConfigPtr get_config() const noexcept { return std::atomic_load(&config_); }

private:
    ConfigPtr config_;
    ArmorColor detect_color_ = ArmorColor::Neutral;

    /// Preprocess image to binary
    [[nodiscard]] cv::Mat
        preprocess_image(const cv::Mat& rgb_img, const AtLegacyTraditionalConfig& cfg) const;

    /// Find light bars in image
    [[nodiscard]] std::vector<AtLegacyLight> find_lights(
        const cv::Mat& bgr_img, const cv::Mat& binary_img,
        const AtLegacyTraditionalConfig& cfg) const;

    /// Check if contour is a valid light bar
    [[nodiscard]] bool
        is_light(const AtLegacyLight& light, const AtLegacyTraditionalConfig& cfg) const noexcept;

    /// Match light pairs to form armors
    [[nodiscard]] std::vector<AtLegacyArmorDetection> match_lights(
        const std::vector<AtLegacyLight>& lights, const AtLegacyTraditionalConfig& cfg) const;

    /// Check if there's a light between two lights
    [[nodiscard]] static bool
        contains_light(size_t i, size_t j, const std::vector<AtLegacyLight>& lights);

    /// Check if two lights form a valid armor
    [[nodiscard]] ArmorType is_armor(
        const AtLegacyLight& light_1, const AtLegacyLight& light_2,
        const AtLegacyTraditionalConfig& cfg) const noexcept;
};

} // namespace fcs
