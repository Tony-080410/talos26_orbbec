#pragma once

#include "at_legacy_config_holder.hpp"
#include "at_legacy_detector_interface.hpp"
#include "at_legacy_traditional_classifier.hpp"
#include "at_legacy_traditional_types.hpp"

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace fcs::L2 {

struct AtLegacyTraditionalConfig;

class AtLegacyTraditionalBackend : public IDetectorBackend {
public:
    using DetectionResult = IDetectorBackend::DetectionResult;
    using ConfigPtr       = std::shared_ptr<const AtLegacyTraditionalConfig>;

    explicit AtLegacyTraditionalBackend(ConfigPtr config);

    [[nodiscard]] DetectionResult detect(const cv::Mat& image) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "traditional"; }

    void set_detect_color(ArmorColor color) noexcept override { detect_color_ = color; }
    [[nodiscard]] ArmorColor detect_color() const noexcept override { return detect_color_; }

private:
    ConfigHolder<AtLegacyTraditionalConfig> config_holder_;
    std::unique_ptr<AtLegacyTraditionalClassifier> classifier_;
    ArmorColor detect_color_ = ArmorColor::Blue;

    [[nodiscard]] cv::Mat
        preprocess_image(const cv::Mat& bgr_img, const AtLegacyTraditionalConfig& cfg) const;
    [[nodiscard]] std::vector<AtLegacyLight> find_lights(
        const cv::Mat& bgr_img, const cv::Mat& binary_img,
        const AtLegacyTraditionalConfig& cfg) const;
    [[nodiscard]] bool
        is_light(const AtLegacyLight& light, const AtLegacyTraditionalConfig& cfg) const noexcept;
    [[nodiscard]] std::vector<AtLegacyArmorDetection> match_lights(
        std::vector<AtLegacyLight>& lights, const AtLegacyTraditionalConfig& cfg,
        cv::Mat gray_img) const;
    [[nodiscard]] static bool
        contains_light(size_t i, size_t j, const std::vector<AtLegacyLight>& lights);
    [[nodiscard]] ArmorType is_armor(
        const AtLegacyLight& light_1, const AtLegacyLight& light_2,
        const AtLegacyTraditionalConfig& cfg) const noexcept;
    [[nodiscard]] cv::Mat extract_number(
        const cv::Mat& src, const std::array<cv::Point2f, 4>& lights_vertices,
        ArmorType armor_type) const noexcept;
};

} // namespace fcs::L2
