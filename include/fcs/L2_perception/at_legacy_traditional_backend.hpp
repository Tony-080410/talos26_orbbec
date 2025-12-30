#pragma once

#include "at_legacy_detector_backend.hpp"
#include "at_legacy_traditional_types.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace fcs {

// ============================================================================
// AT Legacy Traditional Backend Implementation
// ============================================================================

class AtLegacyTraditionalBackend : public AtLegacyDetectorBackendBase<AtLegacyTraditionalBackend> {
public:
    using DetectionResult = std::expected<std::vector<AtLegacyArmorDetection>, DetectorError>;

    /// Light detection parameters
    struct LightParams {
        double min_ratio      = 0.1;  // width/height min
        double max_ratio      = 0.55; // width/height max
        double max_angle      = 45.0; // max vertical tilt angle (degrees)
        int color_diff_thresh = 50;   // R-B color difference threshold
    };

    /// Armor matching parameters
    struct ArmorParams {
        double min_light_ratio           = 0.7;  // min ratio of light lengths
        double min_small_center_distance = 0.8;  // normalized by light length
        double max_small_center_distance = 3.2;
        double min_large_center_distance = 3.2;
        double max_large_center_distance = 5.5;
        double max_angle                 = 35.0; // max horizontal angle (degrees)
    };

    explicit AtLegacyTraditionalBackend(int binary_threshold = 160)
        : binary_threshold_(binary_threshold)
        , light_params_()
        , armor_params_() {}

    AtLegacyTraditionalBackend(
        int binary_threshold, LightParams light_params, ArmorParams armor_params)
        : binary_threshold_(binary_threshold)
        , light_params_(std::move(light_params))
        , armor_params_(std::move(armor_params)) {}

    /// Detect armors in image
    [[nodiscard]] DetectionResult detect_impl(const cv::Mat& input) {
        if (input.empty()) {
            return std::unexpected(DetectorError::InvalidImage);
        }

        // 1. Preprocess the image
        cv::Mat binary = preprocess_image(input);

        // 2. Find lights
        auto lights = find_lights(input, binary);

        // 3. Match lights to armors
        auto detections = match_lights(lights);

        return detections;
    }

    [[nodiscard]] bool is_initialized_impl() const noexcept { return true; }
    [[nodiscard]] std::string_view name_impl() const noexcept { return "traditional"; }

    void set_detect_color(ArmorColor color) noexcept { detect_color_ = color; }
    [[nodiscard]] ArmorColor detect_color() const noexcept { return detect_color_; }

    // Parameters (public for configuration)
    int binary_threshold_;
    LightParams light_params_;
    ArmorParams armor_params_;

private:
    ArmorColor detect_color_ = ArmorColor::Neutral; // 0=BLUE, 1=RED

    /// Preprocess image to binary
    [[nodiscard]] cv::Mat preprocess_image(const cv::Mat& rgb_img) const {
        cv::Mat gray;
        cv::cvtColor(rgb_img, gray, cv::COLOR_BGR2GRAY);

        cv::Mat binary;
        cv::threshold(gray, binary, binary_threshold_, 255, cv::THRESH_BINARY);

        return binary;
    }

    /// Find light bars in image
    [[nodiscard]] std::vector<AtLegacyLight>
        find_lights(const cv::Mat& bgr_img, const cv::Mat& binary_img) const {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        std::vector<AtLegacyLight> lights;
        lights.reserve(contours.size());

        for (const auto& contour : contours) {
            if (contour.size() < 6)
                continue;

            AtLegacyLight light(contour);

            if (is_light(light)) {
                // Determine color from contour pixels
                int sum_r = 0, sum_b = 0;
                for (const auto& point : contour) {
                    const auto& pixel = bgr_img.at<cv::Vec3b>(point.y, point.x);
                    sum_r += pixel[2]; // R channel (BGR order)
                    sum_b += pixel[0]; // B channel
                }

                const int avg_diff = std::abs(sum_r - sum_b) / static_cast<int>(contour.size());
                if (avg_diff > light_params_.color_diff_thresh) {
                    light.color = (sum_r > sum_b) ? ArmorColor::Red : ArmorColor::Blue;
                } else {
                    continue;
                }

                lights.emplace_back(light);
            }
        }

        // Sort by x coordinate
        std::sort(
            lights.begin(), lights.end(), [](const AtLegacyLight& l1, const AtLegacyLight& l2) {
                return l1.center.x < l2.center.x;
            });

        return lights;
    }

    /// Check if contour is a valid light bar
    [[nodiscard]] bool is_light(const AtLegacyLight& light) const noexcept {
        // Check width/height ratio
        const float ratio = static_cast<float>(light.width) / static_cast<float>(light.length);
        const bool ratio_ok =
            (light_params_.min_ratio < ratio) && (ratio < light_params_.max_ratio);

        // Check tilt angle
        const bool angle_ok = light.tilt_angle < light_params_.max_angle;

        return ratio_ok && angle_ok;
    }

    /// Match light pairs to form armors
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        match_lights(const std::vector<AtLegacyLight>& lights) const {
        std::vector<AtLegacyArmorDetection> detections;

        // Target color based on detect_color_
        const ArmorColor target_color = detect_color_;

        // Loop all pairs of lights
        for (auto it1 = lights.begin(); it1 != lights.end(); ++it1) {
            // Filter by color
            if (it1->color != target_color && it1->color != ArmorColor::Neutral)
                continue;

            const double max_iter_width = it1->length * armor_params_.max_large_center_distance;

            for (auto it2 = it1 + 1; it2 != lights.end(); ++it2) {
                // Filter by color
                if (it2->color != target_color && it2->color != ArmorColor::Neutral)
                    continue;

                // Check if there's a light between them
                const auto i = static_cast<size_t>(std::distance(lights.begin(), it1));
                const auto j = static_cast<size_t>(std::distance(lights.begin(), it2));
                if (contains_light(i, j, lights))
                    continue;

                // Early termination if too far apart
                if (it2->center.x - it1->center.x > max_iter_width)
                    break;

                // Check if this pair forms a valid armor
                const auto type = is_armor(*it1, *it2);
                if (type == ArmorType::Invalid)
                    continue;

                // Create detection from lights
                auto detection = make_detection_from_lights(*it1, *it2);
                detection.type = type;
                detections.emplace_back(std::move(detection));
            }
        }

        return detections;
    }

    /// Check if there's a light between two lights
    [[nodiscard]] static bool
        contains_light(size_t i, size_t j, const std::vector<AtLegacyLight>& lights) {
        const auto& light_1 = lights[i];
        const auto& light_2 = lights[j];

        // Bounding rect of the two lights
        std::vector<cv::Point2f> points = {
            light_1.top, light_1.bottom, light_2.top, light_2.bottom};
        auto bounding_rect = cv::boundingRect(points);

        const double avg_length = (light_1.length + light_2.length) / 2.0;
        const double avg_width  = (light_1.width + light_2.width) / 2.0;

        // Check lights in between
        for (size_t k = i + 1; k < j; ++k) {
            const auto& test_light = lights[k];

            // Skip if too wide (might be number)
            if (test_light.width > 2 * avg_width)
                continue;

            // Skip if too short (might be noise)
            if (test_light.length < 0.5 * avg_length)
                continue;

            // Check if any key point is inside the bounding rect
            if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom)
                || bounding_rect.contains(test_light.center)) {
                return true;
            }
        }
        return false;
    }

    /// Check if two lights form a valid armor
    [[nodiscard]] ArmorType
        is_armor(const AtLegacyLight& light_1, const AtLegacyLight& light_2) const noexcept {
        // Check light length ratio
        const float light_length_ratio = (light_1.length < light_2.length)
                                           ? static_cast<float>(light_1.length / light_2.length)
                                           : static_cast<float>(light_2.length / light_1.length);
        if (light_length_ratio <= armor_params_.min_light_ratio) {
            return ArmorType::Invalid; // Invalid, using Small as placeholder
        }

        // Check center distance (normalized by light length)
        const float avg_light_length = static_cast<float>((light_1.length + light_2.length) / 2.0);
        const float center_distance =
            static_cast<float>(cv::norm(light_1.center - light_2.center) / avg_light_length);
        const auto center_distance_ok = armor_params_.min_small_center_distance <= center_distance;
        if (!center_distance_ok) {
            return ArmorType::Invalid;
        }

        // Check horizontal angle
        const cv::Point2f diff = light_1.center - light_2.center;
        const float angle =
            static_cast<float>(std::abs(std::atan(diff.y / diff.x)) / std::numbers::pi * 180.0);
        if (angle >= armor_params_.max_angle) {
            return ArmorType::Invalid; // Invalid
        }

        return center_distance > armor_params_.max_large_center_distance ? ArmorType::Large
                                                                         : ArmorType::Small;
    }
};

} // namespace fcs
