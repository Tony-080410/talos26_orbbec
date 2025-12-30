#include "fcs/L2_perception/at_legacy_traditional_backend.hpp"

#include "fcs/L2_perception/at_legacy_config.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <opencv2/imgproc.hpp>

namespace fcs {

// ============================================================================
// Constructor
// ============================================================================

AtLegacyTraditionalBackend::AtLegacyTraditionalBackend(ConfigPtr config)
    : config_(std::move(config)) {}

// ============================================================================
// Detection Entry Point
// ============================================================================

AtLegacyTraditionalBackend::DetectionResult
    AtLegacyTraditionalBackend::detect_impl(const cv::Mat& input) {
    if (input.empty()) {
        return std::unexpected(DetectorError::InvalidImage);
    }

    auto cfg = get_config();
    if (!cfg) {
        return std::unexpected(DetectorError::NotInitialized);
    }

    // 1. Preprocess the image
    cv::Mat binary = preprocess_image(input, *cfg);

    // 2. Find lights
    auto lights = find_lights(input, binary, *cfg);

    // 3. Match lights to armors
    auto detections = match_lights(lights, *cfg);

    return detections;
}

// ============================================================================
// Preprocessing
// ============================================================================

cv::Mat AtLegacyTraditionalBackend::preprocess_image(
    const cv::Mat& rgb_img, const AtLegacyTraditionalConfig& cfg) const {
    cv::Mat gray;
    cv::cvtColor(rgb_img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat binary;
    cv::threshold(gray, binary, cfg.binary_threshold, 255, cv::THRESH_BINARY);

    return binary;
}

// ============================================================================
// Light Detection
// ============================================================================

std::vector<AtLegacyLight> AtLegacyTraditionalBackend::find_lights(
    const cv::Mat& bgr_img, const cv::Mat& binary_img, const AtLegacyTraditionalConfig& cfg) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<AtLegacyLight> lights;
    lights.reserve(contours.size());

    for (const auto& contour : contours) {
        if (contour.size() < 6)
            continue;

        AtLegacyLight light(contour);

        if (is_light(light, cfg)) {
            // Determine color from contour pixels
            int sum_r = 0, sum_b = 0;
            for (const auto& point : contour) {
                const auto& pixel = bgr_img.at<cv::Vec3b>(point.y, point.x);
                sum_r += pixel[2]; // R channel (BGR order)
                sum_b += pixel[0]; // B channel
            }

            const int avg_diff = std::abs(sum_r - sum_b) / static_cast<int>(contour.size());
            if (avg_diff > cfg.light_color_diff_thresh) {
                light.color = (sum_r > sum_b) ? ArmorColor::Red : ArmorColor::Blue;
            } else {
                continue;
            }

            lights.emplace_back(light);
        }
    }

    // Sort by x coordinate
    std::sort(lights.begin(), lights.end(), [](const AtLegacyLight& l1, const AtLegacyLight& l2) {
        return l1.center.x < l2.center.x;
    });

    return lights;
}

bool AtLegacyTraditionalBackend::is_light(
    const AtLegacyLight& light, const AtLegacyTraditionalConfig& cfg) const noexcept {
    // Check width/height ratio
    const float ratio   = static_cast<float>(light.width) / static_cast<float>(light.length);
    const bool ratio_ok = (cfg.light_min_ratio < ratio) && (ratio < cfg.light_max_ratio);

    // Check tilt angle
    const bool angle_ok = light.tilt_angle < cfg.light_max_angle;

    return ratio_ok && angle_ok;
}

// ============================================================================
// Armor Matching
// ============================================================================

std::vector<AtLegacyArmorDetection> AtLegacyTraditionalBackend::match_lights(
    const std::vector<AtLegacyLight>& lights, const AtLegacyTraditionalConfig& cfg) const {
    std::vector<AtLegacyArmorDetection> detections;

    // Target color based on detect_color_
    const ArmorColor target_color = detect_color_;

    // Loop all pairs of lights
    for (auto it1 = lights.begin(); it1 != lights.end(); ++it1) {
        // Filter by color
        if (it1->color != target_color && it1->color != ArmorColor::Neutral)
            continue;

        const double max_iter_width = it1->length * cfg.armor_max_large_center_distance;

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
            const auto type = is_armor(*it1, *it2, cfg);
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

bool AtLegacyTraditionalBackend::contains_light(
    size_t i, size_t j, const std::vector<AtLegacyLight>& lights) {
    const auto& light_1 = lights[i];
    const auto& light_2 = lights[j];

    // Bounding rect of the two lights
    std::vector<cv::Point2f> points = {light_1.top, light_1.bottom, light_2.top, light_2.bottom};
    auto bounding_rect              = cv::boundingRect(points);

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

ArmorType AtLegacyTraditionalBackend::is_armor(
    const AtLegacyLight& light_1, const AtLegacyLight& light_2,
    const AtLegacyTraditionalConfig& cfg) const noexcept {
    // Check light length ratio
    const float light_length_ratio = (light_1.length < light_2.length)
                                       ? static_cast<float>(light_1.length / light_2.length)
                                       : static_cast<float>(light_2.length / light_1.length);
    if (light_length_ratio <= cfg.armor_min_light_ratio) {
        return ArmorType::Invalid;
    }

    // Check center distance (normalized by light length)
    const float avg_light_length = static_cast<float>((light_1.length + light_2.length) / 2.0);
    const float center_distance =
        static_cast<float>(cv::norm(light_1.center - light_2.center) / avg_light_length);
    const auto center_distance_ok = cfg.armor_min_small_center_distance <= center_distance;
    if (!center_distance_ok) {
        return ArmorType::Invalid;
    }

    // Check horizontal angle
    const cv::Point2f diff = light_1.center - light_2.center;
    const float angle =
        static_cast<float>(std::abs(std::atan(diff.y / diff.x)) / std::numbers::pi * 180.0);
    if (angle >= cfg.armor_max_angle) {
        return ArmorType::Invalid;
    }

    return center_distance > cfg.armor_max_large_center_distance ? ArmorType::Large
                                                                 : ArmorType::Small;
}

} // namespace fcs
