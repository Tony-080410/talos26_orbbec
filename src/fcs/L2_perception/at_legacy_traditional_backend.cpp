#include "fcs/L2_perception/at_legacy_traditional_backend.hpp"

#include "fcs/L2_perception/at_legacy_config.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <spdlog/spdlog.h>

namespace fcs {

// ============================================================================
// Constructor
// ============================================================================

AtLegacyTraditionalBackend::AtLegacyTraditionalBackend(ConfigPtr config)
    : config_(std::move(config)) {
    // Initialize classifier if enabled in config
    if (config_ && !config_->classifier_model_path.empty()) {
        classifier_ = std::make_unique<AtLegacyTraditionalClassifier>(
            config_->classifier_model_path, config_->classifier_use_softmax);
        // Initialize classifier - log error if initialization fails
        if (const auto init_result = classifier_->init(); !init_result) {
            // Log error but continue - detection will work without classification
            spdlog::critical(
                "Failed to initialize ONNX classifier: {}", config_->classifier_model_path);
            std::abort();
        }
    } else {
        spdlog::critical("Classifier is enabled but model path is empty or config is null.");
        std::abort();
    }

    // Always create corner corrector for optional sub-pixel accuracy improvement
    corner_corrector_ = std::make_unique<AtLegacyLightCornerCorrector>();
}

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

    // Save gray image for corner correction (precompute for efficiency)
    cv::Mat gray_img;
    cv::cvtColor(input, gray_img, cv::COLOR_BGR2GRAY);

    // 3. Match lights to armors
    auto detections = match_lights(lights, *cfg, gray_img);

    // 5. Classify armors if classifier is available
    if (classifier_ && classifier_->is_initialized()) {
        for (size_t d = 0; d < detections.size(); ++d) {
            auto& detection = detections[d];

            // Extract number image: perspective transform + OTSU binarization
            cv::Mat number_img = extract_number(input, detection.corners, detection.type);

            if (!number_img.empty()) {
                float confidence = 0.0f;
                auto result      = classifier_->classify(number_img, confidence);

                if (result) {
                    detection.name       = *result;
                    detection.confidence = confidence;
                } else {
                    // Classification failed, mark as invalid
                    detection.name       = ArmorName::Invalid;
                    detection.confidence = 0.0f;
                }
            }
        }

        // 5. Filter out invalid detections (matching ROS2 logic)
        detections.erase(
            std::remove_if(
                detections.begin(), detections.end(),
                [cfg](const AtLegacyArmorDetection& detection) {
                    // Filter by confidence threshold
                    if (detection.confidence < cfg->classifier_confidence_threshold) {
                        return true;
                    }

                    // Filter by name validity
                    if (detection.name == ArmorName::Invalid) {
                        return true;
                    }

                    // Filter by type-class matching (when type filtering enabled)
                    if (cfg->classifier_enable_type_filtering) {
                        if (detection.type == ArmorType::Large) {
                            // Large armor cannot be: outpost, 2, sentry
                            if (detection.name == ArmorName::Outpost
                                || detection.name == ArmorName::Two
                                || detection.name == ArmorName::Sentry) {
                                return true;
                            }
                        } else if (detection.type == ArmorType::Small) {
                            // Small armor cannot be: 1, base
                            if (detection.name == ArmorName::One
                                || detection.name == ArmorName::Base
                                || detection.name == ArmorName::BaseLarge) {
                                return true;
                            }
                        }
                    }

                    return false; // Keep this detection
                }),
            detections.end());
    }

    return detections;
}

// ============================================================================
// Preprocessing
// ============================================================================

cv::Mat AtLegacyTraditionalBackend::preprocess_image(
    const cv::Mat& bgr_img, const AtLegacyTraditionalConfig& cfg) const {
    if (!cfg.advanced_binary) {
        cv::Mat gray;
        cv::cvtColor(bgr_img, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, gray, cfg.binary_threshold, 255, cv::THRESH_BINARY);
        return gray;
    }
    std::vector<cv::Mat> ch;
    cv::split(bgr_img, ch);

    cv::Mat max01, min01, maxc, minc, chroma;
    cv::max(ch[0], ch[1], max01);
    cv::max(max01, ch[2], maxc);

    cv::min(ch[0], ch[1], min01);
    cv::min(min01, ch[2], minc);

    cv::subtract(maxc, minc, chroma);

    cv::Mat mask;
    cv::threshold(chroma, mask, cfg.binary_threshold, 255, cv::THRESH_BINARY);

    return mask;
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
            // Only set color if difference exceeds threshold (matching armor_detector_ov)
            if (avg_diff > cfg.light_color_diff_thresh) {
                light.color = sum_r > sum_b ? ArmorColor::Red : ArmorColor::Blue;
            }
            // If threshold not met, light.color remains Neutral (WHITE)
            lights.emplace_back(light);
        }
    }

    // Sort by x coordinate
    std::ranges::sort(
        lights.begin(), lights.end(),
        [](const AtLegacyLight& l1, const AtLegacyLight& l2) { return l1.center.x < l2.center.x; });

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
    std::vector<AtLegacyLight>& lights, const AtLegacyTraditionalConfig& cfg,
    const cv::Mat gray_img) const {
    std::vector<AtLegacyArmorDetection> detections;

    // Target color based on detect_color_
    const ArmorColor target_color = detect_color_;

    // Loop all pairs of lights
    for (auto it1 = lights.begin(); it1 != lights.end(); ++it1) {
        // Filter by color (allow Neutral/WHITE to pass through, matching armor_detector_ov)
        if (it1->color != target_color && it1->color != ArmorColor::Neutral)
            continue;

        const double max_iter_width = it1->length * cfg.armor_max_large_center_distance;

        for (auto it2 = it1 + 1; it2 != lights.end(); ++it2) {
            // Filter by color (allow Neutral/WHITE to pass through)
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
            if (type == ArmorType::Invalid) {
                continue;
            }

            // NOTE: armor_detector_ov does NOT use corner_corrector
            // corner_corrector_ is available but disabled to match behavior

            // Create detection from lights
            auto detection = make_detection_from_lights(*it1, *it2);
            detection.type = type;
            detections.emplace_back(detection);
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
    const auto center_distance_ok = (cfg.armor_min_small_center_distance <= center_distance
                                     && cfg.armor_max_small_center_distance > center_distance)
                                 || (cfg.armor_min_large_center_distance <= center_distance
                                     && cfg.armor_max_large_center_distance > center_distance);
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

    return center_distance > cfg.armor_min_large_center_distance ? ArmorType::Large
                                                                 : ArmorType::Small;
}

// ============================================================================
// Number Extraction (Perspective Transform + OTSU Binarization)
// ============================================================================

cv::Mat AtLegacyTraditionalBackend::extract_number(
    const cv::Mat& src, const std::array<cv::Point2f, 4>& lights_vertices,
    ArmorType armor_type) const noexcept {
    // Light length in image
    static const int light_length = 12;
    // Image size after warp
    static const int warp_height       = 28;
    static const int small_armor_width = 32;
    static const int large_armor_width = 54;
    // Number ROI size
    static const cv::Size roi_size(20, 28);
    // Classifier input size (matching armor_detector_ov)
    static const cv::Size input_size(28, 28);

    const int top_light_y    = (warp_height - light_length) / 2 - 1;
    const int bottom_light_y = top_light_y + light_length;
    const int warp_width = armor_type == ArmorType::Small ? small_armor_width : large_armor_width;
    cv::Point2f target_vertices[4] = {
        cv::Point(0, bottom_light_y),
        cv::Point(0, top_light_y),
        cv::Point(warp_width - 1, top_light_y),
        cv::Point(warp_width - 1, bottom_light_y),
    };
    cv::Mat number_image;
    cv::Point2f lights_vertices2[4] = {
        lights_vertices[3], lights_vertices[0], lights_vertices[1], lights_vertices[2]};

    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices2, target_vertices);
    cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

    // Get ROI
    number_image =
        number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

    // Convert to grayscale and binarize (matching armor_detector_ov)
    cv::cvtColor(number_image, number_image, cv::COLOR_BGR2GRAY);
    cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Output is already 20x28, no resize needed (matching atvision_ws)
    return number_image;
}

} // namespace fcs
