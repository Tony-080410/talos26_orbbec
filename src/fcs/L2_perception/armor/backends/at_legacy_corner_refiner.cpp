#include "at_legacy_corner_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// LightCornerCorrector Implementation
// ============================================================================

void LightCornerCorrector::correctCorners(AtLegacyArmorDetection& armor, const cv::Mat& gray_img) {
    // Reconstruct lights from corners (since we only have detection, not original lights)
    // Left light: TL → BL (corners[0] → corners[3])
    // Right light: TR → BR (corners[1] → corners[2])
    AtLegacyLight left_light(armor.corners[0], armor.corners[3]);
    AtLegacyLight right_light(armor.corners[1], armor.corners[2]);

    // Refine left light
    if (left_light.width > PASS_OPTIMIZE_WIDTH) {
        SymmetryAxis left_axis = findSymmetryAxis(gray_img, left_light);

        // Update light fields (for potential future use)
        left_light.center = left_axis.centroid;
        left_light.axis   = left_axis.direction;

        // Find and update corners
        if (cv::Point2f t = findCorner(gray_img, left_light, left_axis, CornerOrder::Top);
            t.x > 0.0f) {
            armor.corners[0] = t; // TL
        }
        if (cv::Point2f b = findCorner(gray_img, left_light, left_axis, CornerOrder::Bottom);
            b.x > 0.0f) {
            armor.corners[3] = b; // BL
        }
    }

    // Refine right light
    if (right_light.width > PASS_OPTIMIZE_WIDTH) {
        SymmetryAxis right_axis = findSymmetryAxis(gray_img, right_light);

        // Update light fields
        right_light.center = right_axis.centroid;
        right_light.axis   = right_axis.direction;

        // Find and update corners
        if (cv::Point2f t = findCorner(gray_img, right_light, right_axis, CornerOrder::Top);
            t.x > 0.0f) {
            armor.corners[1] = t; // TR
        }
        if (cv::Point2f b = findCorner(gray_img, right_light, right_axis, CornerOrder::Bottom);
            b.x > 0.0f) {
            armor.corners[2] = b; // BR
        }
    }
}

SymmetryAxis
    LightCornerCorrector::findSymmetryAxis(const cv::Mat& gray_img, const AtLegacyLight& light) {
    // Step 1: Scale the bounding box (expand by 7%)
    cv::Rect light_box = cv::Rect(
        light.center.x - light.width / 2, light.center.y - light.length / 2, light.width,
        light.length);

    light_box.x -= static_cast<int>(light_box.width * ROI_SCALE);
    light_box.y -= static_cast<int>(light_box.height * ROI_SCALE);
    light_box.width += static_cast<int>(light_box.width * ROI_SCALE * 2);
    light_box.height += static_cast<int>(light_box.height * ROI_SCALE * 2);

    // Step 2: Check boundary (clamp to image bounds)
    light_box.x      = std::max(light_box.x, 0);
    light_box.y      = std::max(light_box.y, 0);
    light_box.width  = std::min(light_box.width, gray_img.cols - light_box.x);
    light_box.height = std::min(light_box.height, gray_img.rows - light_box.y);

    // Step 3: Get normalized light image
    cv::Mat roi    = gray_img(light_box);
    float mean_val = cv::mean(roi)[0];

    // Convert to float and normalize
    cv::Mat roi_float;
    roi.convertTo(roi_float, CV_32F);
    cv::normalize(roi_float, roi_float, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

    // Step 4: Calculate centroid using image moments
    cv::Moments moments = cv::moments(roi_float, false);

    // Check for zero-area region (empty ROI)
    if (moments.m00 < 1e-6) {
        SPDLOG_WARN("Zero-area ROI in corner refinement");
        return SymmetryAxis{
            .centroid = light.center, .direction = cv::Point2f(0, -1), .mean_val = mean_val};
    }

    cv::Point2f centroid = cv::Point2f(
        static_cast<float>(moments.m10 / moments.m00),
        static_cast<float>(moments.m01 / moments.m00));
    centroid += cv::Point2f(static_cast<float>(light_box.x), static_cast<float>(light_box.y));

    // Step 5: Initialize point cloud for PCA
    std::vector<cv::Point2f> points;
    const size_t estimated_points =
        static_cast<size_t>(roi_float.rows) * static_cast<size_t>(roi_float.cols);
    points.reserve(estimated_points); // Pre-allocate

    const int rows = roi_float.rows;
    const int cols = roi_float.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Add multiple copies based on brightness (weighted sampling)
            int n_points = static_cast<int>(std::round(roi_float.at<float>(i, j)));
            for (int k = 0; k < n_points; k++) {
                points.emplace_back(static_cast<float>(j), static_cast<float>(i));
            }
        }
    }

    // Step 6: PCA (Principal Component Analysis)
    cv::Mat points_mat = cv::Mat(points).reshape(1);
    cv::PCA pca(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Get symmetry axis (first principal component)
    cv::Point2f axis =
        cv::Point2f(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));

    // Normalize
    float axis_norm = cv::norm(axis);
    if (axis_norm > 1e-6f) {
        axis = axis / axis_norm;
    }

    // Force axis.y < 0 (pointing upward in image coordinates)
    if (axis.y > 0) {
        axis = -axis;
    }

    return SymmetryAxis{.centroid = centroid, .direction = axis, .mean_val = mean_val};
}

cv::Point2f LightCornerCorrector::findCorner(
    const cv::Mat& gray_img, const AtLegacyLight& light, const SymmetryAxis& axis,
    CornerOrder order) {

    // Helper: check if point is inside image (compile-time inlined)
    auto inImage = [&gray_img](const cv::Point2f& pt) -> bool {
        return pt.x >= 0.0f && pt.x < static_cast<float>(gray_img.cols) && pt.y >= 0.0f
            && pt.y < static_cast<float>(gray_img.rows);
    };

    // Helper: check if integer coordinates are valid for pixel access
    auto validPixelCoords = [&gray_img](int x, int y) -> bool {
        return x >= 0 && x < gray_img.cols && y >= 0 && y < gray_img.rows;
    };

    // Helper: calculate Euclidean distance squared (avoid sqrt for performance)
    auto distanceSq = [](float x0, float y0, float x1, float y1) -> float {
        const float dx = x1 - x0;
        const float dy = y1 - y0;
        return dx * dx + dy * dy;
    };

    // Direction: top (oper=1) or bottom (oper=-1) [compile-time]
    const int oper = (order == CornerOrder::Top) ? 1 : -1;
    float L        = static_cast<float>(light.length);
    float dx       = axis.direction.x * static_cast<float>(oper);
    float dy       = axis.direction.y * static_cast<float>(oper);

    std::vector<cv::Point2f> candidates;

    // Multiple rays: average to reduce noise
    const int n      = std::max(1, static_cast<int>(light.width) - 2);
    const int half_n = n / 2;

    for (int i = -half_n; i <= half_n; i++) {
        // Starting point: 40% along the axis + perpendicular offset
        float x0 = axis.centroid.x + L * SEARCH_START_RATIO * dx + static_cast<float>(i);
        float y0 = axis.centroid.y + L * SEARCH_START_RATIO * dy;

        cv::Point2f prev          = cv::Point2f(x0, y0);
        cv::Point2f corner        = cv::Point2f(x0, y0);
        float max_brightness_diff = 0.0f;
        bool has_corner           = true;

        // Search along the ray for maximum brightness gradient
        float search_dist   = L * (SEARCH_END_RATIO - SEARCH_START_RATIO);
        float dist_traveled = 0.0f;

        for (float x = x0 + dx, y = y0 + dy; dist_traveled < search_dist; x += dx, y += dy) {
            cv::Point2f cur = cv::Point2f(x, y);

            if (!inImage(cur)) {
                break;
            }

            // Check bounds for pixel access
            const int prev_x = static_cast<int>(std::round(prev.x));
            const int prev_y = static_cast<int>(std::round(prev.y));
            const int cur_x  = static_cast<int>(std::round(cur.x));
            const int cur_y  = static_cast<int>(std::round(cur.y));

            if (!validPixelCoords(prev_x, prev_y) || !validPixelCoords(cur_x, cur_y)) {
                break;
            }

            float brightness_diff = static_cast<float>(gray_img.at<uint8_t>(prev_y, prev_x))
                                  - static_cast<float>(gray_img.at<uint8_t>(cur_y, cur_x));

            // Update corner if: 1) brightness drops, 2) prev is brighter than background
            if (brightness_diff > max_brightness_diff
                && static_cast<float>(gray_img.at<uint8_t>(prev_y, prev_x)) > axis.mean_val) {
                max_brightness_diff = brightness_diff;
                corner              = prev;
                has_corner          = true;
            }

            prev          = cur;
            dist_traveled = std::sqrt(distanceSq(x0, y0, x, y));
        }

        if (has_corner) {
            candidates.emplace_back(corner);
        }
    }

    // Average all candidate points
    if (!candidates.empty()) {
        cv::Point2f result(0.0f, 0.0f);
        for (const auto& c : candidates) {
            result += c;
        }
        result = result / static_cast<float>(candidates.size());
        return result;
    }

    // No corner found
    return cv::Point2f{-1.0f, -1.0f};
}

} // namespace fcs::L2
