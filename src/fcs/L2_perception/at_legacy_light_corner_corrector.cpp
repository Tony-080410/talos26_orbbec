#include "fcs/L2_perception/at_legacy_light_corner_corrector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace fcs::L2 {

// ============================================================================
// Corner Correction
// ============================================================================

void AtLegacyLightCornerCorrector::correct_corners(
    AtLegacyLight& left_light, AtLegacyLight& right_light, const cv::Mat& gray_img) const noexcept {
    // Correct left light if width is sufficient
    if (left_light.width > PASS_OPTIMIZE_WIDTH) {
        SymmetryAxis left_axis = find_symmetry_axis(gray_img, left_light);
        left_light.center      = left_axis.centroid;
        left_light.axis        = left_axis.direction;

        cv::Point2f t = find_corner(gray_img, left_light, left_axis, true);
        cv::Point2f b = find_corner(gray_img, left_light, left_axis, false);
        if (t.x > 0) {
            left_light.top = t;
        }
        if (b.x > 0) {
            left_light.bottom = b;
        }
    }

    // Correct right light if width is sufficient
    if (right_light.width > PASS_OPTIMIZE_WIDTH) {
        SymmetryAxis right_axis = find_symmetry_axis(gray_img, right_light);
        right_light.center      = right_axis.centroid;
        right_light.axis        = right_axis.direction;

        cv::Point2f t = find_corner(gray_img, right_light, right_axis, true);
        cv::Point2f b = find_corner(gray_img, right_light, right_axis, false);
        if (t.x > 0) {
            right_light.top = t;
        }
        if (b.x > 0) {
            right_light.bottom = b;
        }
    }
}

// ============================================================================
// Symmetry Axis Detection (PCA-based)
// ============================================================================

SymmetryAxis AtLegacyLightCornerCorrector::find_symmetry_axis(
    const cv::Mat& gray_img, const AtLegacyLight& light) const noexcept {
    constexpr float MAX_BRIGHTNESS = 25.0f;
    constexpr float SCALE          = 0.07f;

    // Scale the bounding box
    cv::Rect light_box = cv::boundingRect(std::vector<cv::Point2f>{light.top, light.bottom});
    light_box.x -= static_cast<int>(light_box.width * SCALE);
    light_box.y -= static_cast<int>(light_box.height * SCALE);
    light_box.width  = static_cast<int>(light_box.width * (1.0f + SCALE * 2.0f));
    light_box.height = static_cast<int>(light_box.height * (1.0f + SCALE * 2.0f));

    // Boundary check
    light_box.x      = std::max(light_box.x, 0);
    light_box.y      = std::max(light_box.y, 0);
    light_box.width  = std::min(light_box.width, gray_img.cols - light_box.x);
    light_box.height = std::min(light_box.height, gray_img.rows - light_box.y);

    if (light_box.width <= 0 || light_box.height <= 0) {
        // Return default axis if box is invalid
        return SymmetryAxis{.centroid = light.center, .direction = light.axis, .mean_val = 0.0f};
    }

    // Get ROI and normalize
    cv::Mat roi    = gray_img(light_box);
    float mean_val = static_cast<float>(cv::mean(roi)[0]);

    cv::Mat roi_float;
    roi.convertTo(roi_float, CV_32F);
    cv::normalize(roi_float, roi_float, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

    // Calculate centroid
    cv::Moments moments = cv::moments(roi_float, false);
    if (moments.m00 <= 1e-6) {
        return SymmetryAxis{
            .centroid = light.center, .direction = light.axis, .mean_val = mean_val};
    }

    cv::Point2f centroid(
        static_cast<float>(moments.m10 / moments.m00) + light_box.x,
        static_cast<float>(moments.m01 / moments.m00) + light_box.y);

    // Build point cloud from brightness-weighted samples
    std::vector<cv::Point2f> points;
    for (int i = 0; i < roi_float.rows; ++i) {
        for (int j = 0; j < roi_float.cols; ++j) {
            float brightness = roi_float.at<float>(i, j);
            int count        = static_cast<int>(std::round(brightness));
            for (int k = 0; k < count; ++k) {
                points.emplace_back(static_cast<float>(j), static_cast<float>(i));
            }
        }
    }

    if (points.size() < 2) {
        return SymmetryAxis{.centroid = centroid, .direction = light.axis, .mean_val = mean_val};
    }

    // Perform PCA
    cv::Mat points_mat = cv::Mat(points).reshape(1);
    cv::PCA pca(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Get symmetry axis from first principal component
    cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));

    // Normalize axis
    float axis_norm = cv::norm(axis);
    if (axis_norm > 1e-6) {
        axis = axis / axis_norm;
    }

    // Ensure axis points downward (axis.y should be negative)
    if (axis.y > 0) {
        axis = -axis;
    }

    return SymmetryAxis{.centroid = centroid, .direction = axis, .mean_val = mean_val};
}

// ============================================================================
// Corner Finding (Gradient-based search along symmetry axis)
// ============================================================================

cv::Point2f AtLegacyLightCornerCorrector::find_corner(
    const cv::Mat& gray_img, const AtLegacyLight& light, const SymmetryAxis& axis,
    bool is_top) const noexcept {
    constexpr float START = 0.8f / 2.0f;
    constexpr float END   = 1.2f / 2.0f;

    auto in_image = [&gray_img](float x, float y) -> bool {
        int ix = static_cast<int>(x);
        int iy = static_cast<int>(y);
        return ix >= 0 && ix < gray_img.cols && iy >= 0 && iy < gray_img.rows;
    };

    auto distance = [](float x0, float y0, float x1, float y1) -> float {
        return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    };

    int oper = is_top ? 1 : -1;
    float L  = static_cast<float>(light.length);
    float dx = axis.direction.x * oper;
    float dy = axis.direction.y * oper;

    std::vector<cv::Point2f> candidates;

    // Search along axis at multiple offsets
    int n      = static_cast<int>(light.width) - 2;
    int half_n = std::round(n / 2.0);

    for (int i = -half_n; i <= half_n; ++i) {
        float x0 = axis.centroid.x + L * START * dx + i;
        float y0 = axis.centroid.y + L * START * dy;

        if (!in_image(x0, y0)) {
            continue;
        }

        cv::Point2f prev          = cv::Point2f(x0, y0);
        cv::Point2f corner        = cv::Point2f(x0, y0);
        float max_brightness_diff = 0.0f;
        bool has_corner           = false;

        // Search along symmetry axis for brightness gradient maximum
        for (float step = 1.0f; step < L * (END - START); step += 1.0f) {
            float x = x0 + dx * step;
            float y = y0 + dy * step;

            if (!in_image(x, y)) {
                break;
            }

            cv::Point2f cur = cv::Point2f(x, y);

            int prev_val = gray_img.at<uchar>(static_cast<int>(prev.y), static_cast<int>(prev.x));
            int cur_val  = gray_img.at<uchar>(static_cast<int>(cur.y), static_cast<int>(cur.x));

            float brightness_diff = static_cast<float>(prev_val - cur_val);

            if (brightness_diff > max_brightness_diff && prev_val > axis.mean_val) {
                max_brightness_diff = brightness_diff;
                corner              = prev;
                has_corner          = true;
            }

            prev = cur;
        }

        if (has_corner) {
            candidates.emplace_back(corner);
        }
    }

    // Average all candidate corners
    if (!candidates.empty()) {
        cv::Point2f result =
            std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0));
        return result / static_cast<float>(candidates.size());
    }

    return cv::Point2f(-1, -1);
}

} // namespace fcs::L2
