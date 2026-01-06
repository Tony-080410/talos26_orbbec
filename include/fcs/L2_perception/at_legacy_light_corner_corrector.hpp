#pragma once

#include "at_legacy_traditional_types.hpp"

#include <opencv2/core.hpp>

namespace fcs {

// ============================================================================
// Symmetry Axis Structure
// ============================================================================

struct SymmetryAxis {
    cv::Point2f centroid;  // Center point of the light
    cv::Point2f direction; // Direction vector (normalized)
    float mean_val = 0.0f; // Mean brightness value
};

// ============================================================================
// Light Corner Corrector (PCA-based)
// ============================================================================

/// Sub-pixel accurate corner detection using PCA and gradient search
class AtLegacyLightCornerCorrector {
public:
    /// Correct corners of armor lights using PCA-based method
    /// Updates the top and bottom points of left and right lights in-place
    void correct_corners(
        AtLegacyLight& left_light, AtLegacyLight& right_light,
        const cv::Mat& gray_img) const noexcept;

private:
    /// Minimum light width threshold for applying optimization
    static constexpr int PASS_OPTIMIZE_WIDTH = 3;

    /// Find the symmetry axis of a light bar using PCA
    [[nodiscard]] SymmetryAxis
        find_symmetry_axis(const cv::Mat& gray_img, const AtLegacyLight& light) const noexcept;

    /// Find the corner of a light bar along its symmetry axis
    [[nodiscard]] cv::Point2f find_corner(
        const cv::Mat& gray_img, const AtLegacyLight& light, const SymmetryAxis& axis,
        bool is_top) const noexcept;
};

} // namespace fcs
