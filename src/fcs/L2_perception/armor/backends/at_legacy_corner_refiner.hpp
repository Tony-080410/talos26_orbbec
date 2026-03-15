#pragma once

#include "at_legacy_traditional_types.hpp"
#include "core/types.hpp"

#include <numbers>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace fcs::L2 {

// ============================================================================
// Light Corner Correction (PCA + Gradient Search)
// 移植自 at_vision (armor_detector/light_corner_corrector.cpp)
// ============================================================================

/// Symmetry axis of a light bar (PCA result)
struct SymmetryAxis {
    cv::Point2f centroid;  ///< Brightness-weighted center
    cv::Point2f direction; ///< Principal component direction (normalized)
    float mean_val;        ///< Mean brightness of ROI
};

/// Light bar corner corrector using PCA symmetry axis + gradient search
///
/// Algorithm (移植自 at_vision):
/// 1. PCA 找灯条对称轴: 扩展 ROI → 归一化亮度 → 构建点云 → PCA → 质心 + 主成分
/// 2. 沿对称轴搜索角点: 从 40%-60% length 搜索亮度梯度最大位置
/// 3. 修改 Light 字段: center → 亮度质心, axis → PCA 主成分, top/bottom → 梯度角点
///
/// 关键参数 (from at_vision):
/// - MAX_BRIGHTNESS = 25: 亮度归一化上限
/// - SCALE = 0.07: ROI 扩展 7%
/// - START = 0.4: 搜索起点 (40% length)
/// - END = 0.6: 搜索终点 (60% length)
class LightCornerCorrector {
public:
    LightCornerCorrector() = default;

    /// Correct the corners of armor lights using PCA + gradient search
    /// @param armor Detection to refine (modifies corners in-place)
    /// @param gray_img Grayscale image (uint8_t, single-channel)
    static void correctCorners(AtLegacyArmorDetection& armor, const cv::Mat& gray_img);

private:
    /// Find the symmetry axis of a light using PCA
    /// @param gray_img Grayscale image
    /// @param light Light bar (provides initial bounding rect)
    /// @return Symmetry axis with centroid, direction, and mean brightness
    static SymmetryAxis findSymmetryAxis(const cv::Mat& gray_img, const AtLegacyLight& light);

    /// Corner search direction (compile-time type safe)
    enum class CornerOrder { Top, Bottom };

    /// Find a corner point by searching along the symmetry axis
    /// @param gray_img Grayscale image
    /// @param light Light bar (provides length and width)
    /// @param axis Symmetry axis from PCA
    /// @param order Top or bottom corner
    /// @return Corner point (cv::Point2f), or (-1, -1) if not found
    static cv::Point2f findCorner(
        const cv::Mat& gray_img, const AtLegacyLight& light, const SymmetryAxis& axis,
        CornerOrder order);

    // Constants from at_vision (light_corner_corrector.cpp)
    static constexpr float MAX_BRIGHTNESS     = 25.0f; ///< Brightness normalization upper bound
    static constexpr float ROI_SCALE          = 0.07f; ///< ROI expansion (7%)
    static constexpr float SEARCH_START_RATIO = 0.4f;  ///< Search start (40% of length)
    static constexpr float SEARCH_END_RATIO   = 0.6f;  ///< Search end (60% of length)
    static constexpr int PASS_OPTIMIZE_WIDTH  = 3;     ///< Minimum width for refinement
};

} // namespace fcs::L2
