#include "fcs/L2_perception/detector.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// LightbarDetector Implementation
// ============================================================================

LightbarDetector::LightbarInternal LightbarDetector::make_lightbar(
    const cv::RotatedRect& rect, size_t id, ArmorColor color) noexcept {
    std::array<cv::Point2f, 4> corners;
    rect.points(corners.data());

    // 按 y 坐标排序找 top/bottom
    std::ranges::sort(
        corners, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    Lightbar lb;
    lb.top    = (corners[0] + corners[1]) * 0.5f;
    lb.bottom = (corners[2] + corners[3]) * 0.5f;
    lb.center = rect.center;
    lb.width  = cv::norm(corners[0] - corners[1]);

    const auto top2bottom = lb.bottom - lb.top;
    lb.length             = cv::norm(top2bottom);
    lb.angle = std::atan2(top2bottom.y, top2bottom.x) - static_cast<float>(CV_PI / 2.0);
    lb.color = color;

    return {id, lb, rect};
}

Detection LightbarDetector::make_armor(
    const LightbarInternal& left, const LightbarInternal& right, uint64_t timestamp_ns,
    uint64_t frame_id) noexcept {
    Detection det;
    det.corners[0]   = left.lb.top;     // tl
    det.corners[1]   = right.lb.top;    // tr
    det.corners[2]   = right.lb.bottom; // br
    det.corners[3]   = left.lb.bottom;  // bl
    det.center       = (left.lb.center + right.lb.center) * 0.5f;
    det.color        = left.lb.color;
    det.timestamp_ns = timestamp_ns;
    det.frame_id     = frame_id;

    // 计算 ratio
    const auto width   = cv::norm(right.lb.center - left.lb.center);
    const auto max_len = std::max(left.lb.length, right.lb.length);
    const auto ratio   = max_len > 0.1f ? width / max_len : 0.0f;

    // 判断类型
    det.type = (ratio > 3.0f) ? ArmorType::Big : ArmorType::Small;

    return det;
}

LightbarDetector::LightbarDetector(const DetectorConfig& config)
    : config_(config) {}

std::vector<Detection> LightbarDetector::detect(
    const cv::Mat& bgr, const cv::Mat& binary, uint64_t timestamp_ns, uint64_t frame_id) noexcept {
    detections_.clear();

    if (bgr.empty() || bgr.channels() != 3) {
        return detections_;
    }

    // 1. 预处理
    preprocess(bgr, binary);

    // 2. 检测灯条
    detect_lightbars(bgr);

    // 3. 配对灯条
    pair_lightbars(timestamp_ns, frame_id);

    // 4. 去重
    remove_duplicates();

    return detections_;
}

void LightbarDetector::preprocess(const cv::Mat& bgr, const cv::Mat& binary) {
    if (binary.empty()) {
        cv::cvtColor(bgr, gray_, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_, binary_internal_, config_.binary_threshold, 255, cv::THRESH_BINARY);
    } else {
        binary_internal_ = binary;
    }
}

void LightbarDetector::detect_lightbars(const cv::Mat& bgr) {
    lightbars_.clear();

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_internal_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    size_t id = 0;
    for (const auto& contour : contours) {
        if (contour.size() < 5)
            continue;

        auto rect = cv::minAreaRect(contour);
        if (rect.size.width < 1.0f || rect.size.height < 1.0f)
            continue;

        // 获取颜色
        auto color = get_color(bgr, contour);

        // 构造灯条
        auto lb = make_lightbar(rect, id, color);

        // 几何检查
        if (!check_lightbar(lb.lb))
            continue;

        lightbars_.push_back(lb);
        id++;
    }

    // 按 x 坐标排序
    std::ranges::sort(lightbars_, [](const LightbarInternal& a, const LightbarInternal& b) {
        return a.lb.center.x < b.lb.center.x;
    });

    spdlog::debug("[Detector] {} lightbars detected", lightbars_.size());
}

void LightbarDetector::pair_lightbars(uint64_t timestamp_ns, uint64_t frame_id) {
    for (size_t i = 0; i < lightbars_.size(); ++i) {
        for (size_t j = i + 1; j < lightbars_.size(); ++j) {
            const auto& left  = lightbars_[i];
            const auto& right = lightbars_[j];

            // 颜色必须相同
            if (left.lb.color != right.lb.color)
                continue;

            // 几何检查
            if (!check_armor_geometry(left.lb, right.lb))
                continue;

            // 构造装甲板
            auto det = make_armor(left, right, timestamp_ns, frame_id);
            detections_.push_back(det);
        }
    }

    spdlog::debug("[Detector] {} armors detected", detections_.size());
}

void LightbarDetector::remove_duplicates() {
    // 标记重复
    std::vector<bool> duplicated(detections_.size(), false);

    for (size_t i = 0; i < detections_.size(); ++i) {
        if (duplicated[i])
            continue;

        for (size_t j = i + 1; j < detections_.size(); ++j) {
            if (duplicated[j])
                continue;

            // 检查中心点距离
            const auto dist     = cv::norm(detections_[i].center - detections_[j].center);
            const auto min_size = std::min(
                cv::norm(detections_[i].corners[0] - detections_[i].corners[2]),
                cv::norm(detections_[j].corners[0] - detections_[j].corners[2]));

            if (dist < min_size * 0.5f) {
                // 保留置信度高的
                if (detections_[i].confidence < detections_[j].confidence) {
                    duplicated[i] = true;
                } else {
                    duplicated[j] = true;
                }
            }
        }
    }

    // 移除重复项
    std::erase_if(
        detections_, [&, idx = size_t{0}](const Detection&) mutable { return duplicated[idx++]; });
}

bool LightbarDetector::check_lightbar(const Lightbar& lb) const noexcept {
    const auto angle_error = std::abs(lb.angle);
    const auto ratio       = lb.width > 0.1f ? lb.length / lb.width : 0.0f;

    const bool angle_ok  = angle_error < config_.max_angle_error;
    const bool ratio_ok  = ratio > config_.min_lightbar_ratio && ratio < config_.max_lightbar_ratio;
    const bool length_ok = lb.length > config_.min_lightbar_length;

    return angle_ok && ratio_ok && length_ok;
}

bool LightbarDetector::check_armor_geometry(
    const Lightbar& left, const Lightbar& right) const noexcept {
    const auto width   = cv::norm(right.center - left.center);
    const auto max_len = std::max(left.length, right.length);
    const auto min_len = std::min(left.length, right.length);

    const auto ratio      = max_len > 0.1f ? width / max_len : 0.0f;
    const auto side_ratio = min_len > 0.1f ? max_len / min_len : 99.0f;

    // 计算矩形误差
    const auto left2right = right.center - left.center;
    const auto roll       = std::atan2(left2right.y, left2right.x);
    const auto left_err   = std::abs(left.angle + static_cast<float>(CV_PI / 2.0) - roll);
    const auto right_err  = std::abs(right.angle + static_cast<float>(CV_PI / 2.0) - roll);
    const auto rect_error = std::max(left_err, right_err);

    const bool ratio_ok      = ratio > config_.min_armor_ratio && ratio < config_.max_armor_ratio;
    const bool side_ratio_ok = side_ratio < config_.max_side_ratio;
    const bool rect_error_ok = rect_error < config_.max_rectangular_error;

    return ratio_ok && side_ratio_ok && rect_error_ok;
}

ArmorColor
    LightbarDetector::get_color(const cv::Mat& bgr, std::span<const cv::Point> contour) noexcept {
    int red_sum = 0, blue_sum = 0;

    for (const auto& pt : contour) {
        if (pt.x >= 0 && pt.x < bgr.cols && pt.y >= 0 && pt.y < bgr.rows) {
            const auto& pixel = bgr.at<cv::Vec3b>(pt);
            blue_sum += pixel[0];
            red_sum += pixel[2];
        }
    }

    return blue_sum > red_sum ? ArmorColor::Blue : ArmorColor::Red;
}

ArmorType LightbarDetector::get_type(const Lightbar& left, const Lightbar& right) noexcept {
    const auto width   = cv::norm(right.center - left.center);
    const auto max_len = std::max(left.length, right.length);
    const auto ratio   = max_len > 0.1f ? width / max_len : 0.0f;

    return ratio > 3.0f ? ArmorType::Big : ArmorType::Small;
}

bool LightbarDetector::refine(Detection& det, const cv::Mat& bgr) const noexcept {
    // TODO: 实现 ROI 精修
    (void)det;
    (void)bgr;
    return true;
}

} // namespace fcs::L2
