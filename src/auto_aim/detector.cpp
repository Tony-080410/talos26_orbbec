#include "detector.hpp"
#include <algorithm>
#include <fmt/format.h>
#include <numeric>
#include <ranges>

namespace talos {

// ============ LightbarFull 构造函数 ============

LightbarFull::LightbarFull(const cv::RotatedRect& rect, const std::size_t id)
    : id(id)
    , rotated_rect(rect) {
    std::vector<cv::Point2f> corners(4);
    rect.points(&corners[0]);
    std::ranges::sort(
        corners, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    center     = rect.center;
    top        = (corners[0] + corners[1]) / 2.0;
    bottom     = (corners[2] + corners[3]) / 2.0;
    top2bottom = bottom - top;

    width       = cv::norm(corners[0] - corners[1]);
    angle       = std::atan2(top2bottom.y, top2bottom.x);
    angle_error = std::abs(angle - CV_PI / 2.0);
    length      = cv::norm(top2bottom);
    ratio       = width > 0.1 ? length / width : 0.0;
}

// ============ ArmorFull 构造函数 ============

ArmorFull::ArmorFull(const LightbarFull& l, const LightbarFull& r)
    : left(l)
    , right(r) {
    color  = left.color;
    center = (left.center + right.center) / 2.0;

    points.reserve(4);
    points.emplace_back(left.top);
    points.emplace_back(right.top);
    points.emplace_back(right.bottom);
    points.emplace_back(left.bottom);

    const auto left2right          = right.center - left.center;
    const auto width               = cv::norm(left2right);
    const auto max_lightbar_length = std::max(left.length, right.length);
    const auto min_lightbar_length = std::min(left.length, right.length);

    // 简单 ratio 计算，不做透视矫正
    // 正确的透视矫正应该在 PnP 解算后通过相机模型完成
    ratio      = width / max_lightbar_length;
    side_ratio = max_lightbar_length / min_lightbar_length;

    const auto roll                    = std::atan2(left2right.y, left2right.x);
    const auto left_rectangular_error  = std::abs(left.angle - roll - CV_PI / 2);
    const auto right_rectangular_error = std::abs(right.angle - roll - CV_PI / 2);
    rectangular_error                  = std::max(left_rectangular_error, right_rectangular_error);
}

} // namespace talos

namespace auto_aim {

// ============ TraditionalDetector ============

TraditionalDetector::TraditionalDetector(const SpVisionConfig& config, const bool debug)
    : config_(DetectorConfig::from(config.detector))
    , classifier_(config.classifier)
    , debug_(debug) {
    // 预转换角度为弧度
    max_angle_error_rad_       = config_.max_angle_error / 57.3;
    max_rectangular_error_rad_ = config_.max_rectangular_error / 57.3;
}

TraditionalDetector::TraditionalDetector(const SpVisionConfig& config, Classifier classifier)
    : config_(DetectorConfig::from(config.detector))
    , classifier_(classifier)
    , debug_(false) {
    // 预转换角度为弧度
    max_angle_error_rad_       = config_.max_angle_error / 57.3;
    max_rectangular_error_rad_ = config_.max_rectangular_error / 57.3;
}

void TraditionalDetector::preprocess(
    const cv::Mat& bgr_img, const cv::Mat& binary_img) const noexcept {
    cv::cvtColor(bgr_img, binary_img, cv::COLOR_BGR2GRAY);
    cv::threshold(binary_img, binary_img, config_.threshold, 255, cv::THRESH_BINARY);
}

talos::ArmorSlots TraditionalDetector::detect(
    DetectorWorkBuffers& buffers, const cv::Mat& bgr_img, const cv::Mat& binary_img,
    const uint64_t frame_id, const uint64_t timestamp_ns) const {
    // 清空上一帧结果
    buffers.clear();

    if (bgr_img.empty() || bgr_img.channels() != 3) {
        return to_armor_slots(buffers, frame_id, timestamp_ns);
    }

    // 获取轮廓点
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 获取灯条
    std::size_t lightbar_id = 0;
    for (const auto& contour : contours) {
        if (contour.size() < 5)
            continue;

        auto rotated_rect = cv::minAreaRect(contour);
        if (rotated_rect.size.width < 1.0f || rotated_rect.size.height < 1.0f)
            continue;

        auto lightbar = talos::LightbarFull(rotated_rect, lightbar_id);

        if (!check_geometry(lightbar))
            continue;

        lightbar.color = get_color(bgr_img, contour);
        buffers.lightbars.emplace_back(lightbar);
        lightbar_id += 1;
    }

    // 将灯条从左到右排序
    buffers.lightbars.sort([](const talos::LightbarFull& a, const talos::LightbarFull& b) {
        return a.center.x < b.center.x;
    });

    spdlog::debug("{} lightbars detected", buffers.lightbars.size());
    // 获取装甲板 - 双重循环配对灯条
    for (auto left = buffers.lightbars.begin(); left != buffers.lightbars.end(); ++left) {
        for (auto right = std::next(left); right != buffers.lightbars.end(); ++right) {
            if (left->color != right->color)
                continue;

            auto armor = talos::ArmorFull(*left, *right);
            if (!check_geometry(armor)) {
                spdlog::debug("armor geometry check failed");
                continue;
            }

            // 设置装甲板类型 (不依赖分类器)
            armor.pattern = get_pattern(bgr_img, armor);
            classifier_.classify_ov(armor);
            if (!check_name(armor)) {
                spdlog::debug("armor name check failed");
                continue;
            }

            armor.type = get_type(armor);
            if (!check_type(armor)) {
                spdlog::debug("armor type check failed");
                continue;
            }
            spdlog::debug("armor passed all checks");
            armor.center_norm = get_center_norm(bgr_img, armor.center);
            buffers.armors.emplace_back(armor);
        }
    }
    spdlog::debug("{} armors detected", buffers.armors.size());

    // 去重处理
    remove_duplicates(buffers);

    // 调试输出
    if (debug_) {
        show_result(buffers, binary_img, bgr_img, static_cast<int>(frame_id));
    }

    return to_armor_slots(buffers, frame_id, timestamp_ns);
}

bool TraditionalDetector::refine(talos::ArmorFull& armor, const cv::Mat& bgr_img) const {
    // 取得四个角点
    if (armor.points.size() < 4)
        return false;

    auto tl = armor.points[0];
    auto tr = armor.points[1];
    auto br = armor.points[2];
    auto bl = armor.points[3];

    // 计算向量和调整后的点
    auto lt2b  = bl - tl;
    auto rt2b  = br - tr;
    auto tl1   = (tl + bl) / 2.0 - lt2b;
    auto bl1   = (tl + bl) / 2.0 + lt2b;
    auto br1   = (tr + br) / 2.0 + rt2b;
    auto tr1   = (tr + br) / 2.0 - rt2b;
    auto tl2tr = tr1 - tl1;
    auto bl2br = br1 - bl1;
    auto tl2   = (tl1 + tr) / 2.0 - 0.75f * tl2tr;
    auto tr2   = (tl1 + tr) / 2.0 + 0.75f * tl2tr;
    auto bl2   = (bl1 + br) / 2.0 - 0.75f * bl2br;
    auto br2   = (bl1 + br) / 2.0 + 0.75f * bl2br;

    // 构造新的四个角点
    std::vector<cv::Point> points = {tl2, tr2, br2, bl2};
    auto armor_rotate_rect        = cv::minAreaRect(points);
    cv::Rect boundingBox          = armor_rotate_rect.boundingRect();

    // 检查boundingBox是否超出图像边界
    if (boundingBox.x < 0 || boundingBox.y < 0 || boundingBox.x + boundingBox.width > bgr_img.cols
        || boundingBox.y + boundingBox.height > bgr_img.rows) {
        return false;
    }

    // 在图像上裁剪出这个矩形区域（ROI）
    cv::Mat armor_roi = bgr_img(boundingBox);
    if (armor_roi.empty()) {
        return false;
    }

    // 彩色图转灰度图
    cv::Mat gray_img;
    cv::cvtColor(armor_roi, gray_img, cv::COLOR_BGR2GRAY);

    // 进行二值化
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, config_.threshold, 255, cv::THRESH_BINARY);

    // 获取轮廓点
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat binary_copy = binary_img.clone();
    cv::findContours(binary_copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 获取灯条
    std::list<talos::LightbarFull> roi_lightbars;
    std::size_t lightbar_id = 0;
    for (const auto& contour : contours) {
        if (contour.size() < 5)
            continue;

        auto rotated_rect = cv::minAreaRect(contour);
        auto lightbar     = talos::LightbarFull(rotated_rect, lightbar_id);

        if (!check_geometry(lightbar))
            continue;

        lightbar.color = get_color(armor_roi, contour);
        roi_lightbars.emplace_back(lightbar);
        lightbar_id += 1;
    }

    if (roi_lightbars.size() < 2)
        return false;

    // 将灯条从左到右排序
    roi_lightbars.sort([](const talos::LightbarFull& a, const talos::LightbarFull& b) {
        return a.center.x < b.center.x;
    });

    // 计算与 tl, bl 和 br, tr 距离最近的灯条
    talos::LightbarFull* closest_left_lightbar  = nullptr;
    talos::LightbarFull* closest_right_lightbar = nullptr;
    double min_distance_tl_bl                   = std::numeric_limits<double>::max();
    double min_distance_br_tr                   = std::numeric_limits<double>::max();

    cv::Point2f offset(static_cast<float>(boundingBox.x), static_cast<float>(boundingBox.y));

    for (auto& lightbar : roi_lightbars) {
        if (auto distance_tl_bl =
                cv::norm(tl - (lightbar.top + offset)) + cv::norm(bl - (lightbar.bottom + offset));
            distance_tl_bl < min_distance_tl_bl) {
            min_distance_tl_bl    = distance_tl_bl;
            closest_left_lightbar = &lightbar;
        }

        if (auto distance_br_tr =
                cv::norm(br - (lightbar.bottom + offset)) + cv::norm(tr - (lightbar.top + offset));
            distance_br_tr < min_distance_br_tr) {
            min_distance_br_tr     = distance_br_tr;
            closest_right_lightbar = &lightbar;
        }
    }

    if (closest_left_lightbar && closest_right_lightbar
        && min_distance_br_tr + min_distance_tl_bl < 15) {
        // 将四个点从armor_roi坐标系转换到原始图像坐标系
        armor.points[0] = closest_left_lightbar->top + offset;
        armor.points[1] = closest_right_lightbar->top + offset;
        armor.points[2] = closest_right_lightbar->bottom + offset;
        armor.points[3] = closest_left_lightbar->bottom + offset;
        return true;
    }

    return false;
}

// ============ 几何检查 ============

bool TraditionalDetector::check_geometry(const talos::LightbarFull& lightbar) const {
    const auto angle_ok = lightbar.angle_error < max_angle_error_rad_;
    const auto ratio_ok =
        lightbar.ratio > config_.min_lightbar_ratio && lightbar.ratio < config_.max_lightbar_ratio;
    const auto length_ok = lightbar.length > config_.min_lightbar_length;
    return angle_ok && ratio_ok && length_ok;
}

bool TraditionalDetector::check_geometry(const talos::ArmorFull& armor) const {
    const auto ratio_ok =
        armor.ratio > config_.min_armor_ratio && armor.ratio < config_.max_armor_ratio;
    const auto side_ratio_ok        = armor.side_ratio < config_.max_side_ratio;
    const auto rectangular_error_ok = armor.rectangular_error < max_rectangular_error_rad_;
    return ratio_ok && side_ratio_ok && rectangular_error_ok;
}

bool TraditionalDetector::check_name(const talos::ArmorFull& armor) const {
    if (armor.confidence < config_.min_confidence) {
        return false;
    }
    const auto name_ok = armor.name != talos::ArmorName::not_armor;

    // 保存不确定的图案，用于分类器的迭代
    // if (name_ok && !confidence_ok) save(armor);

    // 出现 5号 则显示 debug 信息。但不过滤。
    if (armor.name == talos::ArmorName::five) {
        // std::cerr << "classifier detected an armor five, which is outdated";
    }

    return name_ok;
}

bool TraditionalDetector::check_type(const talos::ArmorFull& armor) {
    const auto name_ok =
        armor.type == talos::ArmorType::small
            ? armor.name != talos::ArmorName::one && armor.name != talos::ArmorName::base
            : armor.name == talos::ArmorName::one || armor.name == talos::ArmorName::base;

    // cv::imshow("ROI", armor.pattern);
    // 保存异常的图案，用于分类器的迭代
    if (!name_ok) {
        // spdlog::warn("classifier detected a strange armor {} with type {}",
        //              talos::ARMOR_NAMES[static_cast<uint8_t>(armor.name)],
        //              talos::ARMOR_TYPES[static_cast<uint8_t>(armor.type)]);
        // if (debug_) {
        //   cv::imshow("ROI", armor.pattern);
        //}
        return false;
    }

    return name_ok;
}

// ============ 属性获取 ============

talos::Color
    TraditionalDetector::get_color(const cv::Mat& bgr_img, const std::vector<cv::Point>& contour) {
    int red_sum = 0, blue_sum = 0;

    for (const auto& point : contour) {
        if (point.x >= 0 && point.x < bgr_img.cols && point.y >= 0 && point.y < bgr_img.rows) {
            red_sum += bgr_img.at<cv::Vec3b>(point)[2];
            blue_sum += bgr_img.at<cv::Vec3b>(point)[0];
        }
    }

    return blue_sum > red_sum ? talos::Color::blue : talos::Color::red;
}

talos::ArmorType TraditionalDetector::get_type(const talos::ArmorFull& armor) {
    // 优先根据当前armor.ratio判断
    if (armor.ratio > 3.0) {
        return talos::ArmorType::big;
    }
    if (armor.ratio < 2.5) {
        return talos::ArmorType::small;
    }
    if (armor.name == talos::ArmorName::one || armor.name == talos::ArmorName::base) {
        return talos::ArmorType::big;
    }
    // 默认小装甲板
    return talos::ArmorType::small;
}

cv::Point2f
    TraditionalDetector::get_center_norm(const cv::Mat& bgr_img, const cv::Point2f& center) {
    const auto h = static_cast<float>(bgr_img.rows);
    const auto w = static_cast<float>(bgr_img.cols);
    return {center.x / w, center.y / h};
}

cv::Mat TraditionalDetector::get_pattern(const cv::Mat& bgr_img, const talos::ArmorFull& armor) {
    // 延长灯条获得装甲板角点
    // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
    const auto& left  = armor.left;
    const auto& right = armor.right;

    const auto tl = left.center - left.top2bottom * 1.125f;
    const auto bl = left.center + left.top2bottom * 1.125f;
    const auto tr = right.center - right.top2bottom * 1.125f;
    const auto br = right.center + right.top2bottom * 1.125f;

    const auto roi_left   = std::max<int>(static_cast<int>(std::min(tl.x, bl.x)), 0);
    const auto roi_top    = std::max<int>(static_cast<int>(std::min(tl.y, tr.y)), 0);
    const auto roi_right  = std::min<int>(static_cast<int>(std::max(tr.x, br.x)), bgr_img.cols);
    const auto roi_bottom = std::min<int>(static_cast<int>(std::max(bl.y, br.y)), bgr_img.rows);
    const auto roi_tl     = cv::Point(roi_left, roi_top);
    const auto roi_br     = cv::Point(roi_right, roi_bottom);
    const auto roi        = cv::Rect(roi_tl, roi_br);
    /*if (roi_right <= roi_left || roi_bottom <= roi_top) {
        return {};
    }*/

    return bgr_img(roi);
}

// ============ 去重处理 ============

void TraditionalDetector::remove_duplicates(DetectorWorkBuffers& buffers) {
    // 检查装甲板是否存在共用灯条的情况
    for (auto armor1 = buffers.armors.begin(); armor1 != buffers.armors.end(); ++armor1) {
        for (auto armor2 = std::next(armor1); armor2 != buffers.armors.end(); ++armor2) {
            if (armor1->left.id != armor2->left.id && armor1->left.id != armor2->right.id
                && armor1->right.id != armor2->left.id && armor1->right.id != armor2->right.id) {
                continue;
            }

            // 装甲板重叠，保留较小的
            if (armor1->left.id == armor2->left.id || armor1->right.id == armor2->right.id) {
                const auto area1 = armor1->left.length * armor1->right.length;
                if (const auto area2 = armor2->left.length * armor2->right.length; area1 < area2)
                    armor2->duplicated = true;
                else
                    armor1->duplicated = true;
            }

            // 装甲板相连，保留置信度大的
            if (armor1->left.id == armor2->right.id || armor1->right.id == armor2->left.id) {
                if (armor1->confidence < armor2->confidence)
                    armor1->duplicated = true;
                else
                    armor2->duplicated = true;
            }
        }
    }

    buffers.armors.remove_if([](const talos::ArmorFull& a) { return a.duplicated; });
}

// ============ 结果转换 ============

talos::ArmorSlots TraditionalDetector::to_armor_slots(
    const DetectorWorkBuffers& buffers, const uint64_t frame_id, const uint64_t timestamp_ns) {
    talos::ArmorSlots result;
    result.frame_id     = frame_id;
    result.timestamp_ns = timestamp_ns;
    result.clear();

    // 填充灯条
    uint8_t lb_count = 0;
    for (const auto& lb : buffers.lightbars) {
        if (lb_count >= 20)
            break;
        result.lightbars[lb_count].x           = static_cast<float>(lb.center.x);
        result.lightbars[lb_count].y           = static_cast<float>(lb.center.y);
        result.lightbars[lb_count].angle_error = static_cast<float>(lb.angle_error);
        result.lightbars[lb_count].ratio       = static_cast<float>(lb.ratio);
        result.lightbars[lb_count].color       = static_cast<uint8_t>(lb.color);
        lb_count++;
    }
    result.lightbar_count = lb_count;

    // 填充装甲板
    uint8_t armor_count = 0;
    for (const auto& armor : buffers.armors) {
        if (armor_count >= 10)
            break;

        auto& out      = result.armors[armor_count];
        out.x2d        = static_cast<float>(armor.center.x);
        out.y2d        = static_cast<float>(armor.center.y);
        out.confidence = static_cast<float>(armor.confidence);
        out.color      = armor.color;
        out.type       = armor.type;
        out.name       = armor.name;

        // 四个角点
        if (armor.points.size() >= 4) {
            out.pts[0] = armor.points[0].x; // tl_x
            out.pts[1] = armor.points[0].y; // tl_y
            out.pts[2] = armor.points[1].x; // tr_x
            out.pts[3] = armor.points[1].y; // tr_y
            out.pts[4] = armor.points[2].x; // br_x
            out.pts[5] = armor.points[2].y; // br_y
            out.pts[6] = armor.points[3].x; // bl_x
            out.pts[7] = armor.points[3].y; // bl_y
        }

        result.set_valid(armor_count);
        armor_count++;
    }
    result.armor_count = armor_count;

    return result;
}

// ============ PCA角点修正 ============

void TraditionalDetector::lightbar_points_corrector(
    talos::LightbarFull& lightbar, const cv::Mat& gray_img) {
    // 配置参数
    constexpr float MAX_BRIGHTNESS = 25;
    constexpr float ROI_SCALE      = 0.07f;
    constexpr float SEARCH_START   = 0.4f;
    constexpr float SEARCH_END     = 0.6f;

    // 扩展并裁剪ROI
    cv::Rect roi_box = lightbar.rotated_rect.boundingRect();
    roi_box.x -= static_cast<int>(static_cast<float>(roi_box.width) * ROI_SCALE);
    roi_box.y -= static_cast<int>(static_cast<float>(roi_box.height) * ROI_SCALE);
    roi_box.width += static_cast<int>(2.0 * roi_box.width * ROI_SCALE);
    roi_box.height += static_cast<int>(2.0 * roi_box.height * ROI_SCALE);

    // 边界约束
    roi_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);
    if (roi_box.width <= 0 || roi_box.height <= 0)
        return;

    // 归一化ROI
    cv::Mat roi          = gray_img(roi_box);
    const float mean_val = static_cast<float>(cv::mean(roi)[0]);
    cv::Mat roi_float;
    roi.convertTo(roi_float, CV_32F);
    cv::normalize(roi_float, roi_float, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

    // 计算质心
    const cv::Moments moments = cv::moments(roi_float);
    if (moments.m00 < 1e-6)
        return;

    const cv::Point2f centroid(
        static_cast<float>(moments.m10 / moments.m00 + roi_box.x),
        static_cast<float>(moments.m01 / moments.m00 + roi_box.y));

    // 生成稀疏点云
    std::vector<cv::Point2f> points;
    for (int i = 0; i < roi_float.rows; ++i) {
        for (int j = 0; j < roi_float.cols; ++j) {
            if (const float weight = roi_float.at<float>(i, j); weight > 1e-3f) {
                points.emplace_back(static_cast<float>(j), static_cast<float>(i));
            }
        }
    }

    if (points.size() < 5)
        return;

    // PCA计算对称轴方向
    cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
    auto axis_norm = cv::norm(axis);
    if (axis_norm < 1e-6f)
        return;
    axis /= axis_norm;
    if (axis.y > 0)
        axis = -axis;

    const auto find_corner = [&](const float direction) -> cv::Point2f {
        const float dx            = axis.x * direction;
        const float dy            = axis.y * direction;
        const auto length         = static_cast<float>(lightbar.length);
        const float search_length = length * (SEARCH_END - SEARCH_START);

        std::vector<cv::Point2f> candidates;

        const int half_width = static_cast<int>((lightbar.width - 2) / 2);
        for (int i_offset = -half_width; i_offset <= half_width; ++i_offset) {
            const cv::Point2f start_point(
                centroid.x + length * SEARCH_START * dx + static_cast<float>(i_offset),
                centroid.y + length * SEARCH_START * dy);

            cv::Point2f corner = start_point;
            float max_diff     = 0;
            bool found         = false;

            for (float step = 0; step < search_length; ++step) {
                const cv::Point2f cur_point(start_point.x + dx * step, start_point.y + dy * step);

                const double cols = gray_img.cols;
                const double rows = gray_img.rows;
                if (cur_point.x < 0 || cur_point.x >= cols || cur_point.y < 0
                    || cur_point.y >= rows) {
                    break;
                }

                cv::Point2f prev_point = cur_point - cv::Point2f(dx, dy);
                if (prev_point.x < 0 || prev_point.x >= cols || prev_point.y < 0
                    || prev_point.y >= rows) {
                    continue;
                }

                const float prev_val = gray_img.at<uchar>(cv::Point2i(prev_point));
                const float cur_val  = gray_img.at<uchar>(cv::Point2i(cur_point));

                if (const float diff = static_cast<float>(prev_val) - static_cast<float>(cur_val);
                    diff > max_diff && prev_val > mean_val) {
                    max_diff = diff;
                    corner   = prev_point;
                    found    = true;
                }
            }

            if (found) {
                candidates.push_back(corner);
            }
        }

        if (candidates.empty()) {
            return {-1.0, -1.0};
        }
        return std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0))
             / static_cast<float>(candidates.size());
    };

    lightbar.top    = find_corner(1);
    lightbar.bottom = find_corner(-1);
}

// ============ 调试输出 ============

void TraditionalDetector::show_result(
    const DetectorWorkBuffers& buffers, const cv::Mat& binary_img, const cv::Mat& bgr_img,
    int frame_count) const {
    auto detection = bgr_img.clone();

    // 绘制帧号
    cv::putText(
        detection, fmt::format("[{}]", frame_count), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
        {255, 255, 255}, 2);

    // 绘制灯条
    for (const auto& lightbar : buffers.lightbars) {
        auto info = fmt::format(
            "{:.1f} {:.1f} {:.1f}", lightbar.angle_error * 57.3, lightbar.ratio, lightbar.length);
        cv::putText(detection, info, lightbar.top, cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 255, 255}, 1);
        cv::circle(detection, lightbar.top, 3, {0, 255, 255}, -1);
        cv::circle(detection, lightbar.bottom, 3, {0, 255, 255}, -1);
    }

    // 绘制装甲板
    for (const auto& armor : buffers.armors) {
        if (armor.points.size() >= 4) {
            std::vector<cv::Point> pts;
            for (const auto& p : armor.points) {
                pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
            }
            cv::polylines(detection, pts, true, {0, 255, 0}, 2);
        }

        auto info = fmt::format(
            "{:.2f} {:.2f} {:.1f}", armor.ratio, armor.side_ratio, armor.rectangular_error * 57.3);
        cv::putText(detection, info, armor.center, cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 255, 0}, 1);
    }

    cv::Mat binary_resized, detection_resized;
    cv::resize(binary_img, binary_resized, {}, 0.5, 0.5);
    cv::resize(detection, detection_resized, {}, 0.5, 0.5);

    cv::imshow("binary", binary_resized);
    cv::imshow("detection", detection_resized);
}

} // namespace auto_aim
