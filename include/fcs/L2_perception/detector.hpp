#pragma once

#include "../core/types.hpp"

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <span>
#include <vector>

namespace fcs::L2 {

struct Lightbar {
    bool valid;
    ArmorColor color;                   // 颜色
    uint8_t pad_[2];
    cv::Point2f center;                 // 中心点
    float angle_error;                  // Error from ideal angle (4B)
    float ratio;                        // length / width (4B)
};
static_assert(sizeof(Lightbar) == 20, "Lightbar must be exactly 20 bytes");

struct Detection {
    cv::Point2f center;                 // 中心点
    std::array<cv::Point2f, 4> corners; // 四角点: [tl, tr, br, bl]
    float confidence;                   // 分类置信度
    ArmorColor color;                   // RED/BLUE
    ArmorType type;                     // BIG/SMALL
    ArmorName name;                     // 装甲板编号
    uint64_t timestamp_ns;              // 时间戳
    uint64_t frame_id;                  // 帧ID
};
static_assert(sizeof(Detection) == 64, "Detection must be exactly 48 bytes");

// ============================================================================
// DetectorConfig - 检测器配置
// ============================================================================

struct DetectorConfig {
    // 二值化
    uint8_t binary_threshold{150};

    // 灯条几何约束
    float max_angle_error{0.785f};   // 45度 (弧度)
    float min_lightbar_ratio{1.5f};  // 长/宽 最小比
    float max_lightbar_ratio{20.0f}; // 长/宽 最大比
    float min_lightbar_length{8.0f}; // 最小长度 (像素)

    // 装甲板几何约束
    float min_armor_ratio{1.0f};         // 宽/高 最小比
    float max_armor_ratio{5.0f};         // 宽/高 最大比
    float max_side_ratio{1.5f};          // 灯条长度比
    float max_rectangular_error{0.436f}; // 25度 (弧度)

    // 分类
    float min_confidence{0.8f};
};

// ============================================================================
// LightbarDetector - 灯条配对检测器
// ============================================================================

class LightbarDetector {
public:
    explicit LightbarDetector(const DetectorConfig& config = {});

    /**
     * @brief 检测装甲板
     * @param bgr 彩色图像
     * @param binary 二值化图像 (可选，传空则内部生成)
     * @param timestamp_ns 时间戳
     * @param frame_id 帧ID
     * @return 检测到的装甲板列表
     */
    [[nodiscard]] std::span<Detection> detect(
        const cv::Mat& bgr, const cv::Mat& binary, uint64_t timestamp_ns,
        uint64_t frame_id) noexcept;

    /**
     * @brief ROI精修 - 在已知位置精确定位角点
     * @param det 检测结果 (会被修改)
     * @param bgr 彩色图像
     * @return 是否成功
     */
    [[nodiscard]] bool refine(Detection& det, const cv::Mat& bgr) const noexcept;

private:
    // 灯条内部数据 (用于配对，包含几何信息)
    struct LightbarInternal {
        size_t id{0};
        Lightbar lb;
        cv::RotatedRect rect;
    };

    DetectorConfig config_;

    // 工作缓冲区 (复用)
    std::vector<LightbarInternal> lightbars_;
    std::vector<Detection> detections_;
    cv::Mat gray_;
    cv::Mat binary_internal_;

    // ======== 流水线阶段 ========

    // 1. 预处理
    void preprocess(const cv::Mat& bgr, const cv::Mat& binary);

    // 2. 灯条检测
    void detect_lightbars(const cv::Mat& bgr);

    // 3. 灯条配对
    void pair_lightbars(uint64_t timestamp_ns, uint64_t frame_id);

    // 4. 去重
    void remove_duplicates();

    // ======== 几何检查 ========
    [[nodiscard]] bool check_lightbar(const Lightbar& lb) const noexcept;
    [[nodiscard]] bool
        check_armor_geometry(const Lightbar& left, const Lightbar& right) const noexcept;

    // ======== 工具函数 ========
    [[nodiscard]] static ArmorColor
        get_color(const cv::Mat& bgr, std::span<const cv::Point> contour) noexcept;
    [[nodiscard]] static ArmorType get_type(const Lightbar& left, const Lightbar& right) noexcept;

    // ======== 内部构造 ========
    [[nodiscard]] static LightbarInternal
        make_lightbar(const cv::RotatedRect& rect, size_t id, ArmorColor color) noexcept;
    [[nodiscard]] static Detection make_armor(
        const LightbarInternal& left, const LightbarInternal& right, uint64_t timestamp_ns,
        uint64_t frame_id) noexcept;
};

} // namespace fcs::L2
