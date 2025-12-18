#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
#include "classifier.hpp"
#include "detector_state.hpp"
#include "sp_vision/config.hpp"

namespace spike {

    // ============ 传统检测器内部结构 (需要 OpenCV) ============

    /**
     * @brief LightbarFull - 完整灯条数据 (用于传统检测器内部)
     * 包含所有几何信息，用于配对和检验
     */
    struct LightbarFull {
        std::size_t id{0};
        Color color{red};
        cv::Point2f center, top, bottom, top2bottom;
        double angle{0}, angle_error{0}, length{0}, width{0}, ratio{0};
        cv::RotatedRect rotated_rect;

        LightbarFull() = default;
        LightbarFull(const cv::RotatedRect &rect, std::size_t id);
    };

    /**
     * @brief ArmorFull - 完整装甲板数据 (用于传统检测器内部)
     * 包含灯条引用、几何属性、分类结果
     */
    struct ArmorFull {
        Color color{red};
        LightbarFull left, right;
        cv::Point2f center;
        cv::Point2f center_norm; // 归一化坐标
        std::vector<cv::Point2f> points; // 四个角点: tl, tr, br, bl

        double ratio{0}; // 两灯条中点连线 / 长灯条长度
        double side_ratio{0}; // 长灯条 / 短灯条
        double rectangular_error{0}; // 灯条与中点连线夹角误差

        ArmorType type{small};
        ArmorName name{not_armor};
        double confidence{0};
        bool duplicated{false};

        cv::Mat pattern; // 装甲板图案 ROI (用于分类器)

        ArmorFull() = default;
        ArmorFull(const LightbarFull &l, const LightbarFull &r);
    };

} // namespace spike

namespace auto_aim {
    /**
     * @brief 传统装甲板检测器配置
     */
    struct DetectorConfig {
        double threshold{150};
        double max_angle_error{45}; // degree (内部转换为弧度)
        double min_lightbar_ratio{1.5};
        double max_lightbar_ratio{20};
        double min_lightbar_length{8};
        double min_armor_ratio{1};
        double max_armor_ratio{5};
        double max_side_ratio{1.5};
        double min_confidence{0.8};
        double max_rectangular_error{25}; // degree (内部转换为弧度)

        static constexpr auto from(const SpVisionDetectorConfig &config) {
            return DetectorConfig{config.threshold,           config.max_angle_error,
                                  config.min_lightbar_ratio,  config.max_lightbar_ratio,
                                  config.min_lightbar_length, config.min_armor_ratio,
                                  config.max_armor_ratio,     config.max_side_ratio,
                                  config.min_confidence,      config.max_rectangular_error};
        }
    };

    /**
     * @brief 传统装甲板检测器
     *
     * 移植自 sp_vision_25/tasks/auto_aim/detector
     * 功能：
     * 1. 全图检测：灰度化→二值化→轮廓检测→灯条筛选→配对装甲板→去重
     * 2. ROI精修：在已知装甲板位置的基础上，精确定位角点
     * 3. PCA角点修正（可选）
     */
    class TraditionalDetector {
    public:
        explicit TraditionalDetector(const SpVisionConfig &config, bool debug = false);

        /**
         * @brief 全图检测装甲板
         * @param bgr_img BGR彩色图像
         * @param binary_img
         * @param frame_id 帧ID
         * @param timestamp_ns 时间戳
         * @return ArmorSlots 检测结果
         */
        spike::ArmorSlots detect(const cv::Mat &bgr_img, const cv::Mat &binary_img, uint64_t frame_id,
                                 uint64_t timestamp_ns);

        /**
         * @brief 在已有装甲板基础上精修角点 (ROI检测)
         * @param armor 输入/输出装甲板，points会被更新
         * @param bgr_img BGR彩色图像
         * @return 是否成功精修
         */
        bool refine(spike::ArmorFull &armor, const cv::Mat &bgr_img) const;

        /**
         * @brief 获取内部检测到的完整装甲板列表 (用于后续处理)
         */
        [[nodiscard]] const std::list<spike::ArmorFull> &get_armors() const { return armors_; }

        /**
         * @brief 获取内部检测到的完整灯条列表 (用于调试)
         */
        [[nodiscard]] const std::list<spike::LightbarFull> &get_lightbars() const { return lightbars_; }

        /**
         * @brief 从图像和装甲板灯条信息提取图案 ROI
         * @param bgr_img BGR彩色图像
         * @param armor 装甲板 (需要 left/right 灯条信息)
         * @return 提取的图案 ROI
         */
        [[nodiscard]] static cv::Mat get_pattern(const cv::Mat &bgr_img, const spike::ArmorFull &armor);

    private:
        DetectorConfig config_;
        Classifier classifier_;
        bool debug_;

        // 内部状态 (每次detect后更新)
        std::list<spike::LightbarFull> lightbars_;
        std::list<spike::ArmorFull> armors_;

        // 配置值 (预转换为弧度)
        double max_angle_error_rad_;
        double max_rectangular_error_rad_;

        // ============ 几何检查 ============
        [[nodiscard]] bool check_geometry(const spike::LightbarFull &lightbar) const;
        [[nodiscard]] bool check_geometry(const spike::ArmorFull &armor) const;
        [[nodiscard]] bool check_name(const spike::ArmorFull &armor) const;
        [[nodiscard]] static bool check_type(const spike::ArmorFull &armor);

        // ============ 属性获取 ============
        [[nodiscard]] static spike::Color get_color(const cv::Mat &bgr_img, const std::vector<cv::Point> &contour);
        [[nodiscard]] static spike::ArmorType get_type(const spike::ArmorFull &armor);
        [[nodiscard]] static cv::Point2f get_center_norm(const cv::Mat &bgr_img, const cv::Point2f &center) ;

        // ============ 去重处理 ============
        void remove_duplicates();

        // ============ 结果转换 ============
        [[nodiscard]] spike::ArmorSlots to_armor_slots(uint64_t frame_id, uint64_t timestamp_ns) const;

        // ============ PCA角点修正 (可选) ============
        static void lightbar_points_corrector(spike::LightbarFull &lightbar, const cv::Mat &gray_img);

        // ============ 调试输出 ============
        void show_result(const cv::Mat &binary_img, const cv::Mat &bgr_img, int frame_count) const;
    };

} // namespace auto_aim
