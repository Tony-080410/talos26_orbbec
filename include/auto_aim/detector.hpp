#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
#include "classifier.hpp"
#include "config.hpp"
#include "detector_state.hpp"

namespace talos {

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

} // namespace talos

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
     * @brief 检测器工作缓冲区 (每帧复用)
     *
     * 由调度器/策略持有，传递给Detector方法
     */
    struct DetectorWorkBuffers {
        std::list<talos::LightbarFull> lightbars{};
        std::list<talos::ArmorFull> armors{};

        void clear() {
            lightbars.clear();
            armors.clear();
        }
    };

    /**
     * @brief 传统装甲板检测器 (无状态)
     *
     * 移植自 sp_vision_25/tasks/auto_aim/detector
     * 功能：
     * 1. 全图检测：灰度化→二值化→轮廓检测→灯条筛选→配对装甲板→去重
     * 2. ROI精修：在已知装甲板位置的基础上，精确定位角点
     * 3. PCA角点修正（可选）
     *
     * 所有工作缓冲区通过 DetectorWorkBuffers 参数传递
     */
    class TraditionalDetector {
    public:
        explicit TraditionalDetector(const SpVisionConfig &config, bool debug = false);
        explicit TraditionalDetector(const SpVisionConfig &config, Classifier classifier);

        void preprocess(const cv::Mat &bgr_img, const cv::Mat &binary_img) const noexcept;

        /**
         * @brief 全图检测装甲板
         * @param buffers 工作缓冲区 (输入/输出)
         * @param bgr_img BGR彩色图像
         * @param binary_img 二值化图像
         * @param frame_id 帧ID
         * @param timestamp_ns 时间戳
         * @return ArmorSlots 检测结果
         */
        talos::ArmorSlots detect(DetectorWorkBuffers &buffers, const cv::Mat &bgr_img, const cv::Mat &binary_img,
                                 uint64_t frame_id, uint64_t timestamp_ns) const;

        /**
         * @brief 在已有装甲板基础上精修角点 (ROI检测)
         * @param armor 输入/输出装甲板，points会被更新
         * @param bgr_img BGR彩色图像
         * @return 是否成功精修
         */
        [[nodiscard]] bool refine(talos::ArmorFull &armor, const cv::Mat &bgr_img) const;

        /**
         * @brief 从图像和装甲板灯条信息提取图案 ROI
         * @param bgr_img BGR彩色图像
         * @param armor 装甲板 (需要 left/right 灯条信息)
         * @return 提取的图案 ROI
         */
        [[nodiscard]] static cv::Mat get_pattern(const cv::Mat &bgr_img, const talos::ArmorFull &armor);

    private:
        DetectorConfig config_;
        Classifier classifier_;
        bool debug_;

        // 配置值 (预转换为弧度)
        double max_angle_error_rad_;
        double max_rectangular_error_rad_;

        // ============ 几何检查 ============
        [[nodiscard]] bool check_geometry(const talos::LightbarFull &lightbar) const;
        [[nodiscard]] bool check_geometry(const talos::ArmorFull &armor) const;
        [[nodiscard]] bool check_name(const talos::ArmorFull &armor) const;
        [[nodiscard]] static bool check_type(const talos::ArmorFull &armor);

        // ============ 属性获取 ============
        [[nodiscard]] static talos::Color get_color(const cv::Mat &bgr_img, const std::vector<cv::Point> &contour);
        [[nodiscard]] static talos::ArmorType get_type(const talos::ArmorFull &armor);
        [[nodiscard]] static cv::Point2f get_center_norm(const cv::Mat &bgr_img, const cv::Point2f &center);

        // ============ 去重处理 ============
        static void remove_duplicates(DetectorWorkBuffers &buffers);

        // ============ 结果转换 ============
        [[nodiscard]] static talos::ArmorSlots to_armor_slots(const DetectorWorkBuffers &buffers, uint64_t frame_id,
                                                              uint64_t timestamp_ns);

        // ============ PCA角点修正 (可选) ============
        static void lightbar_points_corrector(talos::LightbarFull &lightbar, const cv::Mat &gray_img);

        // ============ 调试输出 ============
        void show_result(const DetectorWorkBuffers &buffers, const cv::Mat &binary_img, const cv::Mat &bgr_img,
                         int frame_count) const;
    };

} // namespace auto_aim
