#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "sp_vision/config.hpp"

namespace talos {
    struct ArmorFull;
}

namespace auto_aim {

    /**
     * @brief 装甲板数字分类器
     *
     * 使用 ONNX 模型对装甲板图案进行分类，识别数字 1-5、哨兵、前哨站、基地
     * 支持 OpenCV DNN 和 OpenVINO 两种推理后端
     */
    class Classifier {
    public:
        explicit Classifier(const SpVisionClassifierConfig &config);

        /**
         * @brief 对装甲板进行分类 (使用 OpenCV DNN)
         * @param armor 装甲板 (pattern 必须已填充)
         * 结果会更新 armor.name 和 armor.confidence
         */
        void classify(talos::ArmorFull &armor);

        /**
         * @brief 对装甲板进行分类 (使用 OpenVINO)
         * @param armor 装甲板 (pattern 必须已填充)
         * 结果会更新 armor.name 和 armor.confidence
         */
        void classify_ov(talos::ArmorFull &armor);

        /**
         * @brief 从图像和装甲板角点提取图案
         * @param bgr_img BGR 图像
         * @param armor 装甲板 (需要 left/right 灯条信息)
         * @return 提取的图案 ROI
         */
        static cv::Mat extract_pattern(const cv::Mat &bgr_img, const talos::ArmorFull &armor);

    private:
        SpVisionClassifierConfig config_;
        cv::dnn::Net net_;
        ov::Core core_;
        ov::CompiledModel compiled_model_;
        bool ov_initialized_{false};

        /**
         * @brief 预处理图像到 32x32 灰度归一化输入
         * @param pattern BGR 图案
         * @param output 输出 32x32 CV_32F 归一化图像
         * @return 是否成功
         */
        static bool preprocess(const cv::Mat &pattern, cv::Mat &output);
    };

} // namespace auto_aim
