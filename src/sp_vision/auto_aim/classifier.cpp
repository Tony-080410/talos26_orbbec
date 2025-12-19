#include "auto_aim/classifier.hpp"
#include "auto_aim/detector.hpp"

#include <algorithm>
#include <utility>

namespace auto_aim {

    Classifier::Classifier(const SpVisionClassifierConfig &config) {
        // 加载 OpenCV DNN 模型

        // 加载 OpenVINO 模型 (如果启用)
        if (config_.use_openvino) {
            try {
                std::cout << "try openvino classifier" << std::endl;
                const auto ov_model = core_.read_model(config.model_path);
                std::cout << "compiling openvino classifier" << std::endl;
                compiled_model_ = core_.compile_model(ov_model, config_.device_type(),
                                                      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
                ov_initialized_ = true;
            } catch (const std::exception &e) {
                std::cerr << "error initializing openvino" << std::endl;
                std::cerr << e.what() << std::endl;
                // OpenVINO 初始化失败，回退到 OpenCV DNN
                ov_initialized_ = false;
            }
        }
        if (!ov_initialized_) {
            spdlog::warn("using cv::dnn as fallback");
            net_ = cv::dnn::readNetFromONNX(config.model_path);
        }
    }

    bool Classifier::preprocess(const cv::Mat &pattern, cv::Mat &output) {
        if (pattern.empty()) {
            return false;
        }

        // 转灰度
        cv::Mat gray;
        if (pattern.channels() == 3) {
            cv::cvtColor(pattern, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = pattern;
        }

        // 等比例缩放到 32x32，保持宽高比
        output = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
        const auto x_scale = 32.0 / gray.cols;
        const auto y_scale = 32.0 / gray.rows;
        const auto scale = std::min(x_scale, y_scale);
        const auto h = static_cast<int>(gray.rows * scale);
        const auto w = static_cast<int>(gray.cols * scale);

        if (h == 0 || w == 0) {
            return false;
        }

        const auto roi = cv::Rect(0, 0, w, h);
        cv::resize(gray, output(roi), {w, h});

        return true;
    }

    void Classifier::classify(talos::ArmorFull &armor) {
        if (armor.pattern.empty()) {
            armor.name = talos::ArmorName::not_armor;
            return;
        }

        cv::Mat input;
        if (!preprocess(armor.pattern, input)) {
            armor.name = talos::ArmorName::not_armor;
            return;
        }

        // 创建 blob: NCHW 格式, 归一化到 [0, 1]
        const auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

        net_.setInput(blob);
        cv::Mat outputs = net_.forward();

        // Softmax
        const float max_val = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::exp(outputs - max_val, outputs);
        const float sum = static_cast<float>(cv::sum(outputs)[0]);
        outputs /= sum;

        // 找最大值
        double confidence;
        cv::Point label_point;
        cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
        int label_id = label_point.x;

        armor.confidence = confidence;
        armor.name = static_cast<talos::ArmorName>(label_id);
    }

    void Classifier::classify_ov(talos::ArmorFull &armor) {
        if (!ov_initialized_) {
            // 回退到 OpenCV DNN
            classify(armor);
            return;
        }

        if (armor.pattern.empty()) {
            armor.name = talos::ArmorName::not_armor;
            return;
        }

        cv::Mat input;
        if (!preprocess(armor.pattern, input)) {
            armor.name = talos::ArmorName::not_armor;
            return;
        }

        // 归一化到 [0, 1]
        input.convertTo(input, CV_32F, 1.0 / 255.0);

        // 创建 OpenVINO tensor: NCHW = [1, 1, 32, 32]
        const ov::Tensor input_tensor(ov::element::f32, {1, 1, 32, 32}, input.data);

        ov::InferRequest infer_request = compiled_model_.create_infer_request();
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor();
        cv::Mat outputs(1, 9, CV_32F, output_tensor.data());

        // Softmax
        const float max_val = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::exp(outputs - max_val, outputs);
        const float sum = static_cast<float>(cv::sum(outputs)[0]);
        outputs /= sum;

        // 找最大值
        double confidence;
        cv::Point label_point;
        cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
        int label_id = label_point.x;

        armor.confidence = confidence;
        armor.name = static_cast<talos::ArmorName>(label_id);
    }

    cv::Mat Classifier::extract_pattern(const cv::Mat &bgr_img, const talos::ArmorFull &armor) {
        // 延长灯条获得装甲板角点
        // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
        const auto &left = armor.left;
        const auto &right = armor.right;

        const auto tl = left.center - left.top2bottom * 1.125f;
        const auto bl = left.center + left.top2bottom * 1.125f;
        const auto tr = right.center - right.top2bottom * 1.125f;
        const auto br = right.center + right.top2bottom * 1.125f;

        const auto roi_left = std::max<int>(static_cast<int>(std::min(tl.x, bl.x)), 0);
        const auto roi_top = std::max<int>(static_cast<int>(std::min(tl.y, tr.y)), 0);
        const auto roi_right = std::min<int>(static_cast<int>(std::max(tr.x, br.x)), bgr_img.cols);
        const auto roi_bottom = std::min<int>(static_cast<int>(std::max(bl.y, br.y)), bgr_img.rows);

        if (roi_right <= roi_left || roi_bottom <= roi_top) {
            return {};
        }

        const auto roi = cv::Rect(roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top);
        return bgr_img(roi).clone();
    }

} // namespace auto_aim
