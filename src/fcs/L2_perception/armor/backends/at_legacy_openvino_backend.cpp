#include "at_legacy_openvino_backend.hpp"
#include "../at_legacy_config.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// Constructor/Destructor
// ============================================================================

template <int INPUT_W, int INPUT_H>
AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::AtLegacyOpenVINOBackend(ConfigPtr config)
    : config_(std::move(config)) {}

template <int INPUT_W, int INPUT_H>
AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::~AtLegacyOpenVINOBackend() {
    model_loaded_ = false;
    initialized_  = false;
}

// ============================================================================
// Initialization
// ============================================================================

template <int INPUT_W, int INPUT_H>
std::expected<void, DetectorError> AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::init() {
    if (!config_) {
        return std::unexpected(DetectorError::InitFailed);
    }

    try {
        core_      = ov::Core();
        auto model = core_.read_model(config_->model_path);
        model->reshape({1, 3, INPUT_H, INPUT_W});

        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::f32).set_layout("NCHW");
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);
        model = ppp.build();

        ov::AnyMap compile_opts;
        compile_opts.emplace(
            ov::hint::performance_mode.name(), config_->enable_multi_thread
                                                   ? ov::hint::PerformanceMode::THROUGHPUT
                                                   : ov::hint::PerformanceMode::LATENCY);

        if (config_->enable_multi_thread) {
            compile_opts.emplace(ov::num_streams.name(), ov::streams::AUTO);
        }

        compiled_ = core_.compile_model(model, config_->device, compile_opts);
        request_  = compiled_.create_infer_request();

        model_loaded_ = true;
        initialized_  = true;

        SPDLOG_INFO("[OpenVINO<{},{}>] Model loaded: {}", INPUT_W, INPUT_H, config_->model_path);

        const auto devices = execution_devices();
        if (!devices.empty()) {
            std::string dev_str;
            for (size_t i = 0; i < devices.size(); ++i) {
                if (i > 0)
                    dev_str += ", ";
                dev_str += devices[i];
            }
            SPDLOG_INFO("[OpenVINO<{},{}>] Execution devices: {}", INPUT_W, INPUT_H, dev_str);
        }

        return {};

    } catch (const std::exception& e) {
        SPDLOG_ERROR("[OpenVINO<{},{}>] Init failed: {}", INPUT_W, INPUT_H, e.what());
        return std::unexpected(DetectorError::ModelLoadError);
    }
}

template <int INPUT_W, int INPUT_H>
std::vector<std::string> AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::execution_devices() const {
    if (!model_loaded_) {
        return {};
    }
    try {
        return compiled_.get_property(ov::execution_devices);
    } catch (const std::exception& e) {
        SPDLOG_WARN(
            "[OpenVINO<{},{}>] Failed to get execution devices: {}", INPUT_W, INPUT_H, e.what());
        return {};
    } catch (...) {
        SPDLOG_WARN(
            "[OpenVINO<{},{}>] Failed to get execution devices: unknown error", INPUT_W, INPUT_H);
        return {};
    }
}

// ============================================================================
// Preprocess
// ============================================================================

template <int INPUT_W, int INPUT_H>
void* AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::preprocess_impl(const cv::Mat& image) {
    if (!config_) {
        return nullptr;
    }
    if (!config_) {
        return nullptr;
    }

    // Allocate new context
    auto ctx = std::make_unique<PreprocessContext>();

    // Store original dimensions
    ctx->orig_w = image.cols;
    ctx->orig_h = image.rows;

    // Letterbox resize
    float scale = std::min(
        static_cast<float>(INPUT_W) / image.cols, static_cast<float>(INPUT_H) / image.rows);
    int new_w = static_cast<int>(image.cols * scale);
    int new_h = static_cast<int>(image.rows * scale);

    ctx->scale_x = scale;
    ctx->scale_y = scale;
    ctx->pad_x   = (INPUT_W - new_w) / 2;
    ctx->pad_y   = (INPUT_H - new_h) / 2;

    // Resize and pad
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    ctx->preprocessed = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
    resized.copyTo(ctx->preprocessed(cv::Rect(ctx->pad_x, ctx->pad_y, new_w, new_h)));

    // Convert HWC to CHW and normalize to [0, 1]
    std::vector<cv::Mat> channels(3);
    cv::split(ctx->preprocessed, channels);
    for (auto& ch : channels) {
        ch.convertTo(ch, CV_32F, 1.0 / 255.0);
    }
    cv::merge(channels, ctx->preprocessed);

    // Store context for later use
    context_ = std::move(ctx);
    return context_.get();
}

// ============================================================================
// Infer
// ============================================================================

template <int INPUT_W, int INPUT_H>
bool AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::infer_impl(void* ctx) {
    if (!model_loaded_ || !ctx) {
        return false;
    }

    auto* context = static_cast<PreprocessContext*>(ctx);

    try {
        // Get input tensor
        auto input_port   = compiled_.input();
        const auto& shape = input_port.get_shape();

        // Convert HWC to CHW
        std::vector<cv::Mat> channels(3);
        cv::split(context->preprocessed, channels);

        // Create input tensor
        ov::Tensor input_tensor(input_port.get_element_type(), shape);
        float* data = input_tensor.data<float>();

        const size_t plane_size = INPUT_W * INPUT_H;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(data + c * plane_size, channels[c].data, plane_size * sizeof(float));
        }

        // Set input and infer
        request_.set_input_tensor(input_tensor);
        request_.infer();

        return true;

    } catch (const std::exception& e) {
        SPDLOG_ERROR("[OpenVINO<{},{}>] Inference failed: {}", INPUT_W, INPUT_H, e.what());
        return false;
    }
}

// ============================================================================
// Postprocess
// ============================================================================

template <int INPUT_W, int INPUT_H>
std::vector<AtLegacyArmorDetection>
    AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::postprocess_impl(void* ctx) {
    std::vector<AtLegacyArmorDetection> detections;

    if (!ctx) {
        return detections;
    }

    auto* context = static_cast<PreprocessContext*>(ctx);
    if (!config_) {
        return detections;
    }

    try {
        // Get output tensor
        ov::Tensor output = request_.get_output_tensor(0);
        const float* data = output.data<const float>();
        if (!data) {
            return detections;
        }

        const auto& shape = output.get_shape();

        // YOLO format: [1, max_det, 6 + nk]
        // Format: [x, y, w, h, conf, class_id, kpts...]
        if (shape.size() != 3 || shape[0] != 1) {
            SPDLOG_WARN("[OpenVINO<{},{}>] Unexpected output shape", INPUT_W, INPUT_H);
            return detections;
        }

        const int max_det    = shape[1];
        const int output_dim = shape[2];

        for (int i = 0; i < max_det; ++i) {
            const float* det = data + i * output_dim;

            float conf = det[4];
            if (conf < config_->confidence_threshold) {
                continue;
            }

            AtLegacyArmorDetection detection;
            detection.confidence = conf;

            // Bbox center coordinates
            float cx = det[0];
            float cy = det[1];
            float w  = det[2];
            float h  = det[3];

            // Convert from letterbox space to original image space
            cx = (cx - context->pad_x) / context->scale_x;
            cy = (cy - context->pad_y) / context->scale_y;
            w /= context->scale_x;
            h /= context->scale_y;

            // Store as corners
            detection.corners[0] = {cx - w / 2, cy - h / 2}; // Top-left
            detection.corners[1] = {cx + w / 2, cy - h / 2}; // Top-right
            detection.corners[2] = {cx + w / 2, cy + h / 2}; // Bottom-right
            detection.corners[3] = {cx - w / 2, cy + h / 2}; // Bottom-left

            // Class ID (TODO: Map to ArmorName and ArmorColor based on model output)
            // int class_id = static_cast<int>(det[5]);

            // Create detection with placeholder type/color
            // TODO: Implement proper class_id to ArmorName/ArmorColor mapping

            detection.rect = cv::Rect(
                static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2), static_cast<int>(w),
                static_cast<int>(h));

            detections.push_back(detection);
        }

        return detections;

    } catch (const std::exception& e) {
        SPDLOG_ERROR("[OpenVINO<{},{}>] Postprocess failed: {}", INPUT_W, INPUT_H, e.what());
        return detections;
    }
}

// ============================================================================
// NMS
// ============================================================================

template <int INPUT_W, int INPUT_H>
std::vector<AtLegacyArmorDetection> AtLegacyOpenVINOBackend<INPUT_W, INPUT_H>::nms_impl(
    std::vector<AtLegacyArmorDetection> detections) {
    if (!config_ || detections.empty()) {
        return detections;
    }

    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<AtLegacyArmorDetection> result;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i])
            continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j])
                continue;

            // Calculate IoU
            cv::Rect box_i = detections[i].rect;
            cv::Rect box_j = detections[j].rect;

            cv::Rect intersection = box_i & box_j;
            cv::Rect union_box    = box_i | box_j;

            float iou =
                static_cast<float>(intersection.area()) / static_cast<float>(union_box.area());

            if (iou > config_->nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class AtLegacyOpenVINOBackend<640, 640>;
template class AtLegacyOpenVINOBackend<416, 416>;

} // namespace fcs::L2
