#include "L2_perception/armor/backends/at_legacy_nn_backend.hpp"

#include "L2_perception/armor/at_legacy_config.hpp"
#include "L2_perception/common/at_legacy_geometry.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <Eigen/Core>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// PIMPL Implementation
// ============================================================================

struct AtLegacyNNBackend::Impl {
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled;
    ov::InferRequest request;

    bool loaded = false;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

AtLegacyNNBackend::AtLegacyNNBackend(Config config)
    : config_{std::move(config)}
    , impl_(std::make_unique<Impl>()) {}

AtLegacyNNBackend::~AtLegacyNNBackend() = default;

AtLegacyNNBackend::AtLegacyNNBackend(AtLegacyNNBackend&& other) noexcept
    : config_{std::move(other.config_)}
    , detect_color_(other.detect_color_)
    , initialized_(other.initialized_)
    , impl_(std::move(other.impl_)) {
    other.initialized_ = false;
}

AtLegacyNNBackend& AtLegacyNNBackend::operator=(AtLegacyNNBackend&& other) noexcept {
    if (this != &other) {
        config_            = std::move(other.config_);
        detect_color_      = other.detect_color_;
        initialized_       = other.initialized_;
        impl_              = std::move(other.impl_);
        other.initialized_ = false;
    }
    return *this;
}

// ============================================================================
// Initialization
// ============================================================================

std::expected<void, DetectorError> AtLegacyNNBackend::init() {
    if (initialized_) {
        return {};
    }

    const auto& cfg = get_config();

    try {
        // Enable profiling if requested
        if (cfg.enable_profiling) {
            impl_->core.set_property(cfg.device, ov::enable_profiling(true));
        }

        // Load model
        impl_->model = impl_->core.read_model(cfg.model_path);
        impl_->model->reshape({1, 3, INPUT_H, INPUT_W});

        // Configure preprocessing
        ov::preprocess::PrePostProcessor ppp(impl_->model);
        ppp.input().tensor().set_element_type(ov::element::f32).set_layout("NCHW");
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);
        impl_->model = ppp.build();

        // Compile options
        ov::AnyMap compile_opts;
        compile_opts.emplace(
            ov::hint::performance_mode.name(), cfg.enable_multi_thread
                                                   ? ov::hint::PerformanceMode::THROUGHPUT
                                                   : ov::hint::PerformanceMode::LATENCY);

        if (cfg.enable_multi_thread) {
            compile_opts.emplace(ov::num_streams.name(), ov::streams::AUTO);
        } else {
            compile_opts.emplace(ov::num_streams.name(), 1);
        }

        // Set device priorities for AUTO/MULTI/HETERO
        const auto& device = cfg.device;
        if (!cfg.device_priorities.empty()
            && (device == "AUTO" || device.rfind("AUTO:", 0) == 0 || device == "MULTI"
                || device.rfind("MULTI:", 0) == 0 || device == "HETERO"
                || device.rfind("HETERO:", 0) == 0)) {
            compile_opts.emplace(ov::device::priorities.name(), cfg.device_priorities);
        }

        // Compile model
        impl_->compiled = impl_->core.compile_model(impl_->model, cfg.device, compile_opts);
        impl_->request  = impl_->compiled.create_infer_request();
        impl_->loaded   = true;
        initialized_    = true;

        SPDLOG_INFO("[AtLegacyNNBackend] Model loaded: {}", cfg.model_path);
        const auto devices = execution_devices();
        if (!devices.empty()) {
            // Reserve upfront to avoid reallocations (estimate: ~12 chars per device)
            std::string dev_str;
            dev_str.reserve(devices.size() * 12);
            bool first = true;
            for (const auto& d : devices) {
                if (!first) {
                    dev_str += ", ";
                }
                dev_str += d;
                first = false;
            }
            SPDLOG_INFO("[AtLegacyNNBackend] Execution devices: {}", dev_str);
        }

        return {};
    } catch (const std::exception& e) {
        SPDLOG_ERROR("[AtLegacyNNBackend] Init failed: {}", e.what());
        return std::unexpected(DetectorError::InitFailed);
    }
}

// ============================================================================
// Preprocessing
// ============================================================================

AtLegacyNNBackend::PreprocContext
    AtLegacyNNBackend::preprocess(const cv::Mat& image, const AtLegacyNNConfig& /*cfg*/) const {
    PreprocContext ctx;
    ctx.orig_w = image.cols;
    ctx.orig_h = image.rows;

    // Get input shape from model
    auto ip         = impl_->compiled.input();
    const auto& shp = ip.get_shape(); // [1, 3, H, W]

    if (shp.size() != 4) {
        SPDLOG_ERROR("[AtLegacyNNBackend] Input shape {} not supported", shp.size());
        return ctx;
    }
    if (shp[0] != 1 || shp[1] != 3) {
        SPDLOG_ERROR(
            "[AtLegacyNNBackend] Input shape mismatch, get [{},{},{},{}], expect [{},{},{},{}]",
            shp[0], shp[1], shp[2], shp[3], 1, 3, INPUT_H, INPUT_W);
        return ctx;
    }

    // Letterbox resize with symmetric padding
    const float r = std::min(
        static_cast<float>(INPUT_W) / static_cast<float>(image.cols),
        static_cast<float>(INPUT_H) / static_cast<float>(image.rows));

    const int unpad_w = static_cast<int>(r * static_cast<float>(image.cols));
    const int unpad_h = static_cast<int>(r * static_cast<float>(image.rows));

    int dw = INPUT_W - unpad_w;
    int dh = INPUT_H - unpad_h;
    dw /= 2;
    dh /= 2;

    // Store inverse transform parameters
    ctx.scale_x = 1.0f / r;
    ctx.scale_y = 1.0f / r;
    ctx.pad_x   = static_cast<float>(dw);
    ctx.pad_y   = static_cast<float>(dh);

    // Resize and pad
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(unpad_w, unpad_h));
    cv::copyMakeBorder(resized, resized, dh, dh, dw, dw, cv::BORDER_CONSTANT);

    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
    ctx.preprocessed = resized;

    return ctx;
}

// ============================================================================
// Inference
// ============================================================================

bool AtLegacyNNBackend::infer(const PreprocContext& ctx) {
    if (!impl_->loaded || ctx.preprocessed.empty()) {
        return false;
    }

    try {
        auto ip         = impl_->compiled.input();
        const auto& shp = ip.get_shape();

        // Split channels (HWC -> CHW)
        cv::Mat ch[3];
        cv::split(ctx.preprocessed, ch);

        // Create input tensor
        ov::Tensor in(ip.get_element_type(), shp);
        float* dst         = in.data<float>();
        const size_t plane = static_cast<size_t>(INPUT_W) * static_cast<size_t>(INPUT_H);

        std::memcpy(dst + plane * 0, ch[2].data, plane * sizeof(float));
        std::memcpy(dst + plane * 1, ch[1].data, plane * sizeof(float));
        std::memcpy(dst + plane * 2, ch[0].data, plane * sizeof(float));

        impl_->request.set_input_tensor(in);
        impl_->request.infer();
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("[AtLegacyNNBackend] Inference failed: {}", e.what());
        return false;
    }
}

// ============================================================================
// Postprocessing
// ============================================================================

std::vector<AtLegacyArmorDetection>
    AtLegacyNNBackend::postprocess(const PreprocContext& ctx, const AtLegacyNNConfig& cfg) const {
    std::vector<AtLegacyArmorDetection> detections;

    ov::Tensor out = impl_->request.get_output_tensor(0);
    const float* p = out.data<const float>();
    if (!p)
        return detections;

    const auto& shape = out.get_shape();

    // YOLO26 end-to-end output: [1, max_det, 6 + nk]
    // Format per detection: [x, y, w, h, conf, class_id, kpts...]
    if (shape.size() != 3 || shape[0] != 1) {
        SPDLOG_WARN("[AtLegacyNNBackend] Unexpected output shape");
        return detections;
    }

    const int max_det    = static_cast<int>(shape[1]);
    const int output_dim = static_cast<int>(shape[2]);

    constexpr int nkpt     = NUM_KPTS;       // 4 keypoints
    constexpr int kpt_dim  = 2;              // x, y per keypoint
    constexpr int nk       = nkpt * kpt_dim; // 8
    constexpr int bbox_dim = 6;              // x, y, w, h, conf, class_id

    SPDLOG_DEBUG(
        "[AtLegacyNNBackend] Output shape: [{}, {}, {}], max_det={}, output_dim={}", shape[0],
        shape[1], shape[2], max_det, output_dim);

    if (output_dim < bbox_dim + nk) {
        SPDLOG_ERROR(
            "[AtLegacyNNBackend] Output dim {} < bbox_dim({}) + nkpts({})", output_dim, bbox_dim,
            nk);
        return detections;
    }

    auto clampf = [](float v, float lo, float hi) -> float {
        return std::max(lo, std::min(v, hi));
    };

    // Scale coordinate from model input (letterboxed) to original image
    // Following ops.scale_coords from Ultralytics
    auto scale_coords = [&](float x, float y) -> cv::Point2f {
        // Inverse letterbox: (coord - pad) * scale
        float px = (x - ctx.pad_x) * ctx.scale_x;
        float py = (y - ctx.pad_y) * ctx.scale_y;
        if (ctx.orig_w > 0 && ctx.orig_h > 0) {
            px = clampf(px, 0.0f, static_cast<float>(ctx.orig_w - 1));
            py = clampf(py, 0.0f, static_cast<float>(ctx.orig_h - 1));
        }
        return cv::Point2f(px, py);
    };

    auto num_id_to_name = [](int num_id) -> ArmorName {
        switch (num_id) {
        case 0: return ArmorName::Sentry;  // G
        case 1: return ArmorName::One;
        case 2: return ArmorName::Two;
        case 3: return ArmorName::Three;
        case 4: return ArmorName::Four;
        case 5: return ArmorName::Five;
        case 6: return ArmorName::Outpost; // O
        case 7: return ArmorName::Base;    // B
        default: return ArmorName::Invalid;
        }
    };

    // Decode YOLO26 class_id (0-63) to (color, size, num)
    // Format: {color}{size}{num} - Bs0, Bb0, Rs0, Rb0, ...
    // 64 classes = 4 colors * 2 sizes * 8 numbers
    auto decode_yolo26_class = [](int cls_id, int& color_id, int& size_id, int& num_id) {
        num_id   = cls_id % 8;
        int tmp  = cls_id / 8;
        size_id  = tmp % 2; // 0=small, 1=big
        color_id = tmp / 2; // 0=Blue, 1=Red, 2=Gray, 3=Purple
    };

    detections.reserve(128);
    const float conf_thresh = static_cast<float>(cfg.confidence_threshold);

    for (int i = 0; i < max_det; ++i) {
        const size_t row_offset = static_cast<size_t>(i) * output_dim;

        // Extract confidence and class_id
        const float conf = p[row_offset + 4];
        const int cls_id = static_cast<int>(p[row_offset + 5]);

        if (conf < conf_thresh) {
            continue;
        }

        // Decode class
        int color_id = -1, size_id = -1, num_id = -1;
        decode_yolo26_class(cls_id, color_id, size_id, num_id);

        if (color_id < 0 || color_id >= 4 || size_id < 0 || size_id >= 2) {
            continue;
        }

        ArmorName cls = num_id_to_name(num_id);
        if (cls == ArmorName::Invalid) {
            continue;
        }
        if (cls == ArmorName::Base && size_id == 1) {
            cls = ArmorName::BaseLarge;
        }
        const auto color = static_cast<ArmorColor>(color_id);

        // Extract keypoints (flattened: kx1, ky1, kx2, ky2, ...)
        // YOLO26 outputs keypoints in order: TL, BL, BR, TR (or similar)
        // We need to reorder to: TL, TR, BR, BL to match model_points in PnP solver
        std::array<cv::Point2f, 4> corners;
        bool kp_valid = true;

        // YOLO26 keypoint order to desired order (TL, TR, BR, BL)
        // Based on YOLO pose convention: 0=TL, 1=BL, 2=BR, 3=TR
        // We want: 0=TL, 1=TR, 2=BR, 3=BL
        constexpr int yolo_to_model[4] = {0, 3, 2, 1};

        for (int k = 0; k < nkpt; ++k) {
            const int yolo_idx     = yolo_to_model[k];
            const size_t kx_offset = row_offset + bbox_dim + 2 * yolo_idx;
            const size_t ky_offset = row_offset + bbox_dim + 2 * yolo_idx + 1;

            // Bounds check
            const size_t total_elements =
                static_cast<size_t>(max_det) * static_cast<size_t>(output_dim);
            if (ky_offset >= total_elements) {
                SPDLOG_ERROR(
                    "[AtLegacyNNBackend] Keypoint access out of bounds: "
                    "ky_offset={} >= total_elements={}, i={}, k={}, yolo_idx={}, "
                    "row_offset={}, bbox_dim={}, output_dim={}",
                    ky_offset, total_elements, i, k, yolo_idx, row_offset, bbox_dim, output_dim);
                kp_valid = false;
                break;
            }

            const float kx = p[kx_offset];
            const float ky = p[ky_offset];

            if (!std::isfinite(kx) || !std::isfinite(ky)) {
                kp_valid = false;
                break;
            }

            corners[k] = scale_coords(kx, ky);
        }

        if (!kp_valid) {
            continue;
        }

        // Optional: verify keypoints are within image bounds
        bool all_in_bounds = true;
        for (const auto& pt : corners) {
            if (pt.x < 0 || pt.x >= ctx.orig_w || pt.y < 0 || pt.y >= ctx.orig_h) {
                all_in_bounds = false;
                break;
            }
        }
        if (!all_in_bounds) {
            continue;
        }

        // Create detection
        AtLegacyArmorDetection det(corners, cls, color, conf);
        if (det.area > 0) {
            detections.push_back(det);
        }
    }

    return detections;
}

// ============================================================================
// Coordinate Transform
// ============================================================================

void AtLegacyNNBackend::transform_coordinates(
    std::vector<AtLegacyArmorDetection>& detections, const PreprocContext& ctx) const {
    // Already done in postprocess via map_point
    (void)detections;
    (void)ctx;
}

// ============================================================================
// NMS
// ============================================================================

std::vector<AtLegacyArmorDetection> AtLegacyNNBackend::nms(
    std::vector<AtLegacyArmorDetection>& detections, const AtLegacyNNConfig& cfg) const {
    if (detections.empty()) {
        return {};
    }

    // Sort by confidence (descending)
    std::sort(
        detections.begin(), detections.end(),
        [](const AtLegacyArmorDetection& a, const AtLegacyArmorDetection& b) {
            return a.confidence > b.confidence;
        });

    // Keep top K
    if (static_cast<int>(detections.size()) > cfg.top_k) {
        detections.resize(cfg.top_k);
    }

    // NMS
    const float nms_thresh = static_cast<float>(cfg.nms_threshold);
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<AtLegacyArmorDetection> result;
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i])
            continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j])
                continue;

            if (at_legacy::detail::iou(detections[i].rect, detections[j].rect) > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// ============================================================================
// Detection Entry Point
// ============================================================================

AtLegacyNNBackend::DetectionResult AtLegacyNNBackend::detect_impl(const cv::Mat& image) {
    if (!initialized_) {
        return std::unexpected(DetectorError::NotInitialized);
    }

    if (image.empty()) {
        return std::unexpected(DetectorError::InvalidImage);
    }

    const auto& cfg = get_config();

    // Preprocess
    auto ctx = preprocess(image, cfg);
    if (ctx.preprocessed.empty()) {
        return std::unexpected(DetectorError::PreprocessFailed);
    }

    // Infer
    if (!infer(ctx)) {
        return std::unexpected(DetectorError::InferenceFailed);
    }

    // Postprocess
    auto detections = postprocess(ctx, cfg);

    // NMS
    auto result = nms(detections, cfg);

    // Debug: print detection results
    // SPDLOG_INFO("[AtLegacyNNBackend] ===== Detection Results =====");
    // SPDLOG_INFO("[AtLegacyNNBackend] Total detections: {}", result.size());
    // for (size_t i = 0; i < result.size(); ++i) {
    //     const auto& det = result[i];
    //     const auto center = det.center();

    //     SPDLOG_INFO("[AtLegacyNNBackend] [{}] {} {} | conf={:.3f} | area={}",
    //         i,
    //         to_string(det.name),
    //         to_string(det.color),
    //         det.confidence,
    //         det.area);

    //     SPDLOG_INFO("[AtLegacyNNBackend]     Center: ({:.1f}, {:.1f})", center.x, center.y);

    //     SPDLOG_INFO("[AtLegacyNNBackend]     Corners: TL=({:.1f},{:.1f}) TR=({:.1f},{:.1f})
    //     BR=({:.1f},{:.1f}) BL=({:.1f},{:.1f})",
    //         det.corners[0].x, det.corners[0].y,
    //         det.corners[1].x, det.corners[1].y,
    //         det.corners[2].x, det.corners[2].y,
    //         det.corners[3].x, det.corners[3].y);

    //     SPDLOG_INFO("[AtLegacyNNBackend]     BBox: x={:.1f} y={:.1f} w={:.1f} h={:.1f}",
    //         det.rect.x, det.rect.y, det.rect.width, det.rect.height);
    // }
    // SPDLOG_INFO("[AtLegacyNNBackend] =================================");

    return result;
}

// ============================================================================
// Utility
// ============================================================================

std::vector<std::string> AtLegacyNNBackend::execution_devices() const {
    if (!impl_->loaded) {
        return {};
    }
    try {
        return impl_->compiled.get_property(ov::execution_devices);
    } catch (const std::exception& e) {
        SPDLOG_WARN("[NNBackend] Failed to get execution devices: {}", e.what());
        return {};
    } catch (...) {
        SPDLOG_WARN("[NNBackend] Failed to get execution devices: unknown error");
        return {};
    }
}

} // namespace fcs::L2
// mod
