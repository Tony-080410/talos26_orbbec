#include "fcs/L2_perception/at_legacy_nn_tup_backend.hpp"

#include "fcs/L2_perception/at_legacy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

namespace fcs {

// ============================================================================
// PIMPL Implementation
// ============================================================================

struct AtLegacyNNTupBackend::Impl {
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled;
    ov::InferRequest request;

    bool loaded = false;
};

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/// Calculate triangle area
[[nodiscard]] inline float
    tri_area(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) noexcept {
    return std::fabs(0.5f * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)));
}

/// Calculate quadrilateral area from 4 points
[[nodiscard]] inline float quad_area(const std::array<cv::Point2f, 4>& p) noexcept {
    return tri_area(p[0], p[1], p[2]) + tri_area(p[0], p[2], p[3]);
}

/// Calculate IoU between two rectangles
[[nodiscard]] inline float iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept {
    const float inter = (a & b).area();
    const float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

} // namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

AtLegacyNNTupBackend::AtLegacyNNTupBackend(ConfigPtr config)
    : config_(std::move(config))
    , impl_(std::make_unique<Impl>()) {}

AtLegacyNNTupBackend::~AtLegacyNNTupBackend() = default;

AtLegacyNNTupBackend::AtLegacyNNTupBackend(AtLegacyNNTupBackend&&) noexcept            = default;
AtLegacyNNTupBackend& AtLegacyNNTupBackend::operator=(AtLegacyNNTupBackend&&) noexcept = default;

// ============================================================================
// Initialization
// ============================================================================

std::expected<void, DetectorError> AtLegacyNNTupBackend::init() {
    if (initialized_) {
        return {};
    }

    auto cfg = get_config();
    if (!cfg) {
        spdlog::error("[AtLegacyNNTupBackend] Config is null");
        return std::unexpected(DetectorError::InitFailed);
    }

    try {
        // Enable profiling if requested
        if (cfg->enable_profiling) {
            impl_->core.set_property(cfg->device, ov::enable_profiling(true));
        }

        // Load model
        impl_->model = impl_->core.read_model(cfg->model_path);

        // Configure preprocessing
        ov::preprocess::PrePostProcessor ppp(impl_->model);
        ppp.input().tensor().set_element_type(ov::element::f32).set_layout("NCHW");
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);
        impl_->model = ppp.build();

        // Compile options
        ov::AnyMap compile_opts;
        compile_opts.emplace(
            ov::hint::performance_mode.name(), cfg->enable_multi_thread
                                                   ? ov::hint::PerformanceMode::THROUGHPUT
                                                   : ov::hint::PerformanceMode::LATENCY);

        if (cfg->enable_multi_thread) {
            compile_opts.emplace(ov::num_streams.name(), ov::streams::AUTO);
        } else {
            compile_opts.emplace(ov::num_streams.name(), 1);
        }

        // Set device priorities for AUTO/MULTI/HETERO
        const auto& device = cfg->device;
        if (!cfg->device_priorities.empty()
            && (device == "AUTO" || device.rfind("AUTO:", 0) == 0 || device == "MULTI"
                || device.rfind("MULTI:", 0) == 0 || device == "HETERO"
                || device.rfind("HETERO:", 0) == 0)) {
            compile_opts.emplace(ov::device::priorities.name(), cfg->device_priorities);
        }

        // Compile model
        impl_->compiled = impl_->core.compile_model(impl_->model, cfg->device, compile_opts);
        impl_->request  = impl_->compiled.create_infer_request();
        impl_->loaded   = true;
        initialized_    = true;

        spdlog::info("[AtLegacyNNTupBackend] Model loaded: {}", cfg->model_path);
        const auto devices = execution_devices();
        if (!devices.empty()) {
            std::string dev_str;
            for (const auto& d : devices) {
                if (!dev_str.empty())
                    dev_str += ", ";
                dev_str += d;
            }
            spdlog::info("[AtLegacyNNTupBackend] Execution devices: {}", dev_str);
        }

        return {};
    } catch (const std::exception& e) {
        spdlog::error("[AtLegacyNNTupBackend] Init failed: {}", e.what());
        return std::unexpected(DetectorError::InitFailed);
    }
}

// ============================================================================
// Grid Generation
// ============================================================================

void AtLegacyNNTupBackend::generate_grids_and_strides(
    int width, int height, const std::array<int, 3>& strides, std::vector<GridAndStride>& grids) {
    grids.clear();
    grids.reserve(
        (width / strides[0]) * (height / strides[0]) + (width / strides[1]) * (height / strides[1])
        + (width / strides[2]) * (height / strides[2]));

    for (int s : strides) {
        const int gw = width / s;
        const int gh = height / s;
        for (int g1 = 0; g1 < gh; ++g1) {
            for (int g0 = 0; g0 < gw; ++g0) {
                grids.push_back({g0, g1, s});
            }
        }
    }
}

// ============================================================================
// Preprocessing
// ============================================================================

AtLegacyNNTupBackend::PreprocContext
    AtLegacyNNTupBackend::preprocess(const cv::Mat& image, const AtLegacyNNTupConfig& cfg) const {
    PreprocContext ctx;

    const int input_w = cfg.input_width;
    const int input_h = cfg.input_height;

    // Letterbox resize with symmetric padding
    const float r = std::min(
        static_cast<float>(input_w) / static_cast<float>(image.cols),
        static_cast<float>(input_h) / static_cast<float>(image.rows));

    const int unpad_w = static_cast<int>(r * static_cast<float>(image.cols));
    const int unpad_h = static_cast<int>(r * static_cast<float>(image.rows));

    int dw = input_w - unpad_w;
    int dh = input_h - unpad_h;
    dw /= 2;
    dh /= 2;

    // Build transform matrix: network coordinates -> original image coordinates
    // T * [x_net, y_net, 1]^T = [x_orig, y_orig, 1]^T
    ctx.T << 1.f / r, 0.f, -static_cast<float>(dw) / r, 0.f, 1.f / r, -static_cast<float>(dh) / r,
        0.f, 0.f, 1.f;

    // Resize and pad
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(unpad_w, unpad_h));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, dh, dw, dw, cv::BORDER_CONSTANT);

    // Convert to float32 WITHOUT normalization (TUP model expects [0, 255])
    padded.convertTo(ctx.preprocessed, CV_32F);

    // Generate grids
    generate_grids_and_strides(input_w, input_h, cfg.strides, ctx.grids);
    ctx.rows        = static_cast<int>(ctx.grids.size());
    ctx.cols        = 9 + NUM_COLORS + cfg.num_classes;
    ctx.num_classes = cfg.num_classes;

    return ctx;
}

// ============================================================================
// Inference
// ============================================================================

bool AtLegacyNNTupBackend::infer(const PreprocContext& ctx) {
    if (!impl_->loaded || ctx.preprocessed.empty()) {
        return false;
    }

    try {
        auto ip           = impl_->compiled.input();
        const auto& shp   = ip.get_shape();
        const int input_h = static_cast<int>(shp[2]);
        const int input_w = static_cast<int>(shp[3]);

        // Split channels (HWC -> CHW), keep BGR order
        cv::Mat ch[3];
        cv::split(ctx.preprocessed, ch);

        // Create input tensor
        ov::Tensor in(ip.get_element_type(), shp);
        float* dst         = in.data<float>();
        const size_t plane = static_cast<size_t>(input_w) * static_cast<size_t>(input_h);

        // Keep BGR order (unlike AT model which converts to RGB)
        std::memcpy(dst + plane * 0, ch[0].data, plane * sizeof(float));
        std::memcpy(dst + plane * 1, ch[1].data, plane * sizeof(float));
        std::memcpy(dst + plane * 2, ch[2].data, plane * sizeof(float));

        impl_->request.set_input_tensor(in);
        impl_->request.infer();
        return true;
    } catch (const std::exception& e) {
        spdlog::error("[AtLegacyNNTupBackend] Inference failed: {}", e.what());
        return false;
    }
}

// ============================================================================
// Postprocessing
// ============================================================================

std::vector<AtLegacyArmorDetection> AtLegacyNNTupBackend::postprocess(
    const PreprocContext& ctx, const AtLegacyNNTupConfig& cfg) const {
    std::vector<AtLegacyArmorDetection> detections;

    ov::Tensor out = impl_->request.get_output_tensor(0);
    const float* p = out.data<const float>();
    if (!p)
        return detections;

    const auto& shape = out.get_shape();

    // Verify output shape
    if (shape.size() != 3 || shape[0] != 1) {
        spdlog::warn("[AtLegacyNNTupBackend] Unexpected output shape");
        return detections;
    }

    const int out_rows = static_cast<int>(shape[1]);
    const int out_cols = static_cast<int>(shape[2]);

    // Use minimum of generated grids and output rows
    const int valid_rows = std::min(out_rows, static_cast<int>(ctx.grids.size()));

    // Verify column count
    const int color_from = 9;
    const int color_to   = 9 + NUM_COLORS;
    const int num_from   = color_to;
    const int num_to     = color_to + ctx.num_classes;

    if (num_to > out_cols) {
        spdlog::warn(
            "[AtLegacyNNTupBackend] Class columns ({}) exceed output width ({})", num_to, out_cols);
        return detections;
    }

    // Create cv::Mat view of output
    cv::Mat obuf(out_rows, out_cols, CV_32F, const_cast<float*>(p));

    detections.reserve(256);
    const float conf_thresh = static_cast<float>(cfg.confidence_threshold);

    auto finite = [](float x) { return std::isfinite(x); };

    for (int i = 0; i < valid_rows; ++i) {
        // Read confidence
        float conf = obuf.at<float>(i, 8);
        if (conf < conf_thresh)
            continue;

        const auto& gs = ctx.grids[i];
        if (gs.stride <= 0 || gs.grid0 < 0 || gs.grid1 < 0)
            continue;

        // Decode 4 corners: (dx + grid) * stride
        const float stride_f = static_cast<float>(gs.stride);
        const float grid0_f  = static_cast<float>(gs.grid0);
        const float grid1_f  = static_cast<float>(gs.grid1);

        float x1 = (obuf.at<float>(i, 0) + grid0_f) * stride_f;
        float y1 = (obuf.at<float>(i, 1) + grid1_f) * stride_f;
        float x2 = (obuf.at<float>(i, 2) + grid0_f) * stride_f;
        float y2 = (obuf.at<float>(i, 3) + grid1_f) * stride_f;
        float x3 = (obuf.at<float>(i, 4) + grid0_f) * stride_f;
        float y3 = (obuf.at<float>(i, 5) + grid1_f) * stride_f;
        float x4 = (obuf.at<float>(i, 6) + grid0_f) * stride_f;
        float y4 = (obuf.at<float>(i, 7) + grid1_f) * stride_f;

        if (!(finite(x1) && finite(y1) && finite(x2) && finite(y2) && finite(x3) && finite(y3)
              && finite(x4) && finite(y4))) {
            continue;
        }

        // Transform to original image coordinates using matrix T
        Eigen::Matrix<float, 3, 4> Pn;
        Pn << x1, x2, x3, x4, y1, y2, y3, y4, 1.f, 1.f, 1.f, 1.f;
        Eigen::Matrix<float, 3, 4> Pd = ctx.T * Pn;

        std::array<cv::Point2f, 4> corners;
        for (int k = 0; k < 4; ++k) {
            corners[k] = cv::Point2f(Pd(0, k), Pd(1, k));
        }

        // Calculate area
        const float poly_area = quad_area(corners);
        if (!std::isfinite(poly_area) || poly_area <= 0.f)
            continue;

        // Decode color (argmax of color scores)
        cv::Mat color_scores = obuf.row(i).colRange(color_from, color_to);
        cv::Point color_id;
        cv::minMaxLoc(color_scores, nullptr, nullptr, nullptr, &color_id);

        // Decode class (argmax of class scores)
        cv::Mat num_scores = obuf.row(i).colRange(num_from, num_to);
        cv::Point num_id;
        cv::minMaxLoc(num_scores, nullptr, nullptr, nullptr, &num_id);

        const int color = color_id.x;
        const int cls   = num_id.x;

        // Filter by detect color (optional)
        // 0=BLUE, 1=RED, 2=GRAY, 3=PURPLE
        if (detect_color_ >= 0 && detect_color_ <= 1) {
            if (color != detect_color_ && color != 2 && color != 3) {
                // Not matching color and not gray/purple
                continue;
            }
        }

        // Create detection
        AtLegacyArmorDetection det(corners, cls, color, conf);
        if (det.area > 0) {
            detections.push_back(std::move(det));
        }

        // Early cap
        if (detections.size() >= 512)
            break;
    }

    return detections;
}

// ============================================================================
// NMS
// ============================================================================

std::vector<AtLegacyArmorDetection> AtLegacyNNTupBackend::nms(
    std::vector<AtLegacyArmorDetection>& detections, const AtLegacyNNTupConfig& cfg) const {
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

            if (iou(detections[i].rect, detections[j].rect) > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// ============================================================================
// Detection Entry Point
// ============================================================================

AtLegacyNNTupBackend::DetectionResult AtLegacyNNTupBackend::detect_impl(const cv::Mat& image) {
    if (!initialized_) {
        return std::unexpected(DetectorError::NotInitialized);
    }

    if (image.empty()) {
        return std::unexpected(DetectorError::InvalidImage);
    }

    // Get current config (atomic load for thread-safety)
    auto cfg = get_config();
    if (!cfg) {
        return std::unexpected(DetectorError::NotInitialized);
    }

    // Preprocess
    auto ctx = preprocess(image, *cfg);
    if (ctx.preprocessed.empty()) {
        return std::unexpected(DetectorError::PreprocessFailed);
    }

    // Infer
    if (!infer(ctx)) {
        return std::unexpected(DetectorError::InferenceFailed);
    }

    // Postprocess
    auto detections = postprocess(ctx, *cfg);

    // NMS
    auto result = nms(detections, *cfg);

    return result;
}

// ============================================================================
// Utility
// ============================================================================

std::vector<std::string> AtLegacyNNTupBackend::execution_devices() const {
    if (!impl_->loaded) {
        return {};
    }
    try {
        return impl_->compiled.get_property(ov::execution_devices);
    } catch (...) {
        return {};
    }
}

} // namespace fcs
