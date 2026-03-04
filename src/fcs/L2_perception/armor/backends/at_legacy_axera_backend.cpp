#include "L2_perception/armor/backends/at_legacy_axera_backend.hpp"

#include "L2_perception/armor/at_legacy_config.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// Scoped Timer (RAII)
// ============================================================================

namespace {
class ScopedTimer {
public:
    explicit ScopedTimer(std::string name)
        : name_(std::move(name))
        , start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        const auto end      = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        SPDLOG_INFO("[AxeraBackend] {} elapsed: {} us", name_, duration.count());
    }

    [[nodiscard]] double elapsed_us() const {
        const auto now = std::chrono::high_resolution_clock::now();
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count());
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};
} // anonymous namespace

// ============================================================================
// Constants
// ============================================================================

namespace {
constexpr int AX_CMM_ALIGN_SIZE           = 128;
constexpr const char* AX_CMM_SESSION_NAME = "ax-samples-cmm";

// Memory allocation strategy (matching AXCL)
enum class AX_ENGINE_ALLOC_BUFFER_STRATEGY_T {
    DEFAULT = 0,
    CACHED  = 1,
};

// Helper: free IO buffer by index
void free_io_index(AX_ENGINE_IO_BUFFER_T* io_buf, size_t index) {
    for (size_t i = 0; i < index; ++i) {
        AX_ENGINE_IO_BUFFER_T* pBuf = io_buf + i;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
}

// Helper: free all IO
void free_io(AX_ENGINE_IO_T* io) {
    for (size_t j = 0; j < io->nInputSize; ++j) {
        AX_ENGINE_IO_BUFFER_T* pBuf = io->pInputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    for (size_t j = 0; j < io->nOutputSize; ++j) {
        AX_ENGINE_IO_BUFFER_T* pBuf = io->pOutputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    delete[] io->pInputs;
    delete[] io->pOutputs;
}

// Helper: prepare IO with cached input for optimal CPU->NPU transfer
std::expected<void, DetectorError> prepare_io(
    AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T* io_data,
    [[maybe_unused]] std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T>
        strategy) {
    memset(io_data, 0, sizeof(*io_data));
    io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);
    io_data->nInputSize = info->nInputSize;

    // Allocate input buffers with CACHED memory for faster CPU writes
    // Critical: CPU writes to cached memory, then flush before NPU reads
    for (int i = 0; i < static_cast<int>(info->nInputSize); ++i) {
        auto meta     = info->pInputs[i];
        auto buffer   = &io_data->pInputs[i];
        buffer->nSize = meta.nSize;

        auto ret = AX_SYS_MemAllocCached(
            reinterpret_cast<AX_U64*>(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize,
            AX_CMM_ALIGN_SIZE, reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));

        if (ret != 0) {
            free_io_index(io_data->pInputs, i);
            SPDLOG_ERROR("[AxeraBackend] Allocate input{} failed", i);
            return std::unexpected(DetectorError::InitFailed);
        }
    }

    // Allocate output buffers (use cached for output)
    io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
    io_data->nOutputSize = info->nOutputSize;

    for (int i = 0; i < static_cast<int>(info->nOutputSize); ++i) {
        auto meta     = info->pOutputs[i];
        auto buffer   = &io_data->pOutputs[i];
        buffer->nSize = meta.nSize;

        auto ret = AX_SYS_MemAllocCached(
            reinterpret_cast<AX_U64*>(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize,
            AX_CMM_ALIGN_SIZE, reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));

        if (ret != 0) {
            free_io_index(io_data->pInputs, io_data->nInputSize);
            free_io_index(io_data->pOutputs, i);
            delete[] io_data->pOutputs;
            SPDLOG_ERROR("[AxeraBackend] Allocate output{} failed", i);
            return std::unexpected(DetectorError::InitFailed);
        }
    }

    return {};
}

// Helper: read model file
bool read_file(const std::string& path, std::vector<char>& data) {
    std::fstream fs(path, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        return false;
    }

    fs.seekg(std::ios::end);
    auto fs_end = fs.tellg();
    fs.seekg(std::ios::beg);
    auto fs_beg = fs.tellg();

    auto file_size = static_cast<size_t>(fs_end - fs_beg);
    data.reserve(file_size);
    data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());

    fs.close();
    return true;
}

// Helper: sigmoid
constexpr float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Helper: reduce_max
template <typename T>
T reduce_max(T x) {
    return x;
}

template <typename T, typename... Ts>
T reduce_max(T x, Ts... xs) {
    return std::max(x, reduce_max(xs...));
}

// Helper: reduce_min
template <typename T>
T reduce_min(T x) {
    return x;
}

template <typename T, typename... Ts>
T reduce_min(T x, Ts... xs) {
    return std::min(x, reduce_min(xs...));
}

// Helper: gen_rect_bbox from keypoints
cv::Rect2f gen_rect_bbox(const AtLegacyArmorDetection& box) {
    cv::Rect2f rect_bbox;
    rect_bbox.x =
        reduce_min(box.corners[0].x, box.corners[1].x, box.corners[2].x, box.corners[3].x);
    rect_bbox.y =
        reduce_min(box.corners[0].y, box.corners[1].y, box.corners[2].y, box.corners[3].y);
    rect_bbox.width =
        reduce_max(box.corners[0].x, box.corners[1].x, box.corners[2].x, box.corners[3].x)
        - rect_bbox.x;
    rect_bbox.height =
        reduce_max(box.corners[0].y, box.corners[1].y, box.corners[2].y, box.corners[3].y)
        - rect_bbox.y;
    return rect_bbox;
}

// Strides for multi-scale detection (P3, P4, P5)
constexpr std::array<int, 3> STRIDES = {8, 16, 32};
constexpr float GRID_OFFSET          = 0.5f;
} // anonymous namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

AtLegacyAxeraBackend::AtLegacyAxeraBackend(ConfigPtr config)
    : config_(std::move(config)) {}

AtLegacyAxeraBackend::~AtLegacyAxeraBackend() {
    if (handle_) {
        free_io(&io_data_);
        AX_ENGINE_DestroyHandle(handle_);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
    }
}

AtLegacyAxeraBackend::AtLegacyAxeraBackend(AtLegacyAxeraBackend&&) noexcept            = default;
AtLegacyAxeraBackend& AtLegacyAxeraBackend::operator=(AtLegacyAxeraBackend&&) noexcept = default;

// ============================================================================
// Initialization
// ============================================================================

std::expected<void, DetectorError> AtLegacyAxeraBackend::init() {
    if (initialized_) {
        return {};
    }

    auto cfg = get_config();
    if (!cfg) {
        SPDLOG_ERROR("[AxeraBackend] Config is null");
        return std::unexpected(DetectorError::InitFailed);
    }

    // Check model file exists
    if (!std::filesystem::exists(cfg->model_path)) {
        SPDLOG_ERROR("[AxeraBackend] Model file not found: {}", cfg->model_path);
        return std::unexpected(DetectorError::ModelLoadError);
    }

    // Initialize AX system
    auto ret = AX_SYS_Init();
    if (ret != 0) {
        SPDLOG_ERROR("[AxeraBackend] AX_SYS_Init failed: {}", ret);
        return std::unexpected(DetectorError::InitFailed);
    }

    // Initialize engine
    AX_ENGINE_NPU_ATTR_T npu_attr{};
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    ret                = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        SPDLOG_ERROR("[AxeraBackend] AX_ENGINE_Init failed: {}", ret);
        AX_SYS_Deinit();
        return std::unexpected(DetectorError::InitFailed);
    }

    // Load model
    if (!read_file(cfg->model_path, model_buffer_)) {
        SPDLOG_ERROR("[AxeraBackend] Failed to read model file: {}", cfg->model_path);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return std::unexpected(DetectorError::ModelLoadError);
    }

    // Create handle
    ret = AX_ENGINE_CreateHandle(&handle_, model_buffer_.data(), model_buffer_.size());
    if (ret != 0) {
        SPDLOG_ERROR("[AxeraBackend] AX_ENGINE_CreateHandle failed: {}", ret);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return std::unexpected(DetectorError::ModelLoadError);
    }

    // Create context
    ret = AX_ENGINE_CreateContext(handle_);
    if (ret != 0) {
        SPDLOG_ERROR("[AxeraBackend] AX_ENGINE_CreateContext failed: {}", ret);
        AX_ENGINE_DestroyHandle(handle_);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return std::unexpected(DetectorError::InitFailed);
    }

    // Get IO info
    AX_ENGINE_IO_INFO_T* io_info = nullptr;
    ret                          = AX_ENGINE_GetIOInfo(handle_, &io_info);
    if (ret != 0) {
        SPDLOG_ERROR("[AxeraBackend] AX_ENGINE_GetIOInfo failed: {}", ret);
        AX_ENGINE_DestroyHandle(handle_);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return std::unexpected(DetectorError::InitFailed);
    }

    // Log model info
    SPDLOG_INFO("[AxeraBackend] Model: {}", cfg->model_path);
    SPDLOG_INFO("[AxeraBackend] Inputs: {}", io_info->nInputSize);
    for (int i = 0; i < static_cast<int>(io_info->nInputSize); ++i) {
        const auto& in = io_info->pInputs[i];
        SPDLOG_INFO("[AxeraBackend]   Input[{}]: name={}, size={} bytes", i, in.pName, in.nSize);
    }
    SPDLOG_INFO("[AxeraBackend] Outputs: {}", io_info->nOutputSize);
    for (int i = 0; i < static_cast<int>(io_info->nOutputSize); ++i) {
        const auto& out = io_info->pOutputs[i];
        SPDLOG_INFO("[AxeraBackend]   Output[{}]: name={}, size={} bytes", i, out.pName, out.nSize);
    }

    // Allocate IO (use CACHED for both input and output for optimal CPU->NPU transfer)
    auto result = prepare_io(
        io_info, &io_data_,
        std::make_pair(
            AX_ENGINE_ALLOC_BUFFER_STRATEGY_T::CACHED, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T::CACHED));
    if (!result) {
        AX_ENGINE_DestroyHandle(handle_);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return std::unexpected(result.error());
    }

    // Build and cache decode grid once (avoids rebuilding every frame)
    cached_grid_ = build_decode_grid();
    SPDLOG_INFO("[AxeraBackend] Cached decode grid with {} anchors", cached_grid_.grid_xy.size());

    initialized_ = true;
    SPDLOG_INFO("[AxeraBackend] Initialization complete");
    return {};
}

// ============================================================================
// Preprocessing (Letterbox)
// ============================================================================

AtLegacyAxeraBackend::PreprocContext AtLegacyAxeraBackend::preprocess_letterbox(
    const cv::Mat& image, const AtLegacyAxeraConfig& /*cfg*/) const {
    PreprocContext ctx;
    ctx.orig_w = image.cols;
    ctx.orig_h = image.rows;

    // Calculate scale factor (maintain aspect ratio)
    const float ratio_w = static_cast<float>(INPUT_W) / static_cast<float>(image.cols);
    const float ratio_h = static_cast<float>(INPUT_H) / static_cast<float>(image.rows);
    ctx.ratio           = std::min(ratio_w, ratio_h);

    // Calculate new size after scaling
    const int new_unpad_w = static_cast<int>(std::round(image.cols * ctx.ratio));
    const int new_unpad_h = static_cast<int>(std::round(image.rows * ctx.ratio));

    // Calculate padding (center the image)
    ctx.pad_w = (INPUT_W - new_unpad_w) / 2.0f;
    ctx.pad_h = (INPUT_H - new_unpad_h) / 2.0f;

    return ctx;
}

void AtLegacyAxeraBackend::preprocess_image(const cv::Mat& image, const PreprocContext& ctx) {
    auto* npu_input         = static_cast<uint8_t*>(io_data_.pInputs[0].pVirAddr);
    const size_t input_size = INPUT_W * INPUT_H * 3;
    const int new_unpad_w = static_cast<int>(std::round(image.cols * ctx.ratio));
    const int new_unpad_h = static_cast<int>(std::round(image.rows * ctx.ratio));
    const int top         = static_cast<int>(std::round(ctx.pad_h - 0.1f));
    const int left        = static_cast<int>(std::round(ctx.pad_w - 0.1f));
    const int bottom      = INPUT_H - top - new_unpad_h;
    const int right       = INPUT_W - left - new_unpad_w;

    // Create cv::Mat wrapper around NPU input buffer (zero-copy)
    cv::Mat npu_mat(INPUT_H, INPUT_W, CV_8UC3, npu_input);

    // Fill only padding regions to reduce write traffic.
    const cv::Scalar kPadColor(114, 114, 114);
    if (top > 0) {
        npu_mat.rowRange(0, top).setTo(kPadColor);
    }
    if (bottom > 0) {
        npu_mat.rowRange(INPUT_H - bottom, INPUT_H).setTo(kPadColor);
    }
    if (left > 0 || right > 0) {
        cv::Mat center_rows = npu_mat.rowRange(top, top + new_unpad_h);
        if (left > 0) {
            center_rows.colRange(0, left).setTo(kPadColor);
        }
        if (right > 0) {
            center_rows.colRange(INPUT_W - right, INPUT_W).setTo(kPadColor);
        }
    }

    // Define ROI for the scaled image region
    cv::Rect roi(left, top, new_unpad_w, new_unpad_h);
    cv::Mat roi_mat = npu_mat(roi);

    // Resize directly into ROI, then swap channels in place.
    if (image.cols != new_unpad_w || image.rows != new_unpad_h) {
        cv::resize(image, roi_mat, {new_unpad_w, new_unpad_h}, 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(roi_mat, roi_mat, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(image, roi_mat, cv::COLOR_BGR2RGB);
    }

    // Flush CPU cache to ensure NPU sees the data
    AX_SYS_MflushCache(io_data_.pInputs[0].phyAddr, npu_input, input_size);
}

// ============================================================================
// Inference (Pure NPU)
// ============================================================================

bool AtLegacyAxeraBackend::infer() {
    if (!initialized_ || !handle_) {
        return false;
    }

    auto ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
        SPDLOG_ERROR("[AxeraBackend] AX_ENGINE_RunSync failed: {}", ret);
        return false;
    }

    return true;
}

// ============================================================================
// Decoding (4-Head Predecode Outputs)
// ============================================================================

AtLegacyAxeraBackend::DecodeGrid AtLegacyAxeraBackend::build_decode_grid() const {
    DecodeGrid grid;
    grid.grid_xy.reserve(8400); // Total anchors: 80*80 + 40*40 + 20*20 = 8400
    grid.strides.reserve(8400);

    for (float stride : STRIDES) {
        const int feat_size = INPUT_H / static_cast<int>(stride); // 80, 40, 20

        for (int y = 0; y < feat_size; ++y) {
            for (int x = 0; x < feat_size; ++x) {
                grid.grid_xy.emplace_back(x + GRID_OFFSET, y + GRID_OFFSET);
                grid.strides.push_back(stride);
            }
        }
    }

    return grid;
}

std::vector<AtLegacyAxeraBackend::DecodedBox> AtLegacyAxeraBackend::decode_four_head_outputs(
    const float* raw_boxes,    // Shape: (1, 4, num_anchors) -> flattened, channel-major
    const float* color_scores, // Shape: (1, 4, num_anchors) -> flattened, channel-major
    const float* obj_scores,   // Shape: (1, num_pairs, num_anchors) -> flattened, channel-major
    const float* raw_kpts,     // Shape: (1, 4, 2, num_anchors) -> flattened
    int num_anchors, const DecodeGrid& grid) const {
    std::vector<DecodedBox> results;
    results.reserve(num_anchors);

    for (int i = 0; i < num_anchors; ++i) {
        const float grid_x = grid.grid_xy[i].first;
        const float grid_y = grid.grid_xy[i].second;
        const float stride = grid.strides[i];

        // Extract raw_boxes: layout [batch, box_channel, anchor] (channel-major)
        // box_channel 0 = lt_x, 1 = lt_y, 2 = rb_x, 3 = rb_y
        const float lt_x = raw_boxes[0 * num_anchors + i];
        const float lt_y = raw_boxes[1 * num_anchors + i];
        const float rb_x = raw_boxes[2 * num_anchors + i];
        const float rb_y = raw_boxes[3 * num_anchors + i];

        // Decode bounding box (xywh format)
        // boxes_xy = (grid_xy + (rb - lt) / 2) * stride
        // boxes_wh = (lt + rb) * stride
        const float center_x = (grid_x + (rb_x - lt_x) / 2.0f) * stride;
        const float center_y = (grid_y + (rb_y - lt_y) / 2.0f) * stride;
        const float width    = (lt_x + rb_x) * stride;
        const float height   = (lt_y + rb_y) * stride;

        // Extract raw_kpts: layout [batch, kpt_idx, xy, anchor] (channel-major)
        // NPU output memory order: [k0_x_a0, k0_x_a1, ..., k0_x_aN, k0_y_a0, k0_y_a1, ..., k0_y_aN,
        //                          k1_x_a0, k1_x_a1, ..., k1_y_aN, ..., k3_x_a0, ..., k3_y_aN]
        // For anchor i, keypoint k, coordinate xy: offset = k * 2 * num_anchors + xy * num_anchors
        // + i
        //
        // CRITICAL: Do NOT use interleaved access! NPU outputs are channel-major!
        // Wrong: const float* kpt_raw = raw_kpts + i * 8;

        // Find max object score across all pairs
        float max_obj_conf = obj_scores[i];
        for (int p = 1; p < NUM_PAIRS; ++p) {
            const float conf = obj_scores[p * num_anchors + i];
            if (conf > max_obj_conf) {
                max_obj_conf = conf;
            }
        }

        DecodedBox decoded;

        // Store center x, center y, width, height (temporarily, used for bbox if needed)
        decoded.kpts[0] = center_x;
        decoded.kpts[1] = center_y;
        decoded.kpts[2] = width;
        decoded.kpts[3] = height;

        // Decode keypoints: kpts = (raw_kpts + grid_xy) * stride
        // Use channel-major indexing: (kpt_idx * 2 + xy) * num_anchors + anchor_idx
        for (int k = 0; k < 4; ++k) {
            const float kpt_x       = raw_kpts[(k * 2 + 0) * num_anchors + i];
            const float kpt_y       = raw_kpts[(k * 2 + 1) * num_anchors + i];
            decoded.kpts[k * 2]     = (kpt_x + grid_x) * stride;
            decoded.kpts[k * 2 + 1] = (kpt_y + grid_y) * stride;
        }

        decoded.conf = max_obj_conf;

        // Extract color scores: layout [batch, color_idx, anchor] (channel-major)
        for (int c = 0; c < 4; ++c) {
            decoded.class1[c] = color_scores[c * num_anchors + i];
        }

        // Store pair class scores (from obj_scores)
        for (int p = 0; p < NUM_PAIRS; ++p) {
            decoded.class2[p] = obj_scores[p * num_anchors + i];
        }

        results.push_back(std::move(decoded));
    }

    return results;
}

cv::Point2f AtLegacyAxeraBackend::scale_point(const cv::Point2f& pt, const PreprocContext& ctx) {
    return cv::Point2f{(pt.x - ctx.pad_w) / ctx.ratio, (pt.y - ctx.pad_h) / ctx.ratio};
}

// ============================================================================
// Postprocessing (4-Head Network)
// ============================================================================

std::vector<AtLegacyArmorDetection> AtLegacyAxeraBackend::postprocess_four_head(
    const PreprocContext& ctx, const AtLegacyAxeraConfig& cfg) const {
    // Get output pointers
    const float* color_scores =
        static_cast<const float*>(io_data_.pOutputs[1].pVirAddr); // /model.23/Sigmoid_output_0
    const float* obj_scores =
        static_cast<const float*>(io_data_.pOutputs[2].pVirAddr); // /model.23/Sigmoid_1_output_0
    const float* raw_kpts =
        static_cast<const float*>(io_data_.pOutputs[3].pVirAddr); // /model.23/Reshape_12_output_0

    // Use cached decode grid (built once at init, avoids rebuilding every frame)
    const DecodeGrid& grid = cached_grid_;
    const int num_anchors  = static_cast<int>(grid.grid_xy.size());
    std::vector<AtLegacyArmorDetection> candidates;
    candidates.reserve(std::min(num_anchors, std::max(cfg.top_k * 4, 256)));

    // Map pair ID to ArmorName (using PAIR_NAMES_DEFAULT mapping)
    // Python: ["Gs", "2s", "3s", "4s", "5s", "Os", "Gb", "1b", "3b", "4b", "5b", "Bb"]
    // ArmorName: Sentry(0), One(1), Two(2), Three(3), Four(4), Five(5), Outpost(6), Base(7),
    // BaseLarge(8)
    static constexpr std::array<ArmorName, 12> PAIR_TO_ARMOR{
        ArmorName::Sentry,   // 0: Gs (Guard sentry)
        ArmorName::Two,      // 1: 2s
        ArmorName::Three,    // 2: 3s
        ArmorName::Four,     // 3: 4s
        ArmorName::Five,     // 4: 5s
        ArmorName::Outpost,  // 5: Os (Outpost)
        ArmorName::Sentry,   // 6: Gb (Guard base - large armor)
        ArmorName::One,      // 7: 1b (Base - hero number 1)
        ArmorName::Three,    // 8: 3b
        ArmorName::Four,     // 9: 4b
        ArmorName::Five,     // 10: 5b
        ArmorName::BaseLarge // 11: Bb (Balance/Buffer base)
    };

    for (int i = 0; i < num_anchors; ++i) {
        int pair_id         = 0;
        float max_obj_conf  = obj_scores[i];
        int best_color_id   = 0;
        float best_color_sc = color_scores[i];

        for (int p = 1; p < NUM_PAIRS; ++p) {
            const float conf = obj_scores[p * num_anchors + i];
            if (conf > max_obj_conf) {
                max_obj_conf = conf;
                pair_id      = p;
            }
        }
        if (max_obj_conf <= cfg.confidence_threshold) {
            continue;
        }

        for (int c = 1; c < 4; ++c) {
            const float sc = color_scores[c * num_anchors + i];
            if (sc > best_color_sc) {
                best_color_sc = sc;
                best_color_id = c;
            }
        }

        // Color filtering (skip gray/purple: 2,3)
        if (best_color_id == 2 || best_color_id == 3) {
            continue;
        }

        // Map color: 0->Blue(0), 1->Red(1)
        const ArmorColor color    = (best_color_id == 0) ? ArmorColor::Blue : ArmorColor::Red;
        const ArmorName armor_name = PAIR_TO_ARMOR[pair_id];
        const float grid_x         = grid.grid_xy[i].first;
        const float grid_y         = grid.grid_xy[i].second;
        const float stride         = grid.strides[i];

        // Extract and scale keypoints back to original image
        std::array<cv::Point2f, 4> corners;
        for (int k = 0; k < 4; ++k) {
            const float kpt_x =
                (raw_kpts[(k * 2 + 0) * num_anchors + i] + grid_x) * stride;
            const float kpt_y =
                (raw_kpts[(k * 2 + 1) * num_anchors + i] + grid_y) * stride;
            corners[k].x = (kpt_x - ctx.pad_w) / ctx.ratio;
            corners[k].y = (kpt_y - ctx.pad_h) / ctx.ratio;
        }

        AtLegacyArmorDetection det(corners, armor_name, color, max_obj_conf);
        if (det.area > 0) {
            candidates.push_back(std::move(det));
        }
    }

    return candidates;
}

// ============================================================================
// Helper Functions
// ============================================================================

// ============================================================================
// NMS
// ============================================================================

bool AtLegacyAxeraBackend::is_overlap(
    const AtLegacyArmorDetection& box1, const AtLegacyArmorDetection& box2) {
    cv::Rect2f bbox1 = gen_rect_bbox(box1);
    cv::Rect2f bbox2 = gen_rect_bbox(box2);
    return (bbox1 & bbox2).area() > 0;
}

std::vector<AtLegacyArmorDetection> AtLegacyAxeraBackend::nms(
    std::vector<AtLegacyArmorDetection>& detections, const AtLegacyAxeraConfig& cfg) const {
    if (detections.empty()) {
        return {};
    }

    // Sort by confidence (descending)
    std::sort(
        detections.begin(), detections.end(),
        [](const AtLegacyArmorDetection& a, const AtLegacyArmorDetection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<AtLegacyArmorDetection> result;
    std::vector<bool> removed(detections.size(), false);

    for (size_t i = 0; i < detections.size() && result.size() < static_cast<size_t>(cfg.top_k);
         i++) {
        if (removed[i])
            continue;

        result.push_back(detections[i]);

        // Suppress overlapping detections
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (!removed[j] && is_overlap(detections[i], detections[j])) {
                removed[j] = true;
            }
        }
    }

    return result;
}

// ============================================================================
// Detection Entry Point
// ============================================================================

AtLegacyAxeraBackend::DetectionResult AtLegacyAxeraBackend::detect_impl(const cv::Mat& image) {
    const auto total_start = std::chrono::high_resolution_clock::now();

    if (!initialized_) {
        return std::unexpected(DetectorError::NotInitialized);
    }

    if (image.empty()) {
        return std::unexpected(DetectorError::InvalidImage);
    }

    // Get current config
    auto cfg = get_config();
    if (!cfg) {
        return std::unexpected(DetectorError::NotInitialized);
    }

    // Preprocess: compute letterbox params (negligible time)
    auto ctx = preprocess_letterbox(image, *cfg);

    // Preprocess: actual image processing (resize + BGR2RGB + cache flush)
    {
        ScopedTimer timer("preprocess");
        preprocess_image(image, ctx);
    }

    // Infer: pure NPU execution
    {
        ScopedTimer timer("infer");
        if (!infer()) {
            return std::unexpected(DetectorError::InferenceFailed);
        }
    }

    // Postprocess (4-head outputs)
    auto detections = ([&]() {
        ScopedTimer timer("postprocess");
        return postprocess_four_head(ctx, *cfg);
    })();

    // NMS
    auto result = ([&]() {
        ScopedTimer timer("nms");
        return nms(detections, *cfg);
    })();

    // Total time
    const auto total_end = std::chrono::high_resolution_clock::now();
    const auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    SPDLOG_INFO(
        "[AxeraBackend] Total detection time: {} us ({:.2f} ms)", total_us, total_us / 1000.0);

    return result;
}

} // namespace fcs::L2
