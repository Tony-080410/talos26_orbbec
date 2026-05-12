#include "quanta/ax_venc_encoder.hpp"
#include "quanta/encoder_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <format>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" {
#include "ax_global_type.h"
#include "ax_venc_rc.h"
}

namespace quanta::detail {

// ===========================================================================
// Axera error code decoder
// ===========================================================================

namespace {
[[nodiscard]] const char* ax_module_name(int id) noexcept {
    switch (id) {
    case 0x01: return "AX_ID_ISP";
    case 0x02: return "AX_ID_CE";
    case 0x03: return "AX_ID_VO";
    case 0x04: return "AX_ID_VDSP";
    case 0x06: return "AX_ID_NPU";
    case 0x07: return "AX_ID_VENC";
    case 0x08: return "AX_ID_VDEC";
    case 0x09: return "AX_ID_JENC";
    case 0x0A: return "AX_ID_JDEC";
    case 0x0B: return "AX_ID_SYS";
    case 0x0D: return "AX_ID_IVPS";
    case 0x12: return "AX_ID_USER";
    default: return "AX_ID_?";
    }
}

[[nodiscard]] const char* ax_err_name(int id) noexcept {
    switch (id) {
    case 0x01: return "AX_ERR_INVALID_MODID";
    case 0x02: return "AX_ERR_INVALID_DEVID";
    case 0x03: return "AX_ERR_INVALID_GRPID";
    case 0x04: return "AX_ERR_INVALID_CHNID";
    case 0x05: return "AX_ERR_INVALID_PIPEID";
    case 0x0A: return "AX_ERR_ILLEGAL_PARAM";
    case 0x0B: return "AX_ERR_NULL_PTR";
    case 0x0C: return "AX_ERR_BAD_ADDR";
    case 0x10: return "AX_ERR_SYS_NOTREADY";
    case 0x11: return "AX_ERR_BUSY";
    case 0x12: return "AX_ERR_NOT_INIT";
    case 0x13: return "AX_ERR_NOT_CONFIG";
    case 0x14: return "AX_ERR_NOT_SUPPORT";
    case 0x15: return "AX_ERR_NOT_PERM";
    case 0x16: return "AX_ERR_EXIST";
    case 0x17: return "AX_ERR_UNEXIST";
    case 0x18: return "AX_ERR_NOMEM";
    case 0x19: return "AX_ERR_NOBUF";
    case 0x1A: return "AX_ERR_NOT_MATCH";
    case 0x20: return "AX_ERR_BUF_EMPTY";
    case 0x21: return "AX_ERR_BUF_FULL";
    case 0x22: return "AX_ERR_QUEUE_EMPTY";
    case 0x23: return "AX_ERR_QUEUE_FULL";
    case 0x27: return "AX_ERR_TIMED_OUT";
    case 0x28: return "AX_ERR_FLOW_END";
    case 0x29: return "AX_ERR_UNKNOWN";
    default: return "AX_ERR_?";
    }
}
} // namespace

std::string ax_error_string(AX_S32 err) noexcept {
    if (err >= 0)
        return std::format("OK(0x{:X})", static_cast<unsigned>(err));

    const unsigned code       = static_cast<unsigned>(err);
    const unsigned module     = (code >> 16) & 0xFF;
    const unsigned sub_module = (code >> 8) & 0xFF;
    const unsigned err_id     = code & 0xFF;

    return std::format(
        "{}/{} {}(0x{:02X}) [raw=0x{:08X}]", ax_module_name(static_cast<int>(module)), sub_module,
        ax_err_name(static_cast<int>(err_id)), err_id, code);
}

// ===========================================================================
// VencModule RAII
// ===========================================================================

std::atomic<int> VencModule::refcount_{0};

VencModule::VencModule(AX_VENC_ENCODER_TYPE_E encoder_type)
    : encoder_type_(encoder_type) {}

std::expected<std::shared_ptr<VencModule>, std::string>
    VencModule::acquire(AX_VENC_ENCODER_TYPE_E encoder_type) noexcept {
    int prev = refcount_.fetch_add(1);
    if (prev == 0) {
        AX_VENC_MOD_ATTR_T mod_attr{};
        mod_attr.enVencType                     = encoder_type;
        mod_attr.stModThdAttr.bExplicitSched    = AX_FALSE;
        mod_attr.stModThdAttr.u32TotalThreadNum = 1;

        AX_S32 ret = AX_VENC_Init(&mod_attr);
        if (ret != AX_SUCCESS) {
            refcount_.fetch_sub(1);
            return std::unexpected(std::format("AX_VENC_Init failed: {}", ax_error_string(ret)));
        }
    }
    return std::shared_ptr<VencModule>(
        new VencModule(encoder_type), [](VencModule* p) { delete p; });
}

VencModule::~VencModule() {
    int prev = refcount_.fetch_sub(1);
    if (prev == 1) {
        AX_VENC_Deinit();
    }
}

// ===========================================================================
// VencChannel RAII
// ===========================================================================

namespace {
std::atomic<int> g_next_chn{0}; // round-robin channel allocation
constexpr int kMaxVencChn = 64;

// Some deployed AX650 runtime images ship libax_venc.so that memcpy()s
// more bytes for AX_VENC_CHN_ATTR_T than older headers define (e.g. 224 vs 208).
// Use a zero-padded staging buffer (with extra headroom) to keep compatibility
// across minor SDK drifts.
constexpr std::size_t kCompatVencChnAttrBytes = 256;
static_assert(std::is_trivially_copyable_v<AX_VENC_CHN_ATTR_T>);

AX_S32 ax_venc_create_chn_compat(VENC_CHN chn, const AX_VENC_CHN_ATTR_T& chn_attr) noexcept {
    if constexpr (sizeof(AX_VENC_CHN_ATTR_T) >= kCompatVencChnAttrBytes) {
        return AX_VENC_CreateChn(chn, &chn_attr);
    } else {
        alignas(AX_VENC_CHN_ATTR_T) std::array<std::byte, kCompatVencChnAttrBytes> attr_raw{};
        std::memcpy(attr_raw.data(), &chn_attr, sizeof(AX_VENC_CHN_ATTR_T));
        return AX_VENC_CreateChn(chn, reinterpret_cast<const AX_VENC_CHN_ATTR_T*>(attr_raw.data()));
    }
}
} // namespace

VencChannel::VencChannel(std::shared_ptr<VencModule> module, VENC_CHN chn)
    : module_(std::move(module))
    , chn_(chn)
    , active_(true) {}

std::expected<VencChannel, std::string> VencChannel::create(
    std::shared_ptr<VencModule> module, const AX_VENC_CHN_ATTR_T& chn_attr) noexcept {
    // Round-robin channel allocation.  AX_VENC_CreateChn will fail with an
    // appropriate error code if the channel is already in use — the caller
    // should retry with a different index.  In practice a single encoder
    // instance uses at most one channel.
    VENC_CHN chn = g_next_chn.fetch_add(1) % kMaxVencChn;

    AX_S32 ret = ax_venc_create_chn_compat(chn, chn_attr);
    if (ret != AX_SUCCESS) {
        // If the preferred channel is taken, try the next few indices
        for (int attempt = 1; attempt < kMaxVencChn; ++attempt) {
            chn = (chn + 1) % kMaxVencChn;
            ret = ax_venc_create_chn_compat(chn, chn_attr);
            if (ret == AX_SUCCESS)
                return VencChannel(std::move(module), chn);
        }
        return std::unexpected(
            std::format(
                "AX_VENC_CreateChn failed: all {} channels occupied, "
                "last {}",
                kMaxVencChn, ax_error_string(ret)));
    }

    return VencChannel(std::move(module), chn);
}

VencChannel::~VencChannel() { destroy(); }

VencChannel::VencChannel(VencChannel&& other) noexcept
    : module_(std::move(other.module_))
    , chn_(other.chn_)
    , active_(other.active_) {
    other.active_ = false;
    other.chn_    = -1;
}

VencChannel& VencChannel::operator=(VencChannel&& other) noexcept {
    if (this != &other) {
        destroy();
        module_       = std::move(other.module_);
        chn_          = other.chn_;
        active_       = other.active_;
        other.active_ = false;
        other.chn_    = -1;
    }
    return *this;
}

void VencChannel::destroy() noexcept {
    if (active_) {
        AX_VENC_StopRecvFrame(chn_);
        AX_VENC_DestroyChn(chn_);
        active_ = false;
    }
}

AX_S32 VencChannel::send_frame(const AX_VIDEO_FRAME_INFO_T& frame, AX_S32 timeout_ms) noexcept {
    return AX_VENC_SendFrame(chn_, &frame, timeout_ms);
}

AX_S32 VencChannel::get_stream(AX_VENC_STREAM_T& stream, AX_S32 timeout_ms) noexcept {
    return AX_VENC_GetStream(chn_, &stream, timeout_ms);
}

AX_S32 VencChannel::release_stream(const AX_VENC_STREAM_T& stream) noexcept {
    return AX_VENC_ReleaseStream(chn_, &stream);
}

AX_S32 VencChannel::start_recv_frame(const AX_VENC_RECV_PIC_PARAM_T& param) noexcept {
    return AX_VENC_StartRecvFrame(chn_, &param);
}

AX_S32 VencChannel::stop_recv_frame() noexcept { return AX_VENC_StopRecvFrame(chn_); }

AX_S32 VencChannel::request_idr(bool instant) noexcept {
    return AX_VENC_RequestIDR(chn_, instant ? AX_TRUE : AX_FALSE);
}

AX_S32 VencChannel::set_rc_param(const AX_VENC_RC_PARAM_T& rc_param) noexcept {
    return AX_VENC_SetRcParam(chn_, &rc_param);
}

AX_S32 VencChannel::set_vui_param(const AX_VENC_VUI_PARAM_T& vui_param) noexcept {
    return AX_VENC_SetVuiParam(chn_, &vui_param);
}

AX_S32 VencChannel::set_intra_refresh(const AX_VENC_INTRA_REFRESH_T& cfg) noexcept {
    return AX_VENC_SetIntraRefresh(chn_, &cfg);
}

AX_S32 VencChannel::set_roi_attr(const AX_VENC_ROI_ATTR_T& roi) noexcept {
    return AX_VENC_SetRoiAttr(chn_, &roi);
}

AX_S32 VencChannel::query_status(AX_VENC_CHN_STATUS_T& status) noexcept {
    return AX_VENC_QueryStatus(chn_, &status);
}

// ===========================================================================
// BGR24 :  NV12 conversion (BT.601)
// ===========================================================================

namespace {

// BT.601 RGB→Y coefficients (16..235 range, 8-bit)
constexpr int kRY   = 66;  //  0.257 * 256 ≈ 66
constexpr int kGY   = 129; //  0.504 * 256 ≈ 129
constexpr int kBY   = 25;  //  0.098 * 256 ≈ 25
constexpr int kYOff = 16;

// BT.601 RGB→Cb coefficients (16..240 range)
constexpr int kRCb   = -38; // -0.148 * 256 ≈ -38
constexpr int kGCb   = -74; // -0.291 * 256 ≈ -74
constexpr int kBCb   = 112; //  0.439 * 256 ≈ 112
constexpr int kCbOff = 128;

// BT.601 RGB→Cr coefficients
constexpr int kRCr   = 112; //  0.439 * 256 ≈ 112
constexpr int kGCr   = -94; // -0.368 * 256 ≈ -94
constexpr int kBCr   = -18; // -0.071 * 256 ≈ -18
constexpr int kCrOff = 128;

[[nodiscard]] inline uint8_t clamp_u8(int v) noexcept {
    return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

// Convert a single BGR pixel to YUV
inline void
    bgr_to_yuv(uint8_t b, uint8_t g, uint8_t r, uint8_t& y, uint8_t& u, uint8_t& v) noexcept {
    y = clamp_u8(((kRY * r + kGY * g + kBY * b) >> 8) + kYOff);
    u = clamp_u8(((kRCb * r + kGCb * g + kBCb * b) >> 8) + kCbOff);
    v = clamp_u8(((kRCr * r + kGCr * g + kBCr * b) >> 8) + kCrOff);
}

} // namespace

std::expected<void, std::string>
    bgr24_to_nv12(Nv12Frame& frame, const uint8_t* bgr_data, int linesize) noexcept {
    if (frame.vir_addr == nullptr || frame.alloc_size == 0) {
        return std::unexpected("Nv12Frame not allocated");
    }

    const int bgr_linesize = (linesize > 0) ? linesize : frame.width * 3;

    uint8_t* y_ptr  = frame.y_data();
    uint8_t* uv_ptr = frame.uv_data();

    for (int row = 0; row < frame.height; ++row) {
        const uint8_t* bgr_row = bgr_data + row * bgr_linesize;
        uint8_t* y_row         = y_ptr + row * frame.y_stride;
        for (int col = 0; col < frame.width; ++col) {
            uint8_t b = bgr_row[col * 3 + 0];
            uint8_t g = bgr_row[col * 3 + 1];
            uint8_t r = bgr_row[col * 3 + 2];

            uint8_t y, u, v;
            bgr_to_yuv(b, g, r, y, u, v);

            y_row[col] = y;

            // 4:2:0 chroma subsampling: nearest-neighbor 2×2 blocks
            if ((row & 1) == 0 && (col & 1) == 0) {
                const int uv_row   = row / 2;
                const int uv_col   = col / 2;
                const int uv_off   = uv_row * frame.uv_stride + uv_col * 2;
                uv_ptr[uv_off + 0] = u;
                uv_ptr[uv_off + 1] = v;
            }
        }
    }

    return {};
}

// ===========================================================================
// BGR scale-to-fit
// ===========================================================================

ScaledBgr scale_bgr_fit(
    const uint8_t* bgr_data, int src_width, int src_height, int src_linesize, int max_width,
    int max_height) noexcept {
    const int src_ls = (src_linesize > 0) ? src_linesize : src_width * 3;

    double scale = std::min(
        static_cast<double>(max_width) / src_width, static_cast<double>(max_height) / src_height);

    ScaledBgr result;
    if (scale >= 1.0) {
        // No downscale needed — copy as-is
        result.width  = src_width;
        result.height = src_height;
        result.data.resize(src_height * src_ls);
        for (int y = 0; y < src_height; ++y) {
            std::memcpy(result.data.data() + y * src_ls, bgr_data + y * src_ls, src_ls);
        }
        return result;
    }

    // Downscale with bilinear interpolation (pixel centers)
    int out_w = (static_cast<int>(src_width * scale) / 2) * 2;
    int out_h = (static_cast<int>(src_height * scale) / 2) * 2;
    out_w     = std::max(2, out_w);
    out_h     = std::max(2, out_h);

    result.width     = out_w;
    result.height    = out_h;
    const int out_ls = out_w * 3;
    result.data.resize(out_h * out_ls);

    for (int y = 0; y < out_h; ++y) {
        double src_y = (y + 0.5) * src_height / out_h - 0.5;
        src_y        = std::clamp(src_y, 0.0, static_cast<double>(src_height - 1));
        int y0       = static_cast<int>(src_y);
        int y1       = std::min(y0 + 1, src_height - 1);
        double wy    = src_y - y0;

        for (int x = 0; x < out_w; ++x) {
            double src_x = (x + 0.5) * src_width / out_w - 0.5;
            src_x        = std::clamp(src_x, 0.0, static_cast<double>(src_width - 1));
            int x0       = static_cast<int>(src_x);
            int x1       = std::min(x0 + 1, src_width - 1);
            double wx    = src_x - x0;

            for (int c = 0; c < 3; ++c) {
                double v00 = bgr_data[y0 * src_ls + x0 * 3 + c];
                double v10 = bgr_data[y0 * src_ls + x1 * 3 + c];
                double v01 = bgr_data[y1 * src_ls + x0 * 3 + c];
                double v11 = bgr_data[y1 * src_ls + x1 * 3 + c];

                double top    = v00 + (v10 - v00) * wx;
                double bottom = v01 + (v11 - v01) * wx;
                double val    = top + (bottom - top) * wy;

                result.data[y * out_ls + x * 3 + c] =
                    static_cast<uint8_t>(std::clamp(std::lround(val), 0L, 255L));
            }
        }
    }

    return result;
}

// ===========================================================================
// Channel attribute construction
// ===========================================================================

std::expected<AX_VENC_CHN_ATTR_T, std::string>
    build_hevc_channel_attr(const EncodeParams& params, int width, int height) noexcept {
    if (width < MIN_VENC_PIC_WIDTH || width > MAX_VENC_PIC_WIDTH)
        return std::unexpected(
            std::format(
                "Width {} out of range [{}, {}]", width, MIN_VENC_PIC_WIDTH, MAX_VENC_PIC_WIDTH));
    if (height < MIN_VENC_PIC_HEIGHT || height > MAX_VENC_PIC_HEIGHT)
        return std::unexpected(
            std::format(
                "Height {} out of range [{}, {}]", height, MIN_VENC_PIC_HEIGHT,
                MAX_VENC_PIC_HEIGHT));
    if (width % 2 != 0 || height % 2 != 0)
        return std::unexpected(std::format("Dimensions {}x{} must be even", width, height));

    const auto bitrate_bps = std::max<int64_t>(VENC_MIN_BITRATE * 1000LL, params.target_bitrate);

    // Estimate stream buffer size: ~3 frames worth at target bitrate (in bytes)
    const int framerate   = std::max(1, params.framerate);
    const AX_U32 buf_size = std::max<AX_U32>(
        256 * 1024, // minimum 256KB
        static_cast<AX_U32>((bitrate_bps / framerate / 8) * 3));

    AX_VENC_CHN_ATTR_T attr{};

    // --- stVencAttr ---
    attr.stVencAttr.enType          = PT_H265;
    attr.stVencAttr.u32MaxPicWidth  = static_cast<AX_U32>(std::max(width, params.max_width));
    attr.stVencAttr.u32MaxPicHeight = static_cast<AX_U32>(std::max(height, params.max_height));
    attr.stVencAttr.enMemSource     = AX_MEMORY_SOURCE_CMM;
    attr.stVencAttr.u32BufSize      = buf_size;
    attr.stVencAttr.enProfile       = AX_VENC_HEVC_MAIN_PROFILE;
    attr.stVencAttr.enLevel         = AX_VENC_HEVC_LEVEL_5;
    attr.stVencAttr.enTier          = AX_VENC_HEVC_MAIN_TIER;
    attr.stVencAttr.enStrmBitDepth  = AX_VENC_STREAM_BIT_8;
    attr.stVencAttr.u32PicWidthSrc  = static_cast<AX_U32>(width);
    attr.stVencAttr.u32PicHeightSrc = static_cast<AX_U32>(height);
    attr.stVencAttr.enLinkMode      = AX_VENC_UNLINK_MODE;
    attr.stVencAttr.s32StopWaitTime = -1; // wait for output queue to drain on destroy
    attr.stVencAttr.u8InFifoDepth   = 3;
    attr.stVencAttr.u8OutFifoDepth  = 3;
    // HEADER_ATTACH_TO_PB intentionally NOT set — VPS/SPS/PPS are managed
    // by the consumer layer (ensure_keyframe_headers), matching the reference
    // encoder in ax_video_encoder_ax650.cpp.

    // --- stRcAttr ---
    attr.stRcAttr = build_rc_attr(params, framerate);

    // --- stGopAttr ---
    attr.stGopAttr.enGopMode = AX_VENC_GOPMODE_NORMALP;

    return attr;
}

AX_VENC_RC_ATTR_T build_rc_attr(const EncodeParams& params, int framerate) noexcept {
    AX_VENC_RC_ATTR_T rc{};
    rc.enRcMode                  = AX_VENC_RC_MODE_H265CBR;
    rc.s32FirstFrameStartQp      = -1; // let encoder decide
    rc.stFrameRate.fSrcFrameRate = static_cast<AX_F32>(framerate);
    rc.stFrameRate.fDstFrameRate = static_cast<AX_F32>(framerate);

    const auto bitrate_bps  = std::max<int64_t>(VENC_MIN_BITRATE * 1000LL, params.target_bitrate);
    const auto bitrate_kbps = static_cast<AX_U32>(std::max(
        static_cast<int>(VENC_MIN_BITRATE),
        static_cast<int>(std::lround(static_cast<double>(bitrate_bps) / 1000.0))));

    auto& cbr       = rc.stH265Cbr;
    cbr.u32Gop      = std::max<AX_U32>(1, static_cast<AX_U32>(params.gop_size));
    cbr.u32StatTime = 1;               // 1-second rate statistic window
    cbr.u32BitRate  = bitrate_kbps;

    // Tight QP to hold the wire rate close to target.
    // Match the old x265 strict CBR behavior.
    cbr.u32MaxQp    = 51;
    cbr.u32MinQp    = 10;
    cbr.u32MaxIQp   = 51;
    cbr.u32MinIQp   = 10;
    cbr.u32MaxIprop = 100;
    cbr.u32MinIprop = 1;

    // Map chroma_qp_offset to intra QP delta (approximate; HW has no per-plane QP offset)
    if (params.chroma_qp_offset > 0) {
        cbr.s32IntraQpDelta = std::clamp(-params.chroma_qp_offset, -51, 51);
    }

    // Map aq_strength to CTB-level QP adjustment.
    // x265 aq-strength=0.4 with aq-mode=1 → moderate variance-adaptive QP
    // AX VENC CTB RC: QUALITY_RATE applies both subjective and rate-driven CTB QP
    if (params.aq_strength > 0.0) {
        cbr.stQpmapInfo.enCtbRcMode = AX_VENC_RC_CTBRC_QUALITY_RATE;
    } else {
        cbr.stQpmapInfo.enCtbRcMode = AX_VENC_RC_CTBRC_DISABLE;
    }

    return rc;
}

AX_VENC_VUI_PARAM_T build_vui_param() noexcept {
    AX_VENC_VUI_PARAM_T vui{};

    // Video signal: BT.709, limited range
    vui.stVuiVideoSignal.video_signal_type_present_flag  = 1;
    vui.stVuiVideoSignal.video_format                    = 5; // Unspecified
    vui.stVuiVideoSignal.video_full_range_flag           = 0; // Limited range
    vui.stVuiVideoSignal.colour_description_present_flag = 1;
    vui.stVuiVideoSignal.colour_primaries                = 1; // BT.709
    vui.stVuiVideoSignal.transfer_characteristics        = 1; // BT.709
    vui.stVuiVideoSignal.matrix_coefficients             = 1; // BT.709

    return vui;
}

AX_VENC_INTRA_REFRESH_T build_intra_refresh(const EncodeParams& params) noexcept {
    AX_VENC_INTRA_REFRESH_T ir{};
    ir.bRefresh           = params.intra_refresh ? AX_TRUE : AX_FALSE;
    ir.u32RefreshNum      = std::max<AX_U32>(1, static_cast<AX_U32>(params.gop_size));
    ir.u32ReqIQp          = 30;
    ir.enIntraRefreshMode = AX_VENC_INTRA_REFRESH_ROW;
    return ir;
}

// ===========================================================================
// Frame submission helpers
// ===========================================================================

AX_VIDEO_FRAME_INFO_T
make_frame_info(const Nv12Frame& nv12, int64_t pts, uint64_t seq_num) noexcept {
    AX_VIDEO_FRAME_INFO_T info{};
    info.enModId = AX_ID_USER;

    auto& frame          = info.stVFrame;
    frame.u32Width       = static_cast<AX_U32>(nv12.width);
    frame.u32Height      = static_cast<AX_U32>(nv12.height);
    frame.enImgFormat    = AX_FORMAT_YUV420_SEMIPLANAR; // NV12
    frame.enVscanFormat  = AX_VSCAN_FORMAT_RASTER;
    frame.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
    frame.stColorGamut   = AX_COLOR_GAMUT_BT709;

    // Y plane (aligned stride, CMM physical address)
    frame.u32PicStride[0] = static_cast<AX_U32>(nv12.y_stride);
    frame.u64PhyAddr[0]   = nv12.y_phy_addr();
    frame.u64VirAddr[0]   = reinterpret_cast<AX_U64>(nv12.y_data());

    // UV plane (aligned stride, same as Y for NV12, CMM physical address)
    frame.u32PicStride[1] = static_cast<AX_U32>(nv12.uv_stride);
    frame.u64PhyAddr[1]   = nv12.uv_phy_addr();
    frame.u64VirAddr[1]   = reinterpret_cast<AX_U64>(nv12.uv_data());

    // Total frame size in bytes (Y + UV, with aligned strides)
    frame.u32FrameSize = static_cast<AX_U32>(nv12.y_size() + nv12.uv_size());

    frame.u64PTS       = static_cast<AX_U64>(pts);
    frame.u64SeqNum    = seq_num;
    frame.u32FrameFlag = 0;

    return info;
}

std::expected<EncodedPacket, std::string>
    extract_packet(VencChannel& chn, AX_VENC_STREAM_T& stream) noexcept {
    const auto& pack = stream.stPack;

    EncodedPacket pkt;
    pkt.pts = static_cast<int64_t>(pack.u64PTS);

    // Use enCodingType for keyframe detection (matches Axera SDK reference encoder).
    // Checking NALU types is wrong: HEADER_ATTACH_TO_PB causes VPS/SPS/PPS to appear
    // in PB frames too, which would falsely mark every frame as a keyframe.
    pkt.keyframe = (pack.enCodingType == AX_VENC_INTRA_FRAME)
                || (pack.enCodingType == AX_VENC_VIRTUAL_INTRA_FRAME);

    if (pack.u32Len > 0 && pack.pu8Addr != nullptr) {
        pkt.data = std::unique_ptr<uint8_t[]>(new uint8_t[pack.u32Len]);
        std::copy_n(pack.pu8Addr, pack.u32Len, pkt.data.get());
        pkt.size = pack.u32Len;
    } else {
        pkt.size = 0;
    }

    // Release the stream buffer back to the hardware
    AX_S32 ret = chn.release_stream(stream);
    if (ret != AX_SUCCESS) {
        return std::unexpected(
            std::format("AX_VENC_ReleaseStream failed: {}", ax_error_string(ret)));
    }

    return pkt;
}

// ===========================================================================
// HEVC Annex B NALU parsing — for VPS/SPS/PPS cache management
// ===========================================================================

namespace {

constexpr uint8_t kHevcNaluVps = 32;
constexpr uint8_t kHevcNaluSps = 33;
constexpr uint8_t kHevcNaluPps = 34;

constexpr uint8_t kStartCode4[] = {0x00, 0x00, 0x00, 0x01};

struct HevcNaluView {
    const uint8_t* data;
    size_t size;
    uint8_t type;
};

std::vector<HevcNaluView> parse_hevc_nalus(const uint8_t* data, size_t size) {
    std::vector<HevcNaluView> nalus;
    size_t offset = 0;

    while (offset + 3 <= size) {
        size_t prefix_len = 0;
        size_t code_pos   = offset;
        for (; code_pos + 3 <= size; ++code_pos) {
            if (code_pos + 4 <= size && data[code_pos] == 0 && data[code_pos + 1] == 0
                && data[code_pos + 2] == 0 && data[code_pos + 3] == 1) {
                prefix_len = 4;
                break;
            }
            if (data[code_pos] == 0 && data[code_pos + 1] == 0 && data[code_pos + 2] == 1) {
                prefix_len = 3;
                break;
            }
        }

        if (prefix_len == 0)
            break;

        const size_t payload = code_pos + prefix_len;

        size_t next_pos = payload;
        for (; next_pos + 3 <= size; ++next_pos) {
            if (next_pos + 4 <= size && data[next_pos] == 0 && data[next_pos + 1] == 0
                && data[next_pos + 2] == 0 && data[next_pos + 3] == 1) {
                break;
            }
            if (data[next_pos] == 0 && data[next_pos + 1] == 0 && data[next_pos + 2] == 1) {
                break;
            }
        }
        if (next_pos + 3 > size)
            next_pos = size;

        if (payload >= next_pos) {
            offset = payload;
            continue;
        }

        const uint8_t* nalu    = data + payload;
        const size_t nalu_size = next_pos - payload;
        const uint8_t type     = (nalu_size >= 2) ? static_cast<uint8_t>((nalu[0] >> 1) & 0x3F) : 0;

        nalus.push_back({nalu, nalu_size, type});
        offset = next_pos;
    }

    return nalus;
}

} // anonymous namespace

// ===========================================================================
// AxBackend — AX VENC hardware encoder backend
// ===========================================================================

class AxBackend final : public EncoderBackend {
public:
    static constexpr AX_S32 kSendTimeoutMs   = 30;
    static constexpr AX_S32 kStreamTimeoutMs = 5;
    // Match the official ax-video-sdk mitigation for AX650: keep a small
    // inflight window of CMM-backed input frames alive after SendFrame
    // succeeds, otherwise libax_venc may dereference released memory later.
    static constexpr std::size_t kReusableInflightDepth = 4;

    [[nodiscard]] static std::expected<std::unique_ptr<AxBackend>, std::string>
        create(EncodeParams params, int src_width, int src_height, int framerate) noexcept {
        auto backend = std::unique_ptr<AxBackend>(new AxBackend(std::move(params)));
        auto result  = backend->setup(src_width, src_height, framerate);
        if (!result)
            return std::unexpected(std::move(result.error()));
        return backend;
    }

    ~AxBackend() override {
        if (channel_) {
            drain_encoder();
        }
    }

    AxBackend(const AxBackend&)            = delete;
    AxBackend& operator=(const AxBackend&) = delete;
    AxBackend(AxBackend&&)                 = default;
    AxBackend& operator=(AxBackend&&)      = default;

    std::expected<void, std::string>
        push_frame(const uint8_t* data, int linesize, int64_t pts) noexcept override {
        // --- Step 1: Scale BGR to fit within max dimensions ---
        ScaledBgr scaled = scale_bgr_fit(
            data, src_width_, src_height_, linesize, params_.max_width, params_.max_height);

        // --- Step 2: Allocate CMM-backed NV12 frame and convert BGR ---
        Nv12Frame nv12;
        auto alloc_result = nv12.allocate(scaled.width, scaled.height);
        if (!alloc_result) {
            return std::unexpected(std::move(alloc_result.error()));
        }

        auto convert_result = bgr24_to_nv12(nv12, scaled.data.data(), scaled.width * 3);
        if (!convert_result) {
            return std::unexpected(std::move(convert_result.error()));
        }

        // Flush CPU cache so the hardware DMA engine sees the data
        nv12.flush_cache();

        // --- Step 3: Submit to hardware encoder ---
        const bool force_keyframe = std::exchange(force_keyframe_requested_, false);
        if (force_keyframe) {
            (void)channel_->request_idr(true);
        }

        AX_VIDEO_FRAME_INFO_T frame_info = make_frame_info(nv12, pts, frame_seq_++);

        AX_S32 ret = channel_->send_frame(frame_info, kSendTimeoutMs);
        if (ret == AX_ERR_VENC_QUEUE_FULL || ret == AX_ERR_VENC_BUF_FULL) {
            auto drain_result = drain_packets();
            if (!drain_result) {
                return std::unexpected(std::move(drain_result.error()));
            }
            ret = channel_->send_frame(frame_info, kSendTimeoutMs);
        }
        if (ret != AX_SUCCESS) {
            return std::unexpected(
                std::format("AX_VENC_SendFrame failed: {}", ax_error_string(ret)));
        }

        retain_inflight_frame(std::move(nv12));

        // --- Step 4: Pull encoded packets ---
        return drain_packets();
    }

    void request_keyframe() noexcept override { force_keyframe_requested_ = true; }

    std::optional<EncodedPacket> poll_packet() noexcept override {
        // Try to drain more packets from the encoder
        auto drain_result = drain_packets();
        if (!drain_result) {
            std::fprintf(
                stderr, "[quanta::stream][ax] drain error: %s\n", drain_result.error().c_str());
        }

        if (packet_queue_.empty())
            return std::nullopt;
        auto pkt = std::move(packet_queue_.front());
        packet_queue_.pop_front();
        return pkt;
    }

    std::expected<void, std::string> flush() noexcept override {
        drain_encoder();
        return {};
    }

    const EncodeParams& params() const noexcept override { return params_; }
    std::pair<int, int> dimensions() const noexcept override { return {out_width_, out_height_}; }

private:
    explicit AxBackend(EncodeParams params)
        : params_(std::move(params)) {}

    std::expected<void, std::string> setup(int src_width, int src_height, int framerate) noexcept {
        src_width_  = src_width;
        src_height_ = src_height;
        framerate_  = framerate;

        int out_w{}, out_h{};
        fit_dimensions(src_width, src_height, params_.max_width, params_.max_height, out_w, out_h);
        out_width_  = out_w;
        out_height_ = out_h;

        auto module_result = VencModule::acquire(AX_VENC_VIDEO_ENCODER);
        if (!module_result)
            return std::unexpected(std::move(module_result.error()));
        module_ = std::move(*module_result);

        auto chn_attr_result = build_hevc_channel_attr(params_, out_w, out_h);
        if (!chn_attr_result)
            return std::unexpected(std::move(chn_attr_result.error()));
        AX_VENC_CHN_ATTR_T chn_attr = std::move(*chn_attr_result);

        auto channel_result = VencChannel::create(module_, chn_attr);
        if (!channel_result)
            return std::unexpected(std::move(channel_result.error()));
        channel_ = std::move(*channel_result);

        AX_VENC_VUI_PARAM_T vui = build_vui_param();
        AX_S32 ret              = channel_->set_vui_param(vui);
        if (ret != AX_SUCCESS) {
            return std::unexpected(
                std::format("AX_VENC_SetVuiParam failed: {}", ax_error_string(ret)));
        }

        if (params_.intra_refresh) {
            AX_VENC_INTRA_REFRESH_T ir = build_intra_refresh(params_);
            ret                        = channel_->set_intra_refresh(ir);
            if (ret != AX_SUCCESS) {
                return std::unexpected(
                    std::format("AX_VENC_SetIntraRefresh failed: {}", ax_error_string(ret)));
            }
        }

        AX_VENC_RECV_PIC_PARAM_T recv_param{};
        recv_param.s32RecvPicNum = -1;
        ret                      = channel_->start_recv_frame(recv_param);
        if (ret != AX_SUCCESS) {
            return std::unexpected(
                std::format("AX_VENC_StartRecvFrame failed: {}", ax_error_string(ret)));
        }

        std::fprintf(
            stderr,
            "[quanta::stream][ax] %dx%d @ %dfps -> %dx%d HEVC, target %d bps, "
            "ch=%d, intra_refresh=%d\n",
            src_width, src_height, framerate, out_w, out_h, params_.target_bitrate,
            static_cast<int>(channel_->id()), params_.intra_refresh ? 1 : 0);

        return {};
    }

    std::expected<void, std::string> drain_packets() noexcept {
        while (true) {
            AX_VENC_STREAM_T stream{};
            AX_S32 ret = channel_->get_stream(stream, kStreamTimeoutMs);
            if (ret != AX_SUCCESS)
                break;

            auto pkt_result = extract_packet(*channel_, stream);
            if (!pkt_result)
                return std::unexpected(std::move(pkt_result.error()));
            ensure_keyframe_headers(*pkt_result);
            packet_queue_.push_back(std::move(*pkt_result));
        }
        return {};
    }

    void drain_encoder() noexcept {
        if (!channel_.has_value())
            return;

        (void)channel_->stop_recv_frame();

        constexpr int kDrainTimeoutMs   = 200;
        constexpr int kMaxDrainAttempts = 10;
        for (int i = 0; i < kMaxDrainAttempts; ++i) {
            AX_VENC_STREAM_T stream{};
            AX_S32 ret = channel_->get_stream(stream, kDrainTimeoutMs);
            if (ret != AX_SUCCESS)
                break;

            auto pkt_result = extract_packet(*channel_, stream);
            if (pkt_result) {
                ensure_keyframe_headers(*pkt_result);
                packet_queue_.push_back(std::move(*pkt_result));
            }
        }

        inflight_frames_.clear();
    }

    void retain_inflight_frame(Nv12Frame&& frame) noexcept {
        inflight_frames_.push_back(std::move(frame));
        while (inflight_frames_.size() > kReusableInflightDepth) {
            inflight_frames_.pop_front();
        }
    }

    void ensure_keyframe_headers(EncodedPacket& pkt) noexcept {
        if (pkt.size == 0 || !pkt.data)
            return;

        auto nalus = parse_hevc_nalus(pkt.data.get(), pkt.size);

        bool found_vps = false, found_sps = false, found_pps = false;
        for (const auto& nalu : nalus) {
            switch (nalu.type) {
            case kHevcNaluVps:
                cached_vps_.assign(nalu.data, nalu.data + nalu.size);
                found_vps = true;
                break;
            case kHevcNaluSps:
                cached_sps_.assign(nalu.data, nalu.data + nalu.size);
                found_sps = true;
                break;
            case kHevcNaluPps:
                cached_pps_.assign(nalu.data, nalu.data + nalu.size);
                found_pps = true;
                break;
            default: break;
            }
        }

        if (!pkt.keyframe)
            return;

        const bool need_vps = !found_vps && !cached_vps_.empty();
        const bool need_sps = !found_sps && !cached_sps_.empty();
        const bool need_pps = !found_pps && !cached_pps_.empty();

        if (!need_vps && !need_sps && !need_pps)
            return;

        size_t prefix = 0;
        if (need_vps)
            prefix += 4 + cached_vps_.size();
        if (need_sps)
            prefix += 4 + cached_sps_.size();
        if (need_pps)
            prefix += 4 + cached_pps_.size();

        auto new_buf = std::unique_ptr<uint8_t[]>(new uint8_t[prefix + pkt.size]);
        size_t off   = 0;

        auto append = [&](const std::vector<uint8_t>& nalu) {
            std::copy_n(kStartCode4, 4, new_buf.get() + off);
            off += 4;
            std::copy(nalu.begin(), nalu.end(), new_buf.get() + off);
            off += nalu.size();
        };

        if (need_vps)
            append(cached_vps_);
        if (need_sps)
            append(cached_sps_);
        if (need_pps)
            append(cached_pps_);

        std::copy_n(pkt.data.get(), pkt.size, new_buf.get() + off);

        pkt.data = std::move(new_buf);
        pkt.size = prefix + pkt.size;
    }

    EncodeParams params_;

    int src_width_  = 0;
    int src_height_ = 0;
    int out_width_  = 0;
    int out_height_ = 0;
    int framerate_  = 0;

    std::shared_ptr<VencModule> module_;
    std::optional<VencChannel> channel_;

    std::deque<EncodedPacket> packet_queue_;
    std::deque<Nv12Frame> inflight_frames_;
    uint64_t frame_seq_            = 0;
    bool force_keyframe_requested_ = false;

    std::vector<uint8_t> cached_vps_;
    std::vector<uint8_t> cached_sps_;
    std::vector<uint8_t> cached_pps_;
};

// ===========================================================================
// Factory
// ===========================================================================

std::expected<std::unique_ptr<EncoderBackend>, std::string>
    create_ax_backend(EncodeParams params, int src_width, int src_height, int framerate) noexcept {
    auto backend = AxBackend::create(std::move(params), src_width, src_height, framerate);
    if (!backend)
        return std::unexpected(std::move(backend.error()));
    return std::unique_ptr<EncoderBackend>(std::move(*backend));
}

} // namespace quanta::detail
