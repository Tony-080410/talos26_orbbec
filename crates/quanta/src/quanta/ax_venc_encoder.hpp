#pragma once

// ---------------------------------------------------------------------------
// AX VENC encoder configuration and frame conversion utilities
// ---------------------------------------------------------------------------
// Maps the existing EncodeParams (originally tuned for libx265) to AX VENC
// hardware encoder parameters. Preserves the tuning *intent*:
//
//   x265 parameter           AX VENC mapping
//   ───────────────────────────────────────────────────────────
//   target_bitrate           CBR u32BitRate + u32StatTime
//   gop_size                 u32Gop (IDR interval)
//   framerate                stFrameRate.fDstFrameRate
//   max_width/max_height     u32PicWidthSrc / u32PicHeightSrc
//   intra_refresh            stIntraRefresh config
//   chroma_qp_offset         s32IntraQpDelta (approximate)
//   aq_strength              CTB RC mode (QUALITY_RATE)
//   crf                       :  N/A (hardware uses CBR for realtime)
//   psy_rd / psy_trellis     :  N/A (x265 psychovisual, no HW equivalent)
//   preset / tune            :  N/A (hardware has fixed quality pipeline)
//   me / subme / rd / rdoq   :  N/A (hardware internal)
// ---------------------------------------------------------------------------

#include "quanta/ax_venc_raii.hpp"
#include "quanta/stream_encoder.hpp"
#include "quanta/transcoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <string>
#include <vector>

extern "C" {
#include "ax_sys_api.h"
#include "ax_venc_comm.h"
}

namespace quanta::detail {

// ---------------------------------------------------------------------------
// Dimension fitting: downscale src to fit within max, snap to even
// (Ported from encode_common.hpp — pure math, no FFmpeg dependency)
// ---------------------------------------------------------------------------

inline void fit_dimensions(int src_w, int src_h, int max_w, int max_h, int& out_w, int& out_h) {
    double scale = std::min(static_cast<double>(max_w) / src_w, static_cast<double>(max_h) / src_h);
    if (scale >= 1.0) {
        out_w = src_w;
        out_h = src_h;
        return;
    }
    out_w = (static_cast<int>(src_w * scale) / 2) * 2;
    out_h = (static_cast<int>(src_h * scale) / 2) * 2;
    if (out_w == 0)
        out_w = 2;
    if (out_h == 0)
        out_h = 2;
}

// ---------------------------------------------------------------------------
// BGR24 :  NV12 (YUV 4:2:0 semiplanar) conversion
// ---------------------------------------------------------------------------
// Uses BT.601 coefficients for RGB→YUV conversion.
// NV12 layout: Y plane (w×h) followed by interleaved UV plane (w/2 × h/2 × 2).
// Strides are aligned to 64 bytes (AX VENC hardware requirement).

// ---------------------------------------------------------------------------
// CMM-backed NV12 frame for AX VENC hardware DMA access
// ---------------------------------------------------------------------------
// The AX650 VENC hardware performs DMA on input frames — heap memory
// (std::vector, malloc) is NOT acceptable. Frames must reside in CMM
// (physically contiguous) memory allocated via AX_SYS_MemAllocCached,
// followed by a cache flush before submission.
// ---------------------------------------------------------------------------

constexpr int kAxVencStrideAlign = 64;
constexpr AX_U32 kCmmAlignment   = 64;

/// RAII owner of a single CMM allocation suitable for NV12 frame data.
/// Holds one contiguous CMM block; the Y and UV planes are placed within it.
struct Nv12Frame {
    AX_U64 phy_addr   = 0;       // CMM physical address
    void* vir_addr    = nullptr; // CMM virtual address
    AX_U32 alloc_size = 0;       // total CMM allocation in bytes

    int width            = 0;
    int height           = 0;
    int y_stride         = 0;    // bytes per row of Y plane (aligned to kAxVencStrideAlign)
    int uv_stride        = 0;    // bytes per row of UV plane (same as y_stride for NV12)
    size_t y_plane_size  = 0;
    size_t uv_plane_size = 0;

    Nv12Frame() = default;

    // Move-only: owns CMM memory
    Nv12Frame(const Nv12Frame&)            = delete;
    Nv12Frame& operator=(const Nv12Frame&) = delete;

    Nv12Frame(Nv12Frame&& other) noexcept
        : phy_addr(other.phy_addr)
        , vir_addr(other.vir_addr)
        , alloc_size(other.alloc_size)
        , width(other.width)
        , height(other.height)
        , y_stride(other.y_stride)
        , uv_stride(other.uv_stride)
        , y_plane_size(other.y_plane_size)
        , uv_plane_size(other.uv_plane_size) {
        other.phy_addr   = 0;
        other.vir_addr   = nullptr;
        other.alloc_size = 0;
    }

    Nv12Frame& operator=(Nv12Frame&& other) noexcept {
        if (this != &other) {
            destroy();
            phy_addr         = other.phy_addr;
            vir_addr         = other.vir_addr;
            alloc_size       = other.alloc_size;
            width            = other.width;
            height           = other.height;
            y_stride         = other.y_stride;
            uv_stride        = other.uv_stride;
            y_plane_size     = other.y_plane_size;
            uv_plane_size    = other.uv_plane_size;
            other.phy_addr   = 0;
            other.vir_addr   = nullptr;
            other.alloc_size = 0;
        }
        return *this;
    }

    ~Nv12Frame() { destroy(); }

    /// Release CMM memory back to the system.
    void destroy() noexcept {
        if (vir_addr != nullptr && phy_addr != 0) {
            AX_SYS_MemFree(phy_addr, vir_addr);
        }
        phy_addr   = 0;
        vir_addr   = nullptr;
        alloc_size = 0;
    }

    /// Allocate CMM memory for NV12 frame data.
    [[nodiscard]] std::expected<void, std::string> allocate(int w, int h) noexcept {
        destroy();

        width     = w;
        height    = h;
        y_stride  = ((w + kAxVencStrideAlign - 1) / kAxVencStrideAlign) * kAxVencStrideAlign;
        uv_stride = y_stride;

        y_plane_size  = static_cast<size_t>(y_stride) * h;
        uv_plane_size = static_cast<size_t>(uv_stride) * (h / 2);
        alloc_size    = static_cast<AX_U32>(y_plane_size + uv_plane_size);

        AX_S32 ret = AX_SYS_MemAllocCached(
            &phy_addr, &vir_addr, alloc_size, kCmmAlignment,
            const_cast<AX_S8*>(reinterpret_cast<const AX_S8*>("QuantaNV12")));
        if (ret != AX_SUCCESS) {
            phy_addr   = 0;
            vir_addr   = nullptr;
            alloc_size = 0;
            return std::unexpected(
                std::format(
                    "AX_SYS_MemAllocCached failed for NV12 {}x{} ({} bytes): ret=0x{:X}", w, h,
                    alloc_size, static_cast<unsigned>(ret)));
        }

        // Zero-initialize (stride padding must be zero)
        std::memset(vir_addr, 0, alloc_size);
        return {};
    }

    /// Flush CPU cache so DMA engine sees the written data.
    void flush_cache() noexcept {
        if (phy_addr != 0 && vir_addr != nullptr && alloc_size != 0) {
            AX_SYS_MflushCache(phy_addr, vir_addr, alloc_size);
        }
    }

    // --- Plane accessors (Y plane at offset 0, UV plane after Y) ---

    [[nodiscard]] uint8_t* y_data() noexcept { return static_cast<uint8_t*>(vir_addr); }
    [[nodiscard]] const uint8_t* y_data() const noexcept {
        return static_cast<const uint8_t*>(vir_addr);
    }
    [[nodiscard]] AX_U64 y_phy_addr() const noexcept { return phy_addr; }

    [[nodiscard]] uint8_t* uv_data() noexcept {
        return static_cast<uint8_t*>(vir_addr) + y_plane_size;
    }
    [[nodiscard]] const uint8_t* uv_data() const noexcept {
        return static_cast<const uint8_t*>(vir_addr) + y_plane_size;
    }
    [[nodiscard]] AX_U64 uv_phy_addr() const noexcept { return phy_addr + y_plane_size; }

    [[nodiscard]] size_t y_size() const noexcept { return y_plane_size; }
    [[nodiscard]] size_t uv_size() const noexcept { return uv_plane_size; }
};

/// Convert BGR24 (row-major, interleaved BGR, linesize = width * 3) to NV12.
/// Both width and height must be even (NV12 chroma subsampling requirement).
/// The output frame must already be allocated via Nv12Frame::allocate().
/// After writing, the caller must call flush_cache() before submitting to VENC.
[[nodiscard]] std::expected<void, std::string>
    bgr24_to_nv12(Nv12Frame& out, const uint8_t* bgr_data, int linesize = 0) noexcept;

/// Scale BGR24 image to fit within max_width × max_height, preserving aspect ratio.
/// Output dimensions are snapped to even (required by NV12).
/// Returns {out_width, out_height, out_data} — out_data is row-major BGR24 with
/// linesize = out_width * 3.
struct ScaledBgr {
    std::vector<uint8_t> data;
    int width  = 0;
    int height = 0;
};

[[nodiscard]] ScaledBgr scale_bgr_fit(
    const uint8_t* bgr_data, int src_width, int src_height, int src_linesize, int max_width,
    int max_height) noexcept;

// ---------------------------------------------------------------------------
// AX VENC channel attribute construction
// ---------------------------------------------------------------------------

/// Build AX_VENC_CHN_ATTR_T for HEVC realtime streaming.
/// Maps EncodeParams to hardware encoder parameters.
[[nodiscard]] std::expected<AX_VENC_CHN_ATTR_T, std::string>
    build_hevc_channel_attr(const EncodeParams& params, int width, int height) noexcept;

/// Build initial RC (rate control) parameters from EncodeParams.
[[nodiscard]] AX_VENC_RC_ATTR_T build_rc_attr(const EncodeParams& params, int framerate) noexcept;

/// Build VUI (Video Usability Information) parameters with BT.709 color metadata.
/// Matches the intent of the old FFmpeg setparams filter.
[[nodiscard]] AX_VENC_VUI_PARAM_T build_vui_param() noexcept;

/// Build intra refresh configuration from EncodeParams.
[[nodiscard]] AX_VENC_INTRA_REFRESH_T build_intra_refresh(const EncodeParams& params) noexcept;

// ---------------------------------------------------------------------------
// Frame submission helpers
// ---------------------------------------------------------------------------

/// Populate an AX_VIDEO_FRAME_INFO_T from a pre-converted NV12 frame.
/// Sets physical addresses from the CMM allocation — required for VENC DMA.
[[nodiscard]] AX_VIDEO_FRAME_INFO_T
    make_frame_info(const Nv12Frame& nv12, int64_t pts, uint64_t seq_num = 0) noexcept;

/// Extract an EncodedPacket from an AX_VENC_STREAM_T.
/// Copies the encoded data into a newly allocated buffer.
/// Releases the stream buffer after copying.
[[nodiscard]] std::expected<EncodedPacket, std::string>
    extract_packet(VencChannel& chn, AX_VENC_STREAM_T& stream) noexcept;

/// Decode an Axera error code into a human-readable string.
///
/// Axera error format (AX_DEF_ERR macro):
///   0x80000000 | (module_id << 16) | (sub_module << 8) | error_id
///
/// Returns a string like "AX_ID_VENC:2 AX_ERR_ILLEGAL_PARAM(0x0A)" for 0x8007020A.
[[nodiscard]] std::string ax_error_string(AX_S32 err) noexcept;

} // namespace quanta::detail
