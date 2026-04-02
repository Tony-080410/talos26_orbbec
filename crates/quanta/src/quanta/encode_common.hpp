#pragma once

#include "quanta/ffmpeg_raii.hpp"
#include "quanta/transcoder.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
}

namespace quanta::detail {

// ---------------------------------------------------------------------------
// FFmpeg error → std::string
// ---------------------------------------------------------------------------

inline std::string averr(int code) {
    char buf[128]{};
    av_strerror(code, buf, sizeof(buf));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Dimension fitting: downscale src to fit within max, snap to even
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
// Filter chain: format → hqdn3d → scale
// ---------------------------------------------------------------------------

struct FilterChain {
    FilterGraphPtr graph;
    AVFilterContext* src_ctx  = nullptr; // buffersrc — non-owning
    AVFilterContext* sink_ctx = nullptr; // buffersink — non-owning
};

struct ColorParams {
    AVColorSpace colorspace                 = AVCOL_SPC_UNSPECIFIED;
    AVColorRange range                      = AVCOL_RANGE_UNSPECIFIED;
    AVColorPrimaries color_primaries        = AVCOL_PRI_UNSPECIFIED;
    AVColorTransferCharacteristic color_trc = AVCOL_TRC_UNSPECIFIED;
    AVChromaLocation chroma_location        = AVCHROMA_LOC_UNSPECIFIED;
};

inline std::expected<FilterChain, std::string> build_filter_chain(
    int src_w, int src_h, AVRational sar, AVRational time_base, AVRational input_frame_rate,
    AVRational output_frame_rate, AVPixelFormat pix_fmt, int dst_w, int dst_h,
    const FilterParams& fp, const ColorParams& color = {}) {

    auto graph = alloc_filter_graph();
    if (!graph)
        return std::unexpected("Cannot alloc filter graph");

    const AVFilter* buf_src  = avfilter_get_by_name("buffer");
    const AVFilter* buf_sink = avfilter_get_by_name("buffersink");
    if (!buf_src || !buf_sink)
        return std::unexpected("Missing buffer/buffersink filter");

    AVFilterContext* src_ctx  = nullptr;
    AVFilterContext* sink_ctx = nullptr;

    std::string args = std::format(
        "video_size={}x{}:pix_fmt={}:sar={}/{}:time_base={}/{}:frame_rate={}/"
        "{}:colorspace={}:range={}",
        src_w, src_h, static_cast<int>(pix_fmt), sar.num, sar.den, time_base.num, time_base.den,
        input_frame_rate.num, input_frame_rate.den, static_cast<int>(color.colorspace),
        static_cast<int>(color.range));

    int ret =
        avfilter_graph_create_filter(&src_ctx, buf_src, "in", args.c_str(), nullptr, graph.get());
    if (ret < 0)
        return std::unexpected("Cannot create buffersrc: " + averr(ret));

    ret = avfilter_graph_create_filter(&sink_ctx, buf_sink, "out", nullptr, nullptr, graph.get());
    if (ret < 0)
        return std::unexpected("Cannot create buffersink: " + averr(ret));

    std::string fps_filter;
    if (output_frame_rate.num > 0 && output_frame_rate.den > 0) {
        fps_filter =
            std::format("fps=fps={}/{}:round=near,", output_frame_rate.num, output_frame_rate.den);
    }

    // Pipeline: format → setparams (lock color props) → fps(optional) → scale → eq → denoise
    // → unsharp×3.
    std::string filters = std::format(
        "format=yuv420p,"
        "setparams=range={}:color_primaries={}:color_trc={}:colorspace={}:chroma_location={},"
        "{}"
        "scale={}:{},"
        "eq=saturation={:.2f},"
        "hqdn3d={:.1f}:{:.1f}:{:.1f}:{:.1f},"
        "unsharp=5:5:{:.1f}:5:5:{:.1f},"
        "unsharp=5:5:{:.1f}:5:5:{:.1f},"
        "unsharp=3:3:{:.1f}:3:3:{:.1f}",
        static_cast<int>(color.range), static_cast<int>(color.color_primaries),
        static_cast<int>(color.color_trc), static_cast<int>(color.colorspace),
        static_cast<int>(color.chroma_location), fps_filter, dst_w, dst_h, fp.saturation,
        fp.denoise_luma, fp.denoise_chroma, fp.denoise_tl, fp.denoise_tc, fp.sharpen1_luma,
        fp.sharpen1_chroma, fp.sharpen2_luma, fp.sharpen2_chroma, fp.sharpen3_luma,
        fp.sharpen3_chroma);

    AVFilterInOut* outputs = avfilter_inout_alloc();
    AVFilterInOut* inputs  = avfilter_inout_alloc();
    if (!outputs || !inputs) {
        avfilter_inout_free(&outputs);
        avfilter_inout_free(&inputs);
        return std::unexpected("Cannot alloc filter inouts");
    }

    outputs->name       = av_strdup("in");
    outputs->filter_ctx = src_ctx;
    outputs->pad_idx    = 0;
    outputs->next       = nullptr;

    inputs->name       = av_strdup("out");
    inputs->filter_ctx = sink_ctx;
    inputs->pad_idx    = 0;
    inputs->next       = nullptr;

    ret = avfilter_graph_parse_ptr(graph.get(), filters.c_str(), &inputs, &outputs, nullptr);
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    if (ret < 0)
        return std::unexpected("Cannot parse filter graph: " + averr(ret));

    ret = avfilter_graph_config(graph.get(), nullptr);
    if (ret < 0)
        return std::unexpected("Cannot config filter graph: " + averr(ret));

    return FilterChain{
        .graph    = std::move(graph),
        .src_ctx  = src_ctx,
        .sink_ctx = sink_ctx,
    };
}

// ---------------------------------------------------------------------------
// libx264 encoder context (CRF + VBV)
// ---------------------------------------------------------------------------

inline std::expected<CodecCtxPtr, std::string> create_encoder_context(
    int width, int height, AVRational time_base, AVRational frame_rate, const EncodeParams& params,
    AVRational sample_aspect_ratio = {1, 1}, const ColorParams& color = {}) {

    const auto* enc_codec = avcodec_find_encoder_by_name("libx264");
    if (!enc_codec)
        return std::unexpected("libx264 encoder not found");

    auto enc_ctx = alloc_codec_ctx(enc_codec);
    if (!enc_ctx)
        return std::unexpected("Cannot allocate encoder context");

    // CRF + VBV rate control (original working approach).
    // bit_rate=0 = CRF mode; VBV caps peak bitrate.
    enc_ctx->bit_rate               = 0;
    enc_ctx->width                  = width;
    enc_ctx->height                 = height;
    enc_ctx->time_base              = time_base;
    enc_ctx->framerate              = frame_rate;
    enc_ctx->gop_size               = params.gop_size;
    enc_ctx->max_b_frames           = 0;
    enc_ctx->pix_fmt                = AV_PIX_FMT_YUV420P;
    enc_ctx->profile                = AV_PROFILE_H264_HIGH;
    enc_ctx->sample_aspect_ratio    = sample_aspect_ratio;
    enc_ctx->colorspace             = color.colorspace;
    enc_ctx->color_range            = color.range;
    enc_ctx->color_primaries        = color.color_primaries;
    enc_ctx->color_trc              = color.color_trc;
    enc_ctx->chroma_sample_location = color.chroma_location;

    constexpr int kPhysicalLinkBudgetKbps  = 120;
    constexpr double kLatencyBudgetSeconds = 0.25;

    const auto kbps = std::max(1, params.target_bitrate / 1000);
    const double frame_rate_fps =
        frame_rate.num > 0 && frame_rate.den > 0
            ? static_cast<double>(frame_rate.num) / static_cast<double>(frame_rate.den)
            : static_cast<double>(params.framerate);
    const double gop_seconds =
        frame_rate_fps > 0.0 ? static_cast<double>(params.gop_size) / frame_rate_fps : 0.0;
    const double desired_vbv_window_seconds = std::clamp(gop_seconds, 1.0, 1.5);
    const auto link_safe_vbv_bufsize =
        std::max(1, static_cast<int>(std::floor(kPhysicalLinkBudgetKbps * kLatencyBudgetSeconds)));
    const auto desired_vbv_bufsize = std::max(
        1, static_cast<int>(std::lround(static_cast<double>(kbps) * desired_vbv_window_seconds)));
    const auto vbv_bufsize = std::min(desired_vbv_bufsize, link_safe_vbv_bufsize);

    av_opt_set(enc_ctx->priv_data, "preset", params.preset, 0);
    av_opt_set(enc_ctx->priv_data, "tune", params.tune, 0);

    // CRF quality target + VBV hard cap + strong in-loop deblocking.
    // Let VBV grow toward a 1.0-1.5s smoothing window, but never beyond the crate's
    // 120kbps * 250ms link/latency budget.
    // deblock=1,0: luma deblocking ON (strength 1), chroma deblocking OFF — preserves luma edges
    // while smoothing block boundaries in luma only.
    std::string x264_opts = std::format(
        "crf={}:vbv-maxrate={}:vbv-bufsize={}:"
        "rc-lookahead={}:bframes=0:"
        "deblock=1,0:chroma-qp-offset={}:"
        "aq-mode=3:aq-strength={}:"
        "psy-rd={},{}",
        params.crf, kbps, vbv_bufsize, params.lookahead, params.chroma_qp_offset,
        params.aq_strength, params.psy_rd, params.psy_trellis);
    av_opt_set(enc_ctx->priv_data, "x264-params", x264_opts.c_str(), 0);

    if (avcodec_open2(enc_ctx.get(), enc_codec, nullptr) < 0)
        return std::unexpected("Cannot open encoder");

    return enc_ctx;
}

} // namespace quanta::detail
