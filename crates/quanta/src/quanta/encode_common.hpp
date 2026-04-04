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
// libx265 encoder context
// ---------------------------------------------------------------------------

inline std::expected<CodecCtxPtr, std::string> create_encoder_context(
    int width, int height, AVRational time_base, AVRational frame_rate, const EncodeParams& params,
    AVRational sample_aspect_ratio = {1, 1}, const ColorParams& color = {},
    bool repeat_headers = false, bool global_header = false) {

    const auto* enc_codec = avcodec_find_encoder_by_name("libx265");
    if (!enc_codec)
        return std::unexpected("libx265 encoder not found");

    auto enc_ctx = alloc_codec_ctx(enc_codec);
    if (!enc_ctx)
        return std::unexpected("Cannot allocate encoder context");

    const bool realtime_stream = repeat_headers && !global_header;
    const auto bitrate_bps     = std::max<int64_t>(1'000, params.target_bitrate);
    const auto bitrate_kbps =
        std::max(1, static_cast<int>(std::lround(static_cast<double>(bitrate_bps) / 1000.0)));
    enc_ctx->bit_rate               = realtime_stream ? bitrate_bps : 0;
    enc_ctx->rc_max_rate            = realtime_stream ? bitrate_bps : 0;
    enc_ctx->rc_min_rate            = realtime_stream ? bitrate_bps : 0;
    enc_ctx->width                  = width;
    enc_ctx->height                 = height;
    enc_ctx->time_base              = time_base;
    enc_ctx->framerate              = frame_rate;
    enc_ctx->gop_size               = params.gop_size;
    enc_ctx->max_b_frames           = 0;
    enc_ctx->pix_fmt                = AV_PIX_FMT_YUV420P;
    enc_ctx->profile                = AV_PROFILE_HEVC_MAIN;
    enc_ctx->sample_aspect_ratio    = sample_aspect_ratio;
    enc_ctx->colorspace             = color.colorspace;
    enc_ctx->color_range            = color.range;
    enc_ctx->color_primaries        = color.color_primaries;
    enc_ctx->color_trc              = color.color_trc;
    enc_ctx->chroma_sample_location = color.chroma_location;
    if (global_header)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    constexpr double kLatencyBudgetSeconds = 0.25;

    const double frame_rate_fps =
        frame_rate.num > 0 && frame_rate.den > 0
            ? static_cast<double>(frame_rate.num) / static_cast<double>(frame_rate.den)
            : static_cast<double>(params.framerate);
    const double gop_seconds =
        frame_rate_fps > 0.0 ? static_cast<double>(params.gop_size) / frame_rate_fps : 0.0;
    const double desired_vbv_window_seconds =
        realtime_stream ? kLatencyBudgetSeconds : std::clamp(gop_seconds, 1.0, 1.5);
    const auto vbv_bufsize = std::max(
        1, static_cast<int>(
               std::lround(static_cast<double>(bitrate_kbps) * desired_vbv_window_seconds)));
    enc_ctx->rc_buffer_size = realtime_stream ? vbv_bufsize * 1000 : 0;

    if (params.preset && params.preset[0] != '\0')
        av_opt_set(enc_ctx->priv_data, "preset", params.preset, 0);
    if (realtime_stream)
        av_opt_set(enc_ctx->priv_data, "tune", "zerolatency", 0);

    const auto min_keyint   = std::max(1, params.gop_size / 2);
    const int frame_threads = realtime_stream ? 1 : 0;
    const int ref_count     = 10;
    const int limit_refs    = 3;
    const char* me_mode     = "star";
    const int subme_level   = 7;
    const int rd_level      = 5;
    const int rdoq_level    = 2;
    const int amp_enabled   = 1;
    const int rect_enabled  = 1;
    const int cb_qp_offset  = std::max(0, params.chroma_qp_offset);
    const int cr_qp_offset  = std::max(0, params.chroma_qp_offset);

    std::string x265_opts;
    if (realtime_stream) {
        // Realtime stream path: hold wire rate close to target instead of letting CRF overshoot.
        // Use periodic intra refresh so the reference picture is refreshed progressively instead
        // of spending the whole GOP budget on a single hard IDR.
        x265_opts = std::format(
            "bitrate={}:vbv-maxrate={}:vbv-bufsize={}:vbv-init=0.9:strict-cbr=1:"
            "keyint={}:min-keyint={}:scenecut=0:open-gop=0:intra-refresh=1:"
            "rc-lookahead=0:bframes=0:frame-threads={}:"
            "ref={}:limit-refs={}:me={}:subme={}:rd={}:rdoq-level={}:"
            "rect={}:amp={}:aq-mode=1:aq-strength=0.4:aq-motion=0:cutree=0:"
            "cbqpoffs={}:crqpoffs={}:psy-rd={}:psy-rdoq={}:qcomp=0.5:qpstep=8:"
            "deblock=0,0:no-sao=1:info=0:vui-timing-info=0:vui-hrd-info=0:repeat-headers={}",
            bitrate_kbps, bitrate_kbps, vbv_bufsize, params.gop_size, min_keyint, frame_threads,
            ref_count, limit_refs, me_mode, subme_level, rd_level, rdoq_level, rect_enabled,
            amp_enabled, cb_qp_offset, cr_qp_offset, params.psy_rd, params.psy_trellis,
            repeat_headers ? 1 : 0);
    } else {
        const int lookahead = params.lookahead;

        // File transcode path: keep the original CRF-driven behavior.
        x265_opts = std::format(
            "crf={}:vbv-maxrate={}:vbv-bufsize={}:"
            "keyint={}:min-keyint={}:scenecut=5:open-gop=0:rc-lookahead={}:bframes=0:frame-threads="
            "{}:"
            "ref={}:limit-refs={}:me={}:subme={}:rd={}:rdoq-level={}:"
            "rect={}:amp={}:"
            "aq-mode=3:aq-strength={}:aq-motion=1:cutree=1:"
            "cbqpoffs={}:crqpoffs={}:psy-rd={}:psy-rdoq={}:"
            "deblock=1,0:no-sao=1:repeat-headers={}",
            params.crf, bitrate_kbps, vbv_bufsize, params.gop_size, min_keyint, lookahead,
            frame_threads, ref_count, limit_refs, me_mode, subme_level, rd_level, rdoq_level,
            rect_enabled, amp_enabled, params.aq_strength, cb_qp_offset, cr_qp_offset,
            params.psy_rd, params.psy_trellis, repeat_headers ? 1 : 0);
    }
    av_opt_set(enc_ctx->priv_data, "x265-params", x265_opts.c_str(), 0);

    if (avcodec_open2(enc_ctx.get(), enc_codec, nullptr) < 0)
        return std::unexpected("Cannot open encoder");

    return enc_ctx;
}

} // namespace quanta::detail
