#include "quanta/transcoder.hpp"
#include "quanta/encode_common.hpp"
#include "quanta/ffmpeg_raii.hpp"

#include <cstdio>
#include <cstring>
#include <format>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace quanta {

using detail::alloc_codec_ctx;
using detail::alloc_filter_graph;
using detail::alloc_frame;
using detail::alloc_packet;
using detail::averr;
using detail::build_filter_chain;
using detail::CodecCtxPtr;
using detail::ColorParams;
using detail::create_encoder_context;
using detail::FilterChain;
using detail::fit_dimensions;
using detail::FmtCtxPtr;
using detail::FmtOutPtr;
using detail::FramePtr;
using detail::open_input;
using detail::PacketPtr;

// ---------------------------------------------------------------------------
// Transcoder implementation
// ---------------------------------------------------------------------------

Transcoder::Transcoder(EncodeParams params)
    : params_(params) {}

std::expected<TranscodeStats, std::string>
    Transcoder::run(std::string_view input_path, std::string_view output_path) const {

    const std::string in{input_path};
    const std::string out{output_path};
    if (params_.framerate <= 0)
        return std::unexpected("Target framerate must be positive");

    // ---- Open input ----
    auto fmt_ctx = open_input(in.c_str());
    if (!fmt_ctx)
        return std::unexpected("Cannot open input: " + in);
    if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
        return std::unexpected("Cannot find stream info");
    }

    // ---- Find video stream ----
    int video_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_idx < 0)
        return std::unexpected("No video stream found");
    auto* in_stream    = fmt_ctx->streams[video_idx];
    auto* in_codec_par = in_stream->codecpar;

    // ---- Open decoder ----
    const auto* dec_codec = avcodec_find_decoder(in_codec_par->codec_id);
    if (!dec_codec)
        return std::unexpected("Unsupported input codec");
    auto dec_ctx = alloc_codec_ctx(dec_codec);
    if (avcodec_parameters_to_context(dec_ctx.get(), in_codec_par) < 0) {
        return std::unexpected("Cannot copy decoder params");
    }
    dec_ctx->pkt_timebase = in_stream->time_base;
    if (avcodec_open2(dec_ctx.get(), dec_codec, nullptr) < 0) {
        return std::unexpected("Cannot open decoder");
    }

    // ---- Compute output dimensions ----
    int out_w{}, out_h{};
    fit_dimensions(
        in_codec_par->width, in_codec_par->height, params_.max_width, params_.max_height, out_w,
        out_h);

    AVRational in_fps = av_guess_frame_rate(fmt_ctx.get(), in_stream, nullptr);
    if (in_fps.num <= 0 || in_fps.den <= 0)
        in_fps = in_stream->avg_frame_rate;
    if (in_fps.num <= 0 || in_fps.den <= 0)
        in_fps = in_stream->r_frame_rate;
    if (in_fps.num <= 0 || in_fps.den <= 0)
        in_fps = {params_.framerate, 1};

    const AVRational out_fps = {params_.framerate, 1};

    const char* filter_mode = params_.filter.enable ? "denoise+sharpen" : "passthrough";
    std::fprintf(
        stderr, "[quanta] %dx%d @ %.0ffps -> %dx%d @ %.0ffps, %s, target %d bps\n",
        in_codec_par->width, in_codec_par->height,
        static_cast<double>(in_fps.num) / (in_fps.den ? in_fps.den : 1), out_w, out_h,
        static_cast<double>(out_fps.num) / out_fps.den, filter_mode, params_.target_bitrate);

    // ---- Build filter chain ----
    // If filters disabled, zero out all filter params (chain still used for scale+format).
    FilterParams effective_fp = params_.filter;
    if (!effective_fp.enable) {
        effective_fp = {
            .enable          = false,
            .denoise_luma    = 0,
            .denoise_chroma  = 0,
            .denoise_tl      = 0,
            .denoise_tc      = 0,
            .sharpen1_luma   = 0,
            .sharpen1_chroma = 0,
            .sharpen2_luma   = 0,
            .sharpen2_chroma = 0,
            .sharpen3_luma   = 0,
            .sharpen3_chroma = 0};
    }

    AVRational sar = in_stream->sample_aspect_ratio;
    if (sar.num == 0)
        sar = {1, 1};

    const ColorParams color = {
        .colorspace      = dec_ctx->colorspace,
        .range           = dec_ctx->color_range,
        .color_primaries = dec_ctx->color_primaries,
        .color_trc       = dec_ctx->color_trc,
        .chroma_location = dec_ctx->chroma_sample_location,
    };

    // Filter chain needs a valid time_base; decoder may leave it as {0,1}.
    AVRational filter_tb = in_stream->time_base;
    if (filter_tb.num == 0 || filter_tb.den == 0) {
        filter_tb = in_fps.num > 0 && in_fps.den > 0 ? AVRational{in_fps.den, in_fps.num}
                                                     : AVRational{1, params_.framerate};
    }

    auto filter_result = build_filter_chain(
        dec_ctx->width, dec_ctx->height, sar, filter_tb, in_fps, out_fps, dec_ctx->pix_fmt, out_w,
        out_h, effective_fp, color);
    if (!filter_result)
        return std::unexpected(filter_result.error());
    auto& chain = filter_result.value();

    // ---- Open encoder (shared config via encode_common) ----
    auto enc_result = create_encoder_context(
        out_w, out_h, {out_fps.den, out_fps.num}, out_fps, params_, sar, color);
    if (!enc_result)
        return std::unexpected(enc_result.error());
    auto enc_ctx = std::move(enc_result.value());

    // ---- Open output ----
    AVFormatContext* out_fmt_raw = nullptr;
    int ret = avformat_alloc_output_context2(&out_fmt_raw, nullptr, nullptr, out.c_str());
    if (ret < 0 || !out_fmt_raw) {
        return std::unexpected("Cannot alloc output context: " + averr(ret));
    }
    FmtOutPtr out_fmt(out_fmt_raw);

    auto* out_stream = avformat_new_stream(out_fmt.get(), nullptr);
    if (!out_stream)
        return std::unexpected("Cannot create output stream");
    ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctx.get());
    if (ret < 0)
        return std::unexpected("Cannot copy encoder params: " + averr(ret));
    out_stream->time_base           = enc_ctx->time_base;
    out_stream->avg_frame_rate      = out_fps;
    out_stream->r_frame_rate        = out_fps;
    out_stream->sample_aspect_ratio = sar;

    if (!(out_fmt->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&out_fmt->pb, out.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0)
            return std::unexpected("Cannot open output file: " + averr(ret));
    }
    ret = avformat_write_header(out_fmt.get(), nullptr);
    if (ret < 0)
        return std::unexpected("Cannot write header: " + averr(ret));

    // After write_header, the muxer may have changed the stream time_base.
    // Use this for rescaling — NOT enc_ctx->time_base.
    const AVRational out_tb = out_stream->time_base;
    const int64_t frame_duration_out =
        std::max<int64_t>(1, av_rescale_q(1, enc_ctx->time_base, out_tb));

    // ---- Frame buffers ----
    auto dec_frame  = alloc_frame();
    auto filt_frame = alloc_frame();
    auto pkt        = alloc_packet();
    if (!dec_frame || !filt_frame || !pkt) {
        return std::unexpected("Cannot allocate frame/packet");
    }

    // ---- Transcode loop ----
    TranscodeStats stats{};
    int64_t next_output_pts      = 0;
    int64_t first_packet_pts_out = AV_NOPTS_VALUE;
    int64_t last_packet_end_out  = AV_NOPTS_VALUE;

    auto encode_frame = [&](AVFrame* frame) -> std::expected<void, std::string> {
        if (frame) {
            frame->pts      = next_output_pts++;
            frame->duration = 1;
            frame->flags &= ~AV_FRAME_FLAG_KEY;
            frame->pict_type = AV_PICTURE_TYPE_NONE;
        }
        ret = avcodec_send_frame(enc_ctx.get(), frame);
        if (ret < 0)
            return std::unexpected("Encode send: " + averr(ret));
        while (ret >= 0) {
            ret = avcodec_receive_packet(enc_ctx.get(), pkt.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
                return std::unexpected("Encode recv: " + averr(ret));
            pkt->stream_index = 0;
            stats.frames_out++;
            stats.bytes_out += pkt->size;
            av_packet_rescale_ts(pkt.get(), enc_ctx->time_base, out_tb);
            if (pkt->pts != AV_NOPTS_VALUE) {
                const int64_t pkt_duration = pkt->duration > 0 ? pkt->duration : frame_duration_out;
                if (first_packet_pts_out == AV_NOPTS_VALUE)
                    first_packet_pts_out = pkt->pts;
                last_packet_end_out = pkt->pts + pkt_duration;
            }
            ret = av_interleaved_write_frame(out_fmt.get(), pkt.get());
            if (ret < 0)
                return std::unexpected("Write: " + averr(ret));
            av_packet_unref(pkt.get());
        }
        return {};
    };

    auto drain_filter = [&]() -> std::expected<void, std::string> {
        while (true) {
            ret = av_buffersink_get_frame(chain.sink_ctx, filt_frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
                return std::unexpected("Filter pull failed: " + averr(ret));
            if (auto r = encode_frame(filt_frame.get()); !r)
                return std::unexpected(r.error());
            av_frame_unref(filt_frame.get());
        }
        return {};
    };

    auto push_filter_frame = [&](AVFrame* frame) -> std::expected<void, std::string> {
        ret = av_buffersrc_add_frame_flags(chain.src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (ret < 0)
            return std::unexpected("Filter push failed: " + averr(ret));
        return drain_filter();
    };

    while (av_read_frame(fmt_ctx.get(), pkt.get()) >= 0) {
        if (pkt->stream_index != video_idx) {
            av_packet_unref(pkt.get());
            continue;
        }
        ret = avcodec_send_packet(dec_ctx.get(), pkt.get());
        av_packet_unref(pkt.get());
        if (ret < 0)
            continue;

        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx.get(), dec_frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
                break;

            stats.frames_in++;

            if (auto r = push_filter_frame(dec_frame.get()); !r)
                return std::unexpected(r.error());
        }
    }

    // Flush decoder
    avcodec_send_packet(dec_ctx.get(), nullptr);
    while (avcodec_receive_frame(dec_ctx.get(), dec_frame.get()) >= 0) {
        stats.frames_in++;
        if (auto r = push_filter_frame(dec_frame.get()); !r)
            return std::unexpected(r.error());
    }

    ret = av_buffersrc_add_frame_flags(chain.src_ctx, nullptr, 0);
    if (ret < 0)
        return std::unexpected("Filter flush failed: " + averr(ret));
    if (auto r = drain_filter(); !r)
        return std::unexpected(r.error());

    // Flush encoder
    if (auto r = encode_frame(nullptr); !r) {
        return std::unexpected(r.error());
    }

    if (stats.frames_out == 0)
        return std::unexpected("No output frames produced");

    av_write_trailer(out_fmt.get());

    // Stats
    if (out_stream->duration > 0) {
        stats.duration_sec = static_cast<double>(out_stream->duration) * av_q2d(out_tb);
    } else if (
        first_packet_pts_out != AV_NOPTS_VALUE && last_packet_end_out != AV_NOPTS_VALUE
        && last_packet_end_out > first_packet_pts_out) {
        stats.duration_sec =
            static_cast<double>(last_packet_end_out - first_packet_pts_out) * av_q2d(out_tb);
    }
    if (stats.duration_sec > 0) {
        stats.avg_bitrate =
            static_cast<double>(stats.bytes_out) * 8.0 / stats.duration_sec / 1000.0;
    }

    return stats;
}

} // namespace quanta
