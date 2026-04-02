#include "quanta/stream_encoder.hpp"
#include "quanta/encode_common.hpp"
#include "quanta/ffmpeg_raii.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <format>
#include <memory>
#include <optional>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

namespace quanta {

using detail::alloc_frame;
using detail::alloc_packet;
using detail::averr;
using detail::build_filter_chain;
using detail::CodecCtxPtr;
using detail::create_encoder_context;
using detail::FilterChain;
using detail::fit_dimensions;
using detail::FramePtr;
using detail::PacketPtr;

// ===========================================================================
// StreamEncoder::Impl
// ===========================================================================

class StreamEncoder::Impl {
public:
    explicit Impl(EncodeParams params)
        : params_(params) {}
    ~Impl() = default;

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

    std::expected<void, std::string> init(int src_width, int src_height, int framerate) {
        if (initialized_)
            return std::unexpected("StreamEncoder already initialized");

        src_width_  = src_width;
        src_height_ = src_height;
        framerate_  = framerate;

        // Compute output dimensions (fit within max bounds, snap to even)
        int out_w{}, out_h{};
        fit_dimensions(src_width, src_height, params_.max_width, params_.max_height, out_w, out_h);
        out_width_  = out_w;
        out_height_ = out_h;

        const AVRational time_base  = {1, framerate};
        const AVRational frame_rate = {framerate, 1};

        // Build filter chain: BGR24 input -> format=yuv420p -> filters -> scale -> output
        auto filter_result = build_filter_chain(
            src_width, src_height, {1, 1}, time_base, frame_rate, AV_PIX_FMT_BGR24, out_w, out_h,
            params_.filter);
        if (!filter_result)
            return std::unexpected("Cannot build filter chain: " + filter_result.error());
        chain_ = std::move(filter_result.value());

        // Create encoder
        auto enc_result = create_encoder_context(out_w, out_h, time_base, frame_rate, params_);
        if (!enc_result)
            return std::unexpected(enc_result.error());
        enc_ctx_ = std::move(enc_result.value());

        filt_frame_ = alloc_frame();
        if (!filt_frame_)
            return std::unexpected("Cannot allocate filtered frame");

        initialized_ = true;

        std::fprintf(
            stderr, "[quanta::stream] %dx%d @ %dfps -> %dx%d, target %d bps\n", src_width,
            src_height, framerate, out_w, out_h, params_.target_bitrate);
        return {};
    }

    std::expected<void, std::string> push_frame(const uint8_t* data, int linesize, int64_t pts) {
        if (!initialized_)
            return std::unexpected("StreamEncoder not initialized");

        // Allocate BGR24 frame and copy input data
        auto bgr = alloc_frame();
        if (!bgr)
            return std::unexpected("Cannot allocate BGR frame");

        bgr->format = AV_PIX_FMT_BGR24;
        bgr->width  = src_width_;
        bgr->height = src_height_;
        bgr->pts    = pts;

        int ret = av_frame_get_buffer(bgr.get(), 0);
        if (ret < 0)
            return std::unexpected("Cannot allocate frame buffer: " + averr(ret));

        // Copy BGR data row-by-row
        const int row_bytes = src_width_ * 3;
        for (int y = 0; y < src_height_; ++y) {
            std::memcpy(bgr->data[0] + y * bgr->linesize[0], data + y * linesize, row_bytes);
        }

        // Push through filter graph
        ret = av_buffersrc_add_frame_flags(chain_.src_ctx, bgr.get(), AV_BUFFERSRC_FLAG_KEEP_REF);
        if (ret < 0)
            return std::unexpected("Filter push failed: " + averr(ret));

        // Pull filtered frames and encode
        while (av_buffersink_get_frame(chain_.sink_ctx, filt_frame_.get()) >= 0) {
            auto r = encode_filtered_frame(filt_frame_.get());
            if (!r)
                return r;
            av_frame_unref(filt_frame_.get());
        }

        return {};
    }

    std::optional<EncodedPacket> poll_packet() {
        if (packet_queue_.empty())
            return std::nullopt;
        auto pkt = std::move(packet_queue_.front());
        packet_queue_.pop_front();
        return pkt;
    }

    std::expected<void, std::string> flush() {
        if (!initialized_)
            return std::unexpected("StreamEncoder not initialized");

        // Drain encoder: send NULL frame
        int ret = avcodec_send_frame(enc_ctx_.get(), nullptr);
        if (ret < 0)
            return std::unexpected("Flush send failed: " + averr(ret));

        auto pkt = alloc_packet();
        while (true) {
            ret = avcodec_receive_packet(enc_ctx_.get(), pkt.get());
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
                break;
            if (ret < 0)
                return std::unexpected("Flush recv failed: " + averr(ret));

            enqueue_packet(pkt.get());
            av_packet_unref(pkt.get());
        }

        return {};
    }

    [[nodiscard]] const EncodeParams& params() const { return params_; }
    [[nodiscard]] std::pair<int, int> dimensions() const { return {out_width_, out_height_}; }

private:
    std::expected<void, std::string> encode_filtered_frame(AVFrame* frame) {
        int ret = avcodec_send_frame(enc_ctx_.get(), frame);
        if (ret < 0)
            return std::unexpected("Encode send: " + averr(ret));

        auto pkt = alloc_packet();
        while (ret >= 0) {
            ret = avcodec_receive_packet(enc_ctx_.get(), pkt.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
                return std::unexpected("Encode recv: " + averr(ret));

            enqueue_packet(pkt.get());
            av_packet_unref(pkt.get());
        }
        return {};
    }

    void enqueue_packet(const AVPacket* pkt) {
        EncodedPacket encoded;
        encoded.data = std::unique_ptr<uint8_t[]>(new uint8_t[pkt->size]);
        std::copy_n(pkt->data, pkt->size, encoded.data.get());
        encoded.size     = static_cast<size_t>(pkt->size);
        encoded.pts      = pkt->pts;
        encoded.keyframe = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
        packet_queue_.push_back(std::move(encoded));
    }

    EncodeParams params_;
    TranscodeStats stats_{};
    bool initialized_ = false;

    int src_width_  = 0;
    int src_height_ = 0;
    int out_width_  = 0;
    int out_height_ = 0;
    int framerate_  = 0;

    FilterChain chain_;
    CodecCtxPtr enc_ctx_;
    FramePtr filt_frame_;
    std::deque<EncodedPacket> packet_queue_;
};

// ===========================================================================
// StreamEncoder — pimpl wrapper
// ===========================================================================

StreamEncoder::StreamEncoder(EncodeParams params)
    : impl_(std::make_unique<Impl>(params)) {}

StreamEncoder::~StreamEncoder() = default;

StreamEncoder::StreamEncoder(StreamEncoder&&) noexcept            = default;
StreamEncoder& StreamEncoder::operator=(StreamEncoder&&) noexcept = default;

std::expected<void, std::string> StreamEncoder::init(int src_width, int src_height, int framerate) {
    return impl_->init(src_width, src_height, framerate);
}

std::expected<void, std::string>
    StreamEncoder::push_frame(const uint8_t* data, int linesize, int64_t pts) {
    return impl_->push_frame(data, linesize, pts);
}

std::optional<EncodedPacket> StreamEncoder::poll_packet() { return impl_->poll_packet(); }

std::expected<void, std::string> StreamEncoder::flush() { return impl_->flush(); }

[[nodiscard]] const EncodeParams& StreamEncoder::params() const { return impl_->params(); }

[[nodiscard]] std::pair<int, int> StreamEncoder::dimensions() const { return impl_->dimensions(); }

} // namespace quanta
