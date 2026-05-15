#include "quanta/decoder_backend.hpp"
#include "quanta/ffmpeg_raii.hpp"

#include <cstring>
#include <format>
#include <memory>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

namespace quanta::detail {

using detail::alloc_codec_ctx;
using detail::alloc_frame;
using detail::alloc_packet;
using detail::CodecCtxPtr;
using detail::create_sws;
using detail::FramePtr;
using detail::PacketPtr;
using detail::SwsCtxPtr;

namespace {

[[nodiscard]] std::string ffmpeg_error(int code) {
    char buffer[128]{};
    av_strerror(code, buffer, sizeof(buffer));
    return std::string(buffer);
}

} // namespace

// ===========================================================================
// FfmpegDecoderBackend — FFmpeg/libavcodec HEVC decoder backend
// ===========================================================================

class FfmpegDecoderBackend final : public DecoderBackend {
public:
    [[nodiscard]] static std::expected<std::unique_ptr<FfmpegDecoderBackend>, std::string>
        create(int target_fps) noexcept try {
        auto backend = std::unique_ptr<FfmpegDecoderBackend>(new FfmpegDecoderBackend(target_fps));
        auto result  = backend->setup();
        if (!result)
            return std::unexpected(std::move(result.error()));
        return backend;
    } catch (const std::exception& e) {
        return std::unexpected(std::format("FfmpegDecoderBackend::create: {}", e.what()));
    }

    ~FfmpegDecoderBackend() override = default;

    FfmpegDecoderBackend(const FfmpegDecoderBackend&)            = delete;
    FfmpegDecoderBackend& operator=(const FfmpegDecoderBackend&) = delete;
    FfmpegDecoderBackend(FfmpegDecoderBackend&&)                 = default;
    FfmpegDecoderBackend& operator=(FfmpegDecoderBackend&&)      = default;

    std::expected<DecodeResult, std::string>
        decode(std::span<const uint8_t> encoded_bytes, int64_t pts) noexcept override try {
        auto packet = alloc_packet();
        if (!packet)
            return std::unexpected("Cannot allocate decode packet");

        int ret = av_new_packet(packet.get(), static_cast<int>(encoded_bytes.size()));
        if (ret < 0)
            return std::unexpected("Cannot allocate decode packet bytes: " + ffmpeg_error(ret));
        std::memcpy(packet->data, encoded_bytes.data(), encoded_bytes.size());
        packet->pts = pts;
        packet->dts = pts;

        ret = avcodec_send_packet(dec_ctx_.get(), packet.get());
        if (ret < 0)
            return std::unexpected("Decode send failed: " + ffmpeg_error(ret));

        DecodeResult result;

        while (true) {
            ret = avcodec_receive_frame(dec_ctx_.get(), dec_frame_.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0) {
                result.had_decode_error = true;
                return std::unexpected("Decode receive failed: " + ffmpeg_error(ret));
            }

            if ((dec_frame_->flags & AV_FRAME_FLAG_CORRUPT) != 0) {
                result.had_corrupt_frame = true;
                av_frame_unref(dec_frame_.get());
                continue;
            }

            if (auto sws_ready = ensure_bgr_converter(); !sws_ready)
                return std::unexpected(std::move(sws_ready.error()));

            DecodedImage image;
            image.width    = dec_frame_->width;
            image.height   = dec_frame_->height;
            image.linesize = image.width * 3;
            image.pts      = dec_frame_->pts != AV_NOPTS_VALUE ? dec_frame_->pts : pts;
            image.data.resize(static_cast<size_t>(image.linesize) * image.height);

            uint8_t* dst_data[4] = {image.data.data(), nullptr, nullptr, nullptr};
            int dst_linesize[4]  = {image.linesize, 0, 0, 0};
            sws_scale(
                sws_ctx_.get(), dec_frame_->data, dec_frame_->linesize, 0, dec_frame_->height,
                dst_data, dst_linesize);

            result.frames.push_back(std::move(image));
            av_frame_unref(dec_frame_.get());
        }

        return result;
    } catch (const std::exception& e) {
        return std::unexpected(std::format("FfmpegDecoderBackend::decode: {}", e.what()));
    }

private:
    explicit FfmpegDecoderBackend(int target_fps)
        : target_fps_(target_fps) {}

    std::expected<void, std::string> setup() {
        const AVCodec* decoder = avcodec_find_decoder(AV_CODEC_ID_HEVC);
        if (!decoder)
            return std::unexpected("HEVC decoder not found");

        dec_ctx_   = alloc_codec_ctx(decoder);
        dec_frame_ = alloc_frame();
        if (!dec_ctx_ || !dec_frame_)
            return std::unexpected("Cannot allocate decoder context/frame");

        dec_ctx_->pkt_timebase = AVRational{1, target_fps_};
        int ret                = avcodec_open2(dec_ctx_.get(), decoder, nullptr);
        if (ret < 0)
            return std::unexpected("Cannot open HEVC decoder: " + ffmpeg_error(ret));
        return {};
    }

    std::expected<void, std::string> ensure_bgr_converter() {
        if (sws_ctx_ && sws_width_ == dec_frame_->width && sws_height_ == dec_frame_->height
            && sws_pix_fmt_ == static_cast<AVPixelFormat>(dec_frame_->format)) {
            return {};
        }

        sws_ctx_ = create_sws(
            dec_frame_->width, dec_frame_->height, static_cast<AVPixelFormat>(dec_frame_->format),
            dec_frame_->width, dec_frame_->height, AV_PIX_FMT_BGR24);
        if (!sws_ctx_)
            return std::unexpected("Cannot create BGR swscale context");

        sws_width_   = dec_frame_->width;
        sws_height_  = dec_frame_->height;
        sws_pix_fmt_ = static_cast<AVPixelFormat>(dec_frame_->format);
        return {};
    }

    int target_fps_;
    CodecCtxPtr dec_ctx_;
    FramePtr dec_frame_;
    SwsCtxPtr sws_ctx_;
    AVPixelFormat sws_pix_fmt_ = AV_PIX_FMT_NONE;
    int sws_width_             = 0;
    int sws_height_            = 0;
};

// ===========================================================================
// Factory
// ===========================================================================

std::expected<std::unique_ptr<DecoderBackend>, std::string>
    create_ffmpeg_decoder_backend(int target_fps) noexcept {
    auto backend = FfmpegDecoderBackend::create(target_fps);
    if (!backend)
        return std::unexpected(std::move(backend.error()));
    return std::unique_ptr<DecoderBackend>(std::move(*backend));
}

} // namespace quanta::detail
