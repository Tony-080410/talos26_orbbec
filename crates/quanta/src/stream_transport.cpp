#include "quanta/stream_transport.hpp"

#include "quanta/ffmpeg_raii.hpp"
#include "quanta/stream_encoder.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

namespace quanta {

using detail::alloc_codec_ctx;
using detail::alloc_frame;
using detail::alloc_packet;
using detail::CodecCtxPtr;
using detail::create_sws;
using detail::FramePtr;
using detail::PacketPtr;
using detail::SwsCtxPtr;

namespace {

constexpr uint8_t kFlagKeyframe   = 1u << 0;
constexpr uint8_t kFlagStart      = 1u << 1;
constexpr uint8_t kFlagEnd        = 1u << 2;
constexpr uint8_t kHeaderTag      = 0xA0;
constexpr uint8_t kHeaderFlagMask = 0x0F;
constexpr size_t kWireCommonHeaderSize =
    sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint8_t);
constexpr size_t kWireStartHeaderSize = kWireCommonHeaderSize + sizeof(uint32_t) + sizeof(uint64_t);

struct WireHeader {
    uint8_t flags                 = 0;
    uint32_t frame_seq            = 0;
    uint8_t fragment_index        = 0;
    uint8_t fragment_count        = 0;
    int32_t pts                   = 0;
    uint64_t capture_timestamp_ns = 0;
    size_t header_size            = kWireCommonHeaderSize;
};

template <class UInt>
void write_le(uint8_t* dst, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (size_t i = 0; i < sizeof(UInt); ++i) {
        dst[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xffu);
    }
}

template <class UInt>
[[nodiscard]] UInt read_le(const uint8_t* src) {
    static_assert(std::is_unsigned_v<UInt>);
    UInt value = 0;
    for (size_t i = 0; i < sizeof(UInt); ++i) {
        value |= static_cast<UInt>(src[i]) << (i * 8);
    }
    return value;
}

[[nodiscard]] uint64_t default_now_ns() {
    const auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}

[[nodiscard]] StreamTransportConfig normalize_config(StreamTransportConfig config) {
    if (!config.now_ns) {
        config.now_ns = [] { return default_now_ns(); };
    }
    return config;
}

[[nodiscard]] std::string ffmpeg_error(int code) {
    char buffer[128]{};
    av_strerror(code, buffer, sizeof(buffer));
    return std::string(buffer);
}

[[nodiscard]] std::expected<void, std::string>
    validate_transport_config(const StreamTransportConfig& config) {
    if (config.mtu_bytes <= kWireStartHeaderSize) {
        return std::unexpected(
            std::format(
                "MTU {} is too small for {}-byte transport header", config.mtu_bytes,
                kWireStartHeaderSize));
    }
    if (config.transport_hz <= 0)
        return std::unexpected("transport_hz must be positive");
    if (config.max_stream_fps <= 0)
        return std::unexpected("max_stream_fps must be positive");
    if (config.max_stream_fps > config.transport_hz) {
        return std::unexpected(
            "max_stream_fps cannot exceed transport_hz on a 1-chunk-per-tick link");
    }
    if (config.max_pending_frames == 0)
        return std::unexpected("max_pending_frames must be positive");
    if (config.max_queued_chunks == 0)
        return std::unexpected("max_queued_chunks must be positive");
    return {};
}

void serialize_header(std::span<uint8_t> bytes, const WireHeader& header) {
    bytes[0] = static_cast<uint8_t>(kHeaderTag | (header.flags & kHeaderFlagMask));
    write_le<uint32_t>(bytes.data() + 1, header.frame_seq);
    bytes[5] = header.fragment_index;
    bytes[6] = header.fragment_count;
    if ((header.flags & kFlagStart) == 0)
        return;

    write_le<uint32_t>(bytes.data() + 7, static_cast<uint32_t>(header.pts));
    write_le<uint64_t>(bytes.data() + 11, header.capture_timestamp_ns);
}

[[nodiscard]] std::expected<WireHeader, std::string> parse_header(std::span<const uint8_t> bytes) {
    if (bytes.size() < kWireCommonHeaderSize)
        return std::unexpected("chunk smaller than transport header");
    if ((bytes[0] & 0xF0u) != kHeaderTag)
        return std::unexpected("invalid quanta transport header tag");

    WireHeader header;
    header.flags          = static_cast<uint8_t>(bytes[0] & kHeaderFlagMask);
    header.frame_seq      = read_le<uint32_t>(bytes.data() + 1);
    header.fragment_index = bytes[5];
    header.fragment_count = bytes[6];
    header.header_size    = kWireCommonHeaderSize;

    if (header.fragment_count == 0)
        return std::unexpected("transport header fragment_count must be positive");
    if (header.fragment_index >= header.fragment_count) {
        return std::unexpected("transport header fragment_index exceeds fragment_count");
    }

    const bool start = (header.flags & kFlagStart) != 0;
    const bool end   = (header.flags & kFlagEnd) != 0;
    if (start && header.fragment_index != 0)
        return std::unexpected("start chunk must carry fragment index 0");
    if (end && header.fragment_index + 1 != header.fragment_count) {
        return std::unexpected("end chunk must carry the last fragment index");
    }

    if (!start)
        return header;

    if (bytes.size() < kWireStartHeaderSize)
        return std::unexpected("chunk smaller than transport start header");
    header.pts                  = static_cast<int32_t>(read_le<uint32_t>(bytes.data() + 7));
    header.capture_timestamp_ns = read_le<uint64_t>(bytes.data() + 11);
    header.header_size          = kWireStartHeaderSize;
    return header;
}

[[nodiscard]] bool flag_set(uint8_t flags, uint8_t bit) { return (flags & bit) != 0; }

[[nodiscard]] bool sequence_less(uint32_t lhs, uint32_t rhs) {
    return lhs != rhs && static_cast<int32_t>(lhs - rhs) < 0;
}

[[nodiscard]] double percentile_ms(const std::vector<uint64_t>& samples_ns, double q) {
    if (samples_ns.empty())
        return 0.0;

    std::vector<uint64_t> ordered(samples_ns.begin(), samples_ns.end());
    const size_t idx =
        static_cast<size_t>(std::clamp(q, 0.0, 1.0) * static_cast<double>(ordered.size() - 1));
    std::nth_element(
        ordered.begin(), ordered.begin() + static_cast<std::ptrdiff_t>(idx), ordered.end());
    return static_cast<double>(ordered[idx]) / 1'000'000.0;
}

} // namespace

class StreamSender::Impl {
public:
    Impl(EncodeParams encode_params, StreamTransportConfig transport_config)
        : encode_params_(encode_params)
        , transport_config_(normalize_config(std::move(transport_config)))
        , encoder_(encode_params_) {}

    std::expected<void, std::string> init(int src_width, int src_height) {
        if (initialized_)
            return std::unexpected("StreamSender already initialized");
        if (src_width <= 0 || src_height <= 0)
            return std::unexpected("source dimensions must be positive");
        if (auto valid = validate_transport_config(transport_config_); !valid)
            return valid;

        src_width_             = src_width;
        src_height_            = src_height;
        min_frame_interval_ns_ = std::max<uint64_t>(
            1, static_cast<uint64_t>(
                   std::llround(1'000'000'000.0 / transport_config_.max_stream_fps)));

        auto init_result = encoder_.init(src_width, src_height, transport_config_.max_stream_fps);
        if (!init_result)
            return init_result;

        initialized_ = true;
        return {};
    }

    std::expected<void, std::string> submit_frame(
        const uint8_t* data, int linesize, int64_t pts, uint64_t capture_timestamp_ns) {
        if (!initialized_)
            return std::unexpected("StreamSender not initialized");
        if (!data)
            return std::unexpected("frame data pointer is null");
        if (linesize < src_width_ * 3)
            return std::unexpected("frame linesize is smaller than packed BGR width");
        if (pts < std::numeric_limits<int32_t>::min()
            || pts > std::numeric_limits<int32_t>::max()) {
            return std::unexpected("transport wire format only supports 32-bit pts values");
        }

        const uint64_t now_ns =
            capture_timestamp_ns != 0 ? capture_timestamp_ns : transport_config_.now_ns();

        stats_.frames_submitted++;
        if (pending_frames_.size() >= transport_config_.max_pending_frames) {
            stats_.frames_dropped++;
            pending_frames_.pop_front();
        }

        PendingFrame frame;
        frame.data.resize(static_cast<size_t>(linesize) * src_height_);
        std::memcpy(frame.data.data(), data, static_cast<size_t>(linesize) * src_height_);
        frame.linesize             = linesize;
        frame.pts                  = pts;
        frame.capture_timestamp_ns = now_ns;
        pending_frames_.push_back(std::move(frame));

        return maybe_encode_pending(now_ns);
    }

    std::expected<std::optional<TransportChunk>, std::string> poll_chunk() {
        if (!initialized_)
            return std::unexpected("StreamSender not initialized");

        const uint64_t now_ns = transport_config_.now_ns();
        reset_encode_schedule_if_idle(now_ns);
        if (auto ready = maybe_encode_pending(now_ns); !ready)
            return std::unexpected(ready.error());

        if (chunk_queue_.empty())
            return std::optional<TransportChunk>{};

        auto chunk = std::move(chunk_queue_.front());
        chunk_queue_.pop_front();
        if (auto ready = maybe_encode_pending(now_ns); !ready)
            return std::unexpected(ready.error());
        return std::optional<TransportChunk>{std::move(chunk)};
    }

    [[nodiscard]] StreamSenderStats stats() const {
        auto copy              = stats_;
        copy.queued_chunks     = chunk_queue_.size();
        copy.has_pending_frame = !pending_frames_.empty();
        return copy;
    }

    [[nodiscard]] std::pair<int, int> dimensions() const {
        if (!initialized_)
            return {0, 0};
        return encoder_.dimensions();
    }

private:
    struct PendingFrame {
        std::vector<uint8_t> data;
        int linesize                  = 0;
        int64_t pts                   = 0;
        uint64_t capture_timestamp_ns = 0;
    };

    void reset_encode_schedule_if_idle(uint64_t now_ns) {
        if (!pending_frames_.empty() || !chunk_queue_.empty() || next_encode_deadline_ns_ == 0)
            return;
        if (now_ns > next_encode_deadline_ns_ + min_frame_interval_ns_)
            next_encode_deadline_ns_ = now_ns;
    }

    std::expected<void, std::string> maybe_encode_pending(uint64_t now_ns) {
        reset_encode_schedule_if_idle(now_ns);

        while (!pending_frames_.empty()) {
            if (chunk_queue_.size() >= transport_config_.max_queued_chunks)
                return {};
            if (next_encode_deadline_ns_ != 0 && now_ns < next_encode_deadline_ns_)
                return {};

            PendingFrame frame = std::move(pending_frames_.front());
            pending_frames_.pop_front();

            auto push_result = encoder_.push_frame(frame.data.data(), frame.linesize, frame.pts);
            if (!push_result)
                return push_result;

            while (auto packet = encoder_.poll_packet()) {
                if (auto fragment_result = fragment_packet(*packet, frame.capture_timestamp_ns);
                    !fragment_result) {
                    return std::unexpected(fragment_result.error());
                }
                stats_.frames_encoded++;
            }

            if (next_encode_deadline_ns_ == 0)
                next_encode_deadline_ns_ = now_ns;
            next_encode_deadline_ns_ += min_frame_interval_ns_;
        }
        return {};
    }

    std::expected<void, std::string>
        fragment_packet(const EncodedPacket& packet, uint64_t capture_timestamp_ns) {
        if (packet.size == 0)
            return std::unexpected("encoded packet is empty");
        if (packet.pts < std::numeric_limits<int32_t>::min()
            || packet.pts > std::numeric_limits<int32_t>::max()) {
            return std::unexpected("encoded packet pts exceeds 32-bit wire range");
        }

        const size_t first_payload_bytes = transport_config_.mtu_bytes - kWireStartHeaderSize;
        const size_t continuation_payload_bytes =
            transport_config_.mtu_bytes - kWireCommonHeaderSize;
        size_t fragment_count = 1;
        if (packet.size > first_payload_bytes) {
            const size_t remaining_bytes = packet.size - first_payload_bytes;
            fragment_count +=
                (remaining_bytes + continuation_payload_bytes - 1) / continuation_payload_bytes;
        }
        if (fragment_count == 0 || fragment_count > std::numeric_limits<uint8_t>::max()) {
            return std::unexpected(
                "encoded frame requires too many fragments for 8-bit fragment count");
        }

        const auto bytes = packet.bytes();
        size_t offset    = 0;
        for (size_t i = 0; i < fragment_count; ++i) {
            const bool start_of_frame = i == 0;
            const size_t header_size =
                start_of_frame ? kWireStartHeaderSize : kWireCommonHeaderSize;
            const size_t max_payload_bytes = transport_config_.mtu_bytes - header_size;
            const size_t payload_size      = std::min(max_payload_bytes, bytes.size() - offset);
            const bool end_of_frame        = offset + payload_size >= bytes.size();
            TransportChunk chunk;
            chunk.data.resize(header_size + payload_size);
            chunk.frame_seq            = next_frame_seq_;
            chunk.fragment_index       = static_cast<uint8_t>(i);
            chunk.fragment_count       = static_cast<uint8_t>(fragment_count);
            chunk.pts                  = packet.pts;
            chunk.capture_timestamp_ns = capture_timestamp_ns;
            chunk.keyframe             = packet.keyframe;
            chunk.start_of_frame       = start_of_frame;
            chunk.end_of_frame         = end_of_frame;

            WireHeader header{
                .flags = static_cast<uint8_t>(
                    (packet.keyframe && start_of_frame ? kFlagKeyframe : 0)
                    | (start_of_frame ? kFlagStart : 0) | (end_of_frame ? kFlagEnd : 0)),
                .frame_seq            = next_frame_seq_,
                .fragment_index       = static_cast<uint8_t>(i),
                .fragment_count       = static_cast<uint8_t>(fragment_count),
                .pts                  = static_cast<int32_t>(packet.pts),
                .capture_timestamp_ns = capture_timestamp_ns,
                .header_size          = header_size,
            };
            serialize_header(std::span<uint8_t>(chunk.data.data(), header_size), header);
            std::memcpy(chunk.data.data() + header_size, bytes.data() + offset, payload_size);
            offset += payload_size;

            stats_.chunks_emitted++;
            stats_.bytes_emitted += chunk.data.size();
            chunk_queue_.push_back(std::move(chunk));
        }

        next_frame_seq_++;
        return {};
    }

    EncodeParams encode_params_;
    StreamTransportConfig transport_config_;
    StreamEncoder encoder_;
    StreamSenderStats stats_{};
    std::deque<PendingFrame> pending_frames_;
    std::deque<TransportChunk> chunk_queue_;
    bool initialized_                 = false;
    int src_width_                    = 0;
    int src_height_                   = 0;
    uint64_t min_frame_interval_ns_   = 0;
    uint64_t next_encode_deadline_ns_ = 0;
    uint32_t next_frame_seq_          = 0;
};

class StreamReceiver::Impl {
public:
    explicit Impl(StreamTransportConfig transport_config)
        : transport_config_(normalize_config(std::move(transport_config))) {}

    std::expected<void, std::string>
        push_chunk(std::span<const uint8_t> bytes, uint64_t receive_timestamp_ns) {
        if (auto valid = validate_transport_config(transport_config_); !valid)
            return valid;
        if (bytes.size() > transport_config_.mtu_bytes) {
            return std::unexpected(
                std::format(
                    "chunk size {} exceeds configured MTU {}", bytes.size(),
                    transport_config_.mtu_bytes));
        }

        stats_.chunks_received++;
        stats_.bytes_received += bytes.size();

        auto parsed = parse_header(bytes);
        if (!parsed)
            return std::unexpected(parsed.error());
        const auto& header = *parsed;

        const bool start = flag_set(header.flags, kFlagStart);
        const bool end   = flag_set(header.flags, kFlagEnd);

        if (is_duplicate_or_stale_completed_frame(header.frame_seq)) {
            record_duplicate_or_stale_chunk(header.frame_seq);
            return {};
        }

        if (!assembly_.active) {
            if (!start) {
                stats_.out_of_order_chunks++;
                stats_.frames_dropped++;
                stats_.assembly_resets++;
                return {};
            }
            start_assembly_from_header(header);
        } else if (header.frame_seq != assembly_.frame_seq) {
            if (sequence_less(header.frame_seq, assembly_.frame_seq)) {
                record_duplicate_or_stale_chunk(header.frame_seq);
                return {};
            }

            stats_.frames_dropped++;
            stats_.assembly_resets++;
            reset_assembly();

            if (!start) {
                stats_.out_of_order_chunks++;
                return {};
            }

            start_assembly_from_header(header);
        } else {
            if (header.fragment_count != assembly_.expected_fragment_count) {
                stats_.out_of_order_chunks++;
                stats_.frames_dropped++;
                stats_.assembly_resets++;
                reset_assembly();
                return {};
            }

            if (header.fragment_index < assembly_.next_fragment_index) {
                stats_.duplicate_chunks++;
                return {};
            }

            if (header.fragment_index > assembly_.next_fragment_index) {
                stats_.out_of_order_chunks++;
                stats_.frames_dropped++;
                stats_.assembly_resets++;
                reset_assembly();
                return {};
            }
        }

        const auto payload = bytes.subspan(header.header_size);
        assembly_.bytes.insert(assembly_.bytes.end(), payload.begin(), payload.end());
        assembly_.next_fragment_index++;

        if (!end)
            return {};

        stats_.frames_reassembled++;
        const uint64_t receive_ns =
            receive_timestamp_ns != 0 ? receive_timestamp_ns : transport_config_.now_ns();
        auto decode_result = decode_current_frame(receive_ns);
        if (!decode_result) {
            reset_assembly();
            return decode_result;
        }
        has_completed_frame_seq_  = true;
        last_completed_frame_seq_ = assembly_.frame_seq;
        reset_assembly();
        return decode_result;
    }

    std::optional<DecodedFrame> poll_frame() {
        if (frame_queue_.empty())
            return std::nullopt;
        auto frame = std::move(frame_queue_.front());
        frame_queue_.pop_front();
        return frame;
    }

    [[nodiscard]] StreamReceiverStats stats() const {
        auto copy             = stats_;
        copy.queued_frames    = frame_queue_.size();
        copy.assembling_frame = assembly_.active;
        return copy;
    }

private:
    struct AssemblyState {
        bool active                     = false;
        uint32_t frame_seq              = 0;
        uint8_t expected_fragment_count = 0;
        uint8_t next_fragment_index     = 0;
        int64_t pts                     = 0;
        uint64_t capture_timestamp_ns   = 0;
        bool keyframe                   = false;
        std::vector<uint8_t> bytes;
    };

    void reset_assembly() { assembly_ = {}; }

    [[nodiscard]] bool is_duplicate_or_stale_completed_frame(uint32_t frame_seq) const {
        if (!has_completed_frame_seq_)
            return false;
        return frame_seq == last_completed_frame_seq_
            || sequence_less(frame_seq, last_completed_frame_seq_);
    }

    void record_duplicate_or_stale_chunk(uint32_t frame_seq) {
        if (has_completed_frame_seq_ && frame_seq == last_completed_frame_seq_) {
            stats_.duplicate_chunks++;
            return;
        }
        if (assembly_.active && frame_seq == assembly_.frame_seq) {
            stats_.duplicate_chunks++;
            return;
        }
        stats_.stale_chunks++;
    }

    void start_assembly_from_header(const WireHeader& header) {
        assembly_.active                  = true;
        assembly_.frame_seq               = header.frame_seq;
        assembly_.expected_fragment_count = header.fragment_count;
        assembly_.next_fragment_index     = 0;
        assembly_.pts                     = header.pts;
        assembly_.capture_timestamp_ns    = header.capture_timestamp_ns;
        assembly_.keyframe                = flag_set(header.flags, kFlagKeyframe);
        assembly_.bytes.clear();
    }

    std::expected<void, std::string> ensure_decoder() {
        if (dec_ctx_ && dec_frame_)
            return {};

        const AVCodec* decoder = avcodec_find_decoder(AV_CODEC_ID_HEVC);
        if (!decoder)
            return std::unexpected("HEVC decoder not found");

        dec_ctx_   = alloc_codec_ctx(decoder);
        dec_frame_ = alloc_frame();
        if (!dec_ctx_ || !dec_frame_)
            return std::unexpected("Cannot allocate decoder context/frame");

        dec_ctx_->pkt_timebase = AVRational{1, transport_config_.max_stream_fps};
        int ret                = avcodec_open2(dec_ctx_.get(), decoder, nullptr);
        if (ret < 0)
            return std::unexpected("Cannot open HEVC decoder: " + ffmpeg_error(ret));
        return {};
    }

    std::expected<void, std::string> ensure_bgr_converter(const AVFrame* frame) {
        if (sws_ctx_ && sws_width_ == frame->width && sws_height_ == frame->height
            && sws_pix_fmt_ == static_cast<AVPixelFormat>(frame->format)) {
            return {};
        }

        sws_ctx_ = create_sws(
            frame->width, frame->height, static_cast<AVPixelFormat>(frame->format), frame->width,
            frame->height, AV_PIX_FMT_BGR24);
        if (!sws_ctx_)
            return std::unexpected("Cannot create BGR swscale context");

        sws_width_   = frame->width;
        sws_height_  = frame->height;
        sws_pix_fmt_ = static_cast<AVPixelFormat>(frame->format);
        return {};
    }

    std::expected<void, std::string> decode_current_frame(uint64_t receive_timestamp_ns) {
        (void)receive_timestamp_ns;

        if (auto decoder = ensure_decoder(); !decoder)
            return decoder;

        auto packet = alloc_packet();
        if (!packet)
            return std::unexpected("Cannot allocate decode packet");
        int ret = av_new_packet(packet.get(), static_cast<int>(assembly_.bytes.size()));
        if (ret < 0)
            return std::unexpected("Cannot allocate decode packet bytes: " + ffmpeg_error(ret));
        std::memcpy(packet->data, assembly_.bytes.data(), assembly_.bytes.size());
        packet->pts                      = assembly_.pts;
        packet->dts                      = assembly_.pts;
        uint64_t decode_stage_started_ns = transport_config_.now_ns();

        ret = avcodec_send_packet(dec_ctx_.get(), packet.get());
        if (ret < 0) {
            stats_.decode_errors++;
            return std::unexpected("Decode send failed: " + ffmpeg_error(ret));
        }

        while (true) {
            ret = avcodec_receive_frame(dec_ctx_.get(), dec_frame_.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0) {
                stats_.decode_errors++;
                return std::unexpected("Decode receive failed: " + ffmpeg_error(ret));
            }

            if (auto sws_ready = ensure_bgr_converter(dec_frame_.get()); !sws_ready)
                return sws_ready;

            DecodedFrame frame;
            frame.width     = dec_frame_->width;
            frame.height    = dec_frame_->height;
            frame.linesize  = frame.width * 3;
            frame.pts       = dec_frame_->pts != AV_NOPTS_VALUE ? dec_frame_->pts : assembly_.pts;
            frame.frame_seq = assembly_.frame_seq;
            const uint64_t decode_finished_ns = transport_config_.now_ns();
            const uint64_t decode_elapsed_ns  = decode_finished_ns >= decode_stage_started_ns
                                                  ? decode_finished_ns - decode_stage_started_ns
                                                  : 0;
            frame.capture_timestamp_ns        = assembly_.capture_timestamp_ns;
            frame.decode_timestamp_ns         = decode_finished_ns;
            frame.end_to_end_latency_ns = frame.decode_timestamp_ns >= frame.capture_timestamp_ns
                                            ? frame.decode_timestamp_ns - frame.capture_timestamp_ns
                                            : 0;
            frame.data.resize(static_cast<size_t>(frame.linesize) * frame.height);

            uint8_t* dst_data[4] = {frame.data.data(), nullptr, nullptr, nullptr};
            int dst_linesize[4]  = {frame.linesize, 0, 0, 0};
            sws_scale(
                sws_ctx_.get(), dec_frame_->data, dec_frame_->linesize, 0, dec_frame_->height,
                dst_data, dst_linesize);

            stats_.frames_decoded++;
            stats_.latency_samples++;
            stats_.last_latency_ns = frame.end_to_end_latency_ns;
            if (stats_.latency_samples == 1
                || frame.end_to_end_latency_ns < stats_.min_latency_ns) {
                stats_.min_latency_ns = frame.end_to_end_latency_ns;
            }
            stats_.max_latency_ns = std::max(stats_.max_latency_ns, frame.end_to_end_latency_ns);
            latency_samples_ns_.push_back(frame.end_to_end_latency_ns);
            stats_.p75_latency_ms = percentile_ms(latency_samples_ns_, 0.75);
            decode_samples_ns_.push_back(decode_elapsed_ns);
            stats_.p75_decode_ms = percentile_ms(decode_samples_ns_, 0.75);

            frame_queue_.push_back(std::move(frame));
            av_frame_unref(dec_frame_.get());
            decode_stage_started_ns = decode_finished_ns;
        }

        return {};
    }

    StreamTransportConfig transport_config_;
    StreamReceiverStats stats_{};
    AssemblyState assembly_{};
    CodecCtxPtr dec_ctx_;
    FramePtr dec_frame_;
    SwsCtxPtr sws_ctx_;
    AVPixelFormat sws_pix_fmt_ = AV_PIX_FMT_NONE;
    int sws_width_             = 0;
    int sws_height_            = 0;
    std::vector<uint64_t> latency_samples_ns_;
    std::vector<uint64_t> decode_samples_ns_;
    std::deque<DecodedFrame> frame_queue_;
    bool has_completed_frame_seq_      = false;
    uint32_t last_completed_frame_seq_ = 0;
};

StreamSender::StreamSender(EncodeParams encode_params, StreamTransportConfig transport_config)
    : impl_(std::make_unique<Impl>(encode_params, std::move(transport_config))) {}

StreamSender::~StreamSender() = default;

StreamSender::StreamSender(StreamSender&&) noexcept            = default;
StreamSender& StreamSender::operator=(StreamSender&&) noexcept = default;

std::expected<void, std::string> StreamSender::init(int src_width, int src_height) {
    return impl_->init(src_width, src_height);
}

std::expected<void, std::string> StreamSender::submit_frame(
    const uint8_t* data, int linesize, int64_t pts, uint64_t capture_timestamp_ns) {
    return impl_->submit_frame(data, linesize, pts, capture_timestamp_ns);
}

std::expected<std::optional<TransportChunk>, std::string> StreamSender::poll_chunk() {
    return impl_->poll_chunk();
}

[[nodiscard]] StreamSenderStats StreamSender::stats() const { return impl_->stats(); }

[[nodiscard]] std::pair<int, int> StreamSender::dimensions() const { return impl_->dimensions(); }

StreamReceiver::StreamReceiver(StreamTransportConfig transport_config)
    : impl_(std::make_unique<Impl>(std::move(transport_config))) {}

StreamReceiver::~StreamReceiver() = default;

StreamReceiver::StreamReceiver(StreamReceiver&&) noexcept            = default;
StreamReceiver& StreamReceiver::operator=(StreamReceiver&&) noexcept = default;

std::expected<void, std::string>
    StreamReceiver::push_chunk(std::span<const uint8_t> bytes, uint64_t receive_timestamp_ns) {
    return impl_->push_chunk(bytes, receive_timestamp_ns);
}

std::optional<DecodedFrame> StreamReceiver::poll_frame() { return impl_->poll_frame(); }

[[nodiscard]] StreamReceiverStats StreamReceiver::stats() const { return impl_->stats(); }

} // namespace quanta
