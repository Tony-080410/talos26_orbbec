#include "quanta/stream_transport.hpp"

#include "quanta/decoder_backend.hpp"
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

namespace quanta {

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

[[nodiscard]] uint64_t ms_to_ns(uint64_t ms) { return ms * 1'000'000ULL; }

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
    if (config.soft_deadline_ms == 0)
        return std::unexpected("soft_deadline_ms must be positive");
    if (config.hard_deadline_ms == 0)
        return std::unexpected("hard_deadline_ms must be positive");
    if (config.display_deadline_ms == 0)
        return std::unexpected("display_deadline_ms must be positive");
    if (config.soft_deadline_ms > config.hard_deadline_ms) {
        return std::unexpected("soft_deadline_ms cannot exceed hard_deadline_ms");
    }
    if (config.hard_deadline_ms > config.display_deadline_ms) {
        return std::unexpected("hard_deadline_ms cannot exceed display_deadline_ms");
    }
    if (config.clean_request_freeze_threshold == 0)
        return std::unexpected("clean_request_freeze_threshold must be positive");
    if (config.clean_request_cooldown_ms == 0)
        return std::unexpected("clean_request_cooldown_ms must be positive");
    if (config.clean_request_window_ms == 0)
        return std::unexpected("clean_request_window_ms must be positive");
    if (config.max_clean_requests_per_window == 0)
        return std::unexpected("max_clean_requests_per_window must be positive");
    if (config.motion_hold_max_px < 0)
        return std::unexpected("motion_hold_max_px cannot be negative");
    if (config.prefilter_temporal_shift < 1 || config.prefilter_temporal_shift > 8)
        return std::unexpected("prefilter_temporal_shift must be in [1, 8]");
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

[[nodiscard]] uint64_t elapsed_since_ns(uint64_t timestamp_ns, uint64_t now_ns) {
    if (timestamp_ns == 0 || now_ns < timestamp_ns)
        return 0;
    return now_ns - timestamp_ns;
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

[[nodiscard]] double percentile_count(const std::vector<uint64_t>& samples, double q) {
    if (samples.empty())
        return 0.0;

    std::vector<uint64_t> ordered(samples.begin(), samples.end());
    const size_t idx =
        static_cast<size_t>(std::clamp(q, 0.0, 1.0) * static_cast<double>(ordered.size() - 1));
    std::nth_element(
        ordered.begin(), ordered.begin() + static_cast<std::ptrdiff_t>(idx), ordered.end());
    return static_cast<double>(ordered[idx]);
}

enum class SubstituteReason {
    MissingChunk,
    DecodeError,
    DisplayDeadline,
    DependencyDirty,
};

struct MotionVector {
    int dx     = 0;
    int dy     = 0;
    bool valid = false;
};

} // namespace

class StreamSender::Impl {
public:
    static std::expected<std::unique_ptr<Impl>, std::string> create(
        EncodeParams encode_params, StreamTransportConfig transport_config, int src_width,
        int src_height) {
        auto impl =
            std::unique_ptr<Impl>(new Impl(std::move(encode_params), std::move(transport_config)));
        auto result = impl->setup(src_width, src_height);
        if (!result)
            return std::unexpected(std::move(result.error()));
        return impl;
    }

    ~Impl() = default;

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

    std::expected<void, std::string> submit_frame(
        const uint8_t* data, int linesize, int64_t pts, uint64_t capture_timestamp_ns) {
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
        const uint64_t now_ns = transport_config_.now_ns();
        expire_queued_chunks(now_ns);
        reset_encode_schedule_if_idle(now_ns);
        if (auto ready = maybe_encode_pending(now_ns); !ready)
            return std::unexpected(ready.error());
        expire_queued_chunks(now_ns);

        if (chunk_queue_.empty())
            return poll_repair_chunk();

        auto chunk = std::move(chunk_queue_.front());
        chunk_queue_.pop_front();
        record_chunk_emitted(chunk);
        if (auto ready = maybe_encode_pending(now_ns); !ready)
            return std::unexpected(ready.error());
        return std::optional<TransportChunk>{std::move(chunk)};
    }

    void request_clean_frame() {
        stats_.force_idr_requests++;
        notify_bad_link();
        force_clean_frame_pending_ = true;
    }

    void notify_bad_link() {
        bad_link_prefilter_frames_remaining_ = std::max(
            bad_link_prefilter_frames_remaining_, transport_config_.prefilter_window_frames);
    }

    [[nodiscard]] StreamSenderStats stats() const {
        auto copy              = stats_;
        copy.queued_chunks     = chunk_queue_.size() + repair_queue_.size();
        copy.has_pending_frame = !pending_frames_.empty();
        copy.chunks_per_frame  = stats_.frames_encoded > 0
                                   ? static_cast<double>(stats_.chunks_generated)
                                        / static_cast<double>(stats_.frames_encoded)
                                   : 0.0;
        return copy;
    }

    [[nodiscard]] std::pair<int, int> dimensions() const { return encoder_->dimensions(); }

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

    [[nodiscard]] uint64_t chunk_age_ns(const TransportChunk& chunk, uint64_t now_ns) const {
        return elapsed_since_ns(chunk.capture_timestamp_ns, now_ns);
    }

    [[nodiscard]] uint64_t pending_frame_age_ns(const PendingFrame& frame, uint64_t now_ns) const {
        return elapsed_since_ns(frame.capture_timestamp_ns, now_ns);
    }

    void update_max_queue_depth() {
        stats_.max_queued_chunks_seen =
            std::max(stats_.max_queued_chunks_seen, chunk_queue_.size() + repair_queue_.size());
    }

    void drop_expired_pending_frames(uint64_t now_ns) {
        while (!pending_frames_.empty()
               && pending_frame_age_ns(pending_frames_.front(), now_ns)
                      > ms_to_ns(transport_config_.hard_deadline_ms)) {
            stats_.frames_dropped++;
            stats_.expired_frames_dropped++;
            pending_frames_.pop_front();
        }
    }

    [[nodiscard]] bool has_newer_work_after_front_frame() const {
        if (chunk_queue_.empty())
            return false;

        const uint32_t front_frame_seq = chunk_queue_.front().frame_seq;
        const auto newer_chunk         = std::find_if(
            std::next(chunk_queue_.begin()), chunk_queue_.end(),
            [front_frame_seq](const TransportChunk& chunk) {
                return chunk.frame_seq != front_frame_seq;
            });
        return newer_chunk != chunk_queue_.end() || !pending_frames_.empty();
    }

    void drop_front_queued_frame() {
        const uint32_t expired_frame_seq = chunk_queue_.front().frame_seq;
        size_t dropped_chunks            = 0;
        size_t dropped_tail_chunks       = 0;
        while (!chunk_queue_.empty() && chunk_queue_.front().frame_seq == expired_frame_seq) {
            if (!chunk_queue_.front().start_of_frame)
                dropped_tail_chunks++;
            chunk_queue_.pop_front();
            dropped_chunks++;
        }

        stats_.frames_dropped++;
        stats_.expired_frames_dropped++;
        stats_.expired_chunks_dropped += dropped_chunks;
        stats_.old_frame_tail_dropped += dropped_tail_chunks;
    }

    void expire_queued_chunks(uint64_t now_ns) {
        expire_repair_chunks(now_ns);

        while (!chunk_queue_.empty()
               && chunk_age_ns(chunk_queue_.front(), now_ns)
                      > ms_to_ns(transport_config_.soft_deadline_ms)
               && has_newer_work_after_front_frame()) {
            drop_front_queued_frame();
        }

        while (!chunk_queue_.empty()
               && chunk_age_ns(chunk_queue_.front(), now_ns)
                      > ms_to_ns(transport_config_.hard_deadline_ms)) {
            drop_front_queued_frame();
        }
    }

    void expire_repair_chunks(uint64_t now_ns) {
        while (!repair_queue_.empty()
               && chunk_age_ns(repair_queue_.front(), now_ns)
                      > ms_to_ns(transport_config_.hard_deadline_ms)) {
            repair_queue_.pop_front();
            stats_.expired_chunks_dropped++;
        }
    }

    std::expected<void, std::string> maybe_encode_pending(uint64_t now_ns) {
        reset_encode_schedule_if_idle(now_ns);
        expire_queued_chunks(now_ns);
        drop_expired_pending_frames(now_ns);

        while (!pending_frames_.empty()) {
            if (chunk_queue_.size() >= transport_config_.max_queued_chunks)
                return {};
            if (next_encode_deadline_ns_ != 0 && now_ns < next_encode_deadline_ns_)
                return {};

            PendingFrame frame = std::move(pending_frames_.front());
            pending_frames_.pop_front();

            const bool forced_clean_frame = std::exchange(force_clean_frame_pending_, false);
            if (forced_clean_frame)
                encoder_->request_keyframe();

            std::vector<uint8_t> filtered_frame;
            const uint8_t* encode_data = frame.data.data();
            if (bad_link_prefilter_frames_remaining_ > 0) {
                filtered_frame = apply_bad_link_prefilter(frame);
                encode_data    = filtered_frame.data();
                bad_link_prefilter_frames_remaining_--;
            }

            auto push_result = encoder_->push_frame(encode_data, frame.linesize, frame.pts);
            if (!push_result)
                return push_result;
            update_prefilter_reference(frame);

            while (auto packet = encoder_->poll_packet()) {
                if (packet->keyframe) {
                    if (forced_clean_frame)
                        stats_.force_idr_count++;
                    else
                        stats_.scheduled_idr_count++;
                }
                if (auto fragment_result = fragment_packet(*packet, frame.capture_timestamp_ns);
                    !fragment_result) {
                    return std::unexpected(fragment_result.error());
                }
                stats_.frames_encoded++;
            }

            if (next_encode_deadline_ns_ == 0)
                next_encode_deadline_ns_ = now_ns;
            next_encode_deadline_ns_ += min_frame_interval_ns_;
            expire_queued_chunks(now_ns);
            drop_expired_pending_frames(now_ns);
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

            if (fragment_count == 1)
                enqueue_repair_chunk(chunk);
            stats_.chunks_generated++;
            chunk_queue_.push_back(std::move(chunk));
            update_max_queue_depth();
        }

        next_frame_seq_++;
        return {};
    }

    void enqueue_repair_chunk(const TransportChunk& chunk) {
        repair_queue_.clear();
        repair_queue_.push_back(chunk);
        update_max_queue_depth();
    }

    std::optional<TransportChunk> poll_repair_chunk() {
        if (repair_queue_.empty())
            return std::optional<TransportChunk>{};

        auto chunk = std::move(repair_queue_.back());
        repair_queue_.clear();
        record_chunk_emitted(chunk, true);
        return std::optional<TransportChunk>{std::move(chunk)};
    }

    void record_chunk_emitted(const TransportChunk& chunk, bool repair = false) {
        const uint64_t send_age_ns = chunk_age_ns(chunk, transport_config_.now_ns());
        stats_.chunks_emitted++;
        if (repair)
            stats_.repair_chunks_emitted++;
        stats_.bytes_emitted += chunk.data.size();
        stats_.last_send_age_ns = send_age_ns;
        stats_.max_send_age_ns  = std::max(stats_.max_send_age_ns, send_age_ns);
        send_age_samples_ns_.push_back(send_age_ns);
        stats_.p95_send_age_ms = percentile_ms(send_age_samples_ns_, 0.95);
        stats_.p99_send_age_ms = percentile_ms(send_age_samples_ns_, 0.99);
    }

    [[nodiscard]] static uint8_t clamp_u8(int value) {
        return static_cast<uint8_t>(std::clamp(value, 0, 255));
    }

    [[nodiscard]] static int bgr_luma(const uint8_t* pixel) {
        return (static_cast<int>(pixel[0]) + 2 * static_cast<int>(pixel[1])
                + static_cast<int>(pixel[2]))
             / 4;
    }

    [[nodiscard]] bool can_temporal_prefilter(const PendingFrame& frame) const {
        return !prefilter_reference_frame_.empty()
            && prefilter_reference_frame_.size() == frame.data.size()
            && prefilter_reference_linesize_ == frame.linesize;
    }

    [[nodiscard]] std::vector<uint8_t> apply_bad_link_prefilter(const PendingFrame& frame) const {
        std::vector<uint8_t> blended = frame.data;
        const bool temporal_ready    = can_temporal_prefilter(frame);

        if (temporal_ready) {
            constexpr int kTemporalAlphaNum      = 1;
            const int kTemporalAlphaDen          = 1 << transport_config_.prefilter_temporal_shift;
            constexpr int kTemporalLumaThreshold = 12;

            for (int y = 0; y < src_height_; ++y) {
                uint8_t* row = blended.data() + static_cast<size_t>(y) * frame.linesize;
                const uint8_t* current =
                    frame.data.data() + static_cast<size_t>(y) * frame.linesize;
                const uint8_t* reference =
                    prefilter_reference_frame_.data() + static_cast<size_t>(y) * frame.linesize;
                for (int x = 0; x < src_width_; ++x) {
                    uint8_t* dst       = row + static_cast<size_t>(x) * 3;
                    const uint8_t* cur = current + static_cast<size_t>(x) * 3;
                    const uint8_t* ref = reference + static_cast<size_t>(x) * 3;
                    if (std::abs(bgr_luma(cur) - bgr_luma(ref)) > kTemporalLumaThreshold)
                        continue;

                    for (int c = 0; c < 3; ++c) {
                        dst[c] = clamp_u8(
                            (cur[c] * (kTemporalAlphaDen - kTemporalAlphaNum)
                             + ref[c] * kTemporalAlphaNum + kTemporalAlphaDen / 2)
                            / kTemporalAlphaDen);
                    }
                }
            }
        }

        std::vector<uint8_t> filtered = blended;
        for (int y = 0; y < src_height_; ++y) {
            for (int x = 0; x < src_width_; ++x) {
                const uint8_t* center = blended.data() + static_cast<size_t>(y) * frame.linesize
                                      + static_cast<size_t>(x) * 3;
                const int center_luma = bgr_luma(center);
                int cb_sum            = 0;
                int cr_sum            = 0;
                int sample_count      = 0;

                const int neighbor_offsets[5][2] = {
                    { 0,  0},
                    {-1,  0},
                    { 1,  0},
                    { 0, -1},
                    { 0,  1}
                };
                constexpr int kChromaBlurLumaThreshold = 14;
                for (const auto& [dx, dy] : neighbor_offsets) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || nx >= src_width_ || ny < 0 || ny >= src_height_)
                        continue;

                    const uint8_t* sample = blended.data()
                                          + static_cast<size_t>(ny) * frame.linesize
                                          + static_cast<size_t>(nx) * 3;
                    const int sample_luma = bgr_luma(sample);
                    if (std::abs(sample_luma - center_luma) > kChromaBlurLumaThreshold)
                        continue;
                    cb_sum += static_cast<int>(sample[0]) - sample_luma;
                    cr_sum += static_cast<int>(sample[2]) - sample_luma;
                    sample_count++;
                }

                const int cb = cb_sum / std::max(sample_count, 1);
                const int cr = cr_sum / std::max(sample_count, 1);
                const int b  = center_luma + cb;
                const int r  = center_luma + cr;
                const int g  = (4 * center_luma - b - r) / 2;

                uint8_t* dst = filtered.data() + static_cast<size_t>(y) * frame.linesize
                             + static_cast<size_t>(x) * 3;
                dst[0] = clamp_u8(b);
                dst[1] = clamp_u8(g);
                dst[2] = clamp_u8(r);
            }
        }

        return filtered;
    }

    void update_prefilter_reference(const PendingFrame& frame) {
        prefilter_reference_frame_    = frame.data;
        prefilter_reference_linesize_ = frame.linesize;
    }

    EncodeParams encode_params_;
    StreamTransportConfig transport_config_;
    std::optional<StreamEncoder> encoder_;
    StreamSenderStats stats_{};
    std::deque<PendingFrame> pending_frames_;
    std::deque<TransportChunk> chunk_queue_;
    std::deque<TransportChunk> repair_queue_;
    std::vector<uint64_t> send_age_samples_ns_;
    std::vector<uint8_t> prefilter_reference_frame_;
    int src_width_                                = 0;
    int src_height_                               = 0;
    int prefilter_reference_linesize_             = 0;
    uint64_t min_frame_interval_ns_               = 0;
    uint64_t next_encode_deadline_ns_             = 0;
    uint32_t next_frame_seq_                      = 0;
    bool force_clean_frame_pending_               = false;
    uint64_t bad_link_prefilter_frames_remaining_ = 0;

    explicit Impl(EncodeParams encode_params, StreamTransportConfig transport_config)
        : encode_params_(std::move(encode_params))
        , transport_config_(normalize_config(std::move(transport_config))) {}

    std::expected<void, std::string> setup(int src_width, int src_height) {
        if (src_width <= 0 || src_height <= 0)
            return std::unexpected("source dimensions must be positive");
        if (auto valid = validate_transport_config(transport_config_); !valid)
            return valid;

        src_width_             = src_width;
        src_height_            = src_height;
        min_frame_interval_ns_ = std::max<uint64_t>(
            1, static_cast<uint64_t>(
                   std::llround(1'000'000'000.0 / transport_config_.max_stream_fps)));

        auto enc = StreamEncoder::create(
            encode_params_, src_width, src_height, transport_config_.max_stream_fps);
        if (!enc)
            return std::unexpected(std::move(enc.error()));
        encoder_.emplace(std::move(*enc));

        return {};
    }
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
        const uint64_t receive_ns =
            receive_timestamp_ns != 0 ? receive_timestamp_ns : transport_config_.now_ns();

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
            queue_missing_frame_substitutes_before(header.frame_seq, receive_ns);
            start_assembly_from_header(header);
        } else if (header.frame_seq != assembly_.frame_seq) {
            if (sequence_less(header.frame_seq, assembly_.frame_seq)) {
                record_duplicate_or_stale_chunk(header.frame_seq);
                return {};
            }

            drop_active_assembly_with_substitute(receive_ns);

            if (!start) {
                stats_.out_of_order_chunks++;
                return {};
            }

            queue_missing_frame_substitutes_before(header.frame_seq, receive_ns);
            start_assembly_from_header(header);
        } else {
            if (header.fragment_count != assembly_.expected_fragment_count) {
                stats_.out_of_order_chunks++;
                drop_active_assembly_with_substitute(receive_ns);
                return {};
            }

            if (header.fragment_index < assembly_.next_fragment_index) {
                stats_.duplicate_chunks++;
                return {};
            }

            if (header.fragment_index > assembly_.next_fragment_index) {
                stats_.out_of_order_chunks++;
                drop_active_assembly_with_substitute(receive_ns);
                return {};
            }
        }

        const auto payload = bytes.subspan(header.header_size);
        assembly_.bytes.insert(assembly_.bytes.end(), payload.begin(), payload.end());
        assembly_.next_fragment_index++;

        if (!end)
            return {};

        stats_.frames_reassembled++;
        if (elapsed_since_ns(assembly_.capture_timestamp_ns, receive_ns)
            > ms_to_ns(transport_config_.display_deadline_ms)) {
            stats_.display_deadline_drops++;
            stats_.frames_dropped++;
            queue_last_good_substitute(
                assembly_.frame_seq, assembly_.pts, assembly_.capture_timestamp_ns, receive_ns,
                SubstituteReason::DisplayDeadline);
            mark_completed_frame_seq(assembly_.frame_seq);
            reset_assembly();
            return {};
        }

        const size_t queued_frames_before_decode = frame_queue_.size();
        auto decode_result                       = decode_current_frame(receive_ns);
        if (!decode_result) {
            // 解码失败(丢包): 保留 DPB 历史不 flush, 用上一帧替补显示。
            // 后续周期 IDR 或刷新帧会重新收敛。
            stats_.frames_dropped++;
            queue_last_good_substitute(
                assembly_.frame_seq, assembly_.pts, assembly_.capture_timestamp_ns, receive_ns,
                SubstituteReason::DecodeError);
            mark_completed_frame_seq(assembly_.frame_seq);
            reset_assembly();
            return {};
        }
        if (frame_queue_.size() == queued_frames_before_decode) {
            stats_.frames_dropped++;
            stats_.bad_frames++;
            if (!assembly_.keyframe)
                stats_.bad_p_frames++;
            queue_last_good_substitute(
                assembly_.frame_seq, assembly_.pts, assembly_.capture_timestamp_ns, receive_ns,
                SubstituteReason::DependencyDirty);
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

    void record_substitute_reason(SubstituteReason reason) {
        switch (reason) {
        case SubstituteReason::MissingChunk: stats_.substitute_missing_chunk++; break;
        case SubstituteReason::DecodeError: stats_.substitute_decode_error++; break;
        case SubstituteReason::DisplayDeadline: stats_.substitute_display_deadline++; break;
        case SubstituteReason::DependencyDirty: stats_.substitute_dependency_dirty++; break;
        }
    }

    [[nodiscard]] static int luma_at(const DecodedFrame& frame, int x, int y) {
        const uint8_t* pixel = frame.data.data() + static_cast<size_t>(y) * frame.linesize
                             + static_cast<size_t>(x) * 3;
        return (static_cast<int>(pixel[0]) + 2 * static_cast<int>(pixel[1])
                + static_cast<int>(pixel[2]))
             / 4;
    }

    [[nodiscard]] static uint64_t translation_sad(
        const DecodedFrame& previous, const DecodedFrame& current, int dx, int dy,
        uint64_t& sample_count) {
        constexpr int kSampleStep = 4;

        uint64_t sad = 0;
        sample_count = 0;
        for (int y = 0; y < current.height; y += kSampleStep) {
            const int previous_y = y - dy;
            if (previous_y < 0 || previous_y >= previous.height)
                continue;

            for (int x = 0; x < current.width; x += kSampleStep) {
                const int previous_x = x - dx;
                if (previous_x < 0 || previous_x >= previous.width)
                    continue;

                const int diff = luma_at(current, x, y) - luma_at(previous, previous_x, previous_y);
                sad += static_cast<uint64_t>(diff < 0 ? -diff : diff);
                sample_count++;
            }
        }
        return sad;
    }

    [[nodiscard]] static MotionVector
        estimate_translation(const DecodedFrame& previous, const DecodedFrame& current) {
        if (previous.data.empty() || current.data.empty() || previous.width != current.width
            || previous.height != current.height || previous.linesize < previous.width * 3
            || current.linesize < current.width * 3) {
            return {};
        }

        constexpr int kSearchRadius    = 4;
        constexpr uint64_t kMinSamples = 64;

        uint64_t zero_samples   = 0;
        const uint64_t zero_sad = translation_sad(previous, current, 0, 0, zero_samples);
        if (zero_samples < kMinSamples)
            return {};

        uint64_t best_sad     = zero_sad;
        uint64_t best_samples = zero_samples;
        MotionVector best{.dx = 0, .dy = 0, .valid = true};
        for (int dy = -kSearchRadius; dy <= kSearchRadius; ++dy) {
            for (int dx = -kSearchRadius; dx <= kSearchRadius; ++dx) {
                uint64_t samples   = 0;
                const uint64_t sad = translation_sad(previous, current, dx, dy, samples);
                if (samples < kMinSamples)
                    continue;

                if (sad * best_samples < best_sad * samples) {
                    best_sad     = sad;
                    best_samples = samples;
                    best.dx      = dx;
                    best.dy      = dy;
                }
            }
        }

        const double zero_avg = static_cast<double>(zero_sad) / static_cast<double>(zero_samples);
        const double best_avg = static_cast<double>(best_sad) / static_cast<double>(best_samples);
        if ((best.dx != 0 || best.dy != 0) && zero_avg - best_avg < 1.5)
            return {.dx = 0, .dy = 0, .valid = true};

        return best;
    }

    [[nodiscard]] int clamp_motion_shift(int value) const {
        return std::clamp(
            value, -transport_config_.motion_hold_max_px, transport_config_.motion_hold_max_px);
    }

    [[nodiscard]] static int clamp_pixel_coord(int value, int limit) {
        return std::clamp(value, 0, std::max(0, limit - 1));
    }

    [[nodiscard]] DecodedFrame shift_last_good_frame(int dx, int dy) const {
        DecodedFrame shifted = last_good_frame_;
        if (last_good_frame_.data.empty() || (dx == 0 && dy == 0))
            return shifted;

        shifted.data.resize(last_good_frame_.data.size());
        for (int y = 0; y < shifted.height; ++y) {
            const int source_y = clamp_pixel_coord(y - dy, last_good_frame_.height);
            for (int x = 0; x < shifted.width; ++x) {
                const int source_x = clamp_pixel_coord(x - dx, last_good_frame_.width);
                const uint8_t* src = last_good_frame_.data.data()
                                   + static_cast<size_t>(source_y) * last_good_frame_.linesize
                                   + static_cast<size_t>(source_x) * 3;
                uint8_t* dst = shifted.data.data() + static_cast<size_t>(y) * shifted.linesize
                             + static_cast<size_t>(x) * 3;
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            }
        }
        return shifted;
    }

    [[nodiscard]] DecodedFrame make_motion_hold_substitute(uint64_t substitute_index) const {
        const uint64_t max_motion_hold_frames = transport_config_.motion_hold_max_frames;
        if (!global_motion_.valid || substitute_index == 0
            || substitute_index > max_motion_hold_frames) {
            return last_good_frame_;
        }

        const int dx = clamp_motion_shift(
            global_motion_.dx
            * static_cast<int>(std::min(substitute_index, max_motion_hold_frames)));
        const int dy = clamp_motion_shift(
            global_motion_.dy
            * static_cast<int>(std::min(substitute_index, max_motion_hold_frames)));
        return shift_last_good_frame(dx, dy);
    }

    void update_motion_history(const DecodedFrame& frame) {
        if (!last_good_frame_.data.empty())
            global_motion_ = estimate_translation(last_good_frame_, frame);
        last_good_frame_ = frame;
    }

    void queue_last_good_substitute(
        uint32_t frame_seq, int64_t pts, uint64_t capture_timestamp_ns,
        uint64_t decode_timestamp_ns, SubstituteReason reason) {
        if (last_good_frame_.data.empty())
            return;

        record_substitute_reason(reason);
        DecodedFrame substitute         = make_motion_hold_substitute(stats_.freeze_run_length + 1);
        substitute.frame_seq            = frame_seq;
        substitute.pts                  = pts;
        substitute.capture_timestamp_ns = capture_timestamp_ns;
        substitute.decode_timestamp_ns  = decode_timestamp_ns;
        substitute.end_to_end_latency_ns = decode_timestamp_ns >= capture_timestamp_ns
                                             ? decode_timestamp_ns - capture_timestamp_ns
                                             : 0;
        record_display_frame_metrics(substitute, true, 0);
        frame_queue_.push_back(std::move(substitute));
    }

    void record_display_frame_metrics(
        const DecodedFrame& frame, bool substitute, uint64_t decode_elapsed_ns) {
        stats_.frames_displayed++;
        if (substitute) {
            stats_.substituted_frames++;
            if (!recovery_active_) {
                recovery_active_            = true;
                recovery_started_decode_ns_ = frame.decode_timestamp_ns;
                recovery_substitute_frames_ = 0;
            }
            recovery_substitute_frames_++;
            stats_.frames_until_clean = recovery_substitute_frames_;
            stats_.freeze_run_length++;
            stats_.max_freeze_run_length =
                std::max(stats_.max_freeze_run_length, stats_.freeze_run_length);
        } else {
            stats_.frames_decoded++;
            stats_.good_frames++;
            stats_.clean_frames++;
            record_recovery_if_needed(frame);
            stats_.freeze_run_length = 0;
        }

        stats_.latency_samples++;
        stats_.last_latency_ns = frame.end_to_end_latency_ns;
        if (stats_.latency_samples == 1 || frame.end_to_end_latency_ns < stats_.min_latency_ns)
            stats_.min_latency_ns = frame.end_to_end_latency_ns;
        stats_.max_latency_ns = std::max(stats_.max_latency_ns, frame.end_to_end_latency_ns);
        latency_samples_ns_.push_back(frame.end_to_end_latency_ns);
        stats_.p75_latency_ms = percentile_ms(latency_samples_ns_, 0.75);
        stats_.p95_latency_ms = percentile_ms(latency_samples_ns_, 0.95);
        stats_.p99_latency_ms = percentile_ms(latency_samples_ns_, 0.99);

        if (!substitute) {
            decode_samples_ns_.push_back(decode_elapsed_ns);
            stats_.p75_decode_ms = percentile_ms(decode_samples_ns_, 0.75);
            stats_.p95_decode_ms = percentile_ms(decode_samples_ns_, 0.95);
            stats_.p99_decode_ms = percentile_ms(decode_samples_ns_, 0.99);
        }

        if (has_last_display_timestamp_ns_) {
            const uint64_t frame_gap_ns = frame.decode_timestamp_ns >= last_display_timestamp_ns_
                                            ? frame.decode_timestamp_ns - last_display_timestamp_ns_
                                            : 0;
            stats_.last_frame_gap_ns    = frame_gap_ns;
            stats_.max_frame_gap_ns     = std::max(stats_.max_frame_gap_ns, frame_gap_ns);
            frame_gap_samples_ns_.push_back(frame_gap_ns);
            stats_.p95_frame_gap_ms = percentile_ms(frame_gap_samples_ns_, 0.95);
            stats_.p99_frame_gap_ms = percentile_ms(frame_gap_samples_ns_, 0.99);
        }
        has_last_display_timestamp_ns_ = true;
        last_display_timestamp_ns_     = frame.decode_timestamp_ns;
    }

    void record_recovery_if_needed(const DecodedFrame& frame) {
        if (!recovery_active_)
            return;

        const uint64_t recovery_ms =
            elapsed_since_ns(recovery_started_decode_ns_, frame.decode_timestamp_ns);
        stats_.recovery_run_count++;
        stats_.frames_until_clean = recovery_substitute_frames_;
        stats_.max_frames_until_clean =
            std::max(stats_.max_frames_until_clean, recovery_substitute_frames_);
        recovery_frame_samples_.push_back(recovery_substitute_frames_);
        recovery_ms_samples_ns_.push_back(recovery_ms);
        stats_.recovery_frames_p50 = percentile_count(recovery_frame_samples_, 0.50);
        stats_.recovery_frames_p95 = percentile_count(recovery_frame_samples_, 0.95);
        stats_.recovery_frames_p99 = percentile_count(recovery_frame_samples_, 0.99);
        stats_.recovery_ms_p50     = percentile_ms(recovery_ms_samples_ns_, 0.50);
        stats_.recovery_ms_p95     = percentile_ms(recovery_ms_samples_ns_, 0.95);
        stats_.recovery_ms_p99     = percentile_ms(recovery_ms_samples_ns_, 0.99);

        recovery_active_            = false;
        recovery_started_decode_ns_ = 0;
        recovery_substitute_frames_ = 0;
    }

    void mark_completed_frame_seq(uint32_t frame_seq) {
        has_completed_frame_seq_  = true;
        last_completed_frame_seq_ = frame_seq;
    }

    void drop_active_assembly_with_substitute(uint64_t receive_timestamp_ns) {
        if (!assembly_.active)
            return;

        stats_.frames_dropped++;
        stats_.assembly_resets++;
        queue_last_good_substitute(
            assembly_.frame_seq, assembly_.pts, assembly_.capture_timestamp_ns,
            receive_timestamp_ns, SubstituteReason::MissingChunk);
        mark_completed_frame_seq(assembly_.frame_seq);
        reset_assembly();
    }

    void queue_missing_frame_substitutes_before(uint32_t frame_seq, uint64_t receive_timestamp_ns) {
        if (!has_completed_frame_seq_ || last_good_frame_.data.empty())
            return;

        uint32_t missing_seq                    = last_completed_frame_seq_ + 1;
        size_t synthetic_frames                 = 0;
        constexpr size_t kMaxSyntheticGapFrames = 4;
        while (missing_seq != frame_seq && sequence_less(missing_seq, frame_seq)
               && synthetic_frames < kMaxSyntheticGapFrames) {
            stats_.frames_dropped++;
            queue_last_good_substitute(
                missing_seq, last_good_frame_.pts + static_cast<int64_t>(synthetic_frames + 1),
                receive_timestamp_ns, receive_timestamp_ns, SubstituteReason::MissingChunk);
            mark_completed_frame_seq(missing_seq);
            missing_seq++;
            synthetic_frames++;
        }
    }

    std::expected<void, std::string> ensure_decoder_backend() {
        if (decoder_)
            return {};
        auto result = detail::create_optimal_decoder_backend(transport_config_.max_stream_fps);
        if (!result)
            return std::unexpected(std::move(result.error()));
        decoder_ = std::move(*result);
        return {};
    }

    std::expected<void, std::string> decode_current_frame(uint64_t receive_timestamp_ns) {
        (void)receive_timestamp_ns;

        if (auto dec = ensure_decoder_backend(); !dec)
            return dec;

        const uint64_t decode_start_ns = transport_config_.now_ns();
        auto result                    = decoder_->decode(
            std::span<const uint8_t>(assembly_.bytes.data(), assembly_.bytes.size()),
            assembly_.pts);

        if (!result) {
            // Decode pipeline error — don't flush DPB, let subsequent IDR recover.
            stats_.decode_errors++;
            stats_.bad_frames++;
            if (!assembly_.keyframe)
                stats_.bad_p_frames++;
            return std::unexpected(std::move(result.error()));
        }

        auto& decode = *result;

        // Handle corrupt frames detected within the decode pipeline.
        if (decode.had_corrupt_frame) {
            stats_.decode_errors++;
            stats_.bad_frames++;
            if (!assembly_.keyframe)
                stats_.bad_p_frames++;
            stats_.frames_dropped++;
            const uint64_t now_ns = transport_config_.now_ns();
            queue_last_good_substitute(
                assembly_.frame_seq, assembly_.pts, assembly_.capture_timestamp_ns, now_ns,
                SubstituteReason::DecodeError);
        }

        // Process successfully decoded frames.
        for (auto& image : decode.frames) {
            DecodedFrame frame;
            frame.width                 = image.width;
            frame.height                = image.height;
            frame.linesize              = image.linesize;
            frame.data                  = std::move(image.data);
            frame.pts                   = image.pts;
            frame.frame_seq             = assembly_.frame_seq;
            frame.capture_timestamp_ns  = assembly_.capture_timestamp_ns;
            frame.decode_timestamp_ns   = transport_config_.now_ns();
            frame.end_to_end_latency_ns = frame.decode_timestamp_ns >= frame.capture_timestamp_ns
                                            ? frame.decode_timestamp_ns - frame.capture_timestamp_ns
                                            : 0;

            if (assembly_.keyframe)
                stats_.idr_frames++;

            update_motion_history(frame);

            const uint64_t decode_elapsed_ns = frame.decode_timestamp_ns >= decode_start_ns
                                                 ? frame.decode_timestamp_ns - decode_start_ns
                                                 : 0;
            record_display_frame_metrics(frame, false, decode_elapsed_ns);
            frame_queue_.push_back(std::move(frame));
        }

        return {};
    }

    StreamTransportConfig transport_config_;
    StreamReceiverStats stats_{};
    AssemblyState assembly_{};
    std::unique_ptr<detail::DecoderBackend> decoder_;
    std::vector<uint64_t> latency_samples_ns_;
    std::vector<uint64_t> decode_samples_ns_;
    std::vector<uint64_t> frame_gap_samples_ns_;
    std::vector<uint64_t> recovery_frame_samples_;
    std::vector<uint64_t> recovery_ms_samples_ns_;
    std::deque<DecodedFrame> frame_queue_;
    bool recovery_active_                = false;
    uint64_t recovery_started_decode_ns_ = 0;
    uint64_t recovery_substitute_frames_ = 0;
    bool has_last_display_timestamp_ns_  = false;
    uint64_t last_display_timestamp_ns_  = 0;
    bool has_completed_frame_seq_        = false;
    uint32_t last_completed_frame_seq_   = 0;
    MotionVector global_motion_{};
    DecodedFrame last_good_frame_; // 最近一次成功解码帧, 丢包时作为替补
};

StreamSender::StreamSender(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

StreamSender::~StreamSender() = default;

StreamSender::StreamSender(StreamSender&&) noexcept            = default;
StreamSender& StreamSender::operator=(StreamSender&&) noexcept = default;

std::expected<StreamSender, std::string> StreamSender::create(
    EncodeParams encode_params, StreamTransportConfig transport_config, int src_width,
    int src_height) noexcept {
    auto impl =
        Impl::create(std::move(encode_params), std::move(transport_config), src_width, src_height);
    if (!impl)
        return std::unexpected(std::move(impl.error()));
    return StreamSender(std::move(*impl));
}

std::expected<void, std::string> StreamSender::submit_frame(
    const uint8_t* data, int linesize, int64_t pts, uint64_t capture_timestamp_ns) noexcept {
    return impl_->submit_frame(data, linesize, pts, capture_timestamp_ns);
}

std::expected<std::optional<TransportChunk>, std::string> StreamSender::poll_chunk() noexcept {
    return impl_->poll_chunk();
}

void StreamSender::notify_bad_link() noexcept { impl_->notify_bad_link(); }

void StreamSender::request_clean_frame() noexcept { impl_->request_clean_frame(); }

[[nodiscard]] StreamSenderStats StreamSender::stats() const { return impl_->stats(); }

[[nodiscard]] std::pair<int, int> StreamSender::dimensions() const { return impl_->dimensions(); }

StreamReceiver::StreamReceiver(StreamTransportConfig transport_config)
    : impl_(std::make_unique<Impl>(std::move(transport_config))) {}

StreamReceiver::~StreamReceiver() = default;

StreamReceiver::StreamReceiver(StreamReceiver&&) noexcept            = default;
StreamReceiver& StreamReceiver::operator=(StreamReceiver&&) noexcept = default;

std::expected<void, std::string> StreamReceiver::push_chunk(
    std::span<const uint8_t> bytes, uint64_t receive_timestamp_ns) noexcept {
    return impl_->push_chunk(bytes, receive_timestamp_ns);
}

std::optional<DecodedFrame> StreamReceiver::poll_frame() { return impl_->poll_frame(); }

[[nodiscard]] StreamReceiverStats StreamReceiver::stats() const { return impl_->stats(); }

} // namespace quanta
