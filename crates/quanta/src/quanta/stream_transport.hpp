#pragma once

#include "quanta/transcoder.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace quanta {

using StreamClockFn = std::function<uint64_t()>;

struct StreamTransportConfig {
    size_t mtu_bytes   = 300;
    int transport_hz   = 50;
    int max_stream_fps = 30;
    StreamClockFn now_ns{};
    size_t max_pending_frames               = 1;
    size_t max_queued_chunks                = 4;
    uint64_t soft_deadline_ms               = 80;
    uint64_t hard_deadline_ms               = 120;
    uint64_t display_deadline_ms            = 150;
    uint64_t clean_request_freeze_threshold = 3;
    uint64_t clean_request_cooldown_ms      = 300;
    uint64_t clean_request_window_ms        = 1'000;
    uint64_t max_clean_requests_per_window  = 3;
    uint64_t motion_hold_max_frames         = 3;
    int motion_hold_max_px                  = 6;
    uint64_t prefilter_window_frames        = 6;
    int prefilter_temporal_shift            = 4;
};

struct TransportChunk {
    std::vector<uint8_t> data;
    uint32_t frame_seq            = 0;
    uint8_t fragment_index        = 0;
    uint8_t fragment_count        = 0;
    int64_t pts                   = 0;
    uint64_t capture_timestamp_ns = 0;
    bool keyframe                 = false;
    bool start_of_frame           = false;
    bool end_of_frame             = false;

    [[nodiscard]] std::span<const uint8_t> bytes() const noexcept { return data; }
};

struct DecodedFrame {
    std::vector<uint8_t> data;
    int width                      = 0;
    int height                     = 0;
    int linesize                   = 0;
    int64_t pts                    = 0;
    uint32_t frame_seq             = 0;
    uint64_t capture_timestamp_ns  = 0;
    uint64_t decode_timestamp_ns   = 0;
    uint64_t end_to_end_latency_ns = 0;

    [[nodiscard]] std::span<const uint8_t> bytes() const noexcept { return data; }
};

struct StreamSenderStats {
    uint64_t frames_submitted       = 0;
    uint64_t frames_encoded         = 0;
    uint64_t frames_dropped         = 0;
    uint64_t expired_frames_dropped = 0;
    uint64_t chunks_generated       = 0;
    uint64_t chunks_emitted         = 0;
    uint64_t repair_chunks_emitted  = 0;
    uint64_t force_idr_requests     = 0;
    uint64_t force_idr_count        = 0;
    uint64_t scheduled_idr_count    = 0;
    uint64_t expired_chunks_dropped = 0;
    uint64_t old_frame_tail_dropped = 0;
    uint64_t bytes_emitted          = 0;
    uint64_t last_send_age_ns       = 0;
    uint64_t max_send_age_ns        = 0;
    double p95_send_age_ms          = 0.0;
    double p99_send_age_ms          = 0.0;
    double chunks_per_frame         = 0.0;
    size_t queued_chunks            = 0;
    size_t max_queued_chunks_seen   = 0;
    bool has_pending_frame          = false;
};

struct StreamReceiverStats {
    uint64_t chunks_received             = 0;
    uint64_t bytes_received              = 0;
    uint64_t duplicate_chunks            = 0;
    uint64_t stale_chunks                = 0;
    uint64_t out_of_order_chunks         = 0;
    uint64_t frames_reassembled          = 0;
    uint64_t frames_displayed            = 0;
    uint64_t frames_decoded              = 0;
    uint64_t good_frames                 = 0;
    uint64_t clean_frames                = 0;
    uint64_t bad_frames                  = 0;
    uint64_t bad_p_frames                = 0;
    uint64_t idr_frames                  = 0;
    uint64_t substituted_frames          = 0;
    uint64_t substitute_missing_chunk    = 0;
    uint64_t substitute_decode_error     = 0;
    uint64_t substitute_display_deadline = 0;
    uint64_t substitute_dependency_dirty = 0;
    uint64_t display_deadline_drops      = 0;
    uint64_t frames_dropped              = 0;
    uint64_t assembly_resets             = 0;
    uint64_t decode_errors               = 0;
    uint64_t latency_samples             = 0;
    uint64_t last_latency_ns             = 0;
    uint64_t min_latency_ns              = 0;
    uint64_t max_latency_ns              = 0;
    uint64_t freeze_run_length           = 0;
    uint64_t max_freeze_run_length       = 0;
    uint64_t recovery_run_count          = 0;
    uint64_t frames_until_clean          = 0;
    uint64_t max_frames_until_clean      = 0;
    uint64_t last_frame_gap_ns           = 0;
    uint64_t max_frame_gap_ns            = 0;
    double p75_latency_ms                = 0.0;
    double p95_latency_ms                = 0.0;
    double p99_latency_ms                = 0.0;
    double p75_decode_ms                 = 0.0;
    double p95_decode_ms                 = 0.0;
    double p99_decode_ms                 = 0.0;
    double recovery_frames_p50           = 0.0;
    double recovery_frames_p95           = 0.0;
    double recovery_frames_p99           = 0.0;
    double recovery_ms_p50               = 0.0;
    double recovery_ms_p95               = 0.0;
    double recovery_ms_p99               = 0.0;
    double p95_frame_gap_ms              = 0.0;
    double p99_frame_gap_ms              = 0.0;
    size_t queued_frames                 = 0;
    bool assembling_frame                = false;
};

class StreamSender {
public:
    /// Factory: construct a fully initialized sender.
    /// Returns the sender on success, or an error string on failure.
    [[nodiscard]] static std::expected<StreamSender, std::string> create(
        EncodeParams encode_params, StreamTransportConfig transport_config, int src_width,
        int src_height) noexcept;

    ~StreamSender();

    StreamSender(StreamSender&&) noexcept;
    StreamSender& operator=(StreamSender&&) noexcept;
    StreamSender(const StreamSender&)            = delete;
    StreamSender& operator=(const StreamSender&) = delete;

    std::expected<void, std::string> submit_frame(
        const uint8_t* data, int linesize, int64_t pts, uint64_t capture_timestamp_ns = 0) noexcept;

    std::expected<std::optional<TransportChunk>, std::string> poll_chunk() noexcept;

    void notify_bad_link() noexcept;
    void request_clean_frame() noexcept;

    [[nodiscard]] StreamSenderStats stats() const;
    [[nodiscard]] std::pair<int, int> dimensions() const;

private:
    class Impl;
    explicit StreamSender(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class StreamReceiver {
public:
    explicit StreamReceiver(StreamTransportConfig transport_config = {});
    ~StreamReceiver();

    StreamReceiver(StreamReceiver&&) noexcept;
    StreamReceiver& operator=(StreamReceiver&&) noexcept;
    StreamReceiver(const StreamReceiver&)            = delete;
    StreamReceiver& operator=(const StreamReceiver&) = delete;

    std::expected<void, std::string>
        push_chunk(std::span<const uint8_t> bytes, uint64_t receive_timestamp_ns = 0) noexcept;

    std::optional<DecodedFrame> poll_frame();

    [[nodiscard]] StreamReceiverStats stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace quanta
