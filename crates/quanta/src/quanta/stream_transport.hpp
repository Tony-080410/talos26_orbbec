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
    size_t max_pending_frames = 1;
    size_t max_queued_chunks  = 50;
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
    uint64_t frames_submitted = 0;
    uint64_t frames_encoded   = 0;
    uint64_t frames_dropped   = 0;
    uint64_t chunks_emitted   = 0;
    uint64_t bytes_emitted    = 0;
    size_t queued_chunks      = 0;
    bool has_pending_frame    = false;
};

struct StreamReceiverStats {
    uint64_t chunks_received     = 0;
    uint64_t bytes_received      = 0;
    uint64_t duplicate_chunks    = 0;
    uint64_t stale_chunks        = 0;
    uint64_t out_of_order_chunks = 0;
    uint64_t frames_reassembled  = 0;
    uint64_t frames_decoded      = 0;
    uint64_t frames_dropped      = 0;
    uint64_t assembly_resets     = 0;
    uint64_t decode_errors       = 0;
    uint64_t latency_samples     = 0;
    uint64_t last_latency_ns     = 0;
    uint64_t min_latency_ns      = 0;
    uint64_t max_latency_ns      = 0;
    double p75_latency_ms        = 0.0;
    double p75_decode_ms         = 0.0;
    size_t queued_frames         = 0;
    bool assembling_frame        = false;
};

class StreamSender {
public:
    explicit StreamSender(
        EncodeParams encode_params = {}, StreamTransportConfig transport_config = {});
    ~StreamSender();

    StreamSender(StreamSender&&) noexcept;
    StreamSender& operator=(StreamSender&&) noexcept;
    StreamSender(const StreamSender&)            = delete;
    StreamSender& operator=(const StreamSender&) = delete;

    std::expected<void, std::string> init(int src_width, int src_height);

    std::expected<void, std::string> submit_frame(
        const uint8_t* data, int linesize, int64_t pts, uint64_t capture_timestamp_ns = 0);

    std::expected<std::optional<TransportChunk>, std::string> poll_chunk();

    [[nodiscard]] StreamSenderStats stats() const;
    [[nodiscard]] std::pair<int, int> dimensions() const;

private:
    class Impl;
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
        push_chunk(std::span<const uint8_t> bytes, uint64_t receive_timestamp_ns = 0);

    std::optional<DecodedFrame> poll_frame();

    [[nodiscard]] StreamReceiverStats stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace quanta
