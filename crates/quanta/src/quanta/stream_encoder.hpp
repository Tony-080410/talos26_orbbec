#pragma once

#include "quanta/transcoder.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace quanta {

// ---------------------------------------------------------------------------
// Encoded HEVC access unit (Annex B format)
// ---------------------------------------------------------------------------
struct EncodedPacket {
    std::unique_ptr<uint8_t[]> data; // Owned byte buffer
    size_t size   = 0;
    int64_t pts   = 0;
    bool keyframe = false;

    EncodedPacket()                                    = default;
    EncodedPacket(EncodedPacket&&) noexcept            = default;
    EncodedPacket& operator=(EncodedPacket&&) noexcept = default;
    EncodedPacket(const EncodedPacket&)                = delete;
    EncodedPacket& operator=(const EncodedPacket&)     = delete;

    /// Access encoded bytes for transmission
    [[nodiscard]] std::span<const uint8_t> bytes() const noexcept { return {data.get(), size}; }
};

// ---------------------------------------------------------------------------
// StreamEncoder: realtime HEVC encoding with BGR24 frame input
// ---------------------------------------------------------------------------
class StreamEncoder {
public:
    /// Factory: construct a fully initialized encoder.
    /// Returns the encoder on success, or an error string on failure.
    [[nodiscard]] static std::expected<StreamEncoder, std::string>
        create(EncodeParams params, int src_width, int src_height, int framerate);

    ~StreamEncoder();

    StreamEncoder(StreamEncoder&&) noexcept;
    StreamEncoder& operator=(StreamEncoder&&) noexcept;
    StreamEncoder(const StreamEncoder&)            = delete;
    StreamEncoder& operator=(const StreamEncoder&) = delete;

    /// Push a raw BGR24 frame into the encoder.
    /// `data` points to BGR24 interleaved pixel data, `linesize` is the byte stride per row
    /// (typically width * 3, may be larger if padded).
    /// Data must remain valid until the next call to push_frame.
    /// May produce 0 or more packets — call poll_packet() to retrieve them.
    std::expected<void, std::string> push_frame(const uint8_t* data, int linesize, int64_t pts);

    /// Retrieve next encoded packet.
    /// Returns nullopt if no packet is available yet.
    std::optional<EncodedPacket> poll_packet();

    /// Flush the encoder: drain all remaining frames and push packets to queue.
    std::expected<void, std::string> flush();

    /// Get encoding parameters
    [[nodiscard]] const EncodeParams& params() const;

    /// Get output dimensions
    [[nodiscard]] std::pair<int, int> dimensions() const;

private:
    class Impl;
    explicit StreamEncoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace quanta
