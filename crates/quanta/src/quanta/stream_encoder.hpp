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
        create(EncodeParams params, int src_width, int src_height, int framerate) noexcept;

    ~StreamEncoder();

    StreamEncoder(StreamEncoder&&) noexcept;
    StreamEncoder& operator=(StreamEncoder&&) noexcept;
    StreamEncoder(const StreamEncoder&)            = delete;
    StreamEncoder& operator=(const StreamEncoder&) = delete;

    std::expected<void, std::string>
        push_frame(const uint8_t* data, int linesize, int64_t pts) noexcept;

    void request_keyframe() noexcept;

    std::optional<EncodedPacket> poll_packet();

    std::expected<void, std::string> flush() noexcept;

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
