#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace quanta::detail {

// ---------------------------------------------------------------------------
// BGR24 decoded video frame produced by DecoderBackend
// ---------------------------------------------------------------------------
struct DecodedImage {
    std::vector<uint8_t> data; // BGR24 packed
    int width    = 0;
    int height   = 0;
    int linesize = 0;
    int64_t pts  = 0;

    [[nodiscard]] std::span<const uint8_t> bytes() const noexcept { return data; }
};

// ---------------------------------------------------------------------------
// Result of decoding a single HEVC access unit
// ---------------------------------------------------------------------------
struct DecodeResult {
    std::vector<DecodedImage> frames;
    bool had_decode_error  = false; // send_packet or receive_frame returned error
    bool had_corrupt_frame = false; // AV_FRAME_FLAG_CORRUPT detected, frame skipped
};

// ---------------------------------------------------------------------------
// Abstract HEVC video decoder backend.
// Implementations: FFmpeg (libavcodec), future hardware backends.
// The StreamReceiver delegates all decode work to a backend instance.
// ---------------------------------------------------------------------------
class DecoderBackend {
public:
    virtual ~DecoderBackend() = default;

    /// Decode a single HEVC access unit into zero or more BGR24 frames.
    /// \param encoded_bytes  Annex-B encoded HEVC data
    /// \param pts            Presentation timestamp for the access unit
    /// \return Decoded frames on success, error string on failure.
    virtual std::expected<DecodeResult, std::string>
        decode(std::span<const uint8_t> encoded_bytes, int64_t pts) noexcept = 0;
};

// ---------------------------------------------------------------------------
// Backend factories
// ---------------------------------------------------------------------------

/// Create FFmpeg/libavcodec HEVC decoder backend (always available).
[[nodiscard]] std::expected<std::unique_ptr<DecoderBackend>, std::string>
    create_ffmpeg_decoder_backend(int target_fps) noexcept;

/// Create the optimal decoder backend for the current platform.
[[nodiscard]] inline std::expected<std::unique_ptr<DecoderBackend>, std::string>
    create_optimal_decoder_backend(int target_fps) noexcept {
    // Future: hardware decoder backends (Axera VDEC, VideoToolbox, etc.)
    return create_ffmpeg_decoder_backend(target_fps);
}

} // namespace quanta::detail
