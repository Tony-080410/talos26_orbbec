#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

// Forward declarations — keep FFmpeg out of the public header
extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;
struct AVStream;
}

namespace quanta {

// ---------------------------------------------------------------------------
// Pre-processing filter pipeline: denoise → sharpen → scale → encode
// ---------------------------------------------------------------------------
struct FilterParams {
    bool enable = true;
    // Saturation: applied after scale, before denoise. 1.0=full, 0.75=75%.
    double saturation = 0.9;
    // hqdn3d — extreme spatial denoise: annihilate ALL texture, leave only structure.
    // Zero temporal: robot camera is mostly static between frames, temporal denoise adds latency.
    double denoise_luma   = 0.0;
    double denoise_chroma = 0.0;
    double denoise_tl     = 0.0;
    double denoise_tc     = 0.0;
    // unsharp — recover edges lost by denoising; chroma edges for color boundary, luma stays soft
    double sharpen1_luma   = 1.0;
    double sharpen1_chroma = 1.8;
    double sharpen2_luma   = 0.0;
    double sharpen2_chroma = 0.0;
    double sharpen3_luma   = 0.0;
    double sharpen3_chroma = 0.7;
};

// ---------------------------------------------------------------------------
// Encode parameters for ultra-low bitrate transmission
// ---------------------------------------------------------------------------
struct EncodeParams {
    int target_bitrate = 40'500; // bps
    int max_width      = 200;    // auto-downscale if wider
    int max_height     = 200;    // auto-downscale if taller
    int gop_size       = 5;      // short recovery period; avoid long P-frame corruption chains
    int framerate      = 30;     // target output framerate
    std::string preset = "fast"; // keep the original rate allocation baseline
    std::string tune   = "ssim";
    bool intra_refresh = false;  // periodic IDR lets a lossy stream recover without feedback

    // CRF + VBV rate control (original working approach)
    int crf              = 52;  // quality target; VBV caps actual bitrate
    int lookahead        = 0;   // realtime transport must emit immediately
    int chroma_qp_offset = 0;   // base chroma penalty; encoder internally protects Cr over Cb
    double aq_strength   = 0.4; // max AQ — pump bits into edge macroblocks
    double psy_rd        = 0.0; // extreme: dynamic→outlines only, discard all texture
    double psy_trellis   = 0.0; // match: edge-biased quantization, flatten interiors

    FilterParams filter{};
};

// ---------------------------------------------------------------------------
// Transcode statistics
// ---------------------------------------------------------------------------
struct TranscodeStats {
    int64_t frames_in   = 0;
    int64_t frames_out  = 0;
    int64_t bytes_out   = 0;
    double duration_sec = 0.0;
    double avg_bitrate  = 0.0; // kbps
};

// ---------------------------------------------------------------------------
// Transcoder: input video -> HEVC ultra-low bitrate output
// ---------------------------------------------------------------------------
class Transcoder {
public:
    explicit Transcoder(EncodeParams params = {});
    ~Transcoder() = default;

    Transcoder(const Transcoder&)            = delete;
    Transcoder& operator=(const Transcoder&) = delete;
    Transcoder(Transcoder&&)                 = default;
    Transcoder& operator=(Transcoder&&)      = default;

    /// Run full transcode: input_path -> output_path
    /// Returns statistics on success, error message on failure.
    std::expected<TranscodeStats, std::string>
        run(std::string_view input_path, std::string_view output_path) const;

    const EncodeParams& params() const { return params_; }

private:
    EncodeParams params_;
};

} // namespace quanta
