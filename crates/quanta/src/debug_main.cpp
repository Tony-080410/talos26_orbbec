#include "quanta/stream_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <vector>

#ifdef _WIN32
# include <io.h>
#else
# include <unistd.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// Color bar test pattern generator (BGR24)
// ---------------------------------------------------------------------------

// SMPTE-style color bars: 8 vertical bars
constexpr uint8_t kBarColors[8][3] = {
    {192, 192, 192}, // white
    {192, 192,   0}, // yellow
    {  0, 192, 192}, // cyan
    {  0, 192,   0}, // green
    {192,   0, 192}, // magenta
    {192,   0,   0}, // red
    {  0,   0, 192}, // blue
    {  0,   0,   0}, // black
};

void generate_color_bars(uint8_t* data, int width, int height, int linesize, int frame_idx) {
    const int bar_width = width / 8;

    for (int y = 0; y < height; ++y) {
        uint8_t* row = data + y * linesize;

        // Top 2/3: color bars
        if (y < height * 2 / 3) {
            for (int x = 0; x < width; ++x) {
                int bar_idx    = std::min(x / bar_width, 7);
                row[x * 3 + 0] = kBarColors[bar_idx][0]; // B
                row[x * 3 + 1] = kBarColors[bar_idx][1]; // G
                row[x * 3 + 2] = kBarColors[bar_idx][2]; // R
            }
        }
        // Middle 1/6: blue gradient (shows luminance depth)
        else if (y < height * 5 / 6) {
            for (int x = 0; x < width; ++x) {
                double t = static_cast<double>(x) / width;
                // Animated gradient: shifts per frame
                double phase   = t * 4.0 + static_cast<double>(frame_idx) * 0.1;
                uint8_t val    = static_cast<uint8_t>(128 + 127 * std::sin(phase));
                row[x * 3 + 0] = val;     // B: strong
                row[x * 3 + 1] = val / 3; // G: dim
                row[x * 3 + 2] = val / 2; // R: medium
            }
        }
        // Bottom 1/6: dark gradient with moving white block (tests edge sharpness)
        else {
            for (int x = 0; x < width; ++x) {
                double t   = static_cast<double>(x) / width;
                uint8_t bg = static_cast<uint8_t>(40 + 20 * t);

                // Animated white block
                int block_x       = (frame_idx * 3) % (width + 40) - 20;
                int block_w       = 30;
                int block_h       = 15;
                int block_y_start = height * 5 / 6;

                bool in_block = x >= block_x && x < block_x + block_w && y >= block_y_start + 5
                             && y < block_y_start + 5 + block_h;

                row[x * 3 + 0] = in_block ? 220 : bg;
                row[x * 3 + 1] = in_block ? 220 : bg;
                row[x * 3 + 2] = in_block ? 220 : bg;
            }
        }

        // Draw frame counter in top-left (crude 3x5 pixel font)
        // Just draw a small white rectangle that blinks per frame
        int blink = (frame_idx / 5) % 2;
        if (blink && y < 10) {
            for (int x = 2; x < 8; ++x) {
                row[x * 3 + 0] = 0;
                row[x * 3 + 1] = 0;
                row[x * 3 + 2] = 255; // Red indicator dot
            }
        }
    }
}

// Simple RAII file writer
struct FileWriter {
    int fd = -1;
    explicit FileWriter(const char* path) {
#ifdef _WIN32
        fd = _open(path, _O_WRONLY | _O_CREAT | _O_TRUNC, _S_IWRITE);
#else
        fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
    }
    ~FileWriter() {
        if (fd >= 0)
#ifdef _WIN32
            _close(fd);
#else
            close(fd);
#endif
    }
    bool valid() const { return fd >= 0; }
    bool write(const uint8_t* data, size_t size) {
        if (fd < 0)
            return false;
        size_t written = 0;
        while (written < size) {
            auto n =
#ifdef _WIN32
                _write(fd, data + written, static_cast<unsigned>(size - written));
#else
                ::write(fd, data + written, size - written);
#endif
            if (n <= 0)
                return false;
            written += static_cast<size_t>(n);
        }
        return true;
    }
};

} // namespace

int main(int argc, char* argv[]) {
    const char* output_path = "test_stream.h265";
    int num_frames          = 300; // 10 seconds at 30fps
    int src_width           = 640;
    int src_height          = 480;
    int framerate           = 30;

    if (argc >= 2) {
        output_path = argv[1];
    }
    if (argc >= 3) {
        num_frames = std::atoi(argv[2]);
    }

    // ---- Setup encoder ----
    quanta::EncodeParams params;
    // Use the shared library defaults so the debug path matches Transcoder behavior.
    quanta::StreamEncoder encoder(params);

    auto init_result = encoder.init(src_width, src_height, framerate);
    if (!init_result) {
        std::fprintf(stderr, "[debug] Init failed: %s\n", init_result.error().c_str());
        return 1;
    }

    auto [out_w, out_h] = encoder.dimensions();
    std::fprintf(
        stderr, "[debug] Encoding %d frames: %dx%d @ %dfps -> %dx%d, %d bps target, preset=%s\n",
        num_frames, src_width, src_height, framerate, out_w, out_h, params.target_bitrate,
        params.preset);

    // ---- Allocate frame buffer ----
    const int linesize = src_width * 3;
    std::vector<uint8_t> frame_buf(static_cast<size_t>(linesize * src_height));

    // ---- Open output ----
    FileWriter writer(output_path);
    if (!writer.valid()) {
        std::fprintf(stderr, "[debug] Cannot open output: %s\n", output_path);
        return 1;
    }

    // ---- Encode loop ----
    int64_t total_bytes      = 0;
    int total_packets        = 0;
    double total_encode_ms   = 0.0;
    double total_generate_ms = 0.0;
    double total_poll_ms     = 0.0;
    int frames_encoded       = 0;

    for (int i = 0; i < num_frames; ++i) {
        // Generate test pattern
        auto t_gen0 = std::chrono::high_resolution_clock::now();
        generate_color_bars(frame_buf.data(), src_width, src_height, linesize, i);
        auto t_gen1 = std::chrono::high_resolution_clock::now();

        // Encode
        auto t0 = std::chrono::high_resolution_clock::now();

        auto push_result = encoder.push_frame(frame_buf.data(), linesize, i);
        if (!push_result) {
            std::fprintf(
                stderr, "[debug] Push frame %d failed: %s\n", i, push_result.error().c_str());
            return 1;
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        // Drain packets
        auto t_poll0 = std::chrono::high_resolution_clock::now();
        while (auto pkt = encoder.poll_packet()) {
            auto bytes = pkt->bytes();
            if (!writer.write(bytes.data(), bytes.size())) {
                std::fprintf(stderr, "[debug] Write failed\n");
                return 1;
            }
            total_bytes += static_cast<int64_t>(bytes.size());
            total_packets++;
        }
        auto t_poll1 = std::chrono::high_resolution_clock::now();

        double encode_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double generate_ms = std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count();
        double poll_ms     = std::chrono::duration<double, std::milli>(t_poll1 - t_poll0).count();
        total_generate_ms += generate_ms;
        total_encode_ms += encode_ms;
        total_poll_ms += poll_ms;
        frames_encoded++;

        // Progress indicator every 30 frames
        if ((i + 1) % 30 == 0) {
            double elapsed_sec  = static_cast<double>(i + 1) / framerate;
            double current_kbps = elapsed_sec > 0 ? (total_bytes * 8.0 / elapsed_sec / 1000.0) : 0;
            std::fprintf(
                stderr,
                "[debug] Frame %d/%d | gen %.1f ms | push %.1f ms | poll %.1f ms | %lld bytes | "
                "%.1f kbps\n",
                i + 1, num_frames, total_generate_ms / frames_encoded,
                total_encode_ms / frames_encoded, total_poll_ms / frames_encoded,
                static_cast<long long>(total_bytes), current_kbps);
        }
    }

    // ---- Flush ----
    auto flush_result = encoder.flush();
    if (!flush_result) {
        std::fprintf(stderr, "[debug] Flush failed: %s\n", flush_result.error().c_str());
        return 1;
    }
    while (auto pkt = encoder.poll_packet()) {
        auto bytes = pkt->bytes();
        if (!writer.write(bytes.data(), bytes.size())) {
            std::fprintf(stderr, "[debug] Write failed during flush\n");
            return 1;
        }
        total_bytes += static_cast<int64_t>(bytes.size());
        total_packets++;
    }

    // ---- Stats ----
    double duration_sec = static_cast<double>(num_frames) / framerate;
    double avg_kbps     = duration_sec > 0 ? (total_bytes * 8.0 / duration_sec / 1000.0) : 0;
    double avg_push_ms  = frames_encoded > 0 ? (total_encode_ms / frames_encoded) : 0;
    double avg_gen_ms   = frames_encoded > 0 ? (total_generate_ms / frames_encoded) : 0;
    double avg_poll_ms  = frames_encoded > 0 ? (total_poll_ms / frames_encoded) : 0;

    std::fprintf(stderr, "\n[debug] === Results ===\n");
    std::fprintf(stderr, "  Output:     %s\n", output_path);
    std::fprintf(stderr, "  Frames:     %d encoded, %d packets\n", frames_encoded, total_packets);
    std::fprintf(stderr, "  Duration:   %.2f sec\n", duration_sec);
    std::fprintf(stderr, "  Size:       %.1f KB\n", static_cast<double>(total_bytes) / 1024.0);
    std::fprintf(stderr, "  Avg bitrate: %.1f kbps (target %d)\n", avg_kbps, params.target_bitrate);
    std::fprintf(stderr, "  Avg generate: %.1f ms/frame\n", avg_gen_ms);
    std::fprintf(
        stderr, "  Avg push:     %.1f ms/frame (budget %.1f ms for %d fps)\n", avg_push_ms,
        1000.0 / framerate, framerate);
    std::fprintf(stderr, "  Avg poll:     %.1f ms/frame\n", avg_poll_ms);
    std::fprintf(stderr, "\n[debug] Play with: ffplay %s\n", output_path);

    return 0;
}
