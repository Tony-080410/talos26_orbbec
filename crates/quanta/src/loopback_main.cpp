#include "quanta/stream_transport.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint8_t kBarColors[8][3] = {
    {192, 192, 192},
    {192, 192,   0},
    {  0, 192, 192},
    {  0, 192,   0},
    {192,   0, 192},
    {192,   0,   0},
    {  0,   0, 192},
    {  0,   0,   0},
};

void generate_color_bars(uint8_t* data, int width, int height, int linesize, int frame_idx) {
    const int bar_width = std::max(1, width / 8);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = data + y * linesize;
        for (int x = 0; x < width; ++x) {
            const int bar_idx = std::min(x / bar_width, 7);
            row[x * 3 + 0]    = kBarColors[bar_idx][0];
            row[x * 3 + 1]    = kBarColors[bar_idx][1];
            row[x * 3 + 2]    = kBarColors[bar_idx][2];
        }

        const int block_x = (frame_idx * 7) % (width + 48) - 24;
        if (y >= height / 2 - 10 && y < height / 2 + 10) {
            for (int x = std::max(0, block_x); x < std::min(width, block_x + 48); ++x) {
                row[x * 3 + 0] = 255;
                row[x * 3 + 1] = 255;
                row[x * 3 + 2] = 255;
            }
        }
    }
}

double percentile_ms(std::vector<double> values, double q) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const size_t idx =
        static_cast<size_t>(std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1));
    return values[idx];
}

uint64_t real_now_ns() {
    const auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}

template <class Fn>
uint64_t run_and_measure_ns(Fn&& fn) {
    const uint64_t start = real_now_ns();
    fn();
    return real_now_ns() - start;
}

} // namespace

int main(int argc, char* argv[]) {
    int input_frames = 150;
    if (argc >= 2)
        input_frames = std::atoi(argv[1]);

    constexpr int kSrcWidth         = 640;
    constexpr int kSrcHeight        = 480;
    constexpr int kInputFps         = 50;
    constexpr uint64_t kOneSecondNs = 1'000'000'000ULL;

    uint64_t virtual_now_ns = real_now_ns();
    auto clock_fn           = [&virtual_now_ns] { return virtual_now_ns; };

    quanta::StreamTransportConfig transport_config{
        .mtu_bytes      = 300,
        .transport_hz   = 50,
        .max_stream_fps = 30,
        .now_ns         = clock_fn,
    };
    quanta::EncodeParams encode_params;
    auto sender_result =
        quanta::StreamSender::create(encode_params, transport_config, kSrcWidth, kSrcHeight);
    if (!sender_result) {
        std::fprintf(
            stderr, "[loopback] sender create failed: %s\n", sender_result.error().c_str());
        return 1;
    }
    quanta::StreamSender sender = std::move(*sender_result);
    quanta::StreamReceiver receiver(transport_config);

    const auto [out_w, out_h]          = sender.dimensions();
    const uint64_t capture_period_ns   = kOneSecondNs / kInputFps;
    const uint64_t transport_period_ns = kOneSecondNs / transport_config.transport_hz;

    const int linesize = kSrcWidth * 3;
    std::vector<uint8_t> frame_buf(static_cast<size_t>(linesize * kSrcHeight));
    std::vector<double> latencies_ms;

    uint64_t next_capture_ns    = virtual_now_ns;
    uint64_t next_transport_ns  = virtual_now_ns;
    const uint64_t sim_start_ns = virtual_now_ns;
    int frames_generated        = 0;
    int frames_decoded          = 0;
    double total_generate_ms    = 0.0;
    double total_submit_ms      = 0.0;
    double total_poll_ms        = 0.0;
    double total_receive_ms     = 0.0;

    while (frames_generated < input_frames || sender.stats().queued_chunks > 0
           || sender.stats().has_pending_frame || receiver.stats().queued_frames > 0) {
        virtual_now_ns = std::max(virtual_now_ns, next_transport_ns);

        if (frames_generated < input_frames && virtual_now_ns >= next_capture_ns) {
            total_generate_ms +=
                static_cast<double>(run_and_measure_ns([&] {
                    generate_color_bars(
                        frame_buf.data(), kSrcWidth, kSrcHeight, linesize, frames_generated);
                }))
                / 1'000'000.0;

            const uint64_t capture_ts    = virtual_now_ns;
            const auto submit_elapsed_ns = run_and_measure_ns([&] {
                auto submit =
                    sender.submit_frame(frame_buf.data(), linesize, frames_generated, capture_ts);
                if (!submit) {
                    std::fprintf(
                        stderr, "[loopback] submit frame %d failed: %s\n", frames_generated,
                        submit.error().c_str());
                    std::exit(1);
                }
            });
            total_submit_ms += static_cast<double>(submit_elapsed_ns) / 1'000'000.0;
            virtual_now_ns += submit_elapsed_ns;

            frames_generated++;
            next_capture_ns += capture_period_ns;
        }

        std::expected<std::optional<quanta::TransportChunk>, std::string> chunk_result;
        const auto poll_elapsed_ns =
            run_and_measure_ns([&] { chunk_result = sender.poll_chunk(); });
        total_poll_ms += static_cast<double>(poll_elapsed_ns) / 1'000'000.0;
        if (!chunk_result) {
            std::fprintf(
                stderr, "[loopback] sender poll failed: %s\n", chunk_result.error().c_str());
            return 1;
        }

        if (chunk_result->has_value()) {
            const auto receive_elapsed_ns = run_and_measure_ns([&] {
                auto recv = receiver.push_chunk((*chunk_result)->bytes(), virtual_now_ns);
                if (!recv) {
                    std::fprintf(
                        stderr, "[loopback] receiver push failed: %s\n", recv.error().c_str());
                    std::exit(1);
                }
            });
            total_receive_ms += static_cast<double>(receive_elapsed_ns) / 1'000'000.0;
            virtual_now_ns += receive_elapsed_ns;
        }

        while (auto frame = receiver.poll_frame()) {
            frames_decoded++;
            latencies_ms.push_back(static_cast<double>(frame->end_to_end_latency_ns) / 1'000'000.0);
        }

        next_transport_ns += transport_period_ns;
    }

    const auto sender_stats   = sender.stats();
    const auto receiver_stats = receiver.stats();
    const double sim_duration_sec =
        static_cast<double>(next_transport_ns - sim_start_ns) / 1'000'000'000.0;
    const double link_kbps =
        sim_duration_sec > 0.0
            ? (static_cast<double>(sender_stats.bytes_emitted) * 8.0 / sim_duration_sec / 1000.0)
            : 0.0;

    std::fprintf(
        stderr, "[loopback] %dx%d input -> %dx%d encoded\n", kSrcWidth, kSrcHeight, out_w, out_h);
    std::fprintf(stderr, "[loopback] Input frames:  %d @ %d fps\n", frames_generated, kInputFps);
    std::fprintf(stderr, "[loopback] Decoded frames: %d\n", frames_decoded);
    std::fprintf(
        stderr, "[loopback] Sender drops:  %llu\n",
        static_cast<unsigned long long>(sender_stats.frames_dropped));
    std::fprintf(
        stderr, "[loopback] Chunks/frame: %.2f | link usage: %.1f kbps\n",
        sender_stats.frames_encoded > 0
            ? static_cast<double>(sender_stats.chunks_emitted) / sender_stats.frames_encoded
            : 0.0,
        link_kbps);
    std::fprintf(
        stderr, "[loopback] Avg generate %.2f ms | submit %.2f ms | receive %.2f ms\n",
        frames_generated > 0 ? total_generate_ms / frames_generated : 0.0,
        frames_generated > 0 ? total_submit_ms / frames_generated : 0.0,
        sender_stats.chunks_emitted > 0 ? total_receive_ms / sender_stats.chunks_emitted : 0.0);
    std::fprintf(
        stderr, "[loopback] Avg poll %.2f ms/tick\n",
        next_transport_ns > sim_start_ns
            ? total_poll_ms
                  / ((next_transport_ns - sim_start_ns) / static_cast<double>(transport_period_ns))
            : 0.0);
    std::fprintf(
        stderr,
        "[loopback] Latency p50=%.1f ms p75=%.1f ms p95=%.1f ms max=%.1f ms dec_p75=%.1f ms\n",
        percentile_ms(latencies_ms, 0.50), receiver_stats.p75_latency_ms,
        percentile_ms(latencies_ms, 0.95), receiver_stats.max_latency_ns / 1'000'000.0,
        receiver_stats.p75_decode_ms);

    return frames_decoded > 0 ? 0 : 1;
}
