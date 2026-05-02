#include "quanta/stream_transport.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
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

uint64_t ms_to_ns(uint64_t ms) { return ms * 1'000'000ULL; }

void print_stream_config(const quanta::StreamTransportConfig& config) {
    std::fprintf(
        stderr,
        "[loopback] stream config: deadline=%llums/%llums/%llums clean_threshold=%llu "
        "cooldown=%llums max_req=%llu/%llums motion_hold=%lluf/%dpx prefilter=%lluf "
        "temporal=1/%d\n",
        static_cast<unsigned long long>(config.soft_deadline_ms),
        static_cast<unsigned long long>(config.hard_deadline_ms),
        static_cast<unsigned long long>(config.display_deadline_ms),
        static_cast<unsigned long long>(config.clean_request_freeze_threshold),
        static_cast<unsigned long long>(config.clean_request_cooldown_ms),
        static_cast<unsigned long long>(config.max_clean_requests_per_window),
        static_cast<unsigned long long>(config.clean_request_window_ms),
        static_cast<unsigned long long>(config.motion_hold_max_frames), config.motion_hold_max_px,
        static_cast<unsigned long long>(config.prefilter_window_frames),
        1 << config.prefilter_temporal_shift);
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
    double packet_loss_rate = 0.0;
    if (argc >= 3)
        packet_loss_rate = std::clamp(std::atof(argv[2]), 0.0, 1.0);

    constexpr int kSrcWidth         = 640;
    constexpr int kSrcHeight        = 480;
    constexpr int kInputFps         = 50;
    constexpr uint64_t kOneSecondNs = 1'000'000'000ULL;

    uint64_t virtual_now_ns = real_now_ns();
    auto clock_fn           = [&virtual_now_ns] { return virtual_now_ns; };

    quanta::EncodeParams encode_params;
    quanta::StreamTransportConfig transport_config{
        .mtu_bytes      = 300,
        .transport_hz   = 50,
        .max_stream_fps = encode_params.framerate,
        .now_ns         = clock_fn,
    };
    print_stream_config(transport_config);
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

    uint64_t next_capture_ns           = virtual_now_ns;
    uint64_t next_transport_ns         = virtual_now_ns;
    const uint64_t sim_start_ns        = virtual_now_ns;
    int frames_generated               = 0;
    int frames_decoded                 = 0;
    double total_generate_ms           = 0.0;
    double total_submit_ms             = 0.0;
    double total_poll_ms               = 0.0;
    double total_receive_ms            = 0.0;
    int dropped_chunks                 = 0;
    uint64_t last_bad_link_substituted = 0;
    uint64_t last_bad_link_bad_p       = 0;
    uint64_t last_clean_request_ns     = 0;
    uint64_t waiting_clean_baseline    = 0;
    bool waiting_clean                 = false;
    std::deque<uint64_t> clean_request_times_ns;
    std::mt19937 rng{0x51525354u};
    std::uniform_real_distribution<double> loss_dist{0.0, 1.0};

    auto maybe_notify_bad_link = [&] {
        const auto receiver_stats = receiver.stats();
        if (receiver_stats.substituted_frames != last_bad_link_substituted
            || receiver_stats.bad_p_frames != last_bad_link_bad_p) {
            sender.notify_bad_link();
            last_bad_link_substituted = receiver_stats.substituted_frames;
            last_bad_link_bad_p       = receiver_stats.bad_p_frames;
        }
    };

    auto maybe_request_clean_frame = [&] {
        const auto receiver_stats = receiver.stats();
        if (waiting_clean && receiver_stats.clean_frames > waiting_clean_baseline) {
            waiting_clean = false;
        } else if (
            waiting_clean && last_clean_request_ns != 0
            && virtual_now_ns - last_clean_request_ns
                   >= ms_to_ns(transport_config.clean_request_cooldown_ms)) {
            waiting_clean = false;
        }
        if (receiver_stats.freeze_run_length == 0)
            return;
        if (waiting_clean)
            return;
        if (receiver_stats.freeze_run_length < transport_config.clean_request_freeze_threshold)
            return;
        if (last_clean_request_ns != 0
            && virtual_now_ns - last_clean_request_ns
                   < ms_to_ns(transport_config.clean_request_cooldown_ms)) {
            return;
        }

        const uint64_t request_window_ns = ms_to_ns(transport_config.clean_request_window_ms);
        while (!clean_request_times_ns.empty()
               && virtual_now_ns - clean_request_times_ns.front() > request_window_ns) {
            clean_request_times_ns.pop_front();
        }
        if (clean_request_times_ns.size() >= transport_config.max_clean_requests_per_window)
            return;

        sender.request_clean_frame();
        last_clean_request_ns  = virtual_now_ns;
        waiting_clean          = true;
        waiting_clean_baseline = receiver_stats.clean_frames;
        clean_request_times_ns.push_back(virtual_now_ns);
    };

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
            if (loss_dist(rng) < packet_loss_rate) {
                dropped_chunks++;
            } else {
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
        }

        while (auto frame = receiver.poll_frame()) {
            frames_decoded++;
            latencies_ms.push_back(static_cast<double>(frame->end_to_end_latency_ns) / 1'000'000.0);
        }
        maybe_notify_bad_link();
        maybe_request_clean_frame();

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
        sender_stats.chunks_per_frame, link_kbps);
    std::fprintf(
        stderr,
        "[loopback] Transport: emitted=%llu repair=%llu maxq=%zu send_age_p99=%.1f ms "
        "tail_drop=%llu expired_chunks=%llu expired_frames=%llu force_idr=%llu/%llu "
        "sched_idr=%llu\n",
        static_cast<unsigned long long>(sender_stats.chunks_emitted),
        static_cast<unsigned long long>(sender_stats.repair_chunks_emitted),
        sender_stats.max_queued_chunks_seen, sender_stats.p99_send_age_ms,
        static_cast<unsigned long long>(sender_stats.old_frame_tail_dropped),
        static_cast<unsigned long long>(sender_stats.expired_chunks_dropped),
        static_cast<unsigned long long>(sender_stats.expired_frames_dropped),
        static_cast<unsigned long long>(sender_stats.force_idr_count),
        static_cast<unsigned long long>(sender_stats.force_idr_requests),
        static_cast<unsigned long long>(sender_stats.scheduled_idr_count));
    std::fprintf(
        stderr, "[loopback] Dropped chunks: %d / %llu (%.1f%% simulated)\n", dropped_chunks,
        static_cast<unsigned long long>(sender_stats.chunks_emitted), packet_loss_rate * 100.0);
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
        "[loopback] Latency p50=%.1f ms p75=%.1f ms p95=%.1f ms p99=%.1f ms max=%.1f ms "
        "dec_p75=%.1f ms\n",
        percentile_ms(latencies_ms, 0.50), receiver_stats.p75_latency_ms,
        receiver_stats.p95_latency_ms, receiver_stats.p99_latency_ms,
        receiver_stats.max_latency_ns / 1'000'000.0, receiver_stats.p75_decode_ms);
    std::fprintf(
        stderr,
        "[loopback] Frame health: displayed=%llu good=%llu substituted=%llu bad=%llu bad_p=%llu "
        "idr=%llu freeze=%llu max_freeze=%llu recovery_p95=%.1ff/%.1fms gap_p99=%.1f ms "
        "display_deadline_drop=%llu\n",
        static_cast<unsigned long long>(receiver_stats.frames_displayed),
        static_cast<unsigned long long>(receiver_stats.good_frames),
        static_cast<unsigned long long>(receiver_stats.substituted_frames),
        static_cast<unsigned long long>(receiver_stats.bad_frames),
        static_cast<unsigned long long>(receiver_stats.bad_p_frames),
        static_cast<unsigned long long>(receiver_stats.idr_frames),
        static_cast<unsigned long long>(receiver_stats.freeze_run_length),
        static_cast<unsigned long long>(receiver_stats.max_freeze_run_length),
        receiver_stats.recovery_frames_p95, receiver_stats.recovery_ms_p95,
        receiver_stats.p99_frame_gap_ms,
        static_cast<unsigned long long>(receiver_stats.display_deadline_drops));
    std::fprintf(
        stderr,
        "[loopback] Substitute reason: missing=%llu decode=%llu deadline=%llu dependency=%llu "
        "recovery_runs=%llu\n",
        static_cast<unsigned long long>(receiver_stats.substitute_missing_chunk),
        static_cast<unsigned long long>(receiver_stats.substitute_decode_error),
        static_cast<unsigned long long>(receiver_stats.substitute_display_deadline),
        static_cast<unsigned long long>(receiver_stats.substitute_dependency_dirty),
        static_cast<unsigned long long>(receiver_stats.recovery_run_count));

    return frames_decoded > 0 ? 0 : 1;
}
