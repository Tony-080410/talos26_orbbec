/// Quanta IPC E2E Demo
///
/// Reads live camera frames from Talos shared memory IPC,
/// encodes them with the Quanta HEVC encoder, transports through
/// an in-process queue (simulating the serial link), decodes on
/// the other end, and displays both sides via cv::imshow.
///
/// Usage: quanta-ipc-demo

#include "quanta/stream_transport.hpp"
#include "shm_client.hpp"
#include "spdlog_hook.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <format>
#include <mutex>
#include <string>
#include <thread>

namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr int kCanvasPad   = 20;
constexpr int kHeaderH     = 64;
constexpr int kTextStep    = 22;
constexpr int kTransportHz = 50;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct DemoState {
    std::mutex mutex;
    cv::Mat latest_source;
    cv::Mat latest_decoded;
    quanta::StreamSenderStats sender_stats{};
    quanta::StreamReceiverStats receiver_stats{};
    int encoded_frames = 0;
    int decoded_frames = 0;
    std::string error;
};

struct TransportPipe {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::vector<uint8_t>> chunks;
    bool sender_finished = false;
};

uint64_t now_ns() { return static_cast<uint64_t>(SteadyClock::now().time_since_epoch().count()); }

std::chrono::nanoseconds period_from_rate(double hz) {
    const double safe_hz = std::max(hz, 1.0);
    return std::chrono::nanoseconds(static_cast<int64_t>(std::llround(1'000'000'000.0 / safe_hz)));
}

// ---------------------------------------------------------------------------
// Simple text overlay
// ---------------------------------------------------------------------------

void draw_lines(
    cv::Mat& image, const std::vector<std::string>& lines, const cv::Point& origin, double scale) {
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::putText(
            image, lines[i], {origin.x, origin.y + static_cast<int>(i) * kTextStep},
            cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    }
}

cv::Mat make_placeholder(const cv::Size& size, const char* msg) {
    cv::Mat panel(size, CV_8UC3, cv::Scalar(18, 18, 18));
    cv::putText(
        panel, msg, {20, panel.rows / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200),
        2, cv::LINE_AA);
    return panel;
}

// ---------------------------------------------------------------------------
// Canvas builder
// ---------------------------------------------------------------------------

cv::Mat build_canvas(
    const cv::Mat& source_bgr, const cv::Mat& decoded_bgr,
    const quanta::StreamSenderStats& sender_stats,
    const quanta::StreamReceiverStats& receiver_stats, uint64_t started_ns, int encoded_frames,
    int decoded_frames, cv::Size display_size) {

    auto make_panel = [&](const cv::Mat& img, const char* label) {
        cv::Mat panel;
        if (img.empty()) {
            panel = make_placeholder(display_size, label);
        } else if (img.size() != display_size) {
            cv::resize(img, panel, display_size, 0.0, 0.0, cv::INTER_AREA);
        } else {
            panel = img.clone();
        }
        // label background
        cv::putText(
            panel, label, {16, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 4,
            cv::LINE_AA);
        cv::putText(
            panel, label, {16, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
            cv::LINE_AA);
        return panel;
    };

    cv::Mat src_panel = make_panel(source_bgr, "IPC Source");
    cv::Mat dec_panel = make_panel(decoded_bgr, "Quanta Decoded");
    cv::Mat panels;
    cv::hconcat(src_panel, dec_panel, panels);

    cv::Mat canvas(
        panels.rows + kHeaderH + kCanvasPad * 2, panels.cols + kCanvasPad * 2, CV_8UC3,
        cv::Scalar(22, 24, 28));
    panels.copyTo(canvas(cv::Rect(kCanvasPad, kHeaderH + kCanvasPad, panels.cols, panels.rows)));

    const double elapsed_sec =
        started_ns != 0 ? static_cast<double>(now_ns() - started_ns) / 1e9 : 0.0;
    const double link_kbps = elapsed_sec > 0.0 ? static_cast<double>(sender_stats.bytes_emitted)
                                                     * 8.0 / elapsed_sec / 1000.0
                                               : 0.0;
    const double display_fps =
        elapsed_sec > 0.0 ? static_cast<double>(decoded_frames) / elapsed_sec : 0.0;

    draw_lines(
        canvas,
        {
            std::format(
                "enc={} dec={} sub={} bad={}", encoded_frames, decoded_frames,
                receiver_stats.substituted_frames, receiver_stats.bad_frames),
            std::format(
                "link={:.1f} kbps  queue={}  chunks={}", link_kbps, sender_stats.queued_chunks,
                sender_stats.chunks_emitted),
            std::format(
                "lat  p50={:.0f}  p95={:.0f}  p99={:.0f} ms", receiver_stats.p75_latency_ms,
                receiver_stats.p95_latency_ms, receiver_stats.p99_latency_ms),
        },
        {kCanvasPad, 26}, 0.5);

    draw_lines(
        canvas,
        {
            std::format(
                "{}x{} @ {}B/{}Hz", display_size.width, display_size.height, 300, kTransportHz),
            std::format(
                "freeze={} max={} rec95={:.0f}ms", receiver_stats.freeze_run_length,
                receiver_stats.max_freeze_run_length, receiver_stats.recovery_ms_p95),
            std::format("display={:.1f} fps  |  q/esc to quit", display_fps),
        },
        {canvas.cols / 2 + 24, 26}, 0.5);

    return canvas;
}

} // namespace

// ============================================================================
// main
// ============================================================================

int main() {
    std::signal(SIGINT, [](int) { /* handled via g_running */ });
    std::signal(SIGTERM, [](int) {});

    init_logger();

    // ── 1. Connect to IPC ──────────────────────────────────────────────
    SPDLOG_INFO("connecting to Talos IPC shared memory...");

    auto client_result = ipc::ShmClient::connect();
    if (!client_result) {
        SPDLOG_CRITICAL("failed to connect IPC (is the simulator running?)");
        return 1;
    }
    auto& client = *client_result;

    SPDLOG_INFO("IPC connected. Waiting for producer heartbeat...");
    if (!client.wait_for_producer(std::chrono::seconds(10))) {
        SPDLOG_CRITICAL("producer did not appear within 10s");
        return 1;
    }
    SPDLOG_INFO("producer alive");

    // Grab the first frame to learn dimensions.
    cv::Mat first_frame;
    while (true) {
        if (auto frame = client.recv_image()) {
            first_frame = frame->image.clone();
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    SPDLOG_INFO(
        "first IPC frame: {}x{} ch={}", first_frame.cols, first_frame.rows, first_frame.channels());

    // ── 2. Quanta encoder / decoder ────────────────────────────────────

    quanta::EncodeParams encode_params;
    encode_params.framerate      = 30;
    encode_params.target_bitrate = 50'000;
    encode_params.max_width      = 300;
    encode_params.max_height     = 300;

    quanta::StreamTransportConfig transport_config{
        .mtu_bytes          = 300,
        .transport_hz       = kTransportHz,
        .max_stream_fps     = encode_params.framerate,
        .now_ns             = [] { return now_ns(); },
        .max_pending_frames = 1,
        .max_queued_chunks  = 4,
    };

    auto sender_result = quanta::StreamSender::create(
        encode_params, transport_config, first_frame.cols, first_frame.rows);
    if (!sender_result) {
        SPDLOG_CRITICAL("sender create: {}", sender_result.error());
        return 1;
    }
    quanta::StreamSender sender = std::move(*sender_result);

    quanta::StreamReceiver receiver(transport_config);

    const auto [out_w, out_h] = sender.dimensions();
    const cv::Size display_size(out_w, out_h);

    SPDLOG_INFO(
        "quanta: {}x{} -> {}x{}, {}B@{}Hz, {}bps", first_frame.cols, first_frame.rows, out_w, out_h,
        transport_config.mtu_bytes, transport_config.transport_hz, encode_params.target_bitrate);

    // ── 3. Thread shared state ─────────────────────────────────────────

    DemoState state;
    state.latest_source = first_frame.clone();

    TransportPipe pipe;
    std::atomic<bool> stop{false};

    const auto capture_period   = period_from_rate(30.0);
    const auto transport_period = period_from_rate(static_cast<double>(kTransportHz));
    const uint64_t started_ns   = now_ns();

    // ── 4. Sender thread (IPC → encode → queue) ────────────────────────

    std::thread sender_thread([&, sender = std::move(sender)]() mutable {
        auto next_capture_tp   = SteadyClock::now();
        auto next_transport_tp = next_capture_tp;
        int encoded            = 0;
        uint64_t last_seq      = 0;

        while (!stop.load()) {
            bool did_work  = false;
            const auto now = SteadyClock::now();

            // ── Capture from IPC ──
            if (now >= next_capture_tp) {
                if (auto frame = client.recv_image()) {
                    // Skip if duplicate / stale seq
                    if (frame->seq > last_seq) {
                        last_seq = frame->seq;

                        // Ensure BGR24
                        cv::Mat bgr;
                        if (frame->image.channels() == 1) {
                            cv::cvtColor(frame->image, bgr, cv::COLOR_GRAY2BGR);
                        } else if (frame->image.channels() == 4) {
                            cv::cvtColor(frame->image, bgr, cv::COLOR_BGRA2BGR);
                        } else {
                            bgr = frame->image;
                        }

                        auto submit = sender.submit_frame(
                            bgr.data, static_cast<int>(bgr.step[0]), encoded, frame->timestamp_ns);
                        if (!submit) {
                            SPDLOG_WARN("submit_frame {}: {}", encoded, submit.error());
                        }

                        encoded++;
                        {
                            std::lock_guard<std::mutex> lock(state.mutex);
                            state.latest_source  = bgr.clone();
                            state.encoded_frames = encoded;
                        }
                    }
                }
                next_capture_tp += capture_period;
                did_work = true;
            }

            // ── Drain chunks ──
            while (!stop.load() && SteadyClock::now() >= next_transport_tp) {
                auto chunk = sender.poll_chunk();
                if (!chunk) {
                    SPDLOG_WARN("poll_chunk: {}", chunk.error());
                    break;
                }
                if (chunk->has_value()) {
                    {
                        std::lock_guard<std::mutex> lock(pipe.mutex);
                        pipe.chunks.push_back(std::move((*chunk)->data));
                    }
                    pipe.cv.notify_one();
                }
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.sender_stats   = sender.stats();
                    state.encoded_frames = encoded;
                }
                next_transport_tp += transport_period;
                did_work = true;
            }

            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.sender_stats   = sender.stats();
                state.encoded_frames = encoded;
            }

            // Back-pressure: if sender queue is drained and input is idle, sleep
            if (!did_work) {
                const auto next_event = std::min(next_capture_tp, next_transport_tp);
                if (next_event > SteadyClock::now()) {
                    std::this_thread::sleep_until(next_event);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(pipe.mutex);
            pipe.sender_finished = true;
        }
        pipe.cv.notify_all();
        SPDLOG_INFO("sender thread done ({} frames encoded)", encoded);
    });

    // ── 5. Receiver thread (queue → decode → state) ────────────────────

    std::thread receiver_thread([&, receiver = std::move(receiver)]() mutable {
        int decoded = 0;

        while (true) {
            std::vector<uint8_t> chunk_bytes;
            {
                std::unique_lock<std::mutex> lock(pipe.mutex);
                pipe.cv.wait(lock, [&] {
                    return stop.load() || !pipe.chunks.empty() || pipe.sender_finished;
                });

                if ((stop.load() && pipe.chunks.empty())
                    || (pipe.chunks.empty() && pipe.sender_finished)) {
                    break;
                }

                chunk_bytes = std::move(pipe.chunks.front());
                pipe.chunks.pop_front();
            }

            auto pushed = receiver.push_chunk(chunk_bytes, now_ns());
            if (!pushed) {
                SPDLOG_WARN("push_chunk: {}", pushed.error());
                continue;
            }

            // Drain all decoded frames
            while (auto decoded_frame = receiver.poll_frame()) {
                cv::Mat view(
                    decoded_frame->height, decoded_frame->width, CV_8UC3,
                    decoded_frame->data.data(), decoded_frame->linesize);
                decoded++;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.latest_decoded = view.clone();
                    state.decoded_frames = decoded;
                    state.receiver_stats = receiver.stats();
                }
            }

            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.receiver_stats = receiver.stats();
                state.decoded_frames = decoded;
            }
        }

        SPDLOG_INFO("receiver thread done ({} frames decoded)", decoded);
    });

    // ── 6. Display loop (main thread) ──────────────────────────────────

    cv::namedWindow("quanta-ipc-e2e", cv::WINDOW_AUTOSIZE);

    int exit_code = 0;
    while (true) {
        cv::Mat source_copy, decoded_copy;
        quanta::StreamSenderStats ss{};
        quanta::StreamReceiverStats rs{};
        int enc = 0, dec = 0;

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            source_copy  = state.latest_source;
            decoded_copy = state.latest_decoded;
            ss           = state.sender_stats;
            rs           = state.receiver_stats;
            enc          = state.encoded_frames;
            dec          = state.decoded_frames;
        }

        cv::Mat canvas =
            build_canvas(source_copy, decoded_copy, ss, rs, started_ns, enc, dec, display_size);

        cv::imshow("quanta-ipc-e2e", canvas);

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            stop.store(true);
            pipe.cv.notify_all();
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    stop.store(true);
    pipe.cv.notify_all();

    if (sender_thread.joinable())
        sender_thread.join();
    if (receiver_thread.joinable())
        receiver_thread.join();

    cv::destroyAllWindows();
    SPDLOG_INFO("exiting (code {})", exit_code);
    return exit_code;
}
