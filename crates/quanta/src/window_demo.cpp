#include "quanta/stream_transport.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <format>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr int kCanvasPad   = 20;
constexpr int kHeaderH     = 88;
constexpr int kTextStep    = 24;
constexpr int kTransportHz = 50;

// 随机丢包仿真参数 (0.0 = 无丢包, 1.0 = 全丢)
constexpr double kPacketLossRate = 0.3;

struct DemoState {
    std::mutex mutex;
    cv::Mat latest_source;
    cv::Mat latest_decoded;
    quanta::StreamSenderStats sender_stats{};
    quanta::StreamReceiverStats receiver_stats{};
    int generated_frames   = 0;
    int decoded_frames     = 0;
    int dropped_chunks     = 0;
    bool sender_finished   = false;
    bool receiver_finished = false;
    std::string error;
};

struct DemoSnapshot {
    cv::Mat latest_source;
    cv::Mat latest_decoded;
    quanta::StreamSenderStats sender_stats{};
    quanta::StreamReceiverStats receiver_stats{};
    int generated_frames   = 0;
    int decoded_frames     = 0;
    int dropped_chunks     = 0;
    bool sender_finished   = false;
    bool receiver_finished = false;
    std::string error;
};

struct TransportPipe {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::vector<uint8_t>> chunks;
    std::atomic<uint64_t> bad_link_events{0};
    std::atomic<uint64_t> clean_frame_requests{0};
    bool sender_finished = false;
};

uint64_t now_ns() {
    const auto now = SteadyClock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}

uint64_t ms_to_ns(uint64_t ms) { return ms * 1'000'000ULL; }

void print_stream_config(const quanta::StreamTransportConfig& config) {
    std::fprintf(
        stderr,
        "[demo] stream config: deadline=%llums/%llums/%llums clean_threshold=%llu "
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

double sanitize_fps(double fps) { return std::isfinite(fps) && fps > 0.0 ? fps : 30.0; }

std::chrono::nanoseconds period_from_rate(double hz) {
    const double safe_hz = std::max(hz, 1.0);
    return std::chrono::nanoseconds(static_cast<int64_t>(std::llround(1'000'000'000.0 / safe_hz)));
}

bool read_next_bgr_frame(cv::VideoCapture& capture, cv::Mat& frame_out) {
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty())
        return false;

    if (frame.depth() != CV_8U) {
        cv::Mat converted;
        frame.convertTo(converted, CV_8U);
        frame = std::move(converted);
    }

    if (frame.channels() == 3) {
        frame_out = std::move(frame);
        return true;
    }

    if (frame.channels() == 1) {
        cv::cvtColor(frame, frame_out, cv::COLOR_GRAY2BGR);
        return true;
    }

    if (frame.channels() == 4) {
        cv::cvtColor(frame, frame_out, cv::COLOR_BGRA2BGR);
        return true;
    }

    return false;
}

void draw_lines(
    cv::Mat& image, const std::vector<std::string>& lines, cv::Point origin, double scale) {
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::putText(
            image, lines[i], {origin.x, origin.y + static_cast<int>(i) * kTextStep},
            cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    }
}

cv::Mat make_placeholder_panel(cv::Size size, const char* message) {
    cv::Mat panel(size, CV_8UC3, cv::Scalar(18, 18, 18));
    cv::putText(
        panel, message, {40, panel.rows / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
        cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
    return panel;
}

DemoSnapshot snapshot_state(DemoState& state) {
    std::lock_guard<std::mutex> lock(state.mutex);

    DemoSnapshot snapshot;
    snapshot.latest_source     = state.latest_source;
    snapshot.latest_decoded    = state.latest_decoded;
    snapshot.sender_stats      = state.sender_stats;
    snapshot.receiver_stats    = state.receiver_stats;
    snapshot.generated_frames  = state.generated_frames;
    snapshot.decoded_frames    = state.decoded_frames;
    snapshot.dropped_chunks    = state.dropped_chunks;
    snapshot.sender_finished   = state.sender_finished;
    snapshot.receiver_finished = state.receiver_finished;
    snapshot.error             = state.error;
    return snapshot;
}

void store_error(
    DemoState& state, TransportPipe& pipe, std::atomic<bool>& stop_requested, std::string message) {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.error.empty())
            state.error = std::move(message);
    }
    stop_requested.store(true);
    pipe.cv.notify_all();
}

void mark_sender_finished(
    DemoState& state, TransportPipe& pipe, const quanta::StreamSenderStats& sender_stats,
    int generated_frames) {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.sender_stats     = sender_stats;
        state.generated_frames = generated_frames;
        state.sender_finished  = true;
    }
    {
        std::lock_guard<std::mutex> lock(pipe.mutex);
        pipe.sender_finished = true;
    }
    pipe.cv.notify_all();
}

void mark_receiver_finished(
    DemoState& state, const quanta::StreamReceiverStats& receiver_stats, int decoded_frames) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.receiver_stats    = receiver_stats;
    state.decoded_frames    = decoded_frames;
    state.receiver_finished = true;
}

cv::Mat build_canvas(
    const cv::Mat& source_bgr, const cv::Mat& decoded_bgr,
    const quanta::StreamTransportConfig& config, const quanta::StreamSenderStats& sender_stats,
    const quanta::StreamReceiverStats& receiver_stats, uint64_t started_ns, int generated_frames,
    int decoded_frames, int dropped_chunks, double source_fps, double stream_fps,
    cv::Size encoded_size) {
    cv::Mat source_panel;
    if (source_bgr.empty()) {
        source_panel = make_placeholder_panel(encoded_size, "Waiting For Source");
    } else if (source_bgr.size() != encoded_size) {
        cv::resize(source_bgr, source_panel, encoded_size, 0.0, 0.0, cv::INTER_AREA);
    } else {
        source_panel = source_bgr.clone();
    }

    cv::Mat decoded_panel;
    if (!decoded_bgr.empty()) {
        if (decoded_bgr.size() != encoded_size) {
            cv::resize(decoded_bgr, decoded_panel, encoded_size, 0.0, 0.0, cv::INTER_LINEAR);
        } else {
            decoded_panel = decoded_bgr.clone();
        }
    } else {
        decoded_panel = make_placeholder_panel(encoded_size, "Waiting For Decoder");
    }

    cv::putText(
        source_panel, "Encoder Input", {20, 36}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(16, 16, 16), 4, cv::LINE_AA);
    cv::putText(
        source_panel, "Encoder Input", {20, 36}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(250, 250, 250), 2, cv::LINE_AA);

    cv::putText(
        decoded_panel, "Decoder Output", {20, 36}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(16, 16, 16), 4, cv::LINE_AA);
    cv::putText(
        decoded_panel, "Decoder Output", {20, 36}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(250, 250, 250), 2, cv::LINE_AA);

    cv::Mat panels;
    cv::hconcat(source_panel, decoded_panel, panels);

    cv::Mat canvas(
        panels.rows + kHeaderH + kCanvasPad * 2, panels.cols + kCanvasPad * 2, CV_8UC3,
        cv::Scalar(22, 24, 28));
    panels.copyTo(canvas(cv::Rect(kCanvasPad, kHeaderH + kCanvasPad, panels.cols, panels.rows)));

    const double elapsed_sec =
        started_ns != 0 ? static_cast<double>(now_ns() - started_ns) / 1'000'000'000.0 : 0.0;
    const double link_kbps =
        elapsed_sec > 0.0
            ? (static_cast<double>(sender_stats.bytes_emitted) * 8.0 / elapsed_sec / 1000.0)
            : 0.0;
    const double displayed_fps =
        elapsed_sec > 0.0 ? static_cast<double>(decoded_frames) / elapsed_sec : 0.0;
    const uint64_t delivered_chunks = receiver_stats.chunks_received;
    const uint64_t simulated_chunks = delivered_chunks + static_cast<uint64_t>(dropped_chunks);
    const double simulated_loss_pct =
        simulated_chunks > 0
            ? static_cast<double>(dropped_chunks) * 100.0 / static_cast<double>(simulated_chunks)
            : 0.0;

    draw_lines(
        canvas,
        {
            std::format(
                "gen={} disp={} sub={} bad={}", generated_frames, decoded_frames,
                receiver_stats.substituted_frames, receiver_stats.bad_frames),
            std::format(
                "queue={} maxq={} link={:.1f}kbps age99={:.0f}ms", sender_stats.queued_chunks,
                sender_stats.max_queued_chunks_seen, link_kbps, sender_stats.p99_send_age_ms),
            std::format(
                "lat p75={:.0f} p95={:.0f} p99={:.0f} max={:.0f}ms", receiver_stats.p75_latency_ms,
                receiver_stats.p95_latency_ms, receiver_stats.p99_latency_ms,
                receiver_stats.max_latency_ns / 1000000.0),
            std::format(
                "freeze={} max={} rec95={:.0f}f/{:.0f}ms gap99={:.0f}ms",
                receiver_stats.freeze_run_length, receiver_stats.max_freeze_run_length,
                receiver_stats.recovery_frames_p95, receiver_stats.recovery_ms_p95,
                receiver_stats.p99_frame_gap_ms),
            std::format(
                "loss={} ({:.0f}%)  rate={:.0f}%", dropped_chunks, simulated_loss_pct,
                kPacketLossRate * 100.0),
        },
        {kCanvasPad, 32}, 0.5);

    draw_lines(
        canvas,
        {
            std::format(
                "{}B@{}Hz {}x{}", config.mtu_bytes, config.transport_hz, encoded_size.width,
                encoded_size.height),
            std::format(
                "emit={} repair={} cpf={:.2f}", sender_stats.chunks_emitted,
                sender_stats.repair_chunks_emitted, sender_stats.chunks_per_frame),
            std::format(
                "idr={} force={}/{} sched={} disp={:.1f}", receiver_stats.idr_frames,
                sender_stats.force_idr_count, sender_stats.force_idr_requests,
                sender_stats.scheduled_idr_count, displayed_fps),
            std::format(
                "miss={} dec={} dep={} tail={}", receiver_stats.substitute_missing_chunk,
                receiver_stats.substitute_decode_error, receiver_stats.substitute_dependency_dirty,
                sender_stats.old_frame_tail_dropped),
            std::format("src={:.2f} push={:.2f} ctrl: q / esc", source_fps, stream_fps),
        },
        {canvas.cols / 2 + 24, 32}, 0.5);

    return canvas;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <video_path> [frame_limit]\n", argv[0]);
        return 1;
    }

    const std::string video_path = argv[1];
    int frame_limit              = 0;
    if (argc >= 3)
        frame_limit = std::atoi(argv[2]);

    cv::VideoCapture probe_capture(video_path);
    if (!probe_capture.isOpened()) {
        std::fprintf(stderr, "[demo] cannot open video: %s\n", video_path.c_str());
        return 1;
    }

    cv::Mat first_frame;
    if (!read_next_bgr_frame(probe_capture, first_frame)) {
        std::fprintf(stderr, "[demo] video has no decodable frames: %s\n", video_path.c_str());
        return 1;
    }

    quanta::EncodeParams encode_params;
    const double source_fps  = sanitize_fps(probe_capture.get(cv::CAP_PROP_FPS));
    const int stream_fps_int = std::max(
        1, static_cast<int>(std::llround(
               std::min(
                   {source_fps, static_cast<double>(kTransportHz),
                    static_cast<double>(std::max(1, encode_params.framerate))}))));
    const double stream_fps = static_cast<double>(stream_fps_int);
    encode_params.framerate = stream_fps_int;
    probe_capture.release();

    quanta::StreamTransportConfig transport_config{
        .mtu_bytes          = 300,
        .transport_hz       = kTransportHz,
        .max_stream_fps     = stream_fps_int,
        .now_ns             = [] { return now_ns(); },
        .max_pending_frames = 1,
        .max_queued_chunks  = 4,
    };

    auto sender_result = quanta::StreamSender::create(
        encode_params, transport_config, first_frame.cols, first_frame.rows);
    if (!sender_result) {
        std::fprintf(stderr, "[demo] sender create failed: %s\n", sender_result.error().c_str());
        return 1;
    }
    quanta::StreamSender sender = std::move(*sender_result);

    quanta::StreamReceiver receiver(transport_config);
    const auto [out_w, out_h] = sender.dimensions();
    const cv::Size encoded_size(out_w, out_h);
    const auto transport_period =
        period_from_rate(static_cast<double>(transport_config.transport_hz));
    const auto capture_period = period_from_rate(stream_fps);
    const uint64_t started_ns = now_ns();
    std::fprintf(
        stderr, "[demo] source %.2ffps -> stream %dfps, %zuB@%dHz, target %d bps, loss %.0f%%\n",
        source_fps, stream_fps_int, transport_config.mtu_bytes, transport_config.transport_hz,
        encode_params.target_bitrate, kPacketLossRate * 100.0);
    print_stream_config(transport_config);

    DemoState state;
    state.latest_source = first_frame.clone();

    TransportPipe pipe;
    std::atomic<bool> stop_requested{false};

    std::thread sender_thread([&, sender = std::move(sender)]() mutable {
        cv::VideoCapture capture(video_path);
        if (!capture.isOpened()) {
            store_error(
                state, pipe, stop_requested,
                std::format("[demo] cannot reopen video in sender thread: {}", video_path));
            mark_sender_finished(state, pipe, sender.stats(), 0);
            return;
        }

        cv::Mat pending_frame;
        if (!read_next_bgr_frame(capture, pending_frame)) {
            store_error(
                state, pipe, stop_requested,
                std::format("[demo] sender thread could not read frames from {}", video_path));
            mark_sender_finished(state, pipe, sender.stats(), 0);
            return;
        }

        auto next_capture_tp          = SteadyClock::now();
        auto next_transport_tp        = next_capture_tp;
        bool input_drained            = false;
        int generated_frames          = 0;
        uint64_t bad_link_events_seen = 0;
        uint64_t clean_requests_seen  = 0;

        auto consume_bad_link_events = [&] {
            const uint64_t events = pipe.bad_link_events.load(std::memory_order_relaxed);
            while (bad_link_events_seen < events) {
                sender.notify_bad_link();
                bad_link_events_seen++;
            }
        };

        auto consume_clean_frame_requests = [&] {
            const uint64_t requests = pipe.clean_frame_requests.load(std::memory_order_relaxed);
            while (clean_requests_seen < requests) {
                sender.request_clean_frame();
                clean_requests_seen++;
            }
        };

        while (!stop_requested.load()) {
            bool did_work      = false;
            const auto current = SteadyClock::now();
            consume_bad_link_events();
            consume_clean_frame_requests();

            if (!input_drained && (frame_limit == 0 || generated_frames < frame_limit)
                && current >= next_capture_tp) {
                cv::Mat submitted_frame   = pending_frame.clone();
                const uint64_t capture_ts = now_ns();

                auto submit = sender.submit_frame(
                    submitted_frame.data, static_cast<int>(submitted_frame.step[0]),
                    generated_frames, capture_ts);
                if (!submit) {
                    store_error(
                        state, pipe, stop_requested,
                        std::format(
                            "[demo] submit frame {} failed: {}", generated_frames, submit.error()));
                    mark_sender_finished(state, pipe, sender.stats(), generated_frames);
                    return;
                }

                generated_frames++;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.latest_source    = std::move(submitted_frame);
                    state.generated_frames = generated_frames;
                    state.sender_stats     = sender.stats();
                }

                next_capture_tp += capture_period;
                if (frame_limit > 0 && generated_frames >= frame_limit) {
                    input_drained = true;
                } else if (!read_next_bgr_frame(capture, pending_frame)) {
                    input_drained = true;
                }
                did_work = true;
            }

            while (!stop_requested.load() && SteadyClock::now() >= next_transport_tp) {
                auto chunk = sender.poll_chunk();
                if (!chunk) {
                    store_error(
                        state, pipe, stop_requested,
                        std::format("[demo] poll_chunk failed: {}", chunk.error()));
                    mark_sender_finished(state, pipe, sender.stats(), generated_frames);
                    return;
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
                    state.sender_stats     = sender.stats();
                    state.generated_frames = generated_frames;
                }

                next_transport_tp += transport_period;
                did_work = true;
            }

            const auto sender_stats = sender.stats();
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.sender_stats     = sender_stats;
                state.generated_frames = generated_frames;
            }

            if (input_drained && sender_stats.queued_chunks == 0 && !sender_stats.has_pending_frame)
                break;

            auto next_event = next_transport_tp;
            if (!input_drained && (frame_limit == 0 || generated_frames < frame_limit))
                next_event = std::min(next_event, next_capture_tp);
            if (!did_work && next_event > SteadyClock::now())
                std::this_thread::sleep_until(next_event);
        }

        mark_sender_finished(state, pipe, sender.stats(), generated_frames);
    });

    std::thread receiver_thread([&, receiver = std::move(receiver)]() mutable {
        int decoded_frames                 = 0;
        uint64_t last_bad_link_substituted = 0;
        uint64_t last_bad_link_bad_p       = 0;
        uint64_t last_clean_request_ns     = 0;
        uint64_t waiting_clean_baseline    = 0;
        bool waiting_clean                 = false;
        std::deque<uint64_t> clean_request_times_ns;
        std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> loss_dist{0.0, 1.0};

        auto maybe_notify_bad_link = [&] {
            const auto receiver_stats = receiver.stats();
            if (receiver_stats.substituted_frames != last_bad_link_substituted
                || receiver_stats.bad_p_frames != last_bad_link_bad_p) {
                pipe.bad_link_events.fetch_add(1, std::memory_order_relaxed);
                last_bad_link_substituted = receiver_stats.substituted_frames;
                last_bad_link_bad_p       = receiver_stats.bad_p_frames;
            }
        };

        auto maybe_request_clean_frame = [&] {
            const auto receiver_stats = receiver.stats();
            const uint64_t current_ns = now_ns();
            if (waiting_clean && receiver_stats.clean_frames > waiting_clean_baseline) {
                waiting_clean = false;
            } else if (
                waiting_clean && last_clean_request_ns != 0
                && current_ns - last_clean_request_ns
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
                && current_ns - last_clean_request_ns
                       < ms_to_ns(transport_config.clean_request_cooldown_ms)) {
                return;
            }

            const uint64_t request_window_ns = ms_to_ns(transport_config.clean_request_window_ms);
            while (!clean_request_times_ns.empty()
                   && current_ns - clean_request_times_ns.front() > request_window_ns) {
                clean_request_times_ns.pop_front();
            }
            if (clean_request_times_ns.size() >= transport_config.max_clean_requests_per_window)
                return;

            pipe.clean_frame_requests.fetch_add(1, std::memory_order_relaxed);
            last_clean_request_ns  = current_ns;
            waiting_clean          = true;
            waiting_clean_baseline = receiver_stats.clean_frames;
            clean_request_times_ns.push_back(current_ns);
        };

        auto publish_decoded_frames = [&]() {
            bool emitted = false;
            while (auto decoded = receiver.poll_frame()) {
                cv::Mat decoded_view(
                    decoded->height, decoded->width, CV_8UC3, decoded->data.data(),
                    decoded->linesize);
                decoded_frames++;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.latest_decoded = decoded_view.clone();
                    state.decoded_frames = decoded_frames;
                    state.receiver_stats = receiver.stats();
                }
                emitted = true;
            }

            if (!emitted) {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.receiver_stats = receiver.stats();
                state.decoded_frames = decoded_frames;
            }
            maybe_notify_bad_link();
            maybe_request_clean_frame();
        };

        while (true) {
            std::vector<uint8_t> chunk_bytes;
            {
                std::unique_lock<std::mutex> lock(pipe.mutex);
                pipe.cv.wait(lock, [&] {
                    return stop_requested.load() || !pipe.chunks.empty() || pipe.sender_finished;
                });

                if ((stop_requested.load() && pipe.chunks.empty())
                    || (pipe.chunks.empty() && pipe.sender_finished)) {
                    break;
                }

                chunk_bytes = std::move(pipe.chunks.front());
                pipe.chunks.pop_front();
            }

            // 随机丢包仿真 (按照 kPacketLossRate 概率丢弃数据包)
            if (loss_dist(rng) < kPacketLossRate) {
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.dropped_chunks++;
                }
                continue;
            }

            auto pushed = receiver.push_chunk(chunk_bytes, now_ns());
            if (!pushed) {
                store_error(
                    state, pipe, stop_requested,
                    std::format("[demo] push_chunk failed: {}", pushed.error()));
                mark_receiver_finished(state, receiver.stats(), decoded_frames);
                return;
            }

            publish_decoded_frames();
        }

        publish_decoded_frames();
        mark_receiver_finished(state, receiver.stats(), decoded_frames);
    });

    cv::namedWindow("quanta-stream-demo", cv::WINDOW_AUTOSIZE);

    int exit_code = 0;
    while (true) {
        const auto snapshot = snapshot_state(state);
        cv::Mat canvas      = build_canvas(
            snapshot.latest_source, snapshot.latest_decoded, transport_config,
            snapshot.sender_stats, snapshot.receiver_stats, started_ns, snapshot.generated_frames,
            snapshot.decoded_frames, snapshot.dropped_chunks, source_fps, stream_fps, encoded_size);
        cv::imshow("quanta-stream-demo", canvas);

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            stop_requested.store(true);
            pipe.cv.notify_all();
            break;
        }

        if (!snapshot.error.empty()) {
            std::fprintf(stderr, "%s\n", snapshot.error.c_str());
            stop_requested.store(true);
            pipe.cv.notify_all();
            exit_code = 1;
            break;
        }

        if (snapshot.sender_finished && snapshot.receiver_finished)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    stop_requested.store(true);
    pipe.cv.notify_all();
    if (sender_thread.joinable())
        sender_thread.join();
    if (receiver_thread.joinable())
        receiver_thread.join();

    const auto final_snapshot = snapshot_state(state);
    if (exit_code == 0 && !final_snapshot.error.empty()) {
        std::fprintf(stderr, "%s\n", final_snapshot.error.c_str());
        exit_code = 1;
    }

    cv::destroyAllWindows();
    return exit_code;
}
