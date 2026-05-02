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
    bool sender_finished = false;
};

uint64_t now_ns() {
    const auto now = SteadyClock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
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
                "gen={} dec={} drp={}", generated_frames, decoded_frames,
                sender_stats.frames_dropped),
            std::format("queue={} link={:.1f}kbps", sender_stats.queued_chunks, link_kbps),
            std::format(
                "lst={:.1f}ms p75={:.1f}ms", receiver_stats.last_latency_ns / 1000000.0,
                receiver_stats.p75_latency_ms),
            std::format(
                "max={:.1f}ms d75={:.1f}ms rst={} err={}",
                receiver_stats.max_latency_ns / 1000000.0, receiver_stats.p75_decode_ms,
                receiver_stats.assembly_resets, receiver_stats.decode_errors),
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
                "emit={} asm={} dec={:.1f}", sender_stats.chunks_emitted,
                receiver_stats.frames_reassembled, displayed_fps),
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

        auto next_capture_tp   = SteadyClock::now();
        auto next_transport_tp = next_capture_tp;
        bool input_drained     = false;
        int generated_frames   = 0;

        while (!stop_requested.load()) {
            bool did_work      = false;
            const auto current = SteadyClock::now();

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
        int decoded_frames = 0;
        std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> loss_dist{0.0, 1.0};

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
