#include "quanta/stream_transport.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>
#include <vector>

namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr int kCanvasPad = 20;
constexpr int kHeaderH   = 88;
constexpr int kTextStep  = 24;

uint64_t now_ns() {
    const auto now = SteadyClock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
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

cv::Mat build_canvas(
    const cv::Mat& source_bgr, const cv::Mat& decoded_bgr,
    const quanta::StreamTransportConfig& config, const quanta::StreamSenderStats& sender_stats,
    const quanta::StreamReceiverStats& receiver_stats, uint64_t started_ns, int generated_frames,
    int decoded_frames, double source_fps, cv::Size encoded_size) {
    cv::Mat source_panel;
    if (source_bgr.size() != encoded_size) {
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
        decoded_panel = cv::Mat(encoded_size, CV_8UC3, cv::Scalar(18, 18, 18));
        cv::putText(
            decoded_panel, "Waiting For Decoder", {60, decoded_panel.rows / 2},
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
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

    draw_lines(
        canvas,
        {
            std::format(
                "gen={} dec={} drp={}", generated_frames, decoded_frames,
                sender_stats.frames_dropped),
            std::format("queue={} link={:.1f} kbps", sender_stats.queued_chunks, link_kbps),
            std::format(
                "lst={:.1f} ms avg={:.1f} ms", receiver_stats.last_latency_ns / 1'000'000.0,
                receiver_stats.avg_latency_ms),
            std::format(
                "max={:.1f} ms rst={} dec_err={}", receiver_stats.max_latency_ns / 1'000'000.0,
                receiver_stats.assembly_resets, receiver_stats.decode_errors),
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
            std::format("src={:.2f} ctrl: q / esc", source_fps),
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

    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::fprintf(stderr, "[demo] cannot open video: %s\n", video_path.c_str());
        return 1;
    }

    cv::Mat pending_frame;
    if (!read_next_bgr_frame(capture, pending_frame)) {
        std::fprintf(stderr, "[demo] video has no decodable frames: %s\n", video_path.c_str());
        return 1;
    }

    const double capture_fps_raw = capture.get(cv::CAP_PROP_FPS);
    const double source_fps =
        std::isfinite(capture_fps_raw) && capture_fps_raw > 0.0 ? capture_fps_raw : 30.0;

    quanta::StreamTransportConfig transport_config{
        .mtu_bytes      = 300,
        .transport_hz   = 50,
        .max_stream_fps = 30,
        .now_ns         = [] { return now_ns(); },
    };
    quanta::EncodeParams encode_params;
    quanta::StreamSender sender(encode_params, transport_config);
    quanta::StreamReceiver receiver(transport_config);

    auto init = sender.init(pending_frame.cols, pending_frame.rows);
    if (!init) {
        std::fprintf(stderr, "[demo] sender init failed: %s\n", init.error().c_str());
        return 1;
    }
    const auto [out_w, out_h] = sender.dimensions();
    const cv::Size encoded_size(out_w, out_h);

    cv::Mat latest_source = pending_frame.clone();
    cv::Mat latest_decoded;

    const auto capture_period =
        std::chrono::nanoseconds(static_cast<int64_t>(std::llround(1'000'000'000.0 / source_fps)));
    const auto transport_period = std::chrono::milliseconds(20);
    auto next_capture_tp        = SteadyClock::now();
    auto next_transport_tp      = next_capture_tp;
    const uint64_t started_ns   = now_ns();

    int generated_frames = 0;
    int decoded_frames   = 0;
    bool input_drained   = false;

    cv::namedWindow("quanta-stream-demo", cv::WINDOW_AUTOSIZE);

    while (true) {
        const auto current = SteadyClock::now();

        if (!input_drained && (frame_limit == 0 || generated_frames < frame_limit)
            && current >= next_capture_tp) {
            latest_source = pending_frame.clone();

            auto submit = sender.submit_frame(
                latest_source.data, static_cast<int>(latest_source.step[0]), generated_frames,
                now_ns());
            if (!submit) {
                std::fprintf(
                    stderr, "[demo] submit frame %d failed: %s\n", generated_frames,
                    submit.error().c_str());
                return 1;
            }
            generated_frames++;
            next_capture_tp += capture_period;

            if (frame_limit > 0 && generated_frames >= frame_limit) {
                input_drained = true;
            } else if (!read_next_bgr_frame(capture, pending_frame)) {
                input_drained = true;
            }
        }

        while (SteadyClock::now() >= next_transport_tp) {
            auto chunk = sender.poll_chunk();
            if (!chunk) {
                std::fprintf(stderr, "[demo] poll_chunk failed: %s\n", chunk.error().c_str());
                return 1;
            }
            if (chunk->has_value()) {
                auto pushed = receiver.push_chunk((*chunk)->bytes(), now_ns());
                if (!pushed) {
                    std::fprintf(stderr, "[demo] push_chunk failed: %s\n", pushed.error().c_str());
                    return 1;
                }
            }
            next_transport_tp += transport_period;
        }

        while (auto decoded = receiver.poll_frame()) {
            latest_decoded = cv::Mat(
                                 decoded->height, decoded->width, CV_8UC3, decoded->data.data(),
                                 decoded->linesize)
                                 .clone();
            decoded_frames++;
        }

        cv::Mat canvas = build_canvas(
            latest_source, latest_decoded, transport_config, sender.stats(), receiver.stats(),
            started_ns, generated_frames, decoded_frames, source_fps, encoded_size);
        cv::imshow("quanta-stream-demo", canvas);

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
            break;

        if (input_drained && sender.stats().queued_chunks == 0 && !sender.stats().has_pending_frame
            && receiver.stats().queued_frames == 0) {
            break;
        }

        auto next_event = next_transport_tp;
        if (!input_drained)
            next_event = std::min(next_event, next_capture_tp);
        const auto sleep_until = next_event - std::chrono::milliseconds(1);
        if (sleep_until > SteadyClock::now())
            std::this_thread::sleep_until(sleep_until);
    }

    cv::destroyAllWindows();
    return 0;
}
