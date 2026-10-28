// 棋盘格内参标定：从 daedalus shm 取图（orbbec_bridge 喂图），复用项目自带标定算法。
// 用法：./calibrate-intrinsic [采样间隔ms，默认 600]

#include "calibration/calibration_config.hpp"
#include "calibration/chessboard_detector.hpp"
#include "calibration/intrinsic_calibrator.hpp"
#include "shm_client.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
    const int interval_ms = argc > 1 ? std::atoi(argv[1]) : 600;

    auto cfg = fcs::calibration::CalibrationConfig::load_from_file("calibration.toml");
    if (!cfg) {
        std::fprintf(stderr, "读 calibration.toml 失败: %s\n", cfg.error().c_str());
        return 1;
    }

    auto client = ipc::ShmClient::connect();
    if (!client) {
        std::fprintf(stderr, "连不上共享内存，先起 ./build/bin/orbbec-bridge\n");
        return 1;
    }

    fcs::calibration::ChessboardDetector detector(
        cfg->board.width, cfg->board.height, cfg->board.square_size);
    fcs::calibration::IntrinsicCalibrator calibrator(cfg->capture, cfg->intrinsic);

    const auto target = static_cast<int>(cfg->capture.min_samples);
    std::printf("棋盘格 %ux%u 内角点，方格 %.1fmm，目标 %d 张，间隔 %dms\n请拿着板子在视野内变换角度/距离/位置\n",
                cfg->board.width, cfg->board.height, cfg->board.square_size * 1000, target,
                interval_ms);

    int accepted = 0;
    int attempts  = 0;
    auto last_try = std::chrono::steady_clock::now() - std::chrono::milliseconds(interval_ms);

    while (accepted < static_cast<int>(cfg->capture.max_samples)) {
        auto img = client->recv_image();
        if (!img) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now - last_try < std::chrono::milliseconds(interval_ms)) {
            continue;
        }
        last_try = now;
        ++attempts;

        // shm 里是 RGB
        cv::Mat gray;
        cv::cvtColor(img->image, gray, cv::COLOR_RGB2GRAY);

        auto det = detector.detect(gray, img->timestamp_ns);
        if (!det) {
            std::printf("\r[%d/%d] 没有棋盘格...          ", accepted, target);
            std::fflush(stdout);
            continue;
        }
        if (!calibrator.is_diverse_enough(*det)) {
            std::printf("\r[%d/%d] 角度太接近，换个姿势      ", accepted, target);
            std::fflush(stdout);
            continue;
        }
        if (auto r = calibrator.add_sample(*det); !r) {
            std::printf("\r[%d/%d] 拒收: %s      ", accepted, target, r.error().c_str());
            std::fflush(stdout);
            continue;
        }

        ++accepted;
        // 打印样本中心位置，便于确认是否覆盖四角
        const auto& pts = det->image_points;
        const cv::Point2f c = pts[pts.size() / 2];
        std::printf("\r[%d/%d] 已采集  中心(%.0f,%.0f)/(%d,%d)     ", accepted, target, c.x, c.y,
                    cfg->width, cfg->height);
        std::fflush(stdout);
        if (accepted >= target + 5) {
            break; // 多采几张，够稳就收
        }
    }

    std::printf("\n采样结束：%d 张（尝试 %d 次），求解中...\n", accepted, attempts);

    auto result = calibrator.calibrate(cv::Size(static_cast<int>(cfg->width),
                                                static_cast<int>(cfg->height)));
    if (!result) {
        std::fprintf(stderr, "标定失败: %s\n", result.error().c_str());
        return 1;
    }

    const auto& r   = *result;
    const auto& K   = r.camera_matrix;
    const auto& D   = r.distort_coefficient;
    std::printf("\n=== 结果：%u 张样本，RMS %.4f px，%ux%u ===\n", r.num_samples, r.rms_error,
                r.width, r.height);
    std::printf("fx=%.4f  fy=%.4f\n", K(0, 0), K(1, 1));
    std::printf("cx=%.4f  cy=%.4f\n", K(0, 2), K(1, 2));
    std::printf("distortion(k1,k2,p1,p2,k3) = %.6f %.6f %.6f %.6f %.6f\n",
                D(0, 0), D(0, 1), D(0, 2), D(0, 3), D(0, 4));
    std::printf("\n把上面四个数填进 src/orbbec_bridge.cpp 顶部的 kFx/kFy/kCx/kCy\n");
    return 0;
}
