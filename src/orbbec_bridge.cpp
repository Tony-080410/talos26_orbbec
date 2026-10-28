// Orbbec Gemini 335 -> talos daedalus 共享内存桥接。
//
// talos 端零改动：以 backend = "Daedalus" 运行，从 /tmp/talos_ipc_* 取图。
// 本程序扮演模拟器/回放的角色，把相机帧写进同一个共享内存。

#include <libobsensor/ObSensor.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "shm_client.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <memory>
#include <thread>

namespace {

std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }

// shm 里图像尺寸是编译期常量（shm_layout.hpp:16），必须是 1440x1080。
// 相机给 1920x1080，居中裁剪而不是缩放：缩放会让 fx/fy 不等，相机模型直接坏掉。
constexpr int kSrcW  = 1920;
constexpr int kSrcH  = 1080;
constexpr int kOutW  = 1440;
constexpr int kOutH  = 1080;
constexpr int kCropX = (kSrcW - kOutW) / 2; // 240

struct Intrinsic {
    double fx, fy, cx, cy;
    double k1, k2, p1, p2, k3;              // OpenCV 顺序
};

// 出厂内参：SDK 读出 @1920x1080，裁剪后主点左移 kCropX
constexpr Intrinsic kFactory{1034.4719, 1034.3972, 972.3774 - kCropX, 544.9583, 0.005418,
                             -0.044486, 0.000492,  0.000234,          0.030413};

// 自标定：35 张棋盘格 11x7@23mm，RMS 0.507 px，1440x1080
constexpr Intrinsic kCalib{1052.4552, 1050.7859, 704.7081,  521.6971, -0.037047,
                           0.091525,  -0.002728, -0.011089, -0.192333};

// 静态外参，抄自 config/robot/infantry_4.toml
constexpr double kCamLinkX = 0.05925, kCamLinkY = 0.0, kCamLinkZ = 0.04313;
constexpr double kMuzzleX = 0.0633901, kMuzzleY = 0.0, kMuzzleZ = 0.0;

uint64_t now_ns() {
    // 用 system_clock（Unix epoch）而不是 steady_clock：Foxglove 按 Unix 时间戳换算
    // 时间轴，steady_clock 是开机计时，界面上会显示成 1970 年。
    // 图像和 pose 时间戳必须同源，talos 的 tf lookup 才能对上。
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
}

ipc::ShmClient::Pose
    identity_pose(uint64_t seq, uint64_t ts, double x = 0, double y = 0, double z = 0) {
    ipc::ShmClient::Pose p{};
    p.x  = x;
    p.y  = y;
    p.z  = z;
    p.qw = 1.0;
    p.qx = p.qy = p.qz = 0.0;
    p.frame_seq        = seq;
    p.timestamp_ns     = ts;
    return p;
}

void publish_static_poses(const ipc::ShmClient& client, uint64_t seq, uint64_t ts) {
    client.publish_pose(ipc::POSE_GIMBAL, identity_pose(seq, ts));
    client.publish_pose(ipc::POSE_ODOM, identity_pose(seq, ts));
    client.publish_pose(ipc::POSE_CAMERA, identity_pose(seq, ts, kCamLinkX, kCamLinkY, kCamLinkZ));
    client.publish_pose(ipc::POSE_MUZZLE, identity_pose(seq, ts, kMuzzleX, kMuzzleY, kMuzzleZ));
}

} // namespace

int main(int argc, char** argv) {
    // --calib 用自标定内参，缺省用出厂内参
    const bool use_calib = argc > 1 && std::strcmp(argv[1], "--calib") == 0;
    const Intrinsic& K   = use_calib ? kCalib : kFactory;
    std::printf("intrinsic: %s\n", use_calib ? "self-calibrated" : "factory");
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    auto shm = ipc::ShmClient::create();
    if (!shm) {
        std::printf("create shm failed\n");
        return 1;
    }
    auto& client = *shm;

    ipc::CameraInfo info{};
    info.timestamp_ns  = now_ns();
    info.fx            = K.fx;
    info.fy            = K.fy;
    info.cx            = K.cx;
    info.cy            = K.cy;
    info.distortion[0] = K.k1;
    info.distortion[1] = K.k2;
    info.distortion[2] = K.p1;
    info.distortion[3] = K.p2;
    info.distortion[4] = K.k3;
    info.width         = kOutW;
    info.height        = kOutH;
    client.publish_camera_info(info);

    std::printf(
        "camera_info: fx=%.3f fy=%.3f cx=%.3f cy=%.3f  %ux%u\n", info.fx, info.fy, info.cx, info.cy,
        info.width, info.height);

    // Context 必须活到程序结束：它持有 SDK 全局状态，提前析构会让 Device/Pipeline 失效。
    ob::Context ctx;
    std::unique_ptr<ob::Pipeline> pipe;
    try {
        auto devices = ctx.queryDeviceList();
        if (devices->deviceCount() == 0) {
            std::printf("no orbbec device\n");
            return 1;
        }
        auto dev    = devices->getDevice(0);
        auto sensor = dev->getSensor(OB_SENSOR_COLOR);
        auto list   = sensor->getStreamProfileList();

        std::shared_ptr<ob::VideoStreamProfile> chosen;
        for (uint32_t i = 0; i < list->getCount(); ++i) {
            auto p = list->getProfile(i)->as<ob::VideoStreamProfile>();
            if (p->getWidth() == kSrcW && p->getHeight() == kSrcH && p->getFps() == 30
                && p->getFormat() == OB_FORMAT_MJPG) {
                chosen = p;
                break;
            }
        }
        if (!chosen) {
            std::printf("no %dx%d@30 MJPG profile\n", kSrcW, kSrcH);
            return 1;
        }

        auto cfg = std::make_shared<ob::Config>();
        cfg->enableStream(chosen);
        // Pipeline 没有移动赋值，只能原地构造/用 unique_ptr，否则 impl_ 被浅拷贝后 double free
        pipe = std::make_unique<ob::Pipeline>(dev);
        pipe->start(cfg);
        std::printf(
            "color stream started: %ux%u @%ufps\n", chosen->getWidth(), chosen->getHeight(),
            chosen->getFps());
    } catch (const ob::Error& e) {
        std::printf("orbbec error: %s\n", e.what());
        return 1;
    }

    std::thread heartbeat([&client] {
        while (g_run.load(std::memory_order_relaxed)) {
            client.update_heartbeat();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    uint64_t seq       = 1;
    uint64_t published = 0;
    const auto start   = std::chrono::steady_clock::now();

    while (g_run.load(std::memory_order_relaxed)) {
        std::shared_ptr<ob::FrameSet> frames;
        try {
            frames = pipe->waitForFrameset(500);
        } catch (const ob::Error& e) {
            std::printf("waitForFrameset: %s\n", e.what());
            continue;
        }
        if (!frames) {
            continue;
        }
        auto color = frames->getFrame(OB_FRAME_COLOR);
        if (!color) {
            continue;
        }
        auto cf = color->as<ob::ColorFrame>();

        // MJPG 帧数据本身就是 JPEG，直接解码
        cv::Mat jpeg(1, static_cast<int>(cf->getDataSize()), CV_8UC1, cf->getData());
        cv::Mat img = cv::imdecode(jpeg, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::printf("imdecode failed, size=%u\n", cf->getDataSize());
            continue;
        }
        if (img.cols != kSrcW || img.rows != kSrcH) {
            std::printf("unexpected frame %dx%d\n", img.cols, img.rows);
            continue;
        }

        // 裁剪 + BGR->RGB：shm 里约定是 RGB，talos 端读出来会再转回 BGR。
        // cvtColor 输出新的连续 Mat，正好满足 publish_image 的整体 memcpy。
        cv::Mat crop;
        cv::cvtColor(img(cv::Rect(kCropX, 0, kOutW, kOutH)), crop, cv::COLOR_BGR2RGB);

        const uint64_t ts = now_ns();
        client.publish_image(crop, seq, ts);
        publish_static_poses(client, seq, ts);
        ++seq;
        ++published;

        if (published % 30 == 0) {
            const auto elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf(
                "published %llu frames  %.1f fps\n", static_cast<unsigned long long>(published),
                published / elapsed);
        }
    }

    g_run.store(false);
    if (heartbeat.joinable()) {
        heartbeat.join();
    }
    if (pipe) {
        try {
            pipe->stop();
        } catch (...) {}
    }
    std::printf("stopped, published %llu frames\n", static_cast<unsigned long long>(published));
    return 0;
}
