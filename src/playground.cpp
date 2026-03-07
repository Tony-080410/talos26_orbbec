#include "shm_client.hpp"
#include "spdlog_hook.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <thread>

using namespace std::chrono_literals;

// Global signal handler
std::atomic<bool> g_running{true};

void signal_handler(int) { g_running = false; }

int main() {
    // Setup signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    init_logger();
    SPDLOG_INFO("=== Talos Daedalus IPC Playground ===");
    SPDLOG_INFO("Press Ctrl+C to quit");
    const auto start = std::chrono::system_clock::now();

    // 创建共享内存 (生产者模式，用于测试)
    SPDLOG_INFO("creating shared memory regions...");
    auto client_result = ipc::ShmClient::create();
    if (!client_result) {
        SPDLOG_CRITICAL(
            "failed to create shared memory: {}", static_cast<int>(client_result.error()));
        return 1;
    }

    auto& client = *client_result;
    SPDLOG_INFO("shared memory created successfully ({}x{})", ipc::IMAGE_WIDTH, ipc::IMAGE_HEIGHT);

    uint64_t frame_count = 0;

    // Main loop
    SPDLOG_INFO("starting main loop...");
    while (g_running) {

        client.update_heartbeat();

        // 测试 2: 接收图像
        if (auto frame = client.recv_image()) {
            if (frame_count % 30 == 0) { // Log every 30 frames
                SPDLOG_INFO(
                    "frame={}, seq={}, size={}x{}, channels={}", frame_count, frame->seq,
                    frame->image.cols, frame->image.rows, frame->image.channels());
            }
        }

        // 测试 3: 定期发送云台命令
        if (frame_count % 10 == 0) {
            client.send_gimbal_cmd(15.0f, -8.0f, 5.2f, frame_count % 30 == 0);
        }

        frame_count++;
        std::this_thread::sleep_for(33ms); // ~30 FPS
    }

    // Cleanup
    SPDLOG_INFO(
        "shutting down... ({} frames, {}ms)", frame_count,
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start)
            .count());

    // Final header info
    const auto& header = client.header();
    SPDLOG_INFO(
        "magic={:#x}, version={}, producer_alive={}", header.magic, header.version,
        client.is_producer_alive());

    const auto& cam_info = client.camera_info();
    SPDLOG_INFO("fx={}, fy={}, cx={}, cy={}", cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy);

    return 0;
}
