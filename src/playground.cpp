#include "chiral/gimbal.hpp"
#include "spdlog_hook.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
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
    SPDLOG_INFO("=== Talos Playground ===");
    SPDLOG_INFO("Press Ctrl+C to quit");
    const auto start = std::chrono::system_clock::now();

    SPDLOG_INFO("creating shared memory regions...");
    auto client_result = talos::chiral::gimbal::GimbalEndpoint::create();
    if (!client_result) {
        SPDLOG_CRITICAL(
            "failed to create shared memory: {}", static_cast<int>(client_result.error()));
        return 1;
    }

    auto& client = *client_result;

    uint64_t frame_count = 0;

    // Main loop
    SPDLOG_INFO("starting main loop...");
    float roll, pitch, yaw;
    while (g_running) {
        // 测试 2: 接收指令
        if (auto frame = client->read_new()) {
            yaw   = frame->target_yaw;
            pitch = frame->target_pitch;
            SPDLOG_INFO("{} {} {}", frame->fire_advice, yaw, pitch);
        }

        // 测试 3: 定期发送云台数据
        if (frame_count % 10 == 0) {
            client->write(
                talos::chiral::gimbal::McuData(
                    talos::chiral::gimbal::Color::Red, yaw, pitch, 0.0, 0.0, 0.0, 0.0));
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

    return 0;
}
