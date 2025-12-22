/**
 * @file sp_vision.cpp
 * @brief 使用共享内存 IPC 的视觉处理示例
 *
 * 从 Rust 模拟器接收图像，处理后发送云台控制命令
 */

#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <thread>
#include "config.hpp"
#include "auto_aim/detector.hpp"
#include "ipc/shm_client.hpp"
#include "spdlog_hook.hpp"
#include "toml++/toml.hpp"

using namespace std::chrono_literals;

int main() {
    hook_cstream();

    // 加载配置
    SpVisionConfig config;
    toml::table tbl;
    try {
        tbl = toml::parse_file("sp_vision.toml");
    } catch (const toml::parse_error &err) {
        const auto &src = err.source();
        std::cerr << fmt::format("parse {}:{}:{}: {}", *src.path, src.begin.line, src.begin.column, err.description());
        return 1;
    }
    if (auto result = config.read_from(tbl); !result) {
        std::cerr << "read sp_vision.toml: " + result.error();
        return 1;
    }

    std::cout << "=== SP Vision IPC Demo ===\n";
    std::cout << "等待 Rust 模拟器启动...\n";

    // 等待连接到共享内存
    std::optional<ipc::ShmClient> client;
    while (!client) {
        if (auto result = ipc::ShmClient::connect()) {
            client = std::move(*result);
            std::cout << "已连接到共享内存!\n";
        } else {
            std::cout << "." << std::flush;
            std::this_thread::sleep_for(500ms);
        }
    }

    // 初始化检测器
    auto_aim::TraditionalDetector detector(config);
    auto_aim::DetectorWorkBuffers detector_buffers;

    std::cout << "开始处理...\n";

    uint64_t frame_count = 0;
    uint64_t last_seq = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        // 接收图像
        auto frame = client->recv_image();
        if (!frame) {
            // 没有新图像，短暂休眠
            std::this_thread::sleep_for(100us);
            continue;
        }

        // 跳帧检测
        if (frame->seq != last_seq + 1 && last_seq > 0) {
            std::cout << "跳帧: " << (frame->seq - last_seq - 1) << " 帧\n";
        }
        last_seq = frame->seq;

        // RGB -> BGR 转换 (OpenCV 使用 BGR)
        cv::Mat bgr;
        cv::cvtColor(frame->image, bgr, cv::COLOR_RGB2BGR);

        // 预处理
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, gray, config.detector.threshold, 255, cv::THRESH_BINARY);

        // 检测装甲板
        auto armor_slots = detector.detect(detector_buffers, bgr, gray, frame->seq, frame->timestamp_ns);

        // 如果检测到装甲板，发送云台命令
        if (armor_slots.armor_count > 0) {
            // 找到最近的装甲板 (简化逻辑)
            int best_idx = 0;
            float min_dist = 1e9f;
            for (int i = 0; i < 10; ++i) {
                if (armor_slots.is_valid(i)) {
                    const auto &armor = armor_slots.armors[i];
                    if (float dist = std::sqrt(armor.x2d * armor.x2d + armor.y2d * armor.y2d); dist < min_dist) {
                        min_dist = dist;
                        best_idx = i;
                    }
                }
            }

            const auto &armor = armor_slots.armors[best_idx];

            // 计算云台角度 (简化: 基于图像中心偏移)
            float cx = static_cast<float>(frame->image.cols) / 2.0f;
            float cy = static_cast<float>(frame->image.rows) / 2.0f;

            float yaw_offset = (armor.x2d - cx) / cx * 45.0f; // 假设 FOV 90度
            float pitch_offset = (armor.y2d - cy) / cy * 30.0f; // 假设垂直 FOV 60度

            // 发送云台命令
            client->send_gimbal_cmd(yaw_offset,
                                    90.0f + pitch_offset, // Rust 端期望的角度格式
                                    min_dist / 100.0f, // 估计距离 (非常粗略)
                                    armor.confidence > 0.8f // 置信度高时建议开火
            );
        } else {
            // 无目标
            client->send_gimbal_cmd(0.0f, 90.0f, -1.0f, false);
        }

        frame_count++;

        // 每秒打印统计
        auto now = std::chrono::steady_clock::now();
        if (auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            elapsed > 0 && frame_count % 60 == 0) {
            std::cout << "\r[" << elapsed << "s] "
                      << "帧数: " << frame_count << ", FPS: " << (frame_count / elapsed)
                      << ", 装甲板: " << static_cast<int>(armor_slots.armor_count) << "   " << std::flush;
        }
    }
}
