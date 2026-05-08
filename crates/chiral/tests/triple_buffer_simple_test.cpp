#include "chiral/talos_triple_buffer_shm.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sys/mman.h>
#include <thread>

using namespace talos::chiral;
using namespace talos::chiral::ipc;

int main() {
    // 清理残留共享内存，避免 O_EXCL 失败
    (void)::shm_unlink(TALOS_SHM_NAME);

    // 创建 Writer
    auto writer = TalosDataWriter::create();
    if (!writer) {
        std::cerr << "Failed to create writer\n";
        return 1;
    }

    std::cout << "Writer created successfully\n";

    // 写入/读取一次基础数据
    TalosData data;
    data.gimbal_link.translation.x = 1.5;
    data.gimbal_link.translation.y = 2.5;
    data.gimbal_link.translation.z = 3.5;

    writer->write(data);
    std::cout << "Data written\n";

    // 创建 Reader
    auto reader = TalosDataReader::open();
    if (!reader) {
        std::cerr << "Failed to open reader\n";
        return 1;
    }

    std::cout << "Reader opened successfully\n";

    // 读取数据
    if (auto read_data = reader->read_new()) {
        std::cout << "Data read: x=" << read_data->gimbal_link.translation.x
                  << " y=" << read_data->gimbal_link.translation.y
                  << " z=" << read_data->gimbal_link.translation.z << "\n";
    } else {
        std::cerr << "No data to read\n";
        return 1;
    }

    // 压测：验证高频下无字段撕裂
    constexpr int iterations = 200000;
    std::atomic<bool> done{false};
    std::atomic<int> reads{0};
    std::atomic<int> bad{0};

    std::thread writer_thread([&]() {
        TalosData d;
        for (int i = 1; i <= iterations; ++i) {
            d.gimbal_link.translation.x = static_cast<double>(i);
            d.gimbal_link.translation.y = static_cast<double>(i * 2);
            d.gimbal_link.translation.z = static_cast<double>(i * 3);
            d.state.robot.yaw           = static_cast<double>(i) * 0.001;
            writer->write(d);
        }
        done.store(true, std::memory_order_release);
    });

    std::thread reader_thread([&]() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
        while (std::chrono::steady_clock::now() < deadline) {
            if (auto d = reader->read_new()) {
                const auto x = static_cast<int>(std::llround(d->gimbal_link.translation.x));
                const auto y = static_cast<int>(std::llround(d->gimbal_link.translation.y));
                const auto z = static_cast<int>(std::llround(d->gimbal_link.translation.z));
                const auto yaw_delta =
                    std::fabs(d->state.robot.yaw - static_cast<double>(x) * 0.001);
                if (y != x * 2 || z != x * 3 || yaw_delta > 1e-9) {
                    bad.fetch_add(1, std::memory_order_relaxed);
                }
                reads.fetch_add(1, std::memory_order_relaxed);
            } else {
                if (done.load(std::memory_order_acquire)) {
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }
    });

    writer_thread.join();
    reader_thread.join();

    if (reads.load(std::memory_order_relaxed) <= 0) {
        std::cerr << "No data captured in stress test\n";
        return 2;
    }
    if (bad.load(std::memory_order_relaxed) != 0) {
        std::cerr << "Data tearing detected: bad=" << bad.load(std::memory_order_relaxed) << "\n";
        return 3;
    }

    std::cout << "Stress read count: " << reads.load(std::memory_order_relaxed) << "\n";
    std::cout << "Triple buffer test PASSED\n";
    return 0;
}
