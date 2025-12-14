#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <iostream>
#include <cstdint>
#include "triple_buffer.hpp"

struct Frame {
    int value = 0;
};

TEST(TripleBufferTest, ConcurrentProducerConsumer) {
    spike::TripleBuffer<Frame> buffer;

    std::atomic<bool> stop{false};
    std::atomic<int> producer_count{0};
    std::atomic<int> consumer_count{0};
    std::atomic<int> version_decrement{0};  // Version decrement (not allowed)

    // Producer: high-speed writing
    std::thread producer([&]() {
        int val = 1;
        while (!stop.load()) {
            Frame* ptr = buffer.begin_write();
            ptr->value = val;
            buffer.publish_write();
            val++;
            producer_count++;
        }
    });

    // Consumer: copy data (avoid race condition)
    std::thread consumer([&]() {
        uint64_t last_version = 0;
        int last_value = 0;
        while (!stop.load()) {
            auto [ptr, version] = buffer.try_read();
            if (ptr != nullptr) {
                // Copy data immediately to avoid producer overwriting
                Frame local_copy = *ptr;
                consumer_count++;

                // Version should monotonically increase (no decrement)
                if (version < last_version) {
                    version_decrement++;
                    std::cout << "Version decrement: " << version << " < " << last_version
                              << std::endl;
                }

                // Data value should also monotonically increase (new value should not be less than previous)
                if (local_copy.value < last_value && local_copy.value > 0) {
                    std::cout << "Data decrement: " << local_copy.value << " < " << last_value
                              << std::endl;
                }

                if (version > last_version) {
                    last_version = version;
                    last_value = local_copy.value;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    });

    // Run for a period of time
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    stop.store(true);

    producer.join();
    consumer.join();

    std::cout << "Producer writes: " << producer_count.load() << std::endl;
    std::cout << "Consumer reads: " << consumer_count.load() << std::endl;
    std::cout << "Version decrements: " << version_decrement.load() << std::endl;

    // Assertion: there should be no version decrement (indicating no corruption)
    EXPECT_EQ(version_decrement.load(), 0)
        << "Triple-buffer should guarantee no version decrements";
    EXPECT_GT(producer_count.load(), 0);
    EXPECT_GT(consumer_count.load(), 0);
}

TEST(TripleBufferTest, DataIntegrity) {
    spike::TripleBuffer<Frame> buffer;

    std::vector<int> written_values;
    std::vector<int> read_values;
    std::atomic<bool> stop{false};

    // Producer writes monotonically increasing values
    std::thread producer([&]() {
        for (int i = 1; i <= 100; i++) {
            Frame* ptr = buffer.begin_write();
            ptr->value = i;
            written_values.push_back(i);
            buffer.publish_write();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        stop.store(true);
    });

    // Consumer reads data (copy)
    std::thread consumer([&]() {
        while (!stop.load()) {
            auto [ptr, version] = buffer.try_read();
            if (ptr != nullptr) {
                Frame local_copy = *ptr;  // Key: copy immediately
                read_values.push_back(local_copy.value);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(150));
        }
    });

    producer.join();
    consumer.join();

    // Verify: all read values are within the range of written values
    for (int val : read_values) {
        EXPECT_GE(val, 1) << "Read value should be >= 1";
        EXPECT_LE(val, 100) << "Read value should be <= 100";
    }

    // Verify: read values are monotonically non-decreasing
    for (size_t i = 1; i < read_values.size(); i++) {
        EXPECT_GE(read_values[i], read_values[i - 1])
            << "Read values should be monotonically non-decreasing";
    }

    std::cout << "Written: " << written_values.size() << ", "
              << "Read: " << read_values.size() << std::endl;
}

