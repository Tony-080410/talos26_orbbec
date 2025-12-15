#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <iostream>
#include "triple_buffer.hpp"

// 大结构体，更容易暴露撕裂问题
struct LargeFrame {
    std::array<uint64_t, 64> data{}; // 512 bytes

    void fill(const auto v) {
        for (auto &x: data) x = v;
    }

    [[nodiscard]] bool is_consistent() const {
        const auto first = data[0];
        return std::all_of(data.begin(), data.end(), [&](const auto v) {
            return v == first;
        });
    }

    [[nodiscard]] auto value() const { return data[0]; }
};

// 1. 基础功能测试
TEST(TripleBufferTest, BasicOperation) {
    spike::TripleBuffer<int> buffer;

    // 初始状态没有新数据
    EXPECT_FALSE(buffer.has_new_frame());
    EXPECT_EQ(buffer.borrow(), std::nullopt);

    // 写入后有新数据
    buffer.borrow_mut() = 42;
    buffer.publish();
    EXPECT_TRUE(buffer.has_new_frame());

    // 读取
    const auto result = buffer.borrow();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(**result, 42);

    // 读完后没有新数据
    EXPECT_FALSE(buffer.has_new_frame());
    EXPECT_EQ(buffer.borrow(), std::nullopt);
}

// 2. 连续写入，只读到最新
TEST(TripleBufferTest, ConsecutiveWrites) {
    spike::TripleBuffer<int> buffer;

    for (int i = 1; i <= 100; ++i) {
        buffer.borrow_mut() = i;
        buffer.publish();
    }

    // 只能读到最新的一个
    const auto result = buffer.borrow();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(**result, 100);

    // 没有更多数据
    EXPECT_EQ(buffer.borrow(), std::nullopt);
}

// 3. current() 总是有效
TEST(TripleBufferTest, CurrentAlwaysValid) {
    spike::TripleBuffer<int> buffer;

    // 即使没有 publish，current() 也能用（返回默认值）
    EXPECT_EQ(buffer.current(), 0);

    buffer.borrow_mut() = 42;
    buffer.publish();
    const auto current = buffer.borrow();
    EXPECT_EQ(current.has_value(), true) << "Previously we published data, so we must receive the data";

    // borrow 后 current 返回刚读的值
    EXPECT_EQ(*current.value(), 42);
}

// 4. 高压单调性测试（无 sleep）
TEST(TripleBufferTest, StressMonotonicity) {
    spike::TripleBuffer<uint64_t> buffer;

    std::atomic stop{false};
    std::atomic<uint64_t> violations{0};
    std::atomic<uint64_t> reads{0};

    std::thread producer([&] {
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            buffer.borrow_mut() = v++;
            buffer.publish();
        }
    });

    std::thread consumer([&] {
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto opt = buffer.borrow()) {
                const auto curr = **opt;
                reads.fetch_add(1, std::memory_order_relaxed);
                if (curr <= last) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                last = curr;
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "Reads: " << reads.load() << ", Violations: " << violations.load() << std::endl;
    EXPECT_EQ(violations.load(), 0);
    EXPECT_GT(reads.load(), 10000); // 确保实际跑了
}

// 5. 高压数据完整性测试（大结构体，无 sleep）
TEST(TripleBufferTest, StressDataIntegrity) {
    spike::TripleBuffer<LargeFrame> buffer;

    std::atomic stop{false};
    std::atomic<uint64_t> corrupted{0};
    std::atomic<uint64_t> reads{0};

    std::thread producer([&] {
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            buffer.borrow_mut().fill(v++);
            buffer.publish();
        }
    });

    std::thread consumer([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto opt = buffer.borrow()) {
                reads.fetch_add(1, std::memory_order_relaxed);
                if (!(*opt)->is_consistent()) {
                    corrupted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "Reads: " << reads.load() << ", Corrupted: " << corrupted.load() << std::endl;
    EXPECT_EQ(corrupted.load(), 0);
    EXPECT_GT(reads.load(), 10000);
}

// 6. 快生产者慢消费者（会丢帧，但不能乱序或撕裂）
TEST(TripleBufferTest, FastProducerSlowConsumer) {
    spike::TripleBuffer<uint64_t> buffer;

    std::atomic stop{false};
    std::atomic<uint64_t> violations{0};

    std::thread producer([&] {
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            buffer.borrow_mut() = v++;
            buffer.publish();
            // 不 sleep，全速写
        }
    });

    std::thread consumer([&] {
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto opt = buffer.borrow()) {
                const uint64_t curr = **opt;
                if (curr <= last) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                last = curr;
            }
            // 慢消费者
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    EXPECT_EQ(violations.load(), 0);
}

// 7. 慢生产者快消费者（大量空读）
TEST(TripleBufferTest, SlowProducerFastConsumer) {
    spike::TripleBuffer<uint64_t> buffer;

    std::atomic stop{false};
    std::atomic<uint64_t> violations{0};
    std::atomic<uint64_t> empty_reads{0};
    std::atomic<uint64_t> valid_reads{0};

    std::thread producer([&] {
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            buffer.borrow_mut() = v++;
            buffer.publish();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::thread consumer([&] {
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto opt = buffer.borrow()) {
                const auto curr = **opt;
                valid_reads.fetch_add(1, std::memory_order_relaxed);
                if (curr <= last) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                last = curr;
            } else {
                empty_reads.fetch_add(1, std::memory_order_relaxed);
            }
            // 不 sleep，全速读
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "Valid: " << valid_reads.load()
            << ", Empty: " << empty_reads.load() << std::endl;
    EXPECT_EQ(violations.load(), 0);
    EXPECT_GT(empty_reads.load(), valid_reads.load()); // 空读应该更多
}

// 8. 多次 borrow_mut 不 publish
TEST(TripleBufferTest, MultipleBorrowMutWithoutPublish) {
    spike::TripleBuffer<int> buffer;

    buffer.borrow_mut() = 1;
    buffer.borrow_mut() = 2;
    buffer.borrow_mut() = 3;
    // 只 publish 一次
    buffer.publish();

    const auto result = buffer.borrow();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(**result, 3);
}

// 9. 交替读写
TEST(TripleBufferTest, AlternatingReadWrite) {
    spike::TripleBuffer<int> buffer;

    for (int i = 1; i <= 1000; ++i) {
        buffer.borrow_mut() = i;
        buffer.publish();

        auto result = buffer.borrow();
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(**result, i);

        // 第二次读应该是空的
        EXPECT_EQ(buffer.borrow(), std::nullopt);
        EXPECT_EQ(buffer.has_new_frame(), false);
    }
}
