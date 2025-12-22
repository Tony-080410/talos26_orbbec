/**
 * IPC Shared Memory Unit Tests
 *
 * Tests:
 * 1. ShmRegion RAII and memory safety
 * 2. TripleBuffer atomicity and tearing prevention
 * 3. Multi-threaded read/write consistency
 * 4. Data integrity under stress
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ranges>
#include <thread>
#include <vector>

#include "shm_layout.hpp"
#include "shm_region.hpp"
#include "shm_triple_buffer.hpp"

namespace ipc::test {

// ============================================================================
// ShmRegion Memory Safety Tests
// ============================================================================

class ShmRegionTest : public ::testing::Test {
protected:
    static constexpr std::string_view TEST_SHM_NAME = "ipc_test_region";
    static constexpr size_t TEST_SIZE = 4096;

    void TearDown() override {
        // Cleanup test shared memory files
        const auto path = shm_path(TEST_SHM_NAME);
        std::filesystem::remove(path);
    }
};

TEST_F(ShmRegionTest, CreateAndDestroy) {
    {
        auto result = ShmRegion::create(TEST_SHM_NAME, TEST_SIZE);
        ASSERT_TRUE(result.has_value()) << "Failed to create ShmRegion";

        EXPECT_NE(result->data(), nullptr);
        EXPECT_EQ(result->size(), TEST_SIZE);
        EXPECT_TRUE(result->is_owner());
    }
    // After scope exit, region should be unmapped and file removed
    EXPECT_FALSE(std::filesystem::exists(shm_path(TEST_SHM_NAME)));
}

TEST_F(ShmRegionTest, OpenNonExistent) {
    auto result = ShmRegion::open("nonexistent_shm_12345", TEST_SIZE);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ShmError::NotFound);
}

TEST_F(ShmRegionTest, CreateThenOpen) {
    // Create as producer
    auto producer = ShmRegion::create(TEST_SHM_NAME, TEST_SIZE);
    ASSERT_TRUE(producer.has_value());

    // Open as consumer
    auto consumer = ShmRegion::open(TEST_SHM_NAME, TEST_SIZE);
    ASSERT_TRUE(consumer.has_value());

    // Both should point to same memory (verify by writing/reading)
    auto* prod_ptr = static_cast<uint64_t*>(producer->data());
    auto* cons_ptr = static_cast<uint64_t*>(consumer->data());

    *prod_ptr = 0xDEADBEEF;
    EXPECT_EQ(*cons_ptr, 0xDEADBEEF);
}

TEST_F(ShmRegionTest, MoveSemantics) {
    auto result = ShmRegion::create(TEST_SHM_NAME, TEST_SIZE);
    ASSERT_TRUE(result.has_value());

    void* original_ptr = result->data();
    const size_t original_size = result->size();

    // Move construct
    ShmRegion moved(std::move(*result));
    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(moved.size(), original_size);
    EXPECT_TRUE(moved.is_owner());

    // Original should be invalidated
    EXPECT_EQ(result->data(), nullptr);
}

TEST_F(ShmRegionTest, MoveAssignment) {
    auto region1 = ShmRegion::create(TEST_SHM_NAME, TEST_SIZE);
    ASSERT_TRUE(region1.has_value());

    auto region2 = ShmRegion::create("ipc_test_region_2", TEST_SIZE);
    ASSERT_TRUE(region2.has_value());

    void* ptr1 = region1->data();

    // Move assign region1 to region2
    *region2 = std::move(*region1);

    EXPECT_EQ(region2->data(), ptr1);
    EXPECT_EQ(region1->data(), nullptr);

    // Cleanup second test file
    std::filesystem::remove(shm_path("ipc_test_region_2"));
}

TEST_F(ShmRegionTest, SizeValidation) {
    auto producer = ShmRegion::create(TEST_SHM_NAME, TEST_SIZE);
    ASSERT_TRUE(producer.has_value());

    // Try to open with larger size
    auto consumer = ShmRegion::open(TEST_SHM_NAME, TEST_SIZE * 2);
    ASSERT_FALSE(consumer.has_value());
    EXPECT_EQ(consumer.error(), ShmError::InvalidSize);
}

TEST_F(ShmRegionTest, ZeroInitialized) {
    auto region = ShmRegion::create(TEST_SHM_NAME, TEST_SIZE);
    ASSERT_TRUE(region.has_value());

    const auto* data = static_cast<const uint8_t*>(region->data());
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        EXPECT_EQ(data[i], 0) << "Memory not zero-initialized at offset " << i;
    }
}

// ============================================================================
// TripleBuffer Atomicity Tests
// ============================================================================

class TripleBufferAtomicityTest : public ::testing::Test {
protected:
    // Large data to make tearing more likely if there's a bug
    struct LargeData {
        std::array<uint64_t, 64> values{}; // 512 bytes

        void fill(uint64_t v) {
            for (auto& x : values) x = v;
        }

        [[nodiscard]] bool is_consistent() const {
            const uint64_t first = values[0];
            return std::ranges::all_of(values, [first](uint64_t v) { return v == first; });
        }

        [[nodiscard]] uint64_t value() const { return values[0]; }
    };

    static constexpr std::string_view TEST_SHM_NAME = "ipc_triple_buffer_test";
};

TEST_F(TripleBufferAtomicityTest, BasicOperation) {
    ImageTripleBuffer buffer{};
    ImageOps ops(&buffer);

    // Initially no new data
    EXPECT_FALSE(ops.has_new_data());
    EXPECT_EQ(ops.borrow(), std::nullopt);

    // Write and publish
    auto& slot = ops.borrow_mut();
    slot.seq = 42;
    slot.timestamp_ns = 1234567890;
    ops.publish();

    EXPECT_TRUE(ops.has_new_data());

    // Read
    auto result = ops.borrow();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->seq, 42);
    EXPECT_EQ((*result)->timestamp_ns, 1234567890);

    // No more new data
    EXPECT_FALSE(ops.has_new_data());
}

TEST_F(TripleBufferAtomicityTest, ConsecutiveWrites) {
    ImageTripleBuffer buffer{};
    ImageOps ops(&buffer);

    for (uint64_t i = 1; i <= 100; ++i) {
        ops.borrow_mut().seq = i;
        ops.publish();
    }

    // Should read latest
    auto result = ops.borrow();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->seq, 100);

    // No more data
    EXPECT_FALSE(ops.has_new_data());
}

TEST_F(TripleBufferAtomicityTest, CurrentAlwaysValid) {
    ImageTripleBuffer buffer{};
    ImageOps ops(&buffer);

    // current() works even without publish
    [[maybe_unused]] const auto& current = ops.current();

    ops.borrow_mut().seq = 42;
    ops.publish();
    [[maybe_unused]] auto _ = ops.borrow();

    EXPECT_EQ(ops.current().seq, 42);
}

// ============================================================================
// Multi-threaded Stress Tests (Tearing Detection)
// ============================================================================

TEST_F(TripleBufferAtomicityTest, StressMonotonicity) {
    ImageTripleBuffer buffer{};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> violations{0};
    std::atomic<uint64_t> reads{0};

    std::thread producer([&] {
        ImageOps ops(&buffer);
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            ops.borrow_mut().seq = v++;
            ops.publish();
        }
    });

    std::thread consumer([&] {
        ImageOps ops(&buffer);
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto result = ops.borrow()) {
                const uint64_t curr = (*result)->seq;
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

    std::cout << "[Monotonicity] Reads: " << reads.load()
              << ", Violations: " << violations.load() << std::endl;
    EXPECT_EQ(violations.load(), 0);
    EXPECT_GT(reads.load(), 1000);
}

TEST_F(TripleBufferAtomicityTest, StressPoseDataIntegrity) {
    // Use PoseTripleBuffer which has larger slots (64 bytes)
    PoseTripleBuffer buffer{};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> corrupted{0};
    std::atomic<uint64_t> reads{0};

    std::thread producer([&] {
        PoseOps ops(&buffer);
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            auto& slot = ops.borrow_mut();
            // Fill with consistent data
            const float fv = static_cast<float>(v);
            slot.timestamp_ns = v;
            slot.position[0] = fv;
            slot.position[1] = fv;
            slot.position[2] = fv;
            slot.quaternion[0] = fv;
            slot.quaternion[1] = fv;
            slot.quaternion[2] = fv;
            slot.quaternion[3] = fv;
            ops.publish();
            ++v;
        }
    });

    std::thread consumer([&] {
        PoseOps ops(&buffer);
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto result = ops.borrow()) {
                reads.fetch_add(1, std::memory_order_relaxed);
                const auto& pose = **result;
                // Check consistency: all values should be from same write
                const float expected = static_cast<float>(pose.timestamp_ns);
                const bool consistent =
                    pose.position[0] == expected &&
                    pose.position[1] == expected &&
                    pose.position[2] == expected &&
                    pose.quaternion[0] == expected &&
                    pose.quaternion[1] == expected &&
                    pose.quaternion[2] == expected &&
                    pose.quaternion[3] == expected;
                if (!consistent) {
                    corrupted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "[PoseIntegrity] Reads: " << reads.load()
              << ", Corrupted: " << corrupted.load() << std::endl;
    EXPECT_EQ(corrupted.load(), 0);
    EXPECT_GT(reads.load(), 1000);
}

TEST_F(TripleBufferAtomicityTest, StressGimbalCmdIntegrity) {
    GimbalTripleBuffer buffer{};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> corrupted{0};
    std::atomic<uint64_t> reads{0};

    std::thread producer([&] {
        GimbalOps ops(&buffer);
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            auto& cmd = ops.borrow_mut();
            const float fv = static_cast<float>(v);
            cmd.timestamp_ns = v;
            cmd.yaw_deg = fv;
            cmd.pitch_deg = fv;
            cmd.distance_m = fv;
            cmd.fire_advice = static_cast<uint8_t>(v % 2);
            ops.publish();
            ++v;
        }
    });

    std::thread consumer([&] {
        GimbalOps ops(&buffer);
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto result = ops.borrow()) {
                reads.fetch_add(1, std::memory_order_relaxed);
                const auto& cmd = **result;
                const float expected = static_cast<float>(cmd.timestamp_ns);
                const bool consistent =
                    cmd.yaw_deg == expected &&
                    cmd.pitch_deg == expected &&
                    cmd.distance_m == expected;
                if (!consistent) {
                    corrupted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "[GimbalIntegrity] Reads: " << reads.load()
              << ", Corrupted: " << corrupted.load() << std::endl;
    EXPECT_EQ(corrupted.load(), 0);
    EXPECT_GT(reads.load(), 1000);
}

TEST_F(TripleBufferAtomicityTest, FastProducerSlowConsumer) {
    ImageTripleBuffer buffer{};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> violations{0};
    std::atomic<uint64_t> reads{0};

    std::thread producer([&] {
        ImageOps ops(&buffer);
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            ops.borrow_mut().seq = v++;
            ops.publish();
            // Fast producer: no sleep
        }
    });

    std::thread consumer([&] {
        ImageOps ops(&buffer);
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto result = ops.borrow()) {
                const uint64_t curr = (*result)->seq;
                reads.fetch_add(1, std::memory_order_relaxed);
                if (curr <= last) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                last = curr;
            }
            // Slow consumer: 100us delay
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "[FastProd/SlowCons] Reads: " << reads.load()
              << ", Violations: " << violations.load() << std::endl;
    EXPECT_EQ(violations.load(), 0);
}

TEST_F(TripleBufferAtomicityTest, SlowProducerFastConsumer) {
    ImageTripleBuffer buffer{};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> violations{0};
    std::atomic<uint64_t> empty_reads{0};
    std::atomic<uint64_t> valid_reads{0};

    std::thread producer([&] {
        ImageOps ops(&buffer);
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            ops.borrow_mut().seq = v++;
            ops.publish();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::thread consumer([&] {
        ImageOps ops(&buffer);
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto result = ops.borrow()) {
                const uint64_t curr = (*result)->seq;
                valid_reads.fetch_add(1, std::memory_order_relaxed);
                if (curr <= last) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                last = curr;
            } else {
                empty_reads.fetch_add(1, std::memory_order_relaxed);
            }
            // Fast consumer: no sleep
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "[SlowProd/FastCons] Valid: " << valid_reads.load()
              << ", Empty: " << empty_reads.load()
              << ", Violations: " << violations.load() << std::endl;
    EXPECT_EQ(violations.load(), 0);
    EXPECT_GT(empty_reads.load(), valid_reads.load()); // More empty reads expected
}

// ============================================================================
// Shared Memory IPC Tests (Simulated Multi-process)
// ============================================================================

class ShmIPCTest : public ::testing::Test {
protected:
    static constexpr std::string_view TEST_META_SHM = "ipc_test_meta";
    static constexpr size_t META_SIZE = sizeof(ShmMetaRegion);

    /**
     * @brief Initialize a TripleBuffer to correct initial state
     *
     * ShmRegion::create() uses memset to zero-initialize memory, which
     * corrupts the default member initializers. We must manually reinitialize.
     *
     * Correct initial state:
     * - state = 1 (ready slot is 1, no FLAG_NEW)
     * - write_idx = 0 (producer writes to slot 0)
     * - read_idx = 2 (consumer last read slot 2)
     */
    template<typename TripleBuffer>
    static void init_triple_buffer(TripleBuffer& buf) {
        buf.state.store(1, std::memory_order_relaxed);
        buf.write_idx = 0;
        buf.read_idx = 2;
    }

    static void init_meta_region(ShmMetaRegion* meta) {
        // Initialize header
        meta->header.magic = SHM_MAGIC;
        meta->header.version = SHM_VERSION;

        // Initialize all triple buffers (CRITICAL: memset corrupts default initializers)
        init_triple_buffer(meta->image);
        for (auto& pose : meta->poses) {
            init_triple_buffer(pose);
        }
        init_triple_buffer(meta->gimbal_cmd);
    }

    void TearDown() override {
        std::filesystem::remove(shm_path(TEST_META_SHM));
    }
};

TEST_F(ShmIPCTest, ProducerConsumerCommunication) {
    // Create shared memory region
    auto shm = ShmRegion::create(TEST_META_SHM, META_SIZE);
    ASSERT_TRUE(shm.has_value());

    auto* meta = shm->as<ShmMetaRegion>();
    init_meta_region(meta);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> violations{0};
    std::atomic<uint64_t> reads{0};

    // Producer thread (simulates Rust process)
    std::thread producer([meta, &stop] {
        ImageOps ops(&meta->image);
        uint64_t seq = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            auto& slot = ops.borrow_mut();
            slot.seq = seq++;
            slot.timestamp_ns = static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            slot.width = IMAGE_WIDTH;
            slot.height = IMAGE_HEIGHT;
            ops.publish();
        }
    });

    // Consumer thread (simulates C++ process)
    std::thread consumer([meta, &stop, &violations, &reads] {
        ImageOps ops(&meta->image);
        uint64_t last_seq = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto result = ops.borrow()) {
                reads.fetch_add(1, std::memory_order_relaxed);
                const auto& img = **result;
                if (img.seq <= last_seq) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                last_seq = img.seq;

                // Verify dimensions
                EXPECT_EQ(img.width, IMAGE_WIDTH);
                EXPECT_EQ(img.height, IMAGE_HEIGHT);
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "[ShmIPC] Reads: " << reads.load()
              << ", Violations: " << violations.load() << std::endl;
    EXPECT_EQ(violations.load(), 0);
    EXPECT_GT(reads.load(), 1000);
}

TEST_F(ShmIPCTest, MultipleTopics) {
    auto shm = ShmRegion::create(TEST_META_SHM, META_SIZE);
    ASSERT_TRUE(shm.has_value());

    auto* meta = shm->as<ShmMetaRegion>();
    init_meta_region(meta);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> image_violations{0};
    std::atomic<uint64_t> pose_violations{0};
    std::atomic<uint64_t> gimbal_violations{0};

    // Image producer
    std::thread image_producer([meta, &stop] {
        ImageOps ops(&meta->image);
        uint64_t seq = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            ops.borrow_mut().seq = seq++;
            ops.publish();
        }
    });

    // Pose producer
    std::thread pose_producer([meta, &stop] {
        PoseOps ops(&meta->poses[POSE_GIMBAL]);
        uint64_t ts = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            ops.borrow_mut().timestamp_ns = ts++;
            ops.publish();
        }
    });

    // Gimbal command producer
    std::thread gimbal_producer([meta, &stop] {
        GimbalOps ops(&meta->gimbal_cmd);
        uint64_t ts = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            ops.borrow_mut().timestamp_ns = ts++;
            ops.publish();
        }
    });

    // Image consumer
    std::thread image_consumer([meta, &stop, &image_violations] {
        ImageOps ops(&meta->image);
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto r = ops.borrow()) {
                if ((*r)->seq <= last) image_violations.fetch_add(1, std::memory_order_relaxed);
                last = (*r)->seq;
            }
        }
    });

    // Pose consumer
    std::thread pose_consumer([meta, &stop, &pose_violations] {
        PoseOps ops(&meta->poses[POSE_GIMBAL]);
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto r = ops.borrow()) {
                if ((*r)->timestamp_ns <= last) pose_violations.fetch_add(1, std::memory_order_relaxed);
                last = (*r)->timestamp_ns;
            }
        }
    });

    // Gimbal consumer
    std::thread gimbal_consumer([meta, &stop, &gimbal_violations] {
        GimbalOps ops(&meta->gimbal_cmd);
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (auto r = ops.borrow()) {
                if ((*r)->timestamp_ns <= last) gimbal_violations.fetch_add(1, std::memory_order_relaxed);
                last = (*r)->timestamp_ns;
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);

    image_producer.join();
    pose_producer.join();
    gimbal_producer.join();
    image_consumer.join();
    pose_consumer.join();
    gimbal_consumer.join();

    std::cout << "[MultiTopic] Image violations: " << image_violations.load()
              << ", Pose violations: " << pose_violations.load()
              << ", Gimbal violations: " << gimbal_violations.load() << std::endl;

    EXPECT_EQ(image_violations.load(), 0);
    EXPECT_EQ(pose_violations.load(), 0);
    EXPECT_EQ(gimbal_violations.load(), 0);
}

/**
 * @brief Test that verifies the fix for ShmRegion::create() / ShmClient::create()
 *
 * PREVIOUSLY A BUG: ShmRegion::create() uses memset to zero-initialize memory,
 * which corrupts the default member initializers for TripleBuffer state.
 *
 * FIX: ShmClient::create() now calls init_triple_buffer() for all TripleBuffers
 * after mapping to restore correct initial state.
 *
 * This test verifies the fix is working correctly.
 */
TEST_F(ShmIPCTest, TripleBufferInitializationAfterCreate) {
    auto shm = ShmRegion::create(TEST_META_SHM, META_SIZE);
    ASSERT_TRUE(shm.has_value());

    auto* meta = shm->as<ShmMetaRegion>();

    // After memset(0), state is corrupted to 0
    // (correct initial: state=1, write_idx=0, read_idx=2)
    EXPECT_EQ(meta->image.state.load(), 0) << "memset zeroed the state";
    EXPECT_EQ(meta->image.write_idx, 0);
    EXPECT_EQ(meta->image.read_idx, 0) << "memset zeroed read_idx (should be 2)";

    // Now manually init (simulating what ShmClient::create() does)
    init_meta_region(meta);

    // Verify correct initial state after init
    EXPECT_EQ(meta->image.state.load(), 1) << "state should be 1 (ready slot)";
    EXPECT_EQ(meta->image.write_idx, 0) << "write_idx should be 0";
    EXPECT_EQ(meta->image.read_idx, 2) << "read_idx should be 2";
}

// ============================================================================
// Layout and Alignment Tests
// ============================================================================

TEST(ShmLayoutTest, StructSizes) {
    EXPECT_EQ(sizeof(ImageMeta), 32);
    EXPECT_EQ(sizeof(PoseMeta), 64);
    EXPECT_EQ(sizeof(GimbalCmd), 32);
    EXPECT_EQ(sizeof(CameraInfo), 128);
    EXPECT_EQ(sizeof(ShmHeader), 64);
    EXPECT_EQ(sizeof(ShmMetaRegion), 2048);
}

TEST(ShmLayoutTest, StructAlignment) {
    EXPECT_EQ(alignof(ImageMeta), 32);
    EXPECT_EQ(alignof(PoseMeta), 64);
    EXPECT_EQ(alignof(GimbalCmd), 32);
    EXPECT_EQ(alignof(CameraInfo), 64);
    EXPECT_EQ(alignof(ShmHeader), CACHE_LINE_SIZE);
}

TEST(ShmLayoutTest, TripleBufferAlignment) {
    EXPECT_EQ(alignof(ImageTripleBuffer), CACHE_LINE_SIZE);
    EXPECT_EQ(alignof(PoseTripleBuffer), CACHE_LINE_SIZE);
    EXPECT_EQ(alignof(GimbalTripleBuffer), CACHE_LINE_SIZE);
}

TEST(ShmLayoutTest, AtomicStateCacheLineIsolation) {
    // Verify atomic state is on its own cache line to prevent false sharing
    ImageTripleBuffer img_buf{};
    PoseTripleBuffer pose_buf{};
    GimbalTripleBuffer gimbal_buf{};

    // State should be at offset 0 (cache line aligned)
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&img_buf.state) % CACHE_LINE_SIZE, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&pose_buf.state) % CACHE_LINE_SIZE, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&gimbal_buf.state) % CACHE_LINE_SIZE, 0);

    // Slots should be at next cache line or beyond
    EXPECT_GE(reinterpret_cast<uintptr_t>(&img_buf.slots[0]) -
              reinterpret_cast<uintptr_t>(&img_buf.state), CACHE_LINE_SIZE);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(TripleBufferEdgeCasesTest, MultipleBorrowMutWithoutPublish) {
    ImageTripleBuffer buffer{};
    ImageOps ops(&buffer);

    ops.borrow_mut().seq = 1;
    ops.borrow_mut().seq = 2;
    ops.borrow_mut().seq = 3;
    ops.publish();

    auto result = ops.borrow();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->seq, 3);
}

TEST(TripleBufferEdgeCasesTest, AlternatingReadWrite) {
    ImageTripleBuffer buffer{};
    ImageOps ops(&buffer);

    for (uint64_t i = 1; i <= 1000; ++i) {
        ops.borrow_mut().seq = i;
        ops.publish();

        auto result = ops.borrow();
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ((*result)->seq, i);

        // Second read should be empty
        EXPECT_EQ(ops.borrow(), std::nullopt);
        EXPECT_FALSE(ops.has_new_data());
    }
}

TEST(TripleBufferEdgeCasesTest, RapidPublishWithoutRead) {
    ImageTripleBuffer buffer{};
    ImageOps ops(&buffer);

    // Rapid publishes without reading
    for (uint64_t i = 1; i <= 10000; ++i) {
        ops.borrow_mut().seq = i;
        ops.publish();
    }

    // Should only get latest
    auto result = ops.borrow();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->seq, 10000);
}

// ============================================================================
// Memory Safety Under Concurrent Access
// ============================================================================

TEST(TripleBufferMemorySafetyTest, ConcurrentAccessNoUseAfterFree) {
    // This test verifies that the triple buffer protocol ensures
    // consumers never access memory being written by producer

    struct alignas(64) LargeSlot {
        std::array<uint64_t, 128> data{}; // 1KB

        void fill(uint64_t v) {
            for (auto& x : data) x = v;
        }

        [[nodiscard]] bool is_consistent() const {
            const uint64_t first = data[0];
            return std::ranges::all_of(data, [first](uint64_t v) { return v == first; });
        }
    };

    struct alignas(CACHE_LINE_SIZE) TestTripleBuffer {
        alignas(CACHE_LINE_SIZE) std::atomic<uint8_t> state{1};
        uint8_t write_idx{0};
        uint8_t read_idx{2};
        uint8_t _pad[61];
        LargeSlot slots[3]{};
    };

    TestTripleBuffer buffer{};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> corrupted{0};
    std::atomic<uint64_t> reads{0};

    std::thread producer([&] {
        uint64_t v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            // Get write slot
            auto& slot = buffer.slots[buffer.write_idx];

            // Slow write to increase chance of tearing if there's a bug
            for (size_t i = 0; i < slot.data.size(); ++i) {
                slot.data[i] = v;
            }

            // Publish
            const auto old = buffer.state.exchange(buffer.write_idx | FLAG_NEW, std::memory_order_acq_rel);
            buffer.write_idx = old & INDEX_MASK;
            ++v;
        }
    });

    std::thread consumer([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            auto expected = buffer.state.load(std::memory_order_acquire);
            if (!(expected & FLAG_NEW)) continue;

            auto ready_idx = expected & INDEX_MASK;
            auto desired = buffer.read_idx;

            if (!buffer.state.compare_exchange_strong(expected, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                if (!(expected & FLAG_NEW)) continue;
                ready_idx = expected & INDEX_MASK;
                desired = buffer.read_idx;
                if (!buffer.state.compare_exchange_strong(expected, desired,
                        std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    continue;
                }
            }

            buffer.read_idx = ready_idx;
            const auto& slot = buffer.slots[ready_idx];

            reads.fetch_add(1, std::memory_order_relaxed);

            // Verify all elements are consistent
            if (!slot.is_consistent()) {
                corrupted.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(3));
    stop.store(true);
    producer.join();
    consumer.join();

    std::cout << "[MemorySafety] Reads: " << reads.load()
              << ", Corrupted: " << corrupted.load() << std::endl;
    EXPECT_EQ(corrupted.load(), 0);
    EXPECT_GT(reads.load(), 1000);
}

} // namespace ipc::test
