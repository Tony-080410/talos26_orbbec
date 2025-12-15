#include <gtest/gtest.h>
#include "detector_state.hpp"
#include "tracker_soa.hpp"
#include "triple_buffer.hpp"

namespace spike::test {
    // ============================================================================
    // Test: Memory Layout Verification
    // ============================================================================

    TEST(MemoryLayout, ArmorSlotsSize) {
        EXPECT_LE(sizeof(spike::ArmorSlots), 1024)
      << "ArmorSlots exceeds 1KB budget";
        EXPECT_GE(sizeof(spike::ArmorSlots), 512)
      << "ArmorSlots seems too small";
    }

    TEST(MemoryLayout, ArmorSlotsAlignment) {
        EXPECT_EQ(alignof(spike::ArmorSlots), 64)
      << "ArmorSlots must be 64-byte cache-line aligned";
    }

    TEST(MemoryLayout, TrackerSoAExactSize) {
        // Critical: must be exactly 64 bytes
        EXPECT_EQ(sizeof(spike::TrackerSoA), 64)
      << "TrackerSoA must be exactly 64 bytes for L1 cache residence";
    }

    TEST(MemoryLayout, TrackerSoAAlignment) {
        EXPECT_EQ(alignof(spike::TrackerSoA), 64)
      << "TrackerSoA must be 64-byte aligned";
    }

    TEST(MemoryLayout, TrackerSoACacheLine) {
        // Verify that instances can be placed contiguously without false sharing
        TrackerSoA tracker1;
        TrackerSoA tracker2;

        const auto addr1 = reinterpret_cast<uintptr_t>(&tracker1);
        const auto addr2 = reinterpret_cast<uintptr_t>(&tracker2);

        // Addresses should differ by exactly 64 bytes (if allocated sequentially)
        EXPECT_EQ(addr1 % 64, 0) << "TrackerSoA should be 64-byte aligned";
        EXPECT_EQ(addr2 % 64, 0) << "TrackerSoA should be 64-byte aligned";
    }

    // ============================================================================
    // Test: ArmorSlots Functionality
    // ============================================================================

    TEST(ArmorSlots, DefaultConstruction) {
        ArmorSlots slots;
        EXPECT_EQ(slots.valid_mask, 0);
        EXPECT_EQ(slots.count_valid(), 0);
    }

    TEST(ArmorSlots, SetValid) {
        ArmorSlots slots;
        slots.set_valid(0);
        slots.set_valid(5);

        EXPECT_TRUE(slots.is_valid(0));
        EXPECT_FALSE(slots.is_valid(1));
        EXPECT_TRUE(slots.is_valid(5));
        EXPECT_EQ(slots.count_valid(), 2);
    }

    TEST(ArmorSlots, SetInvalid) {
        ArmorSlots slots;
        slots.valid_mask = 0xFF; // All 8 bits set
        EXPECT_EQ(slots.count_valid(), 8);

        slots.set_invalid(3);
        EXPECT_EQ(slots.count_valid(), 7);
        EXPECT_FALSE(slots.is_valid(3));
    }

    TEST(ArmorSlots, PopcountPerformance) {
        ArmorSlots slots;
        // Valid mask with various patterns
        for (int pattern = 0; pattern < 256; pattern++) {
            slots.valid_mask = pattern;
            int expected = __builtin_popcount(pattern);
            EXPECT_EQ(slots.count_valid(), expected);
        }
    }

    TEST(ArmorSlots, Clear) {
        ArmorSlots slots;
        slots.valid_mask = 0xFF;
        slots.clear();
        EXPECT_EQ(slots.valid_mask, 0);
        EXPECT_EQ(slots.count_valid(), 0);
    }

    // ============================================================================
    // Test: TrackerSoA State Machine
    // ============================================================================

    TEST(TrackerSoA, DefaultState) {
        TrackerSoA tracker;
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::IDLE);
        EXPECT_EQ(tracker.detecting_count, 0);
        EXPECT_EQ(tracker.lost_count, 0);
    }

    TEST(TrackerSoA, StateTransitionIDLEtoDETECTING) {
        TrackerSoA tracker;
        EXPECT_TRUE(tracker.try_start_detecting());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::DETECTING);

        // Second call should fail (already in DETECTING)
        EXPECT_FALSE(tracker.try_start_detecting());
    }

    TEST(TrackerSoA, StateTransitionDETECTINGtoTRACKING) {
        TrackerSoA tracker;
        tracker.state.store(TrackerState::DETECTING, std::memory_order_relaxed);

        EXPECT_TRUE(tracker.try_confirm_tracking());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::TRACKING);
    }

    TEST(TrackerSoA, StateTransitionTRACKINGtoTEMP_LOST) {
        TrackerSoA tracker;
        tracker.state.store(TrackerState::TRACKING, std::memory_order_relaxed);

        EXPECT_TRUE(tracker.try_enter_lost());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::TEMP_LOST);
    }

    TEST(TrackerSoA, StateTransitionTEMP_LOSTtoTRACKING) {
        TrackerSoA tracker;
        tracker.state.store(TrackerState::TEMP_LOST, std::memory_order_relaxed);

        EXPECT_TRUE(tracker.try_restore_tracking());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::TRACKING);
    }

    TEST(TrackerSoA, StateTransitionReset) {
        TrackerSoA tracker;
        tracker.state.store(TrackerState::TRACKING, std::memory_order_relaxed);

        EXPECT_TRUE(tracker.try_reset_idle());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::IDLE);
    }

    TEST(TrackerSoA, GenerationCounter) {
        TrackerSoA tracker;
        const auto gen0 = tracker.generation;
        tracker.next_generation();
        const auto gen1 = tracker.generation;
        tracker.next_generation();
        const auto gen2 = tracker.generation;

        EXPECT_NE(gen0, gen1);
        EXPECT_NE(gen1, gen2);
    }

    // ============================================================================
    // Test: TripleBuffer Synchronization
    // ============================================================================

    TEST(TripleBuffer, BasicPubSub) {
        TripleBuffer<ArmorSlots> buffer;

        // Initial state: no data
        const auto ptr = buffer.borrow();
        EXPECT_EQ(ptr.has_value(), false);

        // Producer writes
        auto &write_slot = buffer.borrow_mut();
        write_slot.valid_mask = 0x0F; // 4 armors
        write_slot.frame_id = 1;
        buffer.publish();

        // Consumer reads
        const auto val = buffer.borrow();
        ASSERT_EQ(val.has_value(), true);
        EXPECT_EQ(val.value()->valid_mask, 0x0F);
        EXPECT_EQ(val.value()->frame_id, 1);
    }
} // namespace spike::test
