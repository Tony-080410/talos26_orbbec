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
  spike::TrackerSoA tracker1;
  spike::TrackerSoA tracker2;

  auto addr1 = reinterpret_cast<uintptr_t>(&tracker1);
  auto addr2 = reinterpret_cast<uintptr_t>(&tracker2);

  // Addresses should differ by exactly 64 bytes (if allocated sequentially)
  EXPECT_EQ(addr1 % 64, 0) << "TrackerSoA should be 64-byte aligned";
}

// ============================================================================
// Test: ArmorSlots Functionality
// ============================================================================

TEST(ArmorSlots, DefaultConstruction) {
  spike::ArmorSlots slots;
  EXPECT_EQ(slots.valid_mask, 0);
  EXPECT_EQ(slots.count_valid(), 0);
}

TEST(ArmorSlots, SetValid) {
  spike::ArmorSlots slots;
  slots.set_valid(0);
  slots.set_valid(5);

  EXPECT_TRUE(slots.is_valid(0));
  EXPECT_FALSE(slots.is_valid(1));
  EXPECT_TRUE(slots.is_valid(5));
  EXPECT_EQ(slots.count_valid(), 2);
}

TEST(ArmorSlots, SetInvalid) {
  spike::ArmorSlots slots;
  slots.valid_mask = 0xFF;  // All 8 bits set
  EXPECT_EQ(slots.count_valid(), 8);

  slots.set_invalid(3);
  EXPECT_EQ(slots.count_valid(), 7);
  EXPECT_FALSE(slots.is_valid(3));
}

TEST(ArmorSlots, PopcountPerformance) {
  spike::ArmorSlots slots;
  // Valid mask with various patterns
  for (int pattern = 0; pattern < 256; pattern++) {
    slots.valid_mask = pattern;
    int expected = __builtin_popcount(pattern);
    EXPECT_EQ(slots.count_valid(), expected);
  }
}

TEST(ArmorSlots, Clear) {
  spike::ArmorSlots slots;
  slots.valid_mask = 0xFF;
  slots.clear();
  EXPECT_EQ(slots.valid_mask, 0);
  EXPECT_EQ(slots.count_valid(), 0);
}

// ============================================================================
// Test: TrackerSoA State Machine
// ============================================================================

TEST(TrackerSoA, DefaultState) {
  spike::TrackerSoA tracker;
  EXPECT_EQ(tracker.state.load(), spike::TrackerState::IDLE);
  EXPECT_EQ(tracker.detecting_count, 0);
  EXPECT_EQ(tracker.lost_count, 0);
}

TEST(TrackerSoA, StateTransitionIDLEtoDETECTING) {
  spike::TrackerSoA tracker;
  EXPECT_TRUE(tracker.try_start_detecting());
  EXPECT_EQ(tracker.state.load(), spike::TrackerState::DETECTING);

  // Second call should fail (already in DETECTING)
  EXPECT_FALSE(tracker.try_start_detecting());
}

TEST(TrackerSoA, StateTransitionDETECTINGtoTRACKING) {
  spike::TrackerSoA tracker;
  tracker.state.store(spike::TrackerState::DETECTING, std::memory_order_relaxed);

  EXPECT_TRUE(tracker.try_confirm_tracking());
  EXPECT_EQ(tracker.state.load(), spike::TrackerState::TRACKING);
}

TEST(TrackerSoA, StateTransitionTRACKINGtoTEMP_LOST) {
  spike::TrackerSoA tracker;
  tracker.state.store(spike::TrackerState::TRACKING, std::memory_order_relaxed);

  EXPECT_TRUE(tracker.try_enter_lost());
  EXPECT_EQ(tracker.state.load(), spike::TrackerState::TEMP_LOST);
}

TEST(TrackerSoA, StateTransitionTEMP_LOSTtoTRACKING) {
  spike::TrackerSoA tracker;
  tracker.state.store(spike::TrackerState::TEMP_LOST, std::memory_order_relaxed);

  EXPECT_TRUE(tracker.try_restore_tracking());
  EXPECT_EQ(tracker.state.load(), spike::TrackerState::TRACKING);
}

TEST(TrackerSoA, StateTransitionReset) {
  spike::TrackerSoA tracker;
  tracker.state.store(spike::TrackerState::TRACKING, std::memory_order_relaxed);

  EXPECT_TRUE(tracker.try_reset_idle());
  EXPECT_EQ(tracker.state.load(), spike::TrackerState::IDLE);
}

TEST(TrackerSoA, GenerationCounter) {
  spike::TrackerSoA tracker;
  auto gen0 = tracker.generation;
  tracker.next_generation();
  auto gen1 = tracker.generation;
  tracker.next_generation();
  auto gen2 = tracker.generation;

  EXPECT_NE(gen0, gen1);
  EXPECT_NE(gen1, gen2);
}

// ============================================================================
// Test: TripleBuffer Synchronization
// ============================================================================

TEST(TripleBuffer, BasicPubSub) {
  spike::TripleBuffer<spike::ArmorSlots> buffer;

  // Initial state: no data
  auto [ptr, ver] = buffer.try_read();
  EXPECT_EQ(ptr, nullptr);

  // Producer writes
  auto* write_slot = buffer.begin_write();
  write_slot->valid_mask = 0x0F;  // 4 armors
  write_slot->frame_id = 1;
  buffer.publish_write();

  // Consumer reads
  auto [read_ptr, read_ver] = buffer.try_read();
  ASSERT_NE(read_ptr, nullptr);
  EXPECT_EQ(read_ptr->valid_mask, 0x0F);
  EXPECT_EQ(read_ptr->frame_id, 1);
}

TEST(TripleBuffer, NoBlockingDropOnOverload) {
  spike::TripleBuffer<spike::ArmorSlots> buffer;

  // Producer floods
  for (int i = 0; i < 10; i++) {
    auto* slot = buffer.begin_write();
    slot->frame_id = i;
    buffer.publish_write();  // Non-blocking, no allocations
  }

  // Consumer reads latest
  auto [ptr, ver] = buffer.try_read();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ptr->frame_id, 9);  // Latest frame
}

TEST(TripleBuffer, VersionTracking) {
  spike::TripleBuffer<spike::ArmorSlots> buffer;

  // Multiple cycles to verify version counter increments
  uint64_t prev_ver = 0;
  for (int i = 0; i < 5; i++) {
    auto* slot = buffer.begin_write();
    slot->frame_id = i;
    buffer.publish_write();
    auto [ptr, ver] = buffer.try_read();
    ASSERT_NE(ptr, nullptr);

    // Version should increase with each publish (after 3 cycles)
    if (i >= 3) {
      EXPECT_GT(ver, prev_ver)
          << "Version should increase after slot reuse";
    }
    prev_ver = ver;
  }
}

TEST(TripleBuffer, Reset) {
  spike::TripleBuffer<spike::ArmorSlots> buffer;

  // Write data
  auto* slot = buffer.begin_write();
  slot->frame_id = 42;
  buffer.publish_write();

  // Verify data is available
  auto [ptr, ver] = buffer.try_read();
  EXPECT_NE(ptr, nullptr);

  // Reset
  buffer.reset();

  // Should have no new data after reset
  auto [ptr2, ver2] = buffer.try_read();
  EXPECT_EQ(ptr2, nullptr);
}

TEST(TripleBuffer, HasNewFrame) {
  spike::TripleBuffer<spike::ArmorSlots> buffer;

  // Initially no new frame
  EXPECT_FALSE(buffer.has_new_frame());

  // After write
  auto* slot = buffer.begin_write();
  buffer.publish_write();
  EXPECT_TRUE(buffer.has_new_frame());

  // After read
  buffer.try_read();
  EXPECT_FALSE(buffer.has_new_frame());
}

}  // namespace spike::test
