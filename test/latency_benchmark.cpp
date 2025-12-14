#include <benchmark/benchmark.h>
#include "detector_state.hpp"
#include "tracker_soa.hpp"
#include "triple_buffer.hpp"

namespace spike::bench {

// ============================================================================
// Benchmark: TripleBuffer Round-trip
// ============================================================================

static void BM_TripleBufferPublish(benchmark::State& state) {
  spike::TripleBuffer<spike::ArmorSlots> buffer;

  for (auto _ : state) {
    auto* slot = buffer.begin_write();
    slot->valid_mask = 0x0F;
    buffer.publish_write();
    benchmark::DoNotOptimize(slot);
  }
}
BENCHMARK(BM_TripleBufferPublish);

static void BM_TripleBufferRead(benchmark::State& state) {
  spike::TripleBuffer<spike::ArmorSlots> buffer;

  // Pre-populate with data
  auto* slot = buffer.begin_write();
  slot->valid_mask = 0x0F;
  buffer.publish_write();

  for (auto _ : state) {
    auto [ptr, ver] = buffer.try_read();
    benchmark::DoNotOptimize(ptr);
    benchmark::DoNotOptimize(ver);
  }
}
BENCHMARK(BM_TripleBufferRead);

// ============================================================================
// Benchmark: State Machine Transitions
// ============================================================================

static void BM_StateTransitionIDLEtoDETECTING(benchmark::State& state) {
  for (auto _ : state) {
    spike::TrackerSoA tracker;
    benchmark::DoNotOptimize(tracker.try_start_detecting());
  }
}
BENCHMARK(BM_StateTransitionIDLEtoDETECTING);

static void BM_StateTransitionTRACKINGtoTEMP_LOST(benchmark::State& state) {
  spike::TrackerSoA tracker;
  tracker.state.store(spike::TrackerState::TRACKING, std::memory_order_relaxed);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tracker.try_enter_lost());
    // Reset for next iteration
    tracker.state.store(spike::TrackerState::TRACKING, std::memory_order_relaxed);
  }
}
BENCHMARK(BM_StateTransitionTRACKINGtoTEMP_LOST);

// ============================================================================
// Benchmark: ArmorSlots Iteration
// ============================================================================

static void BM_ArmorSlotsCountValid(benchmark::State& state) {
  spike::ArmorSlots slots;
  slots.valid_mask = 0x0F;  // 4 valid armors

  for (auto _ : state) {
    benchmark::DoNotOptimize(slots.count_valid());
  }
}
BENCHMARK(BM_ArmorSlotsCountValid);

// ============================================================================
// Benchmark: Data Access (Cache Locality)
// ============================================================================

static void BM_TrackerSoAMemoryAccess(benchmark::State& state) {
  spike::TrackerSoA tracker;
  tracker.pos[0] = 1.0f;
  tracker.vel[0] = 0.5f;

  for (auto _ : state) {
    // Simulate typical tracker predict operation
    tracker.pos[0] += tracker.vel[0] * 0.004f;  // 4ms timestep
    tracker.pos[1] += tracker.vel[1] * 0.004f;
    tracker.pos[2] += tracker.vel[2] * 0.004f;
    benchmark::DoNotOptimize(tracker);
  }
}
BENCHMARK(BM_TrackerSoAMemoryAccess);

}  // namespace spike::bench
