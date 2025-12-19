#include <benchmark/benchmark.h>
#include "detector_state.hpp"
#include "tracker_soa.hpp"
#include "triple_buffer.hpp"

namespace talos::bench {
    // ============================================================================
    // Benchmark: TripleBuffer Round-trip
    // ============================================================================

    static void BM_TripleBufferPublish(benchmark::State &state) {
        TripleBuffer<ArmorSlots> buffer;

        for ([[maybe_unused]] auto _: state) {
            auto &slot = buffer.borrow_mut();
            slot.valid_mask = 0x0F;
            buffer.publish();
            benchmark::DoNotOptimize(slot);
        }
    }

    BENCHMARK(BM_TripleBufferPublish);

    static void BM_TripleBufferRead(benchmark::State &state) {
        TripleBuffer<ArmorSlots> buffer;

        // Pre-populate with data
        auto &slot = buffer.borrow_mut();
        slot.valid_mask = 0x0F;
        buffer.publish();

        for ([[maybe_unused]] auto _: state) {
            auto ptr = buffer.borrow();
            benchmark::DoNotOptimize(ptr);
        }
    }

    BENCHMARK(BM_TripleBufferRead);

    // ============================================================================
    // Benchmark: State Machine Transitions
    // ============================================================================

    static void BM_StateTransitionIdleToDetecting(benchmark::State &state) {
        for ([[maybe_unused]] auto _: state) {
            TrackerSoA tracker;
            benchmark::DoNotOptimize(tracker.try_start_detecting());
        }
    }

    BENCHMARK(BM_StateTransitionIdleToDetecting);

    static void BM_StateTransitionTrackingtoTempLost(benchmark::State &state) {
        TrackerSoA tracker;
        tracker.state.store(TrackerState::TRACKING, std::memory_order_relaxed);

        for ([[maybe_unused]] auto _: state) {
            benchmark::DoNotOptimize(tracker.try_enter_lost());
            // Reset for next iteration
            tracker.state.store(TrackerState::TRACKING, std::memory_order_relaxed);
        }
    }

    BENCHMARK(BM_StateTransitionTrackingtoTempLost);

    // ============================================================================
    // Benchmark: ArmorSlots Iteration
    // ============================================================================

    static void BM_ArmorSlotsCountValid(benchmark::State &state) {
        ArmorSlots slots;
        slots.valid_mask = 0x0F; // 4 valid armors

        for ([[maybe_unused]] auto _: state) {
            benchmark::DoNotOptimize(slots.count_valid());
        }
    }

    BENCHMARK(BM_ArmorSlotsCountValid);

    // ============================================================================
    // Benchmark: Data Access (Cache Locality)
    // ============================================================================

    static void BM_TrackerSoAMemoryAccess(benchmark::State &state) {
        TrackerSoA tracker;
        tracker.pos[0] = 1.0f;
        tracker.vel[0] = 0.5f;

        for ([[maybe_unused]] auto _: state) {
            // Simulate typical tracker predict operation
            tracker.pos[0] += tracker.vel[0] * 0.004f; // 4ms timestep
            tracker.pos[1] += tracker.vel[1] * 0.004f;
            tracker.pos[2] += tracker.vel[2] * 0.004f;
            benchmark::DoNotOptimize(tracker);
        }
    }

    BENCHMARK(BM_TrackerSoAMemoryAccess);
} // namespace spike::bench
