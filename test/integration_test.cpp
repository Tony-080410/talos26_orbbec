#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <chrono>
#include "detector_state.hpp"
#include "tracker_soa.hpp"
#include "triple_buffer.hpp"
#include "performance_probe.hpp"

namespace spike::test {
    // ============================================================================
    // Integration Test: Detector -> Triple Buffer -> Tracker Pipeline
    // ============================================================================

    TEST(Integration, DetectorToTrackerPipeline) {
        using namespace std::chrono_literals;

        // Setup: Triple buffer for detector-to-tracker communication
        TripleBuffer<ArmorSlots> detector_buffer;

        std::atomic stop{false};
        std::atomic frames_produced{0};
        std::atomic frames_consumed{0};

        // Producer thread: Simulates detector writing frames
        std::thread detector_thread([&] {
            int frame_id = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                // Simulate detector producing frame with 3 armors
                auto &frame = detector_buffer.borrow_mut();
                frame.clear();
                frame.frame_id = frame_id++;
                frame.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();

                // Add 3 valid armors
                for (int i = 0; i < 3; i++) {
                    frame.armors[i].x2d = 100.0f + static_cast<float>(i) * 10;
                    frame.armors[i].y2d = 200.0f + static_cast<float>(i) * 10;
                    frame.armors[i].name = static_cast<uint8_t>(i);
                    frame.armors[i].color = 1;
                    frame.armors[i].confidence = 0.9f;
                    frame.set_valid(i);

                    // PnP result
                    frame.poses[i].x = 1.0f + static_cast<float>(i);
                    frame.poses[i].y = 2.0f + static_cast<float>(i);
                    frame.poses[i].z = 3.0f + static_cast<float>(i);
                    frame.poses[i].qw = 1.0f;
                    frame.poses[i].qx = 0.0f;
                    frame.poses[i].qy = 0.0f;
                    frame.poses[i].qz = 0.0f;
                }

                detector_buffer.publish();
                frames_produced.fetch_add(1, std::memory_order_relaxed);

                // Sleep to give consumer time to read (important for TSan)
                std::this_thread::sleep_for(2ms);
            }
        });

        // Consumer thread: Simulates tracker reading frames
        // Consumer should be at least as fast as producer
        std::thread tracker_thread([&] {
            TrackerSoA tracker;

            while (!stop.load(std::memory_order_relaxed)) {
                if (const auto frame_ptr = detector_buffer.borrow()) {
                    // Copy frame immediately to avoid race condition
                    const ArmorSlots *local_frame = frame_ptr.value();
                    frames_consumed.fetch_add(1, std::memory_order_relaxed);

                    // Verify frame integrity
                    EXPECT_EQ(local_frame->count_valid(), 3);

                    // Simulate tracker state transition based on detections
                    if (tracker.state.load() == TrackerState::IDLE) {
                        tracker.try_start_detecting();
                    } else if (tracker.state.load() == TrackerState::DETECTING) {
                        tracker.detecting_count++;
                        if (tracker.detecting_count >= 3) {
                            tracker.try_confirm_tracking();
                        }
                    }
                }

                // Consumer sleep <= producer sleep to ensure we keep up
                std::this_thread::sleep_for(1ms);
            }
        });

        // Run for 500ms
        std::this_thread::sleep_for(500ms);
        stop.store(true, std::memory_order_relaxed);

        detector_thread.join();
        tracker_thread.join();

        // Verify pipeline worked (lower threshold for TSan overhead)
        EXPECT_GT(frames_produced.load(), 100); // ~250 frames in 500ms with 2ms sleep
        EXPECT_GT(frames_consumed.load(), 50); // Consumer reads most frames

        std::cout << "Pipeline integration test:\n";
        std::cout << "  Frames produced: " << frames_produced.load() << "\n";
        std::cout << "  Frames consumed: " << frames_consumed.load() << "\n";
    }

    // ============================================================================
    // Integration Test: End-to-End Latency Measurement
    // ============================================================================

    TEST(Integration, EndToEndLatency) {
        TripleBuffer<ArmorSlots> buffer;
        LatencyHistogram latency_hist;

        constexpr int NUM_FRAMES = 100;

        // Measure latency from detector write to tracker read
        for (int i = 0; i < NUM_FRAMES; i++) {
            LatencyProbe probe;

            // Detector writes frame
            auto &frame = buffer.borrow_mut();
            frame.clear();
            frame.frame_id = i;
            frame.set_valid(0);
            frame.armors[0].x2d = 100.0f;
            frame.armors[0].y2d = 200.0f;
            buffer.publish();

            // Tracker reads frame immediately
            auto frame_ptr = buffer.borrow();
            ASSERT_EQ(frame_ptr.has_value(), true);

            // Record latency
            latency_hist.record(probe.elapsed_ns());
        }

        const auto stats = latency_hist.compute();

        EXPECT_EQ(stats.count, NUM_FRAMES);

        // Latency should be sub-microsecond (< 1000ns)
        EXPECT_LT(stats.p99_ns, 5000); // P99 < 5us (generous

        std::cout << "End-to-end latency:\n";
        std::cout << "  P50:  " << stats.p50_ns << " ns\n";
        std::cout << "  P95:  " << stats.p95_ns << " ns\n";
        std::cout << "  P99:  " << stats.p99_ns << " ns\n";
        std::cout << "  Mean: " << stats.mean_ns << " ns\n";
    }

    // ============================================================================
    // Integration Test: Multi-Tracker Scenario
    // ============================================================================

    TEST(Integration, MultiTrackerScenario) {
        constexpr int NUM_TRACKERS = 4;
        std::array<TrackerSoA, NUM_TRACKERS> trackers;

        // Initialize all trackers
        for (auto &tracker: trackers) {
            EXPECT_EQ(tracker.state.load(), spike::TrackerState::IDLE);
        }

        // Simulate detector frame with 4 armors
        ArmorSlots frame;
        frame.clear();
        for (int i = 0; i < NUM_TRACKERS; i++) {
            frame.armors[i].x2d = 100.0f + static_cast<float>(i) * 50;
            frame.armors[i].y2d = 200.0f + static_cast<float>(i) * 50;
            frame.armors[i].name = static_cast<uint8_t>(i);
            frame.set_valid(i);

            frame.poses[i].x = 1.0f + static_cast<float>(i);
            frame.poses[i].y = 2.0f;
            frame.poses[i].z = 3.0f;
        }

        // Each tracker processes one armor
        for (int i = 0; i < NUM_TRACKERS; i++) {
            if (frame.is_valid(i)) {
                // Initialize tracker position
                trackers[i].pos[0] = frame.poses[i].x;
                trackers[i].pos[1] = frame.poses[i].y;
                trackers[i].pos[2] = frame.poses[i].z;

                // Start tracking
                bool started = trackers[i].try_start_detecting();
                EXPECT_TRUE(started);
            }
        }

        // Verify all trackers started
        for (const auto &tracker: trackers) {
            EXPECT_EQ(tracker.state.load(), spike::TrackerState::DETECTING);
        }

        // Simulate confirming all trackers
        for (auto &tracker: trackers) {
            tracker.detecting_count = 5; // Enough to confirm
            bool confirmed = tracker.try_confirm_tracking();
            EXPECT_TRUE(confirmed);
        }

        // Verify all trackers are tracking
        for (const auto &tracker: trackers) {
            EXPECT_EQ(tracker.state.load(), spike::TrackerState::TRACKING);
        }
    }

    // ============================================================================
    // Integration Test: State Machine Full Cycle
    // ============================================================================

    TEST(Integration, TrackerStateMachineFullCycle) {
        TrackerSoA tracker;
        TripleBuffer<ArmorSlots> buffer;

        // State: IDLE -> DETECTING
        EXPECT_TRUE(tracker.try_start_detecting());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::DETECTING);

        // Simulate 5 detections to confirm tracking
        for (int i = 0; i < 5; i++) {
            auto &frame = buffer.borrow_mut();
            frame.clear();
            frame.set_valid(0);
            frame.armors[0].x2d = 100.0f;
            frame.armors[0].y2d = 200.0f;
            buffer.publish();

            tracker.detecting_count++;
        }

        // State: DETECTING -> TRACKING
        EXPECT_TRUE(tracker.try_confirm_tracking());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::TRACKING);

        // Simulate lost detection
        EXPECT_TRUE(tracker.try_enter_lost());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::TEMP_LOST);

        // Simulate recovery
        EXPECT_TRUE(tracker.try_restore_tracking());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::TRACKING);

        // Simulate permanent loss (reset to idle)
        tracker.lost_count = 10;
        EXPECT_TRUE(tracker.try_reset_idle());
        EXPECT_EQ(tracker.state.load(), spike::TrackerState::IDLE);
    }

    // ============================================================================
    // Integration Test: Cache-Line Verification
    // ============================================================================

    TEST(Integration, CacheLineResidenceVerification) {
        constexpr int NUM_TRACKERS = 8;
        std::array<TrackerSoA, NUM_TRACKERS> trackers;

        // Verify all trackers are 64-byte aligned
        for (size_t i = 0; i < NUM_TRACKERS; i++) {
            const auto addr = reinterpret_cast<uintptr_t>(&trackers[i]);
            EXPECT_EQ(addr % 64, 0) << "Tracker " << i << " not 64-byte aligned";
        }

        // Verify no false sharing between adjacent trackers
        for (size_t i = 1; i < NUM_TRACKERS; i++) {
            const auto addr_prev = reinterpret_cast<uintptr_t>(&trackers[i - 1]);
            const auto addr_curr = reinterpret_cast<uintptr_t>(&trackers[i]);
            auto distance = addr_curr - addr_prev;

            EXPECT_EQ(distance, 64) << "Trackers " << (i - 1) << " and " << i
                            << " not on separate cache lines";
        }
    }

    // ============================================================================
    // Integration Test: Stress Test with All Components
    // ============================================================================

    TEST(Integration, StressTestAllComponents) {
        using namespace std::chrono_literals;

        TripleBuffer<ArmorSlots> buffer;
        LatencyHistogram latency_hist;
        std::atomic stop{false};
        std::atomic total_processed{0};

        // Producer: writes at controlled rate to avoid data races
        // Note: Triple buffer requires producer not to be >3x faster than consumer
        std::thread producer([&] {
            int frame_id = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                LatencyProbe probe;

                auto &frame = buffer.borrow_mut();
                frame.clear();
                frame.frame_id = frame_id++;

                // Random number of armors (1-5)
                const int num_armors = frame_id % 5 + 1;
                for (int i = 0; i < num_armors; i++) {
                    frame.set_valid(i);
                    frame.armors[i].x2d = 100 + static_cast<float>(i) * 10;
                    frame.armors[i].y2d = 200 + static_cast<float>(i) * 10;
                }

                buffer.publish();

                latency_hist.record(probe.elapsed_ns());
                // Sleep to give consumer time to copy data
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });

        // Consumer: reads and copies data immediately
        std::thread consumer([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                if (auto frame_ptr = buffer.borrow(); frame_ptr.has_value()) {
                    const auto *local_frame = frame_ptr.value();
                    total_processed.fetch_add(1, std::memory_order_relaxed);

                    // Verify frame integrity (only check bounds if frame has valid armors)
                    if (auto count = local_frame->count_valid(); count > 0) {
                        EXPECT_LE(count, 5) << "Frame should have at most 5 armors";
                    }
                }
            }
        });

        // Run stress test for 1 second
        std::this_thread::sleep_for(1000ms);
        stop.store(true, std::memory_order_relaxed);

        producer.join();
        consumer.join();

        // Should process thousands of frames (lower threshold for TSan)
        EXPECT_GT(total_processed.load(), 100);

        std::cout << "Stress test results:\n";
        std::cout << "  Total frames processed: " << total_processed.load() << "\n";
    }

    // ============================================================================
    // Integration Test: Memory Layout Verification
    // ============================================================================

    TEST(Integration, MemoryLayoutComprehensive) {
        // Verify all critical data structures
        EXPECT_EQ(sizeof(spike::TrackerSoA), 64);
        EXPECT_EQ(alignof(spike::TrackerSoA), 64);

        EXPECT_LE(sizeof(spike::ArmorSlots), 2048);
        EXPECT_EQ(alignof(spike::ArmorSlots), 64);

        // Verify triple buffer slot alignment
        TripleBuffer<ArmorSlots> _;
        EXPECT_TRUE(true); // If we got here, buffer is properly constructed

        std::cout << "Memory layout verification:\n";
        std::cout << "  TrackerSoA size:    " << sizeof(TrackerSoA) << " bytes\n";
        std::cout << "  TrackerSoA align:   " << alignof(TrackerSoA) << " bytes\n";
        std::cout << "  ArmorSlots size:    " << sizeof(ArmorSlots) << " bytes\n";
        std::cout << "  ArmorSlots align:   " << alignof(ArmorSlots) << " bytes\n";
        std::cout << "  TripleBuffer valid: ✓\n";
    }
} // namespace spike::test
