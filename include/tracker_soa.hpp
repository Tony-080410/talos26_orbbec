#pragma once

#include <atomic>

namespace spike {
    /**
     * @brief Tracker state machine enumerator
     *
     * State transitions:
     * IDLE --[detection]--> DETECTING --[confirmed]--> TRACKING
     *                                                      |
     *                           <-- [miss] --> TEMP_LOST --+
     *                           [timeout] -> IDLE
     */
    enum class TrackerState : uint8_t {
        IDLE = 0, // No target tracked
        DETECTING = 1, // Candidate detected, waiting for confirmation
        TRACKING = 2, // Actively tracking target
        TEMP_LOST = 3 // Lost tracking, predicting only
    };

    /**
     * @brief Tracker state - optimized for L1 cache residence
     *
     * Critical design constraint: sizeof(TrackerSoA) == 64 bytes EXACTLY
     * This ensures the entire tracker state fits in a single L1 cache line,
     * guaranteeing 100% cache hit rate on all state reads/writes.
     *
     * Precision trade-off: Using float32 instead of double64 to fit in 64 bytes.
     * This is acceptable because:
     * - Armor position accuracy is limited by camera resolution (~0.1-0.2 pixels)
     * - UKF prediction uses numerical integration (Rk4 covariance evolution)
     * - float32 precision (~7 decimal digits) is sufficient for ~1mm accuracy
     *
     * Memory layout (all fields packed):
     * ┌─────────────────────────────────────────────────────────────┐
     * │ Offset  │ Field               │ Size  │ Total Offset         │
     * ├─────────┼─────────────────────┼───────┼──────────────────────┤
     * │  0      │ state (atomic<u8>)  │  1    │  1                   │
     * │  1      │ detecting_count     │  1    │  2                   │
     * │  2      │ lost_count          │  1    │  3                   │
     * │  3      │ generation          │  1    │  4                   │
     * │  4      │ pos[3] (float)      │ 12    │ 16                   │
     * │ 16      │ vel[3] (float)      │ 12    │ 28                   │
     * │ 28      │ yaw, v_yaw (float)  │  8    │ 36                   │
     * │ 36      │ radius[2] (float)   │  8    │ 44                   │
     * │ 44      │ padding             │ 20    │ 64                   │
     * └─────────┴─────────────────────┴───────┴──────────────────────┘
     *
     * Total: 64 bytes (single L1 cache line)
     *
     * Access patterns:
     * - Predict: read pos/vel/yaw/v_yaw, write pos/yaw
     * - Update: read all (11D), compute Kalman gain, write all
     * - State transition: read/write state (atomic)
     * All operations are contained within the 64-byte cache line.
     */
    struct alignas(64) TrackerSoA {
        // === State Machine (4 bytes) ===
        std::atomic<TrackerState> state{TrackerState::IDLE}; // Atomic for CAS operations
        int8_t detecting_count{0}; // Frames since DETECTING started
        int8_t lost_count{0}; // Frames since tracking lost
        uint8_t generation{0}; // Version counter for ABA prevention

        // === 11D State Vector (44 bytes) ===
        // Position (3D)
        float pos[3]{0.0f, 0.0f, 0.0f}; // [x, y, z] world position

        // Velocity (3D)
        float vel[3]{0.0f, 0.0f, 0.0f}; // [vx, vy, vz] linear velocity

        // Rotation (2D)
        float yaw{0.0f}; // Rotation angle (radians)
        float v_yaw{0.0f}; // Angular vel{}ocity (rad/s)

        // Shape (2D)
        float radius[2]{1.0f, 1.0f}; // [radius_x, radius_y] armor dimensions ratio

        // === Cache Line Padding (20 bytes) ===
        char _padding[20];

        /**
         * @brief Verify this tracker fits in exactly one cache line
         */
        static constexpr void static_checks() {
            static_assert(sizeof(TrackerSoA) == 64, "TrackerSoA must be exactly 64 bytes for L1 cache residence");
            static_assert(alignof(TrackerSoA) == 64, "TrackerSoA must be 64-byte aligned (cache line alignment)");
        }

        /**
         * @brief State machine transition: IDLE -> DETECTING
         * Called when first armor detected in IDLE state
         */
        bool try_start_detecting() {
            auto expected = TrackerState::IDLE;
            return state.compare_exchange_strong(expected, TrackerState::DETECTING, std::memory_order_release,
                                                 std::memory_order_relaxed);
        }

        /**
         * @brief State machine transition: DETECTING -> TRACKING
         * Called when detection confidence threshold reached
         */
        bool try_confirm_tracking() {
            auto expected = TrackerState::DETECTING;
            return state.compare_exchange_strong(expected, TrackerState::TRACKING, std::memory_order_release,
                                                 std::memory_order_relaxed);
        }

        /**
         * @brief State machine transition: TRACKING -> TEMP_LOST
         * Called when detection missed
         */
        bool try_enter_lost() {
            auto expected = TrackerState::TRACKING;
            return state.compare_exchange_strong(expected, TrackerState::TEMP_LOST, std::memory_order_release,
                                                 std::memory_order_relaxed);
        }

        /**
         * @brief State machine transition: TEMP_LOST -> TRACKING
         * Called when target re-detected after being lost
         */
        bool try_restore_tracking() {
            auto expected = TrackerState::TEMP_LOST;
            return state.compare_exchange_strong(expected, TrackerState::TRACKING, std::memory_order_release,
                                                 std::memory_order_relaxed);
        }

        /**
         * @brief State machine transition: * -> IDLE
         * Called when timeout (too many consecutive misses)
         */
        bool try_reset_idle() {
            return state.exchange(TrackerState::IDLE, std::memory_order_release) != TrackerState::IDLE;
        }

        /**
         * @brief Increment generation counter for ABA prevention
         */
        void next_generation() {
            generation++; // Wrap-around is intentional and safe
        }
    };

    // Compile-time verification
    static_assert(sizeof(TrackerSoA) == 64, "TrackerSoA size verification failed");
    static_assert(alignof(TrackerSoA) == 64, "TrackerSoA alignment verification failed");
} // namespace spike
