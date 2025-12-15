#pragma once

#include <array>
#include <algorithm>
#include <numeric>

namespace spike {
    /**
     * @brief Armor data - individual armor plate detection result
     * Fits in SoA layout with fixed 10-slot capacity
     */
    struct Armor {
        float x2d, y2d; // 2D image position
        int id; // Armor ID
        uint8_t cls; // Classification (1-5)
        float confidence; // Detection confidence [0, 1]
    };

    /**
     * @brief PnP pose result - 3D location and orientation
     */
    struct Pose {
        float x, y, z; // 3D position in world frame
        float qx, qy, qz, qw; // Unit quaternion (normalized)
    };

    /**
     * @brief ArmorSlots - Structure-of-Arrays layout for detector output
     *
     * Design rationale:
     * - Fixed 10-slot capacity (max armors per frame in RoboMaster)
     * - Uses bitmask to track which slots are valid
     * - Achieves SoA memory layout with O(1) iteration
     * - Total size: ~1.2KB (acceptable for single-frame cache)
     *
     * Memory layout (cache-friendly):
     * ┌────────────────────────────────────────────────────────┐
     * │ valid_mask (1 byte)                                    │
     * │ padding (7 bytes for alignment)                        │
     * │ armors[0..9] (10 × ~20B = 200B)                        │
     * │ poses[0..9] (10 × ~28B = 280B)                         │
     * │ timestamp_ns (8 bytes)                                 │
     * │ frame_id (8 bytes)                                     │
     * └────────────────────────────────────────────────────────┘
     * Total: ~512 bytes (fits in L1 TLB)
     */
    struct alignas(64) ArmorSlots {
        uint8_t valid_mask{0}; // Bitmask: bit i = 1 iff armors[i] valid
        uint8_t padding_[7]; // Alignment padding
        std::array<Armor, 10> armors; // SoA: All armors' x2d in sequence, then y2d, etc.
        std::array<Pose, 10> poses; // SoA: Corresponding PnP results
        uint64_t timestamp_ns{0}; // Frame timestamp (nanoseconds)
        uint64_t frame_id{0}; // Monotonic frame counter

        /**
         * @brief Count valid armors in this frame
         * @return Number of armors (0-10)
         */
        [[nodiscard]] size_t count_valid() const {
            return __builtin_popcount(valid_mask);
        }

        /**
         * @brief Mark armor slot as valid
         */
        void set_valid(const int slot) {
            if (slot >= 0 && slot < 10) {
                valid_mask |= 1u << slot;
            }
        }

        /**
         * @brief Mark armor slot as invalid
         */
        void set_invalid(const int slot) {
            if (slot >= 0 && slot < 10) {
                valid_mask &= ~(1u << slot);
            }
        }

        /**
         * @brief Check if slot is valid
         */
        [[nodiscard]] bool is_valid(const int slot) const {
            return (valid_mask & 1u << slot) != 0;
        }

        /**
         * @brief Clear all slots
         */
        void clear() {
            valid_mask = 0;
        }

        /**
         * @brief Iterate over valid armor indices
         * Example usage:
         *   for (int i = 0; i < 10; i++) {
         *     if (frame.is_valid(i)) {
         *       process(frame.armors[i], frame.poses[i]);
         *     }
         *   }
         */
    };

    // Verify memory layout
    static_assert(sizeof(ArmorSlots) <= 1024, "ArmorSlots exceeds 1KB budget");
    static_assert(alignof(ArmorSlots) == 64, "ArmorSlots must be cache-line aligned");
} // namespace spike
