#pragma once

#include <array>
#include <cstdint>

namespace spike {

    // ============ 枚举类型 ============

    enum Color : uint8_t { red = 0, blue = 1, extinguish = 2, purple = 3 };
    const std::array<std::string, 4> COLORS = {"red", "blue", "extinguish", "purple"};
    enum ArmorType : uint8_t { big = 0, small = 1 };
    const std::array<std::string, 2> ARMOR_TYPES = {"big", "small"};
    enum ArmorName : uint8_t { one = 0, two, three, four, five, sentry, outpost, base, not_armor };
    const std::array<std::string, 9> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                                    "sentry", "outpost", "base",  "not_armor"};
    // ============ 缓存友好的输出结构 ============

    /**
     * @brief Lightbar - 20 bytes, cache-optimized output
     */
    struct Lightbar {
        float x, y; // Center position (8B)
        float angle_error; // Error from ideal angle (4B)
        float ratio; // length / width (4B)
        uint8_t color; // 0=red, 1=blue, 2=extinguish, 3=purple (1B)
        uint8_t _pad[3]; // Padding (3B)
    };
    static_assert(sizeof(Lightbar) == 20, "Lightbar must be exactly 20 bytes");

    /**
     * @brief Armor - 32 bytes, cache-optimized output
     */
    struct Armor {
        float x2d, y2d; // 2D center position (8B)
        float pts[8]; // 4 corner points: tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y (32B)
        float confidence; // Detection confidence [0, 1] (4B)
        uint8_t color; // Color enum (1B)
        uint8_t type; // ArmorType enum (1B)
        uint8_t name; // ArmorName enum (1B)
        uint8_t _pad; // Padding (1B)
    };
    static_assert(sizeof(Armor) == 48, "Armor must be exactly 48 bytes");

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
     * Memory layout (cache-friendly):
     * ┌────────────────────────────────────────────────────────┐
     * │ valid_mask (2 bytes) + lightbar_count (1B) + pad (5B)  │
     * │ armors[0..9] (10 × 48B = 480B)                         │
     * │ poses[0..9] (10 × 28B = 280B)                          │
     * │ lightbars[0..19] (20 × 20B = 400B)                     │
     * │ timestamp_ns (8 bytes)                                 │
     * │ frame_id (8 bytes)                                     │
     * └────────────────────────────────────────────────────────┘
     * Total: ~1184 bytes
     */
    struct alignas(64) ArmorSlots {
        uint16_t valid_mask{0}; // Bitmask: bit i = 1 iff armors[i] valid
        uint8_t lightbar_count{0}; // Number of detected lightbars
        uint8_t armor_count{0}; // Number of valid armors
        uint8_t padding_[4]{}; // Alignment padding

        std::array<Armor, 10> armors{};
        std::array<Pose, 10> poses{};
        std::array<Lightbar, 20> lightbars{};

        uint64_t timestamp_ns{0};
        uint64_t frame_id{0};

        [[nodiscard]] constexpr size_t count_valid() const { return __builtin_popcount(valid_mask); }

        constexpr void set_valid(const int slot) {
            if (slot >= 0 && slot < 10) {
                valid_mask |= 1u << slot;
            }
        }

        constexpr void set_invalid(const int slot) {
            if (slot >= 0 && slot < 10) {
                valid_mask &= ~(1u << slot);
            }
        }

        [[nodiscard]] constexpr bool is_valid(const int slot) const { return (valid_mask & 1u << slot) != 0; }

        constexpr void clear() {
            valid_mask = 0;
            lightbar_count = 0;
            armor_count = 0;
        }
    };

    // Verify memory layout
    static_assert(sizeof(ArmorSlots) <= 2048, "ArmorSlots exceeds 2KB budget");
    static_assert(alignof(ArmorSlots) == 64, "ArmorSlots must be cache-line aligned");

} // namespace spike
