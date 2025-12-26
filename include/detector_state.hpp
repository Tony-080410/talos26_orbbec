#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <string>

namespace talos {

// ============ 枚举类型 ============

enum Color : uint8_t { red = 0, blue = 1, extinguish = 2, purple = 3 };
const std::array<std::string, 4> COLORS = {"red", "blue", "extinguish", "purple"};
enum ArmorType : uint8_t { big = 0, small = 1 };
const std::array<std::string, 2> ARMOR_TYPES = {"big", "small"};
enum ArmorName : uint8_t { one = 0, two, three, four, five, sentry, outpost, base, not_armor };
const std::array<std::string, 9> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                                "sentry", "outpost", "base",  "not_armor"};

constexpr std::string_view color_name(const Color c) noexcept { return COLORS[c]; }
constexpr std::string_view armor_type(const ArmorType c) noexcept { return ARMOR_TYPES[c]; }
constexpr std::string_view armor_name(const ArmorName c) noexcept { return ARMOR_NAMES[c]; }

// ============ 缓存友好的输出结构 ============

/**
 * @brief Lightbar - 20 bytes, cache-optimized output
 */
struct Lightbar {
    float x, y;        // Center position (8B)
    float angle_error; // Error from ideal angle (4B)
    float ratio;       // length / width (4B)
    uint8_t color;     // 0=red, 1=blue, 2=extinguish, 3=purple (1B)
    bool valid;
    uint8_t pad_[2];
};
static_assert(sizeof(Lightbar) == 20, "Lightbar must be exactly 20 bytes");

/**
 * @brief Armor - 48 bytes, cache-optimized output
 */
struct Armor {
    float x2d, y2d;   // 2D center position (8B)
    float pts[8];     // 4 corner points: tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y (32B)
    float confidence; // Detection confidence [0, 1] (4B)
    Color color;      // Color enum (1B)
    ArmorType type;   // ArmorType enum (1B)
    ArmorName name;   // ArmorName enum (1B)
    bool valid;
};
static_assert(sizeof(Armor) == 48, "Armor must be exactly 48 bytes");

/**
 * @brief ArmorSlots - Structure-of-Arrays layout for detector output
 */
struct ArmorSlots {
    uint8_t lightbar_count{0}; // Number of detected lightbars
    uint8_t armor_count{0};    // Number of valid armors

    std::array<Armor, 10> armors{};
    std::array<Lightbar, 20> lightbars{};

    uint64_t timestamp_ns{0};
    uint64_t frame_id{0};

    [[nodiscard]] constexpr uint8_t count_valid() const noexcept {
        if (armor_count == 0) {
            return 0;
        }
        auto count = 0;
        for (auto i = 0; i < armor_count; ++i) {
            if (armors[i].valid) {
                count++;
            }
        }
        return count;
    }

    /**
     * @deprecated set valid in Armor directly
     */
    constexpr void set_valid(const uint8_t slot) noexcept {
        assert(slot < 10);
        armors[slot].valid = true;
    }

    /**
     * @deprecated set invalid in Armor directly
     */
    constexpr void set_invalid(const uint8_t slot) noexcept {
        assert(slot < 10);
        armors[slot].valid = false;
    }

    /**
     * @deprecated use valid in Armor directly
     */
    [[nodiscard]] constexpr bool is_valid(const uint8_t slot) const noexcept {
        assert(slot < 10);
        return armors[slot].valid;
    }

    constexpr void clear() noexcept {
        lightbar_count = 0;
        armor_count    = 0;
        for (auto i = 0; i < 10; ++i) {
            armors[i].valid = false;
        }
        for (auto i = 0; i < 20; ++i) {
            lightbars[i].valid = false;
        }
    }
};

// Verify memory layout
static_assert(sizeof(ArmorSlots) <= 1024, "ArmorSlots exceeds 1KB budget");

} // namespace talos
