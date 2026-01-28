#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/core.h>

namespace fcs {

// ============================================================================
// Armor Enums
// ============================================================================

enum class ArmorType : uint8_t {
    Small   = 0,
    Large   = 1,
    Invalid = 2,
};

const std::array<std::string, 3> ARMOR_TYPES = {"small", "big", "invalid"};
constexpr std::string_view to_string(const ArmorType c) noexcept {
    return ARMOR_TYPES[static_cast<uint8_t>(c)];
}

enum class ArmorColor : uint8_t {
    Blue    = 0,
    Red     = 1,
    Neutral = 2,
    Purple  = 3,
};

const std::array<std::string, 4> COLORS = {"blue", "red", "neutral", "purple"};
constexpr std::string_view to_string(const ArmorColor c) noexcept {
    return COLORS[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}

enum class ArmorName : uint8_t {
    Sentry = 0,
    One,
    Two,
    Three,
    Four,
    Five,
    Outpost,
    Base,
    BaseLarge,
    Invalid
};

const std::array<std::string, 10> ARMOR_NAMES = {
    "sentry", "one", "two", "three", "four", "five", "outpost", "base", "base_large", "invalid"};
constexpr std::string_view to_string(const ArmorName c) noexcept {
    return ARMOR_NAMES[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}

enum class TrackStatus : uint8_t {
    Invalid   = 0, // 无效
    Tentative = 1, // 正在确认
    Confirmed = 2, // 已确认
    Coasting  = 3, // 暂时丢失量测， 使用预测值
};

const std::array<std::string, 4> TRACKER_STATE_NAMES = {
    "INVALID", "TENTATIVE", "CONFIRMED", "COASTING"};
constexpr std::string_view to_string(const TrackStatus c) noexcept {
    return TRACKER_STATE_NAMES[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}

// ============================================================================
// AT Legacy Armor Constants (meters)
// ============================================================================

inline constexpr double AT_LEGACY_SMALL_ARMOR_WIDTH  = 135.0 / 1000.0;
inline constexpr double AT_LEGACY_SMALL_ARMOR_HEIGHT = 55.0 / 1000.0;
inline constexpr double AT_LEGACY_LARGE_ARMOR_WIDTH  = 230.0 / 1000.0;
inline constexpr double AT_LEGACY_LARGE_ARMOR_HEIGHT = 55.0 / 1000.0;

// ============================================================================
// AT Legacy cls/color Mapping (from NN output)
// ============================================================================

/// Determine ArmorType from armor name
/// Hero (One) and BaseLarge are Large armor types
[[nodiscard]] constexpr ArmorType at_legacy_cls_to_armor_type(ArmorName armor_name) noexcept {
    return (armor_name == ArmorName::One || armor_name == ArmorName::BaseLarge) ? ArmorType::Large
                                                                                : ArmorType::Small;
}

} // namespace fcs

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<fcs::ArmorType> : formatter<std::string_view> {
    auto format(const fcs::ArmorType c, format_context& ctx) const {
        return formatter<std::string_view>::format(fcs::to_string(c), ctx);
    }
};

template <>
struct formatter<fcs::ArmorColor> : formatter<std::string_view> {
    auto format(const fcs::ArmorColor c, format_context& ctx) const {
        return formatter<std::string_view>::format(fcs::to_string(c), ctx);
    }
};

template <>
struct formatter<fcs::ArmorName> : formatter<std::string_view> {
    auto format(const fcs::ArmorName c, format_context& ctx) const {
        return formatter<std::string_view>::format(fcs::to_string(c), ctx);
    }
};

template <>
struct formatter<fcs::TrackStatus> : formatter<std::string_view> {
    auto format(const fcs::TrackStatus c, format_context& ctx) const {
        return formatter<std::string_view>::format(fcs::to_string(c), ctx);
    }
};

} // namespace fmt

