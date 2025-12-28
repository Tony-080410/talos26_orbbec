#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace fcs {

enum class ArmorType : uint8_t {
    Small = 0,
    Big   = 1,
};

const std::array<std::string, 2> ARMOR_TYPES = {"big", "small"};
constexpr std::string_view armor_type(const ArmorType c) noexcept {
    return ARMOR_TYPES[static_cast<uint8_t>(c)];
}

enum class ArmorColor : uint8_t {
    Red     = 0,
    Blue    = 1,
    Neutral = 2,
    Purple  = 3,
};

const std::array<std::string, 4> COLORS = {"red", "blue", "neutral", "purple"};
constexpr std::string_view armor_color(const ArmorColor c) noexcept {
    return COLORS[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}

enum class ArmorName : uint8_t { One = 0, Two, Three, Four, Five, Sentry, Outpost, Base, Invalid };

const std::array<std::string, 9> ARMOR_NAMES = {"one",    "two",     "three", "four",   "five",
                                                "sentry", "outpost", "base",  "invalid"};
constexpr std::string_view armor_name(const ArmorName c) noexcept {
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
constexpr std::string_view track_status(const TrackStatus c) noexcept {
    return TRACKER_STATE_NAMES[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}
} // namespace fcs
