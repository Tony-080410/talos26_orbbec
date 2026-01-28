#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <expected>
#include <fmt/core.h>
#include <string>

#include "magic_enum.hpp"

// Forward declaration to avoid including toml++
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs {

enum class RotateType {
    None,
    Clockwise90,
    Clockwise180,
    Clockwise270,
};

struct CameraProfileConfig {
    bool trigger_mode{false};
    bool invert_image{false};
    uint32_t exposure_time_us{3000}; // 3ms in microseconds
    double gain{16.7};
    RotateType rotate_angle{RotateType::None};

    [[nodiscard]] std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

struct CameraConfig {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix;
    Eigen::Matrix<double, 1, 5> distort_coefficient;
    uint32_t width{1440};
    uint32_t height{1080};
    CameraProfileConfig profile{};

    [[nodiscard]] std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<fcs::RotateType> : formatter<std::string_view> {
    auto format(fcs::RotateType type, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(type), ctx);
    }
};

} // namespace fmt
