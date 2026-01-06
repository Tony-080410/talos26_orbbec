#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <expected>
#include <string>

// Forward declaration to avoid including toml++
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs {

struct CameraProfileConfig {
    bool trigger_mode{false};
    bool invert_image{false};
    uint32_t exposure_time_us{3000}; // 3ms in microseconds
    double gain{16.7};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

struct CameraConfig {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix;
    Eigen::Matrix<double, 1, 5> distort_coefficient;
    uint32_t width{1440};
    uint32_t height{1080};
    CameraProfileConfig profile{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs
