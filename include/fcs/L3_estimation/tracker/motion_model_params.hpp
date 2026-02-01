#pragma once

#include <expected>
#include <string>

// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L3 {

struct RobotModelParams {
    double sigma_a_xy  = 10.0;
    double sigma_a_z   = 1.0;
    double sigma_a_yaw = 1.0;
    double sigma_r0    = 1.0;
    double sigma_r1    = 1.0;
    double sigma_h     = 0.5;
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

struct OutpostModelParams {
    double sigma_q_xy  = 10.0;
    double sigma_q_z   = 1.0;
    double sigma_a_yaw = 1.0;
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L3
