#pragma once

#include <expected>

#include "fcs/L4_planning/mpc_planner/config.hpp"

// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L4 {

/// L4 planning configuration (MPC-only)
struct L4Config {
    MpcPlannerConfig mpc_planner{};

    [[nodiscard]] const MpcPlannerConfig& mpc_planner_ref() const noexcept { return mpc_planner; }
    [[nodiscard]] MpcPlannerConfig& mpc_planner_ref() noexcept { return mpc_planner; }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
