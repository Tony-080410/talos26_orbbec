#pragma once

#include <expected>

#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L4_planning/planning/fire_decision.hpp"

// Forward declaration to avoid including toml++ in config headers.
namespace toml::inline v3 {
class table;
}

namespace fcs::L4 {

/// Aimer configuration (shared by all target types)
struct AimerConfig {
    /// System delay compensation (seconds)
    /// Includes: camera exposure, image processing, communication, gimbal response
    double delay{0.015};

    [[nodiscard]] std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

/// L4 planning configuration (Aimer + Planner architecture)
struct L4Config {
    // Fire decision configuration (shared by Aimer and Planner)
    FireDecisionConfig fire_decision{};

    // Shared aimer configuration (used for Robot/Outpost/Rune)
    AimerConfig aimer{};

    // MPC planner configuration
    MpcPlannerConfig mpc_planner{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
