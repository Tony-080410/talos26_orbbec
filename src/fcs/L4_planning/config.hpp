#pragma once

#include <expected>

#include "L4_planning/planning/fire_decision.hpp"
#include "core/trajectory/config.hpp"

// Forward declaration to avoid including toml++ in config headers.
namespace toml::inline v3 {
class table;
}

namespace fcs::L4 {

/// Aimer configuration (shared by all target types)
struct AimerConfig {
    /// System delay compensation (seconds)
    /// Includes: camera exposure, image processing, communication, gimbal response
    double delay{0.01};

    [[nodiscard]] std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

/// L4 planning configuration (Aimer + plan adapter architecture)
struct L4Config {
    // Fire decision configuration shared by L4 aiming and L5 fire gating.
    FireDecisionConfig fire_decision{};

    // Shared aimer configuration (used for Robot/Outpost/Rune)
    AimerConfig aimer{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
