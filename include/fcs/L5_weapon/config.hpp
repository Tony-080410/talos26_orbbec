#pragma once

#include "toml_helper.hpp"
#include <expected>
#include <toml++/toml.hpp>

namespace fcs::L5 {

// ============================================================================
// Enhanced Weapon Controller Configuration (L5 Layer)
// ============================================================================

/// Weapon control configuration - tactical parameters only
struct WeaponControllerConfig {
    /// Fire delay compensation (seconds)
    double fire_delay{0.05};

    /// Minimum confidence threshold for firing
    double min_fire_confidence{0.7};

    /// Maximum angular velocity threshold for firing (rad/s)
    double max_fire_angular_velocity{2.0};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, fire_delay);
        READ(tbl, min_fire_confidence);
        READ(tbl, max_fire_angular_velocity);
        return {};
    }
};

// ============================================================================
// L5 Weapon Full Config
// ============================================================================

struct L5Config {
    WeaponControllerConfig weapon{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const WeaponControllerConfig> weapon_ptr() const {
        return std::make_shared<const WeaponControllerConfig>(weapon);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, weapon);
        return {};
    }
};

} // namespace fcs::L5