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
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept { return {}; }
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
        READ_OPT(tbl, weapon);
        return {};
    }
};

} // namespace fcs::L5
