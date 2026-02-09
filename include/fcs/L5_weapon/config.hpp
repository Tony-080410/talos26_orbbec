#pragma once

#include <expected>
#include <memory>
#include <numbers>

// Forward declaration to avoid including toml++ in config headers.

namespace toml::inline v3 {
class table;
}

namespace fcs::L5 {

// ============================================================================
// Enhanced Weapon Controller Configuration (L5 Layer)
// ============================================================================

/// Weapon control configuration - tactical parameters only
struct WeaponControllerConfig {
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// ============================================================================
// L5 Weapon Full Config
// ============================================================================

struct L5Config {
    WeaponControllerConfig weapon{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L5
