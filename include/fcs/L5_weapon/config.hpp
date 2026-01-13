#pragma once

#include <expected>
#include <memory>

// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

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

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const WeaponControllerConfig> weapon_ptr() const {
        return std::make_shared<const WeaponControllerConfig>(weapon);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L5
