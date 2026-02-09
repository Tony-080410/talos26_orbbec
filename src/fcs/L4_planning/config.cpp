#include "fcs/L4_planning/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L4 {

// ============================================================================
// AimerConfig
// ============================================================================

std::expected<void, std::string> AimerConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, delay);

    return {};
}

// ============================================================================
// L4Config
// ============================================================================

std::expected<void, std::string> L4Config::read_from(const toml::table& tbl) noexcept {
    READ(tbl, fire_decision);
    READ(tbl, aimer);
    return {};
}

} // namespace fcs::L4
