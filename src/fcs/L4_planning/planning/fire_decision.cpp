#include "L4_planning/planning/fire_decision.hpp"

#include "toml_helper.hpp"

namespace fcs::L4 {

std::expected<void, std::string> FireDecisionConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, fire_thresh);
    READ(tbl, shooting_range_h);
    READ(tbl, shooting_range_w);

    // Validate
    if (fire_thresh <= 0.0) {
        return std::unexpected(
            "FireDecisionConfig validation failed: [fire_thresh <= 0]. "
            "fire_thresh must be positive");
    }
    if (shooting_range_h <= 0.0) {
        return std::unexpected(
            "FireDecisionConfig validation failed: [shooting_range_h <= 0]. "
            "shooting_range_h must be positive");
    }
    if (shooting_range_w <= 0.0) {
        return std::unexpected(
            "FireDecisionConfig validation failed: [shooting_range_w <= 0]. "
            "shooting_range_w must be positive");
    }

    return {};
}

} // namespace fcs::L4
