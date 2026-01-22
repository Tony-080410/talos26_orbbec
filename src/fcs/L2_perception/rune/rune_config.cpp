#include "fcs/L2_perception/rune/rune_config.hpp"

#include "toml_helper.hpp"

namespace fcs::rune {

std::expected<void, std::string> RuneDetectorConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, enabled);
    READ(tbl, detect_color);

    READ(tbl, arrow_threshold);
    READ(tbl, target_threshold);
    READ(tbl, rcenter_threshold);

    return {};
}

} // namespace fcs::rune
