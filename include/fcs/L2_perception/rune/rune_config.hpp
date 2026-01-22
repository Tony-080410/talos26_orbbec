#pragma once

#include "core/armor_types.hpp"

#include <expected>
#include <string>

namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::rune {

struct RuneDetectorConfig {
    bool enabled{false};
    ArmorColor detect_color{ArmorColor::Red};

    int arrow_threshold{170};
    int target_threshold{130};
    int rcenter_threshold{120};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::rune
