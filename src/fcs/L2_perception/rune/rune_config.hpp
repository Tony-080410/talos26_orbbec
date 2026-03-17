#pragma once

#include "core/armor_types.hpp"

namespace fcs::rune {

struct RuneDetectorConfig {
    bool enabled{false};
    ArmorColor detect_color{ArmorColor::Red};

    int arrow_threshold{170};
    int target_threshold{130};
    int rcenter_threshold{120};
};

} // namespace fcs::rune
