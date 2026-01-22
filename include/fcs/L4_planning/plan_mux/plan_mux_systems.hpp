#pragma once

#include <expected>
#include <string>

namespace toml::inline v3 {
class table;
}

namespace talos {
class Scheduler;
}

namespace fcs::L4 {

enum class TargetMode {
    Armor,
    Rune,
    Auto,
};

struct PlanMuxConfig {
    TargetMode mode{TargetMode::Auto};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

void register_plan_mux_systems(::talos::Scheduler& scheduler, PlanMuxConfig&& config);

} // namespace fcs::L4
