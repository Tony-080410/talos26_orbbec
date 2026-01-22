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

struct RunePlannerConfig {
    bool enabled{false};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

void register_rune_planning_systems(talos::Scheduler& scheduler, RunePlannerConfig&& config);

} // namespace fcs::L4
