#pragma once

#include "fcs/L3_estimation/energy_meter/types.hpp"
#include "fcs/L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"

namespace toml::inline v3 {
class table;
}

namespace talos {
class Scheduler;
}

namespace fcs::energy_meter {

struct EnergyMeterL3Config {
    ::energy_meter::Tracker::Params tracker{};

    double smooth_alpha{0.8};
    int min_data_points{15};

    double control_delay_s{0.05};

    bool is_big_rune{true};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

void register_energy_meter_systems(::talos::Scheduler& scheduler, EnergyMeterL3Config&& config);

} // namespace fcs::energy_meter
