#pragma once

#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "scheduler/thin.hpp"

namespace fcs::energy_meter {

struct EnergyMeterL3Config {
    ::energy_meter::BigRuneModel::Params big_model{};
    ::energy_meter::SmallRuneModel::Params small_model{};

    int lost_thres{10};
    int tracking_thres{3};
    double matcher_gate{25.0};
    int blade_unlock_frames{10};
};

/// Register energy meter L3 estimation systems with the talos scheduler.
void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config);

} // namespace fcs::energy_meter
