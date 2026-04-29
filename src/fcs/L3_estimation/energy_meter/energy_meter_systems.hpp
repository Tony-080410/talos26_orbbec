#pragma once

#include "L3_estimation/energy_meter_solver/angle_fitter.hpp"
#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "scheduler/thin.hpp"

namespace fcs::energy_meter {

struct EnergyMeterL3Config {
    double min_fit_time{2};
    /// ISRCKF tracker configuration
    ::energy_meter::Tracker::Params tracker{};

    /// Angle fitter configuration (Ceres+RANSAC parameters)
    ::energy_meter::AngleFitterConfig fitter{};
};

/// Register energy meter L3 estimation systems with the talos scheduler.
void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config);

} // namespace fcs::energy_meter
