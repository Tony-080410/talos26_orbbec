#pragma once

#include "L3_estimation/energy_meter_solver/angle_fitter.hpp"
#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "scheduler/thin.hpp"

namespace fcs::energy_meter {

// ============================================================================
// Energy Meter L3 Configuration
// ============================================================================
//
// Migrated from rm_power_rune's Ceres+RANSAC angle fitting.
//
// Big rune model: θ(t) = -a·cos(ω·(t+t0)) + b·t + c
//   Fitted via RANSAC + Ceres least squares with convexity constraints.
//
// Small rune: constant rotation speed ω.
//
// Configuration:
//   [energy_meter]
//   is_big_rune = true
//   control_delay_s = 0.05
//
//   [energy_meter.fitter]
//   min_fit_data_size = 20
//   max_fit_data_size = 1200
//   initial_params = [0.470, 1.942, 0.0, 1.178, 0.0]
//   small_rune_rotation_speed = 1.04719
//   direction_threshold = 3
//   direction_angle_threshold = 0.015
//
//   [energy_meter.tracker]
//   lost_thres = 10
//   tracking_thres = 3
//   matcher_gate = 25.0
//
//   [energy_meter.tracker.energy_params]
//   sigma_xy = 10.0
//   sigma_z = 1.0
//   sigma_roll = 1.0
//   radius = 0.7
//   meas_dist_k = 0.43
//

struct EnergyMeterL3Config {
    /// ISRCKF tracker configuration
    ::energy_meter::Tracker::Params tracker{};

    /// Angle fitter configuration (Ceres+RANSAC parameters)
    ::energy_meter::AngleFitterConfig fitter{};

    /// Prediction delay (seconds).  Includes image latency + system delay.
    double control_delay_s{0.05};

    /// Rune type: true=big (sinusoidal model), false=small (constant speed).
    bool is_big_rune{true};
};

/// Register energy meter L3 estimation systems with the talos scheduler.
void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config);

} // namespace fcs::energy_meter
