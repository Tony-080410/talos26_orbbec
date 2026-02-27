#pragma once

#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "scheduler/thin.hpp"

namespace toml::inline v3 {
class table;
}

namespace fcs::energy_meter {

// ============================================================================
// Energy Meter L3 Configuration
// ============================================================================
//
// Problem:
//   The energy meter (符文) is a rotating target with periodic motion. We need to:
//   1) Track the rotation state (roll angle and velocity)
//   2) Fit a motion model (sinusoidal for big rune, linear for small rune)
//   3) Predict the future position for shooting
//
// Intuition:
//   - Big rune: follows θ(t) = -a/ω·cos(ωt + φ) + b·t + c (sinusoidal)
//   - Small rune: follows θ(t) = ω·t + b (constant rotation)
//   - The angle fitter uses velocity-based LSQ with grid search for ω
//
// Configuration:
//   [energy_meter]
//   is_big_rune = true           # true=big rune (sinusoidal), false=small (linear)
//   control_delay_s = 0.05       # Predictive shooting delay (seconds)
//
//   [energy_meter.tracker]       # ISRCKF tracker params
//   lost_thres = 10              # Lost frames before reset
//   tracking_thres = 3           # Consecutive detections to enter TRACKING
//   matcher_gate = 25.0          # Mahalanobis gate for association
//
//   [energy_meter.tracker.energy_params]  # Process noise covariance
//   sigma_xy = 10.0              # XY position process noise std-dev (m)
//   sigma_z = 1.0                # Z position process noise std-dev (m)
//   sigma_roll = 1.0             # Roll angle process noise std-dev (rad)
//   radius = 0.7                 # Initial radius guess (m)
//   meas_dist_k = 0.43           # Measurement distance scaling factor
//
//   [energy_meter.angle_fitter]  # Angle model fitting params
//   smooth_alpha = 0.8           # Velocity filter exponential smoothing
//   min_data_points = 15         # Minimum observations before fitting
//

// ============================================================================
// Angle Fitter Configuration
// ============================================================================

/// Configuration for the angle model fitter.
struct AngleFitterConfig {
    /// Velocity filter exponential smoothing factor (0-1).
    ///
    /// Problem:
    ///   Raw velocity from finite differences is very noisy. The angle fitter needs
    ///   smooth velocity input for stable LSQ fitting.
    ///
    /// Intuition:
    ///   Closer to 1 → smoother but laggier; closer to 0 → more responsive but noisier.
    ///   Typical value: 0.8 (strong smoothing).
    ///
    /// Precise:
    ///   In AngleFitter::setFilteredVelocity(), updates internal filtered velocity as:
    ///   v_filtered = smooth_alpha * v_filtered + (1 - smooth_alpha) * v_new
    double smooth_alpha{0.8};

    /// Minimum number of observations before fitting the angle model.
    ///
    /// Problem:
    ///   LSQ fitting requires sufficient data points to solve for parameters.
    ///   Fitting too early produces garbage parameters.
    ///
    /// Intuition:
    ///   Larger → more stable but slower convergence; smaller → faster but more noise.
    ///
    /// Precise:
    ///   In AngleFitter::fitAngleModelLSQ(), returns false if observations.size() < this value.
    ///   Big rune needs at least 3 points for [a, ω, φ, b, c]; small rune needs 2 for [ω, b].
    int min_data_points{15};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// ============================================================================
// Energy Meter L3 Full Config
// ============================================================================

struct EnergyMeterL3Config {
    /// ISRCKF tracker configuration (association, state machine, process noise)
    ::energy_meter::Tracker::Params tracker{};

    /// Angle fitter configuration (model fitting parameters)
    AngleFitterConfig angle_fitter{};

    /// Control delay for predictive shooting (seconds).
    ///
    /// Problem:
    ///   By the time we compute and send the shooting command, the target has moved.
    ///   We need to predict where it will be when the projectile arrives.
    ///
    /// Intuition:
    ///   Larger → predict further ahead (requires better model accuracy); smaller →
    ///   shoot at current position (may miss if target is moving fast).
    ///
    /// Precise:
    ///   Total prediction delay = image_delay (now - timestamp) + control_delay_s.
    ///   Used to compute the future angle: θ_future = θ_current - v_roll * total_delay.
    double control_delay_s{0.05};

    /// Rune type: true for big rune (sinusoidal), false for small rune (linear).
    bool is_big_rune{true};

    /// Parse configuration from TOML table.
    /// Uses READ macro for required fields, READ_OPT for optional fields.
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

/// Register energy meter L3 estimation systems with the talos scheduler.
///
/// This creates:
/// - l3_energy_meter system (pool_compute): processes RuneObservation → EnergyMeterState
/// - Tracker and AngleFitter as shared resources
void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config);

} // namespace fcs::energy_meter
