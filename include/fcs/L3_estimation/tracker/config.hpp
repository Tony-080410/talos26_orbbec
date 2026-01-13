#pragma once

#include "inekf_config.hpp"
#include "motion_model.hpp"
#include <expected>
#include <memory>
#include <string>

// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L3 {

// ============================================================================
// Tracker Configuration
// ============================================================================

// ----------------------------------------------------------------------------
// Tracker tuning guide
// ----------------------------------------------------------------------------
//
// Problem:
//   The tracker must convert a noisy per-frame set of armor measurements (false positives, missed
//   detections, multiple armors per target) into a stable target state output.
//
// Intuition:
//   The pipeline is: predict → associate → update → state-machine → output.
//
// Analogy:
//   - `matcher_gate` is the width of an "acceptance corridor" for matching observations.
//   - `tracking_threshold` is a debounce counter before we commit to TRACKING.
//   - `lost_threshold` is a timeout before we drop the track.
//
// Precise (implementation facts you should not guess wrong):
//   1) TOML parsing is strict: `READ(tbl, field)` makes the key/table required. Defaults in this
//      struct are NOT used when reading from TOML. Use `READ_OPT` if you want optional keys.
//   2) Even with InEKF enabled, association still uses the ISRCKF prediction (x_pred, Sx_pred).
//      Therefore `matcher_gate` and `model.*` remain the primary knobs for match stability.
//
// Suggested tuning order:
//   1) Association + state machine: matcher_gate, tracking_threshold, lost_threshold
//   2) ISRCKF process noise: model.*
//   3) InEKF noise / geometry: *_inekf.*

template <typename T>
struct TargetConfig {
    /// Timeout before returning to IDLE from TEMP_LOST (seconds).
    ///
    /// Problem:
    ///   Detection can drop for short intervals (occlusion, blur, lighting). Resetting immediately
    ///   makes the track flicker and downstream behavior unstable.
    ///
    /// Intuition:
    ///   Larger → keep "coasting on prediction" longer; smaller → drop and re-acquire sooner.
    ///
    /// Analogy:
    ///   Like a network timeout / grace period before disconnecting.
    ///
    /// Precise:
    ///   In `Tracker::state_machine` (TempLost), `lost_time_` accumulates by `last_dt_` until
    ///   `lost_time_ >= lost_threshold`, then `reset()` is called. `Tracker::predict()` also does
    ///   an early check using `lost_time_ + last_dt_`.
    double lost_threshold;
    /// Number of consecutive frames required to transition DETECTING -> TRACKING.
    ///
    /// Problem:
    ///   Early detections are the noisiest (false positives / misclassification). We want a short
    ///   confirmation stage before we consider the track "stable".
    ///
    /// Intuition:
    ///   Larger → more conservative (fewer false locks) but slower to enter TRACKING; smaller →
    ///   faster lock but more false tracks.
    ///
    /// Analogy:
    ///   Like debouncing a button / requiring N consecutive votes.
    ///
    /// Precise:
    ///   In `Tracker::state_machine` (Detecting), `detecting_count_` increments on a successful
    ///   association; transition occurs when `detecting_count_ > tracking_threshold`, i.e. it
    ///   effectively requires `tracking_threshold + 1` consecutive "found" frames. Any miss while
    ///   in Detecting triggers an immediate `reset()` (no grace period).
    uint32_t tracking_threshold;
    /// Mahalanobis distance-squared gate (d^2) for observation-to-track association.
    ///
    /// Problem:
    ///   A frame can contain multiple measurements (different armors of the same target, plus
    ///   clutter). We need a principled way to decide which observations are plausible matches.
    ///
    /// Intuition:
    ///   Larger gate → fewer dropouts but more wrong matches; smaller gate → fewer wrong matches
    ///   but more dropouts (TempLost / resets).
    ///
    /// Analogy:
    ///   A chi-square gate: "how wide the match gate is" in normalized units.
    ///
    /// Precise:
    ///   In `DataAssociator::match`, for each observation and armor_id:
    ///     - Build 4D measurement z = [yaw, pitch, distance, armor_yaw].
    ///     - Predict z_pred and Pzz via sigma-point projection of (x_pred, Sx_pred).
    ///     - Form innovation nu with angle wrapping.
    ///     - Form S = Pzz + R, where R comes from `Model::R_diag(z)` (heuristic, NOT from TOML).
    ///     - Compute d^2 = nuᵀ S⁻¹ nu, accept candidate if finite and d^2 < gate_threshold.
    ///   Because Pzz depends on the predicted covariance, increasing process noise in `model.*`
    ///   typically makes gating effectively more permissive even if matcher_gate stays constant.
    double matcher_gate{10.0};
    /// Motion model process-noise parameters (ISRCKF).
    ///
    /// Problem:
    ///   The ISRCKF uses simplified dynamics (roughly constant-velocity / constant-yaw-rate). Real
    ///   targets accelerate, spin, and their geometry parameters are not perfectly known.
    ///
    /// Intuition:
    ///   Larger process noise → more responsive but noisier; smaller → smoother but laggier.
    ///
    /// Analogy:
    ///   Like the "aggressiveness" knob of a low-pass filter: turning it up tracks faster but
    ///   passes more noise.
    ///
    /// Precise:
    ///   These parameters feed into `Model::Q_sqrt(dt)` and inflate the predicted covariance. That
    ///   affects both the measurement update and the association gate (because Pzz depends on Sx).
    ///   Note that ISRCKF measurement noise R is a heuristic in `motion_model.hpp::R_diag` and is
    ///   not configurable via TOML.
    T model;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

struct TrackerConfig {
    /// Robot target parameters (4-armors).
    /// Problem: robots have 4 armors and geometry parameters (r0/r1/h) to estimate.
    TargetConfig<RobotModel::Params> robot;
    /// Outpost target parameters (3-armors).
    /// Problem: outposts have 3 armors and a different (simpler) state/geometry.
    TargetConfig<OutpostModel::Params> outpost;

    /// Enable InEKF parallel filter.
    ///
    /// Problem:
    ///   We want a geometrically consistent pose/velocity estimate with explicit geometry states,
    ///   not just a Euclidean-state CKF output.
    ///
    /// Intuition:
    ///   Turning this on runs an InEKF in parallel and prefers its state output when available.
    ///
    /// Analogy:
    ///   Like running a second estimator in parallel and choosing its output.
    ///
    /// Precise:
    ///   - `first_meet` initializes InEKF when enabled.
    ///   - `predict`/`update` call InEKF alongside ISRCKF.
    ///   - `get_output()` prefers InEKF state if it exists/initialized.
    ///   - Association (armor_id matching) remains ISRCKF-based regardless of this flag.
    bool inekf_enabled{true};
    /// InEKF parameters for robot targets.
    /// Precise: affects InEKF only. ISRCKF measurement noise (R_diag) and ISRCKF geometry
    /// initialization remain unchanged unless you modify code in tracker.hpp.
    RobotInEKFConfig robot_inekf;
    /// InEKF parameters for outpost targets.
    /// Precise: affects InEKF only; outpost radius is fixed in the model.
    OutpostInEKFConfig outpost_inekf;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// ============================================================================
// L3 Estimation Full Config
// ============================================================================

struct L3Config {
    TrackerConfig tracker{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const TrackerConfig> tracker_ptr() const {
        return std::make_shared<const TrackerConfig>(tracker);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L3
