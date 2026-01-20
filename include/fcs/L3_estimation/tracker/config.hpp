#pragma once

#include "inekf_config.hpp"
#include "motion_model_params.hpp"
#include <expected>
#include <memory>
#include <optional>
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

// ============================================================================
// Jpda Matcher Configuration
// ============================================================================

/// JpdaMatcher configuration for advanced phase-permutation JPDA.
///
/// Problem:
///   The standard DataAssociator uses greedy one-to-one matching with Mahalanobis distance gating.
///   This can fail in ambiguous symmetric cases where multiple armors have similar likelihoods.
///
/// Intuition:
///   JpdaMatcher enumerates all 4! = 24 permutations, computes exact set likelihood with
///   log|S| term, and uses implicit phase space to handle yaw wrapping.
///
/// Analogy:
///   Instead of picking the "best-looking" match immediately, JpdaMatcher considers ALL
///   possible matchings and picks the one with highest joint probability.
///
/// Precise:
///   - Enable via `enable_jpda_matcher = true` in TOML
///   - Falls back to DataAssociator when disabled (default behavior)
///   - Phase ambiguity threshold: (max - second) / max < threshold → ambiguous
///   - Semantic fusion: breaks symmetry using detector confidence and class consistency
///   - Viterbi smoothing: 3-frame sliding window to eliminate phase jumps
struct JpdaMatcherConfig {
    /// Enable JpdaMatcher (vs standard DataAssociator).
    ///
    /// Problem:
    ///   JpdaMatcher is more computationally expensive (~600-900μs vs ~100μs) and may not
    ///   always be necessary.
    ///
    /// Intuition:
    ///   Enable this when you frequently encounter symmetric ambiguous cases (fast rotation,
    ///   partial occlusion, high noise).
    ///
    /// Analogy:
    ///   Like upgrading from a "quick heuristic" to an "exact algorithm" when the heuristic
    ///   starts failing.
    ///
    /// Precise:
    ///   When true, Tracker uses JpdaMatcher for association. When false, uses the legacy
    ///   DataAssociator (greedy matching with Mahalanobis gating).
    bool enable_jpda_matcher{false};

    /// Phase ambiguity detection threshold.
    ///
    /// Problem:
    ///   We need to detect when the phase estimate is ambiguous (top-2 phases are close) to
    ///   decide whether to run multi-hypothesis fusion.
    ///
    /// Intuition:
    ///   Smaller → more sensitive to ambiguity (runs multi-hypothesis more often).
    ///   Larger → more tolerant (only runs multi-hypothesis in very ambiguous cases).
    ///
    /// Analogy:
    ///   Like a "confidence margin" - if the top two candidates are within this margin,
    ///   we consider it ambiguous.
    ///
    /// Precise:
    ///   Computed as (max_prob - second_prob) / max_prob < threshold → ambiguous.
    ///   Default 0.3 means if second is within 30% of max, trigger multi-hypothesis.
    double phase_ambiguous_threshold{0.3};

    /// Phase switch penalty for Viterbi smoothing.
    ///
    /// Problem:
    ///   Without temporal smoothing, the phase estimate can jump between frames due to
    ///   measurement noise, causing target state to flicker.
    ///
    /// Intuition:
    ///   Larger → prefer stable phase (resist jumps). Smaller → allow faster phase changes.
    ///
    /// Analogy:
    ///   Like a "hysteresis" or "momentum" term in a control system.
    ///
    /// Precise:
    ///   In Viterbi DP recurrence: cost += -lambda * |phase - prev_phase|.
    ///   Typical range: 0.05 (very loose) to 0.5 (very strict). Default 0.1 is moderate.
    double phase_switch_lambda{0.1};

    /// Semantic reweighting lambda (λ_sem).
    ///
    /// Problem:
    ///   Semantic information may be more or less reliable than geometric information,
    ///   depending on detector quality and operating conditions.
    ///
    /// Intuition:
    ///   Larger → trust semantics more (use class predictions to break geometric ambiguity).
    ///   Smaller → trust geometry more (semantics as tie-breaker only).
    ///
    /// Analogy:
    ///   Like a "confidence weight" for semantic vs geometric likelihood.
    ///
    /// Precise:
    ///   In semantic reweighting: log L_total = log L_geometry + λ_sem * log L_semantic.
    ///   Default 1.0 means equal weight. Set < 1.0 for noisy detectors, > 1.0 for reliable ones.
    double semantic_reweight_lambda{1.0};

    /// Enable semantic fusion (detector confidence + class consistency).
    ///
    /// Problem:
    ///   Pure geometric matching can be ambiguous in symmetric cases. Semantic information
    ///   (armor color/class) can break this symmetry.
    ///
    /// Intuition:
    ///   Enable this when your detector provides reliable class predictions (color/type).
    ///
    /// Analogy:
    ///   Like using "landmark features" in addition to "distance" for matching.
    ///
    /// Precise:
    ///   When true, refines permutation likelihoods using log P(detected | true) from
    ///   SemanticContext class transition matrix.
    bool enable_semantic{false};

    /// Enable Viterbi smoothing (3-frame sliding window).
    ///
    /// Problem:
    ///   Phase estimates can jump between frames due to noise, causing tracker instability.
    ///
    /// Intuition:
    ///   Enable this to smooth out phase transitions and eliminate single-frame outliers.
    ///
    /// Analogy:
    ///   Like a "moving average" but for discrete states with DP.
    ///
    /// Precise:
    ///   When true, runs ViterbiSmoother on phase probabilities and uses smoothed phase
    ///   for final output.
    bool enable_viterbi{true};

    /// Enable geometric consistency constraints.
    ///
    /// Problem:
    ///   Phase ambiguity occurs because 4 armors look geometrically similar in measurement space.
    ///   The filter state contains geometry estimates (r0, r1, h) that could discriminate.
    ///
    /// Intuition:
    ///   Each permutation implies a specific geometric configuration. We can compute the implied
    ///   geometry from observations and compare against the state estimate geometry to add an
    ///   additional term to the likelihood.
    ///
    /// Analogy:
    ///   Like using "shape matching" in addition to "position matching" for association.
    ///
    /// Precise:
    ///   When true, adds geometry error term to permutation likelihood:
    ///   log L_geo(π) = -0.5 * Σ (geometry_error(π)² / σ_geo²)
    bool enable_geometry{true};

    /// Weight for geometry consistency term in likelihood.
    ///
    /// Problem:
    ///   Geometry estimates may be inaccurate early in tracking, so we need to balance
    ///   geometry constraint against measurement innovation.
    ///
    /// Intuition:
    ///   Larger → trust geometry more (stronger constraint). Smaller → trust measurements more.
    ///
    /// Analogy:
    ///   Like a "confidence weight" for geometric vs measurement likelihood.
    ///
    /// Precise:
    ///   Multiplier for geometry penalty in log-likelihood:
    ///   log L += geometry_weight * (-0.5 * geo_penalty)
    double geometry_weight{1.0};

    /// Expected uncertainty in radius estimate (meters).
    ///
    /// Problem:
    ///   Radius estimates have uncertainty that varies with tracking phase.
    ///
    /// Intuition:
    ///   Larger → more tolerant of radius mismatch. Smaller → stricter radius matching.
    ///
    /// Precise:
    ///   Standard deviation for radius error: (r_implied - r_expected)² / σ_r²
    double sigma_radius{0.02};

    /// Expected uncertainty in height estimate (meters).
    ///
    /// Problem:
    ///   Height estimates have uncertainty that varies with tracking phase.
    ///
    /// Intuition:
    ///   Larger → more tolerant of height mismatch. Smaller → stricter height matching.
    ///
    /// Precise:
    ///   Standard deviation for height error: (z_obs - z_expected)² / σ_h²
    double sigma_height{0.03};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

struct TrackerConfig {
    /// Robot target parameters (4-armors).
    /// Problem: robots have 4 armors and geometry parameters (r0/r1/h) to estimate.
    TargetConfig<RobotModelParams> robot;
    /// Outpost target parameters (3-armors).
    /// Problem: outposts have 3 armors and a different (simpler) state/geometry.
    TargetConfig<OutpostModelParams> outpost;

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
    /// JpdaMatcher configuration for advanced phase-permutation JPDA.
    /// Precise: when enable_jpda_matcher=true, uses JpdaMatcher instead of
    /// DataAssociator for armor matching. Falls back to DataAssociator when disabled.
    JpdaMatcherConfig jpda_matcher;

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
