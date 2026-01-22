#pragma once

#include <expected>
#include <string>

// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L3 {

// ----------------------------------------------------------------------------
// InEKF config notes
// ----------------------------------------------------------------------------
//
// Problem:
//   The ISRCKF is a practical Euclidean-state filter, but it is relatively loose about pose-group
//   structure and geometry consistency. The InEKF adds a group-consistent (SE2(3)) state and
//   explicit geometry states (r0/r1/h or dz offsets).
//
// Intuition:
//   - Process noise (sigma_a_*, sigma_yaw, sigma_r, sigma_h) says "how much the target dynamics /
//     geometry can deviate from our simple model".
//   - Measurement noise (sigma_m, sigma_w) says "how much we trust the observations".
//
// Analogy:
//   Think of it as two knobs: "model wiggle room" (process noise) vs "sensor credibility"
//   (measurement noise).
//
// Precise (implementation facts):
//   - State: SE2(3) (R,v,p) with yaw-only rotation; roll/pitch are hard-constrained to 0 in
//   updates.
//   - Extra-state:
//       * Robot (4 armors):  [v_yaw, r0, r1, h]
//       * Outpost (3 armors): [v_yaw, dz1, dz2]  (z offsets relative to p.z for armor 1/2)
//   - update_armor(y, armor_id): uses Rm = I * sigma_m^2 (m^2).
//   - update_yaw(yaw_meas): uses R = sigma_w^2 (rad^2), where yaw_meas is derived from
//     (armor_yaw_meas - yaw_offset) and therefore depends on correct armor_id association.
//   - Naming pitfalls:
//       * sigma_yaw is yaw *process* noise (angular-acceleration std-dev), not a measurement noise.
//       * sigma_w is yaw *measurement* noise (std-dev in rad); "w" is historical and NOT yaw-rate.
//       * sigma_h means "height drift" for robots, but "dz drift" for outposts.
//   - Config parsing uses READ(...): keys are required in TOML (defaults won't apply if missing).

/// InEKF configuration parameters for 4-armor robot targets.
struct RobotInEKFConfig {
    /// XY acceleration process-noise std-dev (≈ m/s^2).
    /// Problem: constant-velocity dynamics cannot explain real acceleration.
    /// Intuition: larger → more responsive; smaller → smoother but laggier.
    /// Analogy: letting the model "wiggle" more.
    /// Precise: used in InEKF::process_noise_covariance() to build CV blocks for x/y (dt^2..dt^4).
    double sigma_a_xy{5.0};
    /// Z acceleration process-noise std-dev (≈ m/s^2).
    /// Problem/Intuition/Analogy: same as sigma_a_xy, but for z.
    /// Precise: used in InEKF::process_noise_covariance() to build CV block for z (dt^2..dt^4).
    double sigma_a_z{1.0};
    /// Yaw angular-acceleration process-noise std-dev (≈ rad/s^2) for [yaw, v_yaw] CV model.
    /// Problem: yaw can accelerate; constant yaw-rate is an approximation.
    /// Intuition: larger → follow fast spins; smaller → smoother yaw but more lag.
    /// Analogy: "turn-rate agility" knob.
    /// Precise: this is *process noise* (not measurement noise). Used in
    /// process_noise_covariance().
    double sigma_yaw{20.0};

    /// Enable coordinated-turn kinematics in prediction.
    ///
    /// Problem:
    ///   CV mean prediction moves in a straight line even when the target is turning, which
    ///   increases lag around turning points.
    ///
    /// Intuition:
    ///   When v_yaw != 0, rotate velocity by yaw increment and integrate position along a circular
    ///   arc (coordinated turn). This tightens tracking for turning targets without increasing
    ///   state dimension.
    bool use_coordinated_turn{true};

    /// Enable slip-gated coupling between yaw-rate and translation.
    ///
    /// Problem:
    ///   In many games/simulations, targets can translate in an arbitrary direction while yaw
    ///   rotates independently ("cat-walk"/strafing). In that case, coupling translation to v_yaw
    ///   (coordinated turn) is wrong and can hurt tracking.
    ///
    /// Solution:
    ///   Estimate a slip angle between the velocity heading and yaw. Only apply coordinated-turn
    ///   integration when heading ≈ yaw; otherwise fall back to straight-line CV for translation.
    bool enable_slip_gated_turn{true};

    /// Enable adaptive scaling of process noise based on innovation magnitude.
    ///
    /// Problem:
    ///   Fixed Q is a compromise: small Q is stable but laggy during maneuvers; large Q follows
    ///   maneuvers but becomes noisy in steady state.
    ///
    /// Intuition:
    ///   Increase Q temporarily when innovations are large (maneuver), then decay back to 1.
    bool enable_adaptive_process_noise{true};

    /// Enable pseudo-velocity updates derived from consecutive spherical armor observations.
    ///
    /// Problem:
    ///   With pose-only observations, velocity is an inferred first-derivative state and can lag
    ///   during fast maneuvers (snake, stop-go, reverse).
    ///
    /// Solution:
    ///   Convert spherical armor observations to a target-center position measurement, then use a
    ///   finite-difference velocity measurement over a short interval to directly update v.
    bool enable_pseudo_velocity_update{true};

    /// Enable soft robust weighting (downweight extreme outliers instead of skipping updates).
    bool enable_robust_update{true};
    /// Geometry radius random-walk noise std-dev (≈ m/√s), shared for r0/r1.
    /// Problem: geometry is not perfectly known and can drift (mounting, calibration bias).
    /// Intuition: larger → geometry adapts faster but can drift; smaller → geometry stays stable.
    /// Analogy: how quickly you allow the "shape" to change.
    /// Precise: enters InEKFModelTraits::add_geometry_process_noise() as sigma_r^2 * dt.
    double sigma_r{1.0};
    /// Geometry height random-walk noise std-dev (≈ m/√s), for h.
    /// Problem/Intuition/Analogy: same as sigma_r, but for height offset h.
    /// Precise: enters add_geometry_process_noise() as sigma_h^2 * dt.
    double sigma_h{0.5};

    /// Armor 3D position measurement noise std-dev (m).
    /// Problem: PnP+TF position has noise/spikes; blindly trusting it makes the estimate jitter.
    /// Intuition: smaller → track measurements tightly; larger → smooth more / trust the model
    /// more. Analogy: measurement "confidence" slider. Precise: used as Rm = I * sigma_m^2 in
    /// InEKF::update() and InEKF::update_armor().
    double sigma_m{0.01};
    /// Yaw measurement noise std-dev (rad) for update_yaw().
    /// Problem: yaw observation can jump, especially if armor_id association is wrong for a frame.
    /// Intuition: smaller → yaw follows measurements (responsive but jumpy); larger → yaw is
    /// stable. Analogy: "how much you believe the compass reading". Precise: used as R = sigma_w^2
    /// in InEKF::update_yaw(). The measurement is derived from armor_yaw_meas - yaw_offset, so
    /// correct armor_id is a prerequisite.
    double sigma_w{0.25};

    /// Enable dynamic anisotropic measurement noise (from motion_model.hpp insights)
    ///
    /// Problem:
    ///   Isotropic noise R = σ²I ignores that visual measurements have direction-dependent
    ///   uncertainty. Depth is unreliable when viewing armor from the side; angular position
    ///   is unreliable at long range due to PnP instability.
    ///
    /// Intuition:
    ///   - Radial (depth) uncertainty ↑ when armor surface is nearly parallel to sight line
    ///   - Tangential (angular) uncertainty ↑ when target is far away
    ///
    /// Analogy:
    ///   Like a flashlight cone - the uncertainty is anisotropic along the viewing direction.
    ///
    /// Precise:
    ///   When true, R becomes a 3×3 SPD matrix:
    ///     R = U × diag(σ_r², σ_t², σ_t²) × U^T
    ///   where U = [e_r, e_h, e_v] (radial/tangential basis in body frame)
    ///   σ_r² = σ_m² × (1 + k_r × log(sin(view_angle) + 1))
    ///   σ_t² = σ_m² × (1 + k_t × log(distance + 1))
    bool use_dynamic_noise{true};

    /// Radial (depth) noise sensitivity to viewing angle (k_r)
    ///
    /// Problem: How much should depth uncertainty increase when viewing armor from the side?
    /// Intuition: Larger → more penalty for oblique views; smaller → more uniform depth noise.
    /// Precise: σ_r² = σ_m² × (1 + k_r × log(sin(view_angle) + 1)), k_r ∈ [0, ∞)
    /// Default 1.0 matches motion_model.hpp behavior.
    double dynamic_noise_k_radial{1.0};

    /// Tangential (angular) noise sensitivity to distance (k_t)
    ///
    /// Problem: How much should angular uncertainty increase with distance?
    /// Intuition: Larger → more penalty for far targets; smaller → more uniform angular noise.
    /// Precise: σ_t² = σ_m² × (1 + k_t × log(distance + 1)), k_t ∈ [0, ∞)
    /// Default 0.1 matches motion_model.hpp behavior (was 1/10).
    double dynamic_noise_k_tangent{0.1};

    /// Target geometry initial values (center -> armor offsets)
    /// Initial radius for armors 0/2 (m).
    /// Problem: we need a reasonable starting geometry to avoid early instability.
    /// Intuition: closer to reality → faster convergence; far off → early jitter and wrong
    /// residuals. Analogy: initial guess for an optimizer. Precise: used only at initialization
    /// (InEKFModelTraits::init_extra), then estimated with process noise + clamp constraints.
    double radius0{0.23};
    /// Initial radius for armors 1/3 (m).
    /// Same semantics as radius0, but for the (1,3) armor pair.
    double radius1{0.23};
    /// Initial height offset h = z1 - z0 for armors 1/3 (m).
    /// Same semantics as radius0, but for the height offset h.
    /// Precise: h is clamped (see InEKFModelTraits) and is allowed to be negative; a negative h
    /// effectively swaps which armor pair is "higher" in the model.
    double height{0.0};

    // -------------------------------------------------------------------------
    // Spherical observation mode (ported from sp_vision_25)
    // -------------------------------------------------------------------------
    //
    // Problem:
    //   Cartesian observations [x,y,z] with large sigma_m cause tracking lag for fast-moving
    //   targets. The filter doesn't trust measurements enough to respond quickly to direction
    //   changes.
    //
    // Solution:
    //   Use spherical observations [yaw, pitch, distance, armor_yaw] with small angular noise.
    //   This matches PnP output characteristics: angles are accurate, depth is noisy.
    //
    // Precise:
    //   When use_spherical_observation=true, update_armor_spherical() is called instead of
    //   update_armor(). The observation model uses:
    //     z = [yaw, pitch, distance, armor_yaw]  (4D)
    //     R = diag(σ_yaw², σ_pitch², R_dist_dynamic, σ_armor_yaw²)
    //   where R_dist_dynamic = log(|delta_angle| + 1) + 1 (view-angle-dependent).

    /// Enable spherical coordinate observation model.
    /// Problem: Cartesian observations don't match PnP output characteristics.
    /// Intuition: Set true for faster response to direction changes (esp. translation).
    /// Default true: spherical is generally better for tracking moving targets.
    bool use_spherical_observation{true};

    /// Yaw observation noise std-dev (rad) for spherical model.
    /// Problem: PnP yaw angle has sub-degree accuracy for frontal views.
    /// Intuition: Smaller → faster response to horizontal movement.
    /// Precise: R(0,0) = sigma_yaw_obs² in spherical measurement noise.
    /// Default 0.063 rad ≈ 3.6° (conservative; sp_vision uses 0.063 which is sqrt(4e-3)).
    double sigma_yaw_obs{0.063};

    /// Pitch observation noise std-dev (rad) for spherical model.
    /// Problem/Intuition/Precise: Same as sigma_yaw_obs but for vertical angle.
    double sigma_pitch_obs{0.063};

    /// Armor orientation (yaw) observation noise std-dev (rad) for spherical model.
    /// Problem: Armor plate orientation is less accurate than position angle, especially at
    /// distance. Intuition: Larger → less sensitive to armor orientation jumps. Precise: Base
    /// noise; actual R(3,3) = (log(dist+1)/200 + sigma_armor_yaw)².
    double sigma_armor_yaw{0.3};

    // -------------------------------------------------------------------------
    // Innovation gating (outlier rejection)
    // -------------------------------------------------------------------------
    //
    // Problem:
    //   Large innovations (measurement - prediction) can occur due to:
    //   - Wrong armor_id association
    //   - PnP outliers / detection glitches
    //   - Rapid direction changes at turning points
    //
    // Trade-off:
    //   - Smaller gate → more robust to outliers, but may skip valid updates during fast maneuvers
    //   - Larger gate → better tracking during fast maneuvers, but more sensitive to outliers
    //
    // Precise:
    //   If |innovation| > max_*_innovation, the entire spherical update is skipped.
    //   Set to large values (e.g., 3.14) to effectively disable gating.

    /// Maximum allowed angle innovation (rad) before rejecting the measurement.
    /// Applies to yaw, pitch, and armor_yaw innovations in spherical mode.
    /// Default 1.5 rad (~86°) allows fast maneuvers; use 0.5 (~30°) for more outlier rejection.
    double max_angle_innovation{1.5};

    /// Maximum allowed distance innovation (m) before rejecting the measurement.
    /// Default 3.0m; set larger if target can have large depth jumps.
    double max_dist_innovation{3.0};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

/// InEKF configuration parameters for 3-armor outpost targets.
struct OutpostInEKFConfig {
    /// XY acceleration process-noise std-dev (≈ m/s^2).
    /// Problem/Intuition/Analogy/Precise: same as RobotInEKFConfig::sigma_a_xy.
    double sigma_a_xy{5.0};
    /// Z acceleration process-noise std-dev (≈ m/s^2).
    /// Problem/Intuition/Analogy/Precise: same as RobotInEKFConfig::sigma_a_z.
    double sigma_a_z{1.0};
    /// Yaw angular-acceleration process-noise std-dev (≈ rad/s^2).
    /// Problem/Intuition/Analogy/Precise: same as RobotInEKFConfig::sigma_yaw.
    double sigma_yaw{20.0};

    /// Enable coordinated-turn kinematics in prediction (same meaning as RobotInEKFConfig).
    bool use_coordinated_turn{true};

    /// Enable slip-gated coupling between yaw-rate and translation (same meaning as
    /// RobotInEKFConfig).
    bool enable_slip_gated_turn{true};

    /// Enable adaptive scaling of process noise based on innovation magnitude.
    bool enable_adaptive_process_noise{true};

    /// Enable pseudo-velocity updates derived from consecutive spherical armor observations.
    bool enable_pseudo_velocity_update{true};

    /// Enable soft robust weighting (downweight extreme outliers instead of skipping updates).
    bool enable_robust_update{true};
    /// dz drift noise std-dev (≈ m/√s) for dz1/dz2 random walk.
    /// Problem: outpost radius is fixed; only relative armor heights (dz1/dz2) are modeled.
    /// Intuition: larger → dz adapts faster but can drift; smaller → dz stays stable.
    /// Analogy: how quickly you allow the "vertical layout" to change.
    /// Precise: enters Outpost InEKF geometry process noise as sigma_h^2 * dt.
    double sigma_h{0.5};

    /// Measurement noise
    /// Armor position measurement noise std-dev (m).
    /// Problem/Intuition/Analogy/Precise: same as RobotInEKFConfig::sigma_m.
    double sigma_m{0.01};
    /// Direct yaw measurement noise std-dev (rad).
    /// Problem/Intuition/Analogy/Precise: same as RobotInEKFConfig::sigma_w.
    double sigma_w{0.25};

    /// Enable dynamic anisotropic measurement noise.
    /// Problem/Intuition/Analogy/Precise: same as RobotInEKFConfig::use_dynamic_noise.
    bool use_dynamic_noise{false};

    /// Radial noise sensitivity to viewing angle.
    /// Precise: same as RobotInEKFConfig::dynamic_noise_k_radial.
    double dynamic_noise_k_radial{1.0};

    /// Tangential noise sensitivity to distance.
    /// Precise: same as RobotInEKFConfig::dynamic_noise_k_tangent.
    double dynamic_noise_k_tangent{0.1};

    // -------------------------------------------------------------------------
    // Spherical observation mode
    // -------------------------------------------------------------------------

    /// Enable spherical coordinate observation model.
    /// Precise: same as RobotInEKFConfig::use_spherical_observation.
    bool use_spherical_observation{true};

    /// Yaw observation noise std-dev (rad) for spherical model.
    /// Precise: same as RobotInEKFConfig::sigma_yaw_obs.
    double sigma_yaw_obs{0.063};

    /// Pitch observation noise std-dev (rad) for spherical model.
    /// Precise: same as RobotInEKFConfig::sigma_pitch_obs.
    double sigma_pitch_obs{0.063};

    /// Armor orientation (yaw) observation noise std-dev (rad) for spherical model.
    /// Precise: same as RobotInEKFConfig::sigma_armor_yaw.
    double sigma_armor_yaw{0.3};

    // -------------------------------------------------------------------------
    // Innovation gating (outlier rejection)
    // -------------------------------------------------------------------------

    /// Maximum allowed angle innovation (rad) before rejecting the measurement.
    /// Precise: same as RobotInEKFConfig::max_angle_innovation.
    double max_angle_innovation{1.5};

    /// Maximum allowed distance innovation (m) before rejecting the measurement.
    /// Precise: same as RobotInEKFConfig::max_dist_innovation.
    double max_dist_innovation{3.0};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L3
