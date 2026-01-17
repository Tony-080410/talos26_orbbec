#pragma once

#include <expected>
#include <numbers>
#include <string>

#include "fcs/L4_planning/trajectory_config.hpp"

// Forward declaration

namespace toml::inline v3 {
class table;
}

namespace fcs::L4 {

// ============================================================================
// MPC Planner Configuration
// ============================================================================

/// MPC planner configuration
///
/// This planner uses TinyMPC (ADMM-based QP solver) to compute optimal
/// gimbal trajectories. It models the gimbal as a double integrator and
/// solves a trajectory optimization problem over a finite horizon.
///
/// The MPC formulation:
///   min  Σ (x - x_ref)ᵀ Q (x - x_ref) + uᵀ R u
///   s.t. x_{k+1} = A x_k + B u_k
///        u_min <= u <= u_max
///
/// where:
///   x = [angle, angular_velocity]  (2D state)
///   u = [angular_acceleration]     (1D control)
///   A, B are double integrator matrices with dt = 0.01s (100Hz)
struct MpcPlannerConfig {
    // ========================================================================
    // MPC Cost Weights
    // ========================================================================

    /// Yaw position tracking weight (Q matrix diagonal element 0).
    ///
    /// Problem:
    ///   The MPC cost balances tracking error and control effort. This weight
    ///   determines how aggressively the gimbal tracks the reference yaw.
    ///
    /// Intuition:
    ///   Larger -> track yaw more tightly but may use more control effort;
    ///   Smaller -> smoother control but worse yaw tracking.
    ///
    /// Precise:
    ///   Used in Q = diag(q_yaw_position, q_yaw_velocity) for yaw axis.
    double q_yaw_position{9e6};

    /// Yaw velocity tracking weight (Q matrix diagonal element 1).
    ///
    /// Problem:
    ///   Controls how much the MPC cares about matching the reference velocity.
    ///
    /// Intuition:
    ///   Set to 0 typically means "don't care about velocity, only position".
    ///   Non-zero values can help smooth transitions but may slow tracking.
    double q_yaw_velocity{0.0};

    /// Yaw control effort weight (R matrix).
    ///
    /// Problem:
    ///   Penalizes large accelerations to achieve smooth gimbal motion.
    ///
    /// Intuition:
    ///   Larger -> smoother but slower; Smaller -> more aggressive control.
    double r_yaw{1.0};

    /// Maximum yaw acceleration (rad/s^2).
    ///
    /// Problem:
    ///   Gimbal motors have limited torque/acceleration capability.
    ///
    /// Precise:
    ///   Used as box constraint: -max_yaw_acc <= u <= max_yaw_acc
    double max_yaw_acc{50.0};

    /// Pitch position tracking weight (Q matrix diagonal element 0).
    ///
    /// Problem:
    ///   The MPC cost balances tracking error and control effort. This weight
    ///   determines how aggressively the gimbal tracks the reference pitch.
    ///
    /// Intuition:
    ///   Larger -> track pitch more tightly but may use more control effort;
    ///   Smaller -> smoother control but worse pitch tracking.
    ///
    /// Precise:
    ///   Used in Q = diag(q_pitch_position, q_pitch_velocity) for pitch axis.
    double q_pitch_position{9e6};

    /// Pitch velocity tracking weight (Q matrix diagonal element 1).
    ///
    /// Problem:
    ///   Controls how much the MPC cares about matching the reference velocity.
    ///
    /// Intuition:
    ///   Set to 0 typically means "don't care about velocity, only position".
    ///   Non-zero values can help smooth transitions but may slow tracking.
    double q_pitch_velocity{0.0};

    /// Pitch control effort weight (R matrix).
    ///
    /// Problem:
    ///   Penalizes large accelerations to achieve smooth gimbal motion.
    ///
    /// Intuition:
    ///   Larger -> smoother but slower; Smaller -> more aggressive control.
    double r_pitch{1.0};

    /// Maximum pitch acceleration (rad/s^2).
    ///
    /// Problem:
    ///   Gimbal pitch motors have limited torque/acceleration capability.
    ///   Pitch axis typically has higher acceleration limits than yaw due to
    ///   smaller inertia.
    ///
    /// Precise:
    ///   Used as box constraint: -max_pitch_acc <= u <= max_pitch_acc
    double max_pitch_acc{100.0};

    /// Minimum pitch angle (radians, negative = looking up).
    ///
    /// Problem:
    ///   Gimbal has mechanical limits on how far it can tilt upward.
    ///   Exceeding this limit could damage the gimbal or cause collisions.
    ///
    /// Intuition:
    ///   More negative -> can aim higher targets;
    ///   Typical range: -0.3 to -0.6 rad (-17° to -34°).
    ///
    /// Precise:
    ///   Output pitch is clamped: pitch = clamp(mpc_output, pitch_min, pitch_max).
    ///   Sign convention: negative pitch = gimbal tilted upward (barrel up).
    double pitch_min{-0.5}; // ~-29°

    /// Maximum pitch angle (radians, positive = looking down).
    ///
    /// Problem:
    ///   Gimbal has mechanical limits on how far it can tilt downward.
    ///   Exceeding this limit could cause the barrel to hit the chassis.
    ///
    /// Intuition:
    ///   More positive -> can aim lower targets;
    ///   Typical range: 0.2 to 0.4 rad (11° to 23°).
    ///
    /// Precise:
    ///   Output pitch is clamped: pitch = clamp(mpc_output, pitch_min, pitch_max).
    ///   Sign convention: positive pitch = gimbal tilted downward (barrel down).
    double pitch_max{0.3};                        // ~+17°

    double fire_thresh{180.0 / std::numbers::pi}; // ~0.17 degrees
    double shooting_range_h{0.135};
    double shooting_range_w{0.135};

    /// Fire decision lookahead (steps into the future).
    ///
    /// Problem:
    ///   The current tracking error may not reflect near-future accuracy.
    ///
    /// Intuition:
    ///   Looking ahead accounts for gimbal inertia and predicts whether
    ///   the gimbal will be on target when the bullet arrives.
    int fire_lookahead{2};

    // ========================================================================
    // Solver Parameters
    // ========================================================================

    /// Maximum ADMM iterations.
    ///
    /// Problem:
    ///   ADMM converges iteratively; more iterations improve accuracy but
    ///   increase latency.
    ///
    /// Precise:
    ///   TinyMPC typically converges in 5-10 iterations for this problem.
    int max_iterations{10};

    /// Absolute tolerance for primal/dual residuals.
    ///
    /// Problem:
    ///   Termination criterion for ADMM convergence.
    double abs_tol{1e-3};

    /// ADMM penalty parameter (rho).
    ///
    /// Problem:
    ///   Affects convergence speed and numerical conditioning.
    ///
    /// Intuition:
    ///   Larger -> faster constraint satisfaction but may oscillate;
    ///   Smaller -> smoother but slower convergence.
    double rho{1.0};

    /// MPC timestep (seconds).
    ///
    /// Precise:
    ///   Used to construct state transition matrices A, B.
    ///   Smaller dt -> more accurate but more steps needed for same horizon.
    double ponder_dt{0.01}; // 100Hz

    int ponder_ahead{100};

    int ponder_back{50};

    // ========================================================================
    // Trajectory Configuration (reused from GimbalPlanner)
    // ========================================================================

    /// Trajectory compensator configuration for ballistic calculations.
    TrajectoryConfig trajectory;

    /// Compensator type for ballistic calculations.
    TrajectoryCompensatorType compensator_type = TrajectoryCompensatorType::RK4;

    // ========================================================================
    // Operational Parameters
    // ========================================================================

    // ========================================================================
    // Adaptive Delay Compensation
    // ========================================================================

    /// Angular velocity threshold for high/low speed classification (rad/s).
    ///
    /// Problem:
    ///   High-speed rotating targets have different system delay characteristics
    ///   compared to slow-moving or stationary targets.
    ///
    /// Intuition:
    ///   Targets spinning faster than this threshold use high_speed_delay;
    ///   slower targets use low_speed_delay.
    ///
    /// Precise:
    ///   delay = |v_yaw| > decision_speed ? high_speed_delay : low_speed_delay
    double decision_speed{8.0};

    /// System delay compensation for high-speed targets (seconds).
    ///
    /// Problem:
    ///   High-speed rotating targets require different lead time compensation
    ///   due to their rapid position changes during system latency.
    ///
    /// Intuition:
    ///   Includes: camera exposure, image processing, communication, gimbal response.
    ///   Typical range: 0.01 to 0.03 seconds.
    double high_speed_delay{0.015};

    /// System delay compensation for low-speed targets (seconds).
    ///
    /// Problem:
    ///   Slow or stationary targets are less affected by system delay,
    ///   but still need compensation for accurate tracking.
    ///
    /// Intuition:
    ///   Can be tuned independently from high_speed_delay.
    ///   Often set to the same value for simplicity.
    double low_speed_delay{0.015};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
