#pragma once

#include <expected>

#include "fcs/L4_planning/trajectory_config.hpp"

// Forward declaration
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

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
    double pitch_max{0.3}; // ~+17°

    // ========================================================================
    // Fire Decision Parameters
    // ========================================================================

    /// Fire threshold (radians).
    ///
    /// Problem:
    ///   Need to determine when the gimbal is close enough to the target
    ///   trajectory to permit firing.
    ///
    /// Intuition:
    ///   Smaller -> more accurate but fires less often;
    ///   Larger -> fires more often but may miss.
    ///
    /// Precise:
    ///   Fire is advised when ||planned_trajectory - reference|| < fire_thresh
    ///   at the lookahead step.
    double fire_thresh{0.003}; // ~0.17 degrees

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
    double dt{0.01}; // 100Hz

    /// Prediction horizon (number of steps).
    ///
    /// Precise:
    ///   Total prediction time = horizon * dt.
    ///   With dt=0.01 and horizon=100, this gives 1 second lookahead.
    int horizon{100};

    /// Index in horizon representing "current time" for control extraction.
    ///
    /// Problem:
    ///   The reference trajectory spans [-T/2, +T/2] relative to now.
    ///   The control to apply is at the "middle" of this trajectory.
    ///
    /// Precise:
    ///   With horizon=100 and half_horizon=50, we extract u[50] and x[50]
    ///   as the current control and state.
    int half_horizon{50};

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

    /// Extra latency from planning to bullet leaving muzzle (seconds).
    ///
    /// Problem:
    ///   Same as GimbalPlannerConfig::plan_to_muzzle_delay.
    double plan_to_muzzle_delay{0.0};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
