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
    /// Larger -> track yaw more tightly but may use more control effort.
    double q_yaw_position{9e6};

    /// Yaw velocity tracking weight (Q matrix diagonal element 1).
    /// Set to 0 means "don't care about velocity, only position".
    double q_yaw_velocity{0.0};

    /// Yaw control effort weight (R matrix).
    /// Larger -> smoother but slower; Smaller -> more aggressive control.
    double r_yaw{1.0};

    /// Maximum yaw acceleration (rad/s^2).
    /// Used as box constraint: -max_yaw_acc <= u <= max_yaw_acc
    double max_yaw_acc{50.0};

    /// Pitch position tracking weight (Q matrix diagonal element 0).
    /// Larger -> track pitch more tightly but may use more control effort.
    double q_pitch_position{9e6};

    /// Pitch velocity tracking weight (Q matrix diagonal element 1).
    double q_pitch_velocity{0.0};

    /// Pitch control effort weight (R matrix).
    double r_pitch{1.0};

    /// Maximum pitch acceleration (rad/s^2).
    double max_pitch_acc{100.0};

    /// Minimum pitch angle (radians, negative = looking up).
    /// Output pitch is clamped: pitch = clamp(mpc_output, pitch_min, pitch_max).
    double pitch_min{-0.5}; // ~-29°

    /// Maximum pitch angle (radians, positive = looking down).
    double pitch_max{0.3};                        // ~+17°

    double fire_thresh{180.0 / std::numbers::pi}; // ~0.17 degrees
    double shooting_range_h{0.135};
    double shooting_range_w{0.135};

    /// Fire decision lookahead (steps into the future).
    int fire_lookahead{2};

    // ========================================================================
    // Solver Parameters
    // ========================================================================

    /// Maximum ADMM iterations. TinyMPC typically converges in 5-10 iterations.
    int max_iterations{10};

    /// Absolute tolerance for primal/dual residuals.
    double abs_tol{1e-3};

    /// ADMM penalty parameter (rho).
    /// Larger -> faster constraint satisfaction but may oscillate.
    double rho{1.0};

    /// MPC timestep (seconds).
    /// Used to construct state transition matrices A, B.
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

    /// Enable/disable MPC optimization.
    /// When disabled, the system directly uses aimer output without trajectory optimization.
    bool enabled{true};

    // ========================================================================
    // Adaptive Delay Compensation
    // ========================================================================

    /// Angular velocity threshold for high/low speed classification (rad/s).
    /// delay = |v_yaw| > decision_speed ? high_speed_delay : low_speed_delay
    double decision_speed{8.0};

    /// System delay compensation for high-speed targets (seconds).
    /// Includes: camera exposure, image processing, communication, gimbal response.
    double high_speed_delay{0.015};

    /// System delay compensation for low-speed targets (seconds).
    double low_speed_delay{0.015};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
