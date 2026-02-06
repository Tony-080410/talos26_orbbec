#pragma once

#include "fcs/L4_planning/trajectory_config.hpp"

#include <Eigen/Core>

#include <memory>
#include <utility>
#include <vector>

namespace fcs::L4::trajectory {

/// Complete ballistic solution result
///
/// Value object representing the complete result of solving for a ballistic
/// trajectory to hit a target. Replaces the old output-parameter pattern.
struct AimSolution {
    double yaw{0.0};            ///< Firing azimuth angle (radians)
    double pitch{0.0};          ///< Firing elevation angle (radians)
    double time_of_flight{0.0}; ///< Time of flight (seconds)
    bool converged{false};      ///< Whether solution converged
    int iterations{0};          ///< Number of iterations performed

    [[nodiscard]] bool is_valid() const noexcept {
        return converged && std::isfinite(yaw) && std::isfinite(pitch) && time_of_flight > 0.0;
    }
};

/// Ballistic solver interface (stateless)
///
/// This interface provides the core solving functionality for ballistic
/// trajectory compensation. It is deliberately stateless - the bullet velocity
/// is passed as a parameter rather than stored, eliminating the need for
/// const_cast tricks.
///
/// All solvers support visualization through generate_trajectory().
class IBallisticSolver {
public:
    virtual ~IBallisticSolver() = default;

    /// Solve for firing angles to hit target position
    ///
    /// @param target_pos Target position in world coordinates (meters)
    /// @param v0 Muzzle velocity (m/s)
    /// @return Complete aim solution with angles, time of flight, and convergence status
    [[nodiscard]] virtual AimSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept = 0;

    /// Generate ballistic trajectory points for visualization
    ///
    /// @param pitch Firing pitch angle (radians)
    /// @param v0 Muzzle velocity (m/s)
    /// @param max_distance Maximum horizontal distance to sample (meters)
    /// @return Vector of (x, z) trajectory points
    [[nodiscard]] virtual std::vector<std::pair<double, double>>
        generate_trajectory(double pitch, double v0, double max_distance) const noexcept = 0;

    /// Get human-readable solver name for debugging
    [[nodiscard]] virtual const char* solver_name() const noexcept = 0;
};

/// Configurable ballistic solver
///
/// Extends IBallisticSolver with runtime configuration management.
/// This is a separate interface (ISP) - clients that don't need runtime
/// configuration can depend only on IBallisticSolver.
///
/// Note: update_config() may rebuild internal caches (e.g., lookup tables).
class IConfigurableSolver : public IBallisticSolver {
public:
    ~IConfigurableSolver() override = default;

    /// Update solver configuration (may rebuild caches)
    ///
    /// @param config New trajectory configuration
    virtual void update_config(const TrajectoryConfig& config) noexcept = 0;

    /// Get current configuration
    ///
    /// @return Current trajectory configuration
    [[nodiscard]] virtual const TrajectoryConfig& config() const noexcept = 0;
};

/// Factory function to create ballistic solver
///
/// @param type Solver type (Ideal, Resistance, RK4)
/// @param config Initial trajectory configuration
/// @return Unique pointer to solver instance
[[nodiscard]] std::unique_ptr<IBallisticSolver>
    create_solver(TrajectoryCompensatorType type, const TrajectoryConfig& config) noexcept;

/// Factory function to create configurable ballistic solver
///
/// Use this when you need runtime configuration updates.
///
/// @param type Solver type (Ideal, Resistance, RK4)
/// @param config Initial trajectory configuration
/// @return Unique pointer to configurable solver instance
[[nodiscard]] std::unique_ptr<IConfigurableSolver> create_configurable_solver(
    TrajectoryCompensatorType type, const TrajectoryConfig& config) noexcept;

} // namespace fcs::L4::trajectory
