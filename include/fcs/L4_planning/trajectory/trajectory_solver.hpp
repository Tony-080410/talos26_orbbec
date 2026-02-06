#pragma once

#include "fcs/L4_planning/trajectory/ballistic_model.hpp"

#include <Eigen/Core>

#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

namespace fcs::L4::trajectory {

/// Result of pitch solving iteration.
struct PitchSolution {
    double pitch{0.0};     ///< Firing pitch angle (radians)
    double tof{0.0};       ///< Time of flight (seconds)
    bool converged{false}; ///< Whether solution converged to tolerance
    int iterations{0};     ///< Number of iterations performed
};

/// Abstract pitch solver for ballistic trajectory compensation.
///
/// Given a target position (x, y, z), computes the required firing pitch
/// to hit that target. Different implementations can use iterative methods,
/// analytic solutions, or precomputed lookup tables.
class PitchSolver {
public:
    virtual ~PitchSolver() = default;

    /// Solve for firing pitch to hit target position.
    ///
    /// @param target_pos Target position in world coordinates (meters)
    /// @param v0 Muzzle velocity (m/s)
    /// @return Pitch solution with angle, time of flight, and convergence status
    [[nodiscard]] virtual PitchSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept = 0;

    /// Get flying time for a target (for visualization/feedback).
    ///
    /// @param target_pos Target position in world coordinates (meters)
    /// @param v0 Muzzle velocity (m/s)
    /// @return Time of flight, or nullopt if computation fails
    [[nodiscard]] virtual std::optional<double>
        get_flying_time(const Eigen::Vector3d& target_pos, double v0) const noexcept = 0;

    /// Get trajectory points for visualization (optional, default empty).
    ///
    /// @param max_distance Maximum horizontal distance to sample (meters)
    /// @param pitch Firing pitch angle (radians)
    /// @param v0 Muzzle velocity (m/s)
    /// @return Vector of (x, z) trajectory points
    [[nodiscard]] virtual std::vector<std::pair<double, double>>
        get_trajectory(double max_distance, double pitch, double v0) const noexcept {
        (void)max_distance;
        (void)pitch;
        (void)v0;
        return {};
    }

    /// Get human-readable solver name for debugging.
    [[nodiscard]] virtual const char* solver_name() const noexcept = 0;
};

// ============================================================================
// Iterative Solver (Stateless, Works with Any BallisticModel)
// ============================================================================

/// Generic iterative pitch solver using successive approximation.
///
/// This solver works with any BallisticModel implementation. It uses
/// a simple iterative method: start with line-of-sight angle, compute
/// impact height error, adjust target height, repeat until convergence.
///
/// This is deliberately stateless - all configuration is passed via
/// the Config struct and stored in the shared_ptr<BallisticModel>.
class IterativeSolver final : public PitchSolver {
public:
    struct Config {
        int max_iterations{20};                   ///< Maximum solving iterations
        double height_tolerance{0.01};            ///< Convergence threshold (meters)
        double max_pitch{std::numbers::pi / 2.5}; ///< Maximum valid pitch (radians)
    };

    // Note: no default config to avoid pedantic errors with default member initializers
    explicit IterativeSolver(
        std::shared_ptr<const BallisticModel> model, const Config& config) noexcept
        : model_(std::move(model))
        , config_(config) {}

    // Factory function for default config
    [[nodiscard]] static IterativeSolver
        create(std::shared_ptr<const BallisticModel> model) noexcept {
        return IterativeSolver{std::move(model), Config{}};
    }

    [[nodiscard]] PitchSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept override;

    [[nodiscard]] std::optional<double>
        get_flying_time(const Eigen::Vector3d& target_pos, double v0) const noexcept override;

    [[nodiscard]] std::vector<std::pair<double, double>>
        get_trajectory(double max_distance, double pitch, double v0) const noexcept override;

    [[nodiscard]] const char* solver_name() const noexcept override { return "Iterative"; }

    [[nodiscard]] const BallisticModel& model() const noexcept { return *model_; }
    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    std::shared_ptr<const BallisticModel> model_;
    Config config_;
};

// ============================================================================
// Table Lookup Solver (Generic Optimization for Any BallisticModel)
// ============================================================================

/// Generic table-based solver with precomputed (range, pitch) -> (z, tof) lookup.
///
/// This solver can optimize ANY BallisticModel by precomputing a 2D lookup
/// table. At runtime, it inverts the table to find pitch for a given (range, z).
///
/// Table inversion: For a fixed range, search the pitch dimension to find
/// the pitch that produces the target z height. Then use the model to refine.
///
/// This works with:
/// - IdealModel + TableLookup (fast analytic ballistic)
/// - ResistanceModel + TableLookup (fast linear drag)
/// - RK4Model + TableLookup (fast quadratic drag, original behavior)
class TableLookupSolver final : public PitchSolver {
public:
    struct TableConfig {
        double range_max{20.0};   ///< Maximum range in table (meters)
        double range_step{0.01};  ///< Range step size (meters)
        double pitch_min{-0.6};   ///< Minimum pitch in table (radians)
        double pitch_max{0.9};    ///< Maximum pitch in table (radians)
        double pitch_step{0.005}; ///< Pitch step size (radians)
    };

    // Note: no default config to avoid pedantic errors with default member initializers
    explicit TableLookupSolver(
        std::shared_ptr<const BallisticModel> model, const TableConfig& config, double v0) noexcept;

    // Factory function for default config
    [[nodiscard]] static TableLookupSolver
        create(std::shared_ptr<const BallisticModel> model, double v0 = 15.0) noexcept;

    /// Rebuild the lookup table (call after model parameters change).
    void rebuild_table(double v0) noexcept;

    [[nodiscard]] PitchSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept override;

    [[nodiscard]] std::optional<double>
        get_flying_time(const Eigen::Vector3d& target_pos, double v0) const noexcept override;

    [[nodiscard]] std::vector<std::pair<double, double>>
        get_trajectory(double max_distance, double pitch, double v0) const noexcept override;

    [[nodiscard]] const char* solver_name() const noexcept override { return "TableLookup"; }

    [[nodiscard]] const BallisticModel& model() const noexcept { return *model_; }
    [[nodiscard]] const TableConfig& config() const noexcept { return config_; }
    [[nodiscard]] bool has_table() const noexcept { return table_ != nullptr; }

private:
    struct LookupTable {
        double range_step;
        double range_max;
        double pitch_min;
        double pitch_max;
        double pitch_step;
        size_t range_count;
        size_t pitch_count;
        std::vector<double> z;   // [pitch][range] - height at impact
        std::vector<double> tof; // [pitch][range] - time of flight
    };

    std::shared_ptr<const BallisticModel> model_;
    TableConfig config_;
    std::shared_ptr<const LookupTable> table_;

    [[nodiscard]] static double
        sample_table_z(const LookupTable& table, double x, double pitch) noexcept;

    [[nodiscard]] static bool lookup_pitch_tof(
        const LookupTable& table, double range, double target_z, double& pitch_out,
        double& tof_out) noexcept;

    [[nodiscard]] static bool estimate_dzdp(
        const LookupTable& table, double range, double pitch, double& dzdp_out) noexcept;
};

// ============================================================================
// Utility Functions
// ============================================================================

namespace detail {

/// Generic iterative pitch solving algorithm (stateless).
///
/// This function implements the core iteration logic used by IterativeSolver.
/// It's separated to allow reuse in different contexts.
[[nodiscard]] PitchSolution iterative_solve_pitch(
    const BallisticModel& model, const Eigen::Vector3d& target_pos, double v0, int max_iterations,
    double height_tolerance, double max_pitch) noexcept;

} // namespace detail

} // namespace fcs::L4::trajectory
