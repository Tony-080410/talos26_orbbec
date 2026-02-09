#pragma once

#include "core/trajectory/model/ballistic_model.hpp"
#include "core/trajectory/solver/solver_interfaces.hpp"

#include <Eigen/Core>

#include <limits>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

namespace fcs::core::trajectory::solver {

// ============================================================================
// Direct Solver (Analytical Solution for Ideal/LinearDrag)
// ============================================================================

/// Direct solver using analytical solutions.
///
/// This solver provides analytical (closed-form) solutions for ballistic models
/// that support direct computation. Works with:
/// - IdealModel: Vacuum ballistic trajectory
/// - LinearDragModel: Linear air resistance
///
/// This is the preferred solver for models with analytical solutions as it's
/// faster and more accurate than iterative methods.
class DirectSolver final : public TrajectorySolver {
public:
    /// Construct DirectSolver with any BallisticModel
    template <typename Model>
    requires(std::derived_from<Model, model::BallisticModel>)
    explicit DirectSolver(std::unique_ptr<Model>&& model) noexcept
        : model_(std::move(model)) {}

    /// Factory function for unique_ptr construction
    [[nodiscard]] static std::unique_ptr<DirectSolver>
        create(std::unique_ptr<model::BallisticModel> model) noexcept {
        return std::unique_ptr<DirectSolver>(new DirectSolver(std::move(model)));
    }

    /// Solve for firing angles to hit target
    [[nodiscard]] AimSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept override;

    /// Generate trajectory points for visualization
    [[nodiscard]] std::vector<std::pair<double, double>>
        generate_trajectory(double pitch, double v0, double max_distance) const noexcept override;

    /// Get solver name for debugging
    [[nodiscard]] std::string_view solver_name() const noexcept override { return "Direct"; }

    /// Access the underlying ballistic model
    [[nodiscard]] const std::unique_ptr<model::BallisticModel>& model() const noexcept {
        return model_;
    }

private:
    std::unique_ptr<model::BallisticModel> model_;
};

// ============================================================================
// Iterative Solver (Numerical Solution for QuadraticDrag)
// ============================================================================

/// Iterative solver using successive approximation.
///
/// This solver works with any BallisticModel implementation. It uses
/// a simple iterative method: start with line-of-sight angle, compute
/// impact height error, adjust target height, repeat until convergence.
///
/// This is deliberately stateless - all configuration is passed via
/// the Config struct and stored in the unique_ptr<BallisticModel>.
class IterativeSolver final : public TrajectorySolver {
public:
    struct Config {
        uint32_t max_iterations{20};              ///< Maximum solving iterations
        double height_tolerance{0.01};            ///< Convergence threshold (meters)
        double max_pitch{std::numbers::pi / 2.5}; ///< Maximum valid pitch (radians)
    };

    /// Construct IterativeSolver with any BallisticModel
    template <typename Model>
    requires(std::derived_from<Model, model::BallisticModel>)
    explicit IterativeSolver(std::unique_ptr<Model>&& model, Config config) noexcept
        : model_(std::move(model))
        , config_(config) {}

    /// Factory function for default config
    [[nodiscard]] static std::unique_ptr<IterativeSolver>
        create(std::unique_ptr<model::BallisticModel> model) noexcept {
        return std::make_unique<IterativeSolver>(std::move(model), Config{});
    }

    /// Solve for firing angles to hit target
    [[nodiscard]] AimSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept override;

    /// Generate trajectory points for visualization
    [[nodiscard]] std::vector<std::pair<double, double>>
        generate_trajectory(double pitch, double v0, double max_distance) const noexcept override;

    /// Get solver name for debugging
    [[nodiscard]] std::string_view solver_name() const noexcept override { return "Iterative"; }

    /// Access the underlying ballistic model
    [[nodiscard]] const std::unique_ptr<model::BallisticModel>& model() const noexcept {
        return model_;
    }

    /// Access solver configuration
    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    std::unique_ptr<model::BallisticModel> model_;
    Config config_;
};

// ============================================================================
// Table Lookup Solver (Decorator Pattern)
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
/// - LinearDragModel + TableLookup (fast linear drag)
/// - QuadraticDragModel + TableLookup (fast quadratic drag, optimized)
///
/// **Decorator Pattern**: This class wraps (decorates) any TrajectorySolver
/// to add table-based acceleration. The decorated solver is used for
/// table building and fallback refinement.
class TableLookupSolver final : public TrajectorySolver {
public:
    struct TableConfig {
        double range_max{20.0};   ///< Maximum range in table (meters)
        double range_step{0.01};  ///< Range step size (meters)
        double pitch_min{-0.6};   ///< Minimum pitch in table (radians)
        double pitch_max{0.9};    ///< Maximum pitch in table (radians)
        double pitch_step{0.005}; ///< Pitch step size (radians)
    };

    /// Decorator constructor: wraps any TrajectorySolver
    ///
    /// @param base_solver The solver to decorate (will be moved)
    /// @param config Table configuration
    /// @param v0 Muzzle velocity for table building (m/s)
    explicit TableLookupSolver(
        std::unique_ptr<TrajectorySolver> base_solver, const TableConfig& config,
        double v0) noexcept;

    /// Factory function for unique_ptr construction
    [[nodiscard]] static std::unique_ptr<TableLookupSolver>
        create(std::unique_ptr<TrajectorySolver> base_solver, double v0) noexcept {
        return std::make_unique<TableLookupSolver>(std::move(base_solver), TableConfig{}, v0);
    }

    /// Rebuild the lookup table (call after model parameters change)
    void rebuild_table(double v0) noexcept;

    /// Solve for firing angles to hit target
    [[nodiscard]] AimSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept override;

    /// Generate trajectory points for visualization
    [[nodiscard]] std::vector<std::pair<double, double>>
        generate_trajectory(double pitch, double v0, double max_distance) const noexcept override;

    /// Get solver name for debugging
    [[nodiscard]] std::string_view solver_name() const noexcept override { return "TableLookup"; }

    /// Access table configuration
    [[nodiscard]] const TableConfig& config() const noexcept { return config_; }

    /// Check if lookup table is built
    [[nodiscard]] bool has_table() const noexcept { return table_ != nullptr; }

    /// Access the decorated solver
    [[nodiscard]] const std::unique_ptr<TrajectorySolver>& base_solver() const noexcept {
        return base_solver_;
    }

private:
    /// Lookup table structure
    struct LookupTable {
        double range_step;
        double range_max;
        double pitch_min;
        double pitch_max;
        double pitch_step;
        size_t range_count;
        size_t pitch_count;
        std::vector<double> z;                      // [pitch][range] - height at impact
        std::vector<double> tof;                    // [pitch][range] - time of flight
    };

    std::unique_ptr<TrajectorySolver> base_solver_; ///< Decorated solver
    TableConfig config_;
    std::unique_ptr<LookupTable> table_;

    /// Sample table using bilinear interpolation
    [[nodiscard]] static double
        sample_table_z(const LookupTable& table, double x, double pitch) noexcept;

    /// Find pitch and TOF for given range and target z
    [[nodiscard]] static bool lookup_pitch_tof(
        const LookupTable& table, double range, double target_z, double& pitch_out,
        double& tof_out) noexcept;

    /// Estimate dz/dp (partial derivative) for Newton refinement
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
[[nodiscard]] AimSolution iterative_solve_pitch(
    const model::BallisticModel& model, const Eigen::Vector3d& target_pos, double v0,
    int max_iterations, double height_tolerance, double max_pitch) noexcept;

} // namespace detail

} // namespace fcs::core::trajectory::solver
