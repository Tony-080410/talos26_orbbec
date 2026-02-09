#pragma once

#include "ballistic_model.hpp"

namespace fcs::core::trajectory::model {

// ============================================================================
// RK4 Quadratic Drag Model
// ============================================================================

/// Ballistic model with quadratic air resistance (numerical RK4 integration).
///
/// Uses a 3D point-mass model with quadratic drag:
/// - F_drag = -0.5 * rho * Cd * A * |v| * v
///
/// This is the most accurate model for real-world ballistics but requires
/// numerical integration. The RK4Integrator provides a stateless implementation.
class QuadraticDragModel final : public BallisticModel {
public:
    struct RK4Params {
        double gravity{9.8};    ///< Gravity acceleration (m/s^2)
        double cd{0.47};        ///< Drag coefficient (dimensionless)
        double rho{1.225};      ///< Air density (kg/m^3)
        double diameter{0.017}; ///< Bullet diameter (m)
        double mass{0.0032};    ///< Bullet mass (kg)
        double dt{0.001};       ///< Desired integration timestep (s)
        int min_steps{50};      ///< Minimum integration steps
        int max_steps{2000};    ///< Maximum integration steps
    };

    // Note: no default params to avoid pedantic errors with default member initializers
    explicit QuadraticDragModel(const RK4Params& params) noexcept
        : gravity_(params.gravity)
        , cd_(params.cd)
        , rho_(params.rho)
        , diameter_(params.diameter)
        , mass_(params.mass)
        , dt_(params.dt)
        , min_steps_(params.min_steps)
        , max_steps_(params.max_steps) {}

    // Factory function for default construction
    [[nodiscard]] static QuadraticDragModel create_default() noexcept {
        return QuadraticDragModel{RK4Params{}};
    }

    [[nodiscard]] std::optional<ImpactResult>
        compute_impact(double range, double pitch, double v0) const noexcept override;

    [[nodiscard]] std::string_view model_name() const noexcept override { return "RK4"; }

    [[nodiscard]] double gravity() const noexcept { return gravity_; }
    [[nodiscard]] double drag_k() const noexcept;
    [[nodiscard]] double select_dt(double range, double pitch, double v0) const noexcept;
    [[nodiscard]] int min_steps() const noexcept { return min_steps_; }
    [[nodiscard]] int max_steps() const noexcept { return max_steps_; }

private:
    double gravity_;
    double cd_;
    double rho_;
    double diameter_;
    double mass_;
    double dt_;
    int min_steps_;
    int max_steps_;
};

// ============================================================================
// RK4 Integrator (Stateless Utility)
// ============================================================================

namespace detail {

/// Stateless RK4 integrator for ballistic trajectories with quadratic drag.
///
/// This class provides a static method for integrating bullet motion until
/// a target horizontal distance is reached. It is deliberately stateless
/// to avoid code duplication between RK4Model::compute_impact() and
/// table building in TableLookupSolver.
class RK4Integrator {
public:
    /// Integrate trajectory until reaching target horizontal distance.
    ///
    /// @param range Target horizontal distance (meters)
    /// @param pitch Firing angle (radians, positive = up)
    /// @param v0 Muzzle velocity (m/s)
    /// @param k Drag coefficient k = 0.5 * Cd * rho * A / m
    /// @param gravity Gravity acceleration (m/s^2)
    /// @param dt Desired timestep (s), will be adjusted based on range
    /// @param min_steps Minimum integration steps
    /// @param max_steps Maximum integration steps
    /// @return Impact result, or nullopt if integration fails
    [[nodiscard]] static std::optional<ImpactResult> integrate_to_range(
        double range, double pitch, double v0, double k, double gravity, double dt, int min_steps,
        int max_steps) noexcept;

    /// Integrate one full trajectory and return all sample points.
    ///
    /// This is used for building lookup tables. Returns samples at regular
    /// horizontal distance intervals.
    ///
    /// @param v0 Muzzle velocity (m/s)
    /// @param pitch Firing angle (radians)
    /// @param k Drag coefficient
    /// @param gravity Gravity acceleration (m/s^2)
    /// @param dt Desired timestep
    /// @param max_steps Maximum integration steps
    /// @param range_step Horizontal distance between samples (m)
    /// @param max_range Maximum horizontal distance to sample (m)
    /// @return Vector of (range, z, tof) samples
    [[nodiscard]] static std::vector<std::tuple<double, double, double>> integrate_full_trajectory(
        double v0, double pitch, double k, double gravity, double dt, int max_steps,
        double range_step, double max_range) noexcept;
};

} // namespace detail

} // namespace fcs::core::trajectory::model
