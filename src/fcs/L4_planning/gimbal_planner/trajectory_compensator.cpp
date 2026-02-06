#include "fcs/L4_planning/trajectory/solver_interfaces.hpp"

#include "fcs/L4_planning/trajectory/ballistic_model.hpp"
#include "fcs/L4_planning/trajectory/trajectory_solver.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string_view>

namespace fcs::L4::trajectory {

namespace {

[[nodiscard]] std::shared_ptr<BallisticModel>
    create_ballistic_model(TrajectoryCompensatorType type, const TrajectoryConfig& config) noexcept;

[[nodiscard]] std::unique_ptr<PitchSolver> create_pitch_solver(
    TrajectoryCompensatorType type, const std::shared_ptr<BallisticModel>& model,
    const TrajectoryConfig& config) noexcept;

// ============================================================================
// BallisticSolverImpl - Core Implementation (Stateless)
// ============================================================================

/// Internal implementation of IBallisticSolver/IConfigurableSolver
///
/// This class wraps the existing PitchSolver + BallisticModel architecture
/// and adapts it to the new interface design. It is deliberately stateless
/// with respect to bullet velocity (v0 is passed as a parameter).
class BallisticSolverImpl final : public IConfigurableSolver {
public:
    BallisticSolverImpl(
        TrajectoryCompensatorType type, std::shared_ptr<const BallisticModel> model,
        std::unique_ptr<PitchSolver> pitch_solver, TrajectoryConfig config) noexcept
        : type_(type)
        , model_(std::move(model))
        , pitch_solver_(std::move(pitch_solver))
        , config_(std::move(config)) {}

    // ========================================================================
    // IBallisticSolver implementation
    // ========================================================================

    [[nodiscard]] AimSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept override {
        // 1. Compute yaw (trigonometry, always solvable)
        const double yaw = std::atan2(target_pos.y(), target_pos.x());

        // 2. Delegate to PitchSolver for pitch and time of flight
        const auto pitch_sol = pitch_solver_->solve(target_pos, v0);

        return AimSolution{
            .yaw            = yaw,
            .pitch          = pitch_sol.pitch,
            .time_of_flight = pitch_sol.tof,
            .converged      = pitch_sol.converged,
            .iterations     = pitch_sol.iterations};
    }

    [[nodiscard]] std::vector<std::pair<double, double>>
        generate_trajectory(double pitch, double v0, double max_distance) const noexcept override {
        return pitch_solver_->get_trajectory(max_distance, pitch, v0);
    }

    [[nodiscard]] const char* solver_name() const noexcept override {
        return pitch_solver_->solver_name();
    }

    // ========================================================================
    // IConfigurableSolver implementation
    // ========================================================================

    void update_config(const TrajectoryConfig& config) noexcept override {
        config_       = config;
        auto model    = create_ballistic_model(type_, config_);
        model_        = model;
        pitch_solver_ = create_pitch_solver(type_, model, config_);
    }

    [[nodiscard]] const TrajectoryConfig& config() const noexcept override { return config_; }

private:
    TrajectoryCompensatorType type_;
    std::shared_ptr<const BallisticModel> model_;
    std::unique_ptr<PitchSolver> pitch_solver_;
    TrajectoryConfig config_;
};

// ============================================================================
// Helper: Create BallisticModel from type and config
// ============================================================================

[[nodiscard]] std::shared_ptr<BallisticModel> create_ballistic_model(
    TrajectoryCompensatorType type, const TrajectoryConfig& config) noexcept {

    using namespace trajectory;

    switch (type) {
    case TrajectoryCompensatorType::Ideal: return std::make_shared<IdealModel>(config.gravity);

    case TrajectoryCompensatorType::Resistance:
        return std::make_shared<ResistanceModel>(ResistanceModel::ResistanceParams{
            .gravity = config.gravity, .resistance = config.resistance});

    case TrajectoryCompensatorType::RK4:
        return std::make_shared<RK4Model>(RK4Model::RK4Params{
            .gravity   = config.gravity,
            .cd        = config.cd,
            .rho       = config.rho,
            .diameter  = config.diameter,
            .mass      = config.mass,
            .dt        = config.rk4_dt,
            .min_steps = config.rk4_min_steps,
            .max_steps = config.rk4_max_steps,
        });

    default:
        // Fallback to RK4
        return std::make_shared<RK4Model>(RK4Model::RK4Params{
            .gravity   = config.gravity,
            .cd        = config.cd,
            .rho       = config.rho,
            .diameter  = config.diameter,
            .mass      = config.mass,
            .dt        = config.rk4_dt,
            .min_steps = config.rk4_min_steps,
            .max_steps = config.rk4_max_steps,
        });
    }
}

// ============================================================================
// Helper: Create PitchSolver from type and config
// ============================================================================

[[nodiscard]] std::unique_ptr<PitchSolver> create_pitch_solver(
    TrajectoryCompensatorType type, const std::shared_ptr<BallisticModel>& model,
    const TrajectoryConfig& config) noexcept {

    using namespace trajectory;
    const auto iterative_cfg = IterativeSolver::Config{
        .max_iterations   = std::max(1, config.iteration_times),
        .height_tolerance = 0.01,
        .max_pitch        = std::numbers::pi / 2.5,
    };
    const auto table_cfg = TableLookupSolver::TableConfig{
        .range_max  = config.table_range_max,
        .range_step = config.table_range_step,
        .pitch_min  = config.table_pitch_min,
        .pitch_max  = config.table_pitch_max,
        .pitch_step = config.table_pitch_step,
    };

    switch (type) {
    case TrajectoryCompensatorType::Ideal:
    case TrajectoryCompensatorType::Resistance:
        // Use IterativeSolver for Ideal and Resistance models
        return std::make_unique<IterativeSolver>(model, iterative_cfg);

    case TrajectoryCompensatorType::RK4:
        if (config.enable_lookup_table) {
            return std::make_unique<TableLookupSolver>(model, table_cfg, config.bullet_speed);
        }
        return std::make_unique<IterativeSolver>(model, iterative_cfg);

    default:
        // Fallback to iterative solving for robustness.
        return std::make_unique<IterativeSolver>(model, iterative_cfg);
    }
}

} // namespace

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] std::unique_ptr<IBallisticSolver>
    create_solver(TrajectoryCompensatorType type, const TrajectoryConfig& config) noexcept {
    auto model        = create_ballistic_model(type, config);
    auto pitch_solver = create_pitch_solver(type, model, config);

    return std::make_unique<BallisticSolverImpl>(
        type, std::move(model), std::move(pitch_solver), config);
}

[[nodiscard]] std::unique_ptr<IConfigurableSolver> create_configurable_solver(
    TrajectoryCompensatorType type, const TrajectoryConfig& config) noexcept {
    auto model        = create_ballistic_model(type, config);
    auto pitch_solver = create_pitch_solver(type, model, config);

    return std::make_unique<BallisticSolverImpl>(
        type, std::move(model), std::move(pitch_solver), config);
}

} // namespace fcs::L4::trajectory
