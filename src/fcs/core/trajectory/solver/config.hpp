#pragma once

#include "../config.hpp"
#include "../model/config.hpp"
#include "../model/ideal.hpp"
#include "../model/linear_drag.hpp"
#include "../model/quadratic_drag.hpp"
#include "solver_interfaces.hpp"
#include "trajectory_solver.hpp"

#include <memory>

namespace fcs::core::trajectory::solver {

[[nodiscard]] inline std::unique_ptr<TrajectorySolver>
    create_solver(const TrajectoryConfig& config) noexcept {

    const auto table_cfg = TableLookupSolver::TableConfig{
        .range_max  = config.model.table_range_max,
        .range_step = config.model.table_range_step,
        .pitch_min  = config.model.table_pitch_min,
        .pitch_max  = config.model.table_pitch_max,
        .pitch_step = config.model.table_pitch_step,
    };

    switch (config.model.type) {
    case model::ModelType::Ideal:
        return std::make_unique<DirectSolver>(
            std::make_unique<model::IdealModel>(config.model.gravity));
    case model::ModelType::LinearDrag:
        return std::make_unique<DirectSolver>(
            std::make_unique<model::LinearDragModel>(model::LinearDragModel::ResistanceParams{
                .gravity = config.model.gravity, .resistance = config.model.resistance}));
    case model::ModelType::QuadraticDrag:
        auto model =
            std::make_unique<model::QuadraticDragModel>(model::QuadraticDragModel::RK4Params{
                .gravity   = config.model.gravity,
                .cd        = config.model.cd,
                .rho       = config.model.rho,
                .diameter  = config.model.diameter,
                .mass      = config.model.mass,
                .dt        = config.model.rk4_dt,
                .min_steps = config.model.rk4_min_steps,
                .max_steps = config.model.rk4_max_steps});
        const auto iterative_cfg = IterativeSolver::Config{
            .max_iterations   = std::max(1u, config.model.iteration_times),
            .height_tolerance = 0.01,
            .max_pitch        = std::numbers::pi / 2.5,
        };
        auto iter = std::make_unique<IterativeSolver>(std::move(model), iterative_cfg);
        return std::make_unique<TableLookupSolver>(
            std::move(iter), table_cfg, config.model.bullet_speed);
    }
}
} // namespace fcs::core::trajectory::solver
