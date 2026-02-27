#pragma once

#include "config.hpp"
#include "scheduler/scheduler.hpp"
#include "solver/config.hpp"
#include "solver/solver_interfaces.hpp"

namespace fcs::core::trajectory {

using trajectory_solver = talos::res<std::unique_ptr<solver::TrajectorySolver>>;
using trajectory_config = talos::res<model::ModelConfig>;

constexpr void register_resource(talos::Scheduler& scheduler, TrajectoryConfig&& config) noexcept {
    scheduler.world().insert_resource(config.model);
    auto solver = solver::create_solver(config);
    scheduler.world().insert_resource(std::move(solver));
}

} // namespace fcs::core::trajectory
