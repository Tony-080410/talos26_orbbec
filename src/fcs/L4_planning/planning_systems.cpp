#include "fcs/L4_planning/planning_systems.hpp"

#include "fcs/L4_planning/aimer/aimer_systems.hpp"
#include "fcs/L4_planning/planner/planner_systems.hpp"
#include <spdlog/spdlog.h>

namespace fcs::L4 {

void register_l4_planning_systems(talos::Scheduler& scheduler, L4Config&& config) {
    SPDLOG_INFO("Registering L4 Planning systems: Aimer + Planner architecture");

    // Register Aimer systems (RobotAimer, OutpostAimer, RuneAimer)
    // This inserts L4Config into world
    register_aimer_systems(scheduler, std::move(config));

    // Register Planner system (MpcPlanner consuming AimerOutput)
    // Note: L4Config is already in world from register_aimer_systems
    register_planner_systems(scheduler);
}

} // namespace fcs::L4
