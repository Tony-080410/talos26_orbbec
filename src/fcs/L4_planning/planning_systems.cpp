#include "L4_planning/planning_systems.hpp"

#include "L4_planning/aimer/aimer_systems.hpp"
#include "core/trajectory/resource.hpp"
#include "scheduler/scheduler.hpp"
#include <spdlog/spdlog.h>

namespace fcs::L4 {

void register_l4_planning_systems(talos::Scheduler& scheduler, L4Config&& config) {
    SPDLOG_INFO("Registering L4 Planning systems: Aimer + Rune Planner");
    // Register Aimer system (Robot/Outpost/Rune)
    register_aimer_systems(scheduler, std::move(config));
}

} // namespace fcs::L4
