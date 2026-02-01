#include "fcs_visualization/foxglove_systems.hpp"

#include "fcs_visualization/foxglove_server.hpp"
#include "fcs_visualization/systems/base.hpp"
#include "fcs_visualization/systems/debug_systems.hpp"
#include "fcs_visualization/systems/l1_sensor_systems.hpp"
#include "fcs_visualization/systems/l2_perception_systems.hpp"
#include "fcs_visualization/systems/l3_estimation_systems.hpp"
#include "fcs_visualization/systems/l4_planning_systems.hpp"
#include "fcs_visualization/systems/rune_systems.hpp"

#include "scheduler/scheduler/scheduler.hpp"

namespace fcs {

std::shared_ptr<FoxgloveServer>
    try_create_foxglove_server(uint16_t port, std::string host) noexcept {
    auto server = create_foxglove_server(port, std::move(host));
    if (!server.has_value()) {
        return {};
    }
    return std::move(*server);
}

void register_foxglove_systems(talos::Scheduler& scheduler, talos::Scheduler* scheduler_ptr) {
    using namespace foxglove::systems;

    // Register all layered systems
    register_l1_sensor_systems(scheduler);
    register_l2_perception_systems(scheduler);
    register_l3_estimation_systems(scheduler);
    register_l4_planning_systems(scheduler);
    register_rune_systems(scheduler);
    register_debug_systems(scheduler, scheduler_ptr);
}

} // namespace fcs
