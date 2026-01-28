#include "fcs_visualization/foxglove_systems.hpp"

#include "fcs_visualization/foxglove_server.hpp"
#include "fcs_visualization/foxglove_systems_impl.hpp"

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
    register_foxglove_systems<talos::Scheduler>(scheduler, scheduler_ptr);
}

} // namespace fcs
