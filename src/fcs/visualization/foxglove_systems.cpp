#include "fcs/visualization/foxglove_systems.hpp"

#include "fcs/visualization/foxglove_server.hpp"
#include "fcs/visualization/foxglove_systems_impl.hpp"

#include "scheduler/scheduler/scheduler.hpp"

namespace fcs {

std::shared_ptr<FoxgloveServer>
    try_create_foxglove_server(uint16_t port, std::string host) noexcept {
    auto server = std::make_shared<FoxgloveServer>(port, std::move(host));
    if (!server->is_initialized()) {
        return {};
    }
    return server;
}

void register_foxglove_systems(talos::Scheduler& scheduler, talos::Scheduler* scheduler_ptr) {
    register_foxglove_systems<talos::Scheduler>(scheduler, scheduler_ptr);
}

} // namespace fcs
