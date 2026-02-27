#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "scheduler/thin.hpp"

namespace fcs {

class FoxgloveServer;

/// Create and return an initialized Foxglove server instance.
/// Returns `nullptr` if initialization fails.
[[nodiscard]] std::shared_ptr<FoxgloveServer>
    try_create_foxglove_server(uint16_t port, std::string host) noexcept;

// Register all Foxglove publisher systems for the Talos scheduler.
// Implemented in `src/fcs/visualization/foxglove_systems.cpp`.
void register_foxglove_systems(
    talos::Scheduler& scheduler, talos::Scheduler* scheduler_ptr = nullptr);

} // namespace fcs
