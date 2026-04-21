#pragma once

#include "config.hpp"
#include "scheduler/thin.hpp"

#include <string>

namespace fcs::runtime {

struct CapturerLaunchContext {
    bool daedalus{false};
    bool mock_mcu{false};
    std::string robot;
    std::string vision;
};

void register_runtime_capturer_system(
    talos::Scheduler& scheduler, const CapturerConfig& config, const CapturerLaunchContext& launch,
    const CameraConfig& camera_config);

} // namespace fcs::runtime
