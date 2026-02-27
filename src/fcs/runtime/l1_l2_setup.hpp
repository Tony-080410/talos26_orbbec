#pragma once

#include <expected>
#include <string>

#include "scheduler/thin.hpp"

namespace fcs {
struct HardwareConfig;
struct VisionConfig;
} // namespace fcs

namespace fcs::runtime {

struct L1L2SetupResult {
    std::string detector_backend_name;
};

// Creates and registers:
// - IMU reader system
// - Camera reader system (publishes AtLegacyImageFrame)
// - Weapon output system (consumes L5::WeaponCommand)
// - L2 detector backend + PnP solver resources
// - CameraConfig resource
//
// `hardware` must be non-null when `daedalus == false`.
[[nodiscard]] std::expected<L1L2SetupResult, std::string> setup_l1_l2(
    talos::World& world, talos::Scheduler& scheduler, const VisionConfig& vision, bool daedalus,
    const HardwareConfig* hardware);

} // namespace fcs::runtime
