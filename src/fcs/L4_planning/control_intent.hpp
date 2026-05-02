#pragma once

#include "core/trajectory/reference_trajectory.hpp"

#include <cstdint>
#include <variant>

namespace fcs::L4 {

// ============================================================================
// L4 → L5 Control Intent
// ============================================================================
//
// L4 translates tracker state into a pure control intention for L5.
// The variant constructors precisely encode the three operational modes:
//
//   TrackCommand  = "I predicted a trajectory. Follow it with MPC."
//   ShotCommand   = "Target is here. Aim and fire."
//   HoldCommand   = "No target. Stay put."
//
// No enum tags, no std::optional fields, no runtime checks needed.
// The variant IS the specification.

/// Track: full reference trajectory for MPC optimization.
struct TrackCommand {
    uint64_t timestamp_ns{0};
    core::trajectory::ReferenceTrajectory trajectory;
};

/// Shot: direct aim without trajectory optimization.
struct ShotCommand {
    uint64_t timestamp_ns{0};
    double yaw{0.0};
    double pitch{0.0};
    double distance{0.0};
};

/// Hold: no target, maintain current position.
struct HoldCommand {
    uint64_t timestamp_ns{0};
};

/// L4 → L5 control intent.
using ControlIntent = std::variant<TrackCommand, ShotCommand, HoldCommand>;

} // namespace fcs::L4
