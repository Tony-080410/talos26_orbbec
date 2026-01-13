#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"

#include "fcs/L4_planning/gimbal_planner/trajectory_compensator_impl.hpp"

namespace fcs::L4 {

std::unique_ptr<TrajectoryCompensator> create_compensator(TrajectoryCompensatorType type) noexcept {
    switch (type) {
    case TrajectoryCompensatorType::Ideal: return std::make_unique<IdealCompensator>();
    case TrajectoryCompensatorType::Resistance: return std::make_unique<ResistanceCompensator>();
    case TrajectoryCompensatorType::RK4: return std::make_unique<RK4Compensator>();
    }
    return std::make_unique<RK4Compensator>();
}

} // namespace fcs::L4

