#pragma once

#include "L4_planning/aimer/types.hpp"
#include "L4_planning/gimbal_planner/types.hpp"

namespace fcs::L4 {

/// Copy common fields from TargetPrediction to GimbalPlan
///
/// This function extracts the common prediction fields from a TargetPrediction
/// and copies them into a GimbalPlan output structure. It handles the following fields:
///
/// Source (TargetPrediction) -> Destination (GimbalPlan):
///   - timestamp_ns           -> timestamp_ns
///   - distance               -> distance
///   - predicted_position     -> predicted_position
///   - predicted_future_ns    -> predicted_future
///   - aimer_fire_advice      -> aimer_fire_advice
///   - aim_yaw                -> aim_yaw
///   - aim_pitch              -> aim_pitch
///
/// @param prediction The target prediction to copy from
/// @param plan_out The gimbal plan to copy to (must be valid)
///
/// @post plan_out contains the copied fields from prediction
///       Other fields in plan_out remain unchanged
///
/// @note This function does not throw exceptions
inline void copy_prediction_to_gimbal_plan(
    const TargetPrediction& prediction, GimbalPlan& plan_out) noexcept {

    plan_out.timestamp_ns       = prediction.timestamp_ns;
    plan_out.distance           = prediction.distance;
    plan_out.predicted_position = prediction.predicted_position;
    plan_out.predicted_future   = prediction.predicted_future_ns;

    // Preserve aimer's original advice
    plan_out.aimer_fire_advice = prediction.aimer_fire_advice;
    plan_out.aim_yaw           = prediction.aim_yaw;
    plan_out.aim_pitch         = prediction.aim_pitch;
}

} // namespace fcs::L4
