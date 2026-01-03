#pragma once

#include "armor_solver/solver.hpp"
#include "armor_solver/tracker.hpp"
#include "armor_solver/types.hpp"
#include "core/time.hpp"

#include "ecs/execution_policy.hpp"
#include "fcs/core/types.hpp"
#include "math/fast_tf/error.hpp"
#include "scheduler/ecs/channel_types.hpp"

#include <memory>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
# include <immintrin.h>
# define SPIN_HINT() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
# define SPIN_HINT() __asm__ volatile("yield" ::: "memory")
#else
# define SPIN_HINT() ((void)0)
#endif

namespace fcs::L3 {

/// Register armor solver pipeline systems
///
/// Pipeline:
///   AtLegacyArmorMeasurementBatch [SPMC] -> TrackerSystem [pool_compute]
///                                               ↓
///                                        TrackerOutput [SPMC]
///                                               ↓
///                                        SolverSystem [pool_compute]
///                                               ↓
///                                        GimbalCommand [SPMC]
template <typename App>
void register_armor_solver_systems(App& app) {
    // TrackerSystem: Measurement -> TrackerOutput
    app.template add_system<talos::pool_compute>(
        "armor_tracker",
        [last_timestamp_ns = uint64_t{0}](
            talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag> meas_in,
            talos::spmc_mut<TrackerOutput, TrackerOutputChannelTag> tracker_out,
            talos::res_mut<std::shared_ptr<Tracker>> tracker) mutable {
            if (!meas_in.has_new()) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            // Calculate dt from timestamps
            double dt = 0.0;
            if (last_timestamp_ns > 0 && batch->timestamp_ns > last_timestamp_ns) {
                dt = static_cast<double>(batch->timestamp_ns - last_timestamp_ns) / 1e9;
            }
            last_timestamp_ns = batch->timestamp_ns;

            // Handle state machine
            if ((*tracker)->status() == TrackerStatus::Idle) {
                // Try to initialize tracking
                auto result = (*tracker)->first_meet(*batch);
                if (!result) {
                    return;
                }
            } else {
                // Normal step
                (*tracker)->step(dt, *batch);
            }

            // Output tracker state
            auto output         = (*tracker)->get_output();
            output.timestamp_ns = batch->timestamp_ns;
            tracker_out.write(output);
        });

    // SolverSystem: TrackerOutput -> GimbalCommand
    app.template add_system<talos::exclusive<250, 2>>(
        "armor_solver", [](talos::spmc<TrackerOutput, TrackerOutputChannelTag> tracker_in,
                           talos::spmc_mut<GimbalCommand, SolverOutputChannelTag> cmd_out,
                           talos::res<std::shared_ptr<Solver>> solver,
                           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) {
            auto target = tracker_in.read_current();
            if (!target || !target->is_tracking()) {
                return;
            }

            // Get current real-time for prediction (not measurement timestamp!)
            // This matches at_vision's: solver_->solve(target_snapshot, now(), tf_buffer)
            const uint64_t current_time_ns = clock::now_ns();

            while (true) {
                // Lookup muzzle transform in odom frame
                // Use current time (not measurement timestamp!) to get latest gimbal pose
                // This matches at_vision's: lookupTransform(..., tf2::TimePointZero)
                auto tf_result =
                    fast_tf::lookup<fast_tf::odom_fuxk_frame, fast_tf::muzzle_link_fuxk_frame>(
                        **tf_system,
                        fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>{
                            fast_tf::TransformMatrix4d{}},
                        current_time_ns);

                if (!tf_result) {
                    if (tf_result.error() != fast_tf::TransformError::FutureExtrapolationRequired) {
                        spdlog::error(
                            "Solver TF lookup failed at {} ns: {}", target->timestamp_ns,
                            fast_tf::to_string(tf_result.error()));
                        return;
                    }
                    SPIN_HINT();
                    continue;
                }

                // Solve gimbal command with real-time current_time_ns
                auto cmd = (*solver)->solve(*target, current_time_ns, **tf_result);
                if (cmd) {
                    cmd_out.write(*cmd);
                }
                return;
            }
        });
}

} // namespace fcs::L3
