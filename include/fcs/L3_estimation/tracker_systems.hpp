#pragma once

#include "tracker/tracker.hpp"
#include "tracker/types.hpp"

#include "core/time.hpp"
#include "ecs/execution_policy.hpp"
#include "fcs/core/types.hpp"
#include "scheduler/ecs/channel_types.hpp"

#include <memory>

namespace fcs::L3 {

/// Register L3 tracker pipeline systems
///
/// Pipeline:
///   AtLegacyArmorMeasurementBatch [SPMC] -> TrackerSystem [pool_compute]
///                                               ↓
///                                        TrackerOutput [SPMC]
///
template <typename App>
void register_tracker_systems(App& app) {
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
}

} // namespace fcs::L3