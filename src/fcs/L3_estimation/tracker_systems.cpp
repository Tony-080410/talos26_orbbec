#include "fcs/L3_estimation/tracker_systems.hpp"

#include "fcs/L3_estimation/tracker/config.hpp"
#include "fcs/L3_estimation/tracker/tracker.hpp"
#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/core/channel_topics.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L3 {

std::shared_ptr<Tracker> create_tracker(const TrackerConfig& config) {
    return std::make_shared<Tracker>(config);
}

void register_tracker_systems(talos::Scheduler& scheduler, TrackerConfig&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::pool_compute>(
        "armor_tracker",
        [last_timestamp_ns = uint64_t{0}](
            talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
            talos::spmc_mut<TrackerOutput, TrackerOutputChannelTopic> tracker_out,
            talos::res<TrackerConfig> config,
            talos::res_mut<std::shared_ptr<Tracker>> tracker) mutable {
            // Reactively update tracker config when changed
            if (config.is_changed()) {
                // Config has changed, reset tracker to apply new configuration
                (*tracker)->reset();
            }

            if (!meas_in.has_new()) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            double dt = 0.0;
            if (last_timestamp_ns > 0 && batch->timestamp_ns > last_timestamp_ns) {
                dt = static_cast<double>(batch->timestamp_ns - last_timestamp_ns) / 1e9;
            }
            last_timestamp_ns = batch->timestamp_ns;

            if ((*tracker)->status() == TrackerStatus::Idle) {
                auto result = (*tracker)->first_meet(*batch);
                if (!result) {
                    return;
                }
            } else {
                (*tracker)->step(dt, *batch);
            }

            auto output         = (*tracker)->get_output();
            output.timestamp_ns = batch->timestamp_ns;
            tracker_out.write(output);
        });
}

} // namespace fcs::L3
