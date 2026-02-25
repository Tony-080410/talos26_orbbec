#include "L3_estimation/tracker_systems.hpp"

#include "L3_estimation/tracker/config.hpp"
#include "L3_estimation/tracker/new_tracker.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "core/channel_topics.hpp"
#include "ecs/channel_types.hpp"
#include "ecs/execution_policy.hpp"
#include "scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L3 {
void register_tracker_systems(talos::Scheduler& scheduler, TrackerConfig&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::pool_compute>(
        "armor_tracker",
        [last_timestamp_ns = uint64_t{0}, tracker_new = std::shared_ptr<TrackerNew>{}](
            talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
            talos::spmc_mut<TrackerOutput, TrackerOutputChannelTopic> tracker_out,
            talos::res<TrackerConfig> config) mutable {
            // Reactively update tracker config when changed
            if (!tracker_new || config.is_changed()) {
                tracker_new       = std::make_shared<TrackerNew>(*config);
                last_timestamp_ns = 0;
            }

            if (!meas_in.has_new()) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            if (last_timestamp_ns == 0) {
                last_timestamp_ns = batch->timestamp_ns;
            }

            double dt = 0.0;
            if (batch->timestamp_ns > last_timestamp_ns) {
                dt = static_cast<double>(batch->timestamp_ns - last_timestamp_ns) / 1e9;
            }
            last_timestamp_ns = batch->timestamp_ns;

            if (tracker_new->status() != TrackerStatus::Idle) {
                tracker_new->predict(dt);
            }

            if (tracker_new->status() == TrackerStatus::Idle) {
                (void)tracker_new->first_meet(*batch);
            } else {
                (void)tracker_new->update(*batch);
            }

            auto output         = tracker_new->get_output();
            output.timestamp_ns = batch->timestamp_ns;
            tracker_out.write(output);
        });
}

} // namespace fcs::L3
