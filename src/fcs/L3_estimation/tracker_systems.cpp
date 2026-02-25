#include "L3_estimation/tracker_systems.hpp"

#include "L3_estimation/tracker/config.hpp"
#include "L3_estimation/tracker/new_tracker.hpp"
#include "L3_estimation/tracker/tracker.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "core/channel_topics.hpp"
#include "ecs/channel_types.hpp"
#include "ecs/execution_policy.hpp"
#include "scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L3 {

std::shared_ptr<Tracker> create_tracker(const TrackerConfig& config) {
    return std::make_shared<Tracker>(config);
}

void register_tracker_systems(talos::Scheduler& scheduler, TrackerConfig&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::pool_compute>(
        "armor_tracker",
        [last_timestamp_ns = uint64_t{0}, tracker_new = std::shared_ptr<TrackerNew>{}](
            talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
            talos::spmc_mut<TrackerOutput, TrackerOutputChannelTopic> tracker_out,
            talos::res<TrackerConfig> config,
            talos::res_mut<std::shared_ptr<Tracker>> tracker) mutable {
            if (!tracker_new) {
                tracker_new = std::make_shared<TrackerNew>(*config);
            }

            // Reactively update tracker config when changed
            if (config.is_changed()) {
                *tracker          = std::make_shared<Tracker>(*config);
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

            const bool use_new_tracker = config->use_new_tracker;
            if (use_new_tracker) {
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
            } else {
                // First read: check if we should predict (matches armor_solver line 307-309)
                if ((*tracker)->status() != TrackerStatus::Idle) {
                    (*tracker)->predict(dt);
                }

                // Second read: state may have changed after first_meet (matches armor_solver line
                // 310-313)
                if ((*tracker)->status() == TrackerStatus::Idle) {
                    (void)(*tracker)->first_meet(*batch);
                } else {
                    (void)(*tracker)->update(*batch);
                }

                auto output         = (*tracker)->get_output();
                output.timestamp_ns = batch->timestamp_ns;
                tracker_out.write(output);
            }
        });
}

} // namespace fcs::L3
