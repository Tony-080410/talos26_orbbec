#include "L3_estimation/ldm_naive/ldm_naive_systems.hpp"

#include "L2_perception/ldm/types.hpp"
#include "L3_estimation/ldm_naive/types.hpp"
#include "core/channel_topics.hpp"
#include "scheduler/scheduler.hpp"

#include <Eigen/Core>
#include <spdlog/spdlog.h>

namespace fcs::L3::ldm {

void register_naive_ldm_systems(talos::Scheduler& scheduler, const NaiveLdmConfig& config) {
    scheduler.world().insert_resource(config);

    scheduler.add_system<talos::pool_compute>(
        "l3_ldm_naive",
        [](talos::res<NaiveLdmConfig> cfg,
           talos::subscribe<fcs::L2::ldm::LdmMeasurement, LdmMeasurementChannelTopic> meas_in,
           talos::publish<LdmState> state_out) mutable {
            (void)cfg;
            auto meas = meas_in.read();
            if (!meas || !meas->valid) {
                return;
            }
            if (!meas->transform_odom.has_value()) {
                SPDLOG_INFO("skip ldm measurement for no transform_odom");
                return;
            }
            LdmState state{};
            state.timestamp_ns = meas->timestamp_ns;
            state.accurate     = false;
            state.position     = meas->transform_odom->translation();
            state_out.write(state);
        });
}

} // namespace fcs::L3::ldm
