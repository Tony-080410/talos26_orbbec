#pragma once
#include "core/armor_types.hpp"
#include "scheduler/system/components.hpp"
namespace fcs::core {

struct ImuState {
    double yaw_vel;
    double pitch_vel;
    double roll_vel;
};
using namespace talos::scheduler::system;
using imu_state     = res<ImuState>;
using imu_state_mut = res_mut<ImuState>;

using detecting_color     = res<ArmorColor>;
using detecting_color_mut = res_mut<ArmorColor>;

} // namespace fcs::core
