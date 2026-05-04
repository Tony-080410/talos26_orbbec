#pragma once

#include "L3_estimation/energy_meter_solver/motion_model.hpp"

namespace fcs::energy_meter {

struct EnergyMeterL3Config {
    ::energy_meter::BigRuneModel::Params big_model{};
    ::energy_meter::SmallRuneModel::Params small_model{};

    int lost_thres{10};
    int tracking_thres{3};
    double matcher_gate{25.0};
    int blade_unlock_frames{10};
};

} // namespace fcs::energy_meter
