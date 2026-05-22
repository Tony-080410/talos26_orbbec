#pragma once

#include "L3_estimation/energy_meter_solver/motion_model.hpp"

namespace fcs::energy_meter {

struct EnergyMeterL3Config {
    ::energy_meter::BigRuneModel::Params big_model{};
    ::energy_meter::SmallRuneModel::Params small_model{};

    double reset_vote_time{2.0};
    int lost_thres{10};
    int tracking_thres{1};
    double matcher_gate{25.0};
    double angle_match_gate_rad{0.45};
    int blade_unlock_frames{10};
    int direction_lock_frames{4};
    int min_fit_samples{140};
    int max_fit_samples{240};
    int fit_iterations{4};
    double fit_window_sec{3.0};
    double fit_inlier_threshold_rad{0.08};
    double fit_penalty_a{0.5};
    double fit_penalty_w{0.5};
    double fit_temporal_weight{0.2};
    double fit_filter_alpha{1.0};
};

} // namespace fcs::energy_meter
