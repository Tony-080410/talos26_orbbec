#pragma once

#include "L3_estimation/tracker/types.hpp"

#include <Eigen/Core>

#include <cstdint>

namespace fcs::L3::ldm {

struct LdmState {
    uint64_t timestamp_ns{0};
    uint64_t last_observation_timestamp_ns{0};
    TrackerStatus status{TrackerStatus::Idle};
    bool accurate{false};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d R_world_body{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d velocity_body{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity_world{Eigen::Vector3d::Zero()};

    [[nodiscard]] bool is_tracking() const noexcept {
        return status == TrackerStatus::Tracking || status == TrackerStatus::TempLost;
    }
};

} // namespace fcs::L3::ldm
