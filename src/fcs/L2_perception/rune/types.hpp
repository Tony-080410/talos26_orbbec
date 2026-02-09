#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstdint>
#include <vector>

namespace fcs::rune {

struct RectF {
    float x{};
    float y{};
    float w{};
    float h{};
};

struct RuneDebugFrame {
    uint64_t timestamp_ns{};
    uint64_t frame_id{};

    bool enabled{false};

    bool detect_reversed{false};
    bool tf_ok{false};
    bool solve_ok{false};
    bool observation_valid{false};

    uint32_t status_code{};

    uint32_t arrows_count{};
    uint32_t targets_count{};

    RectF global_roi{};
    RectF center_roi{};
    std::vector<RectF> target_rois{};

    std::vector<uint8_t> arrow_jpeg{};
    std::vector<uint8_t> target_jpeg{};
    std::vector<uint8_t> rcenter_jpeg{};
};

struct RuneObservation {
    uint64_t timestamp_ns{};
    uint64_t frame_id{};

    bool valid{false};

    Eigen::Vector3d r_center_odom{Eigen::Vector3d::Zero()};

    std::vector<Eigen::Vector3d> target_positions_odom{};
    std::vector<Eigen::Quaterniond> target_quats_odom{};
};

} // namespace fcs::rune
