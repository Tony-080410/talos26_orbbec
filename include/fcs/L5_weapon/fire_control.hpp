#pragma once

#include <Eigen/Core>
#include <cstdint>

namespace fcs::L5 {

// ============================================================================
// WeaponCommand - 武器控制命令
// ============================================================================

struct WeaponCommand {
    uint64_t timestamp_ns{0}; // 时间戳 (纳秒)
    // 云台角度
    double yaw{0.0};   // 目标 yaw (弧度)
    double pitch{0.0}; // 目标 pitch (弧度)

    // 云台角度
    double ref_yaw{0.0};   // 目标 yaw (弧度)
    double ref_pitch{0.0}; // 目标 pitch (弧度)

    // 前馈 (用于预测补偿)
    double yaw_vel{0.0};   // yaw 角速度 (rad/s)
    double pitch_vel{0.0}; // pitch 角速度 (rad/s)

    // 前馈 (用于预测补偿)
    double yaw_accel{0.0};
    double pitch_accel{0.0};

    // 云台角度
    double ref_yaw{0.0};   // 目标 yaw (弧度)
    double ref_pitch{0.0}; // 目标 pitch (弧度)

    // 发射控制
    bool fire{false}; // 发射指令

    // 弹道信息
    double tof{0.0};      // 飞行时间 (秒)
    double distance{0.0}; // 目标距离 (米)
};
} // namespace fcs::L5
