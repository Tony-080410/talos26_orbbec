#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <vector>

namespace fcs::L5 {

// ============================================================================
// WeaponCommand - 武器控制命令
// ============================================================================

struct TrajectoryPlanSample {
    double yaw{0.0};
    double pitch{0.0};
    double distance{0.0};
    double tof{0.0};
    double diff_yaw{0.0};
    std::optional<int> selected_armor_id{};
    std::optional<int> rough_selected_armor_id{};
    Eigen::Vector3d world_position{Eigen::Vector3d::Zero()};
};

struct WeaponVisualizationDebugData {
    std::vector<TrajectoryPlanSample> reference_plan;
    std::vector<TrajectoryPlanSample> optimized_plan;
    int center_index{0};
    int lookahead_index{0};        // Legacy alias, currently mirrors center_index in L5.
    int delay_steps{0};
    double delay_gate_threshold{0.0};
    double delay_gate_forward_error{0.0};
    double delay_gate_backward_error{0.0};
    bool delay_gate_forward_ok{false};
    bool delay_gate_backward_ok{false};
    bool phase_gate_ok{false};
    bool final_fire_gate_ok{false};
};

struct WeaponCommand {
    uint64_t timestamp_ns{0};      // L5 命令生成时间戳 (纳秒)
    uint64_t plan_timestamp_ns{0}; // 对应的 L4 plan 时间戳 (纳秒)
    // L4 原始 plan（优化前，仅调试）
    double plan_yaw{0.0};
    double plan_pitch{0.0};
    bool plan_fire{false};

    // 云台角度
    double yaw{0.0};   // 目标 yaw (弧度)
    double pitch{0.0}; // 目标 pitch (弧度)

    // L5 参考中心点（仅调试）
    double ref_yaw{0.0};   // 目标 yaw (弧度)
    double ref_pitch{0.0}; // 目标 pitch (弧度)

    // 前馈 (用于预测补偿)
    double yaw_vel{0.0};   // yaw 角速度 (rad/s)
    double pitch_vel{0.0}; // pitch 角速度 (rad/s)

    // 前馈 (用于预测补偿)
    double yaw_accel{0.0};
    double pitch_accel{0.0};

    // 发射控制
    bool fire{false}; // 发射指令

    // 弹道信息
    double tof{0.0};      // 飞行时间 (秒)
    double distance{0.0}; // 目标距离 (米)

    // 可视化调试数据
    std::optional<WeaponVisualizationDebugData> viz_debug;
};
} // namespace fcs::L5
