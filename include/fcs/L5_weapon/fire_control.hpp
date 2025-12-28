#pragma once

#include "../L3_estimation/estimator.hpp"
#include "../core/types.hpp"

#include <Eigen/Core>
#include <cstdint>

namespace fcs::L5 {

// ============================================================================
// WeaponCommand - 武器控制命令
// ============================================================================

struct WeaponCommand {
    // 云台角度
    double yaw{0.0};   // 目标 yaw (弧度)
    double pitch{0.0}; // 目标 pitch (弧度)

    // 前馈 (用于预测补偿)
    double yaw_rate{0.0};   // yaw 角速度 (rad/s)
    double pitch_rate{0.0}; // pitch 角速度 (rad/s)

    // 发射控制
    bool fire{false}; // 发射指令

    // 弹道信息
    double tof{0.0};      // 飞行时间 (秒)
    double distance{0.0}; // 目标距离 (米)

    // 有效性
    bool valid{false};
};

// ============================================================================
// FireControlConfig - 火控配置
// ============================================================================

struct FireControlConfig {
    // 弹道参数
    double gravity{9.8};         // 重力加速度 (m/s^2)
    double air_resistance{0.01}; // 空气阻力系数

    // 迭代求解
    int max_iterations{10};              // 最大迭代次数
    double convergence_threshold{0.001}; // 收敛阈值 (秒)

    // 偏置补偿
    double yaw_offset{0.0};   // yaw 偏移 (rad)
    double pitch_offset{0.0}; // pitch 偏移 (rad)

    // 发射判定
    double fire_tolerance_near{0.052}; // 近距离容差 (rad, ~3度)
    double fire_tolerance_far{0.035};  // 远距离容差 (rad, ~2度)
    double distance_threshold{2.0};    // 近/远距离分界 (米)
    bool auto_fire{true};              // 启用自动发射

    // 小陀螺参数
    double spinning_threshold{3.0}; // 小陀螺角速度阈值 (rad/s)
    double coming_angle{0.96};      // 来袭角度阈值 (rad, ~55度)
    double leaving_angle{0.35};     // 离开角度阈值 (rad, ~20度)
};

// ============================================================================
// FireControl - 火控计算机
// ============================================================================

class FireControl {
public:
    explicit FireControl(const FireControlConfig& config = {});

    /**
     * @brief 计算武器控制命令
     *
     * @param track 航迹状态
     * @param bullet_speed 弹速 (m/s)
     * @param gimbal_yaw 当前云台 yaw (rad, 用于发射判定)
     * @param timestamp_ns 当前时间戳
     * @return 武器控制命令
     */
    [[nodiscard]] WeaponCommand solve(
        const L3::TrackState& track, double bullet_speed, double gimbal_yaw,
        uint64_t timestamp_ns) noexcept;

private:
    FireControlConfig config_;
    int lock_armor_id_{-1};  // 锁定的装甲板 ID
    WeaponCommand last_cmd_; // 上次命令 (用于突变检测)

    // ======== 弹道求解 ========

    /**
     * @brief 弹道迭代求解
     * @param track 航迹状态
     * @param bullet_speed 弹速
     * @param target_armor_id 目标装甲板ID
     * @return [yaw, pitch, tof, distance]
     */
    [[nodiscard]] Eigen::Vector4d ballistic_solve(
        const L3::TrackState& track, double bullet_speed, int target_armor_id) const noexcept;

    /**
     * @brief 计算飞行时间
     */
    [[nodiscard]] double
        compute_tof(double distance, double pitch, double bullet_speed) const noexcept;

    /**
     * @brief 计算重力补偿后的 pitch
     */
    [[nodiscard]] double compute_pitch(
        const Eigen::Vector3d& target_xyz, double bullet_speed, double tof) const noexcept;

    // ======== 装甲板选择 ========

    /**
     * @brief 选择目标装甲板
     * @param track 航迹状态
     * @return 装甲板 ID
     */
    [[nodiscard]] int select_armor(const L3::TrackState& track) noexcept;

    /**
     * @brief 判断是否小陀螺
     */
    [[nodiscard]] bool is_spinning(const L3::TrackState& track) const noexcept;

    // ======== 发射判定 ========

    /**
     * @brief 判断是否应该发射
     */
    [[nodiscard]] bool should_fire(
        const WeaponCommand& cmd, const L3::TrackState& track, double gimbal_yaw) const noexcept;
};

} // namespace fcs::L5
