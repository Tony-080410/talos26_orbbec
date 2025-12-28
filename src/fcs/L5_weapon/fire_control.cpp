#include "fcs/L5_weapon/fire_control.hpp"
#include "fcs/L2_perception/solver.hpp" // for math::normalize_angle, xyz_to_ypd

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace fcs::L5 {

// ============================================================================
// 内部常量
// ============================================================================

namespace {

constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;

// 可射击角度范围 (约60度)
constexpr double SHOOTABLE_ANGLE = 60.0 * DEG_TO_RAD;

} // namespace

// ============================================================================
// FireControl 实现
// ============================================================================

FireControl::FireControl(const FireControlConfig& config)
    : config_(config) {}

WeaponCommand FireControl::solve(
    const L3::TrackState& track, double bullet_speed, double gimbal_yaw,
    uint64_t /*timestamp_ns*/) noexcept {
    WeaponCommand cmd;
    cmd.valid = false;

    // 弹速合法性检查
    if (bullet_speed < 14.0) {
        bullet_speed = 23.0;
    }

    // 选择目标装甲板
    const int target_id = select_armor(track);
    if (target_id < 0) {
        spdlog::debug("[FireControl] no valid armor to aim");
        return cmd;
    }

    // 弹道求解
    const auto result = ballistic_solve(track, bullet_speed, target_id);
    if (result[2] <= 0) { // tof <= 0 表示求解失败
        spdlog::debug("[FireControl] ballistic solve failed");
        return cmd;
    }

    // 填充命令
    cmd.yaw      = result[0] + config_.yaw_offset;
    cmd.pitch    = result[1] + config_.pitch_offset;
    cmd.tof      = result[2];
    cmd.distance = result[3];

    // 计算前馈 (yaw_rate 基于角速度预测)
    const double omega = track.omega();
    const double r     = track.radius();
    // 近似: 装甲板切向速度 = omega * r
    // yaw_rate ≈ v_tangent / distance * cos(装甲板角度偏移)
    if (cmd.distance > 0.5) {
        const double v_tangent = omega * r;
        cmd.yaw_rate           = v_tangent / cmd.distance;
    }
    cmd.pitch_rate = 0.0; // 通常不需要 pitch 前馈

    // 发射判定
    cmd.fire  = should_fire(cmd, track, gimbal_yaw);
    cmd.valid = true;

    last_cmd_      = cmd;
    lock_armor_id_ = target_id;

    return cmd;
}

// ============================================================================
// 弹道求解
// ============================================================================

Eigen::Vector4d FireControl::ballistic_solve(
    const L3::TrackState& track, double bullet_speed, int target_armor_id) const noexcept {
    // 获取目标装甲板位置
    const Eigen::Vector4d xyza = track.armor_xyza(target_armor_id);
    Eigen::Vector3d target_xyz = xyza.head<3>();

    // 初始弹道计算
    const double d0 = std::sqrt(target_xyz[0] * target_xyz[0] + target_xyz[1] * target_xyz[1]);
    double tof      = compute_tof(d0, 0.0, bullet_speed);

    if (tof <= 0) {
        return {0, 0, 0, 0};
    }

    // 迭代求解 (考虑目标运动)
    L3::PredictedState predicted;
    predicted.state = track;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // 预测目标位置
        predicted                       = predicted.predict_dt(tof);
        const Eigen::Vector4d pred_xyza = predicted.state.armor_xyza(target_armor_id);
        target_xyz                      = pred_xyza.head<3>();

        // 计算新弹道
        const double d = std::sqrt(target_xyz[0] * target_xyz[0] + target_xyz[1] * target_xyz[1]);
        const double pitch   = compute_pitch(target_xyz, bullet_speed, tof);
        const double new_tof = compute_tof(d, pitch, bullet_speed);

        // 检查收敛
        if (std::abs(new_tof - tof) < config_.convergence_threshold) {
            const double yaw      = std::atan2(target_xyz[1], target_xyz[0]);
            const double distance = target_xyz.norm();
            return {yaw, -pitch, new_tof, distance}; // pitch 取负 (世界坐标系向上为负)
        }

        tof = new_tof;
    }

    // 未收敛，返回最后结果
    const double yaw      = std::atan2(target_xyz[1], target_xyz[0]);
    const double pitch    = compute_pitch(target_xyz, bullet_speed, tof);
    const double distance = target_xyz.norm();
    return {yaw, -pitch, tof, distance};
}

double FireControl::compute_tof(double distance, double pitch, double bullet_speed) const noexcept {
    if (distance <= 0 || bullet_speed <= 0) {
        return -1.0;
    }

    // 简单模型: t = d / (v * cos(pitch))
    const double v_horizontal = bullet_speed * std::cos(pitch);
    if (v_horizontal <= 0) {
        return -1.0;
    }

    return distance / v_horizontal;
}

double FireControl::compute_pitch(
    const Eigen::Vector3d& target_xyz, double bullet_speed, double tof) const noexcept {
    // 带重力补偿的 pitch 计算
    // h = v*sin(pitch)*t - 0.5*g*t^2
    // pitch = asin((h + 0.5*g*t^2) / (v*t))

    const double d = std::sqrt(target_xyz[0] * target_xyz[0] + target_xyz[1] * target_xyz[1]);
    const double h = target_xyz[2];

    if (tof <= 0 || bullet_speed <= 0) {
        return 0.0;
    }

    // 二次方程求解 (抛物线弹道)
    // a*tan^2(theta) + b*tan(theta) + c = 0
    const double g     = config_.gravity;
    const double a     = g * d * d / (2 * bullet_speed * bullet_speed);
    const double b     = -d;
    const double c     = a + h;
    const double delta = b * b - 4 * a * c;

    if (delta < 0) {
        return 0.0;
    }

    // 选择低弹道解
    const double sqrt_delta  = std::sqrt(delta);
    const double tan_pitch_1 = (-b + sqrt_delta) / (2 * a);
    const double tan_pitch_2 = (-b - sqrt_delta) / (2 * a);
    const double pitch_1     = std::atan(tan_pitch_1);
    const double pitch_2     = std::atan(tan_pitch_2);

    // 计算对应的飞行时间，选择较短的
    const double t_1 = d / (bullet_speed * std::cos(pitch_1));
    const double t_2 = d / (bullet_speed * std::cos(pitch_2));

    return (t_1 < t_2) ? pitch_1 : pitch_2;
}

// ============================================================================
// 装甲板选择
// ============================================================================

int FireControl::select_armor(const L3::TrackState& track) noexcept {
    const int armor_num = track.armor_num;

    // 如果装甲板未发生过跳变，只能瞄准当前装甲板
    if (!track.jumped) {
        return 0;
    }

    // 整车旋转中心的球坐标 yaw
    const double center_yaw = std::atan2(track.pos_y(), track.pos_x());

    // 计算各装甲板相对车中心的角度差
    std::array<double, 4> delta_angles{};
    for (int i = 0; i < armor_num; ++i) {
        const Eigen::Vector4d xyza = track.armor_xyza(i);
        delta_angles[i]            = L2::math::normalize_angle(xyza[3] - center_yaw);
    }

    // 非小陀螺模式
    if (!is_spinning(track)) {
        // 选择在可射击范围内的装甲板
        std::array<int, 4> valid_ids{};
        int valid_count = 0;

        for (int i = 0; i < armor_num; ++i) {
            if (std::abs(delta_angles[i]) <= SHOOTABLE_ANGLE) {
                valid_ids[valid_count++] = i;
            }
        }

        if (valid_count == 0) {
            return -1;
        }

        // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
        if (valid_count > 1) {
            const int id0 = valid_ids[0];
            const int id1 = valid_ids[1];

            // 未锁定时，选择 delta_angle 绝对值较小的装甲板
            if (lock_armor_id_ != id0 && lock_armor_id_ != id1) {
                lock_armor_id_ =
                    (std::abs(delta_angles[id0]) < std::abs(delta_angles[id1])) ? id0 : id1;
            }
            return lock_armor_id_;
        }

        // 只有一个装甲板有效时，退出锁定模式
        lock_armor_id_ = -1;
        return valid_ids[0];
    }

    // 小陀螺模式
    const double coming_angle =
        (track.name == ArmorName::Outpost) ? (70.0 * DEG_TO_RAD) : config_.coming_angle;
    const double leaving_angle =
        (track.name == ArmorName::Outpost) ? (30.0 * DEG_TO_RAD) : config_.leaving_angle;

    const double omega = track.omega();

    for (int i = 0; i < armor_num; ++i) {
        if (std::abs(delta_angles[i]) > coming_angle) {
            continue;
        }

        // 顺时针旋转 (omega > 0): 选择正角度侧
        if (omega > 0 && delta_angles[i] < leaving_angle) {
            return i;
        }
        // 逆时针旋转 (omega < 0): 选择负角度侧
        if (omega < 0 && delta_angles[i] > -leaving_angle) {
            return i;
        }
    }

    return -1;
}

bool FireControl::is_spinning(const L3::TrackState& track) const noexcept {
    // 前哨站始终视为小陀螺
    if (track.name == ArmorName::Outpost) {
        return true;
    }
    return std::abs(track.omega()) > config_.spinning_threshold;
}

// ============================================================================
// 发射判定
// ============================================================================

bool FireControl::should_fire(
    const WeaponCommand& cmd, const L3::TrackState& track, double gimbal_yaw) const noexcept {
    if (!config_.auto_fire) {
        return false;
    }

    // 航迹未确认不发射
    if (track.status != TrackStatus::Confirmed) {
        return false;
    }

    // 根据距离选择容差
    const double tolerance = (cmd.distance > config_.distance_threshold)
                               ? config_.fire_tolerance_far
                               : config_.fire_tolerance_near;

    // 检查命令稳定性
    const bool command_stable =
        std::abs(L2::math::normalize_angle(last_cmd_.yaw - cmd.yaw)) < tolerance * 2;

    // 检查云台对准
    const bool gimbal_aligned =
        std::abs(L2::math::normalize_angle(gimbal_yaw - last_cmd_.yaw)) < tolerance;

    return command_stable && gimbal_aligned;
}

} // namespace fcs::L5
