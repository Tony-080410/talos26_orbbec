#include "aimer.hpp"

#include <cmath>
#include <vector>

#include "solver.hpp"     // math::limit_rad
#include "trajectory.hpp" // solve_trajectory

namespace auto_aim {

namespace {
/**
 * @brief 计算时间差 (秒)
 */
double delta_time(
    std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) noexcept {
    return std::chrono::duration<double>(a - b).count();
}
} // namespace

Aimer::Aimer(const AimerConfig& config) noexcept
    : config_(config) {}

GimbalCommand Aimer::aim(
    Target& target, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    const bool to_now) noexcept {
    // 检查弹速合法性
    if (bullet_speed < 14) {
        bullet_speed = 23.0;
    }

    // 确定延时补偿
    // ekf_x()[7] = w (角速度)
    const Eigen::VectorXd ekf_x = target.ekf_x();
    const double delay_time     = std::abs(ekf_x[7]) > config_.decision_speed
                                    ? config_.high_speed_delay_time
                                    : config_.low_speed_delay_time;

    // 预测目标到指定时刻
    auto future = timestamp;
    if (to_now) {
        const double dt = delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
        future += std::chrono::microseconds(static_cast<int>(dt * 1e6));
    } else {
        const double dt = 0.005 + delay_time; // detector-aimer 耗时 0.005 + 发弹延时
        future += std::chrono::microseconds(static_cast<int>(dt * 1e6));
    }
    target.predict(future);

    // 初始瞄准点计算
    const auto aim_point0 = choose_aim_point(target);
    debug_aim_point_      = aim_point0;
    if (!aim_point0.valid) {
        return {false, false, 0, 0, 0};
    }

    // 初始弹道计算
    const Eigen::Vector3d xyz0 = aim_point0.xyza.head<3>();
    const double d0            = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
    auto traj0                 = solve_trajectory(bullet_speed, d0, xyz0[2]);
    if (!traj0) {
        debug_aim_point_.valid = false;
        return {false, false, 0, 0, 0};
    }

    // 迭代求解飞行时间 (最多10次, 收敛条件: 相邻两次 fly_time 差 < 0.001s)
    double prev_fly_time          = traj0->fly_time;
    TrajectoryResult current_traj = *traj0;
    std::vector<Target> iteration_targets(10, target);

    for (int iter = 0; iter < 10; ++iter) {
        // 预测目标在 future + prev_fly_time 时刻的位置
        const auto predict_time =
            future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
        iteration_targets[iter].predict(predict_time);

        // 计算瞄准点
        const auto aim_point = choose_aim_point(iteration_targets[iter]);
        debug_aim_point_     = aim_point;
        if (!aim_point.valid) {
            return {false, false, 0, 0, 0};
        }

        // 计算新弹道
        const Eigen::Vector3d xyz = aim_point.xyza.head<3>();
        const double d            = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
        auto traj                 = solve_trajectory(bullet_speed, d, xyz.z());
        if (!traj) {
            debug_aim_point_.valid = false;
            return {false, false, 0, 0, 0};
        }
        current_traj = *traj;

        // 检查收敛条件
        if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
            break;
        }
        prev_fly_time = current_traj.fly_time;
    }

    // 计算最终角度
    const Eigen::Vector3d final_xyz = debug_aim_point_.xyza.head<3>();
    const double yaw                = std::atan2(final_xyz.y(), final_xyz.x()) + config_.yaw_offset;
    const double pitch =
        -(current_traj.pitch + config_.pitch_offset); // 世界坐标系下 pitch 向上为负
    const double distance = std::sqrt(
        final_xyz.x() * final_xyz.x() + final_xyz.y() * final_xyz.y()
        + final_xyz.z() * final_xyz.z());

    return {true, false, yaw, pitch, distance};
}

AimPoint Aimer::choose_aim_point(const Target& target) noexcept {
    const Eigen::VectorXd ekf_x                        = target.ekf_x();
    const std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
    const auto armor_num                               = static_cast<int>(armor_xyza_list.size());

    // 如果装甲板未发生过跳变, 只有当前装甲板的位置已知
    if (!target.jumped) {
        return {true, armor_xyza_list[0]};
    }

    // 整车旋转中心的球坐标 yaw
    const double center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

    // 计算各装甲板相对车中心的角度差
    std::vector<double> delta_angle_list;
    delta_angle_list.reserve(armor_num);
    for (int i = 0; i < armor_num; i++) {
        const double delta_angle = math::limit_rad(armor_xyza_list[i][3] - center_yaw);
        delta_angle_list.push_back(delta_angle);
    }

    // 非小陀螺模式 (角速度 <= 2 rad/s 且不是前哨站)
    // ekf_x[7] = w (角速度)
    if (std::abs(ekf_x[7]) <= 2 && target.name != talos::ArmorName::outpost) {
        // 选择在可射击范围内的装甲板 (±60度)
        std::vector<int> id_list;
        constexpr double shootable_angle = 60.0 / 57.3;
        for (int i = 0; i < armor_num; i++) {
            if (std::abs(delta_angle_list[i]) > shootable_angle) {
                continue;
            }
            id_list.push_back(i);
        }

        // 无有效装甲板
        if (id_list.empty()) {
            return {false, armor_xyza_list[0]};
        }

        // 锁定模式: 防止在两个都呈45度的装甲板之间来回切换
        if (id_list.size() > 1) {
            const int id0 = id_list[0];
            const int id1 = id_list[1];

            // 未处于锁定模式时, 选择 delta_angle 绝对值较小的装甲板, 进入锁定模式
            if (lock_id_ != id0 && lock_id_ != id1) {
                lock_id_ =
                    (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;
            }

            return {true, armor_xyza_list[lock_id_]};
        }

        // 只有一个装甲板在可射击范围内时, 退出锁定模式
        lock_id_ = -1;
        return {true, armor_xyza_list[id_list[0]]};
    }

    // 小陀螺模式
    double coming_angle;
    double leaving_angle;
    if (target.name == talos::ArmorName::outpost) {
        coming_angle  = 70.0 / 57.3;
        leaving_angle = 30.0 / 57.3;
    } else {
        coming_angle  = config_.comming_angle;
        leaving_angle = config_.leaving_angle;
    }

    // 在小陀螺时, 一侧的装甲板不断出现, 另一侧的装甲板不断消失
    // 选择正在出现的装甲板 (被打中的概率更高)
    for (int i = 0; i < armor_num; i++) {
        if (std::abs(delta_angle_list[i]) > coming_angle) {
            continue;
        }
        // 顺时针旋转 (w > 0): 选择正角度侧
        if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) {
            return {true, armor_xyza_list[i]};
        }
        // 逆时针旋转 (w < 0): 选择负角度侧
        if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) {
            return {true, armor_xyza_list[i]};
        }
    }

    return {false, armor_xyza_list[0]};
}

} // namespace auto_aim
