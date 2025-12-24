#pragma once

#include <Eigen/Core>
#include <array>

namespace auto_aim {

/**
 * @brief 扩展的ArmorSlots结构，包含完整的3D位姿信息
 *
 * ========== 坐标系约定 ==========
 * 数据来源分工:
 * - Solver: 填充 camera_optical_frame 坐标系数据 (PnP直接输出)
 *   - xyz_in_camera, R_in_camera
 * - Tracker: 通过 fast_tf 查询填充其他坐标系数据
 *   - xyz_in_gimbal, ypr_in_gimbal
 *   - xyz_in_world, ypr_in_world, ypd_in_world
 *
 * 坐标系定义：
 * - camera_optical: OpenCV/PnP输出 (x右, y下, z前)
 * - gimbal_link: 云台坐标系 (ROS标准: x前, y左, z上)
 * - world/odom: 世界坐标系
 *
 * 坐标系转换通过 fast_tf 完成:
 * object → camera_optical_frame → camera_link → gimbal_link → odom → world
 * ================================
 */
struct ArmorPose3D {
    // ===== Solver 填充 (camera_optical_frame 坐标系) =====
    Eigen::Vector3d xyz_in_camera{0, 0, 0}; // [camera_optical系] 相机坐标系下的位置 (PnP原始输出)
    Eigen::Matrix3d R_in_camera; // [camera_optical系] 相机坐标系下的旋转矩阵 (PnP原始输出)
    double yaw_raw{0};           // PnP原始yaw值 (rad)

    // ===== Tracker 填充 (通过 fast_tf 转换) =====
    Eigen::Vector3d xyz_in_gimbal{0, 0, 0}; // [gimbal_link系] 云台坐标系下的位置 (m)
    Eigen::Vector3d ypr_in_gimbal{0, 0, 0}; // [gimbal_link系] 云台坐标系下的欧拉角 (rad)
    Eigen::Vector3d xyz_in_world{0, 0, 0};  // [world系] 世界坐标系下的位置 (m)
    Eigen::Vector3d ypr_in_world{0, 0, 0};  // [world系] 世界坐标系下的欧拉角 (rad)
    Eigen::Vector3d ypd_in_world{0, 0, 0}; // [world系] 球坐标系: yaw, pitch, distance (rad, rad, m)

    bool valid{false};                     // 位姿是否有效
};

using Poses3D = std::array<ArmorPose3D, 10>;

} // namespace auto_aim
