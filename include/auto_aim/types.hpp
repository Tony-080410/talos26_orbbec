#pragma once

#include <Eigen/Core>
#include <array>

namespace auto_aim {

/**
 * @brief 扩展的ArmorSlots结构，包含完整的3D位姿信息
 *
 * ========== 坐标系约定 ==========
 * 所有带后缀的字段都明确标注了坐标系：
 * - xyz_in_gimbal: gimbal坐标系下的位置
 * - xyz_in_camera: camera_optical坐标系下的位置 (PnP原始输出)
 * - xyz_in_world: world坐标系下的位置 (最终使用的坐标系)
 * - ypr_in_gimbal: gimbal坐标系下的欧拉角
 * - ypr_in_world: world坐标系下的欧拉角 (最终使用的坐标系)
 * - ypd_in_world: world坐标系下的球坐标 (yaw, pitch, distance)
 * - R_in_camera: camera_optical坐标系下的旋转矩阵 (PnP原始输出)
 *
 * 坐标系定义：
 * - camera_optical: OpenCV/PnP输出 (x右, y下, z前)
 * - ROS camera: REP0103标准 (x前, y左, z上)
 * - gimbal: 云台坐标系 (与ROS camera对齐)
 * - world: 世界坐标系 (与gimbal动态变换)
 *
 * 坐标系转换链：
 * camera_optical → [R_cameraOptical_to_ros] → ROS camera
 *                → [R_camera2gimbal_] → gimbal
 *                → [R_gimbal2world_] → world
 * ================================
 */
struct ArmorPose3D {
    Eigen::Vector3d xyz_in_gimbal{0, 0, 0}; // [gimbal系] 云台坐标系下的位置 (m)
    Eigen::Vector3d xyz_in_camera{0, 0, 0}; // [camera_optical系] 相机坐标系下的位置 (PnP原始输出)
    Eigen::Vector3d xyz_in_world{0, 0, 0};  // [world系] 世界坐标系下的位置 (m)
    Eigen::Vector3d ypr_in_gimbal{0, 0, 0}; // [gimbal系] 云台坐标系下的欧拉角 (rad)
    Eigen::Matrix3d R_in_camera; // [camera_optical系] 相机坐标系下的旋转矩阵 (PnP原始输出)
    Eigen::Vector3d ypr_in_world{0, 0, 0}; // [world系] 世界坐标系下的欧拉角 (rad)
    Eigen::Vector3d ypd_in_world{0, 0, 0}; // [world系] 球坐标系: yaw, pitch, distance (rad, rad, m)
    double yaw_raw{0};                     // [world系] 优化前的yaw值 (rad)
    bool valid{false};                     // 位姿是否有效
};

using Poses3D = std::array<ArmorPose3D, 10>;

} // namespace auto_aim
