#pragma once

#include <Eigen/Core>
#include <array>
#include <cstdint>

namespace auto_aim {

/**
 * @brief 装甲板3D位姿 (仅存储camera_optical坐标系原始数据)
 *
 * ========== 设计原则 ==========
 * 1. 只保存Solver产生的原始PnP数据
 * 2. 其他坐标系数据通过fast_tf按需查询
 * 3. timestamp_ns用于tf时间插值查询
 *
 * 坐标系定义：
 * - camera_optical: OpenCV/PnP输出 (x右, y下, z前)
 *
 * 使用时通过fast_tf查询其他坐标系:
 * ```cpp
 * auto& tf = tf_manager.get_singleton();
 * auto T = tf.get_camera_optical_to_world_at(pose3d.timestamp_ns);
 * Eigen::Vector3d xyz_world = T.transform_point(pose3d.xyz_in_camera);
 * ```
 * ================================
 */
struct ArmorPose3D {
    // ===== 时间戳 (用于tf查询) =====
    uint64_t timestamp_ns{0};

    // ===== Solver 填充 (camera_optical_frame 坐标系) =====
    Eigen::Vector3d xyz_in_camera{0, 0, 0};                   // [camera_optical系] 位置 (m)
    Eigen::Matrix3d R_in_camera{Eigen::Matrix3d::Identity()}; // [camera_optical系] 旋转矩阵
    double yaw_raw{0};                                        // PnP原始yaw值 (rad)

    bool valid{false};                                        // 位姿是否有效
};

using Poses3D = std::array<ArmorPose3D, 10>;

} // namespace auto_aim
