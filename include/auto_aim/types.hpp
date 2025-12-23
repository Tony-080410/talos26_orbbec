#pragma once

#include <Eigen/Core>
#include <array>

namespace auto_aim {

    /**
     * @brief 扩展的ArmorSlots结构，包含完整的3D位姿信息
     */
    struct ArmorPose3D {
        Eigen::Vector3d xyz_in_gimbal{0, 0, 0}; // 云台坐标系下的位置 (m)
        Eigen::Vector3d xyz_in_camera{0, 0, 0};
        Eigen::Vector3d xyz_in_world{0, 0, 0};  // 世界坐标系下的位置 (m)
        Eigen::Vector3d ypr_in_gimbal{0, 0, 0}; // 云台坐标系下的欧拉角 (rad)
        Eigen::Matrix3d R_in_camera;
        Eigen::Vector3d ypr_in_world{0, 0, 0};  // 世界坐标系下的欧拉角 (rad)
        Eigen::Vector3d ypd_in_world{0, 0, 0};  // 球坐标系: yaw, pitch, distance
        double yaw_raw{0};                      // 优化前的yaw值 (rad)
        bool valid{false};
    };

    using Poses3D = std::array<ArmorPose3D, 10>;

} // namespace auto_aim
