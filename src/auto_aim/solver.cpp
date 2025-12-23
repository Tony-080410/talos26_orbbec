#include "solver.hpp"
#include <opencv2/core/eigen.hpp>

namespace auto_aim {

// 装甲板3D点（物体坐标系）
const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
    {0,  BIG_ARMOR_WIDTH / 2.0,  LIGHTBAR_LENGTH / 2.0},
    {0, -BIG_ARMOR_WIDTH / 2.0,  LIGHTBAR_LENGTH / 2.0},
    {0, -BIG_ARMOR_WIDTH / 2.0, -LIGHTBAR_LENGTH / 2.0},
    {0,  BIG_ARMOR_WIDTH / 2.0, -LIGHTBAR_LENGTH / 2.0}
};

const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
    {0,  SMALL_ARMOR_WIDTH / 2.0,  LIGHTBAR_LENGTH / 2.0},
    {0, -SMALL_ARMOR_WIDTH / 2.0,  LIGHTBAR_LENGTH / 2.0},
    {0, -SMALL_ARMOR_WIDTH / 2.0, -LIGHTBAR_LENGTH / 2.0},
    {0,  SMALL_ARMOR_WIDTH / 2.0, -LIGHTBAR_LENGTH / 2.0}
};

// ============ Solver ============

Solver::Solver(const SpVisionSolverConfig& config)
    : R_gimbal2world_(Eigen::Matrix3d::Identity()) {
    // 转换外参矩阵
    R_gimbal2imubody_ = config.extrinsic_gimbal_imu;
    R_camera2gimbal_  = config.extrinsic_camera_gimbal;
    t_camera2gimbal_  = config.extrinsic_camera_gimbal_tf;

    cv::eigen2cv(config.camera_matrix, camera_matrix_);
    cv::eigen2cv(config.distort_coefficient, distort_coeffs_);

    // 初始化camera_optical到ROS标准相机坐标系的转换矩阵
    // camera_optical (OpenCV/PnP): x右, y下, z前
    // ROS标准相机: x前, y左, z上
    R_cameraOptical_to_ros_ << 0, 0, 1, -1, 0, 0, 0, -1, 0;
}

void Solver::set_R_gimbal2world(const Eigen::Quaterniond& q) {
    const Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

bool Solver::solve(const talos::Armor& armor, ArmorPose3D& pose3d, const bool optimize_yaw) const {
    // 从armor.pts提取角点
    std::vector<cv::Point2f> image_points(4);
    image_points[0] = {armor.pts[0], armor.pts[1]}; // tl
    image_points[1] = {armor.pts[2], armor.pts[3]}; // tr
    image_points[2] = {armor.pts[4], armor.pts[5]}; // br
    image_points[3] = {armor.pts[6], armor.pts[7]}; // bl

    // 选择物体点
    const auto type = armor.type;
    const auto& object_points =
        type == talos::ArmorType::big ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

    // PnP解算
    cv::Vec3d rvec, tvec;
    const bool success = cv::solvePnP(
        object_points, image_points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
        cv::SOLVEPNP_IPPE);

    if (!success) {
        pose3d.valid = false;
        return false;
    }

    // ========== 坐标系转换说明 ==========
    // PnP解算结果在camera_optical坐标系 (x右, y下, z前)
    // 目标：转换到world坐标系 (ROS标准: x前, y左, z上)
    // 转换链：camera_optical → ROS camera → gimbal → world
    // ========================================

    // 位置转换：camera_optical → ROS camera → gimbal → world
    Eigen::Vector3d xyz_in_camera_optical;
    cv::cv2eigen(tvec, xyz_in_camera_optical);

    pose3d.xyz_in_gimbal = R_cameraOptical_to_ros_ * xyz_in_camera_optical + t_camera2gimbal_;
    pose3d.xyz_in_world  = R_gimbal2world_ * pose3d.xyz_in_gimbal;

    // 深度有效性检查
    assert(xyz_in_camera_optical[2] > 0 && "PnP depth must be positive (camera_optical z-axis)");

    // 距离合理性检查
    const double distance = pose3d.xyz_in_world.norm();
    if (distance < 0.5 || distance > 15.0) {
        spdlog::warn("[Solver] Unreasonable distance: {:.2f}m (armor at camera_optical: [{:.3f}, {:.3f}, {:.3f}])",
                     distance, xyz_in_camera_optical[0], xyz_in_camera_optical[1], xyz_in_camera_optical[2]);
        pose3d.valid = false;
        return false;
    }

    // 姿态转换：camera_optical → ROS camera → gimbal → world
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    Eigen::Matrix3d R_armor2camera_optical;  // 装甲板在camera_optical系的姿态
    cv::cv2eigen(rmat, R_armor2camera_optical);

    // CRITICAL: 应用camera_optical到ROS标准系的转换
    const Eigen::Matrix3d R_armor2camera_ros = R_cameraOptical_to_ros_ * R_armor2camera_optical;
    const Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera_ros;
    const Eigen::Matrix3d R_armor2world  = R_gimbal2world_ * R_armor2gimbal;

    // 旋转矩阵有效性检查
    assert(std::abs(R_armor2camera_optical.determinant() - 1.0) < 1e-6 && "Rotation matrix must have determinant 1");
    assert(std::abs(R_armor2camera_ros.determinant() - 1.0) < 1e-6 && "Transformed rotation matrix must have determinant 1");

    // 填充pose3d结构
    pose3d.R_in_camera   = R_armor2camera_optical;  // [camera_optical系] 用于调试
    pose3d.xyz_in_camera = xyz_in_camera_optical;   // [camera_optical系] 用于调试
    pose3d.ypr_in_gimbal = math::eulers(R_armor2gimbal, 2, 1, 0);  // [gimbal系]
    pose3d.ypr_in_world  = math::eulers(R_armor2world, 2, 1, 0);   // [world系]
    pose3d.ypd_in_world  = math::xyz2ypd(pose3d.xyz_in_world);     // [world系, 球坐标]

    if (optimize_yaw) {
        this->optimize_yaw(armor, pose3d);
    } else {
        pose3d.yaw_raw = pose3d.ypr_in_world[0];
    }

    pose3d.valid = true;
    return true;
}

void Solver::solve_all(const talos::ArmorSlots& slots, std::array<ArmorPose3D, 10>& poses3d) const {
    for (int i = 0; i < 10; ++i) {
        if (!slots.is_valid(i)) {
            poses3d[i].valid = false;
            continue;
        }

        if (ArmorPose3D pose3d; solve(slots.armors[i], pose3d, true)) {
            poses3d[i] = pose3d;
        } else {
            poses3d[i].valid = false;
        }
    }
}

std::vector<cv::Point2f> Solver::reproject_armor(
    const Eigen::Vector3d& xyz_in_world, const double yaw, const talos::ArmorType type,
    const talos::ArmorName name) const {
    const auto sin_yaw = std::sin(yaw);
    auto cos_yaw       = std::cos(yaw);

    // 装甲板pitch角：前哨站向下，其他向上
    const auto pitch =
        name == talos::ArmorName::outpost ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
    const auto sin_pitch = std::sin(pitch);
    auto cos_pitch       = std::cos(pitch);

    // clang-format off
    const Eigen::Matrix3d R_armor2world {
        {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
        {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
        {         -sin_pitch,        0,           cos_pitch}
    };
    // clang-format on

    // ========== 重投影坐标系转换说明 ==========
    // 目标：将world系的装甲板姿态转换回camera_optical系用于OpenCV投影
    // 转换链：world → gimbal → ROS camera → camera_optical
    // ==========================================

    // 计算 R_armor2camera_ros 和 t_armor2camera_ros (ROS camera系)
    const Eigen::Vector3d& t_armor2world = xyz_in_world;
    const Eigen::Matrix3d R_armor2camera_ros =
        R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
    Eigen::Vector3d t_armor2camera_ros =
        R_camera2gimbal_.transpose()
        * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

    // CRITICAL FIX: 转换到camera_optical系用于OpenCV投影
    // ROS camera → camera_optical
    const Eigen::Matrix3d R_armor2camera_optical =
        R_cameraOptical_to_ros_.transpose() * R_armor2camera_ros;
    const Eigen::Vector3d t_armor2camera_optical =
        R_cameraOptical_to_ros_.transpose() * t_armor2camera_ros;

    // 深度有效性检查
    if (t_armor2camera_optical[2] <= 0) {
        spdlog::warn("[Solver] Reprojection depth negative: {:.3f} (world pos: [{:.3f}, {:.3f}, {:.3f}])",
                     t_armor2camera_optical[2], xyz_in_world[0], xyz_in_world[1], xyz_in_world[2]);
        return {};  // 返回空向量表示投影失败
    }

    // 转换为OpenCV格式 (使用camera_optical系的转换)
    cv::Vec3d rvec;
    cv::Mat R_armor2camera_cv;
    cv::eigen2cv(R_armor2camera_optical, R_armor2camera_cv);
    cv::Rodrigues(R_armor2camera_cv, rvec);
    const cv::Vec3d tvec(t_armor2camera_optical[0], t_armor2camera_optical[1], t_armor2camera_optical[2]);

    // 重投影
    std::vector<cv::Point2f> image_points;
    const auto& object_points =
        type == talos::ArmorType::big ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);

    return image_points;
}

std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f>& world_points) const {
    // ========== 世界到相机坐标系转换说明 ==========
    // 目标：将world系点转换回camera_optical系用于OpenCV投影
    // 转换链：world → gimbal → ROS camera → camera_optical
    // ==========================================

    // world → ROS camera
    const Eigen::Matrix3d R_world2camera_ros =
        R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
    const Eigen::Vector3d t_world2camera_ros = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

    // CRITICAL FIX: ROS camera → camera_optical
    const Eigen::Matrix3d R_world2camera_optical =
        R_cameraOptical_to_ros_.transpose() * R_world2camera_ros;
    const Eigen::Vector3d t_world2camera_optical =
        R_cameraOptical_to_ros_.transpose() * t_world2camera_ros;

    cv::Mat rvec, tvec;
    cv::eigen2cv(R_world2camera_optical, rvec);
    cv::eigen2cv(t_world2camera_optical, tvec);

    // 过滤掉相机后方的点
    std::vector<cv::Point3f> valid_world_points;
    for (const auto& world_point : world_points) {
        Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);

        if (Eigen::Vector3d camera_point = R_world2camera_optical * world_point_eigen + t_world2camera_optical;
            camera_point.z() > 0) {
            valid_world_points.push_back(world_point);
        }
    }

    if (valid_world_points.empty()) {
        return {};
    }

    std::vector<cv::Point2f> pixel_points;
    cv::projectPoints(
        valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixel_points);
    return pixel_points;
}

void Solver::optimize_yaw(const talos::Armor& armor, ArmorPose3D& pose3d) const {
    Eigen::Vector3d gimbal_ypr = math::eulers(R_gimbal2world_, 2, 1, 0);

    constexpr double SEARCH_RANGE = 140; // degree
    const auto yaw0 = math::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

    auto min_error = 1e10;
    auto best_yaw  = pose3d.ypr_in_world[0];

    const auto type = static_cast<talos::ArmorType>(armor.type);
    const auto name = static_cast<talos::ArmorName>(armor.name);

    for (int i = 0; i < static_cast<int>(SEARCH_RANGE); i++) {
        const double yaw = math::limit_rad(yaw0 + i * CV_PI / 180.0);

        if (const auto error =
                armor_reprojection_error(armor, pose3d.xyz_in_world, yaw, type, name);
            error < min_error) {
            min_error = error;
            best_yaw  = yaw;
        }
    }

    pose3d.yaw_raw         = pose3d.ypr_in_world[0];
    pose3d.ypr_in_world[0] = best_yaw;
}

double Solver::armor_reprojection_error(
    const talos::Armor& armor, const Eigen::Vector3d& xyz_in_world, const double yaw,
    const talos::ArmorType type, const talos::ArmorName name) const {
    const auto image_points = reproject_armor(xyz_in_world, yaw, type, name);

    // 提取原始角点
    const std::array original_points = {
        cv::Point2f{armor.pts[0], armor.pts[1]},
        cv::Point2f{armor.pts[2], armor.pts[3]},
        cv::Point2f{armor.pts[4], armor.pts[5]},
        cv::Point2f{armor.pts[6], armor.pts[7]}
    };

    double error = 0.0;
    for (int i = 0; i < 4; i++) {
        error += cv::norm(original_points[i] - image_points[i]);
    }

    return error;
}

// ============ 数学工具函数 ============

namespace math {

double limit_rad(double angle) {
    while (angle > CV_PI)
        angle -= 2 * CV_PI;
    while (angle <= -CV_PI)
        angle += 2 * CV_PI;
    return angle;
}

Eigen::Vector3d eulers(
    const Eigen::Quaterniond& q, int axis0, const int axis1, int axis2, const bool extrinsic) {
    if (!extrinsic)
        std::swap(axis0, axis2);

    const auto i         = axis0;
    const auto j         = axis1;
    auto k               = axis2;
    const auto is_proper = i == k;
    if (is_proper)
        k = 3 - i - j;
    const auto sign = (i - j) * (j - k) * (k - i) / 2;

    double a, b, c, d;
    Eigen::Vector4d xyzw = q.coeffs();
    if (is_proper) {
        a = xyzw[3];
        b = xyzw[i];
        c = xyzw[j];
        d = xyzw[k] * sign;
    } else {
        a = xyzw[3] - xyzw[j];
        b = xyzw[i] + xyzw[k] * sign;
        c = xyzw[j] + xyzw[3];
        d = xyzw[k] * sign - xyzw[i];
    }

    Eigen::Vector3d result;
    const auto n2 = a * a + b * b + c * c + d * d;
    result[1]     = std::acos(2 * (a * a + b * b) / n2 - 1);

    const auto half_sum  = std::atan2(b, a);
    const auto half_diff = std::atan2(-d, c);

    constexpr auto eps = 1e-7;
    const auto safe1   = std::abs(result[1]) >= eps;
    const auto safe2   = std::abs(result[1] - CV_PI) >= eps;

    if (safe1 && safe2) {
        result[0] = half_sum + half_diff;
        result[2] = half_sum - half_diff;
    } else {
        if (!extrinsic) {
            result[0] = 0;
            if (!safe1)
                result[2] = 2 * half_sum;
            if (!safe2)
                result[2] = -2 * half_diff;
        } else {
            result[2] = 0;
            if (!safe1)
                result[0] = 2 * half_sum;
            if (!safe2)
                result[0] = 2 * half_diff;
        }
    }

    for (int idx = 0; idx < 3; idx++)
        result[idx] = limit_rad(result[idx]);

    if (!is_proper) {
        result[2] *= sign;
        result[1] -= CV_PI / 2;
    }

    if (!extrinsic)
        std::swap(result[0], result[2]);

    return result;
}

Eigen::Vector3d eulers(
    const Eigen::Matrix3d& R, const int axis0, const int axis1, const int axis2,
    const bool extrinsic) {
    const Eigen::Quaterniond q(R);
    return eulers(q, axis0, axis1, axis2, extrinsic);
}

Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz) {
    const auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto yaw      = std::atan2(y, x);
    auto pitch    = std::atan2(z, std::sqrt(x * x + y * y));
    auto distance = std::sqrt(x * x + y * y + z * z);
    return {yaw, pitch, distance};
}

Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d& xyz) {
    const auto x = xyz[0], y = xyz[1], z = xyz[2];

    auto dyaw_dx = -y / (x * x + y * y);
    auto dyaw_dy = x / (x * x + y * y);
    auto dyaw_dz = 0.0;

    auto dpitch_dx = -(x * z) / ((z * z / (x * x + y * y) + 1) * std::pow(x * x + y * y, 1.5));
    auto dpitch_dy = -(y * z) / ((z * z / (x * x + y * y) + 1) * std::pow(x * x + y * y, 1.5));
    auto dpitch_dz = 1 / ((z * z / (x * x + y * y) + 1) * std::pow(x * x + y * y, 0.5));

    auto ddistance_dx = x / std::pow(x * x + y * y + z * z, 0.5);
    auto ddistance_dy = y / std::pow(x * x + y * y + z * z, 0.5);
    auto ddistance_dz = z / std::pow(x * x + y * y + z * z, 0.5);

    Eigen::MatrixXd J{
        {     dyaw_dx,      dyaw_dy,      dyaw_dz},
        {   dpitch_dx,    dpitch_dy,    dpitch_dz},
        {ddistance_dx, ddistance_dy, ddistance_dz}
    };
    return J;
}

Eigen::Vector3d ypd2xyz(const Eigen::Vector3d& ypd) {
    const auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
    auto x = distance * std::cos(pitch) * std::cos(yaw);
    auto y = distance * std::cos(pitch) * std::sin(yaw);
    auto z = distance * std::sin(pitch);
    return {x, y, z};
}

} // namespace math

} // namespace auto_aim
