#include "solver.hpp"

#include "fast_tf/frame.hpp"
#include "fast_tf/matrix.hpp"
#include "math.hpp"

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

Solver::Solver(const SpVisionSolverConfig& config) {
    cv::eigen2cv(config.camera_matrix, camera_matrix_);
    cv::eigen2cv(config.distort_coefficient, distort_coeffs_);
}

bool Solver::solve(
    const talos::Armor& armor, ArmorPose3D& pose3d, const uint64_t timestamp_ns) const {
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

    // ========== PnP输出说明 ==========
    // PnP解算结果在camera_optical_frame坐标系 (x右, y下, z前)
    // tvec: object原点在camera_optical_frame中的位置
    // rvec: object坐标系相对于camera_optical_frame的旋转
    // ====================================
    cv::Vec3d rvec, tvec;
    const bool success = cv::solvePnP(
        object_points, image_points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
        cv::SOLVEPNP_IPPE);

    if (!success) {
        pose3d.valid = false;
        return false;
    }

    // 填充帧时间戳 (用于fast_tf查询)
    pose3d.timestamp_ns = timestamp_ns;
    pose3d.pnp          = fast_tf::FuckTransform<fast_tf::camera_optical_fuxk_frame>(
        fast_tf::TransformMatrix4d::from_pnp(rvec, tvec));

    const auto translation = pose3d.pnp->translation();
    // 深度有效性检查
    if (translation[2] <= 0) {
        spdlog::warn("[Solver] PnP depth negative: {:.3f}", translation[2]);
        pose3d.valid = false;
        return false;
    }

    // 距离合理性检查
    if (const double distance = translation.norm(); distance < 0.5 || distance > 15.0) {
        spdlog::warn(
            "[Solver] Unreasonable distance: {:.2f}m (camera_optical: [{:.3f}, {:.3f}, {:.3f}])",
            distance, translation[0], translation[1], translation[2]);
        pose3d.valid = false;
        return false;
    }

    const auto [r, p, y] = math_fuxk::rpy(pose3d.pnp->rotation()).rpy();
    // 提取yaw角 (camera_optical_frame坐标系)
    pose3d.yaw_raw = decltype(pose3d.yaw_raw)(y);

    // NOTE: object变换的发布由Tracker负责，Solver只做PnP解算

    pose3d.valid = true;
    return true;
}

void Solver::solve_all(
    const talos::ArmorSlots& slots, std::array<ArmorPose3D, 10>& poses3d,
    const uint64_t timestamp_ns) const {
    for (int i = 0; i < 10; ++i) {
        if (!slots.is_valid(i)) {
            poses3d[i].valid = false;
            continue;
        }

        if (ArmorPose3D pose3d; solve(slots.armors[i], pose3d, timestamp_ns)) {
            poses3d[i] = pose3d;
        } else {
            poses3d[i].valid = false;
        }
    }
}

// ============ SolverConcept 方法实现 ============

void Solver::solve(const DetectionResult& det, SolveResult& out, DebugData& debug) noexcept {
    const auto* slots = get_armor_slots(det);
    if (!slots) {
        out = std::monostate{};
        return;
    }

    auto& poses = out.emplace<Poses3D>();
    // 使用 debug 中的时间戳，如果为0则使用当前时间
    const uint64_t timestamp_ns =
        debug.timestamp_ns != 0
            ? debug.timestamp_ns
            : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
    solve_all(*slots, poses, timestamp_ns);
}

std::vector<cv::Point2f> Solver::reproject_armor(
    const Eigen::Vector3d& xyz_in_camera, const Eigen::Matrix3d& R_in_camera,
    const talos::ArmorType type) const {
    // 转换为OpenCV格式
    cv::Vec3d rvec;
    cv::Mat R_cv;
    cv::eigen2cv(R_in_camera, R_cv);
    cv::Rodrigues(R_cv, rvec);
    const cv::Vec3d tvec(xyz_in_camera[0], xyz_in_camera[1], xyz_in_camera[2]);

    // 重投影
    std::vector<cv::Point2f> image_points;
    const auto& object_points =
        type == talos::ArmorType::big ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);

    return image_points;
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
