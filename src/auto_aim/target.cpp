#include "target.hpp"

#include "fast_tf/frame.hpp"

#include <ranges>

namespace auto_aim {

Target::Target(
    const talos::Armor& armor, const ArmorPose3D& pose3d,
    const std::reference_wrapper<fast_tf::CoordinateSystem<100>> tf_buffer,
    const uint64_t timestamp_ns, const double radius, const int armor_num,
    const Eigen::VectorXd& P0_dig)
    : name(static_cast<talos::ArmorName>(armor.name))
    , armor_type(static_cast<talos::ArmorType>(armor.type))
    , armor_num_(armor_num)
    , tf_buffer(tf_buffer)
    , timestamp_ns_(timestamp_ns) {
    // ========== 坐标系说明 ==========
    // 输入数据在camera_optical坐标系，通过tf_buffer查询world坐标
    // 状态向量表示旋转中心在world系的位置和速度
    // ================================

    // 输入参数有效性检查
    assert(pose3d.valid && "pose3d must be valid");
    assert(radius > 0.05 && radius < 0.5 && "Radius must be in range (0.05, 0.5)m");
    assert((armor_num == 2 || armor_num == 3 || armor_num == 4) && "Armor num must be 2, 3, or 4");
    assert(P0_dig.size() == 11 && "P0_dig must have 11 elements");
    // 通过 tf_buffer 查询 world 坐标
    auto result = fast_tf::lookup<fast_tf::world_fuxk_frame>(
        tf_buffer.get(), pose3d.pnp, pose3d.timestamp_ns);
    if (!result) {
        spdlog::error("FIXME: unreasonable fallback: {}", fast_tf::to_string(result.error()));
        abort();
    }
    const auto transform = *result;

    const auto xyz                = transform->translation();
    const auto [roll, pitch, yaw] = transform->euler_rot().rpy();

    // Runtime检查: pose3d坐标系一致性
    if (const double distance = xyz.norm(); distance < 0.5 || distance > 15.0) {
        spdlog::warn("[Target] Initial position unreasonable: distance={:.2f}m", distance);
    }

    auto r = radius;

    // 计算旋转中心的坐标 (world系)
    // 从装甲板位置xyz反推旋转中心，假设装甲板在半径r处
    auto center_x = xyz[0] + r * std::cos(yaw); // [world系, m], rpy[2]=yaw
    auto center_y = xyz[1] + r * std::sin(yaw); // [world系, m]
    auto center_z = xyz[2];                     // [world系, m]

    // 状态向量: x vx y vy z vz a w r l h (所有在world坐标系)
    // x, y, z: 旋转中心位置 [m]
    // vx, vy, vz: 旋转中心速度 [m/s]
    // a: 装甲板角度 [rad]
    // w: 角速度 [rad/s]
    // r: 装甲板半径 [m]
    // l: r2 - r1 (长短轴差) [m]
    // h: z2 - z1 (高度差) [m]
    const Eigen::VectorXd x0{
        {center_x, 0, center_y, 0, center_z, 0, yaw, 0, r, 0, 0}
    };
    const Eigen::MatrixXd P0 = P0_dig.asDiagonal();

    // 防止夹角求和出现异常值
    auto x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) -> Eigen::VectorXd {
        Eigen::VectorXd c = a + b;
        c[6]              = math::limit_rad(c[6]); // 限制角度在(-pi, pi]
        return c;
    };

    ekf_ = ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(const uint64_t timestamp_ns) {
    const double dt = static_cast<double>(timestamp_ns - timestamp_ns_) * 1e-9;
    predict_impl(dt);
    timestamp_ns_ = timestamp_ns;
}

void Target::predict_impl(double dt) {
    // 状态转移矩阵
    // clang-format off
    const Eigen::MatrixXd F{
        {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
        {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
        {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
        {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
        {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
        {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
        {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
        {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
        {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
        {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
        {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
    };
    // clang-format on

    // Piecewise White Noise Model
    double v1, v2;
    if (name == talos::ArmorName::outpost) {
        v1 = 10;  // 前哨站加速度方差
        v2 = 0.1; // 前哨站角加速度方差
    } else {
        v1 = 100; // 加速度方差
        v2 = 400; // 角加速度方差
    }
    const auto a = dt * dt * dt * dt / 4;
    const auto b = dt * dt * dt / 2;
    const auto c = dt * dt;
    // clang-format off
    const Eigen::MatrixXd Q{
        {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
        {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
        {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
        {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
        {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
        {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
        {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
        {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
        {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
        {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
        {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
    };
    // clang-format on

    auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd x_prior = F * x;
        x_prior[6]              = math::limit_rad(x_prior[6]);
        return x_prior;
    };

    // 前哨站转速特判
    if (this->converged() && this->name == talos::ArmorName::outpost
        && std::abs(this->ekf_.x[7]) > 2) {
        this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
    }

    ekf_.predict(F, Q, f);
}

void Target::update([[maybe_unused]] const talos::Armor& armor, const ArmorPose3D& pose3d) {
    // 通过 tf_buffer 查询 world 坐标
    auto result = fast_tf::lookup<fast_tf::world_fuxk_frame>(
        tf_buffer.get(), pose3d.pnp, pose3d.timestamp_ns);
    // 变换失败时跳过此次更新
    if (!result) {
        spdlog::warn(
            "[Target::update] lookup failed at timestamp {}ns: {}", pose3d.timestamp_ns,
            fast_tf::to_string(result.error()));
        return;
    }

    const Eigen::Vector3d xyz_world = (*result)->translation();
    const Eigen::Vector3d ypd_world = math::xyz2ypd(xyz_world);

    const auto euler              = (*result)->euler_rot();
    const auto [roll, pitch, yaw] = euler.rpy();

    // 装甲板匹配
    int id             = 0;
    auto min_yaw_error = 1e10;

    const std::vector<Eigen::Vector4d>& xyza_list = armor_xyza_list();

    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    xyza_i_list.reserve(armor_num_);
    for (int i = 0; i < armor_num_; i++) {
        xyza_i_list.emplace_back(xyza_list[i], i);
    }

    std::ranges::sort(
        xyza_i_list,
        [](const std::pair<Eigen::Vector4d, int>& a, const std::pair<Eigen::Vector4d, int>& b) {
            Eigen::Vector3d ypd1 = math::xyz2ypd(a.first.head(3));
            Eigen::Vector3d ypd2 = math::xyz2ypd(b.first.head(3));
            return ypd1[2] < ypd2[2];
        });

    // 取前3个distance最小的装甲板
    const int compare_num = std::min(3, armor_num_);
    for (int i = 0; i < compare_num; i++) {
        const auto& xyza     = xyza_i_list[i].first;
        Eigen::Vector3d ypd  = math::xyz2ypd(xyza.head(3));
        const auto yaw_error = std::abs(math::limit_rad(yaw - xyza[3])) // rpy[2]=yaw
                             + std::abs(math::limit_rad(ypd_world[0] - ypd[0]));

        if (std::abs(yaw_error) < std::abs(min_yaw_error)) {
            id            = xyza_i_list[i].second;
            min_yaw_error = yaw_error;
        }
    }

    if (id != 0) {
        jumped = true;
    }

    if (id != last_id) {
        is_switch_ = true;
    } else {
        is_switch_ = false;
    }

    if (is_switch_) {
        switch_count_++;
    }

    last_id = id;
    update_count_++;

    update_ypda(xyz_world, euler, ypd_world, id);
}

void Target::update_ypda(
    const Eigen::Vector3d& xyz_world, const math_fuxk::Ros2EulerRotd& rpy_world,
    const Eigen::Vector3d& ypd_world, const int id) {
    // ========== 坐标系说明 ==========
    // 所有观测数据都在world坐标系 (ROS标准: x前, y左, z上)
    // ypd_world: 球坐标 [rad, rad, m]
    // rpy_world: 欧拉角 [roll, pitch, yaw] [rad]
    // EKF状态向量也在world坐标系
    // ================================

    // 检查: 观测距离合理性
    if (const double observed_distance = ypd_world[2];
        observed_distance < 0.5 || observed_distance > 15.0) {
        spdlog::warn("[Target] Observation distance unreasonable: {:.2f}m", observed_distance);
        return; // 跳过此次更新
    }

    // 观测雅可比
    const Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

    const auto [roll, pitch, yaw] = rpy_world.rpy();
    // 计算观测噪声权重 (world系)
    const auto center_yaw  = std::atan2(xyz_world[1], xyz_world[0]);
    const auto delta_angle = math::limit_rad(yaw - center_yaw); // rpy[2]=yaw
    const Eigen::VectorXd R_dig{
        {4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
         std::log(std::abs(ypd_world[2]) + 1) / 200 + 9e-2}
    };

    const Eigen::MatrixXd R = R_dig.asDiagonal();

    // 定义非线性观测函数 h: x -> z (world系)
    auto h = [&](const Eigen::VectorXd& x) -> Eigen::Vector4d {
        const Eigen::VectorXd xyz = h_armor_xyz(x, id); // [world系]
        Eigen::VectorXd ypd       = math::xyz2ypd(xyz); // [world系, 球坐标]
        auto angle                = math::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
        return {ypd[0], ypd[1], ypd[2], angle};
    };

    // 防止角度环绕问题
    auto z_subtract = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) -> Eigen::VectorXd {
        Eigen::VectorXd c = a - b;
        c[0]              = math::limit_rad(c[0]); // yaw wrap
        c[1]              = math::limit_rad(c[1]); // pitch wrap
        c[3]              = math::limit_rad(c[3]); // angle wrap
        return c;
    };

    // 观测向量 (world系)
    const Eigen::VectorXd z{
        {ypd_world[0], ypd_world[1], ypd_world[2], yaw}  // rpy[2]=yaw
    };

    // 检查: 观测与预测的创新度
    const Eigen::VectorXd predicted_xyz = h_armor_xyz(ekf_.x, id);
    if (const double innovation = (predicted_xyz - xyz_world).norm(); innovation > 2.0) {
        spdlog::warn(
            "[Target] Large innovation: {:.2f}m (may indicate coordinate mismatch)", innovation);
    }

    ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const ExtendedKalmanFilter& Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const {
    std::vector<Eigen::Vector4d> _armor_xyza_list;

    for (int i = 0; i < armor_num_; i++) {
        auto angle          = math::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
        Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
        _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return _armor_xyza_list;
}

bool Target::diverged() const {
    const auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
    const auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

    return !(r_ok && l_ok);
}

bool Target::converged() {
    if (this->name != talos::ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
        is_converged_ = true;
    }

    // 前哨站特殊判断
    if (this->name == talos::ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
        is_converged_ = true;
    }

    return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd& x, const int id) const {
    // ========== 坐标系说明 ==========
    // 输入: x - EKF状态向量 (world系)
    // 输出: 装甲板位置 (world系)
    // x[0], x[2], x[4]: 旋转中心的x, y, z (world系)
    // x[6]: 装甲板角度 (world系)
    // x[8]: 半径r (world系)
    // ================================

    const auto angle   = math::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    const auto use_l_h = armor_num_ == 4 && (id == 1 || id == 3);

    const auto r = use_l_h ? x[8] + x[9] : x[8];

    // 从旋转中心推算装甲板位置 (world系)
    auto armor_x = x[0] - r * std::cos(angle);    // [world系, m]
    auto armor_y = x[2] - r * std::sin(angle);    // [world系, m]
    auto armor_z = use_l_h ? x[4] + x[10] : x[4]; // [world系, m]

    return {armor_x, armor_y, armor_z};           // [world系]
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd& x, const int id) const {
    const auto angle   = math::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    const auto use_l_h = armor_num_ == 4 && (id == 1 || id == 3);

    const auto r = use_l_h ? x[8] + x[9] : x[8];
    auto dx_da   = r * std::sin(angle);
    auto dy_da   = -r * std::cos(angle);

    auto dx_dr = -std::cos(angle);
    auto dy_dr = -std::sin(angle);
    auto dx_dl = use_l_h ? -std::cos(angle) : 0.0;
    auto dy_dl = use_l_h ? -std::sin(angle) : 0.0;

    auto dz_dh = use_l_h ? 1.0 : 0.0;

    // clang-format off
    const Eigen::MatrixXd H_armor_xyza{
        {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
        {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
        {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
        {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
    };
    // clang-format on

    const Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
    Eigen::MatrixXd H_armor_ypd     = math::xyz2ypd_jacobian(armor_xyz);
    // clang-format off
    const Eigen::MatrixXd H_armor_ypda{
        {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
        {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
        {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
        {                0,                 0,                 0, 1}
    };
    // clang-format on

    return H_armor_ypda * H_armor_xyza;
}

} // namespace auto_aim
