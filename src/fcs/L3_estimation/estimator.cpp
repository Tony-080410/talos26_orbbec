#include "fcs/L3_estimation/estimator.hpp"
#include "fcs/L2_perception/solver.hpp" // for math::normalize_angle, xyz_to_ypd

#include <cmath>
#include <spdlog/spdlog.h>

namespace fcs::L3 {

// ============================================================================
// TrackState 实现
// ============================================================================

Eigen::Vector4d TrackState::armor_xyza(int i) const noexcept {
    const double angle_offset = i * 2.0 * M_PI / armor_num;
    const double angle        = L2::math::normalize_angle(x(idx::A) + angle_offset);

    // 四装甲板模式：1、3号使用不同半径和高度
    const bool use_l_h    = (armor_num == 4) && (i == 1 || i == 3);
    const double r        = use_l_h ? x(idx::R) + x(idx::L) : x(idx::R);
    const double z_offset = use_l_h ? x(idx::H) : 0.0;

    // 从旋转中心计算装甲板位置
    const double armor_x = x(idx::X) - r * std::cos(angle);
    const double armor_y = x(idx::Y) - r * std::sin(angle);
    const double armor_z = x(idx::Z) + z_offset;

    return {armor_x, armor_y, armor_z, angle};
}

std::array<Eigen::Vector4d, 4> TrackState::all_armor_xyza() const noexcept {
    std::array<Eigen::Vector4d, 4> result;
    for (int i = 0; i < 4; ++i) {
        if (i < armor_num) {
            result[i] = armor_xyza(i);
        } else {
            result[i] = Eigen::Vector4d::Zero();
        }
    }
    return result;
}

// ============================================================================
// PredictedState 实现
// ============================================================================

PredictedState PredictedState::predict_dt(double dt) const noexcept {
    PredictedState result;
    result.state = state;

    // 应用状态转移
    auto& x = result.state.x;
    x(idx::X) += x(idx::VX) * dt;
    x(idx::Y) += x(idx::VY) * dt;
    x(idx::Z) += x(idx::VZ) * dt;
    x(idx::A) = L2::math::normalize_angle(x(idx::A) + x(idx::W) * dt);

    // 前哨站角速度限制
    if (state.name == ArmorName::Outpost && std::abs(x(idx::W)) > 2.0) {
        x(idx::W) = x(idx::W) > 0 ? 2.51 : -2.51;
    }

    return result;
}

// ============================================================================
// Estimator 实现
// ============================================================================

Estimator::Estimator(const EstimatorConfig& config)
    : config_(config) {}

void Estimator::init(
    const L2::Measurement& meas, const Eigen::Vector3d& world_xyz, double world_yaw,
    int armor_num) {
    // 初始化状态向量
    const double r = config_.initial_radius;

    // 从装甲板位置反推旋转中心
    const double center_x = world_xyz[0] + r * std::cos(world_yaw);
    const double center_y = world_xyz[1] + r * std::sin(world_yaw);
    const double center_z = world_xyz[2];

    state_.x          = Eigen::Matrix<double, idx::STATE_DIM, 1>::Zero();
    state_.x(idx::X)  = center_x;
    state_.x(idx::VX) = 0.0;
    state_.x(idx::Y)  = center_y;
    state_.x(idx::VY) = 0.0;
    state_.x(idx::Z)  = center_z;
    state_.x(idx::VZ) = 0.0;
    state_.x(idx::A)  = world_yaw;
    state_.x(idx::W)  = 0.0;
    state_.x(idx::R)  = r;
    state_.x(idx::L)  = 0.0;
    state_.x(idx::H)  = 0.0;

    // 初始化协方差
    state_.P = config_.P0_diag.asDiagonal();

    // 元数据
    state_.timestamp_ns  = meas.timestamp_ns;
    state_.status        = TrackStatus::Tentative;
    state_.hit_count     = 1;
    state_.miss_count    = 0;
    state_.name          = meas.name;
    state_.armor_num     = armor_num;
    state_.last_armor_id = 0;
    state_.jumped        = false;

    has_track_ = true;

    spdlog::debug(
        "[Estimator] initialized: center=({:.3f}, {:.3f}, {:.3f}), yaw={:.3f}", center_x, center_y,
        center_z, world_yaw);
}

void Estimator::predict(uint64_t timestamp_ns) noexcept {
    if (!has_track_)
        return;

    const double dt = static_cast<double>(timestamp_ns - state_.timestamp_ns) * 1e-9;
    if (dt <= 0.0 || dt > 1.0) {
        spdlog::warn("[Estimator] invalid dt: {:.6f}s", dt);
        return;
    }

    predict_impl(dt);
    state_.timestamp_ns = timestamp_ns;
}

void Estimator::update(
    const L2::Measurement& meas, const Eigen::Vector3d& world_xyz, double world_yaw) noexcept {
    if (!has_track_)
        return;

    // 匹配最近的装甲板
    const int id = match_armor(world_xyz, world_yaw);

    // 计算观测雅可比
    const auto H = compute_H(id);

    // 计算观测噪声
    const auto ypd           = L2::math::xyz_to_ypd(world_xyz);
    const double center_yaw  = std::atan2(world_xyz[1], world_xyz[0]);
    const double delta_angle = L2::math::normalize_angle(world_yaw - center_yaw);

    Eigen::Matrix<double, idx::MEAS_DIM, idx::MEAS_DIM> R;
    R.setZero();
    R(0, 0) = 4e-3;                                        // yaw
    R(1, 1) = 4e-3;                                        // pitch
    R(2, 2) = std::log(std::abs(delta_angle) + 1) + 1;     // distance weight
    R(3, 3) = std::log(std::abs(ypd[2]) + 1) / 200 + 9e-2; // angle

    // 预测的装甲板位置
    const Eigen::Vector3d pred_xyz = armor_xyz(id);
    const Eigen::Vector3d pred_ypd = L2::math::xyz_to_ypd(pred_xyz);
    const double pred_angle =
        L2::math::normalize_angle(state_.x(idx::A) + id * 2.0 * M_PI / state_.armor_num);

    // 观测向量
    Eigen::Matrix<double, idx::MEAS_DIM, 1> z;
    z << ypd[0], ypd[1], ypd[2], world_yaw;

    // 预测观测向量
    Eigen::Matrix<double, idx::MEAS_DIM, 1> z_pred;
    z_pred << pred_ypd[0], pred_ypd[1], pred_ypd[2], pred_angle;

    // 创新向量 (处理角度环绕)
    Eigen::Matrix<double, idx::MEAS_DIM, 1> y = z - z_pred;
    y(0)                                      = L2::math::normalize_angle(y(0)); // yaw
    y(1)                                      = L2::math::normalize_angle(y(1)); // pitch
    y(3)                                      = L2::math::normalize_angle(y(3)); // angle

    // 卡尔曼增益
    const auto S = H * state_.P * H.transpose() + R;
    const auto K = state_.P * H.transpose() * S.inverse();

    // 更新状态
    state_.x         = state_.x + K * y;
    state_.x(idx::A) = L2::math::normalize_angle(state_.x(idx::A));

    // 更新协方差 (Joseph form for numerical stability)
    const auto I   = Eigen::Matrix<double, idx::STATE_DIM, idx::STATE_DIM>::Identity();
    const auto IKH = I - K * H;
    state_.P       = IKH * state_.P * IKH.transpose() + K * R * K.transpose();

    // 更新元数据
    state_.hit_count++;
    state_.miss_count    = 0;
    state_.jumped        = (id != state_.last_armor_id);
    state_.last_armor_id = id;

    // 更新状态
    if (converged()) {
        state_.status = TrackStatus::Confirmed;
    }

    spdlog::debug(
        "[Estimator] updated: id={}, hit={}, r={:.3f}", id, state_.hit_count, state_.x(idx::R));
}

PredictedState Estimator::predict_at(double dt) const noexcept {
    PredictedState result;
    result.state = state_;

    // 应用状态转移
    auto& x = result.state.x;
    x(idx::X) += x(idx::VX) * dt;
    x(idx::Y) += x(idx::VY) * dt;
    x(idx::Z) += x(idx::VZ) * dt;
    x(idx::A) = L2::math::normalize_angle(x(idx::A) + x(idx::W) * dt);

    return result;
}

bool Estimator::converged() const noexcept {
    const uint32_t threshold = (state_.name == ArmorName::Outpost) ? config_.converge_count_outpost
                                                                   : config_.converge_count;
    return state_.hit_count >= threshold && !diverged();
}

bool Estimator::diverged() const noexcept {
    const double r1 = state_.x(idx::R);
    const double r2 = r1 + state_.x(idx::L);

    return r1 < config_.min_radius || r1 > config_.max_radius || r2 < config_.min_radius
        || r2 > config_.max_radius;
}

void Estimator::reset() noexcept {
    has_track_ = false;
    state_     = TrackState{};
}

// ============================================================================
// EKF 内部实现
// ============================================================================

void Estimator::predict_impl(double dt) noexcept {
    const auto F = compute_F(dt);
    const auto Q = compute_Q(dt);

    // 状态预测
    state_.x         = F * state_.x;
    state_.x(idx::A) = L2::math::normalize_angle(state_.x(idx::A));

    // 前哨站角速度限制
    if (state_.name == ArmorName::Outpost && std::abs(state_.x(idx::W)) > 2.0) {
        state_.x(idx::W) = state_.x(idx::W) > 0 ? 2.51 : -2.51;
    }

    // 协方差预测
    state_.P = F * state_.P * F.transpose() + Q;

    // 增加 miss_count
    state_.miss_count++;
    if (state_.status == TrackStatus::Confirmed && state_.miss_count > 3) {
        state_.status = TrackStatus::Coasting;
    }
}

Eigen::Matrix<double, idx::STATE_DIM, idx::STATE_DIM> Estimator::compute_F(double dt) noexcept {
    Eigen::Matrix<double, idx::STATE_DIM, idx::STATE_DIM> F;
    F.setIdentity();

    F(idx::X, idx::VX) = dt;
    F(idx::Y, idx::VY) = dt;
    F(idx::Z, idx::VZ) = dt;
    F(idx::A, idx::W)  = dt;

    return F;
}

Eigen::Matrix<double, idx::STATE_DIM, idx::STATE_DIM>
    Estimator::compute_Q(double dt) const noexcept {
    // 过程噪声参数
    const double v1 =
        (state_.name == ArmorName::Outpost) ? config_.accel_var_outpost : config_.accel_var;
    const double v2 = (state_.name == ArmorName::Outpost) ? config_.angular_accel_var_outpost
                                                          : config_.angular_accel_var;

    // Piecewise White Noise Model
    const double a = dt * dt * dt * dt / 4;
    const double b = dt * dt * dt / 2;
    const double c = dt * dt;

    Eigen::Matrix<double, idx::STATE_DIM, idx::STATE_DIM> Q;
    Q.setZero();

    // X, VX
    Q(idx::X, idx::X)   = a * v1;
    Q(idx::X, idx::VX)  = b * v1;
    Q(idx::VX, idx::X)  = b * v1;
    Q(idx::VX, idx::VX) = c * v1;

    // Y, VY
    Q(idx::Y, idx::Y)   = a * v1;
    Q(idx::Y, idx::VY)  = b * v1;
    Q(idx::VY, idx::Y)  = b * v1;
    Q(idx::VY, idx::VY) = c * v1;

    // Z, VZ
    Q(idx::Z, idx::Z)   = a * v1;
    Q(idx::Z, idx::VZ)  = b * v1;
    Q(idx::VZ, idx::Z)  = b * v1;
    Q(idx::VZ, idx::VZ) = c * v1;

    // A, W
    Q(idx::A, idx::A) = a * v2;
    Q(idx::A, idx::W) = b * v2;
    Q(idx::W, idx::A) = b * v2;
    Q(idx::W, idx::W) = c * v2;

    return Q;
}

Eigen::Vector3d Estimator::armor_xyz(int id) const noexcept {
    const double angle_offset = id * 2.0 * M_PI / state_.armor_num;
    const double angle        = L2::math::normalize_angle(state_.x(idx::A) + angle_offset);

    const bool use_l_h    = (state_.armor_num == 4) && (id == 1 || id == 3);
    const double r        = use_l_h ? state_.x(idx::R) + state_.x(idx::L) : state_.x(idx::R);
    const double z_offset = use_l_h ? state_.x(idx::H) : 0.0;

    const double armor_x = state_.x(idx::X) - r * std::cos(angle);
    const double armor_y = state_.x(idx::Y) - r * std::sin(angle);
    const double armor_z = state_.x(idx::Z) + z_offset;

    return {armor_x, armor_y, armor_z};
}

Eigen::Matrix<double, idx::MEAS_DIM, idx::STATE_DIM> Estimator::compute_H(int id) const noexcept {
    const double angle_offset = id * 2.0 * M_PI / state_.armor_num;
    const double angle        = L2::math::normalize_angle(state_.x(idx::A) + angle_offset);

    const bool use_l_h = (state_.armor_num == 4) && (id == 1 || id == 3);
    const double r     = use_l_h ? state_.x(idx::R) + state_.x(idx::L) : state_.x(idx::R);

    // 装甲板位置对状态的雅可比
    const double dx_da = r * std::sin(angle);
    const double dy_da = -r * std::cos(angle);
    const double dx_dr = -std::cos(angle);
    const double dy_dr = -std::sin(angle);
    const double dx_dl = use_l_h ? -std::cos(angle) : 0.0;
    const double dy_dl = use_l_h ? -std::sin(angle) : 0.0;
    const double dz_dh = use_l_h ? 1.0 : 0.0;

    // H_xyza: 装甲板位置(x,y,z,angle)对状态向量的雅可比
    Eigen::Matrix<double, 4, idx::STATE_DIM> H_xyza;
    H_xyza.setZero();
    // x row
    H_xyza(0, idx::X) = 1.0;
    H_xyza(0, idx::A) = dx_da;
    H_xyza(0, idx::R) = dx_dr;
    H_xyza(0, idx::L) = dx_dl;
    // y row
    H_xyza(1, idx::Y) = 1.0;
    H_xyza(1, idx::A) = dy_da;
    H_xyza(1, idx::R) = dy_dr;
    H_xyza(1, idx::L) = dy_dl;
    // z row
    H_xyza(2, idx::Z) = 1.0;
    H_xyza(2, idx::H) = dz_dh;
    // angle row
    H_xyza(3, idx::A) = 1.0;

    // xyz_to_ypd 的雅可比
    const Eigen::Vector3d xyz   = armor_xyz(id);
    const Eigen::Matrix3d H_ypd = L2::math::xyz_to_ypd_jacobian(xyz);

    // H_ypda: 观测(yaw,pitch,dist,angle)对装甲板位置的雅可比
    Eigen::Matrix<double, 4, 4> H_ypda;
    H_ypda.setZero();
    H_ypda.block<3, 3>(0, 0) = H_ypd;
    H_ypda(3, 3)             = 1.0;

    return H_ypda * H_xyza;
}

int Estimator::match_armor(const Eigen::Vector3d& world_xyz, double world_yaw) const noexcept {
    int best_id      = 0;
    double min_error = 1e10;

    // 计算观测的球坐标
    const Eigen::Vector3d obs_ypd = L2::math::xyz_to_ypd(world_xyz);

    // 按距离排序候选装甲板
    std::array<std::pair<double, int>, 4> candidates;
    for (int i = 0; i < state_.armor_num; ++i) {
        const Eigen::Vector3d pred_xyz = armor_xyz(i);
        const Eigen::Vector3d pred_ypd = L2::math::xyz_to_ypd(pred_xyz);
        candidates[i]                  = {pred_ypd[2], i}; // distance, id
    }

    // 按距离排序
    std::sort(
        candidates.begin(), candidates.begin() + state_.armor_num,
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // 取前3个最近的装甲板比较 yaw 误差
    const int compare_num = std::min(3, state_.armor_num);
    for (int i = 0; i < compare_num; ++i) {
        const int id                   = candidates[i].second;
        const Eigen::Vector4d xyza     = state_.armor_xyza(id);
        const Eigen::Vector3d pred_ypd = L2::math::xyz_to_ypd(xyza.head<3>());

        // yaw 误差 = 装甲板朝向误差 + 球坐标yaw误差
        const double yaw_error = std::abs(L2::math::normalize_angle(world_yaw - xyza[3]))
                               + std::abs(L2::math::normalize_angle(obs_ypd[0] - pred_ypd[0]));

        if (yaw_error < min_error) {
            min_error = yaw_error;
            best_id   = id;
        }
    }

    return best_id;
}

} // namespace fcs::L3
