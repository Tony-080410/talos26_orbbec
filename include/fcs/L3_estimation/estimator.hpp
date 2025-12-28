#pragma once

#include "../L2_perception/solver.hpp"
#include "../core/types.hpp"

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <optional>

namespace fcs::L3 {

// ============================================================================
// TrackState - 航迹状态
// ============================================================================

struct TrackState {
    // 时间戳
    uint64_t timestamp_ns{0};

    // 状态
    TrackStatus status{TrackStatus::Tentative};
    uint32_t hit_count{0};
    uint32_t miss_count{0};

    // 目标信息
    ArmorName name{ArmorName::Invalid};
    int armor_num{4};

    // 装甲板跳变标记
    int last_armor_id{0};
    bool jumped{false};

    /**
     * @brief 计算第i个装甲板的位置和角度
     * @return [x, y, z, angle]
     */
    [[nodiscard]] Eigen::Vector4d armor_xyza(int i) const noexcept;

    /**
     * @brief 获取所有装甲板位置和角度
     */
    [[nodiscard]] std::array<Eigen::Vector4d, 4> all_armor_xyza() const noexcept;
};

// ============================================================================
// PredictedState - 预测状态 (无副作用)
// ============================================================================

struct PredictedState {
    TrackState state;

    /**
     * @brief 继续预测 (用于弹道迭代)
     */
    [[nodiscard]] PredictedState predict_dt(double dt) const noexcept;
};

// ============================================================================
// EstimatorConfig - 估计器配置
// ============================================================================

struct EstimatorConfig {
    // 初始协方差对角线
    Eigen::Matrix<double, 11, 1> P0_diag{
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01, 0.01}
    };

    // 初始半径估计
    double initial_radius{0.22};

    // 过程噪声 (加速度方差)
    double accel_var{100.0};        // m/s^2
    double accel_var_outpost{10.0}; // 前哨站

    // 过程噪声 (角加速度方差)
    double angular_accel_var{400.0};       // rad/s^2
    double angular_accel_var_outpost{0.1}; // 前哨站

    // 收敛判定
    uint32_t converge_count{3};
    uint32_t converge_count_outpost{10};

    // 发散判定
    double min_radius{0.05};
    double max_radius{0.50};
};

// ============================================================================
// Estimator - 状态估计器 (EKF)
// ============================================================================

class Estimator {
public:
    explicit Estimator(const EstimatorConfig& config = {});

    // ======== 主接口 ========

    /**
     * @brief 初始化航迹
     * @param meas 首次量测
     * @param world_xyz 世界坐标系位置
     * @param world_yaw 世界坐标系yaw角
     * @param armor_num 装甲板数量 (2/3/4)
     */
    void init(
        const L2::Measurement& meas, const Eigen::Vector3d& world_xyz, double world_yaw,
        int armor_num);

    /**
     * @brief 预测到指定时间
     */
    void predict(uint64_t timestamp_ns) noexcept;

    /**
     * @brief 用量测更新
     * @param meas 量测数据
     * @param world_xyz 世界坐标系位置
     * @param world_yaw 世界坐标系yaw角
     */
    void update(
        const L2::Measurement& meas, const Eigen::Vector3d& world_xyz, double world_yaw) noexcept;

    /**
     * @brief 无副作用预测 (用于输出和弹道迭代)
     */
    [[nodiscard]] PredictedState predict_at(double dt) const noexcept;

    // ======== 状态查询 ========

    [[nodiscard]] const TrackState& state() const noexcept { return state_; }
    [[nodiscard]] bool has_track() const noexcept { return has_track_; }
    [[nodiscard]] bool converged() const noexcept;
    [[nodiscard]] bool diverged() const noexcept;

    /**
     * @brief 重置估计器
     */
    void reset() noexcept;

private:
    EstimatorConfig config_;
    TrackState state_;
    bool has_track_{false};

    // ======== EKF 内部 ========

    /**
     * @brief 状态转移 (有副作用)
     */
    void predict_impl(double dt) noexcept;

    /**
     * @brief 计算状态转移矩阵 F
     */
    [[nodiscard]] static Eigen::Matrix<double, 11, 11> compute_F(double dt) noexcept;

    /**
     * @brief 计算过程噪声矩阵 Q
     */
    [[nodiscard]] Eigen::Matrix<double, 11, 11> compute_Q(double dt) const noexcept;

    /**
     * @brief 计算装甲板位置
     */
    [[nodiscard]] Eigen::Vector3d armor_xyz(int id) const noexcept;

    /**
     * @brief 计算观测雅可比矩阵 H
     */
    [[nodiscard]] Eigen::Matrix<double, 11, 11> compute_H(int id) const noexcept;

    /**
     * @brief 匹配最近的装甲板
     */
    [[nodiscard]] int
        match_armor(const Eigen::Vector3d& world_xyz, double world_yaw) const noexcept;
};

} // namespace fcs::L3
