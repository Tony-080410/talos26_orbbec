#pragma once

#include <Eigen/Dense>
#include <vector>

#include "detector_state.hpp"
#include "extended_kalman_filter.hpp"
#include "fast_tf/frame.hpp"
#include "solver.hpp"

namespace auto_aim {

/**
 * @brief 预测状态（无副作用预测的返回值）
 */
struct PredictedState {
    Eigen::VectorXd x;                       // 预测后的状态向量
    std::vector<Eigen::Vector4d> armor_xyza; // 各装甲板 [x, y, z, angle]
    talos::ArmorName name{talos::ArmorName::not_armor};
    int armor_num{4};
    bool jumped{false};

    /**
     * @brief 继续预测（用于迭代求解）
     */
    [[nodiscard]] PredictedState predict_dt(double dt) const noexcept;
};

/**
 * @brief 跟踪目标类
 *
 * 移植自 sp_vision_25/tasks/auto_aim/target
 * 使用EKF跟踪单个装甲板目标
 *
 * 状态向量 (11D):
 * x = [x, vx, y, vy, z, vz, a, w, r, l, h]
 * - x, y, z: 旋转中心位置
 * - vx, vy, vz: 速度
 * - a: 装甲板角度
 * - w: 角速度
 * - r: 装甲板半径 (基准)
 * - l: r2 - r1 (长短轴差)
 * - h: z2 - z1 (高度差)
 *
 * 坐标系说明:
 * - 所有输入使用ArmorPose3D (camera_optical坐标系 + timestamp_ns)
 * - 内部通过fast_tf查询转换到world坐标系进行EKF计算
 */
class Target {
public:
    talos::ArmorName name{talos::ArmorName::not_armor};
    talos::ArmorType armor_type{talos::ArmorType::small};
    bool jumped{false};
    int last_id{0};
    bool valid{false};

    explicit Target(const std::reference_wrapper<fast_tf::CoordinateSystem<100>> buf)
        : timestamp_ns_(0)
        , tf_buffer(buf) {}

    /**
     * @brief 从检测结果初始化目标
     * @param armor 检测到的装甲板（用于名称、类型）
     * @param pose3d 解算得到的3D位姿 (camera_optical坐标系 + timestamp_ns)
     * @param tf_buffer TF缓冲区引用，用于坐标变换
     * @param timestamp_ns 时间戳
     * @param radius 初始半径估计
     * @param armor_num 装甲板数量（2/3/4）
     * @param P0_dig 初始协方差对角线
     */
    Target(
        const talos::Armor& armor, const ArmorPose3D& pose3d,
        std::reference_wrapper<fast_tf::CoordinateSystem<100>> tf_buffer,
        const uint64_t timestamp_ns, double radius, int armor_num, const Eigen::VectorXd& P0_dig);

    Target& operator=(const Target& other) {
        if (this != &other) {
            name          = other.name;
            armor_type    = other.armor_type;
            jumped        = other.jumped;
            last_id       = other.last_id;
            armor_num_    = other.armor_num_;
            switch_count_ = other.switch_count_;
            update_count_ = other.update_count_;
            tf_buffer     = other.tf_buffer;
            is_switch_    = other.is_switch_;
            is_converged_ = other.is_converged_;
            ekf_          = other.ekf_;
            timestamp_ns_ = other.timestamp_ns_;
            valid         = other.valid;
        }
        return *this;
    }
    /**
     * @brief 预测到指定时间
     */
    void predict(uint64_t timestamp_ns);

    /**
     * @brief 预测指定时间间隔
     */
    void predict_impl(double dt);

    /**
     * @brief 预测指定时间间隔（无副作用）
     * @param dt 时间间隔（秒）
     * @return 预测后的状态
     */
    [[nodiscard]] PredictedState predict_at(double dt) const noexcept;

    /**
     * @brief 用观测更新滤波器
     * @param armor 检测到的装甲板
     * @param pose3d 解算得到的3D位姿 (camera_optical坐标系 + timestamp_ns)
     */
    void update(const talos::Armor& armor, const ArmorPose3D& pose3d);

    /**
     * @brief 获取EKF状态向量
     */
    [[nodiscard]] Eigen::VectorXd ekf_x() const;

    /**
     * @brief 获取EKF引用（用于诊断）
     */
    [[nodiscard]] const ExtendedKalmanFilter& ekf() const;

    /**
     * @brief 获取所有装甲板位置和角度列表
     * @return 每个元素为 [x, y, z, angle]
     */
    [[nodiscard]] std::vector<Eigen::Vector4d> armor_xyza_list() const;

    /**
     * @brief 检查是否发散
     */
    [[nodiscard]] bool diverged() const;

    /**
     * @brief 检查是否收敛
     */
    [[nodiscard]] bool converged();

    uint64_t timestamp_ns_;

private:
    int armor_num_{4};
    int switch_count_{0};
    int update_count_{0};
    std::reference_wrapper<fast_tf::CoordinateSystem<100>> tf_buffer;
    bool is_switch_{false};
    bool is_converged_{false};

    ExtendedKalmanFilter ekf_;

    /**
     * @brief 使用球坐标系更新 (world坐标系)
     * @param xyz_world 装甲板在world系的位置
     * @param rpy_world 装甲板在world系的欧拉角 [roll, pitch, yaw]
     * @param ypd_world 装甲板在world系的球坐标
     * @param id 装甲板ID
     */
    void update_ypda(
        const Eigen::Vector3d& xyz_world, const math_fuxk::Ros2EulerRotd& rpy_world,
        const Eigen::Vector3d& ypd_world, int id);

    /**
     * @brief 计算装甲板在世界坐标系中的位置
     */
    [[nodiscard]] Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd& x, int id) const;

    /**
     * @brief 计算观测雅可比矩阵
     */
    [[nodiscard]] Eigen::MatrixXd h_jacobian(const Eigen::VectorXd& x, int id) const;
};

} // namespace auto_aim
