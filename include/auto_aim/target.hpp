#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <vector>

#include "detector_state.hpp"
#include "extended_kalman_filter.hpp"
#include "solver.hpp"

namespace auto_aim {

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
     */
    class Target {
    public:
        talos::ArmorName name{talos::ArmorName::not_armor};
        talos::ArmorType armor_type{talos::ArmorType::small};
        bool jumped{false};
        int last_id{0};

        Target() = default;

        /**
         * @brief 从检测结果初始化目标
         * @param armor 检测到的装甲板（用于名称、类型）
         * @param pose3d 解算得到的3D位姿
         * @param t 时间戳
         * @param radius 初始半径估计
         * @param armor_num 装甲板数量（2/3/4）
         * @param P0_dig 初始协方差对角线
         */
        Target(const talos::Armor &armor, const ArmorPose3D &pose3d, std::chrono::steady_clock::time_point t,
               double radius, int armor_num, const Eigen::VectorXd &P0_dig);

        /**
         * @brief 用于测试的简化构造
         */
        Target(double x, double vyaw, double radius, double h);

        /**
         * @brief 预测到指定时间
         */
        void predict(std::chrono::steady_clock::time_point t);

        /**
         * @brief 预测指定时间间隔
         */
        void predict(double dt);

        /**
         * @brief 用观测更新滤波器
         * @param armor 检测到的装甲板
         * @param pose3d 解算得到的3D位姿
         */
        void update(const talos::Armor &armor, const ArmorPose3D &pose3d);

        /**
         * @brief 获取EKF状态向量
         */
        [[nodiscard]] Eigen::VectorXd ekf_x() const;

        /**
         * @brief 获取EKF引用（用于诊断）
         */
        [[nodiscard]] const ExtendedKalmanFilter &ekf() const;

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

    private:
        int armor_num_{4};
        int switch_count_{0};
        int update_count_{0};

        bool is_switch_{false};
        bool is_converged_{false};

        ExtendedKalmanFilter ekf_;
        std::chrono::steady_clock::time_point t_;

        /**
         * @brief 使用球坐标系更新
         */
        void update_ypda(const ArmorPose3D &pose3d, int id);

        /**
         * @brief 计算装甲板在世界坐标系中的位置
         */
        [[nodiscard]] Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd &x, int id) const;

        /**
         * @brief 计算观测雅可比矩阵
         */
        [[nodiscard]] Eigen::MatrixXd h_jacobian(const Eigen::VectorXd &x, int id) const;
    };

} // namespace auto_aim
