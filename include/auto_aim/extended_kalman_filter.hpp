#pragma once

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>

namespace auto_aim {

/**
 * @brief 扩展卡尔曼滤波器
 *
 * 移植自 sp_vision_25/tools/extended_kalman_filter
 * 支持非线性观测函数和自定义的状态加法/观测减法
 */
class ExtendedKalmanFilter {
public:
    Eigen::VectorXd x; // 状态向量
    Eigen::MatrixXd P; // 协方差矩阵

    // 卡方检验数据
    std::map<std::string, double> data;
    std::deque<int> recent_nis_failures{0};
    size_t window_size = 100;
    double last_nis{0};

    ExtendedKalmanFilter() = default;

    /**
     * @brief 构造EKF
     * @param x0 初始状态
     * @param P0 初始协方差
     * @param x_add 状态加法函数（用于角度wrap）
     */
    ExtendedKalmanFilter(
        const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>&
            x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a + b; });

    /**
     * @brief 线性预测
     * @param F 状态转移矩阵
     * @param Q 过程噪声
     * @return 预测后的状态
     */
    Eigen::VectorXd predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q);

    /**
     * @brief 非线性预测
     * @param F 状态转移矩阵的雅可比
     * @param Q 过程噪声
     * @param f 非线性状态转移函数
     * @return 预测后的状态
     */
    Eigen::VectorXd predict(
        const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f);

    /**
     * @brief 线性更新
     * @param z 观测向量
     * @param H 观测矩阵
     * @param R 观测噪声
     * @param z_subtract 观测减法函数（用于角度wrap）
     * @return 更新后的状态
     */
    Eigen::VectorXd update(
        const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>&
            z_subtract = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a - b; });

    /**
     * @brief 非线性更新
     * @param z 观测向量
     * @param H 观测矩阵的雅可比
     * @param R 观测噪声
     * @param h 非线性观测函数
     * @param z_subtract 观测减法函数（用于角度wrap）
     * @return 更新后的状态
     */
    Eigen::VectorXd update(
        const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& h,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>&
            z_subtract = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a - b; });

private:
    Eigen::MatrixXd I_;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add_;

    int nees_count_{0};
    int nis_count_{0};
    int total_count_{0};
};

} // namespace auto_aim
