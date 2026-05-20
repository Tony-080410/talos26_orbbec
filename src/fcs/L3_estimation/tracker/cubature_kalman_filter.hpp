#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include <Eigen/Dense>

#include <ceres/jet.h>

template <int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class CubatureKalmanFilter {
public:
    CubatureKalmanFilter() = default;

    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;
    using JetX     = ceres::Jet<double, N_X>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1& z)>;

    enum class ConvergenceStatus : uint8_t {
        Unknown,
        Converging,
        Converged,
        Diverging,
    };

    struct ConvergenceCriteria {
        double converged_nis{2.0 * static_cast<double>(N_Z)};
        double diverged_nis{25.0 * static_cast<double>(N_Z)};
        double converged_max_covariance_diag{1e6};
        double diverged_max_covariance_diag{1e12};
        double min_covariance_diag{-1e-9};
        int converged_updates{3};
        int diverged_updates{2};
    };

    explicit CubatureKalmanFilter(
        const PredicFunc& f, const MeasureFunc& h, const UpdateQFunc& u_q, const UpdateRFunc& u_r,
        const MatrixXX& P0, ConvergenceCriteria convergence_criteria = {}) noexcept
        : f(f)
        , h(h)
        , update_Q(u_q)
        , update_R(u_r)
        , P_post(P0)
        , convergence_criteria_(convergence_criteria) {
        update_covariance_stat_(P_post);
    }

    void setState(const MatrixX1& x0) noexcept { x_post = x0; }

    [[nodiscard]] const MatrixX1& X() const noexcept { return x_post; }
    [[nodiscard]] const MatrixXX& P() const noexcept { return P_post; }
    [[nodiscard]] const MatrixXZ& K_gain() const noexcept { return K_; }
    [[nodiscard]] const MatrixXX& Q_mat() const noexcept { return Q; }
    [[nodiscard]] const MatrixZZ& R_mat() const noexcept { return R; }
    [[nodiscard]] ConvergenceStatus convergence_status() const noexcept {
        return convergence_status_;
    }
    [[nodiscard]] bool converging() const noexcept {
        return convergence_status_ == ConvergenceStatus::Converging;
    }
    [[nodiscard]] bool converged() const noexcept {
        return convergence_status_ == ConvergenceStatus::Converged;
    }
    [[nodiscard]] bool diverging() const noexcept {
        return convergence_status_ == ConvergenceStatus::Diverging;
    }
    [[nodiscard]] double normalized_innovation_squared() const noexcept {
        return normalized_innovation_squared_;
    }
    [[nodiscard]] double max_covariance_diag() const noexcept { return max_covariance_diag_; }
    [[nodiscard]] int consecutive_converged_updates() const noexcept {
        return consecutive_converged_updates_;
    }
    [[nodiscard]] int consecutive_diverged_updates() const noexcept {
        return consecutive_diverged_updates_;
    }
    [[nodiscard]] const ConvergenceCriteria& convergence_criteria() const noexcept {
        return convergence_criteria_;
    }

    void setConvergenceCriteria(ConvergenceCriteria criteria) noexcept {
        convergence_criteria_ = criteria;
        resetConvergenceStatus();
    }

    void resetConvergenceStatus() noexcept {
        convergence_status_            = ConvergenceStatus::Unknown;
        consecutive_converged_updates_ = 0;
        consecutive_diverged_updates_  = 0;
        normalized_innovation_squared_ = std::numeric_limits<double>::infinity();
        max_covariance_diag_           = std::numeric_limits<double>::infinity();
    }

    void setPredictFunc(const PredicFunc& predict_func) noexcept { f = predict_func; }
    void setMeasureFunc(const MeasureFunc& measure_func) noexcept { h = measure_func; }

    MatrixX1 predict() noexcept {
        const auto cubature_points = generate_cubature_points_(x_post, P_post);
        if (!cubature_points.has_value()) {
            Q     = update_Q();
            P_pri = P_post;
            x_pri = x_post;
            mark_immediate_divergence_();
            return x_pri;
        }

        Eigen::Matrix<double, N_X, 2 * N_X> propagated_points;
        for (int point_index = 0; point_index < 2 * N_X; ++point_index) {
            propagated_points.col(point_index) =
                propagate_state_(cubature_points->col(point_index));
        }

        x_pri = propagated_points.rowwise().mean();
        P_pri = MatrixXX::Zero();
        for (int point_index = 0; point_index < 2 * N_X; ++point_index) {
            const MatrixX1 delta = propagated_points.col(point_index) - x_pri;
            P_pri += cubature_weight_() * (delta * delta.transpose());
        }

        Q      = update_Q();
        P_pri  = symmetrize_xx_(P_pri + Q);
        x_post = x_pri;
        update_prediction_health_();
        return x_pri;
    }

    MatrixX1 update(const MatrixZ1& z) noexcept {
        const auto cubature_points = generate_cubature_points_(x_pri, P_pri);
        if (!cubature_points.has_value()) {
            K_                             = MatrixXZ::Zero();
            normalized_innovation_squared_ = std::numeric_limits<double>::infinity();
            mark_immediate_divergence_();
            return x_post;
        }

        Eigen::Matrix<double, N_Z, 2 * N_X> measurement_points;
        for (int point_index = 0; point_index < 2 * N_X; ++point_index) {
            measurement_points.col(point_index) =
                propagate_measurement_(cubature_points->col(point_index));
        }

        MatrixZ1 z_pri = measurement_points.rowwise().mean();
        MatrixZZ S     = MatrixZZ::Zero();
        MatrixXZ Pxz   = MatrixXZ::Zero();
        for (int point_index = 0; point_index < 2 * N_X; ++point_index) {
            const MatrixX1 dx = cubature_points->col(point_index) - x_pri;
            const MatrixZ1 dz = measurement_points.col(point_index) - z_pri;
            S += cubature_weight_() * (dz * dz.transpose());
            Pxz += cubature_weight_() * (dx * dz.transpose());
        }

        R = update_R(z);
        S = symmetrize_zz_(S + R);

        const MatrixZ1 nu = z - z_pri;
        Eigen::LDLT<MatrixZZ> ldlt(S);
        if (!S.allFinite() || ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
            K_                             = MatrixXZ::Zero();
            normalized_innovation_squared_ = std::numeric_limits<double>::infinity();
            mark_immediate_divergence_();
            return x_post;
        }

        const MatrixZ1 solved_nu = ldlt.solve(nu);
        const MatrixZX gain_t    = ldlt.solve(Pxz.transpose());
        if (!solved_nu.allFinite() || !gain_t.allFinite()) {
            K_                             = MatrixXZ::Zero();
            normalized_innovation_squared_ = std::numeric_limits<double>::infinity();
            mark_immediate_divergence_();
            return x_post;
        }

        normalized_innovation_squared_ = std::max(0.0, static_cast<double>(nu.dot(solved_nu)));
        K_                             = gain_t.transpose();
        x_post                         = x_pri + K_ * nu;
        P_post                         = symmetrize_xx_(P_pri - K_ * S * K_.transpose());
        update_convergence_status_();
        return x_post;
    }

    void decorrelate(int i, int j) noexcept {
        P_post(i, j) = 0.0;
        P_post(j, i) = 0.0;
        update_covariance_stat_(P_post);
    }

private:
    using CubaturePoints = Eigen::Matrix<double, N_X, 2 * N_X>;

    [[nodiscard]] static constexpr double cubature_weight_() noexcept {
        return 1.0 / static_cast<double>(2 * N_X);
    }

    [[nodiscard]] static MatrixXX symmetrize_xx_(const MatrixXX& matrix) noexcept {
        return 0.5 * (matrix + matrix.transpose());
    }

    [[nodiscard]] static MatrixZZ symmetrize_zz_(const MatrixZZ& matrix) noexcept {
        return 0.5 * (matrix + matrix.transpose());
    }

    [[nodiscard]] std::optional<CubaturePoints>
        generate_cubature_points_(const MatrixX1& mean, const MatrixXX& covariance) const noexcept {
        if (!mean.allFinite() || !covariance.allFinite()) {
            return std::nullopt;
        }

        Eigen::LLT<MatrixXX> llt(symmetrize_xx_(covariance));
        if (llt.info() != Eigen::Success) {
            return std::nullopt;
        }

        const MatrixXX sqrt_covariance = llt.matrixL();
        const double scale             = std::sqrt(static_cast<double>(N_X));
        CubaturePoints points;
        for (int axis = 0; axis < N_X; ++axis) {
            const MatrixX1 offset  = scale * sqrt_covariance.col(axis);
            points.col(axis)       = mean + offset;
            points.col(axis + N_X) = mean - offset;
        }
        return points;
    }

    [[nodiscard]] MatrixX1 propagate_state_(const MatrixX1& point) const noexcept {
        JetX input[N_X]{};
        JetX output[N_X]{};
        for (int i = 0; i < N_X; ++i) {
            input[i].a = point[i];
        }
        f(input, output);

        MatrixX1 propagated = MatrixX1::Zero();
        for (int i = 0; i < N_X; ++i) {
            propagated[i] = output[i].a;
        }
        return propagated;
    }

    [[nodiscard]] MatrixZ1 propagate_measurement_(const MatrixX1& point) const noexcept {
        JetX input[N_X]{};
        JetX output[N_Z]{};
        for (int i = 0; i < N_X; ++i) {
            input[i].a = point[i];
        }
        h(input, output);

        MatrixZ1 measurement = MatrixZ1::Zero();
        for (int i = 0; i < N_Z; ++i) {
            measurement[i] = output[i].a;
        }
        return measurement;
    }

    void update_prediction_health_() noexcept {
        update_covariance_stat_(P_pri);
        if (!x_pri.allFinite() || !P_pri.allFinite()
            || min_covariance_diag(P_pri) < convergence_criteria_.min_covariance_diag) {
            mark_immediate_divergence_();
            return;
        }

        if (max_covariance_diag_ > convergence_criteria_.diverged_max_covariance_diag) {
            register_divergence_sample_();
            return;
        }

        if (convergence_status_ == ConvergenceStatus::Unknown) {
            convergence_status_ = ConvergenceStatus::Converging;
        }
    }

    void update_convergence_status_() noexcept {
        update_covariance_stat_(P_post);
        if (!x_post.allFinite() || !P_post.allFinite()
            || !std::isfinite(normalized_innovation_squared_)
            || min_covariance_diag(P_post) < convergence_criteria_.min_covariance_diag) {
            mark_immediate_divergence_();
            return;
        }

        if (normalized_innovation_squared_ > convergence_criteria_.diverged_nis
            || max_covariance_diag_ > convergence_criteria_.diverged_max_covariance_diag) {
            register_divergence_sample_();
            return;
        }

        consecutive_diverged_updates_ = 0;
        const bool covariance_converged =
            max_covariance_diag_ <= convergence_criteria_.converged_max_covariance_diag;
        const bool innovation_converged =
            normalized_innovation_squared_ <= convergence_criteria_.converged_nis;
        if (covariance_converged && innovation_converged) {
            ++consecutive_converged_updates_;
            convergence_status_ = (consecutive_converged_updates_
                                   >= std::max(1, convergence_criteria_.converged_updates))
                                    ? ConvergenceStatus::Converged
                                    : ConvergenceStatus::Converging;
            return;
        }

        consecutive_converged_updates_ = 0;
        convergence_status_            = ConvergenceStatus::Converging;
    }

    void register_divergence_sample_() noexcept {
        consecutive_converged_updates_ = 0;
        ++consecutive_diverged_updates_;
        convergence_status_ =
            (consecutive_diverged_updates_ >= std::max(1, convergence_criteria_.diverged_updates))
                ? ConvergenceStatus::Diverging
                : ConvergenceStatus::Converging;
    }

    void mark_immediate_divergence_() noexcept {
        consecutive_converged_updates_ = 0;
        consecutive_diverged_updates_  = std::max(1, convergence_criteria_.diverged_updates);
        convergence_status_            = ConvergenceStatus::Diverging;
    }

    void update_covariance_stat_(const MatrixXX& covariance) noexcept {
        max_covariance_diag_ = max_abs_covariance_diag(covariance);
    }

    static double max_abs_covariance_diag(const MatrixXX& covariance) noexcept {
        double max_diag = 0.0;
        for (int i = 0; i < N_X; ++i) {
            const double diag = covariance(i, i);
            if (!std::isfinite(diag)) {
                return std::numeric_limits<double>::infinity();
            }
            max_diag = std::max(max_diag, std::abs(diag));
        }
        return max_diag;
    }

    static double min_covariance_diag(const MatrixXX& covariance) noexcept {
        double min_diag = std::numeric_limits<double>::infinity();
        for (int i = 0; i < N_X; ++i) {
            const double diag = covariance(i, i);
            if (!std::isfinite(diag)) {
                return -std::numeric_limits<double>::infinity();
            }
            min_diag = std::min(min_diag, diag);
        }
        return min_diag;
    }

    PredicFunc f;
    MeasureFunc h;
    UpdateQFunc update_Q;
    MatrixXX Q;
    UpdateRFunc update_R;
    MatrixZZ R;
    MatrixXX P_pri;
    MatrixXX P_post;
    MatrixXZ K_;
    MatrixX1 x_pri;
    MatrixX1 x_post;

    ConvergenceCriteria convergence_criteria_{};
    ConvergenceStatus convergence_status_{ConvergenceStatus::Unknown};
    int consecutive_converged_updates_{0};
    int consecutive_diverged_updates_{0};
    double normalized_innovation_squared_{std::numeric_limits<double>::infinity()};
    double max_covariance_diag_{std::numeric_limits<double>::infinity()};
};
