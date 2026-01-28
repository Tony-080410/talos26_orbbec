#pragma once

#include "../euler.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <groups/SEn3.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <utility>

namespace fast_tf {

template <typename T>
class TransformMatrix4 {
public:
    using SE3     = group::SEn3<T, 1>;
    using Matrix  = typename SE3::MatrixType;
    using Tangent = typename SE3::VectorType;

    TransformMatrix4() = default;

    explicit TransformMatrix4(const SE3& pose)
        : pose_(pose) {}

    explicit TransformMatrix4(SE3&& pose)
        : pose_(std::move(pose)) {}

    explicit TransformMatrix4(const Matrix& t)
        : pose_(t) {}

    // ========================================================================
    // Factory Methods
    // ========================================================================

    static TransformMatrix4 from_pnp(cv::Vec<T, 3> rvec, cv::Vec<T, 3> tvec) {
        cv::Mat rmat_cv;
        cv::Rodrigues(rvec, rmat_cv);

        Eigen::Matrix<T, 3, 3> R;
        cv::cv2eigen(rmat_cv, R);

        // Stricter determinant check for numerical stability
        const T det = R.determinant();
        if (std::abs(det - T(1)) > T(1e-10)) {
            // SVD orthogonalization to fix numerical errors
            Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(
                R, Eigen::ComputeFullU | Eigen::ComputeFullV);
            R = svd.matrixU() * svd.matrixV().transpose();

            // Ensure right-handed coordinate system
            if (R.determinant() < T(0)) {
                R.col(2) = -R.col(2);
            }

            // Final validation
            const T det_new = R.determinant();
            assert(std::abs(det_new - T(1)) < T(1e-12));
        }

        return from_rt(R, Eigen::Matrix<T, 3, 1>(tvec[0], tvec[1], tvec[2]));
    }

    /// Create EulerRot::XYZ intrinsic
    static TransformMatrix4 from_rpy(T roll, T pitch, T yaw, T x = T(0), T y = T(0), T z = T(0)) {
        const auto q = math_fuxk::rpy<T>(roll, pitch, yaw).quat();
        return from_rt(q.toRotationMatrix(), Eigen::Matrix<T, 3, 1>(x, y, z));
    }

    /// Create pure translation (identity rotation)
    static TransformMatrix4 from_translation(T x, T y, T z) {
        return from_rt(Eigen::Matrix<T, 3, 3>::Identity(), Eigen::Matrix<T, 3, 1>(x, y, z));
    }

    static TransformMatrix4 from_quaternion(
        const Eigen::Quaternion<T>& q_in,
        const Eigen::Matrix<T, 3, 1>& t = Eigen::Matrix<T, 3, 1>::Zero()) {
        Eigen::Quaternion<T> q = q_in;

        // Check if normalization is needed (avoid sqrt precision loss)
        const T norm_sq = q.squaredNorm();
        if (std::abs(norm_sq - T(1)) > T(1e-12)) {
            // Only normalize when necessary to reduce precision loss
            q.normalize();
        }

        return from_rt(q.toRotationMatrix(), t);
    }

    static TransformMatrix4
        from_quaternion_xyz(const Eigen::Quaternion<T>& q, T x = T(0), T y = T(0), T z = T(0)) {
        return from_quaternion(q, Eigen::Matrix<T, 3, 1>(x, y, z));
    }

    static TransformMatrix4
        from_rt(const Eigen::Matrix<T, 3, 3>& R, const Eigen::Matrix<T, 3, 1>& t) {
        Matrix tt                     = Matrix::Identity();
        tt.template block<3, 3>(0, 0) = R;
        tt.template block<3, 1>(0, 3) = t;
        return TransformMatrix4(tt);
    }

    // ========================================================================
    // Operators
    // ========================================================================

    TransformMatrix4 operator*(const TransformMatrix4& other) const {
        return TransformMatrix4(pose_ * other.pose_);
    }

    [[nodiscard]] TransformMatrix4 reparent_to(const TransformMatrix4& parent_tf) const {
        return parent_tf * *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] TransformMatrix4 inverse() const { return TransformMatrix4(pose_.inv()); }

    [[nodiscard]] const SE3& se3() const { return pose_; }
    [[nodiscard]] Matrix matrix() const { return pose_.T(); }

    [[nodiscard]] Eigen::Matrix<T, 3, 1> translation() const { return pose_.x(); }

    [[nodiscard]] Eigen::Matrix<T, 3, 3> rotation() const { return pose_.R(); }

    [[nodiscard]] Eigen::Quaternion<T> quaternion() const { return pose_.q(); }

    // ========================================================================
    // Right-invariant Lie ops
    // ========================================================================

    /// Right-minus: δ = Log(this^{-1} · other)
    [[nodiscard]] Tangent rminus(const TransformMatrix4& other) const {
        return SE3::log(pose_.inv() * other.pose_);
    }

    /// Right-plus: this ⊕ δ = this · Exp(δ)
    [[nodiscard]] TransformMatrix4 rplus(const Tangent& delta) const {
        return TransformMatrix4(pose_ * SE3::exp(delta));
    }

    [[nodiscard]] Tangent log() const { return SE3::log(pose_); }
    static TransformMatrix4 exp(const Tangent& xi) { return TransformMatrix4(SE3::exp(xi)); }

    [[nodiscard]] math_fuxk::Ros2EulerRot<T> euler_rot() const {
        return math_fuxk::rpy<T>(quaternion());
    }

    // ========================================================================
    // Interpolation
    // ========================================================================

    static TransformMatrix4 lerp(const TransformMatrix4& a, const TransformMatrix4& b, T t) {
        if (t <= T(0)) {
            return a;
        }
        if (t >= T(1)) {
            return b;
        }
        const Eigen::Matrix<T, 3, 1> pos = a.translation() * (T(1) - t) + b.translation() * t;
        const Eigen::Quaternion<T> q     = a.quaternion().slerp(t, b.quaternion());
        return from_quaternion(q, pos);
    }

    /// SE(3) geodesic interpolation (right-invariant):
    /// γ(t) = a · Exp(t · Log(a^{-1} · b))
    static TransformMatrix4 lerp_se3(const TransformMatrix4& a, const TransformMatrix4& b, T t) {
        constexpr T epsilon = T(1e-10);

        // Use relative error for boundary conditions
        if (t < epsilon) {
            return a;
        }
        if (t > T(1) - epsilon) {
            return b;
        }

        // Midpoint interpolation: use standard formula
        const SE3 delta  = a.pose_.inv() * b.pose_;
        const auto xi    = SE3::log(delta);
        const SE3 interp = a.pose_ * SE3::exp(xi * t);
        return TransformMatrix4(interp);
    }

private:
    SE3 pose_{};
};

// ===== Type Aliases =====
using TransformMatrix4d = TransformMatrix4<double>;
using TransformMatrix4f = TransformMatrix4<float>;

} // namespace fast_tf
