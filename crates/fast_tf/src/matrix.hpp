#pragma once

#include "euler.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <concepts>
#include <groups/SEn3.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <type_traits>
#include <utility>

namespace fast_tf {

/// Strongly typed SE(3) transform.
///
/// The algebra follows:
/// - T<X, Y> * T<Y, Z> -> T<X, Z>
/// - inv(T<X, Y>) -> T<Y, X>
/// - I<X> -> T<X, X>
///
/// Semantically, `TransformMatrix<T, From, To>` maps coordinates expressed in `From` into `To`.
template <typename T, typename From, typename To>
class TransformMatrix {
public:
    template <typename, typename, typename>
    friend class TransformMatrix;

    using SE3     = group::SEn3<T, 1>;
    using SO3     = group::SO3<T>;
    using Matrix  = typename SE3::MatrixType;
    using Tangent = typename SE3::VectorType;
    using Scalar  = T;
    using FromTag = From;
    using ToTag   = To;

    TransformMatrix() = default;

    explicit TransformMatrix(const SE3& pose)
        : pose_(pose) {}

    explicit TransformMatrix(SE3&& pose)
        : pose_(std::move(pose)) {}

    explicit TransformMatrix(const Matrix& t)
        : pose_(t) {}

    // ========================================================================
    // Factory Methods
    // ========================================================================

    static TransformMatrix from_pnp(cv::Vec<T, 3> rvec, cv::Vec<T, 3> tvec) {
        cv::Mat rmat_cv;
        cv::Rodrigues(rvec, rmat_cv);

        Eigen::Matrix<T, 3, 3> R;
        cv::cv2eigen(rmat_cv, R);

        assert(
            std::abs(R.determinant() - T(1)) < T(1e-6)
            && "Rotation matrix must have determinant 1");

        return from_rt(R, Eigen::Matrix<T, 3, 1>(tvec[0], tvec[1], tvec[2]));
    }

    /// Create EulerRot::XYZ intrinsic
    static TransformMatrix from_rpy(T roll, T pitch, T yaw, T x = T(0), T y = T(0), T z = T(0)) {
        const auto q = math_fuxk::rpy<T>(roll, pitch, yaw).quat();
        return from_rt(q.toRotationMatrix(), Eigen::Matrix<T, 3, 1>(x, y, z));
    }

    /// Create pure translation (identity rotation)
    static TransformMatrix from_translation(T x, T y, T z) {
        return from_rt(Eigen::Matrix<T, 3, 3>::Identity(), Eigen::Matrix<T, 3, 1>(x, y, z));
    }

    static TransformMatrix from_quaternion(
        const Eigen::Quaternion<T>& q_in,
        const Eigen::Matrix<T, 3, 1>& t = Eigen::Matrix<T, 3, 1>::Zero()) {
        Eigen::Quaternion<T> q = q_in;
        q.normalize();
        return from_rt(q.toRotationMatrix(), t);
    }

    static TransformMatrix
        from_quaternion_xyz(const Eigen::Quaternion<T>& q, T x = T(0), T y = T(0), T z = T(0)) {
        return from_quaternion(q, Eigen::Matrix<T, 3, 1>(x, y, z));
    }

    static TransformMatrix
        from_rt(const Eigen::Matrix<T, 3, 3>& R, const Eigen::Matrix<T, 3, 1>& t) {
        Matrix tt                     = Matrix::Identity();
        tt.template block<3, 3>(0, 0) = R;
        tt.template block<3, 1>(0, 3) = t;
        return TransformMatrix(tt);
    }

    static TransformMatrix identity() requires(std::same_as<From, To>) { return TransformMatrix{}; }

    static TransformMatrix Identity() requires(std::same_as<From, To>) { return identity(); }

    // ========================================================================
    // Type-level Transform Algebra
    // ========================================================================

    template <typename Next>
    [[nodiscard]] TransformMatrix<T, From, Next>
        operator*(const TransformMatrix<T, To, Next>& other) const {
        return TransformMatrix<T, From, Next>(other.pose_ * pose_);
    }

    [[nodiscard]] TransformMatrix<T, To, From> inverse() const {
        return TransformMatrix<T, To, From>(pose_.inv());
    }

    [[nodiscard]] TransformMatrix<T, To, From> inv() const { return inverse(); }

    // ========================================================================
    // Compatibility Helpers
    // ========================================================================

    template <typename Parent>
    [[nodiscard]] TransformMatrix<T, Parent, To>
        reparent_to(const TransformMatrix<T, Parent, From>& parent_tf) const {
        return parent_tf * *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const SE3& se3() const { return pose_; }
    [[nodiscard]] SO3 so3() const { return SO3(pose_.R()); }
    [[nodiscard]] Matrix matrix() const { return pose_.T(); }

    [[nodiscard]] Eigen::Matrix<T, 3, 1> translation() const { return pose_.x(); }

    [[nodiscard]] Eigen::Matrix<T, 3, 3> rotation() const { return pose_.R(); }

    [[nodiscard]] Eigen::Quaternion<T> quaternion() const { return pose_.q(); }

    // ========================================================================
    // Right-invariant Lie ops
    // ========================================================================

    /// Right-minus: δ = Log(this^{-1} · other)
    [[nodiscard]] Tangent rminus(const TransformMatrix& other) const {
        return SE3::log(pose_.inv() * other.pose_);
    }

    /// Right-plus: this ⊕ δ = this · Exp(δ)
    [[nodiscard]] TransformMatrix rplus(const Tangent& delta) const {
        return TransformMatrix(pose_ * SE3::exp(delta));
    }

    [[nodiscard]] Tangent log() const { return SE3::log(pose_); }
    static TransformMatrix exp(const Tangent& xi) { return TransformMatrix(SE3::exp(xi)); }

    [[nodiscard]] math_fuxk::Ros2EulerRot<T> euler_rot() const {
        return math_fuxk::rpy<T>(quaternion());
    }

    // ========================================================================
    // Interpolation
    // ========================================================================

    static TransformMatrix lerp(const TransformMatrix& a, const TransformMatrix& b, T t) {
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
    static TransformMatrix lerp_se3(const TransformMatrix& a, const TransformMatrix& b, T t) {
        if (t <= T(0)) {
            return a;
        }
        if (t >= T(1)) {
            return b;
        }

        const SE3 delta  = a.pose_.inv() * b.pose_;
        const auto xi    = SE3::log(delta);
        const SE3 interp = a.pose_ * SE3::exp(xi * t);
        return TransformMatrix(interp);
    }

private:
    SE3 pose_{};
};

template <typename Frame, typename Scalar = double>
[[nodiscard]] TransformMatrix<Scalar, Frame, Frame> I() {
    return TransformMatrix<Scalar, Frame, Frame>::identity();
}

template <typename T, typename From, typename To>
[[nodiscard]] TransformMatrix<T, To, From> inv(const TransformMatrix<T, From, To>& tf) {
    return tf.inv();
}

template <typename T, typename From, typename To>
using TypedTransformMatrix = TransformMatrix<T, From, To>;

template <typename From, typename To>
using TransformMatrixd = TransformMatrix<double, From, To>;

template <typename From, typename To>
using TransformMatrixf = TransformMatrix<float, From, To>;

} // namespace fast_tf
