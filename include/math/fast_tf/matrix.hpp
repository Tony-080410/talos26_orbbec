#pragma once

#include "math/euler.hpp"
#include "types.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <groups/SEn3.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <utility>

namespace fast_tf {

class TransformMatrix4d {
public:
    using SE3     = group::SE3d;
    using Matrix  = SE3::MatrixType;
    using Tangent = SE3::VectorType; // se(3) vector (right-trivialized)

    TransformMatrix4d() = default;

    explicit TransformMatrix4d(const SE3& pose)
        : pose_(pose) {}

    explicit TransformMatrix4d(SE3&& pose)
        : pose_(std::move(pose)) {}

    explicit TransformMatrix4d(const Matrix& T)
        : pose_(T) {}

    // ========================================================================
    // Factory Methods
    // ========================================================================

    static TransformMatrix4d from_pnp(cv::Vec3d rvec, cv::Vec3d tvec) {
        cv::Mat rmat_cv;
        cv::Rodrigues(rvec, rmat_cv);

        Eigen::Matrix3d R;
        cv::cv2eigen(rmat_cv, R);

        assert(std::abs(R.determinant() - 1.0) < 1e-6 && "Rotation matrix must have determinant 1");

        return from_rt(R, Eigen::Vector3d(tvec[0], tvec[1], tvec[2]));
    }

    /// Create EulerRot::XYZ intrinsic
    static TransformMatrix4d
        from_rpy(fp_t roll, fp_t pitch, fp_t yaw, fp_t x = 0.0, fp_t y = 0.0, fp_t z = 0.0) {
        const auto q = math_fuxk::rpy(roll, pitch, yaw).quat();
        return from_rt(q.toRotationMatrix(), Eigen::Vector3d(x, y, z));
    }

    /// Create pure translation (identity rotation)
    static TransformMatrix4d from_translation(fp_t x, fp_t y, fp_t z) {
        return from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(x, y, z));
    }

    static TransformMatrix4d from_quaternion(
        const Eigen::Quaterniond& q_in, const Eigen::Vector3d& t = Eigen::Vector3d::Zero()) {
        Eigen::Quaterniond q = q_in;
        q.normalize();
        return from_rt(q.toRotationMatrix(), t);
    }

    static TransformMatrix4d
        from_quaternion_xyz(const Eigen::Quaterniond& q, fp_t x = 0.0, fp_t y = 0.0, fp_t z = 0.0) {
        return from_quaternion(q, Eigen::Vector3d(x, y, z));
    }

    static TransformMatrix4d from_rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        Matrix T                     = Matrix::Identity();
        T.template block<3, 3>(0, 0) = R;
        T.template block<3, 1>(0, 3) = t;
        return TransformMatrix4d(T);
    }

    // ========================================================================
    // Operators
    // ========================================================================

    TransformMatrix4d operator*(const TransformMatrix4d& other) const {
        return TransformMatrix4d(pose_ * other.pose_);
    }

    [[nodiscard]] TransformMatrix4d reparent_to(const TransformMatrix4d& parent_tf) const {
        return parent_tf * *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] TransformMatrix4d inverse() const { return TransformMatrix4d(pose_.inv()); }

    [[nodiscard]] const SE3& se3() const { return pose_; }
    [[nodiscard]] Matrix matrix() const { return pose_.T(); }

    [[nodiscard]] Eigen::Vector3d translation() const { return pose_.x(); }

    [[nodiscard]] Eigen::Matrix3d rotation() const { return pose_.R(); }

    [[nodiscard]] Eigen::Quaterniond quaternion() const { return pose_.q(); }

    // ========================================================================
    // Right-invariant Lie ops
    // ========================================================================

    /// Right-minus: δ = Log(this^{-1} · other)
    [[nodiscard]] Tangent rminus(const TransformMatrix4d& other) const {
        return SE3::log(pose_.inv() * other.pose_);
    }

    /// Right-plus: this ⊕ δ = this · Exp(δ)
    [[nodiscard]] TransformMatrix4d rplus(const Tangent& delta) const {
        return TransformMatrix4d(pose_ * SE3::exp(delta));
    }

    [[nodiscard]] Tangent log() const { return SE3::log(pose_); }
    static TransformMatrix4d exp(const Tangent& xi) { return TransformMatrix4d(SE3::exp(xi)); }

    [[nodiscard]] math_fuxk::Ros2EulerRotd euler_rot() const {
        return math_fuxk::rpy<double>(quaternion());
    }

    // ========================================================================
    // Interpolation
    // ========================================================================

    static TransformMatrix4d lerp(const TransformMatrix4d& a, const TransformMatrix4d& b, fp_t t) {
        if (t <= 0.0) {
            return a;
        }
        if (t >= 1.0) {
            return b;
        }
        const Eigen::Vector3d pos  = a.translation() * (1.0 - t) + b.translation() * t;
        const Eigen::Quaterniond q = a.quaternion().slerp(t, b.quaternion());
        return from_quaternion(q, pos);
    }

    /// SE(3) geodesic interpolation (right-invariant):
    /// γ(t) = a · Exp(t · Log(a^{-1} · b))
    static TransformMatrix4d
        lerp_se3(const TransformMatrix4d& a, const TransformMatrix4d& b, fp_t t) {
        if (t <= 0.0) {
            return a;
        }
        if (t >= 1.0) {
            return b;
        }
        const SE3 delta  = a.pose_.inv() * b.pose_;
        const auto xi    = SE3::log(delta);
        const SE3 interp = a.pose_ * SE3::exp(xi * t);
        return TransformMatrix4d(interp);
    }

private:
    SE3 pose_{};
};

} // namespace fast_tf
