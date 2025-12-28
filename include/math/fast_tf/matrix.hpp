#pragma once

#include "../euler.hpp"
#include "types.hpp"

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/matx.hpp>
#include <utility>

namespace fast_tf {

// ============================================================================
// Transform Matrix Wrapper
// ============================================================================

/// 4x4 Transform matrix wrapper with utility methods

class TransformMatrix4d {
public:
    using Matrix = Eigen::Matrix4d;

    TransformMatrix4d()
        : matrix_(Matrix::Identity()) {}

    explicit TransformMatrix4d(Matrix m)
        : matrix_(std::move(m)) {}

    // ========================================================================
    // Factory Methods
    // ========================================================================

    static TransformMatrix4d from_pnp(cv::Vec3d rvec, cv::Vec3d tvec) {
        cv::Mat rmat_;
        cv::Rodrigues(rvec, rmat_);
        Eigen::Matrix3d rmat;
        cv::cv2eigen(rmat_, rmat);

        // 旋转矩阵有效性检查
        assert(
            std::abs(rmat.determinant() - 1.0) < 1e-6 && "Rotation matrix must have determinant 1");

        Matrix m            = Matrix::Identity();
        m.block<3, 3>(0, 0) = rmat;
        m(0, 3)             = tvec[0];
        m(1, 3)             = tvec[1];
        m(2, 3)             = tvec[2];
        return TransformMatrix4d(m);
    }

    /// Create EulerRot::XYZ intrinsic
    static TransformMatrix4d from_rpy(
        const fp_t roll, const fp_t pitch, const fp_t yaw, const fp_t x = 0.0, const fp_t y = 0.0,
        const fp_t z = 0.0) {
        const auto q = math_fuxk::rpy(roll, pitch, yaw).quat();

        Matrix m            = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m(0, 3)             = x;
        m(1, 3)             = y;
        m(2, 3)             = z;
        return TransformMatrix4d(m);
    }

    /// Create pure translation (identity rotation)
    static TransformMatrix4d from_translation(const fp_t x, const fp_t y, const fp_t z) {
        Matrix m = Matrix::Identity();
        m(0, 3)  = x;
        m(1, 3)  = y;
        m(2, 3)  = z;
        return TransformMatrix4d(m);
    }

    // ========================================================================
    // Operators
    // ========================================================================

    TransformMatrix4d operator*(const TransformMatrix4d& other) const {
        return TransformMatrix4d(matrix_ * other.matrix_);
    }

    [[nodiscard]] TransformMatrix4d reparent_to(const TransformMatrix4d& parent_tf) const {
        return parent_tf * *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] TransformMatrix4d inverse() const { return TransformMatrix4d(matrix_.inverse()); }

    [[nodiscard]] const Matrix& matrix() const { return matrix_; }
    [[nodiscard]] Matrix& matrix() { return matrix_; }

    [[nodiscard]] Eigen::Vector3d translation() const { return matrix_.block<3, 1>(0, 3); }
    [[nodiscard]] math_fuxk::Ros2EulerRotd euler_rot() const {
        return math_fuxk::rpy<double>(quaternion());
    }

    [[nodiscard]] Eigen::Matrix3d rotation() const { return matrix_.block<3, 3>(0, 0); }

    [[nodiscard]] Eigen::Quaterniond quaternion() const { return Eigen::Quaterniond(rotation()); }

    // ========================================================================
    // Interpolation
    // ========================================================================

    static TransformMatrix4d
        lerp(const TransformMatrix4d& a, const TransformMatrix4d& b, const fp_t t) {
        // Position linear interpolation
        const Eigen::Vector3d pos = a.translation() * (1.0 - t) + b.translation() * t;

        // Quaternion spherical linear interpolation
        // Note: Eigen's slerp handles quaternion double-cover (q and -q represent same rotation)
        // by checking dot product sign internally, ensuring shortest arc interpolation
        const Eigen::Quaterniond qa(a.rotation());
        const Eigen::Quaterniond qb(b.rotation());
        const Eigen::Quaterniond q = qa.slerp(t, qb);

        Matrix m            = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m.block<3, 1>(0, 3) = pos;

        return TransformMatrix4d(m);
    }

private:
    Matrix matrix_;
};
} // namespace fast_tf
