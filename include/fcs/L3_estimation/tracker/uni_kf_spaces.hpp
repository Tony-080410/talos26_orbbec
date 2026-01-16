#pragma once

#include <uni_kf/concepts/differential_space.hpp>
#include <uni_kf/core/types.hpp>
#include <uni_kf/spaces/euclidean.hpp>
#include <uni_kf/spaces/product_space.hpp>

#include <groups/SO3.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace fcs::L3::ukf {

struct AxisZ {
    [[nodiscard]] static auto unit() noexcept -> Eigen::Vector3d { return {0.0, 0.0, 1.0}; }
};

struct AxisY {
    [[nodiscard]] static auto unit() noexcept -> Eigen::Vector3d { return {0.0, 1.0, 0.0}; }
};

template <typename AxisTag, typename Scalar_ = double>
struct AxisSO3 {
    using Point  = group::SO3<Scalar_>;
    using Scalar = Scalar_;
    using Fiber  = uni_kf::VecN<1, Scalar>;

    static constexpr int FiberDim = 1;

    using JacobianType = uni_kf::SquareMatN<FiberDim, Scalar>;

    [[nodiscard]] static Point identity() noexcept { return Point{}; }

    [[nodiscard]] static Point act(const Point& p, const Fiber& v) noexcept {
        const Scalar angle = v.data(0);
        const auto axis    = AxisTag::unit().template cast<Scalar>();
        const auto so3     = Point::exp(axis * angle);
        return p * so3;
    }

    [[nodiscard]] static Fiber diff(const Point& p, const Point& q) noexcept {
        const auto axis = AxisTag::unit().template cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> w;
        w = Point::log(p.inv() * q);
        return Fiber(typename Fiber::EigenType(axis.dot(w)));
    }

    [[nodiscard]] static JacobianType left_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }
    [[nodiscard]] static JacobianType right_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }
    [[nodiscard]] static JacobianType inv_left_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }
    [[nodiscard]] static JacobianType inv_right_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }

    [[nodiscard]] static Fiber adjoint(const Point& /*p*/, const Fiber& v) noexcept { return v; }
    [[nodiscard]] static JacobianType adjoint_matrix(const Point& /*p*/) noexcept {
        return JacobianType::Identity();
    }

    [[nodiscard]] static Fiber fiber_from_angle(Scalar angle) noexcept {
        typename Fiber::EigenType a;
        a << angle;
        return Fiber(a);
    }

    [[nodiscard]] static Point from_angle(Scalar angle) noexcept {
        return act(identity(), fiber_from_angle(angle));
    }

    [[nodiscard]] static Scalar angle(const Point& p) noexcept {
        return diff(identity(), p).data(0);
    }
};

using YawSO3Z   = AxisSO3<AxisZ, double>;
using PitchSO3Y = AxisSO3<AxisY, double>;

using R1 = uni_kf::Euclidean<1, double>;

// ----------------------------------------------------------------------------
// S2 direction space (unit vector on sphere)
// ----------------------------------------------------------------------------

struct S2 {
    using Scalar = double;
    using Point  = Eigen::Matrix<Scalar, 3, 1>; // unit vector
    using Fiber  = uni_kf::VecN<2, Scalar>;     // 2-DOF tangent coordinates

    static constexpr int FiberDim = 2;

    using JacobianType = uni_kf::SquareMatN<FiberDim, Scalar>;

    [[nodiscard]] static Point identity() noexcept { return Point{1.0, 0.0, 0.0}; }

    [[nodiscard]] static Point normalize_or_identity(const Point& p) noexcept {
        const Scalar n = p.norm();
        if (!(n > Scalar{0}) || !std::isfinite(n)) {
            return identity();
        }
        return (p / n).eval();
    }

    static void basis_at(const Point& p_unit, Point& b1, Point& b2) noexcept {
        const Point p = normalize_or_identity(p_unit);
        const Point ref =
            (std::abs(p.z()) < Scalar{0.9}) ? Point{0.0, 0.0, 1.0} : Point{1.0, 0.0, 0.0};
        b1              = ref.cross(p);
        const Scalar n1 = b1.norm();
        if (!(n1 > Scalar{0})) {
            // extremely degenerate; fall back
            b1 = Point{0.0, 1.0, 0.0}.cross(p);
        }
        b1.normalize();
        b2 = p.cross(b1);
        b2.normalize();
    }

    [[nodiscard]] static Point act(const Point& p_unit, const Fiber& v) noexcept {
        const Point p = normalize_or_identity(p_unit);
        Point b1, b2;
        basis_at(p, b1, b2);

        const Scalar a     = v.data(0);
        const Scalar b     = v.data(1);
        const Point t      = a * b1 + b * b2;
        const Scalar theta = t.norm();
        if (!(theta > Scalar{0}) || !std::isfinite(theta)) {
            return p;
        }
        const Point u = t / theta;
        const Point q = std::cos(theta) * p + std::sin(theta) * u;
        return normalize_or_identity(q);
    }

    [[nodiscard]] static Fiber diff(const Point& p_unit, const Point& q_unit) noexcept {
        const Point p = normalize_or_identity(p_unit);
        const Point q = normalize_or_identity(q_unit);

        const Scalar dot       = std::clamp<Scalar>(p.dot(q), Scalar{-1}, Scalar{1});
        const Point c          = p.cross(q);
        const Scalar sin_theta = c.norm();
        const Scalar theta     = std::atan2(sin_theta, dot);

        if (!(theta > Scalar{0}) || !(sin_theta > Scalar{0}) || !std::isfinite(theta)) {
            return Fiber::Zero();
        }

        // Tangent vector in R^3 at p
        const Point t3 = (theta / sin_theta) * (q - dot * p);

        Point b1, b2;
        basis_at(p, b1, b2);

        typename Fiber::EigenType v;
        v << b1.dot(t3), b2.dot(t3);
        return Fiber(v);
    }

    [[nodiscard]] static JacobianType left_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }
    [[nodiscard]] static JacobianType right_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }
    [[nodiscard]] static JacobianType inv_left_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }
    [[nodiscard]] static JacobianType inv_right_jacobian(const Fiber& /*v*/) noexcept {
        return JacobianType::Identity();
    }

    [[nodiscard]] static Fiber adjoint(const Point& /*p*/, const Fiber& v) noexcept { return v; }
    [[nodiscard]] static JacobianType adjoint_matrix(const Point& /*p*/) noexcept {
        return JacobianType::Identity();
    }
};

// Measurement space for tracker:
// - direction: S2 (2DOF)
// - distance: R1
// - armor_yaw: SO3_z (1DOF)
using ArmorMeasSpace = uni_kf::ProductSpace<S2, uni_kf::ProductSpace<R1, YawSO3Z>>;

struct ArmorMeas {
    using Space = ArmorMeasSpace;
    using Point = Space::Point;

    [[nodiscard]] static Point make(const Eigen::Vector3d& xyz, double armor_yaw) noexcept {
        const double distance = xyz.norm();
        const Eigen::Vector3d dir =
            (distance > 1e-9) ? (xyz / distance) : Eigen::Vector3d{1.0, 0.0, 0.0};
        typename R1::Point::EigenType d;
        d << distance;
        return {
            dir, {R1::Point(d), YawSO3Z::from_angle(armor_yaw)}
        };
    }

    [[nodiscard]] static Eigen::Vector3d direction(const Point& z) noexcept { return z.first; }
    [[nodiscard]] static double distance(const Point& z) noexcept { return z.second.first.data(0); }
    [[nodiscard]] static double armor_yaw(const Point& z) noexcept {
        return YawSO3Z::angle(z.second.second);
    }
};

} // namespace fcs::L3::ukf
