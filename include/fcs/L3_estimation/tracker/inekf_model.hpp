#pragma once

#include "inekf_config.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <concepts>

namespace fcs::L3 {

// ============================================================================
// InEKFModel Concept
// ============================================================================

/// Minimal concept for models compatible with InEKF geometry conventions.
template <typename M>
concept InEKFModel = requires {
    { M::ARMORS_NUM } -> std::convertible_to<int>;
};

// ============================================================================
// InEKFModelTraits
// ============================================================================

template <InEKFModel Model, bool IsRobot = (Model::ARMORS_NUM == 4)>
struct InEKFModelTraits;

/// Robot-like target (4 armors):
/// extra = [v_yaw, r0, r1, h]
template <InEKFModel Model>
struct InEKFModelTraits<Model, true> {
    static_assert(Model::ARMORS_NUM == 4, "Robot-like InEKF model must have ARMORS_NUM == 4");
    static_assert(
        requires {
            Model::kHighArmorIdA;
            Model::kHighArmorIdB;
        }, "Robot-like model must define static constexpr kHighArmorIdA/kHighArmorIdB (see "
           "motion_model.hpp::RobotModel)");

    static constexpr int kArmorsNum  = Model::ARMORS_NUM;
    static constexpr bool kIsRobot   = true;
    static constexpr bool kIsOutpost = false;
    static constexpr int kExtraDim   = 4;
    using Config                     = RobotInEKFConfig;

    static constexpr int kExtraIdxVYaw = 0;
    static constexpr int kExtraIdxR0   = 1;
    static constexpr int kExtraIdxR1   = 2;
    static constexpr int kExtraIdxH    = 3;

    static constexpr double kMinRadius = 0.17;
    static constexpr double kMaxRadius = 0.27;
    static constexpr double kMinHeight = -0.2;
    static constexpr double kMaxHeight = 0.2;

    template <typename VecE>
    static void init_extra(VecE& extra, const Config& config) noexcept {
        extra.setZero();
        extra(kExtraIdxVYaw) = 0.0;
        extra(kExtraIdxR0)   = config.radius0;
        extra(kExtraIdxR1)   = config.radius1;
        extra(kExtraIdxH)    = config.height;
        clamp_extra_in_place(extra);
    }

    template <typename MatX>
    static void init_geometry_covariance(MatX& P, int group_dim) noexcept {
        P(group_dim + kExtraIdxR0, group_dim + kExtraIdxR0) = 0.01;
        P(group_dim + kExtraIdxR1, group_dim + kExtraIdxR1) = 0.01;
        P(group_dim + kExtraIdxH, group_dim + kExtraIdxH)   = 0.01;
    }

    template <typename MatX>
    static void add_geometry_process_noise(
        MatX& Q, double dt, const Config& config, int group_dim) noexcept {
        const double sigma_r2 = config.sigma_r * config.sigma_r;
        const double sigma_h2 = config.sigma_h * config.sigma_h;
        Q(group_dim + kExtraIdxR0, group_dim + kExtraIdxR0) += sigma_r2 * dt; // r0
        Q(group_dim + kExtraIdxR1, group_dim + kExtraIdxR1) += sigma_r2 * dt; // r1
        Q(group_dim + kExtraIdxH, group_dim + kExtraIdxH) += sigma_h2 * dt;   // h
    }

    template <typename VecE>
    static void clamp_extra_in_place(VecE& extra) noexcept {
        if (!std::isfinite(extra(kExtraIdxVYaw))) {
            extra(kExtraIdxVYaw) = 0.0;
        }
        if (!std::isfinite(extra(kExtraIdxR0))) {
            extra(kExtraIdxR0) = 0.23;
        }
        if (!std::isfinite(extra(kExtraIdxR1))) {
            extra(kExtraIdxR1) = 0.23;
        }
        if (!std::isfinite(extra(kExtraIdxH))) {
            extra(kExtraIdxH) = 0.0;
        }
        extra(kExtraIdxR0) = std::clamp(extra(kExtraIdxR0), kMinRadius, kMaxRadius);
        extra(kExtraIdxR1) = std::clamp(extra(kExtraIdxR1), kMinRadius, kMaxRadius);
        extra(kExtraIdxH)  = std::clamp(extra(kExtraIdxH), kMinHeight, kMaxHeight);
    }

    template <typename Vec3, typename Mat3xX, typename VecE>
    static void fill_armor_offset_and_geometry_jacobian(
        int armor_id, const Eigen::Matrix3d& R_id, const VecE& extra, Vec3& b, Mat3xX& C,
        int group_dim) noexcept {
        static constexpr int kHighArmorIdA = Model::kHighArmorIdA;
        static constexpr int kHighArmorIdB = Model::kHighArmorIdB;

        const bool is_high    = (armor_id == kHighArmorIdA || armor_id == kHighArmorIdB);
        const double radius   = is_high ? extra(kExtraIdxR1) : extra(kExtraIdxR0);
        const double z_offset = is_high ? extra(kExtraIdxH) : 0.0;
        b                     = (R_id * Vec3{-radius, 0.0, z_offset}).eval();

        const Vec3 db_dr = (R_id * Vec3{-1.0, 0.0, 0.0}).eval();
        const Vec3 db_dh = (R_id * Vec3{0.0, 0.0, 1.0}).eval();
        if (is_high) {
            C.template block<3, 1>(0, group_dim + kExtraIdxR1) = db_dr;       // r1
            C.template block<3, 1>(0, group_dim + kExtraIdxH)  = db_dh;       // h
        } else {
            C.template block<3, 1>(0, group_dim + kExtraIdxR0) = db_dr;       // r0
        }
    }
};

/// Outpost-like target (3 armors):
/// extra = [v_yaw, dz1, dz2]
template <InEKFModel Model>
struct InEKFModelTraits<Model, false> {
    static_assert(Model::ARMORS_NUM == 3, "Outpost-like InEKF model must have ARMORS_NUM == 3");
    static_assert(
        requires { Model::OUTPOST_RADIUS; },
        "Outpost-like model must define static constexpr OUTPOST_RADIUS");

    static constexpr int kArmorsNum  = Model::ARMORS_NUM;
    static constexpr bool kIsRobot   = false;
    static constexpr bool kIsOutpost = true;
    static constexpr int kExtraDim   = 3;
    using Config                     = OutpostInEKFConfig;

    static constexpr int kExtraIdxVYaw = 0;
    static constexpr int kExtraIdxDz1  = 1;
    static constexpr int kExtraIdxDz2  = 2;

    template <typename VecE>
    static void init_extra(VecE& extra, const Config&) noexcept {
        extra.setZero();
        extra(kExtraIdxVYaw) = 0.0;
        extra(kExtraIdxDz1)  = 0.0;
        extra(kExtraIdxDz2)  = 0.0;
        clamp_extra_in_place(extra);
    }

    template <typename MatX>
    static void init_geometry_covariance(MatX& P, int group_dim) noexcept {
        P(group_dim + kExtraIdxDz1, group_dim + kExtraIdxDz1) = 0.01;
        P(group_dim + kExtraIdxDz2, group_dim + kExtraIdxDz2) = 0.01;
    }

    template <typename MatX>
    static void add_geometry_process_noise(
        MatX& Q, double dt, const Config& config, int group_dim) noexcept {
        const double sigma_h2 = config.sigma_h * config.sigma_h;
        Q(group_dim + kExtraIdxDz1, group_dim + kExtraIdxDz1) += sigma_h2 * dt; // dz1
        Q(group_dim + kExtraIdxDz2, group_dim + kExtraIdxDz2) += sigma_h2 * dt; // dz2
    }

    template <typename VecE>
    static void clamp_extra_in_place(VecE& extra) noexcept {
        if (!std::isfinite(extra(kExtraIdxVYaw))) {
            extra(kExtraIdxVYaw) = 0.0;
        }
        if (!std::isfinite(extra(kExtraIdxDz1))) {
            extra(kExtraIdxDz1) = 0.0;
        }
        if (!std::isfinite(extra(kExtraIdxDz2))) {
            extra(kExtraIdxDz2) = 0.0;
        }
    }

    template <typename Vec3, typename Mat3xX, typename VecE>
    static void fill_armor_offset_and_geometry_jacobian(
        int armor_id, const Eigen::Matrix3d& R_id, const VecE& extra, Vec3& b, Mat3xX& C,
        int group_dim) noexcept {
        const double radius = static_cast<double>(Model::OUTPOST_RADIUS);
        double z_offset     = 0.0;
        if (armor_id == 1) {
            z_offset = extra(kExtraIdxDz1);
        } else if (armor_id == 2) {
            z_offset = extra(kExtraIdxDz2);
        }
        b = (R_id * Vec3{-radius, 0.0, z_offset}).eval();

        const Vec3 db_dz = (R_id * Vec3{0.0, 0.0, 1.0}).eval();
        if (armor_id == 1) {
            C.template block<3, 1>(0, group_dim + kExtraIdxDz1) = db_dz;        // dz1
        } else if (armor_id == 2) {
            C.template block<3, 1>(0, group_dim + kExtraIdxDz2) = db_dz;        // dz2
        }
    }
};

} // namespace fcs::L3
