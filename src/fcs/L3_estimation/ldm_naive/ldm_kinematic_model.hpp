#pragma once

#include "L3_estimation/tracker/util.hpp"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <groups/SEn3.hpp>

namespace fcs::L3::ldm {

enum LdmPoseUpdate : uint8_t {
    ROT_X,
    ROT_Y,
    ROT_Z,
    BEARING_YAW,
    BEARING_PITCH,
    BEARING_DISTANCE,
    POSE_UPDATE_MAX
};

struct LdmKinematic {
    using Scalar            = double;
    static constexpr int NZ = POSE_UPDATE_MAX;

    using Nominal           = group::SEn3<Scalar, 2>;
    using Xi                = Nominal::VectorType;
    using CovXi             = Nominal::TMatrixType;
    static constexpr int NX = Xi::RowsAtCompileTime;

    using Innovation         = Eigen::Matrix<Scalar, NZ, 1>;
    using CovInnovation      = Eigen::Matrix<Scalar, NZ, NZ>;
    using InnovationJacobian = Eigen::Matrix<Scalar, NZ, NX>;

    struct PoseMeasurement {
        Eigen::Matrix3d R_world_body{Eigen::Matrix3d::Identity()};
        Eigen::Vector3d p_world_body{Eigen::Vector3d::Zero()};
    };

    struct Params {
        Scalar sigma_inert_omega = Scalar(50.0); // rad/s
        Scalar sigma_inert_accel = Scalar(50.0); // m/s^2

        // SO(3) log residual noise, rad.
        Scalar sigma_rot_x = Scalar(0.5);
        Scalar sigma_rot_y = Scalar(0.5);
        Scalar sigma_rot_z = Scalar(1.0);

        // Bearing measurement noise, rad.
        Scalar sigma_r_bearing_yaw   = Scalar(5e-1);
        Scalar sigma_r_bearing_pitch = Scalar(2e-1);

        // Distance measurement noise:
        // sigma_d = sigma_distance_min
        //         + k_distance_depth * abs(depth)
        //         + k_distance_planar * planar_offset
        Scalar sigma_distance_min = Scalar(0.01); // m
        Scalar k_distance_depth   = Scalar(0.5);  // m / m
        Scalar k_distance_planar  = Scalar(0.5);  // m / m
    };
    Params params{};

    static Nominal predict_state(const Nominal& X, Scalar dt) noexcept {
        if (dt <= Scalar(0)) {
            return X;
        }

        const auto v = X.v();

        Xi xi = Xi::Zero();

        // SEn3<*,2> stores [rotation, body-frame velocity, world position].
        // Right-multiplying dp integrates p_dot = R * v_body.
        xi.template segment<3>(6).noalias() = v * dt;

        return X * Nominal::exp(xi);
    }

    [[nodiscard]] Nominal f(const Nominal& x, double dt) const noexcept {
        return predict_state(x, dt);
    }

    [[nodiscard]] static Innovation
        pose_innovation(const Nominal& predicted, const PoseMeasurement& observed) noexcept {
        using SO3 = typename Nominal::SO3Type;

        Innovation nu = Innovation::Zero();
        const SO3 dR(predicted.R().transpose() * observed.R_world_body);
        nu.template segment<3>(0) = SO3::log(dR);

        const Eigen::Vector3d predicted_ypd = fcs::L3::xyz2ypd(predicted.p());
        const Eigen::Vector3d observed_ypd  = fcs::L3::xyz2ypd(observed.p_world_body);
        nu[BEARING_YAW]      = fcs::L3::shortest_rad(predicted_ypd.x(), observed_ypd.x());
        nu[BEARING_PITCH]    = fcs::L3::shortest_rad(predicted_ypd.y(), observed_ypd.y());
        nu[BEARING_DISTANCE] = observed_ypd.z() - predicted_ypd.z();
        return nu;
    }

    [[nodiscard]] static InnovationJacobian pose_update_H(const Nominal& predicted) noexcept {
        InnovationJacobian H = InnovationJacobian::Zero();
        H.template block<3, 3>(ROT_X, 0).setIdentity();
        H.template block<3, 3>(BEARING_YAW, 6).noalias() =
            fcs::L3::xyz2ypd_jacobian(predicted.p()) * predicted.R();
        return H;
    }

    [[nodiscard]] CovXi Q(double dt) const noexcept {
        CovXi Q = CovXi::Zero();
        if (dt <= Scalar(0)) {
            return Q;
        }

        using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

        const Mat3 I3 = Mat3::Identity();

        const Scalar dt2 = dt * dt;
        const Scalar dt3 = dt2 * dt;
        const Scalar dt4 = dt2 * dt2;

        const Scalar omega_var = params.sigma_inert_omega * params.sigma_inert_omega; // (rad/s)^2
        const Scalar accel_var = params.sigma_inert_accel * params.sigma_inert_accel; // (m/s^2)^2

        // Xi = [dtheta, dv, dp]

        // dtheta_noise = omega_noise * dt
        Q.template block<3, 3>(0, 0) = omega_var * dt2 * I3; // rad^2

        // dv_noise = accel_noise * dt
        Q.template block<3, 3>(3, 3) = accel_var * dt2 * I3; // (m/s)^2

        // dp_noise = 0.5 * accel_noise * dt^2
        Q.template block<3, 3>(6, 6) = Scalar(0.25) * accel_var * dt4 * I3; // m^2

        // Cov(dv, dp)
        Q.template block<3, 3>(3, 6) = Scalar(0.5) * accel_var * dt3 * I3; // m^2/s

        Q.template block<3, 3>(6, 3) = Q.template block<3, 3>(3, 6).transpose();

        return Q;
    }

    [[nodiscard]] CovInnovation R(const PoseMeasurement& z) const noexcept {
        using std::abs;
        using std::cos;
        using std::sin;
        using std::sqrt;

        Eigen::Matrix<Scalar, NZ, 1> diag;
        diag.setZero();

        const Eigen::Vector3d ypd = fcs::L3::xyz2ypd(z.p_world_body);

        const Scalar yaw      = ypd.x();
        const Scalar pitch    = ypd.y();
        const Scalar distance = abs(ypd.z());

        // Bearing convention:
        //
        // x = d cos(pitch) cos(yaw)
        // y = d cos(pitch) sin(yaw)
        // z = -d sin(pitch)
        //
        // depth: forward component
        // planar_offset: off-axis component
        const Scalar cp = cos(pitch);
        const Scalar sp = sin(pitch);
        const Scalar cy = cos(yaw);
        const Scalar sy = sin(yaw);

        const Scalar depth = abs(distance * cp * cy);

        const Scalar lateral       = distance * cp * sy;
        const Scalar vertical      = -distance * sp;
        const Scalar planar_offset = sqrt(lateral * lateral + vertical * vertical);

        const Scalar sigma_distance = params.sigma_distance_min + params.k_distance_depth * depth
                                    + params.k_distance_planar * planar_offset;

        diag[ROT_X] = params.sigma_rot_x * params.sigma_rot_x;

        diag[ROT_Y] = params.sigma_rot_y * params.sigma_rot_y;

        diag[ROT_Z] = params.sigma_rot_z * params.sigma_rot_z;

        diag[BEARING_YAW] = params.sigma_r_bearing_yaw * params.sigma_r_bearing_yaw;

        diag[BEARING_PITCH] = params.sigma_r_bearing_pitch * params.sigma_r_bearing_pitch;

        diag[BEARING_DISTANCE] = sigma_distance * sigma_distance;

        CovInnovation R = CovInnovation::Zero();
        R.diagonal()    = diag;
        return R;
    }
};

} // namespace fcs::L3::ldm
