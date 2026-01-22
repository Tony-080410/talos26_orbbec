#pragma once

#include "fcs/L3_estimation/energy_meter_solver/isrckf.hpp"
#include "fcs/L3_estimation/energy_meter_solver/types.hpp"

#include <Eigen/Dense>
#include <expected>
#include <string>

// Forward declaration to avoid including toml++ in headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

#include <cassert>
#include <cmath>

namespace energy_meter {

struct EnergyMeter {
    using Scalar            = double;
    static constexpr int NX = STATE_MAX;
    static constexpr int NZ = MEASURE_MAX;

    using VecX = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ = Eigen::Matrix<Scalar, NZ, 1>;

    struct Params {
        Scalar sigma_xy{10.0};
        Scalar sigma_z{1.0};
        Scalar sigma_roll{1.0};

        Scalar radius{0.7};

        Scalar meas_dist_k{0.43};

        std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
    };

    int armor_id{0};
    static constexpr int ARMORS_NUM = 5;

    Params params{};

    EnergyMeter() = default;

    explicit EnergyMeter(const Params& p)
        : params(p) {}

    auto f(const VecX& x, Scalar dt) const -> VecX {
        VecX xn;
        xn << x(XC), x(YC), x(ZC), x(ROLL) + x(V_ROLL) * dt, x(V_ROLL);
        return xn;
    }

    auto h(const VecX& x) const -> VecZ { return h(x, armor_id); }

    auto h(const VecX& x, int id) const -> VecZ {
        VecZ z;
        assert(id >= 0 && id <= ARMORS_NUM - 1);

        const Scalar angle_step = 2.0 * M_PI / static_cast<Scalar>(ARMORS_NUM);
        const double tmp_roll   = x(ROLL) + id * angle_step;

        z[3] = tmp_roll;

        const Scalar radius = params.radius;

        z[0] = x(XC);
        z[1] = x(YC) + radius * std::sin(tmp_roll);
        z[2] = x(ZC) - radius * std::cos(tmp_roll);

        return z;
    }

    auto Q_sqrt(Scalar dt) const -> Eigen::Matrix<Scalar, NX, NX> {
        using MatX = Eigen::Matrix<Scalar, NX, NX>;
        MatX Q     = MatX::Zero();
        if (dt <= Scalar(0)) {
            return Q;
        }

        const Scalar sigma_xy2   = params.sigma_xy * params.sigma_xy;
        const Scalar sigma_z2    = params.sigma_z * params.sigma_z;
        const Scalar sigma_roll2 = params.sigma_roll * params.sigma_roll;

        const Scalar dt2 = dt * dt;
        const Scalar dt3 = dt2 * dt;
        const Scalar dt4 = dt2 * dt2;

        Q(XC, XC) = sigma_xy2 * dt2;
        Q(YC, YC) = sigma_xy2 * dt2;
        Q(ZC, ZC) = sigma_z2 * dt2;

        Q(ROLL, ROLL)     = sigma_roll2 * dt4 / 4.0;
        Q(ROLL, V_ROLL)   = sigma_roll2 * dt3 / 2.0;
        Q(V_ROLL, ROLL)   = Q(ROLL, V_ROLL);
        Q(V_ROLL, V_ROLL) = sigma_roll2 * dt2;

        Eigen::LLT<MatX> llt(Q);
        return llt.matrixL();
    }

    auto R_sqrt(const VecZ& z) const -> Eigen::Matrix<Scalar, NZ, Eigen::Dynamic> {
        const auto ypd         = xyz2ypd({z[0], z[1], z[2]});
        const double delta_yaw = shortest_rad(ypd[0], z[3]);

        const double K      = params.meas_dist_k;
        const double weight = std::log(K * ypd[2] + 1.0);

        Eigen::Matrix<Scalar, NZ, NZ> Rs = Eigen::Matrix<Scalar, NZ, NZ>::Zero();
        Rs.diagonal() << weight + std::abs(delta_yaw / 3 + 1), weight + std::abs(delta_yaw / 3 + 1),
            weight, std::log(std::abs(ypd[2]) + 1) / 200 + 9e-2;
        return Rs;
    }
};

} // namespace energy_meter
