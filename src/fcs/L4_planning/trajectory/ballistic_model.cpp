#include "fcs/L4_planning/trajectory/ballistic_model.hpp"

#include <algorithm>
#include <limits>
#include <optional>

namespace fcs::L4::trajectory {

// ============================================================================
// RK4Model Implementation
// ============================================================================

double RK4Model::drag_k() const noexcept {
    const double d = std::max(1e-6, diameter_);
    const double A = std::numbers::pi * (d * 0.5) * (d * 0.5);
    const double m = std::max(1e-6, mass_);
    return 0.5 * cd_ * rho_ * A / m;
}

double RK4Model::select_dt(double range, double pitch, double v0) const noexcept {
    const double v0_clamped = std::max(1e-3, v0);
    const double c          = std::max(0.05, std::abs(std::cos(pitch)));
    const double t_est      = range / (v0_clamped * c);

    const double desired_dt = std::clamp(dt_, 1e-5, 0.02);
    const int min_steps     = std::max(1, min_steps_);
    const int max_steps     = std::max(min_steps, max_steps_);
    const int steps =
        std::clamp(static_cast<int>(std::ceil(t_est / desired_dt)), min_steps, max_steps);
    return t_est / static_cast<double>(steps);
}

std::optional<ImpactResult>
    RK4Model::compute_impact(double range, double pitch, double v0) const noexcept {
    const double k  = drag_k();
    const double dt = select_dt(range, pitch, v0);
    return detail::RK4Integrator::integrate_to_range(
        range, pitch, v0, k, gravity_, dt, min_steps_, max_steps_);
}

// ============================================================================
// RK4Integrator Implementation
// ============================================================================

namespace detail {

std::optional<ImpactResult> RK4Integrator::integrate_to_range(
    double range, double pitch, double v0, double k, double gravity, double dt, int min_steps,
    int max_steps) noexcept {

    if (range < 1e-6) {
        return ImpactResult{};
    }

    double x  = 0.0;
    double z  = 0.0;
    double vx = v0 * std::cos(pitch);
    double vz = v0 * std::sin(pitch);
    double t  = 0.0;

    double prev_x = x, prev_z = z, prev_vx = vx, prev_vz = vz, prev_t = t;

    auto accel = [&](double _vx, double _vz) noexcept {
        const double speed = std::hypot(_vx, _vz);
        const double ax    = speed > 1e-8 ? (-k * speed * _vx) : 0.0;
        const double az    = -gravity + (speed > 1e-8 ? (-k * speed * _vz) : 0.0);
        return std::pair<double, double>{ax, az};
    };

    for (int step = 0; step < max_steps; ++step) {
        prev_x  = x;
        prev_z  = z;
        prev_vx = vx;
        prev_vz = vz;
        prev_t  = t;

        const auto [ax1, az1] = accel(vx, vz);
        const double k1x      = vx;
        const double k1z      = vz;
        const double k1vx     = ax1;
        const double k1vz     = az1;

        const double vx2      = vx + 0.5 * dt * k1vx;
        const double vz2      = vz + 0.5 * dt * k1vz;
        const auto [ax2, az2] = accel(vx2, vz2);
        const double k2x      = vx2;
        const double k2z      = vz2;
        const double k2vx     = ax2;
        const double k2vz     = az2;

        const double vx3      = vx + 0.5 * dt * k2vx;
        const double vz3      = vz + 0.5 * dt * k2vz;
        const auto [ax3, az3] = accel(vx3, vz3);
        const double k3x      = vx3;
        const double k3z      = vz3;
        const double k3vx     = ax3;
        const double k3vz     = az3;

        const double vx4      = vx + dt * k3vx;
        const double vz4      = vz + dt * k3vz;
        const auto [ax4, az4] = accel(vx4, vz4);
        const double k4x      = vx4;
        const double k4z      = vz4;
        const double k4vx     = ax4;
        const double k4vz     = az4;

        x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        vx += (dt / 6.0) * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx);
        vz += (dt / 6.0) * (k1vz + 2.0 * k2vz + 2.0 * k3vz + k4vz);
        t += dt;

        if (x >= range) {
            const double denom = (x - prev_x);
            const double alpha =
                denom > 1e-9 ? std::clamp((range - prev_x) / denom, 0.0, 1.0) : 0.0;

            ImpactResult out;
            out.tof = prev_t + alpha * dt;
            out.x   = range;
            out.z   = prev_z + alpha * (z - prev_z);
            return out;
        }

        if (step + 1 >= min_steps && std::hypot(vx, vz) < 1e-3) {
            break;
        }
    }

    return std::nullopt;
}

std::vector<std::tuple<double, double, double>> RK4Integrator::integrate_full_trajectory(
    double v0, double pitch, double k, double gravity, double dt, int max_steps, double range_step,
    double max_range) noexcept {

    std::vector<std::tuple<double, double, double>> samples;
    if (range_step <= 0.0 || max_range <= 0.0) {
        return samples;
    }

    const size_t estimated_samples = static_cast<size_t>(max_range / range_step) + 1;
    samples.reserve(estimated_samples);

    double x  = 0.0;
    double z  = 0.0;
    double vx = v0 * std::cos(pitch);
    double vz = v0 * std::sin(pitch);
    double t  = 0.0;

    samples.emplace_back(0.0, 0.0, 0.0);

    double prev_x = x, prev_z = z, prev_t = t;
    double next_x = range_step;

    auto accel = [&](double _vx, double _vz) noexcept {
        const double speed = std::hypot(_vx, _vz);
        const double ax    = speed > 1e-8 ? (-k * speed * _vx) : 0.0;
        const double az    = -gravity + (speed > 1e-8 ? (-k * speed * _vz) : 0.0);
        return std::pair<double, double>{ax, az};
    };

    for (int step = 0; step < max_steps; ++step) {
        prev_x = x;
        prev_z = z;
        prev_t = t;

        const auto [ax1, az1] = accel(vx, vz);
        const double k1x      = vx;
        const double k1z      = vz;
        const double k1vx     = ax1;
        const double k1vz     = az1;

        const double vx2      = vx + 0.5 * dt * k1vx;
        const double vz2      = vz + 0.5 * dt * k1vz;
        const auto [ax2, az2] = accel(vx2, vz2);
        const double k2x      = vx2;
        const double k2z      = vz2;
        const double k2vx     = ax2;
        const double k2vz     = az2;

        const double vx3      = vx + 0.5 * dt * k2vx;
        const double vz3      = vz + 0.5 * dt * k2vz;
        const auto [ax3, az3] = accel(vx3, vz3);
        const double k3x      = vx3;
        const double k3z      = vz3;
        const double k3vx     = ax3;
        const double k3vz     = az3;

        const double vx4      = vx + dt * k3vx;
        const double vz4      = vz + dt * k3vz;
        const auto [ax4, az4] = accel(vx4, vz4);
        const double k4x      = vx4;
        const double k4z      = vz4;
        const double k4vx     = ax4;
        const double k4vz     = az4;

        x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        vx += (dt / 6.0) * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx);
        vz += (dt / 6.0) * (k1vz + 2.0 * k2vz + 2.0 * k3vz + k4vz);
        t += dt;

        if (x <= prev_x) {
            if (std::hypot(vx, vz) < 1e-3) {
                break;
            }
            continue;
        }

        while (x >= next_x && next_x <= max_range) {
            const double alpha    = std::clamp((next_x - prev_x) / (x - prev_x), 0.0, 1.0);
            const double z_interp = prev_z + alpha * (z - prev_z);
            const double t_interp = prev_t + alpha * dt;
            samples.emplace_back(next_x, z_interp, t_interp);
            next_x += range_step;
        }

        if (x >= max_range) {
            break;
        }
    }

    return samples;
}

} // namespace detail

} // namespace fcs::L4::trajectory
