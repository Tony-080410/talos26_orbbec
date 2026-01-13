#pragma once

#include "trajectory_compensator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace fcs::L4 {

/// Ideal trajectory compensator (no air resistance)
class IdealCompensator final : public TrajectoryCompensator {
public:
    [[nodiscard]] double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept override {
        const double distance = std::hypot(target_position.x(), target_position.y());
        const double angle    = std::atan2(target_position.z(), distance);
        const double t        = distance / (velocity * std::cos(angle));
        return t;
    }

protected:
    [[nodiscard]] double calculate_trajectory(double x, double pitch) const noexcept override {
        const double t = x / (velocity * std::cos(pitch));
        const double y = velocity * std::sin(pitch) * t - 0.5 * gravity * t * t;
        return y;
    }
};

/// Resistance trajectory compensator (with air resistance)
class ResistanceCompensator final : public TrajectoryCompensator {
public:
    [[nodiscard]] double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept override {
        const double r        = resistance < 1e-4 ? 1e-4 : resistance;
        const double distance = std::hypot(target_position.x(), target_position.y());
        const double angle    = std::atan2(target_position.z(), distance);
        const double t        = (std::exp(r * distance) - 1.0) / (r * velocity * std::cos(angle));
        return t;
    }

protected:
    [[nodiscard]] double calculate_trajectory(double x, double pitch) const noexcept override {
        const double r = resistance < 1e-4 ? 1e-4 : resistance;
        const double t = (std::exp(r * x) - 1.0) / (r * velocity * std::cos(pitch));
        const double y = velocity * std::sin(pitch) * t - 0.5 * gravity * t * t;
        return y;
    }
};

/// RK4 trajectory compensator (quadratic drag, 3D point-mass model).
///
/// Performance notes:
/// - Uses a precomputed lookup table (range,pitch)->(z,tof) built when params update.
/// - Runtime solve uses table inversion (range,z)->pitch + optional 1~2 RK4 refinements.
class RK4Compensator final : public TrajectoryCompensator {
public:
    void on_params_updated() noexcept override { rebuild_table(); }

    [[nodiscard]] std::vector<std::pair<double, double>>
        get_trajectory(double max_distance, double pitch) const noexcept override {
        if (max_distance <= 0.0) {
            return {};
        }

        constexpr double sample_step = 0.015;
        const size_t num_points      = static_cast<size_t>(max_distance / sample_step) + 1;
        std::vector<std::pair<double, double>> trajectory;
        trajectory.reserve(num_points);

        if (const auto table = table_) {
            for (double x = 0; x < max_distance; x += sample_step) {
                const double z = sample_table_z(*table, x, pitch);
                trajectory.emplace_back(x, std::isfinite(z) ? z : 0.0);
            }
            return trajectory;
        }

        for (double x = 0; x < max_distance; x += sample_step) {
            trajectory.emplace_back(x, calculate_trajectory(x, pitch));
        }
        return trajectory;
    }

    [[nodiscard]] double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept override {
        const double range = std::hypot(target_position.x(), target_position.y());
        if (range < 1e-6) {
            return 0.0;
        }

        if (const auto table = table_) {
            double pitch_est = 0.0;
            double tof_est   = 0.0;
            if (lookup_pitch_tof(*table, range, target_position.z(), pitch_est, tof_est)) {
                if (std::isfinite(tof_est) && tof_est >= 0.0) {
                    return tof_est;
                }
            }
        }

        const double pitch = std::atan2(target_position.z(), range);
        const auto impact  = simulate_to_range_2d(range, pitch);
        return impact ? impact->time_s : 0.0;
    }

    [[nodiscard]] bool compensate(
        const Eigen::Vector3d& target_position, double& yaw,
        double& pitch) const noexcept override {
        const double target_height = target_position.z();
        const double range         = std::hypot(target_position.x(), target_position.y());
        if (range < 1e-6) {
            return false;
        }

        yaw                           = std::atan2(target_position.y(), target_position.x());
        constexpr double kDhThreshold = 0.01;
        constexpr double kMaxAngle    = std::numbers::pi / 2.5;

        double pitch_est = std::atan2(target_height, range);
        double tof_est   = 0.0;

        if (const auto table = table_) {
            (void)lookup_pitch_tof(*table, range, target_height, pitch_est, tof_est);

            bool have_impact = false;
            double last_dh   = std::numeric_limits<double>::infinity();

            for (int refine = 0; refine < 2; ++refine) {
                const auto impact = simulate_to_range_2d(range, pitch_est);
                if (!impact) {
                    break;
                }
                have_impact     = true;
                const double dh = target_height - impact->z;
                last_dh         = dh;
                if (std::abs(dh) < kDhThreshold) {
                    break;
                }
                double dzdp = 0.0;
                if (!estimate_dzdp(*table, range, pitch_est, dzdp) || std::abs(dzdp) < 1e-6) {
                    break;
                }
                pitch_est = std::clamp(pitch_est + dh / dzdp, table->pitch_min, table->pitch_max);
            }

            if (!have_impact) {
                return false;
            }

            if (std::abs(last_dh) > kDhThreshold) {
                double iterative_height = target_height;
                double dh               = last_dh;

                for (int i = 0; i < 6; ++i) {
                    pitch_est = std::atan2(iterative_height, range);
                    if (std::abs(pitch_est) > kMaxAngle) {
                        break;
                    }

                    const auto impact = simulate_to_range_2d(range, pitch_est);
                    if (!impact) {
                        break;
                    }

                    dh = target_height - impact->z;
                    if (std::abs(dh) < kDhThreshold) {
                        break;
                    }
                    iterative_height += dh;
                }

                if (std::abs(dh) > kDhThreshold) {
                    return false;
                }
            }
        } else {
            // Fallback: limit iterations (avoid 20x RK4 in real-time loop).
            double iterative_height = target_height;
            double dh               = 0.0;

            for (int i = 0; i < std::min(iteration_times, 6); ++i) {
                pitch_est = std::atan2(iterative_height, range);
                if (std::abs(pitch_est) > kMaxAngle) {
                    break;
                }

                const auto impact = simulate_to_range_2d(range, pitch_est);
                if (!impact) {
                    break;
                }

                dh = target_height - impact->z;
                if (std::abs(dh) < kDhThreshold) {
                    break;
                }
                iterative_height += dh;
            }

            if (std::abs(dh) > kDhThreshold) {
                return false;
            }
        }

        if (std::abs(pitch_est) > kMaxAngle) {
            return false;
        }
        pitch = pitch_est;
        return true;
    }

protected:
    [[nodiscard]] double calculate_trajectory(double x, double pitch) const noexcept override {
        if (x < 0.0) {
            return 0.0;
        }
        if (const auto table = table_) {
            const double z = sample_table_z(*table, x, pitch);
            return std::isfinite(z) ? z : 0.0;
        }
        const auto impact = simulate_to_range_2d(x, pitch);
        return impact ? impact->z : 0.0;
    }

private:
    struct Impact2D {
        double time_s{0.0};
        double x{0.0};
        double z{0.0};
        double vx{0.0};
        double vz{0.0};
    };

    struct LookupTable {
        double range_step{0.05};
        double range_max{20.0};
        double pitch_min{-0.4};
        double pitch_max{0.8};
        double pitch_step{0.01};
        size_t range_count{0};
        size_t pitch_count{0};
        std::vector<double> z;   // [pitch][range]
        std::vector<double> tof; // [pitch][range]
    };

    std::shared_ptr<const LookupTable> table_;

    [[nodiscard]] double drag_k() const noexcept {
        const double d = std::max(1e-6, diameter);
        const double A = std::numbers::pi * (d * 0.5) * (d * 0.5);
        const double m = std::max(1e-6, mass);
        return 0.5 * cd * rho * A / m;
    }

    [[nodiscard]] double select_dt(double range, double pitch) const noexcept {
        const double v0    = std::max(1e-3, velocity);
        const double c     = std::max(0.05, std::abs(std::cos(pitch)));
        const double t_est = range / (v0 * c);

        const double desired_dt = std::clamp(rk4_dt, 1e-5, 0.02);
        const int min_steps     = std::max(1, rk4_min_steps);
        const int max_steps     = std::max(min_steps, rk4_max_steps);
        const int steps =
            std::clamp(static_cast<int>(std::ceil(t_est / desired_dt)), min_steps, max_steps);
        return t_est / static_cast<double>(steps);
    }

    [[nodiscard]] std::optional<Impact2D>
        simulate_to_range_2d(double range, double pitch) const noexcept {
        if (range < 1e-6) {
            return Impact2D{};
        }

        const double k      = drag_k();
        const double dt     = select_dt(range, pitch);
        const int min_steps = std::max(1, rk4_min_steps);
        const int max_steps = std::max(min_steps, rk4_max_steps);

        double x  = 0.0;
        double z  = 0.0;
        double vx = velocity * std::cos(pitch);
        double vz = velocity * std::sin(pitch);
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

                Impact2D out;
                out.time_s = prev_t + alpha * dt;
                out.x      = range;
                out.z      = prev_z + alpha * (z - prev_z);
                out.vx     = prev_vx + alpha * (vx - prev_vx);
                out.vz     = prev_vz + alpha * (vz - prev_vz);
                return out;
            }

            if (step + 1 >= min_steps && std::hypot(vx, vz) < 1e-3) {
                break;
            }
        }

        return std::nullopt;
    }

    void rebuild_table() noexcept {
        constexpr double kTableRangeMax  = 20.0;
        constexpr double kTableRangeStep = 0.01;
        constexpr double kPitchMin       = -0.6;
        constexpr double kPitchMax       = 0.9;
        constexpr double kPitchStep      = 0.005;

        auto tbl         = std::make_shared<LookupTable>();
        tbl->range_max   = kTableRangeMax;
        tbl->range_step  = kTableRangeStep;
        tbl->pitch_min   = kPitchMin;
        tbl->pitch_max   = kPitchMax;
        tbl->pitch_step  = kPitchStep;
        tbl->range_count = static_cast<size_t>(std::floor(tbl->range_max / tbl->range_step)) + 1;
        tbl->pitch_count =
            static_cast<size_t>(std::floor((tbl->pitch_max - tbl->pitch_min) / tbl->pitch_step))
            + 1;

        const size_t total = tbl->pitch_count * tbl->range_count;
        tbl->z.assign(total, std::numeric_limits<double>::quiet_NaN());
        tbl->tof.assign(total, std::numeric_limits<double>::quiet_NaN());

        const double k = drag_k();

        auto accel = [&](double _vx, double _vz) noexcept {
            const double speed = std::hypot(_vx, _vz);
            const double ax    = speed > 1e-8 ? (-k * speed * _vx) : 0.0;
            const double az    = -gravity + (speed > 1e-8 ? (-k * speed * _vz) : 0.0);
            return std::pair<double, double>{ax, az};
        };

        const auto idx = [&](size_t pi, size_t ri) { return pi * tbl->range_count + ri; };

        for (size_t pi = 0; pi < tbl->pitch_count; ++pi) {
            const double pitch  = tbl->pitch_min + static_cast<double>(pi) * tbl->pitch_step;
            const double dt     = select_dt(tbl->range_max, pitch);
            const int max_steps = std::max(1, rk4_max_steps);

            double x  = 0.0;
            double z  = 0.0;
            double vx = velocity * std::cos(pitch);
            double vz = velocity * std::sin(pitch);
            double t  = 0.0;

            tbl->z[idx(pi, 0)]   = 0.0;
            tbl->tof[idx(pi, 0)] = 0.0;

            size_t ri     = 1;
            double next_x = tbl->range_step;

            double prev_x = x, prev_z = z, prev_vx = vx, prev_vz = vz, prev_t = t;

            for (int step = 0; step < max_steps && ri < tbl->range_count; ++step) {
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

                if (x <= prev_x) {
                    if (std::hypot(vx, vz) < 1e-3) {
                        break;
                    }
                    continue;
                }

                while (ri < tbl->range_count && x >= next_x) {
                    const double alpha    = std::clamp((next_x - prev_x) / (x - prev_x), 0.0, 1.0);
                    tbl->z[idx(pi, ri)]   = prev_z + alpha * (z - prev_z);
                    tbl->tof[idx(pi, ri)] = prev_t + alpha * dt;
                    ++ri;
                    next_x += tbl->range_step;
                }
            }
        }

        table_ = std::move(tbl);
    }

    [[nodiscard]] static double
        sample_table_z(const LookupTable& table, double x, double pitch) noexcept {
        if (x <= 0.0) {
            return 0.0;
        }
        if (x >= table.range_max) {
            x = table.range_max;
        }
        pitch = std::clamp(pitch, table.pitch_min, table.pitch_max);

        const double r_pos = x / table.range_step;
        const size_t r0    = std::min(static_cast<size_t>(r_pos), table.range_count - 2);
        const double ar    = r_pos - static_cast<double>(r0);

        const double p_pos = (pitch - table.pitch_min) / table.pitch_step;
        const size_t p0    = std::min(static_cast<size_t>(p_pos), table.pitch_count - 2);
        const double ap    = p_pos - static_cast<double>(p0);

        const auto idx = [&](size_t pi, size_t ri) { return pi * table.range_count + ri; };

        const double z00 = table.z[idx(p0, r0)];
        const double z01 = table.z[idx(p0, r0 + 1)];
        const double z10 = table.z[idx(p0 + 1, r0)];
        const double z11 = table.z[idx(p0 + 1, r0 + 1)];

        if (!std::isfinite(z00) || !std::isfinite(z01) || !std::isfinite(z10)
            || !std::isfinite(z11)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double z0 = z00 + ar * (z01 - z00);
        const double z1 = z10 + ar * (z11 - z10);
        return z0 + ap * (z1 - z0);
    }

    [[nodiscard]] static bool lookup_pitch_tof(
        const LookupTable& table, double range, double target_z, double& pitch_out,
        double& tof_out) noexcept {

        if (range <= 0.0 || range > table.range_max) {
            return false;
        }

        const double r_pos = range / table.range_step;
        const size_t r0    = std::min(static_cast<size_t>(r_pos), table.range_count - 2);
        const double ar    = r_pos - static_cast<double>(r0);

        const auto idx = [&](size_t pi, size_t ri) { return pi * table.range_count + ri; };

        auto z_at = [&](size_t pi) -> double {
            const double z0 = table.z[idx(pi, r0)];
            const double z1 = table.z[idx(pi, r0 + 1)];
            if (!std::isfinite(z0) || !std::isfinite(z1)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return z0 + ar * (z1 - z0);
        };

        auto t_at = [&](size_t pi) -> double {
            const double t0 = table.tof[idx(pi, r0)];
            const double t1 = table.tof[idx(pi, r0 + 1)];
            if (!std::isfinite(t0) || !std::isfinite(t1)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return t0 + ar * (t1 - t0);
        };

        double z_prev = z_at(0);
        if (!std::isfinite(z_prev)) {
            return false;
        }
        if (target_z < z_prev) {
            return false;
        }

        for (size_t pi = 1; pi < table.pitch_count; ++pi) {
            const double z_cur = z_at(pi);
            if (!std::isfinite(z_cur)) {
                continue;
            }
            if (target_z <= z_cur) {
                const double p0 = table.pitch_min + static_cast<double>(pi - 1) * table.pitch_step;
                const double p1 = table.pitch_min + static_cast<double>(pi) * table.pitch_step;
                const double denom = (z_cur - z_prev);
                const double ap =
                    denom > 1e-9 ? std::clamp((target_z - z_prev) / denom, 0.0, 1.0) : 0.0;

                pitch_out = p0 + ap * (p1 - p0);

                const double t0 = t_at(pi - 1);
                const double t1 = t_at(pi);
                if (std::isfinite(t0) && std::isfinite(t1)) {
                    tof_out = t0 + ap * (t1 - t0);
                } else {
                    tof_out = std::numeric_limits<double>::quiet_NaN();
                }
                return true;
            }
            z_prev = z_cur;
        }

        return false;
    }

    [[nodiscard]] static bool estimate_dzdp(
        const LookupTable& table, double range, double pitch, double& dzdp_out) noexcept {
        pitch              = std::clamp(pitch, table.pitch_min, table.pitch_max);
        const double p_pos = (pitch - table.pitch_min) / table.pitch_step;
        size_t pi          = static_cast<size_t>(std::round(p_pos));
        if (pi < 1) {
            pi = 1;
        }
        if (pi + 1 >= table.pitch_count) {
            pi = table.pitch_count - 2;
        }

        const double p_lo  = table.pitch_min + static_cast<double>(pi - 1) * table.pitch_step;
        const double p_hi  = table.pitch_min + static_cast<double>(pi + 1) * table.pitch_step;
        const double z_lo  = sample_table_z(table, range, p_lo);
        const double z_hi  = sample_table_z(table, range, p_hi);
        const double denom = (p_hi - p_lo);
        if (!std::isfinite(z_lo) || !std::isfinite(z_hi) || std::abs(denom) < 1e-12) {
            return false;
        }
        dzdp_out = (z_hi - z_lo) / denom;
        return true;
    }
};

} // namespace fcs::L4
