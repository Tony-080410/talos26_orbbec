#pragma once

#include "toml_helper.hpp"
#include <cstdint>

namespace toml::inline v3 {
class table;
}

namespace fcs::core::trajectory::model {

enum class ModelType {
    Ideal,
    LinearDrag,
    QuadraticDrag,
};

struct ModelConfig {
    ModelType type{ModelType::LinearDrag};

    /// Number of iterations for pitch solving / compensation loop.
    uint32_t iteration_times{20};

    /// Bullet velocity at muzzle (m/s).
    double bullet_speed{25.0};

    /// Gravity acceleration (m/s^2)
    double gravity{9.8};

    /// Linearized air resistance coefficient (used by ResistanceModel only).
    double resistance{0.001};

    /// Drag coefficient Cd (dimensionless), used by RK4Model only.
    double cd{0.47};

    /// Air density rho (kg/m^3), used by RK4Model only.
    double rho{1.225};

    /// Bullet diameter (meters), used by RK4Model only.
    double diameter{0.017};

    /// Bullet mass (kg), used by RK4Model only.
    double mass{0.0032};

    /// Desired RK4 integration dt (seconds).
    double rk4_dt{0.001};

    /// Minimum RK4 steps per integration.
    int rk4_min_steps{50};

    /// Maximum RK4 steps per integration.
    int rk4_max_steps{2000};

    // ========================================================================
    // Lookup Table Configuration (for RK4 with table optimization)
    // ========================================================================

    /// Maximum range in lookup table (meters).
    double table_range_max{20.0};

    /// Range step size in lookup table (meters).
    double table_range_step{0.01};

    /// Minimum pitch in lookup table (radians).
    double table_pitch_min{-0.6};

    /// Maximum pitch in lookup table (radians).
    double table_pitch_max{0.9};

    /// Pitch step size in lookup table (radians).
    double table_pitch_step{0.005};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, type);
        READ(tbl, iteration_times);
        READ(tbl, bullet_speed);
        READ(tbl, gravity);
        READ(tbl, resistance);
        READ(tbl, cd);
        READ(tbl, rho);
        READ(tbl, diameter);
        READ(tbl, mass);
        READ(tbl, rk4_dt);
        READ(tbl, rk4_min_steps);
        READ(tbl, rk4_max_steps);
        READ(tbl, table_range_max);
        READ(tbl, table_range_step);
        READ(tbl, table_pitch_min);
        READ(tbl, table_pitch_max);
        READ(tbl, table_pitch_step);
        return {};
    }
};

} // namespace fcs::core::trajectory::model
