#pragma once

#include <expected>
#include <string>

// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L4 {

// ============================================================================
// Trajectory Compensator Types (shared between planners)
// ============================================================================

enum class TrajectoryCompensatorType {
    Ideal,
    Resistance,
    RK4,
};

// ============================================================================
// Trajectory Configuration (shared between planners)
// ============================================================================

struct TrajectoryConfig {
    /// Number of iterations for pitch solving / compensation loop.
    int iteration_times{20};

    /// Bullet velocity at muzzle (m/s).
    double bullet_speed{25.0};

    /// Gravity acceleration (m/s^2)
    double gravity{9.8};

    /// Linearized air resistance coefficient (used by ResistanceCompensator only).
    double resistance{0.001};

    /// Drag coefficient Cd (dimensionless), used by RK4Compensator only.
    double cd{0.47};

    /// Air density rho (kg/m^3), used by RK4Compensator only.
    double rho{1.225};

    /// Bullet diameter (meters), used by RK4Compensator only.
    double diameter{0.017};

    /// Bullet mass (kg), used by RK4Compensator only.
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

    /// Enable lookup table acceleration for RK4 model.
    /// When true, precomputes (range, pitch) -> (z, tof) table for fast pitch solving.
    bool enable_lookup_table{true};

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

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
