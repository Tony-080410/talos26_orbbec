#pragma once

#include <expected>

// Forward declaration to avoid including toml++ in config headers.

namespace toml::inline v3 {
class table;
}

namespace fcs::L5 {

struct MpcAxisConfig {
    double q_pos{9e6};
    double q_vel{0.0};
    double r{1.0};
    double max_acc{50.0};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

struct MpcSolverConfig {
    int horizon_ahead{50};
    int horizon_back{50};
    double dt{0.01};
    int max_iterations{10};
    double abs_tol{1e-3};
    double rho{1.0};
    MpcAxisConfig yaw{9e6, 0.0, 1.0, 50.0};
    MpcAxisConfig pitch{9e6, 0.0, 1.0, 100.0};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// ============================================================================
// Enhanced Weapon Controller Configuration (L5 Layer)
// ============================================================================

struct WeaponControllerConfig {
    bool enabled{true};
    bool enable_debug{true};

    double pitch_min{-0.69813169};
    double pitch_max{0.69813169};

    MpcSolverConfig mpc{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// ============================================================================
// L5 Weapon Full Config
// ============================================================================

struct L5Config {
    WeaponControllerConfig weapon{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L5
