#include "L5_weapon/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L5 {

std::expected<void, std::string> MpcAxisConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, q_pos);
    READ(tbl, q_vel);
    READ(tbl, r);
    READ(tbl, max_acc);

    if (q_pos < 0.0 || q_vel < 0.0 || r <= 0.0 || max_acc <= 0.0) {
        return std::unexpected(
            "MpcAxisConfig validation failed: q_pos/q_vel must be non-negative and r/max_acc "
            "must be positive");
    }
    return {};
}

std::expected<void, std::string> MpcSolverConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, horizon_ahead);
    READ(tbl, horizon_back);
    READ(tbl, dt);
    READ(tbl, max_iterations);
    READ(tbl, abs_tol);
    READ(tbl, rho);
    READ(tbl, yaw);
    READ(tbl, pitch);

    if (horizon_ahead <= 0 || horizon_back <= 0 || dt <= 0.0 || max_iterations <= 0
        || abs_tol <= 0.0 || rho <= 0.0) {
        return std::unexpected(
            "MpcSolverConfig validation failed: horizon/dt/max_iterations/abs_tol/rho must be "
            "positive");
    }

    return {};
}

std::expected<void, std::string>
    WeaponControllerConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, enabled);
    READ(tbl, enable_debug);
    READ(tbl, pitch_min);
    READ(tbl, pitch_max);
    READ(tbl, mpc);

    if (pitch_min >= pitch_max) {
        return std::unexpected(
            "WeaponControllerConfig validation failed: pitch_min must be less than pitch_max");
    }

    return {};
}

std::expected<void, std::string> L5Config::read_from(const toml::table& tbl) noexcept {
    if (const toml::node* legacy_weapon = tbl.get("weapon")) {
        const auto* table = legacy_weapon->as_table();
        if (!table) {
            return std::unexpected("Invalid value for key 'weapon': expected table");
        }
        if (!table->empty()) {
            return std::unexpected(
                "Legacy [weapon] table is no longer supported; move settings under [mpc_weapon]");
        }
        (void)const_cast<toml::table&>(tbl).erase("weapon");
    }

    auto res = toml_helper::take<WeaponControllerConfig>(tbl, "mpc_weapon");
    if (!res) {
        return std::unexpected(res.error());
    }
    weapon = std::move(*res);
    return {};
}

} // namespace fcs::L5
