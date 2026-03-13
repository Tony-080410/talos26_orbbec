#pragma once

#include <expected>
#include <string>

#include "core/trajectory/model/config.hpp"
#include "toml_helper.hpp"

namespace toml::inline v3 {
class table;
}
namespace fcs::core::trajectory {

struct TrajectoryConfig {
    model::ModelConfig model{};

    /// Default bullet velocity at muzzle (m/s).
    double default_bullet_speed{25.0};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_FLATTEN(tbl, model);
        READ(tbl, default_bullet_speed);
        return {};
    }
};

} // namespace fcs::core::trajectory
