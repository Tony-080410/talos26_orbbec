#pragma once

#include <expected>
#include <string>

#include "core/trajectory/model/config.hpp"

namespace toml::inline v3 {
class table;
}
namespace fcs::core::trajectory {

struct TrajectoryConfig {
    model::ModelConfig model{};
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::core::trajectory
