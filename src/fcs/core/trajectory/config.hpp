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
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_FLATTEN(tbl, model);
        return {};
    }
};

} // namespace fcs::core::trajectory
