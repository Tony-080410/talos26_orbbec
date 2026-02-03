#include "fcs/L4_planning/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L4 {

std::expected<void, std::string> L4Config::read_from(const toml::table& tbl) noexcept {
    READ(tbl, mpc_planner);
    return {};
}

} // namespace fcs::L4
