#include "fcs/L5_weapon/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L5 {

std::expected<void, std::string>
    WeaponControllerConfig::read_from(const toml::table& /*tbl*/) noexcept {
    return {};
}

std::expected<void, std::string> L5Config::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, weapon);
    return {};
}

} // namespace fcs::L5
