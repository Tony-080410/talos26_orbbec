#include "auto_aim/config.hpp"

#include "toml_helper.hpp"

namespace fcs {

std::expected<void, std::string> FoxgloveConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, enabled);
    READ(tbl, host);
    READ(tbl, port);
    return {};
}

std::expected<void, std::string> HardwareConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, camera);
    READ(tbl, untrust_roll);
    READ_OPT(tbl, imu_vendor_id);
    READ_OPT(tbl, imu_product_id);
    return {};
}

std::expected<void, std::string> VisionConfig::read_from(const toml::table& tbl) noexcept {
    READ_FLATTEN(tbl, at_legacy);
    READ(tbl, rune_detector);

    READ_FLATTEN(tbl, l3);
    READ(tbl, energy_meter);

    READ_FLATTEN(tbl, l4);
    READ(tbl, rune_planner);
    READ(tbl, plan_mux);

    READ_FLATTEN(tbl, l5);
    return {};
}

} // namespace fcs
