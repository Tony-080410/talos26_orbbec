#include "config.hpp"

#include "toml_helper.hpp"
#include "toml_helper_containers.hpp"
#include "toml_helper_eigen.hpp"

namespace fcs {

std::expected<void, std::string> RobotExtrinsicConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, gimbal_yaw);
    return {};
}

std::expected<void, std::string>
    RobotExtrinsicConfig::gimbal_yaw_t::read_from(const toml::table& tbl) noexcept {
    READ(tbl, gimbal_pitch);
    return {};
}

std::expected<void, std::string>
    RobotExtrinsicConfig::gimbal_yaw_t::gimbal_pitch_t::read_from(const toml::table& tbl) noexcept {
    READ(tbl, translation);
    READ(tbl, camera_link);
    READ(tbl, muzzle_link);
    return {};
}

std::expected<void, std::string>
    RobotExtrinsicConfig::gimbal_yaw_t::gimbal_pitch_t::camera_link_t::read_from(
        const toml::table& tbl) noexcept {
    READ(tbl, translation);
    READ(tbl, roll);
    READ(tbl, pitch);
    READ(tbl, yaw);
    return {};
}

std::expected<void, std::string>
    RobotExtrinsicConfig::gimbal_yaw_t::gimbal_pitch_t::muzzle_link_t::read_from(
        const toml::table& tbl) noexcept {
    READ(tbl, translation);
    return {};
}

std::expected<void, std::string> FoxgloveConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, enabled);
    READ(tbl, host);
    READ(tbl, port);
    return {};
}

std::expected<void, std::string> McuConfig::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, mcu_vendor_id);
    READ_OPT(tbl, mcu_product_id);
    READ(tbl, untrust_roll);
    READ(tbl, dynamic_self_color);
    READ(tbl, dynamic_bullet_speed);
    READ(tbl, bullet_speed_default);
    READ(tbl, bullet_speed_min);
    READ(tbl, bullet_speed_max);
    return {};
}

std::expected<void, std::string> HardwareConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, camera);
    READ(tbl, mcu);
    READ(tbl, extrinsic);
    return {};
}

std::expected<void, std::string> VisionConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, trajectory);
    READ_FLATTEN(tbl, at_legacy);
    READ(tbl, rune_detector);

    READ_FLATTEN(tbl, l3);
    READ(tbl, energy_meter);

    READ_FLATTEN(tbl, l4);

    READ_FLATTEN(tbl, l5);
    return {};
}

} // namespace fcs
