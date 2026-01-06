#include "fcs/camera_config.hpp"
#include "toml_helper.hpp"
#include <toml++/toml.h>

namespace fcs {

std::expected<void, std::string> CameraProfileConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, trigger_mode);
    READ(tbl, invert_image);
    READ(tbl, exposure_time_us);
    READ(tbl, gain);
    return {};
}

std::expected<void, std::string> CameraConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, camera_matrix);
    READ(tbl, distort_coefficient);
    READ(tbl, width);
    READ(tbl, height);
    READ_OPT(tbl, profile);
    return {};
}

} // namespace fcs
