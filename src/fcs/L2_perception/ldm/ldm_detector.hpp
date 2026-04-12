#pragma once

#include "core/armor_types.hpp"
#include "ldm_config.hpp"
#include "types.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <expected>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace fcs::L2::ldm {

class LdmDetector {
public:
    explicit LdmDetector(LdmDetectorConfig config) noexcept;
    ~LdmDetector() = default;

    void set_target_color(ArmorColor color) noexcept;
    ArmorColor get_target_color() const noexcept;

    std::expected<std::optional<LdmDetection>, DetectorError>
        detect(const cv::Mat& image) const noexcept;

private:
    LdmDetectorConfig config_;
    ArmorColor target_color_;
};

} // namespace fcs::L2::ldm
