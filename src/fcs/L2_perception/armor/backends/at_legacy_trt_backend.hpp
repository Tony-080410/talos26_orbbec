#pragma once

#include "../at_legacy_config.hpp"
#include "at_legacy_detector_backend_base.hpp"
#include "core/armor_types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace cv {
class Mat;
}

namespace fcs::L2 {

class AtLegacyTrtBackend : public AtLegacyDetectorBackendBase<AtLegacyTrtBackend> {
public:
    static constexpr int INPUT_W     = 640;
    static constexpr int INPUT_H     = 640;
    static constexpr int NUM_COLORS  = 4;
    static constexpr int NUM_SIZES   = 2;
    static constexpr int NUM_CLASSES = 64;
    static constexpr int NUM_KPTS    = 4;
    static constexpr int OUTPUT_DIM  = 14;

    using DetectionResult = std::expected<std::vector<fcs::AtLegacyArmorDetection>, std::string>;
    using Config          = AtLegacyTrtConfig;

    struct PreprocContext {
        float scale_x;
        float scale_y;
        int pad_x;
        int pad_y;
        int orig_width;
        int orig_height;
    };

    /// Factory: construct a fully-initialized backend (load TensorRT engine).
    /// Construction IS initialization — no separate init() needed.
    [[nodiscard]] static std::expected<AtLegacyTrtBackend, std::string>
        create(Config config) noexcept;

    ~AtLegacyTrtBackend();

    AtLegacyTrtBackend(AtLegacyTrtBackend&&) noexcept;
    AtLegacyTrtBackend& operator=(AtLegacyTrtBackend&&) noexcept;
    AtLegacyTrtBackend(const AtLegacyTrtBackend&)            = delete;
    AtLegacyTrtBackend& operator=(const AtLegacyTrtBackend&) = delete;

    void set_detect_color(fcs::ArmorColor color) noexcept;

    DetectionResult detect_impl(const cv::Mat& image) noexcept;

private:
    /// Private constructor — use create() factory
    explicit AtLegacyTrtBackend(Config config) noexcept;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    Config config_;
    int detect_color_ = 0;
};

} // namespace fcs::L2
