#pragma once

#include <Eigen/Eigen>
#include <array>
#include <cstdint>
#include <expected>
#include <optional>

#include "fcs/L2_perception/detector.hpp"
#include "spdlog_hook.hpp"
#include "toml_helper.hpp"

#include <toml++/toml.hpp>

namespace fcs {

/**
 * @brief 相机内参 (统一格式)
 */
struct CameraInfo {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix;
    Eigen::Matrix<double, 1, 5> distort_coefficient;
    uint32_t width{1440};
    uint32_t height{1080};
};

/**
 * @brief PnP解算器配置
 */
struct AtVisionSolverConfig {
    // 相机内参
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix; // 3x3 row-major
    Eigen::Matrix<double, 1, 5> distort_coefficient;            // 1x5

    // 外参：相机到云台、云台到IMU
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> extrinsic_camera_gimbal; // 3x3 row-major
    Eigen::Matrix<double, 3, 1> extrinsic_camera_gimbal_tf;               // 3x1
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> extrinsic_gimbal_imu;    // 3x3 row-major

    // 统一格式的相机内参 (从上述矩阵填充)
    fcs::CameraInfo camera_info{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["solver"].as_table()) {
            auto t = *_t;
            READ_MATRIX(c, t, camera_matrix);
            READ_MATRIX(d, t, distort_coefficient);
            READ_MATRIX(r, t, extrinsic_camera_gimbal);
            READ_MATRIX(tt, t, extrinsic_camera_gimbal_tf);
            READ_MATRIX(rr, t, extrinsic_gimbal_imu);

            // 填充 camera_info
            camera_info.camera_matrix       = camera_matrix;
            camera_info.distort_coefficient = distort_coefficient;

            return {};
        }
        return std::unexpected("solver section not found");
    }
};

} // namespace fcs
