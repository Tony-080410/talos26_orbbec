#pragma once

// ============================================================================
// Precompiled Header for FCS Library
// ============================================================================
// This file contains commonly-used heavy headers that rarely change.
// Precompiling this reduces compilation time by 50-70%.

// C++ Standard Library
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// Third-party libraries (heavy dependencies)
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

// Conditionally include OpenVINO (only on platforms where it's available)
#if __has_include(<openvino/openvino.hpp>)
# include <openvino/openvino.hpp>
#endif

#include "magic_enum.hpp"

// Project-wide utilities
#include "toml_helper.hpp"
#include <toml++/toml.hpp>
