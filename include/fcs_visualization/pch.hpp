#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <thread>
#include <variant>

#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>
