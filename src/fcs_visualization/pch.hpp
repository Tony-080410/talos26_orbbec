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

#include "scheduler/scheduler.hpp"

#include <spdlog/common.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include <tbb/concurrent_queue.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <fmt/core.h>
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

#include "foxglove_server.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "system_helpers.hpp"
#include "systems/base.hpp"
#include "tactical_palette.hpp"
