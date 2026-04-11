#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <variant>

#include "scheduler/scheduler.hpp"

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <tbb/concurrent_queue.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fmt/core.h>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

#include <foxglove/channel.hpp>
#include <foxglove/context.hpp>
#include <foxglove/error.hpp>
#include <foxglove/expected.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/messages.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>
