#pragma once

#include "frame.hpp"
#include <memory>

namespace fast_tf {

std::shared_ptr<CoordinateSystem> create_infantry_system(timestamp_ns_t ts = 0);
std::shared_ptr<CoordinateSystem> create_hero_system(timestamp_ns_t ts = 0);
std::shared_ptr<CoordinateSystem> create_sentry_system(timestamp_ns_t ts = 0);
std::shared_ptr<CoordinateSystem> create_drone_system(timestamp_ns_t ts = 0);

} // namespace fast_tf
