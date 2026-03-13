#pragma once

#include "frame.hpp"
#include <memory>

namespace fast_tf {

void init_infantry_system(CoordinateSystem& system, timestamp_ns_t ts = 0);
void init_hero_system(CoordinateSystem& system, timestamp_ns_t ts = 0);
void init_sentry_system(CoordinateSystem& system, timestamp_ns_t ts = 0);
void init_drone_system(CoordinateSystem& system, timestamp_ns_t ts = 0);

CoordinateSystem create_infantry_system(timestamp_ns_t ts = 0);
CoordinateSystem create_hero_system(timestamp_ns_t ts = 0);
CoordinateSystem create_sentry_system(timestamp_ns_t ts = 0);
CoordinateSystem create_drone_system(timestamp_ns_t ts = 0);

} // namespace fast_tf
