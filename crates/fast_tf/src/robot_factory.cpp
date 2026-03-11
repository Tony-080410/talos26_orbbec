#include "robot_factory.hpp"

#include "drone.hpp"
#include "hero.hpp"
#include "infantry.hpp"
#include "sentry.hpp"

namespace fast_tf {

std::shared_ptr<CoordinateSystem> create_infantry_system(timestamp_ns_t ts) {
    return infantry_new(ts);
}

std::shared_ptr<CoordinateSystem> create_hero_system(timestamp_ns_t ts) { return hero_new(ts); }

std::shared_ptr<CoordinateSystem> create_sentry_system(timestamp_ns_t ts) { return sentry_new(ts); }

std::shared_ptr<CoordinateSystem> create_drone_system(timestamp_ns_t ts) { return drone_new(ts); }

} // namespace fast_tf
