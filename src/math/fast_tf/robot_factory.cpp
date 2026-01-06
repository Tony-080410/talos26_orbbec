#include "math/fast_tf/robot_factory.hpp"

#include "math/fast_tf/hero.hpp"
#include "math/fast_tf/infantry.hpp"
#include "math/fast_tf/sentry.hpp"

namespace fast_tf {

std::shared_ptr<CoordinateSystem<1024>> create_infantry_system(timestamp_ns_t ts) {
    return infantry_new<1024>(ts);
}

std::shared_ptr<CoordinateSystem<1024>> create_hero_system(timestamp_ns_t ts) {
    return hero_new<1024>(ts);
}

std::shared_ptr<CoordinateSystem<1024>> create_sentry_system(timestamp_ns_t ts) {
    return sentry_new<1024>(ts);
}

} // namespace fast_tf
