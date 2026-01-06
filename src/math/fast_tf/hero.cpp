#include "math/fast_tf/hero.hpp"

namespace fast_tf {

template std::shared_ptr<CoordinateSystem<1024>> hero_new<1024>(timestamp_ns_t);

} // namespace fast_tf
