#include "math/fast_tf/infantry.hpp"

namespace fast_tf {

template std::shared_ptr<CoordinateSystem<1024>> infantry_new<1024>(timestamp_ns_t);

} // namespace fast_tf
