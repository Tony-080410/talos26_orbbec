#include "fast_tf/mod.hpp"
#include <Eigen/Dense>

namespace fast_tf {

// ============================================================================
// SIMD Optimization Warm-up
// ============================================================================

void init_simd_optimization() {
    // Warm up SIMD instructions for improved first-use performance
    Eigen::Matrix4d a       = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d b       = Eigen::Matrix4d::Identity();
    [[maybe_unused]] auto _ = a * b;
}

} // namespace fast_tf
