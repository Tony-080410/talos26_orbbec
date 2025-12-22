#include "fast_tf/frame_types.hpp"
#include "fast_tf/frame_builder.hpp"
#include <unordered_map>
#include <string>

namespace fast_tf {

// ============================================================================
// FrameRegistry 静态成员初始化
// ============================================================================

std::unordered_map<uint64_t, std::string> FrameRegistry::frame_map;
std::unordered_map<std::string, std::string> FrameRegistry::canonical_map;

// ============================================================================
// Frame buffer configuration static initialization
// ============================================================================
// Note: FrameBufferConfig::buffer_sizes is now defined as an inline static
// in frame_builder.hpp for header-only usage

// ============================================================================
// 编译期字符串哈希实现（确保在优化器不内联时可用）
// ============================================================================

// 显式实例化常用的CompileTimeString大小（仅实例化一次）
// template struct CompileTimeString<5>;    // "world"
// template struct CompileTimeString<9>;    // "base_link"
// template struct CompileTimeString<6>;    // "camera"

// ============================================================================
// CircularTransformBuffer 显式实例化
// ============================================================================
// Note: For header-only library, explicit instantiations may not be necessary
// Uncomment if needed for compilation optimization

// 常用的缓冲区大小
// template class CircularTransformBuffer<30>;   // 30条目
// template class CircularTransformBuffer<50>;   // 50条目
// template class CircularTransformBuffer<100>;  // 100条目
// template class CircularTransformBuffer<200>;  // 200条目
// template class CircularTransformBuffer<500>;  // 500条目

// ============================================================================
// 性能优化：预热SIMD
// ============================================================================

void init_simd_optimization() {
    // 这个函数可以在程序启动时调用以预热SIMD指令
    // 虽然Eigen会自动处理，但显式初始化可以改善首次使用的性能

    // 创建示例矩阵并执行基本操作
    Eigen::Matrix4d a = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d b = Eigen::Matrix4d::Identity();
    auto _ = a * b;  // SIMD操作预热
}

} // namespace fast_tf
