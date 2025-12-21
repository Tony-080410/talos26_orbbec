#pragma once

// ============================================================================
// Fast TF - 编译期变换树系统
// 一个高性能的ROS2 tf2替代方案，具有完整的编译期验证
// ============================================================================

// 核心依赖
#include <expected>  // C++23 standard library
#include "compile_time_string.hpp"
#include "concepts.hpp"
#include "frame_types.hpp"
#include "transform_buffer.hpp"
#include "tree_structure.hpp"
#include "transform_tree.hpp"
#include "frame_id.hpp"
#include "frame_builder.hpp"
#include "frame_hierarchy.hpp"
#include "static_transforms.hpp"

namespace fast_tf {

// ============================================================================
// 公共API导出
// ============================================================================

// 编译期字符串工具已包含

// 框架类型和注册已包含

// 变换矩阵
using Matrix4d = TransformMatrix4d;

// 查询选项
using QueryOptions = TransformQueryOptions;

// 查询结果
using QueryResult = TransformQueryResult;

// ============================================================================
// Modern Frame Definition
// ============================================================================

/**
 * Modern frame definition using DECLARE_FRAME macro.
 * Provides compile-time type safety and hierarchy validation.
 *
 * Usage:
 *   DECLARE_FRAME(world);              // Root frame
 *   DECLARE_FRAME(base_link, world);   // Child frame
 *   DECLARE_FRAME(camera, base_link);  // Grandchild frame
 *
 * This creates types: world_Frame, base_link_Frame, camera_Frame
 * These can be used in template queries like:
 *   tree.lookupTransform<camera_Frame, world_Frame>(timestamp)
 */

// ============================================================================
// Type-Safe Template Queries
// ============================================================================

/**
 * Modern query API (RECOMMENDED):
 *   auto result = tree.lookupTransform<FromFrame, ToFrame>(timestamp);
 *   if (result.has_value()) {
 *       TransformMatrix4d tf = result.value();
 *   }
 *
 * Modern set API:
 *   tree.setTransform<FromFrame, ToFrame>(timestamp, transform_data);
 *
 * This replaces string-based APIs with compile-time type safety.
 */

// ============================================================================
// Static Transforms
// ============================================================================

/**
 * Register static transforms for fixed calibrations:
 *   StaticTransformStore::register_transform<camera_Frame, base_link_Frame>(offset);
 *
 * Static transforms:
 * - Reduce memory footprint
 * - Are composed with dynamic transforms: static × dynamic = result
 * - Don't require timestamp lookups
 * - Perfect for sensor calibrations and fixed offsets
 */

// ============================================================================
// 性能优化提示
// ============================================================================

/**
 * 性能使用建议：
 *
 * 1. 编译期验证：
 *    - 所有树结构验证在编译期完成，零运行时开销
 *    - 坐标系路径在编译期验证
 *    - 类型不匹配会导致编译错误
 *
 * 2. 缓冲区管理：
 *    - 使用固定大小的循环缓冲区（std::array）
 *    - 避免动态内存分配
 *    - 使用TBB自旋读写互斥体实现无锁读操作
 *
 * 3. 查询优化：
 *    - 使用 parallel_lookup_transforms 进行批量查询
 *    - 启用插值获得平滑的变换
 *    - 缓存常用的变换对
 *
 * 4. SIMD优化：
 *    - Eigen会自动使用SIMD指令
 *    - 矩阵乘法和四元数操作已优化
 *    - 在支持AVX2的系统上自动启用向量化
 *
 * 5. 内存布局：
 *    - 所有缓冲区都在堆栈上分配（对于小树）
 *    - 或使用std::unique_ptr进行堆分配（大树）
 *    - 避免false sharing通过对齐
 */

// ============================================================================
// 错误处理
// ============================================================================

/**
 * 常见编译期错误及其含义：
 *
 * "Invalid tree structure"
 *   -> 树的拓扑结构有问题（可能存在循环或不连接的节点）
 *
 * "Duplicate frame names"
 *   -> 多个节点使用了相同的名称
 *
 * "Concept constraint violated"
 *   -> TransformInput不符合RPY或四元数格式
 *   -> FrameIdentifier未提供必需的方法
 */

} // namespace fast_tf
