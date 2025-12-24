#pragma once

#include "compile_time_string.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <concepts>
#include <cstdint>

namespace fast_tf {

// ============================================================================
// 基础概念定义
// ============================================================================

// 时间戳概念 - 任何可以用 uint64_t 纳秒表示的时间类型
template <typename T>
concept Timestamp = requires(T t) {
    { static_cast<uint64_t>(t.count()) } -> std::convertible_to<uint64_t>;
} || std::integral<T>;

// 浮点数概念
template <typename T>
concept FloatingPoint = std::floating_point<T>;

// ============================================================================
// 变换输入概念
// ============================================================================

// RPY + 平移向量（三个旋转角 + 三个平移）
template <typename T>
concept RPYTransform = requires(T t) {
    { t.roll } -> std::convertible_to<double>;
    { t.pitch } -> std::convertible_to<double>;
    { t.yaw } -> std::convertible_to<double>;
    { t.x } -> std::convertible_to<double>;
    { t.y } -> std::convertible_to<double>;
    { t.z } -> std::convertible_to<double>;
};

// 四元数 + 平移向量
template <typename T>
concept QuaternionTransform = requires(T t) {
    { t.w } -> std::convertible_to<double>;
    { t.x } -> std::convertible_to<double>;
    { t.y } -> std::convertible_to<double>;
    { t.z } -> std::convertible_to<double>;
    { t.tx } -> std::convertible_to<double>;
    { t.ty } -> std::convertible_to<double>;
    { t.tz } -> std::convertible_to<double>;
};

// 通用变换输入（RPY或四元数）
template <typename T>
concept TransformInput = RPYTransform<T> || QuaternionTransform<T>;

// ============================================================================
// 坐标系标识符概念
// ============================================================================

// 编译期字符串帧标识符
template <typename T>
concept FrameIdentifier = requires(T t) {
    { t.view() } -> std::convertible_to<std::string_view>;
    { t.hash() } -> std::convertible_to<uint64_t>;
};

// 编译期已知的坐标系ID（类型标记）
template <typename T>
concept KnownFrameId = requires {
    typename T::frame_id_type;
    { T::id() } -> std::convertible_to<uint64_t>;
    { T::name() } -> std::convertible_to<std::string_view>;
};

// ============================================================================
// 时间查询概念
// ============================================================================

// 基于时间的变换查询
template <typename T>
concept TemporalQuery = requires(T q) {
    { q.timestamp } -> std::convertible_to<uint64_t>;
    { q.allow_extrapolation } -> std::convertible_to<bool>;
};

// ============================================================================
// 缓冲区概念
// ============================================================================

// 能够存储变换的缓冲区
template <typename T>
concept TransformBuffer = requires(T buf) {
    typename T::value_type;
    { buf.capacity() } -> std::convertible_to<std::size_t>;
    { buf.size() } -> std::convertible_to<std::size_t>;
};

// ============================================================================
// 树节点概念
// ============================================================================

// 有效的树节点（运行期概念）
template <typename T>
concept TreeNodeConcept = requires(T node) {
    typename T::frame_id_type;
    typename T::timestamp_type;
    { node.frame_name() } -> std::convertible_to<std::string_view>;
    { node.get_transform(uint64_t{}) } -> std::convertible_to<Eigen::Matrix4d>;
};

// ============================================================================
// 树结构验证概念
// ============================================================================

// 编译期验证的树结构
template <typename T>
concept ValidTransformTree = requires {
    typename T::root_frame;
    typename T::all_frames;
    { T::verify_connectivity() } -> std::convertible_to<bool>;
};

// ============================================================================
// 锁概念
// ============================================================================

// 读写互斥体（用于TBB集成）
template <typename T>
concept ReadWriteMutex = requires(T mtx) {
    { mtx.lock() } -> std::same_as<void>;
    { mtx.unlock() } -> std::same_as<void>;
    { mtx.lock_read() } -> std::same_as<void>;
    { mtx.unlock_read() } -> std::same_as<void>;
};

// ============================================================================
// 变换矩阵概念
// ============================================================================

template <typename T>
concept TransformMatrix = requires(T m) {
    { m * m } -> std::convertible_to<T>;
    { m.inverse() } -> std::convertible_to<T>;
};

// ============================================================================
// 辅助概念（内部使用）
// ============================================================================

// 可哈希类型
template <typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// 可比较类型
template <typename T>
concept Comparable = requires(T a, T b) {
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
    { a < b } -> std::convertible_to<bool>;
};

} // namespace fast_tf
