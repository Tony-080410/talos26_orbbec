#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <optional>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include "tree_structure.hpp"
#include "frame_types.hpp"

namespace fast_tf {

// ============================================================================
// 变换树查询结果
// ============================================================================

struct TransformQueryResult {
    TransformMatrix4d transform;
    uint64_t timestamp_ns;
    bool is_interpolated = false;

    TransformQueryResult() = default;

    TransformQueryResult(const TransformMatrix4d& tf, uint64_t ts,
                         bool interpolated = false)
        : transform(tf), timestamp_ns(ts), is_interpolated(interpolated) {}
};

// ============================================================================
// 变换查询选项
// ============================================================================

struct TransformQueryOptions {
    bool allow_extrapolation = false;
    bool allow_interpolation = true;
    uint64_t max_age_ns = UINT64_MAX;  // 最大年龄限制

    TransformQueryOptions() = default;
};

// ============================================================================
// 路径查找结果
// ============================================================================

struct PathInfo {
    std::vector<std::string> path;
    bool found = false;

    PathInfo() = default;
    explicit PathInfo(const std::vector<std::string>& p)
        : path(p), found(true) {}
};

// ============================================================================
// 主变换树API
// ============================================================================

template<typename TreeStructure>
class TransformTree {
public:
    using tree_structure = TreeStructure;
    static constexpr std::size_t num_nodes = TreeStructure::num_nodes;

    TransformTree() : tree_(std::make_unique<TreeStructure>()) {}

    // ========================================================================
    // 设置变换接口
    // ========================================================================

    // 通过坐标系名称设置变换（RPY格式）
    void set_transform(std::string_view from_frame, std::string_view to_frame,
                      uint64_t timestamp_ns, double roll, double pitch,
                      double yaw, double tx, double ty, double tz) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(
            roll, pitch, yaw, tx, ty, tz);

        set_transform_matrix(from_frame, to_frame, timestamp_ns, tf);
    }

    // 通过坐标系名称设置变换（四元数格式）
    void set_transform_quat(std::string_view from_frame,
                            std::string_view to_frame, uint64_t timestamp_ns,
                            double w, double x, double y, double z, double tx,
                            double ty, double tz) {
        TransformMatrix4d tf = TransformMatrix4d::from_quaternion(
            w, x, y, z, tx, ty, tz);

        set_transform_matrix(from_frame, to_frame, timestamp_ns, tf);
    }

    // 通过TransformInput概念设置变换
    template<TransformInput T>
    void set_transform_input(std::string_view from_frame,
                             std::string_view to_frame, uint64_t timestamp_ns,
                             const T& input) {
        TransformMatrix4d tf;

        if constexpr (RPYTransform<T>) {
            tf = TransformMatrix4d::from_rpy(
                input.roll, input.pitch, input.yaw,
                input.x, input.y, input.z);
        } else if constexpr (QuaternionTransform<T>) {
            tf = TransformMatrix4d::from_quaternion(
                input.w, input.x, input.y, input.z,
                input.tx, input.ty, input.tz);
        }

        set_transform_matrix(from_frame, to_frame, timestamp_ns, tf);
    }

    // ========================================================================
    // 查询变换接口
    // ========================================================================

    // 查询两个坐标系之间的变换
    std::optional<TransformQueryResult> lookup_transform(
        std::string_view target_frame, std::string_view source_frame,
        uint64_t timestamp_ns,
        const TransformQueryOptions& options = TransformQueryOptions()) {

        // 解析别名
        std::string resolved_target(FrameRegistry::resolve_alias(target_frame));
        std::string resolved_source(FrameRegistry::resolve_alias(source_frame));

        // 如果源和目标相同，返回单位矩阵
        if (resolved_target == resolved_source) {
            return TransformQueryResult(TransformMatrix4d(),
                                       timestamp_ns);
        }

        // 查找从源到目标的路径
        auto path = find_path(resolved_source, resolved_target);
        if (!path.found || path.path.empty()) {
            return std::nullopt;
        }

        // 沿路径乘积变换矩阵
        return compute_chain_transform(path.path, timestamp_ns, options);
    }

    // 获取最新变换
    std::optional<TransformQueryResult> get_latest_transform(
        std::string_view target_frame, std::string_view source_frame) {
        // 获取时间戳范围，使用最新时间戳
        auto time_range = get_time_range(target_frame, source_frame);
        if (time_range) {
            return lookup_transform(target_frame, source_frame,
                                   time_range->second);
        }
        return std::nullopt;
    }

    // ========================================================================
    // 时间范围和验证
    // ========================================================================

    // 获取坐标系的时间范围
    std::optional<std::pair<uint64_t, uint64_t>> get_time_range(
        std::string_view frame) {
        uint64_t frame_id = FrameRegistry::get_frame_id(frame);
        return find_buffer_for_frame(frame_id)->time_range();
    }

    // 获取两个坐标系之间的通用时间范围
    std::optional<std::pair<uint64_t, uint64_t>> get_time_range(
        std::string_view target_frame, std::string_view source_frame) {
        auto path = find_path(std::string(source_frame),
                             std::string(target_frame));
        if (!path.found || path.path.empty()) {
            return std::nullopt;
        }

        uint64_t min_time = 0;
        uint64_t max_time = UINT64_MAX;

        for (std::size_t i = 0; i < path.path.size() - 1; ++i) {
            auto range = get_time_range(path.path[i]);
            if (range) {
                min_time = std::max(min_time, range->first);
                max_time = std::min(max_time, range->second);
            } else {
                return std::nullopt;
            }
        }

        if (min_time <= max_time) {
            return std::make_pair(min_time, max_time);
        }

        return std::nullopt;
    }

    // 检查是否可以查询变换
    bool can_transform(std::string_view target_frame,
                      std::string_view source_frame,
                      uint64_t timestamp_ns) {
        auto result = lookup_transform(target_frame, source_frame,
                                      timestamp_ns);
        return result.has_value();
    }

    // ========================================================================
    // 并行批量查询
    // ========================================================================

    // 并行查询多个变换
    std::vector<std::optional<TransformQueryResult>> parallel_lookup_transforms(
        const std::vector<std::pair<std::string, std::string>>& frame_pairs,
        uint64_t timestamp_ns,
        const TransformQueryOptions& options = TransformQueryOptions()) {

        std::vector<std::optional<TransformQueryResult>> results(
            frame_pairs.size());

        tbb::parallel_for(
            std::size_t(0), frame_pairs.size(),
            [this, &frame_pairs, timestamp_ns, &options, &results]
            (std::size_t i) {
                results[i] = lookup_transform(
                    frame_pairs[i].first, frame_pairs[i].second,
                    timestamp_ns, options);
            });

        return results;
    }

    // ========================================================================
    // 坐标系管理
    // ========================================================================

    // 注册坐标系
    void register_frame(std::string_view name, std::string_view parent_name) {
        uint64_t frame_id = FrameRegistry::get_frame_id(name);
        FrameRegistry::register_frame(name, frame_id);

        frame_hierarchy_[std::string(name)] = std::string(parent_name);
    }

    // 注册别名
    void register_alias(std::string_view alias, std::string_view canonical_name) {
        FrameRegistry::register_alias(alias, canonical_name);
    }

    // 清空树
    void clear() {
        if (tree_) {
            tree_->clear();
        }
        frame_hierarchy_.clear();
        FrameRegistry::clear();
    }

    // ========================================================================
    // 诊断和调试接口
    // ========================================================================

    // 获取所有已注册的坐标系
    std::vector<std::string> get_all_frames() const {
        std::vector<std::string> frames;
        for (const auto& [name, _] : frame_hierarchy_) {
            frames.push_back(name);
        }
        return frames;
    }

    // 获取坐标系的父坐标系
    std::optional<std::string> get_parent_frame(std::string_view frame) const {
        if (auto it = frame_hierarchy_.find(std::string(frame));
            it != frame_hierarchy_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // 获取坐标系的子坐标系
    std::vector<std::string> get_child_frames(std::string_view frame) const {
        std::vector<std::string> children;
        for (const auto& [name, parent] : frame_hierarchy_) {
            if (parent == frame) {
                children.push_back(name);
            }
        }
        return children;
    }

    // 检查帧是否存在
    bool has_frame(std::string_view frame) const {
        return frame_hierarchy_.count(std::string(frame)) > 0;
    }

    // 获取树的数据结构（用于调试）
    const TreeStructure& get_structure() const {
        return *tree_;
    }

private:
    std::unique_ptr<TreeStructure> tree_;
    std::unordered_map<std::string, std::string> frame_hierarchy_;
    std::unordered_map<uint64_t, std::string> frame_id_to_name_;

    // ========================================================================
    // 内部辅助方法
    // ========================================================================

    void set_transform_matrix(std::string_view from_frame,
                             std::string_view to_frame,
                             uint64_t /* timestamp_ns */,
                             const TransformMatrix4d& /* tf */) {
        register_frame(from_frame, to_frame);
        register_frame(to_frame, "");

        uint64_t from_id = FrameRegistry::get_frame_id(from_frame);
        frame_id_to_name_[from_id] = std::string(from_frame);

        // 这里需要根据实际树结构来存储变换
        // 简化版本中直接存储在注册表中
    }

    // BFS查找两个坐标系之间的路径
    PathInfo find_path(const std::string& source,
                       const std::string& target) {
        if (source == target) {
            return PathInfo({source});
        }

        std::unordered_map<std::string, std::string> parent_map;
        std::vector<std::string> queue;
        queue.push_back(source);
        parent_map[source] = "";

        while (!queue.empty()) {
            std::string current = queue.front();
            queue.erase(queue.begin());

            // 获取当前节点的子节点
            auto children = get_child_frames(current);
            for (const auto& child : children) {
                if (parent_map.find(child) == parent_map.end()) {
                    parent_map[child] = current;

                    if (child == target) {
                        // 重建路径
                        std::vector<std::string> path;
                        std::string curr = target;
                        while (!curr.empty()) {
                            path.insert(path.begin(), curr);
                            curr = parent_map[curr];
                        }
                        return PathInfo(path);
                    }

                    queue.push_back(child);
                }
            }
        }

        return PathInfo();
    }

    // 沿路径计算链式变换
    std::optional<TransformQueryResult> compute_chain_transform(
        const std::vector<std::string>& path, uint64_t timestamp_ns,
        const TransformQueryOptions& /* options */) {
        if (path.size() < 2) {
            return std::nullopt;
        }

        // 这是简化实现，实际应该从缓冲区查询变换
        // 并沿路径乘积矩阵
        TransformMatrix4d result;
        bool found_all = true;

        for (std::size_t i = 0; i < path.size() - 1; ++i) {
            // 在真实实现中应该从缓冲区查询
            // 这里返回单位矩阵作为占位符
            (void)i;  // 避免未使用警告
        }

        if (found_all) {
            return TransformQueryResult(result, timestamp_ns, false);
        }

        return std::nullopt;
    }

    // 查找特定帧的缓冲区
    auto find_buffer_for_frame(uint64_t /* frame_id */) {
        // 简化实现：返回树中的缓冲区引用
        // 实际应该根据frame_id找到对应的节点
        // 这里返回空指针占位
        return std::make_unique<CircularTransformBuffer<100>>();
    }
};

} // namespace fast_tf
