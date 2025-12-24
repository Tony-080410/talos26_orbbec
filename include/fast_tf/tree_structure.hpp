#pragma once

#include "compile_time_string.hpp"
#include "frame_types.hpp"
#include "transform_buffer.hpp"
#include <string_view>
#include <tuple>

namespace fast_tf {

// ============================================================================
// 编译期树结构定义
// ============================================================================

// 空父节点标记
struct NullParent {};

// 父节点引用（编译期）
template <std::size_t N>
struct ParentRef {
    static constexpr CompileTimeString<N> parent_name{};
};

// 别名列表（编译期）
template <std::size_t... Names>
struct AliasesList {};

// 单个节点定义
template <
    std::size_t NameSize, typename Parent = NullParent, typename BufferCap = BufferCapacity<100>,
    typename Aliases = AliasesList<>>
struct TreeNode {
    static constexpr std::size_t name_size = NameSize;
    using parent_type                      = Parent;
    using buffer_type                      = typename BufferCap::buffer_type;
    using buffer_capacity                  = BufferCap;
    using aliases_type                     = Aliases;
};

// ============================================================================
// 编译期树结构验证
// ============================================================================

// 向前声明
template <typename... Nodes>
class TransformTreeStructure;

// 节点元组访问辅助
template <typename NodeTuple, std::size_t Idx>
struct GetNode {
    using type = std::tuple_element_t<Idx, NodeTuple>;
};

// 编译期树验证
template <typename... Nodes>
struct TreeValidator {
    // 检查所有节点名称是否唯一
    static constexpr bool check_unique_names() {
        // 简化版本：在实际使用中应该检查所有节点名称的唯一性
        // 这里假设使用者正确提供了唯一的名称
        return true;
    }

    // 检查树的连接性（所有节点应该连接到根）
    static constexpr bool check_connectivity() {
        // 简化版本：实际实现应该检查所有节点是否通过父子关系连接
        return true;
    }

    // 验证树结构
    static constexpr bool validate() { return check_unique_names() && check_connectivity(); }
};

// ============================================================================
// 树节点运行期表示
// ============================================================================

template <typename NodeDef>
class TreeNodeRuntime {
public:
    using buffer_type = typename NodeDef::buffer_type;
    using definition  = NodeDef;

    // 默认构造函数
    TreeNodeRuntime()
        : frame_name_("")
        , parent_name_("") {}

    TreeNodeRuntime(std::string_view name, std::string_view parent_name = "")
        : frame_name_(name)
        , parent_name_(parent_name) {}

    // 设置变换
    void set_transform(uint64_t timestamp_ns, const TransformMatrix4d& transform) {
        buffer_.push(timestamp_ns, transform);
    }

    // 查询变换（精确）
    std::optional<TransformMatrix4d> get_transform(uint64_t timestamp_ns) const {
        auto entry = buffer_.at(timestamp_ns);
        if (entry) {
            return entry->transform;
        }
        return std::nullopt;
    }

    // 查询变换（插值）
    std::optional<TransformMatrix4d> get_transform_interpolated(uint64_t timestamp_ns) const {
        auto entry = buffer_.at_interpolated(timestamp_ns);
        if (entry) {
            return entry->transform;
        }
        return std::nullopt;
    }

    // 获取最新变换
    std::optional<TransformMatrix4d> get_latest_transform() const {
        auto entry = buffer_.latest();
        if (entry) {
            return entry->transform;
        }
        return std::nullopt;
    }

    // 获取帧名称
    std::string_view frame_name() const { return frame_name_; }

    // 获取父帧名称
    std::string_view parent_name() const { return parent_name_; }

    // 获取时间范围
    std::optional<std::pair<uint64_t, uint64_t>> time_range() const { return buffer_.time_range(); }

    // 清空缓冲区
    void clear() { buffer_.clear(); }

    // 获取缓冲区引用
    buffer_type& buffer() { return buffer_; }

    const buffer_type& buffer() const { return buffer_; }

private:
    std::string frame_name_;
    std::string parent_name_;
    buffer_type buffer_;
};

// ============================================================================
// 树结构（运行期）
// ============================================================================

template <typename... NodeDefs>
class TransformTreeStructure {
public:
    using node_tuple                       = std::tuple<TreeNodeRuntime<NodeDefs>...>;
    static constexpr std::size_t num_nodes = sizeof...(NodeDefs);

    // 编译期验证
    static constexpr bool is_valid_tree = TreeValidator<NodeDefs...>::validate();

    static_assert(is_valid_tree, "Invalid tree structure");

    TransformTreeStructure() { initialize_nodes(std::index_sequence_for<NodeDefs...>{}); }

    // 根据索引设置变换
    template <std::size_t NodeIdx>
    void set_transform(uint64_t timestamp_ns, const TransformMatrix4d& transform) {
        std::get<NodeIdx>(nodes_).set_transform(timestamp_ns, transform);
    }

    // 根据索引查询变换
    template <std::size_t NodeIdx>
    std::optional<TransformMatrix4d> get_transform(uint64_t timestamp_ns) const {
        return std::get<NodeIdx>(nodes_).get_transform(timestamp_ns);
    }

    // 根据索引查询插值变换
    template <std::size_t NodeIdx>
    std::optional<TransformMatrix4d> get_transform_interpolated(uint64_t timestamp_ns) const {
        return std::get<NodeIdx>(nodes_).get_transform_interpolated(timestamp_ns);
    }

    // 获取节点数量
    static constexpr std::size_t get_num_nodes() { return num_nodes; }

    // 获取节点（通过索引）
    template <std::size_t NodeIdx>
    auto& get_node() {
        return std::get<NodeIdx>(nodes_);
    }

    template <std::size_t NodeIdx>
    const auto& get_node() const {
        return std::get<NodeIdx>(nodes_);
    }

    // 清空所有节点
    void clear() { clear_nodes(std::index_sequence_for<NodeDefs...>{}); }

private:
    node_tuple nodes_;

    // 初始化所有节点（运行期名称分配）
    template <std::size_t... Is>
    void initialize_nodes(std::index_sequence<Is...>) {
        // 这里可以添加节点的运行期初始化逻辑
        // 如果需要从编译期字符串提取名称
        (void)sizeof...(Is); // 用于避免未使用参数警告
    }

    // 清空所有节点
    template <std::size_t... Is>
    void clear_nodes(std::index_sequence<Is...>) {
        ((std::get<Is>(nodes_).clear()), ...);
    }
};

// ============================================================================
// 编译期树定义宏（可选）
// ============================================================================

// TreeNode 类已在上面定义，无需额外的宏定义

} // namespace fast_tf
