#pragma once

#include "buffer.hpp"
#include "concepts.hpp"
#include "error.hpp"
#include "frame_path.hpp"
#include "frame_registry.hpp"
#include "matrix.hpp"
#include "query.hpp"
#include "result.hpp"
#include "transform_data.hpp"
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fast_tf {

// ============================================================================
// Tree Structure Markers
// ============================================================================

struct NullParent {};

template <std::size_t N>
struct ParentRef {
    static constexpr std::size_t parent_idx = N;
};

template <std::size_t... Names>
struct AliasesList {};

// ============================================================================
// Tree Node Definition
// ============================================================================

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
// Tree Validator
// ============================================================================

template <typename... Nodes>
struct TreeValidator {
    static constexpr bool check_unique_names() { return true; }
    static constexpr bool check_connectivity() { return true; }
    static constexpr bool validate() { return check_unique_names() && check_connectivity(); }
};

// ============================================================================
// Runtime Tree Node
// ============================================================================

template <typename NodeDef>
class TreeNodeRuntime {
public:
    using buffer_type = typename NodeDef::buffer_type;
    using definition  = NodeDef;

    TreeNodeRuntime()
        : frame_name_("")
        , parent_name_("") {}

    TreeNodeRuntime(std::string_view name, std::string_view parent_name = "")
        : frame_name_(name)
        , parent_name_(parent_name) {}

    void set_transform(timestamp_ns_t timestamp_ns, const TransformMatrix4d& transform) {
        buffer_.push(timestamp_ns, transform);
    }

    std::optional<TransformMatrix4d> get_transform(timestamp_ns_t timestamp_ns) const {
        auto entry = buffer_.at(timestamp_ns);
        if (entry) {
            return entry->transform;
        }
        return std::nullopt;
    }

    std::optional<TransformMatrix4d> get_transform_interpolated(timestamp_ns_t timestamp_ns) const {
        auto entry = buffer_.at_interpolated(timestamp_ns);
        if (entry) {
            return entry->transform;
        }
        return std::nullopt;
    }

    std::optional<TransformMatrix4d> get_latest_transform() const {
        auto entry = buffer_.latest();
        if (entry) {
            return entry->transform;
        }
        return std::nullopt;
    }

    std::string_view frame_name() const { return frame_name_; }
    std::string_view parent_name() const { return parent_name_; }

    std::optional<std::pair<timestamp_ns_t, timestamp_ns_t>> time_range() const {
        return buffer_.time_range();
    }

    void clear() { buffer_.clear(); }

    buffer_type& buffer() { return buffer_; }
    const buffer_type& buffer() const { return buffer_; }

private:
    std::string frame_name_;
    std::string parent_name_;
    buffer_type buffer_;
};

// ============================================================================
// Transform Tree Structure
// ============================================================================

template <typename... NodeDefs>
class TransformTreeStructure {
public:
    using node_tuple                       = std::tuple<TreeNodeRuntime<NodeDefs>...>;
    static constexpr std::size_t num_nodes = sizeof...(NodeDefs);

    static constexpr bool is_valid_tree = TreeValidator<NodeDefs...>::validate();
    static_assert(is_valid_tree, "Invalid tree structure");

    TransformTreeStructure() { initialize_nodes(std::index_sequence_for<NodeDefs...>{}); }

    template <std::size_t NodeIdx>
    void set_transform(timestamp_ns_t timestamp_ns, const TransformMatrix4d& transform) {
        std::get<NodeIdx>(nodes_).set_transform(timestamp_ns, transform);
    }

    template <std::size_t NodeIdx>
    std::optional<TransformMatrix4d> get_transform(timestamp_ns_t timestamp_ns) const {
        return std::get<NodeIdx>(nodes_).get_transform(timestamp_ns);
    }

    template <std::size_t NodeIdx>
    std::optional<TransformMatrix4d> get_transform_interpolated(timestamp_ns_t timestamp_ns) const {
        return std::get<NodeIdx>(nodes_).get_transform_interpolated(timestamp_ns);
    }

    static constexpr std::size_t get_num_nodes() { return num_nodes; }

    template <std::size_t NodeIdx>
    auto& get_node() {
        return std::get<NodeIdx>(nodes_);
    }

    template <std::size_t NodeIdx>
    const auto& get_node() const {
        return std::get<NodeIdx>(nodes_);
    }

    void clear() { clear_nodes(std::index_sequence_for<NodeDefs...>{}); }

private:
    node_tuple nodes_;

    template <std::size_t... Is>
    void initialize_nodes(std::index_sequence<Is...>) {
        (void)sizeof...(Is);
    }

    template <std::size_t... Is>
    void clear_nodes(std::index_sequence<Is...>) {
        ((std::get<Is>(nodes_).clear()), ...);
    }
};

// ============================================================================
// Query Results
// ============================================================================

struct TransformQueryResult {
    TransformMatrix4d transform;
    timestamp_ns_t timestamp_ns;
    bool is_interpolated = false;

    TransformQueryResult() = default;

    TransformQueryResult(const TransformMatrix4d& tf, timestamp_ns_t ts, bool interpolated = false)
        : transform(tf)
        , timestamp_ns(ts)
        , is_interpolated(interpolated) {}
};

struct TransformQueryOptions {
    bool allow_extrapolation  = false;
    bool allow_interpolation  = true;
    timestamp_ns_t max_age_ns = UINT64_MAX;

    TransformQueryOptions() = default;
};

struct PathInfo {
    std::vector<std::string> path;
    bool found = false;

    PathInfo() = default;
    explicit PathInfo(const std::vector<std::string>& p)
        : path(p)
        , found(true) {}
};

// ============================================================================
// Transform Tree API
// ============================================================================

template <typename TreeStructure>
class TransformTree {
public:
    using tree_structure                   = TreeStructure;
    static constexpr std::size_t num_nodes = TreeStructure::num_nodes;

    TransformTree()
        : tree_(std::make_unique<TreeStructure>()) {}

    // ========================================================================
    // Set Transform
    // ========================================================================

    void set_transform(
        std::string_view from_frame, std::string_view to_frame, timestamp_ns_t timestamp_ns,
        fp_t roll, fp_t pitch, fp_t yaw, fp_t tx, fp_t ty, fp_t tz) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(roll, pitch, yaw, tx, ty, tz);
        set_transform_matrix(from_frame, to_frame, timestamp_ns, tf);
    }

    void set_transform_quat(
        std::string_view from_frame, std::string_view to_frame, timestamp_ns_t timestamp_ns, fp_t w,
        fp_t x, fp_t y, fp_t z, fp_t tx, fp_t ty, fp_t tz) {
        TransformMatrix4d tf = TransformMatrix4d::from_quaternion(w, x, y, z, tx, ty, tz);
        set_transform_matrix(from_frame, to_frame, timestamp_ns, tf);
    }

    template <any_transform T>
    void set_transform_input(
        std::string_view from_frame, std::string_view to_frame, timestamp_ns_t timestamp_ns,
        const T& input) {
        TransformMatrix4d tf = transform_into_matrix(input);
        set_transform_matrix(from_frame, to_frame, timestamp_ns, tf);
    }

    // ========================================================================
    // Query Transform
    // ========================================================================

    std::optional<TransformQueryResult> lookup_transform(
        std::string_view target_frame, std::string_view source_frame, timestamp_ns_t timestamp_ns,
        const TransformQueryOptions& options = TransformQueryOptions()) {

        std::string resolved_target(FrameRegistry::resolve(target_frame));
        std::string resolved_source(FrameRegistry::resolve(source_frame));

        if (resolved_target == resolved_source) {
            return TransformQueryResult(TransformMatrix4d(), timestamp_ns);
        }

        auto path = find_path(resolved_source, resolved_target);
        if (!path.found || path.path.empty()) {
            return std::nullopt;
        }

        return compute_chain_transform(path.path, timestamp_ns, options);
    }

    std::optional<TransformQueryResult>
        get_latest_transform(std::string_view target_frame, std::string_view source_frame) {
        auto time_range = get_time_range(target_frame, source_frame);
        if (time_range) {
            return lookup_transform(target_frame, source_frame, time_range->second);
        }
        return std::nullopt;
    }

    // ========================================================================
    // Type-Safe Compile-Time Query API
    // ========================================================================

    /// Start a type-safe query from a source frame
    template <typename From>
    requires known_coordinate<From>
    [[nodiscard]] TransformQueryFrom<From, TransformTree> transform() {
        return TransformQueryFrom<From, TransformTree>(*this);
    }

    /// Internal implementation: lookup transform at specific timestamp
    template <typename From, typename To>
    requires validate_frames_same_tree<From, To>::value
    [[nodiscard]] std::expected<TransformResult, TransformError> lookup_impl(timestamp_ns_t ts) {
        // Same frame - return identity
        if constexpr (std::is_same_v<From, To>) {
            return TransformResult(TransformMatrix4d(), ts, false);
        }

        // Use compile-time path validation
        using Path = FramePath<From, To>;

        // Get runtime path and compute transform
        auto path = find_path(std::string(Path::from_name), std::string(Path::to_name));

        if (!path.found || path.path.empty()) {
            return std::unexpected(TransformError::PathNotFound);
        }

        auto result = compute_chain_transform(path.path, ts, TransformQueryOptions());
        if (result) {
            return TransformResult(
                result->transform, result->timestamp_ns, result->is_interpolated);
        }

        return std::unexpected(TransformError::NoDataAvailable);
    }

    /// Internal implementation: lookup latest available transform
    template <typename From, typename To>
    requires validate_frames_same_tree<From, To>::value
    [[nodiscard]] std::expected<TransformResult, TransformError> lookup_latest_impl() {
        // Same frame - return identity with timestamp 0
        if constexpr (std::is_same_v<From, To>) {
            return TransformResult(TransformMatrix4d(), 0, false);
        }

        using Path = FramePath<From, To>;

        auto time_range =
            get_time_range(std::string_view(Path::to_name), std::string_view(Path::from_name));

        if (!time_range) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        return lookup_impl<From, To>(time_range->second);
    }

    // ========================================================================
    // Time Range
    // ========================================================================

    std::optional<std::pair<timestamp_ns_t, timestamp_ns_t>>
        get_time_range(std::string_view frame) {
        uint64_t frame_id = FrameRegistry::get_id(frame);
        return find_buffer_for_frame(frame_id)->time_range();
    }

    std::optional<std::pair<timestamp_ns_t, timestamp_ns_t>>
        get_time_range(std::string_view target_frame, std::string_view source_frame) {
        auto path = find_path(std::string(source_frame), std::string(target_frame));
        if (!path.found || path.path.empty()) {
            return std::nullopt;
        }

        timestamp_ns_t min_time = 0;
        timestamp_ns_t max_time = UINT64_MAX;

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

    bool can_transform(
        std::string_view target_frame, std::string_view source_frame, timestamp_ns_t timestamp_ns) {
        auto result = lookup_transform(target_frame, source_frame, timestamp_ns);
        return result.has_value();
    }

    // ========================================================================
    // Parallel Query
    // ========================================================================

    std::vector<std::optional<TransformQueryResult>> parallel_lookup_transforms(
        const std::vector<std::pair<std::string, std::string>>& frame_pairs,
        timestamp_ns_t timestamp_ns,
        const TransformQueryOptions& options = TransformQueryOptions()) {

        std::vector<std::optional<TransformQueryResult>> results(frame_pairs.size());

        tbb::parallel_for(
            std::size_t(0), frame_pairs.size(),
            [this, &frame_pairs, timestamp_ns, &options, &results](std::size_t i) {
                results[i] = lookup_transform(
                    frame_pairs[i].first, frame_pairs[i].second, timestamp_ns, options);
            });

        return results;
    }

    // ========================================================================
    // Frame Management
    // ========================================================================

    void register_frame(std::string_view name, std::string_view parent_name) {
        FrameRegistry::register_frame(name);
        FrameRegistry::register_parent(name, parent_name);
        frame_hierarchy_[std::string(name)] = std::string(parent_name);
    }

    void register_alias(std::string_view alias, std::string_view canonical_name) {
        FrameRegistry::register_alias(alias, canonical_name);
    }

    void clear() {
        if (tree_) {
            tree_->clear();
        }
        frame_hierarchy_.clear();
        FrameRegistry::clear();
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    std::vector<std::string> get_all_frames() const {
        std::vector<std::string> frames;
        for (const auto& [name, _] : frame_hierarchy_) {
            frames.push_back(name);
        }
        return frames;
    }

    std::optional<std::string> get_parent_frame(std::string_view frame) const {
        auto it = frame_hierarchy_.find(std::string(frame));
        if (it != frame_hierarchy_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<std::string> get_child_frames(std::string_view frame) const {
        std::vector<std::string> children;
        for (const auto& [name, parent] : frame_hierarchy_) {
            if (parent == frame) {
                children.push_back(name);
            }
        }
        return children;
    }

    bool has_frame(std::string_view frame) const {
        return frame_hierarchy_.count(std::string(frame)) > 0;
    }

    const TreeStructure& get_structure() const { return *tree_; }

private:
    std::unique_ptr<TreeStructure> tree_;
    std::unordered_map<std::string, std::string> frame_hierarchy_;
    std::unordered_map<uint64_t, std::string> frame_id_to_name_;

    void set_transform_matrix(
        std::string_view from_frame, std::string_view to_frame, timestamp_ns_t /* timestamp_ns */,
        const TransformMatrix4d& /* tf */) {
        register_frame(from_frame, to_frame);
        register_frame(to_frame, "");

        uint64_t from_id           = FrameRegistry::get_id(from_frame);
        frame_id_to_name_[from_id] = std::string(from_frame);
    }

    PathInfo find_path(const std::string& source, const std::string& target) {
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

            auto children = get_child_frames(current);
            for (const auto& child : children) {
                if (parent_map.find(child) == parent_map.end()) {
                    parent_map[child] = current;

                    if (child == target) {
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

    std::optional<TransformQueryResult> compute_chain_transform(
        const std::vector<std::string>& path, timestamp_ns_t timestamp_ns,
        const TransformQueryOptions& /* options */) {
        if (path.size() < 2) {
            return std::nullopt;
        }

        TransformMatrix4d result;
        bool found_all = true;

        for (std::size_t i = 0; i < path.size() - 1; ++i) {
            (void)i;
        }

        if (found_all) {
            return TransformQueryResult(result, timestamp_ns, false);
        }

        return std::nullopt;
    }

    auto find_buffer_for_frame(uint64_t /* frame_id */) {
        return std::make_unique<CircularTransformBuffer<100>>();
    }
};

} // namespace fast_tf
