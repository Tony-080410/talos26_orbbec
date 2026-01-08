#pragma once

/**
 * @file
 * @brief Flow graph tracking for data pipeline visualization
 *
 * This module provides infrastructure for tracking data flow through
 * the pipeline. When enabled, it records producer-consumer relationships
 * between channels, enabling visualization and analysis of the
 * pipeline topology.
 *
 * ## Enabling flow graph
 *
 * Define `ENABLE_FLOW_GRAPH=1` before including this header, or compile
 * with `-DENABLE_FLOW_GRAPH=1`.
 *
 * ## Output formats
 *
 * - **DOT**: Graphviz format for visualization
 * - **JSON**: Machine-readable format for analysis
 * - **Callback**: Custom handlers for real-time processing
 */

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace primitive {

// ============================================================================
// Compile-time switch: define ENABLE_FLOW_GRAPH=1 to enable
// ============================================================================
#ifdef ENABLE_FLOW_GRAPH
inline constexpr bool kFlowGraphEnabled = true;
#else
inline constexpr bool kFlowGraphEnabled = false;
#endif

// ============================================================================
// Node identifier: file + line
// ============================================================================

/**
 * @brief Identifies a source location in the code
 *
 * Combines file path and line number to uniquely identify
 * where a channel was created or used.
 */
struct NodeId {
    std::string_view file;
    int line = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return !file.empty() && line > 0; }

    [[nodiscard]] bool operator==(const NodeId& other) const noexcept {
        if (line != other.line)
            return false;
        if (file == other.file)
            return true;
        return std::string_view{file} == std::string_view{other.file};
    }
};

struct NodeIdHash {
    [[nodiscard]] std::size_t operator()(const NodeId& id) const noexcept {
        std::size_t h = std::hash<int>{}(id.line);
        if (!id.file.empty()) {
            h ^= std::hash<std::string_view>{}(id.file) << 1;
        }
        return h;
    }
};

// ============================================================================
// Flow graph: directed graph with nodes and edges
// ============================================================================

/**
 * @brief Directed graph representing data flow through the pipeline
 *
 * Records producer-consumer relationships between channels, enabling
 * visualization and analysis of pipeline topology.
 *
 * ## Usage
 *
 * ```cpp
 * FlowGraph graph;
 * graph.add_node({"producer.cpp", 42}, "Frame");
 * graph.add_node({"consumer.cpp", 84}, "Frame");
 * graph.add_edge({...}, {...}, "Frame");
 *
 * // Export for visualization
 * auto dot = graph.to_dot("Pipeline");
 *
 * // Topological analysis
 * auto result = graph.topological_sort();
 * if (result.has_cycle) {
 *     // Handle cycle detection
 * }
 * ```
 *
 * ## Thread safety
 *
 * The graph is not thread-safe for concurrent modification.
 * Initialize and modify in a single-threaded context.
 */
class FlowGraph {
public:
    /**
     * @brief A node in the flow graph
     *
     * Represents a channel endpoint (producer or consumer).
     */
    struct Node {
        NodeId id;
        std::string type_name;
        std::vector<std::size_t> successors;   // outgoing edges (node indices)
        std::vector<std::size_t> predecessors; // incoming edges (node indices)
    };

    /**
     * @brief An edge in the flow graph
     *
     * Represents a data flow from one node to another.
     */
    struct Edge {
        std::size_t from;
        std::size_t to;
        std::string type_name;
    };

    FlowGraph()                            = default;
    FlowGraph(const FlowGraph&)            = default;
    FlowGraph& operator=(const FlowGraph&) = default;
    FlowGraph(FlowGraph&&) noexcept        = default;
    FlowGraph& operator=(FlowGraph&&)      = default;

    // ========================================================================
    // Graph construction
    // ========================================================================

    /**
     * @brief Add a node to the graph
     *
     * If a node with the same ID already exists, returns its index.
     *
     * ## Parameters
     *
     * - `id`: source location identifying the node
     * - `type_name`: name of the data type flowing through
     *
     * ## Returns
     *
     * Index of the node in the graph
     */
    std::size_t add_node(const NodeId& id, const std::string_view type_name) {
        if (const auto it = node_map_.find(id); it != node_map_.end()) {
            return it->second;
        }
        const auto idx = nodes_.size();
        nodes_.push_back(Node{id, std::string{type_name}, {}, {}});
        node_map_[id] = idx;
        return idx;
    }

    /**
     * @brief Add an edge representing data flow
     *
     * Creates a directed edge from producer to consumer. Automatically
     * creates the nodes if they don't exist. Duplicate edges are ignored.
     *
     * ## Parameters
     *
     * - `producer`: source node (data producer)
     * - `consumer`: destination node (data consumer)
     * - `type_name`: data type flowing through this edge
     */
    void
        add_edge(const NodeId& producer, const NodeId& consumer, const std::string_view type_name) {
        const auto from_idx = add_node(producer, type_name);
        const auto to_idx   = add_node(consumer, type_name);

        // Avoid duplicate edges
        auto& succs = nodes_[from_idx].successors;
        if (std::find(succs.begin(), succs.end(), to_idx) == succs.end()) {
            succs.push_back(to_idx);
            nodes_[to_idx].predecessors.push_back(from_idx);
            edges_.push_back(Edge{from_idx, to_idx, std::string{type_name}});
        }
    }

    void clear() noexcept {
        nodes_.clear();
        edges_.clear();
        node_map_.clear();
    }

    // ========================================================================
    // Graph queries
    // ========================================================================

    [[nodiscard]] std::span<const Node> nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::span<const Edge> edges() const noexcept { return edges_; }
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t edge_count() const noexcept { return edges_.size(); }

    [[nodiscard]] std::optional<std::size_t> find_node(const NodeId id) const {
        if (auto it = node_map_.find(id); it != node_map_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    [[nodiscard]] const Node& node_at(const std::size_t idx) const { return nodes_.at(idx); }

    // Sources: nodes with no predecessors (entry points)
    /**
     * @brief Find entry points in the data flow
     *
     * Returns indices of nodes with no incoming edges. These are
     * the data sources in the pipeline.
     *
     * ## Returns
     *
     * Vector of node indices that have no predecessors
     */
    [[nodiscard]] std::vector<std::size_t> sources() const {
        std::vector<std::size_t> result;
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i].predecessors.empty()) {
                result.push_back(i);
            }
        }
        return result;
    }

    // Sinks: nodes with no successors (exit points)
    /**
     * @brief Find exit points in the data flow
     *
     * Returns indices of nodes with no outgoing edges. These are
     * the data sinks (final consumers) in the pipeline.
     *
     * ## Returns
     *
     * Vector of node indices that have no successors
     */
    [[nodiscard]] std::vector<std::size_t> sinks() const {
        std::vector<std::size_t> result;
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i].successors.empty()) {
                result.push_back(i);
            }
        }
        return result;
    }

    // ========================================================================
    // Scheduling: topological sort (Kahn's algorithm)
    // ========================================================================

    /**
     * @brief Result of topological sorting
     */
    struct TopoResult {
        std::vector<std::size_t> order;
        bool has_cycle = false;
    };

    /**
     * @brief Perform topological sort using Kahn's algorithm
     *
     * Orders nodes such that all edges go from earlier to later nodes.
     * Detects cycles in the graph.
     *
     * ## Returns
     *
     * TopoResult containing:
     * - `order`: nodes in topological order
     * - `has_cycle`: true if the graph contains a cycle
     *
     * ## Performance
     *
     * O(V + E) where V is nodes and E is edges.
     */
    [[nodiscard]] TopoResult topological_sort() const {
        TopoResult result;
        if (nodes_.empty()) {
            return result;
        }

        std::vector<std::size_t> in_degree(nodes_.size());
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            in_degree[i] = nodes_[i].predecessors.size();
        }

        std::vector<std::size_t> queue;
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            if (in_degree[i] == 0) {
                queue.push_back(i);
            }
        }

        while (!queue.empty()) {
            const auto u = queue.back();
            queue.pop_back();
            result.order.push_back(u);

            for (const auto v : nodes_[u].successors) {
                if (--in_degree[v] == 0) {
                    queue.push_back(v);
                }
            }
        }

        result.has_cycle = result.order.size() != nodes_.size();
        return result;
    }

    [[nodiscard]] bool has_cycle() const { return topological_sort().has_cycle; }

    // ========================================================================
    // Export: DOT format (Graphviz)
    // ========================================================================

    /**
     * @brief Export graph in DOT format for Graphviz
     *
     * Generates a DOT digraph that can be rendered with:
     * `dot -Tpng graph.dot -o graph.png`
     *
     * ## Parameters
     *
     * - `graph_name`: name for the digraph (default: "FlowGraph")
     *
     * ## Returns
     *
     * DOT format string
     */
    [[nodiscard]] std::string to_dot(std::string_view graph_name = "FlowGraph") const {
        std::string out;
        out.reserve(256 + nodes_.size() * 64 + edges_.size() * 48);

        out += "digraph ";
        out += graph_name;
        out += " {\n";
        out += "    rankdir=LR;\n";
        out += "    node [shape=box, fontname=\"monospace\"];\n\n";

        // Nodes
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& n = nodes_[i];
            out += "    n";
            out += std::to_string(i);
            out += " [label=\"";
            out += extract_filename(n.id.file);
            out += ":";
            out += std::to_string(n.id.line);
            out += "\\n";
            out += n.type_name;
            out += "\"];\n";
        }

        out += "\n";

        // Edges
        for (const auto& e : edges_) {
            out += "    n";
            out += std::to_string(e.from);
            out += " -> n";
            out += std::to_string(e.to);
            out += ";\n";
        }

        out += "}\n";
        return out;
    }

    // ========================================================================
    // Export: JSON format
    // ========================================================================

    /**
     * @brief Export graph in JSON format
     *
     * Machine-readable format for programmatic analysis.
     *
     * ## Returns
     *
     * JSON string with nodes and edges arrays
     */
    [[nodiscard]] std::string to_json() const {
        std::string out;
        out.reserve(256 + nodes_.size() * 128 + edges_.size() * 64);

        out += "{\n  \"nodes\": [\n";
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& n = nodes_[i];
            out += "    {\"id\": ";
            out += std::to_string(i);
            out += ", \"file\": \"";
            out += escape_json(n.id.file);
            out += "\", \"line\": ";
            out += std::to_string(n.id.line);
            out += ", \"type\": \"";
            out += escape_json(n.type_name);
            out += "\"}";
            if (i + 1 < nodes_.size())
                out += ",";
            out += "\n";
        }
        out += "  ],\n  \"edges\": [\n";
        for (std::size_t i = 0; i < edges_.size(); ++i) {
            const auto& e = edges_[i];
            out += "    {\"from\": ";
            out += std::to_string(e.from);
            out += ", \"to\": ";
            out += std::to_string(e.to);
            out += ", \"type\": \"";
            out += escape_json(e.type_name);
            out += "\"}";
            if (i + 1 < edges_.size())
                out += ",";
            out += "\n";
        }
        out += "  ]\n}\n";
        return out;
    }

    // ========================================================================
    // Export: callback-based
    // ========================================================================

    template <typename NodeSink, typename EdgeSink>
    void export_to(NodeSink&& node_sink, EdgeSink&& edge_sink) const {
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            node_sink(i, nodes_[i]);
        }
        for (const auto& e : edges_) {
            edge_sink(e);
        }
    }

private:
    [[nodiscard]] static std::string_view extract_filename(const std::string_view path) {
        if (path.empty()) {
            return "";
        }
        if (const auto pos = path.rfind('/'); pos != std::string_view::npos) {
            return path.substr(pos + 1);
        }
        if (const auto pos = path.rfind('\\'); pos != std::string_view::npos) {
            return path.substr(pos + 1);
        }
        return path;
    }

    [[nodiscard]] static std::string escape_json(const std::string_view s) {
        std::string out;
        out.reserve(s.size());
        for (const char c : s) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
            }
        }
        return out;
    }

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::unordered_map<NodeId, std::size_t, NodeIdHash> node_map_;
};

// ============================================================================
// Global graph instance (only when enabled)
// ============================================================================

#ifdef ENABLE_FLOW_GRAPH
[[nodiscard]] inline FlowGraph& global_flow_graph() noexcept {
    static FlowGraph g;
    return g;
}

inline void register_flow_edge(
    const NodeId& producer, const NodeId& consumer, const std::string_view type) {
    global_flow_graph().add_edge(producer, consumer, type);
}
#else
inline void register_flow_edge(const NodeId&, const NodeId&, std::string_view) noexcept {}
#endif

// ============================================================================
// Conditional storage: zero-cost when disabled (empty base optimization)
// ============================================================================

namespace detail {

#ifdef ENABLE_FLOW_GRAPH
struct LocStorage {
    NodeId loc_;

    constexpr LocStorage() noexcept = default;
    constexpr explicit LocStorage(const std::string_view file, const int line) noexcept
        : loc_{file, line} {}

    [[nodiscard]] constexpr NodeId loc() const noexcept { return loc_; }
};
#else
struct LocStorage {
    constexpr LocStorage() noexcept = default;
    constexpr explicit LocStorage(const std::string_view, int) noexcept {}

    [[nodiscard]] static constexpr NodeId loc() noexcept { return {}; }
};
#endif

} // namespace detail

} // namespace primitive
