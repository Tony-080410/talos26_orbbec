#pragma once

#include "error.hpp"
#include "frame_path.hpp"
#include "result.hpp"
#include "types.hpp"
#include <expected>

namespace fast_tf {

// ============================================================================
// Query Builder - Compile-Time Type-Safe Transform Queries
// ============================================================================

// Forward declaration
template <typename TreeStructure>
class TransformTree;

/// Second stage of query builder - To frame selected
template <typename From, typename To, typename TreeImpl>
class TransformQueryTo {
public:
    explicit TransformQueryTo(TreeImpl& tree)
        : tree_(tree) {}

    /// Query transform at specific timestamp
    [[nodiscard]] std::expected<TransformResult, TransformError> at(timestamp_ns_t ts) const {
        return tree_.template lookup_impl<From, To>(ts);
    }

    /// Query latest available transform
    [[nodiscard]] std::expected<TransformResult, TransformError> latest() const {
        return tree_.template lookup_latest_impl<From, To>();
    }

private:
    TreeImpl& tree_;
};

/// First stage of query builder - From frame selected
template <typename From, typename TreeImpl>
class TransformQueryFrom {
public:
    explicit TransformQueryFrom(TreeImpl& tree)
        : tree_(tree) {}

    /// Select target frame (To)
    template <typename To>
    requires validate_frames_same_tree<From, To>::value
    [[nodiscard]] TransformQueryTo<From, To, TreeImpl> to() {
        return TransformQueryTo<From, To, TreeImpl>(tree_);
    }

private:
    TreeImpl& tree_;
};

} // namespace fast_tf
