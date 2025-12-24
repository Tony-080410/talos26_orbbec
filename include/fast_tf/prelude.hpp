#pragma once

/// @file prelude.hpp
/// @brief Fast TF prelude - commonly used types for quick import

#include "concepts.hpp"
#include "matrix.hpp"
#include "transform_data.hpp"
#include "tree.hpp"
#include "types.hpp"

namespace fast_tf {

// Re-export commonly used names at namespace level
using TransformMatrix = TransformMatrix4d;
using RPY             = RPYTransformData;
using Quaternion      = QuaternionTransformData;

} // namespace fast_tf
