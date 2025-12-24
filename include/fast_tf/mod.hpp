#pragma once

/// @file mod.hpp
/// @brief Fast TF module entry point - includes all public APIs

// Core types and concepts
#include "concepts.hpp"
#include "hash.hpp"
#include "into.hpp"
#include "types.hpp"

// Error handling
#include "error.hpp"
#include "result.hpp"

// Transform types
#include "matrix.hpp"
#include "transform_data.hpp"

// Frame system
#include "frame_id.hpp"
#include "frame_path.hpp"
#include "frame_registry.hpp"

// Query builder
#include "query.hpp"

// Buffer and tree
#include "buffer.hpp"
#include "tree.hpp"

// Static transforms
#include "static_tf.hpp"
