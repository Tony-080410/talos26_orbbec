#pragma once

// Re-export existing TripleBuffer from spike namespace
#include "../../triple_buffer.hpp"

namespace scheduler {

// 使用现有的高质量 TripleBuffer 实现
template<typename T>
using TripleBuffer = spike::TripleBuffer<T>;

} // namespace scheduler
