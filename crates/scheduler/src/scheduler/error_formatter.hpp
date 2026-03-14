#pragma once

/**
 * @file fmt::formatter specializations for scheduler error types
 *
 * Enables all scheduler error types to be formatted with fmt library.
 * This allows errors to be used directly with fmt::format(), spdlog, etc.
 *
 * Example:
 *   fmt::print("Error: {}\n", error);
 *   SPDLOG_ERROR("Build failed: {}", error);
 */

#include "scheduler.hpp"

#include <fmt/format.h>
#include <fmt/std.h> // For std::variant, std::vector, etc.
#include <magic_enum.hpp>
#include <span>
#include <string>

// ============================================================================
// Helper: join strings with separator
// ============================================================================

namespace talos::scheduler::detail {

inline std::string join_strings(std::span<const std::string> strs, std::string_view sep) {
    if (strs.empty())
        return "";
    std::string result = strs[0];
    for (size_t i = 1; i < strs.size(); ++i) {
        result += sep;
        result += strs[i];
    }
    return result;
}

} // namespace talos::scheduler::detail

// ============================================================================
// fmt::formatter specializations - must be in fmt namespace
// ============================================================================

namespace fmt {
using namespace talos::scheduler;
using namespace talos::scheduler::detail;

// Formatter for simple enum types
template <>
struct formatter<SchedulerError> : formatter<std::string_view> {
    auto format(const SchedulerError e, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(e), ctx);
    }
};

// Formatter base class (eliminates duplicate parse methods)
template <typename T>
struct error_formatter_base {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Formatter for errors with two system names
template <>
struct formatter<ChannelKindConflict> : error_formatter_base<ChannelKindConflict> {
    auto format(const ChannelKindConflict& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "ChannelKindConflict: channel used as both SPSC and SPMC by '{}' and '{}'",
            e.first_system, e.second_system);
    }
};

// Formatter for errors with string lists
template <>
struct formatter<MultipleWritersError> : error_formatter_base<MultipleWritersError> {
    auto format(const MultipleWritersError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "MultipleWriters: channel has multiple writers: {}",
            join_strings(e.writers, ", "));
    }
};

template <>
struct formatter<MultipleReadersError> : error_formatter_base<MultipleReadersError> {
    auto format(const MultipleReadersError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "MultipleReaders: SPSC channel has multiple readers: {}",
            join_strings(e.readers, ", "));
    }
};

template <>
struct formatter<OrphanedReaderError> : error_formatter_base<OrphanedReaderError> {
    auto format(const OrphanedReaderError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "OrphanedReader: reader(s) without writer: {}",
            join_strings(e.readers, ", "));
    }
};

template <>
struct formatter<DependencyCycleError> : error_formatter_base<DependencyCycleError> {
    auto format(const DependencyCycleError& e, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "DependencyCycle: {}", join_strings(e.cycle, " -> "));
    }
};

template <>
struct formatter<TooManyComputeSystemsError> : error_formatter_base<TooManyComputeSystemsError> {
    auto format(const TooManyComputeSystemsError& e, format_context& ctx) const {
        return format_to(ctx.out(), "TooManyComputeSystems: {} (max {})", e.count, e.max_count);
    }
};

template <>
struct formatter<UnreachableComputeSystemsError>
    : error_formatter_base<UnreachableComputeSystemsError> {
    auto format(const UnreachableComputeSystemsError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "UnreachableComputeSystems: {}", join_strings(e.systems, ", "));
    }
};

template <>
struct formatter<BuildError> : error_formatter_base<BuildError> {
    auto format(const BuildError& err, format_context& ctx) const {
        return std::visit(
            [&ctx](const auto& e) { return fmt::format_to(ctx.out(), "{}", e); }, err);
    }
};

} // namespace fmt
