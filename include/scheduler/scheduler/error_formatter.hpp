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

#include "scheduler/scheduler/scheduler.hpp"

#include <fmt/format.h>
#include <fmt/std.h> // For std::variant, std::vector, etc.

// ============================================================================
// Helper: join strings with separator
// ============================================================================

namespace talos::detail {

inline std::string join_strings(const std::vector<std::string>& strs, std::string_view sep) {
    if (strs.empty())
        return "";
    std::string result = strs[0];
    for (size_t i = 1; i < strs.size(); ++i) {
        result += sep;
        result += strs[i];
    }
    return result;
}

} // namespace talos::detail

// ============================================================================
// fmt::formatter specializations - must be in fmt namespace
// ============================================================================

namespace fmt {

template <>
struct formatter<talos::SchedulerError> : formatter<std::string_view> {
    auto format(const talos::SchedulerError e, format_context& ctx) const {
        return formatter<std::string_view>::format(talos::to_string(e), ctx);
    }
};

template <>
struct formatter<talos::ChannelKindConflict> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos::ChannelKindConflict& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "ChannelKindConflict: channel used as both SPSC and SPMC by '{}' and '{}'",
            e.first_system, e.second_system);
    }
};

template <>
struct formatter<talos::MultipleWritersError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos::MultipleWritersError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "MultipleWriters: channel has multiple writers: {}",
            talos::detail::join_strings(e.writers, ", "));
    }
};

template <>
struct formatter<talos::MultipleReadersError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos::MultipleReadersError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "MultipleReaders: SPSC channel has multiple readers: {}",
            talos::detail::join_strings(e.readers, ", "));
    }
};

template <>
struct formatter<talos::OrphanedReaderError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos::OrphanedReaderError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "OrphanedReader: reader(s) without writer: {}",
            talos::detail::join_strings(e.readers, ", "));
    }
};

template <>
struct formatter<talos::DependencyCycleError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos::DependencyCycleError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "DependencyCycle: {}", talos::detail::join_strings(e.cycle, " -> "));
    }
};

template <>
struct formatter<talos::TooManyComputeSystemsError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos::TooManyComputeSystemsError& e, format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "TooManyComputeSystems: {} (max {})", e.count, e.max_count);
    }
};

template <>
struct formatter<talos::BuildError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos::BuildError& err, format_context& ctx) const {
        return std::visit(
            [&ctx](const auto& e) { return fmt::format_to(ctx.out(), "{}", e); }, err);
    }
};

} // namespace fmt
