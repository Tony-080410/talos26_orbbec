#pragma once

#include <cstdint>

namespace talos::scheduler::rclcompat {

// ============================================================================
// X-Macro: List of supported timer frequencies
// ============================================================================
// Used to generate:
// - Frequency enum values
// - Timer factory switch cases
// ============================================================================
#define TALOS_TIMER_FREQUENCIES       \
    X(1, "Low frequency monitoring")  \
    X(2, "Low frequency monitoring")  \
    X(5, "Low frequency monitoring")  \
    X(10, "Standard control")         \
    X(20, "Standard control")         \
    X(27, "Standard control (~37ms)") \
    X(30, "Sensors/cameras")          \
    X(50, "Sensors/cameras")          \
    X(60, "Sensors/cameras")          \
    X(100, "Sensors/cameras")         \
    X(120, "High-frequency")          \
    X(150, "High-frequency")          \
    X(200, "High-frequency")          \
    X(250, "High-frequency")          \
    X(500, "High-frequency")          \
    X(1000, "High-frequency")

/**
 * @brief Supported timer frequencies
 *
 * Pre-defined timer frequencies for create_wall_timer.
 * Only these frequencies are allowed - no custom frequencies.
 */
enum class Frequency : std::uint32_t {
#define X(freq, desc) Hz_##freq = (freq),
    TALOS_TIMER_FREQUENCIES
#undef X
};

[[nodiscard]] constexpr std::uint32_t to_uint32(Frequency freq) noexcept {
    return static_cast<std::uint32_t>(freq);
}

} // namespace talos::scheduler::rclcompat
