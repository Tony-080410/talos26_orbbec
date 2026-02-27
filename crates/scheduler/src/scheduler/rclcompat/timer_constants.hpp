#pragma once

#include <cstdint>

namespace talos::scheduler::rclcompat {

/**
 * @brief Supported timer frequencies
 *
 * Pre-defined timer frequencies for create_wall_timer.
 * Only these frequencies are allowed - no custom frequencies.
 */
enum class Frequency : std::uint32_t {
    Hz_1    = 1,   ///< Low frequency monitoring
    Hz_2    = 2,   ///< Low frequency monitoring
    Hz_5    = 5,   ///< Low frequency monitoring
    Hz_10   = 10,  ///< Standard control
    Hz_20   = 20,  ///< Standard control
    Hz_27   = 27,  ///< Standard control (~37ms period)
    Hz_30   = 30,  ///< Sensors/cameras
    Hz_50   = 50,  ///< Sensors/cameras
    Hz_60   = 60,  ///< Sensors/cameras
    Hz_100  = 100, ///< Sensors/cameras
    Hz_120  = 120, ///< High-frequency
    Hz_150  = 150, ///< High-frequency
    Hz_200  = 200, ///< High-frequency
    Hz_250  = 250, ///< High-frequency
    Hz_500  = 500, ///< High-frequency
    Hz_1000 = 1000 ///< High-frequency
};

[[nodiscard]] constexpr std::uint32_t to_uint32(Frequency freq) noexcept {
    return static_cast<std::uint32_t>(freq);
}

} // namespace talos::scheduler::rclcompat
