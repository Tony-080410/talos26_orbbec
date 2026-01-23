#pragma once

#include "timer_constants.hpp"
#include "timer_system.hpp"
#include <functional>
#include <memory>

namespace talos::rclcompat {

/**
 * @brief Create a timer system (compile-time frequency dispatch)
 *
 * Dispatches to the appropriate RclTimerSystem template specialization
 * based on the Frequency enum value.
 *
 * ## Supported Frequencies
 *
 * **Low frequency monitoring:** 1, 2, 5 Hz
 *
 * **Standard control:** 10, 20, 27 Hz
 *
 * **Common sensors/cameras:** 30, 50, 60, 100 Hz
 *
 * **High-frequency sensors:** 120, 150, 200, 250, 500, 1000 Hz
 *
 * ## Parameters
 *
 * - `name`: System name for identification
 * - `frequency`: Timer frequency from Frequency enum
 * - `callback`: Function to call at each timer tick
 *
 * ## Returns
 *
 * Unique pointer to the timer system (type-erased as SystemBase)
 */
[[nodiscard]] std::unique_ptr<SystemBase>
    create_timer_system(std::string name, Frequency frequency, std::function<void()> callback);

} // namespace talos::rclcompat
