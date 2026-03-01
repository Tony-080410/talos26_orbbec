#include "scheduler/rclcompat/timer_factory.hpp"

#include "scheduler/rclcompat/timer_constants.hpp"
#include "timer_system.hpp"
#include <utility>

namespace talos::scheduler::rclcompat {

std::unique_ptr<SystemBase> create_timer_system(
    std::string&& name, const Frequency frequency, std::function<void()> callback) {
    switch (frequency) {
#define X(freq, desc)          \
    case Frequency::Hz_##freq: \
        return std::make_unique<RclTimerSystem<(freq)>>(name, std::move(callback));
        TALOS_TIMER_FREQUENCIES
#undef X
    }
    std::unreachable();
}

} // namespace talos::scheduler::rclcompat
