#include "spdlog_hook.hpp"
#include "talos_gimbal/stm32.hpp"

int main() {
    auto device = talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>{};
    if (const auto result = device.connect(0x0483, 0x5740); !result) {
        spdlog::critical("fuck error {}", talos_gimbal::to_string(result.error()));
        return 1;
    }
    while (true) {
        device.handle_events();
        // std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}
