#include "fcs/runtime/boot.hpp"
#include "fcs_visualization/foxglove_systems.hpp"
#include "scheduler/error_formatter.hpp"
#include "scheduler/scheduler.hpp"
#include "spdlog_hook.hpp"

#include <chrono>

#ifndef TALOS_DISABLE_PROGRAM_MAIN
int main() {
    init_logger();
    SPDLOG_INFO("talos is initializing...");
    const auto start = std::chrono::system_clock::now();

    auto config = fcs::load_config("at_vision.toml");
    if (!config) {
        SPDLOG_CRITICAL("{}", config.error());
        return 1;
    }

    // Extract foxglove config before boot() consumes the rest
    // (foxglove setup requires fcs_visualization, which is not linked by the fcs library)
    auto foxglove_cfg = std::move(config->foxglove);

    talos::World world;
    talos::Scheduler scheduler(world);

    // Foxglove visualization setup (separate library dependency)
    if (foxglove_cfg.enabled) {
        auto server = fcs::try_create_foxglove_server(foxglove_cfg.port, foxglove_cfg.host);
        if (server) {
            world.insert_resource(std::move(server));
            fcs::register_foxglove_systems(scheduler, &scheduler);
            SPDLOG_INFO(
                "Foxglove server enabled on ws://{}:{}", foxglove_cfg.host, foxglove_cfg.port);
        } else {
            SPDLOG_WARN("Foxglove server failed to initialize, visualization disabled");
        }
    } else {
        SPDLOG_INFO("Foxglove visualization disabled by config");
    }

    if (auto r = fcs::boot(world, scheduler, std::move(*config)); !r) {
        SPDLOG_CRITICAL("{}", r.error());
        return 1;
    }

    SPDLOG_INFO(
        "init done. ({}ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now() - start)
                                 .count());

    if (const auto result = scheduler.run(); !result) {
        SPDLOG_CRITICAL("run scheduler: {}", result.error());
    }
}
#endif
