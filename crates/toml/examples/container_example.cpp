// Example usage of container type specializations in toml_helper
//
// This example demonstrates how to use:
// - std::optional<T>
// - std::vector<T>
// - std::array<T, N>

#include <array>
#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <vector>

#include "toml/core.hpp"
#include "toml/ext/containers.hpp"

// Example config structure using all three container types
struct RobotConfig {
    // std::optional: May or may not be present in config
    std::optional<std::string> team_color;
    std::optional<double> max_speed;

    // std::vector: Dynamic-sized list
    std::vector<int> camera_ids;
    std::vector<std::string> allies;

    // std::array: Fixed-size list with validation
    std::array<double, 3> position;    // x, y, z
    std::array<double, 4> orientation; // quaternion w, x, y, z

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, team_color);
        READ_OPT(tbl, max_speed);
        READ(tbl, camera_ids);
        READ(tbl, allies);
        READ(tbl, position);
        READ(tbl, orientation);

        return {};
    }
};

int main() {
    // Example TOML config (as string for demonstration)
    constexpr std::string_view config_toml = R"(
        # Optional fields (can be omitted)
        team_color = "blue"
        # max_speed is omitted → std::nullopt

        # Dynamic arrays
        camera_ids = [0, 1, 2]
        allies = ["robot1", "robot2"]

        # Fixed-size arrays (must have exact length)
        position = [1.0, 2.0, 3.0]
        orientation = [1.0, 0.0, 0.0, 0.0]
    )";

    // Parse TOML
    auto result = toml::parse(config_toml);
    if (!result) {
        std::cerr << "Failed to parse TOML\n";
        return 1;
    }

    const toml::table& tbl = result.table();

    // Deserialize into config struct
    RobotConfig config;
    auto res = config.read_from(tbl);
    if (!res) {
        std::cerr << fmt::format("Failed to read config: {}\n", res.error());
        return 1;
    }

    // Print results
    std::cout << "=== Robot Configuration ===\n";

    // std::optional demonstration
    if (config.team_color) {
        std::cout << fmt::format("Team Color: {}\n", *config.team_color);
    } else {
        std::cout << "Team Color: not specified\n";
    }

    if (config.max_speed) {
        std::cout << fmt::format("Max Speed: {} m/s\n", *config.max_speed);
    } else {
        std::cout << "Max Speed: not specified (using default)\n";
    }

    // std::vector demonstration
    std::cout << fmt::format("Camera IDs: [{}]\n", fmt::join(config.camera_ids, ", "));
    std::cout << fmt::format("Allies: [{}]\n", fmt::join(config.allies, ", "));

    // std::array demonstration
    std::cout << fmt::format(
        "Position: [{}, {}, {}]\n", config.position[0], config.position[1], config.position[2]);
    std::cout << fmt::format(
        "Orientation: [{}, {}, {}, {}]\n", config.orientation[0], config.orientation[1],
        config.orientation[2], config.orientation[3]);

    return 0;
}
