#define ENABLE_FLOW_GRAPH 1

#include "../include/math/euler.hpp"

#include "auto_aim/config.hpp"

#include "fcs/L1_sensor/input_interface.hpp"
#include "math/fast_tf/infantry.hpp"
#include "primitives/channel.hpp"
#include "primitives/flow_graph.hpp"
#include "spdlog_hook.hpp"
#include "talos_gimbal/stm32.hpp"

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <tbb/parallel_pipeline.h>

// ============================================================================
// Config
// ============================================================================

struct TalosConfig : SpVisionConfig {
    bool daedalus;
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (auto read = SpVisionConfig::read_from(tbl); !read) {
            return read;
        }
        if (const auto _t = tbl["runtime"].as_table()) {
            auto t = *_t;
            READ_KEY(t, daedalus);
            return {};
        }
        return std::unexpected("runtime section not found");
    }
};

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using read_config_error = std::variant<toml::parse_error, std::string>;
using namespace std::chrono_literals;

inline std::expected<TalosConfig, read_config_error> read_config() {
    TalosConfig config;
    toml::table tbl;
    try {
        tbl = toml::parse_file("sp_vision.toml");
    } catch (const toml::parse_error& err) {
        return std::unexpected(err);
    }
    if (auto result = config.read_from(tbl); !result) {
        return std::unexpected(result.error());
    }
    return config;
}

static uint64_t now_ns() {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    init_logger();
    spdlog::info("talos is initializing...");

    auto ch      = primitive::make_spmc_channel<uint64_t>();
    auto reader  = ch.clone_reader();
    auto reader2 = ch.clone_reader();
    auto reader3 = ch.clone_reader();
    auto ch2     = primitive::make_spsc_channel<uint64_t>();

    ch2.take_reader();

    const auto dot = primitive::global_flow_graph().to_dot("TestGraph");
    spdlog::info(dot);
    spdlog::info("pipeline stopped");
    return 0;
}