#pragma once

namespace talos {
class Scheduler;
}

namespace fcs::rune {

struct RuneDetectorConfig;

void register_rune_detection_systems(talos::Scheduler& scheduler, RuneDetectorConfig&& config);

} // namespace fcs::rune
