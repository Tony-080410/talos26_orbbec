#pragma once

// ============================================================================
// Talos ECS - Bevy-style Entity Component System
//
// Modular headers:
//   - ecs/execution_policy.hpp : Scheduling policies (exclusive, pool)
//   - ecs/channel_types.hpp    : Channel wrappers (spsc, spmc, res, atomic)
//   - ecs/system_meta.hpp      : System metadata and function traits
//   - ecs/world.hpp            : World container (resources, channels)
//   - ecs/system.hpp           : System base class and implementation
//   - ecs/app.hpp              : App builder
//
// Usage:
//   #include "scheduler/bevy_ecs.hpp"
//
//   void camera_system(talos::spsc_mut<Frame, CameraPipe> out) {
//       out.write(Frame{...});
//   }
//
//   void detector_system(talos::spsc<Frame, CameraPipe> in,
//                        talos::res<Config> cfg,
//                        talos::spsc_mut<Detection, DetectorPipe> out) {
//       if (auto f = in.read()) {
//           out.write(Detection{...});
//       }
//   }
//
//   int main() {
//       talos::App app;
//       app.add_resource(Config{...})
//          .add_system<talos::exclusive<30>>("camera", camera_system)
//          .add_system<talos::pool_compute>("detector", detector_system);
//       app.run_once();
//   }
// ============================================================================

#include "ecs/app.hpp"
#include "ecs/channel_types.hpp"
#include "ecs/execution_policy.hpp"
#include "ecs/system.hpp"
#include "ecs/system_meta.hpp"
#include "ecs/world.hpp"
