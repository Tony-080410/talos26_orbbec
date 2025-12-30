#pragma once

/// @file at_legacy.hpp
/// @brief AT Legacy Armor Detection Module (migrated from at_vision ROS2 project)
///
/// This module provides a complete armor detection pipeline:
///   - Image preprocessing
///   - Armor detection (Traditional or NN backend)
///   - PnP pose estimation
///   - Scheduler system integration
///
/// Usage example:
/// @code
///   #include <fcs/L2_perception/at_legacy.hpp>
///
///   // Create backend
///   auto backend = fcs::create_traditional_backend(160);
///
///   // Create PnP solver
///   std::array<double, 9> camera_matrix = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
///   std::vector<double> dist_coeffs = {k1, k2, p1, p2, k3};
///   auto solver = fcs::create_pnp_solver(camera_matrix, dist_coeffs);
///
///   // Setup scheduler
///   auto app = talos::App::create();
///   app.insert_resource<fcs::AtLegacyDetectorBackend>(std::move(backend));
///   app.insert_resource<fcs::AtLegacyPnPSolver>(std::move(solver));
///
///   app.init_spmc_channel<fcs::AtLegacyImageFrame, fcs::AtLegacyImageChannelTag>();
///   app.init_spmc_channel<fcs::AtLegacyArmorDetectionBatch, fcs::AtLegacyDetectionChannelTag>();
///   app.init_spmc_channel<fcs::AtLegacyArmorMeasurementBatch,
///   fcs::AtLegacyMeasurementChannelTag>();
///
///   fcs::register_at_legacy_detection_systems(app);
/// @endcode

// Core types
#include "../core/channel_tags.hpp"
#include "../core/types.hpp"

// Detection backends
#include "at_legacy_detector_backend.hpp"
#include "at_legacy_traditional_backend.hpp"
#include "at_legacy_traditional_types.hpp"
// #include "at_legacy_nn_backend.hpp"  // Requires OpenVINO, include separately

// PnP solver
#include "at_legacy_pnp_solver.hpp"

// Configuration
#include "at_legacy_config.hpp"

// Scheduler systems
#include "at_legacy_systems.hpp"
