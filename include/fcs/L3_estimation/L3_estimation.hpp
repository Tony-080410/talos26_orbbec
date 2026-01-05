#pragma once

/// @file L3_estimation.hpp
/// @brief Aggregation header for L3 estimation module
///
/// This module provides robot state estimation functionality:
/// - ISRCKF-based tracking for 4-armor robots and 3-armor outposts
/// - State estimation outputs for higher layers
/// - ECS system integration for talos scheduler
///
/// Usage:
/// @code
/// #include "fcs/L3_estimation/L3_estimation.hpp"
///
/// // Create tracker
/// fcs::L3::Tracker tracker(config.tracker);
///
/// // Insert as resources
/// world.insert_resource<fcs::L3::Tracker>(tracker);
///
/// // Register ECS systems
/// fcs::L3::register_tracker_systems(app);
/// @endcode

// Core types
#include "tracker/config.hpp"
#include "tracker/types.hpp"

// Math utilities
#include "tracker/util.hpp"

// Kalman filter implementations
#include "tracker/isrckf.hpp"
#include "tracker/srukf.hpp"

// Motion models
#include "tracker/motion_model.hpp"

// Data association
#include "tracker/data_associator.hpp"

// Tracker
#include "tracker/tracker.hpp"

// ECS system registration
#include "tracker_systems.hpp"