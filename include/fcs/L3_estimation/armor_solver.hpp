#pragma once

/// @file armor_solver.hpp
/// @brief Aggregation header for armor solver module
///
/// This module provides armor tracking and gimbal solving functionality:
/// - ISRCKF-based tracking for 4-armor robots and 3-armor outposts
/// - Ballistic compensation for precise aiming
/// - ECS system integration for talos scheduler
///
/// Usage:
/// @code
/// #include "fcs/L3_estimation/armor_solver.hpp"
///
/// // Create tracker and solver
/// fcs::L3::Tracker tracker(config.tracker);
/// fcs::L3::Solver solver(config.solver);
///
/// // Insert as resources
/// world.insert_resource<fcs::L3::Tracker>(tracker);
/// world.insert_resource<fcs::L3::Solver>(solver);
///
/// // Register ECS systems
/// fcs::L3::register_armor_solver_systems(app);
/// @endcode

// Core types
#include "armor_solver/config.hpp"
#include "armor_solver/types.hpp"

// Math utilities
#include "armor_solver/util.hpp"

// Kalman filter implementations
#include "armor_solver/isrckf.hpp"
#include "armor_solver/srukf.hpp"

// Motion models
#include "armor_solver/motion_model.hpp"

// Data association
#include "armor_solver/data_associator.hpp"

// Tracker
#include "armor_solver/tracker.hpp"

// Trajectory compensation
#include "armor_solver/trajectory_compensator.hpp"

// Solver
#include "armor_solver/solver.hpp"

// ECS system registration
#include "armor_solver_systems.hpp"
