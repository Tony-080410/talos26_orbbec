---
name: fast-tf
description: Generate boilerplate code for talos fast_tf coordinate system and math utilities. Use when creating new coordinate frames, type-safe coordinate wrappers (Fuck types), or robot coordinate systems.
---

# Talos Fast TF Code Generator

## Instructions

When the user wants to add new fast_tf/math components, follow these steps:

### 1. Ask what to generate

Ask the user which component they need:
- **Frame**: New coordinate frame in the transform tree
- **Fuck Type Alias**: Type-safe coordinate wrapper
- **Robot Coordinate System**: Factory function for robot-specific transforms
- **Euler Rotation**: Custom rotation type or conversion

### 2. Gather parameters interactively

**For Frame:**
- Frame name (e.g., `lidar_link`, `imu_link`)
- Parent frame (must exist in tree)
- frame_id string (if different from name)

**For Fuck Type Alias:**
- Alias name
- Coordinate frame
- Inner type (e.g., `Eigen::Vector3d`, `double`)

**For Robot Coordinate System:**
- Robot type name
- Frame tree structure
- Buffer size

### 3. Generate code and ask where to place it

## Current Frame Tree

```
world (root)
└── odom
    └── gimbal_link
        └── camera_link
            ├── camera_optical_frame
            └── muzzle_link
```

## Templates

### Frame Declaration Template

Add to `include/math/fast_tf/frame.hpp` before `#undef`:

```cpp
DECL(${NAME}, ${PARENT});
```

For custom frame_id (different from name):
```cpp
struct ${NAME}_fuxk_frame {
    static constexpr std::string_view frame_id = "${FRAME_ID}";
    static constexpr size_t index              = __COUNTER__ - _base - 1;
    using ancestor                             = ${PARENT}_fuxk_frame;
};
```

### Fuck Type Alias Template

```cpp
template <frame Coordinate>
using Fuck${NAME} = Fuck<Coordinate, ${INNER_TYPE}>;
```

Common patterns:
```cpp
using FuckVec3d = Fuck<Coordinate, Eigen::Vector3d>;
using FuckPose = Fuck<Coordinate, Eigen::Isometry3d>;
```

### Robot Coordinate System Factory

```cpp
/// @brief Create coordinate system for ${ROBOT_TYPE}
/// @tparam N Buffer size for interpolation
template <std::size_t N = 64>
[[nodiscard]] constexpr auto ${ROBOT_TYPE}_coordinate_system() -> CoordinateSystem<N> {
    return CoordinateSystem<N>{};
}
```

### Euler Rotation Usage

```cpp
// From angles
auto rot = math_fuxk::rpy(roll, pitch, yaw);

// To quaternion
Eigen::Quaterniond q = rot.quat();

// To rotation matrix
Eigen::Matrix3d m = rot.matrix();

// From matrix
auto rot2 = math_fuxk::rpy(matrix);

// From quaternion
auto rot3 = math_fuxk::rpy(quaternion);
```

## Type Reference

| Type | Description |
|------|-------------|
| `Fuck<Coordinate, T>` | Type-safe wrapper, compile-time coordinate check |
| `FuckTransform<Coordinate>` | `Fuck<Coordinate, TransformMatrix4d>` |
| `FuckDouble<Coordinate>` | `Fuck<Coordinate, double>` |
| `FuckIntrinsicEulerRotd<Coordinate>` | `Fuck<Coordinate, Ros2EulerRotd>` |
| `CoordinateSystem<N>` | Array of CircularTransformBuffer |

## API Reference

### Update transform
```cpp
// Update transform from child to parent
update<${CHILD}_fuxk_frame>(system, FuckTransform<${CHILD}_fuxk_frame>{matrix}, timestamp_ns);
```

### Lookup transform
```cpp
// Lookup transform, converting from source frame to target frame
auto result = lookup<${TARGET}_fuxk_frame, ${SOURCE}_fuxk_frame>(
    system,
    source_transform,
    timestamp_ns
);
if (result) {
    auto& tf = *result;
    // use tf
}
```

## Examples

### Example 1: Add lidar frame

In `frame.hpp`:
```cpp
DECL(lidar_link, gimbal_link);
```

Usage:
```cpp
// Update lidar transform
update<lidar_link_fuxk_frame>(
    coord_system,
    FuckTransform<lidar_link_fuxk_frame>{lidar_to_gimbal_matrix},
    timestamp
);

// Lookup lidar pose in world frame
auto world_pose = lookup<world_fuxk_frame, lidar_link_fuxk_frame>(
    coord_system,
    FuckTransform<lidar_link_fuxk_frame>{identity},
    timestamp
);
```

### Example 2: Add custom Fuck type for velocity

```cpp
template <frame Coordinate>
using FuckVelocity = Fuck<Coordinate, Eigen::Vector3d>;

// Usage
FuckVelocity<camera_link_fuxk_frame> camera_vel{Eigen::Vector3d{1.0, 0.0, 0.0}};
```

### Example 3: Coordinate transform chain

```cpp
// Transform point from camera_optical to world
auto point_camera = FuckTransform<camera_optical_fuxk_frame>{point_matrix};

auto point_world = lookup<world_fuxk_frame, camera_optical_fuxk_frame>(
    system, point_camera, now_ns()
);

if (point_world) {
    Eigen::Vector3d world_pos = (*point_world)->translation();
}
```

## Frame Concept Constraints

```cpp
// Any frame
template <frame T> void foo();

// Only root frame (no parent)
template <root_frame T> void bar();

// Only non-root frame (has parent)
template <non_root_frame T> void baz();
```
