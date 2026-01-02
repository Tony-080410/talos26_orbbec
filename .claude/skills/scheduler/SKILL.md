---
name: scheduler
description: Generate boilerplate code for talos scheduler ECS system. Use when creating new Channel Tags, data types for channels, or System definitions (exclusive/pool_compute).
---

# Talos Scheduler Code Generator

## Instructions

When the user wants to add new scheduler components, follow these steps:

### 1. Ask what to generate

Ask the user which component they need:

- **Channel Tag**: Empty struct for channel identification
- **Data Type**: POD struct for channel transmission
- **System**: Lambda-based system with execution policy

### 2. Gather parameters interactively

**For Channel Tag:**

- Tag name (e.g., `ImageData`, `DetectionResult`)
- Brief description

**For Data Type:**

- Struct name
- Fields (name, type)
- Whether to add timestamp field

**For System:**

- System name
- Execution policy: `exclusive<Hz>` or `pool_compute`
- Input channels (type + tag)
- Output channels (type + tag)
- Resources needed

### 3. Generate code and ask where to place it

## Templates

### Channel Tag Template

```cpp
/// @brief ${DESCRIPTION}
struct ${NAME}Tag {};
```

**Location**: Usually in `include/scheduler/ecs/channel_types.hpp` or a dedicated tags header.

### Data Type Template

```cpp
/// @brief ${DESCRIPTION}
struct ${NAME} {
    ${FIELDS}
    // timestamp_ns_t timestamp{};  // optional
};
```

**Location**: Domain-specific header or `include/types/`.

### System Template (Exclusive)

```cpp
// ${NAME} system - runs at ${HZ}Hz on dedicated thread
app.add_system<exclusive<${HZ}>>(
    "${NAME}",
    [](${PARAMS}) {
        // TODO: implement
    });
```

### System Template (Pool Compute)

```cpp
// ${NAME} system - runs in thread pool
app.add_system<pool_compute>(
    "${NAME}",
    [](${PARAMS}) {
        // TODO: implement
    });
```

## Channel Type Quick Reference

| Type                 | Usage                        | Example                        |
|----------------------|------------------------------|--------------------------------|
| `spsc<T, Tag>`       | Single reader from SPSC      | `spsc<Frame, CameraTag>`       |
| `spsc_mut<T, Tag>`   | Single writer to SPSC        | `spsc_mut<Frame, CameraTag>`   |
| `spmc<T, Tag>`       | Reader from SPMC (broadcast) | `spmc<Pose, OdomTag>`          |
| `spmc_mut<T, Tag>`   | Writer to SPMC               | `spmc_mut<Pose, OdomTag>`      |
| `res<T>`             | Read-only resource           | `res<Config>`                  |
| `res_mut<T>`         | Mutable resource             | `res_mut<State>`               |
| `atomic_ref<T, Tag>` | Atomic value access          | `atomic_ref<bool, RunningTag>` |

## Execution Policy Reference

| Policy                 | Thread Model                    | Use Case                      |
|------------------------|---------------------------------|-------------------------------|
| `exclusive<Hz>`        | Dedicated thread, fixed rate    | Sensor polling, control loops |
| `exclusive_silent<Hz>` | Same, no logging                | High-frequency loops          |
| `pool_compute`         | Thread pool, triggered by input | Processing pipelines          |

## Examples

### Example 1: Add image processing pipeline

**Tags:**

```cpp
struct RawImageTag {};
struct ProcessedImageTag {};
```

**System:**

```cpp
app.add_system<pool_compute>(
    "image_processor",
    [](spsc<cv::Mat, RawImageTag> input,
       spsc_mut<cv::Mat, ProcessedImageTag> output) {
        if (auto frame = input.read()) {
            cv::Mat processed;
            cv::cvtColor(*frame, processed, cv::COLOR_BGR2GRAY);
            output.write(std::move(processed));
        }
    });
```

### Example 2: Add periodic status publisher

```cpp
StatusMsg msg{
        .cpu_usage = state->cpu_usage,
        .memory_usage = state->memory_usage,
        .timestamp = now_ns()
    };
app.add_system<exclusive<10>>(
    "status_publisher",
    [](res<SystemState> state,
       spmc_mut<StatusMsg, StatusTag> status_out) {

        status_out.write(msg);
    });
```
