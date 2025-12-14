#pragma once

#include <atomic>
#include <array>
#include <utility>
#include <cstring>

namespace spike {

/**
 * @brief Lock-free triple-buffer for producer-consumer synchronization
 *
 * Design pattern:
 * - Producer (detector thread): writes to slots[write_idx], publishes via ready_idx
 * - Consumer (tracker thread): reads from ready_idx, never blocks
 * - Never blocks either party - supports drop-on-overload
 *
 * Memory ordering strategy:
 * - relaxed: Non-critical fields (timestamps, counters)
 * - release/acquire: Frame publish/consume synchronization point
 * - NO seq_cst: Minimizes synchronization overhead
 *
 * Latency target: < 50ns per round-trip (write + read)
 *
 * Example usage:
 *
 * // Producer thread (detector)
 * auto* slot = buffer.begin_write();
 * slot->data = compute_armors();
 * buffer.publish_write();
 *
 * // Consumer thread (tracker) - 250Hz timer
 * auto [data, version] = buffer.try_read();
 * if (data != nullptr) {
 *   tracker.update(data);  // New frame
 * } else {
 *   tracker.predict();     // No new data, predict only
 * }
 */
template <typename T>
class TripleBuffer {
 public:
  struct alignas(64) Slot {
    T data;
    std::atomic<uint64_t> version{0};  // Detects stale reads
  };

  TripleBuffer()
      : write_idx_(0),
        ready_idx_(1),
        read_idx_(1) {}  // Initially: ready_idx == read_idx (no new data)

  /**
   * @brief Get writable buffer slot
   * Producer calls this, fills data, then calls publish_write()
   */
  inline T* begin_write() {
    return &slots_[write_idx_.load(std::memory_order_relaxed)].data;
  }

  /**
   * @brief Publish completed write
   * Called by producer after filling buffer
   * Atomically rotates write -> ready
   * Memory order: release (ensures data visibility to reader)
   */
  inline void publish_write() {
    auto idx = write_idx_.load(std::memory_order_relaxed);
    // Increment version for stale-read detection
    slots_[idx].version.fetch_add(1, std::memory_order_release);
    // Rotate write index: 0->1, 1->2, 2->0
    write_idx_.store((idx + 1) % 3, std::memory_order_relaxed);
    // Publish: idx becomes the ready slot
    ready_idx_.store(idx, std::memory_order_release);
  }

  /**
   * @brief Try to read latest frame
   * Returns {nullptr, 0} if no new frame available (non-blocking)
   * Returns {data_ptr, version} if new frame available
   *
   * Memory order: acquire (synchronize with producer's release)
   */
  inline std::pair<const T*, uint64_t> try_read() {
    auto ready = ready_idx_.load(std::memory_order_acquire);
    auto last_read = read_idx_.load(std::memory_order_relaxed);

    // Check if we have a new frame
    if (ready == last_read) {
      return {nullptr, 0};  // No new frame, consumer can skip or predict-only
    }

    // Update our read pointer
    read_idx_.store(ready, std::memory_order_relaxed);

    // Return data pointer and version for verification
    auto version = slots_[ready].version.load(std::memory_order_acquire);
    return {&slots_[ready].data, version};
  }

  /**
   * @brief Non-blocking check if new data is available
   * Useful for drop-on-overload logic in detector
   */
  inline bool has_new_frame() const {
    return ready_idx_.load(std::memory_order_acquire) !=
           read_idx_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Force reset all indices (use with caution)
   * Utility for testing or recovery
   */
  inline void reset() {
    write_idx_.store(0, std::memory_order_release);
    ready_idx_.store(1, std::memory_order_release);
    read_idx_.store(1, std::memory_order_release);  // Must equal ready_idx (no new data)
    for (auto& slot : slots_) {
      slot.version.store(0, std::memory_order_release);
    }
  }

 private:
  std::array<Slot, 3> slots_;  // Triple buffer: 3 independent slots
  std::atomic<uint8_t> write_idx_;    // Producer writes here
  std::atomic<uint8_t> ready_idx_;    // Completed frame index
  std::atomic<uint8_t> read_idx_;     // Consumer's last read

  // Prevent false sharing between atomic indices
  char _padding[64 - 3];  // Ensure each atomic is in separate cache line
};

} // namespace spike
