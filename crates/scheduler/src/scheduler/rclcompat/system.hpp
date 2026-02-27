#pragma once

#include "../demangle.hpp"
#include "../system/system.hpp"
#include "registry.hpp"
#include "system_builder.hpp"
#include <spdlog/spdlog.h>

namespace talos::scheduler::rclcompat {

// Forward declaration
class RclSubSystemBase;

/**
 * @brief Thread-local context for tracking the currently executing callback
 *
 * This enables automatic binding of Publishers to the callback that first uses them,
 * preventing multiple callbacks from sharing the same Publisher instance.
 *
 * ## Architecture
 *
 * ```
 * RclSubSystem::run()
 *   ├─ CallbackContextScope scope(this)  // Set g_callback_context.current_system = this
 *   ├─ callback_(*msg)  ─────────────────┐
 *   │   └─ publisher.publish(msg)        │
 *   │       ├─ First use: Auto-bind to this
 *   │       └─ Subsequent: Verify binding ─────┘
 *   └─ ~CallbackContextScope()  // Restore previous context
 * ```
 */
struct CallbackContext {
    RclSubSystemBase* current_system = nullptr; ///< Pointer to the currently executing RclSubSystem
};

/// Global thread-local callback context
inline thread_local CallbackContext g_callback_context;

/**
 * @brief RAII scope guard for setting/restoring callback context
 *
 * Automatically restores the previous context when destroyed,
 * ensuring exception safety.
 */
struct CallbackContextScope {
    CallbackContext* ctx_;
    RclSubSystemBase* prev_system_;

    explicit CallbackContextScope(RclSubSystemBase* current) noexcept
        : ctx_(&g_callback_context)
        , prev_system_(ctx_->current_system) {
        ctx_->current_system = current;
    }

    ~CallbackContextScope() noexcept { ctx_->current_system = prev_system_; }

    // Non-copyable, non-movable
    CallbackContextScope(const CallbackContextScope&)            = delete;
    CallbackContextScope& operator=(const CallbackContextScope&) = delete;
};

/**
 * @brief Base class for RclSubSystem
 *
 * Provides access to SystemMeta for callback ownership tracking.
 */
class RclSubSystemBase : public SystemBase {
public:
    ~RclSubSystemBase() override = default;
};

// ============================================================================
// PubSlot: Publisher state (owned by Node)
// ============================================================================

/**
 * @brief Publisher slot - holds channel writer and write state
 *
 * Owned by the Node, shared with Publisher handles and RclPubSystem.
 *
 * ## Template parameters
 *
 * - `T`: message type
 * - `Topic`: topic tag type
 */
template <typename T, typename Topic = DefaultTopic>
struct PubSlot {
    std::optional<spmc_mut<T, Topic>> writer;
    std::atomic<bool> bound{false};         ///< Has bind() been called?
    std::atomic<bool> pending_write{false}; ///< Is there a pending write for run() to consume?

    // Callback ownership tracking (prevents multiple callbacks from sharing the same Publisher)
    std::atomic<RclSubSystemBase*> owner_callback{
        nullptr};                        ///< The callback that owns this publisher
    std::atomic<bool> auto_bound{false}; ///< Whether auto-binding has occurred

    /**
     * @brief Bind this slot to the world
     *
     * Called by RclPubSystem::bind()
     *
     * ## Parameters
     *
     * - `world`: the world
     */
    void bind(World& world) noexcept {
        writer = world.get_spmc_writer<T, Topic>();
        bound.store(true, std::memory_order_release);
    }

    /**
     * @brief Publish a message
     *
     * Called by Publisher::publish()
     *
     * ## Parameters
     *
     * - `msg`: message to publish (moved)
     */
    void publish(T msg) noexcept {
        if (bound.load(std::memory_order_acquire)) [[likely]] {
            writer->write(std::move(msg));
            pending_write.store(true, std::memory_order_release);
        }
    }

    /**
     * @brief Check if the slot is ready to publish
     *
     * ## Returns
     *
     * `true` if bound to the world
     */
    [[nodiscard]] bool ready() const noexcept { return bound.load(std::memory_order_acquire); }
};

// ============================================================================
// Publisher: Move-only handle to PubSlot
// ============================================================================

/**
 * @brief Move-only publisher handle
 *
 * Lightweight handle that references a PubSlot owned by a Node.
 * Move-only to enforce single-ownership at compile time.
 *
 * ## Template parameters
 *
 * - `T`: message type
 * - `Topic`: topic tag type
 *
 * ## Example
 *
 * ```cpp
 * auto pub = node.create_publisher<ImageFrame, CameraTag>();
 * pub.publish(ImageFrame{...});
 *
 * // Move to another scope
 * auto pub2 = std::move(pub);
 * pub2.publish(ImageFrame{...});
 *
 * // pub is now invalid
 * ```
 */
template <typename T, typename Topic = DefaultTopic>
class Publisher {
public:
    Publisher() noexcept = default;

    /**
     * @brief Construct a publisher handle
     *
     * ## Parameters
     *
     * - `slot`: pointer to the PubSlot (non-owning)
     * - `registry`: pointer to the OwnershipRegistry
     * - `key`: channel key for this publisher
     */
    Publisher(PubSlot<T, Topic>* slot, OwnershipRegistry* registry, const ChannelKey key) noexcept
        : slot_(slot)
        , registry_(registry)
        , key_(key) {}

    // Move-only
    Publisher(const Publisher&)            = delete;
    Publisher& operator=(const Publisher&) = delete;

    Publisher(Publisher&& other) noexcept
        : slot_(other.slot_)
        , registry_(other.registry_)
        , key_(other.key_) {
        other.slot_     = nullptr;
        other.registry_ = nullptr;
    }

    Publisher& operator=(Publisher&& other) noexcept {
        if (this != &other) {
            // Release current claim if any
            if (slot_ && registry_) {
                registry_->release_claim(key_);
            }
            slot_           = other.slot_;
            registry_       = other.registry_;
            key_            = other.key_;
            other.slot_     = nullptr;
            other.registry_ = nullptr;
        }
        return *this;
    }

    ~Publisher() noexcept {
        if (slot_ && registry_) {
            registry_->release_claim(key_);
        }
    }

    /**
     * @brief Publish a message
     *
     * ## Parameters
     *
     * - `msg`: message to publish (moved)
     *
     * ## Panics
     *
     * Aborts if the publisher is invalid (moved-from) or if used from a different
     * callback than the one that first claimed ownership.
     */
    void publish(T msg) noexcept {
        if (!slot_) [[unlikely]] {
            SPDLOG_CRITICAL("Publisher::publish() called on invalid (moved-from) publisher");
            std::abort();
        }

        // Automatic callback binding check
        // This prevents multiple different callbacks from sharing the same Publisher instance
        auto* current_callback = g_callback_context.current_system;

        if (current_callback && !slot_->auto_bound.load(std::memory_order_acquire)) {
            // First use: attempt to bind to the current callback
            RclSubSystemBase* expected = nullptr;
            if (slot_->owner_callback.compare_exchange_strong(
                    expected, current_callback, std::memory_order_release,
                    std::memory_order_acquire)) {
                // Successfully bound to this callback
                slot_->auto_bound.store(true, std::memory_order_release);
            } else if (expected != current_callback) {
                // Another callback already owns this publisher
                SPDLOG_CRITICAL(
                    "Publisher {}@{} owned by callback '{}', used by '{}'",
                    scheduler::detail::demangle(typeid(T).name()),
                    scheduler::detail::demangle(typeid(Topic).name()),
                    expected ? expected->meta().name.c_str() : "<unknown>",
                    current_callback ? current_callback->meta().name.c_str() : "<unknown>");
                std::abort();
            }
        } else if (current_callback) {
            // Already bound: verify the current callback is the owner
            auto* owner = slot_->owner_callback.load(std::memory_order_acquire);
            if (owner && owner != current_callback) {
                SPDLOG_CRITICAL(
                    "Publisher {}@{} owned by callback '{}', used by '{}'",
                    scheduler::detail::demangle(typeid(T).name()),
                    scheduler::detail::demangle(typeid(Topic).name()), owner->meta().name.c_str(),
                    current_callback->meta().name.c_str());
                std::abort();
            }
        }

        slot_->publish(std::move(msg));
    }

    /**
     * @brief Check if the publisher is valid
     *
     * ## Returns
     *
     * `true` if this publisher handle is valid
     */
    [[nodiscard]] bool valid() const noexcept { return slot_ != nullptr; }

    /**
     * @brief Check if the publisher is ready to publish
     *
     * ## Returns
     *
     * `true` if the underlying slot is bound
     */
    [[nodiscard]] bool ready() const noexcept { return slot_ && slot_->ready(); }

private:
    PubSlot<T, Topic>* slot_     = nullptr;
    OwnershipRegistry* registry_ = nullptr;
    ChannelKey key_{typeid(void), typeid(void)};
};

/**
 * @brief System that manages a publisher channel
 *
 * This system's run() checks if any writes have been made to the PubSlot
 * and returns true if so, triggering downstream systems.
 *
 * ## Template parameters
 *
 * - `T`: message type
 * - `Topic`: topic tag type
 * - `Policy`: execution policy (default: pool_compute)
 */
template <typename T, typename Topic = DefaultTopic, typename Policy = pool_compute>
class RclPubSystem : public SystemBase {
public:
    /**
     * @brief Construct a publisher system
     *
     * ## Parameters
     *
     * - `name`: system name
     * - `slot`: pointer to the PubSlot to manage
     */
    explicit RclPubSystem(std::string name, PubSlot<T, Topic>* slot) noexcept
        : slot_(slot) {
        meta_ = SystemMetaBuilder(std::move(name))
                    .policy(system::make_policy_info<Policy>())
                    .add_spmc_writer(typeid(T), typeid(Topic))
                    .build();
    }

    void bind(World& world) noexcept override { slot_->bind(world); }

    bool run([[maybe_unused]] World& world) noexcept override {
        // Consume the pending_write flag atomically
        return slot_->pending_write.exchange(false, std::memory_order_acq_rel);
    }

    const SystemMeta& meta() const noexcept override { return meta_; }

private:
    SystemMeta meta_;
    PubSlot<T, Topic>* slot_;
};

/**
 * @brief System that manages a subscription
 *
 * Reads from an SPMC channel and invokes a callback.
 * Can optionally track associated publishers for write detection.
 *
 * ## Template parameters
 *
 * - `T`: message type
 * - `Topic`: topic tag type
 * - `Policy`: execution policy (default: pool_compute)
 */
template <typename T, typename Topic = DefaultTopic, typename Policy = pool_compute>
class RclSubSystem : public RclSubSystemBase {
public:
    using Callback = std::function<void(const T&)>;

    /**
     * @brief Construct a subscription system
     *
     * ## Parameters
     *
     * - `name`: system name
     * - `callback`: function to call when a message is received
     */
    explicit RclSubSystem(std::string name, Callback callback) noexcept
        : callback_(std::move(callback)) {
        meta_ = SystemMetaBuilder(std::move(name))
                    .policy(system::make_policy_info<Policy>())
                    .add_spmc_reader(typeid(T), typeid(Topic))
                    .build();
    }

    /**
     * @brief Track a publisher that the callback may write to
     *
     * This updates the SystemMeta to include the publisher's channel as a writer,
     * enabling proper wake chain computation.
     *
     * ## Parameters
     *
     * - `write_flag`: pointer to the PubSlot's pending_write flag
     * - `type`: data type of the publisher
     * - `topic`: topic type of the publisher
     */
    void track_publisher(
        std::atomic<bool>* write_flag, const std::type_index type,
        const std::type_index topic) noexcept {
        tracked_write_flags_.push_back(write_flag);

        // Update meta to include this writer channel
        meta_.spmc_channels.push_back(
            ChannelMeta{
                .type  = type,
                .topic = topic,
                .kind  = channel_kind::spmc_writer,
            });
    }

    void bind(World& world) noexcept override { reader_ = world.get_spmc_reader<T, Topic>(); }

    bool run([[maybe_unused]] World& world) noexcept override {
        if (!reader_.has_new()) {
            return false;
        }

        auto msg = reader_.read();
        if (msg) {
            // Set thread-local context to enable automatic Publisher binding
            // The RAII scope guard automatically restores the previous context on destruction
            CallbackContextScope scope(this);
            callback_(*msg);
        }

        // Check if any tracked publishers wrote data
        bool any_written = false;
        for (auto* flag : tracked_write_flags_) {
            if (flag->exchange(false, std::memory_order_acq_rel)) {
                any_written = true;
            }
        }
        return any_written;
    }

    const SystemMeta& meta() const noexcept override { return meta_; }

private:
    SystemMeta meta_;
    Callback callback_;
    spmc<T, Topic> reader_;
    std::vector<std::atomic<bool>*> tracked_write_flags_;
};

// ============================================================================
// ResourceAccessor: Proxy for accessing World resources
// ============================================================================

/**
 * @brief Proxy for accessing resources from the World
 *
 * Similar to Publisher, this is a lightweight handle that provides access
 * to resources stored in the World. Can be stored as a class member and
 * used from callbacks.
 *
 * ## Template parameters
 *
 * - `T`: resource type
 *
 * ## Example
 *
 * ```cpp
 * class Detector {
 *     ResourceAccessor<TfBuffer> tf_buffer_;
 *     Publisher<Detection> det_pub_;
 *
 *     void init(Node& node) {
 *         tf_buffer_ = node.create_resource<TfBuffer>();
 *         det_pub_ = node.create_publisher<Detection>();
 *
 *         node.create_subscription<Image>([this](const Image& img) {
 *             auto tf = tf_buffer_.get();  // Access resource
 *             det_pub_.publish(process(img, tf));
 *         });
 *     }
 * };
 * ```
 */
template <typename T>
class ResourceAccessor {
public:
    ResourceAccessor() noexcept = default;

    /**
     * @brief Construct a resource accessor
     *
     * ## Parameters
     *
     * - `world`: pointer to the world (non-owning)
     */
    explicit ResourceAccessor(World* world) noexcept
        : world_(world) {}

    /**
     * @brief Get immutable access to the resource
     *
     * ## Returns
     *
     * `res<T>` handle for reading the resource
     *
     * ## Thread safety
     *
     * Multiple threads can call get() concurrently.
     */
    [[nodiscard]] res<T> get() const noexcept {
        if (!world_) [[unlikely]] {
            SPDLOG_CRITICAL("ResourceAccessor::get() called on invalid accessor");
            std::abort();
        }
        return world_->get_res<T>();
    }

    /**
     * @brief Get mutable access to the resource
     *
     * ## Returns
     *
     * `res_mut<T>` handle for reading/writing the resource
     *
     * ## Thread safety
     *
     * Multiple threads can call get_mut(), but concurrent writes must be
     * synchronized externally. The resource version is automatically
     * incremented on write.
     */
    [[nodiscard]] res_mut<T> get_mut() const noexcept {
        if (!world_) [[unlikely]] {
            SPDLOG_CRITICAL("ResourceAccessor::get_mut() called on invalid accessor");
            std::abort();
        }
        return world_->get_res_mut<T>();
    }

    /**
     * @brief Check if a resource accessor is valid
     *
     * ## Returns
     *
     * `true` if this accessor is bound to a World
     */
    [[nodiscard]] bool valid() const noexcept { return world_ != nullptr; }

    // Copyable (multiple accessors can reference the same World)
    ResourceAccessor(const ResourceAccessor&)            = default;
    ResourceAccessor& operator=(const ResourceAccessor&) = default;

private:
    World* world_ = nullptr;
};

} // namespace talos::scheduler::rclcompat
