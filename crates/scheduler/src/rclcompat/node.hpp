#pragma once

#include "demangle.hpp"
#include "ecs/channel_types.hpp"
#include "scheduler/rclcompat/system.hpp"
#include "scheduler/rclcompat/timer_factory.hpp"
#include "scheduler/scheduler.hpp"

namespace talos::rclcompat {

// ============================================================================
// Node: Container for publishers and subscriptions
// ============================================================================

enum class Qos {
    Volatile, ///< No history, latest value only
};

/**
 * @brief ROS2-style node container
 *
 * A Node owns publishers and subscriptions and manages their lifecycle.
 * Call finalize() after creating all pub/sub to register systems with the scheduler.
 *
 * ## Thread safety
 *
 * - create_publisher() / create_subscription() must be called from a single thread
 * - finalize() must be called before scheduler.run() or while scheduler is paused
 * - Publisher::publish() is thread-safe after finalize()
 */
class Node {
public:
    /**
     * @brief Construct a node
     *
     * ## Parameters
     *
     * - `name`: node name (used as prefix for system names)
     * - `scheduler`: reference to the scheduler to register systems with
     */
    explicit Node(std::string name, Scheduler& scheduler) noexcept
        : name_(std::move(name))
        , scheduler_(scheduler) {}

    // Non-copyable, non-movable (owns references to internal data)
    Node(const Node&)            = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&)                 = delete;
    Node& operator=(Node&&)      = delete;

    /**
     * @brief Create a publisher
     *
     * Returns a move-only Publisher handle. The underlying PubSlot is owned by this Node.
     *
     * ## Template parameters
     *
     * - `T`: message type
     * - `Topic`: topic tag type (default: DefaultTopic)
     *
     * ## Parameters
     *
     * - `qos`: quality of service settings (currently unused)
     *
     * ## Returns
     *
     * Move-only Publisher handle
     *
     * ## Panics
     *
     * Aborts if a publisher for this channel already exists
     */
    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] Publisher<T, Topic> create_publisher([[maybe_unused]] Qos qos = Qos::Volatile) {
        const ChannelKey key{typeid(T), typeid(Topic)};
        const std::string system_name = name_ + "/pub/" + detail::demangle(typeid(Topic).name())
                                      + "/" + detail::demangle(typeid(T).name());

        // Check if this channel is already owned
        if (pub_slots_.contains(key)) {
            SPDLOG_CRITICAL(
                "Node '{}': Publisher for channel {}@{} already exists", name_,
                detail::demangle(key.type.name()), detail::demangle(key.topic.name()));
            std::abort();
        }

        // Create PubSlot using shared_ptr (std::any requires copy-constructibility)
        auto slot           = std::make_shared<PubSlot<T, Topic>>();
        auto* slot_ptr      = slot.get();
        pub_slots_[key]     = slot; // Store as std::any (shared_ptr is copyable)
        pub_slot_ptrs_[key] = slot_ptr;

        // Register ownership
        registry_.register_owner(key, system_name);

        // Claim the publisher handle
        if (!registry_.try_claim(key)) {
            SPDLOG_CRITICAL(
                "Node '{}': Failed to claim publisher for channel {}@{}", name_,
                detail::demangle(key.type.name()), detail::demangle(key.topic.name()));
            std::abort();
        }

        // Create the system (deferred registration until finalize())
        pending_systems_.push_back(std::make_unique<RclPubSystem<T, Topic>>(system_name, slot_ptr));

        return Publisher<T, Topic>(slot_ptr, &registry_, key);
    }

    /**
     * @brief Create a subscription
     *
     * Registers a callback to be invoked when messages are received.
     *
     * ## Template parameters
     *
     * - `T`: message type
     * - `Topic`: topic tag type (default: DefaultTopic)
     *
     * ## Parameters
     *
     * - `callback`: function to call when a message is received
     * - `qos`: quality of service settings (currently unused)
     */
    template <typename T, typename Topic = DefaultTopic>
    void create_subscription(
        std::function<void(const T&)> callback, [[maybe_unused]] Qos qos = Qos::Volatile) {
        const ChannelKey key{typeid(T), typeid(Topic)};
        const std::string system_name = name_ + "/sub/" + detail::demangle(typeid(Topic).name())
                                      + "/" + detail::demangle(typeid(T).name());

        pending_systems_.push_back(
            std::make_unique<RclSubSystem<T, Topic>>(system_name, std::move(callback)));
    }

    /**
     * @brief Create a wall timer
     *
     * Creates a timer that invokes a callback at a fixed frequency.
     * The callback runs on a dedicated fixed_rate thread.
     *
     * ## Parameters
     *
     * - `frequency`: Timer frequency from the Frequency enum (Hz_1 to Hz_1000)
     * - `callback`: function to call at each timer tick
     *
     * ## Example
     *
     * ```cpp
     * // Simple timer at 10Hz
     * node.create_wall_timer(Frequency::Hz_10, []() {
     *     // Executes at 10Hz
     * });
     *
     * // Timer with Publisher at 30Hz
     * auto pub = node.create_publisher<ImageFrame>();
     * node.create_wall_timer(Frequency::Hz_30, [&pub]() {
     *     pub.publish(...);
     * });
     *
     * // Timer with ResourceAccessor at 20Hz
     * auto tf = node.create_resource<TfBuffer>();
     * node.create_wall_timer(Frequency::Hz_20, [&tf]() {
     *     auto transform = tf.get();
     * });
     * ```
     */
    void create_wall_timer(Frequency frequency, std::function<void()> callback) {
        const std::string system_name = name_ + "/timer/" + std::to_string(pending_systems_.size());

        auto system = create_timer_system(system_name, frequency, std::move(callback));
        pending_systems_.push_back(std::move(system));
    }

    /**
     * @brief Create a resource accessor
     *
     * Returns a copyable ResourceAccessor handle that can be used to access
     * resources from the World. Can be stored as a class member and used from
     * callbacks.
     *
     * ## Template parameters
     *
     * - `T`: resource type
     *
     * ## Returns
     *
     * ResourceAccessor handle for accessing the resource
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
    [[nodiscard]] ResourceAccessor<T> create_resource() noexcept {
        return ResourceAccessor<T>(&scheduler_.world());
    }

    /**
     * @brief Check if a resource exists in the World
     *
     * ## Template parameters
     *
     * - `T`: resource type
     *
     * ## Returns
     *
     * `true` if the resource exists
     */
    template <typename T>
    [[nodiscard]] bool has_resource() const noexcept {
        return scheduler_.world().template has_resource<T>();
    }

    /**
     * @brief Insert a resource into the World
     *
     * ## Template parameters
     *
     * - `T`: resource type
     *
     * ## Parameters
     *
     * - `resource`: the resource to insert (moved)
     */
    template <typename T>
    void insert_resource(T resource) {
        scheduler_.world().insert_resource(std::move(resource));
    }

    /**
     * @brief Update a resource in the World
     *
     * ## Template parameters
     *
     * - `T`: resource type
     *
     * ## Parameters
     *
     * - `new_value`: the new resource value (moved)
     *
     * ## Returns
     *
     * `std::expected<uint64_t, std::string>` - the new version number on success
     */
    template <typename T>
    [[nodiscard]] std::expected<uint64_t, std::string> update_resource(T new_value) {
        return scheduler_.world().update_resource(std::move(new_value));
    }

    /**
     * @brief Get the node name
     *
     * ## Returns
     *
     * Node name as string_view
     */
    [[nodiscard]] std::string_view name() const noexcept { return name_; }

    /**
     * @brief Finalize and register all systems with the scheduler
     *
     * Must be called after creating all publishers and subscriptions.
     * If the scheduler is already running, this will use hot_add_system()
     * to pause, add, and resume.
     *
     * ## Returns
     *
     * BuildResult indicating success or failure
     *
     * ## Thread safety
     *
     * Must be called from a single thread, either before scheduler.run()
     * or while the scheduler is paused.
     */
    [[nodiscard]] BuildResult finalize() {
        for (auto& sys : pending_systems_) {
            if (scheduler_.is_running()) {
                // Hot-add: scheduler is running, need to pause/rebuild/resume
                if (auto result = scheduler_.hot_add_system(std::move(sys)); !result) {
                    return result;
                }
            } else {
                if (auto result = scheduler_.add_system(std::move(sys)); !result) {
                    return std::unexpected(BuildError{result.error()});
                }
            }
        }
        pending_systems_.clear();

        // If not running, we need to explicitly call build
        if (!scheduler_.is_running()) {
            return scheduler_.build();
        }

        return {};
    }

    /**
     * @brief Get the ownership registry
     *
     * ## Returns
     *
     * Reference to the ownership registry
     */
    [[nodiscard]] OwnershipRegistry& registry() noexcept { return registry_; }

    /**
     * @brief Get a raw pointer to a PubSlot (for advanced use)
     *
     * ## Template parameters
     *
     * - `T`: message type
     * - `Topic`: topic tag type
     *
     * ## Returns
     *
     * Pointer to the PubSlot, or nullptr if not found
     */
    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] PubSlot<T, Topic>* get_pub_slot() noexcept {
        const ChannelKey key{typeid(T), typeid(Topic)};
        auto it = pub_slot_ptrs_.find(key);
        if (it == pub_slot_ptrs_.end()) {
            return nullptr;
        }
        return static_cast<PubSlot<T, Topic>*>(it->second);
    }

private:
    std::string name_;
    Scheduler& scheduler_;
    OwnershipRegistry registry_;

    // Type-erased storage for PubSlots using std::any
    // We store the unique_ptr in std::any and keep raw pointers separately
    std::unordered_map<ChannelKey, std::any, ChannelKeyHash> pub_slots_;
    std::unordered_map<ChannelKey, void*, ChannelKeyHash> pub_slot_ptrs_;

    std::vector<std::unique_ptr<SystemBase>> pending_systems_;
};

} // namespace talos::rclcompat
