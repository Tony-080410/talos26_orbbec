#pragma once

#include "buffer.hpp"
#include "euler.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace stdx {

// ---------- copy const/volatile ----------
template <class From, class To>
struct copy_cv {
    using type = To;
};

template <class From, class To>
struct copy_cv<const From, To> {
    using type = const To;
};

template <class From, class To>
struct copy_cv<volatile From, To> {
    using type = volatile To;
};

template <class From, class To>
struct copy_cv<const volatile From, To> {
    using type = const volatile To;
};

template <class From, class To>
using copy_cv_t = typename copy_cv<From, To>::type;

// ---------- copy reference ----------
template <class From, class To>
struct copy_ref {
    using type = To;
};

template <class From, class To>
struct copy_ref<From&, To> {
    using type = To&;
};

template <class From, class To>
struct copy_ref<From&&, To> {
    using type = To&&;
};

template <class From, class To>
using copy_ref_t = typename copy_ref<From, To>::type;

// ---------- combine ----------
template <class From, class To>
using forward_like_t =
    copy_ref_t<From, copy_cv_t<std::remove_reference_t<From>, std::remove_reference_t<To>>>;

// ---------- forward_like ----------
template <class Like, class T>
constexpr forward_like_t<Like, T&&> forward_like(T&& x) noexcept {
    return static_cast<forward_like_t<Like, T&&>>(x);
}

} // namespace stdx
namespace fast_tf {

constexpr size_t _base = __COUNTER__;

#define DECL_ROOT(name)                                                       \
    struct name##_fuxk_frame {                                                \
        static constexpr std::string_view frame_id = #name;                   \
        static constexpr size_t index              = __COUNTER__ - _base - 1; \
        using ancestor                             = void;                    \
    };

#define DECL(name, parent)                                                    \
    struct name##_fuxk_frame {                                                \
        static constexpr std::string_view frame_id = #name;                   \
        static constexpr size_t index              = __COUNTER__ - _base - 1; \
        using ancestor                             = parent##_fuxk_frame;     \
    };

DECL_ROOT(world);
DECL(odom, world);
DECL(gimbal_yaw, odom);
DECL(gimbal_pitch, gimbal_yaw);
DECL(camera_link, gimbal_pitch);
struct camera_optical_fuxk_frame {
    static constexpr std::string_view frame_id = "camera_optical_frame";
    static constexpr size_t index              = __COUNTER__ - _base - 1;
    using ancestor                             = camera_link_fuxk_frame;
};
DECL(muzzle_link, gimbal_pitch);

// 防止外部使用
#undef DECL_ROOT
#undef DECL

using odom           = odom_fuxk_frame;
using gimbal         = gimbal_yaw_fuxk_frame;
using camera         = camera_link_fuxk_frame;
using camera_optical = camera_optical_fuxk_frame;
using muzzle         = muzzle_link_fuxk_frame;

constexpr size_t frame_count = __COUNTER__ - _base - 1;

template <typename T>
concept frame = requires {
    { T::frame_id } -> std::convertible_to<std::string_view>;
    { T::index } -> std::convertible_to<size_t>;
    typename T::ancestor;
};

template <typename T>
concept root_frame = frame<T> && std::is_void_v<typename T::ancestor>;

template <typename T>
concept non_root_frame = frame<T> && !std::is_void_v<typename T::ancestor>;

template <frame Child, frame Ancestor>
constexpr bool is_descendant_of() noexcept {
    if constexpr (std::is_same_v<Child, Ancestor>) {
        return true;
    } else if constexpr (std::is_void_v<typename Child::ancestor>) {
        return false;
    } else {
        return is_descendant_of<typename Child::ancestor, Ancestor>();
    }
}

template <frame Coordinate, typename T>
struct [[nodiscard]] Fuck {
    explicit Fuck(T t)
        : val(std::move(t)) {}
    explicit Fuck() = default;

    // ========================================================================
    // Transparent Dereference Operators (C++23 Deducing This)
    // ========================================================================
    // Single implementation handles all value categories:
    // - Fuck&         → T&
    // - const Fuck&   → const T&
    // - Fuck&&        → T&&
    // - const Fuck&&  → const T&

    template <typename Self>
    [[nodiscard]] constexpr auto operator*(this Self&& self) noexcept -> decltype(auto) {
        // stdx::forward_like preserves the value category:
        // If self is an lvalue reference, returns lvalue reference to val
        // If self is an rvalue reference, returns rvalue reference to val
        return stdx::forward_like<Self>(self.val);
    }

    template <typename Self>
    [[nodiscard]] constexpr auto operator->(this Self&& self) noexcept -> decltype(auto) {
        // std::addressof works for both const and non-const
        // Return type is automatically deduced as pointer-to-T
        // (with const-ness matching Self's const-ness)
        return std::addressof(self.val);
    }

    // ========================================================================
    // Coordinate-Preserving Binary Operators
    // ========================================================================
    // Result retains the left-hand side's coordinate system

    template <typename Self>
    [[nodiscard]] constexpr auto operator*(this Self&& self, const T& other) noexcept
        -> Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) * other)> {
        return Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) * other)>(
            stdx::forward_like<Self>(self.val) * other);
    }

    template <typename Self, frame OtherCoord>
    [[nodiscard]] constexpr auto
        operator*(this Self&& self, const Fuck<OtherCoord, T>& other) noexcept
        -> Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) * (*other))> {
        return Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) * (*other))>(
            stdx::forward_like<Self>(self.val) * (*other));
    }

    template <typename Self, frame OtherCoord>
    [[nodiscard]] constexpr auto
        operator+(this Self&& self, const Fuck<OtherCoord, T>& other) noexcept
        -> Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) + (*other))> {
        return Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) + (*other))>(
            stdx::forward_like<Self>(self.val) + (*other));
    }

    template <typename Self, frame OtherCoord>
    [[nodiscard]] constexpr auto
        operator-(this Self&& self, const Fuck<OtherCoord, T>& other) noexcept
        -> Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) - (*other))> {
        return Fuck<Coordinate, decltype(stdx::forward_like<Self>(self.val) - (*other))>(
            stdx::forward_like<Self>(self.val) - (*other));
    }

    // ========================================================================
    // Transparent Function Call Operator
    // ========================================================================
    // Perfectly forwards arguments to the wrapped callable

    template <typename Self, typename... Args>
    [[nodiscard]] constexpr auto operator()(this Self&& self, Args&&... args) noexcept(
        noexcept(stdx::forward_like<Self>(self.val)(std::forward<Args>(args)...)))
        -> decltype(stdx::forward_like<Self>(self.val)(std::forward<Args>(args)...)) {
        return stdx::forward_like<Self>(self.val)(std::forward<Args>(args)...);
    }

private:
    T val;
};

template <typename Coordinate, typename T>
inline auto fucked(T&& t) {
    return Fuck<Coordinate, T>(std::forward<T>(t));
}

template <frame From, frame To>
using FrameTransform = TransformMatrixd<From, To>;

template <frame Coordinate>
using IdentityTransform = FrameTransform<Coordinate, Coordinate>;

template <non_root_frame Layer>
using EdgeTransform = FrameTransform<Layer, typename Layer::ancestor>;

template <frame Coordinate>
using FuckDouble = Fuck<Coordinate, double>;

template <frame Coordinate>
using FuckFloat = Fuck<Coordinate, float>;

template <frame Coordinate>
using FuckIntrinsicEulerRotd = Fuck<Coordinate, math_fuxk::Ros2EulerRotd>;

template <frame Coordinate>
using FuckIntrinsicEulerRotf = Fuck<Coordinate, math_fuxk::Ros2EulerRotf>;

inline constexpr sec coordinate_history_range_sec          = 5;
inline constexpr sample_per_sec coordinate_history_density = 500;

using coordinate_buffer_frames = std::tuple<
    odom_fuxk_frame, gimbal_yaw_fuxk_frame, gimbal_pitch_fuxk_frame, camera_link_fuxk_frame,
    camera_optical_fuxk_frame, muzzle_link_fuxk_frame>;

using CoordinateSystem = std::tuple<
    Buffer<
        EdgeTransform<odom_fuxk_frame>, coordinate_history_range_sec, coordinate_history_density>,
    Buffer<
        EdgeTransform<gimbal_yaw_fuxk_frame>, coordinate_history_range_sec,
        coordinate_history_density>,
    Buffer<
        EdgeTransform<gimbal_pitch_fuxk_frame>, coordinate_history_range_sec,
        coordinate_history_density>,
    Buffer<
        EdgeTransform<camera_link_fuxk_frame>, coordinate_history_range_sec,
        coordinate_history_density>,
    Buffer<
        EdgeTransform<camera_optical_fuxk_frame>, coordinate_history_range_sec,
        coordinate_history_density>,
    Buffer<
        EdgeTransform<muzzle_link_fuxk_frame>, coordinate_history_range_sec,
        coordinate_history_density>>;

template <non_root_frame Layer, typename System>
[[nodiscard]] constexpr decltype(auto) buffer_of(System&& system) noexcept {
    return std::get<Layer::index - 1>(std::forward<System>(system));
}

template <typename System, typename Fn, std::size_t... Is>
constexpr void
    for_each_coordinate_buffer_impl(System&& system, Fn&& fn, std::index_sequence<Is...>) {
    (fn(std::type_identity<std::tuple_element_t<Is, coordinate_buffer_frames>>{},
        std::get<Is>(std::forward<System>(system))),
     ...);
}

template <typename System, typename Fn>
constexpr void for_each_coordinate_buffer(System&& system, Fn&& fn) {
    for_each_coordinate_buffer_impl(
        std::forward<System>(system), std::forward<Fn>(fn),
        std::make_index_sequence<std::tuple_size_v<coordinate_buffer_frames>>{});
}

template <non_root_frame Layer>
constexpr void update(
    CoordinateSystem& system, const EdgeTransform<Layer>& transform, timestamp_ns_t ns) noexcept {
    buffer_of<Layer>(system).push(ns, transform);
}

template <non_root_frame Layer>
constexpr void update_rotate_only(
    CoordinateSystem& system, math_fuxk::Ros2EulerRotd rot, timestamp_ns_t ns) noexcept {
    buffer_of<Layer>(system).push_rotate_only(ns, rot);
}

template <frame To, frame Source, frame From>
constexpr auto lookup(
    const CoordinateSystem& system, FrameTransform<Source, From> tf_source,
    timestamp_ns_t ns) noexcept -> std::expected<FrameTransform<Source, To>, TransformError> {
    static_assert(is_descendant_of<From, To>(), "From must be a descendant of To");
    if constexpr (std::is_same_v<From, To>) {
        return tf_source;
    } else if constexpr (root_frame<From>) {
        return std::unexpected(TransformError::FrameNotFound);
    } else {
        const auto& buffer = buffer_of<From>(system);
        auto result        = buffer.lookup(ns, interpolate);
        if (!result) {
            return std::unexpected(result.error());
        }
        return lookup<To>(system, tf_source * result->value, ns);
    }
}

template <frame To, frame From>
constexpr auto lookup(const CoordinateSystem& system, timestamp_ns_t ns) noexcept
    -> std::expected<FrameTransform<From, To>, TransformError> {
    return lookup<To>(system, I<From>(), ns);
}

/// Lookup transform using clamped interpolation.
///
/// This never returns `FutureExtrapolationRequired`/`PastExtrapolationRequired` as long as the
/// buffer has data; it clamps the query time to the buffer's time range (equivalent to TF2
/// `TimePointZero` semantics for "latest available" when `ns` is newer than the newest sample).
template <frame To, frame Source, frame From>
constexpr auto lookup_clamped(
    const CoordinateSystem& system, FrameTransform<Source, From> tf_source,
    timestamp_ns_t ns) noexcept -> std::expected<FrameTransform<Source, To>, TransformError> {
    static_assert(is_descendant_of<From, To>(), "From must be a descendant of To");
    if constexpr (std::is_same_v<From, To>) {
        return tf_source;
    } else if constexpr (root_frame<From>) {
        return std::unexpected(TransformError::FrameNotFound);
    } else {
        const auto& buffer = buffer_of<From>(system);
        auto result        = buffer.lookup(ns, clamped);
        if (!result) {
            return std::unexpected(result.error());
        }
        return lookup_clamped<To>(system, tf_source * result->value, ns);
    }
}

template <frame To, frame From>
constexpr auto lookup_clamped(const CoordinateSystem& system, timestamp_ns_t ns) noexcept
    -> std::expected<FrameTransform<From, To>, TransformError> {
    return lookup_clamped<To, From, From>(system, I<From>(), ns);
}

// ============================================================================
// Frame Metadata Accessors (for Foxglove export)
// ============================================================================

/**
 * @brief 获取 frame_id by index（编译期优化为跳转表）
 * @warning 添加新 frame 时需要在此函数中添加对应的 case
 */
constexpr std::string_view frame_id_by_index(std::size_t index) noexcept {
    switch (index) {
    case world_fuxk_frame::index: return world_fuxk_frame::frame_id;
    case odom_fuxk_frame::index: return odom_fuxk_frame::frame_id;
    case gimbal_yaw_fuxk_frame::index: return gimbal_yaw_fuxk_frame::frame_id;
    case gimbal_pitch_fuxk_frame::index: return gimbal_pitch_fuxk_frame::frame_id;
    case camera_link_fuxk_frame::index: return camera_link_fuxk_frame::frame_id;
    case camera_optical_fuxk_frame::index: return camera_optical_fuxk_frame::frame_id;
    case muzzle_link_fuxk_frame::index: return muzzle_link_fuxk_frame::frame_id;
    default: return "unknown";
    }
}

/**
 * @brief 获取 parent_frame_id by index（编译期优化为跳转表）
 * @warning 添加新 frame 时需要在此函数中添加对应的 case
 * @return root frame 返回空字符串，非 root frame 返回其父节点的 frame_id
 */
constexpr std::string_view parent_frame_id_by_index(std::size_t index) noexcept {
    switch (index) {
    case world_fuxk_frame::index: return ""; // root 无父节点
    case odom_fuxk_frame::index: return odom_fuxk_frame::ancestor::frame_id;
    case gimbal_yaw_fuxk_frame::index: return gimbal_yaw_fuxk_frame::ancestor::frame_id;
    case gimbal_pitch_fuxk_frame::index: return gimbal_pitch_fuxk_frame::ancestor::frame_id;
    case camera_link_fuxk_frame::index: return camera_link_fuxk_frame::ancestor::frame_id;
    case camera_optical_fuxk_frame::index: return camera_optical_fuxk_frame::ancestor::frame_id;
    case muzzle_link_fuxk_frame::index: return muzzle_link_fuxk_frame::ancestor::frame_id;
    default: return "";
    }
}

} // namespace fast_tf
