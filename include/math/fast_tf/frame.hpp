#pragma once

#include "buffer.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include <string_view>
#include <utility>

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
DECL(gimbal_link, odom);
DECL(camera_link, gimbal_link);
struct camera_optical_fuxk_frame {
    static constexpr std::string_view frame_id = "camera_optical_frame";
    static constexpr size_t index              = __COUNTER__ - _base - 1;
    using ancestor                             = camera_link_fuxk_frame;
};
;
DECL(muzzle_link, gimbal_link);

// 防止外部使用
#undef DECL_ROOT
#undef DECL

using odom           = odom_fuxk_frame;
using gimbal         = gimbal_link_fuxk_frame;
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
constexpr bool is_descendant_of() {
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
        // std::forward_like preserves the value category:
        // If self is an lvalue reference, returns lvalue reference to val
        // If self is an rvalue reference, returns rvalue reference to val
        return std::forward_like<Self>(self.val);
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
        -> Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) * other)> {
        return Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) * other)>(
            std::forward_like<Self>(self.val) * other);
    }

    template <typename Self, frame OtherCoord>
    [[nodiscard]] constexpr auto
        operator*(this Self&& self, const Fuck<OtherCoord, T>& other) noexcept
        -> Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) * (*other))> {
        return Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) * (*other))>(
            std::forward_like<Self>(self.val) * (*other));
    }

    template <typename Self, frame OtherCoord>
    [[nodiscard]] constexpr auto
        operator+(this Self&& self, const Fuck<OtherCoord, T>& other) noexcept
        -> Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) + (*other))> {
        return Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) + (*other))>(
            std::forward_like<Self>(self.val) + (*other));
    }

    template <typename Self, frame OtherCoord>
    [[nodiscard]] constexpr auto
        operator-(this Self&& self, const Fuck<OtherCoord, T>& other) noexcept
        -> Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) - (*other))> {
        return Fuck<Coordinate, decltype(std::forward_like<Self>(self.val) - (*other))>(
            std::forward_like<Self>(self.val) - (*other));
    }

    // ========================================================================
    // Transparent Function Call Operator
    // ========================================================================
    // Perfectly forwards arguments to the wrapped callable

    template <typename Self, typename... Args>
    [[nodiscard]] constexpr auto operator()(this Self&& self, Args&&... args) noexcept(
        noexcept(std::forward_like<Self>(self.val)(std::forward<Args>(args)...)))
        -> decltype(std::forward_like<Self>(self.val)(std::forward<Args>(args)...)) {
        return std::forward_like<Self>(self.val)(std::forward<Args>(args)...);
    }

private:
    T val;
};

template <typename Coordinate, typename T>
inline auto fucked(T&& t) {
    return Fuck<Coordinate, T>(std::forward<T>(t));
}

template <frame Coordinate>
using FuckTransform = Fuck<Coordinate, TransformMatrix4d>;

using FuckWorld = FuckTransform<world_fuxk_frame>;

using FuckOdom   = FuckTransform<odom_fuxk_frame>;
using FuckGimbal = FuckTransform<gimbal_link_fuxk_frame>;

using FuckMuzzle = FuckTransform<muzzle_link_fuxk_frame>;

using FuckCamera  = FuckTransform<camera_link_fuxk_frame>;
using FuckOptical = FuckTransform<camera_optical_fuxk_frame>;

template <frame Coordinate>
using FuckDouble = Fuck<Coordinate, double>;

template <frame Coordinate>
using FuckFloat = Fuck<Coordinate, float>;

template <frame Coordinate>
using FuckIntrinsicEulerRotd = Fuck<Coordinate, math_fuxk::Ros2EulerRotd>;

template <frame Coordinate>
using FuckIntrinsicEulerRotf = Fuck<Coordinate, math_fuxk::Ros2EulerRotf>;

template <std::size_t N>
using CoordinateSystem = std::array<CircularTransformBuffer<N>, frame_count>;

template <non_root_frame Layer, std::size_t N>
constexpr void
    update(CoordinateSystem<N>& system, FuckTransform<Layer> transform, timestamp_ns_t ns) {
    system[Layer::index].push(ns, *transform);
}

template <frame To, frame From, std::size_t N>
constexpr auto
    lookup(const CoordinateSystem<N>& system, FuckTransform<From> tf_source, timestamp_ns_t ns)
        -> std::expected<FuckTransform<To>, TransformError> {
    static_assert(is_descendant_of<From, To>(), "From must be a descendant of To");
    if constexpr (std::is_same_v<From, To>) {
        return tf_source;
    } else if constexpr (root_frame<From>) {
        // 到根了还没找到，不应该发生（static_assert 应该拦住）
        return std::unexpected(TransformError::FrameNotFound);
    } else {
        const auto& buffer = system[From::index];
        auto result        = buffer.at_interpolated(ns);
        if (!result) {
            return std::unexpected(result.error());
        }
        FuckTransform<typename From::ancestor> parent_tf{tf_source->reparent_to(result->transform)};
        return lookup<To, typename From::ancestor, N>(system, parent_tf, ns);
    }
}

/// Lookup transform using clamped interpolation.
///
/// This never returns `FutureExtrapolationRequired`/`PastExtrapolationRequired` as long as the
/// buffer has data; it clamps the query time to the buffer's time range (equivalent to TF2
/// `TimePointZero` semantics for "latest available" when `ns` is newer than the newest sample).
template <frame To, frame From, std::size_t N>
constexpr auto lookup_clamped(
    const CoordinateSystem<N>& system, FuckTransform<From> tf_source, timestamp_ns_t ns)
    -> std::expected<FuckTransform<To>, TransformError> {
    static_assert(is_descendant_of<From, To>(), "From must be a descendant of To");
    if constexpr (std::is_same_v<From, To>) {
        return tf_source;
    } else if constexpr (root_frame<From>) {
        return std::unexpected(TransformError::FrameNotFound);
    } else {
        const auto& buffer = system[From::index];
        auto result        = buffer.at_interpolated_clamped(ns);
        if (!result) {
            return std::unexpected(result.error());
        }
        FuckTransform<typename From::ancestor> parent_tf{tf_source->reparent_to(result->transform)};
        return lookup_clamped<To, typename From::ancestor, N>(system, parent_tf, ns);
    }
}

// ============================================================================
// Frame Metadata Accessors (for Foxglove export)
// ============================================================================

/**
 * @brief 获取 frame_id by index（编译期优化为跳转表）
 * @warning 添加新 frame 时需要在此函数中添加对应的 case
 */
constexpr std::string_view frame_id_by_index(std::size_t index) {
    switch (index) {
    case world_fuxk_frame::index: return world_fuxk_frame::frame_id;
    case odom_fuxk_frame::index: return odom_fuxk_frame::frame_id;
    case gimbal_link_fuxk_frame::index: return gimbal_link_fuxk_frame::frame_id;
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
constexpr std::string_view parent_frame_id_by_index(std::size_t index) {
    switch (index) {
    case world_fuxk_frame::index: return ""; // root 无父节点
    case odom_fuxk_frame::index: return odom_fuxk_frame::ancestor::frame_id;
    case gimbal_link_fuxk_frame::index: return gimbal_link_fuxk_frame::ancestor::frame_id;
    case camera_link_fuxk_frame::index: return camera_link_fuxk_frame::ancestor::frame_id;
    case camera_optical_fuxk_frame::index: return camera_optical_fuxk_frame::ancestor::frame_id;
    case muzzle_link_fuxk_frame::index: return muzzle_link_fuxk_frame::ancestor::frame_id;
    default: return "";
    }
}

} // namespace fast_tf
