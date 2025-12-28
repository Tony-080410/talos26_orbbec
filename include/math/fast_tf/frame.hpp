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
DECL(muzzle_link, camera_link);

// 防止外部使用
#undef DECL_ROOT
#undef DECL

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

    [[nodiscard]] const T& operator*() const noexcept { return val; }
    [[nodiscard]] T& operator*() noexcept { return val; }

    [[nodiscard]] const T* operator->() const noexcept { return &val; }
    [[nodiscard]] T* operator->() noexcept { return &val; }

private:
    T val;
};

template <typename T>
using FuckWorld = Fuck<world_fuxk_frame, T>;

template <frame Coordinate>
using FuckTransform = Fuck<Coordinate, TransformMatrix4d>;

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
        auto result        = buffer.at_interpolated_clamped(ns);
        if (!result) {
            return std::unexpected(result.error());
        }
        FuckTransform<typename From::ancestor> parent_tf{tf_source->reparent_to(result->transform)};
        return lookup<To, typename From::ancestor, N>(system, parent_tf, ns);
    }
}

} // namespace fast_tf
