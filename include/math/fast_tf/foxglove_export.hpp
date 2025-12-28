#pragma once

#include "frame.hpp"

#include <foxglove/schemas.hpp>

namespace fast_tf {

class FoxgloveExporter {
public:
    explicit FoxgloveExporter(foxglove::schemas::FrameTransformsChannel channel)
        : channel_(std::move(channel)) {}

    /**
     * @brief 发布从 Descendant 到 root 的整条链上的所有变换（单条消息）
     */
    template <non_root_frame Descendant, std::size_t N>
    void publish_entire(const CoordinateSystem<N>& system, const uint64_t ts) noexcept {
        foxglove::schemas::FrameTransforms transforms_msg;

        // 直接从 Descendant 开始递归，统一从 system 取
        append_ancestors<Descendant, N>(system, transforms_msg, ts);

        [[maybe_unused]] auto error = channel_.log(transforms_msg);
    }

    /**
     * @brief 只发布单个变换 Descendant -> Descendant::ancestor
     */
    template <non_root_frame Descendant, std::size_t N>
    void publish_single(
        [[maybe_unused]] const CoordinateSystem<N>& system,
        const FuckTransform<Descendant>& transform, const uint64_t ts) noexcept {

        foxglove::schemas::FrameTransforms transforms_msg;
        append_transform<Descendant>(transforms_msg, transform, ts);

        [[maybe_unused]] auto error = channel_.log(transforms_msg);
    }

private:
    foxglove::schemas::FrameTransformsChannel channel_;

    /**
     * @brief 递归添加祖先链的变换
     */
    template <frame Descendant, std::size_t N>
    void append_ancestors(
        const CoordinateSystem<N>& system, foxglove::schemas::FrameTransforms& msg,
        uint64_t ts) noexcept {
        if constexpr (!root_frame<Descendant>) {
            if (auto result = system[Descendant::index].at_interpolated_clamped(ts); result) {
                append_transform<Descendant>(msg, FuckTransform<Descendant>(result->transform), ts);
                append_ancestors<typename Descendant::ancestor, N>(system, msg, ts);
            }
        }
    }

    template <non_root_frame Descendant>
    static void append_transform(
        foxglove::schemas::FrameTransforms& msg, const FuckTransform<Descendant>& transform,
        const uint64_t ts) noexcept {
        using Ancestor = Descendant::ancestor;

        const auto translation = transform->translation();
        const auto quat        = transform->quaternion();

        foxglove::schemas::FrameTransform frame_transform;
        frame_transform.parent_frame_id = std::string(Ancestor::frame_id);
        frame_transform.child_frame_id  = std::string(Descendant::frame_id);
        frame_transform.timestamp       = {
            static_cast<uint32_t>(ts / 1'000'000'000), static_cast<uint32_t>(ts % 1'000'000'000)};
        frame_transform.translation = {translation.x(), translation.y(), translation.z()};
        frame_transform.rotation    = {quat.x(), quat.y(), quat.z(), quat.w()};

        msg.transforms.push_back(frame_transform);
    }
};

} // namespace fast_tf