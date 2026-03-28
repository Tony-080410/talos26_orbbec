#pragma once

#include "foxglove_server.hpp"

namespace fcs::detail {

// ============================================================================
// Unified Channel Registry - 统一通道注册宏系统
// ============================================================================
//
// 这个文件提供了一套统一的宏，用于：
// 1. 声明通道成员变量（在 FoxgloveChannels 中）
// 2. 生成 ChannelTraits 特化（编译期类型到通道的映射）
// 3. 生成通道创建代码（在 Factory::create 中）
//
// 使用方式：只需在一个地方添加通道定义，所有相关代码自动生成。
// ============================================================================

// ---------------------------------------------------------------------------
// Channel Member Macros - 生成 channel 成员变量声明
// ---------------------------------------------------------------------------

#define FOXGLOVE_SCENE_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::SceneUpdateChannel> MemberName;
#define FOXGLOVE_IMAGE_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::CompressedImageChannel> MemberName;
#define FOXGLOVE_JSON_CHANNEL(MemberName) std::optional<foxglove::RawChannel> MemberName;
#define FOXGLOVE_TF_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::FrameTransformsChannel> MemberName;
#define FOXGLOVE_CALIB_CHANNEL(MemberName) \
    std::optional<foxglove::schemas::CameraCalibrationChannel> MemberName;
#define FOXGLOVE_LOG_CHANNEL(MemberName) std::optional<foxglove::schemas::LogChannel> MemberName;

// ---------------------------------------------------------------------------
// Channel Traits Macros - 生成 ChannelTraits 特化
// ---------------------------------------------------------------------------

#define FOXGLOVE_JSON_TRAIT(MsgType, ChMember)                                                   \
    template <>                                                                                  \
    struct ChannelTraits<MsgType> {                                                              \
        static auto get(FoxgloveChannels& ch) noexcept -> std::optional<foxglove::RawChannel>* { \
            return std::addressof(ch.ChMember);                                                  \
        }                                                                                        \
    };

#define FOXGLOVE_SCENE_TRAIT(MsgType, ChMember)                        \
    template <>                                                        \
    struct ChannelTraits<MsgType> {                                    \
        static auto get(FoxgloveChannels& ch) noexcept                 \
            -> std::optional<foxglove::schemas::SceneUpdateChannel>* { \
            return std::addressof(ch.ChMember);                        \
        }                                                              \
    };

#define FOXGLOVE_IMAGE_TRAIT(MsgType, ChMember)                            \
    template <>                                                            \
    struct ChannelTraits<MsgType> {                                        \
        static auto get(FoxgloveChannels& ch) noexcept                     \
            -> std::optional<foxglove::schemas::CompressedImageChannel>* { \
            return std::addressof(ch.ChMember);                            \
        }                                                                  \
    };

// ---------------------------------------------------------------------------
// Channel Creation Macros - 生成通道创建代码
// ---------------------------------------------------------------------------

#define TRY_CREATE_SCENE_CHANNEL(ChName, Topic)                                             \
    do {                                                                                    \
        auto ch = detail::try_create_channel<foxglove::schemas::SceneUpdateChannel>(Topic); \
        if (!ch)                                                                            \
            return std::unexpected(fmt::format("{} channel create failed", Topic));         \
        channels.ChName = std::move(*ch);                                                   \
    } while (0)

#define TRY_CREATE_IMAGE_CHANNEL(ChName, Topic)                                                 \
    do {                                                                                        \
        auto ch = detail::try_create_channel<foxglove::schemas::CompressedImageChannel>(Topic); \
        if (!ch)                                                                                \
            return std::unexpected(fmt::format("{} channel create failed", Topic));             \
        channels.ChName = std::move(*ch);                                                       \
    } while (0)

#define TRY_CREATE_JSON_CHANNEL(ChName, Topic)                                      \
    do {                                                                            \
        auto ch = detail::try_create_raw_channel(Topic, "json");                    \
        if (!ch)                                                                    \
            return std::unexpected(fmt::format("{} channel create failed", Topic)); \
        channels.ChName = std::move(*ch);                                           \
    } while (0)

#define TRY_CREATE_SPECIAL_CHANNEL(ChName, Topic, ChType)                           \
    do {                                                                            \
        auto ch = detail::try_create_channel<ChType>(Topic);                        \
        if (!ch)                                                                    \
            return std::unexpected(fmt::format("{} channel create failed", Topic)); \
        channels.ChName = std::move(*ch);                                           \
    } while (0)

#define TRY_CREATE_LOG_CHANNEL(ChName, Topic)                                       \
    do {                                                                            \
        auto ch = detail::try_create_channel<foxglove::schemas::LogChannel>(Topic); \
        if (!ch)                                                                    \
            return std::unexpected(fmt::format("{} channel create failed", Topic)); \
        channels.ChName = std::move(*ch);                                           \
    } while (0)

// Note: try_create_channel and try_create_raw_channel are defined in foxglove_server.hpp
// in the fcs::detail namespace. This header only provides the macros.

} // namespace fcs::detail
