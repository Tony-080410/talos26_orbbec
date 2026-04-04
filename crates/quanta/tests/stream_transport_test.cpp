#include "quanta/stream_transport.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

void fill_test_pattern(uint8_t* data, int width, int height, int linesize, int seed) {
    for (int y = 0; y < height; ++y) {
        auto* row = data + y * linesize;
        for (int x = 0; x < width; ++x) {
            row[x * 3 + 0] = static_cast<uint8_t>((x + seed * 7) % 256);
            row[x * 3 + 1] = static_cast<uint8_t>((y * 2 + seed * 13) % 256);
            row[x * 3 + 2] = static_cast<uint8_t>(((x + y) * 3 + seed * 17) % 256);
        }
    }
}

} // namespace

TEST(StreamTransport, SenderRespectsMtuAndSplitsLargeFrames) {
    uint64_t now_ns = 1'000'000'000ULL;
    quanta::StreamTransportConfig config{
        .mtu_bytes      = 64,
        .transport_hz   = 50,
        .max_stream_fps = 30,
        .now_ns         = [&now_ns] { return now_ns; },
    };

    quanta::StreamSender sender(quanta::EncodeParams{}, config);
    ASSERT_TRUE(sender.init(320, 240).has_value());

    const int linesize = 320 * 3;
    std::vector<uint8_t> frame(static_cast<size_t>(linesize * 240));
    fill_test_pattern(frame.data(), 320, 240, linesize, 1);

    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 1, now_ns).has_value());

    std::vector<quanta::TransportChunk> chunks;
    for (int tick = 0; tick < 256; ++tick) {
        auto chunk = sender.poll_chunk();
        ASSERT_TRUE(chunk.has_value()) << chunk.error();
        if (!chunk->has_value() && sender.stats().queued_chunks == 0)
            break;
        if (!chunk->has_value())
            continue;
        chunks.push_back(std::move(**chunk));
        now_ns += 20'000'000ULL;
    }

    ASSERT_GT(chunks.size(), 1U);
    ASSERT_FALSE(chunks.front().data.empty());
    EXPECT_TRUE(chunks.front().start_of_frame);
    EXPECT_TRUE(chunks.back().end_of_frame);

    for (const auto& chunk : chunks) {
        EXPECT_LE(chunk.data.size(), 64U);
        EXPECT_EQ(chunk.fragment_count, chunks.front().fragment_count);
        EXPECT_EQ(chunk.frame_seq, 0U);
    }
}

TEST(StreamTransport, SenderBuffersShortBurstWhileBacklogged) {
    uint64_t now_ns = 1'000'000'000ULL;
    quanta::StreamTransportConfig config{
        .mtu_bytes          = 64,
        .transport_hz       = 50,
        .max_stream_fps     = 30,
        .now_ns             = [&now_ns] { return now_ns; },
        .max_pending_frames = 3,
        .max_queued_chunks  = 8,
    };

    quanta::StreamSender sender(quanta::EncodeParams{}, config);
    ASSERT_TRUE(sender.init(320, 240).has_value());

    const int linesize = 320 * 3;
    std::vector<uint8_t> frame(static_cast<size_t>(linesize * 240));
    fill_test_pattern(frame.data(), 320, 240, linesize, 2);

    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 0, now_ns).has_value());

    auto first_chunk = sender.poll_chunk();
    ASSERT_TRUE(first_chunk.has_value()) << first_chunk.error();
    ASSERT_TRUE(first_chunk->has_value());

    now_ns += 20'000'000ULL;
    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 1, now_ns).has_value());

    now_ns += 20'000'000ULL;
    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 2, now_ns).has_value());

    const auto stats = sender.stats();
    EXPECT_EQ(stats.frames_submitted, 3U);
    EXPECT_EQ(stats.frames_dropped, 0U);
    EXPECT_TRUE(stats.has_pending_frame);
    EXPECT_GT(stats.queued_chunks, 0U);
}

TEST(StreamTransport, SenderDropsOldPendingFramesWhenBufferIsFull) {
    uint64_t now_ns = 1'000'000'000ULL;
    quanta::StreamTransportConfig config{
        .mtu_bytes          = 64,
        .transport_hz       = 50,
        .max_stream_fps     = 30,
        .now_ns             = [&now_ns] { return now_ns; },
        .max_pending_frames = 2,
        .max_queued_chunks  = 8,
    };

    quanta::StreamSender sender(quanta::EncodeParams{}, config);
    ASSERT_TRUE(sender.init(320, 240).has_value());

    const int linesize = 320 * 3;
    std::vector<uint8_t> frame(static_cast<size_t>(linesize * 240));
    fill_test_pattern(frame.data(), 320, 240, linesize, 4);

    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 0, now_ns).has_value());

    auto first_chunk = sender.poll_chunk();
    ASSERT_TRUE(first_chunk.has_value()) << first_chunk.error();
    ASSERT_TRUE(first_chunk->has_value());

    now_ns += 20'000'000ULL;
    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 1, now_ns).has_value());

    now_ns += 20'000'000ULL;
    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 2, now_ns).has_value());

    now_ns += 20'000'000ULL;
    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 3, now_ns).has_value());

    const auto stats = sender.stats();
    EXPECT_EQ(stats.frames_submitted, 4U);
    EXPECT_GE(stats.frames_dropped, 1U);
    EXPECT_TRUE(stats.has_pending_frame);
    EXPECT_GT(stats.queued_chunks, 0U);
}

TEST(StreamTransport, LoopbackProducesDecodedFrameAndLatency) {
    uint64_t now_ns = 1'000'000'000ULL;
    quanta::StreamTransportConfig config{
        .mtu_bytes      = 300,
        .transport_hz   = 50,
        .max_stream_fps = 30,
        .now_ns         = [&now_ns] { return now_ns; },
    };

    quanta::StreamSender sender(quanta::EncodeParams{}, config);
    quanta::StreamReceiver receiver(config);
    ASSERT_TRUE(sender.init(160, 120).has_value());

    const int linesize = 160 * 3;
    std::vector<uint8_t> frame(static_cast<size_t>(linesize * 120));
    fill_test_pattern(frame.data(), 160, 120, linesize, 3);

    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 7, now_ns).has_value());

    std::optional<quanta::DecodedFrame> decoded;
    for (int tick = 0; tick < 64 && !decoded.has_value(); ++tick) {
        auto chunk = sender.poll_chunk();
        ASSERT_TRUE(chunk.has_value()) << chunk.error();
        if (chunk->has_value()) {
            auto pushed = receiver.push_chunk((*chunk)->bytes(), now_ns);
            ASSERT_TRUE(pushed.has_value()) << pushed.error();
        }
        decoded = receiver.poll_frame();
        now_ns += 20'000'000ULL;
    }

    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(decoded->frame_seq, 0U);
    EXPECT_EQ(decoded->pts, 7);
    EXPECT_GT(decoded->width, 0);
    EXPECT_GT(decoded->height, 0);
    EXPECT_FALSE(decoded->data.empty());
    EXPECT_GT(decoded->end_to_end_latency_ns, 0U);

    const auto receiver_stats = receiver.stats();
    EXPECT_EQ(receiver_stats.frames_decoded, 1U);
    EXPECT_EQ(receiver_stats.latency_samples, 1U);
    EXPECT_GT(receiver_stats.max_latency_ns, 0U);
}

TEST(StreamTransport, ReceiverIgnoresDuplicateChunksFromQos1Redelivery) {
    uint64_t now_ns = 1'000'000'000ULL;
    quanta::StreamTransportConfig config{
        .mtu_bytes      = 96,
        .transport_hz   = 50,
        .max_stream_fps = 30,
        .now_ns         = [&now_ns] { return now_ns; },
    };

    quanta::StreamSender sender(quanta::EncodeParams{}, config);
    quanta::StreamReceiver receiver(config);
    ASSERT_TRUE(sender.init(160, 120).has_value());

    const int linesize = 160 * 3;
    std::vector<uint8_t> frame(static_cast<size_t>(linesize * 120));
    fill_test_pattern(frame.data(), 160, 120, linesize, 5);

    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 9, now_ns).has_value());

    std::vector<std::vector<uint8_t>> chunks;
    for (int tick = 0; tick < 128; ++tick) {
        auto chunk = sender.poll_chunk();
        ASSERT_TRUE(chunk.has_value()) << chunk.error();
        if (!chunk->has_value() && sender.stats().queued_chunks == 0)
            break;
        if (!chunk->has_value())
            continue;
        chunks.push_back((*chunk)->data);
        now_ns += 20'000'000ULL;
    }

    ASSERT_GT(chunks.size(), 1U);
    for (const auto& chunk : chunks) {
        ASSERT_TRUE(receiver.push_chunk(chunk, now_ns).has_value());
        now_ns += 20'000'000ULL;
    }
    for (const auto& chunk : chunks) {
        ASSERT_TRUE(receiver.push_chunk(chunk, now_ns).has_value());
        now_ns += 20'000'000ULL;
    }

    auto decoded = receiver.poll_frame();
    ASSERT_TRUE(decoded.has_value());
    EXPECT_FALSE(receiver.poll_frame().has_value());

    const auto receiver_stats = receiver.stats();
    EXPECT_EQ(receiver_stats.frames_decoded, 1U);
    EXPECT_GT(receiver_stats.duplicate_chunks, 0U);
}

TEST(StreamTransport, ReceiverDropsOutOfOrderFragments) {
    uint64_t now_ns = 1'000'000'000ULL;
    quanta::StreamTransportConfig config{
        .mtu_bytes      = 96,
        .transport_hz   = 50,
        .max_stream_fps = 30,
        .now_ns         = [&now_ns] { return now_ns; },
    };

    quanta::StreamSender sender(quanta::EncodeParams{}, config);
    quanta::StreamReceiver receiver(config);
    ASSERT_TRUE(sender.init(160, 120).has_value());

    const int linesize = 160 * 3;
    std::vector<uint8_t> frame(static_cast<size_t>(linesize * 120));
    fill_test_pattern(frame.data(), 160, 120, linesize, 6);

    ASSERT_TRUE(sender.submit_frame(frame.data(), linesize, 11, now_ns).has_value());

    std::vector<std::vector<uint8_t>> chunks;
    for (int tick = 0; tick < 128; ++tick) {
        auto chunk = sender.poll_chunk();
        ASSERT_TRUE(chunk.has_value()) << chunk.error();
        if (!chunk->has_value() && sender.stats().queued_chunks == 0)
            break;
        if (!chunk->has_value())
            continue;
        chunks.push_back((*chunk)->data);
        now_ns += 20'000'000ULL;
    }

    ASSERT_GT(chunks.size(), 1U);
    ASSERT_TRUE(receiver.push_chunk(chunks[1], now_ns).has_value());
    now_ns += 20'000'000ULL;
    for (const auto& chunk : chunks) {
        ASSERT_TRUE(receiver.push_chunk(chunk, now_ns).has_value());
        now_ns += 20'000'000ULL;
    }

    auto decoded = receiver.poll_frame();
    ASSERT_TRUE(decoded.has_value());

    const auto receiver_stats = receiver.stats();
    EXPECT_EQ(receiver_stats.frames_decoded, 1U);
    EXPECT_GT(receiver_stats.out_of_order_chunks, 0U);
    EXPECT_GE(receiver_stats.frames_dropped, 1U);
}
