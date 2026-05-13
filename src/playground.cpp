#include "chiral/gimbal.hpp"
#include "spdlog_hook.hpp"
#include "talos_gimbal/mcu_device.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

using namespace std::chrono_literals;

// Global signal handler
std::atomic<bool> g_running{true};

void signal_handler(int) { g_running = false; }

int main() {
    // Setup signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    init_logger();
    SPDLOG_INFO("=== Talos Playground ===");
    SPDLOG_INFO("Press Ctrl+C to quit");
    const auto start = std::chrono::system_clock::now();

    SPDLOG_INFO("creating shared memory regions...");
    auto client = talos_gimbal::McuDeviceHandle::create_usb(0x0483, std::nullopt);
    if (!client) {
        SPDLOG_CRITICAL("failed to connect to usb");
        return 1;
    }

    uint64_t frame_count = 0;
    std::array<uint8_t, 300> bytes{};

    bytes.front() = 0x6A;
    bytes.back()  = 0xA6;

    for (std::size_t i = 1; i + 1 < bytes.size(); ++i) {
        //    bytes[i] = static_cast<uint8_t>(i - 1);
    }

    bytes[51] = 'R';
    bytes[52] = 'o';
    bytes[53] = 'b';
    bytes[54] = 'o';
    bytes[55] = 'M';
    bytes[56] = 'a';
    bytes[57] = 's';
    bytes[58] = 't';
    bytes[59] = 'e';
    bytes[60] = 'r';

    bytes[184] = 'A';
    bytes[185] = 'C';
    bytes[186] = 'T';
    bytes[187] = 'O';
    bytes[188] = 'R';

    bytes[190] = 'T';
    bytes[191] = 'h';
    bytes[192] = 'i';
    bytes[193] = 'n';
    bytes[194] = 'k';
    bytes[195] = 'e';
    bytes[196] = 'r';

    bytes[297] = 'A';
    bytes[298] = 'C';
    bytes[299] = 'K';

    // Main loop
    SPDLOG_INFO("starting main loop...");

    // evil hack for extended data length
    constexpr auto data_length_raw = sizeof(talos_gimbal::SendQuantaData::data);
    // maximum extended data length is stored as u12
    static_assert(data_length_raw < 4096);
    // lower bits
    // higher bits, 4bits available
    constexpr uint8_t data_length_hi = data_length_raw >> 8;
    // double check for only 4bits
    static_assert((data_length_hi & 0b1111) == data_length_hi);
    constexpr uint8_t data_length_protocol = data_length_raw & 0b11111111;
    // actual id is 0x04
    constexpr uint8_t data_id_protocol = (data_length_hi << 4) | 0x04;

    constexpr std::size_t kMaxCustomBlockBytes = 300;

    while (g_running) {
        // 测试 2: 接收指令
        frame_count++;

        const auto* ptr = bytes.data();

        std::size_t remaining = bytes.size();
        std::size_t offset    = 0;

        while (remaining > 0) {
            const std::size_t this_len = std::min(remaining, kMaxCustomBlockBytes);

            talos_gimbal::SendQuantaData packet{
                .header =
                    {
                             .sof = talos_gimbal::HeaderFrame::SoF(),
                             .len = data_length_protocol,
                             .id  = data_id_protocol,
                             },
                // ffs
                .time_stamp = static_cast<uint32_t>(frame_count),
                .data =
                    {
                             .custom_byte_block_len = static_cast<uint16_t>(this_len),
                             .custom_byte_block     = {},
                             },
                .eof = talos_gimbal::HeaderFrame::EoF(),
            };

            std::memcpy(packet.data.custom_byte_block, ptr + offset, this_len);
            packet.data.custom_byte_block[19] = frame_count;
            client->send_sync(reinterpret_cast<uint8_t*>(&packet), sizeof(packet));

            offset += this_len;
            remaining -= this_len;
        }
        std::this_thread::sleep_for(20ms); // ~30 FPS
    }

    talos_gimbal::SendQuantaData packet{
        .header =
            {
                     .sof = talos_gimbal::HeaderFrame::SoF(),
                     .len = data_length_protocol,
                     .id  = data_id_protocol,
                     },
        // ffs
        .time_stamp = static_cast<uint32_t>(frame_count),
        .data =
            {
                     .custom_byte_block_len = static_cast<uint16_t>(10),
                     .custom_byte_block     = {},
                     },
        .eof = talos_gimbal::HeaderFrame::EoF(),
    };
    std::memset(packet.data.custom_byte_block, 0, 300);

    client->send_sync(reinterpret_cast<uint8_t*>(&packet), sizeof(packet));

    // Cleanup
    SPDLOG_INFO(
        "shutting down... ({} frames, {}ms)", frame_count,
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start)
            .count());
    client.reset();
    return 0;
}
