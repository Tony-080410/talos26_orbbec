#include "quanta/transcoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
    quanta::EncodeParams params;

    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        std::fprintf(
            stderr, "  Encodes input video to H.265/HEVC @ %.1f kbps ultra-low bitrate.\n",
            static_cast<double>(params.target_bitrate) / 1000.0);
        return 1;
    }

    quanta::Transcoder t(params);
    auto result = t.run(argv[1], argv[2]);

    if (!result) {
        std::fprintf(stderr, "[quanta] ERROR: %s\n", result.error().c_str());
        return 1;
    }

    auto& s = result.value();
    std::fprintf(stderr, "\n[quanta] Done!\n");
    std::fprintf(
        stderr, "  Frames:    %lld in -> %lld out\n", static_cast<long long>(s.frames_in),
        static_cast<long long>(s.frames_out));
    std::fprintf(stderr, "  Duration:  %.2f sec\n", s.duration_sec);
    std::fprintf(stderr, "  Size:      %.1f KB\n", static_cast<double>(s.bytes_out) / 1024.0);
    std::fprintf(stderr, "  Avg rate:  %.1f kbps\n", s.avg_bitrate);

    return 0;
}
