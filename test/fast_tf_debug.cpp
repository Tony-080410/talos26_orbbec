#include "fast_tf/mod.hpp"
#include <iostream>

using namespace fast_tf;

int main() {
    using Buffer = CircularTransformBuffer<10>;
    Buffer buffer;

    // Create simple transforms
    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 2.0, 0.0, 0.0);
    auto tf3 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 3.0, 0.0, 0.0);

    // Push three transforms
    std::cout << "Pushing 3 transforms...\n";
    buffer.push(1000, tf1);
    buffer.push(2000, tf2);
    buffer.push(3000, tf3);

    // Try to retrieve each
    for (uint64_t ts : {1000UL, 2000UL, 3000UL}) {
        std::cout << "Querying at timestamp " << ts << "...\n";
        auto result = buffer.at(ts);
        if (result.has_value()) {
            std::cout << "  Found at (" << result->transform.translation().x() << ")\n";
        } else {
            std::cout << "  NOT FOUND\n";
        }
    }

    // Check time range
    std::cout << "\nTime range:\n";
    auto range = buffer.time_range();
    if (range.has_value()) {
        std::cout << "  Min: " << range->first << ", Max: " << range->second << "\n";
    }

    // Check size via latest
    std::cout << "\nLatest:\n";
    auto latest = buffer.latest();
    if (latest.has_value()) {
        std::cout << "  Timestamp: " << latest->timestamp_ns << "\n";
        std::cout << "  Translation X: " << latest->transform.translation().x() << "\n";
    }

    return 0;
}
