#pragma once
namespace fcs::L3::ldm {
struct LdmState {
    uint64_t timestamp_ns{0};
    bool accurate{false};
    Eigen::Vector3d position;
};
} // namespace fcs::L3::ldm
