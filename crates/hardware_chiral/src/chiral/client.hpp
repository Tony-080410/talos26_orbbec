#pragma once
#include "chiral/chiral_endpoint.hpp"
#include "packet.hpp"
namespace talos::chiral::gimbal {

using TalosSide = ipc::ChiralEndpoint<McuRequest, McuData>;
using RadarSide = ipc::ChiralEndpoint<McuData, McuRequest>;

} // namespace talos::chiral::gimbal
namespace talos::chiral::ipc {

template <>
struct ShmName<gimbal::McuRequest> {
    static constexpr const char* value = "/chiral_mcu_request";
};

template <>
struct ShmName<gimbal::McuData> {
    static constexpr const char* value = "/chiral_mcu_data";
};
}; // namespace talos::chiral::ipc
