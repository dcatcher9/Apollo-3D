/**
 * @file src/platform/windows/capture_dxgi_device.h
 * @brief Internal DXGI capture-device scheduling setup.
 */
#pragma once

#include <unknwn.h>

namespace platf::dxgi::detail {
  // Query failure prevents capture initialization; setter failures only warn.
  [[nodiscard]] bool configure_capture_dxgi_device(IUnknown *device);
}  // namespace platf::dxgi::detail
