// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// ReShade's UUID templates require the Windows declarations first on MinGW.
// clang-format off
#include <windows.h>
#include <reshade_api.hpp>
// clang-format on
#include <cstdint>
#include <memory>

namespace sunshine::overlay {
  // One instance belongs to one exporter ring slot. The caller must retain it
  // until that slot's GPU fence completes, including after cancel(). No
  // ReShade wrappers are retained by the destructor: all GPU ownership is COM.
  class compositor_t {
  public:
    compositor_t();
    ~compositor_t();
    compositor_t(const compositor_t &) = delete;
    compositor_t &operator=(const compositor_t &) = delete;

    // Call from the source technique callback, after reserving/copying the slot.
    // Width is ONE eye. Source and destination are both exactly 2*width x height.
    // Destination enters/leaves COMMON on D3D12; source stays shader-readable.
    // A new prepare requires completion of this instance's previous GPU work.
    bool prepare(reshade::api::effect_runtime *runtime, reshade::api::resource_view native_rtv, reshade::api::resource source, reshade::api::resource destination, std::uint32_t width, std::uint32_t height, reshade::api::color_space source_color, reshade::api::color_space export_color);

    // Call from reshade_overlay while the runtime's ImGui context is current.
    // The add-on must register with the pinned ImGui 1.92.5 forwarding table.
    void append_capture_callback();

    // Call from reshade_present, before the slot's ready fence is signaled.
    // False means this slot must NOT be published. No waits or readbacks occur.
    bool finish();
    void cancel();

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace sunshine::overlay
