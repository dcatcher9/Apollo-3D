// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <reshade.hpp>
#include <cstdint>
#include <vector>

// Diagnostic census of offscreen UI layers. HDR pipelines (Frostbite; Unreal's
// HDR UI composite) draw UI into a color target at the output resolution that
// is cleared to transparent black each frame and composited in the final pass.
// Its alpha is a per-pixel UI mask that no vendor API exposes. While a Dump 3D
// is armed, qualifying clears are recorded and the target is copied just before
// its clear erases it, so a dump shows the previous frame's layer. Nothing here
// feeds detection; see docs/reshade-sbs.md.
namespace sunshine_game3d::ui_layer {
  namespace api = reshade::api;
  inline constexpr unsigned max_candidates = 3; // game3d_debug::ui_layer_count

  // Formats that can carry UI coverage in alpha.
  bool alpha_format(api::format format);
  // A single-sample 2D color target at the output size cleared to exactly
  // (0, 0, 0, 0). Callers separately exclude swapchain back buffers.
  bool qualifies(const api::resource_desc &desc, const float color[4], std::uint32_t width, std::uint32_t height);

  struct candidate {
    api::resource copy{};    // Add-on owned, shader_resource state; empty unless captured.
    std::uint64_t source{};  // Game resource handle, identity only.
    std::uint32_t width{}, height{}, format{}; // DXGI format of the copy.
    std::uint32_t clears{};  // Qualifying clears seen while armed.
    const char *status = "observed";
  };

  void register_events();
  void unregister_events();
  // The foreground swapchain while a dump is pending: output size and back buffers.
  void observe_output(api::swapchain *swapchain);
  void arm();
  // Stops the census and hands over its candidates. Each copy stays valid
  // while its device lives; the caller either keeps a reference and destroys
  // its handle, or passes it to retire().
  std::vector<candidate> take(api::device *&device);
  // Destroys a copy once any game command list that wrote it has executed.
  void retire(api::device *device, api::resource copy);
  // Drops an armed census whose dump was withdrawn.
  void cancel();
}
