// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "foreground_window_region.h"

#include <chrono>
#include <cstdint>
#include <d3d11.h>
#include <functional>
#include <memory>
#include <optional>

namespace platf::reshade_bridge {
  struct frame_t {
    ID3D11Texture2D *texture = nullptr;
    ID3D11ShaderResourceView *view = nullptr;
    bool linear = false;
    std::chrono::steady_clock::time_point timestamp {};
    std::uint64_t sequence = 0;
    // Validated publisher identity. Stable across frames within one resource generation;
    // unlike sequence, it distinguishes a restarted/replaced source for readiness revisions.
    std::uint32_t producer_process_id = 0;
    std::uint64_t producer_creation_time = 0;
    std::uint64_t resource_generation = 0;
  };

  class receiver_t {
  public:
    // Tests can supply foreground observations without moving real windows or displays.
    using observer_t = std::function<foreground_window::observation_t()>;
    receiver_t(ID3D11Device *device, ID3D11DeviceContext *context, observer_t observe = foreground_window::sample);
    ~receiver_t();
    receiver_t(const receiver_t &) = delete;
    receiver_t &operator=(const receiver_t &) = delete;

    // Nonblocking. A returned frame is private to this receiver and may be reused while its
    // exact foreground producer remains valid. Pointers live until the next poll/destruction.
    // Foreground ownership must cover source_rect; the authored raster must match output_width
    // and output_height exactly. Fullscreen display scaling can make those extents independent.
    std::optional<frame_t> poll(RECT source_rect, int output_width, int output_height);

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::reshade_bridge
