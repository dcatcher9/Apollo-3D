// SPDX-License-Identifier: GPL-3.0-only
#include "hook_host.h"

#include "streamline_camera_probe.h"
#include "streamline_depth_capture.h"
#include "upscaler_call_trace.h"
#include "game3d_clock.h"

namespace sunshine_game3d::hook_host {
  void poll() {
    // Shared capture observation also runs when every API source is disabled.
    // Deferred hook discovery separately serves lightweight source nomination.
    if (!sunshine_streamline::depth_capture::active() &&
        !sunshine_streamline::source_enabled() && !sunshine_upscaler_trace::enabled() &&
        !sunshine_upscaler_trace::capture_enabled()) return;
    const bool api_requested = sunshine_streamline::middleware_requested_for_poll() || sunshine_upscaler_trace::enabled() ||
      sunshine_upscaler_trace::capture_enabled();
    if (!api_requested && !sunshine_streamline::depth_capture::active()) return;
    auto guard = sunshine_streamline::try_poll();
    if (!guard) return;
    const auto now = sunshine_game3d::clock::game_now();
    sunshine_streamline::discovery_step(api_requested, now);
    sunshine_streamline::coverage_step();
    sunshine_upscaler_trace::poll();
    sunshine_streamline::camera_availability_step(now);
    // The common copy owner also serves Generic depth without an upscaler API.
    // Middleware presence and selection preferences cannot gate its lifecycle.
    if (sunshine_streamline::depth_capture::active()) {
      sunshine_streamline::depth_capture_step();
      sunshine_streamline::queue_order_step(now);
    }
    sunshine_streamline::pending_installs_step();
  }

  void initialize(HMODULE addon) {
    sunshine_streamline::initialize(addon);
  }

  void shutdown() {
    sunshine_streamline::shutdown();
  }
}
