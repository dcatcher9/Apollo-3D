// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_depth_input.h"

#include "streamline_camera_probe.h"
#include "streamline_depth_capture.h"
#include "streamline_depth_provider.h"
#include "upscaler_call_trace.h"

namespace sunshine_game3d::depth_input {
  namespace api = reshade::api;

  bool query(api::effect_runtime *runtime, sunshine_depth::frame_depth &out,
      sunshine_depth::depth_orientation orientation, bool request_calibration) {
    return sunshine_depth::get_frame_depth(runtime, out, orientation, request_calibration);
  }

  bool provider_selected(api::effect_runtime *runtime) {
    return sunshine_streamline::provider::selected(runtime);
  }

  bool latest_sample(api::effect_runtime *runtime, sample &out) {
    return sunshine_streamline::provider::latest_center(runtime, out);
  }

  void invalidate_reuse(api::effect_runtime *runtime) {
    sunshine_streamline::provider::invalidate_reused_depth(runtime);
  }

  void set_ready(api::effect_runtime *runtime, bool ready) {
    sunshine_streamline::provider::set_depth_ready(runtime, ready);
  }

  void observe(api::effect_runtime *, api::command_list *, api::resource_view, const sunshine_depth::frame_depth &) {
    // Shared capture observation also runs when every API source is disabled.
    // Deferred hook discovery separately serves lightweight source nomination.
    if (!sunshine_streamline::depth_capture::active() &&
        !sunshine_streamline::source_enabled() && !sunshine_upscaler_trace::enabled() &&
        !sunshine_upscaler_trace::capture_enabled()) return;
    sunshine_streamline::poll();
  }

  void initialize_observers(HMODULE addon) {
    sunshine_streamline::initialize(addon);
  }

  void shutdown_observers() {
    sunshine_streamline::shutdown();
  }
}
