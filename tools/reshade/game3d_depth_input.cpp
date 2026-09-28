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

  void observe(api::effect_runtime *runtime, api::command_list *commands,
      api::resource_view effects_rtv, const sunshine_depth::frame_depth &depth) {
    // Shared capture observation also runs when every API source is disabled.
    // Deferred hook discovery separately serves lightweight source nomination.
    // Detailed camera/content observations remain diagnostic-only.
    if (!sunshine_streamline::depth_capture::active() && !sunshine_streamline::enabled() &&
        !sunshine_streamline::source_enabled() && !sunshine_upscaler_trace::enabled() &&
        !sunshine_upscaler_trace::capture_enabled()) return;
    auto selected = sunshine_depth::camera_selection_snapshot(depth);
    // ReShade can resolve/copy the swapchain before effects. Observe the actual
    // effects input while its view is valid, never a guessed backbuffer. This
    // is identity/extent evidence, not proof that color and depth share UVs.
    if (sunshine_streamline::enabled() && effects_rtv.handle && commands) {
      auto *device = runtime->get_device();
      const auto resource = device->get_resource_from_view(effects_rtv);
      if (resource.handle) {
        const auto desc = device->get_resource_desc(resource);
        if (desc.type == api::resource_type::texture_2d || desc.type == api::resource_type::surface) {
          auto &input = selected.effects_input;
          input.runtime = reinterpret_cast<std::uint64_t>(runtime);
          input.device = device->get_native();
          input.command = commands->get_native();
          input.resource = resource.handle;
          input.width = desc.texture.width;
          input.height = desc.texture.height;
          input.format = static_cast<std::uint32_t>(desc.texture.format);
          input.ready = input.width && input.height;
        }
      }
    }
    sunshine_streamline::poll(selected);
    if (sunshine_streamline::enabled())
      sunshine_depth::report_camera_observations(runtime, selected);
  }

  void initialize_observers(HMODULE addon) {
    sunshine_streamline::initialize(addon);
  }

  void shutdown_observers() {
    sunshine_streamline::shutdown();
  }
}
