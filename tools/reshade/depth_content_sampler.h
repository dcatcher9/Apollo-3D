// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include <unknwn.h> // Defines MinGW's __uuidof helper before the ReShade SDK.
#include <reshade.hpp>
#include "camera_sample_binding.h"

namespace sunshine_depth {
  struct sample_request {
    std::uint64_t token = 0;
    // Independent immutable sample association. Legacy token remains source ID.
    std::uint64_t capture_id = 0, source_lifetime = 0;
    sunshine_camera_binding::scope_key capture_scope;
    reshade::api::resource source {};
    reshade::api::resource_usage before = reshade::api::resource_usage::copy_dest;
    std::uint32_t source_width = 0, source_height = 0;
    // A nonempty rectangle fully inside the source is sampled. Otherwise use the
    // full source. Coordinates are pixels of the original depth backup.
    std::uint32_t x = 0, y = 0, width = 0, height = 0;
  };

  struct sample_result {
    std::uint64_t token = 0;
    std::uint64_t capture_id = 0, source_lifetime = 0;
    sunshine_camera_binding::scope_key capture_scope;
    // Native source/format are read from the actual retained sampling resource;
    // scope/lifetime above are the owner's immutable submission observations.
    reshade::api::resource source {};
    std::uint32_t source_format = 0;
    bool valid = false;
    std::uint32_t width = 32, height = 18;
    std::uint32_t source_width = 0, source_height = 0;
    std::uint32_t viewport_x = 0, viewport_y = 0, viewport_width = 0, viewport_height = 0;
    // Row-major raw hardware depth; failure never returns fabricated flat data.
    std::vector<float> values;
  };

  // Call after end_render_effects has restored the tracked backup's state. This
  // prepares a PRIVATE command list; it does not alter ReShade/game GPU state.
  // The backup must remain in `before` until the matching finish_present event.
  // Unsupported formats, stale sizes and a busy sampler return false.
  bool submit(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands,
              const sample_request &request);
  bool busy();
  std::optional<sample_result> poll(reshade::api::effect_runtime *runtime);
  void begin_present(reshade::api::swapchain *swapchain);
  void finish_present(reshade::api::command_queue *queue, reshade::api::swapchain *swapchain);
  // Discards completed or unsubmitted work; submitted work retains every native
  // resource until its fence/query completes, even after the runtime disappears.
  // Also releases this runtime's idle immutable-pipeline cache. The cache holds
  // one retained native device; pending samples own their program independently.
  void retire_runtime(reshade::api::effect_runtime *runtime);
}
