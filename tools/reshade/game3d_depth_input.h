// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "depth_addon.h"
#include "depth_moments.h"

#include <array>

// Presentation-facing depth/camera input. Source adapters own selection,
// preservation and frame association; stereo consumes their paired evidence.
namespace sunshine_game3d::depth_input {
  struct sample {
    sunshine_scene_depth::frame metadata;
    sunshine_depth::frame_depth::projection_t projection;
    std::uint64_t id{}, tick{};
    // Exact point grid for diagnostics only; live scene geometry uses moments.
    std::uint32_t width{}, height{};
    std::array<float, sunshine_depth_statistics::maximum_tiles> raw{};
    bool range_valid{};
    float range_min{}, range_max{};
    sunshine_depth_statistics::moments moments{};
  };

  // Borrowed handles are valid only inside the depth owner's current read lease.
  // The projection and provenance belong to these exact captured pixels. This
  // does not perform a second camera query or infer a source from a loaded SDK.
  // The ordinary Game 3D path is passive; only the independent reference effect
  // explicitly requests the legacy raw-depth calibration.
  bool query(reshade::api::effect_runtime *runtime, sunshine_depth::frame_depth &out,
    sunshine_depth::depth_orientation orientation = sunshine_depth::depth_orientation::automatic,
    bool request_calibration = false);

  // Provider ownership and readiness are deliberately distinct. A selected API
  // source with pending pixels must not silently enter Generic calibration.
  bool provider_selected(reshade::api::effect_runtime *runtime);

  // Completed immutable evidence retains its original identity, projection and
  // capture tick. Repeated reads do not create fresh calibration observations.
  bool latest_sample(reshade::api::effect_runtime *runtime, sample &out);

  // Focus/technique/runtime invalidation ends reuse eligibility, while keeping
  // the existing capture-owner lifetime and numerical-history rules unchanged.
  void invalidate_reuse(reshade::api::effect_runtime *runtime);
  void set_ready(reshade::api::effect_runtime *runtime, bool ready);

  // Passive hook discovery and optional camera/effects-input diagnostics.
  // The selected frame is immutable input; this cannot select a different
  // resource, enable calibration or associate unrelated camera/depth evidence.
  void observe(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands,
    reshade::api::resource_view effects_rtv, const sunshine_depth::frame_depth &depth);

  // Keep source-observer lifecycle ordering explicit at add-on initialization
  // and shutdown, independently of renderer and per-runtime resource lifetime.
  void initialize_observers(HMODULE addon);
  void shutdown_observers();
}
