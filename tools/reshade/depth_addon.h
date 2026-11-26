// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>
#include <Unknwn.h>
#include <reshade_api.hpp>
#include "depth_orientation.h"
#include "raw_scene_policy.h"
#include "raw_scene_pool.h"
#include "camera_sample_binding.h"
#include "streamline_camera_probe.h"
#include "scene_depth_source.h"
#include "depth_frame.h"

namespace sunshine_streamline::depth_capture { struct packet; struct consumer_diagnostic; }

namespace sunshine_depth {
  using raw_source_roster = sunshine_raw_scene::source_roster;
  // Call after the containing Sunshine 3D module has registered with ReShade.
  // An inactive selector still exposes its setup message in the overlay; keep
  // the containing add-on loaded when this returns false.
  bool initialize();
  void shutdown();
  bool active();

  // Native presentation owns the same capture lease without requiring an FX
  // technique. Enabled native rendering suppresses duplicate effects callbacks.
  void set_native_driver(reshade::api::effect_runtime *runtime, bool enabled);
  bool begin_native_frame(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands);
  void end_native_frame(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands);

  // Read inside this runtime's native frame (begin_native_frame to
  // end_native_frame). Handles are borrowed for that frame only; the selected
  // resource is then in shader_resource state. False clears all
  // handles; ready means safe to sample in THIS pass. reused_depth distinguishes
  // bounded FG reuse from fresh depth. The query is passive: scene selection,
  // histogram evidence and capture state remain unchanged.
  bool get_frame_depth(reshade::api::effect_runtime *runtime, frame_depth &output);

  // Exporter-owned opt-in basis. Call each native frame; disabled releases only
  // this raw-scene request, preserving the selector's own sampling.
  void set_raw_scene_request(reshade::api::effect_runtime *runtime, std::uint64_t basis_epoch, bool enabled);
  // Completed immutable CPU evidence, independent of current rendering. Each
  // sample keeps its own source/basis/layout/runtime and authenticated readback.
  std::size_t get_raw_samples(reshade::api::effect_runtime *runtime,
    std::array<sunshine_raw_scene::sample, sunshine_raw_scene::maximum_sources> &output);
  // Membership only, never fabricated current-frame readiness or GPU handles.
  bool get_raw_source_roster(reshade::api::effect_runtime *runtime, raw_source_roster &output);
  // Validate retained numeric histories against actual resource lifetime and
  // layout, independently of which resources currently own capture slots.
  // The mask indexes the supplied immutable history snapshot; no GPU access.
  std::uint32_t validate_raw_history(reshade::api::effect_runtime *runtime,
    const std::array<sunshine_raw_scene::source_basis, sunshine_raw_scene::maximum_history_sources> &history);
}
