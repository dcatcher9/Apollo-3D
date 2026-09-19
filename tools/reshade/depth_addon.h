// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>
#include <Unknwn.h>
#include <reshade_api.hpp>
#include "native_depth_calibration.h"
#include "streamline_command_association.h"
#include "streamline_depth_content.h"
#include "raw_scene_policy.h"
#include "raw_scene_pool.h"
#include "camera_sample_binding.h"
#include "streamline_camera_probe.h"
#include "scene_depth_source.h"

namespace sunshine_streamline::depth_capture { struct packet; struct consumer_diagnostic; }

namespace sunshine_depth {
  using raw_source_roster = sunshine_raw_scene::source_roster;
  // Call after the containing Sunshine 3D module has registered with ReShade.
  // An inactive selector still exposes its setup message in the overlay; keep
  // the containing add-on loaded when this returns false.
  bool initialize(HMODULE addon_module);
  void shutdown();
  bool active();
  const char *status_message();

  // Native presentation owns the same capture lease without requiring an FX
  // technique. Enabled native rendering suppresses duplicate effects callbacks.
  void set_native_driver(reshade::api::effect_runtime *runtime, bool enabled);
  bool begin_native_frame(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands);
  void end_native_frame(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands);

  struct frame_depth {
    // Value-only API metadata, separate from the borrowed rendering resources.
    sunshine_scene_depth::frame provided;
    struct projection_t {
      bool supplied = false;
      std::uint64_t epoch = 0;
      std::uint32_t viewport = 0;
      double A = 0, B = 0;
      double raw_scale = 1, raw_bias = 0;
    } projection;
    // Original game resource, before any copy-before-clear backup. Identity-only
    // observation for camera metadata matching; never sampled through this handle.
    reshade::api::resource source_resource {};
    reshade::api::resource resource {};
    reshade::api::resource_view shader_resource {};
    std::uint32_t width = 0, height = 0;
    std::uint32_t x = 0, y = 0, active_width = 0, active_height = 0;
    std::uint64_t source_id = 0, layout_epoch = 0, frame_index = 0;
    std::uint64_t runtime_epoch = 0;
    bool aligned_viewport_assumed = false;
    // Observed command recording for the actual preserved copy. This is not a
    // source-content or final-color correspondence proof.
    sunshine_streamline::commands::recording_marker capture_marker {};
    sunshine_streamline::content::copy_snapshot depth_copy {};
    std::uint64_t backup_id = 0;
    std::uint64_t command_queue = 0;
    bool ready = false;
    // FG may sample the last real frame's private display depth with new color.
    // Its metadata sequence/tick remain original; this is not fresh depth evidence.
    bool reused_depth = false;
    // Provider-resolved FG policy for this source/pass. A busy options query
    // retains confirmed mode; lost or ambiguous scope cannot authorize reuse.
    bool frame_generation_active = false;
    depth_orientation detected_orientation = depth_orientation::automatic;
    depth_orientation orientation = depth_orientation::automatic;
    unsigned orientation_agreeing_frames = 0;
    bool calibrated = false;
    unsigned calibration_samples = 0;
    std::uint64_t calibration_frame = 0;
    // Positive nearness is toward the viewer: (raw - raw_anchor) * raw_gain.
    float raw_anchor = 0, raw_gain = 0;
  };

  // Resolve the selected source's capture opportunity, then publish through
  // the shared owner's read lease. Renderer/calibration never choose a capture
  // implementation or issue texture transitions. Generic uses the same owner.
  bool copy_selected_depth(reshade::api::effect_runtime *runtime,
    reshade::api::command_list *commands, const sunshine_streamline::depth_capture::packet &selection,
    reshade::api::resource destination, frame_depth &captured,
    sunshine_streamline::depth_capture::consumer_diagnostic *diagnostic = nullptr);

  // Shared value-only conversion for both effects diagnostics and the exact
  // preserved source/backup that is about to be sampled. No GPU calls or proof.
  inline sunshine_streamline::selected_depth camera_selection_snapshot(const frame_depth &depth) {
    sunshine_streamline::selected_depth selected;
    selected.resource = depth.source_resource.handle; selected.source_id = depth.source_id;
    selected.layout_epoch = depth.layout_epoch; selected.frame_index = depth.frame_index;
    selected.capture_marker = depth.capture_marker; selected.command_queue = depth.command_queue;
    selected.depth_copy = depth.depth_copy; selected.backup_resource = depth.resource.handle; selected.backup_id = depth.backup_id;
    selected.width = depth.width; selected.height = depth.height; selected.x = depth.x; selected.y = depth.y;
    selected.active_width = depth.active_width; selected.active_height = depth.active_height; selected.ready = depth.ready;
    return selected;
  }

  // Read immediately after this module's reshade_begin_effects callback and
  // before reshade_finish_effects. Handles are borrowed for that callback only;
  // the selected resource is then in shader_resource state. False clears all
  // handles; ready means safe to sample in THIS pass. reused_depth distinguishes
  // bounded FG reuse from fresh depth; calibrated describes older raw samples.
  // Set request_calibration=false for passive capture-identity observation. This
  // releases any earlier independent-renderer request and clears its status UI;
  // scene selection, histogram evidence and capture state remain unchanged.
  bool get_frame_depth(reshade::api::effect_runtime *runtime, frame_depth &output,
    depth_orientation orientation_override = depth_orientation::automatic,
    bool request_calibration = true);

  // Exporter-owned opt-in basis. Call each begin_effects; disabled releases only
  // this raw-scene request, preserving legacy selector/calibration sampling.
  void set_raw_scene_request(reshade::api::effect_runtime *runtime, std::uint64_t basis_epoch, bool enabled);
  // Value copy of a completed, submission-bound selected-source readback. No
  // camera projection or final-color registration proof is implied. Repeated
  // reads retain the same sample ID; the policy rejects duplicate observations.
  bool get_latest_raw_sample(reshade::api::effect_runtime *runtime, sunshine_raw_scene::sample &output);
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
  // Probe-only value copy; changed selection/lifetime/layout and stale samples
  // fail closed. This API cannot enable either numerical policy or rendering.
  bool get_latest_camera_observation(reshade::api::effect_runtime *runtime,
    sunshine_camera_binding::submission &output, std::uint64_t &capture_id);
  // Throttled lookup of up to three observed depth tags in the existing resource
  // map, and reporting of submission-bound observations. Never selects a buffer.
  void report_camera_observations(reshade::api::effect_runtime *runtime,
    const sunshine_streamline::selected_depth &selected);
}
