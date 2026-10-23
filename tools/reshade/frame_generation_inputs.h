// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_ui_mask.h"
#include "scene_depth_source.h"
#include "streamline_depth_capture.h"

#include <array>
#include <cstdint>

// DLSS-G's own inputs of one real frame, read from its NGX evaluation
// (feature 11, NVSDK_NGX_Feature_FrameGeneration) by ngx_depth_source.cpp and
// supplied to that viewport's Streamline FG scope by streamline_camera_probe.cpp.
// The names are NVIDIA's public DLSS SDK (NVIDIA/DLSS 37495948,
// include/nvsdk_ngx_defs_dlssg.h). No API states a resource's state at this
// call, so every copy requires a state observed on the evaluation's own command
// list (state_proof::observed_nonzero); a refusal changes nothing.
namespace sunshine_streamline::frame_generation {
  // A named parameter's getter result. absent: neither the D3D12 resource
  // getter nor the void-pointer getter succeeded (nothing changes); null: a
  // successful getter named no resource (an explicit "none this frame").
  enum class parameter : std::uint8_t { absent, null, present };
  struct resource_input {
    parameter state{parameter::absent};
    std::uint64_t native{}; // Identity only, until the capture owner retains it.
    // The input's own subrectangle; all zero (or every key absent) is the
    // whole allocation. area_valid is false for a partial or malformed one.
    sunshine_scene_depth::extent area;
    bool area_valid{true};
  };
  // ui_mask::capture_order: UIAlpha, UI (premultiplied color and alpha),
  // Backbuffer (the real frame's final color), HUDLess (its scene color).
  inline constexpr std::array<const char *, 4> ui_keys{"DLSSG.UIAlpha", "DLSSG.UI", "DLSSG.Backbuffer", "DLSSG.HUDLess"};
  inline constexpr std::array<const char *, 4> ui_subrects{"DLSSG.UIAlphaSubrect", "DLSSG.UISubrect",
    "DLSSG.InputBackbufferSubrect", "DLSSG.HUDLessSubrect"};
  struct inputs {
    std::uint64_t command{}; // The evaluation's own command list.
    resource_input depth;    // DLSSG.Depth with DLSSG.DepthSubrect*.
    // DLSSG.DepthInverted: nonzero means reversed depth (nearer is larger).
    bool inverted_known{}, inverted{};
    std::array<resource_input, 4> ui{};
  };
  // Why DLSSG.Depth supplied no FG copy. scope_changed: the observation, its
  // epoch or the viewport's metadata changed or was busy. inactive: no depth
  // source is requested. tag_copied: the viewport's newest Streamline FG tag
  // boundary already recorded this frame's copy (the tag keeps precedence;
  // nothing changes). capture: the capture owner refused the
  // copy (attempt::depth_stage names the stage, missing_state when the
  // recording observed no state for it).
  enum class depth_outcome : std::uint8_t {
    not_offered, copied, no_fg_scope, absent, null_resource, subrect, scope_changed, inactive, tag_copied, capture
  };
  constexpr const char *name(depth_outcome value) {
    switch (value) {
      case depth_outcome::copied: return "copied";
      case depth_outcome::no_fg_scope: return "no_fg_scope";
      case depth_outcome::absent: return "absent";
      case depth_outcome::null_resource: return "null";
      case depth_outcome::subrect: return "subrect";
      case depth_outcome::scope_changed: return "scope_changed";
      case depth_outcome::inactive: return "inactive";
      case depth_outcome::tag_copied: return "tag_copied";
      case depth_outcome::capture: return "capture";
      default: return "not_offered";
    }
  }
  // One UI input: absent (nothing changes), null (revokes this origin's
  // snapshots of the kind), a malformed subrect (refused unseen), no FG scope,
  // no live UI request for the viewport, tag_copied (the newest Streamline tag
  // of that kind for the viewport recorded a copy within the source age: the
  // tag keeps precedence per kind, as for depth, and nothing changes), or
  // offered to ui_mask::begin (the attempt names its refusal or record stage).
  enum class ui_outcome : std::uint8_t { absent, null_resource, subrect, no_fg_scope, no_request, tag_copied, offered };
  constexpr const char *name(ui_outcome value) {
    switch (value) {
      case ui_outcome::null_resource: return "null";
      case ui_outcome::subrect: return "subrect";
      case ui_outcome::no_fg_scope: return "no_fg_scope";
      case ui_outcome::no_request: return "no_request";
      case ui_outcome::tag_copied: return "tag_copied";
      case ui_outcome::offered: return "offered";
      default: return "absent";
    }
  }
  struct attempt {
    std::uint64_t observation{}, loss{}, depth_ticket{};
    depth_outcome depth{depth_outcome::not_offered};
    depth_capture::record_stage depth_stage{};
    std::array<ui_outcome, 4> ui_result{};
    std::array<sunshine_game3d::ui_mask::attempt, 4> ui{};
  };
}
