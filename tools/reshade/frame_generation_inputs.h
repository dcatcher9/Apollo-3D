// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "freshness.h"
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
  // source is requested. tag_copied: the frame it evaluates has its own
  // Streamline FG tag copy, recorded or waiting for that frame's Present (the
  // tag keeps precedence, depth_tag_precedence; nothing changes). capture: the
  // capture owner refused the
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
  // tag keeps precedence per kind, and nothing changes), or
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
  // The frame a Streamline call belongs to, as DLSS-G's depth precedence
  // compares frames: key, its frame's identity (unique within the
  // observation; 0 when the call named no frame the probe observed), and
  // sequence, the call's observation sequence.
  struct frame_mark {
    std::uint64_t key{}, sequence{};
  };
  // Whether a tag call belongs to frame, the one DLSS-G evaluates: the
  // viewport's newest frame, marked by the constants call that began it.
  // When both name a frame, the tag names that one; otherwise the tag came
  // after that constants call.
  constexpr bool of_frame(const frame_mark &tag, const frame_mark &frame) {
    return tag.key && frame.key ? tag.key == frame.key : tag.sequence > frame.sequence;
  }
  // The precedence of a viewport's Streamline FG depth tags over DLSS-G's own
  // DLSSG.Depth (depth_outcome::tag_copied), per FG scope (epoch and
  // viewport). An FG tag copies its frame first, so DLSS-G's evaluation of
  // that frame adds no second copy: when the scope's newest FG tag copy was
  // recorded for the evaluated frame, or a tag of that frame waits for the
  // frame's Present to be copied (pending; the deferred copy, R2), while that
  // newest copy is younger than the source age. A refused or aged tag copy
  // hands the frame to DLSS-G. Every later frame without an FG tag of its own
  // is DLSS-G's: until C6 closed (10-26) any tag copy younger than the source
  // age kept those frames' DLSSG.Depth out too, so DLSS-G's fresher depth was
  // copied only once that copy aged out (on its compute list, readable three
  // evaluations later, past the copy's hold).
  struct depth_tag_precedence {
    std::uint64_t epoch{}, tick{};
    std::uint32_t viewport{};
    frame_mark frame; // The newest copy's tag call.

    // An FG tag call's depth outcome for its scope: a copy recorded at now
    // (tag: the call), or none (refused, cancelled or lapsed).
    void remember(std::uint64_t scope_epoch, std::uint32_t scope_viewport, bool recorded, const frame_mark &tag,
        std::uint64_t now) {
      if (recorded) *this = {scope_epoch, now, scope_viewport, tag};
      else if (epoch == scope_epoch && viewport == scope_viewport) tick = 0;
    }
    // Whether DLSS-G's evaluation of the scope's frame evaluated stands aside
    // (adds no copy). pending: the newest tag call whose copy waits for its
    // frame's Present, or null.
    bool tags_supply(std::uint64_t scope_epoch, std::uint32_t scope_viewport, const frame_mark &evaluated,
        const frame_mark *pending, std::uint64_t now) const {
      if (!tick || epoch != scope_epoch || viewport != scope_viewport ||
          !sunshine_freshness::ahead_tolerant_lt(tick, now, sunshine_freshness::source_age_ms)) return false;
      return of_frame(frame, evaluated) || (pending && of_frame(*pending, evaluated));
    }
  };
}
