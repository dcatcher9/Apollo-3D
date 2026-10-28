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
  // The viewports the Streamline probe keeps state for (its viewport
  // records and deferred frames); every per-viewport table of an FG scope
  // is sized by it, and the one tagged longest ago gives way to a fifth.
  inline constexpr unsigned viewport_slots = 4;
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
  // Whether a viewport's constants call with key begins a frame after the
  // one keyed previous: a new key, or none (every unkeyed call begins one).
  // A repeated key is the same frame.
  constexpr bool frame_begins(std::uint64_t previous, std::uint64_t key) { return !key || key != previous; }
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
  // is DLSS-G's (docs/reshade-sbs.md, DLSS-G's own inputs; its history in the
  // README's C6 paragraph).
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
  // Which Present a viewport's copy deferred to its frame's Present (R2, the
  // probe's present_deferred) is made at while it generates frames. DLSS-G
  // presents earlier frames (paced real ones and generated ones) while the
  // game submits the next frame's work, so a game execution proves nothing,
  // and it evaluates a frame inside that frame's Present, after the frame's
  // tags. So the frame's Present is the first after DLSS-G evaluated a frame
  // following the deferral's newest tag call, and the deferral waits for that
  // evaluation while DLSS-G evaluates every frame: an evaluation in the
  // deferral's scope within the source age, and one since the viewport's
  // previous deferred frame's newest tag call, which is that frame's own
  // (its evaluation comes before its Present, which comes before this frame's
  // tag calls or with DLSS-G's pacing after them). With no earlier deferred
  // frame of the viewport in its epoch, it waits as well. Otherwise the
  // frame's Present is the first after a game list executed on the presenting
  // queue since the tag call (queue_executed); a deferral of a frame DLSS-G
  // does evaluate is still made at the first Present after that evaluation
  // (docs/reshade-sbs.md, copies deferred to the Present; its history in the
  // README's C7 paragraph).
  // Present ordinals (depth_capture::next_present_ordinal) are stamped at a
  // tag call and at an evaluation: the ordinal of the Present that presents
  // the frame being recorded then. A Present counts the frame's own while no
  // more Presents came since that stamp than the earlier frame's generated
  // ones DLSS-G may still present before it, however long the frame took:
  // the source age bounds a deferral or an evaluation only at a Present that
  // cannot be its frame's (10-28; until then a hitch of the source age
  // between a frame's calls and its Present lapsed its deferred FG copy at
  // that Present, and DLSS-G's evaluation of the frame had stood aside for
  // it, so the Present was mono: open item A, seed 2d28df102b516f4d).
  struct deferral_evaluations {
    // DLSS-G's NGX evaluations of real frames: their count, and the newest
    // one's scope, tick, Present ordinal and generated count (the FG
    // options' numFramesToGenerate then).
    std::uint64_t count{}, epoch{}, tick{}, present{};
    std::uint32_t viewport{}, generated{};
    // A deferral's counts: at its frame's newest tag call (at_tag), and at the
    // viewport's previous deferred frame's newest tag call (previous, when
    // previous_known); present: the newest tag call's Present ordinal.
    struct marks {
      std::uint64_t at_tag{}, previous{}, present{};
      bool previous_known{};
    };
    // Per viewport (viewport_slots; the oldest gives way), the count at its
    // newest deferred frame's newest tag call.
    struct tagged_frame {
      bool used{};
      std::uint64_t epoch{}, serial{}, at_tag{};
      std::uint32_t viewport{};
    };
    std::array<tagged_frame, viewport_slots> tagged{};

    void evaluated(std::uint64_t scope_epoch, std::uint32_t scope_viewport, std::uint64_t now, std::uint64_t ordinal = 0,
        std::uint32_t frames = 0) {
      ++count; epoch = scope_epoch; viewport = scope_viewport; tick = now; present = ordinal; generated = frames;
    }
    // A tag call (serial) that defers a copy of its viewport's frame: the
    // first of that frame's deferral (begins), or a later one of the same
    // frame, whose deferral's marks are kept. ordinal: the call's Present
    // ordinal. Returns the deferral's marks.
    marks tag(std::uint64_t scope_epoch, std::uint32_t tag_viewport, std::uint64_t serial, bool begins, marks kept,
        std::uint64_t ordinal = 0) {
      tagged_frame *slot = nullptr;
      for (auto &value : tagged) if (value.used && value.viewport == tag_viewport) { slot = &value; break; }
      if (begins) {
        kept.previous_known = slot && slot->epoch == scope_epoch;
        kept.previous = kept.previous_known ? slot->at_tag : 0;
      }
      kept.at_tag = count;
      kept.present = ordinal;
      if (!slot) {
        for (auto &value : tagged) if (!value.used) { slot = &value; break; }
        if (!slot) {
          slot = &tagged[0];
          for (auto &value : tagged) if (value.serial < slot->serial) slot = &value;
        }
      }
      *slot = {true, scope_epoch, serial, count, tag_viewport};
      return kept;
    }
    // Whether a Present of a frame-generating viewport is the deferral's
    // frame's (scope: its epoch and viewport). queue_executed: a game list
    // executed on the presenting queue since its newest tag call. ordinal:
    // the Present's own. DLSS-G evaluates every frame while its newest
    // evaluation is within the source age, or is of the frame this Present
    // may still present (at most its generated count of Presents since it),
    // so a hitch between that evaluation and the frame's Present keeps it.
    bool presented(std::uint64_t scope_epoch, std::uint32_t scope_viewport, const marks &deferral, bool queue_executed,
        std::uint64_t now, std::uint64_t ordinal = 0) const {
      const bool evaluating = tick && epoch == scope_epoch && viewport == scope_viewport &&
        (sunshine_freshness::fresh_lt(tick, now, sunshine_freshness::source_age_ms) ||
         (present && ordinal >= present && ordinal - present <= generated));
      if (evaluating && count != deferral.at_tag) return true;
      const bool every_frame = !deferral.previous_known || deferral.previous != deferral.at_tag;
      return !(evaluating && every_frame) && queue_executed;
    }
    // Whether the source age ends a deferral (its newest tag call at tick) at
    // the Present ordinal: only at a Present that cannot be its frame's own.
    // That Present may come after the earlier frames' generated Presents
    // (frames: the viewport's generated count while it generates, else 0)
    // that DLSS-G still presents after the tag call, and it is the first
    // after DLSS-G evaluated a frame following that call. The deferral's
    // other ends (its observation, epoch, a later frame's constants or tag
    // call) stay.
    bool expired(std::uint64_t scope_epoch, std::uint32_t scope_viewport, const marks &deferral, std::uint64_t tag_tick,
        std::uint32_t frames, std::uint64_t ordinal, std::uint64_t now) const {
      if (sunshine_freshness::fresh_lt(tag_tick, now, sunshine_freshness::source_age_ms)) return false;
      if (deferral.present && ordinal >= deferral.present && ordinal - deferral.present <= frames) return false;
      const bool evaluated_since = tick && epoch == scope_epoch && viewport == scope_viewport &&
        count != deferral.at_tag && present && present == ordinal;
      return !evaluated_since;
    }
  };
}
