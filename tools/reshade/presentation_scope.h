// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "freshness.h"
#include "game3d_ui_mask.h"
#include "scene_depth_source.h"
#include "streamline_depth_capture.h"
#include "game3d_controls.h"

#include <array>
#include <cstdint>

// presentation_scope (docs/reshade-sbs.md, Presentation scope): the FG value
// types, the one FG state machine (fg_view) and each Present's FG snapshot
// (present_fg), and their pure predicates, each under its own name. The FG
// mode itself is the probe's FG options store (presentation_scope_mode.inl),
// whose reads always get the last committed options: no FG read is busy since
// 11-01, and the busy rules each reader had for one are deleted. The SL
// adapter bodies that read live state stay in sl_nomination.inl and
// sl_frame_generation.inl.

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
    // Whether a Present is the deferral's frame's (present_deferred).
    // executions: the game's command list executions counted at the
    // deferral's newest tag call (at_tag), and at this Present on any queue
    // (now) and on the presenting queue (on_queue). Without frame generation
    // the frame's Present is the first after a game list executed following
    // the tag call: a count above the tag call's. The Present reads its counts
    // before it takes the deferrals' lock, so a tag call that read after
    // them can already stand in the table with a larger count: that Present
    // is an earlier frame's. Until 11-01 any other count reached it (!=), so
    // the next frame's copy could be made at the previous frame's Present
    // (F67). With frame generation, presented().
    bool reached(bool generating, std::uint64_t scope_epoch, std::uint32_t scope_viewport, const marks &deferral,
        std::uint64_t at_tag, std::uint64_t now_executions, std::uint64_t on_queue, std::uint64_t now,
        std::uint64_t ordinal) const {
      if (!generating) return now_executions > at_tag;
      return presented(scope_epoch, scope_viewport, deferral, on_queue > at_tag, now, ordinal);
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

namespace sunshine_streamline {
  struct frame_generation_snapshot {
    std::uint64_t epoch{}, sequence{}, tick{}, loss_revision{};
    std::uint32_t viewport{}, mode{}, generated_frames{};
    bool known{}, enabled{}, automatic{};
    // Enabled: the tick of the options call that turned FG on in this epoch,
    // kept while later calls leave it on (depth_capture::frame_generation_scope_live).
    std::uint64_t enabled_tick{};
    // The Presents DLSS-G adds per real frame under these options: their
    // generated count, 1 while enabled without one (Dynamic mode with
    // numFramesToGenerate 0, whose count DLSS-G may pick), 0 without FG.
    // Every reader of a viewport's count takes it so: the deferral clock
    // (present_deferred) and DLSS-G's evaluation window
    // (deferral_evaluations), as the Present's grant (present_fg) and the UI
    // (ui_evidence) do. Until 11-01 the deferral clock and the evaluation
    // window took Dynamic's 0 as no generated Present (F4).
    std::uint32_t generated_presents() const { return enabled ? (generated_frames ? generated_frames : 1u) : 0u; }
  };
  // Successful, version-checked game options only. Enabled means requested by
  // the game; it does not identify any particular real/generated presentation.
  // busy is retired: no read produces it since the FG options store (11-01),
  // and it keeps its value so that the provider log's fg_query= numbers do
  // not shift.
  enum class frame_generation_query_status { unavailable, busy, ambiguous, observed };
}

namespace sunshine_streamline::presentation_scope {
  enum class fg_route { none, owned, silent };

  // The one FG state machine (docs/reshade-sbs.md, Presentation scope). Each
  // Present's FG read updates the process's one view once (take_present_fg),
  // and depth and UI read their two projections of that state:
  // - depth: an observed mode names the source choice; until the first one FG
  //   is not required (the owner's 11-01 ruling: an unknown mode is no FG,
  //   Generic allowed). An ambiguous read, or an unavailable one while FG is
  //   required, revokes FG scope while independent ordinary depth remains
  //   eligible (exclude_unconfirmed_fg); this never turns a failed Off
  //   request into confirmed Off.
  // - ui: the last observed mode, kept through an ambiguous or unavailable
  //   read (the UI's safe side: it keeps generated frames' current color out
  //   of UI detection), cleared while the observer is inactive. A restart
  //   (the probe's release_camera_records) resets the whole view.
  // So under an ambiguous read depth excludes FG input while the UI keeps the
  // last confirmed mode: their one documented difference (owner ruling Q1,
  // 11-01). Until 11-01 the provider runtime and each exporter runtime kept a
  // policy of their own, updated from two reads at two points of the Present
  // (PD6, PD11): an options call between them gave the UI one mode and depth
  // the other for that Present, and a runtime created during an ambiguous
  // period started with an unknown UI mode beside an older runtime's known one.
  struct fg_view {
    // Returns the depth projection after the read.
    depth_capture::selection_policy update(frame_generation_query_status status,
        const frame_generation_snapshot &value, bool observer_active = true) {
      if (status == frame_generation_query_status::observed) {
        depth_ = value.enabled ? depth_capture::selection_policy{true, value.epoch, value.viewport, false, value.enabled_tick} :
          depth_capture::selection_policy{};
        // Its viewport breaks a tie of active Streamline views (owner ruling
        // R5) only with a declared generated count. hook_fg_options accepts
        // Dynamic mode with numFramesToGenerate 0 as enabled FG, whose count
        // Dynamic may pick (dynamicTargetFrameRate), but sl_dlss_g.h does not
        // say that it then generates any frame: no confirmed binding, so the
        // tie stays (fail safe, S19). Its own routing is unchanged.
        if (value.enabled && value.generated_frames) {
          depth_.tie_epoch = value.epoch;
          depth_.tie_viewport = value.viewport;
        }
        generated_frames_ = value.enabled ? value.generated_frames : 0;
      } else if (status == frame_generation_query_status::ambiguous || depth_.require_frame_generation) {
        depth_ = {};
        depth_.exclude_unconfirmed_fg = true;
        generated_frames_ = 0;
      }
      if (!observer_active) ui_ = {};
      else if (status == frame_generation_query_status::observed && value.known)
        ui_ = {value.known, value.enabled, value.automatic, value.generated_frames, value.viewport, value.epoch, value.sequence};
      return depth_;
    }
    depth_capture::selection_policy depth() const { return depth_; }
    std::uint32_t generated_frames() const { return generated_frames_; }
    sunshine_game3d::frame_generation_mode ui() const { return ui_; }
  private:
    depth_capture::selection_policy depth_;
    std::uint32_t generated_frames_{};
    sunshine_game3d::frame_generation_mode ui_;
  };

  // One Present's FG snapshot (take_present_fg): taken once, first thing in
  // the Present (depth's present event), and read by every consumer of that
  // Present on its thread: present_deferred, the UI and the provider.
  struct present_fg {
    // The native swapchain it was taken for (0: none taken).
    std::uint64_t swapchain{};
    // The any-viewport read (query_frame_generation) and the view after it.
    frame_generation_query_status read{frame_generation_query_status::unavailable};
    frame_generation_snapshot fg;
    depth_capture::selection_policy depth;
    std::uint32_t generated_frames{};
    sunshine_game3d::frame_generation_mode ui;
    bool observer_active{};
    // Each viewport's committed FG options at the take (viewport_slots of
    // them; an unused entry has sequence 0).
    std::array<frame_generation_snapshot, frame_generation::viewport_slots> modes{};
    frame_generation_snapshot mode(std::uint32_t viewport) const {
      for (const auto &value : modes) if (value.sequence && value.viewport == viewport) return value;
      return {};
    }
    // The Presents DLSS-G adds per real frame while depth requires FG (also
    // for a silent scope; 1 when the count is unknown), 0 without FG: the
    // Present's grant (depth_capture::present, frame_presenting).
    std::uint32_t generated_presents() const {
      return depth.require_frame_generation ? (generated_frames ? generated_frames : 1u) : 0u;
    }
    // The frame this Present presents by depth's window (PD7): its API depth
    // acquisition's presenting floor (the newest evaluated frame's present
    // ordinal while this Present still presents that frame,
    // frame_presenting), else 0, as the provider published it after its
    // acquisition (present_depth_frame). The UI holds a generated Present's
    // decision by it.
    std::uint64_t depth_frame{};
  };
}

namespace sunshine_streamline::provider {
  // Enabled FG owns its viewport's depth only while its scope is live
  // (depth_capture::frame_generation_scope_live). A silent scope selects as
  // without FG for this acquisition, so that viewport's ordinary SR/RR depth
  // may supply depth once FG's own newest copy can no longer be held
  // (depth_capture::frame_generation_holds), and the next accepted FG copy
  // takes ownership back. Its viewport still breaks a tie of active
  // Streamline views (owner ruling R5): the selection keeps the scope's
  // tie_epoch and tie_viewport (fg_view::update), which select_latest_capture
  // reads as a required scope's.
  // Any other policy selects as it is.
  struct frame_generation_selection {
    depth_capture::selection_policy selection;
    bool silent{};
  };
  inline frame_generation_selection select_frame_generation_scope(const depth_capture::selection_policy &policy) {
    const bool silent = policy.require_frame_generation &&
      !depth_capture::frame_generation_scope_live(policy.epoch, policy.viewport, policy.enabled_tick);
    if (!silent) return {policy, false};
    depth_capture::selection_policy selection;
    selection.tie_epoch = policy.tie_epoch;
    selection.tie_viewport = policy.tie_viewport;
    return {selection, true};
  }
}

namespace sunshine_game3d::depth_output {
  // A retained Streamline FG copy younger than the source age: what
  // frame_generation_holds and frame_generation_retained keep.
  inline bool frame_generation_copy(const retained_depth &previous, std::uint64_t now) {
    const auto &before = previous.metadata;
    return previous.capture_id && before.frame_generation_input &&
      before.provider == sunshine_scene_depth::provider_kind::streamline &&
      sunshine_freshness::within_source_age(before.tick, now);
  }
}

namespace sunshine_streamline::depth_capture {
  // FG keeps its viewport while it can supply. An enabled FG scope that is
  // silent at this Present (its newest frame had no FG copy) lets the same
  // viewport's ordinary SR/RR depth stand in only once its own newest copy
  // can no longer be held: while the display holds a copy of that scope made
  // since FG On (policy.enabled_tick), younger than the source age, that copy
  // is held over a fresh or pending ordinary candidate. A frame without an FG
  // copy is then held as any missing frame of a source is, instead of
  // switching the viewport's source (which restarts the placement
  // calibration) to SR and back. policy: the confirmed FG policy, not the
  // silent scope's selection.
  inline bool frame_generation_holds(const sunshine_game3d::depth_output::retained_depth &previous, selection_policy policy, std::uint64_t now) {
    const auto &before = previous.metadata;
    return policy.require_frame_generation && policy.epoch && sunshine_game3d::depth_output::frame_generation_copy(previous, now) &&
      before.epoch == policy.epoch && before.viewport == policy.viewport && before.tick >= policy.enabled_tick;
  }

  // FG Off retires the FG source, which releases its queue's API association
  // (retire_source), while the display may still hold that scope's newest
  // copy, or the copy of the viewport's ordinary source a silent scope let
  // it show. Within the source age it is held as any source's missing frame
  // is: while the display holds an FG copy, or any Streamline copy of a
  // viewport whose FG role FG Off retired (frame_generation_retired, until
  // FG On), younger than the source age and FG is not revoked (an
  // unconfirmed scope refuses its copies), the queue stays the API's, so
  // decide_display holds the copy until it expires, a readable frame of
  // another source replaces it or a source outside its lineage drops it.
  // Only then may the queue hand over to Generic. The successor's copies
  // recorded since the shown copy stay admissible at FG Off (retire_source
  // retires only the FG role), so the first of them readable replaces the
  // held copy at once. Until 11-01 only an FG copy kept the queue: with the
  // display on the silent scope's SR copy and the FG role's owner
  // established by a pending pick, FG Off released the queue while the FG
  // copies it kept were still unreadable (DLSS-G's compute copy in flight
  // past the next Present), and those Presents went mono (source_handoff).
  inline bool frame_generation_retained(const sunshine_game3d::depth_output::retained_depth &previous, selection_policy policy, std::uint64_t now) {
    const auto &before = previous.metadata;
    return !policy.exclude_unconfirmed_fg && (sunshine_game3d::depth_output::frame_generation_copy(previous, now) ||
      (previous.capture_id && before.provider == sunshine_scene_depth::provider_kind::streamline &&
       sunshine_freshness::within_source_age(before.tick, now) && frame_generation_retired(before.epoch, before.viewport, 0)));
  }
}

namespace sunshine_ngx {
  // Whether confirmed Streamline FG owns depth, so that an NGX evaluation
  // records no copy: FG options observed enabled (the probe's
  // frame_generation_snapshot: enabled, epoch, viewport, enabled_tick) whose
  // scope is live (depth_capture::frame_generation_scope_live). Selection
  // never considers NGX while FG is required, and FG Off raises the admission
  // watermark past every snapshot taken meanwhile, so a copy would have no
  // consumer. FG Off, an ambiguous FG read, which observes no options (nested
  // NGX's rule, owns_nested_depth_capture), and a silent scope, which owns
  // nothing, copy.
  inline bool frame_generation_owns_depth(bool enabled, std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick) {
    return enabled && sunshine_streamline::depth_capture::frame_generation_scope_live(epoch, viewport, enabled_tick);
  }
}

