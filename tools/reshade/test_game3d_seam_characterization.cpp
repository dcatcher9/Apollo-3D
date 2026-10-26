// SPDX-License-Identifier: GPL-3.0-only
// Characterization of the depth selection seams that the API contract matrix
// does not cover (refactor plan P0a). These tables pin what the add-on does
// today; they are not an oracle of what an API requires. The Stage 4 moves
// (presentation_scope in P5, depth_selection in P6, depth_output in P7) must
// keep every row. A row changes only in the commit that changes that
// behaviour on purpose; the API contract matrix stays the authority on what
// Streamline, D3D12 and NGX permit.
//
// It drives the production functions through the capture owner's
// test-access header, with no device: the frame generation policy, the FG
// scope's liveness and the selection the provider derives from it, NGX's
// copy rule under frame generation, the frames an FG scope owns, DLSS-G's
// depth precedence for its frame's FG tag, the Present a deferred copy waits
// for under frame generation, select_provider and establish_provider,
// the exact freshness edges of selection, association and the FG scope, the
// display policy decide_display, and the UI adapter's admission of each UI
// candidate (P11).
#ifndef SUNSHINE_STREAMLINE_PROBE_TEST
  #error "Built only into the Streamline probe tests' view of the production sources"
#endif
#include "depth_cache_update.h"
#include "freshness.h"
#include "game3d_ui_input_provider.h"
#include "game3d_ui_mask.h"
#include "ngx_depth_source.h"
#include "streamline_depth_capture.h"
#include "streamline_depth_capture_test_access.h"
#include "streamline_depth_provider.h"
#include "test_streamline_depth_capture_fixture.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace sunshine_streamline::depth_capture {
  namespace {
    unsigned failures {};

    void fail(const std::string &text) {
      ++failures;
      std::fprintf(stderr, "FAIL %s\n", text.c_str());
    }

    constexpr std::uint64_t native = 77;
    constexpr std::uint64_t fg_source = 1ull << 63;  // Viewport 0's FG input.
    constexpr auto source_age = sunshine_freshness::source_age_ms;

    // The capture owner's globals, emptied before each row.
    void reset() {
      fixture::reset_owner();
    }

    // One submitted, successful capture on the native queue (an FG input when
    // its source is an FG source), and the API evaluation that nominated it.
    slot &make(unsigned index, provider_kind provider, std::uint64_t epoch, std::uint64_t sequence, std::uint64_t source, std::uint32_t viewport = 0, std::uint64_t superseded = UINT64_MAX) {
      auto &value = fixture::make_capture(index, native, provider, epoch, sequence, source, viewport, superseded);
      value.metadata.frame_generation_input = (source >> 63) != 0;
      return value;
    }

    // complete_frame()'s bookkeeping of a delivered (copied) frame.
    void deliver(queue_state &owner, const slot &value, bool recent = true) {
      owner.last_epoch = value.metadata.epoch;
      owner.last_sequence = value.metadata.sequence;
      owner.last_source_tick = recent ? GetTickCount64() : GetTickCount64() - source_age - 1;
    }

    // The picked capture's label, or - when nothing readable was picked.
    char label(const capture_pick &pick, const slot *first, char first_label, const slot *second, char second_label) {
      if (!pick.value) {
        return '-';
      }
      if (pick.value == first) {
        return first_label;
      }
      return pick.value == second ? second_label : '?';
    }

    // R a required FG scope, X excluded FG input, - neither.
    char policy_kind(const selection_policy &value) {
      if (value.require_frame_generation) {
        return value.exclude_unconfirmed_fg ? '?' : 'R';
      }
      return value.exclude_unconfirmed_fg ? 'X' : '-';
    }

    // P5: provider::frame_generation_policy turns each Present's FG options
    // query into the selection policy. Each timeline is a sequence of query
    // results and the policy each one returns.
    void frame_generation_policy_timelines() {
      using query = frame_generation_query_status;

      // FG options as the probe publishes them (frame_generation_snapshot).
      enum snapshot_id {
        none,
        on_vp0,
        off_vp0,
        on_vp1_mfg,
        on_new_epoch_dynamic
      };

      struct options {
        bool enabled;
        std::uint64_t epoch, enabled_tick;
        std::uint32_t viewport, generated_frames;
      };

      // By snapshot_id.
      constexpr options published[] {
        {false, 0, 0, 0, 0},
        {true, 5, 100, 0, 1},
        {false, 5, 0, 0, 0},
        {true, 5, 200, 1, 3},
        {true, 6, 300, 0, 0},
      };
      const auto snapshot = [&](snapshot_id id) {
        frame_generation_snapshot out;
        const auto &value = published[id];
        out.known = id != none;
        out.enabled = value.enabled;
        out.epoch = value.epoch;
        out.enabled_tick = value.enabled_tick;
        out.viewport = value.viewport;
        out.generated_frames = value.generated_frames;
        return out;
      };

      // The returned policy: R required FG scope, X excluded FG input, - none.
      struct step {
        query status;
        snapshot_id read;
        char policy;
        std::uint64_t epoch, enabled_tick;
        std::uint32_t viewport, generated_frames;
      };

      struct timeline {
        const char *pins;
        std::vector<step> steps;
      };

      const timeline timelines[] {
        {"Before any observation a busy read requires an FG scope it cannot name (fg_scope_missing) and is not kept",
         {{query::unavailable, none, '-', 0, 0, 0, 0}, {query::busy, none, 'R', 0, 0, 0, 0}, {query::unavailable, none, '-', 0, 0, 0, 0}}},
        {"FG On names its scope; busy reads keep it",
         {{query::observed, on_vp0, 'R', 5, 100, 0, 1}, {query::busy, none, 'R', 5, 100, 0, 1}, {query::busy, none, 'R', 5, 100, 0, 1}}},
        {"FG Off is confirmed: busy and unavailable reads require nothing",
         {{query::observed, on_vp0, 'R', 5, 100, 0, 1}, {query::observed, off_vp0, '-', 0, 0, 0, 0}, {query::busy, none, '-', 0, 0, 0, 0}, {query::unavailable, none, '-', 0, 0, 0, 0}}},
        {"An unavailable read after FG On revokes FG input until the next observed On",
         {{query::observed, on_vp0, 'R', 5, 100, 0, 1}, {query::unavailable, none, 'X', 0, 0, 0, 0}, {query::busy, none, 'X', 0, 0, 0, 0}, {query::unavailable, none, 'X', 0, 0, 0, 0}, {query::observed, on_vp0, 'R', 5, 100, 0, 1}}},
        {"An ambiguous read (two enabled viewports, a failed Off, lost options) revokes FG input; a confirmed Off lifts it",
         {{query::observed, on_vp0, 'R', 5, 100, 0, 1}, {query::ambiguous, none, 'X', 0, 0, 0, 0}, {query::busy, none, 'X', 0, 0, 0, 0}, {query::observed, off_vp0, '-', 0, 0, 0, 0}}},
        {"An ambiguous first read excludes FG input through busy and unavailable reads",
         {{query::ambiguous, none, 'X', 0, 0, 0, 0}, {query::busy, none, 'X', 0, 0, 0, 0}, {query::unavailable, none, 'X', 0, 0, 0, 0}}},
        {"The scope follows the newest observed On: another viewport, multi frame generation, a new epoch, a dynamic count",
         {{query::observed, on_vp0, 'R', 5, 100, 0, 1}, {query::observed, on_vp1_mfg, 'R', 5, 200, 1, 3}, {query::observed, on_new_epoch_dynamic, 'R', 6, 300, 0, 0}}},
      };
      unsigned rows = 0;
      for (const auto &line : timelines) {
        provider::frame_generation_policy policy;
        unsigned index = 0;
        for (const auto &s : line.steps) {
          ++rows;
          const auto got = policy.update(s.status, snapshot(s.read));
          const char kind = policy_kind(got);
          if (kind != s.policy || got.epoch != s.epoch || got.viewport != s.viewport || got.enabled_tick != s.enabled_tick || policy.generated_frames() != s.generated_frames) {
            char text[512];
            std::snprintf(text, sizeof(text),
                          "P5 FG policy, %s, step %u: policy %c epoch %llu viewport %u tick %llu "
                          "frames %u",
                          line.pins,
                          index,
                          kind,
                          static_cast<unsigned long long>(got.epoch),
                          got.viewport,
                          static_cast<unsigned long long>(got.enabled_tick),
                          policy.generated_frames());
            fail(text);
          }
          ++index;
        }
      }
      std::printf("PASS P5 frame generation policy: %zu timelines, %u reads\n", std::size(timelines), rows);
    }

    // P5 and P6: whether an enabled FG scope owns its viewport this Present
    // (frame_generation_scope_live), and what select_provider then picks
    // with the policy provider::begin derives from it (a silent scope selects
    // as without FG). Accepted copies count alike whether a Streamline FG tag
    // or DLSS-G's own NGX evaluation (DLSSG.Depth) supplied them: both are the
    // viewport's FG source. The scope owns only frames FG supplied: a frame
    // without an FG copy (frame_generation_missed) newer than the newest
    // accepted copy makes it silent, and withdraws nothing.
    void frame_generation_scope_rows() {
      enum class fg_copy {
        none,
        accepted,
        missed_after_copy,
        copied_after_miss,
        failed,
        metadata_only,
        other_viewport
      };

      struct row {
        const char *pins;
        bool just_enabled;  // FG options turned FG on less than a second ago.
        fg_copy copy;
        bool sr_ready;  // The viewport's ordinary SR depth is readable.
        bool live;
        char picked;  // S the SR capture, F the FG capture, - none.
        status result;
        selection_reason selection;
      };

      const row rows[] {
        {"FG just turned on before its first copy lets the viewport's SR depth through", true, fg_copy::none, true, false, 'S', status::ready, selection_reason::current_capture},
        {"A silent scope (on for over a second, no accepted FG copy) lets the viewport's SR depth through", false, fg_copy::none, true, false, 'S', status::ready, selection_reason::current_capture},
        {"A silent scope without any API depth supplies nothing", false, fg_copy::none, false, false, '-', status::stale, selection_reason::inactive_views},
        {"An accepted FG copy (tag or DLSS-G) makes the scope live and supplies its depth", false, fg_copy::accepted, true, true, 'F', status::ready, selection_reason::current_capture},
        {"A frame missed after the newest accepted copy turns the scope silent and withdraws nothing", false, fg_copy::missed_after_copy, true, false, 'F', status::ready, selection_reason::current_capture},
        {"An accepted copy of a frame newer than the missed one keeps the scope live", false, fg_copy::copied_after_miss, true, true, 'F', status::ready, selection_reason::current_capture},
        {"A failed FG evaluation is no accepted copy; its attempt still replaced the SR view", false, fg_copy::failed, true, false, '-', status::failed, selection_reason::no_admissible_capture},
        {"A metadata-only FG nomination is no accepted copy", false, fg_copy::metadata_only, true, false, '-', status::unavailable, selection_reason::no_admissible_capture},
        {"Another viewport's accepted FG copy leaves this scope silent and makes Streamline ambiguous", false, fg_copy::other_viewport, true, false, '-', status::ambiguous, selection_reason::ambiguous_views},
      };
      constexpr std::uint64_t epoch = 7;
      for (const auto &r : rows) {
        reset();
        const slot *sr = r.sr_ready ? &make(0, provider_kind::streamline, epoch, 1, 0) : nullptr;
        const slot *fg = nullptr;
        if (r.copy != fg_copy::none) {
          const bool other = r.copy == fg_copy::other_viewport;
          auto &value = make(1, provider_kind::streamline, epoch, 2, other ? fg_source | 1 : fg_source, other ? 1 : 0, 0);
          value.nomination_only = r.copy == fg_copy::metadata_only;
          finish(value.id, r.copy != fg_copy::failed);
          // The copy's frame is sequence 2.
          if (r.copy == fg_copy::missed_after_copy) {
            frame_generation_missed(epoch, 0, 3);
          }
          if (r.copy == fg_copy::copied_after_miss) {
            frame_generation_missed(epoch, 0, 1);
          }
          fg = &value;
        }
        const auto now = GetTickCount64();
        const selection_policy policy {true, epoch, 0, false, r.just_enabled ? now : now - sunshine_freshness::fg_silent_ms - 1};
        // provider::begin's selection for this Present.
        const auto scope = provider::select_frame_generation_scope(policy);
        const bool live = !scope.silent;
        queue_state owner;
        const auto pick = select_provider(owner, native, 1, GetTickCount64(), scope.selection);
        const char picked = label(pick, sr, 'S', fg, 'F');
        if (live != r.live || picked != r.picked || pick.result != r.result || pick.selection != r.selection) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5 FG scope, %s: live %d picked %c result %s selection %s", r.pins, live, picked, name(pick.result), name(pick.selection));
          fail(text);
        }
      }
      // Only a required scope can be silent; any other policy selects as it is.
      selection_policy excluded;
      excluded.exclude_unconfirmed_fg = true;
      const auto none = provider::select_frame_generation_scope({});
      const auto kept = provider::select_frame_generation_scope(excluded);
      if (none.silent || none.selection.require_frame_generation || none.selection.exclude_unconfirmed_fg || kept.silent || !kept.selection.exclude_unconfirmed_fg) {
        fail("P5 FG scope: a policy without a required scope changed its selection");
      }
      std::printf("PASS P5/P6 FG scope liveness and selection: %zu rows\n", std::size(rows));
    }

    // P5: NGX's copy rule under frame generation
    // (sunshine_ngx::frame_generation_owns_depth). Confirmed FG options whose
    // scope is live suppress an NGX evaluation's copy; FG Off, a busy or
    // ambiguous read (which observes no options) and a silent scope copy.
    void ngx_frame_generation_rows() {
      enum class fg_copy {
        none,
        accepted,
        other_viewport,
        other_epoch
      };

      struct row {
        const char *pins;
        bool enabled;  // The FG options query observed FG on (false: Off, busy or ambiguous).
        bool just_enabled;  // FG options turned FG on less than a second ago.
        fg_copy copy;
        bool suppressed;
      };

      const row rows[] {
        {"FG just turned on before its first copy owns nothing: NGX copies", true, true, fg_copy::none, false},
        {"A live scope (an accepted FG copy within a second) owns depth", true, false, fg_copy::accepted, true},
        {"A silent scope owns nothing: NGX copies", true, false, fg_copy::none, false},
        {"Another viewport's accepted FG copy leaves this scope silent", true, false, fg_copy::other_viewport, false},
        {"Another epoch's accepted FG copy leaves this scope silent", true, false, fg_copy::other_epoch, false},
        {"FG Off, a busy or an ambiguous read copies, even beside a live FG copy", false, false, fg_copy::accepted, false},
        {"FG Off copies just after FG was on", false, true, fg_copy::none, false},
      };
      constexpr std::uint64_t epoch = 7;
      for (const auto &r : rows) {
        reset();
        if (r.copy != fg_copy::none) {
          const bool viewport = r.copy == fg_copy::other_viewport;
          auto &value = make(1, provider_kind::streamline, r.copy == fg_copy::other_epoch ? epoch + 1 : epoch, 2, viewport ? fg_source | 1 : fg_source, viewport ? 1 : 0, 0);
          finish(value.id, true);
        }
        const auto now = GetTickCount64();
        const bool suppressed = sunshine_ngx::frame_generation_owns_depth(r.enabled, epoch, 0,
          r.just_enabled ? now : now - sunshine_freshness::fg_silent_ms - 1);
        if (suppressed != r.suppressed) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5 NGX under FG, %s: suppressed %d", r.pins, suppressed);
          fail(text);
        }
      }
      std::printf("PASS P5 NGX copy rule under frame generation: %zu rows\n", std::size(rows));
    }

    // P5: the frames an FG scope owns (frame_generation_frame), in one
    // scope's order of calls. Each frame of the viewport begins with its
    // constants; a live scope expects the frame to record an FG copy, in
    // flight or accepted, and when the next frame begins with none recorded
    // since, that frame missed: the scope turns silent, so the viewport's
    // SR/RR depth supplies it, and only a copy recorded after the miss makes
    // the scope live again (known gap C4 until 10-26: the scope stayed live
    // for a second after its newest copy, keeping SR out once that copy's
    // hold expired).
    void frame_generation_frame_steps() {
      enum class call {
        accepted,  // An FG copy recorded and accepted (the API call succeeded).
        recorded,  // An FG copy recorded, still in flight (a compute copy).
        frame  // A frame of the viewport begins (its key; 0 unknown).
      };

      struct step {
        const char *pins;
        call what;
        std::uint64_t value;  // A copy's observation sequence, or the frame's key.
        bool live;  // The scope afterwards.
      };

      const step steps[] {
        {"An accepted FG copy makes the scope live", call::accepted, 2, true},
        {"A frame begins: the live scope expects its FG copy", call::frame, 1, true},
        {"The frame's own FG copy (DLSS-G's, at its Present)", call::accepted, 4, true},
        {"The next frame begins after a copy was recorded: still live", call::frame, 2, true},
        {"The same frame's constants again begin no frame", call::frame, 2, true},
        {"A copy recorded but still in flight supplies its frame", call::recorded, 6, true},
        {"So the next frame begins live", call::frame, 3, true},
        {"A frame begins with no copy recorded since the previous one: that frame missed, the scope turns silent", call::frame, 4, false},
        {"The copy in flight before the miss, readable after it, leaves the scope silent", call::accepted, 6, false},
        {"A frame of a silent scope changes nothing", call::frame, 5, false},
        {"A copy recorded after the miss makes the scope live again", call::accepted, 9, true},
        {"A frame begins", call::frame, 6, true},
        {"The next frame without a copy misses again", call::frame, 7, false},
        {"A copy recorded later makes the scope live again", call::accepted, 11, true},
        {"A frame without a key begins", call::frame, 0, true},
        {"So does every later call without one: the frame between had no copy", call::frame, 0, false},
      };
      constexpr std::uint64_t epoch = 7;
      reset();
      const auto enabled_tick = GetTickCount64() - sunshine_freshness::fg_silent_ms - 1;
      unsigned index {};
      for (const auto &s : steps) {
        if (s.what == call::frame) {
          frame_generation_frame(epoch, 0, enabled_tick, s.value);
        } else {
          auto &value = make(index++, provider_kind::streamline, epoch, s.value, fg_source, 0, 0);
          {
            std::lock_guard lock(mutex);
            record_frame_generation_copy(value.metadata);
          }
          if (s.what == call::accepted) {
            finish(value.id, true);
          } else {
            value.finished = value.success = false;
          }
        }
        const bool live = frame_generation_scope_live(epoch, 0, enabled_tick);
        if (live != s.live) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5 FG scope frames, %s: live %d", s.pins, live);
          fail(text);
        }
      }
      std::printf("PASS P5 frames an FG scope owns: %zu steps\n", std::size(steps));
    }

    // P5: DLSS-G's own depth input stands aside only for the FG tag copy of
    // the frame it evaluates (frame_generation::depth_tag_precedence), in one
    // scope's order of calls. The frame is the viewport's newest, marked by
    // the constants call that began it; a tag call is that frame's when both
    // name the same frame, or (where either names none) when it came after
    // those constants. Its copy counts recorded, or waiting for the frame's
    // Present (pending), while the scope's newest tag copy is younger than
    // the source age. Known gap C6 until 10-26: any tag copy within the
    // source age kept every later frame's DLSSG.Depth out.
    void dlssg_tag_precedence_steps() {
      namespace fg = sunshine_streamline::frame_generation;
      enum class call {
        copy,  // An FG tag call of the scope records its copy (mark).
        refuse,  // An FG tag call of the scope leaves its frame without a copy.
        evaluate  // DLSS-G evaluates the frame mark, with pending's tag call waiting.
      };

      struct step {
        const char *pins;
        call what;
        fg::frame_mark mark;  // The tag call, or the evaluated frame.
        fg::frame_mark pending;  // key and sequence 0: none.
        std::uint64_t at;  // Milliseconds after the first step.
        bool stands_aside;  // The evaluation adds no copy (tag_copied).
      };

      constexpr auto age = sunshine_freshness::source_age_ms;
      const step steps[] {
        {"A frame's FG tag records its copy", call::copy, {101, 11}, {}, 0, false},
        {"DLSS-G's evaluation of that frame adds no second copy", call::evaluate, {101, 10}, {}, 5, true},
        {"The next frame, without an FG tag of its own, is DLSS-G's (C6)", call::evaluate, {102, 20}, {}, 16, false},
        {"So is every later one while that copy is younger than the source age (C6)", call::evaluate, {103, 30}, {}, 32, false},
        {"A frame tagged before its constants by its own frame token", call::copy, {104, 40}, {}, 48, false},
        {"keeps its precedence once those constants began it", call::evaluate, {104, 41}, {}, 50, true},
        {"A tag naming no frame, recorded after the frame began", call::copy, {0, 52}, {}, 64, false},
        {"keeps its precedence by call order", call::evaluate, {105, 51}, {}, 66, true},
        {"but not over the next frame", call::evaluate, {106, 60}, {}, 80, false},
        {"A frame of unknown key began after the tagged one", call::evaluate, {0, 70}, {}, 96, false},
        {"Without constants nothing marks the frame: any copy within the source age stands", call::evaluate, {}, {}, 98, true},
        {"A later frame's tag, made before that frame's constants", call::copy, {108, 81}, {}, 112, false},
        {"keeps no precedence over the frame DLSS-G evaluates", call::evaluate, {107, 80}, {}, 114, false},
        {"A frame whose tag copy waits for its Present stands aside while the tags copy", call::evaluate, {109, 90}, {109, 91}, 128, true},
        {"An earlier frame's waiting copy keeps nothing", call::evaluate, {110, 100}, {109, 91}, 144, false},
        {"A refused tag copy hands even its own frame to DLSS-G", call::refuse, {111, 111}, {}, 160, false},
        {"its waiting copy included", call::evaluate, {111, 110}, {111, 111}, 162, false},
        {"A recorded copy takes its frame back", call::copy, {112, 121}, {}, 176, false},
        {"for its own evaluation", call::evaluate, {112, 120}, {}, 178, true},
        {"but not once older than the source age", call::evaluate, {112, 120}, {}, 176 + age, false},
      };
      constexpr std::uint64_t epoch = 7;
      constexpr std::uint32_t viewport = 0;
      const auto start = GetTickCount64();
      fg::depth_tag_precedence value;
      for (const auto &s : steps) {
        const auto now = start + s.at;
        bool aside = false;
        if (s.what == call::evaluate) {
          const bool pending = s.pending.key || s.pending.sequence;
          aside = value.tags_supply(epoch, viewport, s.mark, pending ? &s.pending : nullptr, now);
        } else {
          value.remember(epoch, viewport, s.what == call::copy, s.mark, now);
        }
        if (aside != s.stands_aside) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5 DLSS-G tag precedence, %s: stands aside %d", s.pins, aside);
          fail(text);
        }
      }
      // The precedence belongs to its scope.
      value.remember(epoch, viewport, true, {201, 200}, start);
      if (value.tags_supply(epoch + 1, viewport, {201, 199}, nullptr, start) || value.tags_supply(epoch, viewport + 1, {201, 199}, nullptr, start) || !value.tags_supply(epoch, viewport, {201, 199}, nullptr, start)) {
        fail("P5 DLSS-G tag precedence: a tag copy decided another scope's frame, or not its own");
      }
      std::printf("PASS P5 DLSS-G's depth precedence for its frame's FG tag: %zu steps\n", std::size(steps) + 1);
    }

    // P5: which Present makes a copy deferred to its frame's Present (R2)
    // while its viewport generates frames
    // (frame_generation::deferral_evaluations), in one session's order of
    // calls. DLSS-G evaluates a frame inside that frame's Present, so the
    // copy waits for an evaluation after its newest tag call while DLSS-G
    // evaluates every frame: within the source age, and since the viewport's
    // previous deferred frame's newest tag call (or with no earlier deferred
    // frame of the viewport in the epoch). Otherwise the first Present after
    // a game list executed on the presenting queue since the tag call makes
    // it. Known gap C7 until 10-26: any evaluation within the source age kept
    // every copy waiting.
    void deferral_present_steps() {
      namespace fg = sunshine_streamline::frame_generation;
      enum class call {
        tag,  // A tag call begins a frame's deferral.
        again,  // A later tag call of the same frame.
        evaluate,  // DLSS-G evaluates a frame of the scope.
        present  // A Present decides the viewport's deferral.
      };

      struct step {
        const char *pins;
        call what;
        std::uint64_t at;  // Milliseconds after the first step.
        // present: a game list executed on the presenting queue since the
        // deferral's newest tag call.
        bool queue_executed;
        bool made;  // present: the Present is the deferral's frame's.
        std::uint32_t viewport;
        std::uint64_t epoch;
      };

      constexpr auto age = sunshine_freshness::source_age_ms;
      const step steps[] {
        {"The viewport's first deferral, before DLSS-G evaluates", call::tag, 0, false, false, 0, 7},
        {"A Present before the game's work on the presenting queue is an earlier frame's", call::present, 2, false, false, 0, 7},
        {"without DLSS-G's evaluations the first after it is the frame's", call::present, 4, true, true, 0, 7},
        {"DLSS-G evaluates the frame", call::evaluate, 6, false, false, 0, 7},
        {"The next frame's deferral", call::tag, 16, false, false, 0, 7},
        {"waits for DLSS-G, which evaluated the previous deferred frame, through the presenting queue's work", call::present, 18, true, false, 0, 7},
        {"DLSS-G evaluates it", call::evaluate, 20, false, false, 0, 7},
        {"and the first Present after that evaluation is its frame's", call::present, 22, false, true, 0, 7},
        {"A frame tagged twice", call::tag, 32, false, false, 0, 7},
        {"another viewport's first deferral in between", call::tag, 33, false, false, 1, 7},
        {"the frame's later tag call", call::again, 34, false, false, 0, 7},
        {"keeps the previous frame's count: the frame still waits", call::present, 36, true, false, 0, 7},
        {"DLSS-G evaluates it", call::evaluate, 38, false, false, 0, 7},
        {"another viewport, outside DLSS-G's scope, takes the presenting queue's work", call::present, 40, true, true, 1, 7},
        {"the frame is made after that evaluation", call::present, 42, false, true, 0, 7},
        {"DLSS-G stops evaluating: the next frame's deferral", call::tag, 48, false, false, 0, 7},
        {"still waits, as DLSS-G evaluated the previous deferred frame", call::present, 50, true, false, 0, 7},
        {"The frame after it", call::tag, 64, false, false, 0, 7},
        {"waits for no evaluation, but a Present before the game's work on the presenting queue is still an earlier frame's", call::present, 66, false, false, 0, 7},
        {"its Present is the first after that work, within the source age of DLSS-G's last evaluation (C7)", call::present, 68, true, true, 0, 7},
        {"So is every later frame's", call::tag, 80, false, false, 0, 7},
        {"(C7)", call::present, 82, true, true, 0, 7},
        {"DLSS-G evaluates some frames only: one it evaluates", call::tag, 96, false, false, 0, 7},
        {"DLSS-G evaluates it", call::evaluate, 98, false, false, 0, 7},
        {"is made at the Present after its evaluation", call::present, 100, false, true, 0, 7},
        {"the next frame", call::tag, 112, false, false, 0, 7},
        {"waits for DLSS-G again", call::present, 114, true, false, 0, 7},
        {"the frame after a frame it did not evaluate", call::tag, 128, false, false, 0, 7},
        {"does not (C7)", call::present, 130, true, true, 0, 7},
        {"DLSS-G evaluates again", call::evaluate, 140, false, false, 0, 7},
        {"the next frame", call::tag, 144, false, false, 0, 7},
        {"waits for DLSS-G within the source age of its evaluation", call::present, 140 + age - 1, true, false, 0, 7},
        {"but not after it", call::present, 140 + age, true, true, 0, 7},
        {"A new epoch's DLSS-G evaluation", call::evaluate, 420, false, false, 0, 8},
        {"and its first deferral of the viewport", call::tag, 424, false, false, 0, 8},
        {"waits for DLSS-G, with no earlier deferred frame to judge it by", call::present, 426, true, false, 0, 8},
        {"DLSS-G evaluates it", call::evaluate, 428, false, false, 0, 8},
        {"made after that evaluation", call::present, 430, false, true, 0, 8},
      };
      const auto start = GetTickCount64();
      fg::deferral_evaluations value;
      std::array<fg::deferral_evaluations::marks, 2> deferrals {};
      std::uint64_t serial {};
      for (const auto &s : steps) {
        const auto now = start + s.at;
        auto &deferral = deferrals[s.viewport];
        bool made = false;
        if (s.what == call::tag || s.what == call::again) {
          deferral = value.tag(s.epoch, s.viewport, ++serial, s.what == call::tag, deferral);
        } else if (s.what == call::evaluate) {
          value.evaluated(s.epoch, 0, now);
        } else {
          made = value.presented(s.epoch, s.viewport, deferral, s.queue_executed, now);
        }
        if (made != s.made) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5 deferred copy's Present, %s: made %d", s.pins, made);
          fail(text);
        }
      }
      // Four viewports keep their marks; a fifth takes the place of the one
      // deferred longest ago.
      fg::deferral_evaluations table;
      for (std::uint32_t viewport = 0; viewport != 5; ++viewport) {
        table.tag(7, viewport, viewport + 1, true, {});
      }
      if (table.tag(7, 0, 10, true, {}).previous_known || !table.tag(7, 2, 11, true, {}).previous_known) {
        fail("P5 deferred copy's Present: a fifth viewport did not take the place of the one deferred longest ago");
      }
      std::printf("PASS P5 deferred copy's Present under frame generation: %zu steps\n", std::size(steps) + 1);
    }

    // P6: select_provider at one Present. Each row builds the queue's owner
    // (its previous, delivered frame), the newest capture of each provider and
    // the FG policy, then names the pick. Correction 2 of the refactor plan: a
    // delivering owner keeps its own live source against a readable
    // competitor; only without a protected owner does readable Streamline beat
    // readable NGX.
    void select_provider_rows() {
      enum class capture {
        none,
        ready,
        pending,
        failed,
        expired
      };
      enum class owner_kind {
        none,
        sl,
        ngx
      };
      enum class fg_policy {
        none,
        required,
        unknown_epoch,
        other_viewport,
        excluded
      };

      struct row {
        const char *pins;
        owner_kind owner;
        bool delivering;  // The owner copied a frame within the source age.
        capture sl, ngx;
        bool sl_fg;  // The Streamline source is viewport 0's FG input.
        fg_policy policy;
        char picked;  // S the newest Streamline capture, N the newest NGX capture, - none.
        status result;
        selection_reason selection;
      };

      using o = owner_kind;
      using c = capture;
      using p = fg_policy;
      using r = selection_reason;
      const row rows[] {
        // A protected owner against a readable competitor.
        {"A delivering NGX owner keeps its readable source against readable SL", o::ngx, true, c::ready, c::ready, false, p::none, 'N', status::ready, r::current_capture},
        {"An NGX owner that stopped delivering yields to readable SL", o::ngx, false, c::ready, c::ready, false, p::none, 'S', status::ready, r::current_capture},
        {"A delivering owner's pending successor holds against readable SL", o::ngx, true, c::ready, c::pending, false, p::none, '-', status::recorded, r::no_completed_snapshot},
        {"A delivering owner's failed successor never holds against readable SL", o::ngx, true, c::ready, c::failed, false, p::none, 'S', status::ready, r::current_capture},
        {"A delivering owner's expired successor never holds against readable SL", o::ngx, true, c::ready, c::expired, false, p::none, 'S', status::ready, r::current_capture},
        {"A delivering SL owner keeps its readable source against readable NGX", o::sl, true, c::ready, c::ready, false, p::none, 'S', status::ready, r::current_capture},
        {"A delivering SL owner's pending successor holds against readable NGX", o::sl, true, c::pending, c::ready, false, p::none, '-', status::recorded, r::no_completed_snapshot},
        {"An SL owner that stopped delivering yields its pending successor to readable NGX", o::sl, false, c::pending, c::ready, false, p::none, 'N', status::ready, r::current_capture},
        {"A delivering owner without a readable source keeps its failed attempt against pending SL", o::ngx, true, c::pending, c::failed, false, p::none, '-', status::failed, r::no_admissible_capture},
        {"An owner that stopped delivering yields its failed attempt to a pending source", o::ngx, false, c::pending, c::failed, false, p::none, '-', status::recorded, r::no_completed_snapshot},
        // Readable against pending, without an owner.
        {"Readable SL beats readable NGX on a fresh queue", o::none, false, c::ready, c::ready, false, p::none, 'S', status::ready, r::current_capture},
        {"Readable NGX beats pending SL", o::none, false, c::pending, c::ready, false, p::none, 'N', status::ready, r::current_capture},
        {"Readable SL beats pending NGX", o::none, false, c::ready, c::pending, false, p::none, 'S', status::ready, r::current_capture},
        {"Both pending: SL is reported", o::none, false, c::pending, c::pending, false, p::none, '-', status::recorded, r::no_completed_snapshot},
        {"A pending source ranks above a failed one", o::none, false, c::failed, c::pending, false, p::none, '-', status::recorded, r::no_completed_snapshot},
        {"A failed live source ranks above an expired one", o::none, false, c::expired, c::failed, false, p::none, '-', status::failed, r::no_admissible_capture},
        {"Nothing reported: Streamline's inactive views", o::none, false, c::none, c::none, false, p::none, '-', status::stale, r::inactive_views},
        // FG required or excluded.
        {"Required FG picks its own readable input", o::none, false, c::ready, c::ready, true, p::required, 'S', status::ready, r::current_capture},
        {"Required FG overrides a delivering NGX owner", o::ngx, true, c::ready, c::ready, true, p::required, 'S', status::ready, r::current_capture},
        {"Required FG refuses the viewport's ordinary SL depth", o::none, false, c::ready, c::ready, false, p::required, '-', status::unavailable, r::fg_scope_mismatch},
        {"Required FG never falls back to NGX", o::none, false, c::none, c::ready, true, p::required, '-', status::stale, r::inactive_views},
        {"Required FG with its input pending holds", o::none, false, c::pending, c::ready, true, p::required, '-', status::recorded, r::no_completed_snapshot},
        {"Required FG before its scope is known (a busy first read) is missing", o::none, false, c::ready, c::ready, true, p::unknown_epoch, '-', status::unavailable, r::fg_scope_missing},
        {"Required FG of another viewport refuses this viewport's FG input", o::none, false, c::ready, c::ready, true, p::other_viewport, '-', status::unavailable, r::fg_scope_mismatch},
        {"Excluded FG refuses the FG input and lets NGX supply", o::none, false, c::ready, c::ready, true, p::excluded, 'N', status::ready, r::current_capture},
        {"Excluded FG without another source is missing", o::none, false, c::ready, c::none, true, p::excluded, '-', status::stale, r::fg_scope_missing},
        {"Excluded FG keeps ordinary SL depth", o::none, false, c::ready, c::ready, false, p::excluded, 'S', status::ready, r::current_capture},
      };
      constexpr std::uint64_t sl_epoch = 5, ngx_epoch = 6, ngx_source = 99;
      for (const auto &row : rows) {
        reset();
        const auto sl_id = row.sl_fg ? fg_source : 0;
        queue_state owner;
        // The owner's previous frame: established and delivered at Present 1.
        if (row.owner != o::none) {
          const bool sl = row.owner == o::sl;
          auto &previous = sl ? make(0, provider_kind::streamline, sl_epoch, 1, sl_id) :
                                make(1, provider_kind::ngx, ngx_epoch, 1, ngx_source);
          establish_provider(owner, previous, 1);
          deliver(owner, previous, row.delivering);
        }
        const auto newest = [&](unsigned index, capture state, provider_kind provider, std::uint64_t epoch, std::uint64_t source) -> const slot * {
          if (state == c::none) {
            return nullptr;
          }
          auto &value = make(index, provider, epoch, 2, source);
          if (state == c::pending) {
            value.finished = false;
          }
          if (state == c::failed) {
            value.success = false;
          }
          if (state == c::expired) {
            value.metadata.tick = GetTickCount64() - source_age - 1;
          }
          return &value;
        };
        const auto *sl = newest(2, row.sl, provider_kind::streamline, sl_epoch, sl_id);
        const auto *ngx = newest(3, row.ngx, provider_kind::ngx, ngx_epoch, ngx_source);
        selection_policy policy;
        if (row.policy == p::required) {
          policy = {true, sl_epoch, 0};
        }
        if (row.policy == p::unknown_epoch) {
          policy = {true};
        }
        if (row.policy == p::other_viewport) {
          policy = {true, sl_epoch, 1};
        }
        if (row.policy == p::excluded) {
          policy.exclude_unconfirmed_fg = true;
        }
        const auto pick = select_provider(owner, native, 2, GetTickCount64(), policy);
        const char picked = label(pick, sl, 'S', ngx, 'N');
        if (picked != row.picked || pick.result != row.result || pick.selection != row.selection) {
          char text[512];
          std::snprintf(text, sizeof(text), "P6 select_provider, %s: picked %c result %s selection %s", row.pins, picked, name(pick.result), name(pick.selection));
          fail(text);
        }
      }
      std::printf("PASS P6 select_provider: %zu rows\n", std::size(rows));
    }

    void check_pick(const char *timeline, unsigned step, const capture_pick &pick, char expected, const slot *sl, const slot *ngx) {
      const char picked = label(pick, sl, 'S', ngx, 'N');
      if (picked == expected) {
        return;
      }
      char text[512];
      std::snprintf(text, sizeof(text), "P6 %s, step %u: picked %c (expected %c) result %s selection %s", timeline, step, picked, expected, name(pick.result), name(pick.selection));
      fail(text);
    }

    // P6: ownership is per queue (queue_state, refactor plan correction 2),
    // and establish_provider's watermark belongs to one exact source.
    void ownership_timelines() {
      constexpr std::uint64_t epoch = 5, ngx_source = 99;
      {
        // Two queues read the same captures. Queue A's delivering NGX owner
        // keeps NGX; queue B has no owner and takes readable SL; B's owner
        // never moves A's, and a released NGX source frees only the queue it
        // owned.
        const char *timeline = "per-queue ownership";
        reset();
        queues[0] = std::make_unique<queue_state>();
        queues[1] = std::make_unique<queue_state>();
        auto &a = *queues[0];
        auto &b = *queues[1];
        auto &previous = make(1, provider_kind::ngx, epoch, 1, ngx_source);
        establish_provider(a, previous, 1);
        deliver(a, previous);
        const auto *ngx = &make(3, provider_kind::ngx, epoch, 2, ngx_source);
        const auto *sl = &make(2, provider_kind::streamline, epoch, 2, 0);
        check_pick(timeline, 1, select_provider(a, native, 2, GetTickCount64(), {}), 'N', sl, ngx);
        const auto on_b = select_provider(b, native, 2, GetTickCount64(), {});
        check_pick(timeline, 2, on_b, 'S', sl, ngx);
        if (on_b.value) {
          establish_provider(b, *on_b.value, 2);
        }
        check_pick(timeline, 3, select_provider(a, native, 2, GetTickCount64(), {}), 'N', sl, ngx);
        if (!a.provider_established || a.provider != provider_kind::ngx || !b.provider_established || b.provider != provider_kind::streamline) {
          fail("P6 per-queue ownership, step 3: an owner moved another queue's");
        }
        retire_source(provider_kind::ngx, epoch, ngx_source);
        if (a.provider_established || !b.provider_established || b.provider != provider_kind::streamline) {
          fail("P6 per-queue ownership, step 4: a released source freed another queue's owner");
        }
        // The freed queue admits only captures recorded after the release.
        check_pick(timeline, 5, select_provider(a, native, 3, GetTickCount64(), {}), '-', sl, ngx);
        check_pick(timeline, 6, select_provider(b, native, 3, GetTickCount64(), {}), 'S', sl, ngx);
        sl = &make(2, provider_kind::streamline, epoch, 3, 0);
        check_pick(timeline, 7, select_provider(a, native, 3, GetTickCount64(), {}), 'S', sl, ngx);
        queues[0].reset();
        queues[1].reset();
      }
      {
        // establish_provider keeps the delivered watermark for the same exact
        // source and restarts it for any other provider, source or epoch.
        reset();
        queue_state owner;

        struct step {
          provider_kind provider;
          std::uint64_t epoch, sequence, source;
          std::uint64_t kept_sequence;  // The watermark after establishing (0: restarted).
        };

        const step steps[] {
          {provider_kind::ngx, epoch, 1, ngx_source, 0},
          {provider_kind::ngx, epoch, 2, ngx_source, 1},
          {provider_kind::ngx, epoch + 1, 1, ngx_source, 0},
          {provider_kind::ngx, epoch + 1, 2, ngx_source + 1, 0},
          {provider_kind::streamline, epoch + 1, 3, ngx_source + 1, 0},
        };
        unsigned index = 0;
        for (const auto &s : steps) {
          auto &value = make(index % 2, s.provider, s.epoch, s.sequence, s.source);
          establish_provider(owner, value, 10 + index);
          if (!owner.provider_established || owner.provider != s.provider || owner.source_id != s.source || owner.nominated_epoch != s.epoch || owner.present != 10 + index || owner.present_capture != value.id || owner.last_sequence != s.kept_sequence) {
            char text[256];
            std::snprintf(text, sizeof(text), "P6 establish_provider, step %u: watermark %llu", index, static_cast<unsigned long long>(owner.last_sequence));
            fail(text);
          }
          deliver(owner, value);
          ++index;
        }
      }
      {
        // Repeated effects in one presentation keep the provider it chose,
        // even one that has not delivered; the next presentation chooses again.
        const char *timeline = "repeated presentation";
        reset();
        queue_state owner;
        const auto *ngx = &make(3, provider_kind::ngx, epoch, 1, ngx_source);
        establish_provider(owner, *ngx, 2);
        const auto *sl = &make(2, provider_kind::streamline, epoch, 1, 0);
        check_pick(timeline, 1, select_provider(owner, native, 2, GetTickCount64(), {}), 'N', sl, ngx);
        check_pick(timeline, 2, select_provider(owner, native, 3, GetTickCount64(), {}), 'S', sl, ngx);
      }
      std::puts("PASS P6 ownership: per-queue owners, establish_provider watermarks and repeated presentations");
    }

    // P2 and P6: the exact freshness edges, against an explicit clock, one ms
    // either side of each limit. The source age (250 ms) admits only an age
    // below it (<): a capture's own tick, its evaluation's view, a pending FG
    // nomination, a completed snapshot, a pending latest capture and the
    // owner's last delivery. The association timeout (1000 ms) admits its
    // limit (<=): expire_association's evaluations and the handoff window of
    // evaluation_live, neither of which admits a tick ahead of the clock; an
    // FG scope's enabled and accepted ticks admit both
    // (frame_generation_scope_live). The handoff limit (250 ms) admits its
    // limit (<=) in the provider's handoff_hold and in the evaluation_live
    // window the provider passes it. Each limit is named in freshness.h.
    enum class edge_site {
      capture_tick,
      view_tick,
      pending_nomination,
      completed_snapshot,
      pending_latest,
      owner_delivery,
      association_view,
      association_rejected,
      association_pending,
      evaluation_window,
      scope_enabled_alone,
      scope_enabled,
      scope_accepted,
      handoff_association,
      handoff_evaluation
    };

    std::string pick_outcome(const capture_pick &pick, const slot *first, char first_label, const slot *second, char second_label) {
      return std::string(1, label(pick, first, first_label, second, second_label)) + ' ' + name(pick.result) + ' ' + name(pick.selection);
    }

    // The outcome at one site for a tick age ms before now (a negative age is
    // ahead of the clock).
    std::string edge_outcome(edge_site site, std::int64_t age) {
      constexpr std::uint64_t epoch = 5, ngx_epoch = 6, ngx_source = 99;
      constexpr std::uint64_t window = 1000;
      reset();
      queue_state owner;
      auto &views = evaluations[provider_index(provider_kind::streamline)].views;
      switch (site) {
        case edge_site::capture_tick:
          {
            auto &sl = make(0, provider_kind::streamline, epoch, 1, 0);
            const auto now = GetTickCount64();
            sl.metadata.tick = now - age;
            return pick_outcome(select_provider(owner, native, 1, now, {}), &sl, 'S', nullptr, '?');
          }
        case edge_site::view_tick:
          {
            auto &sl = make(0, provider_kind::streamline, epoch, 1, 0);
            const auto now = GetTickCount64();
            sl.metadata.tick = now;
            for (auto &view : views) {
              if (view.sequence) {
                view.tick = now - age;
              }
            }
            return pick_outcome(select_provider(owner, native, 1, now, {}), &sl, 'S', nullptr, '?');
          }
        case edge_site::pending_nomination:
          {
            // The owner's FG source has delivered sequence 1; sequence 2 is
            // nominated, its capture slot not yet recorded.
            auto &previous = make(0, provider_kind::streamline, epoch, 1, fg_source);
            establish_provider(owner, previous, 1);
            deliver(owner, previous);
            begin_evaluation(epoch, 2, 0, provider_kind::streamline, fg_source);
            const auto now = GetTickCount64();
            evaluations[provider_index(provider_kind::streamline)].pending_nomination = {epoch, 2, now - age, fg_source, 0};
            return pick_outcome(select_provider(owner, native, 2, now, {}), &previous, 'P', nullptr, '?');
          }
        case edge_site::completed_snapshot:
          {
            // The newest capture is still on the GPU; its completed predecessor
            // of the same storage may stand in.
            const auto storage = std::make_shared<source_reference>();
            auto &completed = make(0, provider_kind::streamline, epoch, 1, 0);
            completed.texture = std::make_shared<texture_reference>();
            completed.source_nominated = true;
            completed.metadata.source = storage;
            auto &latest = make(1, provider_kind::streamline, epoch, 2, 0);
            latest.finished = false;
            latest.metadata.source = storage;
            const auto now = GetTickCount64();
            completed.metadata.tick = now - age;
            latest.metadata.tick = now;
            return pick_outcome(select_provider(owner, native, 1, now, {}), &completed, 'C', &latest, 'L');
          }
        case edge_site::pending_latest:
          {
            // A pending Streamline capture ranks above NGX's failed one while
            // it is within the source age.
            auto &latest = make(0, provider_kind::streamline, epoch, 1, 0);
            latest.finished = false;
            auto &failed = make(1, provider_kind::ngx, ngx_epoch, 1, ngx_source);
            failed.success = false;
            const auto now = GetTickCount64();
            latest.metadata.tick = now - age;
            failed.metadata.tick = now;
            return pick_outcome(select_provider(owner, native, 1, now, {}), &latest, 'S', &failed, 'N');
          }
        case edge_site::owner_delivery:
          {
            // A delivering NGX owner keeps its readable source against readable SL.
            auto &previous = make(1, provider_kind::ngx, ngx_epoch, 1, ngx_source);
            establish_provider(owner, previous, 1);
            deliver(owner, previous);
            const auto *sl = &make(2, provider_kind::streamline, epoch, 2, 0);
            const auto *ngx = &make(3, provider_kind::ngx, ngx_epoch, 2, ngx_source);
            const auto now = GetTickCount64();
            owner.last_source_tick = now - age;
            return pick_outcome(select_provider(owner, native, 2, now, {}), sl, 'S', ngx, 'N');
          }
        case edge_site::association_view:
        case edge_site::association_rejected:
        case edge_site::association_pending:
          {
            // An established Streamline SR owner (source 0) and its own view.
            auto &previous = make(0, provider_kind::streamline, epoch, 1, 0);
            establish_provider(owner, previous, 1);
            const auto now = GetTickCount64();
            for (auto &view : views) {
              if (!view.sequence) {
                continue;
              }
              if (site == edge_site::association_view) {
                view.tick = now - age;
              } else if (site == edge_site::association_rejected) {
                view.tick = now;
                view.rejected_since = now - age;
              } else {
                view.tick = now - window - 1;
              }
            }
            if (site == edge_site::association_pending) {
              evaluations[provider_index(provider_kind::streamline)].pending_nomination = {epoch, 2, now - age, 0, 0};
            }
            expire_association(owner, now);
            return owner.provider_established ? "kept" : "expired";
          }
        case edge_site::evaluation_window:
          {
            make(0, provider_kind::streamline, epoch, 1, 0);
            const auto now = GetTickCount64();
            for (auto &view : views) {
              if (view.sequence) {
                view.tick = now - age;
              }
            }
            return evaluation_live(now, window) ? "live" : "silent";
          }
        case edge_site::scope_enabled_alone:
        case edge_site::scope_enabled:
        case edge_site::scope_accepted:
          {
            // Viewport 0's FG scope, on the fixed clock: FG turned on age ms
            // ago, with no copy (alone) or one accepted now; or turned on long
            // ago with a copy accepted age ms ago.
            const auto now = GetTickCount64();
            scope_clock = now;
            std::uint64_t enabled_tick = now - age;
            if (site != edge_site::scope_enabled_alone) {
              auto &view = views[0];
              view = {epoch, 1, now, fg_source, 0};
              view.accepted_tick = now; view.accepted_sequence = 1;
            }
            if (site == edge_site::scope_accepted) {
              enabled_tick = now - sunshine_freshness::fg_silent_ms - 1;
              views[0].accepted_tick = now - age;
            }
            const bool live = frame_generation_scope_live(epoch, 0, enabled_tick);
            scope_clock = 0;
            return live ? "live" : "silent";
          }
        case edge_site::handoff_association:
          {
            // The provider owned the previous Present, the size is unchanged
            // and an evaluation is live; the last associated Present was age
            // ms ago.
            const auto now = GetTickCount64();
            const provider::handoff_evidence evidence {true, true, now, now - age, 1920, 1080, 1920, 1080};
            return provider::handoff_hold(evidence, sunshine_freshness::handoff_ms) ? "hold" : "release";
          }
        case edge_site::handoff_evaluation:
          {
            make(0, provider_kind::streamline, epoch, 1, 0);
            const auto now = GetTickCount64();
            for (auto &view : views) {
              if (view.sequence) {
                view.tick = now - age;
              }
            }
            return evaluation_live(now, sunshine_freshness::handoff_ms) ? "live" : "silent";
          }
      }
      return "?";
    }

    void freshness_edges() {
      constexpr std::int64_t limit = source_age, timeout = sunshine_freshness::association_ms;
      constexpr std::int64_t handoff = sunshine_freshness::handoff_ms;
      using e = edge_site;

      struct row {
        const char *pins;
        edge_site site;
        std::int64_t age;
        const char *outcome;
      };

      const row rows[] {
        {"A readable capture just under the source age", e::capture_tick, limit - 1, "S ready current_capture"},
        {"A readable capture exactly the source age old is stale", e::capture_tick, limit, "- stale no_admissible_capture"},
        {"A readable capture past the source age is stale", e::capture_tick, limit + 1, "- stale no_admissible_capture"},
        {"An evaluation view just under the source age is active", e::view_tick, limit - 1, "S ready current_capture"},
        {"An evaluation view exactly the source age old is inactive", e::view_tick, limit, "- stale inactive_views"},
        {"An evaluation view past the source age is inactive", e::view_tick, limit + 1, "- stale inactive_views"},
        {"A pending FG nomination just under the source age holds", e::pending_nomination, limit - 1, "- recorded pending_nomination"},
        {"A pending FG nomination exactly the source age old lapses", e::pending_nomination, limit, "- unavailable no_current_nomination"},
        {"A pending FG nomination past the source age lapses", e::pending_nomination, limit + 1, "- unavailable no_current_nomination"},
        {"A completed snapshot just under the source age stands in", e::completed_snapshot, limit - 1, "C ready completed_snapshot"},
        {"A completed snapshot exactly the source age old does not", e::completed_snapshot, limit, "- recorded no_completed_snapshot"},
        {"A completed snapshot past the source age does not", e::completed_snapshot, limit + 1, "- recorded no_completed_snapshot"},
        {"A pending capture just under the source age is live and ranks above a failed one", e::pending_latest, limit - 1, "- recorded no_completed_snapshot"},
        {"A pending capture exactly the source age old is expired", e::pending_latest, limit, "- failed no_admissible_capture"},
        {"A pending capture past the source age is expired", e::pending_latest, limit + 1, "- failed no_admissible_capture"},
        {"An owner that delivered just under the source age ago is protected", e::owner_delivery, limit - 1, "N ready current_capture"},
        {"An owner that delivered exactly the source age ago is not", e::owner_delivery, limit, "S ready current_capture"},
        {"An owner that delivered past the source age ago is not", e::owner_delivery, limit + 1, "S ready current_capture"},
        {"An owner's view exactly the association timeout old keeps it", e::association_view, timeout, "kept"},
        {"An owner's view past the association timeout expires it", e::association_view, timeout + 1, "expired"},
        {"An owner's view ahead of the clock does not keep it", e::association_view, -1, "expired"},
        {"A run of rejections exactly the association timeout long keeps it", e::association_rejected, timeout, "kept"},
        {"A run of rejections past the association timeout expires it", e::association_rejected, timeout + 1, "expired"},
        {"An owner's pending nomination exactly the association timeout old keeps it", e::association_pending, timeout, "kept"},
        {"An owner's pending nomination past the association timeout expires it", e::association_pending, timeout + 1, "expired"},
        {"An evaluation exactly the handoff window old is live", e::evaluation_window, 1000, "live"},
        {"An evaluation past the handoff window is not", e::evaluation_window, 1001, "silent"},
        {"An evaluation ahead of the clock is not", e::evaluation_window, -1, "silent"},
        {"FG turned on just now with no accepted copy is silent", e::scope_enabled_alone, 0, "silent"},
        {"FG turned on by another thread ahead of this read with no accepted copy is silent", e::scope_enabled_alone, -1, "silent"},
        {"An FG copy accepted in the tick FG was turned on is live", e::scope_enabled, 0, "live"},
        {"An FG copy accepted past the association timeout after FG On is live", e::scope_enabled, timeout + 1, "live"},
        {"An FG copy accepted before FG was turned on (another thread's) is silent", e::scope_enabled, -1, "silent"},
        {"An FG copy accepted exactly the association timeout ago is live", e::scope_accepted, timeout, "live"},
        {"An FG copy accepted past the association timeout is silent", e::scope_accepted, timeout + 1, "silent"},
        {"An FG copy accepted on another thread ahead of this read is live", e::scope_accepted, -1, "live"},
        {"A Present associated exactly the handoff limit ago holds the handoff", e::handoff_association, handoff, "hold"},
        {"A Present associated past the handoff limit releases it", e::handoff_association, handoff + 1, "release"},
        {"A Present associated ahead of the clock releases it", e::handoff_association, -1, "release"},
        {"An evaluation exactly the handoff limit old keeps a handoff live", e::handoff_evaluation, handoff, "live"},
        {"An evaluation past the handoff limit does not", e::handoff_evaluation, handoff + 1, "silent"},
      };
      for (const auto &r : rows) {
        const auto got = edge_outcome(r.site, r.age);
        if (got != r.outcome) {
          char text[512];
          std::snprintf(text, sizeof(text), "P2 freshness edge, %s: %s", r.pins, got.c_str());
          fail(text);
        }
      }
      std::printf("PASS P2/P6 freshness edges: %zu rows\n", std::size(rows));
    }

    // P5 and P6: a single FG copy amid frames without one while the
    // viewport's SR copies are readable only frames after their evaluation
    // (Assassin's Creed Shadows 10-24: another queue, 110-235 ms). The FG
    // copy's delivery moves the queue owner's consumed watermark past every
    // SR snapshot evaluated before it, so none of those is picked (the newer
    // one in flight is reported: no_completed_snapshot), and the provider
    // holds the FG copy (frame_generation_holds) while it is
    // within the source age, also once the SR snapshot evaluated after it is
    // readable. Past the source age that SR snapshot supplies; while it is
    // still in flight the Present has no depth. That loss is the cross-queue
    // delay, not the shared watermark: an SR snapshot evaluated before the FG
    // copy is older than the FG copy, so it is past the source age as well.
    void stray_frame_generation_copy_steps() {
      struct step {
        const char *pins;
        bool fg_fresh;  // The delivered FG copy is within the source age.
        bool sr_retired;  // The SR snapshot evaluated after it is readable.
        bool held;  // frame_generation_holds keeps the FG copy on display.
        char picked;  // O the older SR snapshot, S the newer one, - none.
        status result;
        selection_reason selection;
      };

      const step steps[] {
        {"The newer SR snapshot in flight: the older one is consumed and the FG copy is held", true, false, true, '-', status::recorded, selection_reason::no_completed_snapshot},
        {"The newer SR snapshot readable within the FG copy's source age: the FG copy is still held", true, true, true, 'S', status::ready, selection_reason::current_capture},
        {"The FG copy past the source age: the newer SR snapshot supplies", false, true, false, 'S', status::ready, selection_reason::current_capture},
        {"The FG copy past the source age, the newer SR snapshot still in flight: no depth until it is readable", false, false, false, '-', status::recorded, selection_reason::no_completed_snapshot},
      };
      constexpr std::uint64_t epoch = 7;
      for (const auto &s : steps) {
        reset();
        const auto now = GetTickCount64();
        const auto fg_tick = s.fg_fresh ? now - 16 : now - source_age - 1;
        // The older SR snapshot (sequence 2), evaluated a frame before the FG copy.
        auto &older = make(0, provider_kind::streamline, epoch, 2, 0, 0, fg_source);
        older.metadata.tick = fg_tick - 16;
        // The stray FG copy (sequence 3), accepted and delivered at its Present.
        auto &fg = make(1, provider_kind::streamline, epoch, 3, fg_source, 0, 0);
        fg.metadata.tick = fg_tick;
        finish(fg.id, true);
        queue_state owner;
        establish_provider(owner, fg, 3);
        deliver(owner, fg, s.fg_fresh);
        owner.last_source_tick = fg_tick;
        // The next frame has no FG copy; its SR evaluation replaced the FG view.
        frame_generation_missed(epoch, 0, 4);
        auto &newer = make(2, provider_kind::streamline, epoch, 4, 0, 0, fg_source);
        newer.metadata.tick = fg_tick + 16;
        newer.finished = s.sr_retired;
        const selection_policy policy {true, epoch, 0, false, now - sunshine_freshness::fg_silent_ms - 1};
        const auto scope = provider::select_frame_generation_scope(policy);
        const auto pick = select_provider(owner, native, 4, now, scope.selection);
        retained_depth shown;
        shown.metadata = fg.metadata;
        shown.capture_id = fg.id;
        const bool held = frame_generation_holds(shown, policy, now);
        const char picked = label(pick, &older, 'O', &newer, 'S');
        if (!scope.silent || held != s.held || picked != s.picked || pick.result != s.result || pick.selection != s.selection) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5/P6 stray FG copy, %s: silent %d held %d picked %c result %s selection %s", s.pins, scope.silent, held, picked, name(pick.result), name(pick.selection));
          fail(text);
        }
      }
      std::printf("PASS P5/P6 stray FG copy with late SR snapshots: %zu steps\n", std::size(steps));
    }

    struct display_case {
      acquisition_decision current;
      packet candidate;
      retained_depth previous;
      selection_policy policy;
      std::uint64_t now {};
    };

    // The case's sources become viewport 2's FG input, which the policy requires.
    void fg_retained(display_case &d) {
      d.previous.metadata.frame_generation_input = true;
      d.previous.metadata.source_id = (1ull << 63) | 2;
      d.current.source_id = d.candidate.metadata.source_id = d.previous.metadata.source_id;
      d.candidate.metadata.frame_generation_input = true;
      d.policy = {true, 8, 2};
    }

    // The retained copy becomes viewport 2's FG input, which the case's
    // candidate (the viewport's SR source, 0) follows: FG is off, or the
    // selection is a silent scope's, which selects as without FG.
    void fg_successor(display_case &d) {
      d.previous.metadata.frame_generation_input = true;
      d.previous.metadata.source_id = (1ull << 63) | 2;
    }

    // P5 and P7: decide_display, the display policy that provider::begin
    // applies to the acquisition (moving into depth_output). Each row edits
    // one field of a base case (a pending successor of a retained SR copy,
    // 100 ms old) and names the action and reason.
    void decide_display_rows() {
      const auto base = [] {
        display_case out;
        auto &before = out.previous;
        before.capture_id = 500;
        before.width = 1920;
        before.height = 1080;
        before.format = 39;
        before.view_format = 41;
        before.area = {0, 0, 1920, 1080};
        before.metadata.provider = provider_kind::streamline;
        before.metadata.epoch = 8;
        before.metadata.viewport = 2;
        before.metadata.sequence = 30;
        before.metadata.tick = 1000;
        auto &next = out.candidate;
        static_cast<sunshine_scene_depth::frame &>(next.metadata) = before.metadata;
        next.metadata.sequence = 31;
        next.capture_id = 501;
        next.width = 1920;
        next.height = 1080;
        next.format = 39;
        next.srv_format = 41;
        next.area = before.area;
        auto &now = out.current;
        now.provider = provider_kind::streamline;
        now.epoch = 8;
        now.viewport = 2;
        now.sequence = 31;
        now.capture_id = 501;
        now.source_valid = now.pending_frame = true;
        out.now = 1100;
        return out;
      };
      using action = display_action;

      struct row {
        const char *pins;
        void (*edit)(display_case &);
        action expected;
        const char *reason;
      };

      const row rows[] {
        {"A pending successor holds the retained copy", [](display_case &) {
         },
         action::hold,
         "retained_copy"},
        {"A selected candidate with ready pixels is copied", [](display_case &d) {
           d.current.source_selected = d.candidate.pixel_ready = true;
         },
         action::copy_fresh,
         "fresh_copy_candidate"},
        {"A selected candidate without pixels holds", [](display_case &d) {
           d.current.source_selected = true;
         },
         action::hold,
         "retained_copy"},
        {"Nothing retained", [](display_case &d) {
           d.previous.capture_id = 0;
         },
         action::invalidate,
         "no_retained_depth"},
        {"A retained copy without an epoch", [](display_case &d) {
           d.previous.metadata.epoch = 0;
         },
         action::invalidate,
         "no_retained_depth"},
        {"Excluded FG refuses a ready FG candidate", [](display_case &d) {
           fg_retained(d);
           d.policy = {};
           d.policy.exclude_unconfirmed_fg = true;
           d.current.source_selected = d.candidate.pixel_ready = true;
         },
         action::invalidate,
         "FG_scope_unconfirmed"},
        {"Excluded FG refuses a retained FG copy", [](display_case &d) {
           fg_retained(d);
           d.policy = {};
           d.policy.exclude_unconfirmed_fg = true;
         },
         action::invalidate,
         "FG_scope_unconfirmed"},
        {"Excluded FG keeps a retained SR copy", [](display_case &d) {
           d.policy.exclude_unconfirmed_fg = true;
         },
         action::hold,
         "retained_copy"},
        {"Required FG refuses a retained SR copy", [](display_case &d) {
           d.policy = {true, 8, 2};
         },
         action::invalidate,
         "FG_scope_changed"},
        {"Required FG refuses another provider's retained FG copy", [](display_case &d) {
           fg_retained(d);
           d.previous.metadata.provider = d.current.provider = provider_kind::ngx;
         },
         action::invalidate,
         "FG_scope_changed"},
        {"Required FG refuses another epoch's retained FG copy", [](display_case &d) {
           fg_retained(d);
           d.policy.epoch = 9;
         },
         action::invalidate,
         "FG_scope_changed"},
        {"Required FG refuses another viewport's retained FG copy", [](display_case &d) {
           fg_retained(d);
           d.policy.viewport = 3;
         },
         action::invalidate,
         "FG_scope_changed"},
        {"Required FG holds its own retained copy through a generated Present (no nomination, sequence 0)", [](display_case &d) {
           fg_retained(d);
           d.current.sequence = d.current.capture_id = 0;
           d.current.pending_frame = false;
         },
         action::hold,
         "retained_copy"},
        {"Another provider's nomination", [](display_case &d) {
           d.current.provider = provider_kind::ngx;
         },
         action::invalidate,
         "source_changed"},
        {"Another epoch's nomination", [](display_case &d) {
           d.current.epoch = 9;
         },
         action::invalidate,
         "source_changed"},
        {"Another source's nomination", [](display_case &d) {
           d.current.source_id = 4;
         },
         action::invalidate,
         "source_changed"},
        {"Another viewport's nomination", [](display_case &d) {
           d.current.viewport = 3;
         },
         action::invalidate,
         "source_changed"},
        {"An older sequence than the retained copy", [](display_case &d) {
           d.current.sequence = 29;
         },
         action::invalidate,
         "source_changed"},
        {"No nomination (epoch 0) does not contradict", [](display_case &d) {
           d.current = {};
         },
         action::hold,
         "retained_copy"},
        {"A selected candidate of another storage family", [](display_case &d) {
           d.current.source_selected = true;
           ++d.candidate.format;
         },
         action::invalidate,
         "depth_format_changed"},
        {"A selected candidate read through another view", [](display_case &d) {
           d.current.source_selected = true;
           ++d.candidate.srv_format;
         },
         action::invalidate,
         "depth_format_changed"},
        {"A selected candidate of another resource kind", [](display_case &d) {
           d.current.source_selected = true;
           d.candidate.metadata.resource.kind = sunshine_scene_depth::resource_kind::display_depth;
         },
         action::invalidate,
         "depth_format_changed"},
        {"Storage is compared only for a selected candidate", [](display_case &d) {
           ++d.candidate.format;
         },
         action::hold,
         "retained_copy"},
        {"A selected candidate of another depth encoding", [](display_case &d) {
           d.current.source_selected = true;
           d.candidate.metadata.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
         },
         action::invalidate,
         "depth_encoding_changed"},
        {"A new render size of the same source holds", [](display_case &d) {
           d.current.source_selected = true;
           d.candidate.width = 2206;
           d.candidate.height = 1240;
           d.candidate.area = {0, 0, 2048, 1152};
         },
         action::hold,
         "retained_copy"},
        {"A camera reset or later observation of the successor holds", [](display_case &d) {
           d.candidate.metadata.feedback.reset = true;
           ++d.candidate.metadata.observation_revision;
         },
         action::hold,
         "retained_copy"},
        {"A retained copy without a tick", [](display_case &d) {
           d.previous.metadata.tick = 0;
         },
         action::invalidate,
         "retained_depth_expired"},
        {"A retained copy from the future", [](display_case &d) {
           d.now = 999;
         },
         action::invalidate,
         "retained_depth_expired"},
        {"A retained copy exactly the source age old", [](display_case &d) {
           d.now = 1000 + source_age;
         },
         action::invalidate,
         "retained_depth_expired"},
        {"A retained copy just under the source age", [](display_case &d) {
           d.now = 1000 + source_age - 1;
         },
         action::hold,
         "retained_copy"},
        // An FG copy's successor, the viewport's ordinary source once FG is
        // off (or its scope silent), takes the viewport at its first readable
        // frame; until then the FG copy is held (continuity gap C8 until
        // 10-26: dropped as source_changed while an SR copy on another queue
        // was in flight).
        {"After an FG copy, its viewport's pending SR successor holds it", fg_successor, action::hold, "FG_successor_pending"},
        {"After an FG copy, a selected SR successor without pixels holds it", [](display_case &d) {
           fg_successor(d);
           d.current.source_selected = true;
         },
         action::hold,
         "FG_successor_pending"},
        {"After an FG copy, a readable SR successor is copied", [](display_case &d) {
           fg_successor(d);
           d.current.source_selected = d.candidate.pixel_ready = true;
         },
         action::copy_fresh,
         "fresh_copy_candidate"},
        {"After an FG copy, the SR successor's own storage and encoding do not end the hold", [](display_case &d) {
           fg_successor(d);
           d.current.source_selected = true;
           ++d.candidate.format;
           d.candidate.metadata.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
         },
         action::hold,
         "FG_successor_pending"},
        {"After an FG copy, another viewport's SR source", [](display_case &d) {
           fg_successor(d);
           d.current.viewport = 3;
         },
         action::invalidate,
         "source_changed"},
        {"After an FG copy, another epoch's SR source", [](display_case &d) {
           fg_successor(d);
           d.current.epoch = 9;
         },
         action::invalidate,
         "source_changed"},
        {"After an FG copy, an NGX source", [](display_case &d) {
           fg_successor(d);
           d.current.provider = provider_kind::ngx;
         },
         action::invalidate,
         "source_changed"},
        {"After an FG copy, an older sequence of the FG source itself", [](display_case &d) {
           fg_successor(d);
           d.current.source_id = d.previous.metadata.source_id;
           d.current.sequence = 29;
         },
         action::invalidate,
         "source_changed"},
        {"After an NGX FG copy, the SR source", [](display_case &d) {
           fg_successor(d);
           d.previous.metadata.provider = provider_kind::ngx;
         },
         action::invalidate,
         "source_changed"},
        {"Excluded FG refuses an FG copy before its SR successor", [](display_case &d) {
           fg_successor(d);
           d.policy.exclude_unconfirmed_fg = true;
         },
         action::invalidate,
         "FG_scope_unconfirmed"},
        {"After an FG copy exactly the source age old, its SR successor", [](display_case &d) {
           fg_successor(d);
           d.now = 1000 + source_age;
         },
         action::invalidate,
         "retained_depth_expired"},
      };
      for (const auto &r : rows) {
        auto d = base();
        r.edit(d);
        const auto got = decide_display(d.current, d.candidate, d.previous, d.policy, d.now);
        if (got.action != r.expected || std::strcmp(got.reason, r.reason) != 0) {
          char text[512];
          std::snprintf(text, sizeof(text), "P7 decide_display, %s: action %d reason %s", r.pins, int(got.action), got.reason);
          fail(text);
        }
      }
      std::printf("PASS P5/P7 decide_display: %zu rows\n", std::size(rows));

      // decide_failed_copy: provider::begin's decision once a copy_fresh
      // attempt failed (the consumer refused it, as an observation loss
      // refuses its recording). The failure records nothing into the display
      // storage, so the retained copy is held by the rules above; until
      // 10-26 every failure dropped it (continuity gap C3). Each row edits
      // the copied case (a selected candidate with ready pixels).
      const row failed[] {
        {"A failed copy of the same source holds the retained copy", [](display_case &) {
         },
         action::hold,
         "retained_after_failed_copy"},
        {"A failed copy of a new render size of the same source holds", [](display_case &d) {
           d.candidate.width = 2206;
           d.candidate.height = 1240;
           d.candidate.area = {0, 0, 2048, 1152};
         },
         action::hold,
         "retained_after_failed_copy"},
        {"A failed copy of required FG's input holds its scope's retained copy", fg_retained, action::hold,
         "retained_after_failed_copy"},
        {"A failed copy with the retained copy just under the source age", [](display_case &d) {
           d.now = 1000 + source_age - 1;
         },
         action::hold,
         "retained_after_failed_copy"},
        {"A failed copy with nothing retained", [](display_case &d) {
           d.previous.capture_id = 0;
         },
         action::invalidate,
         "display_copy_failed"},
        {"A failed copy of another source", [](display_case &d) {
           d.current.source_id = d.candidate.metadata.source_id = 4;
         },
         action::invalidate,
         "display_copy_failed"},
        {"A failed copy of an older sequence than the retained copy", [](display_case &d) {
           d.current.sequence = d.candidate.metadata.sequence = 29;
         },
         action::invalidate,
         "display_copy_failed"},
        {"A failed copy of another storage family", [](display_case &d) {
           ++d.candidate.format;
         },
         action::invalidate,
         "display_copy_failed"},
        {"A failed copy of another depth encoding", [](display_case &d) {
           d.candidate.metadata.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
         },
         action::invalidate,
         "display_copy_failed"},
        {"A failed copy under required FG with a retained SR copy", [](display_case &d) {
           d.policy = {true, 8, 2};
         },
         action::invalidate,
         "display_copy_failed"},
        {"A failed copy with the retained copy exactly the source age old", [](display_case &d) {
           d.now = 1000 + source_age;
         },
         action::invalidate,
         "display_copy_failed"},
        {"A failed copy of an FG copy's SR successor holds the FG copy", fg_successor, action::hold, "retained_after_failed_copy"},
      };
      for (const auto &r : failed) {
        auto d = base();
        d.current.source_selected = d.candidate.pixel_ready = true;
        r.edit(d);
        const auto attempt = decide_display(d.current, d.candidate, d.previous, d.policy, d.now);
        const auto got = decide_failed_copy(d.current, d.candidate, d.previous, d.policy, d.now);
        if (attempt.action != action::copy_fresh || got.action != r.expected || std::strcmp(got.reason, r.reason) != 0) {
          char text[512];
          std::snprintf(text, sizeof(text), "P7 decide_failed_copy, %s: attempt %d, action %d reason %s", r.pins,
            int(attempt.action), int(got.action), got.reason);
          fail(text);
        }
      }
      std::printf("PASS P7 decide_failed_copy: %zu rows\n", std::size(failed));
    }

    // P11: ui_input::acquire's admission of each ready UI snapshot
    // (ui_mask::admit_candidate) and of the Present's own color
    // (ui_input::current_color_refusal). Owner ruling R4: UIAlpha is used in
    // Auto only when its tag number is authenticated or DLSS-G names it.
    void ui_admission_rows() {
      using kind = sunshine_game3d::ui_mask::source_kind;
      using refusal = sunshine_game3d::ui_mask::candidate_refusal;

      struct candidate_row {
        const char *pins;
        kind source;
        bool manual, authenticated, frame_generation_evaluation, paired;
        refusal expected;
      };

      const candidate_row candidates[] {
        {"Auto refuses an unauthenticated UIAlpha tag", kind::alpha, false, false, false, false, refusal::ui_alpha_unauthenticated},
        {"Auto admits an authenticated UIAlpha tag", kind::alpha, false, true, false, false, refusal::none},
        {"Auto admits DLSS-G's own DLSSG.UIAlpha", kind::alpha, false, false, true, false, refusal::none},
        {"A manual choice admits an unauthenticated UIAlpha tag", kind::alpha, true, false, false, false, refusal::none},
        {"Auto admits an unauthenticated UI color tag", kind::color_and_alpha, false, false, false, false, refusal::none},
        {"Auto admits the tagged Backbuffer", kind::backbuffer, false, false, false, false, refusal::none},
        {"An unpaired HUD-less image is refused in Auto", kind::hudless, false, true, false, false, refusal::hudless_unpaired},
        {"An unpaired HUD-less image is refused to a manual choice", kind::hudless, true, true, false, false, refusal::hudless_unpaired},
        {"A paired HUD-less image is admitted", kind::hudless, false, false, false, true, refusal::none},
        {"DLSS-G's paired HUD-less image is admitted", kind::hudless, false, false, true, true, refusal::none},
      };
      for (const auto &r : candidates) {
        const auto got = sunshine_game3d::ui_mask::admit_candidate(r.source, r.manual, r.authenticated, r.frame_generation_evaluation, r.paired);
        if (got != r.expected) {
          char text[512];
          std::snprintf(text, sizeof(text), "P11 UI candidate, %s: refusal %d", r.pins, int(got));
          fail(text);
        }
      }

      using choice = sunshine_game3d::ui_qualification::choice;
      constexpr std::uint32_t rgba8 = 28, rgba8_typeless = 27, bgra8 = 87, bgrx8 = 88, rgb10a2 = 24, rgba16f = 10;

      struct current_row {
        const char *pins;
        bool requested, fg_known, fg_enabled;
        std::uint32_t format;
        choice wanted;
        const char *refused;  // nullptr: offered.
      };

      const current_row presented[] {
        {"Auto offers the presented color", true, false, false, rgba8, choice::automatic, nullptr},
        {"An explicit current-color choice offers it", true, false, false, bgra8, choice::current_color, nullptr},
        {"A typeless format with alpha is judged by its typed format", true, false, false, rgba8_typeless, choice::automatic, nullptr},
        {"A 10-bit format with alpha", true, false, false, rgb10a2, choice::automatic, nullptr},
        {"A float format with alpha", true, false, false, rgba16f, choice::automatic, nullptr},
        {"Confirmed FG Off offers it", true, true, false, rgba8, choice::automatic, nullptr},
        {"Active FG excludes it", true, true, true, rgba8, choice::automatic, "FG_active"},
        {"Active FG excludes an explicit current-color choice", true, true, true, rgba8, choice::current_color, "FG_active"},
        {"FG enabled before it is known does not exclude it", true, false, true, rgba8, choice::automatic, nullptr},
        {"A format without alpha", true, false, false, bgrx8, choice::automatic, "no_present_alpha"},
        {"Another explicit choice", true, false, false, rgba8, choice::sl_ui_alpha, "not_selected"},
        {"Another explicit choice: HUD-less", true, false, false, rgba8, choice::sl_hudless, "not_selected"},
        {"Without a request", false, false, false, rgba8, choice::automatic, "not_requested"},
        {"A missing request outranks active FG", false, true, true, bgrx8, choice::sl_backbuffer, "not_requested"},
        {"Active FG outranks a format without alpha", true, true, true, bgrx8, choice::sl_backbuffer, "FG_active"},
        {"A format without alpha outranks another choice", true, false, false, bgrx8, choice::sl_backbuffer, "no_present_alpha"},
      };
      for (const auto &r : presented) {
        sunshine_game3d::source_alpha_ui_decision status;
        status.requested = r.requested;
        status.fg.known = r.fg_known;
        status.fg.enabled = r.fg_enabled;
        const char *got = sunshine_game3d::ui_input::current_color_refusal(status, r.format, r.wanted);
        if ((got == nullptr) != (r.refused == nullptr) || (got && std::strcmp(got, r.refused) != 0)) {
          char text[512];
          std::snprintf(text, sizeof(text), "P11 presented color, %s: %s", r.pins, got ? got : "offered");
          fail(text);
        }
      }
      std::printf("PASS P11 UI admission: %zu candidates, %zu presented colors\n", std::size(candidates), std::size(presented));
    }
  }  // namespace
}  // namespace sunshine_streamline::depth_capture

int main() {
  using namespace sunshine_streamline::depth_capture;
  frame_generation_policy_timelines();
  frame_generation_scope_rows();
  ngx_frame_generation_rows();
  frame_generation_frame_steps();
  dlssg_tag_precedence_steps();
  deferral_present_steps();
  select_provider_rows();
  ownership_timelines();
  freshness_edges();
  stray_frame_generation_copy_steps();
  decide_display_rows();
  ui_admission_rows();
  reset();
  if (failures) {
    std::fprintf(stderr, "Seam characterization: %u rows changed\n", failures);
    return 1;
  }
  std::puts("Seam characterization: every pinned row holds");
  return 0;
}
