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
#include "depth_hold.h"
#include "freshness.h"
#include "game3d_ui_input_provider.h"
#include "game3d_ui_mask.h"
#include "ngx_depth_source.h"
#include "streamline_camera_probe.h"
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
  using sunshine_game3d::depth_output::retained_depth;
  using sunshine_game3d::depth_output::display_action;
  using sunshine_game3d::depth_output::display_decision;
  using sunshine_game3d::depth_output::frame_generation_copy;
  using sunshine_game3d::depth_output::decide_display;
  using sunshine_game3d::depth_output::decide_failed_copy;
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

    // A row that reads now once before its setup: the stamps its fixture
    // calls made after that read (make, begin_evaluation, mark_pending, each
    // a later GetTickCount64, which steps about every 15.6 ms) are now, as
    // when no step fell between them. A tick ahead of now is not fresh
    // (fresh_lt), so a step inside a row's setup failed it (open item C,
    // one run in 46: the FG owner whose own copies are gone). Stamps a row
    // ages on purpose (now - source_age - 1) stay as they are.
    void pin_clock(std::uint64_t now) {
      for (auto &value : capture_pool.slots) {
        if (value.metadata.tick > now) {
          value.metadata.tick = now;
        }
      }
      for (auto &space : evaluations) {
        for (auto &view : space.views) {
          if (view.tick > now) {
            view.tick = now;
          }
        }
        if (space.pending_nomination.tick > now) {
          space.pending_nomination.tick = now;
        }
      }
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
      // The viewport's newest frame key, as the probe's record keeps it.
      std::uint64_t key {};
      for (const auto &s : steps) {
        if (s.what == call::frame) {
          // The probe decides a frame's begin once (store_camera), by the
          // same rule.
          if (sunshine_streamline::frame_generation::frame_begins(std::exchange(key, s.value), s.value)) {
            frame_generation_frame(epoch, 0, enabled_tick);
          }
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

      // The frame key both that begin and DLSS-G's depth precedence compare
      // (sunshine_streamline::frame_key): every constants call of a frame on
      // a token the probe did not see minted repeats its address, so the
      // frame's second call begins none (until 10-27 the precedence keyed
      // each such call apart).
      using sunshine_streamline::frame_identity;
      using sunshine_streamline::frame_identity_kind;
      const frame_identity numeric {frame_identity_kind::v1_numeric, 0, 41, 0, true};
      const frame_identity minted {frame_identity_kind::v2_observed_token, 17, 41, 0x5000, true};
      const frame_identity first_call {frame_identity_kind::v2_constants_call, 30, 0, 0x7000, false};
      auto second_call = first_call;
      second_call.generation = 31;
      if (sunshine_streamline::frame_key(numeric) != 42 || sunshine_streamline::frame_key(minted) != 17 ||
          sunshine_streamline::frame_key(first_call) != 0x7000 || sunshine_streamline::frame_key({}) != 0 ||
          sunshine_streamline::frame_generation::frame_begins(sunshine_streamline::frame_key(first_call),
            sunshine_streamline::frame_key(second_call))) {
        fail("P5 frame key: a numeric frame, a minted token, a constants call's token or none keyed apart from its rule");
      }
      std::printf("PASS P5 frame key\n");
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

    // P5: the source age ends a deferral, and DLSS-G's evaluation, only at a
    // Present that cannot be its frame's own (deferral_evaluations::expired
    // and presented, by Present ordinals). Until 10-28 a deferral lapsed by
    // the clock at its own Present after a hitch of the source age between
    // its frame's calls and that Present, so the frame had no FG copy and the
    // Present was mono (open item A, seed 2d28df102b516f4d).
    void deferral_age_rows() {
      namespace fg = sunshine_streamline::frame_generation;
      struct row {
        const char *pins;
        std::uint64_t elapsed;  // Milliseconds from the tag call to the Present.
        std::uint32_t frames;  // The viewport's generated count while it generates.
        int evaluated;  // DLSS-G evaluated a frame after the tag call, at this many Presents after it (-1: none).
        unsigned presents;  // Presents since the tag call before this one.
        bool queue_executed;
        bool expired, presented;
      };
      constexpr auto age = sunshine_freshness::source_age_ms;
      const row rows[] {
        {"Within the source age a deferral never expires", age - 1, 0, -1, 5, false, false, false},
        {"Its frame's Present after a hitch of the source age, no Present since its tag call", age + 150, 0, -1, 0, true, false, true},
        {"after an earlier frame's generated Present at 2x", age + 150, 1, -1, 1, false, false, false},
        {"two Presents since at 2x: one of them was its frame's", age + 150, 1, -1, 2, false, true, false},
        {"three earlier generated Presents at 4x", age + 150, 3, -1, 3, false, false, false},
        {"four Presents since at 4x", age + 150, 3, -1, 4, false, true, false},
        {"Without frame generation a Present since its tag call was its frame's", age + 150, 0, -1, 1, true, true, true},
        {"DLSS-G evaluated its frame after the tag call, no Present since: the Present is that frame's", age + 150, 0, 2, 2, false, false, true},
        {"a Present came after that evaluation: past", age + 150, 0, 1, 2, false, true, false},
      };
      for (const auto &r : rows) {
        fg::deferral_evaluations value;
        constexpr std::uint64_t epoch = 7, start = 1000, tagged_at = 40;
        // An earlier deferred frame DLSS-G evaluated, so the deferral waits
        // for an evaluation of its own while DLSS-G evaluates every frame.
        value.tag(epoch, 0, 1, true, {}, tagged_at - 1);
        value.evaluated(epoch, 0, start - 10, tagged_at - 1, r.frames);
        const auto deferral = value.tag(epoch, 0, 2, true, {}, tagged_at);
        const auto ordinal = tagged_at + r.presents;
        if (r.evaluated >= 0) {
          value.evaluated(epoch, 0, start + 5, tagged_at + std::uint64_t(r.evaluated), r.frames);
        }
        const auto now = start + r.elapsed;
        const bool expired = value.expired(epoch, 0, deferral, start, r.frames, ordinal, now);
        const bool presented = value.presented(epoch, 0, deferral, r.queue_executed, now, ordinal);
        if (expired != r.expired || presented != r.presented) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5 deferral age, %s: expired %d presented %d", r.pins, expired, presented);
          fail(text);
        }
      }
      std::printf("PASS P5 deferral age by the frame's own Present: %zu rows\n", std::size(rows));
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
        capture_queues.queues[0] = std::make_unique<queue_state>();
        capture_queues.queues[1] = std::make_unique<queue_state>();
        auto &a = *capture_queues.queues[0];
        auto &b = *capture_queues.queues[1];
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
        capture_queues.queues[0].reset();
        capture_queues.queues[1].reset();
      }
      {
        // establish_provider keeps the delivered watermark for the same exact
        // source and restarts it for any other provider, source or epoch.
        // Ownership moving to the other provider also makes every copy
        // recorded before the new owner's historical (admission_after): the
        // restarted watermark cannot compare the two namespaces' sequences.
        reset();
        queue_state owner;

        struct step {
          provider_kind provider;
          std::uint64_t epoch, sequence, source;
          std::uint64_t kept_sequence;  // The watermark after establishing (0: restarted).
          bool historical;  // admission_after is the new owner's copy's id - 1 (else 0).
        };

        const step steps[] {
          {provider_kind::ngx, epoch, 1, ngx_source, 0, false},
          {provider_kind::ngx, epoch, 2, ngx_source, 1, false},
          {provider_kind::ngx, epoch + 1, 1, ngx_source, 0, false},
          {provider_kind::ngx, epoch + 1, 2, ngx_source + 1, 0, false},
          {provider_kind::streamline, epoch + 1, 3, ngx_source + 1, 0, true},
        };
        unsigned index = 0;
        for (const auto &s : steps) {
          auto &value = make(index % 2, s.provider, s.epoch, s.sequence, s.source);
          establish_provider(owner, value, 10 + index);
          if (!owner.provider_established || owner.provider != s.provider || owner.source_id != s.source || owner.nominated_epoch != s.epoch || owner.present != 10 + index || owner.present_capture != value.id || owner.last_sequence != s.kept_sequence ||
              owner.admission_after != (s.historical ? value.id - 1 : 0)) {
            char text[256];
            std::snprintf(text, sizeof(text), "P6 establish_provider, step %u: watermark %llu admission %llu", index, static_cast<unsigned long long>(owner.last_sequence), static_cast<unsigned long long>(owner.admission_after));
            fail(text);
          }
          deliver(owner, value);
          ++index;
        }
        // A handover to a copy recorded before the previous owner's last
        // selected one (a required FG scope's pick, which select_provider's
        // older_than_shown does not judge): that previous copy is historical
        // too, so it never returns as its provider's stand-in (10-28).
        reset();
        queue_state handed;
        auto &earlier = make(0, provider_kind::streamline, epoch, 1, 0);
        auto &shown = make(1, provider_kind::ngx, epoch, 1, ngx_source);
        establish_provider(handed, shown, 20);
        deliver(handed, shown);
        establish_provider(handed, earlier, 21);
        if (handed.provider != provider_kind::streamline || handed.admission_after != shown.id) {
          char text[256];
          std::snprintf(text, sizeof(text), "P6 establish_provider, handover to an earlier copy: admission %llu, the previous owner's copy %llu", static_cast<unsigned long long>(handed.admission_after), static_cast<unsigned long long>(shown.id));
          fail(text);
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
      // The same at the Present of the frame that made them: no Present came
      // since their call but this one (frame_presenting).
      capture_tick_own_frame,
      view_tick_own_frame,
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
        case edge_site::view_tick_own_frame:
          {
            auto &sl = make(0, provider_kind::streamline, epoch, 1, 0);
            const auto now = GetTickCount64();
            sl.metadata.tick = now;
            for (auto &view : views) {
              if (view.sequence) {
                view.tick = now - age;
              }
            }
            // An earlier Present and this one since the evaluation, or only
            // this one.
            fixture::presents_since(site == edge_site::view_tick ? 2 : 1);
            return pick_outcome(select_provider(owner, native, 1, now, {}), &sl, 'S', nullptr, '?');
          }
        case edge_site::capture_tick_own_frame:
          {
            auto &sl = make(0, provider_kind::streamline, epoch, 1, 0);
            sl.present = present_ordinal();
            const auto now = GetTickCount64();
            sl.metadata.tick = now - age;
            for (auto &view : views) {
              if (view.sequence) {
                view.tick = now - age;
              }
            }
            fixture::presents_since(1);
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
        // A frame's own view and capture at that frame's Present, after a
        // hitch between its call and its Present (10-27: inactive_views).
        {"An evaluation view exactly the source age old is active at its own frame's Present", e::view_tick_own_frame, limit, "S ready current_capture"},
        {"An evaluation view 700 ms old is active at its own frame's Present", e::view_tick_own_frame, limit + 450, "S ready current_capture"},
        {"A readable capture exactly the source age old is current at its own frame's Present", e::capture_tick_own_frame, limit, "S ready current_capture"},
        {"A readable capture 700 ms old is current at its own frame's Present", e::capture_tick_own_frame, limit + 450, "S ready current_capture"},
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

    // P5 and P6: a viewport's Streamline FG input is pending with none of its
    // snapshots completed (FG On's first frame, or DLSS-G's compute copy
    // still in flight at its own Present), while the frame's own SR snapshot
    // of that viewport (its display lineage) is readable on this queue: it
    // stands in (lineage_snapshot), also under the FG scope's policy. Only an
    // unconsumed snapshot of the lineage, within the source age, that this
    // queue can read does (until 10-27 none did: the Present held the
    // display's retained copy, mono once that was past the source age).
    void lineage_stand_in_rows() {
      enum class variant {
        plain,
        fg_policy,
        consumed,
        expired,
        other_epoch,
        other_viewport,
        unreadable
      };
      struct row {
        const char *pins;
        variant kind;
        char picked;  // S the SR snapshot, - none.
        status result;
        selection_reason selection;
      };

      const row rows[] {
        {"The FG input in flight: the frame's own SR snapshot stands in", variant::plain, 'S', status::ready, selection_reason::lineage_snapshot},
        {"The FG input in flight under its scope's policy: the SR snapshot stands in", variant::fg_policy, 'S', status::ready, selection_reason::lineage_snapshot},
        {"An SR snapshot this queue already showed does not stand in", variant::consumed, '-', status::recorded, selection_reason::no_completed_snapshot},
        {"An SR snapshot past the source age does not stand in", variant::expired, '-', status::recorded, selection_reason::no_completed_snapshot},
        {"An SR snapshot of an earlier epoch does not stand in", variant::other_epoch, '-', status::recorded, selection_reason::no_completed_snapshot},
        {"An SR snapshot of another viewport does not stand in", variant::other_viewport, '-', status::recorded, selection_reason::no_completed_snapshot},
        {"An SR snapshot on another queue that this queue cannot read does not stand in", variant::unreadable, '-', status::recorded, selection_reason::no_completed_snapshot},
      };
      constexpr std::uint64_t epoch = 9;
      for (const auto &r : rows) {
        reset();
        const auto now = GetTickCount64();
        queue_state owner;
        // The frame's SR evaluation (sequence 2), then DLSS-G's evaluation of
        // its FG input (sequence 3), which replaces the viewport's SR view.
        const auto sr_epoch = r.kind == variant::other_epoch ? epoch - 1 : epoch;
        const std::uint32_t sr_viewport = r.kind == variant::other_viewport ? 1 : 0;
        auto &sr = make(0, provider_kind::streamline, sr_epoch, 2, 0, sr_viewport);
        // Its evaluation nominated it, and its copy has storage.
        sr.texture = std::make_shared<texture_reference>();
        sr.source_nominated = true;
        if (r.kind == variant::expired) {
          sr.metadata.tick = now - source_age - 1;
        }
        if (r.kind == variant::unreadable) {
          // Recorded on another queue, which this queue cannot read.
          sr.queue = native + 1;
        }
        if (r.kind == variant::other_viewport) {
          // That viewport's evaluations stopped (no ambiguity of views).
          for (auto &view : evaluations[provider_index(provider_kind::streamline)].views) {
            if (view.viewport == sr_viewport) {
              view = {};
            }
          }
        }
        auto &fg = make(1, provider_kind::streamline, epoch, 3, fg_source, 0, 0);
        fg.finished = false;
        if (r.kind == variant::consumed) {
          establish_provider(owner, sr, 1);
          deliver(owner, sr);
        }
        const bool required = r.kind == variant::fg_policy;
        const selection_policy policy {required, required ? epoch : 0, 0, false, now - 16};
        pin_clock(now);
        const auto pick = select_provider(owner, native, 2, now, policy);
        const char picked = label(pick, &sr, 'S', &fg, 'F');
        if (picked != r.picked || pick.result != r.result || pick.selection != r.selection) {
          char text[512];
          std::snprintf(text, sizeof(text), "P5/P6 lineage stand-in, %s: picked %c result %s selection %s", r.pins, picked, name(pick.result), name(pick.selection));
          fail(text);
        }
      }
      std::printf("PASS P5/P6 lineage stand-in for a pending FG input: %zu rows\n", std::size(rows));
    }

    // P6: a missing frame (capture_pick::missing_frame, 10-28). The fresh
    // head of the one active view failed, is invalid or nomination-invalid,
    // is a nomination without readable pixels, or was refused before its
    // capture slot existed, while the newest copy of its display lineage
    // completed on another queue and was never shown. That copy stands in at
    // once (completed_snapshot of the head's own source; lineage_snapshot of
    // the viewport's SR source for an FG head), the pick moves no admission
    // watermark (interrupts_admission), and the next pending head finds the
    // copy consumed, never completed_before_gap. Until 10-28 the head revoked
    // it queue-wide: the Present held an older copy, and mono followed once
    // that passed the source age. Only a legal copy stands in, so each
    // protection keeps its row: a copy this queue cannot read (its fence open,
    // its recording not retired), past the source age, already shown, of
    // another epoch or viewport, or of another storage than a head with a
    // slot names does not, and ambiguous or inactive views, a head past the
    // source age, the FG scope's policy picks and a retired source revoke as
    // before.
    void missing_frame_rows() {
      enum class lineage {
        sr,  // SR copies on another queue, an SR head.
        fg_own,  // DLSS-G's copies on its compute queue, an FG head (FG On's scope).
        fg_from_sr  // The viewport's SR copy on another queue, an FG head (FG On).
      };
      enum class head {
        failed,
        invalid,
        nomination_invalid,
        nomination_only,
        refused_without_slot,
        // Not missing: its evaluation began and its nomination is in progress
        // (pending_nomination), its copy not recorded yet.
        in_progress
      };
      enum class variant {
        plain,
        fence_open,
        recording_open,
        expired,
        consumed,
        other_epoch,
        other_viewport,
        other_storage,
        ambiguous,
        inactive,
        expired_head,
        fg_excluded,
        fg_excluded_unreadable,
        fg_mismatch,
        fg_mismatch_unreadable,
        retired,
        // The owner's association expired (expire_association: every attempt
        // of its views rejected for the association window).
        association_expired
      };
      struct row {
        const char *pins;
        lineage line;
        head kind;
        variant change;
        const char *outcome;  // The pick: C the copy, H the head's own slot, - none.
        bool interrupts;  // acquire moves the admission watermark.
        const char *next;  // The next pending head's pick, after this one is shown; null: not checked.
      };
      using l = lineage;
      using h = head;
      using v = variant;

      constexpr const char *consumed_next = "- recorded completed_already_consumed";
      constexpr const char *fg_next = "- recorded no_completed_snapshot";
      // After an interruption the next pending head finds the copy revoked.
      constexpr const char *gap_next = "- recorded completed_before_gap";
      const row rows[] {
        {"SR, a failed head: the SR copy completed on another queue stands in", l::sr, h::failed, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"SR, an invalid head: the SR copy stands in", l::sr, h::invalid, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"SR, a nomination-invalid head: the SR copy stands in", l::sr, h::nomination_invalid, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"SR, a nomination-only head without pixels: the SR copy stands in for it", l::sr, h::nomination_only, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"SR, a head refused before its slot existed: the SR copy stands in", l::sr, h::refused_without_slot, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG, a failed DLSS-G attempt: its previous compute copy stands in", l::fg_own, h::failed, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG, an invalid DLSS-G attempt: its previous compute copy stands in", l::fg_own, h::invalid, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG, a nomination-invalid DLSS-G attempt: its previous compute copy stands in", l::fg_own, h::nomination_invalid, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG, a DLSS-G nomination without pixels: its previous compute copy stands in", l::fg_own, h::nomination_only, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG, a DLSS-G attempt refused before its slot existed: its previous compute copy stands in", l::fg_own, h::refused_without_slot, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG On, a failed first DLSS-G attempt: the frame's SR copy stands in", l::fg_from_sr, h::failed, v::plain, "C ready lineage_snapshot", false, fg_next},
        {"FG On, an invalid first DLSS-G attempt: the SR copy stands in", l::fg_from_sr, h::invalid, v::plain, "C ready lineage_snapshot", false, fg_next},
        {"FG On, a nomination-invalid first DLSS-G attempt: the SR copy stands in", l::fg_from_sr, h::nomination_invalid, v::plain, "C ready lineage_snapshot", false, fg_next},
        {"FG On, a first DLSS-G nomination without pixels: the SR copy stands in", l::fg_from_sr, h::nomination_only, v::plain, "C ready lineage_snapshot", false, fg_next},
        {"FG On, a first DLSS-G attempt refused before its slot existed: the SR copy stands in", l::fg_from_sr, h::refused_without_slot, v::plain, "C ready lineage_snapshot", false, fg_next},
        // A nomination in progress ends recorded (pending) or refused (missing):
        // either takes this copy, so a Present meanwhile shows it now (until
        // 10-28 it held the copy before, one Present late).
        {"SR, a nomination in progress: the SR copy completed before it stands in", l::sr, h::in_progress, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG, a DLSS-G nomination in progress: its previous compute copy stands in", l::fg_own, h::in_progress, v::plain, "C ready completed_snapshot", false, consumed_next},
        {"FG, a DLSS-G nomination in progress of an FG owner whose own copies are gone: the frame's SR copy stands in", l::fg_from_sr, h::in_progress, v::plain, "C ready lineage_snapshot", false, fg_next},
        {"A nomination in progress whose copy this queue already showed holds as pending", l::sr, h::in_progress, v::consumed, "- recorded pending_nomination", false, nullptr},
        {"A nomination in progress with no readable copy holds as pending", l::sr, h::in_progress, v::fence_open, "- recorded pending_nomination", false, nullptr},
        // Per-copy protections: no stand-in, and the missing frame still moves nothing.
        {"A copy whose producer fence is open does not stand in", l::sr, h::failed, v::fence_open, "- failed no_admissible_capture", false, nullptr},
        {"A copy whose recording has not retired (no copy-out) does not stand in", l::sr, h::failed, v::recording_open, "- failed no_admissible_capture", false, nullptr},
        {"A copy past the source age does not stand in", l::sr, h::failed, v::expired, "- failed no_admissible_capture", false, nullptr},
        {"A copy this queue already showed does not stand in", l::sr, h::refused_without_slot, v::consumed, "- missing_state no_current_nomination", false, nullptr},
        {"An FG copy this queue already showed does not stand in", l::fg_own, h::failed, v::consumed, "- failed no_admissible_capture", false, nullptr},
        {"An SR copy this queue already showed does not stand in for an FG head", l::fg_from_sr, h::refused_without_slot, v::consumed, "- missing_state no_current_nomination", false, nullptr},
        {"A copy of an earlier epoch does not stand in", l::sr, h::failed, v::other_epoch, "- failed no_admissible_capture", false, nullptr},
        {"A copy of another viewport does not stand in", l::sr, h::failed, v::other_viewport, "- failed no_admissible_capture", false, nullptr},
        {"An SR copy of another viewport does not stand in for an FG head", l::fg_from_sr, h::failed, v::other_viewport, "- failed no_admissible_capture", false, nullptr},
        {"A copy of another storage than the failed head's does not stand in", l::sr, h::failed, v::other_storage, "- failed no_admissible_capture", false, nullptr},
        {"A head refused before its slot existed names no storage: the copy stands in", l::sr, h::refused_without_slot, v::other_storage, "C ready completed_snapshot", false, nullptr},
        // Identity interruptions revoke as before: once the interruption is
        // over (the second view stopped, the FG policy changed), the next
        // pending head finds the copy completed before the gap.
        {"A second active view is ambiguous and revokes", l::sr, h::failed, v::ambiguous, "- ambiguous ambiguous_views", true, gap_next},
        {"No active view revokes", l::sr, h::failed, v::inactive, "- stale inactive_views", true, gap_next},
        {"A head past the source age that this Present does not present revokes", l::sr, h::failed, v::expired_head, "- failed no_admissible_capture", true, gap_next},
        {"An unconfirmed FG scope excludes a failed FG head and revokes", l::fg_own, h::failed, v::fg_excluded, "- stale fg_scope_missing", true, gap_next},
        {"An unconfirmed FG scope excludes an FG head refused before its slot existed and revokes", l::fg_own, h::refused_without_slot, v::fg_excluded, "- stale fg_scope_missing", true, gap_next},
        {"An unconfirmed FG scope excludes an FG head refused before its slot existed, with no copy to stand in, and revokes", l::fg_own, h::refused_without_slot, v::fg_excluded_unreadable, "- stale fg_scope_missing", true, gap_next},
        {"An unconfirmed FG scope still admits the SR stand-in of an FG head", l::fg_from_sr, h::refused_without_slot, v::fg_excluded, "C ready lineage_snapshot", false, nullptr},
        {"A required FG scope refuses the SR stand-in of an SR head refused before its slot existed and revokes", l::sr, h::refused_without_slot, v::fg_mismatch, "- unavailable fg_scope_mismatch", true, gap_next},
        {"A required FG scope refuses an SR head refused before its slot existed, with no copy to stand in, and revokes", l::sr, h::refused_without_slot, v::fg_mismatch_unreadable, "- unavailable fg_scope_mismatch", true, gap_next},
        {"A required FG scope refuses a failed SR head and revokes", l::sr, h::failed, v::fg_mismatch, "- unavailable fg_scope_mismatch", true, gap_next},
        {"FG Off retired the FG copies: FG On's refused attempt has no historical copy", l::fg_own, h::refused_without_slot, v::retired, "- missing_state no_current_nomination", false, nullptr},
        // expire_association revokes the owner's copies itself: the failed
        // head finds nothing to stand in, nor does the next pending one.
        {"An expired association revokes the copy completed before it", l::sr, h::failed, v::association_expired, "- failed no_admissible_capture", false, gap_next},
      };
      constexpr std::uint64_t epoch = 11, device = 41;
      for (const auto &r : rows) {
        reset();
        const auto now = GetTickCount64();
        // The producer queue (another render queue, or DLSS-G's compute queue):
        // selection compares its identity and reads its fence.
        fixture::completion_fence fence;
        auto &place = capture_queues.queues[0];
        place = std::make_unique<queue_state>();
        place->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&fence);
        place->fence.p = &fence;
        place->device_identity = device;
        struct release_place {
          std::unique_ptr<queue_state> &entry;
          ~release_place() { entry->queue.p = nullptr; entry->fence.p = nullptr; entry.reset(); }
        } release{place};
        const auto producer = reinterpret_cast<std::uint64_t>(&fence);
        queue_state owner;
        owner.device_identity = device;
        const bool fg_head = r.line != l::sr;
        const auto copy_source = r.line == l::fg_own ? fg_source : 0;
        const auto head_source = fg_head ? fg_source : 0;
        const auto layout = [&](DXGI_FORMAT format) {
          auto value = std::make_shared<source_reference>();
          value->device_identity = device;
          value->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
          value->desc.Format = format;
          return value;
        };
        const auto storage = layout(DXGI_FORMAT_R32_TYPELESS);
        const auto copy_epoch = r.change == v::other_epoch ? epoch - 1 : epoch;
        const std::uint32_t copy_viewport = r.change == v::other_viewport ? 1 : 0;
        // The owner's delivered copy (sequence 10), then the lineage's copy
        // completed on the producer queue (sequence 20), never shown. An FG
        // owner's nomination in progress over the frame's SR copy: the owner
        // showed an FG copy, which the pool has since recycled.
        const bool fg_owner = r.kind == h::in_progress && r.line == l::fg_from_sr;
        auto &shown = make(0, provider_kind::streamline, copy_epoch, 10, fg_owner ? fg_source : copy_source, copy_viewport);
        establish_provider(owner, shown, 1);
        deliver(owner, shown);
        if (fg_owner) {
          shown = {};
        }
        auto &copy = make(1, provider_kind::streamline, copy_epoch, 20, copy_source, copy_viewport);
        copy.queue = producer;
        copy.producer_fence = 5;
        const bool unreadable = r.change == v::fence_open || r.change == v::fg_excluded_unreadable ||
          r.change == v::fg_mismatch_unreadable;
        fence.completed = unreadable ? 4 : 5;
        copy.texture = std::make_shared<texture_reference>();
        copy.source_nominated = true;
        copy.compute_list = r.line == l::fg_own;
        copy.metadata.source = storage;
        if (r.change == v::other_storage) {
          copy.metadata.source = layout(DXGI_FORMAT_R16_UNORM);
        }
        if (r.change == v::recording_open) {
          // Its game recording is still open (not Reset), and no copy-out was made.
          copy.command = 900;
          copy.producer_recording = std::make_shared<sunshine_native_command::recording_lifetime>();
          copy.producer_recording->cookie = 900;
        }
        if (r.change == v::expired) {
          copy.metadata.tick = now - source_age - 1;
        }
        if (r.change == v::consumed) {
          deliver(owner, copy);
        }
        if (r.change == v::other_viewport) {
          // That viewport's evaluations stopped (no ambiguity of views).
          for (auto &view : evaluations[provider_index(provider_kind::streamline)].views) {
            if (view.viewport == copy_viewport) {
              view = {};
            }
          }
        }
        if (r.change == v::retired) {
          // FG Off retires the FG source, its copies with it; FG On follows,
          // which restores the role (frame_generation_restored).
          retire_source(provider_kind::streamline, epoch, fg_source);
          frame_generation_restored(epoch, 0);
        }
        if (r.change == v::ambiguous) {
          // Another viewport evaluates concurrently.
          begin_evaluation(epoch, 25, 1, provider_kind::streamline, 0);
        }
        // The head (sequence 30): an FG head replaces the viewport's SR view.
        slot *own_slot = nullptr;
        evaluation_nomination nominating;
        if (r.kind == h::refused_without_slot) {
          begin_evaluation(epoch, 30, 0, provider_kind::streamline, head_source, fg_head ? 0 : UINT64_MAX);
          attempt_status[provider_index(provider_kind::streamline)] = status::missing_state;
        } else if (r.kind == h::in_progress) {
          nominating.begin(epoch, 30, 0, provider_kind::streamline, head_source, fg_head ? 0 : UINT64_MAX);
        } else {
          own_slot = &make(2, provider_kind::streamline, epoch, 30, head_source, 0, fg_head ? 0 : UINT64_MAX);
          own_slot->metadata.source = storage;
          if (r.kind == h::failed) {
            own_slot->success = false;
          } else if (r.kind == h::invalid) {
            own_slot->invalid = true;
            own_slot->failure = capture_failure::replay;
          } else if (r.kind == h::nomination_invalid) {
            own_slot->texture = std::make_shared<texture_reference>();
            own_slot->source_nominated = own_slot->nomination_invalid = true;
          } else {
            own_slot->source_nominated = own_slot->nomination_only = true;
            own_slot->pixel_failure = status::unsupported_resource;
          }
        }
        auto &views = evaluations[provider_index(provider_kind::streamline)].views;
        if (r.change == v::inactive || r.change == v::expired_head) {
          // Two Presents since: none presents the head's frame.
          fixture::presents_since(2);
          if (r.change == v::inactive) {
            for (auto &view : views) {
              if (view.sequence) {
                view.tick = now - source_age - 1;
              }
            }
          }
          if (own_slot) {
            own_slot->metadata.tick = now - source_age - 1;
          }
        }
        if (r.change == v::association_expired) {
          // Every attempt of the owner's views rejected for longer than the
          // association window.
          for (auto &view : views) {
            if (view.sequence) {
              view.rejected_since = now - sunshine_freshness::association_ms - 1;
            }
          }
          expire_association(owner, now);
        }
        selection_policy policy {};
        if (fg_head || r.change == v::fg_mismatch || r.change == v::fg_mismatch_unreadable) {
          policy = {true, epoch, 0, false, now - 16};
        }
        if (r.change == v::fg_excluded || r.change == v::fg_excluded_unreadable) {
          policy = {};
          policy.exclude_unconfirmed_fg = true;
        }
        // The pool keeps the copy that stands in, the head's own completed
        // copy as beneath a pending head, and an FG head's SR copy (until
        // 10-28 kept by no rule: readable and reclaimable at once, the
        // Present's idle sweep or another record could recycle it first).
        const bool stands_in = std::string(r.outcome) == "C ready completed_snapshot" || std::string(r.outcome) == "C ready lineage_snapshot";
        const bool kept = !reclaimable(copy);
        pin_clock(now);
        const auto pick = select_provider(owner, native, 2, now, policy);
        const auto got = pick_outcome(pick, &copy, 'C', own_slot, 'H');
        const bool interrupts = interrupts_admission(owner, pick);
        std::string next = "-";
        if (r.next) {
          // acquire()'s watermark and, for a shown copy, its establishment and
          // complete_frame().
          if (interrupts) {
            owner.admission_after = serial.load();
          }
          if (pick.value && pick.result == status::ready) {
            establish_provider(owner, *pick.value, 2);
            deliver(owner, *pick.value);
          }
          // The interruption ends: the second view stops, the FG policy
          // follows the head (a confirmed scope for an FG head, none for an
          // SR one), and the copy's work completes, so only the gap can
          // refuse it.
          if (r.change == v::ambiguous) {
            for (auto &view : views) {
              if (view.viewport == 1) {
                view = {};
              }
            }
          }
          auto next_policy = policy;
          if (r.change == v::fg_excluded || r.change == v::fg_excluded_unreadable || r.change == v::fg_mismatch ||
              r.change == v::fg_mismatch_unreadable) {
            next_policy = fg_head ? selection_policy {true, epoch, 0, false, now - 16} : selection_policy {};
          }
          fence.completed = 5;
          // The next frame's head is recorded and still in flight.
          auto &pending = make(3, provider_kind::streamline, epoch, 40, head_source, 0, fg_head ? 0 : UINT64_MAX);
          pending.metadata.source = storage;
          pending.finished = false;
          fixture::presents_since(1);
          next = pick_outcome(select_provider(owner, native, 3, GetTickCount64(), next_policy), &copy, 'C', &pending, 'P');
        }
        if (got != r.outcome || interrupts != r.interrupts || (r.next && next != r.next) || (stands_in && !kept)) {
          char text[640];
          std::snprintf(text, sizeof(text), "P6 missing frame, %s: %s, interrupts %d, next %s, kept %d", r.pins, got.c_str(), interrupts, next.c_str(), kept);
          fail(text);
        }
      }
      std::printf("PASS P6 missing frame: %zu rows\n", std::size(rows));
    }

    // P6: FG Off retires only the FG role of its viewport's display lineage
    // (retire_source, 10-28). Its copies are invalidated and the queue's
    // association ends, but the owner's admission floor is its last selected
    // copy (present_capture) and it keeps its consumed frame, and the FG
    // nomination that was the namespace's newest passes to the viewport's
    // ordinary source as a missing frame of the lineage. So the SR copies
    // recorded since the shown FG copy stand in at once. Until 10-28 FG Off
    // moved an FG owner's watermark to serial, or left a head no view named,
    // which acquire read as an interruption (an SR owner whose newest
    // nomination was an FG one), and either revoked those copies: the
    // Present held older depth (completed_before_gap) or went mono. An FG
    // owner meets a later SR copy at FG Off only when no Present selected
    // that copy first: its frame's options call came after its evaluations
    // (a late Off), or its list was executed after its Present. Each row
    // runs acquire()'s bookkeeping at each Present: the gap watermark, the
    // establishment of any selected copy (source authority, ready or not),
    // and complete_frame of a shown one; the owner sits in the owner table,
    // where retire_source finds it. F is the shown FG copy (sequence 10), S
    // an SR copy of the viewport, H the FG-Off frame's own SR copy, G an FG
    // copy recorded after FG Off, N an NGX copy, C an SR owner's shown copy.
    void fg_off_role_rows() {
      enum class shape {
        unsubmitted_sr,  // S: a silent frame's SR copy after F, its list executed after its Present.
        late_off,  // S: the same, then FG Off before any Present (a late Off).
        sr_same_queue,  // S readable at its own Present.
        sr_before_fg,  // S published before F (an earlier frame).
        sr_called_before_fg,  // S published after F, called before it (sequence 9).
        sr_called_before_fg_then_on,  // The same, then FG On within the source age.
        no_sr_before_off,  // No SR copy before FG Off.
        off_then_on,  // S as unsubmitted_sr, then FG On again within the source age.
        sr_owner_fg_head,  // An SR owner; the newest nomination DLSS-G's FG one, in flight.
        sr_owner_fg_head_sr_view_kept,  // The same, the viewport's SR view kept (no production adapter does).
        other_viewport,  // Another viewport's view in the epoch, past the source age; an FG owner.
        other_viewport_active,  // Another viewport's view active at FG Off (the guard); an SR owner.
        same_present,  // A repeated effect of the presentation that showed F, after FG Off.
        fg_after_off_then_on,  // G of a frame before the Off, recorded after it; SR shown; FG On.
        fg_after_off_sr_unshown_then_on,  // The same with S readable and never shown.
        fg_after_off_alone_then_on,  // The same with no S.
        fg_after_off_rekeyed,  // G as the add-on records it after the Off: the viewport's ordinary source.
        off_before_complete,  // FG Off between F's selection and its copy (complete_frame); then S of an earlier frame.
        ngx_before_fg,  // N published before F.
        ngx_after_fg_then_on,  // N published after S takes the queue; FG On.
        ngx_release  // An NGX owner's feature released: still a source end.
      };

      struct row {
        const char *pins;
        shape kind;
        const char *trace;
      };

      const row rows[] {
        {"A silent frame's SR copy after F, executed after its Present, stands in at the Present after FG Off", shape::unsubmitted_sr, "P1 F ready current_capture; P2 - recorded no_completed_snapshot; OFF floor=F; P3 S ready completed_snapshot; P4 H ready current_capture"},
        {"A late Off: the frame's SR copy after F shows at its own Present", shape::late_off, "P1 F ready current_capture; OFF floor=F; P2 S ready current_capture"},
        {"SR readable at its own Present took the queue before FG Off", shape::sr_same_queue, "P1 F ready current_capture; P2 S ready current_capture; OFF floor=0; P3 - recorded completed_already_consumed; P4 H ready current_capture"},
        {"An SR copy published before F stays historical", shape::sr_before_fg, "P1 F ready current_capture; OFF floor=F; P2 - recorded completed_before_gap; P3 H ready current_capture"},
        {"An SR copy published after F of a frame called before it stays consumed", shape::sr_called_before_fg, "P1 F ready current_capture; P2 - submitted no_completed_snapshot; OFF floor=F; P3 - recorded completed_already_consumed; P4 H ready current_capture"},
        {"The same S never stands in for FG On's pending input either", shape::sr_called_before_fg_then_on, "P1 F ready current_capture; P2 - submitted no_completed_snapshot; OFF floor=F; P3 - recorded no_completed_snapshot"},
        {"No SR copy before FG Off: the FG-Off frame's own SR copy shows once readable", shape::no_sr_before_off, "P1 F ready current_capture; OFF floor=F; P2 - recorded no_completed_snapshot; P3 H ready current_capture"},
        {"FG Off then On within the source age: S stands in for FG On's pending input", shape::off_then_on, "P1 F ready current_capture; P2 - recorded no_completed_snapshot; OFF floor=F; P3 S ready lineage_snapshot"},
        {"An SR owner whose newest nomination was DLSS-G's: S stands in, no gap", shape::sr_owner_fg_head, "P1 C ready current_capture; P2 S submitted no_completed_snapshot; OFF floor=0; P3 S ready completed_snapshot; P4 - recorded completed_already_consumed; P5 H ready current_capture"},
        {"The same with the viewport's SR view kept beside the FG one", shape::sr_owner_fg_head_sr_view_kept, "P1 C ready current_capture; P2 S submitted no_completed_snapshot; OFF floor=0; P3 S ready completed_snapshot; P4 - recorded completed_already_consumed; P5 H ready current_capture"},
        {"Another viewport's view past the source age makes nothing ambiguous: DLSS-G's head passes to SR, S stands in", shape::other_viewport, "P1 F ready current_capture; P2 - recorded no_completed_snapshot; OFF floor=F views=2; P3 S ready completed_snapshot; P4 H ready current_capture"},
        {"Another viewport's view active at FG Off: DLSS-G's head goes without an SR head, ambiguous and revoking as before", shape::other_viewport_active, "P1 F ready current_capture; P2 S submitted no_completed_snapshot; OFF floor=0 views=1; P3 - unavailable no_current_nomination [gap]; P4 - ambiguous ambiguous_views [gap]; P5 - ambiguous ambiguous_views [gap]"},
        {"A repeated effect of F's presentation selects nothing new after FG Off", shape::same_present, "P1 F ready current_capture; OFF floor=F; P1 - submitted completed_not_older; P2 S ready current_capture"},
        {"An FG copy of a frame before the Off, recorded after it, never shows once SR was shown", shape::fg_after_off_then_on, "P1 F ready current_capture; OFF floor=F; P2 S ready current_capture; P3 - recorded completed_already_consumed"},
        {"The same FG copy beside a newer readable SR copy never shown: FG On shows the newer frame (until 10-28 G, completed_snapshot)", shape::fg_after_off_sr_unshown_then_on, "P1 F ready current_capture; OFF floor=F; P2 S ready lineage_snapshot"},
        {"The same FG copy alone: the lineage's own finished frame within the source age shows at FG On", shape::fg_after_off_alone_then_on, "P1 F ready current_capture; OFF floor=F; P2 G ready completed_snapshot"},
        {"An FG copy recorded while FG stays off is the viewport's ordinary source: it shows after FG Off, one view (until 10-28 the FG view beside SR, ambiguous)", shape::fg_after_off_rekeyed, "P1 F ready current_capture; OFF floor=F; P2 G ready current_capture; P3 H ready current_capture"},
        {"FG Off between F's selection and its copy: F is still consumed, so an SR copy of an earlier frame published after it stays refused (until 10-28 it showed: older after newer)", shape::off_before_complete, "P1 C ready current_capture; P2 F ready current_capture; OFF floor=F; P3 - unavailable no_current_nomination"},
        {"An NGX copy published before F stays historical", shape::ngx_before_fg, "P1 F ready current_capture; OFF floor=F; P2 - submitted completed_before_gap"},
        {"An NGX copy after S takes the queue; FG On's stand-in never steps back to S", shape::ngx_after_fg_then_on, "P1 F ready current_capture; P2 - recorded no_completed_snapshot; OFF floor=F; P3 N ready current_capture; P4 - recorded no_completed_snapshot"},
        {"An NGX feature's release still revokes every copy before it", shape::ngx_release, "P1 N ready current_capture; OFF floor=serial; P2 - stale inactive_views"},
      };
      constexpr std::uint64_t epoch = 11, device = 41, ngx_epoch = 30, ngx_source = 400;
      for (const auto &r : rows) {
        reset();
        // The producer queue (the game's own queue), whose fence selection reads.
        fixture::completion_fence fence;
        auto &place = capture_queues.queues[0];
        place = std::make_unique<queue_state>();
        place->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&fence);
        place->fence.p = &fence;
        place->device_identity = device;
        capture_queues.queues[1] = std::make_unique<queue_state>();
        // The owner's own queue, by which complete_frame finds it.
        fixture::completion_fence owner_queue;
        capture_queues.queues[1]->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&owner_queue);
        struct release_places {
          ~release_places() {
            capture_queues.queues[0]->queue.p = nullptr;
            capture_queues.queues[0]->fence.p = nullptr;
            capture_queues.queues[0].reset();
            capture_queues.queues[1]->queue.p = nullptr;
            capture_queues.queues[1].reset();
          }
        } release;
        const auto producer = reinterpret_cast<std::uint64_t>(&fence);
        auto &owner = *capture_queues.queues[1];
        owner.device_identity = device;
        auto storage = std::make_shared<source_reference>();
        storage->device_identity = device;
        storage->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        storage->desc.Format = DXGI_FORMAT_R32_TYPELESS;
        // A copy of the viewport's SR input, on the producer queue or this one.
        const auto sr_copy = [&](slot &value, bool other_queue) -> slot & {
          value.metadata.source = storage;
          value.texture = std::make_shared<texture_reference>();
          value.source_nominated = true;
          if (other_queue) {
            value.queue = producer;
            value.producer_fence = 5;
          }
          return value;
        };
        slot *f = nullptr, *s = nullptr, *h = nullptr, *g = nullptr, *n = nullptr, *c = nullptr;
        unsigned index = 0;
        std::uint64_t present = 1;
        std::string trace;
        const auto who = [&](const capture_pick &pick) {
          char out = '-';
          for (const auto &[value, name] : {std::pair {f, 'F'}, std::pair {s, 'S'}, std::pair {h, 'H'}, std::pair {g, 'G'}, std::pair {n, 'N'}, std::pair {c, 'C'}}) {
            if (pick.value && pick.value == value) {
              out = name;
            }
          }
          return pick.value && out == '-' ? '?' : out;
        };
        // One Present's selection and acquire()'s bookkeeping: the gap
        // watermark, the establishment of any selected copy (source
        // authority, whatever its pixels' state), and complete_frame of a
        // shown one. advance: the next selection is another presentation.
        // copied: the consumer copied a ready copy at once (complete_frame);
        // otherwise its copy comes later (copy_later).
        const slot *selected_later = nullptr;
        std::uint64_t selected_present = 0;
        const auto pick = [&](selection_policy policy = {}, bool advance = true, bool copied = true) {
          const auto chosen = select_provider(owner, native, present, GetTickCount64(), policy);
          trace += std::string(trace.empty() ? "" : "; ") + "P" + std::to_string(present) + " " + who(chosen) + " " + name(chosen.result) + " " + name(chosen.selection);
          if (interrupts_admission(owner, chosen)) {
            owner.admission_after = serial.load();
            trace += " [gap]";
          }
          if (chosen.value) {
            establish_provider(owner, *chosen.value, present);
            if (chosen.result == status::ready && copied) {
              deliver(owner, *chosen.value);
            } else if (chosen.result == status::ready) {
              selected_later = chosen.value;
              selected_present = present;
            }
          }
          fixture::presents_since(1);
          present += advance;
        };
        // The consumer's copy of the copy selected without one, as
        // complete_frame commits it.
        const auto copy_later = [&] {
          packet value;
          value.queue = reinterpret_cast<std::uint64_t>(&owner_queue);
          value.capture_id = selected_later->id;
          value.metadata = selected_later->metadata;
          complete_frame(value, selected_present);
        };
        // FG Off: the options call after the evaluations before it (its
        // observation serial at or above their sequences, 30 at most).
        const auto off = [&](provider_kind provider, std::uint64_t retired_epoch, std::uint64_t source) {
          retire_source(provider, retired_epoch, source, 30);
          const auto floor = owner.admission_after;
          const auto text = floor == 0 ? std::string("0") : f && floor == f->id ? std::string("F") : floor == serial.load() ? std::string("serial") : std::to_string(floor);
          trace += "; OFF floor=" + text;
          if (r.kind == shape::other_viewport || r.kind == shape::other_viewport_active) {
            unsigned views {};
            for (const auto &view : evaluations[provider_index(provider_kind::streamline)].views) {
              views += view.sequence != 0;
            }
            trace += " views=" + std::to_string(views);
          }
        };
        const auto fg_copy = [&](std::uint64_t sequence) -> slot & {
          auto &value = make(index++, provider_kind::streamline, epoch, sequence, fg_source, 0, 0);
          value.metadata.source = storage;
          return value;
        };
        const auto sr = [&](std::uint64_t sequence, bool other_queue) -> slot & {
          return sr_copy(make(index++, provider_kind::streamline, epoch, sequence, 0, 0, fg_source), other_queue);
        };
        const auto ngx = [&](std::uint64_t sequence) -> slot & {
          return sr_copy(make(index++, provider_kind::ngx, ngx_epoch, sequence, ngx_source), false);
        };
        // The FG-Off frame's own SR copy H, in flight, then complete.
        const auto own_frame = [&] {
          h = &sr(30, true);
          h->finished = false;
          pick();
          h->finished = true;
          pick();
        };
        // A Present while S's list is not executed yet (no selection of S),
        // then its execution, still in flight.
        const auto unsubmitted = [&] {
          s->producer_submitted = false;
          pick();
          s->producer_submitted = true;
        };
        const auto fg_on = [&] {
          // FG On: its options call restores the FG role
          // (frame_generation_restored), then its first FG nomination is
          // pending; the scope requires FG.
          frame_generation_restored(epoch, 0);
          h = &fg_copy(30);
          h->finished = false;
          pick({true, epoch, 0, false, GetTickCount64() - 16});
        };
        fence.completed = 5;
        switch (r.kind) {
          case shape::sr_owner_fg_head:
          case shape::sr_owner_fg_head_sr_view_kept:
            // An SR owner: C shown; its next copy S in flight at its Present.
            c = &sr(10, true);
            pick();
            s = &sr(20, true);
            fence.completed = 4;
            pick();
            // DLSS-G's FG nomination of the next frame, its copy in flight.
            f = &make(index++, provider_kind::streamline, epoch, 30, fg_source, 0, r.kind == shape::sr_owner_fg_head ? 0 : UINT64_MAX);
            f->finished = false;
            off(provider_kind::streamline, epoch, fg_source);
            fence.completed = 5;
            pick();
            h = &sr(40, true);
            h->finished = false;
            pick();
            h->finished = true;
            pick();
            break;
          case shape::ngx_release:
            n = &ngx(1);
            pick();
            off(provider_kind::ngx, ngx_epoch, ngx_source);
            pick();
            break;
          case shape::off_before_complete:
            // C (sequence 5) shown; F (sequence 10) selected at the next
            // Present, FG Off on the game's thread before its copy is
            // committed; then S, an SR copy of the frame before F (sequence 9,
            // called before F's tag) published after F, readable.
            c = &sr(5, false);
            pick();
            f = &fg_copy(10);
            pick({}, true, false);
            off(provider_kind::streamline, epoch, fg_source);
            copy_later();
            s = &sr(9, false);
            pick();
            break;
          default:
            {
              if (r.kind == shape::sr_before_fg) {
                s = &sr(5, true);
              }
              if (r.kind == shape::ngx_before_fg) {
                n = &ngx(1);
              }
              if (r.kind == shape::other_viewport) {
                // Viewport 1 evaluated in this epoch, past the source age now.
                begin_evaluation(epoch, 2, 1, provider_kind::streamline, 0);
                for (auto &view : evaluations[provider_index(provider_kind::streamline)].views) {
                  if (view.viewport == 1) {
                    view.tick = GetTickCount64() - source_age - 1;
                  }
                }
                fixture::presents_since(2);
              }
              // FG's copy F of a live scope's frame, on the presenting queue, shown.
              f = &fg_copy(10);
              pick();
              switch (r.kind) {
                case shape::unsubmitted_sr:
                case shape::off_then_on:
                case shape::other_viewport:
                case shape::ngx_after_fg_then_on:
                  // A silent frame's SR copy (its scope missed the frame
                  // before), executed on its queue after its Present.
                  s = &sr(20, true);
                  fence.completed = 4;
                  unsubmitted();
                  break;
                case shape::late_off:
                  s = &sr(20, true);
                  fence.completed = 4;
                  break;
                case shape::sr_same_queue:
                  s = &sr(20, false);
                  pick();
                  break;
                case shape::other_viewport_active:
                  s = &sr(20, true);
                  fence.completed = 4;
                  pick();
                  // Viewport 1 evaluates in this epoch now.
                  begin_evaluation(epoch, 22, 1, provider_kind::streamline, 0);
                  break;
                case shape::sr_called_before_fg:
                case shape::sr_called_before_fg_then_on:
                  // Called before F (sequence 9): its evaluation keeps F's frame.
                  s = &sr(9, true);
                  fence.completed = 4;
                  pick();
                  break;
                case shape::same_present:
                  // F's presentation again (its effects); S recorded meanwhile.
                  --present;
                  s = &sr(20, false);
                  break;
                default:
                  break;
              }
              if (r.kind == shape::ngx_after_fg_then_on) {
                n = &ngx(1);
              }
              if (r.kind == shape::other_viewport || r.kind == shape::other_viewport_active) {
                // DLSS-G's FG nomination of the next frame, the newest, in flight.
                fg_copy(25).finished = false;
              }
              off(provider_kind::streamline, epoch, fg_source);
              if (!f->invalid) {
                fail(std::string("P6 FG Off role, ") + r.pins + ": F survived its retirement");
              }
              fence.completed = 5;
              switch (r.kind) {
                case shape::late_off:
                case shape::ngx_before_fg:
                  pick();
                  break;
                case shape::off_then_on:
                case shape::sr_called_before_fg_then_on:
                  fg_on();
                  break;
                case shape::other_viewport_active:
                  pick();
                  own_frame();
                  break;
                case shape::same_present:
                  pick({}, true);
                  pick();
                  break;
                case shape::fg_after_off_then_on:
                  // S, readable, takes the queue; then G, DLSS-G's copy of a
                  // frame before the Off, recorded after it (sequence 15);
                  // then FG On's first nomination, pending.
                  s = &sr(20, false);
                  pick();
                  g = &sr_copy(make(index++, provider_kind::streamline, epoch, 15, fg_source, 0, 0), false);
                  fg_on();
                  break;
                case shape::fg_after_off_sr_unshown_then_on:
                case shape::fg_after_off_alone_then_on:
                  // S (sequence 20), readable, no Present selects it; then G,
                  // an FG copy of an earlier frame recorded after the Off as a
                  // race past the capture owner's checks would leave it (the
                  // add-on records such a copy as the ordinary source); then
                  // FG On's first nomination, pending.
                  if (r.kind == shape::fg_after_off_sr_unshown_then_on) {
                    s = &sr(20, false);
                  }
                  g = &sr_copy(make(index++, provider_kind::streamline, epoch, 15, fg_source, 0, 0), false);
                  g->metadata.frame_generation_input = true;
                  fg_on();
                  break;
                case shape::fg_after_off_rekeyed:
                  // G, the late Off frame's FG copy made at its Present after
                  // the Off: recorded with the role retired, so its view and
                  // its metadata are the viewport's ordinary source; then the
                  // next frame's own SR copy.
                  g = &sr_copy(make(index++, provider_kind::streamline, epoch, 15, fg_source, 0, 0), false);
                  g->metadata.source_id = 0;
                  pick();
                  h = &sr(30, false);
                  pick();
                  break;
                case shape::ngx_after_fg_then_on:
                  // N takes the queue while S is still in flight; then FG
                  // On's first nomination is pending with S readable.
                  fence.completed = 4;
                  pick();
                  fence.completed = 5;
                  fg_on();
                  break;
                default:
                  own_frame();
              }
            }
        }
        if (trace != r.trace) {
          fail(std::string("P6 FG Off role, ") + r.pins + ":\n    got      " + trace + "\n    expected " + r.trace);
        }
      }
      std::printf("PASS P6 FG Off retires only the FG role: %zu rows\n", std::size(rows));
    }

    // P6: a pending head's own copy, on another queue, that completes between
    // the selection's look at it (select_latest_capture) and the stand-in's
    // (completed_snapshot), each reading its producer fence anew: it is
    // decided again as the current capture (10-28). Until then it read
    // completed_not_older and the Present held an older copy beside the
    // frame's own readable one (the 10-28 review's J2: the FG Off frame of
    // seed dda773a575be6561, once in a few fixed-set runs).
    void head_completes_between_looks() {
      constexpr std::uint64_t epoch = 13, device = 41;
      for (const bool between : {false, true}) {
        reset();
        fixture::completion_fence fence;
        fence.completed = 4;
        if (between) {
          fence.completed_later = 5;
          fence.later_after = 2;  // select_latest_capture's two looks.
        }
        auto &place = capture_queues.queues[0];
        place = std::make_unique<queue_state>();
        place->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&fence);
        place->fence.p = &fence;
        place->device_identity = device;
        struct release_place {
          std::unique_ptr<queue_state> &entry;
          ~release_place() { entry->queue.p = nullptr; entry->fence.p = nullptr; entry.reset(); }
        } release{place};
        queue_state owner;
        owner.device_identity = device;
        // The frame's own copy on the presenting queue (sequence 10), then
        // its head, an SR copy on the game's queue (sequence 20), copied out
        // (immutable) and complete only from the stand-in's look on; both of
        // one source.
        auto storage = std::make_shared<source_reference>();
        storage->device_identity = device;
        storage->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        storage->desc.Format = DXGI_FORMAT_R32_TYPELESS;
        auto &own = make(0, provider_kind::streamline, epoch, 10, 0);
        own.texture = std::make_shared<texture_reference>();
        own.source_nominated = true;
        own.metadata.source = storage;
        auto &head = make(1, provider_kind::streamline, epoch, 20, 0);
        head.texture = std::make_shared<texture_reference>();
        head.source_nominated = true;
        head.metadata.source = storage;
        head.queue = reinterpret_cast<std::uint64_t>(&fence);
        head.producer_fence = 5;
        head.copied_out = true;
        const auto now = GetTickCount64();
        pin_clock(now);
        const auto pick = select_provider(owner, native, 2, now, {});
        const auto got = std::string(1, label(pick, &own, 'O', &head, 'H')) + " " + name(pick.result) + " " + name(pick.selection);
        const char *expected = between ? "H ready current_capture" : "O ready completed_snapshot";
        if (got != expected) {
          fail(std::string("P6 head completing between looks, ") + (between ? "completes between them" : "still in flight") + ": got " + got + ", expected " + expected);
        }
      }
      std::printf("PASS P6 a pending head completing between the selection's looks: 2 rows\n");
    }

    // P6: an FG head whose own completed copy is refused (published before a
    // gap) or not readable: its lineage's newest legal SR copy stands in, as
    // where it has none (10-28; until then the Present held the display's
    // older copy beside that readable SR copy: the extra shard's seed
    // d0dd7845ac141ad2 at frame 113). Without an SR copy the head keeps its
    // own refusal.
    void lineage_after_refused_completed() {
      enum class variant { before_gap, before_gap_no_sr, not_readable };
      struct row {
        const char *pins;
        variant kind;
        const char *outcome;
      };
      const row rows[] {
        {"The FG copy completed before a gap, a newer SR copy of the lineage readable: SR stands in", variant::before_gap, "S ready lineage_snapshot"},
        {"The same without an SR copy: the head keeps its refusal", variant::before_gap_no_sr, "- recorded completed_before_gap"},
        {"The FG copy complete on another queue but not readable here, a newer SR copy readable: SR stands in", variant::not_readable, "S ready lineage_snapshot"},
      };
      constexpr std::uint64_t epoch = 17, device = 41;
      for (const auto &r : rows) {
        reset();
        fixture::completion_fence fence;
        fence.completed = 4;
        auto &place = capture_queues.queues[0];
        place = std::make_unique<queue_state>();
        place->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&fence);
        place->fence.p = &fence;
        place->device_identity = device;
        struct release_place {
          std::unique_ptr<queue_state> &entry;
          ~release_place() { entry->queue.p = nullptr; entry->fence.p = nullptr; entry.reset(); }
        } release{place};
        queue_state owner;
        owner.device_identity = device;
        auto storage = std::make_shared<source_reference>();
        storage->device_identity = device;
        storage->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        storage->desc.Format = DXGI_FORMAT_R32_TYPELESS;
        const auto copy = [&](slot &value) -> slot & {
          value.texture = std::make_shared<texture_reference>();
          value.source_nominated = true;
          value.metadata.source = storage;
          return value;
        };
        // The scope's completed FG copy (sequence 10).
        auto &fg = copy(make(0, provider_kind::streamline, epoch, 10, fg_source, 0, 0));
        if (r.kind == variant::not_readable) {
          // On DLSS-G's compute queue, its producer fence not passed yet.
          fg.queue = reinterpret_cast<std::uint64_t>(&fence);
          fg.producer_fence = 5;
          fg.copied_out = true;
        } else {
          // A gap since: the queue's admission floor above it.
          owner.admission_after = fg.id;
        }
        // The viewport's SR copy of a later frame (sequence 20), readable.
        slot *sr = nullptr;
        if (r.kind != variant::before_gap_no_sr) {
          sr = &copy(make(1, provider_kind::streamline, epoch, 20, 0, 0, UINT64_MAX));
        }
        // The scope's next FG copy, pending (sequence 30).
        auto &head = copy(make(2, provider_kind::streamline, epoch, 30, fg_source, 0, 0));
        head.finished = false;
        const auto now = GetTickCount64();
        pin_clock(now);
        const selection_policy policy {true, epoch, 0, false, now - 16};
        const auto pick = select_provider(owner, native, 2, now, policy);
        const auto got = std::string(1, label(pick, sr, 'S', &fg, 'F')) + " " + name(pick.result) + " " + name(pick.selection);
        if (got != r.outcome) {
          fail(std::string("P6 lineage after a refused completed FG copy, ") + r.pins + ": got " + got + ", expected " + r.outcome);
        }
      }
      std::printf("PASS P6 lineage stand-in after a refused completed FG copy: %zu rows\n", std::size(rows));
    }

    // P6/P7: a head refused before its capture slot existed names its source
    // to the display (capture_pick::named, classify_acquisition), also
    // beside ambiguous or inactive views. The display then holds only a copy
    // of that source or its FG/SR lineage, never another viewport's or NGX
    // feature generation's. Until 10-28 such a pick named no source (epoch
    // zero, which contradicts nothing), and the display held the previous
    // viewport's or feature's copy for up to the source age.
    void named_head_rows() {
      enum class variant {
        viewport_ambiguous,  // The new viewport's first attempt while the old view is active.
        viewport_alone,  // The same once the old view expired: a missing frame with nothing to stand in.
        ngx_feature_ambiguous,  // A recreated NGX feature (new source) whose first evaluate lacks depth.
        ngx_feature_alone,
        same_source,  // The shown source's own next attempt: held as any missing frame.
        fg_lineage  // FG On's first DLSS-G attempt over the viewport's shown SR copy: held as its lineage.
      };
      struct row {
        const char *pins;
        variant kind;
        const char *selection;  // The pick's result and reason.
        display_action action;
        const char *reason;
      };
      const row rows[] {
        {"Another viewport's first attempt, refused before its slot, beside the active one: the old copy is dropped", variant::viewport_ambiguous, "ambiguous ambiguous_views", display_action::invalidate, "source_changed"},
        {"Another viewport's first attempt, refused before its slot, alone: the old copy is dropped", variant::viewport_alone, "missing_state no_current_nomination", display_action::invalidate, "source_changed"},
        {"A recreated NGX feature's first evaluate without depth, beside the old feature's view: the old copy is dropped", variant::ngx_feature_ambiguous, "ambiguous ambiguous_views", display_action::invalidate, "source_changed"},
        {"A recreated NGX feature's first evaluate without depth, alone: the old copy is dropped", variant::ngx_feature_alone, "missing_state no_current_nomination", display_action::invalidate, "source_changed"},
        {"The shown source's next attempt refused before its slot: its copy is held", variant::same_source, "missing_state no_current_nomination", display_action::hold, "retained_copy"},
        {"FG On's first DLSS-G attempt refused before its slot over the viewport's SR copy: held as its lineage", variant::fg_lineage, "missing_state no_current_nomination", display_action::hold, "FG_copy_pending"},
      };
      constexpr std::uint64_t epoch = 13, ngx_source = 97;
      for (const auto &r : rows) {
        reset();
        const bool ngx = r.kind == variant::ngx_feature_ambiguous || r.kind == variant::ngx_feature_alone;
        const auto provider = ngx ? provider_kind::ngx : provider_kind::streamline;
        const auto source = ngx ? ngx_source : 0;
        queue_state owner;
        // The shown copy (sequence 10), established and delivered, which the
        // display retains; consumed, so nothing stands in for the head.
        auto &shown = make(0, provider, epoch, 10, source);
        establish_provider(owner, shown, 1);
        deliver(owner, shown);
        retained_depth retained;
        retained.metadata = shown.metadata;
        retained.capture_id = shown.id;
        retained.width = 1920;
        retained.height = 1080;
        retained.format = 39;
        retained.view_format = 41;
        retained.area = {0, 0, 1920, 1080};
        if (r.kind == variant::viewport_alone || r.kind == variant::ngx_feature_alone) {
          // The old view stopped: past the source age, and not the frame this
          // Present presents.
          for (auto &view : evaluations[provider_index(provider)].views) {
            if (view.sequence) {
              view.tick = GetTickCount64() - source_age - 1;
            }
          }
          fixture::presents_since(2);
        }
        // The head (sequence 11), refused before its capture slot existed.
        const std::uint32_t viewport = r.kind == variant::viewport_ambiguous || r.kind == variant::viewport_alone ? 1 : 0;
        const auto head_source = ngx ? ngx_source + 1 : r.kind == variant::fg_lineage ? fg_source : 0;
        begin_evaluation(epoch, 11, viewport, provider, head_source, r.kind == variant::fg_lineage ? 0 : UINT64_MAX);
        attempt_status[provider_index(provider)] = status::missing_state;
        const auto now = GetTickCount64();
        const auto pick = select_provider(owner, native, 2, now, {});
        const auto decision = classify_acquisition(pick, owner, {});
        const auto update = decide_display(decision, packet {}, retained, {}, now);
        const auto selection = std::string(name(pick.result)) + ' ' + name(pick.selection);
        if (pick.value || selection != r.selection || update.action != r.action || std::string(update.reason) != r.reason) {
          char text[512];
          std::snprintf(text, sizeof(text), "P6/P7 named head, %s: %s, display %s (decision epoch %llu viewport %u source %llu)", r.pins, selection.c_str(), update.reason, static_cast<unsigned long long>(decision.epoch), decision.viewport, static_cast<unsigned long long>(decision.source_id));
          fail(text);
        }
      }
      std::printf("PASS P6/P7 a head refused before its slot names its source: %zu rows\n", std::size(rows));
    }

    // F1 (the 10-27 review): which Presents count a stamped frame as the
    // frame they present (frame_presenting). Without frame generation only
    // its first Present (no other Present since its call); with FG on, a
    // frame that made a frame-generation call presents at its first Present
    // and the generated count after it (DLSS-G's real and generated Presents
    // in either order),
    // until the next frame's first evaluation began.
    void frame_presenting_rows() {
      struct row {
        const char *pins;
        unsigned generated;
        bool fg_call;
        unsigned presents;  // Presents since the frame's call, the decided one included.
        bool later_evaluation;
        bool expected;
      };

      const row rows[] {
        {"No Present since its call yet", 0, false, 0, false, true},
        {"Its first Present, without FG", 0, false, 1, false, true},
        {"The Present after its first, without FG", 0, false, 2, false, false},
        {"FG 2x: its first Present", 1, true, 1, false, true},
        {"FG 2x: its second Present (the real one after its generated one, or a generated one after its real one)", 1, true, 2, false, true},
        {"FG 2x: a third Present", 1, true, 3, false, false},
        {"FG 4x: its fourth Present", 3, true, 4, false, true},
        {"FG 4x: a fifth Present", 3, true, 5, false, false},
        {"FG 2x: its second Present after the next frame's first evaluation began", 1, true, 2, true, false},
        {"FG on, but the frame made no frame-generation call", 1, false, 2, false, false},
      };
      for (const auto &r : rows) {
        reset();
        const auto stamp = present_ordinal();
        scope.newest_evaluation_present = stamp;
        if (r.fg_call) {
          scope.frame_generation_present = stamp;
        }
        scope.presenting_generated = r.generated;
        if (r.presents > 1) {
          fixture::presents_since(r.presents - 1);
        }
        // The decided Present: counted before its selection.
        if (r.presents) {
          fixture::presents_since(1);
        }
        if (r.later_evaluation) {
          scope.newest_evaluation_present = present_ordinal();
        }
        const bool got = frame_presenting(stamp);
        if (got != r.expected || (!r.generated && got != (stamp && stamp >= scope.presents.load(std::memory_order_acquire)))) {
          fail(std::string("F1 frame_presenting, ") + r.pins + ": got " + (got ? "presenting" : "not presenting"));
        }
      }
      std::printf("PASS F1 frame_presenting: %zu rows\n", std::size(rows));
    }

    // F1, anchored (deferred item 9 B, 10-31; soak seed d40ba4d5b7c456ee at
    // frame 126): DLSS-G presents an earlier frame's generated Presents while
    // the game records this one (inside its SR evaluation: Assassin's Creed
    // Shadows), so up to the generated count of Presents come between a
    // frame's stamp and its real Present. The first Present after the frame's
    // submission on a presenting queue, or after DLSS-G's evaluation of a
    // frame whose tag copy was recorded, anchors its window: its generated
    // Presents count from there, its own evaluations up to the anchor end
    // nothing, and a later frame's evaluation after it does. Until 10-31 the
    // window counted from the stamp, so after a hitch before the real Present
    // the frame's own generated Presents were mono.
    void frame_presenting_anchor_rows() {
      enum class anchor {
        none,
        submission,  // Its copy's producer list on a presenting queue.
        dlssg,  // DLSS-G's evaluation standing aside for its recorded tag copy.
        earlier_copy,  // A submission of a copy recorded before its stamp.
        anchored_stamp,  // A submission, then an FG stamp at the anchor.
        later_stamp  // A submission, then an FG stamp after its real Present.
      };
      struct row {
        const char *pins;
        unsigned generated;
        unsigned earlier;  // The earlier frame's Presents after its stamp.
        anchor kind;
        bool own_evaluation;  // Its evaluation after them (its SR evaluation).
        unsigned after;  // Presents after its real one, the decided one included.
        bool later_evaluation;  // The next frame's evaluation before the decided Present.
        bool expected;
      };

      const row rows[] {
        {"MFG 4x, three earlier generated Presents after its stamp, no submission seen: its first generated Present", 3, 3, anchor::none, false, 1, false, false},
        {"The same with its copy's submission seen after them: its first generated Present", 3, 3, anchor::submission, false, 1, false, true},
        {"The same: its third generated Present", 3, 3, anchor::submission, false, 3, false, true},
        {"The same: a fourth Present after its real one", 3, 3, anchor::submission, false, 4, false, false},
        {"Anchored by DLSS-G's evaluation of its recorded tag copy: its third generated Present", 3, 3, anchor::dlssg, false, 3, false, true},
        {"MFG 3x, two earlier generated Presents, anchored: its second generated Present", 2, 2, anchor::submission, false, 2, false, true},
        {"MFG 2x, one earlier generated Present, anchored: its generated Present", 1, 1, anchor::submission, false, 1, false, true},
        {"MFG 2x anchored: a second Present after its real one", 1, 1, anchor::submission, false, 2, false, false},
        {"Its own evaluation after the earlier Presents, before its submission: its third generated Present", 3, 3, anchor::submission, true, 3, false, true},
        {"The same without the submission seen: that evaluation ends its window", 3, 3, anchor::none, true, 1, false, false},
        {"Anchored, the next frame's evaluation after its real Present ends its window", 3, 3, anchor::submission, false, 1, true, false},
        {"MFG 2x, two Presents after its stamp: a submission beyond its generated count anchors nothing", 1, 2, anchor::submission, false, 1, false, false},
        {"A submission of a copy recorded before its stamp anchors nothing", 3, 3, anchor::earlier_copy, false, 1, false, false},
        {"An FG stamp at its anchor (DLSS-G's inputs copied at its Present) is its own: its third generated Present", 3, 3, anchor::anchored_stamp, false, 3, false, true},
        {"An FG stamp after its real Present begins the next FG frame", 3, 3, anchor::later_stamp, false, 2, false, false},
      };
      for (const auto &r : rows) {
        reset();
        scope.presenting_generated = r.generated;
        const auto stamp = present_ordinal();
        scope.newest_evaluation_present = stamp;
        stamp_frame_generation(stamp);
        fixture::presents_since(r.earlier);
        if (r.own_evaluation) {
          scope.newest_evaluation_present = present_ordinal();
        }
        // The first Present after the frame's submission: its real one.
        const auto at = present_ordinal();
        if (r.kind == anchor::dlssg) {
          frame_generation_evaluated(9);
        } else if (r.kind == anchor::earlier_copy) {
          anchor_frame_generation(stamp - 1, at);
        } else if (r.kind != anchor::none) {
          anchor_frame_generation(stamp, at);
        }
        if (r.kind == anchor::anchored_stamp) {
          stamp_frame_generation(at);
        }
        fixture::presents_since(1);
        if (r.kind == anchor::later_stamp) {
          stamp_frame_generation(present_ordinal());
        }
        if (r.later_evaluation) {
          scope.newest_evaluation_present = present_ordinal();
        }
        fixture::presents_since(r.after);
        const bool got = frame_presenting(stamp);
        // The copy's own stamp, its submission's (presented_by), decides alike.
        const bool copy = r.kind == anchor::submission || r.kind == anchor::anchored_stamp ? frame_presenting(at) : got;
        if (got != r.expected || copy != got) {
          fail(std::string("F1 anchored frame_presenting, ") + r.pins + ": got " + (got ? "presenting" : "not presenting") +
               (copy != got ? " (its submission's stamp differs)" : ""));
        }
      }
      std::printf("PASS F1 anchored frame_presenting: %zu rows\n", std::size(rows));
    }

    // F2 and P2 (the 10-27 review): a hitch between a frame's evaluation and
    // a later Present of that frame. The FG input in flight (DLSS-G's copy)
    // and the frame's own SR snapshot, both past the source age by the
    // clock: the SR snapshot stands in at the frame's own Presents
    // (frame_presenting), never at a later frame's, and never an earlier
    // frame's snapshot. A copy whose producer list was submitted on a
    // presenting queue after an earlier Present (a game that records frame
    // N+1 before its submission thread presents frame N) is the frame of
    // that submission: current at its Present after a hitch, by its copy and
    // by its view (presented_by, view_presenting).
    void hitch_presenting_rows() {
      enum class variant {
        own_present,
        second_present,
        second_present_without_fg,
        earlier_frame_sr,
        third_present,
        submitted_later,
        submitted_later_unseen
      };
      struct row {
        const char *pins;
        variant kind;
        char picked;
        status result;
        selection_reason selection;
      };

      const row rows[] {
        {"The FG input in flight at its frame's own Present 400 ms after the call: the frame's SR snapshot stands in", variant::own_present, 'S', status::ready, selection_reason::lineage_snapshot},
        {"The same at the frame's second Present with FG 2x: the SR snapshot stands in", variant::second_present, 'S', status::ready, selection_reason::lineage_snapshot},
        {"The same with no frame-generation call: the Present after the first ages it by the clock", variant::second_present_without_fg, '-', status::stale, selection_reason::inactive_views},
        {"An earlier frame's SR snapshot past the source age does not stand in", variant::earlier_frame_sr, '-', status::recorded, selection_reason::no_completed_snapshot},
        {"A third Present with FG 2x ages it by the clock", variant::third_present, '-', status::stale, selection_reason::inactive_views},
        {"A copy submitted on the presenting queue after the Present its call was stamped for is current at its own Present", variant::submitted_later, 'S', status::ready, selection_reason::current_capture},
        {"Without that submission seen, the call's stamp ages it by the clock", variant::submitted_later_unseen, '-', status::stale, selection_reason::inactive_views},
      };
      constexpr std::uint64_t epoch = 9;
      for (const auto &r : rows) {
        reset();
        const auto now = GetTickCount64();
        const auto old = now - source_age - 150;
        queue_state owner;
        auto &views = evaluations[provider_index(provider_kind::streamline)].views;
        const slot *expected {};
        std::uint32_t generated = 1;
        if (r.kind == variant::submitted_later || r.kind == variant::submitted_later_unseen) {
          // Frame N+1's evaluation and copy, recorded before Present N.
          auto &sr = make(0, provider_kind::streamline, epoch, 2, 0);
          sr.present = present_ordinal();
          sr.metadata.tick = old;
          for (auto &view : views) {
            if (view.sequence) {
              view.tick = old;
            }
          }
          // Present N, then the submission thread executes N+1's list.
          fixture::presents_since(1);
          if (r.kind == variant::submitted_later) {
            sr.submitted_present = present_ordinal();
          }
          // N+1's own Present.
          fixture::presents_since(1);
          generated = 0;
          expected = &sr;
        } else {
          // The frame's SR evaluation (sequence 2), then DLSS-G's evaluation
          // of its FG input (sequence 3), which replaces the SR view.
          if (r.kind == variant::earlier_frame_sr) {
            fixture::presents_since(1);
          }
          auto &sr = make(0, provider_kind::streamline, epoch, 2, 0);
          sr.texture = std::make_shared<texture_reference>();
          sr.source_nominated = true;
          sr.present = present_ordinal();
          sr.metadata.tick = old;
          if (r.kind == variant::earlier_frame_sr) {
            // A Present came between: the FG evaluation is the next frame's.
            fixture::presents_since(1);
          }
          auto &fg = make(1, provider_kind::streamline, epoch, 3, fg_source, 0, 0);
          fg.finished = false;
          fg.present = present_ordinal();
          fg.metadata.tick = old;
          for (auto &view : views) {
            if (view.sequence) {
              view.tick = old;
            }
          }
          if (r.kind == variant::second_present_without_fg) {
            scope.frame_generation_present = 0;
          }
          const unsigned since = r.kind == variant::own_present ? 1 : r.kind == variant::third_present ? 3 : 2;
          fixture::presents_since(since);
          expected = &sr;
        }
        selection_policy policy {};
        policy.generated_presents = generated;
        pin_clock(now);
        const auto pick = select_provider(owner, native, 2, now, policy);
        const char picked = label(pick, expected, 'S', nullptr, '?');
        if (picked != r.picked || pick.result != r.result || pick.selection != r.selection) {
          char text[512];
          std::snprintf(text, sizeof(text), "F2/P2 hitch, %s: picked %c result %s selection %s", r.pins, picked, name(pick.result), name(pick.selection));
          fail(text);
        }
      }
      // producer_submitted stamps a copy's submission on a presenting queue
      // (one that acquired), and leaves one on another queue to its call.
      {
        reset();
        int stand_in {};
        auto &place = capture_queues.queues[0];
        place = std::make_unique<queue_state>();
        place->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&stand_in);
        place->chosen = true;
        const auto presenting = reinterpret_cast<std::uint64_t>(place->queue.p);
        auto &copy = make(0, provider_kind::streamline, epoch, 2, 0);
        copy.producer_submitted = false;
        copy.present = present_ordinal();
        fixture::presents_since(1);
        const auto at = present_ordinal();
        producer_submitted(copy, presenting + 8, 7, true);
        auto &second = make(1, provider_kind::streamline, epoch, 3, 0);
        second.producer_submitted = false;
        second.present = copy.present;
        producer_submitted(second, presenting, 7, true);
        const bool stamped = !copy.submitted_present && presented_by(copy) == copy.present && second.submitted_present == at &&
                             presented_by(second) == at;
        place->queue.p = nullptr;
        place.reset();
        if (!stamped) {
          fail("P2 producer_submitted: a presenting queue's submission was not stamped, or another queue's was");
        }
      }
      std::printf("PASS F2/P2 hitch between a frame's evaluation and its Presents: %zu rows\n", std::size(rows) + 1);
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

    // The acquisition names viewport 2's FG input, which follows the case's
    // retained SR copy: FG is on (a required scope) or the selection is a
    // silent scope's.
    void fg_input(display_case &d) {
      d.current.source_id = d.candidate.metadata.source_id = (1ull << 63) | 2;
      d.candidate.metadata.frame_generation_input = true;
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
        // The viewport's ordinary copy of the scope's epoch is its FG
        // input's predecessor (one display lineage): a required scope holds
        // it until its first readable FG copy (FG On before that copy is
        // readable; the round 3 review's soak, until 10-27 FG_scope_changed),
        // and refuses any other.
        {"Required FG holds its viewport's retained SR copy for its pending FG input", [](display_case &d) {
           d.policy = {true, 8, 2};
           fg_input(d);
         },
         action::hold,
         "FG_copy_pending"},
        {"Required FG copies its readable FG input over its viewport's SR copy", [](display_case &d) {
           d.policy = {true, 8, 2};
           fg_input(d);
           d.current.source_selected = d.candidate.pixel_ready = true;
         },
         action::copy_fresh,
         "fresh_copy_candidate"},
        {"Required FG holds its viewport's retained SR copy without a nomination", [](display_case &d) {
           d.policy = {true, 8, 2};
           d.current = {};
         },
         action::hold,
         "retained_copy"},
        {"Required FG refuses another viewport's retained SR copy", [](display_case &d) {
           d.policy = {true, 8, 3};
           fg_input(d);
           d.current.viewport = 3;
           d.current.source_id = (1ull << 63) | 3;
         },
         action::invalidate,
         "FG_scope_changed"},
        {"Required FG refuses another epoch's retained SR copy", [](display_case &d) {
           d.policy = {true, 9, 2};
           fg_input(d);
           d.current.epoch = 9;
         },
         action::invalidate,
         "FG_scope_changed"},
        {"Required FG refuses a retained NGX copy", [](display_case &d) {
           d.policy = {true, 8, 2};
           fg_input(d);
           d.previous.metadata.provider = provider_kind::ngx;
         },
         action::invalidate,
         "FG_scope_changed"},
        {"A silent scope's pending FG input holds its viewport's SR copy", fg_input, action::hold, "FG_copy_pending"},
        {"A silent scope's pending FG input of another viewport drops the SR copy", [](display_case &d) {
           fg_input(d);
           d.current.viewport = 3;
           d.current.source_id = (1ull << 63) | 3;
         },
         action::invalidate,
         "source_changed"},
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
        {"A failed copy of required FG's input holds its viewport's retained SR copy (one lineage)", [](display_case &d) {
           d.policy = {true, 8, 2};
           fg_input(d);
         },
         action::hold,
         "retained_after_failed_copy"},
        {"A failed copy under required FG with another viewport's retained SR copy", [](display_case &d) {
           d.policy = {true, 8, 3};
           fg_input(d);
           d.current.viewport = 3;
           d.current.source_id = (1ull << 63) | 3;
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
  deferral_age_rows();
  head_completes_between_looks();
  lineage_after_refused_completed();
  select_provider_rows();
  ownership_timelines();
  freshness_edges();
  stray_frame_generation_copy_steps();
  lineage_stand_in_rows();
  missing_frame_rows();
  fg_off_role_rows();
  named_head_rows();
  frame_presenting_rows();
  frame_presenting_anchor_rows();
  hitch_presenting_rows();
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
