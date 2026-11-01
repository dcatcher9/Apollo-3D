// SPDX-License-Identifier: GPL-3.0-only
// The continuity judge (test_game3d_continuity.h) on synthetic frame
// records: the O1 rules a session timeline cannot show failing without an
// add-on regression. A late signal of a game's order fence makes only the
// copies read through that fence unholdable (each copy names its own fence),
// and a hold a generated Present drops after such a signal is
// R1-late-signaller's only where the real Present showed such a copy. A copy
// from another queue that completed after the add-on's decision neither
// excuses nor demands a mono Present there. A real Present's copy captured
// inside the frame it presents is no older than that frame for the stale
// rule, however long the frame took. A frame's own copy recorded at its
// call is readable at its own Present: the freshness rule counts it there,
// and a stall inside such a frame is no warm-up. Where each copy's capture id
// is known, a Present that shows a copy older than a readable copy of the
// same kind of a later frame is stale_shown with no bound (a single
// Present's lag; since 11-01 a copy of the shown copy's own frame is no
// newer depth), a real
// Present never shows a copy recorded before the one the last real Present
// showed (a handover to the other API, and since 10-31 the same API's other
// source too), and a copy deferred to its frame's Present counts by the
// capture id the fixture learns after that Present. A copy whose id the
// fixture never learned counts by its frame: one of a frame before the
// shown copy's is no holdable copy (11-01).
#include "test_game3d_continuity.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
  namespace continuity = game3d_continuity;
  using sunshine_game3d::frame_record;
  using sunshine_game3d::source_slot;
  using sunshine_game3d::source_state;

  unsigned failures {};

  void require(bool value, const char *message) {
    if (!value) {
      std::fprintf(stderr, "FAIL: %s\n", message);
      ++failures;
    }
  }

  // A Present's record: the rung that showed depth (with the copy's tick),
  // or mono under API ownership.
  frame_record shown(std::uint64_t present, std::uint64_t now, source_slot owner, std::uint64_t copy_tick) {
    frame_record value;
    value.present = present;
    value.tick_ms = now;
    value.depth_owner = owner;
    value.depth_tick_ms = copy_tick;
    value[owner].state = source_state::ready;
    value[owner].reason = "test";
    return value;
  }

  frame_record mono(std::uint64_t present, std::uint64_t now) {
    frame_record value;
    value.present = present;
    value.tick_ms = now;
    value.depth_owner = source_slot::none;
    value[source_slot::hold].state = source_state::mono;
    value[source_slot::hold].reason = "test";
    return value;
  }

  const bool complete = true;
  bool done() {
    return complete;
  }

  int r1_row() {
    for (std::size_t i = 0; i != continuity::gap_count; ++i) {
      if (continuity::known_gaps[i].limit == continuity::limitation::late_signaller) {
        return int(i);
      }
    }
    return -1;
  }

  // A generated Present after a real Present that showed an FG tag copy
  // made on the presenting queue, while SR was read in flight through a
  // fence a late signal reached after that Present: the add-on keeps
  // holding the tag copy (only the SR read is conflicted), so a dropped
  // hold there is no row's. Where the real Present showed the copy read
  // through that fence, the drop is R1-late-signaller's.
  void generated_drop_after_late_signal() {
    const int fence_a = 0, fence_b = 0;
    for (const bool shows_relied_copy : {false, true}) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      // Three real Presents of FG tag copies (the judge needs three to know
      // a generated Present's hold).
      for (int i = 0; i != 3; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::tag_copy, now);
        judge.present(shown(present++, now, source_slot::fg_depth, now), frame, "test");
      }
      continuity::supply frame;
      frame.late_after = true;
      if (shows_relied_copy) {
        // DLSS-G's compute copy, read in flight through fence_b, shown.
        frame.add_elsewhere(continuity::dlssg_input, now);
        frame.completions.clear();
        frame.completes(continuity::dlssg_input, done, &fence_b);
      } else {
        // The FG tag copy on the presenting queue, shown; SR from another
        // queue read in flight through fence_a, kept out by the FG scope.
        frame.add(continuity::tag_copy, now);
        frame.add_elsewhere(continuity::sr_evaluate, now);
        frame.owned = continuity::sr_evaluate;
        frame.completions.clear();
        frame.completes(continuity::sr_evaluate, done, &fence_a);
      }
      judge.present(shown(present++, now, source_slot::fg_depth, now), frame, "test");
      judge.take();
      judge.late_signal(shows_relied_copy ? &fence_b : &fence_a);
      now += 8;
      judge.present(mono(present++, now), {}, "test", true);
      const auto violations = judge.take();
      require(violations.size() == 1 && violations[0].kind == continuity::violation_kind::dropped_hold,
        "a generated Present that dropped a fresh hold was not judged");
      if (violations.size() == 1) {
        if (shows_relied_copy) {
          require(continuity::known(violations[0]) == r1_row(),
            "a generated Present's dropped hold of a copy read through the late-signalled fence was not R1's");
        } else {
          require(continuity::known(violations[0]) < 0,
            "a generated Present's dropped hold of a presenting-queue copy was excused as R1-late-signaller's "
            "because a late signal reached another copy's fence");
        }
      }
    }
  }

  // Seed c28965dca8d7c76c's frame 38 (10-27 full check): the presenting
  // queue's last copy is held for the source age through frames that supply
  // nothing, then FG On's first frame supplies only DLSS-G's compute copy,
  // which its own Present cannot read (unordered: the order fence is new at
  // the toggle). The add-on goes mono there. Whether that copy completed
  // between the add-on's decision and the judge's record must not decide
  // the verdict; one complete before the Present call must (the add-on could
  // read it), and so must a live hold.
  void unordered_copy_completed_after_decision() {
    struct trial {
      std::uint64_t held_ms;
      bool complete_before_present, complete_at_record, gap;
      const char *message;
    };
    const trial trials[] {
      {250, false, true, false, "an unordered frame's copy that completed after the add-on's decision made its expired hold's mono Present a gap"},
      {250, false, false, false, "an unordered frame's mono Present after its hold expired was judged a gap"},
      {250, true, true, true, "an unordered frame's copy complete before its Present call did not make a mono Present a gap"},
      {150, false, false, true, "a mono Present that dropped a live hold was not judged a gap"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
      }
      const auto copied = now - 16;
      for (std::uint64_t at = copied + 16; at < copied + value.held_ms; at += 16) {
        auto held = shown(present++, at, source_slot::scene_depth, copied);
        held[source_slot::scene_depth].state = source_state::held;
        judge.present(held, {}, "test");
      }
      now = copied + value.held_ms;
      bool completed = value.complete_before_present;
      continuity::supply frame;
      frame.add_elsewhere(continuity::dlssg_input, now, [&completed] {
        return completed;
      });
      frame.unordered = true;
      judge.before_present(frame);
      completed = value.complete_at_record;
      judge.take();
      judge.present(mono(present++, now), frame, "test");
      const auto violations = judge.take();
      const bool gap = violations.size() == 1 && violations[0].kind == continuity::violation_kind::gap;
      require(value.gap ? gap : violations.empty(), value.message);
    }
  }

  // The stale rule ages the newest source at the Present, except a real
  // Present's copy captured inside the frame it presents (the 10-27 full
  // check's C8-fg-off-color-only, frame 39: a 683 ms hitch inside the frame,
  // before its SR evaluation, then that frame's own copy shown, judged
  // stale at age 672). A held copy shown after such a hitch, inside a frame
  // with or without an API source, or after a hitch between frames, stays
  // stale.
  void stale_rule_ages_a_frames_own_copy_from_its_capture() {
    struct trial {
      // The hitch: inside the frame (after its stamp) or before it.
      bool inside, api;
      // The shown copy: the frame's own (captured after the hitch), else
      // the last copy before the hitch.
      bool own;
      bool stale;
      const char *message;
    };
    const trial trials[] {
      {true, true, true, false, "a Present that showed its frame's own copy after a hitch inside that frame was stale"},
      {true, true, false, true, "a Present that held an older copy after a hitch inside its API frame was not stale"},
      {true, false, false, true, "a Present that held a copy after a hitch inside a frame with no API source was not stale"},
      {false, false, false, true, "a Present that held a copy after a hitch between frames was not stale"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
      }
      const auto copied = now - 16;
      const auto stamp = value.inside ? now : now + 400;
      const auto at = now + 416;
      continuity::supply frame;
      if (value.api) {
        frame.add(continuity::sr_evaluate, stamp);
      } else {
        frame.allow(continuity::generic_boundary, stamp);
      }
      auto record = shown(present++, at, source_slot::scene_depth, value.own ? at - 2 : copied);
      if (!value.own) {
        record[source_slot::scene_depth].state = source_state::held;
      }
      judge.take();
      judge.present(record, frame, "test");
      bool stale = false;
      for (const auto &violation : judge.take()) {
        stale = stale || violation.kind == continuity::violation_kind::stale;
      }
      require(stale == value.stale, value.message);
    }
  }

  // A frame's own copy recorded at its call on the presenting queue
  // (supply::immediate: an SR evaluation's input) is readable at its own
  // Present. A Present that holds a copy older than the freshness bound
  // beside it shows stale depth (the session tests' 10-27 review: at FG On
  // with DLSS-G's first copy in flight the add-on held an SR copy 187 ms old
  // while the frame's own SR copy was fresh, and the judge, which counted a
  // frame's copies only from the Present after it, passed it). A copy
  // deferred to the frame's Present (R2) still counts only from the next.
  void immediate_copy_counts_at_its_own_present() {
    struct trial {
      bool immediate, shows_own;
      bool stale;
      const char *message;
    };
    const trial trials[] {
      {true, false, true, "a Present that held a 160 ms old copy beside its frame's own fresh SR copy was not stale_shown"},
      {true, true, false, "a Present that showed its frame's own SR copy was judged"},
      {false, false, false, "a Present that held a copy beside its frame's copy deferred to that Present was stale_shown"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
      }
      const auto copied = now - 16;
      for (; now < copied + 160; now += 16) {
        auto held = shown(present++, now, source_slot::scene_depth, copied);
        held[source_slot::scene_depth].state = source_state::held;
        judge.present(held, {}, "test");
      }
      continuity::supply frame;
      frame.add(value.immediate ? continuity::sr_evaluate : continuity::tag_copy, now);
      frame.immediate = value.immediate ? continuity::sr_evaluate : 0;
      auto record = shown(present++, now, source_slot::scene_depth, value.shows_own ? now : copied);
      if (!value.shows_own) {
        record[source_slot::scene_depth].state = source_state::held;
      }
      judge.take();
      judge.present(record, frame, "test");
      const auto violations = judge.take();
      const bool stale = violations.size() == 1 && violations[0].kind == continuity::violation_kind::stale_shown;
      require(value.stale ? stale : violations.empty(), value.message);
    }
  }

  // A stall leaves a frame's own source readable at its own Present when
  // that Present reads it (supply::immediate), wherever the stall came:
  // inside the frame after its first call, or before that call (until the
  // 10-27 review a stall before it still disarmed the judge and excused up to
  // establish_ms of mono). A frame whose source is pipelined stays a
  // warm-up, and so does one whose only own copy a live FG scope keeps out
  // (owned: the scope refuses that copy, so its Present cannot read it) and
  // a silence of the association window.
  void stall_inside_a_frame_that_reads_its_own_source() {
    struct trial {
      std::uint64_t stamp_after_ms;
      bool immediate, owned, gap;
      std::uint64_t present_after_ms;
      const char *message;
    };
    const trial trials[] {
      {16, true, false, true, 416, "a mono Present after a stall inside a frame whose own SR copy it reads was excused as a warm-up"},
      {400, true, false, true, 416, "a mono Present after a stall before its frame's first call, whose own SR copy it reads, was excused as a warm-up"},
      {16, true, true, false, 416, "a mono Present after a stall inside a frame whose only SR copy a live FG scope refused was judged a gap"},
      {16, false, false, false, 416, "a mono Present after a stall inside a frame with a pipelined source was judged a gap"},
      {1100, true, false, false, 1116, "a mono Present after a silence of the association window was judged a gap"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
      }
      const auto last = now - 16;
      continuity::supply frame;
      frame.add(value.immediate ? continuity::sr_evaluate : continuity::tag_copy, last + value.stamp_after_ms);
      frame.immediate = value.immediate ? continuity::sr_evaluate : 0;
      frame.owned = value.owned ? continuity::sr_evaluate : 0;
      judge.take();
      judge.present(mono(present++, last + value.present_after_ms), frame, "test");
      const auto violations = judge.take();
      const bool gap = violations.size() == 1 && violations[0].kind == continuity::violation_kind::gap;
      require(value.gap ? gap : violations.empty(), value.message);
    }
  }

  // J3 (10-27 review): after a hitch between a frame's evaluation and its
  // Present, the next frame's own copy is deferred to the Present after it
  // (R2): its Present can only hold the copy before the hitch, past the
  // source age. Holding it is stale whatever newer supply the frame made,
  // and going mono there is no gap (nothing holdable). A hold younger than
  // the source age is no violation.
  void hold_past_the_source_age_after_a_hitch() {
    struct trial {
      bool shows_held;
      std::uint64_t hitch_ms;
      bool stale, violation;
      const char *message;
    };
    const trial trials[] {
      {true, 400, true, true, "a Present that held a copy 418 ms old beside its frame's deferred copy was not stale"},
      {false, 400, false, false, "a mono Present whose frame's copy only a later Present reads, with nothing holdable, was a violation"},
      {true, 100, false, false, "a Present that held a copy 118 ms old was a violation"},
      {false, 100, false, true, "a mono Present that dropped a copy 118 ms old was no violation"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
      }
      // The frame with the hitch after its evaluation: its own copy shown.
      continuity::supply hitched;
      hitched.add(continuity::sr_evaluate, now);
      hitched.immediate = continuity::sr_evaluate;
      const auto copied = now;
      now += value.hitch_ms;
      judge.present(shown(present++, now, source_slot::scene_depth, copied), hitched, "test");
      judge.take();
      now += 16;
      continuity::supply deferred;
      deferred.add(continuity::tag_copy, now - 8);
      auto record = value.shows_held ? shown(present++, now, source_slot::scene_depth, copied) : mono(present++, now);
      if (value.shows_held) {
        record[source_slot::scene_depth].state = source_state::held;
      }
      judge.present(record, deferred, "test");
      bool stale = false;
      const auto violations = judge.take();
      for (const auto &violation : violations) {
        stale = stale || violation.kind == continuity::violation_kind::stale;
      }
      require(stale == value.stale && violations.empty() != value.violation, value.message);
    }
  }

  // J4 (10-27 review): a copy's age runs from when it was made, not from
  // its frame's first call. A frame whose first call came before a 400 ms
  // hitch and whose SR copy came after it: a generated Present after its
  // real Present that drops that 20 ms old copy is a dropped hold, and an
  // unordered frame after it that goes mono beside it is a gap.
  void copies_age_from_when_they_were_made() {
    for (const bool generated : {true, false}) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
      }
      continuity::supply hitched;
      hitched.allow(continuity::generic_boundary, now);
      now += 400;
      hitched.add(continuity::sr_evaluate, now);
      hitched.immediate = continuity::sr_evaluate;
      judge.present(shown(present++, now + 4, source_slot::scene_depth, now), hitched, "test");
      judge.take();
      now += 20;
      if (generated) {
        judge.present(mono(present++, now), {}, "test", true);
      } else {
        continuity::supply unordered;
        unordered.add_elsewhere(continuity::sr_evaluate, now);
        unordered.unordered = true;
        judge.present(mono(present++, now), unordered, "test");
      }
      const auto violations = judge.take();
      const auto kind = generated ? continuity::violation_kind::dropped_hold : continuity::violation_kind::gap;
      require(violations.size() == 1 && violations[0].kind == kind,
        generated ? "a generated Present that dropped a 20 ms old copy made after its frame's hitch was no dropped hold"
                  : "an unordered frame's mono Present beside a 20 ms old copy made after a hitch was no gap");
    }
  }

  // J5 (10-27 review): under strict_holds a frame with no legal source holds
  // the last depth within the source age; its mono Present while a copy is
  // holdable is a dropped hold, unless its events permit the drop. Without
  // strict_holds (the native provider fixture's scripts) it may go mono.
  void no_source_frame_holds_under_strict_holds() {
    struct trial {
      bool strict, permitted, violation;
      const char *message;
    };
    const trial trials[] {
      {true, false, true, "a frame with no source dropped a 16 ms old hold under strict_holds without a violation"},
      {true, true, false, "a frame whose events permit the drop was a violation"},
      {false, false, false, "a frame with no source was a violation without strict_holds"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      judge.strict_holds = value.strict;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
      }
      judge.take();
      continuity::supply none;
      none.drop_permitted = value.permitted;
      judge.present(mono(present++, now), none, "test");
      const auto violations = judge.take();
      require(value.violation ? violations.size() == 1 && violations[0].kind == continuity::violation_kind::dropped_hold : violations.empty(),
        value.message);
    }
  }

  // F4 (10-27 review): a generated Present after its FG frame's real Present
  // (between) presents that frame. After a pause between the two (a hitch
  // after the real Present) it is no stall: showing the copy the real
  // Present showed is age 0, and dropping it is a dropped hold whatever the
  // clock. A generated Present of a frame without an FG copy keeps the
  // source age of the clock.
  void generated_present_of_its_fg_frame_after_a_pause() {
    struct trial {
      bool fg_frame, shows, violation;
      continuity::violation_kind kind;
      const char *message;
    };
    const trial trials[] {
      {true, true, false, continuity::violation_kind::stale, "a generated Present that showed its FG frame's own copy 400 ms after its real Present was a violation"},
      {true, false, true, continuity::violation_kind::dropped_hold, "a generated Present that dropped its FG frame's own copy after a pause was no dropped hold"},
      {false, false, false, continuity::violation_kind::dropped_hold, "a generated Present of a frame without an FG copy that dropped a copy past the source age was a violation"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (int i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::tag_copy, now);
        frame.immediate = continuity::tag_copy;
        frame.fg_frame = value.fg_frame;
        judge.present(shown(present++, now, source_slot::fg_depth, now), frame, "test");
      }
      const auto copied = now - 16;
      judge.take();
      now = copied + 4 + 400;
      auto record = value.shows ? shown(present++, now, source_slot::fg_depth, copied) : mono(present++, now);
      if (value.shows) {
        record[source_slot::fg_depth].state = source_state::held;
      }
      judge.present(record, {}, "test", true, true);
      const auto violations = judge.take();
      require(value.violation ? violations.size() == 1 && violations[0].kind == value.kind : violations.empty(), value.message);
      require(!judge.stalls(), "a generated Present after its frame's real Present was counted a stall");
    }
  }

  // Two copies of one frame from another queue, one read through fence_a,
  // the other readable with no fence (complete): a late signal of fence_a
  // makes only the first unholdable, so a following frame whose own copy
  // its Present cannot read (unordered) may not go mono while the second is
  // younger than the source age.
  void late_signal_conflicts_only_its_copies() {
    const int fence_a = 0;
    continuity::judge judge;
    std::uint64_t now = 1000, present = 1;
    // Its Present shows depth (the judge is armed); no other copy is holdable.
    continuity::supply frame;
    frame.add_elsewhere(continuity::sr_evaluate | continuity::dlssg_input, now);
    frame.completions.clear();
    frame.completes(continuity::sr_evaluate, done, &fence_a);
    frame.completes(continuity::dlssg_input, done);
    judge.present(shown(present++, now, source_slot::scene_depth, now - 16), frame, "test");
    judge.late_signal(&fence_a);
    judge.take();
    now += 16;
    continuity::supply unordered;
    unordered.add_elsewhere(continuity::sr_evaluate, now);
    unordered.unordered = true;
    judge.present(mono(present++, now), unordered, "test");
    const auto violations = judge.take();
    require(violations.size() == 1 && violations[0].kind == continuity::violation_kind::gap &&
        continuity::known(violations[0]) < 0,
      "a late signal of one copy's fence made the frame's other copy unholdable too, excusing a mono Present");
  }

  // A copy published before the one the last real Present showed is never
  // holdable (10-28): showing it would be older depth after newer. A frame's
  // copy from another queue, complete only after its own Present (capture
  // 11, never shown), then the next frame's copy read through fence_a and
  // shown (capture 13), whose read a late signal of fence_a conflicts: the
  // following frame, whose own copy its Present cannot read, may go mono
  // (the extra shard's seeds 722b4878c6c27f14 at frame 90 and
  // 7bd2fe012beab724 at frame 179 failed as a gap). Where the earlier copy
  // is the newer one (capture 15), the mono Present is a gap.
  void holdable_never_older_than_shown() {
    for (const std::uint64_t early : {11ull, 15ull}) {
      const int fence_a = 0;
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      bool complete = false;
      continuity::supply first;
      first.add_elsewhere(continuity::sr_evaluate, now);
      first.completions.clear();
      first.completes(continuity::sr_evaluate, [&complete] { return complete; }, nullptr, early);
      first.unordered = true;
      judge.present(mono(present++, now), first, "test");
      now += 16;
      complete = true;
      continuity::supply second;
      second.add_elsewhere(continuity::sr_evaluate, now);
      second.completions.clear();
      second.completes(continuity::sr_evaluate, done, &fence_a, 13);
      second.immediate = continuity::sr_evaluate;
      auto record = shown(present++, now, source_slot::scene_depth, now);
      record.depth_capture = 13;
      judge.present(record, second, "test");
      judge.late_signal(&fence_a);
      judge.take();
      now += 16;
      continuity::supply third;
      third.add_elsewhere(continuity::sr_evaluate, now);
      third.unordered = true;
      judge.present(mono(present++, now), third, "test");
      const auto violations = judge.take();
      const bool gap = violations.size() == 1 && violations[0].kind == continuity::violation_kind::gap;
      require(early < 13 ? violations.empty() : gap,
        early < 13 ? "a copy published before the shown one made a mono Present after the shown copy's conflict a gap"
                   : "a newer complete copy beside the shown copy's conflict did not make the mono Present a gap");
    }
  }

  // A copy whose capture id the fixture never learned (supply::capture 0:
  // the add-on published none it saw at the call) follows the frames'
  // order as ids do: one of a frame before the shown copy's was published
  // before it, so after the shown copy's conflict it is no holdable copy
  // and the mono Present is O1's. One of the shown copy's own frame still
  // is. Until 11-01 any copy without an id counted (soak seeds
  // 2bda274f7f2ca5de at frame 63 and afcf3faa4ef87af5 at frame 12, in
  // loaded runs, failed as a gap).
  void holdable_unknown_id_by_frame() {
    for (const bool own_frame : {false, true}) {
      const int fence_a = 0;
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      continuity::supply first;
      first.add_elsewhere(continuity::sr_evaluate, now);
      first.completions.clear();
      first.completes(continuity::sr_evaluate, done, nullptr, own_frame ? 11 : 0);
      first.unordered = true;
      judge.present(mono(present++, now), first, "test");
      now += 16;
      continuity::supply second;
      second.add_elsewhere(continuity::sr_evaluate, now);
      second.completions.clear();
      second.completes(continuity::sr_evaluate, done, &fence_a, 13);
      if (own_frame) {
        second.add_elsewhere(continuity::dlssg_input, now);
        second.completes(continuity::dlssg_input, done, nullptr, 0);
      }
      second.immediate = continuity::sr_evaluate;
      auto record = shown(present++, now, source_slot::scene_depth, now);
      record.depth_capture = 13;
      judge.present(record, second, "test");
      judge.late_signal(&fence_a);
      judge.take();
      now += 16;
      continuity::supply third;
      third.add_elsewhere(continuity::sr_evaluate, now);
      third.unordered = true;
      judge.present(mono(present++, now), third, "test");
      const auto violations = judge.take();
      const bool gap = violations.size() == 1 && violations[0].kind == continuity::violation_kind::gap;
      require(own_frame ? gap : violations.empty(),
        own_frame ? "a copy without a capture id of the shown copy's own frame did not make the mono Present after the conflict a gap"
                  : "a copy without a capture id of a frame before the shown copy's made the mono Present after the conflict a gap");
    }
  }

  // The freshness rule by capture id (10-28): a Present that holds the
  // previous frame's copy beside its own frame's readable copy, one frame
  // (16 ms) late, passed the tick bound (two frame intervals and the slack);
  // with both capture ids known it is stale_shown. A newer copy of the other
  // kind (an FG scope's input beside ordinary depth: the scope keeps its
  // viewport while it can supply) or without a known id is no evidence.
  void single_present_lag_by_capture() {
    struct trial {
      std::uint64_t newer;  // The frame's own copy (capture id, kind flag); 0 unknown.
      bool stale;
      const char *message;
    };
    const trial trials[] {
      {20, true, "a Present that held the previous frame's copy beside its frame's readable newer copy was not stale_shown"},
      {0, false, "a one-frame lag with no capture id known was stale_shown inside the tick bound"},
      {20 | continuity::copy_fg_input, false, "a newer copy of the other kind made the shown copy stale"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (std::uint64_t i = 0; i != 4; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        frame.capture[0] = 10 + i;
        auto record = shown(present++, now, source_slot::scene_depth, now);
        record.depth_capture = 10 + i;
        judge.present(record, frame, "test");
      }
      continuity::supply frame;
      frame.add(continuity::sr_evaluate, now);
      frame.immediate = continuity::sr_evaluate;
      frame.capture[0] = value.newer;
      auto record = shown(present++, now, source_slot::scene_depth, now - 16);
      record.depth_capture = 13;
      record[source_slot::scene_depth].state = source_state::held;
      judge.take();
      judge.present(record, frame, "test");
      const auto violations = judge.take();
      const bool stale = violations.size() == 1 && violations[0].kind == continuity::violation_kind::stale_shown &&
                         violations[0].shown_capture == 13 && violations[0].newer_capture == 20;
      require(value.stale ? stale : violations.empty(), value.message);
    }
  }

  // An FG copy is held as its display lineage's: a newer readable copy of
  // either kind makes the hold stale, FG on or off (since the 10-28
  // review's J1), and after FG Off (supply::fg_off) also an SR copy the FG
  // hold kept off-screen while FG was on (supply::fg_held), which counts
  // only after FG Off. While FG is on the FG hold over the SR copies it
  // keeps off-screen is the owner's rule and is not judged.
  void fg_hold_after_fg_off_judged_by_lineage() {
    struct trial {
      bool fg_off, held;  // The judged frame comes after FG Off; the SR copy was the FG hold's.
      bool before_stale;  // The silent frame's Present (FG on) is stale already.
      bool stale;
      const char *message;
    };
    const trial trials[] {
      {true, true, false, true, "an FG copy held after FG Off beside a readable newer SR copy the FG hold had kept off-screen was not stale_shown"},
      {true, false, true, true, "an FG copy held after FG Off beside a readable newer SR copy was not stale_shown"},
      {false, true, false, false, "an FG copy a silent scope holds over the SR copies it keeps off-screen was stale_shown while FG was on"},
      {false, false, true, true, "an FG copy held while FG is on beside a readable newer SR copy no FG hold keeps off-screen was not stale_shown"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      // An FG tag copy (capture 10), shown.
      continuity::supply fg;
      fg.add(continuity::tag_copy, now);
      fg.immediate = continuity::tag_copy;
      fg.capture[1] = 10 | continuity::copy_fg_input;
      auto shown_fg = shown(present++, now, source_slot::fg_depth, now);
      shown_fg.depth_capture = 10;
      judge.present(shown_fg, fg, "test");
      now += 16;
      // A silent frame's SR copy (capture 12) on the presenting queue, kept
      // off-screen by the FG hold (or an ordinary one).
      continuity::supply silent;
      silent.add(continuity::sr_evaluate, now);
      silent.immediate = continuity::sr_evaluate;
      silent.capture[0] = 12;
      if (value.held) {
        silent.owned = silent.fg_held = continuity::sr_evaluate;
      }
      auto held_fg = shown(present++, now, source_slot::fg_depth, now - 16);
      held_fg.depth_capture = 10;
      held_fg[source_slot::fg_depth].state = source_state::held;
      judge.present(held_fg, silent, "test");
      const auto before = judge.take();
      now += 16;
      // The next frame's Present holds the FG copy again; its own SR copy
      // is in flight on another queue.
      continuity::supply next;
      next.add_elsewhere(continuity::sr_evaluate, now, [] { return false; });
      next.unordered = true;
      next.fg_off = value.fg_off;
      auto again = shown(present++, now, source_slot::fg_depth, now - 32);
      again.depth_capture = 10;
      again[source_slot::fg_depth].state = source_state::held;
      judge.present(again, next, "test");
      const auto violations = judge.take();
      const auto stale_at = [](const std::vector<continuity::violation> &found) {
        return found.size() == 1 && found[0].kind == continuity::violation_kind::stale_shown &&
               found[0].shown_capture == 10 && found[0].newer_capture == 12;
      };
      require((value.before_stale ? stale_at(before) : before.empty()) &&
              (value.stale ? stale_at(violations) : violations.empty()), value.message);
    }
  }

  // A copy deferred to its frame's Present (R2) has no capture id at its
  // call: the fixture learns it once that Present published it
  // (judge::learned_capture), and a later Present that holds an older copy
  // of its kind beside it is stale_shown by capture id. Until 10-31 such a
  // copy was judged by the tick bound only, which a one-frame lag passes.
  void deferred_copy_counts_by_its_learned_capture() {
    for (const bool learned : {false, true}) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (std::uint64_t i = 0; i != 3; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        frame.capture[0] = 10 + i;
        auto record = shown(present++, now, source_slot::scene_depth, now);
        record.depth_capture = 10 + i;
        judge.present(record, frame, "test");
      }
      // A tag copy deferred to its frame's Present, which holds copy 12.
      continuity::supply deferred;
      deferred.add(continuity::tag_copy, now);
      auto held = shown(present++, now, source_slot::scene_depth, now - 16);
      held.depth_capture = 12;
      held[source_slot::scene_depth].state = source_state::held;
      judge.present(held, deferred, "test");
      if (learned) {
        judge.learned_capture(continuity::tag_copy, 20);
      }
      judge.take();
      now += 16;
      // The next frame supplies nothing; its Present holds copy 12 again.
      continuity::supply none;
      none.allow(continuity::generic_boundary, now);
      auto again = shown(present++, now, source_slot::scene_depth, now - 32);
      again.depth_capture = 12;
      again[source_slot::scene_depth].state = source_state::held;
      judge.present(again, none, "test");
      const auto violations = judge.take();
      const bool stale = violations.size() == 1 && violations[0].kind == continuity::violation_kind::stale_shown &&
                         violations[0].shown_capture == 12 && violations[0].newer_capture == 20;
      require(learned ? stale : violations.empty(),
        learned ? "a Present holding an older copy beside a deferred copy whose capture id was learned was not stale_shown"
                : "a one-frame lag behind a deferred copy with no capture id known was stale_shown inside the tick bound");
    }
  }

  // A copy of the shown copy's own frame is that frame's depth too: the
  // frame's SR copy on another queue, readable only after its Present beside
  // the frame's tag copy that Present showed, is no newer depth, so a later
  // Present without a newer frame may hold the tag copy (the add-on's
  // consumed frame). Until 11-01 the judge took any higher capture id for
  // newer depth (seed dda773a575be6561 at frame 92, stale_shown alone, every
  // run). A later frame's copy readable beside that hold still makes it
  // stale_shown.
  void same_frame_copy_is_no_newer() {
    for (const bool later_frame : {false, true}) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (std::uint64_t i = 0; i != 3; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        frame.immediate = continuity::sr_evaluate;
        frame.capture[0] = 10 + i;
        auto record = shown(present++, now, source_slot::scene_depth, now);
        record.depth_capture = 10 + i;
        judge.present(record, frame, "test");
      }
      // The frame's tag copy at its call, which its Present shows, and its
      // SR copy on another queue, complete at the look after that Present.
      bool complete = false;
      continuity::supply both;
      both.add(continuity::tag_copy | continuity::sr_evaluate, now);
      both.immediate = continuity::tag_copy;
      both.elsewhere = continuity::sr_evaluate;
      both.capture[1] = 20;
      both.completions.push_back({continuity::sr_evaluate, [&complete] { return complete; }, nullptr, false, 22, false});
      auto tag = shown(present++, now, source_slot::scene_depth, now);
      tag.depth_capture = 20;
      complete = true;
      judge.present(tag, both, "test");
      require(judge.take().empty(), "a Present that showed its frame's tag copy, its SR copy in flight, was not clean");
      // Two Presents hold the tag copy: one with no source, one with a later
      // frame's own copy (later_frame) or none again.
      for (unsigned i = 0; i != 2; ++i) {
        now += 16;
        continuity::supply next;
        if (later_frame && i == 1) {
          next.add(continuity::sr_evaluate, now);
          next.immediate = continuity::sr_evaluate;
          next.capture[0] = 24;
        } else {
          next.allow(continuity::generic_boundary, now);
        }
        auto held = shown(present++, now, source_slot::scene_depth, now - 16 * (i + 1));
        held.depth_capture = 20;
        held[source_slot::scene_depth].state = source_state::held;
        judge.present(held, next, "test");
      }
      const auto violations = judge.take();
      const bool stale = violations.size() == 1 && violations[0].kind == continuity::violation_kind::stale_shown &&
                         violations[0].shown_capture == 20 && violations[0].newer_capture == 24;
      require(later_frame ? stale : violations.empty(),
        later_frame ? "a Present holding a frame's tag copy beside a later frame's readable copy was not stale_shown"
                    : "a Present holding a frame's tag copy beside the same frame's SR copy, readable after its Present, was stale_shown");
    }
  }

  // The stale rule ages a Present that holds an API copy from the newest
  // offered API copy's made tick, not its frame's stamp: after a brief hitch
  // before a frame's calls its copy is that much younger than the frame
  // (until 10-28 the Present of a frame with no API source holding a 157 ms
  // old copy made after such a hitch was stale at 297 ms).
  void stale_rule_ages_from_the_newest_copy() {
    continuity::judge judge;
    std::uint64_t now = 1000, present = 1;
    for (int i = 0; i != 4; ++i, now += 16) {
      continuity::supply frame;
      frame.add(continuity::sr_evaluate, now);
      frame.immediate = continuity::sr_evaluate;
      judge.present(shown(present++, now, source_slot::scene_depth, now), frame, "test");
    }
    // A frame stamped at now whose SR copy is made after a 140 ms hitch.
    continuity::supply hitched;
    hitched.allow(continuity::generic_boundary, now);
    hitched.add(continuity::sr_evaluate, now + 140);
    hitched.immediate = continuity::sr_evaluate;
    judge.present(shown(present++, now + 156, source_slot::scene_depth, now + 140), hitched, "test");
    // A frame with no API source, after a second brief hitch, holds that copy.
    continuity::supply none;
    none.allow(continuity::generic_boundary, now + 290);
    auto held = shown(present++, now + 297, source_slot::scene_depth, now + 140);
    held[source_slot::scene_depth].state = source_state::held;
    judge.take();
    judge.present(held, none, "test");
    require(judge.take().empty(), "a Present holding a 157 ms old copy made after its frame's hitch was stale by the frame's stamp");
  }

  // A real Present shows a copy recorded no earlier than the one the last
  // real Present showed (capture ids follow publication order across
  // providers and sources): a copy recorded before it is older depth after
  // newer, at an ownership handover to the other API (the 10-28 review's P2)
  // and, since 10-31 (owner ruling Q3), within one API too: FG Off's SR
  // successor published before the held FG copy (seed c40a42f921da5ba0,
  // frame 149), or FG On's FG input published before the shown SR copy.
  void handover_never_steps_back() {
    struct trial {
      sunshine_game3d::source_api next_api;
      std::uint64_t next_capture;
      bool stale;
      const char *message;
    };
    using sunshine_game3d::source_api;
    const trial trials[] {
      {source_api::ngx, 25, true, "a handover to the other API's copy recorded before the shown one was not stale_shown"},
      {source_api::ngx, 35, false, "a handover to a later copy was stale_shown"},
      {source_api::streamline, 31, false, "the same API's next copy was stale_shown"},
      {source_api::streamline, 25, true, "the same API's copy recorded before the shown one was not stale_shown"},
    };
    for (const auto &value : trials) {
      continuity::judge judge;
      std::uint64_t now = 1000, present = 1;
      for (std::uint64_t i = 0; i != 2; ++i, now += 16) {
        continuity::supply frame;
        frame.add(continuity::sr_evaluate, now);
        auto record = shown(present++, now, source_slot::scene_depth, now);
        record[source_slot::scene_depth].key.api = source_api::streamline;
        record.depth_capture = 29 + i;
        judge.present(record, frame, "test");
      }
      continuity::supply frame;
      frame.add(continuity::sr_evaluate, now);
      auto record = shown(present++, now, source_slot::scene_depth, now);
      record[source_slot::scene_depth].key.api = value.next_api;
      record.depth_capture = value.next_capture;
      judge.take();
      judge.present(record, frame, "test");
      const auto violations = judge.take();
      const bool stale = violations.size() == 1 && violations[0].kind == continuity::violation_kind::stale_shown &&
                         violations[0].shown_capture == value.next_capture && violations[0].newer_capture == 30;
      require(value.stale ? stale : violations.empty(), value.message);
    }
  }
}

int main() {
  single_present_lag_by_capture();
  handover_never_steps_back();
  deferred_copy_counts_by_its_learned_capture();
  same_frame_copy_is_no_newer();
  fg_hold_after_fg_off_judged_by_lineage();
  stale_rule_ages_from_the_newest_copy();
  generated_drop_after_late_signal();
  late_signal_conflicts_only_its_copies();
  holdable_never_older_than_shown();
  holdable_unknown_id_by_frame();
  unordered_copy_completed_after_decision();
  stale_rule_ages_a_frames_own_copy_from_its_capture();
  immediate_copy_counts_at_its_own_present();
  stall_inside_a_frame_that_reads_its_own_source();
  hold_past_the_source_age_after_a_hitch();
  copies_age_from_when_they_were_made();
  no_source_frame_holds_under_strict_holds();
  generated_present_of_its_fg_frame_after_a_pause();
  if (failures) {
    return EXIT_FAILURE;
  }
  std::puts("PASS continuity judge: a late signal conflicts only the copies read through its fence; a generated "
            "Present's dropped hold is R1's only where the real Present showed such a copy; a copy that completed "
            "after the add-on's decision is no holdable copy there; a frame's own copy is no older than its frame "
            "for the stale rule; a frame's own copy at its call is readable at its Present after a stall wherever it "
            "came; a hold past the source age is stale; copies age from when they were made; a no-source frame holds "
            "under strict_holds; a generated Present after its FG frame's real Present presents that frame; a "
            "Present one copy behind a later frame's readable copy of its kind is stale_shown by capture id, a copy of its own frame is no newer, also a deferred copy's "
            "learned one; a real Present never shows a copy recorded before the last real Present's, of either API, "
            "and a copy without a known capture id of a frame before the shown copy's is no holdable copy; "
            "after FG Off a held FG copy is judged by its lineage, "
            "also against the SR copies the FG hold kept off-screen; the stale rule ages a hold from its newest copy's "
            "made tick");
  return EXIT_SUCCESS;
}
