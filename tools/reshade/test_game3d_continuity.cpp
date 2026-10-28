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
// rule, however long the frame took. A stall inside a frame whose own
// Present reads its legal source, after its first call, is no warm-up.
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

  // A stall inside a frame whose own Present reads its legal source
  // (supply::immediate), after the frame's first call, leaves that source
  // readable: a mono Present there is a gap (the session tests' 10-27
  // review: a hitch between a frame's SR evaluation and its Present made
  // that Present mono, and the Present-to-Present stall rule excused it as
  // a warm-up). A stall before the frame's first call, or inside a frame
  // whose source is pipelined, stays a warm-up.
  void stall_inside_a_frame_that_reads_its_own_source() {
    struct trial {
      std::uint64_t stamp_after_ms;
      bool immediate, gap;
      const char *message;
    };
    const trial trials[] {
      {16, true, true, "a mono Present after a stall inside a frame whose own SR copy it reads was excused as a warm-up"},
      {400, true, false, "a mono Present after a stall before its frame's first call was judged a gap"},
      {16, false, false, "a mono Present after a stall inside a frame with a pipelined source was judged a gap"},
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
      judge.take();
      judge.present(mono(present++, last + 416), frame, "test");
      const auto violations = judge.take();
      const bool gap = violations.size() == 1 && violations[0].kind == continuity::violation_kind::gap;
      require(value.gap ? gap : violations.empty(), value.message);
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
}

int main() {
  generated_drop_after_late_signal();
  late_signal_conflicts_only_its_copies();
  unordered_copy_completed_after_decision();
  stale_rule_ages_a_frames_own_copy_from_its_capture();
  stall_inside_a_frame_that_reads_its_own_source();
  if (failures) {
    return EXIT_FAILURE;
  }
  std::puts("PASS continuity judge: a late signal conflicts only the copies read through its fence; a generated "
            "Present's dropped hold is R1's only where the real Present showed such a copy; a copy that completed "
            "after the add-on's decision is no holdable copy there; a frame's own copy is no older than its frame "
            "for the stale rule; a stall inside a frame that reads its own source is no warm-up");
  return EXIT_SUCCESS;
}
