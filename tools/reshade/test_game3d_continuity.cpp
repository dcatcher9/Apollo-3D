// SPDX-License-Identifier: GPL-3.0-only
// The continuity judge (test_game3d_continuity.h) on synthetic frame
// records: the O1 rules a session timeline cannot show failing without an
// add-on regression. A late signal of a game's order fence makes only the
// copies read through that fence unholdable (each copy names its own fence),
// and a hold a generated Present drops after such a signal is
// R1-late-signaller's only where the real Present showed such a copy. A copy
// from another queue that completed after the add-on's decision neither
// excuses nor demands a mono Present there.
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
  if (failures) {
    return EXIT_FAILURE;
  }
  std::puts("PASS continuity judge: a late signal conflicts only the copies read through its fence; a generated "
            "Present's dropped hold is R1's only where the real Present showed such a copy; a copy that completed "
            "after the add-on's decision is no holdable copy there");
  return EXIT_SUCCESS;
}
