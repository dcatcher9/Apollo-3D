// SPDX-License-Identifier: GPL-3.0-only
#include "depth_presentation_stats.h"
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
  using namespace sunshine_depth_stats;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

  void stable_statistics_include_missing_depth() {
    presentation_window window;
    require(window.snapshot(100).total() == 0, "An unobserved window invented samples");
    window.observe(1, 100, presentation_kind::fresh);
    window.observe(2, 200, presentation_kind::reused);
    window.observe(3, 300, presentation_kind::unavailable);
    require(window.snapshot(1099).total() == 0, "Partial first second escaped the collecting state");
    const auto initial = window.snapshot(1100);
    require(initial.window_ms == 1000 && initial.total() == 3 && initial.fresh == 1 &&
      initial.reused == 1 && initial.unavailable == 1, "Unavailable depth was excluded from the denominator");
    window.observe(4, 1100, presentation_kind::fresh);
    window.observe(5, 1200, presentation_kind::fresh);
    const auto stable = window.snapshot(2099);
    require(stable.total() == initial.total() && stable.fresh == initial.fresh &&
      stable.reused == initial.reused && stable.unavailable == initial.unavailable &&
      stable.window_ms == initial.window_ms, "Live fresh/reused alternation changed the current UI sample");
    const auto next = window.snapshot(2100);
    require(next.window_ms == 2000 && next.total() == 5 && next.fresh == 3,
      "The next completed second was not incorporated");
  }

  void rolling_window_expires_without_inventing_presentations() {
    presentation_window window;
    for (std::uint64_t i = 0; i != 7; ++i)
      window.observe(i, i * 1000, i < 2 ? presentation_kind::unavailable : presentation_kind::fresh);
    const auto active = window.snapshot(6000);
    require(active.window_ms == 5000 && active.total() == 5 && active.fresh == 4 && active.unavailable == 1,
      "The five-second window kept an expired or incomplete bucket");
    const auto expired = window.snapshot(12000);
    require(expired.window_ms == 5000 && expired.total() == 0,
      "An idle gap retained expired depth or counted unobserved presentations");
    window.observe(7, 12000, presentation_kind::reused);
    const auto resumed = window.snapshot(13000);
    require(resumed.total() == 1 && resumed.reused == 1,
      "Resuming after a gap resurrected an old ring bucket");
  }

  void one_observation_per_present_and_interval_reset() {
    presentation_window window;
    window.observe(0, 0, presentation_kind::fresh);
    window.observe(0, 100, presentation_kind::unavailable);
    const auto unique = window.snapshot(1000);
    require(unique.total() == 1 && unique.fresh == 1 && unique.unavailable == 0,
      "Duplicate effects passes counted a presentation twice");
    window.reset(); // Provider/source, FG mode or lifecycle change.
    window.observe(0, 2000, presentation_kind::reused);
    require(window.snapshot(2999).total() == 0, "Source reset exposed previous interval's statistics");
    const auto reset = window.snapshot(3000);
    require(reset.total() == 1 && reset.reused == 1 && reset.window_ms == 1000,
      "Explicit interval reset retained depth from another source");
  }

  void discontinuities_cannot_retain_future_samples() {
    presentation_window window;
    window.observe(10, 2000, presentation_kind::fresh);
    require(window.snapshot(1999).total() == 0, "Backward time exposed future samples");
    window.observe(11, 1000, presentation_kind::unavailable);
    const auto clock_reset = window.snapshot(2000);
    require(clock_reset.total() == 1 && clock_reset.unavailable == 1,
      "Clock rollback mixed observation intervals");
    window.observe(1, 3000, presentation_kind::reused);
    const auto present_reset = window.snapshot(4000);
    require(present_reset.total() == 1 && present_reset.reused == 1,
      "Presentation-index rollback mixed observation intervals");
    window.reset();
    const auto near_limit = std::numeric_limits<std::uint64_t>::max() - 1500;
    window.observe(1, near_limit, presentation_kind::fresh);
    require(window.snapshot(near_limit + 1000).fresh == 1,
      "A large monotonic timestamp overflowed bucket boundaries");
  }
}

int main() {
  try {
    stable_statistics_include_missing_depth();
    rolling_window_expires_without_inventing_presentations();
    one_observation_per_present_and_interval_reset();
    discontinuities_cannot_retain_future_samples();
    std::puts("Depth presentation statistics: 4 groups passed");
    return 0;
  }
  catch (const std::exception &e) {
    std::fprintf(stderr, "Depth presentation statistics failed: %s\n", e.what());
    return 1;
  }
}
