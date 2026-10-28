// SPDX-License-Identifier: GPL-3.0-only
#include "automatic_scene_transition.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

static void check(float actual, float expected) {
  if (std::abs(actual - expected) > 0.00001f) {
    std::cerr << "Unexpected stereo recovery: " << actual << " expected " << expected << '\n';
    std::exit(1);
  }
}
int main() {
  sunshine_raw_scene::reentry_transition state;
  check(state.update(false, 100), 0);
  check(state.update(true, 1000), 0); // Waiting earns no strength credit.
  check(state.update(true, 1100), .2f);
  check(state.update(true, 1350), .7f);
  check(state.update(true, 1500), 1);
  check(state.update(true, 1700), 1);
  check(state.update(false, 1750), 0); // Actual current frame must be mono.
  check(state.update(true, 1760), 1);
  check(state.update(true, 1860), 1);
  // A pause in Presents between two Presents with ready depth loses no depth:
  // strength continues, however long the pause (owner ruling 10-27).
  check(state.update(true, 2111), 1);
  check(state.update(true, 2900), 1);
  // Presents shown without ready depth for more than 250 ms restart the ramp,
  // counted from the first of them.
  check(state.update(false, 2910), 0);
  check(state.update(false, 3100), 0);
  check(state.update(true, 3150), 1); // 240 ms since the first flat Present.
  check(state.update(false, 3160), 0);
  check(state.update(false, 3400), 0);
  check(state.update(true, 3411), 0); // 251 ms since the first flat Present.
  check(state.update(true, 3511), .2f);
  check(state.update(true, 3761), .7f);
  check(state.update(true, 4011), 1);
  // A pause whose first Present after it has no ready depth loses depth only
  // from that Present: the last ready image stayed on screen through the
  // pause, so a ready Present 10 ms later keeps the strength (the 10-27
  // review: a generated Present of a frame whose DLSS-G copy was still in
  // flight; until then the loss ran from the last ready Present, 4011).
  check(state.update(false, 4400), 0);
  check(state.update(true, 4410), 1);
  // A pause after a Present without ready depth is part of the loss: that
  // flat image stayed on screen, so 400 ms of it restart the ramp.
  check(state.update(false, 4500), 0);
  check(state.update(true, 4900), 0);
  check(state.update(true, 5000), .2f);
  // A pause inside the ramp keeps its progress and counts as at most 250 ms.
  check(state.update(true, 5400), .7f);
  check(state.update(true, 5500), .9f);
  check(state.update(true, 5490), 0); // Clock rollback restarts and does not underflow.
  check(state.update(true, 5490), 0); // Same timestamp earns no credit.
  check(state.update(true, 5740), .5f);
  check(state.update(true, 5990), 1);
  sunshine_raw_scene::reentry_transition alternating;
  check(alternating.update(true, 0), 0);
  for (std::uint64_t time = 10; time <= 1000; time += 10) {
    const bool ready = time % 20 == 0;
    check(alternating.update(ready, time), ready ? float(time / 2) / 500.f : 0.f);
  }
  check(alternating.update(false, 1010), 0);
  check(alternating.update(false, 1260), 0);
  check(alternating.update(true, 1270), 0); // A stalled last ready frame is not exposure.
  check(alternating.update(false, 1297), 0);
  check(alternating.update(true, 1320), .054f); // Only the 27 ms ready interval counted.
  check(alternating.update(true, 1320), .054f);
  check(alternating.update(true, 1400), .214f);
  using basis = sunshine_raw_scene::stereo_basis;
  sunshine_raw_scene::reentry_transition route;
  check(route.update(true, 0, basis::projection), 0);
  check(route.update(true, 250, basis::projection), .5f);
  check(route.update(true, 500, basis::projection), 1);
  // Missing projection uses the estimator, even if depth remains ready. Its
  // unrelated numeric gain must not inherit a full-strength presentation.
  check(route.update(true, 510, basis::raw), 0);
  check(route.update(true, 760, basis::raw), .5f);
  check(route.update(true, 1010, basis::raw), 1);
  check(route.update(true, 1020, basis::projection), 0);
  check(route.update(true, 1270, basis::projection), .5f);
  check(route.update(true, 1520, basis::projection), 1);
  check(route.update(true, 1530, basis::projection), 1); // Resource rotation leaves the basis unchanged.
  // A basis change restarts the ramp across a pause between ready Presents too.
  check(route.update(true, 2000, basis::raw), 0);
  check(route.update(true, 2250, basis::raw), .5f);
  check(route.update(true, 2650, basis::raw), 1);
  // A Present shown without Game 3D stereo keeps the basis: briefly, it ends
  // no exposure; for more than 250 ms, it restarts the ramp.
  check(route.update(true, 2700, basis::projection), 0);
  check(route.update(true, 3200, basis::projection), .5f);
  check(route.update(true, 3450, basis::projection), 1);
  route.shown_without_depth(3460);
  check(route.update(true, 3470, basis::projection), 1);
  route.shown_without_depth(3480);
  route.shown_without_depth(3700);
  check(route.update(true, 3740, basis::projection), 0);
  std::cout << "Automatic stereo recovery timing passed\n";
}
