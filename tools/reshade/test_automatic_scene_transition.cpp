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
  check(state.update(true, 2111), 0); // Rendering interruption is not elapsed recovery.
  check(state.update(true, 2211), .2f);
  check(state.update(true, 2200), 0); // Clock rollback does not underflow.
  check(state.update(true, 2200), 0); // Same timestamp earns no credit.
  check(state.update(true, 2450), .5f);
  check(state.update(true, 2700), 1);
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
  std::cout << "Automatic stereo recovery timing passed\n";
}
