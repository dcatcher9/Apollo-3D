// SPDX-License-Identifier: GPL-3.0-only
#include "depth_orientation.h"

#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_depth;
  void require(bool condition, const char *message) {
    if (!condition)
      throw std::runtime_error(message);
  }

  void test_orientation() {
    depth_orientation_evidence evidence;
    evidence.observe(1, true, false, false);
    evidence.observe(1, true, false, false);
    require(evidence.agreeing_frames() == 1 && evidence.detected() == depth_orientation::automatic,
      "repeated clears in one frame cannot establish orientation");
    evidence.observe(2, true, false, false);
    evidence.observe(3, true, false, false);
    require(evidence.detected() == depth_orientation::reversed, "three exact zero-clear frames establish reversed depth");
    evidence.observe(4, false, false, false);
    require(evidence.detected() == depth_orientation::reversed, "absence of a clear does not invert known orientation");
    evidence.observe(5, true, true, false);
    require(evidence.detected() == depth_orientation::automatic, "mixed clear endpoints invalidate orientation immediately");
    for (unsigned frame = 6; frame != 9; ++frame)
      evidence.observe(frame, false, true, false);
    require(evidence.detected() == depth_orientation::normal, "three exact one-clear frames establish normal depth");
    evidence.observe(9, false, false, true);
    require(evidence.detected() == depth_orientation::automatic, "a non-endpoint clear is not reversed-depth evidence");
  }
}

int main() {
  try {
    test_orientation();
    std::puts("PASS: depth orientation evidence");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
