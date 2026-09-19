// SPDX-License-Identifier: GPL-3.0-only
// CPU checks for measured fitting boundaries, independent of shader geometry.
#include "test_matched_disparity.h"
#include <array>
#include <cassert>

namespace {
  struct shifts {
    double eye[3][2] {};
    double disparity(unsigned i) const { return eye[i][0] - eye[i][1]; }
  };
  shifts observed(double gain, double zero, const char *) {
    shifts result;
    const double raw[] {.02, .005, .0125};
    for (unsigned i = 0; i < 3; ++i) {
      const double shift = 10 * gain * (raw[i] - zero);
      result.eye[i][0] = shift;
      result.eye[i][1] = -shift;
    }
    return result;
  }
}

int main() {
  std::ostringstream log;
  // A large gain change and a different zero must be fitted without the
  // negative-zero overshoot of a bilinear H,t0 parameterization.
  const auto target = observed(1200, .006, "target");
  const auto fitted = sunshine_matched_disparity::fit(target, observed, log);
  assert(sunshine_matched_disparity::endpoint_eye_error(fitted, target) < 1e-8);

  bool rejected = false;
  try {
    sunshine_matched_disparity::fit(target, [](double, double, const char *) { return shifts {}; }, log);
  } catch (const std::runtime_error &) { rejected = true; }
  assert(rejected);  // Flat/clamped observations must not be declared matched.

  auto displaced = fitted;
  for (auto &plane : displaced.eye) for (auto &eye : plane) eye += 1;
  assert(sunshine_matched_disparity::endpoint_eye_error(displaced, target) >= 1 - 1e-8);
  return 0;
}
