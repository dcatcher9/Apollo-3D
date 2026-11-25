// SPDX-License-Identifier: GPL-3.0-only
#include "native_depth_calibration.h"

#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_depth;
  void require(bool condition, const char *message) {
    if (!condition)
      throw std::runtime_error(message);
  }
  depth_calibration_key key(std::uint64_t id = 7, std::uint64_t layout = 1) {
    return { id, layout, 1920, 1080, 0, 0, 1280, 720 };
  }
  std::vector<float> gradient(float low = .1f, float high = .8f) {
    std::vector<float> raw(32 * 18);
    for (std::size_t i = 0; i != raw.size(); ++i)
      raw[i] = static_cast<float>(static_cast<double>(low) +
        (static_cast<double>(high) - low) * i / (raw.size() - 1));
    return raw;
  }
  void establish(native_depth_calibration &calibration, const std::vector<float> &raw,
                 depth_calibration_key source = key(), std::uint64_t first = 1) {
    for (unsigned i = 0; i != 3; ++i)
      calibration.sample(source, first + i, raw.data(), raw.size());
  }
  double nearness(float raw, const depth_calibration_result &value) {
    return (static_cast<double>(raw) - value.raw_anchor) * value.raw_gain;
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

  void test_fresh_sustained_calibration() {
    const auto raw = gradient();
    native_depth_calibration calibration;
    require(!calibration.sample(key(), 1, raw.data(), raw.size()), "first capture must not establish calibration");
    calibration.sample(key(), 1, raw.data(), raw.size());
    calibration.sample(key(), 0, raw.data(), raw.size());
    require(calibration.result(depth_orientation::normal).samples == 1, "duplicate or old captures do not increase confidence");
    calibration.sample(key(), 2, raw.data(), raw.size());
    require(!calibration.result(depth_orientation::normal).calibrated, "two captures still wait for calibration");
    calibration.sample(key(), 3, raw.data(), raw.size());
    const auto result = calibration.result(depth_orientation::normal);
    require(result.calibrated && result.samples == 3 && result.sampled_frame == 3, "three fresh captures establish calibration");
    require(std::abs(result.raw_anchor - .45f) < 1e-6 && std::abs(result.raw_gain + 1.f / .56f) < 1e-5,
      "raw median anchor and 10–90 percent span use the declared affine domain");
    require(!calibration.result(depth_orientation::automatic).calibrated, "unknown orientation keeps stereo unavailable");
    require(calibration.result(depth_orientation::reversed).calibrated, "an explicit orientation can resolve an unknown convention");
  }

  void test_affine_equivalence() {
    const auto normal = gradient(.08f, .92f);
    std::vector<float> reversed(normal.size());
    std::vector<float> affine(normal.size());
    for (std::size_t i = 0; i != normal.size(); ++i) {
      reversed[i] = 1.f - normal[i];
      affine[i] = .23f + .3f * normal[i];
    }
    native_depth_calibration n, r, a;
    establish(n, normal); establish(r, reversed); establish(a, affine);
    const auto nc = n.result(depth_orientation::normal), rc = r.result(depth_orientation::reversed), ac = a.result(depth_orientation::normal);
    require(nc.calibrated && rc.calibrated && ac.calibrated, "all affine depth representations calibrate");
    for (std::size_t i = 0; i != normal.size(); ++i) {
      require(std::abs(nearness(normal[i], nc) - nearness(reversed[i], rc)) < 1e-6,
        "normal and reversed representations produce the same signed geometry");
      require(std::abs(nearness(normal[i], nc) - nearness(affine[i], ac)) < 1e-6,
        "raw affine projection units cancel out of calibrated geometry");
    }
  }

  void test_narrow_raw_precision() {
    // This span disappears in FP16. Both source and complement are exactly
    // representable FP32 here, so reversal must retain the same geometry.
    const auto normal = gradient(1.f - 128.f * std::numeric_limits<float>::epsilon(),
      1.f - 8.f * std::numeric_limits<float>::epsilon());
    std::vector<float> reversed(normal.size());
    for (std::size_t i = 0; i != normal.size(); ++i)
      reversed[i] = 1.f - normal[i];
    native_depth_calibration n, r;
    establish(n, normal); establish(r, reversed);
    const auto nc = n.result(depth_orientation::normal), rc = r.result(depth_orientation::reversed);
    require(nc.calibrated && rc.calibrated && std::abs(nc.raw_gain) > 50000,
      "tiny near-one raw ranges remain usable without a fixed range threshold");
    for (std::size_t i = 0; i != normal.size(); ++i)
      require(std::abs(nearness(normal[i], nc) - nearness(reversed[i], rc)) < .006,
        "near-one reversal differs by at most the FP32 anchor rounding budget");
    const auto tiny = gradient(1.e-25f, 2.e-25f);
    native_depth_calibration t;
    establish(t, tiny);
    require(t.result(depth_orientation::reversed).calibrated, "tiny reversed raw values need no arbitrary epsilon");
  }

  void test_stability_and_invalidation() {
    native_depth_calibration calibration;
    const auto raw = gradient();
    establish(calibration, raw);
    const auto baseline = calibration.result(depth_orientation::normal);
    const auto changed = gradient(.4f, .41f);
    const std::vector<float> flat(raw.size(), 1.f);
    calibration.sample(key(), 4, changed.data(), changed.size());
    calibration.sample(key(), 5, flat.data(), flat.size(), false);
    const auto stable = calibration.result(depth_orientation::normal);
    require(stable.calibrated && stable.raw_anchor == baseline.raw_anchor && stable.raw_gain == baseline.raw_gain,
      "later narrow or flat scenes cannot pump a stable camera scale or anchor");
    require(stable.sampled_frame == baseline.sampled_frame, "calibration provenance is its establishing capture");
    // Slider values are consumers, not calibration inputs. Two strengths must
    // scale the same coordinate rather than recalibrate to the same intensity.
    const double near = nearness(raw.front(), stable);
    require(near > 0 && std::abs((.65 * near) / (.15 * near) - .65 / .15) < 1e-12,
      "nonzero strength changes cannot be normalized away by calibration");
    calibration.sample(key(8), 6, raw.data(), raw.size());
    require(!calibration.result(depth_orientation::normal).calibrated, "resource lifetime replacement invalidates calibration");
    establish(calibration, raw, key(8), 7);
    auto moved = key(8); moved.x = 8;
    calibration.sample(moved, 10, raw.data(), raw.size());
    require(!calibration.result(depth_orientation::normal).calibrated, "copied rectangle changes invalidate calibration");
    establish(calibration, raw, moved, 11);
    ++moved.layout_epoch;
    calibration.sample(moved, 14, raw.data(), raw.size());
    require(!calibration.result(depth_orientation::normal).calibrated, "layout epoch changes invalidate calibration");
  }

  void test_bad_and_flat_samples() {
    const auto raw = gradient();
    for (float value : { 0.f, 1.f, .5f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -1.f, 2.f }) {
      native_depth_calibration calibration;
      establish(calibration, std::vector<float>(raw.size(), value));
      require(!calibration.result(depth_orientation::normal).calibrated, "flat, nonfinite, and out-of-range data cannot calibrate");
    }
    native_depth_calibration calibration;
    calibration.sample(key(), 1, raw.data(), raw.size());
    calibration.sample(key(), 2, raw.data(), raw.size(), false);
    calibration.sample(key(), 3, raw.data(), raw.size());
    calibration.sample(key(), 4, raw.data(), raw.size());
    require(!calibration.result(depth_orientation::normal).calibrated, "bad content interrupts the three-capture warmup");
    calibration.sample(key(), 5, raw.data(), raw.size());
    require(calibration.result(depth_orientation::normal).calibrated, "three new good captures recover after bad data");
    auto cleared = raw;
    for (std::size_t i = 0; i != cleared.size() * 3 / 4; ++i)
      cleared[i] = 1;
    native_depth_calibration partial;
    establish(partial, cleared);
    require(partial.result(depth_orientation::normal).calibrated && partial.result(depth_orientation::normal).raw_anchor < .8f,
      "exact clear endpoints cannot become the calibration anchor");
  }
}

int main() {
  try {
    test_orientation();
    test_fresh_sustained_calibration();
    test_affine_equivalence();
    test_narrow_raw_precision();
    test_stability_and_invalidation();
    test_bad_and_flat_samples();
    std::puts("PASS: native raw-depth calibration (6 groups)");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
