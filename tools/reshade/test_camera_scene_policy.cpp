// SPDX-License-Identifier: GPL-3.0-only
#include "camera_scene_policy.h"

#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
  using namespace sunshine_camera_scene;
  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }
  void close(double actual, double expected, double tolerance, const char *message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
  }
  registration camera(bool reversed = true, double near_plane = 0.1, double far_plane = 0.0) {
    registration r;
    r.unit_epoch = 7;
    r.source = {0x100, 9, 0, 1920, 1080, 0, 0, 1920, 1080};
    r.proof_admitted = true;
    if (far_plane == 0.0) {
      r.A = reversed ? 0.0 : 1.0;
      r.B = reversed ? near_plane : -near_plane;
    } else {
      r.A = reversed ? -near_plane / (far_plane - near_plane) : far_plane / (far_plane - near_plane);
      r.B = (reversed ? 1.0 : -1.0) * near_plane * far_plane / (far_plane - near_plane);
    }
    return r;
  }
  sample plane(registration r, std::uint64_t id, std::uint64_t time, double z) {
    r.frame = {id, id + 100};
    sample s;
    s.id = id;
    s.capture_ms = time;
    s.metadata = r;
    s.readback_frame = r.frame;
    s.readback_source = r.source;
    s.raw.fill(static_cast<float>(r.A + r.B / z));
    return s;
  }
  output calibrate(policy &p, registration r = camera(), double z = 10.0,
      std::uint64_t start = 0, std::uint64_t first_id = 1) {
    p.reset(r.unit_epoch, start);
    sample s;
    for (unsigned i = 0; i != 4; ++i) {
      s = plane(r, first_id + i, start + 250 * i, z);
      require(p.observe(s, s.capture_ms) == (i == 3 ? status::calibrated : status::calibrating),
        "Stable reference failed at the prescribed sample count/span");
    }
    auto o = p.tick(s.metadata, start + 750);
    require(o.ready && o.calibrated, "Registered calibrated frame is not ready");
    return o;
  }

  void startup_and_sampling() {
    policy p;
    auto r = camera();
    require(p.tick(r, 0).reason == status::uninitialized, "Default policy became ready");
    p.reset(7, 0);
    sample s;
    const std::uint64_t times[]{0, 16, 32, 749, 750};
    for (unsigned i = 0; i != 5; ++i) {
      s = plane(r, i + 1, times[i], 10.0);
      // Outside-center endpoints must not change the defined 4x4 statistic.
      for (unsigned y = 0; y != grid_height; ++y)
        for (unsigned x = 0; x != grid_width; ++x)
          if (x < 14 || x >= 18 || y < 7 || y >= 11) s.raw[y * grid_width + x] = 0.0f;
      const auto state = p.observe(s, times[i]);
      require(state == (i == 4 ? status::calibrated : status::calibrating), "Minimum wall span not enforced");
    }
    auto o = p.tick(s.metadata, 750);
    require(o.calibration_samples == 5 && o.ready, "High-cadence startup discarded observations");
    close(o.K, 10.0, 0.00001, "Reference is not the fixed center arithmetic inverse-distance mean");
    close(o.referenceZPD, 0.05, 1e-8, "Fixed referenceZPD changed");
    close(o.normalized_zero, 1.0, 1e-6, "Initial reference zero plane changed");

    policy three;
    three.reset(7, 0);
    for (unsigned i = 0; i != 3; ++i) {
      s = plane(r, i + 1, i * 500, 10);
      three.observe(s, s.capture_ms);
    }
    require(!three.tick(s.metadata, 1000).calibrated, "Elapsed time replaced the four-sample requirement");
  }

  void temporal_stability_and_timeout() {
    policy p;
    p.reset(7, 0);
    sample s;
    // A return to the initial q must not hide the observed interval excursion.
    const double distances[]{100, 100, 100, 2, 2, 2, 100};
    const unsigned times[]{0, 16, 32, 48, 300, 700, 750};
    for (unsigned i = 0; i != 7; ++i) {
      s = plane(camera(), i + 1, times[i], distances[i]);
      p.observe(s, times[i]);
    }
    require(!p.tick(s.metadata, 750).calibrated, "Unstable waiting interval was ignored");
    for (unsigned i = 0; i != 3; ++i) {
      s = plane(camera(), i + 8, 1000 + i * 250, 100);
      p.observe(s, s.capture_ms);
    }
    require(p.tick(s.metadata, 1500).ready, "A fresh stable window did not recover before deadline");

    policy timeout;
    timeout.reset(7, 0);
    for (unsigned i = 0; i != 3; ++i) {
      s = plane(camera(), i + 1, i * 2000, 10);
      timeout.observe(s, s.capture_ms);
    }
    timeout.scene_cut(4500);
    s = plane(camera(), 4, 5001, 10);
    require(timeout.observe(s, 5001) == status::calibration_timeout, "Cut extended the bounded initialization deadline");
    require(timeout.tick(s.metadata, 5001).reason == status::calibration_timeout, "Timeout did not remain visible");
    auto o = calibrate(timeout, camera(), 20, 6000);
    close(o.K, 20, 0.0001, "Explicit recalibration could not recover timeout");
  }

  void proof_and_order() {
    policy p;
    auto o = calibrate(p);
    const float fixed = o.K;
    auto s = plane(camera(), 5, 1000, 8);
    auto bad = s;
    bad.metadata.unit_epoch = 8;
    require(p.observe(bad, 1000) == status::unit_epoch_mismatch, "Mismatched unit epoch consumed");
    bad = s; bad.metadata.proof_admitted = false;
    require(p.observe(bad, 1000) == status::proof_missing, "Weak/unadmitted metadata consumed");
    bad = s; ++bad.readback_frame.token_generation;
    require(p.observe(bad, 1000) == status::frame_mismatch, "Readback from another token consumed");
    bad = s; ++bad.readback_source.lifetime;
    require(p.observe(bad, 1000) == status::source_mismatch, "Readback source lifetime mismatch consumed");
    require(p.observe(s, 1000) == status::calibrated, "Rejected packets consumed valid sample ID");
    require(p.observe(s, 1001) == status::duplicate_or_backward_sample, "Duplicate sample was accepted");
    bad = plane(camera(), 6, 999, 8);
    require(p.observe(bad, 1001) == status::duplicate_or_backward_sample, "Backward capture time was accepted");
    bad = plane(camera(), 6, 1002, 8); bad.metadata.frame.frame = 4; bad.readback_frame = bad.metadata.frame;
    require(p.observe(bad, 1002) == status::duplicate_or_backward_sample, "Backward source frame was accepted");
    auto current = s.metadata;
    --current.frame.frame;
    require(p.tick(current, 1002).reason == status::frame_mismatch, "A future target drove an older current frame");
    current = s.metadata; ++current.frame.token_generation;
    require(p.tick(current, 1002).reason == status::frame_mismatch, "Equal frame reused another token generation");
    current = s.metadata; current.proof_admitted = false;
    o = p.tick(current, 1002);
    require(!o.ready && o.calibrated && o.K == fixed, "Proof loss reset established calibration");
    current = s.metadata; current.unit_epoch = 8;
    require(p.tick(current, 1002).reason == status::unit_epoch_mismatch, "Current unit epoch mismatch was accepted");
    current = s.metadata;
    require(p.tick(current, 1001).reason == status::clock_went_backwards, "Backward wall clock accepted");
    require(p.tick(current, 1003).ready, "Valid same-frame current registration rejected");
    current.frame = {20, 120};
    require(p.tick(current, 1004).ready, "Fresh target cannot serve later registered frames");
    current.frame = {19, 119};
    require(p.tick(current, 1005).reason == status::frame_mismatch, "Current-frame sequence regressed");
  }

  void clear_and_invalid_depth() {
    for (const bool reversed : {false, true}) {
      auto r = camera(reversed, 0.1, 1000);
      policy p;
      p.reset(7, 0);
      auto s = plane(r, 1, 0, 10);
      for (float endpoint : {0.0f, 1.0f}) {
        auto bad = s;
        bad.raw[8 * grid_width + 16] = endpoint;
        require(p.observe(bad, 0) == status::clear_depth, "One ambiguous clear cell was trimmed or accepted");
      }
      for (float invalid : {-0.1f, 1.1f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        auto bad = s; bad.raw[8 * grid_width + 16] = invalid;
        require(p.observe(bad, 0) == status::invalid_depth, "Malformed center depth accepted");
      }
      require(p.observe(s, 0) == status::calibrating, "Invalid depth packets consumed the valid sample ID");
    }
    // Interior reversed values may be tiny because of units/projection; no
    // arbitrary raw-depth epsilon is allowed. q_ref=1e-12, K=1e12, max Kq=1000.
    policy tiny;
    const auto o = calibrate(tiny, camera(true, 1e9), 1e12);
    close(o.K / 1e12, 1.0, 0.00001, "Tiny inverse distance was treated as missing depth");
    policy interior;
    auto r = camera(true, 0.1);
    // Raw1e-6 is interior. Eventual full-domain admission is separate from
    // accepting this finite raw sample; it must never be epsilon-rejected.
    r.B = 100;
    interior.reset(7, 0);
    auto s = plane(r, 1, 0, 1e8);
    require(interior.observe(s, 0) == status::calibrating, "Small positive raw depth epsilon-rejected");
  }

  void units_and_projection() {
    policy base, scaled;
    auto a = calibrate(base, camera(true, 0.1), 10);
    auto b = calibrate(scaled, camera(true, 100), 10000);
    close(b.K / a.K, 1000, 0.001, "Changing only game length units altered normalized strength");
    close(b.reference_inverse_distance * 1000, a.reference_inverse_distance, 1e-7, "Reference units did not transform inversely");
    for (bool reversed : {false, true}) {
      for (double far_plane : {0.0, 1000.0}) {
        policy p;
        auto c = calibrate(p, camera(reversed, 0.1, far_plane), 10);
        close(c.K, 10, 0.001, "Normal/reversed finite/infinite projection changed physical reference");
        const float raw = static_cast<float>(camera(reversed, 0.1, far_plane).A + camera(reversed, 0.1, far_plane).B / 20);
        const double q = (raw - c.A) * c.inverseB;
        close(c.K * q, 0.5, 0.0001, "Affine shader coefficients do not recover equivalent inverse distance");
      }
    }
  }

  void scene_relative_reference() {
    policy wall, vista, mixed;
    auto near = calibrate(wall, camera(), 1);
    auto far = calibrate(vista, camera(), 100);
    close(near.K, 1, 0.00001, "Stable wall reference was normalized away");
    close(far.K, 100, 0.001, "Stable vista reference was normalized away");
    mixed.reset(7, 0);
    sample s;
    for (unsigned i = 0; i != 4; ++i) {
      s = plane(camera(), i + 1, 250 * i, 100);
      s.raw[8 * grid_width + 16] = 0.1f; // one z1 cell, fifteen z100 cells
      mixed.observe(s, s.capture_ms);
    }
    const auto m = mixed.tick(s.metadata, 750);
    require(m.ready, "Stable mixed interior reference was unexpectedly filtered");
    close(m.K, 1.0 / ((1.0 + 15 * 0.01) / 16), 0.0001, "Mixed scene reference stopped being arithmetic mean of inverse distance");
    require(std::abs(m.K - far.K) > 80, "Scene-relative calibration was misrepresented as game-independent");
  }

  void rate_tau_and_gaps() {
    policy p;
    auto o = calibrate(p);
    const float fixed = o.K;
    auto s = plane(camera(), 5, 1000, 2);
    p.observe(s, 1000);
    double previous = o.normalized_zero;
    o = p.tick(s.metadata, 1000);
    close(o.normalized_zero - previous, 0.5, 1e-6, "Large move did not honor 2 units/second rate");
    for (unsigned now = 1050; now <= 1250; now += 50) {
      previous = o.normalized_zero;
      o = p.tick(s.metadata, now);
      require(o.normalized_zero - previous <= 0.10000001 && o.K == fixed, "Rate limit or frozen K violated while walking");
    }
    auto lost = s.metadata; lost.proof_admitted = false;
    require(!p.tick(lost, 1300).ready, "Proof loss did not pause output");
    previous = o.normalized_zero;
    o = p.tick(s.metadata, 1400);
    close(o.normalized_zero, previous, 1e-12, "Proof gap accumulated movement credit");
    s = plane(camera(), 6, 2600, 2);
    p.observe(s, 2600); // previous target expired at2500
    previous = o.normalized_zero;
    o = p.tick(s.metadata, 2600);
    close(o.normalized_zero, previous, 1e-12, "Fresh target after gap caused a catch-up jump");

    policy one_step, ten_steps;
    calibrate(one_step); calibrate(ten_steps);
    s = plane(camera(), 5, 750, 10.0 / 1.1);
    s.capture_ms = 751;
    one_step.observe(s, 751); ten_steps.observe(s, 751);
    one_step.tick(s.metadata, 751); ten_steps.tick(s.metadata, 751);
    const double initial = one_step.tick(s.metadata, 751).normalized_zero;
    auto one = one_step.tick(s.metadata, 851);
    output many;
    for (unsigned now = 761; now <= 851; now += 10) many = ten_steps.tick(s.metadata, now);
    close(one.normalized_zero, many.normalized_zero, 1e-10, "Unclipped exponential depends on tick partition");
    const double target = static_cast<double>(one.K) * ((s.raw[8 * grid_width + 16] - one.A) * one.inverseB);
    close(one.normalized_zero, target + (initial - target) * std::exp(-0.1 / 0.5), 1e-10, "Zero-plane tau differs from0.5s");
  }

  void capture_age_and_discontinuity() {
    policy p;
    const auto initial = calibrate(p);
    auto s = plane(camera(), 5, 800, 5);
    require(p.observe(s, 2200) == status::calibrated, "Freshness ignored a valid delayed readback");
    auto o = p.tick(s.metadata, 2200);
    close(o.normalized_zero, initial.normalized_zero, 1e-12, "Delayed readback created accumulated gap credit");
    require(p.tick(s.metadata, 2299).ready, "Target expired before capture-based boundary");
    o = p.tick(s.metadata, 2300);
    require(!o.ready && o.calibrated && o.K == initial.K && o.reason == status::target_expired,
      "Readback arrival refreshed capture age or expiry reset K");
    s = plane(camera(), 6, 900, 5);
    require(p.observe(s, 2450) == status::stale_sample, "Stale delayed readback was admitted");
    require(p.observe(plane(camera(), 6, 2500, 5), 2499) == status::sample_in_future, "Future capture was admitted");

    policy startup;
    startup.reset(7, 0);
    for (unsigned i = 0; i != 3; ++i) {
      s = plane(camera(), i + 1, i * 250, 10); startup.observe(s, s.capture_ms);
    }
    auto replacement = camera(); ++replacement.source.lifetime;
    s = plane(replacement, 4, 750, 10);
    startup.observe(s, 750);
    o = startup.tick(s.metadata, 750);
    require(!o.calibrated && o.calibration_samples == 1, "Startup mixed different source lifetimes");
    s = plane(replacement, 5, 1000, 10); s.metadata.proof_admitted = false;
    startup.observe(s, 1000);
    s = plane(replacement, 6, 1250, 10); startup.observe(s, 1250);
    require(startup.tick(s.metadata, 1250).calibration_samples == 1, "Startup spanned missing registration proof");
  }

  void domain_and_fixed_epoch() {
    policy unsupported;
    unsupported.reset(7, 0);
    sample s;
    status state{};
    for (unsigned i = 0; i != 4; ++i) {
      s = plane(camera(true, 0.01), i + 1, i * 250, 1000);
      state = unsupported.observe(s, s.capture_ms);
    }
    require(unsupported.tick(s.metadata, 750).calibrated && unsupported.tick(s.metadata, 750).ready,
      "Finite R32 coordinate was rejected by a retired FP16 depth denominator limit");

    policy p;
    auto o = calibrate(p, camera(), 100);
    const float fixed = o.K;
    auto changed = plane(camera(true, 0.001), 5, 1000, 100).metadata;
    o = p.tick(changed, 1000);
    require(o.ready && o.K == fixed && o.calibrated,
      "Finite projection change was rejected or silently rescaled established K");
    require(shader_coordinate_domain(1e20f, 1.0f, 0.0) &&
        !shader_coordinate_domain(std::numeric_limits<float>::max(), 1000.0f, 0.0),
      "R32 admission lost finite-coordinate support or failed to reject real arithmetic overflow");
    s = plane(camera(true, 1.0), 5, 1100, 200);
    p.observe(s, 1100);
    o = p.tick(s.metadata, 1100);
    require(o.ready && o.K == fixed, "Supported projection change recalibrated K");
    auto replacement = camera(true, 1.0); ++replacement.source.native; ++replacement.source.lifetime;
    s = plane(replacement, 6, 1200, 200);
    require(p.tick(s.metadata, 1200).reason == status::source_mismatch, "Target crossed unrefreshed source layout");
    p.observe(s, 1200);
    require(p.tick(s.metadata, 1200).K == fixed, "Source replacement recalibrated established K");
    p.scene_cut(1300);
    o = p.tick(s.metadata, 1300);
    require(o.calibrated && !o.ready && o.K == fixed, "Scene cut reset scale or retained target");
    s = plane(replacement, 7, 1250, 20);
    require(p.observe(s, 1300) == status::stale_sample, "Pre-cut queued target leaked across cut");
    s = plane(replacement, 7, 1400, 20);
    p.observe(s, 1400);
    require(p.tick(s.metadata, 1400).K == fixed, "Post-cut target silently recalibrated scale");

    for (const auto bad : {registration{7, {}, camera().source, 0.0, 0.0, true},
        registration{7, {}, camera().source, 0.5, 1.0, true},
        registration{7, {}, camera().source, 0.0, std::numeric_limits<double>::denorm_min(), true},
        registration{7, {}, camera().source, std::numeric_limits<double>::infinity(), 1.0, true}}) {
      require(p.tick(bad, 1400).reason == status::invalid_projection, "Malformed affine projection admitted");
    }
    replacement.unit_epoch = 8;
    auto new_epoch = calibrate(p, replacement, 20, 1500);
    close(new_epoch.K, 20, 0.001, "Explicit unit epoch did not permit new reference");
  }
}

int main() {
  try {
    startup_and_sampling();
    temporal_stability_and_timeout();
    proof_and_order();
    clear_and_invalid_depth();
    units_and_projection();
    scene_relative_reference();
    rate_tau_and_gaps();
    capture_age_and_discontinuity();
    domain_and_fixed_epoch();
    std::puts("PASS camera scene policy: sampling, proof/order, stability/timeout, units/projection, scene-relative reference, zero-plane timing, freshness and fixed epochs");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL camera scene policy: %s\n", e.what());
    return 1;
  }
}
