// SPDX-License-Identifier: GPL-3.0-only
#include "projection_depth_controller.h"

#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_projection_depth;
  using range = sunshine_scene_gain::depth_range;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  void close(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      std::fprintf(stderr, "%s: actual=%.12g expected=%.12g\n", message, actual, expected);
      throw std::runtime_error(message);
    }
  }
  float expected_gain(double maximum) {
    const double requested = 1. / maximum;
    float value = static_cast<float>(requested);
    return value > requested ? std::nextafter(value, 0.f) : value;
  }
  constexpr domain viewport{3, 7};
  sample packet(std::uint64_t id, std::uint64_t capture_ms, range depths,
      coefficients projection = make(0., .0625), domain source = viewport) {
    sample value;
    value.id = id; value.capture_ms = capture_ms; value.logical_domain = source; value.projection = projection;
    const float a = projection.shader_A + float(depths.minimum) / projection.inverseB;
    const float b = projection.shader_A + float(depths.maximum) / projection.inverseB;
    value.range_supplied = value.range_valid = true;
    value.range_min = std::min(a, b); value.range_max = std::max(a, b);
    // The full-range record owns extrema, not this legacy diagnostic grid.
    value.raw.fill(projection.shader_A + float((depths.minimum + depths.maximum) * .5) / projection.inverseB);
    return value;
  }
  struct fixture {
    controller policy;
    coefficients projection;
    sunshine_scene_gain::limits budget{.5, 1., 1.};
    center_output output;
    std::uint64_t id{};
    explicit fixture(coefficients value = make(0., .0625)) : projection(value) {
      policy.reset(viewport, 1000); policy.configure(budget);
    }
    center_output capture(range depths, std::uint64_t tick) {
      policy.observe(packet(++id, tick, depths, projection), tick);
      return output = policy.evaluate(viewport, projection, tick);
    }
    center_output tick(std::uint64_t time) { return output = policy.evaluate(viewport, projection, time); }
    center_output start(range depths = {1., 3.}) {
      for (std::uint64_t time = 1000; time <= 1750; time += 250) {
        capture(depths, time);
        require(output.ready == (time == 1750), "Projection gain did not require four fresh spaced ranges");
      }
      require(output.calibration_samples == 4, "Projection startup accounting changed");
      return output;
    }
  };

  void full_image_moments_reject_mixed_camera_coefficients() {
    for (const auto projection : {make(0., .0625), make(1., -.0625)}) {
      fixture value(projection);
      for (std::uint64_t tick = 1000; tick <= 1750; tick += 250) {
        auto source = packet(++value.id, tick, {0., 4.}, projection);
        auto &m = source.moments;
        m.supplied = m.valid = true;
        m.A = projection.shader_A; m.inverseB = projection.inverseB;
        m.count = 3072 * 2304; m.sum = double(m.count); m.sum_squares = double(m.count) * 2.;
        m.tiles_x = 28; m.tiles_y = 21;
        value.policy.observe(source, tick);
        value.output = value.policy.evaluate(viewport, projection, tick);
      }
      require(value.output.ready && value.output.depth_statistics.mean == 1. &&
        value.output.depth_statistics.mean_square == 2. && value.output.depth_statistics.tiles_x == 28,
        "Projection path did not publish full decoded moments");
      close(value.output.target_K, expected_gain(4.), 0., "Projection path used the mean instead of the exact maximum");
      auto wrong = packet(++value.id, 2000, {0., 4.}, projection);
      wrong.moments.supplied = wrong.moments.valid = true;
      wrong.moments.A = projection.shader_A;
      wrong.moments.inverseB = projection.inverseB * 2.f;
      wrong.moments.count = 1000; wrong.moments.sum = 1000.; wrong.moments.sum_squares = 2000.;
      require(value.policy.observe(wrong, 2000) == center_status::invalid_depth,
        "Camera moments from a different frozen projection were accepted");
      require(!value.policy.evaluate(viewport, projection, 2000).has_depth_statistics,
        "Invalid projected moments retained fresh statistics");
    }
  }

  void measured_statistics_are_decoded_with_the_captured_projection() {
    for (const auto projection : {make(0., .0625), make(1., -.0625)}) {
      fixture value(projection);
      for (std::uint64_t time = 1000; time <= 1750; time += 250) {
        value.capture({1., 3.}, time);
        require(value.output.has_depth_statistics == (time == 1750), "Projection startup exposed partial statistics");
      }
      const auto &initial = value.output.depth_statistics;
      require(initial.minimum == 1. && initial.maximum == 3. && initial.mean == 2.,
        "Projected statistics retained raw encoding instead of decoded inverse depth");
      auto source = packet(++value.id, 2000, {0., 4.}, projection);
      source.raw.fill(projection.shader_A + 1.f / projection.inverseB);
      value.policy.observe(source, 2000); value.tick(2000);
      require(value.output.has_depth_statistics && value.output.depth_statistics.minimum == 0. &&
          value.output.depth_statistics.maximum == 4. && value.output.depth_statistics.mean == 1.,
        "Projection statistics lost exact extrema or replaced the mean with applied gain");
      require(value.output.K != 1.f, "Projection fixture did not exercise a smoothed applied gain");
      require(value.policy.observe(source, 2050) == center_status::stale_sample && value.tick(2050).has_depth_statistics,
        "Rejected duplicate invalidated accepted statistics");
      value.tick(3500);
      require(value.output.ready && !value.output.has_depth_statistics,
        "Projection hold exposed expired statistics as fresh");
      value.capture({1., 3.}, 3750);
      require(value.output.has_depth_statistics, "Fresh projected target did not restore statistics");
      require(value.policy.reset_reference(viewport, 4000) && !value.tick(4000).has_depth_statistics,
        "Projection recalibration retained statistics");
    }
  }

  void unchanged_span_survives_feasible_scene_and_zero_changes() {
    fixture value;
    const auto initial = value.start();
    close(initial.K, expected_gain(3.), 0., "Initial projection gain does not use the exact maximum");
    close(initial.q0, 2., 2e-6, "Initial projection zero does not use the independent startup midpoint");
    for (std::uint64_t time = 1800; time <= 2500; time += 50) {
      const auto next = value.capture({1., 3.}, time);
      require(next.ready && next.K == initial.K && next.q0 == initial.q0,
        "Feasible geometry recentered the plane or changed unchanged-span inverse-depth gain");
    }
    const auto moved = value.capture({2.5, 4.5}, 2750);
    require(moved.q0 > initial.q0 && moved.q0 < 3.5 && moved.target_q0 == 3.5 &&
      moved.K*(moved.q0-initial.q0) <= .25 + 2e-7,
      "Observed range change bypassed bounded midpoint convergence");
    require(moved.K < initial.K && moved.K >= initial.K / std::exp2(.25) * (1.-2e-7),
      "Changed maximum bypassed bounded gain tracking");
    const auto far = value.capture({0., 2.}, 3000);
    require(far.q0 < moved.q0 && far.q0 > 1. && far.target_q0 == 1. &&
      far.K*(moved.q0-far.q0) <= .25 + 2e-7,
      "Infinite-far range bypassed bounded convergence toward its midpoint");
    close(reference_zpd * far.K * (far.q0 - 1.) - reference_zpd * far.K * (far.q0 - 3.),
      .1 * far.K, 2e-8, "Convergence translation changed pairwise disparity");
  }

  void full_extrema_override_the_grid_and_legacy_grid_uses_every_cell() {
    fixture full;
    for (std::uint64_t time = 1000; time <= 1750; time += 250) {
      auto source = packet(++full.id, time, {1., 3.});
      source.raw.fill(.125f); // q=2 everywhere: a grid-derived range would be flat.
      full.policy.observe(source, time); full.tick(time);
    }
    close(full.output.K, expected_gain(3.), 0., "Controller did not use exact maximum independently of point-grid mean");
    fixture legacy;
    for (std::uint64_t time = 1000; time <= 1750; time += 250) {
      auto source = packet(++legacy.id, time, {1., 3.});
      source.range_supplied = source.range_valid = false;
      source.raw.fill(.125f); source.raw.front() = .0625f; source.raw.back() = .1875f;
      legacy.policy.observe(source, time); legacy.tick(time);
    }
    close(legacy.output.K, full.output.K, 2e-6, "Legacy fixture path still uses only the center patch");
    close(legacy.output.q0, full.output.q0, 2e-6, "Legacy grid and exact matching extrema disagree");
    const auto held = legacy.output;
    auto moved = packet(++legacy.id, 2000, {1., 3.}); moved.range_supplied = moved.range_valid = false;
    moved.raw.fill(.125f); moved.raw[8 * grid_width + 16] = .0625f; moved.raw[9 * grid_width + 15] = .1875f;
    legacy.policy.observe(moved, 2000); legacy.tick(2000);
    require(legacy.output.K == held.K && legacy.output.q0 == held.q0,
      "Moving extrema through the image center changed the unchanged-range geometry controls");
  }

  void flat_projected_depth_only_updates_established_convergence() {
    fixture value;
    for (std::uint64_t time = 1000; time <= 2000; time += 250) {
      const double depth = time % 500 ? 1. : 3.;
      const auto flat = value.capture({depth, depth}, time);
      require(!flat.ready && flat.calibration_samples == 0,
        "Flat projected clear/title depth seeded automatic calibration");
    }
    for (std::uint64_t time = 2250; time <= 3000; time += 250) value.capture({1., 3.}, time);
    require(value.output.ready, "Nonflat projected gameplay did not initialize after title frames");
    const auto held = value.capture({1., 2.}, 3250);
    const auto flat = value.capture({3., 3.}, 3500);
    require(flat.ready && flat.reason == center_status::ready && flat.K == held.K && flat.q0 > held.q0 &&
      flat.q0 < 3. && flat.target_q0 == 3. && flat.target_K == 0. && flat.has_depth_statistics &&
      flat.K*(flat.q0-held.q0) <= .25 + 2e-7,
      "Flat projected surface changed gain or failed bounded convergence");
    const auto later = value.tick(3750);
    require(later.K == held.K && later.q0 > flat.q0 && later.q0 < 3.,
      "Flat positive projected depth pursued old gain or stopped converging");
    const auto infinite = value.capture({0., 0.}, 4000);
    require(infinite.ready && infinite.reason == center_status::holding && infinite.K == later.K &&
      infinite.q0 == later.q0 && !infinite.has_depth_statistics,
      "All-infinity projected depth did not hold established controls");
  }

  void transient_near_maximum_has_bounded_zero_and_gain_response() {
    fixture value; value.budget = {.5, 1., 1.}; value.policy.configure(value.budget);
    const auto initial = value.start();
    const auto spike = value.capture({1., 11.}, 1800);
    require(spike.ready && spike.K < initial.K && spike.K >= initial.K / std::exp2(.05) * (1.-2e-7),
      "One projected maximum caused an immediate global gain collapse");
    require(spike.q0 > initial.q0 && spike.q0 < 6. && spike.target_q0 == 6. &&
      spike.K*(spike.q0-initial.q0) <= .05 + 2e-7,
      "Near maximum bypassed zero movement budget");
    close(spike.target_K, expected_gain(11.), 0., "Projection target did not use the full maximum");
    float previous = spike.K, previous_zero = spike.q0;
    for (std::uint64_t time = 1850; time <= 15000; time += 50) {
      const auto next = value.capture({1., 3.}, time);
      require(next.ready && next.K >= previous && next.K <= initial.K &&
        next.K <= previous * std::exp2(.05) * (1. + 2e-7) &&
        next.q0 <= previous_zero && next.q0 >= initial.q0 && next.K*(previous_zero-next.q0) <= .05 + 2e-7,
        "Maximum exit exceeded gain or zero response bounds");
      previous = next.K; previous_zero = next.q0;
    }
    require(value.output.K == initial.K,
      "Projection gain required a reset to recover after the near point exited");
    close(value.output.q0, initial.q0, 2e-6, "Projection zero retained permanent near-point drift");
  }

  void strength_zero_and_restoration_use_the_current_range() {
    fixture value; value.budget = {.5, 1., 1.}; value.policy.configure(value.budget);
    const auto initial = value.start();
    value.budget.strength = 0.; value.policy.configure(value.budget);
    const auto zero = value.capture({1., 11.}, 2000);
    require(zero.ready && zero.K == initial.K && zero.q0 == initial.q0,
      "Zero strength changed gain or zero-plane placement");
    value.budget.strength = .5; value.policy.configure(value.budget);
    const auto restored = value.tick(2001);
    require(restored.ready && restored.K == initial.K && restored.q0 == initial.q0,
      "Strength restoration spent paused time or changed the feasible plane");
    close(restored.target_K, expected_gain(11.), 0., "Strength restoration lost the current maximum target");
    const auto resumed = value.tick(2051);
    require(resumed.K < initial.K && resumed.K >= initial.K / std::exp2(.05) * (1.-2e-7),
      "Fresh motion did not resume bounded maximum-based adaptation");
  }

  void units_orientation_storage_and_clipping_preserve_geometry() {
    for (const bool reversed : {false, true}) for (const double units : {.001, 1., 100., 1e6}) {
      fixture baseline;
      fixture changed(make(reversed ? 0. : 1., (reversed ? .0625 : -.0625) * units, 2., -.5));
      std::uint64_t time = 1000;
      for (const range depths : {range{1., 3.}, {1., 3.}, {1., 3.}, {1., 3.}, {2.5, 4.5}, {0., .5}, {1., 11.}, {5., 7.}}) {
        const auto a = baseline.capture(depths, time);
        const auto b = changed.capture({depths.minimum / units, depths.maximum / units}, time);
        require(a.ready == b.ready, "Unit/orientation/storage change altered readiness");
        if (a.ready) {
          close(a.K, b.K / units, 1e-4, "Unit/orientation/storage change altered live or limited gain");
          close(a.q0, b.q0 * units, 1e-5, "Unit/orientation/storage change altered minimum-change convergence");
          close(reference_zpd * a.K * (a.q0 - depths.maximum),
            reference_zpd * b.K * (b.q0 - depths.maximum / units), 8e-6,
            "Equivalent physical scene produced different signed disparity");
        }
        time += 250;
      }
    }
    fixture clipping;
    const auto original = clipping.start();
    clipping.projection = make(0., .125);
    const auto near = clipping.capture({1., 3.}, 2000);
    require(near.K == original.K && near.q0 == original.q0, "Near clipping changed physical gain or convergence");
    clipping.projection = make(-.125 / (100. - .125), .125 * 100. / (100. - .125));
    const auto finite_far = clipping.capture({1., 3.}, 2250);
    close(finite_far.K, original.K, 2e-6, "Finite-far affine offset recalibrated unchanged-span gain");
    close(finite_far.q0, original.q0, 2e-6, "Finite-far affine offset changed physical zero");
  }

  void stale_invalid_and_feedback_reset_hold_applied_controls() {
    fixture value;
    const auto initial = value.start();
    auto malformed = packet(++value.id, 2000, {1., 3.});
    malformed.range_min = NAN;
    require(value.policy.observe(malformed, 2000) == center_status::invalid_depth,
      "Malformed full-range evidence fell back to a valid diagnostic grid");
    auto held = value.tick(2000);
    require(held.ready && held.K == initial.K && held.q0 == initial.q0, "Invalid measurement erased established geometry");
    require(value.policy.observe(packet(value.id, 2000, {1., 11.}), 2050) == center_status::stale_sample,
      "Repeated capture refreshed range evidence");
    require(value.policy.observe(packet(++value.id, 2100, {1., 11.}), 5000) == center_status::stale_sample,
      "Late arrival timestamp refreshed expired capture evidence");
    held = value.tick(5000);
    require(held.ready && held.reason == center_status::holding && held.K == initial.K && held.q0 == initial.q0,
      "Evidence expiry moved or erased established geometry");
    value.policy.synchronize_feedback({1, false});
    require(value.policy.observe(packet(++value.id, 5250, {1., 11.}), 5250) == center_status::stale_sample,
      "Prior feedback revision replaced current depth controls");
    auto reset = packet(++value.id, 5500, {1., 11.}); reset.feedback = {1, true};
    value.policy.observe(reset, 5500); held = value.tick(5500);
    require(held.ready && held.K == initial.K && held.q0 == initial.q0,
      "Ordinary feedback reset discarded gain or used the reset frame to move convergence");
    require(!value.policy.evaluate(viewport, {}, 5750).ready, "Missing current projection authorized rendering");
    require(!value.tick(5700).ready, "Regressed presentation clock authorized rendering");
    require(value.tick(6000).K == initial.K, "Clock recovery recalibrated gain");
  }

  void explicit_reset_and_domain_watermarks_require_fresh_startup() {
    fixture value;
    const auto initial = value.start();
    const domain other{viewport.epoch, viewport.viewport + 1};
    require(!value.policy.reset_reference(other, 2000), "Different viewport reset established reference");
    require(value.tick(2000).K == initial.K, "Rejected reset changed gain");
    require(value.policy.reset_reference(viewport, 2000), "Explicit reference reset failed");
    require(!value.tick(2000).ready && value.output.K == 0 && value.output.calibration_samples == 0,
      "Explicit reset retained numerical calibration");
    require(value.policy.observe(packet(100, 2000, {.125, .375}), 2000) == center_status::stale_sample,
      "Same-tick pre-action evidence seeded the new reference");
    require(value.policy.observe(packet(value.id, 2001, {.125, .375}), 2001) == center_status::stale_sample,
      "Previously consumed capture seeded the new reference");
    for (std::uint64_t time = 2001; time <= 2751; time += 250) {
      const auto current = value.capture({.125, .375}, time);
      require(current.ready == (time == 2751), "Explicit reset skipped fresh four-observation startup");
    }
    require(value.output.K == expected_gain(.375) && value.output.q0 == .25f && !value.output.limited,
      "Explicit reset did not replace reference and zero together");
    const domain replacement{viewport.epoch + 1, viewport.viewport + 1};
    value.policy.reset(replacement, 3000); value.policy.configure(value.budget);
    require(value.policy.reset_reference(replacement, 3000), "New-domain explicit reset failed");
    require(value.policy.observe(packet(1, 3000, {1., 3.}, value.projection, replacement), 3000) == center_status::stale_sample,
      "New-domain same-tick reset admitted pre-action evidence");
    require(value.policy.observe(packet(2, 3001, {1., 3.}), 3001) == center_status::invalid_domain,
      "Previous logical viewport supplied range evidence");
    require(!value.policy.evaluate(replacement, value.projection, 3001).ready, "New domain inherited established gain");
  }

  void unsupported_current_encoding_cannot_change_applied_gain() {
    fixture value;
    const auto original = value.start({.001, .003});
    value.projection = make(0., 1e-38);
    require(value.projection.valid() && !supports_scale(value.projection, original.K),
      "Unsupported-encoding fixture is representable after all");
    const auto unsafe = value.tick(2000);
    require(!unsafe.ready && unsafe.reason == center_status::unsupported_shader_domain && unsafe.K == original.K,
      "Unsupported encoding rendered or changed the established gain");
    value.projection = make(0., .0625);
    const auto restored = value.tick(2250);
    require(restored.ready && restored.K == original.K && restored.q0 == original.q0,
      "Encoding recovery failed to restore the same numerical controls");
  }
}

int main() {
  try {
    full_image_moments_reject_mixed_camera_coefficients();
    unchanged_span_survives_feasible_scene_and_zero_changes();
    measured_statistics_are_decoded_with_the_captured_projection();
    full_extrema_override_the_grid_and_legacy_grid_uses_every_cell();
    flat_projected_depth_only_updates_established_convergence();
    transient_near_maximum_has_bounded_zero_and_gain_response();
    strength_zero_and_restoration_use_the_current_range();
    units_orientation_storage_and_clipping_preserve_geometry();
    stale_invalid_and_feedback_reset_hold_applied_controls();
    explicit_reset_and_domain_watermarks_require_fresh_startup();
    unsupported_current_encoding_cannot_change_applied_gain();
    std::puts("PASS projection controls: maximum reference, bounded midpoint convergence, flat-wall behavior, units and source history");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL projection controls: %s\n", error.what()); return 1;
  }
}
