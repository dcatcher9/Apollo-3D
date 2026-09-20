// SPDX-License-Identifier: GPL-3.0-only
#include "scene_gain.h"

#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
  using namespace sunshine_scene_gain;
  using feedback = sunshine_scene_feedback::sample;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  void close(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      std::fprintf(stderr, "%s: actual=%.12g expected=%.12g\n", message, actual, expected);
      throw std::runtime_error(message);
    }
  }
  float expected_gain(double maximum, double normalization = 1.) {
    const double requested = normalization / maximum;
    float value = static_cast<float>(requested);
    return value > requested ? std::nextafter(value, 0.f) : value;
  }
  struct fixture {
    policy gain;
    feedback context{1};
    limits budget{.5, 1., 1.};
    fixture() { gain.configure(budget); gain.synchronize(context); }
    void capture(depth_range range, std::uint64_t time) {
      gain.advance(time); gain.synchronize(context); gain.observe(range, time, context); gain.update(time);
    }
    void start(depth_range range = {1., 3.}) {
      for (std::uint64_t time = 1000; time <= 1750; time += 250) {
        capture(range, time);
        require(gain.initialized() == (time == 1750), "Startup did not require four spaced ranges");
      }
    }
    void placed(depth_range range) const {
      require(gain.target_zero() >= range.minimum && gain.target_zero() <= range.maximum, "Zero target escaped observed scene depth");
      require(std::isfinite(gain.zero()) && gain.zero() >= 0, "Applied zero left its numeric domain");
      require(std::isnormal(gain.value()) && gain.value() > 0, "Applied gain left the finite positive domain");
      // The GPU owns the current per-pixel displacement limit. Delayed scene
      // evidence and smoothed gain cannot establish that bound on the CPU.
    }
  };

  void full_moments_preserve_statistics_while_maximum_controls_gain() {
    std::array<float, 32 * 18> grid;
    grid.fill(.75f); // Deliberately unrepresentative selector points.
    sunshine_depth_statistics::moments moments;
    moments.supplied = moments.valid = true;
    moments.count = 1000;
    moments.sum = 250.;
    moments.sum_squares = 125.; // Half zero, half .5.
    moments.tiles_x = 28; moments.tiles_y = 21;
    auto decoded = decode_range(grid, true, true, 0.f, .5f, 0.f, 1.f, 0.f, 1.f, moments);
    require(decoded.valid() && decoded.mean == .25 && decoded.mean_square == .125 &&
      decoded.pixel_count == 1000 && decoded.tiles_x == 28 && decoded.tiles_y == 21,
      "Full-pixel evidence was replaced by the point grid or lost its dimensions");
    fixture value; value.start(decoded);
    close(value.gain.target(), expected_gain(.5), 0., "Full moments or point-grid mean replaced the exact maximum reference");
    for (int failure = 0; failure != 6; ++failure) {
      auto bad = moments;
      switch (failure) {
        case 0: bad.valid = false; break;
        case 1: bad.A = 1.f; break;
        case 2: bad.inverseB = -1.f; break;
        case 3: bad.count = 0; break;
        case 4: bad.sum = std::numeric_limits<double>::infinity(); break;
        case 5: bad.sum_squares = 0.; break;
      }
      require(!decode_range(grid, true, true, 0.f, .5f, 0.f, 1.f, 0.f, 1.f, bad).valid(),
        "Malformed or wrong-basis full moments fell back to sparse points");
    }
    moments.A = 1.f; moments.inverseB = -1.f;
    auto normal = decode_range(grid, true, true, .5f, 1.f, 1.f, -1.f, 0.f, 1.f, moments);
    require(normal.valid() && normal.mean == decoded.mean && normal.mean_square == decoded.mean_square,
      "Normal/reversed capture moments changed scene statistics");
    for (const float scale : {1e-30f, 1e30f}) {
      moments.A = 0.f; moments.inverseB = scale;
      moments.sum = 250. * double(scale);
      moments.sum_squares = 125. * double(scale) * double(scale);
      const auto units = decode_range(grid, true, true, 0.f, .5f, 0.f, scale, 0.f, 1.f, moments);
      require(units.valid(), "Representable full-image moments overflowed/underflowed after unit conversion");
      close(units.mean / scale, .25, 1e-7, "Full-image mean depends on inverse-depth units");
      close(units.mean_square / (double(scale)*scale), .125, 1e-7, "Second moment lost squared units");
    }
    moments.inverseB = 1.f; moments.sum = moments.sum_squares = 0.;
    const auto zero = decode_range(grid, true, true, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, moments);
    require(zero.valid() && zero.mean == 0. && zero.mean_square == 0.,
      "Zero depth was rejected as malformed or replaced by a fabricated epsilon");
  }

  void maximum_reference_is_independent_of_coverage_mean_and_zero() {
    for (const double mean : {1.001, 2., 2.999}) {
      fixture value; value.start({1., 3., mean});
      close(value.gain.value(), expected_gain(3.), 0., "Pixel occupancy changed the maximum reference");
      close(value.gain.reference(), 3., 0., "Reference is not the accepted maximum");
      close(value.gain.zero(), 2., 0., "Pixel occupancy changed midpoint initialization");
      close(value.gain.target_zero(), 2., 0., "Zero target is not the range midpoint");
      for (std::uint64_t time = 1800; time <= 2200; time += 50) {
        value.capture({1., 3., time % 100 ? 1.01 : 2.99}, time);
        require(value.gain.value() == expected_gain(3.) && value.gain.zero() == 2.f,
          "Changed occupancy moved controls despite unchanged extrema");
      }
      value.capture({2.5, 4.5, 2.6}, 2200);
      require(value.gain.zero() == 2.f && value.gain.value() == expected_gain(3.),
        "A new range moved applied controls without elapsed time");
      close(value.gain.target_zero(), 3.5, 0., "Changed range retained an obsolete zero target");
      require(value.gain.zero() < value.gain.lower(), "Fixture failed to retain a transient out-of-range zero");
      value.capture({0., 2., .001}, 2200);
      close(value.gain.target_zero(), 1., 0., "Infinity changed midpoint semantics");
      close(.05 * value.gain.value() * ((value.gain.zero()-3.)-(value.gain.zero()-1.)),
        -.1 * expected_gain(3.), 1e-9, "Pure zero translation changed pairwise disparity");
    }
  }

  void midpoint_is_minimax_epsilon_continuous_and_approach_monotone() {
    for (const double low : {0., 1e-9, .5, 2.}) {
      fixture value; value.start({low, 4., low + .001});
      const double midpoint = (low + 4.) * .5;
      close(value.gain.target_zero(), midpoint, 2e-7, "Zero target is not the exact inverse-depth midpoint");
      const double radius = (4.-low)*.5;
      for (const double offset : {-1., -.25, 0., .25, 1.}) {
        const double alternative = midpoint + offset;
        require(std::max(std::abs(low-alternative), std::abs(4.-alternative)) >= radius,
          "Midpoint failed the minimax endpoint displacement criterion");
      }
    }
    fixture zero, epsilon; zero.start({0., 4., .001}); epsilon.start({1e-9, 4., .001});
    close(epsilon.gain.target_zero()-zero.gain.target_zero(), .5e-9, 1e-12,
      "A tiny positive far endpoint introduced an infinity classification discontinuity");
    for (const double far : {0., 1.}) {
      fixture value; value.start({far, 3.});
      double previous_pop = value.gain.value()*(3.-value.gain.zero());
      std::uint64_t time = 1750;
      for (const double near : {4., 6., 12.}) {
        // Compare settled controls: temporal lag is separately rate-bounded,
        // not claimed to be monotone under arbitrary changing measurements.
        for (unsigned i = 0; i < 400; ++i) value.capture({far, near}, time += 50);
        const double pop = value.gain.value()*(near-value.gain.zero());
        require(pop + 2e-6 >= previous_pop, "Approaching the nearest object reversed its settled pop");
        close(pop, .5*(1.-far/near), 2e-6, "Settled approach disagrees with max/midpoint geometry");
        previous_pop = pop;
      }
    }
  }

  void near_spike_has_bounded_zero_and_gain_response() {
    fixture value; value.start();
    const float original = value.gain.value(), zero = value.gain.zero();
    value.capture({0., 200., 2.}, 1800);
    require(value.gain.value() < original && value.gain.value() >= original / std::exp2(.05) * (1.-2e-7),
      "One near-spike observation bypassed bounded gain response");
    require(value.gain.zero() > zero && value.gain.zero() < value.gain.target_zero() &&
      value.gain.value() * (value.gain.zero()-zero) / value.budget.normalization <= .05 + 2e-7,
      "Near spike forced an instantaneous or over-budget zero movement");
    close(value.gain.target_zero(), 100., 0., "Near spike did not set the midpoint target");
    close(value.gain.target(), expected_gain(200.), 0., "Near maximum did not update gain target");
    require(.05 * value.budget.strength * value.gain.value() * (200.-value.gain.zero()) > value.budget.foreground,
      "Fixture no longer exercises GPU displacement protection during delayed adaptation");
    float previous_gain = value.gain.value(), previous_zero = value.gain.zero();
    for (std::uint64_t time = 1850; time <= 15000; time += 50) {
      value.capture({1., 3., 2.}, time);
      require(value.gain.value() >= previous_gain && value.gain.value() <= original &&
        value.gain.value() <= previous_gain * std::exp2(.05) * (1. + 2e-7),
        "Gain recovery exceeded its target or rate bound");
      require(value.gain.zero() <= previous_zero && value.gain.zero() >= zero &&
        value.gain.value() * (previous_zero-value.gain.zero()) / value.budget.normalization <= .05 + 2e-7,
        "Zero recovery exceeded its target or normalized rate bound");
      previous_gain = value.gain.value(); previous_zero = value.gain.zero();
    }
    require(value.gain.value() == original, "Spike exit needed manual gain recalibration");
    close(value.gain.zero(), zero, 2e-6, "Spike exit left permanent zero drift");
  }

  void recovery_cannot_bank_gap_invalid_or_expired_time() {
    fixture value; value.start({0., 200., 2.}); value.capture({0., 200., 2.}, 2000);
    const auto limited = value.gain.value();
    value.capture({0., 4., 2.}, 2050);
    const auto recovering = value.gain.value();
    require(recovering > limited && recovering < value.gain.target(), "Recovery fixture did not start gradual restoration");
    value.gain.update(2050);
    require(value.gain.value() == recovering, "Repeated evaluation at one timestamp advanced gain recovery");
    value.capture({0., 4., 2.}, 5000);
    require(value.gain.value() == recovering, "Long presentation gap was credited as gain recovery time");
    value.capture({0., 4., 2.}, 5050);
    const auto resumed = value.gain.value();
    require(resumed > recovering && resumed <= recovering * std::exp2(.05) * (1. + 2e-7),
      "Fresh post-gap observation did not restart bounded automatic recovery");
    value.capture({NAN, 7.}, 5100);
    value.capture({0., 4., 2.}, 5150);
    require(value.gain.value() == resumed, "Invalid evidence interval earned gain recovery credit");
    value.capture({0., 4., 2.}, 5200);
    const auto after_invalid = value.gain.value();
    require(after_invalid > resumed && after_invalid <= resumed * std::exp2(.05) * (1. + 2e-7),
      "Recovery did not restart after two valid continuous observations");
    value.gain.update(7000); value.gain.update(7050);
    require(!value.gain.has_target() && value.gain.value() == after_invalid,
      "Expired range evidence continued restoring gain");
    value.capture({0., 4., 2.}, 7100);
    require(value.gain.value() == after_invalid, "First post-expiry capture spent accumulated recovery time");
  }

  void strength_changes_hold_geometry_and_normalization_updates_without_recalibration() {
    fixture value; value.start();
    const auto original_gain = value.gain.value();
    value.budget.strength = 0.; value.gain.configure(value.budget);
    const auto held_zero = value.gain.zero();
    value.capture({0., 200., 2.}, 2000);
    require(value.gain.value() == original_gain && value.gain.zero() == held_zero,
      "Zero artistic strength moved established convergence or gain");
    value.budget.strength = .5; value.gain.configure(value.budget); value.gain.update(2001);
    require(value.gain.value() == original_gain && value.gain.zero() == held_zero,
      "Restoring strength credited paused adaptation or forced a safety zero shift");
    close(value.gain.target(), expected_gain(200.), 0., "Strength restoration lost current maximum evidence");
    fixture changed; changed.start();
    const auto prior = changed.gain.value();
    changed.budget.normalization = 2.; changed.gain.configure(changed.budget);
    changed.gain.update(1750);
    close(changed.gain.target(), expected_gain(3., 2.), 0., "Changed display normalization retained an obsolete target");
    require(changed.gain.value() == prior && changed.gain.zero() == 2.f && changed.gain.samples() == 4,
      "Display normalization reset unit history or changed gain without elapsed time");
    changed.gain.update(1800);
    require(changed.gain.value() > prior && changed.gain.value() <= prior * std::exp2(.05) * (1.+2e-7),
      "New normalization did not use bounded adaptation of existing depth evidence");
    close(changed.gain.reference(), 3., 0., "Normalization change mislabeled reciprocal gain as measured reference");
  }

  void zero_adaptation_cannot_bank_missing_invalid_or_expired_time() {
    fixture value; value.start();
    value.capture({4., 8.}, 1800);
    float held_gain = value.gain.value(), held_zero = value.gain.zero();
    require(held_zero > 2.f && held_zero < 4.f, "Zero-gap fixture lacks a pending out-of-range target");
    value.gain.suspend(); // Owner reports a missing current depth image.
    value.capture({4., 8.}, 2000);
    require(value.gain.value() == held_gain && value.gain.zero() == held_zero,
      "Missing presentation time moved held gain or zero on reentry");
    value.capture({4., 8.}, 2050);
    require(value.gain.zero() > held_zero && value.gain.value()*(value.gain.zero()-held_zero) <=
      .05 * zero_budget_per_second + 2e-7, "Fresh continuity failed bounded zero adaptation");
    held_gain = value.gain.value(); held_zero = value.gain.zero();
    value.capture({NAN, 8.}, 2100);
    value.capture({4., 8.}, 2150);
    require(value.gain.value() == held_gain && value.gain.zero() == held_zero,
      "Invalid measurement interval granted zero catch-up credit");
    value.gain.update(4000);
    require(!value.gain.has_target(), "Zero-gap fixture failed to expire evidence");
    value.capture({4., 8.}, 4050);
    require(value.gain.value() == held_gain && value.gain.zero() == held_zero,
      "Expired evidence granted zero catch-up credit");
    value.capture({4., 8.}, 4100);
    require(value.gain.zero() > held_zero && value.gain.zero() < 6.,
      "Fresh post-expiry evidence failed to resume convergence");
    held_gain = value.gain.value(); held_zero = value.gain.zero();
    value.context = {2, true};
    value.capture({10., 20.}, 4150);
    value.context.reset = false;
    value.capture({10., 20.}, 4200);
    require(value.gain.value() == held_gain && value.gain.zero() == held_zero,
      "SDK reset or first post-reset evidence moved the retained zero");
  }

  void configured_display_budget_has_one_strength_independent_normalization() {
    for (const auto extent : {std::array<unsigned, 2>{3840, 2160}, {1920, 1080}, {3440, 1440}, {1080, 1920}}) {
      float previous_gain = 0.f;
      for (const float strength : {1.f, 50.f, 100.f}) {
        fixture value;
        value.budget = render_limits(strength, extent[0], extent[1]);
        value.gain.configure(value.budget);
        value.start({0., 4., 2.});
        const double separation_uv = 100. * extent[1] / (2160. * extent[0]) *
          sunshine_camera_scene::reference_zpd * value.gain.value() * 4.;
        close(separation_uv, sunshine_game3d::default_disparity_limit_uv, 3e-9,
          "CPU nominal geometry disagrees with the shared full-strength per-eye budget");
        require(!previous_gain || value.gain.value() == previous_gain,
          "Display normalization canceled or compensated artistic strength");
        previous_gain = value.gain.value();
      }
    }
    fixture constrained;
    constrained.budget = {.5, .005, .01, 10.};
    constrained.gain.configure(constrained.budget);
    constrained.start({0., 4., 2.});
    const double span = sunshine_camera_scene::reference_zpd * constrained.gain.value() * 4.;
    close(span, .005, 1e-9, "Custom normalization exceeded the smaller declared displacement capacity");
  }

  void published_float_zero_stays_in_the_representable_observed_range() {
    for (const float far : {.25f, 1.f, 100.f}) {
      const float near = std::nextafter(far, std::numeric_limits<float>::infinity());
      const depth_range range{far, near};
      for (const double strength : {.125, .25, .5, .75, 1.}) {
        fixture value;
        value.budget = {strength, 1., 1.};
        value.gain.configure(value.budget);
        value.start(range);
        require(value.gain.zero() >= far && value.gain.zero() <= near &&
          value.gain.zero() >= value.gain.lower() && value.gain.zero() <= value.gain.upper(),
          "Published FP32 zero lies outside representable observed depth");
        close(value.gain.value(), expected_gain(near), 0., "Retired safety-interval solver still reduces CPU gain");
        const auto applied_gain = value.gain.value(), applied_zero = value.gain.zero();
        value.capture(range, 2000);
        require(value.gain.value() == applied_gain && value.gain.zero() == applied_zero,
          "Repeated near-flat depth continued changing established controls");
      }
    }
  }

  void zero_strength_startup_preserves_feasible_zero_when_enabled() {
    fixture value;
    value.budget.strength = 0.;
    value.gain.configure(value.budget);
    value.start({1., 3.});
    const auto initial_gain = value.gain.value(), held_zero = value.gain.zero();
    close(held_zero, 2., 0., "Zero-strength startup did not retain the initialized screen plane");
    value.capture({2., 4.}, 2000);
    require(value.gain.value() == initial_gain && value.gain.zero() == held_zero,
      "Zero strength changed established gain or convergence");
    value.budget.strength = .5;
    value.gain.configure(value.budget);
    value.gain.update(2001);
    require(value.gain.value() == initial_gain && value.gain.zero() == held_zero,
      "Enabling strength after zero-strength startup recentered an already feasible zero");
    value.placed({2., 4.});
  }

  void startup_uses_maximum_and_midpoint_histories_with_flat_rules() {
    fixture value;
    value.capture({1., 2., 1.01}, 1000); value.capture({2., 3., 2.01}, 1250);
    value.capture({.5, 2., .51}, 1500); value.capture({1., 2.5, 1.01}, 1750);
    close(value.gain.value(), expected_gain(2.375), 0., "Startup did not average four maxima");
    close(value.gain.zero(), 1.75, 0., "Startup did not average four range midpoints independently of means");
    fixture infinity; infinity.start({0., 2.}); infinity.placed({0., 2.});
    fixture flat;
    for (std::uint64_t time = 1000; time <= 3000; time += 250) {
      const double depth = time % 1000 ? .25 : 100.;
      flat.capture({depth, depth}, time);
      require(!flat.gain.initialized() && flat.gain.samples() == 0,
        "Flat clear/title frames seeded scene calibration");
    }
    flat.capture({1., 3.}, 3250); flat.capture({.25, .25}, 3500);
    require(flat.gain.samples() == 0, "Flat title frame retained partial gameplay calibration");
    for (std::uint64_t time = 3750; time <= 4500; time += 250) flat.capture({1., 3.}, time);
    const auto held_gain = flat.gain.value();
    float previous_zero = flat.gain.zero();
    for (std::uint64_t time = 4750; time <= 14750; time += 250) {
      flat.capture({3., 3.}, time);
      require(flat.gain.initialized() && flat.gain.has_target() && flat.gain.value() == held_gain &&
        flat.gain.zero() >= previous_zero && flat.gain.zero() <= 3. &&
        held_gain*(flat.gain.zero()-previous_zero) <= .25 + 2e-7,
        "Flat positive wall changed gain or bypassed bounded zero adaptation");
      close(flat.gain.target_zero(), 3., 0., "Flat positive wall lost its convergence target");
      previous_zero = flat.gain.zero();
    }
    close(flat.gain.zero(), 3., 2e-6, "Flat wall never approached zero disparity");
    const auto held_zero = flat.gain.zero();
    flat.capture({0., 0.}, 15000); flat.gain.update(15250);
    require(!flat.gain.has_target() && flat.gain.value() == held_gain && flat.gain.zero() == held_zero,
      "All-infinity evidence moved established controls");
    fixture no_depth;
    for (std::uint64_t time = 1000; time <= 2000; time += 250) no_depth.capture({0., 0.}, time);
    require(!no_depth.gain.initialized(), "All-infinity evidence fabricated an initial gain");
  }

  void multiplicative_units_preserve_disparity_and_projection() {
    for (const double units : {.001, 100., 1e6}) {
      fixture original, scaled; original.start(); scaled.start({units, 3. * units});
      std::uint64_t time = 2000;
      for (const depth_range range : {depth_range{1.5, 2.5}, {2.5, 4.5}, {0., .5}, {1., 11.}, {5., 7.}}) {
        original.capture(range, time);
        scaled.capture({range.minimum * units, range.maximum * units}, time);
        close(original.gain.zero(), scaled.gain.zero() / units, 3e-6,
          "Changing depth units changed the projected zero plane");
        close(original.gain.value(), scaled.gain.value() * units, 3e-5,
          "Changing depth units changed the live or minimally reduced gain");
        close(.05 * original.gain.value() * (original.gain.zero() - range.maximum),
          .05 * scaled.gain.value() * (scaled.gain.zero() - range.maximum * units), 3e-6,
          "Changing depth units changed scene disparity");
        time += 250;
      }
    }
  }

  void history_invalidity_and_expiry_preserve_applied_values() {
    fixture value; value.start(); value.capture({0., 200., 2.}, 2000);
    const auto zero = value.gain.zero(), gain = value.gain.value();
    const auto old = value.context; ++value.context.revision;
    value.gain.synchronize(value.context);
    value.gain.observe({100., 1000.}, 2100, old); value.gain.update(2100);
    require(!value.gain.has_target() && value.gain.zero() == zero && value.gain.value() == gain,
      "Old feedback revision replaced the held applied controls");
    value.context.reset = true; value.capture({100., 1000.}, 2250);
    require(value.gain.zero() == zero && value.gain.value() == gain,
      "History reset recalibrated or moved the established zero");
    value.context.reset = false; value.capture({0., 200., 2.}, 2500);
    value.gain.update(5000); value.gain.update(5050);
    require(!value.gain.has_target() && value.gain.zero() == zero && value.gain.value() == gain,
      "Expired evidence changed established numerical controls");
    for (const depth_range invalid : {depth_range{-1., 2.}, {2., 1.}, {NAN, 2.}, {1., INFINITY}, {1., 3., NAN}, {1., 3., 4.}}) {
      value.capture(invalid, 5250);
      require(value.gain.zero() == zero && value.gain.value() == gain && !value.gain.has_target(),
        "Malformed range supplied movement or erased established controls");
    }
    value.gain.reset_reference();
    require(!value.gain.initialized() && !value.gain.accepts(old),
      "Explicit recenter retained calibration or lost the revision watermark");
    for (std::uint64_t time = 5500; time <= 6250; time += 250) value.capture({.125, .375}, time);
    require(value.gain.value() == expected_gain(.375) && value.gain.zero() == .25f && !value.gain.limited(),
      "Explicit recenter did not replace gain and zero together");
  }

  void startup_spacing_and_expiry_are_preserved() {
    fixture value;
    for (const auto time : {1000ULL, 1000ULL, 999ULL, 1100ULL}) value.gain.observe({1., 3.}, time, value.context);
    require(value.gain.samples() == 1, "Repeated, regressed or unspaced capture advanced startup");
    value.gain.advance(2500);
    require(!value.gain.initialized() && value.gain.samples() == 0, "Expired partial startup retained its startup means");
    for (std::uint64_t time = 2750; time <= 3500; time += 250) value.capture({.5, 1.5}, time);
    close(value.gain.value(), expected_gain(1.5), 0., "Fresh startup inherited maxima from expired evidence");
  }

  void live_maximum_replaces_startup_with_log_smoothing_and_preserves_strength() {
    fixture near, far; near.start({1., 2.}); far.start({3., 5.});
    float previous_near = near.gain.value(), previous_far = far.gain.value();
    for (std::uint64_t time = 1800; time <= 14000; time += 50) {
      near.capture({1., 3., 2.}, time); far.capture({1., 3., 2.}, time);
      require(near.gain.value() <= previous_near && near.gain.value() >= previous_near / std::exp2(.05) * (1. - 2e-7),
        "Nominal maximum-reference gain decreased faster than one octave per second");
      require(far.gain.value() >= previous_far && far.gain.value() <= previous_far * std::exp2(.05) * (1. + 2e-7),
        "Nominal maximum-reference gain increased faster than one octave per second");
      near.placed({1., 3.}); far.placed({1., 3.});
      previous_near = near.gain.value(); previous_far = far.gain.value();
    }
    require(near.gain.value() == expected_gain(3.) && far.gain.value() == expected_gain(3.),
      "Different initial scenes permanently determined the same gameplay gain");
    for (const double strength : {.25, .5, 1.}) {
      fixture value; value.budget.strength = strength; value.gain.configure(value.budget); value.start();
      close(value.gain.value(), expected_gain(3.), 0., "Maximum normalization canceled the artistic strength slider");
      close(double(sunshine_camera_scene::reference_zpd) * strength * value.gain.value() * 2., double(sunshine_camera_scene::reference_zpd) * expected_gain(3.) * 2. * strength, 2e-9,
        "Steady pairwise disparity is not proportional to artistic strength");
    }
    fixture gradual; gradual.start();
    gradual.capture({1., 3.1, 2.1}, 1800);
    const double alpha = -std::expm1(-.05/sunshine_camera_scene::time_constant_seconds);
    close(gradual.gain.value(), double(expected_gain(3.)) * std::exp(alpha * std::log(double(expected_gain(3.1))/expected_gain(3.))), 5e-8,
      "Small live maximum changes bypassed exponential log-gain smoothing");
    close(gradual.gain.zero(), 2. + alpha * .05, 2e-7, "Small zero-target change bypassed midpoint EMA smoothing");
  }

  void exact_extrema_and_all_grid_mean_have_distinct_roles() {
    std::array<float, 8> grid{0.f, .25f, .25f, .25f, .5f, .5f, .5f, .75f};
    const auto original = grid;
    const auto range = decode_range(grid, true, true, 0.f, 1.f, 0.f, 1.f);
    close(range.mean, .375, 0., "Legacy mean statistics did not include every point, including infinity");
    require(range.minimum == 0. && range.maximum == 1. && grid == original,
      "Statistics clipped raw pixels or replaced authenticated extrema");
    const auto changed = decode_range(grid, true, true, 0.f, .875f, 0.f, 1.f);
    require(changed.mean == range.mean && changed.maximum == .875,
      "Changed exact extrema silently replaced the independent grid mean");
    const auto normal = decode_range(grid, false, false, 0.f, 0.f, 1.f, -1.f);
    close(normal.mean, .625, 0., "Normal depth orientation did not decode every point before averaging");
    require(!decode_range(grid, true, false, 0.f, 1.f, 0.f, 1.f).valid(),
      "Failed full-range evidence silently fell back to diagnostic points");
    grid.front() = NAN;
    require(!decode_range(grid, true, true, 0.f, 1.f, 0.f, 1.f).valid(),
      "Malformed mean input was ignored despite valid exact extrema");
  }

  void depth_statistics_require_an_accepted_current_target() {
    fixture value;
    depth_range measured;
    require(!value.gain.depth_statistics(measured), "Default zeros became measured statistics");
    for (std::uint64_t time = 1000; time <= 1500; time += 250) {
      value.capture({0., 2., .3}, time);
      require(!value.gain.depth_statistics(measured), "Partial startup exposed unaccepted statistics");
    }
    value.capture({0., 2., .3}, 1750);
    require(value.gain.depth_statistics(measured) && measured.minimum == 0. && measured.maximum == 2. && measured.mean == .3,
      "Statistics did not preserve exact accepted mean and valid infinite-far endpoint");
    require(measured.mean != 1. / value.gain.target(), "Fixture failed to distinguish mean from rounded target reciprocal");
    value.gain.observe({1., 4., 3.}, 1800, {0, false});
    require(value.gain.depth_statistics(measured) && measured.mean == .3, "Rejected revision replaced accepted statistics");
    value.capture({1., 4., 3.}, 2000);
    require(value.gain.depth_statistics(measured) && measured.minimum == 1. && measured.maximum == 4. && measured.mean == 3.,
      "Fresh statistics incorrectly followed smoothed applied gain");
    value.capture({3., 3.}, 2250);
    require(value.gain.depth_statistics(measured) && measured.minimum == 3. && measured.maximum == 3. &&
      !value.gain.has_gain_target() && value.gain.target_zero() == 3., "Flat positive convergence lost its fresh statistics");
    value.capture({0., 2., .3}, 2500);
    value.gain.update(4000);
    require(!value.gain.depth_statistics(measured), "Expired target still exposed fresh statistics");
    value.capture({0., 2., .3}, 4250);
    value.capture({NAN, 2.}, 4500);
    require(!value.gain.depth_statistics(measured), "Invalid target retained fresh statistics");
    value.capture({0., 2., .3}, 4750);
    value.context = {2, true};
    value.capture({0., 2., .3}, 5000);
    require(!value.gain.depth_statistics(measured), "Reset evidence supplied fresh statistics");
    value.context.reset = false;
    value.capture({0., 2., .3}, 5250);
    require(value.gain.depth_statistics(measured), "Post-reset accepted target lost statistics");
    value.gain.reset_reference();
    require(!value.gain.depth_statistics(measured), "Explicit recalibration retained statistics");
  }

  void low_strength_startup_does_not_amplify_zero_plane_translation() {
    // Expedition 33 dump 048: the old safety-interval midpoint moved outside
    // this range as strength decreased, retaining about 76.8 pixels/eye even
    // at 1%. Verify the alignment term, not only pairwise depth separation.
    const depth_range scene{1.4917861790308962e-7, .008846094831824303, .0023376106640702827};
    float established_gain = 0, established_zero = 0;
    for (const float percent : {.1f, 1.f, 12.8329296f, 50.f, 100.f}) {
      fixture value;
      value.budget = render_limits(percent, 3840, 2160);
      value.gain.configure(value.budget);
      value.start(scene);
      value.placed(scene);
      if (established_gain == 0) {
        established_gain = value.gain.value(); established_zero = value.gain.zero();
      }
      require(value.gain.value() == established_gain && value.gain.zero() == established_zero,
        "Lower startup strength changed the nominal reference or compensated it with zero-plane movement");
      const double infinity_shift_pixels = 100. * sunshine_camera_scene::reference_zpd *
        value.budget.strength * value.gain.value() * value.gain.zero();
      close(infinity_shift_pixels, 100. * sunshine_camera_scene::reference_zpd *
        percent * .01 * established_gain * established_zero, 1e-6,
        "Low artistic strength retained a large constant whole-image alignment offset");
    }
  }

}

int main() {
  try {
    full_moments_preserve_statistics_while_maximum_controls_gain();
    maximum_reference_is_independent_of_coverage_mean_and_zero();
    midpoint_is_minimax_epsilon_continuous_and_approach_monotone();
    near_spike_has_bounded_zero_and_gain_response();
    recovery_cannot_bank_gap_invalid_or_expired_time();
    strength_changes_hold_geometry_and_normalization_updates_without_recalibration();
    zero_adaptation_cannot_bank_missing_invalid_or_expired_time();
    configured_display_budget_has_one_strength_independent_normalization();
    published_float_zero_stays_in_the_representable_observed_range();
    zero_strength_startup_preserves_feasible_zero_when_enabled();
    startup_uses_maximum_and_midpoint_histories_with_flat_rules();
    multiplicative_units_preserve_disparity_and_projection();
    history_invalidity_and_expiry_preserve_applied_values();
    startup_spacing_and_expiry_are_preserved();
    live_maximum_replaces_startup_with_log_smoothing_and_preserves_strength();
    exact_extrema_and_all_grid_mean_have_distinct_roles();
    depth_statistics_require_an_accepted_current_target();
    low_strength_startup_does_not_amplify_zero_plane_translation();
    std::puts("PASS shared zero plane: maximum gain, midpoint minimax, occupancy/epsilon continuity, approach pop, bounded zero/gain adaptation and history safety");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL shared zero plane: %s\n", error.what()); return 1;
  }
}
