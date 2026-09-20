// SPDX-License-Identifier: GPL-3.0-only
#include "raw_scene_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
  using namespace sunshine_raw_scene;

  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }
  void close(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      std::fprintf(stderr, "%s: actual=%.17g expected=%.17g tolerance=%.17g\n",
        message, actual, expected, tolerance);
      throw std::runtime_error(message);
    }
  }
  float expected_gain(double maximum) {
    const double requested = 1. / maximum;
    float value = static_cast<float>(requested);
    return value > requested ? std::nextafter(value, 0.f) : value;
  }
  selected_frame selected(orientation direction = orientation::reversed) {
    selected_frame value;
    value.basis_epoch = 7;
    value.layout_epoch = 3;
    value.source = {0x100, 9, 0, 1920, 1080, 0, 0, 1920, 1080};
    value.direction = direction;
    value.depth_ready = value.aligned_viewport_assumed = true;
    return value;
  }
  float raw_value(orientation direction, float t) {
    return direction == orientation::normal ? 1.0f - t : t;
  }
  sample scene(const selected_frame &current, std::uint64_t id, std::uint64_t time, float t) {
    sample value;
    value.id = id;
    value.capture_ms = time;
    value.metadata = current;
    value.readback_frame = current.frame;
    value.readback_source = current.source;
    value.readback_layout_epoch = current.layout_epoch;
    for (size_t i = 0; i < value.raw.size(); ++i)
      value.raw[i] = raw_value(current.direction, t * (i % 2 ? 1.5f : .5f));
    return value;
  }
  struct fixture {
    policy controller;
    selected_frame current = selected();
    sample latest;
    output result;
    std::uint64_t now{}, present{}, id{};

    explicit fixture(orientation direction = orientation::reversed) {
      current.direction = direction;
      require(controller.reset(current.basis_epoch, 0), "Initial epoch was refused");
      configure();
    }
    void configure() {
      controller.configure({.5, 1., 1.});
    }
    void next(std::uint64_t time) {
      now = time;
      current.frame = {++present, 17};
    }
    output checked_output(output value) {
      if (value.calibrated) require(std::isnormal(value.H) && value.H > 0.f &&
        std::isfinite(value.t0) && value.t0 >= 0.f,
        "Published gain or independent zero plane is outside the shader domain");
      return result=value;
    }
    output tick(std::uint64_t time) {
      next(time);
      return checked_output(controller.update(current, nullptr, now));
    }
    output inject(std::uint64_t time, const sample &packet) {
      next(time);
      return checked_output(controller.update(current, &packet, now));
    }
    output capture(std::uint64_t time, float t = .25f) {
      next(time);
      latest = scene(current, ++id, time, t);
      return checked_output(controller.update(current, &latest, now));
    }
    output capture_grid(std::uint64_t time, const decltype(sample::raw) &raw) {
      next(time);
      latest = scene(current, ++id, time, .25f);
      latest.raw = raw;
      return checked_output(controller.update(current, &latest, now));
    }
    sample pending(std::uint64_t time, float t = .25f) {
      tick(time);
      return scene(current, ++id, time, t);
    }
    output initialize(float t = .25f, std::uint64_t start = 0) {
      for (unsigned i = 0; i < 4; ++i) {
        capture(start + i * 250, t);
        require(result.ready == (i == 3), "Initialization must be one four-capture, 750 ms window");
      }
      require(result.calibrated, "Ready output lacks initialized numeric state");
      return result;
    }
  };
  void same_output(const output &a, const output &b, const char *message) {
    require(a.ready == b.ready && a.calibrated == b.calibrated && a.reason == b.reason &&
      a.H == b.H && a.t0 == b.t0 && a.normalized_zero == b.normalized_zero &&
      a.calibration_samples == b.calibration_samples, message);
  }

  void full_image_statistics_use_the_captured_basis_not_selector_points() {
    for (const auto direction : {orientation::normal, orientation::reversed}) {
      fixture value(direction);
      const auto capture = [&](std::uint64_t tick, bool wrong_basis = false) {
        value.next(tick);
        auto packet = scene(value.current, ++value.id, tick, .1f);
        packet.raw.fill(raw_value(direction, 0.f)); // Points miss all foreground.
        packet.range_supplied = packet.range_valid = true;
        packet.range_min = direction == orientation::normal ? .5f : 0.f;
        packet.range_max = direction == orientation::normal ? 1.f : .5f;
        auto &m = packet.moments;
        m.supplied = m.valid = true;
        m.A = direction == orientation::normal ? 1.f : 0.f;
        m.inverseB = direction == orientation::normal ? -1.f : 1.f;
        if (wrong_basis) m.inverseB = -m.inverseB;
        m.count = 1920 * 1080;
        m.sum = double(m.count) * .25;
        m.sum_squares = double(m.count) * .125;
        m.tiles_x = 32; m.tiles_y = 18;
        return value.checked_output(value.controller.update(value.current, &packet, tick));
      };
      for (unsigned i = 0; i < 4; ++i) capture(i * 250);
      require(value.result.ready && value.result.depth_statistics.mean == .25 &&
        value.result.depth_statistics.mean_square == .125 && value.result.depth_statistics.pixel_count == 1920 * 1080,
        "Sparse selector points overrode full active-pixel scene statistics");
      require(!capture(1000, true).ready && value.result.reason == status::invalid_depth,
        "A decoded measurement from another depth convention was accepted");
      require(capture(1250).ready, "Fresh matching moments did not restore scene controls");
    }
  }

  void measured_statistics_keep_the_oriented_basis_and_target_lifetime() {
    for (const auto direction : {orientation::normal, orientation::reversed}) {
      fixture value(direction);
      for (unsigned i = 0; i != 4; ++i) {
        value.capture(i * 250, .25f);
        require(value.result.has_depth_statistics == (i == 3), "Raw startup exposed partial statistics");
      }
      const auto &stats = value.result.depth_statistics;
      require(stats.minimum == .125 && stats.maximum == .375 && stats.mean == .25,
        "Raw near/far statistics did not use the same oriented basis as zero");
      value.next(1000);
      auto packet = scene(value.current, ++value.id, 1000, .125f);
      packet.range_supplied = packet.range_valid = true;
      packet.range_min = 0.f; packet.range_max = 1.f;
      value.result = value.controller.update(value.current, &packet, 1000);
      require(value.result.has_depth_statistics && value.result.depth_statistics.minimum == 0. &&
          value.result.depth_statistics.maximum == 1. && value.result.depth_statistics.mean == .125,
        "Raw statistics confused complete extrema with point-grid mean");
      value.current.depth_ready = false;
      require(value.tick(1050).has_depth_statistics, "Missing presentation discarded valid asynchronous statistics");
      value.current.depth_ready = true;
      require(!value.tick(2500).has_depth_statistics && value.result.calibrated,
        "Expired raw target exposed stale statistics or erased established controls");
      value.capture(2750);
      require(value.result.has_depth_statistics, "Fresh raw evidence did not restore statistics");
      value.controller.scene_cut(2800);
      require(!value.tick(2850).has_depth_statistics, "Scene cut retained fresh raw statistics");
      require(value.controller.reset(8, 3000), "Statistics fixture reset failed");
      value.current.basis_epoch = 8;
      require(!value.tick(3000).has_depth_statistics, "Explicit raw reset inherited statistics");
    }
  }

  void clear_center_does_not_reject_valid_off_center_geometry() {
    for (const auto direction : {orientation::normal, orientation::reversed}) {
      decltype(sample::raw) raw;
      raw.fill(raw_value(direction, 0.f));
      raw.front() = raw_value(direction, .125f);
      raw.back() = raw_value(direction, .375f);
      const auto accepted = [](const output &state) {
        require(state.ready && state.has_depth_statistics && state.depth_statistics.minimum == 0. &&
            state.depth_statistics.maximum == .375 && state.depth_statistics.mean > 0.,
          "Clear central samples rejected valid off-center depth or omitted it from statistics");
      };
      // Admission cannot infer missing color/depth correspondence from a clear
      // center. This checks valid depth evidence, not semantic scene matching.
      fixture startup(direction);
      for (unsigned i = 0; i != 4; ++i) {
        const auto state = startup.capture_grid(i * 250, raw);
        require(state.reason != status::clear_depth && state.reason != status::invalid_depth,
          "Finite nonflat depth was declared invalid solely because its center is clear");
      }
      accepted(startup.result);

      fixture initialized(direction);
      initialized.initialize();
      accepted(initialized.capture_grid(1000, raw));
      accepted(initialized.tick(1250));
    }
  }

  void initialization_uses_all_depth_extrema_and_one_window() {
    fixture basic;
    basic.capture(0);
    basic.capture(10);
    basic.capture(20);
    require(!basic.capture(30).ready, "Four captures without the minimum span became ready");
    require(basic.result.calibration_samples == 1, "Rapid captures over-weighted the initial scene");
    basic.capture(250);
    basic.capture(500);
    require(basic.capture(750).ready, "A coherent spanning capture needed a second qualification window");
    close(basic.result.H, expected_gain(.375), 0, "Nonflat span did not initialize full-strength gain");
    close(basic.result.t0, .25, 0, "Nonflat scene did not start at the feasible midpoint");

    fixture late;
    late.tick(6000);
    late.initialize(.125f, 6100);
    close(late.result.H, expected_gain(.1875), 0, "Startup still has a five-second timeout");

    fixture unstable;
    unstable.capture(0, .125f);
    unstable.capture(250, .5f);
    unstable.capture(500, .125f);
    require(unstable.capture(750, .5f).ready, "Moving scene was required to become stationary before calibration");
    close(unstable.result.H, expected_gain(1.5 * .3125), 2e-5,
      "Moving startup did not average all four admitted maxima");

    fixture extremes;
    auto grid = scene(extremes.current, 1, 0, .25f).raw;
    grid[7 * grid_width + 14] = std::numeric_limits<float>::denorm_min();
    grid[7 * grid_width + 15] = std::nextafter(1.0f, 0.0f);
    const auto immutable = grid;
    const double maximum = grid[7 * grid_width + 15];
    for (unsigned i = 0; i < 4; ++i) extremes.capture_grid(i * 250, grid);
    require(extremes.result.ready, "Finite interior extremes were excluded or rejected");
    close(extremes.result.H, expected_gain(maximum), 1e-5,
      "Depth extrema were trimmed, clipped or replaced by an average");
    require(grid == immutable && extremes.latest.raw == immutable, "Reference calculation changed raw input");

    for (const auto direction : {orientation::normal, orientation::reversed}) {
      fixture endpoints(direction);
      auto complete = scene(endpoints.current, 1, 0, .25f).raw;
      complete.front() = 0.f; complete.back() = 1.f;
      for (unsigned i = 0; i < 4; ++i) endpoints.capture_grid(i * 250, complete);
      require(endpoints.result.ready, "Exact zero/one depth endpoints were rejected or treated as missing");
      close(endpoints.result.H, 1., 1e-5,
        "Full-grid endpoints were clipped out of the startup span");
      require(endpoints.latest.raw == complete, "Range calculation changed endpoint depth pixels");
    }

    for (const float bad : {-0.1f, 1.1f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      fixture invalid;
      auto malformed = scene(invalid.current, 1, 0, .25f).raw;
      malformed[7 * grid_width + 14] = bad;
      for (unsigned i = 0; i < 4; ++i) invalid.capture_grid(i * 250, malformed);
      require(!invalid.result.ready && !invalid.result.calibrated,
        "An invalid grid cell was discarded to fabricate a usable range");
      invalid.initialize(.25f, 1000);
    }
  }

  void raw_orientation_and_shader_contract() {
    for (const auto direction : {orientation::normal, orientation::reversed}) {
      fixture f(direction);
      const auto state = f.initialize(.125f);
      close(state.H, expected_gain(.1875), 0, "Orientation changed gain for equivalent raw geometry");
      close(state.t0, .125, 0, "Orientation changed the raw-coordinate zero plane");
      require(state.shader_A == (direction == orientation::normal ? 1.f : 0.f) &&
        state.shader_inverseB == (direction == orientation::normal ? -1.f : 1.f),
        "Shader coefficients do not preserve the oriented raw depth");
      for (const float t : {0.f, .03125f, .125f, .5f, 1.f}) {
        const float converted = (raw_value(direction, t) - state.shader_A) * state.shader_inverseB;
        close(converted, t, 0, "Raw geometry was clamped to a calibration interval");
        const double prepared = 1 / (1 + state.H * static_cast<double>(converted));
        const double reconstructed = state.referenceZPD * (1 + state.H * state.t0 - 1 / prepared);
        close(reconstructed, state.referenceZPD * state.H * (state.t0 - t), 1e-7,
          "Prepared depth no longer represents signed affine displacement");
      }
      close(state.normalized_zero, state.H * state.t0, 1e-7, "Normalized-zero diagnostic is inconsistent");
    }
  }

  void flat_depth_cannot_seed_gain_but_tracks_established_zero() {
    fixture value;
    decltype(sample::raw) raw;
    for (std::uint64_t time = 0; time <= 1000; time += 250) {
      raw.fill(time % 500 ? .25f : 1.f);
      const auto result = value.capture_grid(time, raw);
      require(!result.ready && !result.calibrated && result.calibration_samples == 0,
        "Flat clear/title raw depth seeded automatic calibration");
    }
    value.initialize(.25f, 1250);
    value.capture(2250, .125f);
    const auto held = value.result;
    raw.fill(.5f);
    const auto flat = value.capture_grid(2500, raw);
    require(flat.ready && flat.reason == status::ready && flat.H == held.H && flat.t0 > held.t0 &&
      flat.t0 < .5f && flat.target_t0 == .5 && flat.target_H == 0. && flat.has_depth_statistics &&
      flat.H*(flat.t0-held.t0) <= .25 + 2e-7,
      "Initialized flat raw surface changed gain or failed bounded convergence");
    const auto later = value.tick(2750);
    require(later.ready && later.H == held.H && later.t0 > flat.t0 && later.t0 < .5f,
      "Flat positive raw surface pursued old gain or stopped converging");
    raw.fill(0.f);
    const auto infinite = value.capture_grid(3000, raw);
    require(infinite.ready && infinite.reason == status::holding_reference && infinite.H == later.H &&
      infinite.t0 == later.t0 && !infinite.has_depth_statistics,
      "All-infinity raw depth moved established controls");
  }

  void feasible_composition_holds_gain_and_zero() {
    fixture f;
    const auto initial = f.initialize();
    for (std::uint64_t now = 800; now <= 3000; now += 50) {
      const bool fresh = now % 250 == 0;
      if (fresh) {
        decltype(sample::raw) raw;
        raw.fill(.25f);
        const auto position = (now / 250) % raw.size();
        raw[position] = .125f;
        raw[(position + raw.size() / 2) % raw.size()] = .375f;
        f.capture_grid(now, raw);
      } else f.tick(now);
      require(f.result.ready, "Continuous valid scene became unready");
      require(f.result.H == initial.H && f.result.t0 == initial.t0,
        "Feasible scene motion changed unchanged-span gain or recentered the plane");
      close(f.result.target_H, initial.H, 0, "Feasible scene replaced the requested gain");
    }
  }

  void room_wall_room_preserves_pairwise_depth() {
    fixture f; f.initialize();
    for (const float depth : {.125f, .5f, .25f}) {
      const auto end = f.now + 12000;
      for (auto time = f.now + 50; time <= end; time += 50) {
        const float previous = f.result.H;
        const auto next = f.capture(time, depth);
        require(next.ready && next.H <= previous * std::exp2(.05) * (1. + 2e-7),
          "Live raw range increased gain faster than twice per second");
      }
      close(f.result.H, expected_gain(1.5 * depth), 1e-6, "Raw range retained the startup gain instead of learning the current scene");
      close(f.result.referenceZPD * f.result.H * ((f.result.t0 - .125) - (f.result.t0 - .5)),
        double(timing::reference_zpd) * f.result.H * .375, 1e-7,
        "Minimum-change zero shifted pairwise inverse-depth separation");
    }
  }

  void full_extrema_drive_target_without_safety_zero_projection() {
    fixture f;
    f.controller.configure({.5, 1., 1.});
    const auto capture_range = [&](std::uint64_t time, float minimum, float maximum) {
      f.next(time);
      auto exact = scene(f.current, ++f.id, time, .25f);
      exact.range_supplied = exact.range_valid = true;
      exact.range_min = minimum; exact.range_max = maximum;
      exact.raw.fill((minimum+maximum)*.5f);
      return f.checked_output(f.controller.update(f.current, &exact, time));
    };
    for (unsigned i = 0; i < 4; ++i) capture_range(i * 250, .225f, .275f);
    const auto initial = f.result;
    close(initial.H, expected_gain(.275f), 0., "Full-range initialization ignored the exact maximum");
    const auto spike = capture_range(800, .25f, .9f);
    require(spike.ready && spike.H < initial.H && spike.H >= initial.H / std::exp2(.05) * (1.-2e-7),
      "One raw maximum bypassed bounded gain adaptation");
    require(spike.t0 > initial.t0 && spike.t0 < spike.target_t0 &&
      spike.H*(spike.t0-initial.t0) <= .05 + 2e-7,
      "A new near depth bypassed bounded zero adaptation");
    close(spike.target_t0, (.25 + double(.9f))*.5, 0., "Raw zero target did not use full extrema");
    close(spike.target_H, expected_gain(.9f), 0., "Point-grid mean replaced the full-range maximum target");
    float previous = spike.H, previous_zero = spike.t0;
    for (std::uint64_t time = 850; time <= 10000; time += 50) {
      const auto narrower = capture_range(time, .225f, .275f);
      require(narrower.H >= previous && narrower.H <= initial.H &&
        narrower.H <= previous * std::exp2(.05) * (1. + 2e-7),
        "Near-point exit exceeded automatic gain bounds");
      require(narrower.t0 <= previous_zero && narrower.t0 >= initial.t0 &&
        narrower.H*(previous_zero-narrower.t0) <= .05 + 2e-7,
        "Near-point exit exceeded zero adaptation bounds");
      previous = narrower.H; previous_zero = narrower.t0;
    }
    close(f.result.H, initial.H, 0., "Raw depth did not automatically recover after point exit");
    close(f.result.t0, initial.t0, 2e-6, "Transient point left permanent raw zero drift");
    const auto before_projection = f.result;
    const auto projected = capture_range(10250, .475f, .525f);
    require(projected.t0 > before_projection.t0 && projected.t0 < .475f &&
      projected.H*(projected.t0-before_projection.t0) <= .25 + 2e-7,
      "Translated range instantly clamped zero instead of bounded adaptation");
    close(projected.target_t0, (double(.475f)+double(.525f))*.5, 0., "Translated zero target is not its midpoint");
    require(projected.H < initial.H && projected.H >= initial.H / std::exp2(.25) * (1.-2e-7),
      "Translated maximum bypassed the gain rate limit");
    auto failed = f.pending(10500, .5f);
    failed.range_supplied = true; failed.range_valid = false;
    require(!f.inject(10510, failed).ready, "Failed exact range silently used the sparse diagnostic grid");
  }

  void frame_quantized_capture_cadence_keeps_unchanged_range() {
    for (const std::uint64_t interval : {267ULL, 333ULL, 500ULL}) {
      fixture f;
      f.initialize();
      for (unsigned i = 1; i <= 20; ++i) {
        const std::uint64_t capture_at = 750 + i * interval;
        // Presentation remains responsive while GPU capture completes slightly
        // after its nominal 250 ms interval, or at the allowed 500 ms boundary.
        while (f.now + 50 < capture_at) f.tick(f.now + 50);
        const auto next = f.capture(capture_at, .25f);
        require(next.ready && next.H == expected_gain(.375) && next.t0 == .25f && next.target_H == expected_gain(.375),
          "Frame-quantized readback cadence lost readiness or unchanged range target");
      }
      require(f.result.t0 == .25f, "Frame-quantized captures moved a feasible plane");
    }

    fixture gap;
    gap.initialize();
    while (gap.now + 50 < 1251) gap.tick(gap.now + 50);
    const auto first=gap.capture(1251, .25f);
    require(first.ready && first.H == expected_gain(.375) && first.t0 == .25f, "Sparse capture changed unchanged-range geometry");
    const auto resumed = gap.capture(1301, .25f);
    require(resumed.ready && resumed.H == expected_gain(.375) && resumed.t0 == .25f,
      "Fresh continuity moved feasible depth or changed the reference");
  }

  void ignored_packets_are_identical_to_absent_packets() {
    for (unsigned kind = 0; kind < 9; ++kind) {
      fixture a;
      a.initialize();
      a.capture(1000, .5f);
      fixture b = a;
      auto packet = a.latest;
      if (kind != 0) {
        packet.id = 100000;
        packet.capture_ms = 1050;
        packet.metadata.frame = {a.present + 1, 17};
        packet.readback_frame = packet.metadata.frame;
      }
      switch (kind) {
      case 1: ++packet.metadata.source.native; packet.readback_source = packet.metadata.source; break;
      case 2: ++packet.metadata.basis_epoch; break;
      case 3: ++packet.metadata.layout_epoch; packet.readback_layout_epoch = packet.metadata.layout_epoch; break;
      case 4: packet.capture_ms = 100000; break;
      case 5: packet.capture_ms = 1; break;
      case 6: packet.metadata.frame.frame = 1; packet.readback_frame = packet.metadata.frame; break;
      case 7: packet.metadata.direction = orientation::normal; break;
      case 8: ++packet.metadata.convention_epoch; break;
      }
      for (std::uint64_t time = 1050; time <= 1100; time += 50)
        same_output(a.inject(time, packet), b.tick(time),
          "Duplicate, old or wrong-basis readback differed from an absent packet");
      same_output(a.capture(1250, .4f), b.capture(1250, .4f),
        "Ignored high-ID readback poisoned later ordinary sample admission");
    }

    fixture repeated;
    repeated.initialize();
    const auto duplicate = repeated.latest;
    for (std::uint64_t time = 800; time < 2250; time += 50)
      require(repeated.inject(time, duplicate).ready, "A duplicate removed a still-fresh target");
    require(repeated.inject(2250, duplicate).ready && repeated.result.reason == status::holding_reference,
      "Replay refreshed target age or disabled an unchanged initialized reference");
  }

  void expired_comfort_target_holds_only_a_valid_current_reference() {
    fixture f;
    f.initialize();
    f.capture(1000, .25f);
    const auto duplicate = f.latest;
    for (std::uint64_t time = 1050; time < 2500; time += 50) f.tick(time);
    const auto held = f.result;
    for (std::uint64_t time = 2500; time <= 10000; time += 50) {
      const auto result = f.inject(time, duplicate);
      require(result.ready && result.reason == status::holding_reference,
        "Ordinary measurement expiry disabled valid current depth");
      require(result.H == held.H && result.t0 == held.t0,
        "Held reference moved using expired or replayed evidence");
    }
    auto wrong_generation=f.current;
    wrong_generation.frame={++f.present,18};
    require(!f.controller.evaluate(wrong_generation,10001).ready,"Expired target admitted a new capture generation");
    f.current.depth_ready = false;
    require(!f.tick(10010).ready, "Held numerical state made missing pixels current");
    f.current.depth_ready = true;
    f.current.copy_ambiguous = true;
    require(!f.tick(10020).ready, "Held reference admitted ambiguous current pixels");
    f.current.copy_ambiguous = false;
    require(f.tick(10030).reason == status::holding_reference,
      "Short copy absence destroyed initialized numerical history");
    const auto resumed = f.capture(10050, .25f);
    require(resumed.ready && resumed.reason == status::ready && resumed.H == held.H && resumed.t0 == held.t0,
      "Fresh target return accumulated catch-up movement");
    const auto adapting = f.capture(10300, .25f);
    require(adapting.ready && adapting.H == held.H && adapting.target_H == held.target_H && adapting.t0 == held.t0,
      "Fresh observations altered unchanged-range geometry despite a feasible zero");

    fixture cut = f;
    cut.controller.scene_cut(10400);
    require(!cut.tick(10400).ready && !cut.tick(13000).ready,
      "Explicit cut was mistaken for a harmless target timeout");
    fixture malformed = f;
    auto bad = malformed.pending(10400, .25f);
    bad.raw[7 * grid_width + 14] = -1.f;
    require(!malformed.inject(10410, bad).ready && !malformed.tick(13000).ready,
      "Malformed new evidence was mistaken for a harmless target timeout");
    fixture replaced = f;
    ++replaced.current.source.lifetime;
    require(!replaced.tick(13000).ready, "A new physical source reused a held reference");
  }

  void asynchronous_packet_identity_and_malformed_recovery() {
    fixture delayed;
    delayed.initialize();
    const auto submitted = delayed.pending(800, .25f);
    delayed.tick(850);
    require(delayed.inject(900, submitted).ready, "Legitimate asynchronous readback older than current was rejected");
    require(delayed.result.H == expected_gain(.375) && delayed.result.target_H == expected_gain(.375) && delayed.result.t0 == .25f,
      "Admitted asynchronous readback altered feasible unchanged-range geometry");

    for (unsigned kind = 0; kind < 11; ++kind) {
      fixture f;
      f.initialize();
      auto bad = f.pending(800, .25f);
      switch (kind) {
      case 0: bad.raw[7 * grid_width + 14] = std::numeric_limits<float>::quiet_NaN(); break;
      case 1: bad.range_supplied = true; bad.range_valid = false; break;
      case 2: bad.range_supplied = bad.range_valid = true; bad.range_min = .5f; bad.range_max = .25f; break;
      case 3: bad.raw[7 * grid_width + 14] = -.1f; break;
      case 4: ++bad.readback_frame.token_generation; break;
      case 5: ++bad.readback_source.lifetime; break;
      case 6: ++bad.readback_layout_epoch; break;
      case 7: bad.raw[7 * grid_width + 14] = std::numeric_limits<float>::infinity(); break;
      case 8: bad.metadata.depth_ready = false; break;
      case 9: bad.metadata.copy_ambiguous = true; break;
      case 10: bad.metadata.aligned_viewport_assumed = false; break;
      }
      const auto previous = f.result;
      const auto rejected = f.inject(850, bad);
      require(!rejected.ready && rejected.calibrated && rejected.H == previous.H && rejected.t0 == previous.t0,
        "Malformed new readback moved numeric state or stayed ready");
      auto absent = f;
      same_output(f.inject(900, bad), absent.tick(900), "Repeated immutable malformed ID was rejected twice");
      const auto restored = f.capture(950, .25f);
      require(restored.ready && restored.H == previous.H && restored.t0 == previous.t0,
        "Same-source valid recovery restarted calibration or accumulated gap motion");
      const auto adapting = f.capture(1000, .25f);
      require(adapting.ready && adapting.H == previous.H && adapting.target_H == previous.target_H && adapting.t0 == previous.t0,
        "Fresh recovery changed gain or moved an already feasible zero");
    }
  }

  void missing_evidence_and_cadence_gaps_do_not_earn_motion() {
    for (unsigned kind = 0; kind < 5; ++kind) {
      fixture f;
      f.initialize();
      f.capture(1000, .25f);
      const auto held = f.result;
      const auto original = f.current;
      if (kind == 0) f.current.depth_ready = false;
      if (kind == 1) f.current.source.native = 0;
      if (kind == 2) f.current.copy_ambiguous = true;
      if (kind == 3) f.current.aligned_viewport_assumed = false;
      const auto stopped = f.tick(kind == 4 ? 5000 : 1050);
      require(stopped.ready == (kind == 4) && stopped.calibrated && stopped.H == held.H && stopped.t0 == held.t0,
        "Missing current input or an expired zero target lost its distinct readiness semantics");
      if (kind == 4) require(stopped.reason == status::holding_reference,
        "A long observation gap silently refreshed the zero target");
      f.current = original;
      const std::uint64_t resumed_at = kind == 4 ? 5050 : 1100;
      const auto resumed = f.capture(resumed_at, .25f);
      require(resumed.ready && resumed.t0 == held.t0,
        "First recovered current frame earned zero movement across a gap");
      close(resumed.H, held.H, 0, "Recovery changed the established reference");
      const auto adapting = f.capture(resumed_at + 50, .25f);
      require(adapting.ready && adapting.H == held.H && adapting.target_H == held.target_H && adapting.t0 == held.t0,
        "Recovery moved feasible depth or changed the unchanged range target");
    }

    fixture short_gap;
    short_gap.initialize();
    short_gap.capture(1000, .25f);
    const auto held = short_gap.result;
    const auto gap_return=short_gap.tick(1301);
    require(gap_return.H==held.H && gap_return.t0==held.t0,"Presentation gap banked motion");
    const auto first = short_gap.capture(1350, .25f);
    require(first.H == held.H && first.t0 == held.t0, "Fresh post-gap continuity moved a feasible plane");
    const auto recent = short_gap.capture(1400, .25f);
    require(recent.ready && recent.H == held.H && recent.t0 == first.t0,
      "Presentation recovery changed the unchanged range target or feasible zero");

    fixture sparse;
    sparse.initialize();
    for (std::uint64_t time = 800; time <= 1400; time += 50) sparse.tick(time);
    const auto first_sparse=sparse.capture(1450,.25f);
    require(first_sparse.H == expected_gain(.375) && first_sparse.t0 == .25f, "Sparse captures changed unchanged-range geometry");
    const auto second = sparse.capture(1500, .25f);
    require(second.ready && second.H == expected_gain(.375) && second.target_H == expected_gain(.375) && second.t0 == .25f,
      "Second fresh capture changed the unchanged range target or feasible zero");
  }

  void missing_frames_preserve_asynchronous_evidence_but_strict_errors_do_not() {
    fixture startup;
    // Every actual sample takes 20 ms to complete, crossing a missing present.
    // A changing range must still initialize from four authenticated captures.
    for (unsigned i = 0; i < 4; ++i) {
      const auto time = i * 250ULL;
      const auto packet = startup.pending(time, i % 2 ? .5f : .125f);
      selected_frame no_depth;
      no_depth.basis_epoch = startup.current.basis_epoch;
      const auto missing = startup.controller.update(no_depth, nullptr, time + 10);
      require(!missing.ready && missing.reason == status::depth_unavailable,
        "Missing current pixels were rendered using a cached frame");
      const auto received = startup.inject(time + 20, packet);
      require(received.calibration_samples == i + 1 && received.ready == (i == 3),
        "A short missing frame discarded a valid in-flight startup observation");
    }
    close(startup.result.H, expected_gain(1.5 * .3125), 2e-5,
      "Moving asynchronous startup lost an admitted maximum");
    const auto held = startup.result;
    selected_frame missing;
    missing.basis_epoch = startup.current.basis_epoch;
    require(!startup.controller.update(missing, nullptr, 780).ready, "Missing current frame retained stereo");
    const auto returned = startup.tick(790);
    require(returned.ready && returned.H == held.H && returned.t0 == held.t0,
      "Fresh current pixels could not reuse still-fresh numerical evidence without catch-up");

    for (unsigned kind = 0; kind < 4; ++kind) {
      fixture f;
      f.initialize();
      const auto queued = f.pending(800, .5f);
      f.tick(805);
      auto bad = f.current;
      std::uint64_t failed_at = 810;
      if (kind == 0) bad.frame.frame = 0;
      if (kind == 1) --bad.frame.frame;
      if (kind == 2) failed_at = 804;
      if (kind == 3) bad.source.native = 0;
      require(!f.controller.update(bad, nullptr, failed_at).ready, "Strict provenance error remained ready");
      require(f.inject(850, queued).ready == (kind != 2),
        "Presentation rejection erased authenticated evidence, or a clock rollback admitted old evidence");
      require(f.capture(900, .5f).ready, "Strict provenance failure could not recover with new evidence");
    }

    fixture expired;
    expired.capture(0);
    expired.capture(250);
    expired.capture(500);
    expired.tick(1600);
    require(expired.result.calibration_samples == 0, "Expired startup observations accumulated forever");
    expired.initialize(.25f, 1700);
  }

  void source_changes_have_one_fresh_initialization_without_cache() {
    for (unsigned kind = 0; kind < 7; ++kind) {
      fixture f;
      f.initialize();
      const auto old_source = f.current;
      const auto old_packet = f.latest;
      switch (kind) {
      case 0: ++f.current.source.native; break;
      case 1: ++f.current.source.lifetime; break;
      case 2: ++f.current.layout_epoch; break;
      case 3: ++f.current.convention_epoch; break;
      case 4: f.current.direction = orientation::normal; break;
      case 5: ++f.current.source.left; --f.current.source.extent_width; break;
      case 6: ++f.current.source.viewport; break;
      }
      f.current.depth_ready = false;
      require(!f.tick(800).calibrated, "Changed valid identity kept the previous basis while capture was unavailable");
      f.current.depth_ready = true;
      require(!f.inject(850, old_packet).ready, "Old source readback initialized a newly selected source");
      for (unsigned i = 0; i < 4; ++i)
        require(f.capture(900 + i * 250, .125f).ready == (i == 3),
          "New source has no single fresh 750 ms initialization");
      close(f.result.H, expected_gain(.1875), 0, "New source inherited an old raw-depth basis");
      f.current = old_source;
      require(!f.tick(1700).calibrated, "A to B to A silently restored a cached gain");
      require(!f.inject(1750, old_packet).ready && f.result.calibration_samples == 0,
        "Returning source cleared readback replay watermarks");
      for (unsigned i = 0; i < 4; ++i)
        require(f.capture(1800 + i * 250, .5f).ready == (i == 3),
          "Returning source bypassed or doubled fresh initialization");
      close(f.result.H, expected_gain(.75), 0, "Returning source reused cached scene calibration");
    }

    fixture floor;
    floor.initialize();
    ++floor.current.source.native;
    auto preselected = scene(floor.current, ++floor.id, 775, .125f);
    floor.tick(800);
    require(!floor.inject(850, preselected).ready && floor.result.calibration_samples == 0,
      "New-source packet captured before selection entered its initialization history");
    floor.initialize(.125f, 900);

    fixture same;
    same.initialize();
    for (std::uint64_t time = 800; time <= 1200; time += 50)
      require(same.capture(time).ready && same.result.H == expected_gain(.375) && same.result.t0 == .25f,
        "Repeated same-source selection (pin/unpin) reset numeric state");
  }

  void current_order_reset_and_cut_guards() {
    fixture f;
    f.initialize();
    const auto original = f.result;
    require(!f.controller.reset(0, 800) && !f.controller.reset(7, 800) && !f.controller.reset(6, 800),
      "Same, zero or older explicit epoch reset state");
    require(f.capture(800).ready && f.result.H == original.H,
      "Rejected explicit reset damaged current calibration");
    auto older = f.current;
    --older.frame.frame;
    require(!f.controller.update(older, nullptr, 850).ready, "Backward current-frame sequence was accepted");
    require(f.capture(900).ready, "Current-order rejection could not recover with fresh evidence");
    auto generation_collision = f.current;
    ++generation_collision.frame.token_generation;
    require(!f.controller.update(generation_collision, nullptr, 950).ready,
      "Equal frame number with a different capture generation was accepted");
    require(f.capture(1000).ready, "Capture-generation failure did not recover");
    require(!f.controller.update(f.current, nullptr, 999).ready, "Backward wall time was accepted");
    require(f.capture(1050).ready, "Monotonic clock recovery failed");

    const auto queued = f.pending(1100, .25f);
    f.controller.scene_cut(1150);
    const auto held = f.result;
    require(!f.inject(1200, queued).ready, "Pre-cut queued readback revived a discarded target");
    const auto resumed = f.capture(1250, .25f);
    require(resumed.ready && resumed.H == held.H && resumed.t0 == held.t0,
      "Scene cut recalibrated gain or accumulated zero-plane movement");

    const auto old_packet = f.latest;
    require(f.controller.reset(8, 1300), "New explicit epoch was refused");
    f.current.basis_epoch = 8;
    f.configure();
    require(!f.inject(1350, old_packet).ready, "Queued old-epoch packet survived explicit recalibration");
    f.initialize(.125f, 1400);
    close(f.result.H, expected_gain(.1875), 0, "Explicit new epoch retained the old gain");
  }

  void large_gain_uses_r32_without_clamping_raw() {
    fixture too_large;
    for(unsigned i=0;i<4;++i)too_large.capture(i*250,1e-8f);
    require(too_large.result.ready && too_large.result.calibrated,"Finite large scale was rejected by a retired FP16 limit");
    for(std::uint64_t time=800;time<=15000;time+=50)too_large.capture(time,.25f);
    require(too_large.result.ready,"Unsupported startup could not recover without reset");

    fixture near_limit;
    const auto near_initial = near_limit.initialize(1.f / 16380);
    for(std::uint64_t time=800;time<=4000;time+=50) {
      const auto next=near_limit.capture(time,1e-8f);
      require(next.ready,"Finite scale stopped rendering after crossing the retired FP16 boundary");
      require(next.H >= near_initial.H && next.t0 >= 0.f,
        "Live range failed to increase finite gain or crossed zero");
      require(near_limit.latest.raw[7*grid_width+14]==.5e-8f,"Shader limits rewrote raw input");
    }
    for(std::uint64_t time=4050;time<=15000;time+=50)near_limit.capture(time,.25f);
    require(near_limit.result.ready,"Unsupported shader pair trapped later valid target");
  }

}

int main() {
  try {
    full_image_statistics_use_the_captured_basis_not_selector_points();
    initialization_uses_all_depth_extrema_and_one_window();
    measured_statistics_keep_the_oriented_basis_and_target_lifetime();
    clear_center_does_not_reject_valid_off_center_geometry();
    raw_orientation_and_shader_contract();
    flat_depth_cannot_seed_gain_but_tracks_established_zero();
    feasible_composition_holds_gain_and_zero();
    room_wall_room_preserves_pairwise_depth();
    full_extrema_drive_target_without_safety_zero_projection();
    frame_quantized_capture_cadence_keeps_unchanged_range();
    ignored_packets_are_identical_to_absent_packets();
    expired_comfort_target_holds_only_a_valid_current_reference();
    asynchronous_packet_identity_and_malformed_recovery();
    missing_evidence_and_cadence_gaps_do_not_earn_motion();
    missing_frames_preserve_asynchronous_evidence_but_strict_errors_do_not();
    source_changes_have_one_fresh_initialization_without_cache();
    current_order_reset_and_cut_guards();
    large_gain_uses_r32_without_clamping_raw();
    std::puts("PASS raw scene policy: maximum gain, bounded midpoint convergence, flat-wall behavior and source/async/replay guards");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL raw scene policy: %s\n", error.what());
    return 1;
  }
}
