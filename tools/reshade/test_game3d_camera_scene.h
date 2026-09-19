// SPDX-License-Identifier: GPL-3.0-only
// Feeds the CPU scene policy into the real, separately supplied Depth3D effect.
// Synthetic registration is ground truth owned by this fixture, not middleware proof.
#pragma once
#include "camera_scene_policy.h"

namespace sunshine_camera_fixture {
  template<class Fixture, class Plane, class Capture, class Prepared, class Metadata, class Ready>
  void run_scene_policy(Fixture &f, unsigned width, unsigned height, Plane plane, Capture capture,
      Prepared prepared, Metadata metadata, Ready ready, std::ostream &log, const std::filesystem::path &directory) {
    using sunshine_parity::need;
    namespace scene = sunshine_camera_scene;
    auto registration = [&](double unit, std::uint64_t frame) {
      scene::registration value;
      value.unit_epoch = 1;
      value.frame = {frame, frame};
      value.source = {100, 1, 0, width, height, 0, 0, width, height};
      value.A = 0;
      value.B = unit;
      value.proof_admitted = true;
      return value;
    };
    auto sample = [&](const scene::registration &current, std::uint64_t id, std::uint64_t time, double q) {
      scene::sample value;
      value.id = id;
      value.capture_ms = time;
      value.metadata = current;
      value.readback_frame = current.frame;
      value.readback_source = current.source;
      value.raw.fill(float(current.A + current.B * q));
      return value;
    };
    auto initialize = [&](scene::policy &policy, double unit) {
      policy.reset(1, 1000);
      scene::output result;
      for (unsigned i = 0; i < 4; ++i) {
        auto current = registration(unit, i + 1);
        policy.observe(sample(current, i + 1, 1000 + 250 * i, 1 / (128 * unit)), 1000 + 250 * i);
        result = policy.tick(current, 1000 + 250 * i);
        need(result.calibrated == (i == 3), "Scene policy accepted an incomplete startup reference");
      }
      need(result.ready && result.calibrated, "Known coherent scene did not initialize the automatic scale");
      need(std::abs(result.K / unit - 128) < .002 && std::abs(result.referenceZPD - .05f) < 1e-7,
        "Automatic reference differs from the known startup distance");
      return result;
    };
    auto apply = [&](const scene::output &value) {
      metadata(value.A, value.inverseB, value.K, value.referenceZPD, value.q0);
      ready(value.ready);
    };
    scene::policy policy;
    auto current = registration(1, 4);
    auto state = initialize(policy, 1);
    const float fixed_scale = state.K, fixed_reference = state.referenceZPD;
    apply(state);
    plane(1.f / 128);
    f.set_float("Depth_Adjustment", 0);
    const auto mono = capture();
    f.set_float("Depth_Adjustment", 100);

    struct rendered { std::vector<std::uint8_t> pixels; std::array<float, 2> channels, shifts; float disparity; };
    auto render = [&](const scene::output &value, double q, const char *label) {
      need(value.ready, "Scene fixture tried to render unadmitted camera geometry");
      apply(value);
      plane(float(value.A + q / value.inverseB));
      const auto pixels = capture();
      const auto channels = prepared();
      const auto shifts = f.measure_eye_shifts(mono, pixels, label);
      const double field = value.referenceZPD * value.K * (value.q0 - q);
      const double depth = 1 / (1 + value.K * q);
      need(std::abs(channels[0] - field) < .002 && std::abs(channels[1] - depth) < .002,
        "Automatic policy changed the real shader's inverse-depth gain/preparation");
      const float disparity = shifts[0] - shifts[1];
      log << "scene_render label=" << label << " K=" << value.K << " reference=" << value.referenceZPD
          << " q0=" << value.q0 << " q=" << q << " field=" << channels[0] << " depth=" << channels[1]
          << " left=" << shifts[0] << " right=" << shifts[1] << " disparity=" << disparity << '\n';
      return rendered{pixels, channels, shifts, disparity};
    };
    const auto on_screen = [](const rendered &value) {
      return std::abs(value.disparity) <= .5f && std::abs(value.shifts[0]) <= .5f && std::abs(value.shifts[1]) <= .5f;
    };
    const auto initial_zero = render(state, state.q0, "scene-startup-screen-plane");
    need(on_screen(initial_zero), "Automatic startup does not put its reference on the screen plane");
    const auto reference_image = render(state, 1. / 64, "scene-initial-near-surface");

    // Different world-unit conventions must produce the same final eyes. Each
    // policy initializes itself; neither test copies the other policy's scale.
    for (double unit : {100., .001}) {
      scene::policy scaled;
      const auto scaled_state = initialize(scaled, unit);
      const auto actual = render(scaled_state, 1 / (64 * unit), "scene-equivalent-world-units");
      need(f.image_difference(reference_image.pixels, actual.pixels) < .01f &&
          std::abs(actual.disparity - reference_image.disparity) <= .5f,
        "Changing game world units altered the automatically calibrated stereo result");
      log << "scene_units scale=" << unit << " image_difference="
          << f.image_difference(reference_image.pixels, actual.pixels) << '\n';
    }

    // A new clipping near plane changes raw representation, not geometry scale.
    current = registration(.5, 5);
    state = policy.tick(current, 1800);
    need(state.ready && state.K == fixed_scale, "Clipping-plane change recalibrated the scene scale");
    const auto new_near = render(state, 1. / 64, "scene-new-clipping-near");
    need(f.image_difference(reference_image.pixels, new_near.pixels) < .01f,
      "Equivalent geometry with a changed clipping plane changed exported pixels");

    // An extreme foreground outside the fixed reference patch is deliberately
    // retained in the input grid. It must not resize the scene or move q0.
    current = registration(1, 6);
    auto outside = sample(current, 5, 2000, 1. / 128);
    for (unsigned y = 0; y < 18; ++y)
      for (unsigned x = 0; x < 32; ++x)
        if (x < 14 || x >= 18 || y < 7 || y >= 11) outside.raw[y * 32 + x] = .75f;
    policy.observe(outside, 2000);
    state = policy.tick(current, 2000);
    need(state.K == fixed_scale && std::abs(state.q0 - 1.f / 128) < 1e-7,
      "An object outside the reference patch changed the scene scale or zero plane");
    const auto outside_result = render(state, 1. / 64, "scene-outside-reference-foreground");
    need(outside_result.pixels == reference_image.pixels,
      "Unrelated depth distribution changed the same surface's actual stereo image");

    // Walk 64 scene units toward two fixed world surfaces. Adapt only q0; the
    // inverse distances change with walking and should change pairwise disparity.
    float first_span = 0, last_span = 0;
    for (unsigned step = 0; step <= 4; ++step) {
      const double travel = 16. * step;
      const auto time = std::uint64_t(2250 + step * 250);
      current = registration(1, 10 + step);
      policy.observe(sample(current, 6 + step, time, 1 / (128 - travel)), time);
      // Render ticks are distinct from low-rate asynchronous depth samples.
      for (unsigned tick = 0; tick < 5; ++tick) {
        state = policy.tick(current, time + 50 * tick);
        need(state.ready && state.K == fixed_scale && state.referenceZPD == fixed_reference,
          "Walking silently recalibrated depth strength or lost coherent input");
        log << "scene_tick time=" << time + 50 * tick << " normalized_zero=" << state.normalized_zero << '\n';
      }
      const auto near_surface = render(state, 1 / (128 - travel), "scene-walking-near");
      const auto far_surface = render(state, 1 / (256 - travel), "scene-walking-far");
      const auto zero_surface = render(state, state.q0, "scene-walking-screen-plane");
      need(on_screen(zero_surface), "Adaptive zero plane drifted from the physical screen plane");
      const double expected_difference = fixed_reference * fixed_scale * (1 / (128 - travel) - 1 / (256 - travel));
      need(std::abs((far_surface.channels[0] - near_surface.channels[0]) - expected_difference) < .002,
        "Walking changed pairwise field gain instead of only geometry/zero plane");
      last_span = far_surface.disparity - near_surface.disparity;
      if (!step) first_span = last_span;
    }
    need(last_span > first_span + .5f, "Approaching fixed world surfaces did not increase their actual stereo separation");

    // A missing frame proof and a later cut retain calibration. Neither event
    // grants accumulated zero-plane movement when valid observations resume.
    const double prior_zero = state.normalized_zero;
    current.proof_admitted = false;
    auto missing = policy.tick(current, 3500);
    need(!missing.ready && missing.calibrated && missing.K == fixed_scale,
      "Missing registration lost the stable scale or remained ready");
    policy.scene_cut(10000);
    current = registration(1, 100);
    policy.observe(sample(current, 20, 10001, 1. / 256), 10001);
    state = policy.tick(current, 10001);
    need(state.calibrated && state.K == fixed_scale && std::abs(state.normalized_zero - prior_zero) < 1e-8,
      "Reacquisition after a cut recalibrated or spent accumulated zero-plane movement");
    state = policy.tick(current, 10051);
    need(state.ready && std::abs(state.normalized_zero - prior_zero) <= .1 + 1e-8,
      "Reacquired zero plane exceeded its normalized motion-rate bound");
    need(on_screen(render(state, state.q0, "scene-reacquired-screen-plane")),
      "Reacquired reference did not return to the physical screen plane");

    // Exercise reconstruction and supported final-AA states with depth steps
    // and one/two/four-pixel foreground structures. This checks real-rendering
    // invariance; these synthetic edges do not establish a particular game's halo.
    scene::policy edge_policy;
    const auto edge_state = initialize(edge_policy, 1);
    std::vector<float> edges(size_t(width) * height);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x) {
        const unsigned boundary = width / 2 + (y / 9) % 7;
        const bool foreground = y < height / 2 ? x < boundary :
          (x % 96 == 24 || (x % 96 >= 48 && x % 96 < 50) || (x % 96 >= 72 && x % 96 < 76));
        edges[size_t(y) * width + x] = foreground ? 1.f / 64 : 1.f / 256;
      }
    apply(edge_state);
    f.upload_depth(edges);
    const bool final_aa = f.final_aa_available();
    f.set_int("USE_AA", 0);
    const auto edges_no_aa = capture();
    f.set_int("USE_AA", final_aa ? 1 : 0);
    const auto edges_selected = capture();
    const float aa_difference = f.image_difference(edges_no_aa, edges_selected);
    need(!final_aa || aa_difference > .001f, "Original AA is inert with automatic scene geometry");
    need(final_aa || edges_selected == edges_no_aa, "Repeated unfiltered automatic edge rendering changed");
    const auto full_rgb_difference = [&](const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b) {
      need(a.size() == b.size(), "Equivalent edge exports have different layouts");
      float maximum = 0;
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < 2 * width; ++x)
          for (unsigned c = 0; c < 3; ++c) {
            const float av = f.channel(a, x, y, c), bv = f.channel(b, x, y, c);
            need(std::isfinite(av) && std::isfinite(bv), "Automatic edge export contains non-finite RGB");
            maximum = std::max(maximum, std::abs(av - bv));
          }
      return maximum;
    };
    for (double unit : {100., .001}) {
      scene::policy equivalent_policy;
      apply(initialize(equivalent_policy, unit));
      const auto equivalent_edges = capture();
      // Raw values are identical after a consistent unit conversion of B and z.
      const float unit_difference = full_rgb_difference(edges_selected, equivalent_edges);
      need(unit_difference < .01f,
        "Equivalent units changed step/thin-object reconstruction in the selected final-AA state");
      log << "scene_edge_units scale=" << unit << " full_rgb_max_difference=" << unit_difference << '\n';
    }
    apply(edge_state);
    f.set_int("USE_AA", 0);
    need(capture() == edges_no_aa, "Restoring edge geometry and final-AA state did not restore the same automatic rendering");
    if (final_aa)
      sunshine_parity::write_bytes(directory / "scene-edges-aa.sbs", edges_selected.data(), edges_selected.size());
    sunshine_parity::write_bytes(directory / "scene-edges-no-aa.sbs", edges_no_aa.data(), edges_no_aa.size());
    log << "scene_edges final_aa_available=" << final_aa << " aa_difference=" << aa_difference << " units_invariant=true reversible=true\n";
    need(log.good(), "Cannot write automatic scene policy shader evidence");
    std::puts("PASS automatic scene policy through actual shader: independent startup, world-unit invariance, clipping changes, outside-reference foreground, walking gain, measured screen plane and gap/cut continuity");
  }
}
