// SPDX-License-Identifier: GPL-3.0-only
// Opt-in estimator research through the unchanged actual shader. Explicit
// candidate uniforms isolate renderer response; this is not add-on integration.
#pragma once

namespace sunshine_scale_candidates_fixture {
  template<class Fixture, class Plane, class Capture, class Metadata, class Ready>
  void run(Fixture &f, unsigned width, unsigned height, Plane plane, Capture capture,
      Metadata metadata, Ready ready, reshade::api::effect_runtime *runtime, const std::filesystem::path &directory) {
    using sunshine_parity::need;
    namespace fs = std::filesystem;
    need(f.color == 2, "Scale candidate fixture requires scRGB FP16 output");
    const auto output = directory / "scale-candidates";
    fs::create_directories(output);
    std::ofstream report(output / "measurements.csv");
    report << std::setprecision(17)
      << "case,candidate,gain,center,sigma,mad,span,field_span,min_disparity_px,max_disparity_px,binocular_span_px,mono_error,spatial_finite,stale_gain\n";
    std::ofstream scope(output / "scope.txt");
    scope << "Candidate statistics are calculated independently over 32x18 synthetic raw depths. "
      << "The unchanged full shader renders each spatial scene and constant-depth layer probes. "
      << "Binocular shifts use actual exported layer probes, not an analytical replica of warping. "
      << "No production controller, source selector, lifetime policy or UI is exercised. "
      << "This experiment is functional and contains no performance conclusion.\n"
      << "Shared zero=.25, reference=.05, strength=100; all candidates match gain4 on equal .125/.375 layers. "
      << "Equal-depth candidate denominators retain prior gain4. Stale-gain stress uses each candidate's previous gain "
      << "from equal .249/.251 layers: gain4 for center/scene mean, gain500 for spread candidates. "
      << "Final pixel disparity includes the current renderer's reconstruction and quantization; the field span is not a pixel oracle.\n";
    const auto save = [&](const fs::path &path, const std::vector<std::uint8_t> &bytes) {
      sunshine_parity::write_bytes(path, bytes.data(), bytes.size());
    };
    const auto finite = [&](const std::vector<std::uint8_t> &pixels) {
      for (size_t i = 0; i + 1 < pixels.size(); i += 2) {
        std::uint16_t value;
        std::memcpy(&value, pixels.data() + i, 2);
        need(std::isfinite(sunshine_parity::decode_half(value)), "Candidate shader export has nonfinite pixels");
      }
    };
    ready(true);
    metadata(0.f, 1.f, 4.f, .05f, .25f);
    plane(.25f);
    f.set_float("Depth_Adjustment", 0.f);
    const auto mono = capture();
    finite(mono);
    save(output / "mono.rgba16f", mono);
    save(output / "source.rgba16f", f.source_bytes);
    f.set_float("Depth_Adjustment", 100.f);
    struct scenario { const char *name; std::array<float, 576> raw; bool stale = false; };
    std::vector<scenario> scenes;
    const auto add = [&](const char *name, auto sample) {
      scenario value {name, {}};
      for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x)
        value.raw[y * 32 + x] = sample(x, y);
      scenes.push_back(value);
    };
    add("equal_layers", [](unsigned x, unsigned) { return x < 16 ? .125f : .375f; });
    add("small_foreground", [](unsigned x, unsigned y) { return x >= 4 && x < 6 && y >= 4 && y < 7 ? .375f : .125f; });
    add("multilevel", [](unsigned x, unsigned) { return .0625f + .125f * float(x / 8); });
    add("one_outlier", [](unsigned x, unsigned y) { return !x && !y ? .99f : x < 16 ? .125f : .375f; });
    add("equal_depth", [](unsigned, unsigned) { return .25f; });
    auto stale = scenes.front(); stale.name = "narrow_to_broad_stale"; stale.stale = true; scenes.push_back(stale);
    for (const auto &scene : scenes) {
      double mean = 0, center = 0, variance = 0, mad = 0;
      float lo = 1, hi = 0;
      for (unsigned i = 0; i < 576; ++i) {
        mean += scene.raw[i];
        lo = std::min(lo, scene.raw[i]); hi = std::max(hi, scene.raw[i]);
        const auto x = i % 32, y = i / 32;
        if (x >= 14 && x < 18 && y >= 7 && y < 11) center += scene.raw[i] / 16.;
      }
      mean /= 576.;
      for (float raw : scene.raw) { const auto d = raw - mean; variance += d * d; mad += std::abs(d); }
      variance /= 576.; mad /= 576.;
      const auto sigma = std::sqrt(variance), span = double(hi) - lo;
      const std::array<double, 5> gains {1. / center, sigma > 0 ? .5 / sigma : 4.,
        sigma > 0 ? std::min(.5 / sigma, 2. / span) : 4., variance > 0 ? mad / (2. * variance) : 4., 1. / mean};
      const std::array<const char *, 5> names {"center_mean", "variance", "bounded_variance", "effective_span", "scene_mean"};
      std::vector<float> hardware(size_t(width) * height);
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x)
        hardware[size_t(y) * width + x] = scene.raw[std::min(17u, y * 18 / height) * 32 + std::min(31u, x * 32 / width)];
      sunshine_parity::write_bytes(output / (std::string(scene.name) + ".raw.f32"), hardware.data(), hardware.size() * sizeof(float));
      for (unsigned candidate = 0; candidate < gains.size(); ++candidate) {
        // The stale case deliberately retains a prior narrow scene's gain.
        // It tests renderer behavior, not the temporal controller.
        const float gain = scene.stale ? (candidate == 0 || candidate == 4 ? 4.f : 500.f) : float(gains[candidate]);
        metadata(0.f, 1.f, gain, .05f, .25f);
        f.upload_depth(hardware);
        f.set_float("Depth_Adjustment", 0.f);
        const auto zero_strength = capture();
        const auto mono_error = f.image_difference(mono, zero_strength);
        need(mono_error < .003f, "Candidate gain altered zero-strength color");
        f.set_float("Depth_Adjustment", 100.f);
        const auto spatial = capture();
        finite(spatial);
        const auto prefix = std::string(scene.name) + "-" + names[candidate];
        save(output / (prefix + ".rgba16f"), spatial);
        const auto prepared = f.read(f.linear_depth.p);
        finite(prepared);
        const auto desc = f.linear_depth->GetDesc();
        need(desc.Width == width && desc.Height == height, "Scale probe requires the current full-resolution prepared texture");
        for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x) {
          const auto px = (2 * x + 1) * width / 64, py = (2 * y + 1) * height / 36;
          std::uint16_t value;
          std::memcpy(&value, prepared.data() + (size_t(py) * width + px) * 4 + 2, 2);
          const float expected = 1.f / (1.f + gain * scene.raw[y * 32 + x]);
          need(std::abs(sunshine_parity::decode_half(value) - expected) < .002f,
            "Actual prepared depth does not match explicit candidate gain and uploaded native depth");
        }
        save(output / (prefix + ".prepared.rg16f"), prepared);
        if (scene.name == std::string("equal_layers") && candidate == 0)
          sunshine_parity::uniforms(runtime, sunshine_depth3d_fixture::effect_file, output / "initial-uniforms.txt");
        double min_disparity = 0, max_disparity = 0;
        if (!scene.stale) {
          const auto probe = [&](float depth, const char *label) {
            plane(depth);
            const auto pixels = capture();
            finite(pixels);
            const auto shifts = f.measure_eye_shifts(mono, pixels, label);
            return double(shifts[0] - shifts[1]);
          };
          min_disparity = probe(lo, "scale-candidate-min-layer");
          max_disparity = probe(hi, "scale-candidate-max-layer");
          if (span > 0.) need(std::abs(max_disparity - min_disparity) > .5,
            "Candidate layer probes never established a nonzero actual stereo response");
        }
        report << scene.name << ',' << names[candidate] << ',' << gain << ',' << center << ',' << sigma << ',' << mad
          << ',' << span << ',' << .05 * gain * span << ',';
        if (scene.stale) report << "nan,nan,nan,";
        else report << min_disparity << ',' << max_disparity << ',' << std::abs(max_disparity - min_disparity) << ',';
        report << mono_error << ",1," << scene.stale << '\n';
        report.flush();
        need(report.good(), "Cannot write candidate renderer measurements");
        std::printf("MEASURE scale-candidate case=%s candidate=%s gain=%.9g field_span=%.9g actual_layer_span=%.9g finite=1 mono_error=%.9g stale=%u\n",
          scene.name, names[candidate], gain, .05 * gain * span, std::abs(max_disparity - min_disparity), mono_error, unsigned(scene.stale));
      }
    }
    need(scope.good(), "Cannot write candidate experiment scope");
    std::puts("PASS candidate scale experiment: actual shader spatial output finite, zero-strength mono, measured layer disparity; no production integration claim");
  }
}
