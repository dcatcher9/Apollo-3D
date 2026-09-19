// SPDX-License-Identifier: GPL-3.0-only
// Opt-in actual full-effect color oracles. No stereo reconstruction is copied.
#pragma once

namespace sunshine_depth3d_color {
  namespace fs = std::filesystem;
  inline fs::path output;

  inline bool requested() {
    const char *value = std::getenv("SUNSHINE_DEPTH3D_COLOR_ORACLE");
    sunshine_parity::need(!value || !*value || std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0, "SUNSHINE_DEPTH3D_COLOR_ORACLE must be 0 or 1");
    return value && std::strcmp(value, "1") == 0;
  }

  inline void initialize(const fs::path &runtime, const fs::path &shaders, const fs::path &directory) {
    if (!requested()) {
      return;
    }
    sunshine_parity::need(!sunshine_parity::requested(), "Select either color-oracle or parity mode");
    output = directory / "color-oracle";
    sunshine_parity::need(!fs::exists(output), "Color oracle requires a fresh output directory");
    fs::create_directories(output);
    std::ofstream manifest(output / "provenance.txt");
    wchar_t executable[32768] {};
    sunshine_parity::need(GetModuleFileNameW(nullptr, executable, 32768) != 0, "Cannot identify color-oracle executable");
    sunshine_parity::manifest_file(manifest, executable);
    sunshine_parity::manifest_file(manifest, runtime);
    sunshine_parity::manifest_file(manifest, directory / "ReShade.ini");
    sunshine_parity::manifest_file(manifest, directory / "preset.ini");
    for (const auto &entry : fs::recursive_directory_iterator(shaders)) {
      if (entry.is_regular_file() && (entry.path().extension() == ".fx" || entry.path().extension() == ".fxh")) {
        sunshine_parity::manifest_file(manifest, entry.path());
      }
    }
  }

  // ST 2084 absolute-light decoding, followed by a linear Rec.2020 -> Rec.709
  // basis change. Values are scRGB units (80 cd/m2), not display tone mapping.
  inline double pq_linear(double code) {
    const double p = std::pow(code, 32.0 / 2523.0);
    return 125.0 * std::pow(std::max(p - 3424.0 / 4096.0, 0.0) / (2413.0 / 128.0 - (2392.0 / 128.0) * p), 16384.0 / 2610.0);
  }

  inline double pq_encode(double units) {
    const double p = std::pow(units / 125.0, 2610.0 / 16384.0);
    return std::pow((3424.0 / 4096.0 + (2413.0 / 128.0) * p) / (1.0 + (2392.0 / 128.0) * p), 2523.0 / 32.0);
  }

  using rgb = std::array<double, 3>;

  inline rgb decoded(rgb value, unsigned color) {
    if (color == 1) {
      for (auto &v : value) {
        v = sunshine_parity::linear(float(v));
      }
    } else if (color == 3) {
      for (auto &v : value) {
        v = pq_linear(v);
      }
      const auto v = value;
      value = {1.660491002 * v[0] - .587641139 * v[1] - .072849863 * v[2], -.124550475 * v[0] + 1.132899897 * v[1] - .008349423 * v[2], -.018150763 * v[0] - .100578898 * v[1] + 1.118729661 * v[2]};
    }
    return value;
  }

  template<class Fixture>
  void run(Fixture &f, unsigned width, unsigned height, reshade::api::effect_runtime *runtime, const char *effect) {
    using sunshine_parity::need;
    need(!output.empty(), "Color oracle initialization missing");
    std::ofstream report(output / "measurements.txt");
    report << std::setprecision(12) << "effect=" << effect << " color=" << f.color << " compatibility=" << f.compatible
           << " source_format=" << unsigned(f.source_format) << " dimensions=" << width << 'x' << height
           << " export=float32_RGBA_linear_Rec709_scRGB80nits; SDR decoded for measurement\n";
    sunshine_parity::isolate_techniques(f, runtime, effect, false, report);
    const bool game3d = std::strcmp(effect, "SunshineGame3D.fx") == 0;
    if (!game3d) f.set_int("Depth_Map", 1);
    f.set_int("Depth_Map_View", 0);
    if (!game3d) f.set_float("Zero_Parallax_Distance", .18f);
    f.set_float("Compatibility_Power", 1);
    if (sunshine_depth3d_fixture::sharpening_available(f, effect)) f.set_float("Sharpen_Power", 0);
    f.set_int("Performance_Level", 2);
    const bool final_aa = f.final_aa_available();
    report << "final_AA_available=" << final_aa << '\n';
    f.set_int("USE_AA", 0);
    if (game3d) {
      // Explicit synthetic camera for the automatic-only effect. Color tests
      // must exercise a nonzero warp, not accidentally pass mono fallback.
      f.set_int("Sunshine_CameraCoordinateBasis", 1);
      const float projection[] {0, 1}, convergence[] {.05f, .001f};
      const float rect[] {0, 0, 1, 1}, jitter[] {0, 0};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraProjection"), projection, 2);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthRect"), rect, 4);
      runtime->set_uniform_value_float(f.uniform("Sunshine_DepthJitter"), jitter, 2);
      runtime->set_uniform_value_bool(f.uniform("Sunshine_CameraDepthReady"), true);
      f.set_float("Sunshine_CameraDepthScale", 1000);
      f.set_float("Sunshine_CameraStrengthBlend", 1);
      report << "camera=explicit_synthetic_raw_H1000_q0.001_not_production_calibration\n";
    }
    f.upload_depth(std::vector<float>(size_t(width) * height, .01f));
    bool passed = true;

    auto quantize = [&](rgb v) {
      for (auto &c : v) {
        c = f.color == 2 ? sunshine_parity::decode_half(sunshine_parity::encode_half(float(c))) :
                           std::round(c * (f.color == 1 ? 255 : 1023)) / (f.color == 1 ? 255 : 1023);
      }
      return v;
    };
    auto paint = [&](rgb a, rgb b, bool checker, const std::string &label) {
      a = quantize(a);
      b = quantize(b);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const auto &v = (checker ? ((x + y) & 1) : (((x + y / 7) / 8) & 1)) ? b : a;
          const size_t i = size_t(y) * width + x;
          if (f.color == 2) {
            const std::uint16_t pixel[4] {sunshine_parity::encode_half(float(v[0])), sunshine_parity::encode_half(float(v[1])), sunshine_parity::encode_half(float(v[2])), 0x3c00};
            std::memcpy(f.source_bytes.data() + 8 * i, pixel, 8);
          } else if (f.color == 1) {
            const std::uint8_t pixel[4] {std::uint8_t(std::lround(v[0] * 255)), std::uint8_t(std::lround(v[1] * 255)), std::uint8_t(std::lround(v[2] * 255)), 255};
            std::memcpy(f.source_bytes.data() + 4 * i, pixel, 4);
          } else {
            const std::uint32_t pixel = std::uint32_t(std::lround(v[0] * 1023)) |
                                        (std::uint32_t(std::lround(v[1] * 1023)) << 10) | (std::uint32_t(std::lround(v[2] * 1023)) << 20) | 0xc0000000u;
            std::memcpy(f.source_bytes.data() + 4 * i, &pixel, 4);
          }
        }
      }
      f.fill_upload(f.source_upload, f.backbuffers[0]->GetDesc(), f.source_bytes.data(), f.source_footprint);
      sunshine_parity::write_bytes(output / (label + ".source.bin"), f.source_bytes.data(), f.source_bytes.size());
      return std::array<rgb, 2> {decoded(a, f.color), decoded(b, f.color)};
    };
    auto capture = [&](const std::string &label) {
      // Bounded static settling; no assertion about live game temporal history.
      for (unsigned i = 0; i < 30; ++i) {
        f.step();
      }
      f.measured_frames();  // Also proves the native source remained byte-exact.
      const auto bytes = f.read(f.exported.p);
      std::vector<float> pixels(size_t(width) * 2 * height * 4);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < 2 * width; ++x) {
          for (unsigned c = 0; c < 4; ++c) {
            float value = f.channel(bytes, x, y, c);
            if (f.color == 1 && c == 3) {
              std::uint32_t packed = 0;
              std::memcpy(&packed, bytes.data() + (size_t(y) * width * 2 + x) * 4, 4);
              value = float(packed >> 30) / 3.f;
            }
            if (f.color == 1 && c < 3) {
              value = sunshine_parity::linear(value);
            }
            need(std::isfinite(value), "Color oracle export contains nonfinite values");
            pixels[(size_t(y) * width * 2 + x) * 4 + c] = value;
          }
        }
      }
      sunshine_parity::write_bytes(output / (label + ".rgba.f32"), pixels.data(), pixels.size() * sizeof(float));
      sunshine_parity::uniforms(runtime, effect, output / (label + ".uniforms.txt"));
      return pixels;
    };
    auto sample = [&](const std::vector<float> &pixels, auto &&visit) {
      for (unsigned eye = 0; eye < 2; ++eye) {
        for (unsigned y = height / 4; y < 3 * height / 4; y += 7) {
          for (unsigned x = width / 8; x < 7 * width / 8; ++x) {
            const size_t i = (size_t(y) * width * 2 + eye * width + x) * 4;
            visit(x, y, rgb {pixels[i], pixels[i + 1], pixels[i + 2]});
          }
        }
      }
    };

    // The documented SDR pipeline filters encoded sRGB; HDR filters linear
    // scRGB. In either interpolation domain, mixtures of two endpoint colors
    // lie on their RGB line segment regardless of the warp's coordinates.
    // Saved captures remain linear; convert only SDR metric samples/endpoints.
    const auto interpolation_domain = [&](rgb value) {
      if (f.color == 1) for (auto &c : value) c = sunshine_parity::encoded(float(c));
      return value;
    };
    report << "interpolation_metric_domain=" << (f.color == 1 ? "encoded_sRGB" : "linear_Rec709_scRGB80nits") << '\n';
    auto endpoints = paint(f.color == 2 ? rgb {-.25, .5, 4} : rgb {.05, .25, .85}, f.color == 2 ? rgb {4, 8, -.5} : rgb {.8, .9, .1}, false, "interpolation");
    for (auto &endpoint : endpoints) endpoint = interpolation_domain(endpoint);
    rgb direction {};
    double length2 = 0;
    for (unsigned c = 0; c < 3; ++c) {
      direction[c] = endpoints[1][c] - endpoints[0][c];
      length2 += direction[c] * direction[c];
    }
    size_t total_interior = 0;
    for (float strength : {7.3f, 12.1f, 19.7f}) {
      f.set_float("Depth_Adjustment", strength);
      const auto label = "interpolation-" + std::to_string(strength);
      const auto pixels = capture(label);
      double maximum = 0;
      size_t interior = 0;
      sample(pixels, [&](unsigned, unsigned, rgb value) {
        value = interpolation_domain(value);
        double t = 0;
        for (unsigned c = 0; c < 3; ++c) {
          t += (value[c] - endpoints[0][c]) * direction[c];
        }
        t /= length2;
        double residual2 = 0;
        for (unsigned c = 0; c < 3; ++c) {
          const double error = value[c] - endpoints[0][c] - std::clamp(t, 0.0, 1.0) * direction[c];
          residual2 += error * error;
        }
        maximum = std::max(maximum, std::sqrt(residual2 / length2));
        interior += t > .08 && t < .92 ? 1u : 0u;
      });
      total_interior += interior;
      report << "interpolation strength=" << strength << " max_relative_segment_error=" << maximum << " interior_samples=" << interior << '\n';
      passed &= maximum < .006;
    }
    passed &= total_interior >= width / 8;

    // A 1px achromatic checker gives a strong independent hue-neutrality and
    // scale-equivariance test for the actual HDR AA post pass. The three native
    // scRGB scales keep the gradient above AXAA's absolute edge thresholds.
    if (f.color != 1) {
      f.set_float("Depth_Adjustment", 0);
      std::vector<float> scale_reference;
      for (double scale : f.color == 2 ? std::vector<double> {.25, 1, 4} : std::vector<double> {1}) {
        const double low = .5 * scale, high = 2.5 * scale;
        const rgb a = f.color == 3 ? rgb {pq_encode(low), pq_encode(low), pq_encode(low)} : rgb {low, low, low};
        const rgb b = f.color == 3 ? rgb {pq_encode(high), pq_encode(high), pq_encode(high)} : rgb {high, high, high};
        const std::string label = "hdr-aa-" + std::to_string(scale);
        const auto ends = paint(a, b, true, label);
        f.set_int("USE_AA", 0);
        const auto plain = capture(label + "-off");
        if (final_aa) f.set_int("USE_AA", 1);
        const auto filtered = capture(label + (final_aa ? "-on" : "-no-aa-repeat"));
        double hue_error = 0, response = 0, scale_error = 0;
        double checker_weight_error = 0, plain_endpoint_error = 0;
        size_t checker_levels[2] {};
        sample(filtered, [&](unsigned x, unsigned y, rgb value) {
          hue_error = std::max(hue_error, std::max(std::abs(value[0] - value[1]), std::abs(value[1] - value[2])) / high);
        });
        // In a 1px checker, AXAA's thin-line rule retains the center as the
        // edge sample. The 3x3 mean contains five center-color samples and
        // four opposite-color samples, and the low-pass weight is 3/4.
        // Therefore the result is (2*center + opposite)/3. Derive the phase
        // The retired Game3D path must instead preserve the center endpoint.
        // Derive either expected source phase
        // from the captured AA-off image, not a source-coordinate rounding
        // assumption, and verify that it is actually one of the endpoints.
        for (unsigned eye = 0; eye < 2; ++eye) {
          for (unsigned y = height / 4; y < 3 * height / 4; y += 7) {
            for (unsigned x = width / 8; x < 7 * width / 8; ++x) {
              const size_t i = (size_t(y) * width * 2 + eye * width + x) * 4;
              const unsigned center = std::abs(plain[i] - ends[0][0]) < std::abs(plain[i] - ends[1][0]) ? 0 : 1;
              ++checker_levels[center];
              for (unsigned c = 0; c < 3; ++c) {
                const double range = std::abs(ends[1][c] - ends[0][c]);
                need(range > .3, "HDR AA checker contrast no longer clears the fixed search thresholds");
                const double expected = final_aa ? (2 * ends[center][c] + ends[1 - center][c]) / 3 : ends[center][c];
                plain_endpoint_error = std::max(plain_endpoint_error, std::abs(plain[i + c] - ends[center][c]) / range);
                checker_weight_error = std::max(checker_weight_error, std::abs(filtered[i + c] - expected) / range);
              }
            }
          }
        }
        for (size_t i = 0; i < filtered.size(); ++i) {
          if (i % 4 != 3) {
            response = std::max(response, std::abs(double(filtered[i] - plain[i])) / high);
            if (!scale_reference.empty()) {
              scale_error = std::max(scale_error, std::abs(double(filtered[i]) / scale - scale_reference[i]) / 2.5);
            }
          }
        }
        if (scale_reference.empty() && f.color == 2) {
          scale_reference = filtered;
          for (size_t i = 0; i < scale_reference.size(); ++i) {
            if (i % 4 != 3) {
              scale_reference[i] /= float(scale);
            }
          }
        }
        report << "hdr-aa scale=" << scale << " hue_relative_error=" << hue_error << " response=" << response
               << " scale_relative_error=" << scale_error << " checker_weight_relative_error=" << checker_weight_error
               << " plain_endpoint_relative_error=" << plain_endpoint_error << " checker_dark_samples=" << checker_levels[0]
               << " checker_bright_samples=" << checker_levels[1] << " decoded_endpoints=" << ends[0][0] << ',' << ends[1][0] << '\n';
        passed &= hue_error < .003 && (final_aa ? response > .02 : plain == filtered) && scale_error < .006 && checker_weight_error < .005 &&
                  plain_endpoint_error < .005 && checker_levels[0] >= width / 8 && checker_levels[1] >= width / 8;
      }
    }
    report << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    report.close();
    need(passed, "Actual full-effect color oracle failed; inspect color-oracle/measurements.txt");
    std::puts(final_aa ? "PASS actual full-effect transfer-domain interpolation and original/legacy HDR AA color oracles; native source unchanged" :
      "PASS actual full-effect transfer-domain interpolation, absent Game3D final AA, unchanged HDR checker and source colors");
  }
}  // namespace sunshine_depth3d_color
