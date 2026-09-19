// SPDX-License-Identifier: GPL-3.0-only
// Known scene edges through the actual Game3D depth sampler and exported image.
// No CPU stereo renderer or independently maintained shader copy is used.
#pragma once

namespace sunshine_jitter_fixture {
  namespace fs = std::filesystem;
  inline fs::path output;

  inline bool requested() { return sunshine_camera_fixture::flag("SUNSHINE_GAME3D_JITTER_TEST"); }

  inline void initialize(const fs::path &runtime, const fs::path &shaders, const fs::path &directory) {
    if (!requested()) return;
    using sunshine_parity::need;
    need(std::strcmp(sunshine_depth3d_fixture::effect_file, "SunshineGame3D.fx") == 0,
      "Jitter regression requires the actual SunshineGame3D shader");
    need(!sunshine_camera_fixture::requested() && !sunshine_parity::requested() && !sunshine_depth3d_color::requested(),
      "Jitter regression must not replace another requested shader regression");
    output = directory / "jitter-oracle";
    need(!fs::exists(output), "Jitter evidence requires a fresh isolated output directory");
    fs::create_directories(output);
    std::ofstream manifest(output / "provenance.txt");
    wchar_t executable[32768]{};
    need(GetModuleFileNameW(nullptr, executable, 32768) != 0, "Cannot identify jitter test executable");
    sunshine_parity::manifest_file(manifest, executable);
    sunshine_parity::manifest_file(manifest, runtime);
    sunshine_parity::manifest_file(manifest, directory / "ReShade.ini");
    sunshine_parity::manifest_file(manifest, directory / "preset.ini");
    for (const auto &entry : fs::recursive_directory_iterator(shaders))
      if (entry.is_regular_file() && (entry.path().extension() == ".fx" || entry.path().extension() == ".fxh"))
        sunshine_parity::manifest_file(manifest, entry.path());
    manifest << "scope=synthetic known jitter registration, actual full shader and native mono preservation; "
      "no SDK extraction, frame pairing, game halo elimination or performance claim\n";
    need(manifest.good(), "Cannot write jitter provenance");
  }

  template<class Fixture>
  void run(Fixture &f, unsigned width, unsigned height, reshade::api::effect_runtime *runtime, bool &depth_ready) {
    using sunshine_parity::need;
    const bool automatic = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC");
    const bool control = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_JITTER_CONTROL");
    const unsigned depth_width = width / 2, depth_height = height / 2;
    f.replace_depth(depth_width, depth_height);
    depth_ready = true;
    const bool legacy_controls = f.uniform("WP", false).handle != 0;
    if (legacy_controls) {
    f.set_int("WP", 0);
    f.set_int("Depth_Map", 1);
    f.set_int("Depth_Map_View", 2);
    f.set_float("Depth_Map_Adjust", 40.f);
    f.set_float("Depth_Adjustment", 50.f);
    f.set_float("Auto_Depth_Adjust", 0.f);
    f.set_float("ZPD_OverShoot", 0.f);
    f.set_int("Range_Boost", 0);
    f.set_float("Compatibility_Power", 1.f);
    const float neutral_trim[]{0, 0, 0, .25f}, neutral_edge[]{0, 0, 0, 0};
    runtime->set_uniform_value_float(f.uniform("WZPD_and_WND"), neutral_trim, 4);
    runtime->set_uniform_value_float(f.uniform("Weapon_Depth_Edge"), neutral_edge, 4);
    const float no_offsets[]{0, 0};
    runtime->set_uniform_value_float(f.uniform("Offset"), no_offsets, 2);
    }
    f.set_int("Depth_Map_View", 2);
    f.set_float("Depth_Adjustment", 50.f);
    f.set_float("Compatibility_Power", 1.f);
    if (automatic) {
      const bool ready = true;
      const float projection[]{0, 1}, convergence[]{.05f, 1.f / 128}, range[]{0, 1};
      runtime->set_uniform_value_bool(f.uniform("Sunshine_CameraDepthReady"), &ready, 1);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraProjection"), projection, 2);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraRawDepthRange"), range, 2);
      f.set_int("Sunshine_CameraCoordinateBasis", 1);
      f.set_float("Sunshine_CameraDepthScale", 128.f);
      f.set_float("Sunshine_CameraStrengthBlend", 1.f);
    }
    const auto crop = [&](unsigned left, unsigned top, unsigned w, unsigned h) {
      const float bounds[]{(left + .5f) / depth_width, (top + .5f) / depth_height,
        (left + w - .5f) / depth_width, (top + h - .5f) / depth_height};
      if (!control && f.uniform("Sunshine_DepthJitterBounds", false).handle) runtime->set_uniform_value_float(f.uniform("Sunshine_DepthJitterBounds"), bounds, 4);
      if (automatic) {
        const float rect[]{float(left) / depth_width, float(top) / depth_height,
          float(w) / depth_width, float(h) / depth_height};
        runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthRect"), rect, 4);
      }
    };
    const auto jitter = [&](float x, float y) {
      if (control) { need(x == 0 && y == 0, "Control shader cannot apply jitter"); return; }
      // Actual allocation-UV uniform. Inputs here are known render/depth pixels;
      // provider-unit conversion and frame ownership have separate regressions.
      const float value[]{x / depth_width, y / depth_height};
      runtime->set_uniform_value_float(f.uniform("Sunshine_DepthJitter"), value, 2);
    };
    const auto capture = [&] {
      f.measured_frames(); // Also checks exact preservation of the original color.
      return f.read(f.exported.p);
    };
    std::ofstream log(output / "measurements.csv");
    log << "mode,case,axis,jitter_px,max_depth_error,changed_pixels\n";
    const char *mode = automatic ? "automatic" : "manual";
    std::vector<float> raw(size_t(depth_width) * depth_height);
    constexpr float far_raw = .004f, near_raw = .016f;
    const auto prepared = [&](float value) { return 1.f / (1.f + (automatic ? 128.f : 319.f) * value); };
    const float far_value = prepared(far_raw), near_value = prepared(near_raw);

    for (unsigned axis = 0; axis < 2; ++axis) {
      const unsigned count = axis ? depth_height : depth_width;
      const unsigned edge = count / 2;
      for (unsigned y = 0; y < depth_height; ++y)
        for (unsigned x = 0; x < depth_width; ++x)
          raw[size_t(y) * depth_width + x] = (axis ? y : x) < edge ? far_raw : near_raw;
      f.upload_depth(raw);
      crop(0, 0, depth_width, depth_height);
      jitter(0, 0);
      const auto zero = capture();
      sunshine_parity::write_bytes(output / ("zero-axis" + std::to_string(axis) + ".sbs"), zero.data(), zero.size());
      if (control) continue;
      // Nonzero inactive bounds must not change the zero-jitter path, including
      // Manual's original out-of-range behavior. Auto still owns its full rect.
      const float inactive_bounds[]{.25f, .25f, .75f, .75f};
      if (f.uniform("Sunshine_DepthJitterBounds", false).handle) runtime->set_uniform_value_float(f.uniform("Sunshine_DepthJitterBounds"), inactive_bounds, 4);
      need(capture() == zero, "Zero jitter changed pixels through inactive jitter bounds");
      crop(0, 0, depth_width, depth_height);
      for (float phase : {-.49f, .49f}) {
        jitter(axis ? 0 : phase, axis ? phase : 0);
        const auto corrected = capture();
        float maximum = 0;
        size_t changed = 0;
        for (unsigned y = height / 8; y < height * 7 / 8; ++y) {
          for (unsigned x = width / 8; x < width * 7 / 8; ++x) {
            // Geometric edge at an integer render-pixel boundary. At 2x output
            // its adjacent pixel centers are +/-0.25 render pixels away. Each
            // signed 0.49 jitter must cross one of those centers, in the stated
            // positive-right/down convention. No stereo warp is reimplemented.
            const float position = .5f * (float(axis ? y : x) + .5f) + phase;
            const float expected = position < edge ? far_value : near_value;
            for (unsigned eye = 0; eye < 2; ++eye) {
              const float actual = f.channel(corrected, eye * width + x, y, 0);
              need(std::isfinite(actual), "Jitter correction produced nonfinite depth");
              maximum = std::max(maximum, std::abs(actual - expected));
              changed += std::abs(actual - f.channel(zero, eye * width + x, y, 0)) > .01f;
            }
          }
        }
        log << mode << ",edge," << axis << ',' << phase << ',' << maximum << ',' << changed << '\n';
        log.flush();
        std::printf("MEASURE jitter mode=%s axis=%u phase=%.2f max_error=%.9g changed=%zu\n",
          mode, axis, phase, maximum, changed);
        need(maximum < .002f && changed > 0, "Actual depth edges have wrong jitter sign, magnitude or no correction");
      }
      jitter(0, 0);
      need(capture() == zero, "Clearing jitter did not restore byte-identical depth pixels");
    }

    // Compare the complete stereo pipeline before/after removing Manual mode,
    // including large foreground/background separation. These are actual GPU
    // exports, not a second implementation of the warp.
    if (automatic) {
      crop(0, 0, depth_width, depth_height);
      jitter(0, 0);
      for (unsigned scene = 0; scene < 3; ++scene) {
        for (unsigned y = 0; y < depth_height; ++y)
          for (unsigned x = 0; x < depth_width; ++x) {
            const bool foreground = scene == 0 ? x < depth_width / 2 :
              (x > depth_width / 3 && x < depth_width * 2 / 3 && y > depth_height / 4);
            raw[size_t(y) * depth_width + x] = foreground ? .016f : (scene ? .0004f : .004f);
          }
        f.upload_depth(raw);
        const float scale = scene == 2 ? 2500.f : 128.f;
        const float convergence[]{.05f, 1.f / scale};
        f.set_float("Sunshine_CameraDepthScale", scale);
        runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
        for (int view : {0, 1, 2}) {
          f.set_int("Depth_Map_View", view);
          for (float strength : {0.f, 30.f, 100.f}) {
            f.set_float("Depth_Adjustment", strength);
            const auto pixels = capture();
            const std::string name = "stereo-scene" + std::to_string(scene) + "-view" +
              std::to_string(view) + "-strength" + std::to_string(int(strength)) + ".sbs";
            sunshine_parity::write_bytes(output / name, pixels.data(), pixels.size());
          }
        }
      }
      f.set_int("Depth_Map_View", 2);
      f.set_float("Depth_Adjustment", 50.f);
      f.set_float("Sunshine_CameraDepthScale", 128.f);
      const float convergence[]{.05f, 1.f / 128};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
    }
    if (control) { std::puts("PASS authenticated old-shader zero-jitter and stereo control captures"); return; }

    // Deliberately poison all allocation padding. Corrected sampling must remain
    // on active-pixel centers even at the output's extreme four corners.
    const unsigned left = 3, top = 2, active_width = depth_width - 6, active_height = depth_height - 4;
    for (unsigned y = 0; y < depth_height; ++y)
      for (unsigned x = 0; x < depth_width; ++x)
        raw[size_t(y) * depth_width + x] = x >= left && x < left + active_width &&
          y >= top && y < top + active_height ? far_raw : near_raw;
    f.upload_depth(raw);
    crop(left, top, active_width, active_height);
    for (float phase : {-.49f, .49f}) {
      jitter(phase, -phase);
      const auto bounded = capture();
      float maximum = 0;
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width * 2; ++x) {
          const float actual = f.channel(bounded, x, y, 0);
          need(std::isfinite(actual), "Jitter padding clamp produced nonfinite depth");
          maximum = std::max(maximum, std::abs(actual - far_value));
        }
      log << mode << ",padding,both," << phase << ',' << maximum << ",0\n";
      log.flush();
      need(maximum < .002f, "Jitter correction sampled poisoned allocation padding");
    }
    need(log.good(), "Cannot write actual shader jitter measurements");
    std::printf("PASS actual Game3D %s: signed horizontal/vertical jitter, two-times depth upsampling, zero-jitter exact restore, active-rect padding safety and native mono preservation\n", mode);
  }
}
