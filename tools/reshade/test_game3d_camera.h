// SPDX-License-Identifier: GPL-3.0-only
// Opt-in camera fixtures for the actual Depth3D ray search and reconstruction.
// No per-game calibration, source selection, or alternative renderer lives here.
#pragma once
#include <limits>
#include "test_game3d_camera_scene.h"
#include "test_scale_candidates_runtime.h"
#include "test_stereo_reference_runtime.h"

namespace sunshine_camera_fixture {
  namespace fs = std::filesystem;
  namespace api = reshade::api;
  inline fs::path output;

  inline bool flag(const char *name) {
    const char *value = std::getenv(name);
    sunshine_parity::need(!value || !*value || std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0,
      "Camera fixture flags must be 0 or 1");
    return value && std::strcmp(value, "1") == 0;
  }

  inline bool scene_requested() { return flag("SUNSHINE_GAME3D_CAMERA_SCENE_TEST"); }
  inline bool requested() { return flag("SUNSHINE_GAME3D_CAMERA_TEST") || scene_requested() || flag("SUNSHINE_GAME3D_SCALE_CANDIDATES_TEST") || flag("SUNSHINE_GAME3D_STEREO_REFERENCE_TEST"); }
  inline bool compiled() { return flag("SUNSHINE_GAME3D_CAMERA_DEPTH") || flag("SUNSHINE_GAME3D_AUTOMATIC"); }

  inline std::string definitions() {
    if (flag("SUNSHINE_GAME3D_AUTOMATIC")) {
      sunshine_parity::need(std::strcmp(sunshine_depth3d_fixture::effect_file, "SunshineGame3D.fx") == 0,
        "Automatic setup requires SunshineGame3D");
      return ",SUNSHINE_GAME3D_AUTOMATIC=1";
    }
    const char *value = std::getenv("SUNSHINE_GAME3D_CAMERA_DEPTH");
    if (!value || !*value) return {};
    compiled();
    sunshine_parity::need(std::strcmp(sunshine_depth3d_fixture::effect_file, "SunshineGame3D.fx") == 0,
      "Camera candidate requires SunshineGame3D");
    return std::string(",SUNSHINE_GAME3D_CAMERA_DEPTH=") + value;
  }

  inline void initialize(const fs::path &runtime, const fs::path &shaders, const fs::path &directory) {
    if (!requested()) return;
    sunshine_parity::need(!scene_requested() || compiled(), "Automatic scene fixture requires the camera shader compile gate");
    sunshine_parity::need(!sunshine_parity::requested() && !sunshine_depth3d_color::requested(),
      "Camera-only fixture must not replace a requested color or parity gate");
    sunshine_parity::need(std::strcmp(sunshine_depth3d_fixture::effect_file, "SunshineGame3D.fx") == 0,
      "Camera-only fixture requires SunshineGame3D");
    output = directory / "camera-oracle";
    sunshine_parity::need(!fs::exists(output), "Camera oracle requires a fresh isolated output directory");
    fs::create_directories(output);
    std::ofstream manifest(output / "provenance.txt");
    wchar_t executable[32768] {};
    sunshine_parity::need(GetModuleFileNameW(nullptr, executable, 32768) != 0, "Cannot identify camera fixture executable");
    sunshine_parity::manifest_file(manifest, executable);
    sunshine_parity::manifest_file(manifest, runtime);
    sunshine_parity::manifest_file(manifest, directory / "ReShade.ini");
    sunshine_parity::manifest_file(manifest, directory / "preset.ini");
    for (const auto &entry : fs::recursive_directory_iterator(shaders))
      if (entry.is_regular_file() && (entry.path().extension() == ".fx" || entry.path().extension() == ".fxh"))
        sunshine_parity::manifest_file(manifest, entry.path());
    manifest << "camera_compile_gate=" << compiled() << "\nscale_policy="
             << (scene_requested() ? "automatic_scene_reference_synthetic_registration" : "explicit_ground_truth_only") << '\n';
    sunshine_parity::need(manifest.good(), "Cannot write camera fixture provenance");
  }

  // The sole Host renderer has no reciprocal FP16 preparation texture. Keep
  // the original camera/preparation oracle below intact for frozen shaders;
  // this capability-gated oracle observes actual R32 geometry and final output.
  template<class Fixture>
  void run_native(Fixture &f, unsigned width, unsigned height, api::effect_runtime *runtime, bool &depth_ready) {
    using sunshine_parity::need;
    need(flag("SUNSHINE_GAME3D_CAMERA_TEST") && !scene_requested() &&
      !flag("SUNSHINE_GAME3D_SCALE_CANDIDATES_TEST") && !flag("SUNSHINE_GAME3D_STEREO_REFERENCE_TEST"),
      "Retired camera-estimator experiments require their frozen shader; use CAMERA_TEST for sole Host warp");
    std::ofstream log(output / "measurements.txt");
    log << std::setprecision(12) << "sole_host_warp_capability=2\ngeometry_format=R32_FLOAT\n";
    depth_ready = true;
    f.set_int("Depth_Map_View", 0);
    f.set_float("Depth_Adjustment", 50);
    if (sunshine_depth3d_fixture::sharpening_available(f)) f.set_float("Sharpen_Power", 0);
    f.set_int("Sunshine_CameraCoordinateBasis", 0);
    f.set_float("Sunshine_CameraStrengthBlend", 1);
    const float rect[] {0, 0, 1, 1}, range[] {0, 1}, jitter[] {0, 0};
    runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthRect"), rect, 4);
    runtime->set_uniform_value_float(f.uniform("Sunshine_CameraRawDepthRange"), range, 2);
    runtime->set_uniform_value_float(f.uniform("Sunshine_DepthJitter"), jitter, 2);
    const auto ready = [&](bool value) { runtime->set_uniform_value_bool(f.uniform("Sunshine_CameraDepthReady"), value); };
    const auto metadata = [&](float A, float inverseB, float H, float reference, float q0) {
      const float projection[] {A, inverseB}, convergence[] {reference, q0};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraProjection"), projection, 2);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
      f.set_float("Sunshine_CameraDepthScale", H);
    };
    std::vector<float> raw(size_t(width) * height, .01f);
    f.upload_depth(raw);
    const auto capture = [&] {
      f.measured_frames();
      return f.read(f.exported.p);
    };
    metadata(0, 1, 319, .05f, .02f);
    ready(false);
    const auto flat = capture();
    ready(true);
    const auto active = capture();
    need(f.image_difference(active, flat) > .002f, "Native camera fixture has no visible active displacement");
    f.set_int("Depth_Map_View", 2);
    const auto diagnostic = capture();
    const float expected = 1.f / (1.f + 319.f * .01f);
    float error = 0;
    for (unsigned eye = 0; eye < 2; ++eye)
      for (unsigned c = 0; c < 3; ++c)
        error = std::max(error, std::abs(f.channel(diagnostic, eye * width + width / 2, height / 2, c) - expected));
    need(error < .002f, "Direct native Normal Depth differs from the inverse-distance oracle");
    f.set_float("Depth_Adjustment", 0);
    need(capture() == diagnostic, "Native Normal Depth diagnostic depends on stereo strength");
    f.set_float("Depth_Adjustment", 50);
    f.set_int("Depth_Map_View", 0);
    need(capture() == active, "Native camera diagnostic return changed geometry");
    log << "normal_depth_expected=" << expected << " max_error=" << error << '\n';
    // Prove the retired FP16 restrictions are gone without allowing FP32
    // overflow. Both cases are finite at every actual raw-interval endpoint.
    for (const float H : {32767.f, 1e8f}) {
      metadata(0, 1, H, .05f, .02f);
      const auto pixels = capture();
      need(f.image_difference(pixels, flat) > .002f, "Finite R32 camera was rejected by a stale FP16 limit");
      f.capture_host_warp(output / (H < 1e6f ? "large-finite-32767" : "large-finite-1e8"));
      log << "accepted_H=" << H << " finite_fields=true\n";
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::array<std::array<float, 5>, 15> invalid {{
      {nan,1,319,.05f,.02f}, {0,nan,319,.05f,.02f}, {0,0,319,.05f,.02f},
      {0,1,nan,.05f,.02f}, {0,1,0,.05f,.02f}, {0,1,319,nan,.02f},
      {0,1,319,0,.02f}, {0,1,319,2,.02f}, {0,1,319,.05f,nan},
      {0,1,319,.05f,-1}, {inf,1,319,.05f,.02f}, {.5f,1,319,.05f,.02f},
      {0,1,inf,.05f,.02f}, {0,1,319,.05f,inf}, {0,1,1e38f,1,1e38f}
    }};
    for (size_t i = 0; i < invalid.size(); ++i) {
      const auto &v = invalid[i];
      metadata(v[0], v[1], v[2], v[3], v[4]);
      need(capture() == flat, "Invalid native camera did not return exact current mono color");
      log << "rejected_case=" << i << " exact_flat=true\n";
    }
    metadata(0,1,319,.05f,.02f);
    f.set_int("Sunshine_CameraCoordinateBasis", 1);
    need(capture() == active, "Identical raw/projected coordinates do not produce identical Host output");
    f.set_int("Sunshine_CameraCoordinateBasis", 0);
    f.set_float("Sunshine_CameraStrengthBlend", 0);
    need(capture() == flat, "Zero transition strength did not return exact flat color");
    f.set_float("Sunshine_CameraStrengthBlend", 1);
    depth_ready = false;
    need(capture() == flat, "Missing depth did not return exact flat color");
    depth_ready = true;
    need(capture() == active, "Native camera recovery did not restore exact current stereo");
    need(log.good(), "Cannot save native camera evidence");
    std::puts("PASS actual sole Host camera: direct Normal Depth, finite R32 scales beyond FP16 limits, overflow/invalid rejection, raw equivalence and exact readiness recovery");
  }

  template<class Fixture>
  void run(Fixture &f, unsigned width, unsigned height, api::effect_runtime *runtime, bool &depth_ready) {
    using sunshine_parity::need;
    if (sunshine_parity::sole_host_warp(runtime, sunshine_depth3d_fixture::effect_file)) {
      run_native(f, width, height, runtime, depth_ready);
      return;
    }
    std::ofstream log(output / "measurements.txt");
    log << std::setprecision(12);
    log << "final_aa_available=" << f.final_aa_available() << '\n';
    depth_ready = true;
    f.set_int("WP", 0);
    f.set_int("Depth_Map", 1);
    f.set_int("Depth_Map_View", 0);
    f.set_float("Depth_Map_Adjust", 40.f);
    f.set_float("Depth_Adjustment", 50.f);
    f.set_float("Zero_Parallax_Distance", .05f);
    f.set_float("ZPD_OverShoot", 0.f);
    f.set_float("Auto_Depth_Adjust", 0.f);
    f.set_float("ZPD_Balance", 0.f);
    f.set_int("ZPD_Boundary", 0);
    f.set_int("Range_Boost", 0);
    f.set_float("Compatibility_Power", 1.f);
    if (sunshine_depth3d_fixture::sharpening_available(f)) f.set_float("Sharpen_Power", 0.f);
    f.set_int("USE_AA", 0);
    f.set_int("Auto_Scaler_Adjust", 0);
    const std::array<float, 4> neutral_trim {0.f, 0.f, 0.f, .25f}, neutral_edge {};
    const auto set_vector = [&](const char *name, const std::array<float, 4> &value) {
      runtime->set_uniform_value_float(f.uniform(name), value.data(), 4);
    };
    set_vector("WZPD_and_WND", neutral_trim);
    set_vector("Weapon_Depth_Edge", neutral_edge);
    const float no_offsets[] {0.f, 0.f};
    runtime->set_uniform_value_float(f.uniform("Offset"), no_offsets, 2);

    std::vector<float> hardware(size_t(width) * height, .01f);
    const auto plane = [&](float raw) {
      std::fill(hardware.begin(), hardware.end(), raw);
      f.upload_depth(hardware);
    };
    const auto capture = [&]() {
      f.measured_frames();
      return f.read(f.exported.p);
    };
    const auto check_depth = [&](float expected, const char *label) {
      const auto samples = f.linear_samples();
      float error = 0;
      for (float value : samples) error = std::max(error, std::abs(value - expected));
      log << "depth label=" << label << " expected=" << expected << " max_error=" << error << '\n';
      need(error < .002f, "Camera preparation differs from the independent view-distance oracle");
    };
    // Both channels of the real RG16F production texture: scene convergence
    // (red) and unchanged diagnostic/prepared depth (green), away from corners.
    const auto prepared_center = [&]() {
      const auto desc = f.linear_depth->GetDesc();
      const auto bytes = f.read(f.linear_depth.p);
      const size_t pixel = size_t(desc.Height / 2) * desc.Width + desc.Width / 2;
      std::uint16_t values[2] {};
      std::memcpy(values, bytes.data() + pixel * 4, sizeof(values));
      const std::array<float, 2> value {sunshine_parity::decode_half(values[0]), sunshine_parity::decode_half(values[1])};
      need(std::isfinite(value[0]) && std::isfinite(value[1]), "Camera production depth channels contain non-finite data");
      return value;
    };

    if (flag("SUNSHINE_GAME3D_SCALE_CANDIDATES_TEST") || flag("SUNSHINE_GAME3D_STEREO_REFERENCE_TEST")) {
      const auto metadata = [&](float A, float inverseB, float K, float reference, float q0) {
        const float projection[] {A, inverseB}, convergence[] {reference, q0};
        runtime->set_uniform_value_float(f.uniform("Sunshine_CameraProjection"), projection, 2);
        runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthScale"), &K, 1);
        runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
      };
      const auto ready = [&](bool value) {
        runtime->set_uniform_value_bool(f.uniform("Sunshine_CameraDepthReady"), &value, 1);
      };
      f.set_float("Sunshine_CameraStrengthBlend", 1.f);
      f.set_int("Sunshine_CameraCoordinateBasis", 1);
      const float candidate_rect[] {0.f, 0.f, 1.f, 1.f}, candidate_range[] {0.f, 1.f};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthRect"), candidate_rect, 4);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraRawDepthRange"), candidate_range, 2);
      if (flag("SUNSHINE_GAME3D_STEREO_REFERENCE_TEST"))
        sunshine_stereo_reference_fixture::run(f, width, height, plane, capture, metadata, ready, runtime, output);
      else
        sunshine_scale_candidates_fixture::run(f, width, height, plane, capture, metadata, ready, runtime, output);
      return;
    }

    plane(.01f);
    const auto ready_uniform = f.uniform("Sunshine_CameraDepthReady", false);
    need(bool(ready_uniform.handle) == compiled(), "Camera compile gate did not include/exclude the candidate uniforms");
    if (ready_uniform.handle) {
      bool ready = true;
      runtime->get_uniform_value_bool(ready_uniform, &ready, 1);
      need(!ready, "Camera source readiness did not default to false");
    }
    const auto legacy = capture();
    const auto legacy_channels = prepared_center();
    check_depth(1.f / (1.f + 319.f * .01f), "legacy-default");
    sunshine_parity::write_bytes(output / "legacy-default.sbs", legacy.data(), legacy.size());
    if (const char *control = std::getenv("SUNSHINE_GAME3D_CAMERA_CONTROL"); control && *control) {
      need(sunshine_parity::read_file(control) == legacy,
        "Camera-disabled/default pixels differ byte-for-byte from the supplied frozen control");
      log << "external_control_byte_identical=true\n";
    }
    if (!compiled()) {
      std::puts("PASS camera compile gate zero: no camera inputs, original prepared depth and exported pixels retained");
      need(log.good(), "Cannot write camera measurements");
      return;
    }

    const auto ready = [&](bool value) { runtime->set_uniform_value_bool(ready_uniform, &value, 1); };
    const auto metadata = [&](float A, float inverseB, float K, float reference, float q0) {
      const float projection[] {A, inverseB}, convergence[] {reference, q0};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraProjection"), projection, 2);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthScale"), &K, 1);
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
    };
    if (scene_requested()) {
      run_scene_policy(f, width, height, plane, capture, prepared_center, metadata, ready, log, output);
      // A nondefault depth-coordinate transform cannot silently change the
      // reference patch behind raw-relative automation. Compare final pixels
      // against the original path with that same saved transform.
      f.set_int("Sunshine_CameraCoordinateBasis", 1);
      metadata(0, 1, 128, .05f, 1.f / 128);
      plane(.01f);
      const float shifted[] {2.f, 1.f}, identity[] {1.f, 1.f};
      const auto top_left = f.uniform("Horizontal_and_Vertical_TL", false);
      if (top_left.handle) {
        runtime->set_uniform_value_float(top_left, shifted, 2);
        ready(false);
        const auto shifted_legacy = capture();
        ready(true);
        need(capture() == shifted_legacy, "Raw mode admitted an incompatible top-left depth scale");
        runtime->set_uniform_value_float(top_left, identity, 2);
      } else log << "raw_guard_top_left=static_identity_in_this_permutation\n";
      f.set_float("AR_Side_Shrink", .5f);
      ready(false);
      const auto side_legacy = capture();
      ready(true);
      need(capture() == side_legacy, "Raw mode admitted incompatible depth side scaling");
      f.set_float("AR_Side_Shrink", 0.f);
      const auto autofit = f.uniform("DB_AutoFit", false);
      if (autofit.handle) {
        runtime->set_uniform_value_bool(autofit, true);
        ready(false);
        const auto external_fit_legacy = capture();
        ready(true);
        need(capture() == external_fit_legacy, "Raw mode admitted an external depth-fit provider");
        runtime->set_uniform_value_bool(autofit, false);
      }
      f.set_int("Auto_Scaler_Adjust", 1);
      ready(false);
      const auto edge_fit_legacy = capture();
      ready(true);
      need(capture() == edge_fit_legacy, "Raw mode admitted an untracked automatic edge-depth scale");
      f.set_int("Auto_Scaler_Adjust", 0);
      f.set_int("Sunshine_CameraCoordinateBasis", 0);
      std::puts("PASS actual raw-coordinate guards preserve the original output for incompatible saved depth transforms");
      ready(false);
      f.set_float("Depth_Adjustment", 50);
      plane(.01f);
      need(capture() == legacy, "Automatic scene fixture failed to restore the exact original fallback");
      return;
    }
    constexpr float K = 319.f, reference = .05f, anchor_q0 = (1.f / reference - 1.f) / K;
    metadata(0.f, 1.f, K, reference, anchor_q0);
    need(capture() == legacy, "Ready=false camera metadata changed original exported pixels");
    ready(true);
    const float nan = std::numeric_limits<float>::quiet_NaN(), infinity = std::numeric_limits<float>::infinity();
    const std::array<std::array<float, 5>, 19> invalid {{
      {nan, 1, K, reference, anchor_q0}, {0, nan, K, reference, anchor_q0},
      {0, 0, K, reference, anchor_q0}, {0, 1, nan, reference, anchor_q0},
      {0, 1, 0, reference, anchor_q0}, {0, 1, K, nan, anchor_q0},
      {0, 1, K, 0, anchor_q0}, {0, 1, K, reference, nan},
      {0, 1, K, reference, -1}, {infinity, 1, K, reference, anchor_q0},
      {.5f, 1, K, reference, anchor_q0}, {0, 1, infinity, reference, anchor_q0},
      {0, 1, K, reference, infinity},
      // All inputs below are finite FP32. The first is the review reproducer:
      // at raw=.01, D~1e-18 rounds to half zero and convergence to infinity.
      {0, 1, 1e20f, reference, 1},
      {0, 1, 1e20f, reference, 0}, // Isolated green/prepared-depth underflow.
      {0, 1, 32767.f, reference, 0}, // Endpoint D=2^-15 is subnormal half.
      {1, -1, 32767.f, reference, 0}, // Same bound for normal-Z projection.
      {0, 1, K, reference, 1e20f}, // Ordinary D, isolated red half overflow.
      {0, 1, K, 1, 1000} // Finite field >65504 without enormous FP32 values.
    }};
    for (size_t i = 0; i < invalid.size(); ++i) {
      const auto &v = invalid[i];
      metadata(v[0], v[1], v[2], v[3], v[4]);
      need(capture() == legacy, "Invalid/NaN camera metadata changed legacy pixels instead of falling through exactly");
      need(prepared_center() == legacy_channels, "Rejected camera metadata changed either production depth channel");
      check_depth(1.f / (1.f + 319.f * .01f), "invalid-fallback");
      log << "invalid_case=" << i << " byte_identical=true\n";
    }

    // Use a different, valid projection from the legacy reversed-depth input.
    // This makes rejection observable; a camera-equivalent legacy field could
    // pass these checks even if the unsupported-control guard did not execute.
    metadata(1, -1, K, reference, anchor_q0);
    const auto active_normal = capture();
    need(active_normal != legacy, "Unsupported-control fixture cannot distinguish active camera from legacy");
    const auto check_control_fallback = [&](const char *label) {
      ready(false);
      const auto old_control = capture();
      const auto old_channels = prepared_center();
      ready(true);
      need(capture() == old_control, "Nonneutral weapon/trim controls did not select the exact legacy path");
      need(prepared_center() == old_channels, "Nonneutral controls mixed camera depth with legacy preparation");
      log << "unsupported_control=" << label << " byte_identical=true\n";
    };
    f.set_int("WP", 1);
    check_control_fallback("custom-weapon");
    f.set_int("WP", 0);
    for (size_t component = 0; component < 4; ++component) {
      auto trim = neutral_trim;
      trim[component] += .1f;
      set_vector("WZPD_and_WND", trim);
      check_control_fallback("near-min-auto-trim");
      std::array<float, 4> saved {};
      runtime->get_uniform_value_float(f.uniform("WZPD_and_WND"), saved.data(), 4);
      need(saved == trim, "Camera admission rewrote a saved legacy trim value");
      set_vector("WZPD_and_WND", neutral_trim);
      auto edge = neutral_edge;
      edge[component] = .25f;
      set_vector("Weapon_Depth_Edge", edge);
      check_control_fallback("weapon-depth-edge");
      runtime->get_uniform_value_float(f.uniform("Weapon_Depth_Edge"), saved.data(), 4);
      need(saved == edge, "Camera admission rewrote a saved legacy edge value");
      set_vector("Weapon_Depth_Edge", neutral_edge);
      log << "unsupported_control_component=" << component << '\n';
    }
    need(capture() == active_normal, "Restoring neutral controls did not restore active camera output");

    // An explicit K and normalization anchor prove policy algebra without
    // selecting a game-specific near/scene reference or enabling an add-on.
    // The prepared fields match legacy. Final eyes intentionally omit its
    // post-search compatibility translation, measured against one mono source.
    f.set_float("Depth_Adjustment", 0);
    const auto projection_flat = capture();
    f.set_float("Depth_Adjustment", 50);
    // Compatibility_Power=1 gives TP=.0375; a flat plane has DD_Spread.y=0.
    // The original search adds sign(Diverge)*37.5*TP*2 pixels after refinement.
    constexpr float legacy_compatibility_offset = 37.5f * .0375f * 2.f;
    const std::array<float, 3> distances {25.f, 100.f, 256.f};
    for (float z : distances) {
      ready(false);
      plane(1.f / z);
      const auto old_pixels = capture();
      const auto old_channels = prepared_center();
      const auto old_shifts = f.measure_eye_shifts(projection_flat, old_pixels, "legacy-neutral-field");
      std::vector<std::uint8_t> representation_pixels;
      for (unsigned variant = 0; variant < 4; ++variant) {
        const bool reversed = (variant & 1) != 0, finite = (variant & 2) != 0;
        constexpr double n = 1, far_plane = 512;
        const double A = finite ? (reversed ? -n / (far_plane - n) : far_plane / (far_plane - n)) : (reversed ? 0 : 1);
        const double B = finite ? (reversed ? far_plane * n / (far_plane - n) : -far_plane * n / (far_plane - n)) : (reversed ? n : -n);
        const float Af = float(A), inverseB = float(1 / B);
        metadata(Af, inverseB, K, reference, anchor_q0);
        plane(float(A + B / z));
        ready(true);
        const auto actual = capture();
        const auto channels = prepared_center();
        const auto shifts = f.measure_eye_shifts(projection_flat, actual, "camera-neutral-field-screen-plane");
        const float expected = z / (z + K);
        check_depth(expected, "projection-representation");
        need(std::abs(channels[0] - old_channels[0]) < .002f && std::abs(channels[1] - old_channels[1]) < .002f,
          "Camera branch changed either neutral prepared-depth/convergence channel");
        log << "projection z=" << z << " reversed=" << reversed << " finite=" << finite
            << " red=" << channels[0] << " green=" << channels[1] << '\n';
        for (unsigned eye = 0; eye < 2; ++eye) {
          const float expected_shift = eye == 0 ? legacy_compatibility_offset : -legacy_compatibility_offset;
          const float measured_shift = shifts[eye] - old_shifts[eye];
          log << "legacy_compatibility_bias eye=" << eye << " legacy=" << old_shifts[eye]
              << " camera=" << shifts[eye] << " difference=" << measured_shift << " expected=" << expected_shift << '\n';
          need(std::abs(measured_shift - expected_shift) <= .5f,
            "Matching camera/legacy depth changed final eyes beyond the predicted compatibility translation");
        }
        if (representation_pixels.empty()) representation_pixels = actual;
        else need(f.image_difference(representation_pixels, actual) < .01f,
          "Equivalent camera projections changed final eye pixels");
      }
    }

    for (unsigned variant = 0; variant < 4; ++variant) {
      const bool reversed = (variant & 1) != 0, finite = (variant & 2) != 0;
      constexpr double n = 1, far_plane = 512;
      const double A = finite ? (reversed ? -n / (far_plane - n) : far_plane / (far_plane - n)) : (reversed ? 0 : 1);
      const double B = finite ? (reversed ? far_plane * n / (far_plane - n) : -far_plane * n / (far_plane - n)) : (reversed ? n : -n);
      metadata(float(A), float(1 / B), K, reference, anchor_q0);
      plane(reversed ? 0.f : 1.f);
      capture();
      check_depth(finite ? float(far_plane / (far_plane + K)) : 1.f, "far-endpoint");
    }

    metadata(0, 1, K, reference, anchor_q0);
    plane(.01f);
    const auto neutral = capture();
    f.set_float("Auto_Depth_Adjust", .3f);
    f.set_float("ZPD_Balance", .8f);
    f.set_float("Zero_Parallax_Distance", .4f);
    f.set_float("ZPD_OverShoot", 1.f);
    need(capture() == neutral, "Active camera mode still responds to legacy scene gain/convergence controls");
    float preserved = 0;
    runtime->get_uniform_value_float(f.uniform("Auto_Depth_Adjust"), &preserved, 1);
    need(preserved == .3f, "Camera policy rewrote the saved legacy depth control");
    runtime->get_uniform_value_float(f.uniform("ZPD_Balance"), &preserved, 1);
    need(preserved == .8f, "Camera policy rewrote the saved legacy balance control");
    f.set_float("Auto_Depth_Adjust", 0);
    f.set_float("ZPD_Balance", 0);
    f.set_float("Zero_Parallax_Distance", reference);
    f.set_float("ZPD_OverShoot", 0);

    // Exported-pixel translations and both real production channels. Changing
    // q0 must move the zero crossing and retain pairwise inverse-depth slope.
    f.set_float("Depth_Adjustment", 0);
    const auto flat = capture();
    f.set_float("Depth_Adjustment", 65);
    const std::array<float, 3> inverse_planes {1.f / 64, 1.f / 128, 1.f / 256};
    const std::array<float, 3> zero_planes {1.f / 80, 1.f / 128, 1.f / 192};
    std::array<std::array<float, 3>, 3> fields {}, disparity {};
    for (size_t zero = 0; zero < zero_planes.size(); ++zero) {
      metadata(0, 1, K, reference, zero_planes[zero]);
      for (size_t surface = 0; surface < inverse_planes.size(); ++surface) {
        plane(inverse_planes[surface]);
        const auto pixels = capture();
        const auto channels = prepared_center();
        const auto shifts = f.measure_eye_shifts(flat, pixels, "camera-independent-convergence");
        const float expected_depth = 1.f / (1.f + K * inverse_planes[surface]);
        const float expected_field = reference * K * (zero_planes[zero] - inverse_planes[surface]);
        fields[zero][surface] = channels[0];
        disparity[zero][surface] = shifts[0] - shifts[1];
        need(std::abs(channels[0] - expected_field) < .002f && std::abs(channels[1] - expected_depth) < .002f,
          "Camera zero-plane move changed inverse-depth gain or the diagnostic channel");
        log << "convergence q0=" << zero_planes[zero] << " q=" << inverse_planes[surface]
            << " field=" << channels[0] << " expected=" << expected_field << " depth=" << channels[1]
            << " left=" << shifts[0] << " right=" << shifts[1] << " disparity=" << disparity[zero][surface] << '\n';
      }
    }
    for (size_t zero = 1; zero < zero_planes.size(); ++zero) {
      const float shift = reference * K * (zero_planes[zero] - zero_planes[0]);
      for (size_t surface = 0; surface < inverse_planes.size(); ++surface)
        need(std::abs(fields[zero][surface] - fields[0][surface] - shift) < .002f,
          "Independent convergence did not translate the depth field by one common offset");
      for (size_t surface = 1; surface < inverse_planes.size(); ++surface) {
        const float old_difference = disparity[0][surface] - disparity[0][0];
        const float new_difference = disparity[zero][surface] - disparity[zero][0];
        need(std::abs(new_difference - old_difference) <= 1.f,
          "Moving camera zero plane changed pairwise exported disparity beyond subpixel-fit tolerance");
      }
    }
    need(fields[0][1] > 0 && std::abs(fields[1][1]) < .002f && fields[2][1] < 0,
      "Camera zero plane did not cross the known middle surface");
    need(std::abs(disparity[0][1] - disparity[2][1]) >= 1.f,
      "Camera convergence field did not move the actual exported eyes");
    need(disparity[0][1] > .5f && std::abs(disparity[1][1]) <= .5f && disparity[2][1] < -.5f,
      "Camera convergence crossed a field zero without crossing actual zero binocular disparity");

    // Physical screen-plane semantics, independent of legacy compatibility.
    // q=q0 must leave both source-coordinate shifts at zero, then crossing q0
    // must reverse actual eye disparity. Include auto and foveated compatibility
    // settings, and retain their stored values while camera mode is active.
    constexpr float screen_surface = 1.f / 128;
    const std::array<float, 3> strengths {15.f, 65.f, 100.f}, compatibility {0.f, 1.f, -1.f};
    const std::array<float, 3> crossing_zero {screen_surface * .5f, screen_surface, screen_surface * 1.5f};
    float previous_span = 0;
    for (float strength : strengths) {
      f.set_float("Depth_Adjustment", strength);
      plane(screen_surface);
      std::array<std::vector<std::uint8_t>, 3> compatibility_reference;
      for (size_t cp = 0; cp < compatibility.size(); ++cp) {
        f.set_float("Compatibility_Power", compatibility[cp]);
        std::array<float, 3> screen_disparity {};
        for (size_t crossing = 0; crossing < crossing_zero.size(); ++crossing) {
          metadata(0, 1, K, reference, crossing_zero[crossing]);
          const auto pixels = capture();
          const auto shifts = f.measure_eye_shifts(flat, pixels, "camera-actual-screen-plane");
          screen_disparity[crossing] = shifts[0] - shifts[1];
          log << "screen_plane strength=" << strength << " compatibility=" << compatibility[cp]
              << " q0=" << crossing_zero[crossing] << " q=" << screen_surface << " left=" << shifts[0]
              << " right=" << shifts[1] << " disparity=" << screen_disparity[crossing] << '\n';
          if (crossing == 1) {
            need(std::abs(shifts[0]) <= .5f && std::abs(shifts[1]) <= .5f && std::abs(screen_disparity[crossing]) <= .5f,
              "q=q0 does not map to the actual screen plane at this strength/compatibility setting");
          }
          if (cp == 0) compatibility_reference[crossing] = pixels;
          else need(pixels == compatibility_reference[crossing],
            "Compatibility still shifts active camera output or changes its depth response");
        }
        need(screen_disparity[0] < -.5f && screen_disparity[2] > .5f,
          "Moving q0 across a fixed surface did not reverse actual binocular disparity");
        runtime->get_uniform_value_float(f.uniform("Compatibility_Power"), &preserved, 1);
        need(preserved == compatibility[cp], "Camera screen-plane policy rewrote stored compatibility power");
        if (cp == 0) {
          const float span = screen_disparity[2] - screen_disparity[0];
          need(span > previous_span + .5f, "Increasing strength did not increase actual camera disparity response");
          previous_span = span;
        }
      }
    }
    f.set_float("Compatibility_Power", 1);

    // Readiness revocation restores the exact legacy path, with the user's
    // existing controls, after the candidate has actually rendered geometry.
    ready(false);
    f.set_float("Depth_Adjustment", 50);
    plane(.01f);
    need(capture() == legacy, "Disabling camera readiness did not restore original exported pixels");
    need(log.good(), "Cannot write camera measurements");
    std::puts("PASS controlled camera depth: exact fallback, finite/infinite projections, neutral fields and predicted legacy eye bias, actual screen-plane convergence across strength/compatibility, both depth channels");
  }
}
