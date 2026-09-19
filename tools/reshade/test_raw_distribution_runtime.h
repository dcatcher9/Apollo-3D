// SPDX-License-Identifier: GPL-3.0-only
// Actual DSV distribution permutation and explicit UI-action integration.
// Camera, depth, readiness and strength-blend uniforms are never injected.
#pragma once

namespace sunshine_raw_distribution_fixture {
  template<class Fixture>
  void run(Fixture &f, float initial_gain, const sunshine_raw_oracle::result &initial_reference,
      unsigned width, unsigned height) {
    namespace fs = std::filesystem;
    namespace api = reshade::api;
    const auto need = sunshine_parity::need;
    const bool final_aa = f.final_aa_available();
    const auto directory = f.runtime_directory / "distribution-evidence";
    need(!fs::exists(directory), "Distribution fixture requires fresh evidence directory");
    fs::create_directories(directory);
    std::ofstream report(directory / "measurements.txt");
    report << std::setprecision(17) << "schema=raw-distribution-integration-1\n"
      << "oracle=" << (sunshine_raw_oracle::legacy_center() ? "legacy-center" : "scene-sigma")
      << " geometry_injected=0 test_only_action_adapter=1 native_depth=real_DSV_selected_backup\n";
    const auto preset_before = sunshine_parity::read_file(f.runtime_directory / "preset.ini");
    const auto save = [&](const fs::path &path, const std::vector<std::uint8_t> &bytes) {
      sunshine_parity::write_bytes(path, bytes.data(), bytes.size());
    };
    const auto paint = [&](bool permuted) {
      // Constant interior colors plus a narrow bright marker per depth band.
      // scRGB uses linear Rec709; PQ uses Rec2020 code, both kept in source.bin.
      const std::array<std::array<float, 3>, 4> colors {{{.08f, .16f, 2.f}, {.1f, 1.f, .2f},
        {1.5f, .3f, .08f}, {3.f, .08f, .15f}}};
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        unsigned band = std::min(3u, 4 * x / width);
        if (permuted) band = (band + 1) % 4;
        auto rgb = colors[band];
        const unsigned within = x % (width / 4);
        if (within >= width / 8 - 4 && within < width / 8 + 4) rgb = {8.f, 8.f, 8.f};
        const auto pixel = size_t(y) * width + x;
        if (f.color == 2) {
          const std::uint16_t value[4] {sunshine_parity::encode_half(rgb[0]), sunshine_parity::encode_half(rgb[1]),
            sunshine_parity::encode_half(rgb[2]), 0x3c00};
          std::memcpy(f.source_bytes.data() + pixel * 8, value, 8);
        } else {
          std::uint32_t packed = 0xc0000000u;
          for (unsigned c = 0; c < 3; ++c)
            packed |= std::uint32_t(std::lround(sunshine_depth3d_color::pq_encode(rgb[c]) * 1023)) << (10 * c);
          std::memcpy(f.source_bytes.data() + pixel * 4, &packed, 4);
        }
      }
      f.fill_upload(f.source_upload, f.backbuffers[0]->GetDesc(), f.source_bytes.data(), f.source_footprint);
    };
    const auto capture = [&](const std::string &name, bool settled) {
      const auto output = directory / name;
      fs::create_directories(output);
      // Save the current full outputs before readback/disk latency can cause a
      // subsequent production readiness gap. No frames are advanced here.
      const auto stereo = f.read(f.exported.p), prepared = f.read(f.linear_depth.p);
      const auto selected = f.selected_native_depth();
      save(output / "stereo.rgba16f", stereo);
      save(output / "prepared.rg16f", prepared);
      save(output / "selected-depth.f32", selected);
      save(output / "source.bin", f.source_bytes);
      std::ofstream metadata(output / "metadata.txt");
      const auto zero = f.zero();
      metadata << std::setprecision(17) << "width=" << width << "\nheight=" << height
        << "\nsource_color=" << f.color << "\nsource_format=" << unsigned(f.source_format)
        << "\ndepth_width=" << f.scene->width << "\ndepth_height=" << f.scene->height
        << "\npattern=" << f.scene->pattern << "\nH=" << f.scalar("Sunshine_CameraDepthScale")
        << "\nreference=" << zero[0] << "\nt0=" << zero[1]
        << "\nready=" << f.ready() << "\nblend=" << f.scalar("Sunshine_CameraStrengthBlend")
        << "\nsettled=" << settled << "\nfinal_aa_available=" << final_aa
        << "\nAA=" << (final_aa ? f.scalar("USE_AA") : 0.f)
        << "\nstrength=" << f.scalar("Depth_Adjustment") << '\n';
      // Independent image moments of bright, unoccluded band-interior markers.
      // Keep measured shifts separate from analytical depth-field expectations.
      for (unsigned band = 0; band < 4; ++band) {
        double shifts[2] {};
        const unsigned center = (2 * band + 1) * width / 8, radius = width / 16;
        for (unsigned eye = 0; eye < 2; ++eye) {
          const auto at = [&](unsigned x) { return f.channel(stereo, eye * width + x, height / 2, 0); };
          const double background = .5 * (at(center - radius) + at(center + radius));
          double mass = 0, moment = 0;
          for (unsigned x = center - radius; x < center + radius; ++x) {
            const double weight = std::max(0., double(at(x)) - background);
            mass += weight;
            moment += weight * (x + .5);
          }
          need(mass > 1 && mass < 150, "Distribution interior marker missing/clipped; no reliable pixel disparity");
          shifts[eye] = moment / mass - center;
        }
        const unsigned index = (band + ((f.scene->pattern == 11 || f.scene->pattern == 13) ? 1 : 0)) % 4;
        const double levels[] {.125, .25, .5, .75};
        metadata << "landmark_band=" << band << " raw_oriented=" << levels[index]
          << " left_px=" << shifts[0] << " right_px=" << shifts[1]
          << " binocular_px=" << shifts[0] - shifts[1] << '\n';
      }
      sunshine_parity::uniforms(observed.runtime, effect_file, output / "uniforms.txt");
      sunshine_parity::textures(observed.runtime, effect_file, output / "textures.txt");
      need(metadata.good(), "Cannot record distribution capture metadata");
    };
    const auto settled_pair = [&](const char *name) {
      f.set_float("Depth_Adjustment", 0);
      f.frames_for(800);
      save(directory / (std::string(name) + "-mono.rgba16f"), f.check_current_mono());
      f.set_float("Depth_Adjustment", 100);
      for (unsigned aa = 0; aa < (final_aa ? 2u : 1u); ++aa) {
        f.set_int("USE_AA", aa);
        f.frames_for(800);
        need(f.ready() && f.scalar("Sunshine_CameraStrengthBlend") == 1.f,
          "Distribution snapshot did not regain full ready stereo after readback");
        capture(std::string(name) + "-aa" + std::to_string(aa), true);
      }
    };
    paint(false);
    settled_pair("A");
    f.set_int("USE_AA", 0);
    f.scene->pattern = f.normal ? 13 : 11;
    paint(true);
    f.step();
    capture("A-to-B-one-effect-frame", false);
    f.frames_for(1400);
    const auto permuted_reference = f.reference_from_selected();
    auto sorted_a = initial_reference.oriented, sorted_b = permuted_reference.oriented;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    need(sorted_a == sorted_b && initial_reference.sigma == permuted_reference.sigma,
      "Native scene permutation changed the depth distribution/spread");
    need(initial_reference.center != permuted_reference.center &&
      initial_reference.minimum == .125 && initial_reference.maximum == .75,
      "Native scene permutation lacks its known changed center/full depth range");
    need(f.ready() && f.scalar("Sunshine_CameraDepthScale") == initial_gain,
      "Merely moving the scene recalibrated established gain");
    // The real .5-second zero-plane filter is asymptotic. Allow a bounded
    // settling interval instead of treating1.4seconds as exact convergence.
    const auto settle_deadline = GetTickCount64() + 8000;
    while ((!f.ready() || std::abs(f.zero()[1] - permuted_reference.center) >= 1e-5) &&
           GetTickCount64() < settle_deadline) {
      f.step();
      need(f.scalar("Sunshine_CameraDepthScale") == initial_gain,
        "Zero-plane settling silently changed established gain");
    }
    need(std::abs(f.zero()[1] - permuted_reference.center) < 1e-5,
      "Permuted scene did not independently move the zero plane to the new center");
    settled_pair("B-before-recalibration");

    const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
    using query_t = BOOL (*)(api::effect_runtime *, unsigned *);
    using action_t = BOOL (*)(api::effect_runtime *);
    const auto query = reinterpret_cast<query_t>(GetProcAddress(module, "SunshineGame3DTestQueryAutomatic"));
    const auto action = reinterpret_cast<action_t>(GetProcAddress(module, "SunshineGame3DTestRecalibrate"));
    need(query && action, "Distribution fixture requires the explicit test-only real UI action adapter");
    unsigned mode = 0, flags = 0;
    // Readbacks may age readiness but must not rewrite the action's semantics.
    f.frames_for(800);
    need(query(observed.runtime, &flags) && (flags & 8),
      "Established distribution has no actual Recalibrate action");
    const auto begin = GetTickCount64();
    need(action(observed.runtime) && !action(observed.runtime), "Recalibrate did not accept exactly one pending action");
    do {
      f.step();
      if (GetTickCount64() - begin < 650)
        need(!f.ready() && f.scalar("Sunshine_CameraStrengthBlend") == 0,
          "Distribution recalibration reused preceding samples before a new stable interval");
    } while (!f.ready() && GetTickCount64() - begin < 8000);
    need(f.ready(), "Distribution recalibration never acquired a new native reference");
    f.frames_for(800);
    const float new_gain = f.scalar("Sunshine_CameraDepthScale");
    const auto actual_b = f.reference_from_selected();
    f.require_reference(new_gain, actual_b, "Recalibrated gain differs from independent native distribution statistics");
    if (sunshine_raw_oracle::legacy_center())
      need(std::abs(new_gain - initial_gain) > .1f, "Legacy control did not reproduce center-dependent gain");
    else need(new_gain == initial_gain, "Same native distribution acquired a different scene-spread gain after explicit Recalibrate");
    settled_pair("B-after-recalibration");
    need(sunshine_parity::read_file(f.runtime_directory / "preset.ini") == preset_before,
      "Distribution/recalibration changed the saved preset");
    report << "initial_H=" << initial_gain << " new_H=" << new_gain << " sigma=" << actual_b.sigma
      << " initial_center=" << initial_reference.center << " new_center=" << actual_b.center
      << " raw_min=" << actual_b.minimum << " raw_max=" << actual_b.maximum
      << " native_distribution_identical=1 preset_bytes_preserved=1\n";
    need(report.good(), "Cannot write distribution integration evidence");
    std::printf("PASS real DSV distribution permutation: H %.9g -> %.9g, center %.9g -> %.9g, explicit Recalibrate, original preset bytes preserved; no geometry injection\n",
      initial_gain, new_gain, initial_reference.center, actual_b.center);
  }
}
