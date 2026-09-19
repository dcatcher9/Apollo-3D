// SPDX-License-Identifier: GPL-3.0-only
// Actual game DSV -> integrated selector -> async sampler -> raw policy ->
// original-derived shader. No depth or automatic uniforms are injected.
// The explicit actions flag alone permits the separately named test-only DLL.
#include "test_raw_runtime_fixture.h"
#include "test_raw_reference_oracle.h"
#include "test_raw_distribution_runtime.h"

namespace {
  struct raw_fixture : raw_runtime_fixture {
    fs::path evidence_directory;
    sunshine_raw_oracle::result reference_from_selected() {
      const auto bytes = selected_native_depth();
      std::array<float, sunshine_raw_oracle::cells> raw {};
      for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x) {
        const unsigned px = ((x * 2 + 1) * scene->width) / 64;
        const unsigned py = ((y * 2 + 1) * scene->height) / 36;
        std::memcpy(&raw[y * 32 + x], bytes.data() + (size_t(py) * scene->width + px) * sizeof(float), sizeof(float));
        const double u = (px + .5) / scene->width, v = (py + .5) / scene->height;
        double expected = 0;
        if (scene->pattern >= 10 && scene->pattern <= 13) {
          unsigned band = std::min(3u, unsigned(4 * u));
          if (scene->pattern == 11 || scene->pattern == 13) band = (band + 1) % 4;
          const double levels[] {.125, .25, .5, .75};
          expected = levels[band];
        } else if (scene->pattern == 3 || scene->pattern == 8) expected = .04 + .008 * u + .004 * v;
        else if (scene->pattern == 6 || scene->pattern == 7) expected = std::exp2(-10. + 8. * (.7 * u + .3 * v));
        else throw std::runtime_error("Native oracle has no independent expected draw pattern");
        const float oriented = normal ? 1.f - raw[y * 32 + x] : raw[y * 32 + x];
        require(std::abs(double(oriented) - expected) < 2e-7,
          "Selected native DEPTH does not match the actual known DSV draw at the sampler's texel positions");
      }
      const auto result = sunshine_raw_oracle::calculate(raw, normal);
      std::printf("MEASURE independent native DEPTH oracle: model=%s mean=%.12g sigma=%.12g center=%.12g range=[%.12g,%.12g] expected_H=%.12g\n",
        sunshine_raw_oracle::legacy_center() ? "legacy-center" : "scene-sigma", result.mean, result.sigma,
        result.center, result.minimum, result.maximum, result.expected_gain);
      return result;
    }
    void require_reference(float H, const sunshine_raw_oracle::result &reference, const char *message) {
      require(std::abs(double(H) - reference.expected_gain) < .002 && std::abs(zero()[1] - reference.center) < 1e-5, message);
    }
    void check_manual_handover_scale(float H) {
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
      const auto select_manual = reinterpret_cast<select_t>(GetProcAddress(module, "SunshineDepthTestSelectManual"));
      require(select_manual, "Automatic handover requires the test add-on's UI-selection adapter");
      // The preceding 4K readback may create a legitimate capture-cadence gap.
      // Settle ordinary rendered frames before attributing any reentry to pinning.
      frames_for(1000);
      const auto binding = selected_binding();
      require(ready() && std::isfinite(H) && H > 0 && binding.handle && scalar("Sunshine_CameraStrengthBlend") == 1.f,
        "Automatic handover requires a calibrated current source");
      const auto check = [&] {
        require(ready() && scalar("Sunshine_CameraDepthScale") == H && selected_binding() == binding &&
          scalar("Sunshine_CameraStrengthBlend") == 1.f,
          "Pinning or unpinning the same source restarted Automatic calibration, changed H or reduced strength");
      };
      require(select_manual(observed.runtime, reinterpret_cast<std::uint64_t>(scene->texture.p)),
        "Could not pin Automatic's calibrated source");
      check();
      const auto pinned_until = GetTickCount64() + 1200;
      do { step(); check(); } while (GetTickCount64() < pinned_until);
      require(select_manual(observed.runtime, 0), "Could not release Automatic's calibrated source");
      check();
      const auto released_until = GetTickCount64() + 1500;
      do { step(); check(); } while (GetTickCount64() < released_until);
      std::printf("PASS actual Automatic same-source pin/unpin preserves readiness, shader binding and H=%.9g\n", H);
    }
    void check_legacy_calibration_inactive() {
      if (!sunshine_camera_fixture::flag("SUNSHINE_GAME3D_LEGACY_CALIBRATION_TEST")) return;
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      using query_t = BOOL (*)(api::effect_runtime *, unsigned *);
      const auto query = reinterpret_cast<query_t>(GetProcAddress(module, "SunshineDepthTestLegacyCalibrationState"));
      require(query, "Legacy calibration ownership test requires its read-only test-addon query");
      unsigned requested = UINT_MAX, entries = UINT_MAX;
      require(query(observed.runtime, &requested, &entries), "Cannot inspect legacy calibration ownership");
      std::printf("MEASURE Game3D legacy calibration: requested=%u cached_sources=%u\n", requested, entries);
      require(requested == 0 && entries == 0,
        "Game3D performed unrequested legacy percentile calibration");
      std::puts("PASS Game3D has no legacy percentile calibration cache or request");
    }
    void check_preparation(float H, unsigned pattern) {
      const auto pixels = read(linear_depth.p);
      const auto desc = linear_depth->GetDesc();
      float minimum_field = INFINITY, maximum_field = -INFINITY;
      size_t below_search_floor = 0;
      for (size_t i = 0; i < size_t(desc.Width) * desc.Height; ++i) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, pixels.data() + i * 4, 2);
        const float field = half_float(bits);
        require(std::isfinite(field), "Automatically prepared convergence contains nonfinite values");
        minimum_field = std::min(minimum_field, field);
        maximum_field = std::max(maximum_field, field);
        below_search_floor += field < -1.5f;
      }
      std::printf("MEASURE prepared convergence pattern=%u min=%.9g max=%.9g below_minus1_5_fraction=%.9g; BEFORE final smoothing/boost/search clamp, not actual clamped-pixel count\n",
        pattern, minimum_field, maximum_field, double(below_search_floor) / (double(desc.Width) * desc.Height));
      for (unsigned y = 1; y <= 3; ++y) for (unsigned x = 1; x <= 4; ++x) {
        const auto tx = unsigned(desc.Width * x / 5), ty = unsigned(desc.Height * y / 4);
        const float u = (tx + .5f) / float(desc.Width), v = (ty + .5f) / desc.Height;
        const float t = pattern == 3 || pattern == 8 ? .04f + .008f * u + .004f * v :
          float(std::exp2(-10.0 + 8.0 * (.7 * u + .3 * v)));
        std::uint16_t half = 0;
        std::memcpy(&half, pixels.data() + (size_t(ty) * desc.Width + tx) * 4 + 2, 2);
        require(std::abs(half_float(half) - 1.f / (1.f + H * t)) < .002f,
          "Real shader did not use automatically sampled oriented raw depth");
      }
    }
    void record_evidence(float H, const std::array<float, 2> &initial_zero) {
      if (evidence_directory.empty()) return;
      require(!fs::exists(evidence_directory), "Automatic evidence requires a fresh output directory");
      fs::create_directories(evidence_directory);
      const bool final_aa = final_aa_available();
      const auto save = [&](const char *name, const std::vector<std::uint8_t> &bytes) {
        sunshine_parity::write_bytes(evidence_directory / name, bytes.data(), bytes.size());
      };
      std::ofstream metadata(evidence_directory / "metadata.txt");
      metadata << std::setprecision(12) << "schema=automatic-runtime-snapshot-1\n"
               << "source_width=" << width << "\nsource_height=" << height << "\nsource_color=" << color
               << "\nsource_format=" << unsigned(source_format) << "\nnormal=" << normal
               << "\nscene_width=" << scene->width << "\nscene_height=" << scene->height
               << "\nscene_pattern=" << scene->pattern << "\nH=" << H
               << "\nreference=" << initial_zero[0] << "\nt0=" << initial_zero[1]
               << "\nreference_model=" << (sunshine_raw_oracle::legacy_center() ? "legacy-center" : "scene-sigma")
               << "\nselected_depth=actual_shader_bound_pre_clear_backup\nnative_scene_depth=post_draw_clear_allocation"
               << "\nstrength=100\nsharpen=0\nfinal_aa_available=" << final_aa << '\n';
      save("source.bin", source_bytes);
      save("scene-depth.f32", read(scene->texture.p, D3D12_RESOURCE_STATE_DEPTH_WRITE));
      // The native allocation was cleared after drawing. Preserve that artifact
      // and separately retain the actual shader-bound pre-clear backup.
      save("selected-depth.f32", selected_native_depth());
      set_int("USE_AA", 0);
      set_float("Sharpen_Power", 0);
      set_float("Depth_Adjustment", 0);
      measured_frames();
      save("mono.sbs", check_current_mono());
      set_float("Depth_Adjustment", 100);
      for (unsigned aa = 0; aa < (final_aa ? 2u : 1u); ++aa) {
        set_int("USE_AA", aa);
        // Full 4K readback and disk writes can exceed the production reentry
        // gap bound. Resume continuous presents before recording stereo.
        frames_for(800);
        measured_frames();
        std::printf("MEASURE automatic snapshot aa=%u ready=%d H=%.9g t0=%.9g blend=%.9g\n",
          aa, ready(), scalar("Sunshine_CameraDepthScale"), zero()[1], scalar("Sunshine_CameraStrengthBlend"));
        require(ready() && scalar("Sunshine_CameraDepthScale") == H && zero() == initial_zero &&
                scalar("Sunshine_CameraStrengthBlend") == 1.f,
          "Automatic evidence captured unsettled or different scene geometry");
        save(aa ? "stereo-aa1.sbs" : "stereo-aa0.sbs", read(exported.p));
        save(aa ? "prepared-aa1.rg16f" : "prepared-aa0.rg16f", read(linear_depth.p));
      }
      set_int("USE_AA", 0);
      sunshine_parity::uniforms(observed.runtime, effect_file, evidence_directory / "uniforms.txt");
      sunshine_parity::textures(observed.runtime, effect_file, evidence_directory / "textures.txt");
      require(metadata.good(), "Cannot write automatic evidence metadata");
      std::puts("PASS actual automatic source/depth/preparation/mono/stereo snapshots captured for supported final-AA states without injected depth or camera values");
    }
    void exercise_inactive_depth_work() {
      const auto directory = runtime_directory / "inactive-depth-evidence";
      require(!fs::exists(directory), "Inactive-depth evidence requires a fresh directory");
      fs::create_directories(directory);
      const bool final_aa = final_aa_available();
      com_ptr<ID3D12Resource> reconstructed, smoothed;
      find_texture("texReconBuffer", reconstructed, 0, DXGI_FORMAT_R16_FLOAT);
      find_texture("texSmooth", smoothed, 0, DXGI_FORMAT_R16_FLOAT);
      const auto save = [&](const std::string &name) {
        const auto output = read(exported.p);
        sunshine_parity::write_bytes(directory / (name + ".sbs"), output.data(), output.size());
        const auto recon = read(reconstructed.p), smooth = read(smoothed.p);
        sunshine_parity::write_bytes(directory / (name + ".reconstruction.r16f"), recon.data(), recon.size());
        sunshine_parity::write_bytes(directory / (name + ".smoothing.r16f"), smooth.data(), smooth.size());
        return output;
      };
      for (unsigned aa = 0; aa < (final_aa ? 2u : 1u); ++aa) {
        const std::string prefix = "aa" + std::to_string(aa);
        set_int("USE_AA", aa);
        set_int("Depth_Map_View", 0);
        set_float("Depth_Adjustment", 100);
        frames_for(800);
        require(ready() && scalar("Sunshine_CameraStrengthBlend") == 1.f, "Initial inactive-depth test geometry is not ready");
        const auto active = save(prefix + "-active");
        set_float("Depth_Adjustment", 0);
        frames_for(800);
        check_current_mono();
        save(prefix + "-zero-strength");
        // A ready Normal Depth diagnostic remains visible at zero strength.
        set_int("Depth_Map_View", 2);
        frames_for(800);
        require(ready(), "Normal Depth diagnostic unexpectedly lost source readiness");
        save(prefix + "-normal-depth");
        // Finish all readbacks while the diagnostic is still selected, then
        // restore continuous reentry timing before testing ONE resumed frame.
        frames_for(800);
        require(scalar("Sunshine_CameraStrengthBlend") == 1.f, "Diagnostic readback did not recover continuous rendering");
        set_int("Depth_Map_View", 0);
        set_float("Depth_Adjustment", 100);
        step();
        require(ready() && scalar("Sunshine_CameraStrengthBlend") == 1.f,
          "First resumed stereo frame did not have a valid full-strength source");
        const auto resumed = save(prefix + "-first-resumed");
        require(resumed == active, "First stereo frame after diagnostic used stale spatial-depth work");
      }
      set_int("USE_AA", 0);
      set_int("Depth_Map_View", 0);
      set_float("Depth_Adjustment", 100);
      frames_for(800);
      std::puts("PASS inactive spatial-depth work: mono within HDR export precision, ready Normal Depth diagnostic, first resumed stereo byte-identical for supported final-AA states; full buffers saved");
    }
    void measure_gpu_cost(float H) {
      const auto directory = runtime_directory / "gpu-cost";
      require(!fs::exists(directory), "GPU-cost evidence requires a fresh directory");
      fs::create_directories(directory);
      const bool final_aa = final_aa_available();
      std::ofstream samples(directory / "samples.csv"), cases(directory / "cases.csv");
      samples << "case,sample,begin,end,frequency,gpu_ms,H,t0,blend\n" << std::setprecision(12);
      cases << "case,warmup,samples,median_ms,min_ms,max_ms,H,t0,blend\n" << std::setprecision(12);
      com_ptr<ID3D12QueryHeap> heap;
      com_ptr<ID3D12Resource> readback;
      D3D12_QUERY_HEAP_DESC description {};
      description.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
      description.Count = 2;
      checked(game->CreateQueryHeap(&description, IID_PPV_ARGS(heap.put())), "Create actual-effect timing query heap");
      buffer(readback, 2 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK);
      auto *queue = reinterpret_cast<ID3D12CommandQueue *>(observed.runtime->get_command_queue()->get_native());
      UINT64 frequency {};
      checked(queue->GetTimestampFrequency(&frequency), "Read actual-effect timing frequency");
      require(frequency && !sunshine_parity::armed && !sunshine_parity::queries,
        "Invalid or overlapping actual-effect timing interval");
      struct clear_t {
        ~clear_t() {
          sunshine_parity::armed = false;
          sunshine_parity::queries = nullptr;
          sunshine_parity::query_readback = nullptr;
        }
      } clear;
      sunshine_parity::queries = heap.p;
      sunshine_parity::query_readback = readback.p;
      sunshine_parity::write_bytes(directory / "source.bin", source_bytes.data(), source_bytes.size());
      for (unsigned content = 0; content < 2; ++content) {
        scene->pattern = content ? (normal ? 7 : 6) : (normal ? 8 : 3);
        frames_for(800);
        const auto reference = reference_from_selected();
        // Real zero-plane adaptation, never an injected or refitted camera value.
        const auto deadline = GetTickCount64() + 12000;
        while (std::abs(double(zero()[1]) - reference.center) > 1e-8 && GetTickCount64() < deadline) step();
        require(ready() && scalar("Sunshine_CameraDepthScale") == H &&
          std::abs(double(zero()[1]) - reference.center) <= 1e-8,
          "Actual scene did not settle its zero plane before GPU timing");
        const auto depth = selected_native_depth();
        const std::string scene_name = content ? "broad" : "narrow";
        sunshine_parity::write_bytes(directory / (scene_name + ".depth.f32"), depth.data(), depth.size());
        for (unsigned aa = 0; aa < (final_aa ? 2u : 1u); ++aa) for (unsigned mode = 0; mode < 3; ++mode) {
          const std::string label = scene_name + "-aa" + std::to_string(aa) +
            (mode == 0 ? "-stereo" : mode == 1 ? "-mono" : "-normal-depth");
          set_int("USE_AA", aa);
          set_int("Depth_Map_View", mode == 2 ? 2 : 0);
          set_float("Depth_Adjustment", mode == 0 ? 100.f : 0.f);
          // File readback can interrupt the production 500 ms reentry. No disk
          // writes or large texture readbacks occur in the following GPU samples.
          frames_for(800);
          for (unsigned frame = 0; frame < 16; ++frame) step();
          std::array<double, 64> durations {};
          for (unsigned sample = 0; sample < durations.size(); ++sample) {
            sunshine_parity::started = sunshine_parity::finished = sunshine_parity::invalid = false;
            sunshine_parity::armed = true;
            step();
            sunshine_parity::armed = false;
            require(sunshine_parity::started && sunshine_parity::finished && !sunshine_parity::invalid,
              "GPU timing did not bracket one complete selected technique");
            require(ready() && scalar("Sunshine_CameraDepthScale") == H &&
              scalar("Sunshine_CameraStrengthBlend") == 1.f &&
              std::abs(double(zero()[1]) - reference.center) <= 1e-8,
              "GPU timing included a different or unsettled automatic geometry");
            void *mapped {};
            const D3D12_RANGE range {0, 2 * sizeof(UINT64)};
            checked(readback->Map(0, &range, &mapped), "Read actual-effect GPU timestamps");
            UINT64 ticks[2] {};
            std::memcpy(ticks, mapped, sizeof(ticks));
            const D3D12_RANGE no_write {0, 0};
            readback->Unmap(0, &no_write);
            require(ticks[1] > ticks[0], "Actual-effect GPU timestamp order is invalid");
            durations[sample] = 1000. * double(ticks[1] - ticks[0]) / double(frequency);
            samples << label << ',' << sample << ',' << ticks[0] << ',' << ticks[1] << ',' << frequency << ','
                    << durations[sample] << ',' << H << ',' << zero()[1] << ',' << scalar("Sunshine_CameraStrengthBlend") << '\n';
          }
          std::sort(durations.begin(), durations.end());
          const double median = .5 * (durations[31] + durations[32]);
          cases << label << ",16,64," << median << ',' << durations.front() << ',' << durations.back()
                << ',' << H << ',' << zero()[1] << ',' << scalar("Sunshine_CameraStrengthBlend") << '\n';
          const auto pixels = read(exported.p);
          sunshine_parity::write_bytes(directory / (label + ".sbs"), pixels.data(), pixels.size());
          std::printf("MEASURE actual-effect GPU cost case=%s median_ms=%.9g min_ms=%.9g max_ms=%.9g samples=64\n",
            label.c_str(), median, durations.front(), durations.back());
        }
      }
      require(samples.good() && cases.good(), "Cannot persist actual-effect GPU cost evidence");
      std::puts("PASS GPU-cost sampling covers effects_begin through the selected complete technique; real selector/policy, no injected depth or geometry");
    }
    void startup_clear_center() {
      // A real second game draw writes an endpoint only in the central sample
      // area. Keep the full viewport/source identity and the rest of the scene
      // valid, so depth selection can continue sampling this same allocation.
      // A partial ClearDepthStencilView here would let depth preservation copy
      // the valid image before the clear, defeating the intended input.
      scene->before_final_clear = [&] {
        const D3D12_RECT center {LONG(scene->width * 35 / 100), LONG(scene->height * 35 / 100),
          LONG(scene->width * 65 / 100), LONG(scene->height * 65 / 100)};
        commands->RSSetScissorRects(1, &center);
        struct { float w, h; unsigned pattern; } constants {float(scene->width), float(scene->height), 0};
        commands->SetGraphicsRoot32BitConstants(0, 3, &constants, 0);
        commands->DrawInstanced(3, 1, 0, 0);
        const D3D12_RECT full {0, 0, LONG(scene->width), LONG(scene->height)};
        commands->RSSetScissorRects(1, &full);
      };
    }
    void check_startup_depth() {
      com_ptr<ID3D12Resource> selected;
      observed.runtime->enumerate_texture_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char name[256] {};
        runtime->get_texture_variable_name(variable, name);
        if (!named(name, "DepthBufferTex")) return;
        api::resource_view view {};
        runtime->get_texture_binding(variable, &view, nullptr);
        require(view.handle, "Startup fixture lost the actual selected DEPTH binding");
        const auto resource = runtime->get_device()->get_resource_from_view(view);
        require(resource.handle, "Startup selected DEPTH binding has no actual resource");
        auto *native = reinterpret_cast<ID3D12Resource *>(resource.handle);
        native->AddRef();
        selected.reset();
        selected.p = native;
      });
      require(selected.p != nullptr, "Startup fixture cannot inspect selected depth pixels");
      const auto desc = selected->GetDesc();
      require(desc.Width == scene->width && desc.Height == scene->height &&
              (desc.Format == DXGI_FORMAT_R32_FLOAT || desc.Format == DXGI_FORMAT_R32_TYPELESS),
        "Startup fixture selected the flat decoy or an unexpected depth format");
      const auto pixels = read(selected.p);
      const auto pixel = [&](unsigned x, unsigned y) {
        float value = 0;
        std::memcpy(&value, pixels.data() + (size_t(y) * desc.Width + x) * sizeof(float), sizeof(float));
        return value;
      };
      for (unsigned y = 7; y < 11; ++y) for (unsigned x = 14; x < 18; ++x)
        require(pixel(((x * 2 + 1) * scene->width) / 64, ((y * 2 + 1) * scene->height) / 36) == 0.f,
          "Actual selected startup depth did not contain the drawn clear central samples");
      for (unsigned y : {1u, 4u}) for (unsigned x : {1u, 4u}) {
        const auto value = pixel(scene->width * x / 5, scene->height * y / 5);
        require(std::isfinite(value) && value > 0.f && value < 1.f,
          "Startup unsuitable center also destroyed the surrounding valid scene depth");
      }
      std::puts("PASS startup input: actual selected DEPTH has 16 endpoint center samples and valid surrounding scene pixels");
    }
    void run_startup_recovery(bool replace) {
      const auto begin = GetTickCount64();
      const auto deadline = begin + 15000;
      const auto initialized = [&] {
        // Production logging confirms an actual accepted capture started the
        // reference. Never seed a camera/depth uniform or call a test adapter.
        std::ifstream input(runtime_directory / "ReShade.log");
        const std::string log((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        const std::string marker = "Sunshine 3D raw automation: calibrating; samples=";
        const auto at = log.find(marker);
        if (at == std::string::npos) return false;
        const auto count = std::strtoul(log.c_str() + at + marker.size(), nullptr, 10);
        return count > 0;
      };
      while (!initialized() && GetTickCount64() < begin + 5000) {
        step();
        require(!ready(), "Startup reference committed before the fixture could present unsuitable input");
      }
      require(initialized() && !ready(), "Startup fixture did not observe the first actual uncommitted reference sample");
      if (replace) {
        // Allocate while the old resource is still alive to force a distinct
        // native address, then release it. This change precedes any committed H.
        auto replacement = target(width / 2, height / 2, 1, normal ? 7 : 6, false, true);
        replacement->clear_depth = normal ? 1.f : 0.f;
        require(replacement->texture.p != scene->texture.p, "Startup replacement reused the live source address");
        scene = std::move(replacement);
      }
      startup_clear_center();
      const auto invalid_begin = GetTickCount64();
      unsigned invalid_presents = 0;
      do {
        step();
        ++invalid_presents;
        require(!ready() && scalar("Sunshine_CameraStrengthBlend") == 0.f,
          "Unsuitable startup center initialized stereo or retained nonzero strength");
      } while (GetTickCount64() - invalid_begin < 5700);
      check_startup_depth();
      const auto mono = check_current_mono();
      require(GetTickCount64() - invalid_begin > 5000 && !ready(),
        "Startup fixture did not remain uncalibrated past the former five-second deadline");
      std::printf("PASS startup wait%s: %u actual presents over %llu ms, not ready and current-source mono beyond five seconds\n",
        replace ? " after pre-calibration source replacement" : " on the same source", invalid_presents,
        static_cast<unsigned long long>(GetTickCount64() - invalid_begin));

      scene->before_final_clear = {};
      scene->pattern = normal ? 8 : 3;
      const auto stable_begin = GetTickCount64();
      unsigned stable_presents = 0;
      do {
        step();
        ++stable_presents;
        if (GetTickCount64() - stable_begin < 700)
          require(!ready() && scalar("Sunshine_CameraStrengthBlend") == 0.f,
            "Startup recovery reused old samples before a fresh stable reference window");
      } while (!ready() && GetTickCount64() < deadline);
      require(ready(), replace ?
        "Automatic startup did not recover after an uncommitted source replacement and invalid center" :
        "Automatic startup remained permanently timed out after valid scene depth returned");
      const auto H = scalar("Sunshine_CameraDepthScale");
      const auto plane = zero();
      const auto reference = reference_from_selected();
      require_reference(H, reference, "Startup recovered H/t0 from stale input instead of the new stable scene");
      frames_for(800);
      require(ready() && scalar("Sunshine_CameraDepthScale") == H && scalar("Sunshine_CameraStrengthBlend") == 1.f,
        "Recovered startup did not reach stable full user strength");
      check_preparation(H, scene->pattern);
      require(image_difference(mono, read(exported.p)) > .002f, "Automatic startup recovery remained mono");
      require(GetTickCount64() < deadline, "Startup recovery exceeded its bounded 15-second scenario");
      std::printf("PASS production Automatic startup recovery%s: fresh stable window, %u presents, H=%.9g t0=%.9g, stereo resumed without UI action or injected uniforms\n",
        replace ? " after source replacement" : " on unchanged source", stable_presents, H, plane[1]);
    }
    void run_performance(float H, float initial_t0) {
      const auto stereo = read(exported.p);
      scene->pattern = normal ? 7 : 6;
      frames_for(1400);
      require(ready() && scalar("Sunshine_CameraDepthScale") == H && std::abs(zero()[1] - initial_t0) > .002f,
        "Performance Mode specialized dynamic H/t0 or lost moving-scene readiness");
      check_preparation(H, scene->pattern);
      render_tracked_depth = {};
      frames_for(100);
      require(!ready() && scalar("Sunshine_CameraStrengthBlend") == 0.f, "Performance Mode retained stereo readiness after depth loss");
      const auto mono = check_current_mono();
      require(image_difference(mono, stereo) > .002f, "Performance Mode rendered no actual stereo before losing depth");
      const auto original = source_bytes;
      const unsigned pixel_bytes = bytes_per_pixel(source_format);
      for (unsigned y = 0; y < height; ++y) {
        auto first = source_bytes.begin() + size_t(y) * width * pixel_bytes;
        std::rotate(first, first + size_t(width / 3) * pixel_bytes, first + size_t(width) * pixel_bytes);
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      frames_for(100);
      require(!ready() && scalar("Sunshine_CameraStrengthBlend") == 0.f, "Changing color without depth revived Performance Mode stereo");
      require(image_difference(mono, check_current_mono()) > .01f, "Missing-depth fallback reused an old color frame");
      source_bytes = original;
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      render_tracked_depth = [&] { if (scene) draw(*scene); };
      frames_for(1800);
      require(ready() && scalar("Sunshine_CameraDepthScale") == H && scalar("Sunshine_CameraStrengthBlend") == 1.f,
        "Performance Mode failed recovery with fixed reference and dynamic strength blend");
      check_preparation(H, scene->pattern);
      measured_frames();
      require(image_difference(mono, read(exported.p)) > .002f, "Performance Mode recovery remained mono");
      std::puts("PASS actual Performance Mode: precompiled artistic settings, dynamic source-owned readiness/H/t0, moving screen plane, current-color mono on depth loss and stereo reentry");
      render_tracked_depth = {};
    }
    void exercise_recalibration(float previous_H, const std::vector<std::uint8_t> &mono) {
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      using query_t = BOOL (*)(api::effect_runtime *, unsigned *);
      using action_t = BOOL (*)(api::effect_runtime *);
      const auto query = reinterpret_cast<query_t>(GetProcAddress(module, "SunshineGame3DTestQueryAutomatic"));
      const auto action = reinterpret_cast<action_t>(GetProcAddress(module, "SunshineGame3DTestRecalibrate"));
      require(query && action, "Explicit test-only addon is missing automatic action adapters");
      unsigned mode = 0, flags = 0;
      require(query(observed.runtime, &flags) && (flags & 8) && !(flags & 4),
        "Suspended replacement source did not offer a real recalibration action");
      // Old queued packets contain the preceding linear pattern. The new
      // reference must come from captures made after this different pattern.
      scene->pattern = normal ? 7 : 6;
      const auto begin = GetTickCount64();
      require(action(observed.runtime) && !action(observed.runtime), "Recalibration was not accepted exactly once while pending");
      require(query(observed.runtime, &flags) && !(flags & (4 | 8)),
        "Pending recalibration remained ready or accepted another action");
      bool observed_wait = false;
      do {
        step();
        if (GetTickCount64() - begin < 650) {
          observed_wait = true;
          require(!ready() && scalar("Sunshine_CameraStrengthBlend") == 0.f,
            "Recalibration reused an old packet before a new reference window could complete");
          require(read(exported.p) == mono, "Recalibration waiting did not duplicate current source eyes");
        }
      } while (GetTickCount64() - begin < 600);
      require(observed_wait, "Action test did not observe the bounded recalibration wait");
      const auto deadline = GetTickCount64() + 8000;
      while (!ready() && GetTickCount64() < deadline) step();
      require(ready(), "Explicit recalibration did not recover the replacement source");
      const auto H = scalar("Sunshine_CameraDepthScale");
      const auto reference = reference_from_selected();
      require_reference(H, reference, "Recalibration reference came from stale samples or silently retained the old H/t0");
      require(std::abs(H - previous_H) > 1.f, "Recalibration retained the old gain after a materially different distribution");
      frames_for(800);
      require(query(observed.runtime, &flags) && (flags & 12) == 12 &&
              ready() && scalar("Sunshine_CameraDepthScale") == H && scalar("Sunshine_CameraStrengthBlend") == 1.f,
        "Recalibration did not return the actual automatic controls to ready");
      check_preparation(H, scene->pattern);
      require(image_difference(mono, read(exported.p)) > .002f, "Recalibrated real warp did not resume stereo");
      std::printf("PASS TEST-ONLY actual Recalibrate action: once while pending, mono during new capture window, replacement H %.9g -> %.9g t0=%.9g; no stale reference reuse\n",
        previous_H, H, zero()[1]);
    }
    void exercise_source_recovery(float previous_H, const std::vector<std::uint8_t> &mono) {
      const bool expect_recovery = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_EXPECT_RECOVERY");
      const auto started = GetTickCount64();
      const auto deadline = started + (expect_recovery ? 12000 : 3500);
      while (!ready() && GetTickCount64() < deadline) step();
      const auto elapsed = GetTickCount64() - started;
      if (!expect_recovery) {
        require(!ready() && scalar("Sunshine_CameraDepthScale") == previous_H,
          "Frozen control unexpectedly recovered the lasting replacement source");
        require(read(exported.p) == mono, "Frozen suspended control stopped presenting current mono");
        std::printf("CONTROL persistent replacement remains suspended without Recalibrate: H=%.9g waited_ms=%llu\n",
          previous_H, static_cast<unsigned long long>(elapsed));
        return;
      }
      require(ready(), "Persistent qualified replacement did not recover without Recalibrate");
      const float H = scalar("Sunshine_CameraDepthScale");
      const auto reference = reference_from_selected();
      require_reference(H, reference, "Automatic source recovery used stale calibration instead of the replacement depth");
      require(std::abs(H - previous_H) > 1.f,
        "Different replacement scene silently inherited the original source's gain");
      frames_for(800);
      require(ready() && scalar("Sunshine_CameraDepthScale") == H && scalar("Sunshine_CameraStrengthBlend") == 1.f,
        "Automatic source recovery did not settle at the user strength");
      check_preparation(H, scene->pattern);
      const auto stereo = read(exported.p);
      require(image_difference(mono, stereo) > .002f, "Recovered replacement claimed readiness without actual stereo");
      const auto directory = runtime_directory / "source-recovery-evidence";
      fs::create_directories(directory);
      sunshine_parity::write_bytes(directory / "recovered.sbs", stereo.data(), stereo.size());
      const auto depth = selected_native_depth();
      sunshine_parity::write_bytes(directory / "selected-depth.f32", depth.data(), depth.size());
      std::ofstream report(directory / "result.txt");
      report << std::setprecision(12) << "previous_H=" << previous_H << "\nreplacement_H=" << H
             << "\nexpected_H=" << reference.expected_gain << "\nt0=" << zero()[1]
             << "\nwaited_ms=" << elapsed << "\nready=1\nblend=1\nmanual_action=0\n";
      require(report.good(), "Cannot write source-recovery evidence");
      std::printf("PASS actual lasting replacement recovers without Recalibrate: fresh H %.9g -> %.9g, t0=%.9g, waited_ms=%llu; native depth oracle and stereo pass\n",
        previous_H, H, zero()[1], static_cast<unsigned long long>(elapsed));
    }
    void run(bool reversed) {
      normal = !reversed;
      const bool automatic = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC");
      const bool performance = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_PERFORMANCE");
      const bool actions = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST");
      const bool startup = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_STARTUP_TEST");
      const bool source_return = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_RETURN_TEST");
      const bool source_recovery = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_REPLACE_TEST");
      const bool distribution = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_DISTRIBUTION_TEST");
      create_pipeline();
      scene = target(width / 2, height / 2, 1, distribution ? (normal ? 12 : 10) :
        startup ? (normal ? 7 : 6) : (normal ? 8 : 3), false, true);
      scene->clear_depth = normal ? 1.f : 0.f;
      decoy = target(width, height, 200, 0);
      render_tracked_depth = [&] { if (decoy) draw(*decoy); if (scene) draw(*scene); };
      const auto load_until = GetTickCount64() + 45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64() < load_until) step();
      require(observed.runtime && observed.renders, "Raw automation shader failed to compile; inspect isolated ReShade.log");
      const bool final_aa = final_aa_available();
      find_texture("DoubleTex", exported, width * 2, DXGI_FORMAT_R16G16B16A16_FLOAT);
      find_texture("texzBufferN_P", linear_depth, 0, DXGI_FORMAT_R16G16_FLOAT);
      check_unified_addon();
      require(!observed.inject, "Raw automation fixture injected depth");
      bool runtime_performance = false;
      require(reshade::get_config_value(observed.runtime, "GENERAL", "PerformanceMode", runtime_performance) && runtime_performance == performance,
        "Actual runtime Performance Mode differs from the requested fixture");
      if (performance) {
        for (const char *name : {"Sunshine_CameraDepthReady", "Sunshine_CameraCoordinateBasis", "Sunshine_CameraProjection",
               "Sunshine_CameraDepthScale", "Sunshine_CameraConvergence", "Sunshine_CameraStrengthBlend"}) {
          char source[64] {};
          require(observed.runtime->get_annotation_string_from_uniform_variable(uniform(name), "source", source) && std::strcmp(source, "sunshine_camera") == 0,
            "Performance Mode removed or specialized a required dynamic camera source uniform");
        }
      }
      if (!performance) {
        set_int("WP", 0);
        set_int("Depth_Map_View", 0);
        set_int("Range_Boost", 0);
        set_int("Depth_Map", normal ? 0 : 1);
        set_float("Depth_Map_Adjust", 40);
        set_int("Auto_Scaler_Adjust", 0);
        set_int("USE_AA", 0);
        set_float("Sharpen_Power", 0);
        set_float("Auto_Depth_Adjust", 0);
        set_float("ZPD_OverShoot", 0);
        set_float("Compatibility_Power", 1);
        for (const auto name : {"Depth_Map_Flip", "Eye_Swap", "DB_AutoFit", "Flip_HV_Scale"}) {
          const auto control = uniform(name, false);
          if (control.handle) observed.runtime->set_uniform_value_bool(control, false);
        }
        const float zero_vector[2] {0, 0}, one_vector[2] {1, 1};
        for (const auto name : {"DLSS_FSR_Offset", "Image_Position_Adjust"}) {
          const auto control = uniform(name, false);
          if (control.handle) observed.runtime->set_uniform_value_float(control, zero_vector, 2);
        }
        for (const auto name : {"Horizontal_and_Vertical", "Horizontal_and_Vertical_TL"}) {
          const auto control = uniform(name, false);
          if (control.handle) observed.runtime->set_uniform_value_float(control, one_vector, 2);
        }
        const unsigned zero_uint[2] {0, 0};
        observed.runtime->set_uniform_value_uint(uniform("Starting_Resolution"), zero_uint, 2);
        if (automatic) {
          // Deliberately conflicting saved manual controls must not disable the
          // owned Automatic path or move its sampled reference to different UVs.
          set_int("WP", 1);
          set_int("Auto_Scaler_Adjust", 1);
          set_int("Range_Boost", 4);
          set_float("Auto_Depth_Adjust", .8f);
          set_float("ZPD_OverShoot", .75f);
          set_float("AR_Side_Shrink", .2f);
          const float offset[2] {.02f, -.01f};
          observed.runtime->set_uniform_value_float(uniform("DLSS_FSR_Offset"), offset, 2);
          observed.runtime->set_uniform_value_bool(uniform("Depth_Map_Flip"), true);
          const auto autofit = uniform("DB_AutoFit", false);
          if (autofit.handle) observed.runtime->set_uniform_value_bool(autofit, true);
        }
        set_float("Depth_Adjustment", 100.f);
      }
      if (startup) run_startup_recovery(sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_STARTUP_REPLACE"));
      const auto until = GetTickCount64() + 18000;
      while (!ready() && GetTickCount64() < until) step();
      require(ready(), "Actual selected DSV did not initialize automatic raw controls");
      const auto selected = selected_description();
      require(selected.Width == scene->width && selected.Height == scene->height,
        "Automatic raw calibration used the high-draw flat decoy");
      int basis = -1;
      observed.runtime->get_uniform_value_int(uniform("Sunshine_CameraCoordinateBasis"), &basis, 1);
      require(basis == 1, "Raw automation mislabeled its basis as recovered camera data");
      float projection[2] {};
      observed.runtime->get_uniform_value_float(uniform("Sunshine_CameraProjection"), projection, 2);
      require(projection[0] == (normal ? 1.f : 0.f) && projection[1] == (normal ? -1.f : 1.f),
        "Clear-derived normal/reversed orientation was not applied");
      const float H = scalar("Sunshine_CameraDepthScale");
      const auto reference = reference_from_selected();
      require_reference(H, reference, "Automatic reference differs from independently measured native DEPTH statistics");
      const auto initial_zero = zero();
      require(std::abs(initial_zero[0] - .05f) < 1e-7 && std::abs(initial_zero[1] - reference.center) < 1e-5,
        "Automatic initial screen plane differs from the sampled reference");
      if (distribution) {
        sunshine_raw_distribution_fixture::run(*this, H, reference, width, height);
        render_tracked_depth = {};
        return;
      }
      frames_for(automatic ? 800 : 100);
      if (automatic) {
        require(scalar("Sunshine_CameraStrengthBlend") == 1.f, "Automatic recovery never reached full user strength");
        if (!performance) {
          require(scalar("ZPD_OverShoot") == .75f && scalar("AR_Side_Shrink") == .2f,
            "Automatic setup rewrote saved manual controls");
        }
      }
      check_preparation(H, normal ? 8 : 3);
      check_legacy_calibration_inactive();
      if (actions) check_manual_handover_scale(H);
      if (sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_GPU_COST")) {
        measure_gpu_cost(H);
        render_tracked_depth = {};
        return;
      }
      if (sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_INACTIVE_TEST")) exercise_inactive_depth_work();
      record_evidence(H, initial_zero);
      if (performance) {
        run_performance(H, initial_zero[1]);
        return;
      }
      set_float("Depth_Adjustment", 0);
      measured_frames();
      const auto mono = read(exported.p);
      set_float("Depth_Adjustment", 100);
      measured_frames();
      const auto stereo = read(exported.p);
      require(image_difference(mono, stereo) > .002f, "Automatically driven actual warp produced no stereo difference");
      if (automatic) {
        set_float("Depth_Adjustment", 0);
        set_int("USE_AA", final_aa ? 1 : 0);
        set_float("Sharpen_Power", 1);
        measured_frames();
        require(read(exported.p) == mono, "Zero strength fallback was changed by sharpening or supported final AA");
        set_float("Depth_Adjustment", 100);
        set_int("USE_AA", 0);
        set_float("Sharpen_Power", 0);
      }
      scene->pattern = normal ? 7 : 6;
      frames_for(1400);
      require(ready() && scalar("Sunshine_CameraDepthScale") == H,
        "Moving the scene changed the established strength or lost raw readiness");
      require(std::abs(zero()[1] - initial_zero[1]) > .002f, "Actual scene samples did not move the screen plane");
      check_preparation(H, scene->pattern);
      std::printf("PASS actual %s raw sampling/policy/warp: H=%.9g initial_t0=%.9g moving_t0=%.9g; mono preserved\n",
        normal ? "normal" : "reversed", H, initial_zero[1], zero()[1]);

      // Color-only presents have no current depth; stale automatic state must
      // never claim readiness even while the old backup texture is still alive.
      render_tracked_depth = {};
      frames_for(100);
      require(!ready(), "Color-only presents reused the preceding frame's automatic depth");
      if (automatic) {
        require(scalar("Sunshine_CameraStrengthBlend") == 0.f, "Missing current depth kept nonzero stereo strength");
        require(read(exported.p) == mono, "Missing-depth Automatic output did not duplicate current source eyes");
      }
      render_tracked_depth = [&] { if (scene) draw(*scene); };
      frames_for(1100);
      require(ready() && scalar("Sunshine_CameraDepthScale") == H, "A short capture gap changed fixed H");

      // A different source must not immediately inherit calibration. A sustained
      // replacement may establish its own fresh epoch in the explicit recovery test.
      auto replacement = target(width, height, 1, source_recovery ? (normal ? 7 : 6) : (normal ? 8 : 3), false, true);
      replacement->clear_depth = normal ? 1.f : 0.f;
      // Keep the exact original native allocation alive for the optional live-
      // regression case. Equal dimensions alone must never authorize recovery.
      auto original_scene = std::move(scene);
      scene = std::move(replacement);
      const auto selection_deadline = GetTickCount64() + 8000;
      do { step(); } while (selected_description().Width != scene->width && GetTickCount64() < selection_deadline);
      require(selected_description().Width == scene->width, "Replacement depth source was not actually selected");
      // This is deliberately shorter than qualification of a permanent source
      // change; the exact-source-return case is a transient selection excursion.
      frames_for(150);
      require(!ready() && scalar("Sunshine_CameraDepthScale") == H,
        "A new source lifetime silently reused or recalibrated automatic stereo");
      if (automatic) {
        require(read(exported.p) == mono, "Changed depth source rendered legacy geometry while automatic reference was suspended");
        const auto replacement_selected = selected_description();
        require(replacement_selected.Width == scene->width && replacement_selected.Height == scene->height,
          "Replacement-source suspension did not bind the new selected depth");
      } else {
        require(varied_linear_depth(normal ? 8 : 3),
          "Replacement-source rejection did not acquire the replacement's actual depth pixels");
      }
      std::puts("PASS color-only gap and source-lifetime changes cannot reuse stale automatic depth");
      if (source_recovery) exercise_source_recovery(H, mono);
      if (source_return) {
        scene = std::move(original_scene);
        const auto returned_at = GetTickCount64();
        const auto return_deadline = returned_at + 8000;
        do { step(); } while (!ready() && GetTickCount64() < return_deadline);
        require(ready() && scalar("Sunshine_CameraDepthScale") == H,
          "Returning the exact original depth allocation did not recover fixed-H stereo without Recalibrate");
        frames_for(800);
        require(ready() && scalar("Sunshine_CameraDepthScale") == H &&
            scalar("Sunshine_CameraStrengthBlend") == 1.f,
          "Exact-source recovery did not settle at unchanged full strength");
        check_preparation(H, scene->pattern);
        require(image_difference(mono, read(exported.p)) > .002f,
          "Exact-source recovery claimed readiness without restoring actual stereo pixels");
        std::printf("PASS actual selected A -> B -> A depth return restores stereo without Recalibrate; unchanged H=%.9g, elapsed_ms=%llu\n",
          H, static_cast<unsigned long long>(GetTickCount64() - returned_at));
      }
      if (actions) exercise_recalibration(H, mono);
      check_legacy_calibration_inactive();
      render_tracked_depth = {};
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 7 && argc != 9) {
    std::fputs("usage: raw_scene_runtime <official.dll> <Shaders> <production.addon64> <fresh-output> <scrgb|pq> <normal|reversed> [width height]\n", stderr);
    return 2;
  }
  std::thread([] { Sleep(150000); std::fputs("FAIL raw runtime watchdog\n", stderr); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    width = argc == 9 ? unsigned(std::stoul(argv[7])) : 1280;
    height = argc == 9 ? unsigned(std::stoul(argv[8])) : 720;
    require(std::string(argv[5]) == "scrgb" || std::string(argv[5]) == "pq", "Invalid raw fixture color");
    require(std::string(argv[6]) == "normal" || std::string(argv[6]) == "reversed", "Invalid raw fixture orientation");
    require(sunshine_camera_fixture::compiled() && !sunshine_camera_fixture::requested(), "Raw fixture requires only camera compile gate, not injected camera tests");
    const bool performance = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_PERFORMANCE");
    const bool actions = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST");
    const bool evidence = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_EVIDENCE");
    const bool startup = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_STARTUP_TEST");
    const bool startup_replace = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_STARTUP_REPLACE");
    const bool source_return = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_RETURN_TEST");
    const bool source_recovery = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_REPLACE_TEST");
    const bool inactive_work = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_INACTIVE_TEST");
    const bool gpu_cost = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_GPU_COST");
    const bool distribution = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_DISTRIBUTION_TEST");
    require(!sunshine_camera_fixture::flag("SUNSHINE_GAME3D_LEGACY_CALIBRATION_TEST") || (actions && !distribution),
      "Legacy calibration ownership test requires the explicit test-only action run");
    require(!(performance && actions), "Run Performance Mode and test-only UI actions as separate validations");
    require(!distribution || (actions && !performance && !startup && !source_return && !evidence &&
      sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC")),
      "Distribution permutation requires isolated Automatic test-only actions, without other scenario flags");
    require((!performance && !actions) || sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC"),
      "Performance/action validations require SUNSHINE_GAME3D_AUTOMATIC=1");
    require(!evidence || (!performance && !actions && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC")),
      "Automatic snapshots require production Automatic with editable quality controls");
    require(!startup || (!performance && !actions && !evidence && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC")),
      "Startup recovery requires isolated production Automatic, separate from performance/actions/snapshots");
    require(!startup_replace || startup, "Startup replacement requires SUNSHINE_GAME3D_AUTOMATIC_STARTUP_TEST=1");
    require(!source_return || (!performance && !actions && !startup && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC")),
      "Source-return recovery requires isolated production Automatic, separate from performance/actions/startup");
    require(!source_return || !fs::exists(fs::absolute(argv[4])), "Source-return recovery requires a fresh isolated output directory");
    require(!source_recovery || (sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") &&
      !performance && !actions && !startup && !source_return && !distribution && !evidence && !fs::exists(fs::absolute(argv[4]))),
      "Source-replacement recovery requires an isolated fresh production Automatic run");
    require(!sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_EXPECT_RECOVERY") || source_recovery,
      "Recovery expectation requires the explicit source-replacement scenario");
    require(!inactive_work || (sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") &&
      !performance && !actions && !startup && !source_return && !source_recovery && !distribution && !evidence && !fs::exists(fs::absolute(argv[4]))),
      "Inactive-depth validation requires a fresh isolated Automatic run");
    require(!gpu_cost || (sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") &&
      !performance && !actions && !startup && !source_return && !source_recovery && !distribution && !evidence &&
      !inactive_work && !fs::exists(fs::absolute(argv[4]))),
      "GPU cost requires a fresh isolated actual Automatic run");
    require(!startup || !fs::exists(fs::absolute(argv[4])), "Startup recovery requires a fresh isolated output directory");
    raw_fixture fixture;
    fixture.runtime_directory = fs::absolute(argv[4]);
    if (evidence) fixture.evidence_directory = fs::absolute(argv[4]) / "automatic-evidence";
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fs::absolute(argv[4]),
      std::string(argv[5]) == "pq" ? 3 : 2, 0, fs::absolute(argv[3]));
    fixture.run(std::string(argv[6]) == "reversed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
