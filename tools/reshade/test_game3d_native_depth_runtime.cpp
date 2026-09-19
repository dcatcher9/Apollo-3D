// SPDX-License-Identifier: GPL-3.0-only
// Real DSV draws -> production capture/sampler -> native Game 3D calibration.
// No DEPTH semantic, camera parameters, calibration values or readiness are injected.
#define SUNSHINE_RAW_SCENE_RUNTIME
#include "test_depth_selection_runtime.cpp"
#include "game3d_controls.h"

namespace {
  struct native_depth_fixture : selection_fixture_t {
    using query_automatic_t = BOOL (*)(api::effect_runtime *, unsigned *);
    using query_scale_t = BOOL (*)(api::effect_runtime *, unsigned *, unsigned *, float *);
    using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
    using manual_state_t = BOOL (*)(api::effect_runtime *, std::uint64_t *, BOOL *);
    query_automatic_t query_automatic = nullptr;
    query_scale_t query_scale = nullptr;
    select_t select_manual = nullptr;
    manual_state_t manual_state = nullptr;
    std::ofstream report;
    unsigned fx_renders = 0;

    struct status {
      unsigned flags = 0, basis = 0, state = 0;
      float scale = 0;
      bool ready() const { return (flags & 4u) != 0; }
    };

    void no_effects() {
      require(!observed.inject && observed.ready_uniforms.empty(), "Native depth fixture injected effect state");
      unsigned techniques = 0;
      observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++techniques; });
      require(techniques == 0 && observed.renders == fx_renders,
        "An FX technique participated in native depth capture/calibration");
    }

    status current() {
      status value;
      require(query_automatic(observed.runtime, &value.flags) &&
        query_scale(observed.runtime, &value.basis, &value.state, &value.scale),
        "Native Game 3D status observation failed");
      require(value.flags & 1u, "Native Game 3D unexpectedly became disabled");
      return value;
    }

    void record(const char *label, const status &value) {
      std::printf("MEASURE native-depth %s flags=%u basis=%u state=%u H=%.9g\n",
        label, value.flags, value.basis, value.state, value.scale);
      report << label << " flags=" << value.flags << " basis=" << value.basis <<
        " state=" << value.state << " H=" << value.scale << '\n';
    }

    status wait_ready(const char *label) {
      const auto until = GetTickCount64() + 15000;
      unsigned consecutive = 0;
      status value;
      do {
        step(); no_effects(); value = current();
        const bool valid = value.ready() &&
          value.basis == unsigned(sunshine_game3d::automatic_scale_basis::relative_depth) &&
          value.state == unsigned(sunshine_game3d::automatic_scale_state::active) &&
          std::isfinite(value.scale) && value.scale > 0.f;
        consecutive = valid ? consecutive + 1 : 0;
        if (consecutive >= 16) { record(label, value); return value; }
      } while (GetTickCount64() < until);
      record(label, value);
      throw std::runtime_error("Real drawn DSV did not establish native ready state and active positive relative scale");
    }

    static bool same_scale(float a, float b) {
      return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= std::max(1.f, std::abs(b)) * 1e-5f;
    }

    void stable(unsigned milliseconds, std::uint64_t pin, float scale) {
      const auto until = GetTickCount64() + milliseconds;
      do {
        step(); no_effects();
        const auto value = current();
        std::uint64_t selected = UINT64_MAX;
        BOOL recovering = TRUE;
        require(manual_state(observed.runtime, &selected, &recovering) && selected == pin && !recovering,
          "Same-source pin action left an unexpected override/recovery state");
        require(value.ready() && value.state == unsigned(sunshine_game3d::automatic_scale_state::active) &&
          same_scale(value.scale, scale), "Same-source pin action restarted calibration or changed native scale");
      } while (GetTickCount64() < until);
    }

    void reload_without_depth_reset(float scale) {
      const auto reloads = observed.reloads;
      const auto until = GetTickCount64() + 15000;
      unsigned frames = 0;
      observed.runtime->reload_effect_next_frame(nullptr);
      do {
        step(); no_effects(); ++frames;
        const auto value = current();
        require(value.ready() && value.basis == unsigned(sunshine_game3d::automatic_scale_basis::relative_depth) &&
          value.state == unsigned(sunshine_game3d::automatic_scale_state::active) && same_scale(value.scale, scale),
          "Unrelated FX reload reset native capture readiness or established scale");
      } while (observed.reloads == reloads && GetTickCount64() < until);
      require(observed.reloads > reloads, "Unrelated empty FX reload did not complete");
      // Continue after the reload callback too: delayed source/epoch resets
      // must not hide behind the UI snapshot retained during compilation.
      stable(700, 0, scale);
      record("native-depth-across-fx-reload", current());
      std::printf("PASS native capture/scale survives actual empty FX reload across %u submitted frames\n", frames);
    }

    void run_native_depth(const fs::path &directory) {
      report.open(directory / "native-depth-status.txt");
      report.precision(9);
      const auto loaded_until = GetTickCount64() + 30000;
      while ((!observed.runtime || !observed.reloads) && GetTickCount64() < loaded_until) step();
      require(observed.runtime && observed.reloads, "Official runtime did not initialize the isolated fixture");
      check_unified_addon();
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      query_automatic = reinterpret_cast<query_automatic_t>(GetProcAddress(module, "SunshineGame3DTestQueryAutomatic"));
      query_scale = reinterpret_cast<query_scale_t>(GetProcAddress(module, "SunshineGame3DTestQueryScale"));
      select_manual = reinterpret_cast<select_t>(GetProcAddress(module, "SunshineDepthTestSelectManual"));
      manual_state = reinterpret_cast<manual_state_t>(GetProcAddress(module, "SunshineDepthTestManualState"));
      require(query_automatic && query_scale && select_manual && manual_state,
        "Test add-on lacks native UI observations or existing pin actions");

      // The shared setup copies the frozen effect. Remove it before creating any
      // game depth target: neither calibration nor capture can depend on FX state.
      fs::rename(directory / "effects" / effect_file, directory / "effects" / "SunshineGame3D.fx.disabled");
      const auto reloads = observed.reloads;
      observed.runtime->reload_effect_next_frame(nullptr);
      const auto empty_until = GetTickCount64() + 30000;
      unsigned techniques = 1;
      do {
        step(); techniques = 0;
        observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++techniques; });
      } while ((observed.reloads == reloads || techniques) && GetTickCount64() < empty_until);
      require(observed.reloads > reloads && techniques == 0, "Frozen FX was not unloaded before native depth capture");
      for (const auto &entry : fs::recursive_directory_iterator(directory / "effects"))
        require(entry.path().extension() != ".fx", "Native depth fixture still contains an installed FX");
      fx_renders = observed.renders;
      no_effects();
      require(!current().ready(), "Native fixture claimed ready before any DSV was drawn");
      std::puts("PASS native depth phase starts without installed/loaded effects or injected depth/calibration");

      create_pipeline();
      // Exact binary bands give stable, spatially varied data independent of
      // rasterization rounding, and no high-draw or alternate-resolution decoy.
      scene = target(width, height, 1, 10, false, true);
      render_tracked_depth = [&] { draw(*scene); };
      const auto initial = wait_ready("initial-drawn-depth");
      stable(600, 0, initial.scale);

      const auto source = reinterpret_cast<std::uint64_t>(scene->texture.p);
      require(select_manual(observed.runtime, source), "Could not pin the actual drawn depth source");
      stable(700, source, initial.scale);
      require(select_manual(observed.runtime, 0), "Could not release the actual drawn depth source");
      stable(700, 0, initial.scale);
      record("same-source-pin-unpin", current());

      reload_without_depth_reset(initial.scale);

      render_tracked_depth = {};
      const auto gap_until = GetTickCount64() + 3000;
      unsigned missing_frames = 0;
      status gap;
      do {
        step(); no_effects(); gap = current();
        missing_frames = gap.ready() ? 0 : missing_frames + 1;
      } while (missing_frames < 16 && GetTickCount64() < gap_until);
      record("color-only-gap", gap);
      require(missing_frames >= 16 && gap.state == unsigned(sunshine_game3d::automatic_scale_state::held) &&
        same_scale(gap.scale, initial.scale), "Color-only presents reused stale depth or discarded the established scale");

      render_tracked_depth = [&] { draw(*scene); };
      const auto recovered = wait_ready("same-source-recovered");
      require(same_scale(recovered.scale, initial.scale), "Recovering the same drawn source reset native scale");
      stable(600, 0, initial.scale);
      render_tracked_depth = {};
      no_effects();
      require(report.good(), "Could not write native depth evidence");
      std::puts("PASS actual D3D12 native capture without FX: drawn DSV, automatic positive scale, stable pin/unpin, unrelated FX reload continuity, color-only loss and same-source recovery");
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5 && argc != 7) {
    std::fputs("usage: test_game3d_native_depth_runtime official-ReShade64.dll frozen-shader-directory SunshineSBSTest.addon64 fresh-output-directory [width height]\n", stderr);
    return 2;
  }
  std::thread([] {
    Sleep(120000);
    std::fputs("FAIL native depth runtime watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST", "1") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC", "1") == 0 &&
      _putenv_s("SUNSHINE_DEPTH_GENERIC_ONLY_TEST", "1") == 0,
      "Could not configure explicit test-only native capture fixture");
    width = argc == 7 ? unsigned(std::stoul(argv[5])) : 1280;
    height = argc == 7 ? unsigned(std::stoul(argv[6])) : 720;
    require(width >= 640 && height >= 360 && width <= 3840 && height <= 3840 && width % 2 == 0 && height % 2 == 0,
      "Invalid native depth fixture dimensions");
    const auto directory = fs::absolute(argv[4]);
    require(!fs::exists(directory), "Native depth fixture requires a fresh isolated output directory");
    native_depth_fixture fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), directory, 2, 0, fs::absolute(argv[3]));
    fixture.run_native_depth(directory);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
