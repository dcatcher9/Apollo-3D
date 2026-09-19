// SPDX-License-Identifier: GPL-3.0-only
// Actual DSV -> selector/sampler -> published Automatic controls -> HDR shader.
// Reuse the established native fixture; no depth, gain or readiness injection.
#include "test_raw_runtime_fixture.h"
#include "depth_addon.h"

namespace {
  struct scheduling_stall : std::runtime_error { using std::runtime_error::runtime_error; };
  using depth_query_t = BOOL (*)(api::effect_runtime *, sunshine_depth::frame_depth *);
  depth_query_t depth_query = nullptr;
  struct depth_observation { std::uint64_t native{}, lifetime{}, frame{}; unsigned render{}; bool valid{}; } current_depth;
  unsigned last_depth_render_reload{};
  void observe_depth(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *, api::resource_view, api::resource_view) {
    char name[256] {}; runtime->get_technique_name(technique, name);
    if (!named(name, technique_name)) return;
    last_depth_render_reload = observed.reloads;
    current_depth = {};
    sunshine_depth::frame_depth value;
    if (depth_query && depth_query(runtime, &value) && value.ready)
      current_depth = {value.source_resource.handle, value.source_id, value.frame_index, observed.renders, true};
  }
  struct adaptive_fixture : raw_runtime_fixture {
    std::ofstream trajectory;
    std::uint64_t last_frame_completed_ms = 0;
    bool require_continuous_frames = false;
    using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
    using action_t = BOOL (*)(api::effect_runtime *);
    select_t select_manual = nullptr;
    action_t recalibrate = nullptr;
    unsigned pattern(bool changed) const { return (normal ? 16u : 14u) + unsigned(changed); }
    float gain() { return scalar("Sunshine_CameraDepthScale"); }
    float blend() { return scalar("Sunshine_CameraStrengthBlend"); }
    void record(const char *phase) {
      trajectory << phase << ',' << GetTickCount64() << ',' << observed.renders << ',' << current_depth.native << ','
        << current_depth.lifetime << ',' << current_depth.frame << ',' << ready() << ',' << gain() << ',' << zero()[1] << ',' << blend() << '\n';
      require(trajectory.good(), "Cannot record actual adaptive trajectory");
    }
    void frame(const char *phase) {
      step(); record(phase);
      const auto now = GetTickCount64(), previous = last_frame_completed_ms;
      last_frame_completed_ms = now;
      if (require_continuous_frames && previous && now - previous > 250) {
        trajectory.flush();
        throw scheduling_stall(std::string(phase) + ": completed-frame interval " + std::to_string(now - previous) +
          " ms exceeds the 250 ms presentation contract; continuous-adaptation result is inconclusive");
      }
    }
    bool selected_scene() const {
      return current_depth.valid && current_depth.render == observed.renders && current_depth.native == reinterpret_cast<std::uint64_t>(scene->texture.p);
    }
    template<class Check> void pump(unsigned ms, const char *phase, Check check) {
      const auto end = GetTickCount64() + ms;
      do { frame(phase); check(); } while (GetTickCount64() < end);
    }
    void verify_depth(bool changed) {
      const auto bytes = selected_native_depth();
      for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x) {
        const auto px = (2 * x + 1) * scene->width / 64, py = (2 * y + 1) * scene->height / 36;
        float raw = 0; std::memcpy(&raw, bytes.data() + (size_t(py) * scene->width + px) * 4, 4);
        const bool center = x >= 14 && x < 18 && y >= 7 && y < 11;
        const float expected = center ? (changed ? .25f : .125f) : (float(x / 8 + 1) / 16.f);
        require(std::isfinite(raw) && (normal ? 1.f - raw : raw) == expected,
          "Selected native depth differs from the independent coherent-band/center draw oracle");
      }
      const auto prep = read(linear_depth.p);
      const auto desc = linear_depth->GetDesc();
      const auto tx = unsigned(desc.Width / 2), ty = desc.Height / 2;
      std::uint16_t value = 0; std::memcpy(&value, prep.data() + (size_t(ty) * desc.Width + tx) * 4 + 2, 2);
      const float expected = 1.f / (1.f + gain() * (changed ? .25f : .125f));
      require(std::abs(half_float(value) - expected) < .002f, "Actual shader preparation did not use current adaptive gain and center depth");
    }
    void settle_ready(const char *phase, unsigned timeout = 12000) {
      const auto end = GetTickCount64() + timeout;
      do { frame(phase); } while ((!ready() || !selected_scene() || blend() != 1.f) && GetTickCount64() < end);
      require(ready() && selected_scene() && blend() == 1.f, "Adaptive source did not reach ready full-strength stereo");
    }
    void qualify_source(const char *phase, float expected_gain, float expected_zero, std::uint64_t retired_lifetime = 0) {
      // No draw of this new source/action epoch has occurred before entry.
      // Its samples may arrive before the renderer first selects it, so the
      // fresh-window lower bound belongs to this earliest possible draw, not
      // to first_selected. Keep selection latency as a separate upper bound.
      const auto first_possible_draw = GetTickCount64(), end = first_possible_draw + 10000;
      std::uint64_t first_selected = 0, first_ready = 0;
      do {
        frame(phase);
        if (GetTickCount64() - first_possible_draw < 650)
          require(!ready() && blend() == 0.f, "Source initialization reused old scale before one fresh reference window");
        if (selected_scene()) {
          require(!retired_lifetime || current_depth.lifetime != retired_lifetime,
            "Destroyed source replacement reused the retired lifetime identity");
          if (!first_selected) first_selected = GetTickCount64();
        }
        if (!first_selected) continue;
        const auto elapsed = GetTickCount64() - first_selected;
        if (ready()) { first_ready = GetTickCount64(); break; }
        require(elapsed <= 1400, "Source recovery exceeded one 750 ms reference window plus bounded capture latency");
      } while (GetTickCount64() < end);
      require(first_selected && first_ready && first_ready - first_selected <= 1400,
        "Fresh selected source failed the bounded single-window initialization");
      require(gain() == expected_gain && std::abs(zero()[1] - expected_zero) < 1e-7f,
        "New source inherited an old reference instead of its own center mean");
      std::printf("PASS source initialization %s: draw-entry-to-ready=%llu ms selected-to-ready=%llu ms H=%.9g t0=%.9g; one fresh window\n",
        phase, static_cast<unsigned long long>(first_ready - first_possible_draw),
        static_cast<unsigned long long>(first_ready - first_selected), gain(), zero()[1]);
      settle_ready(phase);
    }
    void check_effect_reload() {
      if (!sunshine_camera_fixture::flag("SUNSHINE_DEPTH_RELOAD_TEST")) return;
      const auto check_depth_readiness = [&](bool expected) {
        unsigned matched = 0;
        observed.runtime->enumerate_uniform_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_uniform_variable variable) {
          char source[32]{};
          if (!runtime->get_annotation_string_from_uniform_variable(variable, "source", source) ||
              std::strcmp(source, "bufready_depth") != 0) return;
          bool value = false;
          runtime->get_uniform_value_bool(variable, &value, 1);
          require(value == expected, "Reloaded effect has stale or uninitialized depth-readiness uniforms");
          ++matched;
        });
        require(matched != 0, "Reloaded effect has no actual depth-readiness uniforms to validate");
      };
      settle_ready("before-effect-reload");
      check_depth_readiness(true);
      const auto previous_reload = observed.reloads, previous_renders = observed.renders;
      // Start with a ready effect, then withhold all depth while replacement
      // effects compile. Their fresh uniforms must not inherit that readiness.
      render_tracked_depth = {};
      exported.reset(); linear_depth.reset();
      observed.runtime->reload_effect_next_frame(nullptr);
      const auto deadline = GetTickCount64() + 45000;
      do {
        // The usual frame logger reads camera uniforms. They are deliberately
        // absent during compilation, so use only the real present pump here.
        step();
      } while ((observed.reloads == previous_reload || observed.renders == previous_renders ||
          last_depth_render_reload != observed.reloads) && GetTickCount64() < deadline);
      require(observed.reloads > previous_reload && observed.renders > previous_renders &&
          last_depth_render_reload == observed.reloads,
        "Official ReShade did not render the newly compiled effect after reload");
      find_texture("DoubleTex", exported, width * 2, DXGI_FORMAT_R16G16B16A16_FLOAT);
      find_texture("texzBufferN_P", linear_depth, 0, DXGI_FORMAT_R16G16_FLOAT);
      set_int("Depth_Map_View", 0); set_float("Sharpen_Power", 0); set_float("Depth_Adjustment", 100);
      pump(100, "reloaded-without-depth", [&] {
        require(!current_depth.valid && !ready() && blend() == 0.f,
          "New effect reused pre-reload capture or calibration without current depth");
        check_depth_readiness(false);
      });
      const auto mono = check_current_mono();
      // Change the actual game depth center from .25 to .125. A retained H=4
      // or old sample cannot satisfy the fresh reference window and H=8 oracle.
      scene->pattern = pattern(false);
      render_tracked_depth = [&] { if (scene) draw(*scene); };
      qualify_source("effect-reload-fresh-reference", 8.f, .125f);
      check_depth_readiness(true);
      verify_depth(false);
      settle_ready("effect-reload-final");
      const auto stereo = read(exported.p);
      require(image_difference(mono, stereo) > .002f, "Reloaded current depth did not produce fresh stereo pixels");
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width * 2; ++x) for (unsigned c = 0; c < 4; ++c)
        require(std::isfinite(channel(stereo, x, y, c)), "Reloaded HDR export contains nonfinite RGBA");
      std::puts("PASS actual effect reload reacquires shader resources, clears stale bufready/calibration, and renders fresh exact-depth HDR stereo");
    }
    void run_adaptive(bool reversed) {
      normal = !reversed;
      trajectory.open(runtime_directory / "adaptive-trajectory.csv");
      trajectory << std::setprecision(17) << "phase,wall_ms,render,source,lifetime,frame,ready,H,t0,blend\n";
      create_pipeline();
      scene = target(width, height, 1, pattern(false), false, true);
      scene->clear_depth = normal ? 1.f : 0.f;
      decoy = target(width / 2, height / 2, 12, 0);
      render_tracked_depth = [&] { if (decoy) draw(*decoy); if (scene) draw(*scene); };
      const auto load_end = GetTickCount64() + 45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64() < load_end) step();
      require(observed.runtime && observed.renders && !observed.inject, "Actual adaptive effect did not initialize without injected depth");
      check_unified_addon();
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      select_manual = reinterpret_cast<select_t>(GetProcAddress(module, "SunshineDepthTestSelectManual"));
      recalibrate = reinterpret_cast<action_t>(GetProcAddress(module, "SunshineGame3DTestRecalibrate"));
      depth_query = reinterpret_cast<depth_query_t>(GetProcAddress(module, "SunshineDepthTestFrame"));
      require(select_manual && recalibrate && depth_query, "Adaptive fixture requires existing test-only real UI/identity adapters");
      reshade::register_event<reshade::addon_event::reshade_render_technique>(observe_depth);
      set_int("Depth_Map_View", 0); set_int("USE_AA", 0); set_float("Sharpen_Power", 0); set_float("Depth_Adjustment", 100);
      find_texture("DoubleTex", exported, width * 2, DXGI_FORMAT_R16G16B16A16_FLOAT);
      find_texture("texzBufferN_P", linear_depth, 0, DXGI_FORMAT_R16G16_FLOAT);
      observed.runtime->enumerate_techniques(effect_file, [&](api::effect_runtime *runtime, api::effect_technique technique) {
        char name[256] {}; runtime->get_technique_name(technique, name);
        std::printf("DIAGNOSTIC actual technique name=%s enabled=%d\n", name, runtime->get_technique_state(technique));
      });
      for (const char *name : {"Sunshine_CameraDepthReady", "Sunshine_CameraCoordinateBasis", "Sunshine_CameraProjection", "Sunshine_CameraDepthScale", "Sunshine_CameraConvergence", "Sunshine_CameraStrengthBlend"}) {
        char source[128] {}; observed.runtime->get_annotation_string_from_uniform_variable(uniform(name), "source", source);
        std::printf("DIAGNOSTIC actual uniform=%s source=%s\n", name, source);
      }
      settle_ready("initial");
      require(gain() == 8.f && zero()[1] == .125f, "Constant center did not initialize H=8 and t0=.125");
      verify_depth(false);
      settle_ready("after-initial-readback");
      const auto binding = selected_binding();
      const auto constant = [&] { require(ready() && blend() == 1.f && gain() == 8.f && zero()[1] == .125f && selected_binding() == binding,
        "Constant-source pin/unpin changed scale, zero, binding or ready strength"); };
      require(select_manual(observed.runtime, reinterpret_cast<std::uint64_t>(scene->texture.p)), "Cannot pin actual adaptive source");
      pump(1200, "constant-pin", constant);
      require(select_manual(observed.runtime, 0), "Cannot release actual adaptive source");
      pump(1200, "constant-unpin", constant);
      std::puts("PASS constant center and same-source pin/unpin retain H=8, t0=.125, binding, readiness and full blend");

      scene->pattern = pattern(true);
      const auto started = GetTickCount64();
      require_continuous_frames = true;
      pump(7000, "zero-plane-reference-changed-scene", [&] {
        const auto elapsed = GetTickCount64() - started;
        const float H = gain();
        require(ready() && blend() == 1.f && selected_scene() && H >= 4.f && H <= 8.f &&
          std::abs(H * zero()[1] - 1.f) < 2e-6f,
          "Changed scene broke coupled normalization or lost readiness/source");
        if (elapsed >= 5000) {
          require(std::abs(zero()[1] - .25f) < 2e-5f, "Independent screen plane did not settle");
        }
      });
      require_continuous_frames = false;
      require(std::abs(gain() - 4.f) < .001f && std::abs(zero()[1] - .25f) < 2e-5f,
        "Screen-plane normalization retained the initial scene's scale");
      std::printf("PASS screen-plane reference=8 -> %.9g; t0=%.9g reciprocal coupled, full blend retained\n", gain(), zero()[1]);
      verify_depth(true);
      settle_ready("after-adaptive-readback");

      const float before_gap = gain(), zero_before_gap = zero()[1];
      const auto gap_started = GetTickCount64();
      render_tracked_depth = {};
      const auto missing_publication = [&] {
        // Mono frames deliberately retain the last published H/t0 uniforms.
        // This checks publication. A pending valid readback may update the zero
        // target but cannot change the established reference.
        require(!current_depth.valid && !ready() && blend() == 0.f && gain() == before_gap && zero()[1] == zero_before_gap,
          "Missing current depth published stereo or changed held camera uniforms");
      };
      pump(100, "depth-gap-drain", missing_publication);
      // Change visible source color during the depth gap, so stale output cannot pass.
      for (size_t i = 0; i < size_t(width) * height; ++i) {
        if (color == 2) { const std::uint16_t red = 0x3a00; std::memcpy(source_bytes.data() + i * 8, &red, 2); }
        else { std::uint32_t pixel = 0; std::memcpy(&pixel, source_bytes.data() + i * 4, 4); pixel = (pixel & ~1023u) | 450u; std::memcpy(source_bytes.data() + i * 4, &pixel, 4); }
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      frame("depth-gap-new-color");
      check_current_mono();
      missing_publication();
      // No depth draws means no new selected captures. Continue presenting past
      // the1500 ms evidence expiry: a stale target/duplicate must not accumulate
      // hidden gain that would be revealed on the next ready publication.
      pump(1600, "depth-gap-no-new-samples", missing_publication);
      render_tracked_depth = [&] { if (scene) draw(*scene); };
      const auto first_ready_until = GetTickCount64() + 12000;
      do { frame("depth-return-first-ready"); } while (!ready() && GetTickCount64() < first_ready_until);
      require(ready() && selected_scene(), "Depth return failed to publish a fresh current scene");
      require(gain() == before_gap,
        "Missing captures or their return changed the fixed reference");
      require(zero()[1] == zero_before_gap, "First ready depth return spent missing-time zero-plane credit");
      std::printf("PASS gap reference hold: elapsed=%llu ms reference=%.9g -> %.9g; zero held\n",
        static_cast<unsigned long long>(GetTickCount64() - gap_started), before_gap, gain());
      settle_ready("depth-return");
      const float before_pause = gain();
      Sleep(650);
      frame("presentation-gap-first");
      require(gain() == before_pause && blend() == 0.f, "Presentation gap accumulated gain/reentry credit");
      settle_ready("presentation-return");
      std::puts("PASS current-color mono while unavailable, fixed reference, and no missing/presentation-time zero catch-up credit");

      // Save the last actual A publication immediately before leaving it.
      // The still-live exact source keeps its own numerical history off-turn.
      require(ready() && selected_scene() && blend() == 1.f, "Cannot save an unready A reference");
      const auto original_lifetime = current_depth.lifetime;
      const float original_gain = gain(), original_zero = zero()[1];
      auto original = std::move(scene);
      scene = target(width, height, 1, pattern(true), false, true); scene->clear_depth = normal ? 1.f : 0.f;
      qualify_source("replacement-B", 4.f, .25f);
      auto replacement = std::move(scene);
      scene = std::move(original); scene->pattern = pattern(false);
      const auto return_end = GetTickCount64() + 10000;
      std::uint64_t return_selected = 0;
      do {
        frame("return-A-retained-reference");
        if (selected_scene() && !return_selected) return_selected = GetTickCount64();
        if (ready() && selected_scene()) break;
        if (return_selected) require(GetTickCount64() - return_selected <= 1400,
          "Retained A did not resume within bounded fresh-capture latency");
      } while (GetTickCount64() < return_end);
      require(return_selected && ready() && selected_scene() && GetTickCount64() - return_selected <= 1400 &&
        current_depth.lifetime == original_lifetime, "Retained A did not resume its exact live source");
      require(gain() == original_gain,
        "Retained A reset its fixed reference while off-turn");
      require(zero()[1] == original_zero, "Retained A reset its zero or spent off-turn motion credit");
      std::printf("PASS retained A: lifetime=%llu H=%.9g -> %.9g t0=%.9g; prior basis resumed, no motion catch-up\n",
        static_cast<unsigned long long>(original_lifetime), original_gain, gain(), zero()[1]);
      settle_ready("return-A-full-strength");
      verify_depth(false);
      // Actual destruction must retire that history even if the native address
      // is reused by D3D12. The new lifetime still needs one fresh window.
      scene.reset();
      scene = target(width, height, 1, pattern(false), false, true); scene->clear_depth = normal ? 1.f : 0.f;
      qualify_source("destroyed-A-replacement", 8.f, .125f, original_lifetime);
      require(current_depth.lifetime != original_lifetime, "Destroyed A replacement reused the old lifetime identity");
      verify_depth(false);
      settle_ready("before-recalibrate");
      scene->pattern = pattern(true);
      pump(1000, "before-recalibrate-changed", [&] { require(ready() && blend() == 1.f, "Pre-action adaptation lost readiness"); });
      require(recalibrate(observed.runtime) && !recalibrate(observed.runtime), "Recalibrate did not admit exactly one pending real UI action");
      qualify_source("explicit-recalibrate", 4.f, .25f);
      verify_depth(true);
      settle_ready("final");
      set_float("Depth_Adjustment", 0); pump(80, "zero-strength", [] {}); const auto mono = check_current_mono();
      set_float("Depth_Adjustment", 100); settle_ready("stereo-restored");
      const auto stereo = read(exported.p);
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width * 2; ++x) for (unsigned c = 0; c < 4; ++c)
        require(std::isfinite(channel(stereo, x, y, c)), "Adaptive HDR export contains nonfinite RGBA");
      require(image_difference(mono, stereo) > .002f, "Adaptive ready state did not produce actual stereo pixels");
      sunshine_parity::write_bytes(runtime_directory / "adaptive-final.sbs", stereo.data(), stereo.size());
      sunshine_parity::write_bytes(runtime_directory / "adaptive-current-source.bin", source_bytes.data(), source_bytes.size());
      std::puts("PASS actual zero-plane RAW HDR: coupled reference/zero, current mono, retained exact source, fresh lifetime/recenter and finite stereo");
      check_effect_reload();
      render_tracked_depth = {};
      reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_depth);
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 9) { std::fputs("usage: adaptive_raw_runtime <official.dll> <Shaders> <test.addon64> <fresh-output> <scrgb|pq> <normal|reversed> width height\n", stderr); return 2; }
  std::thread([] { Sleep(180000); std::fputs("FAIL adaptive runtime watchdog\n", stderr); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    width = unsigned(std::stoul(argv[7])); height = unsigned(std::stoul(argv[8]));
    require(width >= 640 && width <= 3840 && height >= 360 && height <= 2160 && width % 2 == 0 && height % 2 == 0, "Invalid adaptive fixture dimensions");
    require(std::string(argv[5]) == "scrgb" || std::string(argv[5]) == "pq", "Invalid adaptive fixture color");
    require(std::string(argv[6]) == "normal" || std::string(argv[6]) == "reversed", "Invalid adaptive depth convention");
    require(sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST"), "Adaptive fixture requires actual Automatic and explicit test-action add-on");
    require(!fs::exists(fs::absolute(argv[4])), "Adaptive fixture requires a fresh isolated output");
    adaptive_fixture fixture; fixture.runtime_directory = fs::absolute(argv[4]);
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fixture.runtime_directory, std::string(argv[5]) == "pq" ? 3 : 2, 0, fs::absolute(argv[3]));
    fixture.run_adaptive(std::string(argv[6]) == "reversed");
    return 0;
  } catch (const scheduling_stall &error) { std::fprintf(stderr, "INCONCLUSIVE %s\n", error.what()); return 3; }
  catch (const std::exception &error) { std::fprintf(stderr, "FAIL %s\n", error.what()); return 1; }
}
