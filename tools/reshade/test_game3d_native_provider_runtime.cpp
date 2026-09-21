// SPDX-License-Identifier: GPL-3.0-only
// Provider metadata + real GPU UAV production -> native capture/calibration.
// The optional FG SDK exposes only ordinary SL metadata calls. It performs no
// capture and injects no depth binding, readiness, sampler result or fence.
#define SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
#include "test_streamline_direct_runtime.cpp"
#include "game3d_controls.h"

namespace {
  struct native_provider_fixture : direct_fixture {
    using automatic_t = BOOL (*)(api::effect_runtime *, unsigned *);
    using scale_t = BOOL (*)(api::effect_runtime *, unsigned *, unsigned *, float *);
    automatic_t query_automatic{};
    scale_t query_scale{};
    unsigned fx_renders{};
    std::ofstream evidence;
    sunshine_streamline::extent active_area;

    struct status {
      unsigned flags{}, basis{}, scale_state{};
      float scale{};
      sunshine_streamline::provider::source_status source;
      bool ready() const { return flags & 4u; }
    };

    void no_effects() {
      unsigned count{};
      observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++count; });
      require(!count && observed.renders == fx_renders && !observed.inject && observed.ready_uniforms.empty(),
        "Provider capture/calibration depended on an FX technique or injected effect state");
    }
    status inspect() {
      status result;
      require(query_automatic(observed.runtime, &result.flags) &&
        query_scale(observed.runtime, &result.basis, &result.scale_state, &result.scale) &&
        query_provider_status(observed.runtime, &result.source), "Native provider UI observation failed");
      require(result.flags & 1u, "Native Game 3D became disabled");
      return result;
    }
    void log_status(const char *label, const status &s) {
      std::printf("MEASURE native-provider %s ready=%u scale_basis=%u scale_state=%u K=%.9g selected=%u capture_ready=%u reused=%u depth=%ux%u allocation=%ux%u tagged=%u,%u,%u,%u capture=%llu sequence=%llu\n",
        label, unsigned(s.ready()), s.basis, s.scale_state, s.scale, unsigned(s.source.selected), unsigned(s.source.ready),
        unsigned(s.source.reused_depth), s.source.current.width, s.source.current.height,
        selected->width, selected->height, active_area.left, active_area.top, active_area.width, active_area.height,
        static_cast<unsigned long long>(s.source.current.capture), static_cast<unsigned long long>(s.source.current.sequence));
      evidence << label << " ready=" << s.ready() << " basis=" << s.basis << " scale_state=" << s.scale_state <<
        " K=" << s.scale << " capture_ready=" << s.source.ready << " reused=" << s.source.reused_depth <<
        " allocation=" << selected->width << 'x' << selected->height <<
        " tagged=" << active_area.left << ',' << active_area.top << ',' << active_area.width << ',' << active_area.height <<
        " capture=" << s.source.current.capture << " sequence=" << s.source.current.sequence << '\n';
    }
    bool active(const status &s) const {
      return s.ready() && s.basis == unsigned(sunshine_game3d::automatic_scale_basis::camera_matrix) &&
        s.scale_state == unsigned(sunshine_game3d::automatic_scale_state::active) && std::isfinite(s.scale) && s.scale > 0 &&
        s.source.selected && s.source.ready && s.source.provider == sunshine_scene_depth::provider_kind::streamline &&
        s.source.current.resource == native(*selected) && s.source.current.width == active_area.width &&
        s.source.current.height == active_area.height && s.source.current.capture && s.source.current.sequence;
    }
    status settle(const char *label) {
      const auto until = GetTickCount64() + 15000;
      unsigned consecutive{};
      status s;
      do {
        step(); no_effects(); s = inspect();
        consecutive = active(s) && !s.source.reused_depth ? consecutive + 1 : 0;
        if (consecutive >= 12) { log_status(label, s); return s; }
      } while (GetTickCount64() < until);
      log_status(label, s);
      throw std::runtime_error("Native provider did not establish ready projection scale and the actual supplied depth source");
    }
    static bool same_scale(float a, float b) {
      return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= std::max(1.f,std::abs(b))*1e-5f;
    }

    void load_native(const fs::path &directory) {
      evidence.open(directory / "native-provider-status.txt"); evidence.precision(9);
      const auto until = GetTickCount64() + 30000;
      while ((!observed.runtime || !observed.reloads) && GetTickCount64() < until) step();
      require(observed.runtime && observed.reloads, "Provider fixture runtime initialization failed");
      check_unified_addon();
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      capture = reinterpret_cast<capture_t>(GetProcAddress(module, "SunshineStreamlineTestCapture"));
      query_provider_status = reinterpret_cast<provider_status_t>(GetProcAddress(module, "SunshineDepthTestProviderStatus"));
      query_automatic = reinterpret_cast<automatic_t>(GetProcAddress(module, "SunshineGame3DTestQueryAutomatic"));
      query_scale = reinterpret_cast<scale_t>(GetProcAddress(module, "SunshineGame3DTestQueryScale"));
      require(capture && query_provider_status && query_automatic && query_scale,
        "Test add-on lacks native provider metadata adapter or passive UI observations");
      fs::rename(directory / "effects" / effect_file, directory / "effects" / "SunshineGame3D.fx.disabled");
      const auto reloads = observed.reloads;
      observed.runtime->reload_effect_next_frame(nullptr);
      const auto empty_until = GetTickCount64() + 30000;
      unsigned count = 1;
      do {
        step(); count = 0;
        observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++count; });
      } while ((observed.reloads == reloads || count) && GetTickCount64() < empty_until);
      require(observed.reloads > reloads && !count, "FX remained loaded before provider tests");
      for (const auto &entry : fs::recursive_directory_iterator(directory / "effects"))
        require(entry.path().extension() != ".fx", "Provider fixture still has an installed FX");
      fx_renders = observed.renders; no_effects();
      initialize_camera(); create_pipeline(); create_compute();
      // Expedition 33 allocates 2228x1256 at 4K, while its tagged scene uses
      // only 2228x1253. Keep that exact allocation for the SDK crop regression.
      first = target_uav(width == 3840 && height == 2160 ? 2228 : width/2,
        width == 3840 && height == 2160 ? 1256 : height/2, 0);
      selected = first.get();
      active_area = {0, 0, selected->width, selected->height};
      decoy = target(width, height, 1, 9, false, true);
      reshade::register_event<reshade::addon_event::reset_command_list>(observe_reset);
    }

    void run_streamline() {
      render_tracked_depth = [&] { draw_frame(); };
      const auto first_ready = settle("SL-UAV-over-native-DSV");
      require(ticket && v2_capture_calls, "Native provider fixture never nominated actual produced UAV depth");
      const auto before_reload = observed.reloads;
      observed.runtime->reload_effect_next_frame(nullptr);
      const auto reload_until = GetTickCount64() + 15000;
      do {
        step(); no_effects(); const auto s = inspect();
        require(active(s) && same_scale(s.scale, first_ready.scale), "FX reload reset native provider capture or scale");
      } while (observed.reloads == before_reload && GetTickCount64() < reload_until);
      require(observed.reloads > before_reload, "Native provider empty-FX reload did not complete");
      log_status("SL-after-FX-reload", inspect());

      valid_evaluation = false;
      for (unsigned frame = 0; frame < 4; ++frame) {
        step(); no_effects(); const auto s = inspect();
        require(!s.ready() && s.source.selected && !s.source.ready && !s.source.reused_depth &&
          s.scale_state == unsigned(sunshine_game3d::automatic_scale_state::held) && same_scale(s.scale, first_ready.scale),
          "Failed provider evaluation reused stale depth, fell back to Generic, or discarded native scale");
      }
      log_status("SL-failed-evaluation-held", inspect());
      valid_evaluation = true;
      const auto recovered = settle("SL-recovered");
      require(same_scale(recovered.scale, first_ready.scale), "Recovering the same provider source reset native scale");
      std::puts("PASS native SL provider: real lower-resolution UAV/copy/fences/sampling over full-resolution Generic decoy, matrix scale, FX reload, failed evaluation and recovery; no FX");
    }

    void run_fg(HMODULE sdk) {
      using namespace sunshine_streamline;
      const auto set_foreground = reinterpret_cast<void (*)(HWND)>(GetProcAddress(
        GetModuleHandleW(L"SunshineSBSTest.addon64"), "SunshineSbsTestSetForeground"));
      require(set_foreground, "FG reuse fixture needs the existing controlled foreground observer");
      struct foreground_scope {
        void (*set)(HWND);
        ~foreground_scope() { set(nullptr); }
      } foreground{set_foreground};
      // Approximate FG history intentionally ends on focus loss. This selects
      // our real hidden game HWND in the test observer without moving UI focus.
      set_foreground(window);
      struct options { base_structure base; std::uint32_t mode{}, generated_frames{}; };
      static_assert(sizeof(options) == 40);
      using set_options_t = std::int32_t (*)(const abi_v2::viewport &, const options &);
      const auto getter = reinterpret_cast<abi_v2::get_feature_function>(GetProcAddress(sdk, "slGetFeatureFunction"));
      const auto new_token = reinterpret_cast<abi_v2::get_new_frame_token>(GetProcAddress(sdk, "slGetNewFrameToken"));
      const auto constants_call = reinterpret_cast<abi_v2::set_constants>(GetProcAddress(sdk, "slSetConstants"));
      const auto tag_call = reinterpret_cast<abi_v2::set_tag_for_frame>(GetProcAddress(sdk, "slSetTagForFrame"));
      const auto evaluate = reinterpret_cast<abi_v2::evaluate_feature>(GetProcAddress(sdk, "slEvaluateFeature"));
      require(getter && new_token && constants_call && tag_call && evaluate, "FG metadata fixture lacks public SL exports");
      void *function{};
      require(getter(1000, "slDLSSGSetOptions", function) == 0 && function, "Public SDK did not expose FG options");
      set_options_t volatile set_options = reinterpret_cast<set_options_t>(function);
      for (unsigned frame = 0; frame < 4; ++frame) step();
      const abi_v2::viewport viewport{{nullptr, viewport_guid, 1}, 0};
      const auto configure = [&](bool enabled) {
        const options value{{nullptr, {0xfac5f1cb,0x2dfd,0x4f36,{0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}}, 3},
          enabled ? 1u : 0u, enabled ? 1u : 0u};
        require(set_options(viewport, value) == 0, "Observed FG options changed the public SDK return value");
      };
      // This metadata-only SDK exposes the FG tag boundary. Keep its already
      // validated FG nomination path for crop tests; direct SL coverage above
      // separately exercises successful/failed evaluation while FG is off.
      configure(true);
      unsigned fg_frame{};
      abi_v2::frame_token *last_token{};
      std::uint64_t tag_started{}, tag_returned{};
      const auto camera_constants = [&](bool reset) {
        abi_v2::constants constants{}; constants.base = {nullptr, constants_guid, 1};
        constants.common.camera_view_to_clip = camera.projection;
        constants.common.clip_to_camera_view = camera.inverse_projection;
        constants.common.camera_near = camera.near_plane; constants.common.camera_far = camera.far_plane;
        constants.common.camera_fov = camera.fov; constants.common.camera_aspect = camera.aspect;
        constants.common.camera_right[0] = constants.common.camera_up[1] = constants.common.camera_forward[2] = 1;
        constants.depth_inverted = 1;
        constants.reset = reset ? 1u : 0u;
        return constants;
      };
      const auto real_frame = [&] {
        draw(*decoy); write_depth_pixels();
        abi_v2::frame_token *token{}; const auto frame = ++fg_frame;
        require(new_token(token, &frame) == 0 && token, "FG SDK did not supply an explicit frame token");
        last_token = token;
        const auto constants = camera_constants(false);
        require(constants_call(constants, *token, viewport) == 0, "FG constants observation changed the SDK result");
        abi_v2::resource resource{}; resource.base = {nullptr, resource_guid, 1};
        resource.type = 8; resource.native = selected->resource.p; resource.state = unsigned(selected->state);
        resource.width = selected->width; resource.height = selected->height;
        resource.native_format = DXGI_FORMAT_R32_FLOAT; resource.mip_levels = resource.array_layers = 1;
        const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 0, active_area};
        tag_started = GetTickCount64();
        require(tag_call(*token, viewport, &tag, 1, reinterpret_cast<void *>(game_native_command)) == 0,
          "FG tag observation changed the SDK result");
        tag_returned = GetTickCount64();
        const base_structure *inputs[]{&viewport.base};
        require(evaluate(0, *token, inputs, 1, reinterpret_cast<void *>(game_native_command)) == 0,
          "FG same-frame SR observation changed the SDK result");
      };
      render_tracked_depth = real_frame;
      settle("SLFG-SDK-full-allocation");
      // These are public SDK extents, not injected shader coordinates. Native
      // rendering must accept the same crop even though no FX uniform exists.
      active_area = {0, 0, selected->width, selected->height-3};
      settle("SLFG-SDK-padded-three-rows");
      active_area = {8, 16, selected->width-32, selected->height-16};
      settle("SLFG-SDK-nonzero-offset-crop");
      std::puts("PASS native SL camera depth accepts padded and nonzero-offset active rectangles with zero installed FX");
      active_area = {0, 0, selected->width, selected->height-3};
      settle("SLFG-padded-real-depth");
      for (unsigned pair = 0; pair < 8; ++pair) {
        const auto pair_start = GetTickCount64();
        render_tracked_depth = real_frame;
        step(); no_effects(); const auto real = inspect();
        log_status("SLFG-pair-real-depth", real);
        require(active(real) && !real.source.reused_depth, "Fresh FG source did not produce current native depth");
        // One generated presentation has current color but no new provider call
        // or game depth draw. Observe production retention without modifying it.
        render_tracked_depth = {};
        step(); no_effects(); const auto generated = inspect();
        log_status("SLFG-generated-previous-depth", generated);
        std::printf("MEASURE native FG pair=%u elapsed_ms=%llu real_resource=0x%llx generated_resource=0x%llx expected_resource=0x%llx\n",
          pair, static_cast<unsigned long long>(GetTickCount64()-pair_start),
          static_cast<unsigned long long>(real.source.current.resource),
          static_cast<unsigned long long>(generated.source.current.resource),
          static_cast<unsigned long long>(native(*selected)));
        require(active(generated) && generated.source.reused_depth &&
          generated.source.current.capture == real.source.current.capture &&
          generated.source.current.sequence == real.source.current.sequence && same_scale(generated.scale, real.scale),
          "Generated presentation failed to retain completed real depth and its native matrix scale");
      }
      configure(false);
      step(); no_effects();
      const auto off = inspect(); log_status("FG-off-without-new-depth", off);
      require(!off.ready() && !off.source.ready && !off.source.reused_depth,
        "Disabling FG retained previous-frame depth for a color-only presentation");
      active_area = {0, 0, selected->width, selected->height};
      render_tracked_depth = [&] { draw_frame(); };
      settle("SL-recovered-after-FG-off");
      std::puts("PASS native SLFG 2x: actual cropped SDK depth, eight real/generated pairs with retained completed depth, FG-off rejection and fresh SL recovery; no FX");

      // Keep the original measured trajectory above unchanged. These additional
      // native presentations exercise the lifetime of the private display copy,
      // without an FX callback, injected readiness or synthetic GPU completion.
      const auto missing = [&](const char *label) {
        step(); no_effects(); const auto value = inspect(); log_status(label, value);
        require(!value.ready() && !value.source.ready && !value.source.reused_depth &&
          !value.source.current.capture && !value.source.current.sequence,
          "Invalidated native FG cache supplied previous depth without a fresh successful copy");
      };
      const auto held = [&](const status &seed, const char *label) {
        step(); no_effects(); const auto value = inspect();
        require(active(value) && value.source.reused_depth &&
          value.source.current.capture == seed.source.current.capture &&
          value.source.current.sequence == seed.source.current.sequence &&
          value.source.current.resource == seed.source.current.resource && same_scale(value.scale, seed.scale),
          "Native FG hold changed the admitted real depth identity or scale");
        if (label) log_status(label, value);
        return value;
      };
      const auto recovered_fresh = [&](const status &before, const char *label) {
        render_tracked_depth = real_frame;
        const auto value = settle(label);
        require(!value.source.reused_depth && value.source.current.capture != before.source.current.capture &&
          value.source.current.sequence > before.source.current.sequence,
          "Native FG cache recovered without a newly completed real capture");
        return value;
      };

      configure(true); render_tracked_depth = real_frame;
      const auto expiry_seed = settle("SLFG-cache-expiry-seed");
      const auto seed_started = tag_started, seed_returned = tag_returned;
      const auto seed_frame = fg_frame;
      render_tracked_depth = {};
      unsigned holds{};
      std::uint64_t last_held{};
      do {
        held(expiry_seed, holds ? nullptr : "SLFG-cache-expiry-first-hold");
        last_held = GetTickCount64(); ++holds;
      } while (holds < 2 || last_held < seed_returned + 140);
      require(last_held >= seed_started && last_held - seed_started < sunshine_scene_depth::maximum_source_age_ms,
        "Native expiry fixture exhausted its source-age window before exercising repeated holds");
      log_status("SLFG-cache-expiry-last-hold", inspect());
      // The public passive status exposes identity, not the private capture tick.
      // Bracket the actual SDK tag instead: expiry after its upper time bound,
      // while the most recent successful hold is still young, distinguishes the
      // original capture lifetime from a TTL accidentally renewed on each hold.
      const auto expiry_at = seed_returned + sunshine_scene_depth::maximum_source_age_ms + 20;
      const auto before_expiry_wait = GetTickCount64();
      if (before_expiry_wait < expiry_at) Sleep(DWORD(expiry_at - before_expiry_wait));
      missing("SLFG-cache-expired-after-repeated-holds");
      const auto expired = GetTickCount64();
      require(expired >= expiry_at && expired >= last_held &&
        expired - last_held < sunshine_scene_depth::maximum_source_age_ms && fg_frame == seed_frame &&
        tag_started == seed_started && tag_returned == seed_returned,
        "Native expiry fixture did not distinguish the original capture age from renewed hold age");
      missing("SLFG-cache-expired-no-revival");
      std::printf("MEASURE native FG cache expiry holds=%u tag_begin_ms=%llu tag_end_ms=%llu last_hold_ms=%llu expired_ms=%llu\n",
        holds, static_cast<unsigned long long>(seed_started), static_cast<unsigned long long>(seed_returned),
        static_cast<unsigned long long>(last_held), static_cast<unsigned long long>(expired));
      recovered_fresh(expiry_seed, "SLFG-cache-expiry-fresh-recovery");

      const auto reset_seed = settle("SLFG-cache-reset-seed");
      const auto reset_seed_started = tag_started;
      const auto reset_seed_frame = fg_frame;
      render_tracked_depth = {};
      held(reset_seed, "SLFG-cache-reset-before-hold");
      render_tracked_depth = [&] {
        require(last_token, "Native camera-reset fixture has no existing frame token");
        const auto constants = camera_constants(true);
        require(constants_call(constants, *last_token, viewport) == 0, "Camera reset changed the SDK result");
      };
      missing("SLFG-cache-camera-reset");
      require(GetTickCount64() - reset_seed_started < sunshine_scene_depth::maximum_source_age_ms,
        "Native camera-reset fixture expired before reset could be distinguished from age rejection");
      // Restore valid camera metadata on the same token, without a depth tag or
      // evaluation. This must not repair the invalidated copy's observation lease.
      render_tracked_depth = [&] {
        const auto constants = camera_constants(false);
        require(constants_call(constants, *last_token, viewport) == 0, "Camera recovery changed the SDK result");
      };
      missing("SLFG-cache-camera-valid-without-depth");
      render_tracked_depth = {};
      missing("SLFG-cache-camera-reset-no-revival");
      require(fg_frame == reset_seed_frame, "Camera-reset no-revival case accidentally produced a new real frame");
      recovered_fresh(reset_seed, "SLFG-cache-camera-reset-fresh-recovery");

      const auto mode_seed = settle("SLFG-cache-mode-seed");
      const auto mode_seed_frame = fg_frame;
      render_tracked_depth = {};
      held(mode_seed, "SLFG-cache-mode-before-hold");
      configure(false); missing("SLFG-cache-mode-off");
      configure(true); missing("SLFG-cache-mode-on-without-depth");
      missing("SLFG-cache-mode-on-no-revival");
      require(fg_frame == mode_seed_frame, "FG mode no-revival case accidentally produced a new real frame");
      recovered_fresh(mode_seed, "SLFG-cache-mode-fresh-recovery");
      std::puts("PASS native FG cache: repeated holds cannot renew capture age; camera reset and FG off/on cannot revive old depth; fresh copies recover all three intervals; no FX");
    }

    void finish() {
      render_tracked_depth = {};
      reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
      no_effects(); require(evidence.good(), "Could not write native provider evidence");
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5 && argc != 7 && argc != 8) {
    std::fputs("usage: test_game3d_native_provider_runtime official.dll frozenShaders test.addon64 freshOutput [width height [frame_generation_interposer.dll]]\n", stderr);
    return 2;
  }
  std::thread([] { Sleep(150000); std::fputs("FAIL native provider runtime watchdog\n", stderr); TerminateProcess(GetCurrentProcess(),124); }).detach();
  try {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC", "1") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST", "1") == 0 &&
      _putenv_s("SUNSHINE_DEPTH_GENERIC_ONLY_TEST", "0") == 0,
      "Could not configure explicit native provider fixture");
    width = argc >= 7 ? unsigned(std::stoul(argv[5])) : 1280;
    height = argc >= 7 ? unsigned(std::stoul(argv[6])) : 720;
    require(width >= 640 && width <= 3840 && height >= 360 && height <= 2160 && width%4 == 0 && height%4 == 0,
      "Invalid native provider fixture dimensions");
    const auto directory = fs::absolute(argv[4]);
    require(!fs::exists(directory), "Native provider fixture requires a fresh isolated output directory");
    HMODULE sdk{};
    if (argc == 8) {
      const auto original = fs::absolute(argv[7]);
      require(fs::is_regular_file(original), "Requested FG metadata fixture DLL is missing");
      fs::create_directories(directory);
      const auto isolated = directory / "sl.interposer.dll";
      fs::copy_file(original, isolated);
      sdk = LoadLibraryW(isolated.c_str());
      require(sdk, "Could not load isolated metadata-only FG SDK");
    }
    native_provider_fixture fixture; fixture.runtime_directory = directory;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), directory, 2, 0, fs::absolute(argv[3]));
    fixture.load_native(directory); fixture.run_streamline();
    if (sdk) fixture.run_fg(sdk);
    fixture.finish();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what()); return 1;
  }
}
