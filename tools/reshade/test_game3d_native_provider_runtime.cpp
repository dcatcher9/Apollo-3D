// SPDX-License-Identifier: GPL-3.0-only
// Provider metadata + real GPU UAV production -> native capture/calibration.
// The optional FG SDK exposes only ordinary SL metadata calls. It performs no
// capture and injects no depth binding, readiness, sampler result or fence.
#define SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
#include "test_streamline_direct_runtime.cpp"
#include "depth_refusal.h"
#include "game3d_controls.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_layer.h"
#include "../../src/game3d_debug_protocol.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {
  // SUNSHINE_GAME3D_TEST_DIAGNOSTICS_OFF: the run keeps the Diagnostics switch
  // off. Either way candidates are bound directly (leased snapshots, live
  // layer copies) outside dumps: the switch only adds per-pass GPU timing.
  bool diagnostics_off() {
    const char *value = std::getenv("SUNSHINE_GAME3D_TEST_DIAGNOSTICS_OFF");
    return value && *value && std::strcmp(value, "0") != 0;
  }

  // The live layer as the add-on's execute handler left it, read inside
  // ReShade's execute event of the one list armed here (list: that native
  // list only; 0: the first executed): ReShade invokes callbacks in
  // registration order, so the test add-on's own handler ran and the native
  // ExecuteCommandLists has not yet (layer queue order).
  struct execute_probe_t {
    BOOL (*query)(sunshine_game3d::ui_layer::test_live_state *){};
    sunshine_game3d::ui_layer::test_live_state seen{};
    std::uint64_t list{};
    bool armed{}, probed{};
  } execute_probe;
  void probe_execute(api::command_queue *, api::command_list *commands) {
    if (!execute_probe.armed || execute_probe.probed) return;
    if (execute_probe.list && (!commands || commands->get_native() != execute_probe.list)) return;
    execute_probe.probed = execute_probe.query && execute_probe.query(&execute_probe.seen);
  }
}

// The game's NGX SDK: the shared stand-in's exported entry points and
// parameter getters, hooked by production module discovery. The DLSS-G case
// creates feature 11 and sets only its named DLSSG.* inputs.
#include "test_ngx_stand_in.h"

namespace {
  struct native_provider_fixture : direct_fixture {
    using automatic_t = BOOL (*)(api::effect_runtime *, unsigned *);
    using scale_t = BOOL (*)(api::effect_runtime *, unsigned *, unsigned *, float *);
    using refusal_t = BOOL (*)(api::effect_runtime *, sunshine_depth_refusal::record *);
    automatic_t query_automatic{};
    scale_t query_scale{};
    refusal_t query_refusal{};
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
        present_frame(); s = inspect();
        consecutive = active(s) && !s.source.reused_depth ? consecutive + 1 : 0;
        if (consecutive >= 12) { log_status(label, s); return s; }
      } while (GetTickCount64() < until);
      log_status(label, s);
      throw std::runtime_error("Native provider did not establish ready projection scale and the actual supplied depth source");
    }
    // A line of the isolated runtime's ReShade.log containing every part.
    // Add-on lines reach it through the asynchronous log writer, and bounded
    // lines are written at most every 5 s, so keep presenting the current
    // scenario for at most wait_ms while waiting.
    bool reshade_log_contains(std::initializer_list<const char *> parts, std::uint64_t wait_ms = 7000) {
      const auto until = GetTickCount64() + wait_ms;
      do {
        std::ifstream file(runtime_directory / "ReShade.log", std::ios::binary);
        for (std::string line; std::getline(file, line);)
          if (std::all_of(parts.begin(), parts.end(), [&](const char *part) { return line.find(part) != std::string::npos; }))
            return true;
        present_frame();
      } while (GetTickCount64() < until);
      return false;
    }
    // How many lines of the isolated runtime's ReShade.log contain every part
    // now (no wait: add-on lines may still be in the asynchronous writer).
    std::size_t reshade_log_count(std::initializer_list<const char *> parts) {
      std::size_t count{};
      std::ifstream file(runtime_directory / "ReShade.log", std::ios::binary);
      for (std::string line; std::getline(file, line);)
        if (std::all_of(parts.begin(), parts.end(), [&](const char *part) { return line.find(part) != std::string::npos; }))
          ++count;
      return count;
    }
    // Those lines themselves, oldest first (no wait).
    std::vector<std::string> reshade_log_lines(std::initializer_list<const char *> parts) {
      std::vector<std::string> lines;
      std::ifstream file(runtime_directory / "ReShade.log", std::ios::binary);
      for (std::string line; std::getline(file, line);)
        if (std::all_of(parts.begin(), parts.end(), [&](const char *part) { return line.find(part) != std::string::npos; }))
          lines.push_back(line);
      return lines;
    }
    // The number right after field in line, searched from the end of after
    // when given; zero when either is missing.
    static std::uint64_t number_after(const std::string &line, const char *field, const char *after = nullptr) {
      std::size_t from = 0;
      if (after) {
        from = line.find(after);
        if (from == std::string::npos) return 0;
        from += std::strlen(after);
      }
      const auto at = line.find(field, from);
      return at == std::string::npos ? 0 : std::strtoull(line.c_str() + at + std::strlen(field), nullptr, 10);
    }
    // Whether a line of the isolated runtime's ReShade.log containing
    // line_part has a number above minimum right after field, presenting the
    // current scenario for at most wait_ms while waiting (bounded lines).
    bool reshade_log_number_above(const char *line_part, const char *field, std::uint64_t minimum,
        std::uint64_t wait_ms = 7000) {
      const auto until = GetTickCount64() + wait_ms;
      do {
        std::ifstream file(runtime_directory / "ReShade.log", std::ios::binary);
        for (std::string line; std::getline(file, line);) {
          if (line.find(line_part) == std::string::npos) continue;
          const auto at = line.find(field);
          if (at != std::string::npos && std::strtoull(line.c_str() + at + std::strlen(field), nullptr, 10) > minimum)
            return true;
        }
        present_frame();
      } while (GetTickCount64() < until);
      return false;
    }
    // The test add-on's newest depth refusal (SunshineDepthTestLastRefusal).
    sunshine_depth_refusal::record last_refusal() {
      sunshine_depth_refusal::record value;
      require(query_refusal(observed.runtime, &value), "Test add-on refusal query failed");
      return value;
    }
    // The newest refusal once one after serial is recorded, presenting the
    // current scenario for at most wait_ms. No log line is awaited.
    sunshine_depth_refusal::record refusal_after(std::uint64_t serial, std::uint64_t wait_ms = 2000) {
      const auto until = GetTickCount64() + wait_ms;
      auto value = last_refusal();
      while (value.serial <= serial && GetTickCount64() < until) {
        present_frame();
        value = last_refusal();
      }
      return value;
    }
    static bool same_scale(float a, float b) {
      return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= std::max(1.f,std::abs(b))*1e-5f;
    }
    // The production Dump3D mailbox (src/game3d_debug_protocol.h) with this
    // process as its consumer, nonce naming the scenario: request() asks the
    // loaded add-on for one dump while the scenario keeps presenting and
    // returns its metadata. The response stays valid until release(), the
    // next request or the mailbox's end. (The UI hook scenarios above keep
    // their own copies of this protocol.)
    struct dump_mailbox {
      native_provider_fixture &fixture;
      HANDLE handle{};
      game3d_debug::shared_state_t *state{};
      std::uint64_t request_id{};
      dump_mailbox(native_provider_fixture &owner, std::uint64_t nonce) : fixture(owner) {
        namespace dump = ::game3d_debug;
        const auto name = std::wstring(dump::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
        handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        require(handle != nullptr, "Loaded add-on did not publish its production Dump3D mailbox");
        state = static_cast<dump::shared_state_t *>(MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(dump::shared_state_t)));
        require(state && state->signature == dump::magic && state->protocol_version == dump::version,
          "Production Dump3D mailbox is incompatible");
        FILETIME created{}, exited{}, kernel{}, user{};
        require(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user), "Identify the Dump3D consumer");
        state->consumer_pid = GetCurrentProcessId();
        state->consumer_creation_time = (std::uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime;
        state->consumer_nonce = nonce;
      }
      dump_mailbox(const dump_mailbox &) = delete;
      dump_mailbox &operator=(const dump_mailbox &) = delete;
      ~dump_mailbox() {
        if (state) { release(); UnmapViewOfFile(state); }
        if (handle) CloseHandle(handle);
      }
      void release() {
        if (request_id) InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state->released_id), request_id);
      }
      nlohmann::json request(const char *incomplete) {
        namespace dump = ::game3d_debug;
        request_id = state->request_id + 1;
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state->request_id), request_id);
        const auto until = GetTickCount64() + 10000;
        while (std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&state->response_id), 0, 0)) != request_id &&
            GetTickCount64() < until)
          fixture.present_frame();
        require(state->response_id == request_id && state->response.result == dump::status::complete &&
            state->response.json_bytes <= dump::max_json_bytes && state->response.texture_count <= dump::max_textures, incomplete);
        return nlohmann::json::parse(std::string(state->json, state->response.json_bytes));
      }
    };

    // One case as a frame script (README, native provider fixture): warm_up
    // presents until the case's starting state holds, stimulus presents the
    // frames under test and judges them, and recovery restores the fixture's
    // control. Every frame of a script begins before its deadline, the sum of
    // the bounds its own waits allow, so a case that stops making progress
    // fails by name; MEASURE reports each case's time.
    struct frame_script {
      const char *name;
      std::uint64_t deadline_ms;
      std::function<void()> warm_up, stimulus, recovery;
    };
    const char *script_name{};
    std::uint64_t script_deadline{};
    // One game frame and its Present, observed without FX.
    void present_frame() {
      if (script_deadline && GetTickCount64() >= script_deadline)
        throw std::runtime_error(std::string("Native provider case ") + script_name + " passed its frame deadline");
      step(); no_effects();
    }
    void run_script(const frame_script &script) {
      const auto started = GetTickCount64();
      script_name = script.name; script_deadline = started + script.deadline_ms;
      struct end_script {
        native_provider_fixture &fixture;
        ~end_script() { fixture.script_name = nullptr; fixture.script_deadline = 0; }
      } scope{*this};
      if (script.warm_up) script.warm_up();
      script.stimulus();
      if (script.recovery) script.recovery();
      std::printf("MEASURE native-provider case=%s elapsed_ms=%llu deadline_ms=%llu\n", script.name,
        static_cast<unsigned long long>(GetTickCount64() - started), static_cast<unsigned long long>(script.deadline_ms));
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
      query_refusal = reinterpret_cast<refusal_t>(GetProcAddress(module, "SunshineDepthTestLastRefusal"));
      require(capture && query_provider_status && query_automatic && query_scale && query_refusal,
        "Test add-on lacks native provider metadata adapter, passive UI observations or the refusal query");
      // The Diagnostics switch on, as the panel sets it, unless
      // SUNSHINE_GAME3D_TEST_DIAGNOSTICS_OFF keeps it off: it adds per-pass
      // GPU timing only, so candidates stay bound directly either way.
      const auto set_diagnostics = reinterpret_cast<BOOL (*)(BOOL)>(GetProcAddress(module, "SunshineGame3DTestSetDiagnostics"));
      require(set_diagnostics && set_diagnostics(diagnostics_off() ? FALSE : TRUE),
        "Test add-on cannot set the Diagnostics switch");
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
        present_frame(); const auto s = inspect();
        require(active(s) && same_scale(s.scale, first_ready.scale), "FX reload reset native provider capture or scale");
      } while (observed.reloads == before_reload && GetTickCount64() < reload_until);
      require(observed.reloads > before_reload, "Native provider empty-FX reload did not complete");
      const auto fresh = inspect();
      log_status("SL-after-FX-reload", fresh);
      require(active(fresh) && !fresh.source.reused_depth, "Native provider showed no fresh depth before the failed evaluations");
      const auto fresh_tick = GetTickCount64();

      // Failed evaluations hold the newest completed copy within its source
      // age, like any failed or missing successor, then stay current-color
      // mono: never another copy, never a hold after mono, never Generic, and
      // the native scale is held throughout. (The fixture's window used to be
      // in the background here, where every Present invalidated reuse.)
      valid_evaluation = false;
      unsigned mono{}, holds{};
      for (const auto until = GetTickCount64() + 15000; mono < 4 && GetTickCount64() < until;) {
        // The hold is decided after this tick, within the capture's age,
        // and the capture is older than fresh_tick.
        const auto presented = GetTickCount64();
        present_frame(); const auto s = inspect();
        require(s.source.selected && same_scale(s.scale, first_ready.scale),
          "Failed provider evaluation fell back to Generic or discarded native scale");
        if (s.source.ready) {
          const bool held = !mono && s.source.reused_depth && s.source.current.capture == fresh.source.current.capture &&
            s.source.current.sequence == fresh.source.current.sequence &&
            presented - fresh_tick < sunshine_scene_depth::maximum_source_age_ms;
          if (!held) {
            log_status("SL-failed-evaluation-unexpected-depth", s);
            std::printf("MEASURE native-provider SL-failed-evaluation holds=%u mono=%u fresh_capture=%llu fresh_sequence=%llu age_ms=%llu\n",
              holds, mono, static_cast<unsigned long long>(fresh.source.current.capture),
              static_cast<unsigned long long>(fresh.source.current.sequence),
              static_cast<unsigned long long>(presented - fresh_tick));
          }
          require(held, "Failed provider evaluation showed depth other than its newest completed copy, or after its age bound");
          ++holds;
          continue;
        }
        ++mono;
        require(!s.ready() && !s.source.reused_depth &&
          s.scale_state == unsigned(sunshine_game3d::automatic_scale_state::held),
          "Failed provider evaluation reused stale depth after mono or did not hold the native scale");
      }
      require(mono >= 4, "Failed provider evaluations kept showing depth past the held copy's age bound");
      std::printf("MEASURE native-provider SL-failed-evaluation holds=%u mono=%u\n", holds, mono);
      log_status("SL-failed-evaluation-held", inspect());
      valid_evaluation = true;
      const auto recovered = settle("SL-recovered");
      require(same_scale(recovered.scale, first_ready.scale), "Recovering the same provider source reset native scale");
      std::puts("PASS native SL provider: real lower-resolution UAV/copy/fences/sampling over full-resolution Generic decoy, matrix scale, FX reload, failed evaluation and recovery; no FX");
    }

    // The public tag 23 UI hook with one Resource type, then the offscreen UI
    // layer, over the FG cases' real frames.
    void run_ui_hook(std::uint32_t resource_type) {
      using namespace sunshine_streamline;
      const auto real_frame = [this] { real_fg_frame(); };
      namespace dump = ::game3d_debug;
      const auto hook_directory = runtime_directory / ("ui-hook-type-" + std::to_string(resource_type));
      std::filesystem::create_directories(hook_directory);
      // At-call ownership must freeze this mask before the game reuses its
      // allocation. Only the public SDK tag is offered to the loaded add-on.
      com_ptr<ID3D12Resource> mask, before_upload, after_upload;
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width; desc.Height = height;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = DXGI_FORMAT_B8G8R8A8_TYPELESS;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(mask.put())), "Create tagged BGRA allocation");
      std::vector<std::uint8_t> expected(size_t(width) * height * 4), overwritten(expected.size());
      unsigned covered{};
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const auto offset = (size_t(y) * width + x) * 4;
        const bool ui = x >= width / 3 && x < width / 2 && y >= height / 4 && y < height * 3 / 4;
        const std::uint8_t levels[]{1, 64, 255}; covered += ui;
        expected[offset] = static_cast<std::uint8_t>((x * 3 + y * 7 + 17) & 255);
        expected[offset + 1] = static_cast<std::uint8_t>((x * 11 + y * 5 + 31) & 255);
        expected[offset + 2] = static_cast<std::uint8_t>((x * 19 + y * 13 + 47) & 255);
        expected[offset + 3] = ui ? levels[(x + y) % 3] : 0;
        for (unsigned c = 0; c < 3; ++c) overwritten[offset + c] = 255 - expected[offset + c];
        overwritten[offset + 3] = 255;
      }
      require(covered && covered < width * height, "Public UI hook fixture has no selective alpha");
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT before_footprint{}, after_footprint{};
      fill_upload(before_upload, desc, expected.data(), before_footprint);
      fill_upload(after_upload, desc, overwritten.data(), after_footprint);
      // An offscreen UI layer as Frostbite and Unreal's HDR UI composite draw
      // it: output resolution, cleared to transparent black every frame, then UI.
      com_ptr<ID3D12Resource> layer, layer_upload;
      com_ptr<ID3D12DescriptorHeap> layer_heap;
      auto layer_desc = desc;
      layer_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      layer_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &layer_desc,
        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(layer.put())), "Create offscreen UI layer");
      D3D12_DESCRIPTOR_HEAP_DESC layer_heap_desc{};
      layer_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; layer_heap_desc.NumDescriptors = 1;
      checked(game->CreateDescriptorHeap(&layer_heap_desc, IID_PPV_ARGS(layer_heap.put())), "Create UI layer RTV heap");
      const auto layer_rtv = layer_heap->GetCPUDescriptorHandleForHeapStart();
      game->CreateRenderTargetView(layer.p, nullptr, layer_rtv);
      // Translucent UI tinted brighter than white: green 200 over alpha 160 lies
      // above its alpha but within twice it, as Stellar Blade's pulsing markers.
      std::vector<std::uint8_t> layer_pattern(size_t(width) * height * 4);
      for (unsigned y = height / 8; y < height / 4; ++y) for (unsigned x = width / 8; x < width / 4; ++x) {
        const auto offset = (size_t(y) * width + x) * 4;
        layer_pattern[offset] = static_cast<std::uint8_t>((x * 5 + y) & 255);
        layer_pattern[offset + 1] = 200;
        layer_pattern[offset + 2] = static_cast<std::uint8_t>((y * 3) & 255);
        layer_pattern[offset + 3] = 160;
      }
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT layer_footprint{};
      fill_upload(layer_upload, layer_desc, layer_pattern.data(), layer_footprint);
      // The same content with straight alpha: color far beyond twice its alpha,
      // as in a scene buffer rather than UI blended over transparent black.
      auto straight_pattern = layer_pattern;
      for (size_t offset = 0; offset < straight_pattern.size(); offset += 4)
        if (straight_pattern[offset + 3]) straight_pattern[offset + 3] = 40;
      com_ptr<ID3D12Resource> straight_upload;
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT straight_footprint{};
      fill_upload(straight_upload, layer_desc, straight_pattern.data(), straight_footprint);
      bool straight_layer = false;
      // The same content opaque everywhere, as a splash drawn into the layer.
      auto opaque_pattern = layer_pattern;
      for (size_t offset = 3; offset < opaque_pattern.size(); offset += 4) opaque_pattern[offset] = 255;
      com_ptr<ID3D12Resource> opaque_upload;
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT opaque_footprint{};
      fill_upload(opaque_upload, layer_desc, opaque_pattern.data(), opaque_footprint);
      bool opaque_layer = false;
      const auto copy = [&](ID3D12GraphicsCommandList *list, ID3D12Resource *upload,
                            const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &footprint) {
        transition(list, mask.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = upload; from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
        to.pResource = mask.p; to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(list, mask.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      };
      struct restore_callback {
        std::function<void()> &target;
        std::function<void()> previous;
        ~restore_callback() { target = std::move(previous); }
      } restore{render_tracked_depth, render_tracked_depth};
      const auto draw_layer = [&] {
        // The game records through ReShade's command-list proxy, so its clear
        // event fires; the unwrapped native list would bypass it.
        auto *game_list = commands.p;
        const float transparent[4]{};
        game_list->ClearRenderTargetView(layer_rtv, transparent, 0, nullptr);
        transition(game_list, layer.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION layer_from{}, layer_to{};
        layer_from.pResource = opaque_layer ? opaque_upload.p : straight_layer ? straight_upload.p : layer_upload.p;
        layer_from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        layer_from.PlacedFootprint = opaque_layer ? opaque_footprint : straight_layer ? straight_footprint : layer_footprint;
        layer_to.pResource = layer.p; layer_to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        game_list->CopyTextureRegion(&layer_to, 0, 0, 0, &layer_from, nullptr);
        transition(game_list, layer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
      };
      render_tracked_depth = [&] {
        real_frame();
        auto *list = reinterpret_cast<ID3D12GraphicsCommandList *>(game_native_command);
        copy(list, before_upload.p, before_footprint);
        draw_layer();
        abi_v2::resource resource{}; resource.base = {nullptr, resource_guid, 1};
        resource.type = resource_type; resource.native = mask.p; resource.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        resource.width = width; resource.height = height; resource.native_format = DXGI_FORMAT_B8G8R8A8_TYPELESS;
        resource.mip_levels = resource.array_layers = 1;
        const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 23, 0, {0, 0, width, height}};
        require(last_token && (sl.frame_tag ? sl.frame_tag(*last_token, fg_viewport, &tag, 1, list) :
            sl.global_tag(fg_viewport, &tag, 1, list)) == 0,
          "Public UI tag changed the synthetic SDK result");
        copy(list, after_upload.p, after_footprint);
      };
      for (unsigned i = 0; i < 8; ++i) { present_frame(); }
      // The tag is a declared source: its first valid selective sample accepts
      // it (A1), from the render after that sample is read. A few sample
      // intervals pass before the dump.
      for (const auto until = GetTickCount64() + 500; GetTickCount64() < until;) { present_frame(); }

      struct mailbox {
        HANDLE handle{};
        dump::shared_state_t *state{};
        std::uint64_t request{};
        ~mailbox() {
          if (state) {
            if (request) InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state->released_id), request);
            UnmapViewOfFile(state);
          }
          if (handle) CloseHandle(handle);
        }
      } box;
      const auto name = std::wstring(dump::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      box.handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
      require(box.handle != nullptr, "Loaded add-on did not publish its production Dump3D mailbox");
      box.state = static_cast<dump::shared_state_t *>(MapViewOfFile(box.handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(dump::shared_state_t)));
      require(box.state && box.state->signature == dump::magic && box.state->protocol_version == dump::version,
        "Production Dump3D mailbox is incompatible");
      FILETIME created{}, exited{}, kernel{}, user{};
      require(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user), "Identify actual dump consumer");
      box.state->consumer_pid = GetCurrentProcessId();
      box.state->consumer_creation_time = (std::uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime;
      box.state->consumer_nonce = 0x5549484f4f4bULL;
      box.request = box.state->request_id + 1;
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->request_id), box.request);
      const auto until = GetTickCount64() + 10000;
      while (std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->response_id), 0, 0)) != box.request && GetTickCount64() < until) {
        present_frame();
      }
      require(box.state->response_id == box.request && box.state->response.result == dump::status::complete &&
          box.state->response.json_bytes <= dump::max_json_bytes && box.state->response.texture_count <= dump::max_textures,
        "Production Dump3D did not complete the public UI hook capture");
      const auto metadata = nlohmann::json::parse(std::string(box.state->json, box.state->response.json_bytes));
      {
        std::ofstream manifest(hook_directory / "ui-hook-producer-metadata.json");
        manifest << metadata.dump(2) << '\n';
        require(manifest.good(), "Write actual public UI hook provenance");
      }
      // Auto validates every admitted candidate on the GPU and consumes the
      // resolved R32 mask through its red channel (mask channel 1).
      const auto &origin = metadata.at("ui_source");
      const auto &replay = metadata.at("replay");
      // S3's frame identity and rule H2 were removed: the dump carries neither
      // their stamps, proposals and still flag nor the identity shadow.
      {
        const auto &detection = replay.at("ui_detection");
        require(!detection.contains("stamps") &&
            !detection.contains("identity_bits") && !detection.contains("still_bits") && !origin.contains("identity_shadow"),
          ("The dump kept removed S3 or H2 fields: " + detection.dump()).c_str());
      }
      bool tag23_candidate = false;
      for (const auto &candidate : origin.value("candidates", nlohmann::json::array()))
        tag23_candidate |= candidate.at("source") == "sl_ui_color_alpha" && candidate.at("tag_type") == 23 &&
          candidate.at("channel") == "alpha" && candidate.at("format") == 87 &&
          candidate.at("available_for_detection") == true &&
          candidate.at("source_native") == reinterpret_cast<std::uint64_t>(mask.p);
      require(origin.at("source") == "automatic_candidate_set" && tag23_candidate &&
          replay.at("source_alpha_ui") == true && replay.at("ui_alpha_source") == "ui_source_color" &&
          replay.at("ui_constant_binding").at("uint32")[3] == 1,
        "Public UI hook did not offer the actual tag23 alpha source to automatic detection");
      // The tagged UIColorAndAlpha and the offscreen UI layer are offered side
      // by side from the first frame, each in its own slot (E1): the stored
      // flags describe the layer slot (the premultiplied bound, 1), and the
      // accepted tag decides.
      namespace candidate = sunshine_game3d::ui_detection::candidate;
      const auto &tag_detection = replay.at("ui_detection");
      if (tag_detection.at("ran_or_held") == "inactive" ||
          (tag_detection.at("candidates").get<unsigned>() & (candidate::ui_color | candidate::layer)) !=
            (candidate::ui_color | candidate::layer) ||
          !(tag_detection.at("accepted").get<unsigned>() & candidate::ui_color) ||
          (tag_detection.at("flags").get<unsigned>() & ~sunshine_game3d::ui_detection::per_frame_sample) !=
            sunshine_game3d::ui_detection::stored_premultiplied)
        throw std::runtime_error("A tagged UIColorAndAlpha did not reach detection accepted beside the layer: " + tag_detection.dump());
      bool optional = false;
      for (const auto &entry : metadata.at("optional_captures")) if (entry.at("artifact_id") == 10) {
        if (entry.value("status", std::string{}) != "captured" || !entry.contains("capture_diagnostic"))
          throw std::runtime_error("Public UI hook optional tag23 was not captured: " + entry.dump());
        const auto &attempt = entry.at("capture_diagnostic");
        if (!(attempt.at("format") == 90 && attempt.at("flags") == 4 && attempt.at("native_state") == 8 &&
            attempt.at("copy_state") == 192 && attempt.at("used_observed_state") == true))
          throw std::runtime_error("Public UI hook dump lost its actual typeless source/state provenance: " + entry.dump());
        optional = true;
      }
      require(optional, "Production dump omitted the observed tag23 capture");
      // The census copies the layer before its next clear: the previous frame's
      // UI. The frame's list ran on the presenting queue, the dump's own, so
      // the copy is read in queue order once its submission was seen and its
      // list reset (ordering proven), at least one Present after the copy.
      char layer_source[24];
      std::snprintf(layer_source, sizeof(layer_source), "0x%llx", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(layer.p)));
      unsigned layer_artifact{};
      const auto &first_census = metadata.at("ui_layer_census");
      for (const auto &entry : first_census.at("candidates"))
        if (entry.at("source") == layer_source && entry.at("captured") == true && entry.at("dxgi_format") == 28 &&
            entry.at("width") == width && entry.at("height") == height && entry.at("clears_while_armed") >= 1 &&
            entry.at("status") == "captured_before_clear" && entry.at("read_order") == "queue" &&
            entry.at("fence_value") == 0 && entry.at("presents_since_copy") >= 1 && entry.at("unordered_reason").is_null())
          layer_artifact = entry.at("artifact_id");
      if (!layer_artifact || first_census.at("ordering") != "proven" || first_census.at("settled") != true ||
          first_census.at("wait_ms") >= sunshine_game3d::ui_layer::census_wait_ms)
        throw std::runtime_error("Dump census missed the offscreen UI layer or read it without queue order: " + first_census.dump());
      // No S3 ticket remains on the census copy or an offered candidate.
      for (const auto &entry : metadata.at("ui_layer_census").at("candidates"))
        require(!entry.contains("ticket"), ("The dump census copy kept its removed S3 ticket: " + entry.dump()).c_str());
      for (const auto &candidate : origin.at("candidates"))
        require(!candidate.contains("ticket"), ("An offered candidate kept its removed S3 ticket: " + candidate.dump()).c_str());
      bool layer_bytes = false;
      for (unsigned i = 0; i < box.state->response.texture_count; ++i) {
        const auto &item = box.state->response.textures[i];
        if (unsigned(item.kind) != layer_artifact) continue;
        require(item.dxgi_format == DXGI_FORMAT_R8G8B8A8_UNORM && item.width == width && item.height == height,
          "Offscreen UI layer copy changed format or extent");
        com_ptr<ID3D12Resource> copied;
        checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(copied.put())), "Open UI layer copy");
        require(read(copied.p, D3D12_RESOURCE_STATE_COMMON) == layer_pattern, "Offscreen UI layer copy is not the drawn layer");
        layer_bytes = true;
      }
      require(layer_bytes, "Dump omitted the captured offscreen UI layer");
      std::array<com_ptr<ID3D12Resource>, 2> retained;
      std::array<std::vector<std::uint8_t>, 2> retained_bytes;
      unsigned retained_count{};
      bool consumed = false, tagged = false;
      for (unsigned i = 0; i < box.state->response.texture_count; ++i) {
        const auto &item = box.state->response.textures[i];
        const auto kind = unsigned(item.kind);
        if (kind != unsigned(dump::artifact::ui_source_color) && kind != 10) continue;
        const bool resolved = kind == unsigned(dump::artifact::ui_source_color);
        require(item.dxgi_format == (resolved ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM) &&
          item.width == width && item.height == height, "Public UI hook dump changed typed extent");
        require(retained_count < retained.size(), "Public UI hook dump repeated a requested artifact");
        auto &texture = retained[retained_count];
        checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(texture.put())), "Open production UI snapshot");
        const auto bytes = read(texture.p, D3D12_RESOURCE_STATE_COMMON);
        if (resolved) {
          // The resolved mask is the winning candidate's own alpha, alpha/255.
          require(bytes.size() == expected.size(), "Resolved UI mask changed extent");
          bool exact = true;
          for (size_t pixel = 0; pixel * 4 < expected.size(); ++pixel) {
            float value{}; std::memcpy(&value, bytes.data() + pixel * 4, sizeof(value));
            exact &= std::abs(value - float(expected[pixel * 4 + 3]) / 255.f) <= 1e-6f;
          }
          require(exact, "Resolved automatic UI mask did not reproduce the pre-overwrite tag23 alpha");
        } else {
          require(bytes == expected, "Public UI hook captured post-tag overwrite or changed alpha/RGB bytes");
        }
        const auto path = hook_directory / (kind == 10 ? "ui-hook-tag23.bin" : "ui-hook-consumed.bin");
        sunshine_parity::write_bytes(path, bytes.data(), bytes.size());
        retained_bytes[retained_count++] = bytes;
        consumed |= resolved; tagged |= kind == 10;
      }
      require(consumed && tagged && read(mask.p) == overwritten,
        "Hook regression lacks both snapshots or the game allocation was not overwritten after the tag");
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
      present_frame();
      for (unsigned i = 0; i != retained_count; ++i)
        require(read(retained[i].p, D3D12_RESOURCE_STATE_COMMON) == retained_bytes[i],
          "Host acknowledgement or later game write changed UI snapshot");

      // Draws for a while (a second by default), then dumps; the caller
      // releases the dump.
      const auto dump_after_layer_frames = [&](std::uint64_t draw_ms = 1000) {
        for (const auto until = GetTickCount64() + draw_ms; GetTickCount64() < until;) { present_frame(); }
        box.request = box.state->request_id + 1;
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->request_id), box.request);
        for (const auto until = GetTickCount64() + 10000;
             std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->response_id), 0, 0)) != box.request &&
             GetTickCount64() < until;) { present_frame(); }
        require(box.state->response_id == box.request && box.state->response.result == dump::status::complete &&
            box.state->response.json_bytes <= dump::max_json_bytes && box.state->response.texture_count <= dump::max_textures,
          "Production Dump3D did not complete the offscreen UI layer capture");
        return nlohmann::json::parse(std::string(box.state->json, box.state->response.json_bytes));
      };
      // D3D12 reads back the decision texels: the tagged UI color's nearly
      // opaque pixels (texel 4), the layer's own counts (texel 7) and the H1
      // texel 10. Nothing claims here, so no evidence pass runs and texels 5
      // and 6 read as not measured; the layer's claim below measures them.
      {
        unsigned opaque{};
        for (size_t offset = 3; offset < expected.size(); offset += 4) opaque += expected[offset] == 255;
        const auto tagged_metadata = dump_after_layer_frames();
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
        present_frame();
        const auto &automatic = tagged_metadata.at("replay").at("source_alpha_auto");
        const auto &sampled = automatic.at("sampled_evidence");
        const auto &scene = sampled.at("scene");
        if (!(sampled.at("candidates").get<unsigned>() & candidate::ui_color) || sampled.at("alpha_opaque").size() != 2 ||
            sampled.at("alpha_opaque")[1] != opaque || sampled.at("alpha_opaque")[0] != 0u || !opaque ||
            !(sampled.at("accepted").get<unsigned>() & candidate::ui_color) || automatic.at("sampled_source") != 2u ||
            !(sampled.at("candidates").get<unsigned>() & candidate::layer) || !sampled.at("layer").contains("opaque") ||
            automatic.contains("scene_shadow") || automatic.at("scene_guard").at("hidden") != false ||
            automatic.at("scene_guard").at("pre_ui") != false || scene.at("ran") != false || !scene.contains("verdict") ||
            !scene.contains("n") || !scene.contains("d") || !sampled.at("pre_ui_scene").contains("valid") ||
            !sampled.at("pre_ui_scene").contains("image") || !sampled.contains("inferred_opaque") || !sampled.contains("claims") ||
            sampled.at("h1").at("applied") != false || sampled.at("h1").at("winner") != 2u || automatic.at("sampled_source") == 8u ||
            !sampled.at("one_way").contains("strong") || !sampled.at("one_way").contains("contradicted") ||
            sampled.at("reason") != "decided" || sampled.at("refused") != "none" || sampled.at("reused") != false ||
            // Rule H2 (fix 2) was removed with its fields.
            automatic.contains("still_screen") || sampled.contains("still_short_ms") ||
            tagged_metadata.at("replay").at("ui_detection").contains("still_bits"))
          throw std::runtime_error("D3D12 lost a decision texel, measured without a claim or kept a removed field: " +
            automatic.dump());
        evidence << "d3d12-decision-texels opaque_ui_color=" << opaque << " scene_ran=" << scene.at("ran") << '\n';
      }
      // Once tag 23 stops, the offscreen layer is the remaining candidate: live
      // tracking copies it before each clear, and once accepted it decides
      // while premultiplied (V1), tint included.
      render_tracked_depth = [&] { real_frame(); draw_layer(); };
      // H1 on D3D12 (docs/reshade-sbs.md, hidden-scene evidence): unless its
      // selective frames already earned it acceptance, the layer is opaque
      // everywhere over a flat presented frame that shows none of the depth's
      // edges, as a splash: an informative full claim (b). The scene guard
      // holds the hidden verdict after two samples and the frame is full-frame
      // UI (8). The claim makes the evidence passes run, actionably, so D3D12
      // reads back texel 5 (the presented frame) and texel 6 (the layer as the
      // pre-UI scene image) here, accepted layer or not.
      {
        const auto pixel_bytes = source_bytes.size() / (size_t(width) * height);
        std::vector<std::uint8_t> flat_bytes(source_bytes.size());
        for (size_t offset = 0; offset < flat_bytes.size(); offset += pixel_bytes)
          std::memcpy(flat_bytes.data() + offset, source_bytes.data(), pixel_bytes);
        com_ptr<ID3D12Resource> flat_upload;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT flat_footprint{};
        fill_upload(flat_upload, backbuffers[0]->GetDesc(), flat_bytes.data(), flat_footprint);
        struct presented_scope {
          com_ptr<ID3D12Resource> &upload, &other;
          D3D12_PLACED_SUBRESOURCE_FOOTPRINT &footprint, &other_footprint;
          bool &opaque;
          presented_scope(com_ptr<ID3D12Resource> &u, com_ptr<ID3D12Resource> &o, D3D12_PLACED_SUBRESOURCE_FOOTPRINT &f,
              D3D12_PLACED_SUBRESOURCE_FOOTPRINT &of, bool &layer): upload(u), other(o), footprint(f), other_footprint(of), opaque(layer) {
            std::swap(upload.p, other.p); std::swap(footprint, other_footprint); opaque = true;
          }
          ~presented_scope() { std::swap(upload.p, other.p); std::swap(footprint, other_footprint); opaque = false; }
        } flat_frame{source_upload, flat_upload, source_footprint, flat_footprint, opaque_layer};
        const auto hidden_metadata = dump_after_layer_frames();
        const auto &automatic = hidden_metadata.at("replay").at("source_alpha_auto");
        const auto &hidden_sample = automatic.at("sampled_evidence");
        const auto &scene = hidden_sample.at("scene");
        const auto &pre_ui_scene = hidden_sample.at("pre_ui_scene");
        bool full_frame = false;
        for (unsigned i = 0; i < box.state->response.texture_count; ++i) {
          const auto &item = box.state->response.textures[i];
          if (unsigned(item.kind) != unsigned(dump::artifact::ui_source_color)) continue;
          com_ptr<ID3D12Resource> resolved;
          checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(resolved.put())), "Open hidden-scene UI mask");
          const auto bytes = read(resolved.p, D3D12_RESOURCE_STATE_COMMON);
          full_frame = bytes.size() == size_t(width) * height * sizeof(float);
          for (size_t pixel = 0; full_frame && pixel * 4 < bytes.size(); ++pixel) {
            float value{}; std::memcpy(&value, bytes.data() + pixel * 4, sizeof(value));
            full_frame = value == 1.f;
          }
        }
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
        present_frame();
        // An accepted layer (earned by its selective frames, here or in an
        // earlier pass of this hook) is the S1 winner, flat at any coverage
        // (source 10), which H1 relabels 8 under a held hidden verdict
        // (selection revision 10: it pins at weight 1 either way, P1).
        const bool accepted_layer = hidden_sample.at("accepted").get<unsigned>() & candidate::layer;
        const auto flags = hidden_metadata.at("replay").at("ui_detection").at("flags").get<unsigned>();
        if (!full_frame || hidden_sample.at("layer").at("opaque") != size_t(width) * height || scene.at("ran") != true ||
            !scene.contains("n") || !scene.contains("d") || pre_ui_scene.at("ran") != true || pre_ui_scene.at("image") != "layer" ||
            (accepted_layer ?
              hidden_sample.at("h1").at("winner") != sunshine_game3d::ui_detection::source_layer ||
                automatic.at("sampled_source") !=
                  (hidden_sample.at("h1").at("applied") == true ? 8u : sunshine_game3d::ui_detection::source_layer) :
              automatic.at("sampled_source") != 8u || automatic.at("scene_guard").at("hidden") != true ||
              hidden_sample.at("h1").at("applied") != true ||
              !(hidden_sample.at("claims").get<unsigned>() & candidate::layer) || scene.at("valid") != true ||
              scene.at("verdict") != "hidden" ||
              (flags & (sunshine_game3d::ui_detection::stored_mask | sunshine_game3d::ui_detection::per_frame_scene_hidden)) !=
                (sunshine_game3d::ui_detection::stored_premultiplied | sunshine_game3d::ui_detection::per_frame_scene_hidden)))
          throw std::runtime_error("D3D12 did not flatten the layer's claim over a hidden scene (H1): " + automatic.dump() +
            " full_frame=" + std::to_string(full_frame));
        evidence << "d3d12-h1-layer accepted_layer=" << accepted_layer << " sampled_source=" << automatic.at("sampled_source") <<
          " scene_guard_hidden=" << automatic.at("scene_guard").at("hidden") << " scene_n=" << scene.at("n") << " scene_d=" <<
          scene.at("d") << " pre_ui_n=" << pre_ui_scene.at("n") << " pre_ui_d=" << pre_ui_scene.at("d") << '\n';
      }
      // An inferred source: the layer is accepted by valid selective samples
      // over at least 2 s (A1), so it draws selectively for longer first.
      const auto layer_metadata = dump_after_layer_frames(2500);
      bool layer_offered = false, tag23_offered = false;
      for (const auto &candidate : layer_metadata.at("ui_source").value("candidates", nlohmann::json::array())) {
        layer_offered |= candidate.value("source", std::string{}) == "ui_layer" && candidate.at("available_for_detection") == true &&
          candidate.at("format") == DXGI_FORMAT_R8G8B8A8_UNORM;
        tag23_offered |= candidate.value("source", std::string{}) == "sl_ui_color_alpha";
      }
      if (!layer_offered || tag23_offered)
        throw std::runtime_error("Auto did not offer the live offscreen UI layer once tag 23 stopped: " + layer_metadata.at("ui_source").dump());
      // An 8-bit layer forwards its stored flags in its own slot: the
      // premultiplied bound (1).
      const auto &layer_detection = layer_metadata.at("replay").at("ui_detection");
      if (layer_detection.at("ran_or_held") == "inactive" || !(layer_detection.at("candidates").get<unsigned>() & candidate::layer) ||
          !(layer_detection.at("accepted").get<unsigned>() & candidate::layer) ||
          (layer_detection.at("flags").get<unsigned>() & ~sunshine_game3d::ui_detection::per_frame_sample) !=
            sunshine_game3d::ui_detection::stored_premultiplied ||
          !layer_metadata.at("replay").contains("ui_pin"))
        throw std::runtime_error("The offscreen UI layer did not reach detection accepted with flags 1: " + layer_detection.dump());
      // The layer is one frame late, but its mask is its raw alpha like every
      // other source: exactly the 160/255 rectangle, never a texel beyond it.
      const auto &pin = layer_metadata.at("replay").at("ui_pin");
      if (!pin.at("soft_pin_gain").get<unsigned>() || pin.contains("late_margin"))
        throw std::runtime_error("The native shader lost its soft pin marker or still records a late-layer margin: " + pin.dump());
      bool layer_mask = false;
      std::vector<float> layer_field;
      size_t masked{}, rectangle{};
      for (size_t pixel = 0; pixel * 4 < layer_pattern.size(); ++pixel) rectangle += layer_pattern[pixel * 4 + 3] != 0;
      for (unsigned i = 0; i < box.state->response.texture_count; ++i) {
        const auto &item = box.state->response.textures[i];
        const bool resolved_mask = unsigned(item.kind) == unsigned(dump::artifact::ui_source_color);
        if (!resolved_mask && unsigned(item.kind) != unsigned(dump::artifact::final_field)) continue;
        com_ptr<ID3D12Resource> resolved;
        checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(resolved.put())), "Open layer UI mask or field");
        const auto bytes = read(resolved.p, D3D12_RESOURCE_STATE_COMMON);
        require(item.dxgi_format == DXGI_FORMAT_R32_FLOAT && bytes.size() == layer_pattern.size(), "Layer UI mask or field changed format or extent");
        if (!resolved_mask) {
          layer_field.resize(bytes.size() / sizeof(float));
          std::memcpy(layer_field.data(), bytes.data(), bytes.size());
          continue;
        }
        bool exact = true;
        masked = 0;
        for (size_t pixel = 0; pixel * 4 < layer_pattern.size(); ++pixel) {
          float value{}; std::memcpy(&value, bytes.data() + pixel * 4, sizeof(value));
          exact &= std::abs(value - float(layer_pattern[pixel * 4 + 3]) / 255.f) <= 1e-6f;
          masked += value != 0.f;
        }
        require(exact && masked == rectangle, "Resolved automatic UI mask is not the offscreen layer's raw alpha");
        layer_mask = true;
      }
      require(layer_mask && !layer_field.empty(), "Dump omitted the resolved UI mask or final field from the offscreen layer");
      // The rectangle lies on the UI plane, and the rows directly above and
      // below it keep the scene's own depth: no band of scene around the UI
      // is flattened (in columns the pin's collar and slope reach on).
      const unsigned left = width / 8, right = width / 4, top = height / 8, bottom = height / 4;
      const float plane = layer_field[size_t(height * 3 / 16) * width + width * 3 / 16];
      unsigned off_plane{}, scene_adjacent{};
      for (unsigned x = left; x < right; ++x) {
        for (unsigned y = top; y < bottom; ++y) off_plane += std::abs(layer_field[size_t(y) * width + x] - plane) * width > .5f;
        for (const unsigned y : {top - 1, bottom}) scene_adjacent += std::abs(layer_field[size_t(y) * width + x] - plane) * width > .5f;
      }
      if (off_plane || !scene_adjacent)
        throw std::runtime_error("The offscreen UI layer's raw alpha did not keep its UI on the plane beside an off-plane scene: off_plane=" +
          std::to_string(off_plane) + " scene_adjacent=" + std::to_string(scene_adjacent));
      evidence << "late-layer-raw-alpha masked=" << masked << " layer_texels=" << rectangle << " ui_off_plane=" << off_plane <<
        " scene_adjacent=" << scene_adjacent << '\n';
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
      present_frame();

      // A layer with straight alpha is still offered but V1-invalid: it
      // neither decides nor blocks.
      straight_layer = true;
      const auto straight_metadata = dump_after_layer_frames();
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
      straight_layer = false;
      present_frame();
      const auto &sampled = straight_metadata.at("replay").at("source_alpha_auto").at("sampled_evidence");
      if (!(sampled.at("candidates").get<unsigned>() & candidate::layer) || !sampled.at("layer").at("invalid").get<unsigned>() ||
          (sampled.at("valid_bits").get<unsigned>() & candidate::layer) ||
          straight_metadata.at("replay").at("source_alpha_auto").at("sampled_source") == sunshine_game3d::ui_detection::source_layer)
        throw std::runtime_error("The GPU admitted an offscreen layer with straight alpha: " + sampled.dump());
      // The live layer copy's queue facts on D3D12 (game3d_ui_layer.h, queue
      // order): each copy keeps the queue that ran it and is held until the
      // submission hook ran after its native call; one run on the presenting
      // queue is then in queue order; one run on a second queue is held
      // for the fence signal recorded after its submission, then until the CPU
      // sees that fence reach it, and only then offered (in fence order) while
      // the copy offered before it stays offered. The presenting queue never
      // waits for it: a Present while the second queue is gated completes at
      // once, and the schedule that froze Stellar Blade with frame generation
      // (10-06: the second queue waits for presenting-queue work queued after
      // the Present) runs at GPU speed.
      if (resource_type == 0) {
        using order = sunshine_game3d::ui_layer::read_order;
        const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
        const auto live_state = reinterpret_cast<BOOL (*)(sunshine_game3d::ui_layer::test_live_state *)>(
          GetProcAddress(module, "SunshineUILayerTestLive"));
        require(live_state, "Test add-on lacks the live layer observer");
        const auto query = [&] {
          sunshine_game3d::ui_layer::test_live_state value;
          require(live_state(&value) && value.capture_id, "The live layer has no copy");
          return value;
        };
        const auto wait = [&](ID3D12CommandQueue *on) {
          checked(on->Signal(completion.p, ++fence_value), "Fence the layer list");
          checked(completion->SetEventOnCompletion(fence_value, completion_event), "Observe the layer list");
          require(WaitForSingleObject(completion_event, 3000) == WAIT_OBJECT_0, "The layer list exceeded three seconds");
        };
        bool skip_layer = false;
        // Presenting-queue work the game queues before a frame's list.
        std::function<void()> before_frame;
        render_tracked_depth = [&] {
          if (before_frame) before_frame();
          real_frame();
          if (!skip_layer) draw_layer();
        };
        for (unsigned i = 0; i != 4; ++i) { present_frame(); }
        // The copy ran in the frame's own list on the presenting queue before
        // its Present: same queue, and the count names the Present before it.
        auto live = query();
        require(live.in_order && live.presents_since_copy == 1 && live.executed_queue &&
            live.executed_queue == live.presenting_queue,
          "A layer copy executed before its Present was not an in-order copy of the Present its count names");
        require(live.order == unsigned(order::queue) && !live.fence_value,
          "A layer copy on the presenting queue was ordered by a fence instead of queue order");
        // Inside ReShade's execute event of the frame's list, before the
        // native call, its copy on the presenting queue is held too (a Present
        // on another thread would otherwise read it before its write reached
        // the queue) and the copy before it stays offered; once the call
        // returned the submission hook offers it in queue order, no fence.
        {
          const auto before_same = live;
          execute_probe = {};
          execute_probe.query = live_state;
          execute_probe.list = game_native_command;
          require(execute_probe.list, "The frame's native list is not known");
          reshade::register_event<reshade::addon_event::execute_command_list>(probe_execute);
          {
            struct unregister_probe {
              ~unregister_probe() {
                execute_probe.armed = false;
                reshade::unregister_event<reshade::addon_event::execute_command_list>(probe_execute);
              }
            } scope;
            execute_probe.armed = true;
            present_frame();
          }
          const auto &window = execute_probe.seen;
          require(execute_probe.probed && window.capture_id == before_same.capture_id + 1 && window.owing == 1 &&
              !window.awaiting && window.offered_id == before_same.offered_id,
            ("A layer copy on the presenting queue was offered before its native submission returned: probed=" +
              std::to_string(execute_probe.probed) + " owing=" + std::to_string(window.owing) + " offered=" +
              std::to_string(window.offered_id) + " captured=" + std::to_string(window.capture_id) + " before=" +
              std::to_string(before_same.offered_id)).c_str());
          live = query();
          require(!live.owing && !live.awaiting && live.offered_id == window.capture_id && live.in_order &&
              live.order == unsigned(order::queue) && !live.fence_value && live.presents_since_copy == 1 &&
              live.executed_queue == live.presenting_queue,
            ("A layer copy on the presenting queue was not offered in queue order once its native submission returned: "
              "owing=" + std::to_string(live.owing) + " offered=" + std::to_string(live.offered_id) + " order=" +
              std::to_string(live.order) + " presents_since_copy=" + std::to_string(live.presents_since_copy)).c_str());
        }
        // A layer list executed on a second direct queue: the add-on's
        // submission hook signals that queue's fence after it.
        com_ptr<ID3D12CommandQueue> second;
        D3D12_COMMAND_QUEUE_DESC second_desc{};
        second_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        checked(game->CreateCommandQueue(&second_desc, IID_PPV_ARGS(second.put())), "Second direct queue");
        // The list also executes a bundle after its clear, as bundle-heavy
        // engines do for their UI draws: the list keeps its own copy.
        com_ptr<ID3D12CommandAllocator> bundle_allocator;
        com_ptr<ID3D12GraphicsCommandList> bundle;
        checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(bundle_allocator.put())), "Bundle allocator");
        checked(game->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocator.p, nullptr, IID_PPV_ARGS(bundle.put())),
          "Bundle");
        checked(bundle->Close(), "Close the bundle");
        // Two foreign layer lists, so a second frame's copy can be recorded
        // while the first one's is still queued.
        struct foreign_list {
          com_ptr<ID3D12CommandAllocator> allocator;
          com_ptr<ID3D12GraphicsCommandList> list;
          bool open = true;
        };
        std::array<foreign_list, 2> foreign;
        for (auto &l : foreign) {
          checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(l.allocator.put())),
            "Foreign layer allocator");
          checked(game->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, l.allocator.p, nullptr, IID_PPV_ARGS(l.list.put())),
            "Foreign layer list");
        }
        const float transparent[4]{};
        // The layer's first clear after the last Present, recorded into a
        // foreign list (a new list is open; later ones are reset after their
        // queue completed), executed on the second queue behind gate's value.
        const auto run_foreign_on = [&](foreign_list &l, ID3D12Fence *gate, std::uint64_t value) {
          if (!l.open) {
            checked(l.allocator->Reset(), "Reset the foreign layer allocator");
            checked(l.list->Reset(l.allocator.p, nullptr), "Reset the foreign layer list");
          }
          l.open = false;
          l.list->ClearRenderTargetView(layer_rtv, transparent, 0, nullptr);
          l.list->ExecuteBundle(bundle.p);
          checked(l.list->Close(), "Close the foreign layer list");
          if (gate) checked(second->Wait(gate, value), "Gate the second queue");
          ID3D12CommandList *foreign_lists[]{l.list.p};
          second->ExecuteCommandLists(1, foreign_lists);
        };
        // Frames whose layer copies run on the presenting queue confirm the
        // layer again after a long Present; the next frame skips its own clear.
        const auto confirm = [&] {
          skip_layer = false;
          for (unsigned i = 0; i != 4; ++i) { present_frame(); }
          skip_layer = true;
        };
        const auto timed_step = [&] {
          const auto start = GetTickCount64();
          present_frame();
          return GetTickCount64() - start;
        };
        // A fence the test signals from the CPU at open(), or after the limit
        // armed (arm), so a regression that waits on the GPU never strands a
        // queue; every exit opens it (to value).
        struct cpu_gate {
          com_ptr<ID3D12Fence> fence;
          HANDLE opened = CreateEventW(nullptr, TRUE, FALSE, nullptr);
          std::thread opener;
          std::uint64_t value = 1;
          void arm(DWORD limit_ms) {
            opener = std::thread([this, limit_ms] {
              WaitForSingleObject(opened, limit_ms);
              fence->Signal(value);
            });
          }
          void open() { if (opened) SetEvent(opened); }
          ~cpu_gate() {
            open();
            if (opener.joinable()) opener.join();
            if (fence.p) fence->Signal(value);
            if (opened) CloseHandle(opened);
          }
        };
        const auto make_gate = [&](cpu_gate &gate, const char *what) {
          require(gate.opened, "Gate event");
          checked(game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(gate.fence.put())), what);
        };
        // Every exit clears the frame hooks and the deferred completion.
        struct clear_hooks {
          std::function<void()> &before, &after;
          bool &defer;
          ~clear_hooks() { before = {}; after = {}; defer = false; }
        } hooks{before_frame, after_present, defer_completion};
        // A regression that waits on the GPU would hold a Present until the
        // gate opens at this limit.
        constexpr DWORD gate_limit_ms = 1000;
        // Held, then offered: the second queue's copy waits for a gate. Inside
        // ReShade's execute event, after the add-on's own handler and before
        // the native call, the copy is held for its fence signal; once the
        // call returned its signal is recorded but the CPU sees the fence
        // below it, so it stays held and the previous copy stays offered. A
        // Present meanwhile reads that previous copy at once (no GPU wait). A
        // reset of its list right after the submission (legal while it runs)
        // keeps it held. Once the fence reached it, the next Present offers
        // it in fence order.
        skip_layer = true;
        const auto before_held = query();
        std::uint64_t held_present_ms{}, held_fence_value{};
        com_ptr<ID3D12CommandAllocator> reset_allocator;
        checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(reset_allocator.put())),
          "Allocator for the early list reset");
        {
          cpu_gate gate;
          make_gate(gate, "Second-queue gate");
          execute_probe = {};
          execute_probe.query = live_state;
          reshade::register_event<reshade::addon_event::execute_command_list>(probe_execute);
          {
            struct unregister_probe {
              ~unregister_probe() {
                execute_probe.armed = false;
                reshade::unregister_event<reshade::addon_event::execute_command_list>(probe_execute);
              }
            } scope;
            execute_probe.armed = true;
            run_foreign_on(foreign[0], gate.fence.p, 1);
          }
          const auto submitted = query();
          const auto &window = execute_probe.seen;
          require(execute_probe.probed && window.capture_id == before_held.capture_id + 1 && window.owing == 1 &&
              window.offered_id == before_held.offered_id && window.offered_id != window.capture_id,
            ("A second-queue layer copy was offered before its fence signal was recorded: probed=" +
              std::to_string(execute_probe.probed) + " owing=" + std::to_string(window.owing) + " offered=" +
              std::to_string(window.offered_id) + " captured=" + std::to_string(window.capture_id) + " before=" +
              std::to_string(before_held.offered_id)).c_str());
          require(!submitted.owing && submitted.awaiting == 1 && submitted.capture_id == window.capture_id &&
              submitted.offered_id == before_held.offered_id && submitted.order == unsigned(order::queue),
            ("A second-queue layer copy was offered before the CPU saw its fence reach the value signalled after it: "
              "owing=" + std::to_string(submitted.owing) + " awaiting=" + std::to_string(submitted.awaiting) +
              " offered=" + std::to_string(submitted.offered_id)).c_str());
          gate.arm(gate_limit_ms);
          held_present_ms = timed_step();
          live = query();
          require(held_present_ms < gate_limit_ms / 2 && live.awaiting == 1 && live.offered_id == before_held.offered_id &&
              live.in_order && live.order == unsigned(order::queue),
            ("A Present waited for a held second-queue layer copy or read it before its fence reached it: elapsed_ms=" +
              std::to_string(held_present_ms) + " awaiting=" + std::to_string(live.awaiting) + " offered=" +
              std::to_string(live.offered_id) + " held=" + std::to_string(window.capture_id)).c_str());
          checked(foreign[0].list->Reset(reset_allocator.p, nullptr), "Reset the held copy's list after its submission");
          checked(foreign[0].list->Close(), "Close the early-reset list");
          require(query().awaiting == 1, "Resetting a submitted list dropped its held layer copy");
          gate.open();
        }
        wait(second.p);
        present_frame();
        live = query();
        held_fence_value = live.fence_value;
        require(!live.awaiting && live.offered_id == before_held.capture_id + 1 && live.in_order &&
            live.order == unsigned(order::fence_passed) && live.fence_value && live.executed_queue &&
            live.executed_queue != live.presenting_queue,
          ("A second-queue layer copy, after a bundle and an early list reset, was not offered in fence order once its "
            "fence reached it: offered=" + std::to_string(live.offered_id) + " order=" + std::to_string(live.order) +
            " awaiting=" + std::to_string(live.awaiting)).c_str());
        // The schedule that froze Stellar Blade (10-06, frame generation): each
        // frame's copy runs on the second queue behind a signal the presenting
        // queue queues after that frame's Present (presented), and the
        // presenting queue's next frame waits for the second queue's previous
        // copy (copied). The CPU queues frame N+1 while frame N may still run.
        // A GPU wait for frame N's copy before Present N closed this into a
        // cycle that only a CPU rescue left (1078 ms live); now Present N reads
        // the copy before it, both Presents run at GPU speed, and each copy is
        // offered once its fence reached it.
        confirm();
        const auto before_cycle = query();
        std::uint64_t cycle_elapsed{};
        {
          cpu_gate presented, copied;
          presented.value = 2;
          make_gate(presented, "Gate after each Present");
          make_gate(copied, "Second queue's copy of the previous frame");
          com_ptr<ID3D12CommandAllocator> spare;
          checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(spare.put())),
            "Second frame allocator");
          struct restore_allocator {
            com_ptr<ID3D12CommandAllocator> &current, &other;
            bool swapped = false;
            ~restore_allocator() { if (swapped) std::swap(current.p, other.p); }
          } frame_allocator{allocator, spare};
          run_foreign_on(foreign[0], presented.fence.p, 1);
          checked(second->Signal(copied.fence.p, 1), "Signal the second queue's frame-N copy");
          after_present = [&, gate = presented.fence.p] {
            checked(queue->Signal(gate, 1), "Open the second queue's gate after Present N");
          };
          const auto start = GetTickCount64();
          defer_completion = true;
          present_frame();
          defer_completion = false;
          // Present N ran latest() before its own GPU work, so frame N's copy,
          // gated behind it, was held and the copy before it was offered.
          const auto frame_n = query();
          run_foreign_on(foreign[1], presented.fence.p, 2);
          // Frame N's list may still execute: frame N+1 records with another
          // allocator.
          std::swap(allocator.p, spare.p);
          frame_allocator.swapped = true;
          before_frame = [&, gate = copied.fence.p] {
            checked(queue->Wait(gate, 1), "Frame N+1 waits for the second queue's frame-N copy");
          };
          after_present = [&, gate = presented.fence.p] {
            checked(queue->Signal(gate, 2), "Open the second queue's gate after Present N+1");
          };
          present_frame();
          cycle_elapsed = GetTickCount64() - start;
          before_frame = {};
          after_present = {};
          live = query();
          require(cycle_elapsed < gate_limit_ms / 2 && frame_n.awaiting == 1 && frame_n.offered_id == before_cycle.offered_id &&
              frame_n.order == unsigned(order::queue) && frame_n.capture_id == before_cycle.capture_id + 1 &&
              (live.offered_id == before_cycle.offered_id || live.offered_id == frame_n.capture_id) &&
              live.capture_id == frame_n.capture_id + 1,
            ("A frame-generation schedule with second-queue layer copies stalled the presenting queue or offered a copy "
              "before its fence reached it: elapsed_ms=" + std::to_string(cycle_elapsed) + " frame_n_awaiting=" +
              std::to_string(frame_n.awaiting) + " frame_n_offered=" + std::to_string(frame_n.offered_id) +
              " offered=" + std::to_string(live.offered_id) + " captured=" + std::to_string(live.capture_id)).c_str());
          wait(second.p);
        }
        // Both fences reached: the next Present offers frame N+1's copy, in
        // fence order (no queue is ever revoked).
        present_frame();
        live = query();
        require(!live.awaiting && live.offered_id == live.capture_id && live.in_order &&
            live.order == unsigned(order::fence_passed) && live.fence_value > held_fence_value &&
            live.executed_queue != live.presenting_queue,
          "Once their fences reached them, the schedule's second-queue layer copies were not offered in fence order");
        skip_layer = false;
        render_tracked_depth = [&] { real_frame(); draw_layer(); };
        present_frame();
        live = query();
        require(live.in_order && live.order == unsigned(order::queue) && live.executed_queue == live.presenting_queue,
          "The next layer copy run on the presenting queue was not in queue order again");
        evidence << "layer-cross-queue same_queue_held_until_submitted=1 held_until_signal=1 held_until_fence=1 held_present_ms=" << held_present_ms <<
          " early_reset_kept_held=1 fence_passed=1 fg_schedule_ms=" << cycle_elapsed << " presenting_queue_waits=0\n";
        std::printf("MEASURE layer cross-queue: held_present_ms=%llu fg_schedule_ms=%llu\n",
          static_cast<unsigned long long>(held_present_ms), static_cast<unsigned long long>(cycle_elapsed));
        std::puts("PASS D3D12 layer queue order: a copy run on the presenting queue before its Present is held until its "
          "native submission returned, then in queue order and names that Present; one run on a second direct queue, kept "
          "across a bundle, is held (the previous copy stays "
          "offered) until its fence signal is recorded after the native submission and the CPU sees that fence reach it, "
          "also across an early reset of its list; a Present meanwhile never waits for it; the frame-generation schedule "
          "that closed a cycle with a GPU wait runs at GPU speed and its copies are offered in fence order once reached; "
          "the next copy on the presenting queue is in queue order again");

        // C12 #17: a D3D12 render pass whose render target begins with a
        // CLEAR to (0, 0, 0, 0) is a layer clear. ReShade reports it before
        // the native BeginRenderPass, so the copy recorded there, outside the
        // pass, holds the previous frame's layer: the layer, no longer
        // cleared any other way, expires, is confirmed again by render-pass
        // clears alone, and its offered copy holds the drawn layer's alpha
        // exactly (a copy recorded after the native call would sit inside the
        // pass, after the clear). A CLEAR to alpha 1 records no copy.
        {
          com_ptr<ID3D12GraphicsCommandList4> pass_list;
          checked(commands->QueryInterface(IID_PPV_ARGS(pass_list.put())), "The game list has no render passes");
          float pass_clear[4]{};
          const auto pass_layer = [&] {
            D3D12_RENDER_PASS_RENDER_TARGET_DESC target{};
            target.cpuDescriptor = layer_rtv;
            target.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
            target.BeginningAccess.Clear.ClearValue.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            std::memcpy(target.BeginningAccess.Clear.ClearValue.Color, pass_clear, sizeof(pass_clear));
            target.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
            pass_list->BeginRenderPass(1, &target, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
            pass_list->EndRenderPass();
            // The UI drawn after the pass, as draw_layer draws it after its clear.
            transition(commands.p, layer.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION from{}, to{};
            from.pResource = layer_upload.p; from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            from.PlacedFootprint = layer_footprint;
            to.pResource = layer.p; to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            transition(commands.p, layer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
          };
          // Expire the layer: no clear for longer than max_clear_gap_ms.
          render_tracked_depth = [&] { real_frame(); };
          present_frame();
          Sleep(DWORD(sunshine_game3d::ui_layer::max_clear_gap_ms + 50));
          present_frame();
          const auto expired = query().capture_id;
          render_tracked_depth = [&] { real_frame(); pass_layer(); };
          for (unsigned i = 0; i != sunshine_game3d::ui_layer::confirm_clears; ++i) {
            present_frame();
            require(query().capture_id == expired, "A render-pass clear copied the layer before it was confirmed again");
          }
          present_frame();
          present_frame();
          live = query();
          require(live.capture_id > expired && live.offered_id == live.capture_id && live.offered_copy && live.in_order &&
              live.order == unsigned(order::queue) && live.format == unsigned(DXGI_FORMAT_R8G8B8A8_UNORM),
            ("Render-pass clears alone did not confirm the layer and offer its copy in queue order: captured=" +
              std::to_string(live.capture_id) + " expired=" + std::to_string(expired) + " offered=" +
              std::to_string(live.offered_id) + " format=" + std::to_string(live.format)).c_str());
          // The offered copy holds the previous frame's layer, exactly.
          com_ptr<ID3D12Resource> copy_back;
          auto *copy = reinterpret_cast<ID3D12Resource *>(live.offered_copy);
          const auto copy_desc = copy->GetDesc();
          D3D12_PLACED_SUBRESOURCE_FOOTPRINT back_footprint{};
          UINT64 back_bytes = 0;
          game->GetCopyableFootprints(&copy_desc, 0, 1, 0, &back_footprint, nullptr, nullptr, &back_bytes);
          buffer(copy_back, back_bytes, D3D12_HEAP_TYPE_READBACK);
          begin_commands();
          constexpr auto sampled = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
          transition(commands.p, copy, sampled, D3D12_RESOURCE_STATE_COPY_SOURCE);
          D3D12_TEXTURE_COPY_LOCATION back_from{}, back_to{};
          back_from.pResource = copy; back_from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          back_to.pResource = copy_back.p; back_to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
          back_to.PlacedFootprint = back_footprint;
          commands->CopyTextureRegion(&back_to, 0, 0, 0, &back_from, nullptr);
          transition(commands.p, copy, D3D12_RESOURCE_STATE_COPY_SOURCE, sampled);
          submit();
          const std::uint8_t *bytes = nullptr;
          checked(copy_back->Map(0, nullptr, reinterpret_cast<void **>(const_cast<std::uint8_t **>(&bytes))), "Map the layer copy");
          std::size_t mismatched = 0;
          for (unsigned y = 0; y != height; ++y)
            for (unsigned x = 0; x != width; ++x)
              mismatched += bytes[back_footprint.Offset + size_t(y) * back_footprint.Footprint.RowPitch + size_t(x) * 4 + 3] !=
                layer_pattern[(size_t(y) * width + x) * 4 + 3];
          const D3D12_RANGE no_write{};
          copy_back->Unmap(0, &no_write);
          require(!mismatched, ("The render-pass layer copy does not hold the previous frame's layer alpha: " +
            std::to_string(mismatched) + " pixels differ").c_str());
          // A CLEAR to alpha 1 is no layer clear: no copy.
          const auto before_opaque = query().capture_id;
          pass_clear[3] = 1.f;
          for (unsigned i = 0; i != 3; ++i) { present_frame(); }
          require(query().capture_id == before_opaque, "A render-pass CLEAR to alpha 1 copied the layer");
          render_tracked_depth = [&] { real_frame(); draw_layer(); };
          for (unsigned i = 0; i != 2; ++i) { present_frame(); }
          evidence << "layer-render-pass confirmed_by_pass_clears=1 copy_before_native_pass=1 opaque_clear_ignored=1\n";
          std::puts("PASS D3D12 layer render-pass clear: a pass beginning with a CLEAR to (0, 0, 0, 0) confirms the layer "
            "on its own and its copy, recorded before the native BeginRenderPass, holds the previous frame's layer alpha "
            "exactly; a CLEAR to alpha 1 records no copy");
        }

        // Dump 3D census copies follow the same queue order (game3d_ui_layer.h,
        // census_verdict): the dump reads one only with CPU proof that its
        // write completed (here: its second-queue fence reached and its list
        // reset), defers for it at most census_wait_ms after arming, never
        // makes the presenting queue wait, and omits a copy without proof.
        // The layer holds layer_pattern (the last frame drew it); the frames
        // below leave it alone, so only the foreign lists clear it.
        {
          using sunshine_game3d::ui_layer::census_wait_ms;
          skip_layer = true;
          render_tracked_depth = [&] {
            real_frame();
            if (!skip_layer) draw_layer();
          };
          const auto responded = [&] {
            return std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->response_id), 0, 0)) ==
              box.request;
          };
          // A new request; the next Present's poll arms the census, so the
          // clears after it are its copies.
          const auto request_census = [&] {
            box.request = box.state->request_id + 1;
            InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->request_id), box.request);
            present_frame();
            require(!responded(), "A census dump completed at the Present that armed it");
          };
          const auto await_census = [&](std::uint64_t limit_ms) {
            for (const auto until = GetTickCount64() + limit_ms; !responded() && GetTickCount64() < until;) { present_frame(); }
            require(responded() && box.state->response.result == dump::status::complete &&
                box.state->response.json_bytes <= dump::max_json_bytes && box.state->response.texture_count <= dump::max_textures,
              "Production Dump3D did not complete a census capture");
            return nlohmann::json::parse(std::string(box.state->json, box.state->response.json_bytes));
          };
          const auto release_census = [&] {
            InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
            present_frame();
          };
          const auto layer_row = [&](const nlohmann::json &metadata) {
            for (const auto &entry : metadata.at("ui_layer_census").at("candidates"))
              if (entry.at("source") == layer_source) return entry;
            throw std::runtime_error("The dump census has no row for the UI layer: " + metadata.at("ui_layer_census").dump());
          };
          const auto artifact = [&](unsigned kind) {
            for (unsigned i = 0; i < box.state->response.texture_count; ++i)
              if (unsigned(box.state->response.textures[i].kind) == kind) return int(i);
            return -1;
          };
          const auto reset_list = [&](foreign_list &l) {
            checked(l.allocator->Reset(), "Reset the census list's allocator");
            checked(l.list->Reset(l.allocator.p, nullptr), "Reset the census list");
            checked(l.list->Close(), "Close the reset census list");
            l.open = false;
          };
          // (a) Gated on the second queue: the dump is not taken (nothing is
          // published) and no Present waits while the copy's fence is below its
          // value; once the fence reached it and its list was reset, the dump
          // reads it in fence order and holds the layer as it was before the
          // foreign clear.
          std::uint64_t gated_ms{};
          nlohmann::json fenced_row;
          {
            cpu_gate gate;
            make_gate(gate, "Census copy gate");
            request_census();
            run_foreign_on(foreign[0], gate.fence.p, 1);
            gate.arm(gate_limit_ms);
            const auto submitted = query();
            require(submitted.census_copies == 1 && submitted.census_awaiting == 1 && submitted.census_pending == 1,
              ("A gated second-queue census copy was not held for its fence: copies=" +
                std::to_string(submitted.census_copies) + " awaiting=" + std::to_string(submitted.census_awaiting) +
                " pending=" + std::to_string(submitted.census_pending)).c_str());
            for (unsigned i = 0; i != 4; ++i) {
              gated_ms = std::max(gated_ms, timed_step());
              require(!responded(), "The dump read a census copy whose second-queue fence had not reached its value");
            }
            require(gated_ms < gate_limit_ms / 2 && query().census_awaiting == 1,
              ("A Present waited for a gated census copy, or the copy left its fence: slowest_ms=" +
                std::to_string(gated_ms)).c_str());
            gate.open();
          }
          wait(second.p);
          reset_list(foreign[0]);
          {
            const auto metadata = await_census(5000);
            const auto &census = metadata.at("ui_layer_census");
            fenced_row = layer_row(metadata);
            const auto presenting = query().presenting_queue;
            if (!(fenced_row.at("captured") == true && fenced_row.at("status") == "captured_before_clear" &&
                  fenced_row.at("read_order") == "fence_passed" && fenced_row.at("fence_value") > 0 &&
                  fenced_row.at("executed_queue") != 0 && fenced_row.at("executed_queue") != presenting &&
                  fenced_row.at("executed_queue") != sunshine_game3d::ui_layer::mixed_queue &&
                  fenced_row.at("presents_since_copy") >= 1 && census.at("ordering") == "proven" &&
                  census.at("settled") == true && census.at("wait_ms") < census_wait_ms))
              throw std::runtime_error("A second-queue census copy was not read in fence order once its fence reached it: " +
                census.dump());
            const auto index = artifact(fenced_row.at("artifact_id").get<unsigned>());
            require(index >= 0, "The dump omitted a census copy proven by its fence");
            const auto &item = box.state->response.textures[unsigned(index)];
            com_ptr<ID3D12Resource> copied;
            checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(copied.put())),
              "Open the fenced census copy");
            require(read(copied.p, D3D12_RESOURCE_STATE_COMMON) == layer_pattern,
              "The fenced census copy is not the layer as it was before the foreign clear");
            release_census();
          }
          // (b) Gated past census_wait_ms: the dump is taken at the deadline
          // without reading the copy (awaiting_fence, no artifact, settled
          // false) while the gate is still closed, and no Present waits.
          std::uint64_t deadline_ms{};
          {
            constexpr DWORD census_gate_ms = DWORD(census_wait_ms) + 1000;
            cpu_gate gate;
            make_gate(gate, "Census gate past the wait");
            request_census();
            run_foreign_on(foreign[1], gate.fence.p, 1);
            gate.arm(census_gate_ms);
            const auto until = GetTickCount64() + census_gate_ms - 300;
            while (!responded() && GetTickCount64() < until) deadline_ms = std::max(deadline_ms, timed_step());
            require(responded() && gate.fence->GetCompletedValue() < 1,
              "The census dump did not complete at its deadline while the copy's queue was still gated");
            const auto metadata = await_census(0);
            const auto &census = metadata.at("ui_layer_census");
            const auto row = layer_row(metadata);
            if (!(row.at("captured") == false && row.at("status") == "awaiting_fence" && row.at("read_order").is_null() &&
                  row.at("fence_value") > 0 && census.at("settled") == false && census.at("wait_ms") >= census_wait_ms &&
                  artifact(row.at("artifact_id").get<unsigned>()) < 0 && deadline_ms < gate_limit_ms / 2))
              throw std::runtime_error("A census copy gated past the wait was read, waited for, or not reported awaiting "
                "its fence: slowest_ms=" + std::to_string(deadline_ms) + " " + census.dump());
            release_census();
            gate.open();
          }
          wait(second.p);
          reset_list(foreign[1]);
          // (c) A clear recorded into a list that never executes: omitted at the
          // deadline as not_executed; the list is then reset, never executed,
          // so the retired copy is never written.
          {
            auto &l = foreign[0];
            request_census();
            checked(l.allocator->Reset(), "Reset the unexecuted census list's allocator");
            checked(l.list->Reset(l.allocator.p, nullptr), "Reset the unexecuted census list");
            l.list->ClearRenderTargetView(layer_rtv, transparent, 0, nullptr);
            checked(l.list->Close(), "Close the unexecuted census list");
            const auto metadata = await_census(census_wait_ms + 4000);
            const auto &census = metadata.at("ui_layer_census");
            const auto row = layer_row(metadata);
            if (!(row.at("captured") == false && row.at("status") == "not_executed" && row.at("executed_queue") == 0 &&
                  census.at("settled") == false && census.at("wait_ms") >= census_wait_ms &&
                  artifact(row.at("artifact_id").get<unsigned>()) < 0))
              throw std::runtime_error("A census copy whose list never executed was read or not reported not_executed: " +
                census.dump());
            release_census();
            reset_list(l);
          }
          skip_layer = false;
          render_tracked_depth = [&] { real_frame(); draw_layer(); };
          evidence << "layer-census fence_passed=1 gated_present_ms=" << gated_ms << " fence_value=" <<
            fenced_row.at("fence_value") << " deadline_awaiting_fence=1 deadline_present_ms=" << deadline_ms <<
            " not_executed=1 presenting_queue_waits=0\n";
          std::printf("MEASURE layer census: gated_present_ms=%llu deadline_present_ms=%llu\n",
            static_cast<unsigned long long>(gated_ms), static_cast<unsigned long long>(deadline_ms));
          std::puts("PASS D3D12 layer census order: a census copy run on a second queue is not read while its fence is "
            "below its value (no Present waits), and is read in fence order with the layer's content once its fence "
            "reached it and its list was reset; one still gated at the deadline and one never executed are omitted "
            "(awaiting_fence, not_executed) and the dump completes without them");
        }
      }
      evidence << "public-ui-hook descriptor_type=" << resource_type << " source_format=90 captured_format=87 tag=23 covered=" << covered <<
        " exact_consumed_and_optional=1 post_tag_opaque_overwrite=1 host_ack_immutable=1 offscreen_ui_layer=1 live_ui_layer_mask=1 frame_tags=" << bool(sl.frame_tag) << '\n';
      std::puts("PASS actual public SL tag23 hook: typeless90 to typed87, automatic GPU mask equals pre-overwrite alpha and optional dump exact before opaque overwrite; host lease immutable");
      std::puts("PASS live offscreen UI layer: offered in its own slot beside tag 23 from the first frame; without tag 23 and once accepted, the layer copied before its clear is the automatic UI mask exactly as its raw alpha, tint within twice its alpha included, the UI on the plane and the scene rows beside it off it; straight alpha is V1-invalid; opaque everywhere over a flat presented frame it is full-frame UI (by the layer route 8 before acceptance, by itself after)");
    }

    void run_automatic_ui_tags(HMODULE sdk) {
      using namespace sunshine_streamline;
      namespace dump = ::game3d_debug;
      const auto tag_call = reinterpret_cast<abi_v2::set_tag>(GetProcAddress(sdk, "slSetTag"));
      const auto new_token = reinterpret_cast<abi_v2::get_new_frame_token>(GetProcAddress(sdk, "slGetNewFrameToken"));
      const auto constants_call = reinterpret_cast<abi_v2::set_constants>(GetProcAddress(sdk, "slSetConstants"));
      const bool failed_off = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_UI_FAILED_OFF_TEST");
      set_options_t options_call{};
      if (failed_off) {
        const auto getter = reinterpret_cast<abi_v2::get_feature_function>(GetProcAddress(sdk, "slGetFeatureFunction"));
        const auto set_result = reinterpret_cast<void (*)(std::int32_t)>(GetProcAddress(sdk, "SunshineFixtureSetOptionsResult"));
        void *function{};
        require(getter && set_result && getter(1000, "slDLSSGSetOptions", function) == 0 && function,
          "Failed-Off regression requires the controllable public SDK options result");
        set_result(1); options_call = reinterpret_cast<set_options_t>(function);
      }
      const auto set_foreground = reinterpret_cast<void (*)(HWND)>(GetProcAddress(
        GetModuleHandleW(L"SunshineSBSTest.addon64"), "SunshineSbsTestSetForeground"));
      require(tag_call && new_token && constants_call && set_foreground && source_format == DXGI_FORMAT_R10G10B10A2_UNORM,
        "Public UI regression requires the real global SL tag entry and native R10 color");
      struct foreground_scope {
        void (*set)(HWND);
        ~foreground_scope() { set(nullptr); }
      } foreground{set_foreground};
      set_foreground(window); // Select the hidden fixture without moving desktop focus.
      const abi_v2::viewport viewport{{nullptr, viewport_guid, 1}, 0};
      com_ptr<ID3D12Resource> hudless, upload;
      auto desc = backbuffers[0]->GetDesc();
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(hudless.put())), "Create public HUDless input");
      auto hudless_bytes = source_bytes;
      std::vector<float> expected(size_t(width) * height);
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const auto pixel = size_t(y) * width + x;
        const bool covered = x >= width / 8 && x < width / 4 && y >= height * 3 / 4 && y < height * 7 / 8;
        expected[pixel] = covered ? 1.f : 0.f;
        if (covered) {
          std::uint32_t value{}; std::memcpy(&value, hudless_bytes.data() + pixel * 4, 4);
          value = (value & ~1023u) | ((value & 1023u) - 16u);
          std::memcpy(hudless_bytes.data() + pixel * 4, &value, 4);
        }
      }
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
      fill_upload(upload, desc, hudless_bytes.data(), footprint);
      unsigned frame_number{};
      render_tracked_depth = [&] {
        draw(*decoy); write_depth_pixels();
        auto *list = reinterpret_cast<ID3D12GraphicsCommandList *>(game_native_command);
        abi_v2::frame_token *token{}; const auto frame = ++frame_number;
        require(new_token(token, &frame) == 0 && token, "Public tag regression could not obtain frame token");
        abi_v2::constants constants{}; constants.base = {nullptr, constants_guid, 1};
        constants.common.camera_view_to_clip = camera.projection;
        constants.common.clip_to_camera_view = camera.inverse_projection;
        constants.common.camera_near = camera.near_plane; constants.common.camera_far = camera.far_plane;
        constants.common.camera_fov = camera.fov; constants.common.camera_aspect = camera.aspect;
        constants.common.camera_right[0] = constants.common.camera_up[1] = constants.common.camera_forward[2] = 1;
        constants.depth_inverted = 1;
        require(constants_call(constants, *token, viewport) == 0, "Public standalone constants changed the SDK result");
        transition(list, hudless.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = upload.p; from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
        to.pResource = hudless.p; to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(list, hudless.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        // Exact live descriptor pattern: zero Resource header/dimensions and
        // full tagged extents; one global depth/motion/HUDless batch at viewport0.
        abi_v2::resource resources[3]{};
        resources[0].native = selected->resource.p; resources[0].state = unsigned(selected->state);
        resources[1].native = selected->resource.p; resources[1].state = unsigned(selected->state);
        resources[2].native = hudless.p; resources[2].state = D3D12_RESOURCE_STATE_COPY_SOURCE;
        const abi_v2::resource_tag tags[]{
          {{nullptr, tag_guid, 1}, &resources[0], 0, 0, active_area},
          {{nullptr, tag_guid, 1}, &resources[1], 1, 0, active_area},
          {{nullptr, tag_guid, 1}, &resources[2], 2, 0, {0, 0, width, height}}};
        require(tag_call(viewport, tags, 3, list) == 0, "Public mixed UI tags changed the SDK result");
        if (options_call) {
          const fg_options off{{nullptr, fg_options_type, 3}, 0, 0};
          require(options_call(viewport, off) == 1, "Repeated failed Off call changed the SDK result");
        }
      };
      // Only public SDK hooks supply UI/depth. The optional failure case calls
      // unsupported FG Off immediately after each tag, matching the live game.
      // No evaluation or private capture export is used.
      const auto warm_until = GetTickCount64() + 750;
      do { present_frame(); } while (GetTickCount64() < warm_until);
      settle("public-SL-global-tags-no-FG-no-evaluate");
      struct mailbox {
        HANDLE handle{}; dump::shared_state_t *state{}; std::uint64_t request{};
        ~mailbox() {
          if (state) {
            if (request) InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state->released_id), request);
            UnmapViewOfFile(state);
          }
          if (handle) CloseHandle(handle);
        }
      } box;
      const auto name = std::wstring(dump::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      box.handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
      require(box.handle != nullptr, "Public automatic UI test has no production dump mailbox");
      box.state = static_cast<dump::shared_state_t *>(MapViewOfFile(box.handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(dump::shared_state_t)));
      require(box.state && box.state->signature == dump::magic && box.state->protocol_version == dump::version,
        "Public automatic UI test has incompatible dump mailbox");
      FILETIME created{}, exited{}, kernel{}, user{};
      require(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user), "Identify UI dump consumer");
      box.state->consumer_pid = GetCurrentProcessId();
      box.state->consumer_creation_time = (std::uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime;
      box.state->consumer_nonce = 0x55494155544fULL;
      std::uint64_t request_generation{};
      for (unsigned round = 0; round != 2; ++round) {
        for (unsigned i = 0; i != 20; ++i) { present_frame(); }
        box.request = box.state->request_id + 1;
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->request_id), box.request);
        const auto until = GetTickCount64() + 10000;
        while (std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->response_id), 0, 0)) != box.request && GetTickCount64() < until) {
          present_frame();
        }
        require(box.state->response_id == box.request && box.state->response.result == dump::status::complete &&
            box.state->response.json_bytes <= dump::max_json_bytes && box.state->response.texture_count <= dump::max_textures,
          "Production dump did not finish automatic public UI capture");
        const auto metadata = nlohmann::json::parse(std::string(box.state->json, box.state->response.json_bytes));
        {
          std::ofstream output(runtime_directory / ("public-ui-auto-" + std::to_string(round) + ".json"));
          output << metadata.dump(2) << '\n';
          require(output.good(), "Write public UI regression evidence");
        }
        const auto &replay = metadata.at("replay");
        const auto &attempt = replay.at("source_alpha_capture_attempt");
        const auto &gate = attempt.at("hook_gate");
        require(attempt.at("input_observed") == true && gate.at("state") == "admitted" &&
            gate.at("matching_requests") == 1 && gate.at("seen_kinds") == 8 &&
            attempt.at("request").at("viewport") == 0 && gate.at("viewport") == 0,
          "Actual public UI hook never reached the live request: inspect public-ui-auto JSON gate evidence");
        const auto generation = attempt.at("request_generation").get<std::uint64_t>();
        require(!round || generation == request_generation, "Normal public-hook frames repeatedly replaced the UI request");
        request_generation = generation;
        require(replay.at("source_alpha_ui_fg_mode").at("known") == false &&
            replay.at("source_alpha_ui") == true && replay.at("ui_alpha_source") == "ui_source_color",
          "Automatic public UI detection required FG or bypassed the resolved mask");
        bool paired{};
        for (const auto &candidate : metadata.at("ui_source").at("candidates"))
          if (candidate.at("source") == "sl_hudless_difference")
            paired = candidate.at("available_for_detection") == true && candidate.at("current_present_pair") == true;
        require(paired, "Live public HUDless source did not reach current-frame GPU comparison");
        require(active(inspect()), "Public standalone tags did not retain the actual SL depth and finite native scale");
        bool consumed{}, stereo{};
        for (unsigned i = 0; i != box.state->response.texture_count; ++i) {
          const auto &item = box.state->response.textures[i];
          if (item.kind == dump::artifact::sbs) {
            require(item.dxgi_format == DXGI_FORMAT_R16G16B16A16_FLOAT && item.width == 2 * width && item.height == height,
              "Public standalone depth dump has wrong SBS format");
            com_ptr<ID3D12Resource> texture;
            checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(texture.put())), "Open standalone public-tag SBS");
            const auto bytes = read(texture.p, D3D12_RESOURCE_STATE_COMMON);
            const auto eye_row = size_t(width) * 8;
            require(bytes.size() == eye_row * 2 * height, "Public standalone SBS has wrong byte count");
            for (unsigned y = 0; y < height; ++y)
              stereo |= std::memcmp(bytes.data() + y * eye_row * 2, bytes.data() + y * eye_row * 2 + eye_row, eye_row) != 0;
            sunshine_parity::write_bytes(runtime_directory / ("public-depth-sbs-" + std::to_string(round) + ".bin"), bytes.data(), bytes.size());
          }
          if (item.kind != dump::artifact::ui_source_color) continue;
          require(item.dxgi_format == DXGI_FORMAT_R32_FLOAT && item.width == width && item.height == height,
            "Automatic public UI dump has wrong resolved-mask format");
          com_ptr<ID3D12Resource> texture;
          checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(texture.put())), "Open resolved automatic UI mask");
          const auto bytes = read(texture.p, D3D12_RESOURCE_STATE_COMMON);
          require(bytes.size() == expected.size() * sizeof(float) && std::memcmp(bytes.data(), expected.data(), bytes.size()) == 0,
            "Automatic public HUDless comparison did not produce exact selective current-frame coverage");
          sunshine_parity::write_bytes(runtime_directory / ("public-ui-mask-" + std::to_string(round) + ".bin"), bytes.data(), bytes.size());
          consumed = true;
        }
        require(consumed && stereo && !v2_capture_calls && !v1_capture_calls,
          "Public tag result lacked the exact mask, distinct stereo eyes, or used private capture injection");
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
      }
      render_tracked_depth = {};
      evidence << "public-ui-auto viewport=0 zero_header=1 omitted_size=1 mixed_tags=0,1,2 FG_unknown=1 repeated_failed_Off=" << failed_off << " SL_evaluate=0 persistent_request="
        << request_generation << " exact_resolved_mask=1 real_SL_depth=1 distinct_eyes=1\n";
      std::puts("PASS actual public mixed SL tags -> persistent UI request -> same-present HUDless comparison -> exact automatic mask, FG unknown and no SL evaluation");
    }

    // The FG cases (run_fg) on the synthetic Streamline SDK's public entry
    // points, which the production hooks observe: one frame script each.
    static constexpr sunshine_streamline::guid fg_options_type{0xfac5f1cb,0x2dfd,0x4f36,{0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}};
    using fg_options = sunshine_streamline::abi_v2::fg_options_prefix;
    static_assert(sizeof(fg_options) == 40);
    using set_options_t = std::int32_t (*)(const sunshine_streamline::abi_v2::viewport &, const fg_options &);
    struct sl_entry_points {
      sunshine_streamline::abi_v2::get_new_frame_token new_token{};
      sunshine_streamline::abi_v2::set_constants constants{};
      sunshine_streamline::abi_v2::set_tag_for_frame frame_tag{}; // Absent before 2.7: global tags only.
      sunshine_streamline::abi_v2::set_tag global_tag{};
      sunshine_streamline::abi_v2::evaluate_feature evaluate{};
      set_options_t volatile set_options{};
    } sl;
    void (*set_foreground)(HWND){};
    const sunshine_streamline::abi_v2::viewport fg_viewport{{nullptr, sunshine_streamline::viewport_guid, 1}, 0};
    // The newest FG frame's explicit index and token, and when its depth tag
    // call began and returned.
    unsigned fg_frame{};
    sunshine_streamline::abi_v2::frame_token *last_token{};
    std::uint64_t tag_started{}, tag_returned{};

    void configure_fg(bool enabled) {
      const fg_options value{{nullptr, fg_options_type, 3}, enabled ? 1u : 0u, enabled ? 1u : 0u};
      require(sl.set_options(fg_viewport, value) == 0, "Observed FG options changed the public SDK return value");
    }
    sunshine_streamline::abi_v2::constants camera_constants(bool reset) const {
      sunshine_streamline::abi_v2::constants constants{}; constants.base = {nullptr, sunshine_streamline::constants_guid, 1};
      constants.common.camera_view_to_clip = camera.projection;
      constants.common.clip_to_camera_view = camera.inverse_projection;
      constants.common.camera_near = camera.near_plane; constants.common.camera_far = camera.far_plane;
      constants.common.camera_fov = camera.fov; constants.common.camera_aspect = camera.aspect;
      constants.common.camera_right[0] = constants.common.camera_up[1] = constants.common.camera_forward[2] = 1;
      constants.depth_inverted = 1;
      constants.reset = reset ? 1u : 0u;
      return constants;
    }
    // The tagged depth as every FG frame describes it: the selected R32 UAV.
    sunshine_streamline::abi_v2::resource depth_resource() const {
      sunshine_streamline::abi_v2::resource resource{}; resource.base = {nullptr, sunshine_streamline::resource_guid, 1};
      resource.type = 8; resource.native = selected->resource.p; resource.state = unsigned(selected->state);
      resource.width = selected->width; resource.height = selected->height;
      resource.native_format = DXGI_FORMAT_R32_FLOAT; resource.mip_levels = resource.array_layers = 1;
      return resource;
    }
    // Each FG frame begins as the game's: the decoy and the tracked depth
    // drawn, then the frame's own explicit token (message: its case's).
    sunshine_streamline::abi_v2::frame_token &begin_fg_frame(const char *message) {
      draw(*decoy); write_depth_pixels();
      sunshine_streamline::abi_v2::frame_token *token{}; const auto frame = ++fg_frame;
      require(sl.new_token(token, &frame) == 0 && token, message);
      last_token = token;
      return *token;
    }
    // A real FG frame: constants, an OnlyValidNow depth tag on the game's list
    // (a frame tag where the SDK has one), and its same-frame SR evaluation.
    void real_fg_frame() {
      using namespace sunshine_streamline;
      auto &token = begin_fg_frame("FG SDK did not supply an explicit frame token");
      const auto constants = camera_constants(false);
      require(sl.constants(constants, token, fg_viewport) == 0, "FG constants observation changed the SDK result");
      const auto resource = depth_resource();
      const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 0, active_area};
      tag_started = GetTickCount64();
      const auto tagged = sl.frame_tag ? sl.frame_tag(token, fg_viewport, &tag, 1, reinterpret_cast<void *>(game_native_command)) :
        sl.global_tag(fg_viewport, &tag, 1, reinterpret_cast<void *>(game_native_command));
      require(tagged == 0,
        "FG tag observation changed the SDK result");
      tag_returned = GetTickCount64();
      const base_structure *inputs[]{&fg_viewport.base};
      require(sl.evaluate(0, token, inputs, 1, reinterpret_cast<void *>(game_native_command)) == 0,
        "FG same-frame SR observation changed the SDK result");
    }
    // An FG frame whose ValidUntilPresent depth tag has no command list.
    void commandless_fg_frame() {
      using namespace sunshine_streamline;
      auto &token = begin_fg_frame("Command-less FG SDK did not supply a frame token");
      require(sl.constants(camera_constants(false), token, fg_viewport) == 0, "Command-less FG constants changed the SDK result");
      const auto resource = depth_resource();
      const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 1, active_area};
      require(sl.global_tag(fg_viewport, &tag, 1, nullptr) == 0, "Command-less FG tag changed the SDK result");
    }
    // The same viewport's SR evaluation with its own depth input.
    void sr_evaluate(const sunshine_streamline::abi_v2::frame_token &token) {
      using namespace sunshine_streamline;
      const auto resource = depth_resource();
      const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 1, active_area};
      const base_structure *inputs[]{&fg_viewport.base, &tag.base};
      require(sl.evaluate(0, token, inputs, 2, reinterpret_cast<void *>(game_native_command)) == 0,
        "SR evaluation with its own depth changed the SDK result");
    }
    // Assassin's Creed Shadows with FG on (10-22): an SR evaluation (sr), then
    // either an accepted FG tag on the game's list (fg_commands) or a refused
    // command-less one followed by its untag.
    void fallback_frame(bool fg_commands, bool sr) {
      using namespace sunshine_streamline;
      auto &token = begin_fg_frame("FG fallback SDK did not supply a frame token");
      require(sl.constants(camera_constants(false), token, fg_viewport) == 0, "FG fallback constants changed the SDK result");
      if (sr) sr_evaluate(token);
      const auto resource = depth_resource();
      const auto command = reinterpret_cast<void *>(game_native_command);
      if (fg_commands) {
        const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 0, active_area};
        require((sl.frame_tag ? sl.frame_tag(token, fg_viewport, &tag, 1, command) : sl.global_tag(fg_viewport, &tag, 1, command)) == 0,
          "Accepted FG tag changed the SDK result");
        return;
      }
      const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 1, active_area};
      const abi_v2::resource_tag untag{{nullptr, tag_guid, 1}, nullptr, 0, 0, {}};
      require(sl.global_tag(fg_viewport, &tag, 1, nullptr) == 0 && sl.global_tag(fg_viewport, &untag, 1, command) == 0,
        "Refused FG tag or its untag changed the SDK result");
    }

    // An FG gap after its cache was invalidated: no depth and no identity.
    void missing_fg(const char *label) {
      present_frame(); const auto value = inspect(); log_status(label, value);
      require(!value.ready() && !value.source.ready && !value.source.reused_depth &&
        !value.source.current.capture && !value.source.current.sequence,
        "Invalidated native FG cache supplied previous depth without a fresh successful copy");
    }
    // An FG gap that holds seed's admitted real depth and scale unchanged.
    status held_fg(const status &seed, const char *label) {
      present_frame(); const auto value = inspect();
      require(active(value) && value.source.reused_depth &&
        value.source.current.capture == seed.source.current.capture &&
        value.source.current.sequence == seed.source.current.sequence &&
        value.source.current.resource == seed.source.current.resource && same_scale(value.scale, seed.scale),
        "Native FG hold changed the admitted real depth identity or scale");
      if (label) log_status(label, value);
      return value;
    }
    status recovered_fresh(const status &before, const char *label) {
      render_tracked_depth = [this] { real_fg_frame(); };
      const auto value = settle(label);
      require(!value.source.reused_depth && value.source.current.capture != before.source.current.capture &&
        value.source.current.sequence > before.source.current.sequence,
        "Native FG cache recovered without a newly completed real capture");
      return value;
    }
    // Presents until twelve consecutive Presents have fresh depth from the FG
    // input (frame_generation_input) or from the viewport's SR evaluation.
    void settle_route(const char *label, bool frame_generation_input) {
      const auto until = GetTickCount64() + 15000;
      unsigned consecutive{};
      status s;
      do {
        present_frame(); s = inspect();
        consecutive = active(s) && !s.source.reused_depth && s.source.frame_generation_input == frame_generation_input ?
          consecutive + 1 : 0;
      } while (consecutive < 12 && GetTickCount64() < until);
      log_status(label, s);
      require(consecutive >= 12, frame_generation_input ? "An accepted FG copy did not take its viewport's depth back" :
        "A silent FG scope did not let the same viewport's SR depth through");
    }
    std::size_t readiness_losses() {
      return reshade_log_count({"Sunshine depth readiness: lost"}) +
        reshade_log_count({"Sunshine depth readiness:", "explicit_reuse_invalidation"});
    }
    // Eight rounds of a real Present of a silent FG scope (fresh SR depth)
    // and its generated Presents, each holding the real frame's SR copy.
    void silent_cadence(const char *label, unsigned generated_frames) {
      for (unsigned round = 0; round < 8; ++round) {
        render_tracked_depth = [this] { fallback_frame(false, true); };
        present_frame(); const auto real = inspect();
        require(active(real) && !real.source.reused_depth && !real.source.frame_generation_input,
          "A real Present of a silent FG scope lost its fresh SR depth");
        render_tracked_depth = {};
        for (unsigned generated = 0; generated < generated_frames; ++generated) {
          present_frame(); const auto value = inspect();
          if (!round && !generated) log_status(label, value);
          require(active(value) && value.source.reused_depth && !value.source.frame_generation_input &&
            value.source.current.capture == real.source.current.capture &&
            value.source.current.sequence == real.source.current.sequence && same_scale(value.scale, real.scale),
            "A generated Present of a silent FG scope did not hold its real frame's SR depth");
        }
      }
    }

    void run_fg(HMODULE sdk) {
      using namespace sunshine_streamline;
      set_foreground = reinterpret_cast<void (*)(HWND)>(GetProcAddress(
        GetModuleHandleW(L"SunshineSBSTest.addon64"), "SunshineSbsTestSetForeground"));
      require(set_foreground, "FG reuse fixture needs the existing controlled foreground observer");
      struct foreground_scope {
        void (*set)(HWND);
        ~foreground_scope() { set(nullptr); }
      } foreground{set_foreground};
      // The export stop at a focus loss ends FG reuse once. This selects our
      // real hidden game HWND in the test observer without moving UI focus.
      set_foreground(window);
      const auto getter = reinterpret_cast<abi_v2::get_feature_function>(GetProcAddress(sdk, "slGetFeatureFunction"));
      sl.new_token = reinterpret_cast<abi_v2::get_new_frame_token>(GetProcAddress(sdk, "slGetNewFrameToken"));
      sl.constants = reinterpret_cast<abi_v2::set_constants>(GetProcAddress(sdk, "slSetConstants"));
      sl.frame_tag = reinterpret_cast<abi_v2::set_tag_for_frame>(GetProcAddress(sdk, "slSetTagForFrame"));
      sl.global_tag = reinterpret_cast<abi_v2::set_tag>(GetProcAddress(sdk, "slSetTag"));
      sl.evaluate = reinterpret_cast<abi_v2::evaluate_feature>(GetProcAddress(sdk, "slEvaluateFeature"));
      require(getter && sl.new_token && sl.constants && sl.global_tag && sl.evaluate, "FG metadata fixture lacks public SL exports");
      std::printf("MEASURE SL SDK capability frame_tags=%u; using %s tags\n", unsigned(sl.frame_tag != nullptr), sl.frame_tag ? "frame" : "global");
      void *function{};
      require(getter(1000, "slDLSSGSetOptions", function) == 0 && function, "Public SDK did not expose FG options");
      sl.set_options = reinterpret_cast<set_options_t>(function);
      for (unsigned frame = 0; frame < 4; ++frame) step();
      // This metadata-only SDK exposes the FG tag boundary. Keep its already
      // validated FG nomination path for crop tests; direct SL coverage above
      // separately exercises successful/failed evaluation while FG is off.
      configure_fg(true);
      fg_crop_pairs_and_off();
      fg_cache();
      for (const std::uint32_t type : {0u, 8u})
        run_script({type ? "public-ui-hook-type-8" : "public-ui-hook-type-0", 150000, {}, [&, type] { run_ui_hook(type); }, {}});
      fg_until_present_clear();
      fg_untag();
      fg_commandless_refusal();
      fg_silent_scope();
      fg_dlssg_evaluation();
      fg_silent_scope_without_api_source();
      fg_linear_rr();
    }

    // Public SDK crops, real/generated pairs, dynamic resolution and FG off.
    void fg_crop_pairs_and_off() {
      run_script({"sdk-crops", 45000,
        [&] {
          render_tracked_depth = [this] { real_fg_frame(); };
          settle("SLFG-SDK-full-allocation");
        },
        [&] {
          // These are public SDK extents, not injected shader coordinates. Native
          // rendering must accept the same crop even though no FX uniform exists.
          active_area = {0, 0, selected->width, selected->height-3};
          settle("SLFG-SDK-padded-three-rows");
          active_area = {8, 16, selected->width-32, selected->height-16};
          settle("SLFG-SDK-nonzero-offset-crop");
          std::puts("PASS native SL camera depth accepts padded and nonzero-offset active rectangles with zero installed FX");
        }, {}});
      run_script({"generated-pairs", 30000,
        [&] {
          active_area = {0, 0, selected->width, selected->height-3};
          settle("SLFG-padded-real-depth");
        },
        [&] {
          for (unsigned pair = 0; pair < 8; ++pair) {
            const auto pair_start = GetTickCount64();
            render_tracked_depth = [this] { real_fg_frame(); };
            present_frame(); const auto real = inspect();
            log_status("SLFG-pair-real-depth", real);
            require(active(real) && !real.source.reused_depth, "Fresh FG source did not produce current native depth");
            // One generated presentation has current color but no new provider call
            // or game depth draw. Observe production retention without modifying it.
            render_tracked_depth = {};
            present_frame(); const auto generated = inspect();
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
        }, {}});
      // Dynamic resolution (Assassin's Creed Shadows with FG on, 10-23): the
      // tagged extent of the same source changes every second real frame.
      // Each real Present has fresh depth of its own extent and each
      // generated one holds it with the real frame's scale; the matrix scale
      // stays active (a reset would recalibrate). The fixture crops one fixed
      // scene, so a smaller extent shows other content and the scale may
      // adapt to it within its rate bound. The provider reports the changes
      // on its render size line.
      run_script({"dynamic-resolution", 30000, {},
        [&] {
          const sunshine_streamline::extent drs[] = {{0, 0, selected->width, selected->height-3},
            {0, 0, selected->width*7/8, selected->height*7/8}, {0, 0, selected->width*3/4, selected->height*3/4}};
          for (unsigned pair = 0; pair < 12; ++pair) {
            active_area = drs[pair / 2 % 3];
            render_tracked_depth = [this] { real_fg_frame(); };
            present_frame(); const auto real = inspect();
            render_tracked_depth = {};
            present_frame(); const auto generated = inspect();
            if (pair % 2 == 0) log_status("SLFG-dynamic-resolution-generated", generated);
            require(active(real) && !real.source.reused_depth && active(generated) && generated.source.reused_depth &&
              generated.source.current.capture == real.source.current.capture && same_scale(generated.scale, real.scale),
              "A dynamic render size of the same FG source lost depth or reset its scale");
          }
          active_area = {0, 0, selected->width, selected->height-3};
          render_tracked_depth = [this] { real_fg_frame(); };
          // Its line (at most every 5 s) names the smallest extent of this run.
          const auto smallest = "active=" + std::to_string(drs[2].width) + 'x' + std::to_string(drs[2].height) + "..";
          require(reshade_log_contains({"Sunshine depth render size: changed=", smallest.c_str(), "provider=Streamline",
              "depth stays ready"}),
            "Dynamic render sizes produced no render size line");
        }, {}});
      run_script({"fg-off", 30000, {},
        [&] {
          render_tracked_depth = {};
          configure_fg(false);
          present_frame();
          const auto off = inspect(); log_status("FG-off-without-new-depth", off);
          require(!off.ready() && !off.source.ready && !off.source.reused_depth,
            "Disabling FG retained previous-frame depth for a color-only presentation");
        },
        [&] {
          active_area = {0, 0, selected->width, selected->height};
          render_tracked_depth = [this] { draw_frame(); };
          settle("SL-recovered-after-FG-off");
          std::puts("PASS native SLFG 2x: actual cropped SDK depth, eight real/generated pairs with retained completed depth, FG-off rejection and fresh SL recovery; no FX");
        }});
    }

    // Keep the original measured trajectory above unchanged. These additional
    // native presentations exercise the lifetime of the private display copy,
    // without an FX callback, injected readiness or synthetic GPU completion.
    void fg_cache() {
      status expiry_seed;
      std::uint64_t seed_started{}, seed_returned{};
      unsigned seed_frame{};
      run_script({"cache-expiry", 45000,
        [&] {
          configure_fg(true); render_tracked_depth = [this] { real_fg_frame(); };
          expiry_seed = settle("SLFG-cache-expiry-seed");
          seed_started = tag_started; seed_returned = tag_returned;
          seed_frame = fg_frame;
        },
        [&] {
          render_tracked_depth = {};
          unsigned holds{};
          std::uint64_t last_held{};
          do {
            held_fg(expiry_seed, holds ? nullptr : "SLFG-cache-expiry-first-hold");
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
          missing_fg("SLFG-cache-expired-after-repeated-holds");
          const auto expired = GetTickCount64();
          require(expired >= expiry_at && expired >= last_held &&
            expired - last_held < sunshine_scene_depth::maximum_source_age_ms && fg_frame == seed_frame &&
            tag_started == seed_started && tag_returned == seed_returned,
            "Native expiry fixture did not distinguish the original capture age from renewed hold age");
          missing_fg("SLFG-cache-expired-no-revival");
          std::printf("MEASURE native FG cache expiry holds=%u tag_begin_ms=%llu tag_end_ms=%llu last_hold_ms=%llu expired_ms=%llu\n",
            holds, static_cast<unsigned long long>(seed_started), static_cast<unsigned long long>(seed_returned),
            static_cast<unsigned long long>(last_held), static_cast<unsigned long long>(expired));
        },
        [&] { recovered_fresh(expiry_seed, "SLFG-cache-expiry-fresh-recovery"); }});

      status reset_seed;
      std::uint64_t reset_seed_started{};
      unsigned reset_seed_frame{};
      run_script({"camera-reset", 45000,
        [&] {
          reset_seed = settle("SLFG-cache-reset-seed");
          reset_seed_started = tag_started;
          reset_seed_frame = fg_frame;
        },
        [&] {
          render_tracked_depth = {};
          held_fg(reset_seed, "SLFG-cache-reset-before-hold");
          // A reset ends temporal history, not the finished copy of an earlier
          // frame: like any gap, the copy is held only within its own source age.
          render_tracked_depth = [this] {
            require(last_token, "Native camera-reset fixture has no existing frame token");
            const auto constants = camera_constants(true);
            require(sl.constants(constants, *last_token, fg_viewport) == 0, "Camera reset changed the SDK result");
          };
          held_fg(reset_seed, "SLFG-cache-camera-reset-hold");
          render_tracked_depth = [this] {
            const auto constants = camera_constants(false);
            require(sl.constants(constants, *last_token, fg_viewport) == 0, "Camera recovery changed the SDK result");
          };
          held_fg(reset_seed, "SLFG-cache-camera-valid-without-depth");
          require(GetTickCount64() - reset_seed_started < sunshine_scene_depth::maximum_source_age_ms,
            "Native camera-reset fixture expired before its holds were observed");
          render_tracked_depth = {};
          require(fg_frame == reset_seed_frame, "Camera-reset hold case accidentally produced a new real frame");
        },
        [&] { recovered_fresh(reset_seed, "SLFG-cache-camera-reset-fresh-recovery"); }});

      status mode_seed;
      unsigned mode_seed_frame{};
      run_script({"mode-off-on", 45000,
        [&] {
          mode_seed = settle("SLFG-cache-mode-seed");
          mode_seed_frame = fg_frame;
        },
        [&] {
          render_tracked_depth = {};
          held_fg(mode_seed, "SLFG-cache-mode-before-hold");
          configure_fg(false); missing_fg("SLFG-cache-mode-off");
          configure_fg(true); missing_fg("SLFG-cache-mode-on-without-depth");
          missing_fg("SLFG-cache-mode-on-no-revival");
          require(fg_frame == mode_seed_frame, "FG mode no-revival case accidentally produced a new real frame");
        },
        [&] {
          recovered_fresh(mode_seed, "SLFG-cache-mode-fresh-recovery");
          std::puts("PASS native FG cache: repeated holds cannot renew capture age; camera reset keeps the finished copy within its age; FG off/on cannot revive old depth; fresh copies recover all three intervals; no FX");
        }});
    }

    // The Witcher 3 pattern with FG off: each frame tags depth valid until
    // present, evaluates, then clears the tag before present, while another
    // viewport sends reset constants every frame. Neither the clear nor the
    // foreign reset is lost evidence, so every presentation has fresh depth.
    void fg_until_present_clear() {
      run_script({"until-present-clear-foreign-reset", 20000,
        [&] { configure_fg(false); },
        [&] {
          render_tracked_depth = [this] {
            using namespace sunshine_streamline;
            const abi_v2::viewport foreign_viewport{{nullptr, viewport_guid, 1}, 7};
            auto &token = begin_fg_frame("Until-present SDK did not supply a frame token");
            require(sl.constants(camera_constants(true), token, foreign_viewport) == 0 &&
                sl.constants(camera_constants(false), token, fg_viewport) == 0,
              "Until-present constants observation changed the SDK result");
            const auto resource = depth_resource();
            const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 1, active_area};
            const auto command = reinterpret_cast<void *>(game_native_command);
            require(sl.global_tag(fg_viewport, &tag, 1, command) == 0, "Until-present tag observation changed the SDK result");
            const base_structure *inputs[]{&fg_viewport.base};
            require(sl.evaluate(0, token, inputs, 1, command) == 0, "Until-present evaluation changed the SDK result");
            const abi_v2::resource_tag cleared{{nullptr, tag_guid, 1}, nullptr, 0, 1, {}};
            require(sl.global_tag(fg_viewport, &cleared, 1, command) == 0, "Until-present clear changed the SDK result");
          };
          settle("SL-until-present-clear-foreign-reset");
          std::puts("PASS native SL provider: depth cleared after evaluation and another viewport's per-frame camera reset keep fresh depth every presentation; no FX");
        }, {}});
    }

    // Assassin's Creed Shadows with FG on (10-22): after its FG depth tag
    // each frame sends a global depth untag, a null Resource or one whose
    // native is null, with or without a command list (sl_core_api.h: a null
    // resource removes the tag). An untag is not a capture attempt: the FG
    // depth captured at the tag stays the viewport's source, so every
    // presentation keeps fresh FG depth.
    void fg_untag() {
      using namespace sunshine_streamline;
      com_ptr<ID3D12Resource> hudless;
      run_script({"untag", 110000,
        [&] { configure_fg(true); },
        [&] {
          for (const bool null_native : {false, true}) for (const bool untag_commands : {true, false}) {
            render_tracked_depth = [&] {
              real_fg_frame();
              abi_v2::resource empty{};
              const abi_v2::resource_tag untag{{nullptr, tag_guid, 1}, null_native ? &empty : nullptr, 0, 0, {}};
              require(sl.global_tag(fg_viewport, &untag, 1, untag_commands ? reinterpret_cast<void *>(game_native_command) : nullptr) == 0,
                "FG depth untag changed the SDK result");
            };
            settle(null_native ? untag_commands ? "SLFG-untag-null-native" : "SLFG-untag-null-native-without-commands" :
              untag_commands ? "SLFG-untag-null-resource" : "SLFG-untag-null-resource-without-commands");
          }
          // DLSS-G integrations tag HUD-less color every frame, possibly in the
          // same call as the depth untag. Only the call's depth tags make it an
          // untag, so a real color tag beside them keeps fresh FG depth too.
          {
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = width; desc.Height = height;
            desc.DepthOrArraySize = desc.MipLevels = 1; desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
            desc.SampleDesc.Count = 1; desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
            checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
              D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(hudless.put())), "Create FG HUD-less input");
          }
          for (const bool untag_commands : {true, false}) {
            render_tracked_depth = [&] {
              real_fg_frame();
              abi_v2::resource color{}; color.base = {nullptr, resource_guid, 1};
              color.native = hudless.p; color.state = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
              color.width = width; color.height = height; color.native_format = DXGI_FORMAT_R10G10B10A2_UNORM;
              color.mip_levels = color.array_layers = 1;
              const abi_v2::resource_tag tags[]{{{nullptr, tag_guid, 1}, nullptr, 0, 0, {}},
                {{nullptr, tag_guid, 1}, &color, 2, 1, {0, 0, width, height}}};
              require(sl.global_tag(fg_viewport, tags, 2, untag_commands ? reinterpret_cast<void *>(game_native_command) : nullptr) == 0,
                "FG depth untag beside a HUD-less tag changed the SDK result");
            };
            settle(untag_commands ? "SLFG-untag-beside-hudless" : "SLFG-untag-beside-hudless-without-commands");
          }
          const abi_v2::resource_tag no_hudless{{nullptr, tag_guid, 1}, nullptr, 2, 0, {}};
          require(sl.global_tag(fg_viewport, &no_hudless, 1, nullptr) == 0, "HUD-less untag changed the SDK result");
          // A malformed tag that names a native resource is no untag: it still
          // withdraws the FG source, so no presentation copies fresh depth.
          render_tracked_depth = [this] {
            real_fg_frame();
            // A typed Resource of version 0 (a later version is read by its v1 fields).
            abi_v2::resource malformed{}; malformed.base = {nullptr, resource_guid, 0};
            malformed.type = 8; malformed.native = selected->resource.p; malformed.state = unsigned(selected->state);
            malformed.width = selected->width; malformed.height = selected->height;
            const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &malformed, 0, 0, active_area};
            require(sl.global_tag(fg_viewport, &tag, 1, reinterpret_cast<void *>(game_native_command)) == 0,
              "Malformed FG depth tag changed the SDK result");
          };
          for (unsigned frame = 0; frame < 12; ++frame) {
            present_frame(); const auto s = inspect();
            if (!frame) log_status("SLFG-malformed-tag-after-depth", s);
            require(!(active(s) && !s.source.reused_depth), "A malformed FG tag naming a native was treated as an untag");
          }
        },
        [&] {
          render_tracked_depth = [this] { real_fg_frame(); };
          settle("SLFG-recovered-after-malformed-tag");
          std::puts("PASS native SLFG untag: a null Resource or null native after each FG tag, with or without commands and beside a HUD-less tag, keeps fresh FG depth every presentation; a malformed tag still withdraws; no FX");
        }});
    }

    // A ValidUntilPresent FG tag may legally arrive without a command list,
    // which leaves no point to copy it at: the boundary is withdrawn, and
    // the bounded withdrawal line names that reason and the tag. The test
    // add-on's refusal query reports it at once, with its contract class.
    void fg_commandless_refusal() {
      run_script({"commandless-refusal", 30000, {},
        [&] {
          const auto before_commandless = last_refusal().serial;
          render_tracked_depth = [this] { commandless_fg_frame(); };
          {
            namespace refusal = sunshine_depth_refusal;
            const auto refused = refusal_after(before_commandless);
            std::printf("MEASURE native-provider SLFG-commandless-refusal provider=%u gate=%u class=%u name=%s viewport=%u fg=%u sequence=%llu serial=%llu\n",
              unsigned(refused.provider), unsigned(refused.where), unsigned(refused.reason), refused.name, refused.viewport,
              unsigned(refused.frame_generation), static_cast<unsigned long long>(refused.sequence), static_cast<unsigned long long>(refused.serial));
            require(refused.serial > before_commandless && refused.provider == refusal::adapter::streamline &&
                refused.where == refusal::gate::source && refused.reason == refusal::reason_class::no_copy_point &&
                !std::strcmp(refused.name, "boundary_without_commands") && refused.frame_generation && refused.viewport == fg_viewport.value,
              "The refusal query did not report the command-less FG tag as a no_copy_point source refusal");
          }
          require(reshade_log_contains({"Sunshine Streamline FG depth refused: reason=boundary_without_commands", "tag_type=0",
              "scope=global", "lifecycle=1", "command=0x0 ", "fg=1"}),
            "A command-less FG tag boundary was refused without naming its reason");
        },
        [&] {
          render_tracked_depth = [this] { real_fg_frame(); };
          settle("SLFG-recovered-after-commandless-tag");
          std::puts("PASS native SLFG refusal: a command-less ValidUntilPresent FG tag is refused and logged as boundary_without_commands; no FX");
        }});
    }

    // Assassin's Creed Shadows with FG on (10-22): FG options stay on, but
    // every FG depth tag is refused (here: sent without a command list)
    // and then untagged, while the same viewport's SR evaluation tags its
    // own depth with commands. FG owns that viewport's depth only while an
    // FG copy was accepted within a second; after that its scope is silent
    // and SR supplies depth. An accepted FG copy takes ownership back at
    // once, and SR stays shut out while FG copies keep arriving.
    void fg_silent_scope() {
      run_script({"silent-scope", 80000, {},
        [&] {
          render_tracked_depth = [this] { fallback_frame(false, true); };
          const auto silent_start = GetTickCount64();
          for (std::uint64_t now = silent_start; now - silent_start < 700; now = GetTickCount64()) {
            present_frame(); const auto s = inspect();
            require(!(s.ready() && s.source.ready && !s.source.frame_generation_input && !s.source.reused_depth),
              "SR depth replaced FG within a second of its last accepted FG copy");
          }
          settle_route("SLFG-silent-scope-SR-depth", false);
          require(GetTickCount64() - silent_start >= 1000, "SR took the viewport's depth before the FG scope fell silent");
          require(reshade_log_contains({"Sunshine Streamline frame generation depth scope: silent viewport=0"}) &&
              reshade_log_contains({"Sunshine Streamline FG depth refused: reason=boundary_without_commands", "fg_scope=silent", "withdrawn=0"}),
            "The silent FG scope or its refused boundaries were not logged");
          // Its generated Presents (no SL call, no depth draw) hold the real
          // frame's SR copy exactly as FG depth is held, at 2x and 3x, also while
          // the game window is in the background: the export stop invalidates
          // reuse once, never on every Present. Assassin's Creed Shadows (10-23)
          // logged a readiness loss on every generated Present for the 42 s its
          // window was in the background (no_retained_depth).
          const auto losses_before_cadence = readiness_losses();
          silent_cadence("SLFG-silent-scope-SR-generated-2x", 1);
          silent_cadence("SLFG-silent-scope-SR-generated-3x", 2);
          // The window leaves the foreground: one real Present stops the export
          // (its pause line, then one reuse invalidation of a fresh copy, which
          // stays ready); the next real frame arms the hold again.
          const auto pauses = reshade_log_count({"Sunshine SBS: export inactive (not_foreground"});
          set_foreground(nullptr);
          render_tracked_depth = [this] { fallback_frame(false, true); };
          present_frame();
          require(active(inspect()), "The export stop withdrew a fresh SR copy");
          for (const auto until = GetTickCount64() + 7000;
              reshade_log_count({"Sunshine SBS: export inactive (not_foreground"}) == pauses && GetTickCount64() < until;) {
            present_frame();
            require(active(inspect()), "A real Present in the background lost its fresh SR depth");
          }
          require(reshade_log_count({"Sunshine SBS: export inactive (not_foreground"}) == pauses + 1,
            "The fixture's focus loss did not stop the export exactly once");
          silent_cadence("SLFG-silent-scope-SR-background-2x", 1);
          silent_cadence("SLFG-silent-scope-SR-background-3x", 2);
          set_foreground(window);
          silent_cadence("SLFG-silent-scope-SR-foreground-again-2x", 1);
          require(readiness_losses() == losses_before_cadence,
            "Silent-scope generated Presents logged a readiness loss or a reuse invalidation");
        },
        [&] {
          render_tracked_depth = [this] { fallback_frame(true, true); };
          settle_route("SLFG-accepted-copy-owns-again", true);
          for (unsigned frame = 0; frame < 30; ++frame) {
            present_frame(); const auto s = inspect();
            require(active(s) && s.source.frame_generation_input,
              "SR depth competed with a viewport whose FG copies keep being accepted");
          }
          require(reshade_log_contains({"Sunshine Streamline frame generation depth scope: live viewport=0"}),
            "The FG scope's return to live was not logged");
        }});
    }

    // DLSS-G's own NGX evaluation (feature 11) of each real frame, as in
    // Assassin's Creed Shadows (10-23): the game tags FG depth, HUD-less and
    // Backbuffer without a command list (refused), its SR evaluation tags its own
    // depth, and DLSS-G evaluates DLSSG.Depth (with DLSSG.DepthInverted and
    // DLSSG.DepthSubrect*), DLSSG.HUDLess and DLSSG.Backbuffer on its own
    // command list. Copies are made only in a state observed on that list.
    // Without one every copy is refused, counted, and nothing changes: the
    // silent scope keeps SR depth. With one, DLSSG.Depth is the FG source
    // (its accepted copies keep the scope live, so SR stays out and the
    // command-less tag withdraws nothing), generated Presents hold it, and
    // its HUD-less/Backbuffer pair reaches UI detection as one exact batch.
    void fg_dlssg_evaluation() {
      using namespace sunshine_streamline;
      const auto exe = GetModuleHandleW(nullptr);
      const auto ngx_create = reinterpret_cast<decltype(&NVSDK_NGX_D3D12_CreateFeature)>(
        GetProcAddress(exe, "NVSDK_NGX_D3D12_CreateFeature"));
      const auto ngx_evaluate = reinterpret_cast<decltype(&NVSDK_NGX_D3D12_EvaluateFeature)>(
        GetProcAddress(exe, "NVSDK_NGX_D3D12_EvaluateFeature"));
      const auto ngx_release = reinterpret_cast<decltype(&NVSDK_NGX_D3D12_ReleaseFeature)>(
        GetProcAddress(exe, "NVSDK_NGX_D3D12_ReleaseFeature"));
      require(ngx_create && ngx_evaluate && ngx_release, "DLSS-G fixture lacks its exported NGX entry points");
      auto color = backbuffers[0]->GetDesc();
      color.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      com_ptr<ID3D12Resource> hudless, backbuffer;
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &color, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
        nullptr, IID_PPV_ARGS(hudless.put())), "Create the DLSS-G HUD-less input");
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &color, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
        nullptr, IID_PPV_ARGS(backbuffer.put())), "Create the DLSS-G Backbuffer input");
      // The DLSSG.* keys the DLSS-G helpers set (nvsdk_ngx_helpers_dlssg_d3d.h):
      // one real frame per evaluation, inverted depth, and its subrect each frame.
      ngx_fixture::parameters inputs;
      inputs.set_resource("DLSSG.Depth", selected->resource.p);
      inputs.set_resource("DLSSG.HUDLess", hudless.p);
      inputs.set_resource("DLSSG.Backbuffer", backbuffer.p);
      inputs.set_ui("DLSSG.DepthInverted", 1);
      inputs.set_ui("DLSSG.MultiFrameIndex", 1);
      inputs.set_ui("DLSSG.MultiFrameCount", 1);
      void *feature{};
      bool observed_state = false, tag_commands = false, ui_tag_commands = false;
      const auto evaluations_before = ngx_fixture::evaluated;
      const auto dlssg_frame = [&] {
        fallback_frame(tag_commands, true);
        auto *list = reinterpret_cast<ID3D12GraphicsCommandList *>(game_native_command);
        // The game's own HUD-less (2) and Backbuffer (53) tags, in their
        // declared state: command-less as in Shadows, or on the game's list.
        const auto color_resource = [&](ID3D12Resource *native) {
          abi_v2::resource value{}; value.base = {nullptr, resource_guid, 1};
          value.type = 8; value.native = native;
          value.state = unsigned(D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
          value.width = width; value.height = height;
          value.native_format = unsigned(color.Format); value.mip_levels = value.array_layers = 1;
          return value;
        };
        const auto hudless_resource = color_resource(hudless.p), backbuffer_resource = color_resource(backbuffer.p);
        const abi_v2::resource_tag ui_tags[]{{{nullptr, tag_guid, 1}, &hudless_resource, 2, 1, {0, 0, width, height}},
          {{nullptr, tag_guid, 1}, &backbuffer_resource, 53, 1, {0, 0, width, height}}};
        require(sl.global_tag(fg_viewport, ui_tags, 2, ui_tag_commands ? reinterpret_cast<void *>(game_native_command) : nullptr) == 0,
          "The HUD-less and Backbuffer tags changed the SDK result");
        if (!feature)
          require(ngx_create(list, 11, &inputs, &feature) == ngx_fixture::success && feature, "DLSS-G feature creation failed");
        require(selected->state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "DLSS-G fixture depth left its UAV state");
        inputs.set_ui("DLSSG.DepthSubrectBaseX", active_area.left);
        inputs.set_ui("DLSSG.DepthSubrectBaseY", active_area.top);
        inputs.set_ui("DLSSG.DepthSubrectWidth", active_area.width);
        inputs.set_ui("DLSSG.DepthSubrectHeight", active_area.height);
        // DLSS-G's recording transitions its inputs, as Streamline's DLSS-G
        // plugin does; without that no state is observed on it.
        constexpr auto read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (observed_state) {
          transition(list, selected->resource.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read);
          transition(list, hudless.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, read);
          transition(list, backbuffer.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, read);
        }
        require(ngx_evaluate(list, feature, &inputs, nullptr) == ngx_fixture::success,
          "DLSS-G evaluation observation changed the SDK result");
        if (observed_state) {
          transition(list, selected->resource.p, read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
          transition(list, hudless.p, read, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
          transition(list, backbuffer.p, read, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        }
      };
      std::uint64_t before_dlssg{};
      run_script({"dlssg-evaluation", 160000,
        [&] {
          before_dlssg = last_refusal().serial;
          render_tracked_depth = dlssg_frame;
          settle_route("DLSSG-unobserved-state-SR-depth", false);
        },
        [&] {
          for (unsigned frame = 0; frame < 30; ++frame) {
            present_frame(); const auto s = inspect();
            require(active(s) && !s.source.reused_depth && !s.source.frame_generation_input,
              "DLSS-G depth without an observed state changed the viewport's SR depth");
          }
          {
            // DLSS-G's evaluation is each frame's last call: its refusal is the newest.
            namespace refusal = sunshine_depth_refusal;
            const auto refused = last_refusal();
            std::printf("MEASURE native-provider DLSSG-unobserved-refusal provider=%u gate=%u class=%u name=%s stage=%u status=%u fg=%u serial=%llu\n",
              unsigned(refused.provider), unsigned(refused.where), unsigned(refused.reason), refused.name, unsigned(refused.stage),
              unsigned(refused.status), unsigned(refused.frame_generation), static_cast<unsigned long long>(refused.serial));
            require(refused.serial > before_dlssg && refused.provider == refusal::adapter::dlss_g &&
                refused.where == refusal::gate::capture && refused.stage == sunshine_streamline::depth_capture::record_stage::missing_state &&
                refused.reason == refusal::reason_class::state_unprovable && refused.frame_generation,
              "The refusal query did not report DLSS-G's unobserved depth as a state_unprovable capture refusal");
          }
          require(reshade_log_contains({"Sunshine NGX depth:", "dlssg_depth={copied=0 inner_frames=0 refused=missing_state:"}),
            "Refused DLSS-G depth copies were not counted by their missing_state reason");
          require(!reshade_log_number_above("Sunshine UI capture gate:", "hudless_presents={batch=", 0, 0),
            "A same-batch HUD-less pair was counted before DLSS-G's inputs were observed");
          observed_state = true;
          settle_route("DLSSG-evaluation-owns-FG-depth", true);
          for (unsigned pair = 0; pair < 6; ++pair) {
            render_tracked_depth = dlssg_frame;
            present_frame(); const auto real = inspect();
            require(active(real) && !real.source.reused_depth && real.source.frame_generation_input,
              "A real Present lost DLSS-G's FG depth or let SR depth compete with it");
            render_tracked_depth = {};
            present_frame(); const auto generated = inspect();
            if (!pair) log_status("DLSSG-evaluation-generated-hold", generated);
            require(active(generated) && generated.source.reused_depth && generated.source.frame_generation_input &&
                generated.source.current.capture == real.source.current.capture && same_scale(generated.scale, real.scale),
              "A generated Present did not hold its real frame's DLSS-G depth");
          }
          render_tracked_depth = dlssg_frame;
          require(reshade_log_contains({"Sunshine Streamline FG depth refused: reason=boundary_without_commands", "fg_scope=live",
              "withdrawn=0", "dlssg_supplied=1"}),
            "A command-less FG tag withdrew the FG source DLSS-G's evaluation supplies");
          require(reshade_log_number_above("Sunshine NGX depth:", "dlssg_depth={copied=", 0),
            "Accepted DLSS-G depth copies were not counted");
          require(reshade_log_number_above("Sunshine NGX depth:", "dlssg_ui={offered=", 0) &&
              reshade_log_number_above("Sunshine NGX depth:", " captured=", 0),
            "DLSS-G's HUD-less and Backbuffer were not captured for UI detection");
          require(reshade_log_number_above("Sunshine UI capture gate:", "hudless_presents={batch=", 0, 12000),
            "DLSS-G's HUD-less image did not reach UI detection with its same-evaluation Backbuffer as an exact pair");
          // The game's own HUD-less and Backbuffer tags copy on its command list
          // (as in a game whose Streamline UI tags already work with FG on): per
          // kind the tag keeps precedence, as for depth. The evaluation offers
          // neither input again (dlssg_ui tag_copied), so no second full-size
          // copy competes for the kind's two snapshots, and UI detection keeps
          // the tags' own same-batch pair.
          ui_tag_commands = true;
          for (const auto settled = GetTickCount64() + 300; GetTickCount64() < settled;) present_frame();
          const auto ngx_lines = [&] { return reshade_log_lines({"Sunshine NGX depth:", "dlssg_ui={offered="}); };
          const auto gate_lines = [&] { return reshade_log_lines({"Sunshine UI capture gate:", "hudless_presents={batch="}); };
          const auto wait_lines = [&](const auto &lines, std::size_t count) {
            for (const auto until = GetTickCount64() + 16000; lines().size() < count && GetTickCount64() < until;) present_frame();
            auto found = lines();
            require(found.size() >= count, "The bounded NGX or UI capture gate line stopped while Streamline tags copied UI");
            return found;
          };
          const auto ngx_start = ngx_lines().size(), gate_start = gate_lines().size();
          // Every frame between two NGX lines written after the switch had the tags' copies.
          const auto ngx = wait_lines(ngx_lines, ngx_start + 2);
          const auto &first_line = ngx[ngx_start], &second_line = ngx[ngx_start + 1];
          const auto ui_field = [](const std::string &line, const char *field) { return number_after(line, field, "dlssg_ui={"); };
          require(ui_field(second_line, "offered=") == ui_field(first_line, "offered=") &&
              ui_field(second_line, " captured=") == ui_field(first_line, " captured=") &&
              ui_field(second_line, "tag_copied:") > ui_field(first_line, "tag_copied:"),
            "DLSS-G's evaluation offered a HUD-less or Backbuffer input the game's own Streamline tags had copied");
          // A gate window wholly after the switch: the tags' exact pairs.
          const auto gate = wait_lines(gate_lines, gate_start + 2);
          require(number_after(gate[gate_start + 1], "hudless_presents={batch=") > 0,
            "The Streamline tags' HUD-less image did not reach UI detection with its same-batch Backbuffer");
          const auto captured_behind_tags = ui_field(second_line, " captured=");
          ui_tag_commands = false;
          // An FG tag with a command list copies the frame first and keeps
          // precedence: the evaluation adds no second copy (tag_copied). The
          // UI tags are command-less again (refused): DLSS-G's UI inputs are
          // captured again at once.
          tag_commands = true;
          settle_route("DLSSG-evaluation-behind-FG-tag-copies", true);
          require(reshade_log_contains({"Sunshine NGX depth:", "dlssg_depth={copied=", "tag_copied:"}),
            "DLSS-G's evaluation copied a frame its Streamline FG tag had already copied");
          require(reshade_log_number_above("Sunshine NGX depth:", " captured=", captured_behind_tags),
            "Refused Streamline UI tags did not hand their kinds back to DLSS-G's evaluation");
          tag_commands = false;
          // Refused again: the scope falls silent and SR depth returns.
          observed_state = false;
          settle_route("DLSSG-refused-again-SR-depth", false);
        },
        [&] {
          require(ngx_release(feature) == ngx_fixture::success && ngx_fixture::evaluated > evaluations_before,
            "DLSS-G fixture release failed or its evaluations never ran");
          // Accepted FG tag copies own the viewport again, as before this case.
          render_tracked_depth = [this] { fallback_frame(true, true); };
          settle_route("DLSSG-case-done-FG-tags-own-again", true);
          evidence << "dlssg-evaluation unobserved=SR observed=FG-owned generated=held ui=exact-batch ui-tags-with-commands=tag-copied\n";
          std::puts("PASS native DLSS-G evaluation: DLSSG.Depth without an observed state is refused and counted with SR depth unchanged; in an observed state it owns the FG scope, generated Presents hold it, the command-less FG tag withdraws nothing, its HUD-less/Backbuffer pair reaches UI detection exactly, and FG depth and HUD-less/Backbuffer tags that copy keep precedence per kind; no FX");
        }});
    }

    // The same refused FG tags with no SR evaluation (DLSS-G without DLSS
    // SR/RR): a silent scope only lets an API source of the viewport
    // through. With none, enabled FG still holds the queue in mono, so the
    // generic selector never takes real or generated FG presentations.
    void fg_silent_scope_without_api_source() {
      run_script({"silent-scope-without-api-source", 15000, {},
        [&] {
          const auto silent_lines = [&] {
            return reshade_log_count({"Sunshine Streamline frame generation depth scope: silent viewport=0"});
          };
          const auto generic_lines = [&] { return reshade_log_count({"using generic depth fallback"}); };
          const auto silent_before = silent_lines(), generic_before = generic_lines();
          const auto held_frame = [&] {
            present_frame(); const auto s = inspect();
            require(s.source.selected, "A silent FG scope with no API source handed the queue to the generic depth selector");
            return s;
          };
          render_tracked_depth = [this] { fallback_frame(false, false); };
          for (const auto until = GetTickCount64() + 7000; silent_lines() == silent_before && GetTickCount64() < until;) held_frame();
          require(silent_lines() > silent_before, "Refused FG tags without SR did not turn the FG scope silent");
          status mono;
          for (const auto start = GetTickCount64(); GetTickCount64() - start < 1500;) mono = held_frame();
          log_status("SLFG-silent-scope-no-API-source", mono);
          require(!mono.source.ready && generic_lines() == generic_before,
            "A silent FG scope with no API source supplied depth or used the generic depth fallback");
          std::puts("PASS native SLFG scope: refused FG tags let the same viewport's SR depth through after a second, never before; an accepted FG copy owns the viewport again and SR stays out; with no SR the silent scope stays mono, never generic; no FX");
        }, {}});
    }

    // Use the same public SDK boundary for linear RR depth with no usable
    // camera. The native copy, provider acquisition and GPU moment readback
    // remain production paths; only the game-authored input is synthetic.
    void fg_linear_rr() {
      using namespace sunshine_streamline;
      using center_t = bool (*)(api::effect_runtime *, provider::center_sample *);
      center_t query_center{};
      const float original_center = center_raw;
      sunshine_streamline::extent original_area;
      run_script({"linear-rr-without-camera", 95000,
        [&] {
          configure_fg(false);
          query_center = reinterpret_cast<center_t>(GetProcAddress(
            GetModuleHandleW(L"SunshineSBSTest.addon64"), "SunshineStreamlineTestCenter"));
          require(query_center, "Linear provider fixture lacks passive moment observation");
          original_area = active_area;
          active_area = {0, 0, selected->width, selected->height};
          render_tracked_depth = [this] {
            using namespace sunshine_streamline;
            auto &token = begin_fg_frame("Linear RR SDK did not supply a frame token");
            // This frame has no camera metadata. Publishing malformed matrices
            // instead would intentionally invalidate the observation revision and
            // test a different contract from linear depth without a camera.
            const auto resource = depth_resource();
            // Clear the preceding hardware-depth tag so it cannot hide tag49.
            const abi_v2::resource_tag tags[]{
              {{nullptr, tag_guid, 1}, nullptr, 0, 1, {}},
              {{nullptr, tag_guid, 1}, &resource, 49, 1, active_area}};
            const auto tagged = sl.frame_tag ? sl.frame_tag(token, fg_viewport, tags, 2, reinterpret_cast<void *>(game_native_command)) :
              sl.global_tag(fg_viewport, tags, 2, reinterpret_cast<void *>(game_native_command));
            require(tagged == 0, "Linear RR tag observation changed the SDK result");
            const base_structure *inputs[]{&fg_viewport.base};
            require(sl.evaluate(1001, token, inputs, 1, reinterpret_cast<void *>(game_native_command)) == 0,
              "Linear RR evaluation changed the SDK result");
          };
        },
        [&] {
          for (const float center_value : {original_center, 0.f, -1.f,
              std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            const auto preceding = inspect().source.current.sequence;
            center_raw = center_value;
            provider::center_sample sampled;
            status s;
            unsigned consecutive{};
            const auto until = GetTickCount64() + 15000;
            do {
              present_frame(); s = inspect();
              const bool complete = query_center(observed.runtime, &sampled) && sampled.metadata.sequence > preceding &&
                sampled.metadata.resource.native == native(*selected) && !sampled.metadata.projection.supplied &&
                sampled.metadata.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance &&
                sampled.moments.supplied && sampled.moments.valid &&
                sampled.moments.encoding == sunshine_scene_depth::depth_encoding::linear_distance;
              const bool selected_linear = s.ready() && s.source.selected && s.source.ready && !s.source.reused_depth &&
                s.source.provider == sunshine_scene_depth::provider_kind::streamline &&
                s.source.current.resource == native(*selected) && s.source.current.capture &&
                s.basis == unsigned(sunshine_game3d::automatic_scale_basis::linear_distance) &&
                s.scale_state == unsigned(sunshine_game3d::automatic_scale_state::active) && std::isfinite(s.scale) && s.scale > 0;
              consecutive = complete && selected_linear ? consecutive + 1 : 0;
            } while (consecutive < 8 && GetTickCount64() < until);
            log_status("SL-RR-linear-no-camera", s);
            require(consecutive >= 8, "Linear RR input failed native capture/acquisition/calibration without camera matrices");
            const auto raw = read(selected->resource.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            require(raw.size() == size_t(selected->width) * selected->height * sizeof(float),
              "Linear provider source readback has the wrong extent");
            std::uint64_t count{};
            double sum{}, squared{};
            for (size_t offset = 0; offset < raw.size(); offset += sizeof(float)) {
              float distance{}; std::memcpy(&distance, raw.data() + offset, sizeof(distance));
              if (!std::isfinite(distance) || distance <= 0.f) continue;
              const double inverse = 1. / distance;
              ++count; sum += inverse; squared += inverse * inverse;
            }
            const auto &moments = sampled.moments;
            require(moments.count == count && std::abs(moments.sum - sum) <= sum * 2e-5 &&
                std::abs(moments.sum_squares - squared) <= squared * 2e-5,
              "Native provider linear moments included invalid distances or decoded device depth");
            require(count && (std::isfinite(center_value) && center_value > 0.f ?
                count == std::uint64_t(selected->width) * selected->height :
                count < std::uint64_t(selected->width) * selected->height),
              "Linear provider invalid-pixel fixture did not exercise rejection");
            evidence << "linear-no-camera center=" << center_value << " count=" << count << " sum=" << moments.sum
              << " sum_squares=" << moments.sum_squares << " capture=" << sampled.id
              << " sequence=" << sampled.metadata.sequence << '\n';
          }
        },
        [&] {
          center_raw = original_center; active_area = original_area;
          render_tracked_depth = [this] { draw_frame(); };
          settle("SL-recovered-after-linear-RR");
          std::puts("PASS native linear RR provider: public tag49 and feature1001, no camera matrix, actual capture/acquire/calibration and reciprocal moments excluding zero/negative/NaN/Inf; no FX");
        }});
    }

    void finish() {
      render_tracked_depth = {};
      reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
      no_effects(); require(evidence.good(), "Could not write native provider evidence");
    }
  };
}

// The API contract matrix's runtime layer (test_game3d_api_contract_runtime.cpp)
// includes the fixture alone.
#ifndef SUNSHINE_NATIVE_PROVIDER_FIXTURE_ONLY
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
    const bool public_ui = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_UI_PUBLIC_TAG_TEST");
    require(!public_ui || sdk, "Public UI regression requires the metadata-only SDK argument");
    native_provider_fixture fixture; fixture.runtime_directory = directory;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), directory, public_ui ? 3 : 2, 0, fs::absolute(argv[3]));
    fixture.load_native(directory);
    if (public_ui) fixture.run_automatic_ui_tags(sdk);
    else { fixture.run_streamline(); if (sdk) fixture.run_fg(sdk); }
    fixture.finish();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what()); return 1;
  }
}
#endif
