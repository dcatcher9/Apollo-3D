// SPDX-License-Identifier: GPL-3.0-only
// Provider metadata + real GPU UAV production -> native capture/calibration.
// The optional FG SDK exposes only ordinary SL metadata calls. It performs no
// capture and injects no depth binding, readiness, sampler result or fence.
#define SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
#include "test_streamline_direct_runtime.cpp"
#include "game3d_controls.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_layer.h"
#include "../../src/game3d_debug_protocol.h"
#include <nlohmann/json.hpp>
#include <functional>
#include <string>
#include <thread>

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

    void run_ui_hook(const std::function<void()> &real_frame,
        sunshine_streamline::abi_v2::frame_token *&token,
        const sunshine_streamline::abi_v2::viewport &viewport,
        sunshine_streamline::abi_v2::set_tag_for_frame frame_tag,
        sunshine_streamline::abi_v2::set_tag global_tag,
        std::uint32_t resource_type,
        const std::function<void(bool)> &configure_fg) {
      using namespace sunshine_streamline;
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
        require(token && (frame_tag ? frame_tag(*token, viewport, &tag, 1, list) : global_tag(viewport, &tag, 1, list)) == 0,
          "Public UI tag changed the synthetic SDK result");
        copy(list, after_upload.p, after_footprint);
      };
      for (unsigned i = 0; i < 8; ++i) { step(); no_effects(); }
      // The tag is a declared source: its first valid selective sample accepts
      // it (A1), from the render after that sample is read. A few sample
      // intervals pass before the dump.
      for (const auto until = GetTickCount64() + 500; GetTickCount64() < until;) { step(); no_effects(); }

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
        step(); no_effects();
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
      // flags describe the layer slot (5), and the accepted tag decides.
      namespace candidate = sunshine_game3d::ui_detection::candidate;
      const auto &tag_detection = replay.at("ui_detection");
      if (tag_detection.at("ran_or_held") == "inactive" ||
          (tag_detection.at("candidates").get<unsigned>() & (candidate::ui_color | candidate::layer)) !=
            (candidate::ui_color | candidate::layer) ||
          !(tag_detection.at("accepted").get<unsigned>() & candidate::ui_color) ||
          (tag_detection.at("flags").get<unsigned>() & ~sunshine_game3d::ui_detection::per_frame_sample) != 5u)
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
      step(); no_effects();
      for (unsigned i = 0; i != retained_count; ++i)
        require(read(retained[i].p, D3D12_RESOURCE_STATE_COMMON) == retained_bytes[i],
          "Host acknowledgement or later game write changed UI snapshot");

      // Draws for a while (a second by default), then dumps; the caller
      // releases the dump.
      const auto dump_after_layer_frames = [&](std::uint64_t draw_ms = 1000) {
        for (const auto until = GetTickCount64() + draw_ms; GetTickCount64() < until;) { step(); no_effects(); }
        box.request = box.state->request_id + 1;
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->request_id), box.request);
        for (const auto until = GetTickCount64() + 10000;
             std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->response_id), 0, 0)) != box.request &&
             GetTickCount64() < until;) { step(); no_effects(); }
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
        step(); no_effects();
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
        step(); no_effects();
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
                (5u | sunshine_game3d::ui_detection::per_frame_scene_hidden)))
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
      // late-layer identity and the premultiplied bound (5).
      const auto &layer_detection = layer_metadata.at("replay").at("ui_detection");
      if (layer_detection.at("ran_or_held") == "inactive" || !(layer_detection.at("candidates").get<unsigned>() & candidate::layer) ||
          !(layer_detection.at("accepted").get<unsigned>() & candidate::layer) ||
          (layer_detection.at("flags").get<unsigned>() & ~sunshine_game3d::ui_detection::per_frame_sample) != 5u ||
          !layer_metadata.at("replay").contains("ui_pin"))
        throw std::runtime_error("The offscreen UI layer did not reach detection accepted with flags 5: " + layer_detection.dump());
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
      step(); no_effects();

      // A layer with straight alpha is still offered but V1-invalid: it
      // neither decides nor blocks.
      straight_layer = true;
      const auto straight_metadata = dump_after_layer_frames();
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
      straight_layer = false;
      step(); no_effects();
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
        for (unsigned i = 0; i != 4; ++i) { step(); no_effects(); }
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
            step(); no_effects();
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
          for (unsigned i = 0; i != 4; ++i) { step(); no_effects(); }
          skip_layer = true;
        };
        const auto timed_step = [&] {
          const auto start = GetTickCount64();
          step(); no_effects();
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
        step(); no_effects();
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
          step(); no_effects();
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
          step(); no_effects();
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
        step(); no_effects();
        live = query();
        require(!live.awaiting && live.offered_id == live.capture_id && live.in_order &&
            live.order == unsigned(order::fence_passed) && live.fence_value > held_fence_value &&
            live.executed_queue != live.presenting_queue,
          "Once their fences reached them, the schedule's second-queue layer copies were not offered in fence order");
        skip_layer = false;
        render_tracked_depth = [&] { real_frame(); draw_layer(); };
        step(); no_effects();
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
            step(); no_effects();
            require(!responded(), "A census dump completed at the Present that armed it");
          };
          const auto await_census = [&](std::uint64_t limit_ms) {
            for (const auto until = GetTickCount64() + limit_ms; !responded() && GetTickCount64() < until;) { step(); no_effects(); }
            require(responded() && box.state->response.result == dump::status::complete &&
                box.state->response.json_bytes <= dump::max_json_bytes && box.state->response.texture_count <= dump::max_textures,
              "Production Dump3D did not complete a census capture");
            return nlohmann::json::parse(std::string(box.state->json, box.state->response.json_bytes));
          };
          const auto release_census = [&] {
            InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
            step(); no_effects();
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
        " exact_consumed_and_optional=1 post_tag_opaque_overwrite=1 host_ack_immutable=1 offscreen_ui_layer=1 live_ui_layer_mask=1 frame_tags=" << bool(frame_tag) << '\n';
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
      struct options { base_structure base; std::uint32_t mode{}, generated_frames{}; };
      using options_t = std::int32_t (*)(const abi_v2::viewport &, const options &);
      options_t options_call{};
      if (failed_off) {
        const auto getter = reinterpret_cast<abi_v2::get_feature_function>(GetProcAddress(sdk, "slGetFeatureFunction"));
        const auto set_result = reinterpret_cast<void (*)(std::int32_t)>(GetProcAddress(sdk, "SunshineFixtureSetOptionsResult"));
        void *function{};
        require(getter && set_result && getter(1000, "slDLSSGSetOptions", function) == 0 && function,
          "Failed-Off regression requires the controllable public SDK options result");
        set_result(1); options_call = reinterpret_cast<options_t>(function);
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
          const options off{{nullptr, {0xfac5f1cb,0x2dfd,0x4f36,{0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}}, 3}, 0, 0};
          require(options_call(viewport, off) == 1, "Repeated failed Off call changed the SDK result");
        }
      };
      // Only public SDK hooks supply UI/depth. The optional failure case calls
      // unsupported FG Off immediately after each tag, matching the live game.
      // No evaluation or private capture export is used.
      const auto warm_until = GetTickCount64() + 750;
      do { step(); no_effects(); } while (GetTickCount64() < warm_until);
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
        for (unsigned i = 0; i != 20; ++i) { step(); no_effects(); }
        box.request = box.state->request_id + 1;
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->request_id), box.request);
        const auto until = GetTickCount64() + 10000;
        while (std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->response_id), 0, 0)) != box.request && GetTickCount64() < until) {
          step(); no_effects();
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
      const auto global_tag_call = reinterpret_cast<abi_v2::set_tag>(GetProcAddress(sdk, "slSetTag"));
      const auto evaluate = reinterpret_cast<abi_v2::evaluate_feature>(GetProcAddress(sdk, "slEvaluateFeature"));
      require(getter && new_token && constants_call && global_tag_call && evaluate, "FG metadata fixture lacks public SL exports");
      std::printf("MEASURE SL SDK capability frame_tags=%u; using %s tags\n", unsigned(tag_call != nullptr), tag_call ? "frame" : "global");
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
        const auto tagged = tag_call ? tag_call(*token, viewport, &tag, 1, reinterpret_cast<void *>(game_native_command)) :
          global_tag_call(viewport, &tag, 1, reinterpret_cast<void *>(game_native_command));
        require(tagged == 0,
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
      // A reset ends temporal history, not the finished copy of an earlier
      // frame: like any gap, the copy is held only within its own source age.
      render_tracked_depth = [&] {
        require(last_token, "Native camera-reset fixture has no existing frame token");
        const auto constants = camera_constants(true);
        require(constants_call(constants, *last_token, viewport) == 0, "Camera reset changed the SDK result");
      };
      held(reset_seed, "SLFG-cache-camera-reset-hold");
      render_tracked_depth = [&] {
        const auto constants = camera_constants(false);
        require(constants_call(constants, *last_token, viewport) == 0, "Camera recovery changed the SDK result");
      };
      held(reset_seed, "SLFG-cache-camera-valid-without-depth");
      require(GetTickCount64() - reset_seed_started < sunshine_scene_depth::maximum_source_age_ms,
        "Native camera-reset fixture expired before its holds were observed");
      render_tracked_depth = {};
      require(fg_frame == reset_seed_frame, "Camera-reset hold case accidentally produced a new real frame");
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
      std::puts("PASS native FG cache: repeated holds cannot renew capture age; camera reset keeps the finished copy within its age; FG off/on cannot revive old depth; fresh copies recover all three intervals; no FX");
      for (const std::uint32_t type : {0u, 8u})
        run_ui_hook(real_frame, last_token, viewport, tag_call, global_tag_call, type, configure);

      // The Witcher 3 pattern with FG off: each frame tags depth valid until
      // present, evaluates, then clears the tag before present, while another
      // viewport sends reset constants every frame. Neither the clear nor the
      // foreign reset is lost evidence, so every presentation has fresh depth.
      configure(false);
      const abi_v2::viewport foreign_viewport{{nullptr, viewport_guid, 1}, 7};
      render_tracked_depth = [&] {
        draw(*decoy); write_depth_pixels();
        abi_v2::frame_token *token{}; const auto frame = ++fg_frame;
        require(new_token(token, &frame) == 0 && token, "Until-present SDK did not supply a frame token");
        last_token = token;
        require(constants_call(camera_constants(true), *token, foreign_viewport) == 0 &&
            constants_call(camera_constants(false), *token, viewport) == 0,
          "Until-present constants observation changed the SDK result");
        abi_v2::resource resource{}; resource.base = {nullptr, resource_guid, 1};
        resource.type = 8; resource.native = selected->resource.p; resource.state = unsigned(selected->state);
        resource.width = selected->width; resource.height = selected->height;
        resource.native_format = DXGI_FORMAT_R32_FLOAT; resource.mip_levels = resource.array_layers = 1;
        const abi_v2::resource_tag tag{{nullptr, tag_guid, 1}, &resource, 0, 1, active_area};
        const auto command = reinterpret_cast<void *>(game_native_command);
        require(global_tag_call(viewport, &tag, 1, command) == 0, "Until-present tag observation changed the SDK result");
        const base_structure *inputs[]{&viewport.base};
        require(evaluate(0, *token, inputs, 1, command) == 0, "Until-present evaluation changed the SDK result");
        const abi_v2::resource_tag cleared{{nullptr, tag_guid, 1}, nullptr, 0, 1, {}};
        require(global_tag_call(viewport, &cleared, 1, command) == 0, "Until-present clear changed the SDK result");
      };
      settle("SL-until-present-clear-foreign-reset");
      std::puts("PASS native SL provider: depth cleared after evaluation and another viewport's per-frame camera reset keep fresh depth every presentation; no FX");

      // Use the same public SDK boundary for linear RR depth with no usable
      // camera. The native copy, provider acquisition and GPU moment readback
      // remain production paths; only the game-authored input is synthetic.
      configure(false);
      using center_t = bool (*)(api::effect_runtime *, provider::center_sample *);
      const auto query_center = reinterpret_cast<center_t>(GetProcAddress(
        GetModuleHandleW(L"SunshineSBSTest.addon64"), "SunshineStreamlineTestCenter"));
      require(query_center, "Linear provider fixture lacks passive moment observation");
      const float original_center = center_raw;
      const auto original_area = active_area;
      active_area = {0, 0, selected->width, selected->height};
      render_tracked_depth = [&] {
        draw(*decoy); write_depth_pixels();
        abi_v2::frame_token *token{}; const auto frame = ++fg_frame;
        require(new_token(token, &frame) == 0 && token, "Linear RR SDK did not supply a frame token");
        // This frame has no camera metadata. Publishing malformed matrices
        // instead would intentionally invalidate the observation revision and
        // test a different contract from linear depth without a camera.
        abi_v2::resource resource{}; resource.base = {nullptr, resource_guid, 1};
        resource.type = 8; resource.native = selected->resource.p; resource.state = unsigned(selected->state);
        resource.width = selected->width; resource.height = selected->height;
        resource.native_format = DXGI_FORMAT_R32_FLOAT; resource.mip_levels = resource.array_layers = 1;
        // Clear the preceding hardware-depth tag so it cannot hide tag49.
        const abi_v2::resource_tag tags[]{
          {{nullptr, tag_guid, 1}, nullptr, 0, 1, {}},
          {{nullptr, tag_guid, 1}, &resource, 49, 1, active_area}};
        const auto tagged = tag_call ? tag_call(*token, viewport, tags, 2, reinterpret_cast<void *>(game_native_command)) :
          global_tag_call(viewport, tags, 2, reinterpret_cast<void *>(game_native_command));
        require(tagged == 0, "Linear RR tag observation changed the SDK result");
        const base_structure *inputs[]{&viewport.base};
        require(evaluate(1001, *token, inputs, 1, reinterpret_cast<void *>(game_native_command)) == 0,
          "Linear RR evaluation changed the SDK result");
      };
      for (const float center_value : {original_center, 0.f, -1.f,
          std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        const auto preceding = inspect().source.current.sequence;
        center_raw = center_value;
        provider::center_sample sampled;
        status s;
        unsigned consecutive{};
        const auto until = GetTickCount64() + 15000;
        do {
          step(); no_effects(); s = inspect();
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
      center_raw = original_center; active_area = original_area;
      render_tracked_depth = [&] { draw_frame(); };
      settle("SL-recovered-after-linear-RR");
      std::puts("PASS native linear RR provider: public tag49 and feature1001, no camera matrix, actual capture/acquire/calibration and reciprocal moments excluding zero/negative/NaN/Inf; no FX");
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
