// SPDX-License-Identifier: GPL-3.0-only
// Provider metadata + real GPU UAV production -> native capture/calibration.
// The optional FG SDK exposes only ordinary SL metadata calls. It performs no
// capture and injects no depth binding, readiness, sampler result or fence.
#define SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
#include "test_streamline_direct_runtime.cpp"
#include "game3d_controls.h"
#include "game3d_ui_detection_contract.h"
#include "../../src/game3d_debug_protocol.h"
#include <nlohmann/json.hpp>

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

    void run_ui_hook(const std::function<void()> &real_frame,
        sunshine_streamline::abi_v2::frame_token *&token,
        const sunshine_streamline::abi_v2::viewport &viewport,
        sunshine_streamline::abi_v2::set_tag_for_frame frame_tag,
        sunshine_streamline::abi_v2::set_tag global_tag,
        std::uint32_t resource_type) {
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
      // A tagged UIColorAndAlpha fills the UI color slot without layer flags.
      const auto &tag_detection = replay.at("ui_detection");
      if (tag_detection.at("ran_or_held") == "inactive" || !(tag_detection.at("candidates").get<unsigned>() & 2u) ||
          tag_detection.at("flags") != 0u)
        throw std::runtime_error("A tagged UIColorAndAlpha did not reach detection without flags: " + tag_detection.dump());
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
      // The census copies the layer before its next clear: the previous frame's UI.
      char layer_source[24];
      std::snprintf(layer_source, sizeof(layer_source), "0x%llx", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(layer.p)));
      unsigned layer_artifact{};
      for (const auto &entry : metadata.at("ui_layer_census").at("candidates"))
        if (entry.at("source") == layer_source && entry.at("captured") == true && entry.at("dxgi_format") == 28 &&
            entry.at("width") == width && entry.at("height") == height && entry.at("clears_while_armed") >= 1)
          layer_artifact = entry.at("artifact_id");
      if (!layer_artifact) throw std::runtime_error("Dump census missed the offscreen UI layer: " + metadata.at("ui_layer_census").dump());
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

      // Draws for a second, then dumps; the caller releases the dump.
      const auto dump_after_layer_frames = [&] {
        for (const auto until = GetTickCount64() + 1000; GetTickCount64() < until;) { step(); no_effects(); }
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
      // D3D12 reads back all seven decision texels: the tagged UI color's
      // nearly opaque pixels (texel 4) and, in this first session without
      // remembered trust, the first-run shadow's hidden-scene evidence of the
      // presented and HUD-less images (texels 5 and 6), which changes nothing.
      {
        unsigned opaque{};
        for (size_t offset = 3; offset < expected.size(); offset += 4) opaque += expected[offset] == 255;
        const auto tagged_metadata = dump_after_layer_frames();
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
        step(); no_effects();
        const auto &automatic = tagged_metadata.at("replay").at("source_alpha_auto");
        const auto &sampled = automatic.at("sampled_evidence");
        const auto &scene = sampled.at("scene");
        if (!(sampled.at("candidates").get<unsigned>() & 2u) || sampled.at("alpha_opaque").size() != 2 ||
            sampled.at("alpha_opaque")[1] != opaque || sampled.at("alpha_opaque")[0] != 0u || !opaque ||
            automatic.at("scene_shadow") != true || automatic.at("scene_hold") != 0u || scene.at("ran") != true ||
            !scene.contains("verdict") || !scene.contains("n") || !scene.contains("d") || !sampled.at("hudless_scene").contains("valid") ||
            automatic.at("sampled_source") == 8u || automatic.at("sampled_source") == 9u)
          throw std::runtime_error("D3D12 lost a decision texel or the first-run shadow changed a decision: " + automatic.dump());
        evidence << "d3d12-decision-texels opaque_ui_color=" << opaque << " shadow_scene_n=" << scene.at("n") << " shadow_scene_d=" <<
          scene.at("d") << " verdict=" << scene.at("verdict").get<std::string>() << '\n';
      }
      // Once tag 23 stops, the offscreen layer is the UI color candidate: live
      // tracking copies it before each clear and Auto admits it premultiplied,
      // tint included.
      render_tracked_depth = [&] { real_frame(); draw_layer(); };
      // The layer route on D3D12 (docs/reshade-sbs.md, hidden-scene evidence):
      // before any selective frame could earn it trust, the layer is opaque
      // everywhere over a flat presented frame that shows none of the depth's
      // edges, as a splash. The CPU holds the hidden verdict and the frame is
      // full-frame UI (8).
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
        // A layer that earned trust in an earlier pass of this hook decides by
        // itself (source 2), and no route is held.
        const bool trusted_layer = hidden_sample.at("trusted_alpha").get<unsigned>() & 2u;
        const auto flags = hidden_metadata.at("replay").at("ui_detection").at("flags").get<unsigned>();
        if (!full_frame || hidden_sample.at("alpha_opaque")[1] != size_t(width) * height || (trusted_layer ?
              automatic.at("sampled_source") != 2u || automatic.at("scene_hold") != 0u :
              automatic.at("sampled_source") != 8u || automatic.at("scene_hold") != 1u || scene.at("valid") != true ||
              scene.at("verdict") != "hidden" ||
              (flags & (sunshine_game3d::ui_detection::stored_mask | sunshine_game3d::ui_detection::per_frame_scene_hold)) !=
                (5u | sunshine_game3d::ui_detection::per_frame_scene_hold)))
          throw std::runtime_error("D3D12 did not take the layer route over a hidden scene: " + automatic.dump() +
            " full_frame=" + std::to_string(full_frame));
        evidence << "d3d12-layer-route trusted_layer=" << trusted_layer << " sampled_source=" << automatic.at("sampled_source") <<
          " scene_hold=" << automatic.at("scene_hold") << " scene_n=" << scene.at("n") << " scene_d=" << scene.at("d") << '\n';
      }
      const auto layer_metadata = dump_after_layer_frames();
      bool layer_offered = false, tag23_offered = false;
      for (const auto &candidate : layer_metadata.at("ui_source").value("candidates", nlohmann::json::array())) {
        layer_offered |= candidate.value("source", std::string{}) == "ui_layer" && candidate.at("available_for_detection") == true &&
          candidate.at("format") == DXGI_FORMAT_R8G8B8A8_UNORM;
        tag23_offered |= candidate.value("source", std::string{}) == "sl_ui_color_alpha";
      }
      if (!layer_offered || tag23_offered)
        throw std::runtime_error("Auto did not offer the live offscreen UI layer once tag 23 stopped: " + layer_metadata.at("ui_source").dump());
      // An 8-bit layer forwards its stored flags: the late-layer identity and
      // the premultiplied check (5).
      const auto &layer_detection = layer_metadata.at("replay").at("ui_detection");
      if (layer_detection.at("ran_or_held") == "inactive" || !(layer_detection.at("candidates").get<unsigned>() & 2u) ||
          layer_detection.at("flags") != 5u || !layer_metadata.at("replay").contains("ui_pin"))
        throw std::runtime_error("The offscreen UI layer did not reach detection with flags 5: " + layer_detection.dump());
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

      // A layer with straight alpha is still offered but the GPU rejects it.
      straight_layer = true;
      const auto straight_metadata = dump_after_layer_frames();
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&box.state->released_id), box.request);
      straight_layer = false;
      step(); no_effects();
      const auto &sampled = straight_metadata.at("replay").at("source_alpha_auto").at("sampled_evidence");
      if (!(sampled.at("candidates").get<unsigned>() & 2u) || !sampled.at("alpha_invalid")[1].get<unsigned>())
        throw std::runtime_error("The GPU admitted an offscreen layer with straight alpha: " + sampled.dump());
      evidence << "public-ui-hook descriptor_type=" << resource_type << " source_format=90 captured_format=87 tag=23 covered=" << covered <<
        " exact_consumed_and_optional=1 post_tag_opaque_overwrite=1 host_ack_immutable=1 offscreen_ui_layer=1 live_ui_layer_mask=1 frame_tags=" << bool(frame_tag) << '\n';
      std::puts("PASS actual public SL tag23 hook: typeless90 to typed87, automatic GPU mask equals pre-overwrite alpha and optional dump exact before opaque overwrite; host lease immutable");
      std::puts("PASS live offscreen UI layer: without tag 23, the layer copied before its clear is the automatic UI mask exactly as its raw alpha, tint within twice its alpha included, the UI on the plane and the scene rows beside it off it; straight alpha is rejected; opaque everywhere over a flat presented frame it is full-frame UI by the layer route (8)");
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
        run_ui_hook(real_frame, last_token, viewport, tag_call, global_tag_call, type);

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
