// SPDX-License-Identifier: GPL-3.0-only
// Real DSV drawing through the official runtime. DEPTH is never injected.
#define SUNSHINE_DEPTH_SELECTION_RUNTIME
#include "test_depth3d_runtime_d3d12.cpp"
#include "../../src/reshade_bridge_protocol.h"
#include <d3dcompiler.h>
#include <memory>

namespace {
  struct selection_fixture_t : fixture_t {
    struct target_t {
      com_ptr<ID3D12Resource> texture;
      com_ptr<ID3D12DescriptorHeap> heap;
      unsigned width = 0, height = 0, draws = 1, pattern = 0;
      DXGI_FORMAT format = DXGI_FORMAT_D32_FLOAT;
      bool clear_after = false;
      float clear_depth = 0.f;
      bool segmented_viewport = false;
      std::function<void()> before_final_clear;
    };
    std::unique_ptr<target_t> scene, decoy, shadow;
    com_ptr<ID3D12RootSignature> root;
    com_ptr<ID3D12PipelineState> depth32_pipeline, depth32s8_pipeline;

    void check_unified_addon() {
      const char *manual_test = std::getenv("SUNSHINE_DEPTH_MANUAL_RECOVERY_TEST");
      bool actions_test = manual_test && std::strcmp(manual_test, "1") == 0;
#ifdef SUNSHINE_RAW_SCENE_RUNTIME
      actions_test = actions_test || sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST");
#endif
      const HMODULE module = GetModuleHandleW(actions_test ? L"SunshineSBSTest.addon64" : L"SunshineSBS.addon64");
      require(module != nullptr, "Unified SunshineSBS addon was not loaded by the official runtime");
      const auto name = reinterpret_cast<const char *const *>(GetProcAddress(module, "NAME"));
      require(name && *name && std::strcmp(*name, actions_test ? "Sunshine SBS TEST ONLY" : "Sunshine 3D") == 0,
        actions_test ? "Explicit action fixture requires the test-only SunshineSBSTest DLL" : "Selection fixture requires the production unified Sunshine 3D addon");
      require(!actions_test || GetModuleHandleW(L"SunshineSBS.addon64") == nullptr,
        "Test-only action fixture also loaded a production addon");
      require(GetModuleHandleW(L"SunshineDepth.addon64") == nullptr && GetModuleHandleW(L"SunshineDepthProbe.addon64") == nullptr,
              "A legacy depth addon or temporary probe is loaded alongside the unified addon");
      char disabled[256] {};
      size_t length = sizeof(disabled);
      require(reshade::get_config_value(nullptr, "ADDON", "DisabledAddons", disabled, &length) &&
              std::strcmp(disabled, "Generic Depth") == 0,
              "The actual runtime did not disable built-in Generic Depth");
      const auto mapping_name = std::wstring(reshade_bridge::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      HANDLE mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, mapping_name.c_str());
      require(mapping != nullptr, "Unified addon did not initialize the production SBS exporter mapping");
      const auto *state = static_cast<const reshade_bridge::shared_state_t *>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, sizeof(reshade_bridge::shared_state_t)));
      const bool valid = state && state->shared_bytes == sizeof(*state) && state->metadata.signature == reshade_bridge::magic &&
                         state->metadata.protocol_version == reshade_bridge::version && state->metadata.producer_pid == GetCurrentProcessId();
      if (state) UnmapViewOfFile(state);
      CloseHandle(mapping);
      require(valid, "Unified addon producer mapping has an invalid identity or protocol layout");
      // This hidden-window fixture validates registration and depth data. The
      // controlled-foreground Present fixture validates shared frame delivery.
      std::puts(actions_test ?
        "PASS explicitly selected Sunshine SBS TEST ONLY addon initializes selector/exporter for action validation; no production, legacy or probe addon loaded" :
        "PASS one production Sunshine 3D addon initializes both selector and exporter; Generic Depth disabled and no legacy/probe addon loaded");
    }

    void create_pipeline() {
      const char *shader = R"(
cbuffer Draw : register(b0) { float target_width; float target_height; uint pattern; };
float4 vs(uint id : SV_VertexID) : SV_Position {
  float2 uv = float2((id << 1) & 2, id & 2);
  return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float ps(float4 position : SV_Position) : SV_Depth {
  float2 uv = position.xy / float2(target_width, target_height);
  if (pattern == 0) return 0;
  // Content-admission regression: written interior depth, but no scene range.
  if (pattern == 19) return 0.5;
  // Foreground-only and outdoor scene buffers are both valid depth images.
  // Their allocations, workloads and depth convention deliberately match.
  if (pattern == 4 && (uv.x < 0.35 || uv.x > 0.65 || uv.y < 0.1 || uv.y > 0.9)) return 0;
  if (pattern == 5 && uv.y < 0.35) return 0;
  if (pattern == 4 || pattern == 5) return 0.01 + 0.02 * uv.x + 0.015 * uv.y;
  if (pattern == 1) return 0.01 + 0.02 * uv.x + 0.015 * uv.y;
  if (pattern == 3) return 0.04 + 0.008 * uv.x + 0.004 * uv.y;
  // Same camera convention and coverage, but more populated distance ranges.
  if (pattern == 6) return exp2(-10.0 + 8.0 * (0.7 * uv.x + 0.3 * uv.y));
  if (pattern == 7) return 1.0 - exp2(-10.0 + 8.0 * (0.7 * uv.x + 0.3 * uv.y));
  if (pattern == 8) return 1.0 - (0.04 + 0.008 * uv.x + 0.004 * uv.y);
  if (pattern == 9) return 0.01 + 0.02 * (1.0 - uv.x) + 0.015 * uv.y;
  // Rotation fixture only: independent third basis and small exact center steps.
  if (pattern == 18 || (pattern >= 100 && pattern <= 228)) {
    uint2 cell = min(uint2(31, 17), uint2(uv * float2(32, 18)));
    bool center = cell.x >= 14 && cell.x < 18 && cell.y >= 7 && cell.y < 11;
    float t = pattern == 18 ? 0.0625 : 0.125 + float(pattern - 100) / 1024.0;
    return center ? t : float(cell.x / 8 + 1) / 16.0;
  }
  // Adaptive-runtime fixture: coherent large depth bands with an independently fixed
  // central 4x4 sample patch. Only that patch changes between the two scenes.
  if (pattern >= 14 && pattern <= 17) {
    uint2 cell = min(uint2(31, 17), uint2(uv * float2(32, 18)));
    bool center = cell.x >= 14 && cell.x < 18 && cell.y >= 7 && cell.y < 11;
    float t = center ? ((pattern & 1) ? 0.25 : 0.125) : (float(cell.x / 8 + 1) / 16.0);
    return pattern >= 16 ? 1.0 - t : t;
  }
  // Raw-reference integration only: exact binary-representable four-band
  // distributions, permuted spatially without changing their histogram.
  if (pattern >= 10 && pattern <= 13) {
    uint band = min(3u, uint(uv.x * 4.0));
    if (pattern == 11 || pattern == 13) band = (band + 1u) % 4u;
    float t = band == 0 ? 0.125 : band == 1 ? 0.25 : band == 2 ? 0.5 : 0.75;
    return pattern >= 12 ? 1.0 - t : t;
  }
  return 0.1 + 0.5 * uv.x + 0.2 * uv.y;
})";
      com_ptr<ID3DBlob> vs, ps, errors, serialized;
      checked(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "vs", "vs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, vs.put(), errors.put()), "Compile tracked-depth vertex shader");
      checked(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "ps", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, ps.put(), errors.put()), "Compile tracked-depth pixel shader");
      D3D12_ROOT_PARAMETER constants {};
      constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      constants.Constants.Num32BitValues = 3;
      constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC signature {};
      signature.NumParameters = 1;
      signature.pParameters = &constants;
      checked(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, serialized.put(), errors.put()), "Serialize tracked-depth root signature");
      checked(game->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(root.put())), "Create tracked-depth root signature");
      D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
      pipeline.pRootSignature = root.p;
      pipeline.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
      pipeline.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
      pipeline.SampleMask = UINT_MAX;
      pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
      pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
      pipeline.RasterizerState.DepthClipEnable = TRUE;
      pipeline.DepthStencilState.DepthEnable = TRUE;
      pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
      pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
      pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      pipeline.SampleDesc.Count = 1;
      pipeline.DSVFormat = DXGI_FORMAT_D32_FLOAT;
      checked(game->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(depth32_pipeline.put())), "Create tracked D32 pipeline");
      pipeline.DSVFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
      checked(game->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(depth32s8_pipeline.put())), "Create tracked D32S8 pipeline");
    }

    std::unique_ptr<target_t> target(unsigned w, unsigned h, unsigned draws, unsigned pattern, bool stencil = false, bool clear_after = false) {
      auto result = std::make_unique<target_t>();
      result->width = w; result->height = h; result->draws = draws; result->pattern = pattern;
      result->format = stencil ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : DXGI_FORMAT_D32_FLOAT;
      result->clear_after = clear_after;
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = w; desc.Height = h;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = stencil ? DXGI_FORMAT_R32G8X24_TYPELESS : DXGI_FORMAT_R32_TYPELESS;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
      D3D12_CLEAR_VALUE clear {};
      clear.Format = result->format;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(result->texture.put())), "Create tracked game depth target");
      D3D12_DESCRIPTOR_HEAP_DESC descriptors {};
      descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
      descriptors.NumDescriptors = 1;
      checked(game->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(result->heap.put())), "Create game depth view heap");
      D3D12_DEPTH_STENCIL_VIEW_DESC view {};
      view.Format = result->format;
      view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
      game->CreateDepthStencilView(result->texture.p, &view, result->heap->GetCPUDescriptorHandleForHeapStart());
      return result;
    }

    void draw(target_t &target, bool unbind = true) {
      const auto dsv = target.heap->GetCPUDescriptorHandleForHeapStart();
      commands->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
      commands->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, target.clear_depth, 0, 0, nullptr);
      commands->SetGraphicsRootSignature(root.p);
      commands->SetPipelineState(target.format == DXGI_FORMAT_D32_FLOAT ? depth32_pipeline.p : depth32s8_pipeline.p);
      commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      const auto draw_region = [&](unsigned w, unsigned h, unsigned pattern, unsigned draws) {
        const D3D12_VIEWPORT viewport {0, 0, float(w), float(h), 0, 1};
        const D3D12_RECT scissor {0, 0, LONG(w), LONG(h)};
        commands->RSSetViewports(1, &viewport);
        commands->RSSetScissorRects(1, &scissor);
        struct { float w, h; unsigned pattern; } constants {float(w), float(h), pattern};
        commands->SetGraphicsRoot32BitConstants(0, 3, &constants, 0);
        for (unsigned i = 0; i < draws; ++i) commands->DrawInstanced(3, 1, 0, 0);
      };
      if (target.segmented_viewport) {
        // Same allocation/layout every frame, but capture tracking observes the
        // later small scene pass. Starting a probe must not change its lifetime
        // identity and cause perpetual challenger cancellation.
        draw_region(target.width, target.height, 0, 1);
        commands->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.f, 0, 0, nullptr);
        draw_region(target.width / 2, target.height / 2, target.pattern, 4);
      } else draw_region(target.width, target.height, target.pattern, target.draws);
      if (target.before_final_clear) target.before_final_clear();
      if (target.clear_after) commands->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, target.clear_depth, 0, 0, nullptr);
      if (unbind) commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
    }

    api::resource_view selected_binding() {
      api::resource_view selected {};
      if (!observed.runtime) return selected;
      observed.runtime->enumerate_texture_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char name[256] {};
        runtime->get_texture_variable_name(variable, name);
        if (!named(name, "DepthBufferTex")) return;
        runtime->get_texture_binding(variable, &selected, nullptr);
      });
      return selected;
    }

    D3D12_RESOURCE_DESC selected_description() {
      D3D12_RESOURCE_DESC desc {};
      const auto view = selected_binding();
      if (view.handle) {
        const auto resource = observed.runtime->get_device()->get_resource_from_view(view);
        if (resource.handle) desc = reinterpret_cast<ID3D12Resource *>(resource.handle)->GetDesc();
      }
      return desc;
    }

    bool varied_linear_depth(unsigned pattern = 1, float viewport_fraction = 1.f) {
      com_ptr<ID3D12Resource> linear;
      if (!observed.runtime) return false;
      observed.runtime->enumerate_texture_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char name[256] {};
        runtime->get_texture_variable_name(variable, name);
        if (!named(name, "texzBufferN_P")) return;
        api::resource_view view {};
        runtime->get_texture_binding(variable, &view, nullptr);
        if (!view.handle) return;
        auto resource = runtime->get_device()->get_resource_from_view(view);
        linear.p = reinterpret_cast<ID3D12Resource *>(resource.handle);
        if (linear.p) linear->AddRef();
      });
      if (!linear.p) return false;
      const auto desc = linear->GetDesc();
      require(desc.Format == DXGI_FORMAT_R16G16_FLOAT, "Scene-depth oracle expects RG16F linear depth");
      const auto pixels = read(linear.p);
      require(pixels.size() == size_t(desc.Width) * desc.Height * 4, "Unexpected linear-depth readback size");
      float low = 1, high = 0;
      for (unsigned y = 1; y <= 3; ++y) for (unsigned x = 1; x <= 4; ++x) {
        const auto tx = unsigned(double(desc.Width) * viewport_fraction * x / 5);
        const auto ty = unsigned(desc.Height * viewport_fraction * y / 4);
        const size_t offset = (size_t(ty) * size_t(desc.Width) + tx) * 4 + 2;
        std::uint16_t value = 0;
        std::memcpy(&value, pixels.data() + offset, 2);
        const float depth = half_float(value);
        if (!std::isfinite(depth) || depth <= 0 || depth > 1) return false;
        // Independent projection oracle for the known reversed-Z DSV gradient.
        // DMA=40 gives near/far=.125/40; tolerate point-sampling and FP16 rounding.
        const float u = (tx + .5f) / (float(desc.Width) * viewport_fraction), v = (ty + .5f) / (desc.Height * viewport_fraction);
        float raw = (pattern == 3 || pattern == 8) ? .04f + .008f * u + .004f * v : .01f + .02f * u + .015f * v;
        if (pattern == 6 || pattern == 7) raw = float(std::exp2(-10.0 + 8.0 * (.7 * u + .3 * v)));
        if (pattern == 9) raw = .01f + .02f * (1 - u) + .015f * v;
        // Patterns 7/8 use normal-Z with Depth_Map=0; the same projection of
        // their complementary raw depth supplies an independent pixel oracle.
        if ((pattern == 4 && (u < .35f || u > .65f || v < .1f || v > .9f)) || (pattern == 5 && v < .35f)) raw = 0;
        const float expected = 1.f / (1.f + 319.f * raw);
        if (std::abs(depth - expected) > .002f) return false;
        low = std::min(low, depth); high = std::max(high, depth);
      }
      return high - low > .001f;
    }

    void wait_for_scene(const char *label, unsigned expected_width, unsigned expected_height, unsigned pattern = 1, float viewport_fraction = 1.f) {
      const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(14);
      unsigned consecutive = 0, frames = 0;
      while (std::chrono::steady_clock::now() < until) {
        step();
        if (++frames % 8) continue;
        const auto selected = selected_description();
        if (selected.Width == expected_width && selected.Height == expected_height && varied_linear_depth(pattern, viewport_fraction)) {
          if (++consecutive == 3) {
            std::printf("PASS %s: actual DEPTH=%ux%u with spatially varying shader-visible scene depth\n", label, expected_width, expected_height);
            return;
          }
        } else consecutive = 0;
      }
      const auto selected = selected_description();
      std::fprintf(stderr, "Selection timeout: %s expected=%ux%u actual=%llux%u\n", label, expected_width, expected_height, static_cast<unsigned long long>(selected.Width), selected.Height);
      throw std::runtime_error("Automatic scene-depth selection did not converge");
    }

    void check_multiple_viewports() {
      decoy->draws = 200;
      decoy->pattern = 0;
      scene = target(width, height, 4, 1, false, true);
      scene->segmented_viewport = true;
      wait_for_scene("multi-viewport capture keeps its identity and actual copy rectangle", width, height, 1, .5f);
    }

    void check_scene_completeness() {
      scene.reset();
      shadow.reset();
      decoy->draws = 1;
      decoy->pattern = 4;
      wait_for_scene("foreground-only source is valid and initially available", width, height, 4);
      const auto qualify_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < qualify_until) step();
      scene = target(width, height, 1, 1);
      wait_for_scene("full scene replaces useful foreground-only source at identical resolution", width, height);
      scene->pattern = 5;
      wait_for_scene("scene including clear sky remains usable", width, height, 5);
      const auto stable_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < stable_until) step();
      require(varied_linear_depth(5), "Clear sky made the selector return to foreground-only depth");
      scene = target(width / 2, height / 2, 1, 1);
      wait_for_scene("lower-resolution full scene outranks native foreground-only depth", scene->width, scene->height);
      scene->pattern = 5;
      wait_for_scene("lower-resolution scene with sky retains background coverage", scene->width, scene->height, 5);
      const auto scaled_sky_until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (std::chrono::steady_clock::now() < scaled_sky_until) {
        step();
        const auto selected = selected_description();
        require(selected.Width == scene->width && selected.Height == scene->height, "Sky made native foreground depth replace the fuller scaled scene");
      }
      require(varied_linear_depth(5), "Scaled sky scene lost its shader-visible background depth");
    }

    void check_histogram_breadth() {
      scene.reset();
      shadow.reset();
      decoy->draws = 1;
      decoy->pattern = 3;
      set_int("Depth_Map", 1);
      wait_for_scene("narrow-range depth remains valid on its own", width, height, 3);
      const auto qualify_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < qualify_until) step();
      scene = target(width, height, 1, 6);
      wait_for_scene("broader histogram replaces equally complete same-size narrow depth", width, height, 6);
      const auto stable_until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (std::chrono::steady_clock::now() < stable_until) step();
      require(varied_linear_depth(6), "Histogram promotion oscillated between equally sized candidates");
      scene = target(width / 2, height / 2, 1, 6);
      // Histogram breadth breaks ties at equal active-resolution coverage. A
      // similarly complete smaller source must not displace useful native depth.
      wait_for_scene("native narrow depth outranks equally complete scaled broad depth", width, height, 3);
      const auto scaled_until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (std::chrono::steady_clock::now() < scaled_until) {
        step();
        require(selected_description().Width == width, "Broader lower-resolution depth displaced useful native depth");
      }
      require(varied_linear_depth(3), "Native winner lost its expected narrow-range depth contents");
      scene.reset();
      wait_for_scene("native narrow depth remains selected when the smaller candidate disappears", width, height, 3);
      decoy->pattern = 8;
      set_int("Depth_Map", 0);
      wait_for_scene("normal-Z narrow depth remains valid", width, height, 8);
      const auto normal_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < normal_until) step();
      scene = target(width, height, 1, 7);
      wait_for_scene("normal-Z histogram also selects broader same-size depth", width, height, 7);
      std::puts("PASS real histogram comparison: same-size promotion, native-resolution preference, stable retention, narrow-range validity and both depth conventions");
    }

    void check_reload_interleaved() {
      // A depth-active auxiliary present advances the tracker even though the
      // native scene target only renders every other present. Exercise the
      // production probe lifecycle, not just selection_policy in isolation.
      unsigned present = 0;
      render_tracked_depth = [&] {
        draw(*scene);
        if (++present % 2 == 0 && decoy) draw(*decoy);
      };
      shadow.reset();
      for (unsigned reload = 0; reload < 2; ++reload) {
        decoy.reset();
        wait_for_scene("reload falls back to live lower-resolution depth", scene->width, scene->height);
        decoy = target(width, height, 1, 9, reload != 0, reload != 0);
        wait_for_scene("recreated native depth survives interleaved presents and promotes", width, height, 9);
        const auto stable_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < stable_until) {
          step();
          const auto selected = selected_description();
          require(selected.Width == width && selected.Height == height,
                  "Interleaved activity evicted a confirmed native source");
        }
        std::printf("PASS native depth recovery after reload %u, with auxiliary depth-active presents\n", reload + 1);
      }
      render_tracked_depth = {};
    }

    void check_manual_handover() {
      const char *enabled = std::getenv("SUNSHINE_DEPTH_MANUAL_RECOVERY_TEST");
      require(enabled && std::strcmp(enabled, "1") == 0,
              "Manual handover fixture requires explicit SUNSHINE_DEPTH_MANUAL_RECOVERY_TEST=1");
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
      using state_t = BOOL (*)(api::effect_runtime *, std::uint64_t *, BOOL *);
      const auto select_manual = reinterpret_cast<select_t>(GetProcAddress(module, "SunshineDepthTestSelectManual"));
      const auto manual_state = reinterpret_cast<state_t>(GetProcAddress(module, "SunshineDepthTestManualState"));
      require(module && select_manual && manual_state, "Manual handover requires the test add-on's UI-selection adapter");
      const auto ready = [&] {
        bool value = false;
        observed.runtime->get_uniform_value_bool(uniform("Sunshine_DepthReady"), &value, 1);
        return value;
      };
      const auto require_mode = [&](std::uint64_t expected) {
        std::uint64_t selected = 0;
        BOOL recovering = TRUE;
        require(manual_state(observed.runtime, &selected, &recovering) && selected == expected && !recovering,
                "Manual handover retained an override or recovery watch");
      };
      const auto pump = [&](unsigned milliseconds, const std::function<void()> &check) {
        const auto until = GetTickCount64() + milliseconds;
        do { step(); check(); } while (GetTickCount64() < until);
      };

      shadow.reset();
      scene = target(width, height, 1, 9, true, true);
      // Comparable real scene contents: the lower-resolution source wins cold
      // workload fallback, but cannot displace confirmed native depth on workload.
      decoy = target(width / 2, height / 2, 8, 1);
      wait_for_scene("manual handover starts with confirmed native depth", width, height, 9);
      const auto native_handle = reinterpret_cast<std::uint64_t>(scene->texture.p);
      require(select_manual(observed.runtime, native_handle), "Could not pin the live native scene depth");
      pump(3200, [&] { require_mode(native_handle); });
      require(ready() && varied_linear_depth(9), "Pinned native depth did not become ready before handover");
      const auto view = selected_binding();
      const auto backup = observed.runtime->get_device()->get_resource_from_view(view);
      const auto native_description = selected_description();
      require(view.handle && backup.handle, "Pinned native depth has no actual shader binding");

      // Zero invokes the UI's explicit automatic-selection action. A frozen old
      // selector can expose this action through an adapter-only control build;
      // no new export or synthetic depth injection is needed to reach this check.
      require(select_manual(observed.runtime, 0), "Could not invoke the automatic-selection UI action");
      require_mode(0);
      require(selected_binding() == view && ready(),
              "Returning to automatic selection discarded the current depth binding or readiness");
      pump(2500, [&] {
        require_mode(0);
        require(selected_binding() == view && ready(), "Automatic handover recreated or replaced useful native depth");
        require(observed.runtime->get_device()->get_resource_from_view(view) == backup,
                "Automatic handover changed the preserved depth backup");
      });
      require(varied_linear_depth(9), "Automatic handover lost the actual native scene contents");
      std::puts("PASS manual-to-automatic handover preserves the native shader binding, backup, readiness and useful scene depth against higher-workload scaled depth");

      scene->pattern = 0;
      wait_for_scene("released native source may be replaced after its contents become flat", decoy->width, decoy->height);
      require_mode(0);
      scene->pattern = 9;
      wait_for_scene("native source requalifies after real depth returns", width, height, 9);

      // Trying a different manual source must not erase the previously useful
      // native candidate. This flat peer is a real DSV and remains allocated;
      // after release, the first frame should reuse existing good evidence.
      shadow = target(width, height, 1, 0);
      for (unsigned i = 0; i < 3; ++i) step();
      const auto flat_handle = reinterpret_cast<std::uint64_t>(shadow->texture.p);
      require(select_manual(observed.runtime, flat_handle), "Could not pin the flat native peer for recovery");
      pump(1200, [&] { require_mode(flat_handle); });
      require(select_manual(observed.runtime, 0), "Could not release the flat native peer");
      step();
      require_mode(0);
      const auto recovered = selected_description();
      require(recovered.Width == width && recovered.Height == height && recovered.Format == native_description.Format && ready(),
              "Trying a manual peer erased useful native evidence and caused cold workload fallback");
      pump(1800, [&] {
        require_mode(0);
        const auto actual = selected_description();
        require(actual.Width == width && actual.Height == height, "Recovered native depth lost continuity after manual handover");
      });
      // The selected binding changes in the first frame; the existing processed
      // depth history is allowed to settle before checking its pixel contents.
      require(varied_linear_depth(9), "Recovered native binding did not restore its actual scene pixels");
      std::puts("PASS manual peer selection preserves useful native history; invalid released depth recovers immediately, without a permanent pin");
    }

    void check_manual_reload() {
      const char *enabled = std::getenv("SUNSHINE_DEPTH_MANUAL_RECOVERY_TEST");
      require(enabled && std::strcmp(enabled, "1") == 0,
              "Manual recovery fixture requires explicit SUNSHINE_DEPTH_MANUAL_RECOVERY_TEST=1");
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
      using state_t = BOOL (*)(api::effect_runtime *, std::uint64_t *, BOOL *);
      const auto select_manual = reinterpret_cast<select_t>(GetProcAddress(module, "SunshineDepthTestSelectManual"));
      const auto manual_state = reinterpret_cast<state_t>(GetProcAddress(module, "SunshineDepthTestManualState"));
      require(module && select_manual && manual_state,
              "Manual recovery requires the separately named test add-on's UI-selection adapter");

      shadow.reset();
      scene = target(width, height, 1, 1, true, true);
      wait_for_scene("manual reload starts with a captured native D32S8 source", width, height);
      decoy->draws = 1;
      decoy->pattern = 6;
      wait_for_scene("useful peer has positive evidence before the manual choice", width, height, 6);
      const auto original_handle = reinterpret_cast<std::uint64_t>(scene->texture.p);
      require(select_manual(observed.runtime, original_handle), "Could not select the live native source through the UI-selection adapter");

      const auto require_manual = [&](bool expected_recovery, bool check_recovery = true) {
        std::uint64_t selected = 0;
        BOOL recovering = FALSE;
        require(manual_state(observed.runtime, &selected, &recovering), "Could not query manual source state");
        require(selected == original_handle, "A live or temporarily missing manual choice was unexpectedly cleared");
        require(!check_recovery || bool(recovering) == expected_recovery, "Manual recovery state does not match actual source activity");
      };
      const auto pump = [&](unsigned milliseconds, const std::function<void()> &check) {
        const auto until = GetTickCount64() + milliseconds;
        do { step(); check(); } while (GetTickCount64() < until);
      };
      bool draw_original = true, draw_replacement = true, interleaved = false;
      unsigned present = 0;
      render_tracked_depth = [&] {
        if (shadow) draw(*shadow);
        if (draw_replacement && decoy && (!interleaved || ++present % 2 == 0)) draw(*decoy);
        if (draw_original) draw(*scene);
      };
      // A broader useful peer still cannot steal an active manual selection.
      pump(2500, [&] { require_manual(false); });
      require(varied_linear_depth(1), "Active manual source was replaced by a broader useful peer");
      std::puts("PASS active native manual selection retains priority over another useful native source");

      draw_original = false;
      pump(700, [&] { require_manual(false); });
      draw_original = true;
      pump(600, [&] { require_manual(false); });
      require(varied_linear_depth(1), "Manual source failed to resume after a brief rendering gap");
      std::puts("PASS short manual-source rendering gap preserves the choice and resumes normally");

      // Keep the manual resource current through real clears, with no depth
      // draws. The pin remains, but the public shader readiness must go false.
      scene->draws = 0;
      pump(700, [&] {
        require_manual(false);
        bool ready = true;
        observed.runtime->get_uniform_value_bool(uniform("Sunshine_DepthReady"), &ready, 1);
        require(!ready, "A clear-only manual depth resource retained shader readiness");
      });
      scene->draws = 1;
      pump(600, [&] { require_manual(false); });
      bool resumed_ready = false;
      observed.runtime->get_uniform_value_bool(uniform("Sunshine_DepthReady"), &resumed_ready, 1);
      require(resumed_ready && varied_linear_depth(1), "Manual depth failed to recover after clear-only activity");
      std::puts("PASS clear-only manual activity retains the pin, suspends shader readiness and recovers on real draws");

      using accounting_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
      const auto accounting = reinterpret_cast<accounting_t>(GetProcAddress(module, "SunshineDepthTestActivityAccounting"));
      require(accounting && accounting(observed.runtime, original_handle),
              "Mesh/copy/indirect callbacks failed diagnostic-off depth activity accounting");
      std::puts("PASS callback accounting: mesh and copied depth, direct/indexed/indirect draws, no fabricated vertices, merge/reset, nonraster and zero-count rejection; no mesh GPU raster test");

      // The selected DSV is cleared and then filled only by a real GPU copy.
      // A final clear exercises preservation of copied depth without any draw
      // into that destination; donor rendering alone cannot authorize its pin.
      auto copied_scene = target(width, height, 1, 9, true, false);
      const auto ordinary_rendering = render_tracked_depth;
      scene->draws = 0;
      scene->before_final_clear = [&] {
        transition(commands.p, copied_scene->texture.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(commands.p, scene->texture.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyResource(scene->texture.p, copied_scene->texture.p);
        transition(commands.p, scene->texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        transition(commands.p, copied_scene->texture.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
      };
      render_tracked_depth = [&] { draw(*copied_scene); draw(*scene); };
      pump(2500, [&] {
        require_manual(false);
        bool ready = false;
        observed.runtime->get_uniform_value_bool(uniform("Sunshine_DepthReady"), &ready, 1);
        require(ready, "Copy-produced manual scene depth lost current shader readiness");
      });
      require(varied_linear_depth(9), "Copy-only manual source did not preserve its current spatial depth before clear");
      scene->before_final_clear = {};
      scene->draws = 1;
      render_tracked_depth = ordinary_rendering;
      pump(600, [&] { require_manual(false); });
      require(varied_linear_depth(1), "Manual source did not resume raster depth after copy-produced frames");
      std::puts("PASS actual copy-only D32S8 manual depth remains ready and pinned beyond recovery grace, preserves current scene before clear, then resumes raster depth");

      // A loading screen has draws, but they are all clear depth. Time alone
      // cannot revoke the manual choice without fresh useful replacement depth.
      draw_original = false;
      decoy->pattern = 0;
      pump(4200, [&] { require_manual(false, false); });
      require_manual(false, false);
      draw_replacement = false;
      pump(600, [&] { require_manual(false, false); });
      std::puts("PASS sustained flat loading and no-depth presents retain the manual pin without a valid replacement");
      draw_original = true;
      pump(600, [&] { require_manual(false); });
      require(varied_linear_depth(1), "Original source did not cancel recovery when gameplay resumed");
      std::puts("PASS original manual source resumes and cancels pending recovery");

      // The old DSV remains allocated throughout reload. A distinct lower-res
      // target advances the tracker on every present, while recreated native
      // depth renders every other present, matching the reported live sequence.
      draw_original = false;
      decoy = target(width, height, 1, 9, true, true);
      shadow = target(width / 2, height / 2, 1, 1);
      draw_replacement = interleaved = true;
      wait_for_scene("inactive allocated manual source recovers to recreated native depth", width, height, 9);
      std::uint64_t selected = original_handle;
      BOOL recovering = TRUE;
      require(manual_state(observed.runtime, &selected, &recovering) && selected == 0 && !recovering,
              "Confirmed automatic replacement did not clear the inactive manual pin");
      require(scene->texture.p && reinterpret_cast<std::uint64_t>(scene->texture.p) == original_handle,
              "Manual reload fixture accidentally destroyed the old game depth allocation");
      pump(1800, [&] {
        require(manual_state(observed.runtime, &selected, &recovering) && selected == 0,
                "Automatic recovery restored the abandoned manual pin");
        const auto actual = selected_description();
        require(actual.Width == width && actual.Height == height,
                "Interleaved presents evicted the recovered native depth source");
      });
      require(varied_linear_depth(9), "Recovered native depth lost its expected spatial contents");
      std::puts("PASS allocated-but-unused manual source recovers through real DSV sampling to stable native depth; no semantic injection");
      render_tracked_depth = {};
    }

    void run_selection(const std::string &only = {}) {
      create_pipeline();
      scene = target(width / 2, height / 2, 1, 1);
      decoy = target(width, height, 200, 0);
      shadow = target(1024, 1024, 120, 2);
      render_tracked_depth = [&] { if (shadow) draw(*shadow); if (decoy) draw(*decoy); if (scene) draw(*scene); };
      const auto load_limit = std::chrono::steady_clock::now() + std::chrono::seconds(25);
      while ((!observed.runtime || !observed.renders) && std::chrono::steady_clock::now() < load_limit) step();
      require(observed.runtime && observed.renders, "Depth3D fixture did not compile");
      check_unified_addon();
      require(!observed.inject, "Automatic selection fixture must not inject DEPTH");
      set_int("Depth_Map", 1);
      set_float("Depth_Map_Adjust", 40);
      set_float("Depth_Adjustment", 50);
      set_float("Zero_Parallax_Distance", .05f);
      set_int("Range_Boost", 0);
      for (const auto name : {"Depth_Map_Flip", "DB_AutoFit"}) {
        const auto control = uniform(name, false);
        if (control.handle) observed.runtime->set_uniform_value_bool(control, false);
      }
      wait_for_scene("half-resolution DLSS-like input", scene->width, scene->height);
      if (!only.empty()) {
        if (only == "--viewport-only") check_multiple_viewports();
        else if (only == "--coverage-only") check_scene_completeness();
        else if (only == "--reload-only") check_reload_interleaved();
        else if (only == "--manual-reload-only") check_manual_reload();
        else if (only == "--manual-handover-only") check_manual_handover();
        else check_histogram_breadth();
        render_tracked_depth = {};
        return;
      }

      // Reproduces the live bug missed by a flat-decoy-only test: both sources
      // have smooth useful contents, the same distribution and one draw. Mirror
      // the pattern so the pixel oracle distinguishes the resources without
      // giving either candidate a histogram advantage over the other.
      decoy->draws = 1;
      decoy->pattern = 9;
      wait_for_scene("qualified native source replaces confirmed smaller source", width, height, 9);
      const auto stable_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < stable_until) {
        step();
        const auto selected = selected_description();
        require(selected.Width == width && selected.Height == height, "Qualified candidates oscillated after native-source promotion");
      }
      require(varied_linear_depth(9), "Promoted native source changed to another plausible pattern");
      std::puts("PASS qualified native selection remains stable while smaller source stays useful");
      decoy->pattern = 0;
      wait_for_scene("useful lower-resolution source recovers after native source becomes flat", scene->width, scene->height);
      decoy->draws = 200;

      scene = target(width / 3, height / 3, 1, 1, true, true);
      wait_for_scene("one-third resolution D32S8, preserved before clear", scene->width, scene->height);
      scene = target(width * 2 / 3, height * 2 / 3, 1, 1);
      wait_for_scene("two-thirds resolution after resource recreation", scene->width, scene->height);
      scene = target(width, height, 1, 1);
      wait_for_scene("native TAA-like input vs equal-size high-draw decoy", width, height);

      scene->pattern = 0;
      decoy->pattern = 1;
      wait_for_scene("content moves to another equal-size buffer", width, height);
      scene.reset();
      shadow.reset();
      decoy->draws = 1;
      wait_for_scene("sole low-draw scene remains available", width, height);

      check_scene_completeness();
      check_multiple_viewports();
      check_histogram_breadth();
      std::puts("PASS real D3D12 automatic selection: depth histogram, scene completeness, qualified-source comparison, stable promotion, scaled fallback, actual DSV draws, no semantic injection, D32S8 clear preservation, recreation and content recovery");
      render_tracked_depth = {};
    }
  };
}

#if !defined(SUNSHINE_COMMAND_ASSOCIATION_RUNTIME) && !defined(SUNSHINE_RAW_SCENE_RUNTIME)
int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5 && argc != 7 && argc != 8) {
    std::fputs("usage: depth_selection_runtime <official-ReShade64.dll> <Depth3D-Shaders> <SunshineSBS.addon64> <isolated-output> [width height [--viewport-only|--coverage-only|--histogram-only|--reload-only|--manual-reload-only|--manual-handover-only]]\n", stderr);
    return 2;
  }
  std::thread([] { Sleep(150000); std::fputs("FAIL selection runtime watchdog\n", stderr); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    width = argc >= 7 ? unsigned(std::stoul(argv[5])) : 1280;
    height = argc >= 7 ? unsigned(std::stoul(argv[6])) : 720;
    require(argc != 8 || std::string(argv[7]) == "--viewport-only" || std::string(argv[7]) == "--coverage-only" || std::string(argv[7]) == "--histogram-only" || std::string(argv[7]) == "--reload-only" || std::string(argv[7]) == "--manual-reload-only" || std::string(argv[7]) == "--manual-handover-only", "Unknown fixture option");
    require(width >= 640 && width <= 3840 && height >= 360 && height <= 2160 && width % 2 == 0 && height % 2 == 0, "Invalid fixture size");
    selection_fixture_t fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fs::absolute(argv[4]), 2, 0, fs::absolute(argv[3]));
    fixture.run_selection(argc == 8 ? argv[7] : "");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
#endif
