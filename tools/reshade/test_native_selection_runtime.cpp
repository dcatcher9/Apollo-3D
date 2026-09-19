// SPDX-License-Identifier: GPL-3.0-only
// Opt-in actual ReShade D3D12 production selector/calibration integration.
// Real DSV draws feed DEPTH; calibration and DEPTH bindings are never injected.
#define SUNSHINE_NATIVE_STEREO_RUNTIME_HELPER
#include "test_native_stereo_runtime.cpp"
#include "../../src/reshade_bridge_protocol.h"
#include <d3dcompiler.h>
#include <memory>

namespace {
  struct native_selection_fixture_t : fixture_t {
    struct target_t {
      com_ptr<ID3D12Resource> texture;
      com_ptr<ID3D12DescriptorHeap> heap;
      unsigned x = 0, y = 0, w = 0, h = 0, draws = 1;
      bool normal = false, flat = false;
    };
    std::unique_ptr<target_t> scene, decoy;
    bool statistics_mode = false;
    com_ptr<ID3D12RootSignature> root;
    com_ptr<ID3D12PipelineState> pipeline;

    void check_production_module() {
      const HMODULE module = GetModuleHandleW(L"SunshineSBS.addon64");
      require(module != nullptr, "Official runtime did not load the production unified addon");
      const auto name = reinterpret_cast<const char *const *>(GetProcAddress(module, "NAME"));
      require(name && *name && std::strcmp(*name, "Sunshine 3D") == 0, "Fixture requires the production Sunshine 3D addon");
      require(GetModuleHandleW(L"SunshineDepth.addon64") == nullptr && GetModuleHandleW(L"SunshineDepthProbe.addon64") == nullptr,
        "Legacy selector/probe coexists with production module");
      char disabled[256] {};
      size_t length = sizeof(disabled);
      require(reshade::get_config_value(nullptr, "ADDON", "DisabledAddons", disabled, &length) && std::strcmp(disabled, "Generic Depth") == 0,
        "Actual ReShade Generic Depth is not disabled");
      bool automatic = true;
      require(reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "AutoSelectSceneDepth", automatic) && automatic != statistics_mode,
        "Actual runtime did not load the requested automatic/statistics selection mode");
      const auto mapping_name = std::wstring(reshade_bridge::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      HANDLE mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, mapping_name.c_str());
      require(mapping != nullptr, "Production exporter did not initialize its mapping");
      const auto *state = static_cast<const reshade_bridge::shared_state_t *>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, sizeof(reshade_bridge::shared_state_t)));
      const bool valid = state && state->shared_bytes == sizeof(*state) && state->metadata.signature == reshade_bridge::magic &&
        state->metadata.protocol_version == reshade_bridge::version && state->metadata.producer_pid == GetCurrentProcessId();
      if (state) UnmapViewOfFile(state);
      CloseHandle(mapping);
      require(valid, "Unified producer identity/protocol is invalid");
      require(!observed.inject && !observed.depth_view.handle && observed.ready_uniforms.empty(),
        "Production calibration fixture must never inject DEPTH or readiness");
      std::puts("PASS production unified selector/exporter loaded; Generic Depth disabled; no injected depth/calibration");
    }

    void create_scene_pipeline() {
      const char *shader = R"(
cbuffer Scene : register(b0) { float4 rect; uint pattern; };
float4 vs(uint id : SV_VertexID) : SV_Position {
  float2 uv = float2((id << 1) & 2, id & 2);
  return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float ps(float4 position : SV_Position) : SV_Depth {
  if (pattern == 0) return 0;
  float2 uv = (position.xy - rect.xy) / rect.zw;
  float reversed = 0.1 + 0.5 * uv.x + 0.2 * uv.y;
  return pattern == 2 ? 1.0 - reversed : reversed;
})";
      com_ptr<ID3DBlob> vs, ps, errors, serialized;
      checked(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "vs", "vs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, vs.put(), errors.put()), "Compile real scene VS");
      checked(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "ps", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, ps.put(), errors.put()), "Compile real scene depth PS");
      D3D12_ROOT_PARAMETER constants {};
      constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      constants.Constants.Num32BitValues = 5;
      constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC signature {};
      signature.NumParameters = 1;
      signature.pParameters = &constants;
      checked(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, serialized.put(), errors.put()), "Serialize real scene root");
      checked(game->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(root.put())), "Create real scene root");
      D3D12_GRAPHICS_PIPELINE_STATE_DESC desc {};
      desc.pRootSignature = root.p;
      desc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
      desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
      desc.SampleMask = UINT_MAX;
      desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
      desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
      desc.RasterizerState.DepthClipEnable = TRUE;
      desc.DepthStencilState.DepthEnable = TRUE;
      desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
      desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
      desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      desc.SampleDesc.Count = 1;
      desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
      checked(game->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(pipeline.put())), "Create real scene depth pipeline");
    }

    std::unique_ptr<target_t> create_target(bool flat, bool normal, unsigned x = 0, unsigned y = 0) {
      auto result = std::make_unique<target_t>();
      result->x = x; result->y = y;
      result->w = flat ? width : width / 2;
      result->h = flat ? height : height / 2;
      result->draws = statistics_mode ? (flat ? 12 : 200) : (flat ? 200 : 1);
      result->flat = flat; result->normal = normal;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width; desc.Height = height;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = DXGI_FORMAT_R32_TYPELESS;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
      D3D12_CLEAR_VALUE clear {};
      clear.Format = DXGI_FORMAT_D32_FLOAT;
      clear.DepthStencil.Depth = normal ? 1.f : 0.f;
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(result->texture.put())), "Create real game depth target");
      D3D12_DESCRIPTOR_HEAP_DESC descriptors {};
      descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
      descriptors.NumDescriptors = 1;
      checked(game->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(result->heap.put())), "Create real game DSV heap");
      D3D12_DEPTH_STENCIL_VIEW_DESC view {};
      view.Format = DXGI_FORMAT_D32_FLOAT;
      view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
      game->CreateDepthStencilView(result->texture.p, &view, result->heap->GetCPUDescriptorHandleForHeapStart());
      return result;
    }

    void draw_scene(target_t &target) {
      const auto dsv = target.heap->GetCPUDescriptorHandleForHeapStart();
      const float clear = target.normal ? 1.f : 0.f;
      commands->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
      commands->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, clear, 0, 0, nullptr);
      commands->SetGraphicsRootSignature(root.p);
      commands->SetPipelineState(pipeline.p);
      commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      const D3D12_VIEWPORT viewport {float(target.x), float(target.y), float(target.w), float(target.h), 0, 1};
      const D3D12_RECT scissor {LONG(target.x), LONG(target.y), LONG(target.x + target.w), LONG(target.y + target.h)};
      commands->RSSetViewports(1, &viewport);
      commands->RSSetScissorRects(1, &scissor);
      struct { float x, y, w, h; unsigned pattern; } constants {
        float(target.x), float(target.y), float(target.w), float(target.h), target.flat ? 0u : target.normal ? 2u : 1u};
      commands->SetGraphicsRoot32BitConstants(0, 5, &constants, 0);
      for (unsigned i = 0; i != target.draws; ++i)
        commands->DrawInstanced(3, 1, 0, 0);
      // The scene only survives through the selector's actual before-clear copy.
      commands->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, clear, 0, 0, nullptr);
      commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
    }

    bool flag(const char *name) {
      bool value = false;
      observed.runtime->get_uniform_value_bool(uniform(name), &value, 1);
      return value;
    }
    float scalar(const char *name) {
      float value = 0;
      observed.runtime->get_uniform_value_float(uniform(name), &value, 1);
      return value;
    }
    bool calibrated() { return flag("Sunshine_DepthReady") && flag("Sunshine_Calibrated"); }

    void discover_production() {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
      while (!observed.renders && std::chrono::steady_clock::now() < deadline) step();
      require(observed.runtime && observed.renders, "Independent production shader did not compile/execute");
      find_texture("DoubleTex", exported, width * 2, color == 1 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT);
      set_int("DepthDirection", 0);
      set_bool("DepthView", true);
      check_production_module();
    }

    void wait_calibrated(const char *phase, bool normal) {
      const auto started = std::chrono::steady_clock::now();
      const auto deadline = started + std::chrono::seconds(25);
      unsigned frames = 0;
      do {
        step(); ++frames;
      } while ((!calibrated() || (scalar("Sunshine_RawGain") < 0) != normal) && std::chrono::steady_clock::now() < deadline);
      const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      std::printf("MEASURE production %s frames=%u elapsed=%.3f ready=%u calibrated=%u anchor=%.9g gain=%.9g\n",
        phase, frames, elapsed, unsigned(flag("Sunshine_DepthReady")), unsigned(flag("Sunshine_Calibrated")), scalar("Sunshine_RawAnchor"), scalar("Sunshine_RawGain"));
      require(calibrated() && (scalar("Sunshine_RawGain") < 0) == normal, "Production selector never calibrated the scene in its automatic depth convention");
    }

    std::vector<std::uint8_t> verify_scene(const char *phase) {
      require(scene && calibrated(), "Scene oracle requires actual calibrated source");
      std::array<float, 4> rect {};
      observed.runtime->get_uniform_value_float(uniform("Sunshine_DepthRect"), rect.data(), 4);
      const float expected_rect[] {float(scene->w) / width, float(scene->h) / height, float(scene->x) / width, float(scene->y) / height};
      for (unsigned i = 0; i != 4; ++i)
        require(std::abs(rect[i] - expected_rect[i]) < 1.e-7f, "Production copied viewport scale/offset is stale or wrong");
      const float anchor = scalar("Sunshine_RawAnchor"), gain = scalar("Sunshine_RawGain");
      // The distribution .1+.5U+.2V has median .45 and 10–90% span
      // .7-2*sqrt(2*.5*.2*.1). Sparse point sampling accounts for <2%.
      require(std::abs(anchor - (scene->normal ? .55f : .45f)) < .005f, "Calibration anchor is not raw scene median");
      const float expected_gain = 1.f / (.7f - 2.f * std::sqrt(.02f));
      require(std::abs(std::abs(gain) - expected_gain) < .04f, "Calibration gain is not robust raw scene spread");
      require((gain < 0) == scene->normal, "Exact repeated clear convention did not determine gain sign");

      com_ptr<ID3D12Resource> bound;
      observed.runtime->enumerate_texture_variables("SunshineDepth3D.fx", [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char name[256] {}; runtime->get_texture_variable_name(variable, name);
        if (!named(name, "GameDepth")) return;
        api::resource_view view {}; runtime->get_texture_binding(variable, &view, nullptr);
        if (view.handle) {
          const auto source = runtime->get_device()->get_resource_from_view(view);
          bound.p = reinterpret_cast<ID3D12Resource *>(source.handle);
          if (bound.p) bound->AddRef();
        }
      });
      require(bound.p && bound->GetDesc().Width == width && bound->GetDesc().Height == height, "Production DEPTH binding is missing or wrong-sized");
      const auto raw_bytes = read(bound.p, D3D12_RESOURCE_STATE_COPY_DEST);
      require(raw_bytes.size() == size_t(width) * height * sizeof(float), "Production bound depth is not raw R32");
      const auto depth = read(exported.p);
      float minimum = 1, maximum = 0;
      for (unsigned yi = 1; yi != 4; ++yi) for (unsigned xi = 1; xi != 5; ++xi) {
        const unsigned x = width * xi / 5, y = height * yi / 4;
        const unsigned dx = scene->x + unsigned((double(x) + .5) * scene->w / width);
        const unsigned dy = scene->y + unsigned((double(y) + .5) * scene->h / height);
        const float reversed = .1f + .5f * (float(dx - scene->x) + .5f) / scene->w + .2f * (float(dy - scene->y) + .5f) / scene->h;
        const float expected_raw = scene->normal ? 1.f - reversed : reversed;
        float raw = 0; std::memcpy(&raw, raw_bytes.data() + (size_t(dy) * width + dx) * sizeof(float), sizeof(raw));
        require(std::abs(raw - expected_raw) < 1.e-6f, "Actual selected backup is not the real before-clear scene gradient");
        const float expected = std::clamp(.5f + .5f * (expected_raw - anchor) * gain, 0.f, 1.f);
        const float pixel = channel(depth, x, y, 0);
        require(std::abs(pixel - expected) < .002f, "Actual DepthView disagrees with raw-depth/viewport calibration oracle");
        require(pixel == channel(depth, x + width, y, 0), "Production depth view is not identical in both eyes");
        minimum = std::min(minimum, pixel); maximum = std::max(maximum, pixel);
      }
      require(maximum - minimum > .3f, "Production depth diagnostic is flat despite a real scene gradient");
      std::printf("PASS %s: real %s scene selected, before-clear copy, exact offset viewport, raw calibration and full-eye pixels\n",
        phase, statistics_mode ? "statistics-selected" : "low-draw");
      return depth;
    }

    void require_duplicate_eyes() {
      const auto pixels = read(exported.p);
      for (unsigned y = height / 8; y < height * 7 / 8; y += std::max(1u, height / 32))
        for (unsigned x = 0; x != width; ++x)
          for (unsigned c = 0; c != 3; ++c)
            require(channel(pixels, x, y, c) == channel(pixels, x + width, y, c), "Unavailable production depth does not give identical eyes");
    }

    void run_production() {
      // Compile/initialize with no scene, so subsequent startup cannot inherit
      // samples gathered while waiting for shader compilation.
      discover_production();
      require(!calibrated(), "Production shader calibrated without any game depth");
      create_scene_pipeline();
      scene = create_target(false, false, 8, 6);
      decoy = create_target(true, false);
      render_tracked_depth = [&] { if (scene) draw_scene(*scene); if (decoy) draw_scene(*decoy); };
      step();
      require(!calibrated(), "Fresh scene is immediately calibrated without three captures");
      require_duplicate_eyes();
      wait_calibrated("reversed-Z startup", false);
      const auto reversed = verify_scene("reversed-Z startup");

      // Calibration is historical; current-frame validity must be independent.
      render_tracked_depth = {};
      step();
      require(!flag("Sunshine_DepthReady") && !flag("Sunshine_Calibrated"), "Color-only present reused stale depth readiness/calibration");
      require_duplicate_eyes();
      render_tracked_depth = [&] { if (scene) draw_scene(*scene); if (decoy) draw_scene(*decoy); };
      step();
      require(calibrated(), "A single unavailable present discarded stable calibration for the same source");
      verify_scene("same-source recovery");

      // Replace native scene lifetime and its clear convention. Old calibration
      // must be unavailable immediately, then rebuilt from three fresh captures.
      scene = create_target(false, true, 8, 6);
      step();
      require(!calibrated(), "Recreated scene inherited a stale raw calibration");
      require_duplicate_eyes();
      const auto fresh_started = std::chrono::steady_clock::now();
      wait_calibrated("normal-Z recreated source", true);
      require(std::chrono::duration<double>(std::chrono::steady_clock::now() - fresh_started).count() >= .4,
        "Recreated source calibrated before three separately scheduled captures could complete");
      const auto normal = verify_scene("normal-Z recreated source");
      require(image_difference(normal, reversed) < .003f, "Equivalent normal/reversed game depth yields different calibrated geometry");

      // Change only the active copy rectangle in the SAME allocation. This must
      // invalidate historical calibration even when the resource pointer lives.
      scene->x += 12; scene->y += 8;
      step();
      require(!calibrated(), "New copied layout inherited the old region's calibration");
      require_duplicate_eyes();
      wait_calibrated("same-resource offset change", true);
      verify_scene("same-resource offset change");
      require(!observed.inject && observed.ready_uniforms.empty(), "Fixture unexpectedly injected production inputs");
      std::printf("PASS production native depth selection/calibration integration (%s; no manual override)\n",
        statistics_mode ? "Auto-select OFF, draw statistics" : "Auto-select ON, content selection");
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const bool statistics_mode = argc > 6 && std::strcmp(argv[argc - 1], "--statistics") == 0;
  const int arguments = argc - int(statistics_mode);
  if (arguments != 6 && arguments != 8) {
    std::fputs("usage: test_native_selection_runtime <official-ReShade64.dll> <Sunshine-Shaders> <isolated-output-directory> <SunshineSBS.addon64> <srgb|scrgb|pq> [width height] [--statistics]\n", stderr);
    return 2;
  }
  std::thread([] { Sleep(180000); std::fputs("FAIL native selection fixture watchdog\n", stderr); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    const unsigned color = std::strcmp(argv[5], "srgb") == 0 ? 1 : std::strcmp(argv[5], "scrgb") == 0 ? 2 : std::strcmp(argv[5], "pq") == 0 ? 3 : 0;
    require(color != 0, "Invalid fixture source color");
    if (arguments == 8) {
      width = unsigned(std::stoul(argv[6])); height = unsigned(std::stoul(argv[7]));
      require(width >= 640 && height >= 360 && width <= 3840 && height <= 2160 && width % 2 == 0 && height % 2 == 0, "Fixture dimensions must be even, from 640x360 to 3840x2160");
    }
    native_selection_fixture_t fixture;
    fixture.statistics_mode = statistics_mode;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fs::absolute(argv[3]), color, 0, fs::absolute(argv[4]), !statistics_mode);
    fixture.run_production();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
