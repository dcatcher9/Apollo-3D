// SPDX-License-Identifier: GPL-3.0-only
// Opt-in functional test of source-alpha protection in the real native shader.
// No installed FX, CPU warp replica, visible window or performance claim.
#include "game3d_renderer.h"
#include "game3d_controls.h"
#include "test_game3d_debug_dump_runtime.h"
#include <reshade.hpp>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
  namespace api = reshade::api;
  namespace fs = std::filesystem;
  using Microsoft::WRL::ComPtr;
  api::effect_runtime *observed_runtime{};
  unsigned runtime_reloads{};
  void init_runtime(api::effect_runtime *runtime) { observed_runtime = runtime; }
  void reload_runtime(api::effect_runtime *) { ++runtime_reloads; }
  void require(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
  void checked(HRESULT result, const char *message) {
    if (FAILED(result)) { char text[256]; std::snprintf(text, sizeof(text), "%s: 0x%08lx", message, static_cast<unsigned long>(result)); throw std::runtime_error(text); }
  }
  float half_float(std::uint16_t value) {
    const float sign = (value & 0x8000) ? -1.f : 1.f;
    const unsigned exponent = (value >> 10) & 31, fraction = value & 1023;
    if (!exponent) return sign * std::ldexp(float(fraction), -24);
    if (exponent == 31) return fraction ? NAN : sign * INFINITY;
    return sign * std::ldexp(1.f + float(fraction) / 1024.f, int(exponent) - 15);
  }
  std::uint16_t half_bits(float value) {
    // Test inputs are exactly representable finite multiples of 1/64, plus
    // explicit IEEE exceptional alpha encodings for validity tests.
    if (std::isnan(value)) return 0x7e00;
    if (std::isinf(value)) return std::signbit(value) ? 0xfc00 : 0x7c00;
    if (!value) return 0;
    const bool negative = value < 0; value = std::abs(value);
    int exponent{}; const float fraction = std::frexp(value, &exponent);
    return std::uint16_t((negative ? 0x8000 : 0) | ((exponent + 14) << 10) | unsigned(std::lround((fraction * 2 - 1) * 1024)));
  }
  unsigned pixel_bytes(DXGI_FORMAT format) {
    if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) return 8;
    require(format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R8G8B8A8_UNORM ||
      format == DXGI_FORMAT_R10G10B10A2_UNORM, "unexpected readback format");
    return 4;
  }
  struct image {
    std::vector<unsigned char> bytes;
    unsigned width{}, height{};
    DXGI_FORMAT format{};
    float channel(unsigned x, unsigned y, unsigned c) const {
      const auto *p = bytes.data() + (size_t(y) * width + x) * pixel_bytes(format);
      if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) { std::uint16_t v; std::memcpy(&v, p + 2 * c, 2); return half_float(v); }
      if (format == DXGI_FORMAT_R10G10B10A2_UNORM) { std::uint32_t v; std::memcpy(&v, p, 4); return float((v >> (10 * c)) & (c == 3 ? 3 : 1023)) / (c == 3 ? 3.f : 1023.f); }
      if (format == DXGI_FORMAT_R32_FLOAT) { float v; std::memcpy(&v, p, 4); return v; }
      return p[c] / 255.f;
    }
  };
  struct result { image field, output; };
  struct fixture {
    unsigned width, height, color;
    HWND window{};
    HMODULE module{};
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain> swapchain;
    ComPtr<ID3D11Texture2D> runtime_backbuffer, backbuffer, source, depth;
    ComPtr<ID3D11RenderTargetView> rtv;
    ComPtr<ID3D11ShaderResourceView> depth_view;
    sunshine_game3d::renderer renderer, control_renderer;
    std::uint64_t mask_capture = 0;
    bool has_control{};
    std::vector<unsigned char> original;
    fixture(const fs::path &runtime, const fs::path &directory, unsigned c, unsigned w, unsigned h, const fs::path &control): width(w), height(h), color(c) {
      require(width >= 8 && width <= 3840 && height >= 8 && height <= 2160, "test dimensions out of bounds");
      require(!fs::exists(directory), "use a fresh evidence directory");
      fs::create_directories(directory / "effects"); fs::create_directories(directory / "addons");
      fs::copy_file(runtime, directory / "dxgi.dll");
      std::ofstream(directory / "ReShade.ini") << "[ADDON]\nAddonPath=.\\addons\n[GENERAL]\nEffectSearchPaths=.\\effects\nPresetPath=.\\preset.ini\nEffectCachePath=.\\cache\n[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n";
      std::ofstream(directory / "preset.ini") << "Techniques=\n";
      SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", directory.c_str());
      wchar_t system[MAX_PATH]{}; GetSystemDirectoryW(system, MAX_PATH);
      require(LoadLibraryW((fs::path(system) / "d3d11.dll").c_str()) != nullptr, "preload D3D11");
      module = LoadLibraryW((directory / "dxgi.dll").c_str()); require(module, "load official ReShade");
      require(reshade::register_addon(GetModuleHandleW(nullptr), module), "register runtime observer");
      reshade::register_event<reshade::addon_event::init_effect_runtime>(init_runtime);
      reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(reload_runtime);
      WNDCLASSW wc{}; wc.lpfnWndProc = DefWindowProcW; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"SunshineSourceAlphaFunctionalTest";
      RegisterClassW(&wc);
      // Bootstrap the official runtime using its established ordinary-size
      // hidden swapchain. The renderer gets a separate exact-sized source on
      // this same wrapped device, so tiny shader dimensions do not depend on
      // the runtime's small-swapchain admission policy.
      constexpr unsigned runtime_width = 640, runtime_height = 360;
      RECT bounds{0, 0, runtime_width, runtime_height};
      require(AdjustWindowRect(&bounds, WS_OVERLAPPEDWINDOW, FALSE), "adjust hidden client rectangle");
      window = CreateWindowW(wc.lpszClassName, L"Hidden source-alpha regression", WS_OVERLAPPEDWINDOW, 0, 0,
        bounds.right - bounds.left, bounds.bottom - bounds.top, nullptr, nullptr, wc.hInstance, nullptr);
      require(window && !IsWindowVisible(window), "fixture window must remain hidden");
      DXGI_SWAP_CHAIN_DESC desc{}; desc.BufferDesc.Width = runtime_width; desc.BufferDesc.Height = runtime_height;
      desc.BufferDesc.Format = color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count = 1; desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; desc.BufferCount = 2;
      desc.OutputWindow = window; desc.Windowed = TRUE; desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      const auto create = reinterpret_cast<decltype(&D3D11CreateDeviceAndSwapChain)>(GetProcAddress(module, "D3D11CreateDeviceAndSwapChain"));
      require(create != nullptr, "official runtime has no D3D11 proxy");
      checked(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &desc, &swapchain, &device, nullptr, &context), "create wrapped game device");
      ComPtr<IDXGISwapChain3> chain; checked(swapchain.As(&chain), "swapchain3");
      checked(chain->SetColorSpace1(color == 2 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709), "source color space");
      checked(swapchain->GetBuffer(0, IID_PPV_ARGS(&runtime_backbuffer)), "runtime backbuffer");
      checked(device->CreateRenderTargetView(runtime_backbuffer.Get(), nullptr, &rtv), "runtime backbuffer RTV");
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while ((!observed_runtime || !runtime_reloads) && std::chrono::steady_clock::now() < deadline) {
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&message); DispatchMessageW(&message); }
        ID3D11RenderTargetView *target = rtv.Get(); context->OMSetRenderTargets(1, &target, nullptr);
        const float black[4]{}; context->ClearRenderTargetView(rtv.Get(), black); checked(swapchain->Present(0, 0), "initialize runtime");
        Sleep(5);
      }
      require(observed_runtime && runtime_reloads, "official runtime did not initialize");
      unsigned techniques{}; observed_runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++techniques; });
      require(techniques == 0, "test must use only native renderer");
      D3D11_TEXTURE2D_DESC source_desc{}; runtime_backbuffer->GetDesc(&source_desc);
      source_desc.Width = width; source_desc.Height = height; source_desc.MiscFlags = 0;
      source_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
      checked(device->CreateTexture2D(&source_desc, nullptr, &backbuffer), "exact-sized renderer source");
      source_desc.BindFlags = 0;
      checked(device->CreateTexture2D(&source_desc, nullptr, &source), "source pattern");
      D3D11_TEXTURE2D_DESC dd{}; dd.Width = width; dd.Height = height; dd.ArraySize = dd.MipLevels = dd.SampleDesc.Count = 1;
      dd.Format = DXGI_FORMAT_R32_FLOAT; dd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      std::vector<float> raw(size_t(width) * height, .02f); D3D11_SUBRESOURCE_DATA data{raw.data(), width * 4, 0};
      checked(device->CreateTexture2D(&dd, &data, &depth), "raw depth"); checked(device->CreateShaderResourceView(depth.Get(), nullptr, &depth_view), "depth SRV");
      require(renderer.configure(observed_runtime, {reinterpret_cast<std::uint64_t>(backbuffer.Get())}, static_cast<api::color_space>(color)), "configure production renderer");
      if (!control.empty()) {
        std::ifstream input(control, std::ios::binary);
        require(input.good(), "cannot read frozen control shader");
        const std::string shader{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        require(!shader.empty() && control_renderer.configure(observed_runtime, {reinterpret_cast<std::uint64_t>(backbuffer.Get())},
          static_cast<api::color_space>(color), shader), "configure frozen control renderer");
        has_control = true;
      }
    }
    ~fixture() {
      if (observed_runtime) { observed_runtime->get_command_queue()->wait_idle(); renderer.reset_after_runtime_drain(); control_renderer.reset_after_runtime_drain(); }
      if (module) reshade::unregister_addon(GetModuleHandleW(nullptr), module);
      if (window) DestroyWindow(window);
    }
    image read(api::resource value) {
      auto *texture = reinterpret_cast<ID3D11Texture2D *>(value.handle); require(texture, "missing GPU output");
      D3D11_TEXTURE2D_DESC desc{}; texture->GetDesc(&desc); image out{{}, desc.Width, desc.Height, desc.Format};
      const auto pitch = desc.Width * pixel_bytes(desc.Format);
      desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = desc.MiscFlags = 0; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging; checked(device->CreateTexture2D(&desc, nullptr, &staging), "readback texture");
      context->CopyResource(staging.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE map{}; checked(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map), "GPU readback");
      out.bytes.resize(size_t(pitch) * desc.Height);
      for (unsigned y = 0; y < desc.Height; ++y) std::memcpy(out.bytes.data() + size_t(y) * pitch, static_cast<const unsigned char *>(map.pData) + size_t(y) * map.RowPitch, pitch);
      context->Unmap(staging.Get(), 0); return out;
    }
    void pattern(const std::vector<float> &alpha, bool fixed_rgb = false, unsigned color_phase = 0) {
      const auto bpp = color == 2 ? 8u : 4u; original.resize(size_t(width) * height * bpp);
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        const bool ui = std::isfinite(alpha[i]) && alpha[i] > 0;
        // Scene has exactly zero red; red UI becomes an unambiguous ghost
        // marker. Green texture keeps the scene useful for displacement checks.
        // The FG transition regression must change alpha without changing the
        // already-composited picture. Its stationary stripe also makes actual
        // left/right image displacement measurable, beyond field-only checks.
        const auto source_x = (x + color_phase) % width;
        const bool marker = fixed_rgb ? source_x >= width / 3 && source_x < width / 2 : ui;
        const float values[4]{marker ? (color == 2 ? 2.f : 1.f) : 0.f, float(8 + ((source_x * 7 + y * 3) % 48)) / 64.f, marker ? 1.f : 0.f, alpha[i]};
        if (color == 2) for (unsigned c = 0; c != 4; ++c) { const auto half = half_bits(values[c]); std::memcpy(original.data() + i * bpp + c * 2, &half, 2); }
        else for (unsigned c = 0; c != 4; ++c) original[i * bpp + c] = static_cast<unsigned char>(std::lround(std::clamp(values[c], 0.f, 1.f) * 255));
      }
      context->UpdateSubresource(source.Get(), 0, nullptr, original.data(), width * bpp, 0);
    }
    api::resource_view retain_mask(const std::vector<float> &alpha) {
      pattern(alpha, true);
      auto pixels = original;
      const auto bpp = color == 2 ? 8u : 4u;
      // Deliberately unrelated old RGB: consuming these channels would freeze
      // or replace the current game picture, even if the alpha itself is right.
      for (size_t i = 0; i != alpha.size(); ++i) {
        if (color == 2) {
          const std::uint16_t rgb[]{half_bits(4.f), half_bits(0.f), half_bits(4.f)};
          std::memcpy(pixels.data() + i * bpp, rgb, sizeof(rgb));
        } else {
          pixels[i * bpp] = pixels[i * bpp + 1] = 0; pixels[i * bpp + 2] = 255;
        }
      }
      const auto view = renderer.prepare_ui_source(++mask_capture, [&](api::resource texture) {
        context->UpdateSubresource(reinterpret_cast<ID3D11Resource *>(texture.handle), 0, nullptr, pixels.data(), width * bpp, 0);
        return true;
      });
      require(view.handle, "renderer-owned retained UI texture unavailable");
      require(read(renderer.ui_source()).bytes == pixels, "retained real-frame alpha upload changed native RGBA");
      return view;
    }
    result render(bool protect, int sign, bool flat = false, bool control = false, bool frame_generation_active = false,
        api::resource_view alpha_source = {}, const fs::path &dump_directory = {},
        const sunshine_game3d::ui_plane_parameters &plane = {},
        const sunshine_game3d::render_parameters *override_parameters = nullptr,
        const sunshine_game3d::alpha_auto_source *automatic = nullptr) {
      auto &active = control ? control_renderer : renderer;
      sunshine_game3d::render_parameters p;
      p.strength = flat ? 0 : 100; p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 1;
      p.depth_scale = 100000; p.strength_blend = 1; p.projection = {0, 1};
      p.convergence = {.05f, sign > 0 ? .03f : .01f}; p.disparity_limit_uv = .04f;
      if (override_parameters) p = *override_parameters;
      context->CopyResource(backbuffer.Get(), source.Get());
      auto *queue = observed_runtime->get_command_queue();
      require(active.render(queue->get_immediate_command_list(), {reinterpret_cast<std::uint64_t>(backbuffer.Get())},
        {reinterpret_cast<std::uint64_t>(depth_view.Get())}, p,
        alpha_source.handle ? protect : sunshine_game3d::source_alpha_ui_for_present(protect, frame_generation_active),
        alpha_source, plane, automatic), "production render failed");
      require(sunshine_game3d::ui_parameter_words(protect, active.consumed_ui_plane()) ==
          sunshine_game3d::ui_parameter_words(protect, plane), "renderer changed consumed UI-plane bits");
      std::unique_ptr<sunshine_game3d_test::dump_fixture> dump;
      if (!dump_directory.empty()) {
        dump = std::make_unique<sunshine_game3d_test::dump_fixture>();
        dump->begin(observed_runtime, active, p, {reinterpret_cast<std::uint64_t>(depth_view.Get())},
          frame_generation_active, static_cast<api::color_space>(color));
      }
      queue->flush_immediate_command_list(); active.finish_present();
      if (dump) dump->submitted(observed_runtime);
      queue->wait_idle();
      if (dump) dump->verify(observed_runtime, [&](api::resource texture, bool) { return read(texture).bytes; }, dump_directory);
      result out{read(active.diagnostics().final_field), read(active.output())};
      require(out.field.width == width && out.field.height == height && out.output.width == width * 2 && out.output.height == height,
        "renderer used bootstrap swapchain dimensions instead of exact test source");
      require(read({reinterpret_cast<std::uint64_t>(backbuffer.Get())}).bytes == original, "UI protection modified mono source");
      return out;
    }
  };
  double color_error(const image &a, const image &b, unsigned x, unsigned y) {
    double error{}; for (unsigned c = 0; c != 3; ++c) error = std::max(error, double(std::abs(a.channel(x, y, c) - b.channel(x, y, c)))); return error;
  }
  double float_spacing(double value) {
    const float f = static_cast<float>(std::abs(value));
    return double(std::nextafter(f, std::numeric_limits<float>::infinity())) - f;
  }

  // Closed-form projection of one constant UI plane, not a CPU warp replica.
  double plane_parallax(const fixture &gpu, const sunshine_game3d::render_parameters &p,
      const sunshine_game3d::ui_plane_parameters &plane) {
    const double strength = p.strength * .01 * p.strength_blend;
    const double uv = double(gpu.height) * 100. / (2160. * gpu.width);
    const double projected = p.convergence[0] * double(p.depth_scale) *
      (double(plane.inverse_depth) - p.convergence[1]) * strength * uv;
    const double budget = p.disparity_limit_uv * strength;
    return std::clamp(std::clamp(projected, -2.5 * uv, 1.5 * uv), -budget, budget);
  }

  // Diagnosis only: measure consecutive native GPU planes and fields without
  // requiring a temporal stability policy. No production path or clock changes.
  void temporal_ui_probe(fixture &gpu, const fs::path &directory, bool front_limit) {
    using sunshine_game3d::ui_plane_mode;
    using sunshine_game3d::ui_plane_parameters;
    require(gpu.width >= 320 && gpu.height >= 180, "temporal UI probe needs at least 320x180");
    auto &active = gpu.has_control ? gpu.control_renderer : gpu.renderer;
    require(active.active_shader_source().find(front_limit ? "#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1" :
        "#define SUNSHINE_UI_NEAREST_PLANE 1") != std::string_view::npos,
      "temporal UI probe requires a shader supporting the selected mode");
    std::ofstream shader(directory / "temporal-ui-probe-shader.hlsl", std::ios::binary);
    shader << active.active_shader_source();
    require(shader.good(), "cannot freeze temporal probe shader");
    const size_t count = size_t(gpu.width) * gpu.height;
    const unsigned px = gpu.width / 2, py = gpu.height * 5 / 12;
    const size_t peak = size_t(py) * gpu.width + px;
    std::vector<float> alpha(count, 0.f), raw(count, .125f);
    size_t covered{};
    for (unsigned y = gpu.height / 3; y < gpu.height / 2; ++y)
      for (unsigned x = gpu.width / 4; x < gpu.width * 3 / 4; ++x) {
        alpha[size_t(y) * gpu.width + x] = .25f;
        ++covered;
      }
    require(alpha[peak] > 0 && alpha.front() == 0, "temporal probe coverage fixture is invalid");
    raw.front() = .95f; // A closer uncovered sample must not set the UI plane.
    gpu.pattern(alpha, true); // The exact RGB, alpha, crop and jitter stay fixed.
    sunshine_game3d::render_parameters base;
    base.strength = 100; base.depth_ready = base.camera_ready = 1; base.coordinate_basis = 0;
    base.depth_scale = 4320.f / gpu.height; base.strength_blend = 1;
    base.projection = {0, 1}; base.convergence = {.05f, .125f}; base.disparity_limit_uv = .04f;
    const ui_plane_parameters plane{front_limit ? ui_plane_mode::front_limit : ui_plane_mode::depth_midpoint_nearest_ui, .25f};
    const std::array<const char *, 5> scenarios{{
      "steady_control", "alternating_covered_nearest", "static_depth_changing_gain",
      "static_depth_changing_zero", "static_depth_changing_gain_and_zero"}};
    nlohmann::json document{{"schema", "sunshine.game3d.temporal-ui-probe.v1"},
      {"purpose", "Measure current native GPU temporal response; no assertion of a desired smoothing or stability policy."},
      {"measurement", front_limit ? "Actual GPU final_field readback after every render; no inverse-depth scalar exists in front-limit mode and no CPU warp is simulated." :
        "Actual GPU ui_plane_resolved and final_field readback after every render; no CPU warp simulation."},
      {"timing", "Sequential completed fixture renders with synchronous diagnostic readback; frame steps are not game timestamps or performance measurements."},
      {"renderer", gpu.has_control ? "frozen_shader_override" : "embedded_shader"},
      {"ui_plane_mode", static_cast<std::uint32_t>(plane.mode)},
      {"inverse_depth_role", front_limit ? "unused" : "midpoint_floor"},
      {"shader_file", "temporal-ui-probe-shader.hlsl"},
      {"width", gpu.width}, {"height", gpu.height}, {"color_space", gpu.color},
      {"covered_pixels", covered}, {"covered_peak_xy", {px, py}},
      {"uncovered_nearest_q", .95f}, {"midpoint_floor_q", plane.inverse_depth},
      {"fixed_inputs", "Current RGB, positive-alpha coverage, projection A/inverseB, crop, jitter, strength, strength blend and UI floor."},
      {"frames", nlohmann::json::array()}, {"scenarios", nlohmann::json::array()}};
    std::ofstream csv(directory / "temporal-ui-probe.csv");
    require(csv.good(), "cannot create temporal probe CSV");
    csv << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "scenario,frame,sequence,covered_nearest_q,midpoint_floor_q,gain_K,scene_zero_q,gpu_ui_q,gpu_ui_q_delta,ui_source_u,ui_per_eye_pixels,ui_step_pixels,ui_pair_disparity_pixels,candidate_at_ui_pixels,display_budget_pixels,field_nonfinite,field_capped,covered_nonuniform\n";
    unsigned sequence{};
    for (unsigned scenario = 0; scenario != scenarios.size(); ++scenario) {
      const unsigned frames = scenario == 0 ? 8u : 12u;
      double previous_ui{}, previous_q{}, maximum_step{}, maximum_q_step{};
      double minimum_ui = std::numeric_limits<double>::infinity(), maximum_ui = -minimum_ui;
      for (unsigned frame = 0; frame != frames; ++frame) {
        auto parameters = base;
        const bool alternate = (frame & 1) != 0;
        raw[peak] = scenario == 1 ? (alternate ? .8f : .4f) : .6f;
        if ((scenario == 2 || scenario == 4) && alternate) parameters.depth_scale *= 2.f;
        if ((scenario == 3 || scenario == 4) && alternate) parameters.convergence[1] = .325f;
        gpu.context->UpdateSubresource(gpu.depth.Get(), 0, nullptr, raw.data(), gpu.width * sizeof(float), 0);
        const auto actual = gpu.render(true, 1, false, gpu.has_control, false, {}, {}, plane, &parameters);
        const auto diagnostics = active.diagnostics();
        float q{};
        if (front_limit) {
          require(!diagnostics.ui_plane_tiles.handle && !diagnostics.ui_plane_resolved.handle,
            "front-limit temporal probe exposed nearest-depth reduction resources");
        } else {
          const auto resolved = gpu.read(diagnostics.ui_plane_resolved);
          require(resolved.width == 1 && resolved.height == 1 && resolved.format == DXGI_FORMAT_R32_FLOAT,
            "temporal probe did not receive the actual single-plane GPU scalar");
          q = resolved.channel(0, 0, 0);
        }
        const float field = actual.field.channel(px, py, 0);
        const double ui_pixels = double(field) * gpu.width;
        const auto candidate = gpu.read(diagnostics.candidate);
        const double candidate_pixels = double(candidate.channel(px, py, 0)) * gpu.width;
        const float budget = parameters.disparity_limit_uv * parameters.strength * .01f * parameters.strength_blend;
        size_t nonfinite{}, capped{}, nonuniform{};
        float field_min = std::numeric_limits<float>::infinity(), field_max = -field_min;
        for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x) {
          const auto value = actual.field.channel(x, y, 0);
          if (!std::isfinite(value)) ++nonfinite;
          else {
            field_min = std::min(field_min, value); field_max = std::max(field_max, value);
            if (std::abs(value) >= budget) ++capped;
          }
          if (alpha[size_t(y) * gpu.width + x] > 0 && value != field) ++nonuniform;
        }
        require(std::isfinite(q) && std::isfinite(field) && !nonfinite,
          "temporal probe produced nonfinite GPU evidence");
        const double delta = frame ? ui_pixels - previous_ui : 0.;
        const double q_delta = frame ? double(q) - previous_q : 0.;
        maximum_step = std::max(maximum_step, std::abs(delta));
        maximum_q_step = std::max(maximum_q_step, std::abs(q_delta));
        minimum_ui = std::min(minimum_ui, ui_pixels); maximum_ui = std::max(maximum_ui, ui_pixels);
        previous_ui = ui_pixels; previous_q = q;
        std::uint32_t q_bits{}; std::memcpy(&q_bits, &q, sizeof(q));
        nlohmann::json row{{"scenario", scenarios[scenario]}, {"frame", frame}, {"sequence", sequence++},
          {"ui_plane_mode", static_cast<std::uint32_t>(plane.mode)},
          {"covered_nearest_q", raw[peak]}, {"midpoint_floor_q", plane.inverse_depth},
          {"gain_K", parameters.depth_scale}, {"scene_zero_q", parameters.convergence[1]},
          {"gpu_ui_q", front_limit ? nlohmann::json(nullptr) : nlohmann::json(q)},
          {"gpu_ui_q_bits", front_limit ? nlohmann::json(nullptr) : nlohmann::json(q_bits)},
          {"gpu_ui_q_delta", frame && !front_limit ? nlohmann::json(q_delta) : nlohmann::json(nullptr)},
          {"ui_source_u", field}, {"ui_per_eye_pixels", ui_pixels},
          {"ui_step_pixels", frame ? nlohmann::json(delta) : nlohmann::json(nullptr)},
          {"ui_pair_disparity_pixels", 2 * ui_pixels}, {"candidate_at_ui_pixels", candidate_pixels},
          {"display_budget_pixels", double(budget) * gpu.width},
          {"field_min_per_eye_pixels", double(field_min) * gpu.width},
          {"field_max_per_eye_pixels", double(field_max) * gpu.width},
          {"field_nonfinite", nonfinite}, {"field_capped", capped}, {"covered_nonuniform", nonuniform}};
        document["frames"].push_back(row);
        // One JSON object per frame also makes a redirected stdout log useful.
        std::printf("%s\n", row.dump().c_str());
        csv << scenarios[scenario] << ',' << frame << ',' << sequence - 1 << ',' << raw[peak] << ','
          << plane.inverse_depth << ',' << parameters.depth_scale << ',' << parameters.convergence[1] << ',';
        if (!front_limit) csv << q;
        csv << ',';
        if (frame && !front_limit) csv << q_delta;
        csv << ',' << field << ',' << ui_pixels << ',';
        if (frame) csv << delta;
        csv << ',' << 2 * ui_pixels << ',' << candidate_pixels << ',' << double(budget) * gpu.width << ','
          << nonfinite << ',' << capped << ',' << nonuniform << '\n';
      }
      document["scenarios"].push_back({{"scenario", scenarios[scenario]}, {"frames", frames},
        {"maximum_absolute_ui_step_pixels", maximum_step},
        {"maximum_absolute_q_step", front_limit ? nlohmann::json(nullptr) : nlohmann::json(maximum_q_step)},
        {"ui_min_per_eye_pixels", minimum_ui}, {"ui_max_per_eye_pixels", maximum_ui}});
    }
    document["status"] = "diagnostic_complete";
    std::ofstream json(directory / "temporal-ui-probe.json");
    json << document.dump(2) << '\n';
    require(csv.good() && json.good(), "cannot write temporal UI probe evidence");
    std::printf("DIAGNOSTIC COMPLETE native GPU temporal UI probe: %u frames; no temporal stability assertion\n", sequence);
  }
  double translated_color(const image &source, double x, unsigned y, unsigned channel) {
    x = std::clamp(x, 0., double(source.width - 1));
    const unsigned left = static_cast<unsigned>(std::floor(x));
    const unsigned right = std::min(left + 1, source.width - 1);
    return source.channel(left, y, channel) * (1. - (x - left)) + source.channel(right, y, channel) * (x - left);
  }
  void verify_independent_ui_plane(fixture &gpu, std::ostream &report, const fs::path &dump_directory) {
    using sunshine_game3d::ui_plane_mode;
    using sunshine_game3d::ui_plane_parameters;
    sunshine_game3d::render_parameters p;
    p.strength = 100; p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 1;
    p.depth_scale = 432.f / gpu.height; p.strength_blend = 1;
    p.projection = {0, 1}; p.convergence = {.05f, 1.f}; p.disparity_limit_uv = .04f;
    const auto render = [&](bool protect, const ui_plane_parameters &plane,
        const sunshine_game3d::render_parameters &parameters, bool control = false,
        api::resource_view alpha_source = {}, const fs::path &dump = {}) {
      return gpu.render(protect, 1, false, control, alpha_source.handle != 0, alpha_source, dump, plane, &parameters);
    };
    const double color_tolerance = gpu.color == 2 ? .0021 : 1.01 / 1023;
    // The analytic translated image also crosses the hardware linear filter:
    // allow one 8-bit subtexel step and two float UV rounding steps, scaled by
    // the fixture's largest channel edge. Existing zero-plane tests retain
    // their tighter export-quantization-only tolerance below.
    const double translated_tolerance = color_tolerance + (gpu.color == 2 ? 2. : 1.) *
      (1. / 256 + 2 * float_spacing(1.) * gpu.width);
    std::vector<float> alpha(size_t(gpu.width) * gpu.height, 1.f);
    for (float inverse : {0.f, 2.f}) {
      const ui_plane_parameters plane{ui_plane_mode::depth_midpoint, inverse};
      gpu.pattern(alpha, true);
      const auto off = render(false, {}, p);
      const auto off_plane = render(false, plane, p);
      require(off.field.bytes == off_plane.field.bytes && off.output.bytes == off_plane.output.bytes,
        "UI depth changed output with protection disabled");
      const auto legacy = render(true, {}, p);
      if (gpu.has_control) {
        const auto frozen = render(true, {}, p, true);
        require(legacy.field.bytes == frozen.field.bytes && legacy.output.bytes == frozen.output.bytes,
          "legacy screen-plane UI changed frozen output");
      }
      const auto scene_candidate = gpu.read(gpu.renderer.diagnostics().candidate).bytes;
      const auto scene_vertical = gpu.read(gpu.renderer.diagnostics().vertical_field).bytes;
      const image source{gpu.original, gpu.width, gpu.height,
        gpu.color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM};
      // Full white and gray are real planar UI; their color must translate as
      // one rigid layer, including both signs, fractional shifts and clipping.
      for (unsigned variant = 0; variant != 4; ++variant) {
        auto parameters = p;
        if (variant == 1) { parameters.strength = 50; parameters.strength_blend = .5f; }
        if (variant == 2) parameters.disparity_limit_uv = .0001f;
        if (variant == 3) { parameters.depth_scale *= 100; parameters.disparity_limit_uv = .001f; }
        std::fill(alpha.begin(), alpha.end(), variant == 1 ? .25f : 1.f);
        gpu.pattern(alpha, true);
        const auto actual = render(true, plane, parameters);
        const double expected = plane_parallax(gpu, parameters, plane);
        const double observed = actual.field.channel(0, 0, 0);
        require(std::abs(observed - expected) <= 8 * float_spacing(expected), "independent UI plane has wrong signed projection");
        const double limit = parameters.disparity_limit_uv * double(parameters.strength) * .01 * parameters.strength_blend;
        require(std::abs(observed) <= limit + float_spacing(limit), "UI plane exceeds per-eye display budget");
        double color_error_max{};
        for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
          require(actual.field.channel(x, y, 0) == observed, "whole-image UI did not retain one exact plane");
          for (unsigned eye = 0; eye != 2; ++eye) for (unsigned channel = 0; channel != 3; ++channel) {
            const double sample_x = x + (eye ? 1. : -1.) * observed * gpu.width;
            color_error_max = std::max(color_error_max, std::abs(double(actual.output.channel(x + eye * gpu.width, y, channel)) -
              translated_color(source, sample_x, y, channel)));
          }
        }
        require(color_error_max <= translated_tolerance, "constant UI plane did not rigidly translate current RGB");
        if (!variant) {
          require(gpu.read(gpu.renderer.diagnostics().candidate).bytes == scene_candidate &&
              gpu.read(gpu.renderer.diagnostics().vertical_field).bytes == scene_vertical,
            "UI plane changed scene candidate or vertical conditioning");
          if (gpu.has_control && gpu.control_renderer.active_shader_source().find("Sunshine_UIPlaneMode") != std::string_view::npos) {
            const auto frozen = render(true, plane, parameters, true);
            require(actual.field.bytes == frozen.field.bytes && actual.output.bytes == frozen.output.bytes,
              "fixed midpoint mode changed frozen field or RGB bits");
          }
        }
        report << "UI-plane inverse=" << inverse << " variant=" << variant << " per_eye_px=" << observed * gpu.width
          << " rigid_RGB_error=" << color_error_max << std::endl;
      }
      std::fill(alpha.begin(), alpha.end(), 0.f); gpu.pattern(alpha, true);
      const auto empty = render(true, plane, p);
      const auto empty_off = render(false, {}, p);
      require(empty.field.bytes == empty_off.field.bytes && empty.output.bytes == empty_off.output.bytes,
        "all-black alpha changed stereo at a nonzero UI plane");

      // A partial layer must keep the same shape at its projected position.
      // Only its red marker is compared outside UI; the underlying green
      // scene is intentionally free to follow the conditioned scene field.
      for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
        for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) alpha[size_t(y) * gpu.width + x] = .25f;
      alpha[size_t(gpu.height / 8) * gpu.width + gpu.width / 4] = 1.f;
      gpu.pattern(alpha);
      const image partial_source{gpu.original, gpu.width, gpu.height, source.format};
      const auto unprotected = render(false, {}, p);
      const auto partial = render(true, plane, p, false, {}, inverse == 2.f ? dump_directory : fs::path{});
      const double expected = plane_parallax(gpu, p, plane);
      const double anchor = partial.field.channel(gpu.width / 3, gpu.height / 4, 0);
      require(std::abs(anchor - expected) <= 8 * float_spacing(expected), "partial UI has the wrong plane");
      double red_error{};
      for (unsigned y = 0; y < gpu.height; ++y) {
        std::vector<unsigned> distance(gpu.width, gpu.width);
        int nearest = -int(gpu.width);
        for (unsigned x = 0; x < gpu.width; ++x) {
          if (alpha[size_t(y) * gpu.width + x] > 0) nearest = int(x);
          distance[x] = std::min(gpu.width, unsigned(int(x) - nearest));
        }
        nearest = 2 * int(gpu.width);
        for (int x = int(gpu.width) - 1; x >= 0; --x) {
          if (alpha[size_t(y) * gpu.width + unsigned(x)] > 0) nearest = x;
          distance[unsigned(x)] = std::min(distance[unsigned(x)], unsigned(nearest - x));
        }
        for (unsigned x = 0; x < gpu.width; ++x) {
          const double value = partial.field.channel(x, y, 0);
          if (distance[x] == gpu.width) {
            require(std::memcmp(partial.field.bytes.data() + (size_t(y) * gpu.width + x) * 4,
              unprotected.field.bytes.data() + (size_t(y) * gpu.width + x) * 4, 4) == 0,
              "blank UI row changed scene field bits");
          } else {
            const double radius = .5 * std::max(int(distance[x]) - 1, 0) / gpu.width;
            const double tolerance = 4 * (float_spacing(value) + float_spacing(anchor) + float_spacing(radius));
            require(std::abs(value - anchor) <= radius + tolerance, "shifted UI collar exceeded its distance bound");
            if (distance[x] <= 1) require(value == anchor, "UI and bilinear collar were not pinned exactly");
          }
          if (x) {
            const double previous = partial.field.channel(x - 1, y, 0);
            require(std::abs(value - previous) * gpu.width <= .5 +
                4 * (float_spacing(value) + float_spacing(previous)) * gpu.width,
              "shifted UI field exceeded horizontal invertibility bound");
          }
          for (unsigned eye = 0; eye != 2; ++eye)
            red_error = std::max(red_error, std::abs(double(partial.output.channel(x + eye * gpu.width, y, 0)) -
              translated_color(partial_source, x + (eye ? 1. : -1.) * anchor * gpu.width, y, 0)));
        }
      }
      require(red_error <= translated_tolerance, "partial UI shape changed or left a ghost outside translated support");
      const auto external = gpu.retain_mask(alpha);
      gpu.pattern(alpha, true);
      const auto direct = render(true, plane, p);
      std::vector<float> wrong_alpha(alpha.size(), 1.f); gpu.pattern(wrong_alpha, true);
      const auto retained = render(true, plane, p, false, external);
      require(direct.field.bytes == retained.field.bytes && direct.output.bytes == retained.output.bytes,
        "nonzero UI plane changed retained-alpha selection or current RGB");
      report << "UI-plane inverse=" << inverse << " partial_per_eye_px=" << anchor * gpu.width
        << " translated_red_error=" << red_error << " retained_alpha_exact=1" << std::endl;
      std::fill(alpha.begin(), alpha.end(), 1.f);
    }
    gpu.pattern(alpha, true);
    const auto legacy = render(true, {}, p);
    for (const auto &invalid : std::array<ui_plane_parameters, 5>{{
      {static_cast<ui_plane_mode>(4), 1.f}, {ui_plane_mode::depth_midpoint, -1.f},
      {ui_plane_mode::depth_midpoint, std::numeric_limits<float>::infinity()},
      {ui_plane_mode::depth_midpoint, std::numeric_limits<float>::quiet_NaN()},
      {ui_plane_mode::screen, std::numeric_limits<float>::quiet_NaN()}}}) {
      const auto actual = render(true, invalid, p);
      require(actual.field.bytes == legacy.field.bytes && actual.output.bytes == legacy.output.bytes,
        "invalid UI-plane metadata did not preserve legacy screen pinning");
    }
    for (unsigned condition = 0; condition != 3; ++condition) {
      auto mono = p;
      if (!condition) mono.strength = 0;
      if (condition == 1) mono.depth_ready = 0;
      if (condition == 2) mono.camera_ready = 0;
      const auto actual = render(true, {ui_plane_mode::depth_midpoint, 2.f}, mono);
      const auto reference = render(true, {}, mono);
      require(actual.field.bytes == reference.field.bytes && actual.output.bytes == reference.output.bytes,
        "independent UI plane changed mono fallback");
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x)
        require(actual.field.channel(x, y, 0) == 0.f, "mono fallback retained UI disparity");
    }
    std::puts("PASS independent UI plane: both signs, rigid RGB, collar, budget, legacy/disabled parity, retained alpha and invalid/mono fallback");
  }

  void verify_nearest_ui_plane(fixture &gpu, std::ostream &report, const fs::path &dump_directory) {
    using sunshine_game3d::ui_plane_mode;
    using sunshine_game3d::ui_plane_parameters;
    const auto count = size_t(gpu.width) * gpu.height;
    const auto at = [&](unsigned x, unsigned y) { return size_t(y) * gpu.width + x; };
    const unsigned ax = gpu.width / 3, ay = gpu.height / 3;
    const unsigned bx = gpu.width * 2 / 3, by = gpu.height * 2 / 3;
    const unsigned cx = gpu.width - 1, cy = gpu.height - 1;
    std::vector<float> alpha(count, 0.f), q(count, .125f);
    alpha[at(ax, ay)] = .25f; alpha[at(bx, by)] = 1.f; alpha[at(cx, cy)] = 1.f / 64;
    sunshine_game3d::render_parameters p;
    p.strength = 100; p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 0;
    p.depth_scale = 432.f / gpu.height; p.strength_blend = 1;
    p.projection = {0, 1}; p.convergence = {.05f, .125f}; p.disparity_limit_uv = .04f;
    ui_plane_parameters plane{ui_plane_mode::depth_midpoint_nearest_ui, .25f};
    const auto upload = [&](const std::vector<float> &values, bool reverse) {
      auto raw = values;
      if (reverse) for (auto &value : raw) value = 1.f - value;
      gpu.context->UpdateSubresource(gpu.depth.Get(), 0, nullptr, raw.data(), gpu.width * sizeof(float), 0);
    };
    const auto render = [&](bool protect, const ui_plane_parameters &selected,
        const sunshine_game3d::render_parameters &parameters, api::resource_view external = {},
        const fs::path &dump = {}, bool control = false) {
      return gpu.render(protect, 1, false, control, external.handle != 0, external, dump, selected, &parameters);
    };
    const auto resolved = [&] {
      const auto resources = gpu.renderer.diagnostics();
      require(resources.ui_plane_tiles.handle && resources.ui_plane_resolved.handle,
        "current render omitted nearest-UI GPU evidence");
      const auto scalar = gpu.read(resources.ui_plane_resolved);
      require(scalar.width == 1 && scalar.height == 1 && scalar.format == DXGI_FORMAT_R32_FLOAT,
        "nearest-UI result is not one global float32 plane");
      const auto tiles = gpu.read(resources.ui_plane_tiles);
      require(tiles.width == (gpu.width + 15) / 16 && tiles.height == (gpu.height + 15) / 16,
        "nearest-UI tile reduction dimensions are incomplete");
      return scalar.channel(0, 0, 0);
    };
    const auto pinned = [&](const result &value, float inverse, const sunshine_game3d::render_parameters &parameters,
        const std::vector<float> &coverage) {
      const double expected = plane_parallax(gpu, parameters, {ui_plane_mode::depth_midpoint, inverse});
      bool seen = false; float first{};
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        if (!std::isfinite(coverage[at(x, y)]) || coverage[at(x, y)] <= 0.f) continue;
        const float actual = value.field.channel(x, y, 0);
        require(std::abs(double(actual) - expected) <= 8 * float_spacing(expected),
          "UI pixels were not pinned to the resolved global near plane");
        if (seen) require(actual == first, "separate UI rows received different planes");
        else { first = actual; seen = true; }
      }
      return first;
    };
    if (gpu.has_control && gpu.control_renderer.active_shader_source().find("#define SUNSHINE_UI_NEAREST_PLANE 1") == std::string_view::npos)
      require(!gpu.control_renderer.render(observed_runtime->get_command_queue()->get_immediate_command_list(),
          {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())},
          {reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())}, p, true, {}, plane),
        "historical shader silently accepted an unsupported nearest-UI plane");

    for (bool reverse : {false, true}) {
      p.projection = reverse ? std::array<float, 2>{1, -1} : std::array<float, 2>{0, 1};
      q.assign(count, .125f); q[0] = .9375f;
      q[at(ax, ay)] = .5f; q[at(bx, by)] = .75f; q[at(cx, cy)] = .625f;
      upload(q, reverse); gpu.pattern(alpha, true);
      const auto off = render(false, {}, p);
      const auto off_new = render(false, plane, p);
      require(off.field.bytes == off_new.field.bytes && off.output.bytes == off_new.output.bytes &&
          !gpu.renderer.diagnostics().ui_plane_resolved.handle,
        "disabled UI changed output or exposed an obsolete resolved plane");
      const auto candidate = gpu.read(gpu.renderer.diagnostics().candidate).bytes;
      const auto vertical = gpu.read(gpu.renderer.diagnostics().vertical_field).bytes;
      const auto actual = render(true, plane, p, {}, reverse ? fs::path{} : dump_directory);
      require(resolved() == .75f, "nearest UI plane included closer uncovered depth or missed covered depth");
      const float applied = pinned(actual, .75f, p, alpha);
      require(gpu.read(gpu.renderer.diagnostics().candidate).bytes == candidate &&
          gpu.read(gpu.renderer.diagnostics().vertical_field).bytes == vertical,
        "nearest-UI reduction changed scene geometry before UI pinning");
      for (unsigned x = 0; x < gpu.width; ++x)
        require(actual.field.channel(x, 0, 0) == off.field.channel(x, 0, 0),
          "nearest-UI plane changed an unmasked row");
      if (gpu.has_control) {
        const auto frozen = render(true, {ui_plane_mode::depth_midpoint, plane.inverse_depth}, p, {}, {}, true);
        const float prior = frozen.field.channel(ax, ay, 0);
        require(prior < applied && applied - prior > .1f / gpu.width,
          "frozen midpoint control did not demonstrate the covered-foreground counterexample");
        report << "nearest-UI control=midpoint reverse=" << reverse << " prior_px=" << prior * gpu.width
          << " required_px=" << applied * gpu.width << " old_requirement_failed=1" << std::endl;
      }
      // Lower the previously closest covered pixel. A maximum from another
      // render must not survive, even though nearer unmasked geometry remains.
      q[at(bx, by)] = .375f; upload(q, reverse);
      const auto lower = render(true, plane, p);
      require(resolved() == .625f, "nearest-UI maximum retained the previous render");
      pinned(lower, .625f, p, alpha);
      // One faint covered corner texel is enough, including a partial edge tile.
      q[at(cx, cy)] = .875f; upload(q, reverse);
      const auto outlier = render(true, plane, p);
      require(resolved() == .875f, "single soft-alpha corner pixel was missed");
      pinned(outlier, .875f, p, alpha);
      q[at(ax, ay)] = q[at(bx, by)] = q[at(cx, cy)] = .125f; upload(q, reverse);
      const auto floor = render(true, plane, p);
      require(resolved() == plane.inverse_depth, "nearest-UI plane moved behind the midpoint floor");
      pinned(floor, plane.inverse_depth, p, alpha);

      std::vector<float> white(count, 1.f); gpu.pattern(white, true);
      const auto full = render(true, plane, p);
      require(resolved() == .9375f, "all-white UI failed to include the nearest depth");
      pinned(full, .9375f, p, white);
      std::vector<float> black(count, 0.f); gpu.pattern(black, true);
      const auto empty = render(true, plane, p);
      require(resolved() == plane.inverse_depth, "all-black coverage reused an old foreground maximum");
      const auto empty_off = render(false, {}, p);
      require(empty.field.bytes == empty_off.field.bytes && empty.output.bytes == empty_off.output.bytes,
        "all-black UI changed scene output in nearest mode");

      // An independent retained mask must control the reduction and pinning;
      // current output alpha and the old texture's RGB have no authority here.
      q[at(ax, ay)] = .5f; q[at(bx, by)] = .75f; q[at(cx, cy)] = .625f; upload(q, reverse);
      const auto external = gpu.retain_mask(alpha);
      gpu.pattern(alpha, true); const auto direct = render(true, plane, p);
      gpu.pattern(white, true); const auto retained = render(true, plane, p, external);
      require(resolved() == .75f && retained.field.bytes == direct.field.bytes && retained.output.bytes == direct.output.bytes,
        "nearest-UI reduction used current alpha or retained RGB instead of selected coverage");

      // With a cropped allocation and nonzero jitter, the selected source pixel
      // maps to this exact interior texel. A closer unjittered/padding texel must
      // not become the UI plane. All quantities here are exactly representable.
      auto cropped = p; cropped.depth_rect = {.25f, .25f, .5f, .5f};
      cropped.jitter = {1.f / gpu.width, -1.f / gpu.height};
      std::vector<float> point(count, 0.f); point[at(gpu.width / 2, gpu.height / 2)] = .25f;
      q.assign(count, .125f); q[0] = .9375f; q[at(gpu.width / 2, gpu.height / 2)] = .9375f;
      q[at(gpu.width / 2 + 1, gpu.height / 2 - 1)] = .75f;
      upload(q, reverse); gpu.pattern(point, true);
      const auto shifted = render(true, plane, cropped);
      require(resolved() == .75f, "nearest-UI reduction ignored depth crop or projection jitter");
      pinned(shifted, .75f, cropped, point);
      report << "nearest-UI reverse=" << reverse
        << " covered_max_exact=1 uncovered_ignored=1 floor=1 current_frame=1 soft_corner=1 all_white_black=1 retained_alpha=1 crop_jitter=1" << std::endl;
    }

    // Nonfinite/negative decoded depths do not invent a near plane. Positive
    // alpha still pins at the valid midpoint floor when no covered q is valid.
    p.projection = {0, 1};
    q.assign(count, .125f); q[at(ax, ay)] = std::numeric_limits<float>::quiet_NaN();
    q[at(bx, by)] = std::numeric_limits<float>::infinity(); q[at(cx, cy)] = -1.f;
    upload(q, false); gpu.pattern(alpha, true);
    const auto invalid_q = render(true, plane, p);
    require(resolved() == plane.inverse_depth, "invalid covered depth became the nearest plane");
    pinned(invalid_q, plane.inverse_depth, p, alpha);
    if (gpu.color == 2) {
      auto invalid_alpha = alpha;
      invalid_alpha[at(ax, ay)] = std::numeric_limits<float>::quiet_NaN();
      invalid_alpha[at(bx, by)] = std::numeric_limits<float>::infinity(); invalid_alpha[at(cx, cy)] = -1.f;
      q.assign(count, .875f); upload(q, false); gpu.pattern(invalid_alpha, true);
      const auto no_coverage = render(true, plane, p);
      require(resolved() == plane.inverse_depth, "nonfinite or negative alpha entered the maximum");
      const auto reference = render(false, {}, p);
      require(no_coverage.field.bytes == reference.field.bytes && no_coverage.output.bytes == reference.output.bytes,
        "invalid alpha changed nearest-mode output");
    }
    q.assign(count, .75f); upload(q, false); gpu.pattern(alpha, true);
    const auto legacy = render(true, {}, p);
    for (float bad_floor : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      const auto bad = render(true, {ui_plane_mode::depth_midpoint_nearest_ui, bad_floor}, p);
      require(bad.field.bytes == legacy.field.bytes && bad.output.bytes == legacy.output.bytes && resolved() == 0.f,
        "invalid nearest-UI floor did not preserve screen-plane fallback");
    }
    for (unsigned condition = 0; condition != 4; ++condition) {
      auto mono = p;
      if (condition == 0) mono.strength = 0;
      if (condition == 1) mono.depth_ready = 0;
      if (condition == 2) mono.camera_ready = 0;
      if (condition == 3) mono.projection[1] = 0;
      const auto inactive = render(true, plane, mono);
      require(resolved() == 0.f, "inactive stereo retained a prior resolved UI plane");
      const auto reference = render(true, {}, mono);
      require(inactive.field.bytes == reference.field.bytes && inactive.output.bytes == reference.output.bytes,
        "nearest-UI mode changed mono or invalid projection fallback");
    }
    // Leave the longstanding source-alpha fixtures' original depth untouched.
    q.assign(count, .02f); upload(q, false);
    std::puts("PASS nearest-UI global plane: exact current-frame maximum/floor, both depth directions, selected alpha, crop/jitter, invalid input and legacy/mono compatibility");
  }
  void verify_front_limit_ui_plane(fixture &gpu, std::ostream &report, const fs::path &dump_directory) {
    using sunshine_game3d::ui_plane_mode;
    using sunshine_game3d::ui_plane_parameters;
    const size_t count = size_t(gpu.width) * gpu.height;
    const unsigned px = gpu.width / 2, py = gpu.height / 2;
    const size_t peak = size_t(py) * gpu.width + px;
    std::vector<float> alpha(count, 0.f), raw(count, .125f);
    for (unsigned y = gpu.height / 3; y < gpu.height * 2 / 3; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width * 2 / 3; ++x)
        alpha[size_t(y) * gpu.width + x] = .25f;
    alpha[peak] = .25f;
    sunshine_game3d::render_parameters p;
    p.strength = 100; p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 0;
    p.depth_scale = 432.f / gpu.height; p.strength_blend = 1;
    p.projection = {0, 1}; p.convergence = {.05f, .125f}; p.disparity_limit_uv = .04f;
    const ui_plane_parameters front{ui_plane_mode::front_limit, 0.f};
    const auto upload = [&] { gpu.context->UpdateSubresource(gpu.depth.Get(), 0, nullptr, raw.data(), gpu.width * sizeof(float), 0); };
    const auto render = [&](bool protect, const ui_plane_parameters &plane,
        const sunshine_game3d::render_parameters &parameters, bool control = false,
        api::resource_view external = {}, const fs::path &dump = {}) {
      return gpu.render(protect, 1, false, control, external.handle != 0, external, dump, plane, &parameters);
    };
    const auto no_reduction = [&] {
      require(!gpu.renderer.diagnostics().ui_plane_tiles.handle && !gpu.renderer.diagnostics().ui_plane_resolved.handle,
        "front-limit UI exposed a current nearest-depth reduction");
    };
    const auto budget = [](const sunshine_game3d::render_parameters &parameters) {
      const float strength = std::clamp(parameters.strength, 0.f, 100.f) * .01f * std::clamp(parameters.strength_blend, 0.f, 1.f);
      return parameters.disparity_limit_uv * strength;
    };
    const auto pinned = [&](const result &actual, const sunshine_game3d::render_parameters &parameters) {
      const float expected = budget(parameters);
      const float observed = actual.field.channel(px, py, 0);
      require(std::abs(double(observed) - expected) <= 2 * float_spacing(expected), "front-limit UI does not use the positive current display budget");
      for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x) {
        const float value = actual.field.channel(x, y, 0);
        require(std::isfinite(value) && std::abs(value) <= expected + float_spacing(expected), "front-limit field exceeded the current finite display budget");
        if (alpha[size_t(y) * gpu.width + x] > 0) require(value == observed, "front-limit UI did not share one global plane");
      }
      no_reduction();
      return observed;
    };
    if (gpu.has_control && gpu.control_renderer.active_shader_source().find("#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1") == std::string_view::npos)
      require(!gpu.control_renderer.render(observed_runtime->get_command_queue()->get_immediate_command_list(),
          {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())}, {reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())}, p, true, {}, front),
        "historical shader silently accepted the unsupported front-limit UI plane");
    gpu.pattern(alpha, true);
    float first{}, mode2_min = std::numeric_limits<float>::infinity(), mode2_max = -mode2_min;
    std::vector<unsigned char> first_candidate;
    bool candidate_changed = false;
    for (unsigned frame = 0; frame != 12; ++frame) {
      auto parameters = p;
      raw[peak] = frame & 1 ? .8f : .4f;
      raw.front() = .95f;
      parameters.depth_scale *= frame % 3 == 0 ? .5f : frame % 3 == 1 ? 8.f : 2.f;
      parameters.convergence[1] = frame % 4 < 2 ? .025f : .375f;
      upload();
      const auto nearest = render(true, {ui_plane_mode::depth_midpoint_nearest_ui, .25f}, parameters);
      const auto candidate = gpu.read(gpu.renderer.diagnostics().candidate).bytes;
      const auto vertical = gpu.read(gpu.renderer.diagnostics().vertical_field).bytes;
      if (!frame) first_candidate = candidate;
      else candidate_changed |= candidate != first_candidate;
      const float prior = nearest.field.channel(px, py, 0);
      mode2_min = std::min(mode2_min, prior); mode2_max = std::max(mode2_max, prior);
      if (!frame && gpu.has_control && gpu.control_renderer.active_shader_source().find("#define SUNSHINE_UI_NEAREST_PLANE 1") != std::string_view::npos) {
        const auto frozen = render(true, {ui_plane_mode::depth_midpoint_nearest_ui, .25f}, parameters, true);
        require(frozen.field.bytes == nearest.field.bytes && frozen.output.bytes == nearest.output.bytes,
          "new front-limit support changed frozen nearest-mode field or RGB bits");
      }
      const auto actual = render(true, front, parameters, false, {}, !frame ? dump_directory : fs::path{});
      const float observed = pinned(actual, parameters);
      if (!frame) first = observed;
      else require(observed == first, "front-limit UI moved with scene depth, gain or zero");
      require(gpu.read(gpu.renderer.diagnostics().candidate).bytes == candidate && gpu.read(gpu.renderer.diagnostics().vertical_field).bytes == vertical,
        "front-limit UI changed scene candidate or vertical conditioning");
      for (unsigned x = 0; x != gpu.width; ++x)
        require(actual.field.channel(x, 0, 0) == nearest.field.channel(x, 0, 0), "front-limit UI changed an unmasked row");
      report << "front-limit temporal frame=" << frame << " raw_q=" << raw[peak] << " gain=" << parameters.depth_scale
        << " zero=" << parameters.convergence[1] << " nearest_px=" << prior * gpu.width << " front_px=" << observed * gpu.width << std::endl;
    }
    require(candidate_changed && mode2_max > mode2_min, "front-limit temporal fixture did not exercise changing scene geometry and nearest-mode UI");
    for (unsigned variant = 0; variant != 4; ++variant) {
      auto parameters = p;
      if (variant == 0) parameters.strength = 40;
      if (variant == 1) parameters.strength_blend = .25f;
      if (variant == 2) { parameters.strength = 150; parameters.strength_blend = 1.5f; }
      if (variant == 3) parameters.disparity_limit_uv = .006f;
      pinned(render(true, front, parameters), parameters);
    }
    const auto reference = render(true, front, p);
    for (float ignored : {.7f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      const auto actual = render(true, {ui_plane_mode::front_limit, ignored}, p);
      require(actual.field.bytes == reference.field.bytes && actual.output.bytes == reference.output.bytes,
        "front-limit UI consumed its explicitly unused inverse-depth word");
    }
    const auto off = render(false, {}, p), front_off = render(false, front, p);
    require(off.field.bytes == front_off.field.bytes && off.output.bytes == front_off.output.bytes, "disabled front-limit UI changed scene output");
    if (gpu.has_control) {
      const auto frozen_off = render(false, front, p, true);
      require(frozen_off.field.bytes == off.field.bytes && frozen_off.output.bytes == off.output.bytes, "disabled front-limit UI changed frozen shader output");
    }
    // Selected retained alpha still supplies coverage only; current RGB wins.
    const auto external = gpu.retain_mask(alpha);
    gpu.pattern(alpha, true, 3); const auto direct = render(true, front, p);
    std::vector<float> white(count, 1.f);
    gpu.pattern(white, true, 3); const auto retained = render(true, front, p, false, external);
    require(retained.field.bytes == direct.field.bytes && retained.output.bytes == direct.output.bytes,
      "front-limit UI consumed retained RGB or current alpha instead of the selected mask");
    std::fill(alpha.begin(), alpha.end(), 0.f); gpu.pattern(alpha, true);
    const auto empty = render(true, front, p), empty_off = render(false, {}, p);
    require(empty.field.bytes == empty_off.field.bytes && empty.output.bytes == empty_off.output.bytes, "front-limit all-black mask changed stereo");
    std::fill(alpha.begin(), alpha.end(), 1.f); gpu.pattern(alpha, true);
    const image source{gpu.original, gpu.width, gpu.height, gpu.color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM};
    const auto whole = render(true, front, p);
    const float whole_plane = pinned(whole, p);
    double rgb_error{};
    for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x)
      for (unsigned eye = 0; eye != 2; ++eye) for (unsigned channel = 0; channel != 3; ++channel)
        rgb_error = std::max(rgb_error, std::abs(double(whole.output.channel(x + eye * gpu.width, y, channel)) -
          translated_color(source, x + (eye ? 1. : -1.) * whole_plane * gpu.width, y, channel)));
    const double tolerance = (gpu.color == 2 ? .0021 : 1.01 / 1023) + (gpu.color == 2 ? 2. : 1.) *
      (1. / 256 + 2 * float_spacing(1.) * gpu.width);
    require(rgb_error <= tolerance, "front-limit all-white plane did not rigidly translate current RGB");
    // This non-binary strength/blend combination distinguishes regrouped
    // limit*(strength*.01*blend) from the authoritative final scene clamp.
    // A uniform, saturated scene supplies the actual GPU bound as the oracle.
    auto fractional = p;
    fractional.depth_scale *= 1000.f;
    fractional.strength = 3.f; fractional.strength_blend = .7f; fractional.disparity_limit_uv = .02f;
    raw.assign(count, 1.f); upload();
    const auto saturated_scene = render(false, {}, fractional);
    const auto exact_front = render(true, front, fractional);
    no_reduction();
    require(saturated_scene.field.channel(0, 0, 0) > 0 && exact_front.field.bytes == saturated_scene.field.bytes,
      "front-limit float ordering differs from the authoritative final scene bound at .02/3/.7");
    std::uint32_t bound_bits{};
    const float actual_bound = saturated_scene.field.channel(0, 0, 0);
    std::memcpy(&bound_bits, &actual_bound, sizeof(bound_bits));
    report << "front-limit bound_rounding=exact limit=.02 strength=3 blend=.7 source_u_bits=" << bound_bits << std::endl;
    // Depth-plane modes must obey the same final bound in both directions.
    // The actual positive GPU bound and its exact sign flip define the
    // symmetric display interval. A negative uniform scene can round inward
    // during Q30 conditioning, so its field is not the negative-bound oracle.
    for (const float inverse_depth : {1.f, 0.f}) {
      raw.assign(count, inverse_depth); upload();
      const auto saturated = render(false, {}, fractional);
      const float scene_bound = saturated.field.channel(0, 0, 0);
      require(std::isfinite(scene_bound) && (inverse_depth > 0 ? scene_bound > 0 : scene_bound < 0),
        "depth-plane bound fixture did not produce the intended saturated scene sign");
      if (gpu.has_control) {
        const auto frozen_off = render(false, {}, fractional, true);
        require(frozen_off.field.bytes == saturated.field.bytes && frozen_off.output.bytes == saturated.output.bytes,
          "UI-plane final-bound change altered the unprotected scene or RGB");
      }
      for (const auto mode : {ui_plane_mode::depth_midpoint, ui_plane_mode::depth_midpoint_nearest_ui}) {
        // Mode 2 obtains the same q from the real uniform depth texture.
        const ui_plane_parameters plane{mode, mode == ui_plane_mode::depth_midpoint ? inverse_depth : 0.f};
        const auto actual = render(true, plane, fractional);
        const float expected = inverse_depth > 0 ? actual_bound : -actual_bound;
        const float ui_bound = actual.field.channel(0, 0, 0);
        std::uint32_t scene_bits{}, ui_bits{}, expected_bits{};
        std::memcpy(&scene_bits, &scene_bound, sizeof(scene_bits));
        std::memcpy(&ui_bits, &ui_bound, sizeof(ui_bits));
        std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
        report << "depth-plane bound_observation mode=" << static_cast<unsigned>(mode)
          << " q=" << inverse_depth << " scene_bits=" << scene_bits << " ui_bits=" << ui_bits
          << " expected_signed_cap_bits=" << expected_bits << std::endl;
        for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x) {
          const float value = actual.field.channel(x, y, 0);
          require(std::isfinite(value) && value == expected && std::abs(value) <= actual_bound,
            "depth-plane UI is not uniformly pinned to the authoritative signed display bound at .02/3/.7");
        }
        if (mode == ui_plane_mode::depth_midpoint) no_reduction();
        else require(gpu.renderer.diagnostics().ui_plane_tiles.handle && gpu.renderer.diagnostics().ui_plane_resolved.handle,
          "nearest depth-plane bound test did not execute the production GPU reduction");
        report << "depth-plane bound_rounding=exact mode=" << static_cast<unsigned>(mode)
          << " q=" << inverse_depth << " limit=.02 strength=3 blend=.7 source_u_bits=" << ui_bits << std::endl;
      }
    }
    for (unsigned condition = 0; condition != 10; ++condition) {
      auto invalid = p;
      if (condition == 0) invalid.strength = 0;
      if (condition == 1) invalid.strength_blend = 0;
      if (condition == 2) invalid.strength_blend = std::numeric_limits<float>::quiet_NaN();
      if (condition == 3) invalid.depth_ready = 0;
      if (condition == 4) invalid.camera_ready = 0;
      if (condition == 5) invalid.projection[1] = 0;
      if (condition == 6) invalid.depth_scale = 0;
      if (condition == 7) invalid.convergence[1] = std::numeric_limits<float>::quiet_NaN();
      if (condition == 8) invalid.disparity_limit_uv = .05f;
      if (condition == 9) invalid.disparity_limit_uv = -1.f;
      const auto actual = render(true, front, invalid);
      no_reduction();
      const auto screen = render(true, {}, invalid);
      require(actual.field.bytes == screen.field.bytes && actual.output.bytes == screen.output.bytes,
        "front-limit UI changed mono or existing camera-admission fallback");
    }
    raw.assign(count, .02f); upload();
    report << "front-limit stability=exact strength_blend_budget=1 unused_q=1 mono=1 no_reduction=1 current_RGB_error=" << rgb_error << std::endl;
    std::puts("PASS front-limit UI: fixed across depth/gain/zero, current strength/blend/budget, ignored q bits, selected alpha, current RGB and legacy/mono compatibility");
  }
  void verify_fg_transitions(fixture &gpu, std::ostream &report) {
    const size_t pixels = size_t(gpu.width) * gpu.height;
    const unsigned bpp = gpu.color == 2 ? 8 : 4, rgb_bytes = gpu.color == 2 ? 6 : 3;
    const double color_tolerance = gpu.color == 2 ? .0021 : 1.01 / 1023;
    std::vector<float> alpha(pixels, 0);
    struct frame { const char *label; unsigned mask; bool fg; };
    // Repeated return to FG catches stale b1 state after restoring protection.
    // These are controlled alpha states, not claims about which real/generated
    // presentation carries any particular alpha in a game.
    const std::array<frame, 8> frames{{
      {"clear-FG", 0, true}, {"partial-FG", 1, true}, {"opaque-FG", 2, true},
      {"opaque-FG-off", 2, false}, {"opaque-FG-resumed", 2, true},
      {"partial-FG-off", 1, false}, {"partial-FG-resumed", 1, true}, {"clear-FG-off", 0, false}
    }};
    for (int sign : {-1, 1}) {
      std::fill(alpha.begin(), alpha.end(), 0);
      gpu.pattern(alpha, true);
      const auto fixed_rgb = gpu.original;
      const auto flat = gpu.render(false, sign, true);
      const auto baseline = gpu.render(false, sign);
      if (gpu.has_control) {
        const auto frozen = gpu.render(false, sign, false, true);
        require(frozen.field.bytes == baseline.field.bytes && frozen.output.bytes == baseline.output.bytes,
          "FG regression baseline differs from frozen shader");
      }
      double original_shift{}, stereo_difference{}, mono_difference{};
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        original_shift = std::max(original_shift, std::abs(double(baseline.field.channel(x, y, 0))) * gpu.width);
        for (unsigned c = 0; c != 3; ++c)
          stereo_difference = std::max(stereo_difference, double(std::abs(baseline.output.channel(x, y, c) -
            baseline.output.channel(x + gpu.width, y, c))));
        mono_difference = std::max(mono_difference, color_error(baseline.output, flat.output, x, y));
      }
      require(original_shift >= .035 * gpu.width && stereo_difference > .1 && mono_difference > .1,
        "FG regression must start with visibly distinct stereo eyes and mono picture");
      for (const auto &frame : frames) {
        std::fill(alpha.begin(), alpha.end(), frame.mask == 2 ? 1.f : 0.f);
        if (frame.mask == 1) {
          for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
            for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) alpha[size_t(y) * gpu.width + x] = .25f;
        }
        gpu.pattern(alpha, true);
        for (size_t i = 0; i < pixels; ++i)
          require(std::memcmp(gpu.original.data() + i * bpp, fixed_rgb.data() + i * bpp, rgb_bytes) == 0,
            "FG regression changed RGB while varying alpha");
        const auto actual = gpu.render(true, sign, false, false, frame.fg);
        const bool exact_stereo = actual.field.bytes == baseline.field.bytes && actual.output.bytes == baseline.output.bytes;
        if (frame.fg || frame.mask == 0)
          require(exact_stereo, std::string(frame.label) + ": fixed scene lost/changed stereo when only alpha varied");
        size_t nonzero{}, pinned{};
        double ui_error{};
        for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
          const auto i = size_t(y) * gpu.width + x;
          const auto field = actual.field.channel(x, y, 0);
          require(std::isfinite(field), "FG transition produced nonfinite field");
          nonzero += field != 0;
          if (!frame.fg && alpha[i] > 0) {
            require(field == 0, "FG off did not restore selected UI protection");
            ++pinned;
            for (unsigned eye = 0; eye != 2; ++eye)
              ui_error = std::max(ui_error, color_error(actual.output, flat.output, x + eye * gpu.width, y));
          }
        }
        if (frame.fg) require(nonzero == pixels, "FG alpha change flattened any pixel of constant nonzero-depth scene");
        else if (frame.mask == 2) require(nonzero == 0 && pinned == pixels, "FG-off opaque UI did not become flat");
        else if (frame.mask == 1) require(pinned > 0 && nonzero > 0 && !exact_stereo,
          "FG-off partial mask must restore UI pins while leaving stereo outside");
        require(ui_error <= color_tolerance, "FG-off UI restoration changed protected RGB beyond export quantization");
        report << "FG-transition " << frame.label << " sign=" << sign << " requested=1 FG=" << frame.fg
          << " fixed_RGB=1 depth_unchanged=1 exact_off_stereo=" << exact_stereo << " nonzero_field_pixels=" << nonzero
          << " pinned_pixels=" << pinned << " UI_color_error=" << ui_error << std::endl;
      }
      // The user can also disable protection explicitly while FG is off.
      std::fill(alpha.begin(), alpha.end(), 1); gpu.pattern(alpha, true);
      const auto unchecked = gpu.render(false, sign);
      require(unchecked.field.bytes == baseline.field.bytes && unchecked.output.bytes == baseline.output.bytes,
        "Unchecked UI protection changed stereo for opaque alpha with FG off");
      std::printf("PASS FG alpha transitions sign=%d fixed RGB/depth stereo_difference=%.6f mono_difference=%.6f; "
        "FG on bit-exact off, FG off restores partial/opaque UI, unchecked stays stereo\n", sign, stereo_difference, mono_difference);
    }
  }
  void verify(fixture &gpu, const std::vector<float> &alpha, const std::string &label, std::ostream &report) {
    gpu.pattern(alpha);
    const auto flat = gpu.render(false, 1, true);
    const bool empty = std::none_of(alpha.begin(), alpha.end(), [](float a) { return std::isfinite(a) && a > 0; });
    const double color_tolerance = gpu.color == 2 ? .0021 : 1.01 / 1023;
    // Only a distance-bound oracle, never a reproduction of the GPU warp.
    std::vector<unsigned> distances(alpha.size(), gpu.width);
    for (unsigned y = 0; y < gpu.height; ++y) {
      int last = -int(gpu.width);
      for (unsigned x = 0; x < gpu.width; ++x) {
        const auto i = size_t(y) * gpu.width + x;
        if (std::isfinite(alpha[i]) && alpha[i] > 0) last = int(x);
        distances[i] = std::min(gpu.width, unsigned(int(x) - last));
      }
      last = 2 * int(gpu.width);
      for (int x = int(gpu.width) - 1; x >= 0; --x) {
        const auto i = size_t(y) * gpu.width + unsigned(x);
        if (std::isfinite(alpha[i]) && alpha[i] > 0) last = x;
        distances[i] = std::min(distances[i], unsigned(last - x));
      }
    }
    for (const int sign : {-1, 1}) {
      const auto off = gpu.render(false, sign);
      if (gpu.has_control) {
        const auto prior = gpu.render(false, sign, false, true);
        require(prior.field.bytes == off.field.bytes && prior.output.bytes == off.output.bytes,
          label + ": disabled protection changed frozen original field or output");
      }
      const auto on = gpu.render(true, sign);
      if (empty) require(on.field.bytes == off.field.bytes && on.output.bytes == off.output.bytes, label + ": all-clear protection is not bit-exact");
      // Every UI pixel below must have exactly zero displacement and match
      // native mono RGB within export quantization. Native mono bypasses the
      // FP16 eye targets, so whole-buffer byte equality is not its contract.
      double worst_slope{}, worst_ui{}, ghost_red{}, largest_original{}, worst_distance{};
      double worst_distance_tolerance{}, worst_distance_residual{}, worst_slope_residual{};
      unsigned worst_distance_x{}, worst_distance_y{};
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        const auto i = size_t(y) * gpu.width + x; const bool ui = std::isfinite(alpha[i]) && alpha[i] > 0;
        const double f = on.field.channel(x, y, 0); require(std::isfinite(f), label + ": non-finite parallax");
        largest_original = std::max(largest_original, std::abs(double(off.field.channel(x, y, 0))) * gpu.width);
        if (x) {
          const double prior = on.field.channel(x - 1, y, 0);
          const double slope = std::abs(f - prior) * gpu.width;
          // Each clipped endpoint has the two-rounding allowance below. Their
          // difference may accumulate both; this scales with local float ULPs,
          // not with an arbitrary resolution-dependent pixel epsilon.
          const double tolerance = 2 * (float_spacing(f) + float_spacing(prior)) * gpu.width;
          worst_slope = std::max(worst_slope, slope);
          worst_slope_residual = std::max(worst_slope_residual, slope - .5 - tolerance);
        }
        const unsigned distance = distances[i];
        if (distance != gpu.width) {
          const double bound = .5 * std::max(int(distance) - 1, 0);
          // HLSL divides an exact half-integer pixel distance by constant W.
          // The compiler may use a rounded reciprocal and multiply, adding up
          // to two float32 rounding steps. Permit two ULPs at this exact local
          // bound in UV, then convert that allowance to pixels for reporting.
          // At W=3840 and ~0.04 UV, one ULP alone is ~1.43e-5 pixels.
          const double tolerance = bound ? 2 * float_spacing(bound / gpu.width) * gpu.width : 0;
          const double excess = std::abs(f) * gpu.width - bound;
          if (excess > worst_distance) {
            worst_distance = excess; worst_distance_tolerance = tolerance;
            worst_distance_x = x; worst_distance_y = y;
          }
          worst_distance_residual = std::max(worst_distance_residual, excess - tolerance);
          if (!bound) require(f == 0, label + ": UI/bilinear support was not pinned exactly");
        }
        if (ui) require(f == 0, label + ": finite positive alpha was not pinned");
        for (unsigned eye = 0; eye != 2; ++eye) {
          const unsigned ex = x + eye * gpu.width;
          if (ui) worst_ui = std::max(worst_ui, color_error(on.output, flat.output, ex, y));
          else ghost_red = std::max(ghost_red, double(on.output.channel(ex, y, 0)));
          for (unsigned c = 0; c != 3; ++c) require(std::isfinite(on.output.channel(ex, y, c)), label + ": non-finite exported color");
        }
      }
      report << label << " sign=" << sign << " original_shift_px=" << largest_original << " slope_px=" << worst_slope <<
        " UI_color_error=" << worst_ui << " ghost_red=" << ghost_red << " bound_excess_px=" << worst_distance <<
        " bound_2ulp_tolerance_px=" << worst_distance_tolerance << " bound_xy=" << worst_distance_x << ',' << worst_distance_y <<
        " bound_excess_beyond_2ulp_px=" << worst_distance_residual << " slope_excess_beyond_2ulp_px=" << worst_slope_residual << std::endl;
      std::printf("MEASURE %s sign=%d max_bound_excess_px=%.12g local_2ulp_tolerance_px=%.12g xy=%u,%u residual=%.12g slope_px=%.12g slope_residual=%.12g\n",
        label.c_str(), sign, worst_distance, worst_distance_tolerance, worst_distance_x, worst_distance_y,
        worst_distance_residual, worst_slope, worst_slope_residual);
      require(largest_original >= .035 * gpu.width, label + ": fixture failed to exercise large signed displacement");
      require(worst_slope_residual <= 0, label + ": horizontal slope exceeds 0.5 pixel plus endpoint float32 rounding");
      require(worst_distance_residual <= 0, label + ": protection exceeded nearest-UI distance bound plus float32 rounding");
      require(worst_ui <= color_tolerance, label + ": protected UI differs from flat source");
      require(ghost_red <= color_tolerance, label + ": UI color leaked outside its original support");
      std::printf("PASS %s sign=%d original_shift=%.4f slope=%.8f UI_error=%.8g ghost=%.8g\n", label.c_str(), sign, largest_original, worst_slope, worst_ui, ghost_red);
    }
  }
  void verify_retained_fg_alpha(fixture &gpu, std::ostream &report, const fs::path &dump_directory) {
    const size_t pixels = size_t(gpu.width) * gpu.height;
    std::vector<float> real_alpha(pixels, 0), presented_alpha(pixels, 0);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) real_alpha[size_t(y) * gpu.width + x] = .25f;
    const auto retained = gpu.retain_mask(real_alpha);
    bool dumped = false;
    for (int sign : {-1, 1}) {
      std::vector<unsigned char> previous_color;
      for (unsigned phase : {0u, gpu.width / 5 + 3}) {
        gpu.pattern(real_alpha, true, phase);
        const auto reference = gpu.render(true, sign); // Matching non-FG source alpha.
        const auto unprotected = gpu.render(false, sign);
        require(reference.field.bytes != unprotected.field.bytes && reference.output.bytes != unprotected.output.bytes,
          "retained-alpha test must exercise a meaningful nonflat UI mask");
        if (!previous_color.empty()) require(reference.output.bytes != previous_color, "new RGB phase did not change the game image");
        previous_color = reference.output.bytes;
        for (unsigned state = 0; state != 3; ++state) {
          std::fill(presented_alpha.begin(), presented_alpha.end(), state == 1 ? 1.f : 0.f);
          if (state == 2) for (unsigned y = 0; y < gpu.height / 3; ++y)
            for (unsigned x = gpu.width * 2 / 3; x < gpu.width; ++x) presented_alpha[size_t(y) * gpu.width + x] = .5f;
          gpu.pattern(presented_alpha, true, phase);
          const auto reused = gpu.renderer.prepare_ui_source(gpu.mask_capture, [](api::resource) {
            require(false, "generated presentation copied an already retained real-input mask");
            return false;
          });
          require(reused.handle == retained.handle, "generated presentation lost the retained mask view");
          const bool capture = !dumped && state == 1;
          const auto actual = gpu.render(true, sign, false, false, true, reused, capture ? dump_directory : fs::path{});
          dumped |= capture;
          require(actual.field.bytes == reference.field.bytes && actual.output.bytes == reference.output.bytes,
            "FG output alpha overrode retained real alpha, or retained RGB replaced the current picture");
          require(gpu.renderer.diagnostics().ui_source.handle == gpu.renderer.ui_source().handle,
            "renderer did not identify the exact consumed external UI mask");
          report << "retained-FG-alpha sign=" << sign << " RGB_phase=" << phase << " presented_alpha_state=" << state
            << " exact_nonFG_field=1 exact_current_RGB_output=1 retained_RGB_poisoned=1" << std::endl;
        }
      }
    }
    require(dumped, "retained-alpha dump fixture did not capture an external mask");
    // Explicit/replay rendering preserves authored settings, including a
    // full-screen UI mask. Live automatic eligibility is tested separately.
    // The next real all-black alpha must also replace the old mask.
    for (float real : {1.f, 0.f}) {
      std::fill(real_alpha.begin(), real_alpha.end(), real);
      const auto current_mask = gpu.retain_mask(real_alpha);
      gpu.pattern(real_alpha, true);
      const auto reference = gpu.render(true, 1);
      std::fill(presented_alpha.begin(), presented_alpha.end(), 1.f - real);
      gpu.pattern(presented_alpha, true);
      const auto actual = gpu.render(true, 1, false, false, true, current_mask);
      require(actual.field.bytes == reference.field.bytes && actual.output.bytes == reference.output.bytes,
        "whole-screen real alpha was rejected or the next real mask did not replace it");
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x)
        require((actual.field.channel(x, y, 0) == 0) == (real == 1), "whole-screen retained mask has the wrong stereo state");
      const auto off = gpu.render(false, 1, false, false, true, current_mask);
      require(!gpu.renderer.diagnostics().ui_source.handle, "disabled protection still consumed external alpha");
      const auto baseline = gpu.render(false, 1);
      require(off.field.bytes == baseline.field.bytes && off.output.bytes == baseline.output.bytes,
        "disabled external alpha changed the rendered picture");
      report << "retained-FG-alpha real_alpha=" << real << " presented_alpha=" << 1.f - real
        << " exact_reference=1 explicit_disable_preserved=1" << std::endl;
    }
    std::puts("PASS retained FG alpha: changing output alpha ignored, current RGB preserved, real opaque honored, new clear mask replaces old");
  }

  void verify_automatic_source_alpha(fixture &gpu, std::ostream &report) {
    using namespace sunshine_game3d;
    const std::uint64_t pixels = std::uint64_t(gpu.width) * gpu.height;
    std::vector<float> selective(size_t(pixels), 0), clear(size_t(pixels), 0), full(size_t(pixels), 1);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) selective[size_t(y) * gpu.width + x] = .25f;
    selective.back() = .5f; // Isolated corner coverage must not vanish in sampling.
    const auto covered = std::uint64_t(std::count_if(selective.begin(), selective.end(), [](float a) { return a > 0; }));
    require(covered > 0 && covered < pixels, "startup-alpha fixture needs selective coverage");
    gpu.pattern(selective, true);
    const auto off = gpu.render(false, 1);
    const auto selective_on = gpu.render(true, 1);
    gpu.pattern(full, true);
    const auto full_on = gpu.render(true, 1);
    require(selective_on.field.bytes != off.field.bytes && selective_on.output.bytes != off.output.bytes &&
        full_on.field.bytes != off.field.bytes && full_on.output.bytes != off.output.bytes,
      "startup-alpha fixture does not exercise UI protection");
    const auto exact = [](const result &actual, const result &expected) {
      return actual.field.bytes == expected.field.bytes && actual.output.bytes == expected.output.bytes;
    };
    const auto unchanged = [&](alpha_probe_counters before, const char *message) {
      const auto after = gpu.renderer.alpha_probe_activity();
      require(after.submitted == before.submitted && after.mapped == before.mapped, message);
    };
    alpha_auto_source source;
    std::uint64_t epoch = 17;
    const auto begin = [&](alpha_auto_policy &policy, std::uint64_t start) {
      source = {}; source.now_ms = start; source.epoch = ++epoch; source.revision = 1; source.session = &policy;
    };
    const auto render = [&](api::resource_view retained = {}, bool eligible = true) {
      return gpu.render(eligible, 1, false, false, source.retained, retained, {}, {}, nullptr, &source);
    };
    // The fixture already drains actual GPU work. A repeated render at the same
    // fake timestamp consumes one completed sample without advancing its age.
    const auto sample = [&](const std::vector<float> &presented, std::uint64_t tick,
                            std::uint64_t expected_covered, api::resource_view retained = {}, bool monitoring = true) {
      source.now_ms = source.tick_ms = tick; ++source.sequence;
      gpu.pattern(presented, true);
      render(retained);
      const auto actual = render(retained);
      const auto decision = gpu.renderer.consumed_alpha_auto();
      require(decision.sample_sequence == source.sequence && decision.sample_tick_ms == tick,
        "startup alpha lost completed source sample identity");
      require(decision.covered == expected_covered && decision.pixels == pixels,
        "startup alpha GPU reduction did not count every finite positive texel exactly");
      const bool enabled = expected_covered > 0 && expected_covered < pixels;
      require(decision.monitoring == monitoring && decision.enabled == enabled &&
          gpu.renderer.consumed_source_alpha_ui() == enabled && exact(actual, enabled ? selective_on : off),
        "startup detector did not apply the current provisional or confirmed UI decision exactly");
      if (!monitoring) require(decision.state == alpha_auto_state::automatic_on,
        "500 ms of selective alpha did not confirm Auto On early");
    };
    {
      alpha_auto_policy policy; begin(policy, 1000);
      gpu.pattern(selective, true);
      const auto waiting = gpu.renderer.alpha_probe_activity();
      require(exact(render({}, false), off), "ineligible waiting source changed stereo");
      source.now_ms = 14000; source.retained = true;
      require(exact(render(), off), "missing FG alpha source changed stereo");
      auto decision = gpu.renderer.consumed_alpha_auto();
      require(decision.monitoring && !decision.window_started &&
          decision.state == alpha_auto_state::waiting_for_source,
        "unavailable startup source consumed the detection window");
      unchanged(waiting, "unavailable startup source submitted or mapped an alpha probe");
      source.retained = false; source.now_ms = source.tick_ms = 15000; source.sequence = 1;
      require(exact(render(), off), "uncompleted first observation enabled UI");
      decision = gpu.renderer.consumed_alpha_auto();
      require(decision.window_started && decision.window_start_ms == 15000 && decision.monitoring &&
          !decision.sample_sequence && gpu.renderer.alpha_probe_activity().submitted == waiting.submitted + 1,
        "first queued alpha observation did not publish the delayed window start immediately");
      require(exact(render(), selective_on), "delayed first selective observation did not enable provisional UI");
      sample(selective, 15500, covered, {}, false);
      const auto stopped = gpu.renderer.alpha_probe_activity();
      source.now_ms = 16000; gpu.pattern(full, true);
      require(exact(render(), full_on), "delayed startup failed to preserve early confirmed On");
      unchanged(stopped, "delayed startup continued probing after confirmation");
    }
    {
      alpha_auto_policy policy; begin(policy, 30000);
      policy.set_manual(false); source.now_ms = source.tick_ms = 35000; source.sequence = 1;
      const auto waiting = gpu.renderer.alpha_probe_activity();
      gpu.pattern(full, true); render();
      policy.set_manual(true); render();
      require(!gpu.renderer.consumed_alpha_auto().window_started,
        "manual protection started an automatic observation window");
      unchanged(waiting, "manual protection queued an automatic observation");
      policy.set_automatic(39000);
      sample(full, 40000, pixels);
      require(gpu.renderer.consumed_alpha_auto().window_start_ms == 40000,
        "returning Auto started before an eligible probe was queued");
      sample(full, 40400, pixels);
      source.now_ms = 40000 + alpha_startup_window_ms - 1; render();
      require(gpu.renderer.consumed_alpha_auto().monitoring,
        "delayed full-alpha window expired before five minutes");
      const auto stopped = gpu.renderer.alpha_probe_activity();
      source.now_ms = 40000 + alpha_startup_window_ms;
      require(exact(render(), off) && !gpu.renderer.consumed_alpha_auto().monitoring,
        "delayed full-alpha window did not finish Off five minutes after its first probe");
      source.now_ms += 5000; gpu.pattern(selective, true);
      require(exact(render(), off), "selective alpha restarted the delayed completed window");
      unchanged(stopped, "delayed full-alpha window continued monitoring after its deadline");
    }
    std::uint64_t start = 10000;
    for (float value : {0.f, 1.f, .5f}) {
      alpha_auto_policy policy(start); begin(policy, start);
      std::fill(full.begin(), full.end(), value);
      sample(full, start + 100, value > 0 ? pixels : 0);
      sample(full, start + 400, value > 0 ? pixels : 0);
      const auto stopped = gpu.renderer.alpha_probe_activity();
      source.now_ms = start + alpha_startup_window_ms;
      require(exact(render(), off) && !gpu.renderer.consumed_alpha_auto().monitoring &&
          gpu.renderer.consumed_alpha_auto().state == alpha_auto_state::automatic_off,
        "zero/full-only startup did not choose Off at its original deadline");
      gpu.pattern(selective, true); source.now_ms += 5000;
      require(exact(render(), off), "post-startup selective alpha changed the frozen Off choice");
      unchanged(stopped, "full-only startup continued GPU dispatch/map after its deadline");
      policy.set_manual(true); gpu.pattern(full, true);
      require(exact(render(), value > 0 ? full_on : off) && gpu.renderer.consumed_alpha_auto().state == alpha_auto_state::manual_on,
        "manual On failed to override startup Off");
      policy.set_manual(false);
      require(exact(render(), off) && gpu.renderer.consumed_alpha_auto().state == alpha_auto_state::manual_off,
        "manual Off failed to override startup On");
      unchanged(stopped, "manual overrides restarted GPU coverage monitoring");
      start += 20000;
    }
    std::fill(full.begin(), full.end(), 1.f);
    {
      alpha_auto_policy policy(70000); begin(policy, 70000);
      sample(selective, 70100, covered); // The first completed selective mask is provisional On.
      sample(full, 70400, pixels); // Full coverage interrupts an unconfirmed candidate.
      sample(selective, 70600, covered);
      sample(clear, 70900, 0); // Empty coverage also turns provisional protection off.
      sample(selective, 71100, covered);
      ++source.revision;
      sample(selective, 71400, covered); // New scope starts a new 500-ms interval.
      sample(selective, 71600, covered); // 500 ms since the old scope must not confirm.
      sample(selective, 71900, covered, {}, false); // Exactly 500 ms in this scope confirms early.
      const auto stopped = gpu.renderer.alpha_probe_activity();
      ++source.revision; source.now_ms = 72000; gpu.pattern(selective, true);
      require(exact(render(), selective_on) && !gpu.renderer.consumed_alpha_auto().monitoring &&
          gpu.renderer.consumed_alpha_auto().state == alpha_auto_state::automatic_on,
        "confirmed Auto On was lost after a source change");
      source.now_ms = 72100; gpu.pattern(full, true);
      require(exact(render(), full_on), "full alpha changed the early confirmed On choice");
      source.now_ms = 72200; gpu.pattern(clear, true);
      require(exact(render(), off) && gpu.renderer.consumed_source_alpha_ui(),
        "empty alpha changed the early confirmed On choice");
      source.now_ms = 70000 + alpha_startup_window_ms + 5000; gpu.pattern(full, true);
      require(exact(render(), full_on), "post-deadline full alpha changed confirmed On choice");
      unchanged(stopped, "confirmed Auto On continued GPU dispatch/map before or after deadline");
      // Recreating renderer resources cannot recreate the process-owned window.
      gpu.renderer.reset_after_runtime_drain();
      require(gpu.renderer.configure(observed_runtime, {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())},
        static_cast<api::color_space>(gpu.color)), "reconfigure renderer after startup detection");
      gpu.pattern(selective, true);
      require(exact(render(), selective_on), "renderer recreation forgot the frozen process toggle");
      unchanged({}, "new renderer allocated/submitted/mapped a post-deadline observation");
    }
    {
      alpha_auto_policy policy(90000); begin(policy, 90000);
      sample(selective, 90000 + alpha_startup_window_ms - 510, covered);
      source.now_ms = source.tick_ms = 90000 + alpha_startup_window_ms - 10; ++source.sequence;
      gpu.pattern(selective, true); render(); // Completed GPU copy, not yet observed on CPU.
      const auto pending = gpu.renderer.alpha_probe_activity();
      source.now_ms = 90000 + alpha_startup_window_ms;
      require(exact(render(), off) && !gpu.renderer.consumed_alpha_auto().enabled,
        "a readback completed at the deadline qualified startup after its window");
      unchanged(pending, "deadline retirement mapped a pending alpha observation");
    }
    {
      alpha_auto_policy policy(110000); begin(policy, 110000);
      source.retained = true;
      const auto retained = gpu.retain_mask(selective);
      sample(full, 110100, covered, retained);
      const auto first = gpu.renderer.consumed_alpha_auto();
      const auto once = gpu.renderer.alpha_probe_activity();
      source.now_ms = 110600; render(retained);
      require(gpu.renderer.consumed_alpha_auto().sample_sequence == first.sample_sequence &&
          gpu.renderer.consumed_alpha_auto().sample_tick_ms == first.sample_tick_ms &&
          gpu.renderer.consumed_alpha_auto().monitoring,
        "re-presenting one retained input for 500 ms confirmed or advanced startup evidence");
      unchanged(once, "duplicate retained mask was dispatched/mapped twice");
      source.now_ms = 110000 + alpha_startup_window_ms;
      require(exact(render(retained), off), "one retained input incorrectly qualified a stable startup interval");
      unchanged(once, "retained-only startup continued monitoring after deadline");
    }
    {
      alpha_auto_policy policy(130000); begin(policy, 130000);
      source.retained = true;
      auto retained = gpu.retain_mask(selective);
      sample(full, 130100, covered, retained);
      retained = gpu.retain_mask(selective);
      sample(full, 130600, covered, retained, false);
      const auto stopped = gpu.renderer.alpha_probe_activity();
      source.now_ms = 130700;
      require(exact(render(retained), selective_on), "startup detector used presented alpha instead of retained input");
      unchanged(stopped, "retained qualified startup continued GPU monitoring");
    }
    {
      alpha_auto_policy policy(150000); begin(policy, 150000);
      sample(selective, 150100, covered);
      source.now_ms = source.tick_ms = 150300; ++source.sequence;
      gpu.pattern(full, true); render();
      const auto pending = gpu.renderer.alpha_probe_activity();
      policy.set_manual(true); source.now_ms = 150301;
      require(exact(render(), full_on), "manual On during startup did not apply immediately");
      policy.set_manual(false);
      require(exact(render(), off), "manual Off during startup did not apply immediately");
      unchanged(pending, "manual override mapped/submitted a pending startup observation");
    }
    {
      constexpr std::uint64_t cadence_start = 200000;
      alpha_auto_policy policy(cadence_start); begin(policy, cadence_start);
      const auto queued = [&](const std::vector<float> &alpha, std::uint64_t offset,
                              std::uint64_t count, bool monitoring = true) {
        const auto before = gpu.renderer.alpha_probe_activity();
        sample(alpha, cadence_start + offset, count, {}, monitoring);
        const auto after = gpu.renderer.alpha_probe_activity();
        require(after.submitted == before.submitted + 1 && after.mapped == before.mapped + 1,
          "alpha cadence did not queue and map exactly one eligible observation");
      };
      const auto throttled = [&](std::uint64_t offset, std::uint64_t interval) {
        source.now_ms = source.tick_ms = cadence_start + offset; ++source.sequence;
        gpu.pattern(full, true);
        const auto before = gpu.renderer.alpha_probe_activity();
        require(exact(render(), off) && exact(render(), off), "throttled opaque alpha changed UI");
        require(gpu.renderer.consumed_alpha_auto().probe_interval_ms == interval,
          "renderer consumed the wrong alpha probe interval");
        unchanged(before, "alpha cadence admitted a probe before its next interval");
      };
      queued(full, 0, pixels);
      throttled(99, 100);
      queued(full, 100, pixels);
      queued(full, alpha_initial_window_ms - 100, pixels);
      throttled(alpha_initial_window_ms, 1000);
      throttled(alpha_initial_window_ms + 899, 1000);
      queued(full, alpha_initial_window_ms + 900, pixels);
      queued(selective, alpha_initial_window_ms + 1900, covered);
      require(gpu.renderer.consumed_alpha_auto().probe_interval_ms == 100,
        "selective evidence at sparse cadence did not request fast confirmation");
      queued(full, alpha_initial_window_ms + 2000, pixels);
      require(gpu.renderer.consumed_alpha_auto().probe_interval_ms == 1000,
        "failed selective candidate did not resume one-probe-per-second monitoring");
      throttled(alpha_initial_window_ms + 2999, 1000);
      queued(full, alpha_initial_window_ms + 3000, pixels);
      queued(selective, alpha_initial_window_ms + 4000, covered);
      queued(selective, alpha_initial_window_ms + 4100, covered);
      queued(selective, alpha_initial_window_ms + 4400, covered);
      queued(selective, alpha_initial_window_ms + 4500, covered, false);
      require(!gpu.renderer.consumed_alpha_auto().probe_interval_ms,
        "confirmed sparse-phase candidate retained an active probe interval");
      const auto stopped = gpu.renderer.alpha_probe_activity();
      source.now_ms = cadence_start + alpha_startup_window_ms; gpu.pattern(full, true);
      require(exact(render(), full_on), "confirmed sparse-phase choice was lost at five minutes");
      unchanged(stopped, "confirmed sparse-phase choice resumed monitoring at the final deadline");
    }
    {
      constexpr std::uint64_t fg_start = 300000;
      alpha_auto_policy policy(fg_start); begin(policy, fg_start); source.retained = true;
      auto retained = gpu.retain_mask(full);
      sample(full, fg_start + alpha_initial_window_ms, pixels, retained);
      for (unsigned second = 1; second <= 2; ++second) {
        const auto before = gpu.renderer.alpha_probe_activity();
        source.now_ms = fg_start + alpha_initial_window_ms + (second - 1) * 1000 + 251;
        require(exact(render(), off) && gpu.renderer.consumed_alpha_auto().window_start_ms == fg_start,
          "expired retained FG alpha changed stereo or restarted sparse detection");
        unchanged(before, "missing retained FG input dispatched or mapped an observation");
        retained = gpu.retain_mask(second == 2 ? selective : full);
        sample(full, fg_start + alpha_initial_window_ms + second * 1000,
          second == 2 ? covered : pixels, retained);
      }
      require(gpu.renderer.consumed_alpha_auto().probe_interval_ms == 100,
        "fresh selective FG input after a stale gap did not enter fast confirmation");
      for (unsigned step = 1; step <= 5; ++step) {
        retained = gpu.retain_mask(selective);
        sample(full, fg_start + alpha_initial_window_ms + 2000 + step * 100, covered, retained, step != 5);
      }
      const auto stopped = gpu.renderer.alpha_probe_activity();
      source.now_ms = fg_start + alpha_initial_window_ms + 3000;
      require(exact(render(), off) && gpu.renderer.consumed_alpha_auto().enabled,
        "missing retained input erased the confirmed FG choice or consumed current output alpha");
      retained = gpu.retain_mask(full);
      require(exact(render(retained), full_on), "confirmed FG choice did not resume with eligible retained alpha");
      unchanged(stopped, "confirmed FG choice resumed monitoring after retained-input availability changed");
    }
    // No session pointer means deterministic explicit/replay interpretation.
    gpu.pattern(full, true);
    require(exact(gpu.render(true, 1), full_on), "startup policy leaked into explicit full-alpha replay");
    gpu.pattern(selective, true);
    require(exact(gpu.render(true, 1), selective_on), "startup policy leaked into explicit selective replay");
    report << "startup-alpha exact_gpu_counts=1 delayed_first_source=1 ineligible_manual_do_not_start=1 provisional_on=1 zero_full_interrupt=1 confirmation_500ms=1 fast_60s_then_1hz=1 fixed_5min_window=1 scope_resets_candidate=1"
      " post_confirmation_and_deadline_dispatch_and_map=0 pending_deadline_discard=1 renderer_recreation=1 retained_identity=1 manual_override=1 explicit_replay=1" << std::endl;
    std::puts("PASS startup alpha: exact GPU coverage, provisional UI, 500ms confirmation, zero later dispatch/map, retained identity and manual/replay overrides");
  }
  void verify_mask_upload_recovery(fixture &gpu) {
    const auto previous = gpu.read(gpu.renderer.ui_source());
    unsigned uploads = 0;
    const auto upload = [&](api::resource texture) {
      ++uploads;
      gpu.context->UpdateSubresource(reinterpret_cast<ID3D11Resource *>(texture.handle), 0, nullptr,
        previous.bytes.data(), previous.width * pixel_bytes(previous.format), 0);
      return true;
    };
    require(!gpu.renderer.prepare_ui_source(0, upload).handle && !uploads,
      "missing capture identity admitted an upload");
    require(!gpu.renderer.prepare_ui_source(gpu.mask_capture + 1, [](api::resource) { return false; }).handle,
      "failed mask upload returned a usable view");
    require(gpu.renderer.prepare_ui_source(gpu.mask_capture, upload).handle && uploads == 1,
      "failed replacement retained the previous upload identity");
    require(gpu.renderer.prepare_ui_source(gpu.mask_capture, upload).handle && uploads == 1,
      "successful retry was copied again");
    observed_runtime->get_command_queue()->wait_idle();
    gpu.renderer.reset_after_runtime_drain();
    require(gpu.renderer.configure(observed_runtime, {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())},
      static_cast<api::color_space>(gpu.color)), "recreate renderer for mask lifetime test");
    require(gpu.renderer.prepare_ui_source(gpu.mask_capture, upload).handle && uploads == 2,
      "renderer recreation reused an upload into a destroyed texture");
    require(gpu.read(gpu.renderer.ui_source()).bytes == previous.bytes,
      "recreated renderer did not restore the exact retained mask");
    std::puts("PASS retained UI upload identity: no repeat copy; failures and renderer replacement reupload");
  }
}
int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 6 || argc > 8) { std::fputs("usage: source_alpha_runtime official-ReShade64.dll fresh-output srgb|scrgb width height [frozen-control.hlsl] [--temporal-ui-probe|--temporal-ui-front-limit]\n", stderr); return 2; }
  std::thread([] { Sleep(120000); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    const unsigned color = !std::strcmp(argv[3], "srgb") ? 1 : !std::strcmp(argv[3], "scrgb") ? 2 : 0;
    require(color != 0, "unsupported source transfer");
    const auto directory = fs::absolute(argv[2]);
    bool temporal_probe = false, temporal_front_limit = false;
    fs::path control;
    for (int index = 6; index < argc; ++index) {
      if (!std::strcmp(argv[index], "--temporal-ui-probe") || !std::strcmp(argv[index], "--temporal-ui-front-limit")) {
        require(!temporal_probe, "duplicate temporal probe argument"); temporal_probe = true;
        temporal_front_limit = !std::strcmp(argv[index], "--temporal-ui-front-limit");
      } else {
        require(control.empty() && std::strncmp(argv[index], "--", 2), "unknown or duplicate runtime-test argument");
        control = fs::absolute(argv[index]);
      }
    }
    fixture gpu(fs::absolute(argv[1]), directory, color, unsigned(std::stoul(argv[4])), unsigned(std::stoul(argv[5])), control);
    if (temporal_probe) { temporal_ui_probe(gpu, directory, temporal_front_limit); return 0; }
    std::ofstream report(directory / "source-alpha-results.txt");
    std::vector<float> alpha(size_t(gpu.width) * gpu.height, 0);
    verify(gpu, alpha, "all-black", report);
    std::fill(alpha.begin(), alpha.end(), 1); verify(gpu, alpha, "all-white", report);
    std::fill(alpha.begin(), alpha.end(), .25f); verify(gpu, alpha, "all-gray", report);
    std::fill(alpha.begin(), alpha.end(), 0);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y) for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) alpha[size_t(y) * gpu.width + x] = .25f;
    alpha[size_t(gpu.height / 8) * gpu.width + gpu.width / 4] = 1;
    alpha[size_t(gpu.height * 7 / 8) * gpu.width + gpu.width * 3 / 4] = 1;
    verify(gpu, alpha, "gray-rectangle-and-single-texels", report);
    std::fill(alpha.begin(), alpha.end(), 0);
    for (unsigned y = gpu.height / 3; y < gpu.height / 3 + 2; ++y) std::fill(alpha.begin() + size_t(y) * gpu.width, alpha.begin() + size_t(y + 1) * gpu.width, .25f);
    verify(gpu, alpha, "full-width-bars", report);
    if (color == 2) {
      std::fill(alpha.begin(), alpha.end(), 0);
      for (size_t i = 0; i < alpha.size(); i += 3) alpha[i] = std::numeric_limits<float>::quiet_NaN();
      for (size_t i = 1; i < alpha.size(); i += 3) alpha[i] = std::numeric_limits<float>::infinity();
      for (size_t i = 2; i < alpha.size(); i += 3) alpha[i] = -1;
      verify(gpu, alpha, "nonpositive-nonfinite-alpha", report);
    }
    verify_fg_transitions(gpu, report);
    verify_independent_ui_plane(gpu, report, directory / "independent-ui-plane-dump");
    verify_nearest_ui_plane(gpu, report, directory / "nearest-ui-plane-dump");
    verify_front_limit_ui_plane(gpu, report, directory / "front-limit-ui-plane-dump");
    verify_retained_fg_alpha(gpu, report, directory / "retained-alpha-dump");
    verify_automatic_source_alpha(gpu, report);
    verify_mask_upload_recovery(gpu);
    require(report.good(), "cannot write evidence");
    std::printf("PASS actual D3D11 source-alpha renderer %ux%u color=%u; no FX or visible window\n", gpu.width, gpu.height, color);
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
