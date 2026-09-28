// SPDX-License-Identifier: GPL-3.0-only
// Opt-in functional test of source-alpha protection in the real native shader.
// No installed FX, CPU warp replica, visible window or performance claim.
#include "test_game3d_render_input.h"
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
#include <map>
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
    ComPtr<ID3D11Query> completion_query;
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
      const D3D11_QUERY_DESC completion_desc{D3D11_QUERY_EVENT, 0};
      checked(device->CreateQuery(&completion_desc, &completion_query), "create fixture completion query");
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
    void drain_render() {
      // ReShade 6.8's D3D11 wait_idle() is disabled. Explicitly complete all
      // work, including finish_present's fence signal, before a test assumes
      // its next render can consume a pending alpha observation. This wait is
      // test-only; the production renderer must retain nonblocking polling.
      context->End(completion_query.Get());
      context->Flush();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      BOOL complete = FALSE;
      HRESULT status;
      while ((status = context->GetData(completion_query.Get(), &complete, sizeof(complete), 0)) == S_FALSE) {
        require(std::chrono::steady_clock::now() < deadline, "fixture GPU completion timed out");
        Sleep(1);
      }
      checked(status, "wait for fixture GPU completion");
      require(complete == TRUE, "fixture GPU completion query did not complete");
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
        const sunshine_game3d::alpha_auto_source *automatic = nullptr,
        const sunshine_game3d::ui_adaptive::source *adaptive = nullptr,
        sunshine_game3d::renderer *render_target = nullptr,
        const sunshine_game3d::ui_render_input *ui_override = nullptr) {
      auto &active = render_target ? *render_target : control ? control_renderer : renderer;
      sunshine_game3d::render_parameters p;
      p.strength = flat ? 0 : 100; p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 1;
      p.depth_scale = 100000; p.strength_blend = 1; p.projection = {0, 1};
      p.convergence = {.05f, sign > 0 ? .03f : .01f}; p.disparity_limit_uv = .04f;
      if (override_parameters) p = *override_parameters;
      context->CopyResource(backbuffer.Get(), source.Get());
      auto *queue = observed_runtime->get_command_queue();
      if (ui_override) {
        sunshine_game3d::render_frame_input frame;
        frame.color = {reinterpret_cast<std::uint64_t>(backbuffer.Get())};
        frame.depth = {reinterpret_cast<std::uint64_t>(depth_view.Get())};
        frame.scene = p; frame.ui = *ui_override;
        require(active.render(queue->get_immediate_command_list(), frame), "production UI detection render failed");
      } else {
        require(sunshine_game3d::test::render_frame(active, queue->get_immediate_command_list(), {reinterpret_cast<std::uint64_t>(backbuffer.Get())},
          {reinterpret_cast<std::uint64_t>(depth_view.Get())}, p,
          alpha_source.handle ? protect : sunshine_game3d::source_alpha_ui_for_present(protect, frame_generation_active),
          alpha_source, plane, automatic, adaptive), "production render failed");
      }
      require(adaptive || sunshine_game3d::ui_parameter_words(protect, active.consumed_ui_plane()) ==
          sunshine_game3d::ui_parameter_words(protect, plane), "renderer changed consumed UI-plane bits");
      std::unique_ptr<sunshine_game3d_test::dump_fixture> dump;
      if (!dump_directory.empty()) {
        dump = std::make_unique<sunshine_game3d_test::dump_fixture>();
        dump->begin(observed_runtime, active, p, {reinterpret_cast<std::uint64_t>(depth_view.Get())},
          frame_generation_active, static_cast<api::color_space>(color));
      }
      queue->flush_immediate_command_list(); active.finish_present();
      if (dump) dump->submitted(observed_runtime);
      drain_render();
      if (dump) dump->verify(observed_runtime, [&](api::resource texture, bool) { return read(texture).bytes; }, dump_directory);
      result out{read(active.diagnostics().final_field), read(active.output())};
      require(out.field.width == width && out.field.height == height && out.output.width == width * 2 && out.output.height == height,
        "renderer used bootstrap swapchain dimensions instead of exact test source");
      require(read({reinterpret_cast<std::uint64_t>(backbuffer.Get())}).bytes == original, "UI protection modified mono source");
      return out;
    }
  };
  void verify_adaptive_ui_plane(fixture &gpu, std::ostream &report, const fs::path &directory) {
    using namespace sunshine_game3d;
    renderer reference;
    require(reference.configure(observed_runtime, {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())},
      static_cast<api::color_space>(gpu.color)), "configure adaptive reference renderer");
    const ui_plane_parameters adaptive_plane{ui_plane_mode::display_fraction, 0.f};
    render_parameters p;
    p.strength = 100; p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 1;
    p.strength_blend = 1; p.projection = {0, 1}; p.convergence = {.05f, .01f}; p.disparity_limit_uv = .04f;
    std::vector<float> raw(size_t(gpu.width) * gpu.height, .02f);
    gpu.context->UpdateSubresource(gpu.depth.Get(), 0, nullptr, raw.data(), gpu.width * sizeof(float), 0);
    std::vector<float> alpha(size_t(gpu.width) * gpu.height, 0.f);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) alpha[size_t(y) * gpu.width + x] = 1.f;
    gpu.pattern(alpha, true);
    const auto explicit_render = [&](bool enabled, const ui_plane_parameters &plane) {
      return gpu.render(enabled, -1, false, false, false, {}, {}, plane, &p, nullptr, nullptr, &reference);
    };
    ui_adaptive::source input;
    input.now_ms = input.tick_ms = 1000; input.epoch = 107; input.revision = 19;
    input.source_id = 31; input.sequence = 1; input.viewport = 3; input.eligible = true;
    struct observation {
      std::array<std::uint64_t, 5> counts{};
      std::uint64_t covered{}, center_pixels{};
      float required_uv{};
    };
    std::map<std::uint64_t, observation> expected;
    std::array<std::uint64_t, 5> current_counts{};
    const std::uint64_t center_pixels = std::uint64_t(gpu.width * 7 / 8 - gpu.width / 8) *
      (gpu.height * 7 / 8 - gpu.height / 8);
    float required_uv{};
    std::uint64_t covered{};
    image scene;
    const auto ui_pixel = [&](const image &field) {
      const auto first = std::find_if(alpha.begin(), alpha.end(), [](float a) { return a > 0.f; });
      require(first != alpha.end(), "adaptive fixture needs a visible UI pixel");
      const auto index = static_cast<size_t>(first - alpha.begin());
      return field.channel(unsigned(index % gpu.width), unsigned(index / gpu.width), 0);
    };
    const auto candidate_conflicts = [&](unsigned level) {
      return current_counts[level] * 100 > covered * 20 ||
        current_counts[level] * 1000 > center_pixels * 20;
    };
    const auto set_scene = [&](float fraction) {
      const double uv = double(gpu.height) * 100. / (2160. * gpu.width);
      p.depth_scale = float(std::abs(fraction) * .04 / (.05 * .01 * uv));
      p.convergence[1] = fraction < 0.f ? .03f : .01f;
      scene = explicit_render(false, {}).field;
      const auto front = explicit_render(true, {ui_plane_mode::front_limit, 0.f});
      const float cap = ui_pixel(front.field);
      require(cap > 0.f, "adaptive fixture requires positive scene cap");
      current_counts = {}; covered = 0;
      // Derive the exact center rectangle from pixel bounds independently of
      // production's tile aggregation, including odd and tiny dimensions.
      for (unsigned y = gpu.height / 8; y < gpu.height * 7 / 8; ++y)
        for (unsigned x = gpu.width / 8; x < gpu.width * 7 / 8; ++x) {
          if (alpha[size_t(y) * gpu.width + x] <= 0.f) continue;
          ++covered;
          const float required = scene.channel(x, y, 0) + .05f * cap;
          for (unsigned level = 0; level < 5; ++level)
            current_counts[level] += required > std::min(ui_adaptive::levels_uv[level], .5f * cap);
        }
      unsigned selected{};
      while (selected < 4 && candidate_conflicts(selected)) ++selected;
      required_uv = std::min(ui_adaptive::levels_uv[selected], .5f * cap);
      if (gpu.has_control) {
        const auto control = gpu.render(false, -1, false, true, false, {}, {}, {}, &p);
        require(control.field.bytes == scene.bytes, "adaptive shader changed UI-off scene geometry");
      }
    };
    std::uint64_t last_accepted{};
    const auto step = [&](bool fresh = true, unsigned elapsed = 100, const fs::path &dump = fs::path{}) {
      input.now_ms += elapsed;
      if (fresh) { ++input.sequence; input.tick_ms = input.now_ms; }
      expected[input.sequence] = {current_counts, covered, center_pixels, required_uv};
      const auto actual = gpu.render(true, -1, false, false, false, {}, dump, adaptive_plane, &p, nullptr, &input);
      const auto decision = gpu.renderer.consumed_ui_adaptive();
      const auto plane = gpu.renderer.consumed_ui_plane();
      require(plane.mode == ui_plane_mode::display_fraction && plane.inverse_depth == decision.applied_fraction &&
        plane.inverse_depth >= 0.f && plane.inverse_depth <= ui_adaptive::max_live_fraction,
        "adaptive applied b1 fraction exceeds the live comfort ceiling");
      require(decision.applied_uv <= ui_adaptive::comfort_cap_uv &&
          ui_pixel(actual.field) <= ui_adaptive::comfort_cap_uv + 1e-9f,
        "adaptive UI exceeded its absolute disparity ceiling");
      const float cursor_plane = admitted_ui_parallax_uv(p, plane, true, true, gpu.width, gpu.height);
      require(double(gpu.width) * std::abs(double(cursor_plane) - ui_pixel(actual.field)) <= 1e-6,
        "Exported cursor plane disagrees with the actual D3D11 UI field");
      if (decision.accepted_sequence && decision.accepted_sequence != last_accepted) {
        require(expected.contains(decision.accepted_sequence) &&
          decision.conflict_counts == expected.at(decision.accepted_sequence).counts &&
          decision.covered_pixels == expected.at(decision.accepted_sequence).covered &&
          decision.center_pixels == expected.at(decision.accepted_sequence).center_pixels &&
          decision.required_uv == expected.at(decision.accepted_sequence).required_uv,
          "GPU conflict observation did not match pre-UI scene, selected mask and exact central area");
        last_accepted = decision.accepted_sequence;
      }
      // A separate renderer has no temporal state. This is also the replay path.
      const auto frozen = explicit_render(true, plane);
      require(actual.field.bytes == frozen.field.bytes && actual.output.bytes == frozen.output.bytes,
        "split probe/application differed from frozen inline UI rendering");
      require(gpu.read(gpu.renderer.diagnostics().candidate).bytes == gpu.read(reference.diagnostics().candidate).bytes &&
        gpu.read(gpu.renderer.diagnostics().vertical_field).bytes == gpu.read(reference.diagnostics().vertical_field).bytes,
        "adaptive UI changed pre-UI scene passes");
      report << "adaptive " << input.now_ms << " sequence " << input.sequence << " accepted " << decision.accepted_sequence
        << " target " << decision.target_index << " applied " << decision.applied_fraction << " capped " << decision.capped_conflict
        << " status " << decision.status << '\n';
      return decision;
    };
    for (const float fraction : {-.2f, -.04f, -.01f, .025f, .04f, .90f}) {
      set_scene(fraction);
      ui_adaptive::decision decision;
      for (unsigned i = 0; i < 7; ++i) decision = step();
      require(std::abs(decision.target_uv - required_uv) < 1e-9f &&
        decision.applied_uv == decision.target_uv &&
        decision.capped_conflict == candidate_conflicts(4),
        "adaptive GPU absolute level or cap differs from pre-UI measurements");
    }
    // Generic allocations can rotate every present with different layout epochs.
    // Completed counts still describe the same admitted display-space scene.
    input.generic_basis_epoch = 71; input.generic_routing_epoch = 73;
    auto submissions = gpu.renderer.ui_probe_submissions();
    const auto before_rotation = submissions;
    std::uint64_t last_submission_tick{};
    ui_adaptive::decision rotating;
    for (unsigned frame = 0; frame < 40; ++frame) {
      input.source_id = 31 + frame % 3; input.revision = 19 + frame % 3;
      rotating = step(true, 16);
      const auto current = gpu.renderer.ui_probe_submissions();
      if (current != submissions) {
        require(current == submissions + 1 && (!last_submission_tick || input.now_ms - last_submission_tick >= 100),
          "Generic rotation bypassed the conflict probe cadence");
        last_submission_tick = input.now_ms;
      }
      submissions = current;
    }
    require(rotating.accepted_sequence && rotating.applied_uv == ui_adaptive::comfort_cap_uv &&
        rotating.target_uv == required_uv && submissions - before_rotation <= 7,
      "Generic ABC rotation failed to sustain adaptive UI with bounded GPU probes");
    const auto before_churn = submissions;
    for (unsigned frame = 0; frame < 14; ++frame) {
      ++input.generic_routing_epoch;
      const auto changed = step(true, 16);
      require(changed.applied_uv == 0 && changed.accepted_sequence == 0,
        "Unrelated capture group consumed the previous group's pending evidence");
      const auto current = gpu.renderer.ui_probe_submissions();
      if (current != submissions) {
        require(current == submissions + 1 && input.now_ms - last_submission_tick >= 100,
          "Logical scope churn bypassed the renderer probe budget");
        last_submission_tick = input.now_ms;
      }
      submissions = current;
    }
    require(submissions - before_churn <= 3, "Scope churn submitted one scan per presentation");
    report << "adaptive rotation=ABC distinct_layouts=1 pending_off_turn=1 probe_interval_ms=100 scope_churn_bounded=1\n";
    input.generic_basis_epoch = input.generic_routing_epoch = 0;
    set_scene(-.2f);
    ui_adaptive::decision released;
    for (unsigned i = 0; i < 25; ++i) released = step();
    require(released.target_uv == 0.f && released.applied_uv == 0.f,
      "sustained GPU clearance did not return UI to screen depth");
    const auto held = released.applied_fraction;
    const auto identity = released.accepted_sequence;
    input.eligible = false;
    for (unsigned i = 0; i < 4; ++i) {
      const auto missing = step(false);
      require(missing.applied_fraction == held && missing.accepted_sequence == identity, "missing depth moved UI plane");
    }
    input.eligible = true;
    const auto expired = step(false);
    require(expired.applied_fraction == held && expired.accepted_sequence == identity, "expired FG reuse became new evidence");
    for (unsigned i = 0; i < 4; ++i) step();
    // Freeze a non-level position during a new rise, to prove replay does not
    // quantize the applied ramp fraction or need the observation history.
    set_scene(.90f);
    step(); step(); step();
    const auto captured = step(true, 25, directory);
    require(captured.applied_uv > 0.f && captured.applied_uv < ui_adaptive::comfort_cap_uv,
      "adaptive dump fixture did not capture a transition fraction");
    const auto historical = explicit_render(true, {ui_plane_mode::display_fraction, .75f});
    const auto front = explicit_render(true, {ui_plane_mode::front_limit, 0.f});
    require(std::abs(historical.field.channel(gpu.width / 3, gpu.height / 4, 0) -
        .75f * front.field.channel(gpu.width / 3, gpu.height / 4, 0)) <= 1e-8f,
      "Live comfort ceiling changed historical frozen 75-percent rendering");
    // Explicit malformed word2 keeps its bits but safely renders at screen depth.
    for (float bad : {-.01f, .751f, std::numeric_limits<float>::quiet_NaN()}) {
      const auto invalid = explicit_render(true, {ui_plane_mode::display_fraction, bad});
      const auto zero = explicit_render(true, {});
      require(invalid.field.bytes == zero.field.bytes && invalid.output.bytes == zero.output.bytes,
        "invalid display fraction did not fall back to screen plane");
      require(admitted_ui_parallax_uv(p, {ui_plane_mode::display_fraction, bad}, true, true, gpu.width, gpu.height) == 0.f,
        "Malformed display fraction exported a nonzero cursor plane");
    }
    p.strength = 37.f; p.strength_blend = .63f;
    const ui_plane_parameters cursor_plane{ui_plane_mode::display_fraction, .5f};
    const auto partial_strength = explicit_render(true, cursor_plane);
    const float exported = admitted_ui_parallax_uv(p, reference.consumed_ui_plane(), true, true, gpu.width, gpu.height);
    require(exported > 0.f && double(gpu.width) * std::abs(double(exported) -
        partial_strength.field.channel(gpu.width / 3, gpu.height / 4, 0)) <= 1e-6,
      "Partial strength/blend separated the exported cursor and actual D3D11 UI planes");
    require(admitted_ui_parallax_uv(p, cursor_plane, false, true, gpu.width, gpu.height) == 0.f &&
        admitted_ui_parallax_uv(p, cursor_plane, true, false, gpu.width, gpu.height) == 0.f &&
        admitted_ui_parallax_uv(p, cursor_plane, true, true, 4800, gpu.height) == 0.f,
      "Unadmitted, disabled or unsupported UI exported a nonzero cursor plane");
    auto inactive = p; inactive.depth_ready = 0;
    require(admitted_ui_parallax_uv(inactive, cursor_plane, true, true, gpu.width, gpu.height) == 0.f,
      "Missing depth exported a nonzero cursor plane");
    inactive = p; inactive.camera_ready = 0;
    require(admitted_ui_parallax_uv(inactive, cursor_plane, true, true, gpu.width, gpu.height) == 0.f,
      "Missing camera exported a nonzero cursor plane");
    inactive = p; inactive.depth_view = 1;
    require(admitted_ui_parallax_uv(inactive, cursor_plane, true, true, gpu.width, gpu.height) == 0.f,
      "Depth diagnostics exported a shifted cursor plane");
    p.strength = 100.f; p.strength_blend = 1.f;
    const auto edge_mask = [&]() {
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x)
        alpha[size_t(y) * gpu.width + x] = x < gpu.width / 8 || x >= gpu.width * 7 / 8 ||
          y < gpu.height / 8 || y >= gpu.height * 7 / 8 ? 1.f : 0.f;
    };
    edge_mask();
    gpu.pattern(alpha, true); set_scene(.9f);
    ++input.revision; last_accepted = 0;
    ui_adaptive::decision spatial;
    for (unsigned i = 0; i < 4; ++i) spatial = step();
    require(spatial.accepted_sequence && spatial.covered_pixels == 0 && spatial.target_index == 0 &&
        spatial.applied_fraction == 0.f && spatial.conflict_counts == std::array<std::uint64_t, 5>{},
      "Edge-only UI promoted the D3D11 plane or entered the central denominator");
    const auto edge_protected = explicit_render(true, adaptive_plane);
    require(ui_pixel(edge_protected.field) == 0.f && edge_protected.field.bytes != scene.bytes,
      "Ignoring border conflicts disabled protection of border UI");
    // One center texel must be compared to one center UI texel, even with a
    // much larger border mask. There is no absolute minimum conflict count.
    alpha[size_t(gpu.height / 2) * gpu.width + gpu.width / 2] = 1.f;
    gpu.pattern(alpha, true); set_scene(.9f);
    ++input.revision; last_accepted = 0;
    for (unsigned i = 0; i < 6; ++i) spatial = step();
    require(spatial.covered_pixels == 1 && spatial.conflict_counts[3] == 1 &&
        spatial.applied_uv == ui_adaptive::comfort_cap_uv && spatial.capped_conflict,
      "Border UI diluted the center conflict ratio or an absolute floor hid a central texel");
    report << "adaptive-center75 edge_only_ignored=1 center_denominator_only=1 exact_central_area=1 ui_ratio_or_area_entry=1 single_center_texel_triggers=1 border_protection_retained=1 exact_frozen_pixels=1\n";
    gpu.drain_render();
    reference.reset_after_runtime_drain();
    std::puts("PASS adaptive UI D3D11: exact center75 counts and area, UI-ratio-or-area entry, edge-only exclusion, retained border protection, five absolute levels, delayed retreat, frozen pixel/cursor parity and replayable ramp");
  }

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
      {static_cast<ui_plane_mode>(UINT32_MAX), 1.f}, {ui_plane_mode::depth_midpoint, -1.f},
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
      require(!sunshine_game3d::test::render_frame(gpu.control_renderer, observed_runtime->get_command_queue()->get_immediate_command_list(),
          {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())},
          {reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())}, p, true, {}, plane),
        "historical shader silently accepted an unsupported nearest-UI plane");

    // A broken legacy-only entry must not participate in normal configuration.
    // Requesting it later fails before command recording, even after one of its
    // two pipelines and both textures were created successfully.
    {
      std::string source(sunshine_game3d::renderer::shader_source());
      const std::string entry = "void SunshineUINearestReduceCS(";
      const auto found = source.find(entry);
      require(found != std::string::npos, "lazy nearest-UI fixture cannot find the reduction entry");
      source.replace(found, entry.size(), "void SunshineUnavailableNearestReduceCS(");
      sunshine_game3d::renderer lazy;
      const api::resource backbuffer{reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())};
      const api::resource_view depth{reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())};
      require(lazy.configure(observed_runtime, backbuffer, static_cast<api::color_space>(gpu.color), source),
        "normal configuration eagerly compiled a legacy nearest-UI pipeline");
      upload(q, false); gpu.pattern(alpha, true);
      const ui_plane_parameters ordinary{ui_plane_mode::display_fraction, .1f};
      const auto expected = render(true, ordinary, p);
      const auto draw_ordinary = [&] {
        return gpu.render(true, 1, false, false, false, {}, {}, ordinary, &p, nullptr, nullptr, &lazy);
      };
      const auto before = draw_ordinary();
      require(before.field.bytes == expected.field.bytes && before.output.bytes == expected.output.bytes,
        "lazy legacy preparation changed ordinary UI rendering");
      for (unsigned attempt = 0; attempt != 2; ++attempt) {
        require(!sunshine_game3d::test::render_frame(lazy, observed_runtime->get_command_queue()->get_immediate_command_list(),
            backbuffer, depth, p, true, {}, plane), "unavailable nearest-UI entry did not reject its render");
        require(sunshine_game3d::ui_parameter_words(true, lazy.consumed_ui_plane()) ==
            sunshine_game3d::ui_parameter_words(true, ordinary),
          "failed nearest-UI preparation modified the last consumed render");
      }
      // No finish_present between rejection and recovery: any recorded/pending
      // failed frame would prevent this ordinary render from succeeding.
      const auto after = draw_ordinary();
      require(after.field.bytes == before.field.bytes && after.output.bytes == before.output.bytes,
        "partial nearest-UI preparation failure poisoned subsequent ordinary rendering");
      lazy.reset_after_runtime_drain();
      report << "nearest-UI lazy_configure=1 partial_prepare_failure_before_commands=1 ordinary_recovery_exact=1\n";
    }

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
  void verify_front_limit_ui_plane(fixture &gpu, std::ostream &report, const fs::path &dump_directory, bool shallow = false) {
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
    const ui_plane_parameters front{shallow ? ui_plane_mode::shallow_front : ui_plane_mode::front_limit, 0.f};
    const float fraction = shallow ? .25f : 1.f;
    const char *label = shallow ? "shallow-front" : "front-limit";
    const auto comparison_mode = shallow ? ui_plane_mode::depth_midpoint : ui_plane_mode::depth_midpoint_nearest_ui;
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
      const float expected = fraction * budget(parameters);
      const float observed = actual.field.channel(px, py, 0);
      require(std::abs(double(observed) - expected) <= 2 * float_spacing(expected), "front-limit UI does not use the positive current display budget");
      for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x) {
        const float value = actual.field.channel(x, y, 0);
        require(std::isfinite(value) && std::abs(value) <= budget(parameters) + float_spacing(budget(parameters)), "fixed-plane field exceeded the current finite display budget");
        if (alpha[size_t(y) * gpu.width + x] > 0) require(value == observed, "front-limit UI did not share one global plane");
      }
      no_reduction();
      return observed;
    };
    if (gpu.has_control && gpu.control_renderer.active_shader_source().find(shallow ? "#define SUNSHINE_UI_SHALLOW_FRONT_PLANE 1" :
        "#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1") == std::string_view::npos)
      require(!sunshine_game3d::test::render_frame(gpu.control_renderer, observed_runtime->get_command_queue()->get_immediate_command_list(),
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
      const auto nearest = render(true, {comparison_mode, .25f}, parameters);
      const auto candidate = gpu.read(gpu.renderer.diagnostics().candidate).bytes;
      const auto vertical = gpu.read(gpu.renderer.diagnostics().vertical_field).bytes;
      if (!frame) first_candidate = candidate;
      else candidate_changed |= candidate != first_candidate;
      const float prior = nearest.field.channel(px, py, 0);
      mode2_min = std::min(mode2_min, prior); mode2_max = std::max(mode2_max, prior);
      if ((!frame || shallow) && gpu.has_control && gpu.control_renderer.active_shader_source().find("#define SUNSHINE_UI_NEAREST_PLANE 1") != std::string_view::npos) {
        const auto frozen = render(true, {comparison_mode, .25f}, parameters, true);
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
      // Quantify the existing UI neighborhood conditioner against actual GPU
      // UI-off output. The already pinned field cannot reveal those changes.
      const auto unprotected = render(false, {}, parameters);
      if (shallow && gpu.has_control) {
        const auto frozen_off = render(false, {}, parameters, true);
        require(unprotected.field.bytes == frozen_off.field.bytes && unprotected.output.bytes == frozen_off.output.bytes,
          "shallow-front support changed unprotected scene field or RGB during the temporal comparison");
      }
      size_t changed_pixels{}, changed_uncovered{}, conflict_pixels{};
      for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x) {
        const bool covered = alpha[size_t(y) * gpu.width + x] > 0;
        const float before = unprotected.field.channel(x, y, 0);
        if (before != actual.field.channel(x, y, 0)) {
          ++changed_pixels;
          if (!covered) ++changed_uncovered;
        }
        if (covered && before > observed) ++conflict_pixels;
      }
      report << label << " temporal frame=" << frame << " raw_q=" << raw[peak] << " gain=" << parameters.depth_scale
        << " zero=" << parameters.convergence[1] << " comparison_mode=" << static_cast<unsigned>(comparison_mode)
        << " comparison_px=" << prior * gpu.width << " fixed_px=" << observed * gpu.width
        << " changed_pixels=" << changed_pixels << " changed_uncovered=" << changed_uncovered
        << " pre_ui_conflict_pixels=" << conflict_pixels << std::endl;
    }
    require(candidate_changed && mode2_max > mode2_min, "front-limit temporal fixture did not exercise changing scene geometry and nearest-mode UI");
    if (shallow) {
      // Exercise a genuine partial-mask foreground conflict even at 1080p.
      // A shallow UI cannot retain the full foreground cap in its collar.
      const auto saved_raw = raw;
      raw.assign(count, 1.f); upload();
      auto close = p; close.depth_scale *= 1000.f;
      const auto before = render(false, {}, close);
      const auto after = render(true, front, close);
      const float plane = pinned(after, close);
      size_t conflicts{}, changed_uncovered{};
      for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x) {
        const bool covered = alpha[size_t(y) * gpu.width + x] > 0;
        const float original = before.field.channel(x, y, 0);
        if (covered && original > plane) ++conflicts;
        if (!covered && original != after.field.channel(x, y, 0)) ++changed_uncovered;
      }
      require(conflicts && changed_uncovered,
        "shallow partial-mask fixture did not exercise foreground conflict and the existing UI collar");
      require(after.field.channel(0, py, 0) == before.field.channel(0, py, 0) && after.field.channel(0, py, 0) > plane,
        "shallow UI flattened foreground outside the local UI support");
      report << label << " partial_foreground_conflict=" << conflicts << " changed_uncovered=" << changed_uncovered
        << " unaffected_foreground_px=" << after.field.channel(0, py, 0) * gpu.width
        << " ui_per_eye_pixels=" << plane * gpu.width << std::endl;
      raw = saved_raw; upload();
    }
    for (unsigned variant = 0; variant != 5; ++variant) {
      auto parameters = p;
      if (variant == 0) parameters.strength = 40;
      if (variant == 1) parameters.strength_blend = .25f;
      if (variant == 2) { parameters.strength = 150; parameters.strength_blend = 1.5f; }
      if (variant == 3) parameters.disparity_limit_uv = .006f;
      if (variant == 4) { parameters.strength = 50; parameters.disparity_limit_uv = .01f; }
      const float observed = pinned(render(true, front, parameters), parameters);
      report << label << " controls variant=" << variant << " strength=" << parameters.strength
        << " blend=" << parameters.strength_blend << " limit=" << parameters.disparity_limit_uv
        << " ui_per_eye_pixels=" << observed * gpu.width << std::endl;
    }
    const auto reference = render(true, front, p);
    for (float ignored : {.7f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      const auto actual = render(true, {front.mode, ignored}, p);
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
    require(saturated_scene.field.channel(0, 0, 0) > 0, "fixed-plane cap oracle is not saturated positive scene depth");
    for (unsigned y = 0; y != gpu.height; ++y) for (unsigned x = 0; x != gpu.width; ++x)
      require(exact_front.field.channel(x, y, 0) == fraction * saturated_scene.field.channel(x, y, 0),
        "fixed-plane float ordering differs from the authoritative GPU scene cap at .02/3/.7");
    std::uint32_t bound_bits{};
    const float actual_bound = saturated_scene.field.channel(0, 0, 0);
    std::memcpy(&bound_bits, &actual_bound, sizeof(bound_bits));
    report << label << " bound_rounding=exact limit=.02 strength=3 blend=.7 fraction=" << fraction << " cap_source_u_bits=" << bound_bits << std::endl;
    // Depth-plane modes must obey the same final bound in both directions.
    // The actual positive GPU bound and its exact sign flip define the
    // symmetric display interval. A negative uniform scene can round inward
    // during Q30 conditioning, so its field is not the negative-bound oracle.
    if (!shallow) for (const float inverse_depth : {1.f, 0.f}) {
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
    report << label << " stability=exact fraction=" << fraction << " strength_blend_budget=1 unused_q=1 mono=1 no_reduction=1 current_RGB_error=" << rgb_error << std::endl;
    std::printf("PASS %s UI: fixed across depth/gain/zero, current strength/blend/budget, ignored q bits, selected alpha, current RGB and legacy/mono compatibility\n", label);
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
    const auto pixels = size_t(gpu.width) * gpu.height;
    std::array<std::vector<float>, 3> masks{
      std::vector<float>(pixels, 1.f), std::vector<float>(pixels, 0.f), std::vector<float>(pixels, 0.f)};
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) masks[1][size_t(y) * gpu.width + x] = .25f;
    masks[1].back() = .5f; // Include isolated translucent coverage in the exact pixel reference.
    std::array<result, 3> off, on;
    for (unsigned mask = 0; mask != masks.size(); ++mask) {
      gpu.pattern(masks[mask], true);
      off[mask] = gpu.render(false, 1);
      on[mask] = gpu.render(true, 1);
    }
    require(on[0].field.bytes != off[0].field.bytes && on[0].output.bytes != off[0].output.bytes &&
        on[1].field.bytes != off[1].field.bytes && on[1].output.bytes != off[1].output.bytes,
      "Automatic UI fixture does not exercise full/selective protection");
    const auto exact = [](const result &actual, const result &expected) {
      return actual.field.bytes == expected.field.bytes && actual.output.bytes == expected.output.bytes;
    };
    alpha_auto_policy policy;
    alpha_auto_source source;
    source.session = &policy; source.now_ms = source.tick_ms = 1000;
    source.epoch = 17; source.revision = 1; source.sequence = 1;
    const auto verify = [&](unsigned mask, api::resource_view retained, bool eligible = true) {
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      const auto actual = gpu.render(eligible, 1, false, false, source.retained, retained, {}, {}, nullptr, &source);
      const bool protected_pixels = eligible && mask == 1;
      require(exact(actual, protected_pixels ? on[mask] : off[mask]),
        "Automatic alpha detection changed exact field/SBS or accepted a full-scene mask");
      const auto resolved = gpu.renderer.diagnostics().ui_source;
      require(resolved.handle, "Auto did not expose its actual resolved GPU mask");
      const auto selected = gpu.read(resolved);
      require(selected.format == DXGI_FORMAT_R32_FLOAT && selected.width == gpu.width && selected.height == gpu.height,
        "Auto diagnostic is not the resolved full-resolution R32 mask");
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        const auto value = selected.channel(x, y, 0);
        const bool wanted = protected_pixels && masks[mask][size_t(y) * gpu.width + x] > 0.f;
        require(std::isfinite(value) && value >= 0.f && value <= 1.f && (value > 0.f) == wanted,
          "Auto mask retained full-scene/old coverage or lost selective UI pixels");
      }
    };
    // All decisions come from the real GPU inputs. No review, provider approval
    // mutation or injected CPU coverage statistics are used by this fixture.
    for (unsigned kind = 0; kind != 3; ++kind) {
      source.retained = kind != 0; source.dedicated_mask = kind == 2;
      ++source.revision;
      // Switching from selective directly to opaque/empty tests current-frame
      // rejection: yesterday's valid UI may not flatten today's whole scene.
      for (const unsigned mask : {1u, 0u, 2u, 1u}) {
        const auto retained = kind ? gpu.retain_mask(masks[mask]) : api::resource_view{};
        gpu.pattern(masks[mask], true);
        verify(mask, retained);
      }
    }
    if (gpu.color == 2) {
      source.retained = source.dedicated_mask = false;
      for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity(), -1.f, 2.f}) {
        auto malformed = masks[1]; malformed.front() = invalid;
        gpu.pattern(malformed, true);
        source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
        require(exact(gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, &source), off[1]),
          "Automatic alpha accepted nonfinite or out-of-range input as a valid UI mask");
      }
    }
    // A usable retained candidate wins over unusable presented alpha, while
    // its unrelated historical RGB must never enter the current picture.
    source.retained = source.dedicated_mask = true;
    const auto selective = gpu.retain_mask(masks[1]);
    gpu.pattern(masks[0], true);
    source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
    require(exact(gpu.render(true, 1, false, false, true, selective, {}, {}, nullptr, &source), on[1]),
      "Automatic retained UI lost precedence or replaced current RGB");
    policy.set_manual(false);
    require(exact(gpu.render(true, 1, false, false, true, selective, {}, {}, nullptr, &source), off[0]) &&
        gpu.renderer.consumed_alpha_auto().state == alpha_auto_state::manual_off,
      "Manual Off did not suppress automatically detected retained UI");
    policy.set_automatic(source.now_ms);
    require(exact(gpu.render(true, 1, false, false, true, selective, {}, {}, nullptr, &source), on[1]),
      "Returning to Auto required review or lost an available selective mask");
    source.retained = source.dedicated_mask = false;
    gpu.renderer.reset_after_runtime_drain();
    require(gpu.renderer.configure(observed_runtime, {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())},
      static_cast<api::color_space>(gpu.color)), "recreate renderer for automatic source");
    gpu.pattern(masks[1], true); verify(1, {});
    for (unsigned mask = 0; mask != masks.size(); ++mask) {
      gpu.pattern(masks[mask], true);
      require(exact(gpu.render(true, 1), on[mask]), "Automatic detection changed explicit replay rendering");
    }
    report << "automatic-ui D3D11 current_captured_dedicated=1 full_scene_rejected=1 empty_rejected=1 selective_without_review=1 immediate_bad_frame_rejection=1 resolved_gpu_mask_checked=1 current_RGB_preserved=1 renderer_recreation=1 manual_off_wins=1 explicit_field_and_sbs_parity=1\n";
    std::puts("PASS D3D11 automatic UI: full-scene/empty rejected immediately, selective exact field/SBS without review, current RGB and manual Off preserved");
  }
  void verify_automatic_hudless(fixture &gpu, std::ostream &report) {
    using namespace sunshine_game3d;
    const auto pixels = size_t(gpu.width) * gpu.height;
    std::vector<float> mask(pixels, 0.f), empty(pixels, 0.f);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) mask[size_t(y) * gpu.width + x] = 1.f;
    const auto bpp = gpu.color == 2 ? 8u : 4u;
    const auto opaque = [&](std::vector<unsigned char> &bytes) {
      for (size_t i = 0; i != pixels; ++i) {
        if (gpu.color == 2) { const auto one = half_bits(1.f); std::memcpy(bytes.data() + i * bpp + 6, &one, 2); }
        else bytes[i * bpp + 3] = 255;
      }
    };
    gpu.pattern(empty); auto hudless = gpu.original; opaque(hudless);
    gpu.pattern(empty, false, 13); auto moved_scene = gpu.original; opaque(moved_scene);
    gpu.pattern(mask); const auto protected_reference = gpu.render(true, 1);
    const auto off_reference = gpu.render(false, 1);
    auto final_color = gpu.original; opaque(final_color);
    require(protected_reference.field.bytes != off_reference.field.bytes &&
        protected_reference.output.bytes != off_reference.output.bytes,
      "HUDless fixture did not exercise a visible HUD protection change");

    std::vector<ComPtr<ID3D11Texture2D>> textures;
    std::vector<ComPtr<ID3D11ShaderResourceView>> views;
    const auto color_view = [&](const std::vector<unsigned char> &bytes) {
      D3D11_TEXTURE2D_DESC desc{}; gpu.source->GetDesc(&desc); desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data{bytes.data(), gpu.width * bpp, 0};
      ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> view;
      checked(gpu.device->CreateTexture2D(&desc, &data, &texture), "HUDless candidate texture");
      checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "HUDless candidate view");
      const api::resource_view result{reinterpret_cast<std::uint64_t>(view.Get())};
      textures.push_back(std::move(texture)); views.push_back(std::move(view)); return result;
    };
    const auto red_view = [&](const std::vector<float> &values) {
      D3D11_TEXTURE2D_DESC desc{}; gpu.source->GetDesc(&desc);
      desc.Format = DXGI_FORMAT_R32_FLOAT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data{values.data(), gpu.width * unsigned(sizeof(float)), 0};
      ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> view;
      checked(gpu.device->CreateTexture2D(&desc, &data, &texture), "UI alpha candidate texture");
      checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "UI alpha candidate view");
      const api::resource_view result{reinterpret_cast<std::uint64_t>(view.Get())};
      textures.push_back(std::move(texture)); views.push_back(std::move(view)); return result;
    };
    const auto correct = color_view(hudless), shifted = color_view(moved_scene), flattened_ui = color_view(final_color);
    const auto explicit_alpha = red_view(mask), opaque_alpha = red_view(std::vector<float>(pixels, 1.f));
    gpu.original = final_color;
    gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), gpu.width * bpp, 0);
    const auto exact = [](const result &a, const result &b) {
      return a.field.bytes == b.field.bytes && a.output.bytes == b.output.bytes;
    };
    require(exact(gpu.render(false, 1), off_reference), "Opaque final alpha changed the unprotected RGB reference");
    alpha_auto_policy policy;
    alpha_auto_source source;
    source.session = &policy; source.now_ms = source.tick_ms = 1000;
    source.epoch = 29; source.revision = 1; source.sequence = 1;
    ui_render_input ui;
    ui.kind = ui_input_kind::hudless_difference; ui.view = correct; ui.automatic = &source;
    const auto run = [&](bool accepted, const char *label, bool inspect_mask = true) {
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      const auto actual = gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      require(exact(actual, accepted ? protected_reference : off_reference),
        std::string(label) + ": HUDless detection changed RGB, missed HUD or flattened mismatched scene");
      if (inspect_mask) {
        const auto resource = gpu.renderer.diagnostics().ui_source;
        require(resource.handle, std::string(label) + ": missing resolved GPU mask");
        const auto selected = gpu.read(resource);
        require(selected.format == DXGI_FORMAT_R32_FLOAT, "HUDless diagnostic is not an R32 mask");
        for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
          const auto value = selected.channel(x, y, 0);
          const bool wanted = accepted && mask[size_t(y) * gpu.width + x] > 0.f;
          require(std::isfinite(value) && value >= 0.f && value <= 1.f && (value > 0.f) == wanted,
            std::string(label) + ": resolved mask does not match the visible HUD difference");
        }
      }
    };
    run(true, "matching final/HUDless");
    ui.view = shifted;
    run(false, "scene-wide camera shift after a good pair");
    ui.view = flattened_ui;
    run(false, "identical final/HUDless has no separator");
    ui.view = correct;
    run(true, "matching pair recovers automatically");
    // Reproduce the useful part of Hogwarts' discovery: a transport-valid
    // UIColorAndAlpha can be an opaque full scene. Continue to HUDless in the
    // same frame instead of letting that unusable higher candidate win.
    ui_detection_inputs inputs;
    inputs.masks[1] = flattened_ui; inputs.hudless = correct; inputs.current_color = true;
    ui.detection = &inputs;
    run(true, "flattened explicit UI falls through to HUDless");
    inputs.hudless = shifted;
    run(false, "all candidates unsuitable");
    inputs.masks[0] = explicit_alpha;
    run(true, "explicit single-channel alpha wins over unsuitable color inputs");
    inputs.masks[0] = opaque_alpha;
    run(false, "full-scene explicit alpha cannot override unsuitable inputs");
    inputs.hudless = correct;
    run(true, "full-scene explicit alpha falls through to HUDless");
    inputs.masks[0] = {};
    policy.set_manual(false);
    run(false, "manual Off wins over a valid pair", false);
    require(gpu.renderer.consumed_alpha_auto().state == alpha_auto_state::manual_off,
      "HUDless detection changed the user's manual Off setting");
    policy.set_automatic(source.now_ms);
    run(true, "Auto resumes without review");
    report << "automatic-hudless D3D11 final_alpha_opaque=1 paired_HUD_exact=1 scene_wide_motion_rejected=1 identical_pair_empty=1 same_frame_bad_pair_rejection=1 flattened_explicit_fallback=1 current_RGB_preserved=1 manual_off_wins=1 no_review=1\n";
    std::puts("PASS D3D11 HUDless auto: exact HUD difference, scene-wide motion/identical rejection, flattened candidate fallback and manual Off without review");
  }
  void verify_normalized_ui_input(fixture &gpu, std::ostream &report) {
    using namespace sunshine_game3d;
    std::vector<float> selective(size_t(gpu.width) * gpu.height, 0.f);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x)
        selective[size_t(y) * gpu.width + x] = 1.f;
    const auto retained = gpu.retain_mask(std::vector<float>(selective.size(), 0.f));
    gpu.pattern(selective, true);
    const auto protected_pixels = gpu.render(true, 1);
    const auto unprotected_pixels = gpu.render(false, 1);
    require(protected_pixels.field.bytes != unprotected_pixels.field.bytes,
      "Normalized UI input fixture must distinguish protected and unprotected geometry");

    render_frame_input frame;
    frame.color = {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())};
    frame.depth = {reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())};
    frame.scene = gpu.renderer.consumed_parameters();
    const auto verify = [&](const result &expected, bool enabled) {
      gpu.context->CopyResource(gpu.backbuffer.Get(), gpu.source.Get());
      auto *queue = observed_runtime->get_command_queue();
      require(gpu.renderer.render(queue->get_immediate_command_list(), frame),
        "Normalized UI input render was rejected");
      queue->flush_immediate_command_list();
      gpu.renderer.finish_present();
      gpu.drain_render();
      require(gpu.renderer.consumed_source_alpha_ui() == enabled &&
          gpu.read(gpu.renderer.diagnostics().final_field).bytes == expected.field.bytes &&
          gpu.read(gpu.renderer.output()).bytes == expected.output.bytes,
        "Normalized UI kind selected a different alpha source or changed pixels");
    };
    require(!frame.ui.available(), "Default UI input is unexpectedly available");
    verify(unprotected_pixels, false);
    frame.ui.view = retained;
    require(!frame.ui.available(), "A stray view enabled explicitly unavailable UI input");
    verify(unprotected_pixels, false);
    for (const auto kind : {ui_input_kind::captured_color_alpha, ui_input_kind::dedicated_mask}) {
      frame.ui.kind = kind;
      frame.ui.view = {};
      require(!frame.ui.available(), "Missing captured view fell back to current color alpha");
      verify(unprotected_pixels, false);
      frame.ui.view = retained;
      require(frame.ui.available(), "Completed captured mask was not available");
      verify(unprotected_pixels, true); // The retained mask is empty; current alpha is selective.
    }
    frame.ui.kind = ui_input_kind::current_color_alpha;
    frame.ui.view = {};
    require(frame.ui.available(), "Explicit current color requires an unrelated captured view");
    verify(protected_pixels, true);
    frame.ui.view = retained;
    verify(protected_pixels, true); // Source kind, not stray view presence, owns selection.
    report << "normalized-ui-input current_explicit=1 captured_missing_no_fallback=1 unavailable_view_ignored=1 current_view_ignored=1 exact_pixels=1\n";
    std::puts("PASS normalized UI input: explicit current alpha, captured-view availability, no implicit fallback and exact pixels");
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
  std::thread([] { Sleep(180000); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    const unsigned color = !std::strcmp(argv[3], "srgb") ? 1 : !std::strcmp(argv[3], "scrgb") ? 2 : 0;
    require(color != 0, "unsupported source transfer");
    const auto directory = fs::absolute(argv[2]);
    bool temporal_probe = false, temporal_front_limit = false, adaptive_only = false;
    fs::path control;
    for (int index = 6; index < argc; ++index) {
      if (!std::strcmp(argv[index], "--adaptive-ui-only")) {
        require(!adaptive_only, "duplicate adaptive-only argument"); adaptive_only = true;
      } else if (!std::strcmp(argv[index], "--temporal-ui-probe") || !std::strcmp(argv[index], "--temporal-ui-front-limit")) {
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
    if (adaptive_only) {
      verify_adaptive_ui_plane(gpu, report, directory / "adaptive-ui-plane-dump");
      require(report.good(), "cannot write adaptive evidence");
      return 0;
    }
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
    verify_front_limit_ui_plane(gpu, report, directory / "shallow-front-ui-plane-dump", true);
    verify_adaptive_ui_plane(gpu, report, directory / "adaptive-ui-plane-dump");
    verify_retained_fg_alpha(gpu, report, directory / "retained-alpha-dump");
    verify_automatic_source_alpha(gpu, report);
    verify_automatic_hudless(gpu, report);
    verify_normalized_ui_input(gpu, report);
    verify_mask_upload_recovery(gpu);
    require(report.good(), "cannot write evidence");
    std::printf("PASS actual D3D11 source-alpha renderer %ux%u color=%u; no FX or visible window\n", gpu.width, gpu.height, color);
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
