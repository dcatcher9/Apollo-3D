// SPDX-License-Identifier: GPL-3.0-only
// Opt-in functional test of source-alpha protection in the real native shader.
// No installed FX, CPU warp replica, visible window or performance claim.
#include "test_game3d_render_input.h"
#include "game3d_controls.h"
#include "test_game3d_debug_dump_runtime.h"
#include "game3d_ui_darkening.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_layer.h"
#include "game3d_ui_temporal.h"
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
#include <functional>
#include <future>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <map>
#include <set>
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
    // Whether a UI-detection render's depth is this frame's (render_frame_input::depth_current).
    bool depth_current = true;
    std::vector<unsigned char> original;
    // The acceptance key (A1) of a fixture candidate: its typed DXGI format,
    // the source color's by default, in the fixture's color space.
    std::string key(sunshine_game3d::ui_selection::kind kind, DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN) const {
      if (format == DXGI_FORMAT_UNKNOWN) format = color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
      return sunshine_game3d::ui_selection::signature{kind, std::uint32_t(format), color}.key();
    }
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
        frame.scene = p; frame.ui = *ui_override; frame.depth_current = depth_current;
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
      // A dumped render's own retention, owed until the dump's copies (fix 3).
      active.finish_retention(queue->get_immediate_command_list());
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
    ++input.epoch; last_accepted = 0; // A new scope; an observation loss would keep placement.
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
    ++input.epoch; last_accepted = 0; // A new scope; an observation loss would keep placement.
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
    // Alpha at or above the soft pin knee pins exactly. A fainter texel (the
    // 1/64 corner) still places the plane but pins only in proportion.
    const float knee = 1.f / float(sunshine_game3d::shader_marker(sunshine_game3d::renderer::shader_source(), "SUNSHINE_UI_SOFT_PIN_GAIN"));
    const auto pinned = [&](const result &value, float inverse, const sunshine_game3d::render_parameters &parameters,
        const std::vector<float> &coverage) {
      const double expected = plane_parallax(gpu, parameters, {ui_plane_mode::depth_midpoint, inverse});
      bool seen = false; float first{};
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        if (!std::isfinite(coverage[at(x, y)]) || coverage[at(x, y)] < knee) continue;
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
    const float gain = float(sunshine_game3d::shader_marker(sunshine_game3d::renderer::shader_source(), "SUNSHINE_UI_SOFT_PIN_GAIN"));
    const bool binary = std::all_of(alpha.begin(), alpha.end(),
      [gain](float a) { return !(std::isfinite(a) && a > 0) || a * gain >= 1.f; });
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
      // A mask without finite alpha below the soft pin knee keeps the binary
      // distance rule's field bit for bit. The screen plane hides the rule's
      // rounding, so compare on a display-fraction plane; a power-of-two width
      // makes 0.5/W exact and hides it too (use, for example, 250x142).
      if (binary && gpu.has_control &&
          gpu.control_renderer.active_shader_source().find("Sunshine_UIPlaneMode == 5u") != std::string_view::npos) {
        const sunshine_game3d::ui_plane_parameters plane{sunshine_game3d::ui_plane_mode::display_fraction, .25f};
        const auto current = gpu.render(true, sign, false, false, false, {}, {}, plane);
        const auto frozen = gpu.render(true, sign, false, true, false, {}, {}, plane);
        require(current.field.bytes != off.field.bytes || empty, label + ": the display-fraction plane did not pin");
        require(current.field.bytes == frozen.field.bytes && current.output.bytes == frozen.output.bytes,
          label + ": a binary UI mask differs from the frozen control shader");
        std::printf("PASS %s sign=%d binary mask matches the frozen control on a display-fraction plane%s\n", label.c_str(), sign,
          gpu.width & (gpu.width - 1) ? "" : " (power-of-two width: rounding not exercised)");
      }
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
    // Only an accepted candidate decides, and then at any coverage (S1, P1):
    // a selective mask protects its pixels, a full one pins the frame flat and
    // an empty one protects nothing. An unaccepted one decides nothing.
    const auto verify = [&](unsigned mask, api::resource_view retained, bool decides = true) {
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      const auto actual = gpu.render(true, 1, false, false, source.retained, retained, {}, {}, nullptr, &source);
      const bool protected_pixels = decides && mask != 2;
      require(exact(actual, protected_pixels ? on[mask] : off[mask]),
        "Automatic alpha detection changed exact field/SBS, decided unaccepted or ignored accepted coverage (mask " +
        std::to_string(mask) + (decides ? ", accepted" : ", unaccepted") + ", candidates " +
        std::to_string(gpu.renderer.consumed_detection().candidates) + ", accepted " +
        std::to_string(gpu.renderer.consumed_detection().accepted) + ", stored " + policy.stored() + ")");
      const auto resolved = gpu.renderer.diagnostics().ui_source;
      require(resolved.handle, "Auto did not expose its actual resolved GPU mask");
      const auto selected = gpu.read(resolved);
      require(selected.format == DXGI_FORMAT_R32_FLOAT && selected.width == gpu.width && selected.height == gpu.height,
        "Auto diagnostic is not the resolved full-resolution R32 mask");
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        const auto value = selected.channel(x, y, 0);
        const bool wanted = protected_pixels && masks[mask][size_t(y) * gpu.width + x] > 0.f;
        require(std::isfinite(value) && value >= 0.f && value <= 1.f && (value > 0.f) == wanted,
          "Auto mask decided without acceptance, retained old coverage or lost accepted UI pixels");
      }
    };
    // All decisions come from the real GPU inputs. No review, provider approval
    // mutation or injected CPU coverage statistics are used by this fixture.
    // Current color alpha and the Backbuffer are inferred sources, the tagged
    // UI color a declared one (A1).
    const ui_selection::kind kinds[]{ui_selection::kind::current, ui_selection::kind::backbuffer, ui_selection::kind::ui_color};
    for (unsigned kind = 0; kind != 3; ++kind) {
      source.retained = kind != 0; source.dedicated_mask = kind == 2;
      ++source.epoch;
      const auto signature = *ui_selection::signature::parse(gpu.key(kinds[kind]));
      const auto retained = [&](unsigned mask) { return kind ? gpu.retain_mask(masks[mask]) : api::resource_view{}; };
      require(!policy.accepts(signature), "The fixture accepted a source before its evidence");
      // No decision crosses the epoch (identity) change (T1; an observation
      // revision alone would keep the chain): the first frame of each
      // kind starts a new chain, so the previous kind's accepted source, now
      // missing, is never reused.
      const auto starts_chain = [&] {
        require((gpu.renderer.consumed_detection().flags & ui_detection::per_frame_hold_reset) &&
            !(gpu.renderer.consumed_detection().flags & ui_detection::per_frame_accepted_missing),
          "The first frame after a scope change did not start a new T1 chain");
      };
      if (kind == 1) {
        // Acceptance an earlier session earned decides from the first frame.
        gpu.pattern(masks[1], true);
        verify(1, retained(1), false);
        starts_chain();
        require(policy.restore(gpu.key(kinds[kind])).restored == 1, "The Backbuffer signature was not restored");
      } else {
        // Earned live: the declared tag by its first valid selective sample,
        // current alpha by alpha_trust_samples samples over alpha_trust_span_ms.
        // Until the render after that sample's read nothing decides.
        const auto first = source.now_ms + 100;
        unsigned frames = 0;
        while (!policy.accepts(signature)) {
          require(++frames <= 40, "A valid selective source never earned acceptance");
          const auto mask = retained(1);
          gpu.pattern(masks[1], true);
          verify(1, mask, false);
          if (frames == 1) starts_chain();
        }
        require(kind == 2 ? frames == 2 : source.now_ms - first >= alpha_trust_span_ms,
          "A source earned acceptance from too little evidence: " + std::to_string(frames) + " frames");
      }
      // Switching from selective directly to opaque/empty: an accepted source
      // decides each frame's own coverage, a full one as a flat frame.
      for (const unsigned mask : {1u, 0u, 2u, 1u}) {
        const auto view = retained(mask);
        gpu.pattern(masks[mask], true);
        verify(mask, view);
      }
    }
    {
      const auto counted = policy.counters();
      require(counted[ui_counter::none + ui_no_mask::unaccepted] && !counted[ui_counter::untrusted_inferred] &&
          counted[ui_counter::trust_earned] >= 2 && counted[ui_counter::trust_restored] == 1,
        "Selective sources before acceptance were not counted as unaccepted, or an inferred source decided unaccepted");
    }
    if (gpu.color == 2) {
      // More than 1% of malformed pixels makes accepted alpha V1-invalid: it
      // decides nothing that frame. The first such frame (the accepted tag
      // of the previous real frame is missing too) has no decision of its own
      // and reuses that frame's selective tag decision once (T1); the next
      // ones have no mask.
      source.retained = source.dedicated_mask = false;
      bool first = true;
      for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity(), -1.f, 2.f}) {
        auto malformed = masks[1];
        std::fill_n(malformed.begin(), 3 * gpu.width, invalid);
        gpu.pattern(malformed, true);
        source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
        require(exact(gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, &source), first ? on[1] : off[1]),
          first ? "The first V1-invalid frame did not reuse the previous real frame's decision once" :
            "Automatic alpha accepted nonfinite or out-of-range input as a valid UI mask");
        first = false;
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
    policy.set_automatic();
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
    report << "automatic-ui D3D11 current_captured_dedicated=1 unaccepted_decides_nothing=1 declared_one_sample=1 inferred_two_seconds=1 restored_acceptance=1 accepted_full_flat=1 accepted_empty=1 selective_without_review=1 immediate_bad_frame_rejection=1 resolved_gpu_mask_checked=1 current_RGB_preserved=1 renderer_recreation=1 manual_off_wins=1 explicit_field_and_sbs_parity=1\n";
    std::puts("PASS D3D11 automatic UI: nothing decides before acceptance (a tag after one sample, presented alpha after 2 s, or restored), then each frame's own coverage decides (selective, flat or empty) with exact field/SBS, current RGB and manual Off preserved");
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
    // Exact counters (game3d_ui_counters.h): every render below, whether it
    // requested detection, how it ran, and whether it was a generated Present
    // (T1: it either shows the real frame's decision, held.generated, or has
    // no mask, held.none).
    struct counted_frame {
      std::uint64_t tick;
      bool requested;
      ui_detection_snapshot::run_state state;
      bool generated;
    };
    std::vector<counted_frame> counted;
    const auto note = [&] {
      const auto *in = ui.detection;
      counted.push_back({source.now_ms, policy.decision().state != alpha_auto_state::manual_off,
        gpu.renderer.consumed_detection().state, in && in->hold_previous});
    };
    const auto run = [&](bool accepted, const char *label, bool inspect_mask = true) {
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      const auto actual = gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      note();
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
    // A1: a HUD-less pair is a declared source. Its first valid selective
    // sample from an exact pair accepts it; an inexact pair never earns, and
    // before acceptance it decides nothing (S1).
    const auto hudless_signature = *ui_selection::signature::parse(gpu.key(ui_selection::kind::hudless));
    ui_detection_inputs earning;
    earning.hudless = correct;
    ui.detection = &earning;
    run(false, "an unaccepted inexact HUD-less pair decides nothing");
    run(false, "an inexact HUD-less pair never earns acceptance");
    require(!policy.accepts(hudless_signature), "An inexact HUD-less pair earned acceptance");
    earning.hudless_exact = true;
    run(false, "an unaccepted exact HUD-less pair decides nothing yet");
    run(false, "acceptance applies from the render after its sample is read");
    require(policy.accepts(hudless_signature), "A valid selective exact HUD-less sample did not accept the pair");
    ui.detection = nullptr;
    // A frame without protection ends the decision chain (T1).
    source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
    gpu.render(false, 1, false, false, false, {}, {}, {}, nullptr, &source);
    // An accepted change set decides from an inexact pair too (counted as
    // inexact_difference) until frame identity is exact (E2).
    run(true, "matching final/HUDless");
    // T1: an accepted HUD-less pair that is invalid this frame leaves the
    // real frame without a decision of its own. It reuses the previous real
    // frame's decision and mask once, and then has no mask.
    ui.view = shifted;
    run(true, "a scene-wide camera shift after a good pair reuses its decision once");
    run(false, "scene-wide camera shift after a good pair");
    ui.view = flattened_ui;
    run(false, "identical final/HUDless has no separator");
    ui.view = correct;
    run(true, "matching pair recovers automatically");
    run(true, "matching pair publishes its detection evidence");
    {
      // The CPU sample of the previous frame's decision explains which check passed.
      const auto sample = gpu.renderer.consumed_alpha_auto();
      const auto &evidence = sample.evidence;
      require(sample.source_kind == 5 && evidence.candidates == 16u && evidence.hudless_changed == sample.covered &&
          evidence.hudless_changed && evidence.hudless_invalid == 0 &&
          evidence.hudless_unchanged + evidence.hudless_changed <= sample.pixels &&
          evidence.hudless_unchanged * 4u >= sample.pixels * 3u && evidence.matching_tiles >= 128u,
        "HUD-less detection evidence does not describe the accepted decision");
    }
    // The counters' window starts after the first commits, which may also
    // carry an earlier test's last Auto frame.
    const auto window_start = policy.counters();
    // Reproduce the useful part of Hogwarts' discovery: a transport-valid
    // UIColorAndAlpha can be an opaque full scene. Continue to HUDless in the
    // same frame instead of letting that unusable higher candidate win.
    ui_detection_inputs inputs;
    inputs.masks[1] = flattened_ui; inputs.hudless = correct; inputs.current_color = true;
    inputs.real_frame = 7;
    ui.detection = &inputs;
    run(true, "flattened explicit UI falls through to HUDless");
    // Frame generation presents interpolated frames between a HUD-less tag and
    // its real frame. They never detect (T1): each shows the decision of the
    // real frame it shows, the decided one or the next, by the HUD-less tag's
    // present generation, without a multiplier cap.
    inputs.hudless = {}; inputs.hold_previous = true; inputs.real_frame = 8;
    run(true, "generated present holds the real frame's HUD-less mask");
    {
      // A held mask reports the detection run that made it.
      const auto held = gpu.renderer.consumed_detection();
      require(held.state == ui_detection_snapshot::run_state::held && held.held_presents == 1 &&
          held.candidates == (2u | 8u | 16u) && held.flags == 0u && held.stored_flags == 0u,
        "A held HUD-less mask lost the detection constants that made it");
    }
    run(true, "second generated present still holds");
    run(true, "third generated present still holds");
    run(true, "a fourth generated present still holds: no multiplier cap");
    // A Present classified generated that shows neither the decided real
    // frame nor the next one has no mask, and the chain ends (fail safe).
    inputs.real_frame = 9;
    run(false, "a generated present beyond the next real frame has no mask", false);
    require(gpu.renderer.consumed_detection().state == ui_detection_snapshot::run_state::inactive,
      "A generated present beyond the T1 tag bound reported a held mask");
    inputs.real_frame = 8;
    run(false, "a generated present after the chain ended has no mask", false);
    inputs.hold_previous = false; inputs.hudless = correct; inputs.real_frame = 10;
    run(true, "the next real frame detects again");
    require(gpu.renderer.consumed_detection().state == ui_detection_snapshot::run_state::ran &&
        !gpu.renderer.consumed_detection().held_presents &&
        (gpu.renderer.consumed_detection().flags & ui_detection::per_frame_hold_reset),
      "A fresh detection after the chain ended reported a held mask or no hold reset");
    inputs.real_frame = 0;
    inputs.hudless = shifted;
    run(true, "the first unsuitable frame reuses the previous real frame's decision once");
    run(false, "all candidates unsuitable");
    // An unaccepted UIAlpha neither decides nor blocks (S1), and a full one
    // never earns acceptance (A1).
    inputs.masks[0] = opaque_alpha;
    run(false, "an unaccepted full-scene UIAlpha is no full-screen UI");
    inputs.hudless = correct;
    run(true, "an unaccepted full-scene UIAlpha falls through to HUDless");
    inputs.masks[0] = {};
    policy.set_manual(false);
    run(false, "manual Off wins over a valid pair", false);
    require(gpu.renderer.consumed_alpha_auto().state == alpha_auto_state::manual_off,
      "HUDless detection changed the user's manual Off setting");
    policy.set_automatic();
    run(true, "Auto resumes without review");
    // A HUD-less capture from another queue can complete after its own frame
    // was presented. It pairs with that frame's retained color, never with the
    // current frame, and only within the retained history.
    inputs = {}; inputs.masks[1] = flattened_ui; inputs.current_color = true;
    const auto late = [&](const std::vector<unsigned char> &color, std::uint32_t presents_ago, bool accepted,
        const char *label, api::resource_view pair = {}) {
      gpu.original = color;
      gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), gpu.width * bpp, 0);
      inputs.hudless = correct; inputs.hudless_presents_ago = presents_ago; inputs.hudless_pair = pair;
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.renderer.begin_present();
      gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      note();
      const auto selected = gpu.read(gpu.renderer.diagnostics().ui_source);
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        const auto value = selected.channel(x, y, 0);
        require(std::isfinite(value) && (value > 0.f) == (accepted && mask[size_t(y) * gpu.width + x] > 0.f),
          std::string(label) + ": late HUD-less mask does not match its own frame's HUD");
      }
    };
    // A real frame whose accepted HUD-less pair is missing or invalid reuses
    // the previous real frame's decision once (T1); the second such frame
    // shows the pairing's own outcome.
    late(final_color, 1, true, "late HUD-less before any color is retained reuses the previous decision once");
    require(!(gpu.renderer.consumed_detection().candidates & ui_detection::candidate::hudless) &&
        (gpu.renderer.consumed_detection().flags & ui_detection::per_frame_accepted_missing),
      "A late HUD-less image without retained color was offered, or its accepted pair was not missing");
    late(moved_scene, 1, true, "late HUD-less pairs with the previous frame's retained color");
    late(moved_scene, 2, true, "two Presents late still pairs within retained history");
    late(moved_scene, 3, true, "a frame no longer retained reuses the previous decision once");
    late(moved_scene, 3, false, "a frame no longer retained is unpaired");
    late(final_color, 0, true, "an on-time capture still uses the current color");
    late(moved_scene, 0, true, "a mismatched on-time pair reuses the previous decision once");
    late(moved_scene, 0, false, "an on-time capture never pairs with an earlier color");
    // The game's tagged Backbuffer from the HUD-less image's own batch is its
    // exact pair on any Present, whatever the current frame shows.
    late(moved_scene, 0, true, "HUD-less pairs exactly with its batch's tagged Backbuffer", flattened_ui);
    late(moved_scene, 2, true, "a batch pair takes precedence over Present counting", flattened_ui);
    late(final_color, 0, true, "a mismatched batch image reuses the previous decision once", shifted);
    late(final_color, 0, false, "a mismatched batch image is rejected", shifted);
    // A menu or title screen covers the whole frame while the game still renders
    // its scene: HUD-less shows that scene, nearly every pixel differs, and the
    // frame stays flat. A black HUD-less image is not a scene and flattens nothing.
    auto menu = hudless;
    for (auto &byte : menu) byte ^= 0x80;
    opaque(menu);
    std::vector<unsigned char> black(hudless.size(), 0);
    opaque(black);
    const auto dark = color_view(black);
    const auto full_frame = [&](api::resource_view hudless_view, bool exact, bool flat, const char *label) {
      gpu.original = menu;
      gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), gpu.width * bpp, 0);
      inputs.hudless = hudless_view; inputs.hudless_presents_ago = 0; inputs.hudless_pair = {}; inputs.hudless_exact = exact;
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.renderer.begin_present();
      gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      note();
      const auto selected = gpu.read(gpu.renderer.diagnostics().ui_source);
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        const auto value = selected.channel(x, y, 0);
        require(std::isfinite(value) && (flat ? value == 1.f : value == 0.f), std::string(label) + ": wrong full-frame UI mask");
      }
    };
    full_frame(correct, false, false, "an unverified pair differing everywhere is rejected, not flattened");
    full_frame(correct, true, true, "a full-frame menu over a lit scene stays flat");
    // With frame generation on, a real frame outside the tag batch pairs only by
    // Present counting. Right after an exact decision the T1 grace reuses that
    // decision once instead of flipping the menu to 3D; the next such frame
    // has no mask.
    full_frame(correct, false, true, "an inexact real frame reuses the exact full-frame decision once");
    full_frame(correct, false, false, "the grace is spent: a second inexact frame has no mask");
    full_frame(correct, false, false, "an inexact frame after an inexact decision is rejected");
    full_frame(dark, true, false, "a black HUD-less image never flattens the frame");
    full_frame(correct, true, true, "the next exact frame decides again");
    full_frame(dark, true, true, "a black HUD-less frame right after it reuses the exact decision once");
    full_frame(correct, false, false, "an inexact frame after a spent grace stays empty");
    // A channel the session trusts as UI coverage is the mask whatever it covers:
    // nothing, a HUD or a whole menu. Selective coverage earns trust; an exact
    // HUD-less pair showing the scene under a full channel takes it away.
    enum class expected { empty, flat, hud };
    const auto trust_frame = [&](const std::vector<unsigned char> &color, api::resource_view alpha0, api::resource_view alpha1,
        api::resource_view hudless_view, expected want, const char *label, api::resource_view alpha2 = {}) {
      gpu.original = color;
      gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), gpu.width * bpp, 0);
      inputs = {}; inputs.masks[0] = alpha0; inputs.masks[1] = alpha1; inputs.masks[2] = alpha2;
      inputs.hudless = hudless_view; inputs.hudless_exact = hudless_view.handle != 0;
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.renderer.begin_present();
      gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      note();
      if (!label) return;
      const auto selected = gpu.read(gpu.renderer.diagnostics().ui_source);
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        const auto value = selected.channel(x, y, 0);
        const bool hud = mask[size_t(y) * gpu.width + x] > 0.f;
        require(std::isfinite(value) && (want == expected::flat ? value > 0.f : want == expected::empty ? value == 0.f :
          (value > 0.f) == hud), std::string(label) + ": wrong trusted-alpha UI mask");
      }
    };
    const auto nearly_opaque = red_view(std::vector<float>(pixels, .99f)), no_alpha = red_view(std::vector<float>(pixels, 0.f));
    const auto ui_alpha_signature = *ui_selection::signature::parse(gpu.key(ui_selection::kind::ui_alpha, DXGI_FORMAT_R32_FLOAT));
    require(!policy.accepts(ui_alpha_signature), "The fixture accepted the UIAlpha channel too early");
    trust_frame(menu, opaque_alpha, {}, {}, expected::empty, "an unaccepted opaque UIAlpha is not full-screen UI");
    // UIAlpha is a declared source: its first valid selective sample accepts
    // it (A1), and it decides from the render after that sample is read.
    trust_frame(menu, explicit_alpha, {}, {}, expected::empty, "an unaccepted selective UIAlpha decides nothing");
    trust_frame(menu, explicit_alpha, {}, {}, expected::empty, "acceptance waits for its sample to be read");
    require(policy.accepts(ui_alpha_signature), "A selective UIAlpha sample did not accept exactly its channel");
    trust_frame(menu, explicit_alpha, {}, {}, expected::hud, "an accepted UIAlpha yields its HUD mask");
    trust_frame(menu, opaque_alpha, {}, {}, expected::flat, "an accepted channel covering everything is a full-screen menu");
    trust_frame(menu, opaque_alpha, {}, {}, expected::flat, "an accepted channel publishes its evidence");
    {
      const auto sample = gpu.renderer.consumed_alpha_auto();
      require(sample.source_kind == 1 && sample.covered == pixels && sample.evidence.accepted == ui_detection::candidate::ui_alpha &&
          (sample.evidence.valid_bits & ui_detection::candidate::ui_alpha) && sample.evidence.alpha_covered[0] == pixels,
        "Accepted-alpha evidence does not describe the decision");
    }
    trust_frame(menu, nearly_opaque, {}, {}, expected::flat, "an accepted channel above zero everywhere is a full-screen menu");
    trust_frame(menu, no_alpha, {}, {}, expected::empty, "an accepted channel covering nothing means no UI");
    // Acceptance belongs to its own candidate: an unaccepted tag in its place
    // decides nothing, and the T1 grace reuses the missing UIAlpha's empty
    // decision once.
    trust_frame(menu, {}, flattened_ui, {}, expected::empty, "acceptance belongs to its own candidate");
    trust_frame(menu, explicit_alpha, {}, {}, expected::hud, "an accepted channel still yields its HUD mask");
    // A pause menu that tints the live scene changes too many pixels for a HUD
    // mask and too few for full-screen UI: only an accepted channel can say.
    auto tinted = hudless;
    for (size_t i = 0; i != pixels * 6 / 10; ++i)
      for (unsigned byte = 0; byte != 3; ++byte) tinted[i * bpp + byte] ^= 0x80;
    opaque(tinted);
    trust_frame(tinted, opaque_alpha, {}, correct, expected::flat, "an accepted channel decides a menu that tints the scene");
    // An accepted channel missing from a frame (an observation loss refuses its
    // capture until the next tag) leaves the real frame without a decision of
    // its own: the T1 grace reuses the previous one once, then no mask.
    trust_frame(tinted, {}, {}, correct, expected::flat, "a missing accepted channel reuses its mask once");
    trust_frame(tinted, {}, {}, correct, expected::empty, "the grace is spent: a second missing frame has no mask");
    trust_frame(tinted, {}, {}, correct, expected::empty, "without an accepted channel a tinting menu stays 3D");
    // A2: a declared source is never judged. A full accepted UIAlpha keeps
    // deciding while an exact pair shows the lit scene under it (P1).
    for (unsigned frame = 0; frame < 25; ++frame)
      trust_frame(final_color, opaque_alpha, {}, correct, expected::flat,
        frame ? nullptr : "a declared full UIAlpha keeps deciding");
    require(policy.accepts(ui_alpha_signature), "A declared UIAlpha was judged by an exact pair");
    // A2: a full accepted inferred alpha (the Backbuffer) while a valid exact
    // pair shows the lit scene unchanged under it is contradicted one way. It
    // wins at first, then loses acceptance 2 s into the contradicting run.
    const auto backbuffer_signature = *ui_selection::signature::parse(gpu.key(ui_selection::kind::backbuffer));
    require(policy.restore(gpu.key(ui_selection::kind::backbuffer)).restored == 1, "The Backbuffer key was not restored");
    trust_frame(final_color, {}, {}, correct, expected::flat, "an accepted Backbuffer decides before any contradiction",
      flattened_ui);
    for (unsigned frame = 0; frame < 25; ++frame)
      trust_frame(final_color, {}, {}, correct, expected::hud, nullptr, flattened_ui);
    require(!policy.accepts(backbuffer_signature), "An exact pair showing the lit scene did not revoke the Backbuffer");
    trust_frame(final_color, {}, {}, correct, expected::hud, "after losing acceptance the HUD-less pair decides",
      flattened_ui);
    trust_frame(final_color, {}, {}, correct, expected::hud, "the HUD-less pair keeps deciding", flattened_ui);
    {
      // Between the window's two commits every render is one of the frames
      // noted above: Auto frames detect (the T1 grace included), generated
      // Presents show a real frame's decision (held.generated) or have no
      // mask (held.none), or nothing ran.
      const auto end = policy.counters();
      const auto delta = end - window_start;
      ui_counters expected;
      for (const auto &frame : counted) {
        if (frame.tick > window_start.through_ms && frame.tick <= end.through_ms && frame.requested) {
          ++expected[ui_counter::auto_frames];
          if (frame.state == ui_detection_snapshot::run_state::held) ++expected[ui_counter::held_generated];
          else if (frame.state == ui_detection_snapshot::run_state::ran) ++expected[ui_counter::detection_frames];
          else if (frame.generated) ++expected[ui_counter::held_none];
          else ++expected[ui_counter::inactive_no_candidates];
        }
      }
      std::uint64_t decided = 0, none = 0;
      for (std::uint32_t source_kind = 0; source_kind != ui_counter_word::decided_count; ++source_kind)
        decided += delta.decided(source_kind);
      for (std::size_t reason = 0; reason != ui_no_mask::count; ++reason) none += delta[ui_counter::none + reason];
      require(end.reconciled() && end.through_ms + 200 >= source.now_ms &&
          delta[ui_counter::auto_frames] == expected[ui_counter::auto_frames] &&
          delta[ui_counter::detection_frames] == expected[ui_counter::detection_frames] &&
          delta[ui_counter::held_generated] == expected[ui_counter::held_generated] &&
          delta[ui_counter::held_none] == expected[ui_counter::held_none] && delta.inactive() == expected.inactive(),
        "D3D11 exact UI counters differ from the noted renders");
      require(expected[ui_counter::held_generated] >= 4 && expected[ui_counter::held_none] >= 2,
        "The D3D11 counter window lost a scripted generated Present");
      // Every reused frame is a detection frame that applied the previous real
      // frame's decision; the reused count is exact on the GPU.
      require(decided == delta[ui_counter::detection_frames] && none == delta.decided(0) && delta.decided(1) &&
          delta.decided(3) && delta.decided(5) && delta.decided(6) && delta[ui_counter::contradicted] &&
          delta[ui_counter::inexact_difference] && delta[ui_counter::reused] >= 8 &&
          delta[ui_counter::reused] < delta[ui_counter::detection_frames] &&
          !delta[ui_counter::presented_over_dedicated] && !delta[ui_counter::untrusted_inferred] &&
          delta[ui_counter::none + ui_no_mask::unaccepted] && end[ui_counter::trust_earned] >= 2 &&
          end[ui_counter::trust_revoked_exact] >= 1 && !delta[ui_counter::trust_revoked_declared],
        "D3D11 exact UI counters lost a decided source, a no-mask reason, a reuse or a trust event");
      std::printf("PASS D3D11 exact UI counters: %llu Auto frames through %llu ms reconcile; held generated=%llu none=%llu "
        "reused=%llu by frame; decisions and no-mask reasons sum exactly\n",
        static_cast<unsigned long long>(delta[ui_counter::auto_frames]), static_cast<unsigned long long>(end.through_ms),
        static_cast<unsigned long long>(delta[ui_counter::held_generated]),
        static_cast<unsigned long long>(delta[ui_counter::held_none]),
        static_cast<unsigned long long>(delta[ui_counter::reused]));
    }
    inputs = {};
    report << "automatic-hudless D3D11 final_alpha_opaque=1 paired_HUD_exact=1 scene_wide_motion_rejected=1 identical_pair_empty=1 same_frame_bad_pair_rejection=1 flattened_explicit_fallback=1 generated_present_hold=1 t1_tag_bound=1 late_retained_pair=1 tagged_backbuffer_pair=1 full_frame_ui=1 t1_grace_once=1 hudless_earned_exact=1 accepted_alpha=1 declared_never_judged=1 one_way_revocation=1 current_RGB_preserved=1 manual_off_wins=1 no_review=1\n";
    std::puts("PASS D3D11 HUDless auto: exact HUD difference, scene-wide motion/identical rejection, flattened candidate fallback, generated Presents showing the real frame's decision within the T1 tag bound and without a cap, late pairing with retained color, exact tagged-Backbuffer pairing, full-frame UI flattening, the T1 grace reusing a real frame's decision once for a missing or invalid accepted source, acceptance earned from one selective sample (an exact HUD-less pair, UIAlpha) before anything decides, a declared UIAlpha never judged, one-way revocation of a full accepted Backbuffer, and manual Off without review");
  }
  // V1 and S1 for the offscreen UI layer (docs/reshade-sbs.md, UI decision
  // framework). Stellar Blade draws its SDR scene image into the cleared
  // target that holds its UI layer in HDR: color everywhere, alpha nowhere.
  // Such a layer fails the premultiplied bound on more than 1% of its pixels,
  // so it is V1-invalid: it neither decides nor blocks, whether accepted or
  // not, and only accepted candidates decide. A tagged UI color is checked for
  // range only and, accepted, blocks presented alpha; a layer with color on at
  // most 1% of pixels is valid and decides.
  void verify_layer_validity(fixture &gpu, std::ostream &report) {
    using namespace sunshine_game3d;
    const auto pixels = size_t(gpu.width) * gpu.height;
    const auto bpp = gpu.color == 2 ? 8u : 4u;
    std::vector<float> hud(pixels, 0.f), none(pixels, 0.f);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) hud[size_t(y) * gpu.width + x] = 1.f;
    // The presented frame: the scene with HUD alpha, as current color and as
    // the tagged Backbuffer; an opaque copy is a Backbuffer without UI alpha.
    gpu.pattern(hud);
    const auto presented = gpu.original;
    auto opaque = presented, scene = presented;
    for (size_t i = 0; i != pixels; ++i) {
      if (gpu.color == 2) {
        const auto one = half_bits(1.f), zero = half_bits(0.f);
        std::memcpy(opaque.data() + i * bpp + 6, &one, 2);
        std::memcpy(scene.data() + i * bpp + 6, &zero, 2);
      } else {
        opaque[i * bpp + 3] = 255;
        scene[i * bpp + 3] = 0;
      }
    }
    std::vector<ComPtr<ID3D11Texture2D>> textures;
    std::vector<ComPtr<ID3D11ShaderResourceView>> views;
    const auto view_of = [&](const std::vector<unsigned char> &bytes) {
      D3D11_TEXTURE2D_DESC desc{}; gpu.source->GetDesc(&desc); desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data{bytes.data(), gpu.width * bpp, 0};
      ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> view;
      checked(gpu.device->CreateTexture2D(&desc, &data, &texture), "layer-without-alpha texture");
      checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "layer-without-alpha view");
      const api::resource_view result{reinterpret_cast<std::uint64_t>(view.Get())};
      textures.push_back(std::move(texture)); views.push_back(std::move(view)); return result;
    };
    // A layer of gray color and alpha per pixel in the fixture's format.
    const auto layer_of = [&](const std::function<std::array<float, 2>(size_t)> &value) {
      std::vector<unsigned char> bytes(pixels * bpp);
      for (size_t i = 0; i != pixels; ++i) {
        const auto pixel = value(i);
        const float channels[4]{pixel[0], pixel[0], pixel[0], pixel[1]};
        for (unsigned c = 0; c != 4; ++c) {
          if (gpu.color == 2) { const auto half = half_bits(channels[c]); std::memcpy(bytes.data() + i * bpp + c * 2, &half, 2); }
          else bytes[i * bpp + c] = static_cast<unsigned char>(std::lround(std::clamp(channels[c], 0.f, 1.f) * 255));
        }
      }
      return view_of(bytes);
    };
    // The scene image with zero alpha: color on every pixel, alpha nowhere.
    const auto scene_layer = view_of(scene);
    const auto backbuffer = view_of(presented), opaque_backbuffer = view_of(opaque);
    const size_t speck = pixels / 200; // 0.5% of pixels.
    const auto speck_layer = layer_of([&](size_t i) { return std::array<float, 2>{i < speck ? 1.f : 0.f, 0.f}; });
    // HUD alpha with color beside it on the top eighth of the frame: ambiguous.
    const auto ambiguous_layer = layer_of([&](size_t i) {
      return std::array<float, 2>{hud[i] > 0.f || i < pixels / 8 ? 1.f : 0.f, hud[i]};
    });
    const auto empty_layer = layer_of([](size_t) { return std::array<float, 2>{0.f, 0.f}; });
    const auto layer_flags = ui_detection::layer_detection_flags(gpu.color == 2);

    alpha_auto_source source;
    source.now_ms = source.tick_ms = 1000;
    source.epoch = 37; source.revision = 1; source.sequence = 1;
    ui_detection_inputs inputs;
    ui_render_input ui;
    ui.automatic = &source; ui.detection = &inputs;
    gpu.original = presented;
    gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), gpu.width * bpp, 0);
    const auto frame = [&](const char *label, const std::vector<float> *want, std::uint32_t source_kind = ~0u) {
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.renderer.begin_present();
      gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      if (!want) return;
      const auto selected = gpu.read(gpu.renderer.diagnostics().ui_source);
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x) {
        const auto value = selected.channel(x, y, 0);
        require(std::isfinite(value) && (value > 0.f) == ((*want)[size_t(y) * gpu.width + x] > 0.f),
          std::string(label) + ": wrong UI mask");
      }
      // The sample read this frame is the previous frame's, of the same inputs.
      const auto decided = gpu.renderer.consumed_alpha_auto().source_kind;
      if (source_kind != ~0u)
        require(decided == source_kind, std::string(label) + ": wrong deciding source " + std::to_string(decided));
    };
    const auto twice = [&](const char *label, const std::vector<float> &want, std::uint32_t source_kind) {
      frame(nullptr, nullptr);
      frame(label, &want, source_kind);
    };
    // Each case restores acceptance of exactly the listed kinds, then offers
    // the offscreen UI layer in its own slot (t7, 0x40), a tagged UI color, the
    // Backbuffer and current color alpha as given.
    struct offer {
      api::resource_view layer{}, tag{}, backbuffer{};
      bool current{};
    };
    const auto layout = [&](alpha_auto_policy &policy, std::initializer_list<ui_selection::kind> accepted, const offer &offered) {
      for (const auto kind : accepted) require(policy.restore(gpu.key(kind)).restored == 1, "A fixture key was not restored");
      source.session = &policy;
      inputs = {};
      inputs.layer = offered.layer; inputs.layer_flags = offered.layer.handle ? layer_flags : 0u;
      inputs.masks[1] = offered.tag; inputs.masks[2] = offered.backbuffer; inputs.current_color = offered.current;
    };
    using ui_selection::kind;
    const std::vector<float> all(pixels, 1.f);
    // (a) An accepted layer holding the scene without alpha is V1-invalid: it
    // neither decides nor blocks. Unaccepted current alpha decides nothing;
    // accepted current alpha decides its HUD.
    alpha_auto_policy a;
    layout(a, {kind::ui_layer}, {scene_layer, {}, {}, true});
    twice("an accepted invalid layer and unaccepted current alpha decide nothing", none, 0);
    alpha_auto_policy a2;
    layout(a2, {kind::ui_layer, kind::current}, {scene_layer, {}, {}, true});
    twice("an accepted invalid layer lets accepted current alpha decide", hud, 4);
    // (b) The same pixels as an accepted tagged UI color: the tag is checked
    // for range only, so it is a valid empty mask that decides no UI and, as a
    // declared alpha, keeps accepted presented alpha out.
    alpha_auto_policy b;
    layout(b, {kind::ui_color, kind::current}, {{}, scene_layer, {}, true});
    twice("an accepted valid empty tag decides no UI and keeps presented alpha out", none, 2);
    // (c) Color without alpha on at most 1% of pixels is glow: the accepted
    // layer is valid and decides, here no UI.
    alpha_auto_policy c;
    layout(c, {kind::ui_layer, kind::current}, {speck_layer, {}, {}, true});
    twice("an accepted layer with sparse glow decides", none, ui_detection::source_layer);
    // (d) A layer with UI alpha and more than 1% invalid pixels is invalid as
    // well: it neither decides nor blocks accepted presented alpha.
    alpha_auto_policy d;
    layout(d, {kind::ui_layer, kind::current}, {ambiguous_layer, {}, {}, true});
    twice("an accepted ambiguous layer blocks nothing", hud, 4);
    // (e) A clean empty accepted layer means no UI.
    alpha_auto_policy e;
    layout(e, {kind::ui_layer}, {empty_layer, {}, {}, true});
    twice("a clean empty accepted layer decides no UI", none, ui_detection::source_layer);
    // (f) An accepted opaque Backbuffer beside an invalid layer pins the frame
    // flat (P1): acceptance, not a full-frame bound, decides that (A1 keys it
    // by color space).
    alpha_auto_policy f;
    layout(f, {kind::ui_layer, kind::backbuffer}, {scene_layer, {}, opaque_backbuffer, false});
    twice("an accepted opaque Backbuffer beside an invalid layer pins flat", all, 3);
    // A generated Present without the tagged Backbuffer shows the real frame's
    // decision (T1): it never detects, so the accepted V1-invalid layer it
    // still offers cannot stop it.
    inputs.masks[2] = {}; inputs.hold_previous = true;
    frame("a generated Present holds the flat Backbuffer mask", &all);
    require(gpu.renderer.consumed_detection().state == ui_detection_snapshot::run_state::held,
      "An accepted V1-invalid layer stopped the generated-present hold");
    // Stellar Blade in SDR with FG: the scene image in the cleared target never
    // earns acceptance, and the accepted Backbuffer's HUD alpha decides.
    alpha_auto_policy g;
    layout(g, {kind::backbuffer}, {scene_layer, {}, backbuffer, false});
    twice("an accepted Backbuffer beside an invalid layer decides", hud, 3);
    frame("the accepted Backbuffer keeps deciding", &hud, 3);
    // From no sample (a new epoch discards it), as frame generation presents:
    // the tagged Backbuffer comes with real Presents only. Every generated
    // Present shows the real frame's Backbuffer mask instead of dropping to
    // no mask (T1), whatever key the samples carry (F1). No decision crosses
    // the epoch: the first real Present starts a new chain.
    ++source.epoch;
    for (unsigned i = 0; i != 6; ++i) {
      inputs.masks[2] = backbuffer; inputs.hold_previous = false;
      frame("a real Present with the Backbuffer decides", &hud);
      require(!i == ((gpu.renderer.consumed_detection().flags & ui_detection::per_frame_hold_reset) != 0),
        "Only the first real Present after an epoch change starts a new T1 chain");
      inputs.masks[2] = {}; inputs.hold_previous = true;
      frame("an alternating generated Present holds the Backbuffer mask", &hud);
      require(gpu.renderer.consumed_detection().state == ui_detection_snapshot::run_state::held,
        "Alternating generated Presents never held the Backbuffer mask beside an invalid layer");
    }
    require(!g.accepts(*ui_selection::signature::parse(gpu.key(kind::ui_layer))),
      "The scene image in the layer's cleared target earned acceptance");
    inputs = {};
    report << "layer-validity D3D11 invalid_layer_neither_decides_nor_blocks=1 unaccepted_current_decides_nothing=1 "
              "accepted_current_decides=1 valid_empty_tag_decides=1 sparse_glow_decides=1 ambiguous_blocks_nothing=1 "
              "empty_decides=1 accepted_full_flat=1 hold_kept=1 backbuffer_decides=1 alternating_hold=1\n";
    std::puts("PASS D3D11 layer validity (V1/S1): an invalid offscreen UI layer neither decides nor blocks, so only accepted presented alpha decides, at any coverage; a valid empty tag, a layer with sparse glow and a clean empty layer decide; holds stay beside an invalid layer");
  }
  // An offscreen UI layer has its own slot (t7, candidate 0x40) with its
  // stored flags, never a per-frame bit; a tagged UIColorAndAlpha has the UI
  // color slot and no flags (E1). Dump 3D records the detection constants
  // behind the consumed mask, the accepted candidates and the pin markers.
  void verify_layer_detection_dump(fixture &gpu, std::ostream &report, const fs::path &directory) {
    using namespace sunshine_game3d;
    const auto pixels = size_t(gpu.width) * gpu.height;
    std::vector<float> hud(pixels, 0.f);
    for (unsigned y = gpu.height / 4; y < gpu.height * 3 / 4; ++y)
      for (unsigned x = gpu.width / 3; x < gpu.width / 2; ++x) hud[size_t(y) * gpu.width + x] = 1.f;
    gpu.pattern(hud, true);
    // White UI blended over transparent black: premultiplied.
    const auto bpp = gpu.color == 2 ? 8u : 4u;
    std::vector<unsigned char> bytes(pixels * bpp);
    for (size_t i = 0; i != pixels; ++i)
      for (unsigned c = 0; c != 4; ++c) {
        if (gpu.color == 2) { const auto half = half_bits(hud[i]); std::memcpy(bytes.data() + i * bpp + c * 2, &half, 2); }
        else bytes[i * bpp + c] = hud[i] > 0.f ? 255 : 0;
      }
    D3D11_TEXTURE2D_DESC desc{}; gpu.source->GetDesc(&desc); desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA data{bytes.data(), gpu.width * bpp, 0};
    ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> view;
    checked(gpu.device->CreateTexture2D(&desc, &data, &texture), "UI layer texture");
    checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "UI layer view");
    // Both are accepted, so each decides its own raw alpha (S1).
    alpha_auto_policy policy;
    require(policy.restore(gpu.key(ui_selection::kind::ui_layer) + ',' + gpu.key(ui_selection::kind::ui_color)).restored == 2,
      "The layer and tag keys were not restored");
    alpha_auto_source source;
    source.session = &policy; source.now_ms = source.tick_ms = 1000;
    source.epoch = 31; source.revision = 1; source.sequence = 1;
    ui_detection_inputs inputs;
    const api::resource_view layer_view{reinterpret_cast<std::uint64_t>(view.Get())};
    inputs.layer = layer_view;
    inputs.layer_flags = ui_layer::detection_flags(static_cast<api::format>(desc.Format));
    ui_render_input ui;
    ui.automatic = &source; ui.detection = &inputs;
    const auto expected_flags = ui_detection::layer_detection_flags(gpu.color == 2);
    // Every automatic mask is its source's raw alpha: the one-frame-late layer
    // (stored flag 0x4) exactly like a same-frame tagged UI color.
    require(expected_flags & ui_detection::stored_late_layer, "The offscreen UI layer lost its late-layer flag");
    // per_frame: the T1 bits this detection pushes beside the stored flags.
    const auto check = [&](std::uint32_t candidate, std::uint32_t flags, std::uint32_t per_frame, const fs::path &dump,
        const char *label) {
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.render(true, 1, false, false, false, {}, dump, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      const auto consumed = gpu.renderer.consumed_detection();
      require(consumed.state == ui_detection_snapshot::run_state::ran && consumed.candidates == candidate &&
          consumed.accepted == candidate && consumed.flags == (flags | per_frame) && consumed.stored_flags == flags &&
          !(consumed.stored_flags & ui_detection::per_frame_mask),
        std::string(label) + ": wrong detection constants");
      const auto selected = gpu.read(gpu.renderer.diagnostics().ui_source);
      for (unsigned y = 0; y < gpu.height; ++y) for (unsigned x = 0; x < gpu.width; ++x)
        require(selected.channel(x, y, 0) == hud[size_t(y) * gpu.width + x], std::string(label) + ": mask is not the slot's raw alpha");
    };
    // The first detection in this epoch starts a T1 chain (hold reset).
    check(ui_detection::candidate::layer, expected_flags, ui_detection::per_frame_hold_reset, directory, "offscreen UI layer");
    std::ifstream stored(directory / "manifest.json");
    const auto manifest = nlohmann::json::parse(stored);
    const auto &replay = manifest.at("producer_metadata").at("replay");
    require(replay.at("ui_detection").at("flags") == (expected_flags | ui_detection::per_frame_hold_reset) &&
        replay.at("ui_detection").at("ran_or_held") == "ran" &&
        replay.at("ui_detection").at("candidates") == ui_detection::candidate::layer &&
        replay.at("ui_detection").at("accepted") == ui_detection::candidate::layer &&
        replay.at("ui_detection").at("candidate_layout") == ui_detection::candidate_layout &&
        replay.at("ui_detection").at("still_bits") == 0u &&
        (replay.at("ui_detection").at("rules_bits").get<std::uint32_t>() &
          (ui_detection::still::flatten | ui_detection::rules::pin_only_ui)) == 0u &&
        replay.at("ui_pin").at("decision_texels") == ui_detection::change_set_decision_texels &&
        replay.at("ui_pin").at("evidence_images") == ui_detection::max_scene_evidence_images,
      "Dump lost the layer's detection constants or pin markers");
    require(!replay.at("ui_pin").contains("late_margin") &&
        replay.at("ui_pin").at("soft_pin_gain") == shader_marker(renderer::shader_source(), "SUNSHINE_UI_SOFT_PIN_GAIN"),
      "Dump lost the soft pin gain or still records a late-layer margin");
    // The accepted layer the previous real frame offered is missing now (T1):
    // the tag decides by itself, so nothing is reused.
    inputs = {};
    inputs.masks[1] = layer_view;
    check(ui_detection::candidate::ui_color, 0u, ui_detection::per_frame_accepted_missing, {}, "tagged UIColorAndAlpha");
    report << "layer-detection-dump flags=" << expected_flags << " tagged_flags=0 own_slots=1 ran=1 held_reported=1 layer_raw_alpha=1\n";
    std::puts("PASS D3D11 UI detection constants: the UI layer has its own slot with stored flags only, a tagged UI color the UI color slot with none, both accepted masks are their slot's raw alpha, and Dump 3D records them with the accepted candidates and pin markers");
  }
  // Hidden-scene evidence (docs/reshade-sbs.md, hidden-scene evidence). H1,
  // full-frame UI over a hidden scene (source 8), needs this frame's
  // informative full claim, current depth, and the scene guard's hold of a
  // hidden verdict (two valid hidden samples within hold_ms) of the
  // presented frame lacking the consumed depth's edges, and for a pre-UI
  // image's claim also that image reading visible on those samples; nothing
  // else changes a decision. The evidence passes run on sample frames only,
  // after a sample with an acting-capable claim, while a hold is active, for
  // the first-run shadow or after an accepted whole-frame decision, and in
  // SDR Auto for rule H2 (fix 2) while no source other than 11 decided; its
  // own section (m) closes this one.
  void verify_hidden_scene(fixture &gpu, std::ostream &report) {
    using namespace sunshine_game3d;
    using ui_detection::scene_verdict;
    const unsigned width = gpu.width, height = gpu.height;
    const auto pixels = size_t(width) * height;
    const auto bpp = gpu.color == 2 ? 8u : 4u;
    // Depth: two discs nearer than a flat background (parallax 0 there, 100
    // pixels per 2160 rows on the discs), or the flat background alone.
    const auto inside = [&](unsigned x, unsigned y) {
      const auto disc = [&](double cx, double cy, double r) {
        const double dx = ((x + .5) / width - cx) * width / height, dy = (y + .5) / height - cy;
        return dx * dx + dy * dy <= r * r;
      };
      return disc(.3, .5, .25) || disc(.72, .45, .18);
    };
    std::vector<float> silhouette_depth(pixels, .03f), flat_depth(pixels, .03f);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        if (inside(x, y)) silhouette_depth[size_t(y) * width + x] = .0302f;
    std::vector<ComPtr<ID3D11Texture2D>> textures;
    std::vector<ComPtr<ID3D11ShaderResourceView>> views;
    const auto view_of = [&](const void *bytes, unsigned pitch, DXGI_FORMAT format) {
      D3D11_TEXTURE2D_DESC desc{}; gpu.source->GetDesc(&desc);
      desc.Format = format; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      const D3D11_SUBRESOURCE_DATA data{bytes, pitch, 0};
      ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> view;
      checked(gpu.device->CreateTexture2D(&desc, &data, &texture), "hidden-scene texture");
      checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "hidden-scene view");
      textures.push_back(texture); views.push_back(view);
      return view;
    };
    const auto silhouette = view_of(silhouette_depth.data(), width * 4u, DXGI_FORMAT_R32_FLOAT),
      flat = view_of(flat_depth.data(), width * 4u, DXGI_FORMAT_R32_FLOAT);
    struct restore_depth {
      fixture &gpu; ComPtr<ID3D11ShaderResourceView> view;
      ~restore_depth() { gpu.depth_view = view; gpu.depth_current = true; }
    } restore{gpu, gpu.depth_view};
    gpu.depth_view = silhouette;
    // Gray premultiplied color: value * alpha, alpha. Values are multiples of 1/64.
    D3D11_TEXTURE2D_DESC color_desc{}; gpu.source->GetDesc(&color_desc);
    const auto make = [&](const std::function<float(unsigned, unsigned)> &value,
        const std::function<float(unsigned, unsigned)> &alpha) {
      std::vector<unsigned char> bytes(pixels * bpp);
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        const float a = alpha(x, y), v = value(x, y) * a, values[4]{v, v, v, a};
        if (gpu.color == 2) for (unsigned c = 0; c != 4; ++c) { const auto half = half_bits(values[c]); std::memcpy(bytes.data() + i * bpp + c * 2, &half, 2); }
        else for (unsigned c = 0; c != 4; ++c) bytes[i * bpp + c] = static_cast<unsigned char>(std::lround(std::clamp(values[c], 0.f, 1.f) * 255));
      }
      return bytes;
    };
    const auto opaque = [](unsigned, unsigned) { return 1.f; };
    // A logo away from the discs, the discs' own picture, a flat menu with the
    // logo, and black.
    const auto logo = [&](unsigned x, unsigned y) {
      return x >= width / 20 && x < width / 5 && y >= height / 20 && y < height / 6 && (x / 2) % 3 != 0;
    };
    const auto logo_value = [&](unsigned x, unsigned y) { return logo(x, y) ? .75f : 0.f; };
    const auto logo_color = make(logo_value, opaque);
    const auto scene_color = make([&](unsigned x, unsigned y) { return inside(x, y) ? .75f : .25f; }, opaque);
    const auto menu_color = make([&](unsigned x, unsigned y) { return logo(x, y) ? .75f : .5f; }, opaque);
    const auto black_color = make([](unsigned, unsigned) { return 0.f; }, opaque);
    // Hidden but not blank: stripes one scene cell wide, flat within three
    // cells of each disc's edge, so every edge cell is quieter than the
    // striped cells around it (D < 0) and most comparisons are decided.
    const auto near_edge = [&](unsigned x, unsigned y) {
      const auto ring = [&](double cx, double cy, double r) {
        const double dx = ((x + .5) / width - cx) * width / height, dy = (y + .5) / height - cy;
        return std::abs(std::sqrt(dx * dx + dy * dy) - r) * ui_detection::scene::cells_y <= 3.;
      };
      return ring(.3, .5, .25) || ring(.72, .45, .18);
    };
    const auto banded_color = make([&](unsigned x, unsigned y) {
      return near_edge(x, y) ? .5f : (x * ui_detection::scene::cells_x / width) % 2 ? .75f : .25f;
    }, opaque);
    const auto as_view = [](const ComPtr<ID3D11ShaderResourceView> &view) { return api::resource_view{reinterpret_cast<std::uint64_t>(view.Get())}; };
    const auto color_view = [&](const std::vector<unsigned char> &bytes) { return as_view(view_of(bytes.data(), width * bpp, color_desc.Format)); };
    const auto layer_view = [&](const std::function<float(unsigned, unsigned)> &alpha) {
      const auto bytes = make(logo_value, alpha);
      return color_view(bytes);
    };
    const auto opaque_layer = color_view(logo_color);
    // 98.9% opaque; half transparent everywhere; one pixel brighter than its
    // zero alpha allows (not premultiplied).
    const auto nearly_opaque_layer = layer_view([&](unsigned x, unsigned y) { return (size_t(y) * width + x) % 90 ? 1.f : 0.f; });
    const auto half_layer = layer_view([](unsigned, unsigned) { return .5f; });
    const auto logo_layer = layer_view([&](unsigned x, unsigned y) { return logo(x, y) ? 1.f : 0.f; });
    auto straight_bytes = logo_color;
    if (gpu.color == 2) { const auto zero = half_bits(0.f), one = half_bits(1.f); for (unsigned c = 0; c != 3; ++c) std::memcpy(straight_bytes.data() + c * 2, &one, 2); std::memcpy(straight_bytes.data() + 6, &zero, 2); }
    else { straight_bytes[0] = straight_bytes[1] = straight_bytes[2] = 255; straight_bytes[3] = 0; }
    const auto straight_layer = color_view(straight_bytes);
    const auto scene_view = color_view(scene_color), black_view = color_view(black_color);
    const std::vector<float> ones(pixels, 1.f);
    const auto opaque_ui_alpha = as_view(view_of(ones.data(), width * 4u, DXGI_FORMAT_R32_FLOAT));
    const auto layer_flags = ui_layer::detection_flags(static_cast<api::format>(color_desc.Format));
    require(layer_flags & ui_detection::stored_late_layer, "The fixture layer lost its late-layer identity");

    alpha_auto_policy policy;
    alpha_auto_source source;
    source.session = &policy; source.now_ms = source.tick_ms = 1000;
    source.epoch = 41; source.revision = 1; source.sequence = 1;
    ui_detection_inputs inputs;
    ui_render_input ui;
    ui.automatic = &source; ui.detection = &inputs;
    struct outcome { bool flat, empty, held; std::uint64_t evidence; alpha_auto_decision sample; result pixels; };
    // Rule H2 (fix 2) measures every SDR Auto sample frame on which no source
    // other than 11 decided, without acting on it (scene_guard::measure), so
    // in sRGB the evidence passes also run where nothing else asks for them;
    // scRGB keeps every check that they do not.
    const bool h2_measures = gpu.color == 1;
    const auto frame = [&](const std::vector<unsigned char> &color, const render_parameters *parameters = nullptr) {
      gpu.original = color;
      gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), width * bpp, 0);
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      const auto before = gpu.renderer.alpha_probe_activity().scene_evidence;
      gpu.renderer.begin_present();
      const auto rendered = gpu.render(true, 1, false, false, false, {}, {}, {}, parameters, nullptr, nullptr, nullptr, &ui);
      const auto mask = gpu.read(gpu.renderer.diagnostics().ui_source);
      bool all_ones = true, all_zero = true;
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const auto value = mask.channel(x, y, 0);
        all_ones = all_ones && value == 1.f; all_zero = all_zero && value == 0.f;
      }
      const auto flags = gpu.renderer.consumed_detection().flags;
      return outcome{all_ones, all_zero, (flags & ui_detection::per_frame_scene_hidden) != 0,
        gpu.renderer.alpha_probe_activity().scene_evidence - before, gpu.renderer.consumed_alpha_auto(), rendered};
    };
    // Frames until the first flat one, at most limit; each must hold the
    // hidden verdict exactly when flat.
    const auto frames_to_flat = [&](const std::vector<unsigned char> &color, unsigned limit, const char *label) {
      for (unsigned count = 1; count <= limit; ++count) {
        const auto o = frame(color);
        require(o.flat == o.held, std::string(label) + ": the held hidden verdict did not flatten exactly the held frame");
        if (o.flat) return count;
      }
      return 0u;
    };
    const auto never_flat = [&](const std::vector<unsigned char> &color, unsigned count, const char *label,
        const render_parameters *parameters = nullptr) {
      // The sample read by a frame describes the frame before it.
      outcome last{};
      for (unsigned i = 0; i != count; ++i) {
        last = frame(color, parameters);
        require(!last.flat && (!i || last.sample.source_kind != 8u),
          std::string(label) + ": full-frame UI without its hidden-scene evidence");
      }
      return last;
    };

    // A frame without protection ends any mask an earlier section left: an
    // accepted candidate missing from a frame would otherwise hold it (T1).
    source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
    gpu.render(false, 1, false, false, false, {}, {}, {}, nullptr, &source);
    // A frame smaller than the grid has cells without a pixel: no evidence,
    // so nothing ever holds a verdict.
    if (width < ui_detection::scene::cells_x || height < ui_detection::scene::cells_y) {
      inputs.layer = opaque_layer; inputs.layer_flags = layer_flags;
      const auto small = never_flat(logo_color, 4, "frame smaller than the scene grid");
      require(small.sample.evidence.scene.ran && !small.sample.evidence.scene.valid && !small.sample.evidence.scene.n,
        "A frame smaller than the scene grid gave hidden-scene evidence");
      inputs = {};
      report << "hidden-scene D3D11 below_grid=" << width << 'x' << height << " no_evidence=1\n";
      std::puts("PASS D3D11 hidden scene: a frame smaller than the scene grid gives no evidence and never flattens");
      return;
    }
    // (a) An unaccepted opaque offscreen layer over a scene its picture hides
    // (claim (b): a cleared target, V1-valid and opaque-full): the first
    // sample shows the claim, the next two measure, the second valid hidden
    // verdict enters the hold (H1), and the frame pins flat at one plane.
    inputs.layer = opaque_layer; inputs.layer_flags = layer_flags;
    const auto first = frame(logo_color);
    require(!first.flat && (h2_measures || !first.evidence), "Evidence ran or flattened before any sample showed a claim");
    const auto second = frame(logo_color);
    require(!second.flat && second.evidence == 1, "A claim did not measure the next sample frame");
    const auto one_sample = frame(logo_color);
    require(!one_sample.flat && !one_sample.held && one_sample.evidence == 1 &&
        one_sample.sample.evidence.scene.verdict == scene_verdict::hidden, "One hidden sample entered the hold");
    const auto entered = frame(logo_color);
    const auto &measured = entered.sample.evidence.scene;
    require(entered.flat && entered.held && measured.ran && measured.valid && measured.verdict == scene_verdict::hidden &&
        measured.n >= ui_detection::scene::min_edges && measured.d < .15f && entered.sample.evidence.layer_opaque == pixels,
      "An opaque layer over a hidden scene did not become full-frame UI from two valid hidden samples");
    {
      float plane{}; bool uniform = true;
      std::memcpy(&plane, entered.pixels.field.bytes.data(), sizeof(plane));
      for (size_t i = 0; i != pixels; ++i) { float value; std::memcpy(&value, entered.pixels.field.bytes.data() + i * 4, 4); uniform = uniform && value == plane; }
      require(uniform, "Full-frame UI over a hidden scene left a non-uniform field");
    }
    const auto confirmed = frame(logo_color);
    require(confirmed.flat && confirmed.sample.source_kind == 8u && confirmed.sample.covered == pixels &&
        confirmed.sample.evidence.h1_applied && !confirmed.sample.evidence.s1_source &&
        (confirmed.sample.evidence.claims & ui_detection::candidate::layer) &&
        confirmed.sample.scene_guard.hidden && !confirmed.sample.scene_guard.pre_ui,
      "H1 over the layer's claim did not report source 8 over the whole frame");
    // (b) A presented frame showing the depth's edges reads visible and
    // releases the hold; only the frame already decided before that sample
    // stays flat.
    frame(scene_color);
    const auto released = frame(scene_color);
    require(!released.flat && !released.held && released.sample.evidence.scene.verdict == scene_verdict::visible &&
        released.sample.evidence.scene.d >= .25f && released.sample.scene_guard.refuted == 1,
      "A visible scene did not release the hold and refute the layer");
    never_flat(scene_color, 2, "presented frame showing the depth's edges");
    // That visible verdict refuted the layer's signature: like a scene buffer
    // the census took for a layer, its claim acts no more until it is offered
    // below 99% opaque. A selective layer is an overlay again (unaccepted, it
    // decides nothing itself), and the opaque one then re-enters from its
    // claim.
    never_flat(logo_color, 4, "refuted opaque layer");
    inputs.layer = logo_layer;
    const auto selective = frame(logo_color);
    require(!selective.flat && selective.empty && !selective.held,
      "A selective unaccepted layer decided a mask or kept the hidden-scene hold");
    inputs.layer = opaque_layer;
    require(frames_to_flat(logo_color, 5, "re-entry") == 4, "A layer shown transparent again did not re-enter");
    // (c) Invalid evidence never renews a hold: over flat depth the hold
    // expires hold_ms after the last valid hidden sample's tick.
    frame(logo_color);
    gpu.depth_view = flat;
    unsigned held_frames = 0;
    for (outcome o = frame(logo_color); o.flat; o = frame(logo_color)) {
      // The first held frame reads the last sample over the silhouette.
      ++held_frames;
      require((held_frames == 1 || !o.sample.evidence.scene.valid) && o.sample.evidence.scene.ran && held_frames <= 6,
        "Invalid evidence renewed the hold");
    }
    require(held_frames == 5, "A hold did not last exactly hold_ms after its last valid sample: " + std::to_string(held_frames));
    never_flat(logo_color, 3, "flat depth without edge cells");
    // (d) Depth that is not this frame's gives no evidence and disables H1.
    gpu.depth_view = silhouette;
    gpu.depth_current = false;
    const auto stale = never_flat(logo_color, 4, "reused depth");
    require(stale.sample.evidence.scene.ran && !stale.sample.evidence.scene.valid &&
        stale.sample.evidence.scene.n >= ui_detection::scene::min_edges, "Reused depth gave valid hidden-scene evidence");
    gpu.depth_current = true;
    require(frames_to_flat(logo_color, 4, "current depth again") == 3, "Current depth did not re-enter after two hidden samples");
    // (e) Only an identity change (epoch or viewport) clears the guard. An
    // observation revision, detection turning inactive, acceptance changes
    // and a HUD-less image that frame generation pairs on some Presents only
    // keep the hold.
    ++source.epoch;
    never_flat(logo_color, 1, "epoch change");
    require(frames_to_flat(logo_color, 4, "after the epoch change") == 3, "The hold did not re-enter after an epoch change");
    ++source.revision;
    for (unsigned i = 0; i != 3; ++i) {
      const auto kept = frame(logo_color);
      require(kept.flat && kept.held, "A revision change cleared the hidden-scene hold");
    }
    {
      // One frame with automatic UI protection off: detection is inactive.
      const auto parameters = gpu.renderer.consumed_parameters();
      policy.set_manual(false);
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.renderer.begin_present();
      gpu.context->CopyResource(gpu.backbuffer.Get(), gpu.source.Get());
      sunshine_game3d::render_frame_input off;
      off.color = {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())};
      off.depth = {reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())};
      off.scene = parameters; off.ui = ui;
      gpu.renderer.render(observed_runtime->get_command_queue()->get_immediate_command_list(), off);
      require(gpu.renderer.consumed_detection().state == ui_detection_snapshot::run_state::inactive,
        "Manual Off left detection active");
      policy.set_automatic();
    }
    // The guard outlives the inactive frame: the next detecting frame is flat.
    require(frames_to_flat(logo_color, 4, "after inactive detection") == 1, "Inactive detection cleared the hidden-scene hold");
    // HUD-less present on alternate Presents only: the hold stays.
    for (unsigned i = 0; i != 4; ++i) {
      inputs.hudless = i % 2 ? api::resource_view{} : scene_view;
      const auto paired = frame(logo_color);
      require(paired.flat && paired.held, "A HUD-less pairing that comes and goes cleared the hold");
    }
    inputs.hudless = {};
    {
      // The layer accepted while held keeps the hold (acceptance never clears
      // D) and decides by itself: a winner opaque on every pixel (source 10,
      // flat at any coverage) that H1 never overrides.
      alpha_auto_policy trusting;
      require(trusting.restore(gpu.key(ui_selection::kind::ui_layer)).restored == 1, "The layer key was not restored");
      source.session = &trusting;
      const auto gained = frame(logo_color);
      const auto decided = frame(logo_color);
      require(gained.held && gained.flat && decided.held && decided.sample.source_kind == ui_detection::source_layer &&
          !decided.sample.evidence.h1_applied && decided.sample.evidence.s1_source == ui_detection::source_layer,
        "Acceptance gained by the layer cleared the hold, or H1 overrode its whole-frame decision");
      source.session = &policy;
    }
    // Acceptance lost again: the unaccepted layer's claim acts under the same
    // hold at once.
    const auto lost = frame(logo_color);
    require(lost.flat && lost.held, "Acceptance lost again cleared the hidden-scene hold");
    // (f) Without a ready camera there is no evidence.
    render_parameters no_camera = gpu.renderer.consumed_parameters();
    no_camera.camera_ready = 0;
    ++source.epoch;
    const auto unready = never_flat(logo_color, 4, "camera not ready", &no_camera);
    require(unready.sample.evidence.scene.ran && !unready.sample.evidence.scene.valid && !unready.sample.evidence.scene.n,
      "Depth without a ready camera was measured");
    // (g) Without an informative claim nothing flattens, even under a held
    // hidden verdict, which renews while the presented frame reads hidden.
    // Once a visible sample releases it, no evidence pass runs. A tagged
    // UIColorAndAlpha has a slot of its own and is unaccepted here, so it is
    // never informative.
    const auto closed_gate = [&](api::resource_view layer, bool tag, const char *label) {
      // After an epoch change the first sample shows the claim, two measure.
      ++source.epoch;
      inputs = {}; inputs.layer = opaque_layer; inputs.layer_flags = layer_flags;
      require(frames_to_flat(logo_color, 5, label) == 4, std::string(label) + ": the layer's claim did not enter");
      if (tag) { inputs.layer = {}; inputs.layer_flags = 0; inputs.masks[1] = layer; }
      else inputs.layer = layer;
      // Each frame decides from its own claims: the held hidden verdict alone
      // flattens nothing.
      never_flat(logo_color, 5, label);
      // The presented frame showing the scene releases the hold.
      frame(scene_color);
      never_flat(scene_color, 1, label);
      std::uint64_t runs = 0;
      for (unsigned i = 0; i != 3; ++i) {
        const auto quiet = frame(logo_color);
        require(!quiet.flat && !quiet.held, std::string(label) + ": no claim flattened or held");
        runs += quiet.evidence;
      }
      require(h2_measures || !runs, std::string(label) + ": evidence passes ran without a claim, a hold or the first-run shadow");
    };
    closed_gate(nearly_opaque_layer, false, "98.9% opaque layer");
    closed_gate(half_layer, false, "layer at alpha 0.5 over the whole frame");
    closed_gate(opaque_layer, true, "opaque tagged UIColorAndAlpha");
    // V1 tolerates up to 1% invalid pixels: a layer with one pixel beyond the
    // premultiplied bound is still V1-valid and opaque-full, so its claim
    // acts like the opaque layer's.
    ++source.epoch;
    inputs = {}; inputs.layer = straight_layer; inputs.layer_flags = layer_flags;
    require(frames_to_flat(logo_color, 5, "layer with one invalid pixel") == 4,
      "A V1-valid opaque-full layer with one invalid pixel did not claim");
    ++source.epoch;
    inputs = {}; inputs.masks[2] = opaque_layer;
    never_flat(logo_color, 4, "opaque Backbuffer alpha only");
    inputs = {}; inputs.current_color = true;
    never_flat(logo_color, 4, "opaque current alpha only");
    // (h) An unaccepted UIAlpha that is opaque everywhere is no informative
    // claim (H1): it never flattens and runs no evidence. Accepted, it pins
    // flat by itself as source 1 (the opacity ruling, P1).
    ++source.epoch;
    inputs = {}; inputs.masks[0] = opaque_ui_alpha;
    never_flat(logo_color, 4, "unaccepted opaque UIAlpha");
    const auto quiet_ui_alpha = never_flat(logo_color, 2, "unaccepted opaque UIAlpha");
    require(h2_measures || !quiet_ui_alpha.evidence, "An unaccepted UIAlpha ran evidence");
    {
      alpha_auto_policy trusting;
      require(trusting.restore(gpu.key(ui_selection::kind::ui_alpha, DXGI_FORMAT_R32_FLOAT)).restored == 1,
        "The UIAlpha key was not restored");
      ++source.epoch; source.session = &trusting;
      const auto pinned = frame(logo_color);
      const auto decided = frame(logo_color);
      require(pinned.flat && decided.flat && decided.sample.source_kind == 1u && !decided.sample.evidence.h1_applied,
        "An accepted opaque UIAlpha did not pin flat by itself as source 1");
      source.session = &policy;
    }
    // (i) The pre-UI HUD-less image (claim (d)): an inexact HUD-less image
    // that shows the depth's edges while the presented menu does not. Two
    // samples read the presented frame hidden and the HUD-less image visible.
    ++source.epoch;
    inputs = {}; inputs.current_color = true; inputs.hudless = scene_view;
    require(frames_to_flat(menu_color, 5, "pre-UI HUD-less image") == 4,
      "A HUD-less image showing the hidden scene did not flatten under H1");
    const auto hudless_decided = frame(menu_color);
    const auto &pre_ui = hudless_decided.sample.evidence.pre_ui_scene;
    require(hudless_decided.flat && hudless_decided.sample.source_kind == 8u && pre_ui.valid && pre_ui.d >= .25f &&
        hudless_decided.sample.evidence.pre_ui_image == ui_detection::pre_ui_image::hudless &&
        hudless_decided.sample.evidence.claims == ui_detection::claim_pre_ui && hudless_decided.sample.scene_guard.pre_ui &&
        (gpu.renderer.consumed_detection().flags & (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible)) ==
          (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible),
      "The pre-UI HUD-less image did not report source 8 with both held verdicts");
    ++source.epoch;
    inputs.hudless = black_view;
    const auto dark = never_flat(menu_color, 4, "black HUD-less image");
    require(dark.sample.evidence.pre_ui_scene.valid && dark.sample.evidence.pre_ui_scene.d < .25f,
      "A black HUD-less image read visible");
    ++source.epoch;
    inputs.hudless = scene_view;
    never_flat(scene_color, 4, "presented frame showing the scene beside its HUD-less image");
    // Before its acceptance an exact pair differing nearly everywhere does not
    // decide 6 (only an accepted pair decides), but it is an informative
    // full claim (c): under two hidden samples H1 flattens it as 8.
    ++source.epoch;
    inputs.hudless_exact = true;
    require(frames_to_flat(menu_color, 5, "unaccepted exact full-frame pair") == 4,
      "An unaccepted exact full-frame pair did not flatten under H1");
    const auto exact_h1 = frame(menu_color);
    require(exact_h1.sample.source_kind == 8u && (exact_h1.sample.evidence.claims & ui_detection::candidate::hudless),
      "An unaccepted exact full-frame pair decided 6, or H1 did not name its claim");
    {
      // Accepted, the same exact pair over a lit scene is rule 6 at once.
      alpha_auto_policy accepting;
      require(accepting.restore(gpu.key(ui_selection::kind::hudless)).restored == 1, "The HUD-less key was not restored");
      ++source.epoch; source.session = &accepting;
      const auto exact = frame(menu_color);
      require(exact.flat && !exact.held && frame(menu_color).sample.source_kind == 6u, "An accepted exact full-frame pair lost rule 6");
      source.session = &policy;
    }
    // (j) Stellar Blade SDR (dump 50264_218658377782962): frame generation
    // suspended in the settings menu, so no tag; the game's cleared output
    // target holds the scene drawn before the UI at alpha 0 (V1-invalid:
    // colour without alpha) while the presented frame is the opaque menu.
    // The depth describes the pre-UI image, not the frame shown (claim (d)).
    // The layer's claim needs its signature proven the pre-UI scene image by
    // the session's acceptance ledger (game3d_alpha_auto.h, fix 1): three
    // samples over 2 s whose layer equals the presented colour (8 times the
    // pair threshold) on 90% of pixels while lit on half (texel 11). A
    // session of its own keeps the proof out of the later sections.
    const auto zero_alpha = [&](std::vector<unsigned char> bytes) {
      for (size_t i = 0; i != pixels; ++i) {
        if (gpu.color == 2) { const auto zero = half_bits(0.f); std::memcpy(bytes.data() + i * bpp + 6, &zero, 2); }
        else bytes[i * bpp + 3] = 0;
      }
      return bytes;
    };
    const auto scene_layer_bytes = zero_alpha(scene_color), menu_layer_bytes = zero_alpha(menu_color);
    // Dark but beyond the premultiplied bound at alpha 0 (4/255).
    const auto dark_layer_bytes = zero_alpha(make([](unsigned, unsigned) { return 3.f / 64.f; }, opaque));
    const auto scene_layer = color_view(scene_layer_bytes), mismatch_layer = color_view(menu_layer_bytes),
      dark_layer = color_view(dark_layer_bytes);
    // Texel 11 as SunshinePreUIPixel counts it from the pushed pair threshold
    // (b2 word 4): matching pixels, lit layer pixels, lit presented pixels
    // and lit presented pixels that differ, at 8 times the threshold (scRGB
    // relative above one).
    using pre_ui_pixels = std::array<std::uint32_t, 4>;
    const auto channel_of = [&](const std::vector<unsigned char> &bytes, size_t i, unsigned c) {
      if (gpu.color != 2) return bytes[i * bpp + c] / 255.f;
      std::uint16_t half; std::memcpy(&half, bytes.data() + i * bpp + c * 2, 2); return half_float(half);
    };
    const auto expected_pixels = [&](const std::vector<unsigned char> &presented, const std::vector<unsigned char> &layer) {
      float threshold{};
      const auto threshold_bits = gpu.renderer.consumed_detection().pre_ui_threshold_bits;
      std::memcpy(&threshold, &threshold_bits, sizeof(threshold));
      require(threshold > 0.f, "The layer and the presented colour got no pair threshold");
      const float coarse = threshold * 8.f;
      pre_ui_pixels counts{};
      for (size_t i = 0; i != pixels; ++i) {
        float presented_peak = 0.f, layer_peak = 0.f, delta = 0.f;
        for (unsigned c = 0; c != 3; ++c) {
          const float p = channel_of(presented, i, c), l = channel_of(layer, i, c);
          presented_peak = std::max(presented_peak, std::abs(p)); layer_peak = std::max(layer_peak, std::abs(l));
          delta = std::max(delta, std::abs(p - l));
        }
        delta /= gpu.color == 2 ? std::max(1.f, presented_peak) : 1.f;
        const bool match = delta <= coarse, lit = presented_peak > coarse;
        counts[0] += match; counts[1] += layer_peak > coarse; counts[2] += lit; counts[3] += lit && !match;
      }
      return counts;
    };
    const auto counted = [](const alpha_auto_decision &sample) {
      const auto &e = sample.evidence;
      return pre_ui_pixels{e.pre_ui_match, e.pre_ui_image_lit, e.presented_lit, e.presented_lit_differs};
    };
    const auto all_pixels = std::uint32_t(pixels);
    const auto proven_pushed = [&] { return (gpu.renderer.consumed_detection().flags & ui_detection::per_frame_pre_ui_proven) != 0; };
    alpha_auto_policy sdr;
    source.session = &sdr;
    const auto layer_signature = *ui_selection::signature::parse(gpu.key(ui_selection::kind::ui_layer));
    ++source.epoch;
    inputs = {}; inputs.current_color = true; inputs.layer = scene_layer; inputs.layer_flags = layer_flags;
    // Unproven, the layer claims nothing: the menu stays 3D and no sample
    // frame measures, so not even the hidden verdict is held (S2b held it
    // from the V1-invalid layer's own claim). Its pixels mismatch.
    const auto unproven = never_flat(menu_color, 5, "Stellar Blade SDR settings before gameplay (layer unproven)");
    const auto menu_pixels = expected_pixels(menu_color, scene_layer_bytes);
    require(!unproven.held && (h2_measures || !unproven.evidence) && !unproven.sample.evidence.claims && !unproven.sample.scene_guard.proven &&
        !proven_pushed() && counted(unproven.sample) == menu_pixels && menu_pixels[0] * 10u < all_pixels * 9u &&
        !sdr.pre_ui_proven(layer_signature),
      "The unproven SDR scene layer acted or measured, or its pixel counts are not exact");
    // Gameplay shows the scene the layer holds: equal on every pixel and lit,
    // so the third matching sample 2 s after the first proves the layer
    // without any D (nothing claims before the proof). A frame reads the
    // previous frame's sample, 100 ms apart: the 22nd gameplay frame reads
    // the sample 2 s after the first.
    const auto play_pixels = expected_pixels(scene_color, scene_layer_bytes);
    require(play_pixels == pre_ui_pixels{all_pixels, all_pixels, all_pixels, 0u}, "The fixture's gameplay layer does not equal the presented colour");
    unsigned proving_frames = 0;
    outcome proving{};
    while (!sdr.pre_ui_proven(layer_signature) && proving_frames != 30) {
      proving = frame(scene_color);
      ++proving_frames;
      require(!proving.flat && (proving_frames == 1 || counted(proving.sample) == play_pixels) &&
          (sdr.pre_ui_proven(layer_signature) || h2_measures || !proving.evidence),
        "SDR gameplay before the proof flattened or measured, or its pixel counts are not exact");
    }
    require(proving_frames == 22 && proving.sample.scene_guard.proven && proving.evidence == 1 && proven_pushed(),
      "SDR gameplay did not prove the scene layer by its pixels at the sample 2 s after the first: " + std::to_string(proving_frames));
    // A visible sample whose layer differs from the presented frame never
    // withdraws the proof (UI over the scene looks the same).
    inputs.layer = mismatch_layer;
    frame(scene_color);
    const auto mismatched = frame(scene_color);
    require(mismatched.evidence == 1 && mismatched.sample.evidence.scene.verdict == scene_verdict::visible &&
        counted(mismatched.sample) == expected_pixels(scene_color, menu_layer_bytes) &&
        !ui_selection::pre_ui_match(mismatched.sample.evidence.pre_ui_match, mismatched.sample.evidence.pre_ui_image_lit, all_pixels) &&
        mismatched.sample.scene_guard.proven && sdr.pre_ui_proven(layer_signature),
      "A mismatching visible sample withdrew the proof, or was not measured");
    inputs.layer = scene_layer;
    // The menu suspends frame generation and the depth moves to another
    // provider: another viewport, an identity change that clears the scene
    // guard but not the ledger's proof. Flat at its second hidden sample.
    ++source.viewport;
    require(frames_to_flat(menu_color, 4, "Stellar Blade SDR settings") == 3,
      "The SDR settings menu over its proven pre-UI layer did not flatten at its second hidden sample after the identity change");
    const auto sb_menu = frame(menu_color);
    require(sb_menu.flat && sb_menu.sample.source_kind == 8u && sb_menu.sample.evidence.claims == ui_detection::claim_pre_ui &&
        sb_menu.sample.evidence.pre_ui_image == ui_detection::pre_ui_image::layer && sb_menu.sample.evidence.pre_ui_scene.valid &&
        sb_menu.sample.evidence.pre_ui_scene.d >= .25f && sb_menu.sample.evidence.scene.verdict == scene_verdict::hidden &&
        sb_menu.sample.evidence.frame_reason == ui_detection::frame_reason_decided && counted(sb_menu.sample) == menu_pixels &&
        sb_menu.sample.scene_guard.hidden && sb_menu.sample.scene_guard.pre_ui && sb_menu.sample.scene_guard.proven &&
        (gpu.renderer.consumed_detection().flags & (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible |
          ui_detection::per_frame_pre_ui_proven)) == (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible |
          ui_detection::per_frame_pre_ui_proven),
      "The SDR settings menu did not report H1 from its proven pre-UI layer");
    // Gameplay again: the presented frame shows the scene, reads visible and
    // releases both holds; the pre-UI claim stays true but never acts alone,
    // and is never refuted, so the next menu visit enters again.
    frame(scene_color);
    const auto sb_released = frame(scene_color);
    require(!sb_released.flat && !sb_released.held && !sb_released.sample.scene_guard.refuted,
      "SDR gameplay did not release the menu's holds, or refuted the pre-UI claim");
    const auto sb_gameplay = never_flat(scene_color, 4, "SDR gameplay, presented and pre-UI image both visible");
    require(sb_gameplay.evidence == 1 && sb_gameplay.sample.evidence.pre_ui_scene.d >= .25f &&
        sb_gameplay.sample.evidence.scene.verdict == scene_verdict::visible && counted(sb_gameplay.sample) == play_pixels,
      "SDR gameplay did not measure both images visible on every sample");
    // A real frame within 100 ms of the previous sample is not a detection
    // sample: it pushes no pre-UI threshold, so its passes skip the pre-UI
    // counts the CPU never reads; the next sample frame counts them again.
    {
      ++source.sequence;
      gpu.renderer.begin_present();
      gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      const auto between = gpu.renderer.consumed_detection();
      require(between.state == ui_detection_snapshot::run_state::ran && (between.candidates & ui_detection::candidate::layer) &&
          !between.pre_ui_threshold_bits, "A frame that is not a detection sample pushed the pre-UI threshold");
      const auto next = frame(scene_color);
      require(gpu.renderer.consumed_detection().pre_ui_threshold_bits && !next.flat && counted(next.sample) == play_pixels,
        "The sample frame after a skipped one lost its pre-UI threshold or its exact counts");
    }
    require(frames_to_flat(menu_color, 4, "Stellar Blade SDR settings again") == 3,
      "The next SDR menu visit did not enter at its second hidden sample");
    // Dark gameplay: the pre-UI image reads hidden too, so its hold never
    // enters and nothing flattens, though the presented frame reads hidden.
    // The dark layer is lit nowhere (sRGB) or differs (scRGB): no match.
    ++source.epoch;
    inputs.layer = dark_layer;
    const auto sb_dark = never_flat(black_color, 6, "SDR dark gameplay, both images hidden");
    require(sb_dark.sample.evidence.pre_ui_scene.valid && sb_dark.sample.evidence.pre_ui_scene.d < .25f &&
        sb_dark.held && !sb_dark.sample.scene_guard.pre_ui && counted(sb_dark.sample) == expected_pixels(black_color, dark_layer_bytes) &&
        !ui_selection::pre_ui_match(sb_dark.sample.evidence.pre_ui_match, sb_dark.sample.evidence.pre_ui_image_lit, all_pixels) &&
        sdr.pre_ui_proven(layer_signature), "A dark pre-UI image entered the pre-UI hold, matched, or withdrew the proof");
    require(sdr.stored() == ui_selection::pre_ui_key(layer_signature).key(), "The SDR session did not keep the layer's proof as its only key");
    source.session = &policy;
    std::printf("PASS D3D11 Stellar Blade SDR pre-UI proof (fix 1): an unproven layer without alpha claims and measures nothing; %u gameplay frames prove it by exact texel 11 pixel counts (3 matching samples over 2 s, no D), counted on sample frames only; a mismatching visible sample and an identity change keep the proof; the menu flattens at its second hidden sample and a dark pre-UI image never enters the pre-UI hold\n", proving_frames);
    // (k) An accepted layer decides by itself (source 10), whatever the
    // evidence: its opaque-full claim measures, and the hidden verdict it
    // enters is held, but H1 never overrides a winner opaque on every pixel. Its
    // samples count as accepted whole-frame decisions (full_alpha_d).
    {
      alpha_auto_policy trusting;
      require(trusting.restore(gpu.key(ui_selection::kind::ui_layer)).restored == 1, "The layer key was not restored");
      ++source.epoch; source.session = &trusting;
      inputs = {}; inputs.layer = opaque_layer; inputs.layer_flags = layer_flags;
      const auto first = frame(logo_color);
      require(first.sample.source_kind != ui_detection::source_layer || !first.evidence,
        "Evidence ran before a whole-frame alpha sample");
      for (unsigned i = 0; i != 2; ++i) frame(logo_color);
      const auto trusted = frame(logo_color);
      require(trusted.flat && trusted.sample.source_kind == ui_detection::source_layer && trusted.held && trusted.evidence == 1 &&
          !trusted.sample.evidence.h1_applied && trusted.sample.evidence.s1_source == ui_detection::source_layer,
        "An accepted layer lost its own decision, or H1 overrode it");
      const auto counted = trusting.counters();
      require(counted[ui_counter::full_alpha] >= 3 && counted[ui_counter::full_alpha_d_hidden] +
          counted[ui_counter::full_alpha_d_ambiguous] + counted[ui_counter::full_alpha_d_visible] >= 1 &&
          !counted[ui_counter::full_d_hidden] && !counted[ui_counter::full_d_visible] && counted[ui_counter::scene_entered] == 1,
        "The whole-frame layer's frames or measured samples were not counted as full_alpha, or its entry not counted");
      source.session = &policy;
    }
    // (l) The first-run shadow measures on sample frames without a claim,
    // never changes a decision or a hold, and reports how long the hidden run
    // lasts. A frame without protection first ends the accepted layer's flat
    // mask, which its absence would otherwise hold (T1).
    ++source.epoch;
    source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
    gpu.render(false, 1, false, false, false, {}, {}, {}, nullptr, &source);
    inputs = {}; inputs.current_color = true;
    never_flat(logo_color, 2, "no claim before the shadow");
    const auto quiet_no_claim = never_flat(logo_color, 2, "no claim before the shadow");
    require(h2_measures || !quiet_no_claim.evidence, "Evidence ran without a claim");
    policy.set_first_run(true);
    outcome shadow{};
    for (unsigned i = 0; i != 8; ++i) {
      shadow = never_flat(banded_color, 1, "first-run shadow");
      require(shadow.empty && shadow.evidence == 1 && shadow.sample.scene_shadow && !shadow.held &&
          !shadow.sample.scene_guard.hidden, "The first-run shadow changed a decision or a hold, or skipped a sample frame");
    }
    require(shadow.sample.evidence.scene.verdict == scene_verdict::hidden && !shadow.sample.source_kind &&
        shadow.sample.evidence.scene.decided >= ui_detection::scene::min_edges &&
        shadow.sample.evidence.shadow_hidden_ms >= 500, "The first-run shadow did not report its uncovered hidden run");
    // A blank frame reads hidden too, with nothing decided: it is not an
    // uncovered hidden scene and ends the run.
    const auto blank_shadow = never_flat(logo_color, 2, "first-run shadow over a blank frame");
    require(blank_shadow.sample.evidence.scene.verdict == scene_verdict::hidden &&
        blank_shadow.sample.evidence.scene.decided < ui_detection::scene::min_edges &&
        !blank_shadow.sample.evidence.shadow_hidden_ms, "A blank frame extended the uncovered hidden run");
    const auto visible_shadow = never_flat(scene_color, 2, "first-run shadow over a visible scene");
    require(!visible_shadow.sample.evidence.shadow_hidden_ms && visible_shadow.sample.evidence.scene.verdict == scene_verdict::visible,
      "A visible sample did not end the hidden run");
    // With a claim, only evidence the claim or a hold asked for acts: H1
    // enters after the same samples as without the shadow, though the shadow
    // measured the first sample that showed the claim.
    inputs = {}; inputs.layer = opaque_layer; inputs.layer_flags = layer_flags;
    require(frames_to_flat(logo_color, 5, "first-run shadow with a claim") == 4,
      "The first-run shadow changed when H1 enters");
    policy.set_first_run(false);
    inputs = {};
    report << "hidden-scene D3D11 layer_claim=1 two_sample_entry=1 visible_release=1 refuted_layer_quiet=1 hold_expiry_frames="
      << held_frames << " reused_depth_blocks=1 camera_blocks=1 epoch_clears=1 revision_inactive_acceptance_keep=1"
      " alternating_hudless_keeps_hold=1 no_claim_quiet=1 one_invalid_pixel_claims=1 unaccepted_ui_alpha_quiet=1"
      " accepted_ui_alpha_source1=1 pre_ui_hudless=1 black_hudless_rejected=1 unaccepted_exact_pair_h1=1 rule6_accepted=1"
      " sdr_unproven_quiet=1 sdr_proof_frames=" << proving_frames << " sdr_proof_kept_mismatch=1 sdr_proof_kept_identity=1"
      " sdr_pre_ui_pixels_exact=1 sdr_pre_ui_sample_only=1 sdr_pre_ui_layer=1 sdr_gameplay_visible=1 sdr_dark_never_flat=1 accepted_layer_not_overridden=1"
      " first_run_shadow=1"
      " blank_frames_end_run=1 shadow_entry_unchanged=1\n";
    std::puts("PASS D3D11 hidden scene (H1): an informative full claim (an unaccepted opaque layer, an unaccepted exact full change set, or a pre-UI image: an inexact HUD-less image, or Stellar Blade SDR's cleared target without alpha once the session ledger proved it the pre-UI image by its pixels (three matching samples over 2 s, kept across a mismatching sample and an identity change), reading visible while the presented frame reads hidden) flattens as 8 only after two valid hidden samples and while the depth is this frame's; holds expire after hold_ms, release on visible evidence and clear only on an epoch change, not on a revision, inactive detection, acceptance or a HUD-less pairing that comes and goes; a visible verdict refutes the layer's claim until it shows itself transparent; reused depth, an unready camera, no claim, a tagged UI color, presented alpha and an unaccepted UIAlpha never flatten, while an accepted UIAlpha or layer decides by itself and is never overridden; an exact full-frame pair is rule 6 once accepted; SDR gameplay (both images visible) and dark gameplay (both hidden) never flatten; evidence runs only for a claim, a hold, the shadow or a whole-frame decision; the first-run shadow only measures, ignores blank frames and leaves entry unchanged");
    // (m) Rule H2 (fix 2; docs/reshade-sbs.md, still screens without a UI
    // source). Stellar Blade's SDR loading screen offers its cleared output
    // target almost black and without alpha (V1-invalid, unproven) beside an
    // opaque current alpha (0x48), neither accepted: no source decides and
    // nothing claims. Its frame shows none of the depth's edges (the banded
    // image: D < 0) and stays still. In SDR Auto the run enters at the sample
    // 2 s after its first passing one; the default shadow only reports it,
    // and enabled (UIFlattenStillScreens=1) the frames are flat as source 11
    // from the first frame that reads that sample until the first one that
    // reads a moving sample. Depth that is not the frame's own keeps the run.
    // scRGB is out of scope: nothing runs and nothing is pushed.
    {
      constexpr std::uint32_t cells = ui_detection::scene::cells_x * ui_detection::scene::cells_y;
      const auto near_black_layer = color_view(zero_alpha(make([](unsigned, unsigned) { return 1.f / 64.f; }, opaque)));
      // The same stripes moved by one scene cell: every striped cell changes.
      const auto panned_color = make([&](unsigned x, unsigned y) {
        return near_edge(x, y) ? .5f : (x * ui_detection::scene::cells_x / width + 1) % 2 ? .75f : .25f;
      }, opaque);
      alpha_auto_policy loading;
      source.session = &loading;
      gpu.depth_view = silhouette;
      gpu.depth_current = true;
      const auto uniform_field = [&](const outcome &o) {
        float plane{}; bool uniform = true;
        std::memcpy(&plane, o.pixels.field.bytes.data(), sizeof(plane));
        for (size_t i = 0; i != pixels; ++i) { float value; std::memcpy(&value, o.pixels.field.bytes.data() + i * 4, 4); uniform = uniform && value == plane; }
        return uniform;
      };
      // One loading screen of `count` frames after an identity change; returns
      // the frame that first reads an active run (0 without one) and the last
      // outcome. Every frame is flat exactly when the run is active, in scope
      // and enabled, which is exactly when the still word is pushed.
      const auto loading_screen = [&](bool enabled, unsigned count, const char *label) {
        loading.set_still_flatten(enabled);
        inputs = {}; inputs.current_color = true; inputs.layer = near_black_layer; inputs.layer_flags = layer_flags;
        // Another screen before it (the logo), then an identity change.
        frame(logo_color);
        ++source.epoch;
        unsigned entry = 0;
        outcome o{};
        for (unsigned i = 1; i <= count; ++i) {
          o = frame(banded_color);
          const auto bits = gpu.renderer.consumed_detection().still_bits;
          const bool active = o.sample.still.phase == still_screen::phase::active;
          if (active && !entry) entry = i;
          require(o.sample.still.scope == h2_measures && o.sample.still.enabled == (h2_measures && enabled) && !o.held &&
              !o.sample.evidence.claims, std::string(label) + ": wrong scope or switch, a claim or a hidden-scene hold");
          require(o.flat == (h2_measures && enabled && active) && bits == (o.flat ? ui_detection::still::flatten : 0u) &&
              (!o.flat || uniform_field(o)),
            std::string(label) + ": flat outside an enabled active run, the still word pushed without one, or a non-uniform field");
          require(h2_measures || (!o.evidence && o.sample.still.phase == still_screen::phase::none &&
              !o.sample.evidence.still_compared), std::string(label) + ": rule H2 ran outside SDR");
        }
        return std::pair{entry, o};
      };
      if (h2_measures) {
        // The shadow: 2.5 s of the loading screen never flattens, but the run
        // enters at the frame reading the sample 2 s after its first passing
        // one. After the identity change the first sample compares with the
        // screen before (moving), so the run starts at the second (read by
        // frame 3) and its sample 2 s later is read by frame 23.
        const auto [shadow_entry, shadow] = loading_screen(false, 25, "SDR loading screen, shadow");
        require(shadow_entry == 23 && !shadow.flat && shadow.evidence == 1 && !shadow.sample.source_kind &&
            shadow.sample.still.phase == still_screen::phase::active && shadow.sample.still.run_ms >= ui_detection::still::run_ms &&
            shadow.sample.evidence.still_compared == cells && shadow.sample.evidence.still_cells == cells &&
            shadow.sample.evidence.scene.n >= ui_detection::scene::min_edges && shadow.sample.evidence.scene.d <= .05f,
          "The SDR loading screen's shadow did not enter at its 2 s sample with every cell compared and still: " +
            std::to_string(shadow_entry));
        // Enabled: flat from the same frame on, as source 11 in its samples.
        const auto [flat_entry, flat_outcome] = loading_screen(true, 25, "SDR loading screen, enabled");
        require(flat_entry == 23 && flat_outcome.flat && flat_outcome.sample.source_kind == ui_detection::source_still &&
            flat_outcome.sample.covered == pixels && flat_outcome.sample.evidence.frame_reason != ui_detection::frame_reason_decided &&
            flat_outcome.sample.evidence.still_cells == cells,
          "The enabled SDR loading screen was not flat as source 11 from the frame reading its 2 s sample: " +
            std::to_string(flat_entry));
        // Depth that is not the frame's own (about 12% of the live loading
        // screen's samples): its evidence is invalid for H1, but H2 counts D
        // against the depth the warp uses, so the run and the flat frames stay.
        gpu.depth_current = false;
        for (unsigned i = 0; i != 4; ++i) {
          const auto stale = frame(banded_color);
          require(stale.flat && stale.sample.still.phase == still_screen::phase::active &&
              gpu.renderer.consumed_detection().still_bits == ui_detection::still::flatten,
            "Depth that is not the frame's own ended the still screen's run");
        }
        const auto stale_read = frame(banded_color);
        require(stale_read.flat && !stale_read.sample.evidence.scene.valid && stale_read.sample.evidence.scene.ran &&
            stale_read.sample.source_kind == ui_detection::source_still, "A reused-depth sample was valid or not flat as source 11");
        gpu.depth_current = true;
        // The screen moves: the frame that first shows it is still flat (its
        // sample is not read yet); the next frame reads that moving sample and
        // releases at once, and a new run starts from the frame after.
        const auto moved = frame(panned_color);
        const auto released = frame(panned_color);
        require(moved.flat && !released.flat && released.sample.source_kind == ui_detection::source_still &&
            std::uint64_t(released.sample.evidence.still_cells) * 100 < std::uint64_t(cells) * ui_detection::still::still_percent &&
            released.sample.still.phase == still_screen::phase::none && !gpu.renderer.consumed_detection().still_bits,
          "A moving loading screen was not released by the first frame that read its sample");
        const auto again = frame(panned_color);
        require(!again.flat && again.sample.still.phase == still_screen::phase::pending,
          "The still screen after the move did not start a new run");
      } else {
        // scRGB: never in scope, flattening enabled or not.
        const auto [entry, last] = loading_screen(true, 25, "scRGB loading screen, enabled");
        require(!entry && !last.flat, "An scRGB loading screen entered rule H2");
      }
      loading.set_still_flatten(false);
      source.session = &policy;
      inputs = {};
      report << "still-screen D3D11 scope=" << h2_measures << " shadow_entry_frame=" << (h2_measures ? 23 : 0)
             << " enabled_flat_source11=" << h2_measures << " reused_depth_keeps_run=" << h2_measures
             << " release_next_read_sample=" << h2_measures << " scrgb_never=" << !h2_measures << '\n';
      std::printf("PASS D3D11 still screen (H2, fix 2): %s\n", h2_measures ?
        "an SDR loading screen without a UI source (unaccepted 0x48, D < 0, every cell still) enters at the frame reading its 2 s sample; the shadow only reports it, enabled it is flat as source 11 (uniform field), reused depth keeps the run, and the first moving sample read releases it" :
        "scRGB output is out of scope: the same loading screen, flattening enabled, never runs, measures or pushes the still word");
    }
  }
  // Fix 3, the pre-UI change set (docs/reshade-sbs.md, UI decision
  // framework; game3d_ui_change_set.h), as Stellar Blade SDR shows it: its
  // cleared offscreen target holds the scene drawn before the UI at alpha 0,
  // copied before the first clear after a Present, so the copy offered with a
  // render holds the pre-UI image of the Present one before (presents_since_
  // copy 1); the presented alpha is accepted and a uniform 1.0 in menus (the
  // FG-off gameplay alpha that equals the UI colour tag), and the session's
  // ledger proved the layer the pre-UI scene image (its pre_ui key). The
  // scene moves every frame, so only the retained Present the copy shows
  // pairs with it exactly. UIPinOnlyUI=0 (the default) only measures
  // and logs the pair (the change-set shadow) and the menu stays flat by the
  // whole-frame alpha (source 4); =1 offers the pre-UI change set and the
  // refine rule pins only the changed pixels (source 12) while full pages and
  // a black pre-UI image stay flat. Without an exact pairing (the copy's own
  // interval, frame generation) it is never offered. Then the HUD-less
  // pairing's ring isolation, Dump 3D's retained Presents and the GPU cost.
  void verify_pre_ui_change_set(fixture &gpu, std::ostream &report, const fs::path &directory, bool timing_only = false) {
    using namespace sunshine_game3d;
    namespace candidate = ui_detection::candidate;
    const unsigned width = gpu.width, height = gpu.height;
    const auto pixels = size_t(width) * height;
    const auto bpp = gpu.color == 2 ? 8u : 4u;
    D3D11_TEXTURE2D_DESC color_desc{}; gpu.source->GetDesc(&color_desc);
    color_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    using rgba = std::array<float, 4>;
    const auto encode = [&](const std::function<rgba(unsigned, unsigned)> &pixel) {
      std::vector<unsigned char> bytes(pixels * bpp);
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        const auto v = pixel(x, y);
        if (gpu.color == 2) for (unsigned c = 0; c != 4; ++c) { const auto half = half_bits(v[c]); std::memcpy(bytes.data() + i * bpp + c * 2, &half, 2); }
        else for (unsigned c = 0; c != 4; ++c) bytes[i * bpp + c] = static_cast<unsigned char>(std::lround(std::clamp(v[c], 0.f, 1.f) * 255));
      }
      return bytes;
    };
    std::vector<ComPtr<ID3D11Texture2D>> textures;
    std::vector<ComPtr<ID3D11ShaderResourceView>> views;
    const auto view_of = [&](const std::vector<unsigned char> &bytes, ComPtr<ID3D11Texture2D> *out = nullptr) {
      const D3D11_SUBRESOURCE_DATA data{bytes.data(), width * bpp, 0};
      ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> view;
      checked(gpu.device->CreateTexture2D(&color_desc, &data, &texture), "pre-UI change-set texture");
      checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "pre-UI change-set view");
      if (out) *out = texture;
      textures.push_back(texture); views.push_back(view);
      return api::resource_view{reinterpret_cast<std::uint64_t>(view.Get())};
    };
    // The UI of the Equipment page: a tab row, the left icon panel, the
    // bottom-right prompts, a one-pixel hint line and an isolated speck. A
    // changed pixel stays in the mask with at least 3 changed pixels in its
    // 3x3 window: the speck and the line's two ends go.
    std::vector<std::uint8_t> ui(pixels, 0);
    const auto fill = [&](unsigned x0, unsigned y0, unsigned x1, unsigned y1) {
      for (unsigned y = y0; y < y1 && y < height; ++y) for (unsigned x = x0; x < x1 && x < width; ++x) ui[size_t(y) * width + x] = 1;
    };
    fill(width / 4, height / 20, width * 3 / 4, height / 20 + std::max(2u, height / 40));
    fill(width / 40, height / 4, width / 40 + std::max(2u, width / 20), height * 3 / 4);
    fill(width * 3 / 4, height * 9 / 10, width * 19 / 20, height * 19 / 20);
    fill(width / 2, height / 2, width / 2 + width / 10, height / 2 + 1);
    fill(width * 3 / 5, height * 3 / 10, width * 3 / 5 + 1, height * 3 / 10 + 1);
    std::vector<std::uint8_t> filtered(pixels, 0);
    std::uint32_t ui_pixels = 0, kept = 0;
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
      const size_t i = size_t(y) * width + x;
      if (!ui[i]) continue;
      ++ui_pixels;
      unsigned count = 0;
      for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const int nx = int(x) + dx, ny = int(y) + dy;
        if (nx >= 0 && ny >= 0 && nx < int(width) && ny < int(height)) count += ui[size_t(ny) * width + nx];
      }
      if (count >= ui_detection::change_set::min_neighbourhood) { filtered[i] = 1; ++kept; }
    }
    require(kept < ui_pixels && kept * 4 < pixels, "The fixture's UI has nothing for the 3x3 rule to remove, or is not selective");
    // A moving scene (checkers that shift every frame, lit everywhere) and
    // the page's presented colour over it, opaque everywhere.
    const auto scene = [&](unsigned phase, bool lit = true) {
      return [=](unsigned x, unsigned y) -> rgba {
        return lit ? rgba{0.f, ((x + 3 * phase) / 4 + y / 4) % 2 ? .75f : .25f, .25f, 1.f} : rgba{0.f, 0.f, 0.f, 1.f};
      };
    };
    // Fix 4: the Equipment page with its bottom dim band (dump 033), the
    // scene darkened by a black band whose opacity rises to 0.3 over the
    // bottom fifth of the frame (less than 1/64 per row), under the UI; the
    // change set stays selective (under a quarter of the frame changed, half
    // of the tiles unchanged).
    enum class page { equipment, settings, loading, dimmed };
    const unsigned dim_rows = height / 5, dim_top = height - dim_rows;
    const auto presented_of = [&](page kind, unsigned phase) {
      const auto base = scene(phase, kind != page::loading);
      return encode([&](unsigned x, unsigned y) -> rgba {
        if (kind == page::settings) return {.5f, .5f, .5f, 1.f};
        if (ui[size_t(y) * width + x]) return rgba{1.f, .5f, 1.f, 1.f};
        auto v = base(x, y);
        if (kind == page::dimmed && y >= dim_top) {
          const float keep = 1.f - .3f * float(y - dim_top + 1) / float(dim_rows + 1);
          for (unsigned c = 0; c != 3; ++c) v[c] *= keep;
        }
        return v;
      });
    };
    // The colours the shader loads from an encoded image.
    const auto decode = [&](const std::vector<unsigned char> &bytes) {
      std::vector<ui_darkening::rgb> result(pixels);
      for (size_t i = 0; i != pixels; ++i)
        for (unsigned c = 0; c != 3; ++c) {
          if (gpu.color == 2) { std::uint16_t half; std::memcpy(&half, bytes.data() + i * bpp + c * 2, 2); result[i][c] = half_float(half); }
          else result[i][c] = float(bytes[i * bpp + c]) / 255.f;
        }
      return result;
    };
    // The cleared target: the pre-UI scene at alpha 0 (black on the loading
    // screen, lit on far fewer than 1% of pixels).
    const auto layer_of = [&](page kind, unsigned phase) {
      const auto base = scene(phase, kind != page::loading);
      return encode([&](unsigned x, unsigned y) -> rgba { auto v = base(x, y); v[3] = 0.f; return v; });
    };
    ComPtr<ID3D11Texture2D> layer_texture;
    const auto layer_view = view_of(layer_of(page::equipment, 0), &layer_texture);
    const auto layer_flags = ui_layer::detection_flags(static_cast<api::format>(color_desc.Format));

    alpha_auto_policy sb;
    const auto current_key = gpu.key(ui_selection::kind::current), pre_ui_key = gpu.key(ui_selection::kind::pre_ui);
    require(sb.restore(current_key + ',' + pre_ui_key).restored == 2, "The current alpha and pre-UI keys were not restored");
    const auto layer_signature = ui_selection::signature{ui_selection::kind::ui_layer, std::uint32_t(color_desc.Format), gpu.color};
    require(sb.pre_ui_proven(layer_signature), "The restored pre-UI key did not prove the layer");
    alpha_auto_source source;
    source.session = &sb; source.now_ms = source.tick_ms = 500000;
    source.epoch = 61; source.revision = 1; source.sequence = 1;
    ui_detection_inputs inputs;
    ui_render_input ui_input;
    ui_input.automatic = &source; ui_input.detection = &inputs;
    struct outcome { alpha_auto_decision sample; ui_detection_snapshot run; image mask; };
    // One render of a page at the next phase. The layer offered is the
    // previous render's pre-UI image (exact when presents_ago is 1, as the
    // tracker counts it); the next render's layer is this one's.
    unsigned phase = 0;
    page last = page::equipment;
    // The presented colours of the last three renders, newest last.
    std::array<std::vector<unsigned char>, 3> presented_history;
    const auto step = [&](page kind, std::uint32_t presents_ago = 1, bool fg = false, std::uint64_t advance = 100,
        const fs::path &dump = {}) {
      const auto layer = layer_of(last, phase);
      gpu.context->UpdateSubresource(layer_texture.Get(), 0, nullptr, layer.data(), width * bpp, 0);
      ++phase;
      last = kind;
      inputs.current_color = true;
      inputs.layer = layer_view; inputs.layer_flags = layer_flags;
      inputs.layer_presents_ago = presents_ago; inputs.fg_known_off = !fg;
      gpu.original = presented_of(kind, phase);
      std::rotate(presented_history.begin(), presented_history.begin() + 1, presented_history.end());
      presented_history.back() = gpu.original;
      gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), width * bpp, 0);
      source.now_ms += advance; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.renderer.begin_present();
      gpu.render(true, 1, false, false, false, {}, dump, {}, nullptr, nullptr, nullptr, nullptr, &ui_input);
      return outcome{gpu.renderer.consumed_alpha_auto(), gpu.renderer.consumed_detection(),
        gpu.read(gpu.renderer.diagnostics().ui_source)};
    };
    const auto is_flat = [&](const image &mask) {
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x)
        if (mask.channel(x, y, 0) != 1.f) return false;
      return true;
    };
    const auto is_filtered_ui = [&](const image &mask) {
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x)
        if (mask.channel(x, y, 0) != float(filtered[size_t(y) * width + x])) return false;
      return true;
    };
    const auto pre_ui_rules = [](const ui_detection_snapshot &run) {
      return run.rules_bits & (ui_detection::rules::pin_only_ui | ui_detection::change_set::pair_mask);
    };
    // A frame without protection ends any mask an earlier section left.
    gpu.original = presented_of(page::equipment, phase);
    gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), width * bpp, 0);
    source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
    gpu.render(false, 1, false, false, false, {}, {}, {}, nullptr, &source);
    step(page::equipment, 0);

    if (!timing_only) {
      // (a) The shadow (UIPinOnlyUI=0): never offered and the menu is
      // flat by the accepted whole-frame alpha, while each sample measures the
      // layer against the retained Present it shows: exactly the UI pixels
      // changed, verified against the Presents 0 and 2 back, valid, and with
      // the switch it would refine. The ledger keeps its keys.
      // The first frame reads the sample of the copy's own interval.
      step(page::equipment);
      const auto stored = sb.stored();
      const auto counters_before = sb.counters();
      outcome shadow{};
      for (unsigned i = 0; i != 4; ++i) {
        shadow = step(page::equipment);
        require(!(shadow.run.candidates & candidate::pre_ui) && shadow.run.layer_pairing == change_set::pair_class::retained &&
            shadow.run.layer_presents_ago == 1u && !(shadow.run.rules_bits & ui_detection::rules::pin_only_ui) &&
            ui_detection::change_set::pair_offset(shadow.run.rules_bits) == 1u && is_flat(shadow.mask),
          "The shadow offered the pre-UI change set, lost the retained pairing or did not leave the menu flat");
      }
      const auto &cs = shadow.sample.evidence.change_set;
      require(shadow.sample.source_kind == 4u && shadow.sample.covered == pixels && !shadow.sample.evidence.refined &&
          shadow.sample.change_set.measured && shadow.sample.change_set.pairing == change_set::pair_class::retained &&
          shadow.sample.change_set.offset == 1u && shadow.sample.change_set.valid && shadow.sample.change_set.would_refine &&
          !shadow.sample.change_set.enabled && cs.changed == ui_pixels && cs.filtered == kept && cs.changed_1 == ui_pixels &&
          cs.changed_2 > ui_pixels && cs.matching_tiles >= 128u && !cs.nonfinite && !cs.judge_kind,
        "The shadow sample did not measure exactly the UI pixels against the retained Present, or would not refine: changed=" +
          std::to_string(cs.changed) + " filtered=" + std::to_string(cs.filtered) + " ui=" + std::to_string(ui_pixels) +
          " kept=" + std::to_string(kept) + " changed_1=" + std::to_string(cs.changed_1) + " changed_2=" +
          std::to_string(cs.changed_2) + " tiles=" + std::to_string(cs.matching_tiles));
      const auto counted = sb.counters() - counters_before;
      require(sb.stored() == stored && sb.pre_ui_proven(layer_signature) && counted[ui_counter::change_set_samples] >= 3 &&
          counted[ui_counter::change_set_retained] == counted[ui_counter::change_set_samples] &&
          counted[ui_counter::change_set_valid] == counted[ui_counter::change_set_samples] &&
          counted[ui_counter::change_set_would_refine] == counted[ui_counter::change_set_samples] &&
          counted[ui_counter::change_set_pair_verified] == counted[ui_counter::change_set_samples] &&
          !counted[ui_counter::change_set_pair_contradicted] && !counted[ui_counter::refined] && !counted.decided(12) &&
          counted.decided(4) >= 1,
        "The shadow changed the ledger, or its counters are not exact");
      // (b) UIPinOnlyUI=1: offered with the retained pairing, refined
      // from the shapeless alpha, and only the changed pixels the 3x3 rule
      // keeps are pinned; the scene stays unpinned.
      sb.set_pin_only_ui(true);
      outcome pinned{};
      for (unsigned i = 0; i != 3; ++i) {
        pinned = step(page::equipment);
        require((pinned.run.candidates & candidate::pre_ui) && (pinned.run.accepted & candidate::pre_ui) &&
            !(pinned.run.candidates & candidate::exact) &&
            pre_ui_rules(pinned.run) == (ui_detection::rules::pin_only_ui | (1u << ui_detection::change_set::pair_shift)) &&
            is_filtered_ui(pinned.mask),
          "UIPinOnlyUI=1 did not offer the pre-UI change set or pin exactly the changed pixels the 3x3 rule keeps");
      }
      require(pinned.sample.source_kind == ui_detection::source_pre_ui && pinned.sample.covered == ui_pixels &&
          pinned.sample.evidence.refined && pinned.sample.evidence.s1_source == ui_detection::source_pre_ui &&
          pinned.sample.change_set.enabled && pinned.sample.change_set.would_refine && sb.stored() == stored,
        "The refined sample did not report source 12 with the refined bit");
      // (b2) Fix 4, rule P2 (pin only UI): the page with its bottom dim band.
      // The refined pre-UI set (12) holds the band's changed pixels; with the
      // switch on they do not pin unless sharp structure or an unproven
      // region keeps them (beside the prompts), exactly as game3d_ui_darkening.h's
      // reference of the layer pair says, and the UI's colour pixels keep
      // their pins. The next frame's sample carries the same counts.
      std::uint64_t dimmed_unpinned = 0, dimmed_kept = 0;
      {
        step(page::dimmed);
        const auto pre_ui = decode(layer_of(page::dimmed, phase));
        const auto dimmed = step(page::dimmed);
        const auto paired = decode(presented_history[1]);
        float t;
        std::memcpy(&t, &dimmed.run.threshold_bits, sizeof(t));
        const auto k = float(ui_detection::change_set::inferred_scale);
        const auto mask = ui_darkening::change_set_mask(width, height, paired, pre_ui, t, k, gpu.color, true);
        const auto reference = ui_darkening::reference_change_set(width, height, paired, pre_ui, mask, t, k, gpu.color);
        std::uint64_t band_unpinned = 0, band_changed = 0, differs = 0;
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
          const auto i = size_t(y) * width + x;
          const bool unpinned = reference.classes[i] == ui_darkening::pixel_class::unpinned;
          differs += dimmed.mask.channel(x, y, 0) != (mask[i] && !unpinned ? 1.f : 0.f);
          require(!filtered[i] || (mask[i] && reference.classes[i] == ui_darkening::pixel_class::colour),
            "A UI colour pixel of the dimmed page was not pinned as colour by the reference");
          if (y >= dim_top && !ui[i] && mask[i]) { ++band_changed; band_unpinned += unpinned; }
        }
        require(dimmed.sample.source_kind == ui_detection::source_pre_ui && (dimmed.run.candidates & candidate::pre_ui) &&
            pre_ui_rules(dimmed.run) == (ui_detection::rules::pin_only_ui | (1u << ui_detection::change_set::pair_shift)) &&
            (dimmed.run.rules_bits & ui_detection::rules::darkening_measured) && !differs,
          "The dimmed page's mask differs from the CPU reference on " + std::to_string(differs) + " pixels");
        require(band_changed && band_unpinned * 2 >= band_changed && reference.n.unpinned == band_unpinned,
          "The dimmed page's band was not mostly unpinned by the reference, or something else was: " + std::to_string(band_unpinned) + " of " +
            std::to_string(band_changed));
        const auto next = step(page::dimmed);
        require(next.sample.source_kind == ui_detection::source_pre_ui && next.sample.darkening.measured &&
            next.sample.darkening.applied && next.sample.darkening.unpinned == reference.n.unpinned &&
            next.sample.darkening.kept == reference.n.kept,
          "The dimmed page's sample words 62-63 differ from the CPU reference: unpinned=" +
            std::to_string(next.sample.darkening.unpinned) + " kept=" + std::to_string(next.sample.darkening.kept) + " reference " +
            std::to_string(reference.n.unpinned) + "/" + std::to_string(reference.n.kept));
        dimmed_unpinned = reference.n.unpinned;
        dimmed_kept = reference.n.kept;
        step(page::equipment);
      }
      // (c) Settings, a full page: the changed pixels are the whole frame, an
      // invalid set, so the whole-frame alpha keeps it flat. A black loading
      // screen (pre-UI image lit on fewer than 1% of pixels) also stays flat.
      // The first frame of a page pairs the copy of the page before it (the
      // pair describes the Present one before), so the page is checked from
      // its second frame and its sample from the third.
      for (const auto kind : {page::settings, page::loading}) {
        const char *label = kind == page::settings ? "settings page" : "black loading screen";
        step(kind);
        const auto second = step(kind);
        const auto flat = step(kind);
        require((flat.run.candidates & candidate::pre_ui) && is_flat(second.mask) && is_flat(flat.mask) &&
            flat.sample.source_kind == 4u && !flat.sample.evidence.refined && flat.sample.change_set.measured &&
            !flat.sample.change_set.valid,
          std::string(label) + ": the pre-UI change set refined an invalid set");
      }
      step(page::equipment);
      require(is_filtered_ui(step(page::equipment).mask), "The Equipment page did not refine again after a full page");
      // (d) T1: a frame without an exact pairing (the copy's own interval)
      // misses the pre-UI change set the previous real frame offered (T1's
      // change-set gap, b2 word 5; never accepted_missing), so the refined
      // decision is reused once, then the frame is flat; frame generation is
      // a late pairing, never offered, and so is the first Present after it
      // (the Present the copy shows was not known off).
      const auto missing = step(page::equipment, 0);
      require(!(missing.run.candidates & candidate::pre_ui) && missing.run.layer_pairing == change_set::pair_class::unavailable &&
          (missing.run.rules_bits & ui_detection::change_set::gap) &&
          !(missing.run.flags & ui_detection::per_frame_accepted_missing) && is_filtered_ui(missing.mask),
        "A frame without its pair did not reuse the refined decision once through the change-set gap");
      const auto spent = step(page::equipment, 0);
      require(!(spent.run.candidates & candidate::pre_ui) && is_flat(spent.mask) && spent.sample.evidence.reused,
        "A second frame without its pair reused again, or the sample lost the reuse");
      const auto generated = step(page::equipment, 1, true);
      require(!(generated.run.candidates & candidate::pre_ui) && generated.run.layer_pairing == change_set::pair_class::late &&
          !ui_detection::change_set::pair_offset(generated.run.rules_bits) && is_flat(generated.mask),
        "Frame generation offered the pre-UI change set or paired the layer with a retained Present");
      const auto late = step(page::equipment, 1, true);
      require(late.sample.change_set.pairing == change_set::pair_class::late && late.sample.change_set.offset == 0u &&
          !late.sample.change_set.valid && late.sample.evidence.change_set.changed > ui_pixels,
        "The late pair's shadow was not measured against the current Present");
      const auto first_off = step(page::equipment);
      require(!(first_off.run.candidates & candidate::pre_ui) && first_off.run.layer_pairing == change_set::pair_class::late &&
          is_flat(first_off.mask),
        "The first Present after frame generation paired the Present before it, which was not known off");
      require(is_filtered_ui(step(page::equipment).mask), "The retained pairing did not refine again after frame generation");
      // (e) Dump 3D: an armed dump retains every Present and the dumped
      // render defers its own copy, so the package carries both retained
      // Presents and the consumed layer copy; the next render still pairs.
      gpu.renderer.set_dump_retention(true, true);
      const auto dumped = step(page::equipment, 1, false, 100, directory / "pre-ui-change-set-dump");
      gpu.renderer.set_dump_retention(false, false);
      require(is_filtered_ui(dumped.mask), "The dumped render did not refine");
      std::ifstream stored_manifest(directory / "pre-ui-change-set-dump" / "manifest.json");
      const auto manifest = nlohmann::json::parse(stored_manifest);
      const auto &metadata = manifest.at("producer_metadata");
      const auto &detection = metadata.at("replay").at("ui_detection");
      std::set<std::string> kinds;
      std::map<std::string, std::vector<unsigned char>> retained_bytes;
      for (const auto &artifact : manifest.at("artifacts")) {
        const auto kind = artifact.at("kind").get<std::string>();
        kinds.insert(kind);
        if (kind.rfind("retained_present_", 0) == 0) {
          std::ifstream file(directory / "pre-ui-change-set-dump" / artifact.at("file").get<std::string>(), std::ios::binary);
          retained_bytes[kind] = {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        }
      }
      bool all_captured = true;
      for (const auto &row : metadata.at("change_set_artifacts").at("artifacts")) all_captured = all_captured && row.at("captured").get<bool>();
      require(kinds.count("retained_present_1") && kinds.count("retained_present_2") && kinds.count("ui_layer_detected") &&
          all_captured && detection.at("layer_pairing") == "retained" && detection.at("layer_presents_ago") == 1u &&
          detection.at("rules_bits") == dumped.run.rules_bits && detection.at("still_bits") == 0u &&
          metadata.at("replay").at("ui_pin").at("decision_texels") == ui_detection::change_set_decision_texels &&
          retained_bytes["retained_present_1"] == presented_history[1] && retained_bytes["retained_present_2"] == presented_history[0],
        "Dump 3D lost the retained Presents (the colours one and two Presents back), the consumed layer copy or the pairing metadata");
      const auto after_dump = step(page::equipment);
      require((after_dump.run.candidates & candidate::pre_ui) && after_dump.run.layer_pairing == change_set::pair_class::retained &&
          is_filtered_ui(after_dump.mask), "The render after a dump lost its retained pair");
      sb.set_pin_only_ui(false);
      // (f) Ring isolation: a late HUD-less pairing sees only Presents
      // retained for it. After every retention request lapsed (more than its
      // 120 Presents), the first late HUD-less render finds no pair and the
      // second pairs, whether the renders before it retained every Present
      // for the proven layer's shadow (path 1) or nothing (path 0: frame
      // generation).
      {
        const auto hudless = view_of(encode(scene(phase + 50)));
        alpha_auto_policy plain, proven;
        require(proven.restore(current_key + ',' + pre_ui_key).restored == 2, "The ring isolation session was not restored");
        std::array<std::array<std::uint32_t, 2>, 2> candidates_of{};
        std::array<std::array<std::vector<unsigned char>, 2>, 2> masks;
        for (unsigned path = 0; path != 2; ++path) {
          ++source.epoch;
          source.session = &proven;
          for (unsigned i = 0; i != 125; ++i) step(page::equipment, 1, path == 0);
          source.session = &plain;
          inputs.layer = {}; inputs.layer_flags = 0;
          inputs.hudless = hudless; inputs.hudless_presents_ago = 1;
          for (unsigned i = 0; i != 2; ++i) {
            gpu.original = presented_of(page::equipment, ++phase);
            gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), width * bpp, 0);
            source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
            gpu.renderer.begin_present();
            gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui_input);
            candidates_of[path][i] = gpu.renderer.consumed_detection().candidates;
            masks[path][i] = gpu.read(gpu.renderer.diagnostics().ui_source).bytes;
          }
          inputs.hudless = {}; inputs.hudless_presents_ago = 0;
        }
        require(candidates_of[0] == candidates_of[1] && masks[0] == masks[1] && !(candidates_of[0][0] & candidate::hudless) &&
            (candidates_of[0][1] & candidate::hudless),
          "Renders that retained Presents for the layer changed the late HUD-less pairing");
        source.session = &sb;
      }
      report << "pre-ui-change-set D3D11 ui=" << ui_pixels << " kept=" << kept << " shadow_retained_valid_would_refine=1"
        " ledger_unchanged=1 refined_source12=1 settings_flat=1 loading_flat=1 t1_reuse_once=1 fg_late_never_offered=1"
        " dump_retained_presents=1 ring_isolation=1 dimmed_band_unpinned=" << dimmed_unpinned << " dimmed_kept=" << dimmed_kept << "\n";
      std::printf("PASS D3D11 pre-UI change set (fix 3, Stellar Blade SDR): the shadow measures exactly the UI pixels (%u, %u after the 3x3 rule) against the retained Present the layer copy shows, verified against the Presents 0 and 2 back, valid and would refine, while the menu stays flat by the accepted whole-frame alpha and the ledger is unchanged; UIPinOnlyUI=1 refines it to source 12 pinning only those pixels; a settings page and a black loading screen stay flat; a frame without its pair reuses the refined decision once; frame generation is a late pairing, never offered; Dump 3D carries both retained Presents and the consumed layer copy; a late HUD-less pairing is unchanged by layer retention\n",
        ui_pixels, kept);
    }

    // (g) GPU cost of the detection stage (the span from the source copy to
    // the end of detection, the retention copy included), the median per
    // frame by configuration and by frame kind (a sample frame, one per 100
    // ms, or another): no layer offered; frame generation on (no retention,
    // the shadow measures the proven layer's late pair on sample frames); the
    // shadow (retained pairs); the switch on (every Present retained, the 3x3
    // mask of source 12); frame generation on with every Present retained
    // (the copy alone); and a real UI layer that is never proven (Witcher 3,
    // Stellar Blade HDR: no shadow, no retention). Renders only, without readbacks, each completed
    // before the next. With a frozen control shader the control renderer is
    // timed too (an A/B of a shader variant).
    //
    // Timing at sustained clocks (the timing configuration): an idle GPU
    // stays at its lowest clocks (P8, 405 MHz memory on an RTX 5080) between
    // frames that wait on the CPU, where each full-frame read costs about a
    // millisecond, so such medians measure memory latency, not the passes.
    // The scene's eight phases (its checkers shift three pixels a frame, an
    // eight-pixel period) are uploaded once and copied on the GPU, and before
    // each timed render the GPU copies a 256 MB buffer warm_copies times
    // (about 5 ms at full clocks), which keeps it busy and leaves none of the
    // frame's inputs in its L2. nvidia-smi samples the clock state during
    // each configuration's timed frames.
    renderer *timed = &gpu.renderer;
    const page timed_page = last;
    std::array<ComPtr<ID3D11Texture2D>, 8> timed_presented, timed_layer;
    for (unsigned k = 0; k != timed_presented.size(); ++k) {
      view_of(presented_of(page::equipment, k), &timed_presented[k]);
      view_of(layer_of(timed_page, k), &timed_layer[k]);
    }
    const unsigned warm_copies = timing_only ? 8u : 0u;
    ComPtr<ID3D11Buffer> flush_from, flush_to;
    if (warm_copies) {
      D3D11_BUFFER_DESC desc{};
      desc.ByteWidth = 256u << 20;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      checked(gpu.device->CreateBuffer(&desc, nullptr, &flush_from), "timing warm-up buffer");
      checked(gpu.device->CreateBuffer(&desc, nullptr, &flush_to), "timing warm-up buffer");
    }
    const auto warm = [&](unsigned copies) {
      for (unsigned i = 0; i != copies; ++i) gpu.context->CopyResource(flush_to.Get(), flush_from.Get());
    };
    const auto gpu_state = [timing_only]() -> std::string {
      if (!timing_only) return "-";
      std::string text;
      if (FILE *pipe = _popen("nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem --format=csv,noheader,nounits 2>NUL", "r")) {
        char line[128];
        if (std::fgets(line, sizeof(line), pipe)) text = line;
        _pclose(pipe);
      }
      text.erase(std::remove_if(text.begin(), text.end(), [](char c) { return c == ' ' || c == '\r' || c == '\n'; }), text.end());
      std::replace(text.begin(), text.end(), ',', '/');
      return text.empty() ? std::string("unknown") : text;
    };
    const auto timed_step = [&](bool layer, bool fg, std::uint64_t advance) {
      gpu.context->CopyResource(layer_texture.Get(), timed_layer[phase % timed_layer.size()].Get());
      ++phase;
      inputs.current_color = true;
      inputs.layer = layer ? layer_view : api::resource_view{}; inputs.layer_flags = layer ? layer_flags : 0u;
      inputs.layer_presents_ago = 1; inputs.fg_known_off = !fg;
      gpu.context->CopyResource(gpu.source.Get(), timed_presented[phase % timed_presented.size()].Get());
      gpu.context->CopyResource(gpu.backbuffer.Get(), gpu.source.Get());
      warm(warm_copies);
      source.now_ms += advance; source.tick_ms = source.now_ms; ++source.sequence;
      timed->begin_present();
      render_frame_input frame;
      frame.color = {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())};
      frame.depth = {reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())};
      frame.scene = gpu.renderer.consumed_parameters();
      frame.ui = ui_input;
      auto *queue = observed_runtime->get_command_queue();
      require(timed->render(queue->get_immediate_command_list(), frame), "Timed change-set render failed");
      queue->flush_immediate_command_list();
      timed->finish_present();
      gpu.drain_render();
      return timed->consumed_detection().pre_ui_threshold_bits != 0;
    };
    struct cost { double sample_ms{}, other_ms{}; unsigned frames{}; std::string state; };
    const auto median = [](std::vector<double> values) {
      if (values.empty()) return 0.;
      std::sort(values.begin(), values.end());
      return values[values.size() / 2];
    };
    // A session without the pre-UI key: its layer is a real UI layer to the
    // renderer (never proven), as in Witcher 3 or Stellar Blade HDR.
    alpha_auto_policy unproven;
    require(unproven.restore(current_key).restored == 1, "The unproven timing session was not restored");
    const auto measure = [&](bool layer, bool pin, bool fg, bool retain_all, bool proven = true, bool darkening = true) {
      source.session = proven ? &sb : &unproven;
      sb.set_pin_only_ui(pin);
      timed->set_dump_retention(retain_all, false);
      // Fix 4: without the darkening passes, as a shader without them.
      timed->set_darkening_passes(darkening);
      std::vector<double> samples, others;
      gpu_timing discard;
      // Half a second of copies first: clocks ramp up over a few hundred
      // milliseconds of load.
      for (const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(warm_copies ? 500 : 0);
           std::chrono::steady_clock::now() < until;) {
        warm(warm_copies);
        gpu.drain_render();
      }
      for (unsigned i = 0; i != 8; ++i) timed_step(layer, fg, 16);
      timed->take_gpu_timing(discard);
      auto state = std::async(std::launch::async, gpu_state);
      for (unsigned i = 0; i != 64; ++i) {
        // Without a layer no frame pushes a pre-UI threshold: a sample is
        // one 100 ms after the last.
        const auto before = source.now_ms;
        const bool sample = timed_step(layer, fg, 16) || (!layer && (before + 16) / 100 != before / 100);
        gpu_timing t;
        if (!timed->take_gpu_timing(t) || t.frames != 1) continue;
        (sample ? samples : others).push_back(t.mean_ms[gpu_timing::detection]);
      }
      timed->set_dump_retention(false, false);
      timed->set_darkening_passes(true);
      source.session = &sb;
      return cost{median(samples), median(others), unsigned(samples.size() + others.size()), state.get()};
    };
    for (auto *candidate_renderer : {&gpu.renderer, gpu.has_control ? &gpu.control_renderer : nullptr}) {
      if (!candidate_renderer) continue;
      timed = candidate_renderer;
      const auto no_layer = measure(false, false, true, false), fg_on = measure(true, false, true, false),
        shadow_cost = measure(true, false, false, false), pinned_cost = measure(true, true, false, false),
        copy_cost = measure(true, false, true, true), unproven_cost = measure(true, false, false, false, false);
      // Fix 4: the switch on (source 12, the passes on every frame) and the
      // shadow (source 4, ineligible: the passes on sample frames return at
      // once) without the darkening passes.
      const auto pinned_plain = measure(true, true, false, false, true, false),
        shadow_plain = measure(true, false, false, false, true, false);
      sb.set_pin_only_ui(false);
      char text[1280];
      std::snprintf(text, sizeof(text),
        "pre-ui-change-set-gpu %s %ux%u detection_ms(median) no_layer={sample=%.4f other=%.4f} fg_on={sample=%.4f other=%.4f} "
        "shadow={sample=%.4f other=%.4f} switch_on={sample=%.4f other=%.4f} fg_on_retain_every_present={sample=%.4f other=%.4f} "
        "unproven_layer={sample=%.4f other=%.4f} frames=%u/%u/%u/%u/%u/%u retention_copy=%.4f "
        "switch_on_other_minus_shadow_other=%.4f shadow_sample_minus_other=%.4f unproven_sample_minus_other=%.4f "
        "warm_copies=%u gpu_state(pstate/sm/mem)=%s,%s,%s,%s,%s,%s\n",
        timed == &gpu.renderer ? "production" : "control",
        width, height, no_layer.sample_ms, no_layer.other_ms, fg_on.sample_ms, fg_on.other_ms, shadow_cost.sample_ms,
        shadow_cost.other_ms, pinned_cost.sample_ms, pinned_cost.other_ms, copy_cost.sample_ms, copy_cost.other_ms,
        unproven_cost.sample_ms, unproven_cost.other_ms, no_layer.frames, fg_on.frames, shadow_cost.frames, pinned_cost.frames,
        copy_cost.frames, unproven_cost.frames, copy_cost.other_ms - fg_on.other_ms,
        pinned_cost.other_ms - shadow_cost.other_ms, shadow_cost.sample_ms - shadow_cost.other_ms,
        unproven_cost.sample_ms - unproven_cost.other_ms, warm_copies, no_layer.state.c_str(), fg_on.state.c_str(),
        shadow_cost.state.c_str(), pinned_cost.state.c_str(), copy_cost.state.c_str(), unproven_cost.state.c_str());
      report << text;
      std::fputs(text, stdout);
      std::snprintf(text, sizeof(text),
        "pre-ui-change-set-darkening-gpu %s %ux%u detection_ms(median) switch_on={sample=%.4f other=%.4f} "
        "switch_on_without_darkening={sample=%.4f other=%.4f} shadow_without_darkening={sample=%.4f other=%.4f} "
        "darkening_per_enabled_frame=%.4f darkening_per_enabled_sample=%.4f ineligible_shadow_sample=%.4f "
        "ineligible_shadow_other=%.4f gpu_state=%s,%s\n",
        timed == &gpu.renderer ? "production" : "control", width, height, pinned_cost.sample_ms, pinned_cost.other_ms,
        pinned_plain.sample_ms, pinned_plain.other_ms, shadow_plain.sample_ms, shadow_plain.other_ms,
        pinned_cost.other_ms - pinned_plain.other_ms, pinned_cost.sample_ms - pinned_plain.sample_ms,
        shadow_cost.sample_ms - shadow_plain.sample_ms, shadow_cost.other_ms - shadow_plain.other_ms,
        pinned_plain.state.c_str(), shadow_plain.state.c_str());
      report << text;
      std::fputs(text, stdout);
    }
    // The presented colour the fixture keeps beside its source texture.
    gpu.original = presented_of(page::equipment, phase);
    gpu.context->UpdateSubresource(gpu.source.Get(), 0, nullptr, gpu.original.data(), width * bpp, 0);
    inputs = {};
  }
  // Fix 4, rule P2 (pin only UI; docs/reshade-sbs.md, UI decision
  // framework): an accepted offscreen UI layer (source 10) holding a smooth
  // black dim band over the bottom of the frame (alpha rising to 0.6, less
  // than 1/64 per row), sharp black 2-pixel strokes (four separate bars, as
  // the strokes of glyphs; a 2x2 junction's inner pixels would unpin, which
  // the pin band covers) and a white icon. In the default shadow (UIPinOnlyUI=0) the mask stays the
  // layer's raw alpha, the decisions stay 10, and every sample's words 62-63
  // equal game3d_ui_darkening.h's reference of the layer (unpinned, kept);
  // with the switch on the band no longer pins while the strokes and the icon
  // keep their pins, exactly as the reference says. With timing (the
  // --pin-only-ui-only configuration at 4K), the GPU cost of the darkening
  // passes on this layer, A/B against the same renderer without them.
  void verify_pin_only_ui_darkening(fixture &gpu, std::ostream &report, bool timing = false) {
    using namespace sunshine_game3d;
    const unsigned width = gpu.width, height = gpu.height;
    const auto pixels = size_t(width) * height;
    const auto bpp = gpu.color == 2 ? 8u : 4u;
    using rgba = std::array<float, 4>;
    std::vector<rgba> value(pixels, rgba{0.f, 0.f, 0.f, 0.f});
    // 1 the dim band, 2 a stroke, 3 the icon.
    std::vector<std::uint8_t> region(pixels, 0);
    const unsigned band = height * 3 / 8, top = height - band;
    for (unsigned y = top; y < height; ++y)
      for (unsigned x = 0; x < width; ++x) {
        value[size_t(y) * width + x] = {0.f, 0.f, 0.f, .6f * float(y - top + 1) / float(band + 1)};
        region[size_t(y) * width + x] = 1;
      }
    const auto fill = [&](unsigned x0, unsigned y0, unsigned x1, unsigned y1, rgba v, std::uint8_t r) {
      for (unsigned y = y0; y < y1 && y < top; ++y)
        for (unsigned x = x0; x < x1 && x < width; ++x) { value[size_t(y) * width + x] = v; region[size_t(y) * width + x] = r; }
    };
    const rgba black{0.f, 0.f, 0.f, 1.f}, white{1.f, 1.f, 1.f, 1.f};
    const unsigned stroke = 2, glyph = std::max(8u, height / 4), left = width / 8, row = height / 8;
    fill(left, row, left + glyph, row + stroke, black, 2);
    fill(left + glyph / 2 - 1, row + stroke + 2, left + glyph / 2 + 1, row + glyph, black, 2);
    fill(left + glyph + 4, row, left + glyph + 4 + stroke, row + glyph - stroke - 2, black, 2);
    fill(left + glyph + 8, row + glyph - stroke, left + 2 * glyph, row + glyph, black, 2);
    fill(width * 5 / 8, row, width * 5 / 8 + std::max(4u, width / 10), row + std::max(4u, height / 8), white, 3);
    // The layer in the fixture's format, and the values the shader loads.
    std::vector<unsigned char> bytes(pixels * bpp);
    std::vector<ui_darkening::rgb> colour(pixels);
    std::vector<float> alpha(pixels);
    for (size_t i = 0; i != pixels; ++i) {
      std::array<float, 4> loaded{};
      for (unsigned c = 0; c != 4; ++c) {
        if (gpu.color == 2) {
          const auto half = half_bits(value[i][c]);
          std::memcpy(bytes.data() + i * bpp + c * 2, &half, 2);
          loaded[c] = half_float(half);
        } else {
          const auto byte = static_cast<unsigned char>(std::lround(std::clamp(value[i][c], 0.f, 1.f) * 255));
          bytes[i * bpp + c] = byte;
          loaded[c] = float(byte) / 255.f;
        }
      }
      colour[i] = {loaded[0], loaded[1], loaded[2]};
      alpha[i] = loaded[3];
    }
    // The scRGB layer is a float layer, whose tolerance is linear.
    const auto reference = ui_darkening::reference_opacity(width, height, colour, alpha, gpu.color == 2);
    // The band's first rows round to alpha 0 at 4K in sRGB: no UI there.
    std::uint64_t band_pixels = 0, stroke_pixels = 0;
    for (size_t i = 0; i != pixels; ++i) {
      const auto c = reference.classes[i];
      const bool band_ui = region[i] == 1 && alpha[i] > 0.f;
      band_pixels += band_ui;
      stroke_pixels += region[i] == 2;
      require(band_ui ? c == ui_darkening::pixel_class::unpinned : region[i] == 2 ? c == ui_darkening::pixel_class::kept :
          region[i] == 3 ? c == ui_darkening::pixel_class::colour : c == ui_darkening::pixel_class::not_ui,
        "The darkening fixture does not mean what it says: the band must unpin, the strokes stay kept, the icon is colour (pixel " +
          std::to_string(i % width) + "," + std::to_string(i / width) + " region " + std::to_string(region[i]) + " class " +
          std::to_string(int(c)) + " alpha " + std::to_string(alpha[i]) + ")");
    }
    require(reference.n.unpinned == band_pixels && reference.n.kept == stroke_pixels, "The darkening fixture's reference counts differ");
    D3D11_TEXTURE2D_DESC desc{}; gpu.source->GetDesc(&desc); desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA data{bytes.data(), width * bpp, 0};
    ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> view;
    checked(gpu.device->CreateTexture2D(&desc, &data, &texture), "darkening layer texture");
    checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "darkening layer view");
    alpha_auto_policy policy;
    require(policy.restore(gpu.key(ui_selection::kind::ui_layer)).restored == 1, "The layer key was not restored");
    alpha_auto_source source;
    source.session = &policy; source.now_ms = source.tick_ms = 2000000;
    source.epoch = 71; source.revision = 1; source.sequence = 1;
    ui_detection_inputs inputs;
    inputs.layer = {reinterpret_cast<std::uint64_t>(view.Get())};
    inputs.layer_flags = ui_layer::detection_flags(static_cast<api::format>(desc.Format));
    ui_render_input ui;
    ui.automatic = &source; ui.detection = &inputs;
    gpu.pattern(std::vector<float>(pixels, 0.f));
    struct outcome { alpha_auto_decision sample; ui_detection_snapshot run; image mask; };
    const auto frame = [&](bool enabled) {
      policy.set_pin_only_ui(enabled);
      source.now_ms += 100; source.tick_ms = source.now_ms; ++source.sequence;
      gpu.renderer.begin_present();
      gpu.render(true, 1, false, false, false, {}, {}, {}, nullptr, nullptr, nullptr, nullptr, &ui);
      return outcome{gpu.renderer.consumed_alpha_auto(), gpu.renderer.consumed_detection(),
        gpu.read(gpu.renderer.diagnostics().ui_source)};
    };
    const auto masked = [&](const image &mask, bool enabled) {
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const auto i = size_t(y) * width + x;
        const float expected = enabled && reference.classes[i] == ui_darkening::pixel_class::unpinned ? 0.f : alpha[i];
        if (!(std::fabs(mask.channel(x, y, 0) - expected) <= 1e-6f)) return false;
      }
      return true;
    };
    const auto rules = [](const ui_detection_snapshot &run) {
      return run.rules_bits & (ui_detection::rules::pin_only_ui | ui_detection::rules::darkening_measured);
    };
    const auto counters_before = policy.counters();
    // (a) The shadow: the mask is the raw alpha, every frame is a sample that
    // runs the passes, and the samples carry the reference's counts.
    outcome shadow{};
    for (unsigned i = 0; i != 4; ++i) {
      shadow = frame(false);
      require(shadow.run.state == ui_detection_snapshot::run_state::ran && rules(shadow.run) == ui_detection::rules::darkening_measured &&
          masked(shadow.mask, false), "The shadow changed the layer's raw alpha mask or did not run the darkening passes");
      if (i)
        require(shadow.sample.source_kind == ui_detection::source_layer && shadow.sample.covered == reference.n.covered &&
            shadow.sample.darkening.measured && !shadow.sample.darkening.applied &&
            shadow.sample.darkening.unpinned == reference.n.unpinned && shadow.sample.darkening.kept == reference.n.kept,
          "A shadow sample's words 62-63 differ from the CPU reference: unpinned=" + std::to_string(shadow.sample.darkening.unpinned) +
            " kept=" + std::to_string(shadow.sample.darkening.kept) + " reference " + std::to_string(reference.n.unpinned) + "/" +
            std::to_string(reference.n.kept));
    }
    const auto counted = policy.counters() - counters_before;
    require(counted[ui_counter::darkening_samples] >= 2 &&
        counted[ui_counter::darkening_unpinned_samples] == counted[ui_counter::darkening_samples] &&
        counted[ui_counter::darkening_unpinned_px] == counted[ui_counter::darkening_samples] * reference.n.unpinned &&
        counted[ui_counter::darkening_kept_px] == counted[ui_counter::darkening_samples] * reference.n.kept &&
        counted.decided(ui_detection::source_layer) >= 1,
      "The darkening counters do not sum the committed samples");
    // (b) UIPinOnlyUI=1: from the first frame the band does not pin and the
    // strokes and the icon do; the samples say the rule applied.
    outcome pinned{};
    for (unsigned i = 0; i != 3; ++i) {
      pinned = frame(true);
      require(rules(pinned.run) == (ui_detection::rules::pin_only_ui | ui_detection::rules::darkening_measured) && masked(pinned.mask, true),
        "UIPinOnlyUI=1 did not unpin exactly the reference's darkening pixels");
    }
    require(pinned.sample.source_kind == ui_detection::source_layer && pinned.sample.darkening.measured &&
        pinned.sample.darkening.applied && pinned.sample.darkening.unpinned == reference.n.unpinned &&
        pinned.sample.darkening.kept == reference.n.kept, "The applied sample's words 62-63 differ from the CPU reference");
    // Back to the shadow: the raw alpha again, at once.
    require(masked(frame(false).mask, false), "Clearing the switch did not restore the raw alpha mask");
    if (!timing) {
      report << "pin-only-ui D3D11 band=" << band_pixels << " strokes=" << stroke_pixels << " unpinned=" << reference.n.unpinned
             << " kept=" << reference.n.kept << " shadow_raw_alpha=1 words_equal_reference=1 switch_on_band_unpinned=1 strokes_icon_pinned=1\n";
      std::printf("PASS D3D11 pin only UI (fix 4, rule P2): an accepted layer's smooth black dim band (%llu px) is measured in the shadow and unpinned with UIPinOnlyUI=1, its sharp black strokes (%llu px) and white icon keep their pins; words 62-63 and the masks equal the CPU reference\n",
        static_cast<unsigned long long>(band_pixels), static_cast<unsigned long long>(stroke_pixels));
    } else {
      // GPU cost of the detection stage, the median per frame by kind (a
      // sample frame or another), the switch on and off, with and without
      // the darkening passes, at sustained clocks as the pre-UI timing.
      D3D11_BUFFER_DESC buffer{};
      buffer.ByteWidth = 256u << 20;
      buffer.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      ComPtr<ID3D11Buffer> flush_from, flush_to;
      checked(gpu.device->CreateBuffer(&buffer, nullptr, &flush_from), "timing warm-up buffer");
      checked(gpu.device->CreateBuffer(&buffer, nullptr, &flush_to), "timing warm-up buffer");
      const auto warm = [&] { for (unsigned i = 0; i != 8; ++i) gpu.context->CopyResource(flush_to.Get(), flush_from.Get()); };
      const auto timed_step = [&] {
        gpu.context->CopyResource(gpu.backbuffer.Get(), gpu.source.Get());
        warm();
        source.now_ms += 16; source.tick_ms = source.now_ms; ++source.sequence;
        gpu.renderer.begin_present();
        render_frame_input frame_input;
        frame_input.color = {reinterpret_cast<std::uint64_t>(gpu.backbuffer.Get())};
        frame_input.depth = {reinterpret_cast<std::uint64_t>(gpu.depth_view.Get())};
        frame_input.scene = gpu.renderer.consumed_parameters();
        frame_input.ui = ui;
        auto *queue = observed_runtime->get_command_queue();
        require(gpu.renderer.render(queue->get_immediate_command_list(), frame_input), "Timed darkening render failed");
        queue->flush_immediate_command_list();
        gpu.renderer.finish_present();
        gpu.drain_render();
        return gpu.renderer.consumed_detection().pre_ui_threshold_bits != 0;
      };
      struct cost { double sample_ms{}, other_ms{}; };
      const auto median = [](std::vector<double> values) {
        if (values.empty()) return 0.;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
      };
      const auto measure = [&](bool enabled, bool passes) {
        policy.set_pin_only_ui(enabled);
        gpu.renderer.set_darkening_passes(passes);
        for (const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500); std::chrono::steady_clock::now() < until;) {
          warm();
          gpu.drain_render();
        }
        for (unsigned i = 0; i != 8; ++i) timed_step();
        gpu_timing discard;
        gpu.renderer.take_gpu_timing(discard);
        std::vector<double> samples, others;
        for (unsigned i = 0; i != 96; ++i) {
          const bool sample = timed_step();
          gpu_timing t;
          if (!gpu.renderer.take_gpu_timing(t) || t.frames != 1) continue;
          (sample ? samples : others).push_back(t.mean_ms[gpu_timing::detection]);
        }
        gpu.renderer.set_darkening_passes(true);
        return cost{median(samples), median(others)};
      };
      const auto on = measure(true, true), on_without = measure(true, false), shadow_cost = measure(false, true),
        shadow_without = measure(false, false);
      policy.set_pin_only_ui(false);
      char text[768];
      std::snprintf(text, sizeof(text),
        "pin-only-ui-gpu layer %ux%u detection_ms(median) switch_on={sample=%.4f other=%.4f} "
        "switch_on_without_darkening={sample=%.4f other=%.4f} shadow={sample=%.4f other=%.4f} "
        "shadow_without_darkening={sample=%.4f other=%.4f} darkening_per_enabled_frame=%.4f "
        "darkening_per_enabled_sample=%.4f darkening_per_shadow_sample=%.4f shadow_other_delta=%.4f\n",
        width, height, on.sample_ms, on.other_ms, on_without.sample_ms, on_without.other_ms, shadow_cost.sample_ms,
        shadow_cost.other_ms, shadow_without.sample_ms, shadow_without.other_ms, on.other_ms - on_without.other_ms,
        on.sample_ms - on_without.sample_ms, shadow_cost.sample_ms - shadow_without.sample_ms,
        shadow_cost.other_ms - shadow_without.other_ms);
      report << text;
      std::fputs(text, stdout);
    }
    inputs = {};
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
  if (argc < 6 || argc > 8) { std::fputs("usage: source_alpha_runtime official-ReShade64.dll fresh-output srgb|scrgb width height [frozen-control.hlsl] [--temporal-ui-probe|--temporal-ui-front-limit|--adaptive-ui-only|--pre-ui-change-set-only|--pin-only-ui-only]\n", stderr); return 2; }
  // The timing configuration at 4K, with a control renderer, runs longer.
  const bool long_run = std::any_of(argv + 6, argv + argc, [](const char *a) {
    return !std::strcmp(a, "--pre-ui-change-set-only") || !std::strcmp(a, "--pin-only-ui-only");
  });
  std::thread([long_run] { Sleep(long_run ? 900000 : 180000); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    const unsigned color = !std::strcmp(argv[3], "srgb") ? 1 : !std::strcmp(argv[3], "scrgb") ? 2 : 0;
    require(color != 0, "unsupported source transfer");
    const auto directory = fs::absolute(argv[2]);
    bool temporal_probe = false, temporal_front_limit = false, adaptive_only = false, change_set_only = false, pin_only_ui_only = false;
    fs::path control;
    for (int index = 6; index < argc; ++index) {
      if (!std::strcmp(argv[index], "--adaptive-ui-only")) {
        require(!adaptive_only, "duplicate adaptive-only argument"); adaptive_only = true;
      } else if (!std::strcmp(argv[index], "--pre-ui-change-set-only")) {
        // Fix 3's section alone, with its GPU cost (any size, such as 3840x2160).
        require(!change_set_only, "duplicate change-set-only argument"); change_set_only = true;
      } else if (!std::strcmp(argv[index], "--pin-only-ui-only")) {
        // Fix 4's section alone, with its GPU cost on a layer (any size).
        require(!pin_only_ui_only, "duplicate pin-only-UI argument"); pin_only_ui_only = true;
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
    if (change_set_only) {
      verify_pre_ui_change_set(gpu, report, directory, gpu.width * gpu.height > 1u << 20);
      require(report.good(), "cannot write change-set evidence");
      return 0;
    }
    if (pin_only_ui_only) {
      verify_pin_only_ui_darkening(gpu, report, gpu.width * gpu.height > 1u << 20);
      require(report.good(), "cannot write pin-only-UI evidence");
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
    verify_layer_detection_dump(gpu, report, directory / "layer-detection-dump");
    verify_layer_validity(gpu, report);
    verify_hidden_scene(gpu, report);
    verify_pre_ui_change_set(gpu, report, directory);
    verify_pin_only_ui_darkening(gpu, report);
    verify_normalized_ui_input(gpu, report);
    verify_mask_upload_recovery(gpu);
    require(report.good(), "cannot write evidence");
    std::printf("PASS actual D3D11 source-alpha renderer %ux%u color=%u; no FX or visible window\n", gpu.width, gpu.height, color);
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
