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
        api::resource_view alpha_source = {}, const fs::path &dump_directory = {}) {
      auto &active = control ? control_renderer : renderer;
      sunshine_game3d::render_parameters p;
      p.strength = flat ? 0 : 100; p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 1;
      p.depth_scale = 100000; p.strength_blend = 1; p.projection = {0, 1};
      p.convergence = {.05f, sign > 0 ? .03f : .01f}; p.disparity_limit_uv = .04f;
      context->CopyResource(backbuffer.Get(), source.Get());
      auto *queue = observed_runtime->get_command_queue();
      require(active.render(queue->get_immediate_command_list(), {reinterpret_cast<std::uint64_t>(backbuffer.Get())},
        {reinterpret_cast<std::uint64_t>(depth_view.Get())}, p,
        alpha_source.handle ? protect : sunshine_game3d::source_alpha_ui_for_present(protect, frame_generation_active),
        alpha_source), "production render failed");
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
    // All white can be genuine full-screen UI; do not infer invalid alpha from
    // a histogram. The next real all-black alpha must also replace the old mask.
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
  if (argc != 6 && argc != 7) { std::fputs("usage: source_alpha_runtime official-ReShade64.dll fresh-output srgb|scrgb width height [frozen-control.hlsl]\n", stderr); return 2; }
  std::thread([] { Sleep(120000); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    const unsigned color = !std::strcmp(argv[3], "srgb") ? 1 : !std::strcmp(argv[3], "scrgb") ? 2 : 0;
    require(color != 0, "unsupported source transfer");
    const auto directory = fs::absolute(argv[2]);
    fixture gpu(fs::absolute(argv[1]), directory, color, unsigned(std::stoul(argv[4])), unsigned(std::stoul(argv[5])), argc == 7 ? fs::absolute(argv[6]) : fs::path{});
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
    verify_retained_fg_alpha(gpu, report, directory / "retained-alpha-dump");
    verify_mask_upload_recovery(gpu);
    require(report.good(), "cannot write evidence");
    std::printf("PASS actual D3D11 source-alpha renderer %ux%u color=%u; no FX or visible window\n", gpu.width, gpu.height, color);
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
