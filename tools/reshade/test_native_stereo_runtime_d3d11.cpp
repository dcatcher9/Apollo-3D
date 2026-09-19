// SPDX-License-Identifier: GPL-3.0-only
// Actual official ReShade runtime fixture for the independent Sunshine shader.
// DEPTH and calibration injection exist only in this isolated test executable.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <fstream>
#include <reshade.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>

namespace {
  namespace api = reshade::api;
  namespace fs = std::filesystem;
  unsigned width = 640, height = 360, sdr_bits = 8;

  template<class T>
  struct com_ptr {
    T *p = nullptr;

    ~com_ptr() {
      reset();
    }

    void reset() {
      if (p) {
        p->Release();
        p = nullptr;
      }
    }

    T **put() {
      reset();
      return &p;
    }

    T *operator->() const {
      return p;
    }
  };

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  void checked(HRESULT result, const char *message) {
    if (FAILED(result)) {
      char text[384];
      std::snprintf(text, sizeof(text), "%s (0x%08lx)", message, static_cast<unsigned long>(result));
      throw std::runtime_error(text);
    }
  }

  bool named(const char *name, const char *expected) {
    const std::string value(name), tail(expected);
    return value == tail || (value.size() > tail.size() && value.compare(value.size() - tail.size(), tail.size(), tail) == 0 && value[value.size() - tail.size() - 1] == ':');
  }

  float half_float(unsigned value) {
    const float sign = (value & 0x8000) ? -1.f : 1.f;
    const unsigned exponent = (value >> 10) & 31, fraction = value & 1023;
    if (exponent == 0) {
      return sign * std::ldexp(float(fraction), -24);
    }
    if (exponent == 31) {
      return fraction ? NAN : sign * INFINITY;
    }
    return sign * std::ldexp(1.f + float(fraction) / 1024.f, int(exponent) - 15);
  }

  struct observations_t {
    api::effect_runtime *runtime = nullptr;
    api::effect_technique technique {};
    api::color_space color = api::color_space::unknown;
    ID3D11DeviceContext *context = nullptr;
    ID3D11Texture2D *mono = nullptr;
    api::resource_view depth_view {};
    std::vector<api::effect_uniform_variable> depth_ready_uniforms;
    bool inject_depth = false, depth_ready = false, calibrated = false;
    api::effect_uniform_variable calibrated_uniform {};
    bool capture = false;
    unsigned renders = 0, captures = 0, reloads = 0;
  } observed;

  void on_init(api::effect_runtime *runtime) {
    observed.runtime = runtime;
  }

  void on_reload(api::effect_runtime *) {
    ++observed.reloads;
  }

  void on_present(api::command_queue *, api::swapchain *swapchain, const api::rect *, const api::rect *, std::uint32_t, const api::rect *) {
    observed.color = swapchain->get_color_space();
  }

  void on_begin_effects(api::effect_runtime *runtime, api::command_list *, api::resource_view, api::resource_view) {
    if (!observed.inject_depth) {
      return;
    }
    // Inject the fixture's hardware depth and readiness only after ReShade's
    // automatic uniforms, immediately before the actual shader executes.
    const api::resource_view view = observed.depth_ready ? observed.depth_view : api::resource_view {};
    runtime->update_texture_bindings("DEPTH", view, view);
    runtime->set_uniform_value_bool(observed.calibrated_uniform, observed.calibrated);
    for (const auto uniform : observed.depth_ready_uniforms) {
      runtime->set_uniform_value_bool(uniform, observed.depth_ready);
    }
  }

  void on_technique(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *, api::resource_view rtv, api::resource_view) {
    char name[256] {};
    runtime->get_technique_name(technique, name);
    if (!named(name, "SunshineDepth3D")) {
      return;
    }
    observed.technique = technique;
    ++observed.renders;
    if (observed.capture && rtv.handle && observed.context && observed.mono) {
      ID3D11Resource *resource = nullptr;
      reinterpret_cast<ID3D11View *>(rtv.handle)->GetResource(&resource);
      if (resource) {
        // Copy immediately after the actual technique, before ReShade's overlay.
        observed.context->CopyResource(observed.mono, resource);
        resource->Release();
        ++observed.captures;
      }
    }
  }

  struct fixture_t {
    HMODULE runtime_module = nullptr;
    HWND window = nullptr;
    com_ptr<ID3D11Device> device;
    com_ptr<ID3D11DeviceContext> context;
    com_ptr<IDXGISwapChain> swapchain;
    com_ptr<ID3D11Texture2D> backbuffer, mono, before;
    com_ptr<ID3D11RenderTargetView> target;
    com_ptr<ID3D11Texture2D> exported;
    com_ptr<ID3D11Texture2D> depth, source_pattern;
    com_ptr<ID3D11ShaderResourceView> depth_view;
    unsigned color = 0;
    std::array<float, 4> input {0.25f, 0.5f, 0.75f, 1.f};

    ~fixture_t() {
      observed.capture = false;
      observed.inject_depth = false;
      observed.depth_ready_uniforms.clear();
      observed.context = nullptr;
      observed.mono = nullptr;
      if (runtime_module) {
        reshade::unregister_addon(GetModuleHandleW(nullptr), runtime_module);
      }
      exported.reset();
      depth_view.reset();
      depth.reset();
      source_pattern.reset();
      target.reset();
      before.reset();
      mono.reset();
      backbuffer.reset();
      swapchain.reset();
      context.reset();
      device.reset();
      if (window) {
        DestroyWindow(window);
      }
    }

    void initialize(const fs::path &runtime, const fs::path &shader_root, const fs::path &directory, unsigned source_color) {
      color = source_color;
      require(fs::exists(shader_root / "SunshineDepth3D.fx"), "Missing independent SunshineDepth3D.fx");
      fs::create_directories(directory / "effects");
      fs::create_directories(directory / "addons");
      fs::copy_file(shader_root / "SunshineDepth3D.fx", directory / "effects" / "SunshineDepth3D.fx", fs::copy_options::overwrite_existing);
      fs::copy_file(runtime, directory / "dxgi.dll", fs::copy_options::overwrite_existing);
      {
        std::ofstream config(directory / "ReShade.ini");
        config << "[ADDON]\nAddonPath=.\\addons\n[GENERAL]\nEffectSearchPaths=.\\effects\nPresetPath=.\\preset.ini\n"
                  "PerformanceMode=0\nSkipLoadingDisabledEffects=0\nEffectCachePath=.\\cache\n"
                  "[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n";
        std::ofstream preset(directory / "preset.ini");
        preset << "Techniques=SunshineDepth3D@SunshineDepth3D.fx\nTechniqueSorting=SunshineDepth3D@SunshineDepth3D.fx\n";
      }
      SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", directory.c_str());
      // Initialize native D3D11 before loading the DXGI proxy, matching a normal
      // process whose graphics imports exist when ReShade attaches.
      wchar_t system[MAX_PATH] {};
      GetSystemDirectoryW(system, MAX_PATH);
      require(LoadLibraryW((fs::path(system) / "d3d11.dll").c_str()) != nullptr, "Could not preload native D3D11");
      runtime_module = LoadLibraryW((directory / "dxgi.dll").c_str());
      require(runtime_module != nullptr, "Could not load official ReShade runtime");
      require(reshade::register_addon(GetModuleHandleW(nullptr), runtime_module), "Could not register runtime observer");
      reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init);
      reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(on_reload);
      reshade::register_event<reshade::addon_event::present>(on_present);
      reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
      reshade::register_event<reshade::addon_event::reshade_render_technique>(on_technique);
      WNDCLASSW window_class {};
      window_class.lpfnWndProc = DefWindowProcW;
      window_class.hInstance = GetModuleHandleW(nullptr);
      window_class.lpszClassName = L"SunshineNativeStereoD3D11Fixture";
      RegisterClassW(&window_class);
      RECT bounds {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
      AdjustWindowRect(&bounds, WS_OVERLAPPEDWINDOW, FALSE);
      window = CreateWindowW(window_class.lpszClassName, L"Hidden Sunshine native stereo test", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, bounds.right - bounds.left, bounds.bottom - bounds.top, nullptr, nullptr, window_class.hInstance, nullptr);
      require(window != nullptr, "Could not create hidden test window");
      DXGI_SWAP_CHAIN_DESC desc {};
      desc.BufferDesc.Width = width;
      desc.BufferDesc.Height = height;
      desc.BufferDesc.Format = color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : (color == 3 || sdr_bits == 10) ? DXGI_FORMAT_R10G10B10A2_UNORM :
                                                                                                              DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count = 1;
      desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      desc.BufferCount = 2;
      desc.OutputWindow = window;
      desc.Windowed = TRUE;
      desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      using create_t = decltype(&D3D11CreateDeviceAndSwapChain);
      const auto create = reinterpret_cast<create_t>(GetProcAddress(runtime_module, "D3D11CreateDeviceAndSwapChain"));
      require(create != nullptr, "Official runtime lacks D3D11 proxy export");
      checked(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &desc, swapchain.put(), device.put(), nullptr, context.put()), "Create actual ReShade game swapchain");
      com_ptr<IDXGISwapChain3> chain3;
      checked(swapchain->QueryInterface(IID_PPV_ARGS(chain3.put())), "Query native color-space interface");
      checked(chain3->SetColorSpace1(color == 2 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : color == 3 ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 :
                                                                                                         DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),
              "Set native game color space");
      checked(swapchain->GetBuffer(0, IID_PPV_ARGS(backbuffer.put())), "Get actual mono backbuffer");
      checked(device->CreateRenderTargetView(backbuffer.p, nullptr, target.put()), "Create mono target");
      D3D11_TEXTURE2D_DESC texture_desc {};
      backbuffer->GetDesc(&texture_desc);
      texture_desc.Usage = D3D11_USAGE_STAGING;
      texture_desc.BindFlags = texture_desc.MiscFlags = 0;
      texture_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      checked(device->CreateTexture2D(&texture_desc, nullptr, mono.put()), "Create mono readback");
      checked(device->CreateTexture2D(&texture_desc, nullptr, before.put()), "Create source readback");
      observed.context = context.p;
      observed.mono = mono.p;
      std::printf("Official runtime, hidden DX11 window, independent SunshineDepth3D.fx: source=%u SDR-source-bits=%u\n", color, sdr_bits);
    }

    void step() {
      require(IsWindow(window) && !IsWindowVisible(window), "Fixture window became visible or disappeared");
      MSG message;
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      context->OMSetRenderTargets(1, &target.p, nullptr);
      context->ClearRenderTargetView(target.p, input.data());
      if (source_pattern.p) {
        context->CopyResource(backbuffer.p, source_pattern.p);
      }
      if (observed.capture) {
        context->CopyResource(before.p, backbuffer.p);
      }
      checked(swapchain->Present(0, 0), "Natural game Present");
      require(unsigned(observed.color) == color, "Actual runtime source transfer differs from native swapchain");
      Sleep(3);
    }

    void discover() {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
      while (!observed.renders && std::chrono::steady_clock::now() < deadline) {
        step();
        if (observed.runtime && observed.reloads) {
          observed.runtime->enumerate_techniques("SunshineDepth3D.fx", [](api::effect_runtime *runtime, api::effect_technique technique) {
            char name[256] {};
            runtime->get_technique_name(technique, name);
            if (named(name, "SunshineDepth3D")) {
              runtime->set_technique_state(technique, true);
            }
          });
        }
      }
      require(observed.runtime && observed.renders, "Actual full SunshineDepth3D effect did not compile and execute; inspect isolated ReShade.log");
      api::effect_texture_variable variable {};
      observed.runtime->enumerate_texture_variables("SunshineDepth3D.fx", [&](api::effect_runtime *runtime, api::effect_texture_variable texture) {
        int marker = 0;
        if (runtime->get_annotation_int_from_texture_variable(texture, "sunshine_sbs_export", &marker, 1) && marker == 1) {
          require(!variable.handle, "Multiple export textures violate unique effect contract");
          variable = texture;
        }
      });
      require(variable.handle != 0, "Actual Depth3D did not expose explicit export marker");
      char name[256] {}, transfer[32] {}, layout[32] {};
      observed.runtime->get_texture_variable_name(variable, name);
      require(named(name, "DoubleTex"), "Actual export is not named DoubleTex");
      unsigned source = 0, w = 0, h = 0;
      require(observed.runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_color_space", &source, 1) && source == color, "Actual compile-time source color annotation is stale");
      require(observed.runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_width", &w, 1) && w == width && observed.runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_height", &h, 1) && h == height, "Actual source dimensions annotation incorrect");
      require(observed.runtime->get_annotation_string_from_texture_variable(variable, "sunshine_sbs_layout", layout) && std::strcmp(layout, "sbs_lr") == 0, "Actual layout annotation incorrect");
      require(observed.runtime->get_annotation_string_from_texture_variable(variable, "sunshine_sbs_color_space", transfer) && std::strcmp(transfer, color == 1 ? "srgb" : "scrgb") == 0, "Actual export transfer annotation incorrect");
      api::resource_view view {};
      observed.runtime->get_texture_binding(variable, &view, nullptr);
      require(view.handle != 0, "Actual DoubleTex has no resource view");
      com_ptr<ID3D11Resource> resource;
      reinterpret_cast<ID3D11View *>(view.handle)->GetResource(resource.put());
      checked(resource->QueryInterface(IID_PPV_ARGS(exported.put())), "Actual DoubleTex is not a native texture");
      const auto adapted = observed.runtime->get_device()->get_resource_from_view(view);
      require(adapted.handle == reinterpret_cast<std::uint64_t>(resource.p), "Adapted SDK resource getter differs from native COM");
      D3D11_TEXTURE2D_DESC desc {};
      exported->GetDesc(&desc);
      require(desc.Width == 2 * width && desc.Height == height && desc.Format == (color == 1 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT), "Actual DoubleTex dimensions/format incorrect");
      std::printf("PASS full-effect runtime compile/execution + DoubleTex annotations + native %ux%u format=%u\n", desc.Width, desc.Height, unsigned(desc.Format));
    }

    std::vector<std::uint8_t> read(ID3D11Texture2D *texture) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      com_ptr<ID3D11Texture2D> staging;
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      checked(device->CreateTexture2D(&desc, nullptr, staging.put()), "Create bounded readback");
      context->CopyResource(staging.p, texture);
      com_ptr<ID3D11Query> query;
      D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0};
      checked(device->CreateQuery(&query_desc, query.put()), "Create readback completion");
      context->End(query.p);
      context->Flush();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      HRESULT done;
      while ((done = context->GetData(query.p, nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE && std::chrono::steady_clock::now() < deadline) {
        Sleep(1);
      }
      require(done == S_OK, "GPU readback exceeded bound");
      D3D11_MAPPED_SUBRESOURCE mapped {};
      checked(context->Map(staging.p, 0, D3D11_MAP_READ, 0, &mapped), "Map completed readback");
      const unsigned pitch = desc.Width * (desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 16 : desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 :
                                                                                                                                                4);
      std::vector<std::uint8_t> result(size_t(pitch) * desc.Height);
      for (unsigned y = 0; y < desc.Height; ++y) {
        std::memcpy(result.data() + size_t(pitch) * y, static_cast<const std::uint8_t *>(mapped.pData) + size_t(mapped.RowPitch) * y, pitch);
      }
      context->Unmap(staging.p, 0);
      return result;
    }

    api::effect_uniform_variable uniform(const char *name, bool required = true) {
      api::effect_uniform_variable result {};
      observed.runtime->enumerate_uniform_variables("SunshineDepth3D.fx", [&](api::effect_runtime *runtime, api::effect_uniform_variable candidate) {
        char candidate_name[256] {};
        runtime->get_uniform_variable_name(candidate, candidate_name);
        if (named(candidate_name, name)) {
          result = candidate;
        }
      });
      if (required && !result.handle) {
        throw std::runtime_error(std::string("Missing actual depth control: ") + name);
      }
      return result;
    }

    void set_float(const char *name, float value) {
      observed.runtime->set_uniform_value_float(uniform(name), &value, 1);
    }

    void set_bool(const char *name, bool value) {
      observed.runtime->set_uniform_value_bool(uniform(name), value);
    }

    static float decode_srgb(float x) {
      return x <= .04045f ? x / 12.92f : std::pow((x + .055f) / 1.055f, 2.4f);
    }

    static float encode_srgb(float x) {
      return x <= .0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.f / 2.4f) - .055f;
    }

    std::array<float, 3> source_color_at(const std::vector<std::uint8_t> &native, unsigned x, unsigned y) const {
      const size_t pixel = size_t(y) * width + x;
      std::array<float, 3> result {};
      for (unsigned c = 0; c < 3; ++c) {
        if (color == 2) {
          result[c] = half_float(reinterpret_cast<const std::uint16_t *>(native.data())[pixel * 4 + c]);
        } else if (color == 3 || sdr_bits == 10) {
          result[c] = float((reinterpret_cast<const std::uint32_t *>(native.data())[pixel] >> (10 * c)) & 1023) / 1023.f;
        } else {
          result[c] = native[pixel * 4 + c] / 255.f;
        }
      }
      if (color == 3) {
        std::array<double, 3> light {};
        for (unsigned c = 0; c < 3; ++c) {
          const double p = std::pow(double(result[c]), 32.0 / 2523.0);
          light[c] = 125.0 * std::pow(std::max(p - 3424.0 / 4096.0, 0.0) / (2413.0 / 128.0 - (2392.0 / 128.0) * p), 16384.0 / 2610.0);
        }
        result = {float(1.6604910021 * light[0] - .5876411388 * light[1] - .0728498633 * light[2]), float(-.1245504745 * light[0] + 1.1328998971 * light[1] - .0083494226 * light[2]), float(-.0181507634 * light[0] - .1005788980 * light[1] + 1.1187296614 * light[2])};
      }
      return result;
    }

    void prepare_depth() {
      observed.depth_ready_uniforms.push_back(uniform("Sunshine_DepthReady"));
      observed.calibrated_uniform = uniform("Sunshine_Calibrated");
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = desc.ArraySize = 1;
      desc.SampleDesc.Count = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      checked(device->CreateTexture2D(&desc, nullptr, depth.put()), "Create FP32 hardware depth fixture");
      checked(device->CreateShaderResourceView(depth.p, nullptr, depth_view.put()), "Create DEPTH semantic view");
      observed.depth_view = api::resource_view {reinterpret_cast<std::uint64_t>(depth_view.p)};
      observed.inject_depth = observed.depth_ready = observed.calibrated = true;
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      const float rect[] {1, 1, 0, 0};
      observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), rect, 4);
      set_float("ScreenPlane", 0.f);
      set_bool("EdgeAntialias", false);
      set_bool("DepthView", false);
    }

    void upload_depth(const std::vector<float> &values) {
      require(values.size() == size_t(width) * height, "Depth fixture dimensions differ");
      context->UpdateSubresource(depth.p, 0, nullptr, values.data(), width * sizeof(float), 0);
    }

    // Native source patterns: asymmetric stripes, a diagonal scene boundary,
    // or one foreground column. No CPU stereo renderer supplies the oracle.
    void pattern(unsigned kind) {
      D3D11_TEXTURE2D_DESC desc {};
      backbuffer->GetDesc(&desc);
      desc.BindFlags = desc.MiscFlags = 0;
      const unsigned bytes = color == 2 ? 8 : 4;
      std::vector<std::uint8_t> pixels(size_t(width) * height * bytes);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const size_t pixel = size_t(y) * width + x;
          const bool stripe = ((x / 16) % 2) != (((x / 16) / 7 + (x / 16) / 13) % 2);
          const bool bright = kind == 1 ? x < width / 2 + y / 4 : kind == 2 ? x == width / 2 :
                                                                              stripe;
          if (color == 1 && sdr_bits == 8) {
            const std::uint8_t values[4] {std::uint8_t(bright ? 224 : 32), std::uint8_t(y < height / 3 ? 200 : 60), std::uint8_t(!kind && ((x + 2 * y) / 31) % 2 ? 170 : 80), 255};
            std::memcpy(pixels.data() + pixel * 4, values, 4);
          } else if (color == 2) {
            const std::uint16_t values[4] {std::uint16_t(bright ? 0x4000 : 0x3400), std::uint16_t(y < height / 3 ? 0x3c00 : 0x3800), std::uint16_t(!kind && ((x + 2 * y) / 31) % 2 ? 0x3a00 : 0xb000), 0x3c00};
            std::memcpy(pixels.data() + pixel * 8, values, 8);
          } else {
            const std::uint32_t value = (bright ? 700u : 256u) | ((y < height / 3 ? 600u : 300u) << 10) | ((!kind && ((x + 2 * y) / 31) % 2 ? 500u : 200u) << 20) | (3u << 30);
            std::memcpy(pixels.data() + pixel * 4, &value, 4);
          }
        }
      }
      const D3D11_SUBRESOURCE_DATA data {pixels.data(), width * bytes, 0};
      checked(device->CreateTexture2D(&desc, &data, source_pattern.put()), "Create native source pattern");
    }

    void measured_frames() {
      observed.capture = true;
      const unsigned previous = observed.captures;
      for (unsigned i = 0; i < 3; ++i) {
        step();
      }
      observed.capture = false;
      require(observed.captures >= previous + 3, "Actual nonzero-depth technique failed to execute");
      require(read(before.p) == read(mono.p), "Nonzero depth or diagnostic changed the native mono game image");
    }

    float exported_channel(const std::vector<std::uint8_t> &pixels, unsigned x, unsigned y, unsigned channel) const {
      const size_t pixel = size_t(y) * width * 2 + x;
      return color == 1 ? float((reinterpret_cast<const std::uint32_t *>(pixels.data())[pixel] >> (10 * channel)) & 1023) / 1023.f : half_float(reinterpret_cast<const std::uint16_t *>(pixels.data())[4 * pixel + channel]);
    }

    float image_difference(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b) const {
      float maximum = 0;
      for (unsigned y = height / 8; y < height * 7 / 8; y += std::max(1u, height / 360)) {
        for (unsigned x = width / 8; x < width * 7 / 8; x += std::max(1u, width / 640)) {
          for (unsigned eye = 0; eye < 2; ++eye) {
            for (unsigned c = 0; c < 3; ++c) {
              const float av = exported_channel(a, eye * width + x, y, c), bv = exported_channel(b, eye * width + x, y, c);
              require(std::isfinite(av) && std::isfinite(bv), "Export contains a non-finite color");
              maximum = std::max(maximum, std::abs(av - bv));
            }
          }
        }
      }
      return maximum;
    }

    std::array<float, 2> measure_eye_shifts(const std::vector<std::uint8_t> &flat, const std::vector<std::uint8_t> &stereo, const char *setting) const {
      // Use asymmetric stripe spacing, several rows, and the full central image.
      // The reference is the real zero-divergence export, not a second warp.
      // A source-coordinate shift is positive when output[x] reads input[x+shift].
      const int limit = std::max(16, int(width / 16));
      const unsigned margin = std::max(width / 8, unsigned(limit + 3));
      std::array<float, 2> result {};
      for (unsigned eye = 0; eye < 2; ++eye) {
        auto error_at = [&](float shift) {
          double squared = 0, signal = 0;
          size_t samples = 0;
          for (unsigned y : {height / 4, height / 2, height * 3 / 4}) {
            for (unsigned x = margin; x + margin < width; ++x) {
              const float coordinate = float(x) + shift;
              const unsigned sx = unsigned(std::floor(coordinate));
              const float blend = coordinate - sx;
              const float a = exported_channel(flat, sx, y, 0), b = exported_channel(flat, sx + 1, y, 0);
              const float expected = color == 1 ? encode_srgb(decode_srgb(a) + (decode_srgb(b) - decode_srgb(a)) * blend) : a + (b - a) * blend;
              const float actual = exported_channel(stereo, eye * width + x, y, 0);
              require(std::isfinite(actual), "Shift measurement encountered invalid exported color");
              squared += double(actual - expected) * (actual - expected);
              signal += double(a) * a + double(b) * b;
              ++samples;
            }
          }
          require(samples && signal > 0, "Shift measurement has no native source signal");
          return squared / signal;
        };
        double best_error = INFINITY;
        float best_shift = 0;
        for (int shift = -limit; shift <= limit; ++shift) {
          const double error = error_at(float(shift));
          if (error < best_error) {
            best_error = error;
            best_shift = float(shift);
          }
        }
        const float coarse = best_shift;
        for (int fraction = -4; fraction <= 4; ++fraction) {
          const float shift = coarse + float(fraction) * .25f;
          const double error = error_at(shift);
          if (error < best_error) {
            best_error = error;
            best_shift = shift;
          }
        }
        std::printf("MEASURE native shift setting=%s eye=%u source_pixels=%.3f normalized_error=%.9g\n", setting, eye, best_shift, best_error);
        require(std::abs(best_shift) < limit - 1, "Native displacement fit reached its search boundary");
        require(best_error < .015, "Actual export does not correlate with a translated constant-plane source");
        result[eye] = best_shift;
      }
      return result;
    }

    void check_colors(const std::array<float, 4> &value, std::array<float, 3> expected) {
      input = value;
      for (float strength : {0.f, 1.f}) {
        set_float("Strength", strength);
        measured_frames();
        const auto native = read(before.p), stereo = read(exported.p);
        if (color == 1) {
          const size_t pixel = size_t(height / 2) * width + width / 2;
          for (unsigned c = 0; c < 3; ++c) {
            expected[c] = sdr_bits == 10 ? float((reinterpret_cast<const std::uint32_t *>(native.data())[pixel] >> (10 * c)) & 1023) / 1023.f : native[pixel * 4 + c] / 255.f;
          }
        }
        float worst = 0;
        for (unsigned eye = 0; eye < 2; ++eye) {
          for (unsigned y = height / 4; y < 3 * height / 4; ++y) {
            for (unsigned x = width / 4; x < 3 * width / 4; ++x) {
              for (unsigned c = 0; c < 3; ++c) {
                const float actual = exported_channel(stereo, eye * width + x, y, c);
                const float error = std::abs(actual - expected[c]);
                worst = std::max(worst, error);
                require(std::isfinite(actual) && error <= .003f + .002f * std::abs(expected[c]), "Constant color/HDR export differs from the independent color oracle");
              }
            }
          }
        }
        std::printf("PASS native color=%u bits=%u strength=%.1f input=(%.6g,%.6g,%.6g) max-error=%.9g; mono byte-exact\n", color, sdr_bits, strength, value[0], value[1], value[2], worst);
      }
    }

    void check_edges(std::vector<float> &hardware) {
      // Arrange an actual projected boundary between subpixel rays at every
      // supported source width, rather than accepting an AA response of zero.
      const float maximum_shift = .006f * width;  // Strength 1.2, full per-eye resolution.
      const float projected_shift = std::floor(.6f * maximum_shift) + .3f;
      const float foreground = .5f + .5f * projected_shift / maximum_shift;
      set_float("Strength", 1.2f);
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      pattern(1);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          hardware[size_t(y) * width + x] = x < width / 2 + y / 4 ? foreground : .2f;
        }
      }
      upload_depth(hardware);
      set_bool("EdgeAntialias", false);
      measured_frames();
      const auto noaa = read(exported.p);
      set_bool("EdgeAntialias", true);
      measured_frames();
      const auto aa = read(exported.p);
      const float response = image_difference(noaa, aa);
      std::printf("MEASURE stereo-edge AA response=%.9g, source-plane displacement=%.6g\n", response, projected_shift);
      require(response > .005f, "Stereo-edge AA did not change the deliberately fractional boundary");
      set_bool("EdgeAntialias", false);
      measured_frames();
      require(read(exported.p) == noaa, "Disabling stereo-edge AA did not restore the original result");

      // A one-source-column foreground object must survive in both eyes and
      // move to its predicted subpixel position, even with a far background.
      pattern(2);
      std::fill(hardware.begin(), hardware.end(), .2f);
      for (unsigned y = 0; y < height; ++y) {
        hardware[size_t(y) * width + width / 2] = foreground;
      }
      upload_depth(hardware);
      set_float("Strength", 0.f);
      measured_frames();
      const auto thin_flat = read(exported.p);
      set_float("Strength", 1.2f);
      set_bool("EdgeAntialias", true);
      measured_frames();
      const auto thin = read(exported.p);
      const unsigned row = height / 2;
      for (unsigned eye = 0; eye < 2; ++eye) {
        const float expected_x = width / 2.f + (eye ? -projected_shift : projected_shift);
        const int first = int(std::floor(expected_x)) - 3, last = first + 7;
        const float background = exported_channel(thin_flat, width / 2 - 20, row, 0);
        const float foreground_color = exported_channel(thin_flat, width / 2, row, 0);
        const float contrast = foreground_color - background;
        require(contrast > .05f, "Thin-object fixture has no measurable source contrast");
        float mass = 0, weighted_x = 0, peak = 0;
        for (int x = first; x <= last; ++x) {
          const float sample = exported_channel(thin, eye * width + unsigned(x), row, 0);
          require(std::isfinite(sample), "Thin-object export contains nonfinite color");
          const float weight = std::max(0.f, (sample - background) / contrast);
          mass += weight;
          weighted_x += weight * x;
          peak = std::max(peak, weight);
        }
        require(mass > .2f && peak > .15f, "A one-column foreground object disappeared during stereo reprojection");
        const float centroid = weighted_x / mass;
        std::printf("MEASURE thin-column eye=%u source=%u expected=%.6g centroid=%.6g contrast-mass=%.6g peak=%.6g\n", eye, width / 2, expected_x, centroid, mass, peak);
        require(std::abs(centroid - expected_x) < .85f, "The thin object's stereo position differs from its projected depth");
        require(eye ? centroid < width / 2.f - .75f : centroid > width / 2.f + .75f, "The thin object remained at its mono location");
      }
      std::puts("PASS nonzero reversible stereo-edge AA and full-resolution one-column foreground coverage");
    }

    void run() {
      discover();
      prepare_depth();
      std::vector<float> hardware(size_t(width) * height, .75f);
      upload_depth(hardware);
      if (color == 1) {
        check_colors({.25f, .5f, .75f, 1}, {});
        check_colors({1, 0, 1, 1}, {});
      } else if (color == 2) {
        check_colors({4, .5f, -.125f, 1}, {4, .5f, -.125f});
        check_colors({125, 12.5f, 1, 1}, {125, 12.5f, 1});
      } else {
        // Native PQ endpoints have exact code values; these analytical expected
        // values independently test ST.2084 and Rec.2020-to-709 conversion.
        check_colors({0, 0, 0, 1}, {0, 0, 0});
        check_colors({1, 1, 1, 1}, {125, 125, 125});
        check_colors({1, 0, 0, 1}, {207.5613753f, -15.5688093f, -2.2688454f});
      }

      pattern(0);
      set_float("Strength", 0.f);
      measured_frames();
      const auto flat = read(exported.p), native = read(before.p);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const auto expected = source_color_at(native, x, y);
          for (unsigned c = 0; c < 3; ++c) {
            require(exported_channel(flat, x, y, c) == exported_channel(flat, x + width, y, c), "Zero strength does not duplicate the full image in both eyes");
            require(std::abs(exported_channel(flat, x, y, c) - expected[c]) <= .003f + .002f * std::abs(expected[c]), "Zero-strength export does not match the native source pixel");
          }
        }
      }

      float previous = 0;
      for (float strength : {.5f, 1.f, 2.f}) {
        set_float("Strength", strength);
        measured_frames();
        const auto shifts = measure_eye_shifts(flat, read(exported.p), std::to_string(strength).c_str());
        const float expected = .0025f * width * strength;
        require(shifts[0] < 0 && shifts[1] > 0, "Foreground stereo directions are reversed");
        require(std::abs(shifts[0] + expected) < .4f && std::abs(shifts[1] - expected) < .4f, "Measured strength differs from the analytical constant-plane shift");
        require(std::abs(shifts[0]) > previous + .25f, "Positive strength values produce a fixed stereo effect");
        previous = std::abs(shifts[0]);
      }
      set_float("Strength", 1.f);
      set_float("ScreenPlane", .75f);
      measured_frames();
      const auto behind = measure_eye_shifts(flat, read(exported.p), "behind screen plane");
      require(behind[0] > 0 && behind[1] < 0, "Screen plane did not reverse the depth placement");
      set_float("ScreenPlane", 0.f);
      measured_frames();
      const auto reversed = read(exported.p);
      std::fill(hardware.begin(), hardware.end(), .25f);
      upload_depth(hardware);
      set_float("Sunshine_RawGain", -2.f);
      measured_frames();
      require(read(exported.p) == reversed, "Equivalent normal and reversed-Z planes have different geometry");

      observed.depth_ready = false;
      measured_frames();
      require(read(exported.p) == flat, "Missing depth does not return the current flat source");
      observed.depth_ready = true;
      observed.calibrated = false;
      measured_frames();
      require(read(exported.p) == flat, "Missing calibration does not return the current flat source");
      observed.calibrated = true;

      std::fill(hardware.begin(), hardware.end(), .999975f);
      upload_depth(hardware);
      set_float("Sunshine_RawAnchor", .99995f);
      set_float("Sunshine_RawGain", -20000.f);
      measured_frames();
      const auto precise = measure_eye_shifts(flat, read(exported.p), "near-one FP32 depth");
      const float precision_expected = (.999975f - .99995f) * 20000.f * .005f * width;
      require(std::abs(precise[0] - precision_expected) < .4f && std::abs(precise[1] + precision_expected) < .4f, "Near-one FP32 depth was lost before calibration");

      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          hardware[size_t(y) * width + x] = .2f + .6f * float(x) / float(width - 1);
        }
      }
      upload_depth(hardware);
      set_bool("DepthView", true);
      measured_frames();
      const auto diagnostic = read(exported.p);
      set_float("Strength", 0.f);
      set_float("ScreenPlane", .8f);
      measured_frames();
      require(read(exported.p) == diagnostic, "Depth diagnostic depends on stereo strength or screen plane");
      for (unsigned x = 0; x < width; ++x) {
        const float expected = .2f + .6f * float(x) / float(width - 1);
        for (unsigned eye = 0; eye < 2; ++eye) {
          for (unsigned c = 0; c < 3; ++c) {
            require(std::abs(exported_channel(diagnostic, eye * width + x, height / 2, c) - expected) < .002f, "Depth diagnostic differs from the independent raw-depth ramp oracle");
          }
        }
      }
      set_bool("DepthView", false);
      set_float("ScreenPlane", 0.f);
      check_edges(hardware);
      std::printf("PASS independent native stereo DX11 %ux%u color=%u source-bits=%u: color/HDR, slider geometry, screen plane, reverse-Z, FP32 precision, diagnostics, separate fallback states, mono preservation, AA and thin objects; %u actual technique executions\n", width, height, color, sdr_bits, observed.renders);
    }
  };
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 6 && argc != 8 && argc != 9) {
    std::fprintf(stderr, "usage: test_native_stereo_runtime_d3d11 <official-ReShade64.dll> <Sunshine-Shaders> <isolated-ignored-output> <srgb|scrgb|pq> <compatibility:0|1> [width height [SDR-bits:8|10]]\n");
    return 2;
  }
  std::thread([] {
    Sleep(180000);
    std::fputs("FAIL actual D3D11 native stereo fixture watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    const unsigned color = std::strcmp(argv[4], "srgb") == 0 ? 1 : std::strcmp(argv[4], "scrgb") == 0 ? 2 :
                                                                 std::strcmp(argv[4], "pq") == 0      ? 3 :
                                                                                                        0;
    require(color && (std::strcmp(argv[5], "0") == 0 || std::strcmp(argv[5], "1") == 0), "Invalid color or compatibility argument");
    if (argc >= 8) {
      size_t used_width = 0, used_height = 0;
      const auto requested_width = std::stoul(argv[6], &used_width), requested_height = std::stoul(argv[7], &used_height);
      require(used_width == std::strlen(argv[6]) && used_height == std::strlen(argv[7]) && requested_width >= 640 && requested_height >= 360 && requested_width <= 3840 && requested_height <= 2160 && requested_width % 2 == 0 && requested_height % 2 == 0, "Fixture dimensions must be even, from 640x360 up to 3840x2160");
      width = unsigned(requested_width);
      height = unsigned(requested_height);
    }
    if (argc == 9) {
      require(color == 1 && (std::strcmp(argv[8], "8") == 0 || std::strcmp(argv[8], "10") == 0), "The explicit SDR source bit depth must be 8 or 10 for an SDR fixture");
      sdr_bits = unsigned(std::stoul(argv[8]));
    }
    const auto output = fs::absolute(argv[3]);
    require(!fs::exists(output), "Use a fresh isolated output directory");
    fixture_t fixture;
    // Keep the shared harness's compatibility argument for CLI consistency;
    // this independent effect has no upstream HDR compatibility permutation.
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), output, color);
    fixture.run();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
