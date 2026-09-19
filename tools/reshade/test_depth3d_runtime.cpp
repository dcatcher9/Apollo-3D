// SPDX-License-Identifier: GPL-3.0-only
// Opt-in fixture for the separately supplied, licensed Depth3D checkout. No shader
// source is distributed here. Runs the real official ReShade DLL and full effect.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
#include "test_depth3d_effect.h"
#include "test_color_precision.h"

namespace {
  namespace api = reshade::api;
  namespace fs = std::filesystem;
  using sunshine_depth3d_fixture::effect_file;
  using sunshine_depth3d_fixture::technique_name;
  constexpr unsigned width = 640, height = 360;

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
    bool inject_depth = false, depth_ready = false;
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
    // The official generic-depth add-on supplies this exact semantic and every
    // bufready_depth uniform. Run after automatic uniforms, before real passes.
    const api::resource_view view = observed.depth_ready ? observed.depth_view : api::resource_view {};
    runtime->update_texture_bindings("DEPTH", view, view);
    for (const auto uniform : observed.depth_ready_uniforms) {
      runtime->set_uniform_value_bool(uniform, observed.depth_ready);
    }
  }

  void on_technique(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *, api::resource_view rtv, api::resource_view) {
    char name[256] {};
    runtime->get_technique_name(technique, name);
    if (!named(name, technique_name)) {
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
    com_ptr<ID3D11Texture2D> depth, linear_depth, source_pattern;
    com_ptr<ID3D11ShaderResourceView> depth_view;
    unsigned color = 0, compatible = 0;
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
      linear_depth.reset();
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

    void initialize(const fs::path &runtime, const fs::path &shader_root, const fs::path &directory, unsigned source_color, unsigned hdr_compatible) {
      sunshine_depth3d_fixture::select_effect();
      color = source_color;
      compatible = hdr_compatible;
      require(fs::exists(shader_root / effect_file), "Missing selected external full-pipeline shader");
      fs::create_directories(directory / "effects");
      fs::create_directories(directory / "addons");
      // Personal probe copies remain only in the operator's ignored output directory.
      // Do not package these separately licensed external shader files.
      for (const auto &entry : fs::recursive_directory_iterator(shader_root)) {
        if (!entry.is_regular_file() || (entry.path().extension() != ".fxh" && entry.path() != shader_root / effect_file)) {
          continue;
        }
        const auto destination = directory / "effects" / fs::relative(entry.path(), shader_root);
        fs::create_directories(destination.parent_path());
        fs::copy_file(entry.path(), destination, fs::copy_options::overwrite_existing);
      }
      fs::copy_file(runtime, directory / "dxgi.dll", fs::copy_options::overwrite_existing);
      {
        std::ofstream config(directory / "ReShade.ini");
        config << "[ADDON]\nAddonPath=.\\addons\n[GENERAL]\nEffectSearchPaths=.\\effects\nPresetPath=.\\preset.ini\n"
                  "PerformanceMode=0\nSkipLoadingDisabledEffects=0\nEffectCachePath=.\\cache\n"
                  "PreprocessorDefinitions=HDR_Compatible_Mode="
               << compatible << sunshine_depth3d_fixture::configuration_definitions()
               << "\n[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n";
        std::ofstream preset(directory / "preset.ini");
        preset << "Techniques=" << technique_name << "@" << effect_file << "\nTechniqueSorting=" << technique_name << "@" << effect_file << "\n";
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
      window_class.lpszClassName = L"Depth3DOfficialRuntimeFixture";
      RegisterClassW(&window_class);
      RECT bounds {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
      AdjustWindowRect(&bounds, WS_OVERLAPPEDWINDOW, FALSE);
      window = CreateWindowW(window_class.lpszClassName, L"Hidden Depth3D shader probe", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, bounds.right - bounds.left, bounds.bottom - bounds.top, nullptr, nullptr, window_class.hInstance, nullptr);
      require(window != nullptr, "Could not create hidden test window");
      DXGI_SWAP_CHAIN_DESC desc {};
      desc.BufferDesc.Width = width;
      desc.BufferDesc.Height = height;
      desc.BufferDesc.Format = color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : color == 3 ? DXGI_FORMAT_R10G10B10A2_UNORM :
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
      std::printf("Official runtime, hidden DX11 window, actual external %s: source=%u HDR_Compatible_Mode=%u\n", effect_file, color, compatible);
    }

    void step() {
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
          observed.runtime->enumerate_techniques(effect_file, [](api::effect_runtime *runtime, api::effect_technique technique) {
            char name[256] {};
            runtime->get_technique_name(technique, name);
            if (named(name, technique_name)) {
              runtime->set_technique_state(technique, true);
            }
          });
        }
      }
      require(observed.runtime && observed.renders, "Actual full SuperDepth3D effect did not compile and execute; inspect isolated ReShade.log");
      api::effect_texture_variable variable {};
      observed.runtime->enumerate_texture_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_texture_variable texture) {
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
      check_original_only();
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

    void check_colors(const std::array<float, 4> &source, std::array<float, 3> expected) {
      input = source;
      observed.capture = true;
      const auto previous = observed.captures;
      for (unsigned frame = 0; frame < 3; ++frame) {
        step();
      }
      observed.capture = false;
      require(observed.captures >= previous + 3, "Actual selected SuperDepth3D technique did not run on every measured frame");
      const auto original = read(before.p), final_mono = read(mono.p);
      require(original == final_mono, "Full mono backbuffer changed from its original native encoding");
      if (color == 1) {
        const auto *native = original.data() + (size_t(height / 2) * width + width / 2) * 4;
        // UNORM clear rounding at half steps is implementation dependent. Use
        // the observed native source codes, without an alternate color shader.
        for (unsigned channel = 0; channel < 3; ++channel) {
          expected[channel] = native[channel] / 255.f;
        }
      }
      const auto sbs = read(exported.p);
      float maximum_error = 0.f;
      // Original Depth3D retains its existing border/occlusion behavior. Compare
      // the interior of both eyes; this is a color/transport fixture, not geometry.
      for (unsigned eye = 0; eye < 2; ++eye) {
        for (unsigned y = height / 4; y < 3 * height / 4; ++y) {
          for (unsigned x = width / 4; x < 3 * width / 4; ++x) {
            const size_t pixel = size_t(y) * 2 * width + eye * width + x;
            for (unsigned channel = 0; channel < 3; ++channel) {
              const float actual = color == 1 ? float((reinterpret_cast<const std::uint32_t *>(sbs.data())[pixel] >> (10 * channel)) & 1023) / 1023.f : half_float(reinterpret_cast<const std::uint16_t *>(sbs.data())[4 * pixel + channel]);
              const float error = std::abs(actual - expected[channel]);
              maximum_error = std::max(maximum_error, error);
              if (!std::isfinite(actual) || error > (color == 1 ? .003f : .003f + .002f * std::abs(expected[channel]))) {
                std::printf("FAIL color=%u compatible=%u eye=%u xy=%u,%u channel=%u actual=%.9g expected=%.9g\n", color, compatible, eye, x, y, channel, actual, expected[channel]);
                throw std::runtime_error("Actual full effect export pixel differs from analytical color fixture");
              }
            }
          }
        }
      }
      std::printf("PASS source=%u compatible=%u input=(%.6g,%.6g,%.6g) expected=(%.6g,%.6g,%.6g) max_error=%.6g; full mono byte-exact\n", color, compatible, source[0], source[1], source[2], expected[0], expected[1], expected[2], maximum_error);
    }

    api::effect_uniform_variable uniform(const char *name, bool required = true) {
      api::effect_uniform_variable result {};
      observed.runtime->enumerate_uniform_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_uniform_variable candidate) {
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

    bool final_aa_available() {
      const char *legacy = std::getenv("SUNSHINE_GAME3D_LEGACY_FINAL_AA");
      require(!legacy || !*legacy || std::strcmp(legacy, "0") == 0 || std::strcmp(legacy, "1") == 0,
        "SUNSHINE_GAME3D_LEGACY_FINAL_AA must be 0 or 1");
      const bool legacy_enabled = legacy && std::strcmp(legacy, "1") == 0;
      const bool game3d = std::strcmp(effect_file, "SunshineGame3D.fx") == 0;
      require(!legacy_enabled || game3d, "Legacy final-AA opt-in is only for frozen SunshineGame3D shaders");
      const bool expected = !game3d || legacy_enabled;
      require(bool(uniform("USE_AA", false).handle) == expected,
        expected ? "Expected original/legacy final-AA control is missing" : "Retired Game3D final-AA control is still reflected");
      return expected;
    }

    void set_int(const char *name, int value) {
      // This one retired control has an explicit effect/version contract.
      // Enabled-AA requests still fail; no other missing uniform is tolerated.
      if (std::strcmp(name, "USE_AA") == 0 && !final_aa_available()) {
        require(value == 0, "AA1 was requested from Game3D after final-AA removal");
        return;
      }
      observed.runtime->set_uniform_value_int(uniform(name), &value, 1);
    }

    void texture_named(const char *name, com_ptr<ID3D11Texture2D> &result, DXGI_FORMAT format, bool full_resolution = true) {
      observed.runtime->enumerate_texture_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char candidate_name[256] {};
        runtime->get_texture_variable_name(variable, candidate_name);
        if (!named(candidate_name, name)) {
          return;
        }
        api::resource_view view {};
        runtime->get_texture_binding(variable, &view, nullptr);
        require(view.handle != 0, "Actual depth intermediate has no native view");
        com_ptr<ID3D11Resource> resource;
        reinterpret_cast<ID3D11View *>(view.handle)->GetResource(resource.put());
        checked(resource->QueryInterface(IID_PPV_ARGS(result.put())), "Read actual shader intermediate resource");
      });
      require(result.p != nullptr, "Actual shader depth intermediate is missing");
      D3D11_TEXTURE2D_DESC desc {};
      result->GetDesc(&desc);
      const bool geometry = full_resolution ? desc.Width == width && desc.Height == height : desc.Width > 0 && desc.Width <= width && desc.Height > 0 && desc.Height <= height;
      require(geometry && desc.Format == format, "Actual depth intermediate geometry/format differs");
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

    void prepare_depth() {
      set_int("Depth_Map_View", 0);
      set_int("Depth_Map", 0);
      set_float("Depth_Adjustment", 0.f);
      set_float("Zero_Parallax_Distance", .05f);
      set_float("Depth_Map_Adjust", 40.f);
      set_float("Auto_Depth_Adjust", 0.f);
      set_float("ZPD_Balance", 0.f);
      set_int("ZPD_Boundary", 0);
      set_float("PopOut_Target", 0.f);
      set_int("Range_Boost", 0);
      if (sunshine_depth3d_fixture::sharpening_available(*this)) set_float("Sharpen_Power", 0.f);
      set_int("USE_AA", 0);
      set_int("View_Mode", 1);
      // Isolate shared preparation from profile-specific alignment and automatic convergence.
      for (const char *name : {"Depth_Map_Flip", "Eye_Swap", "DB_AutoFit", "Flip_HV_Scale"}) {
        const auto handle = uniform(name, false);
        if (handle.handle) {
          observed.runtime->set_uniform_value_bool(handle, false);
        }
      }
      set_int("Auto_Scaler_Adjust", 0);
      set_float("AR_Side_Shrink", 0.f);
      const float zero2[2] {0, 0}, one2[2] {1, 1};
      for (const char *name : {"DLSS_FSR_Offset", "Image_Position_Adjust"}) {
        const auto handle = uniform(name, false);
        if (handle.handle) {
          observed.runtime->set_uniform_value_float(handle, zero2, 2);
        }
      }
      const auto scale = uniform("Horizontal_and_Vertical", false);
      if (scale.handle) {
        observed.runtime->set_uniform_value_float(scale, one2, 2);
      }
      const unsigned zero_uint2[2] {0, 0};
      observed.runtime->set_uniform_value_uint(uniform("Starting_Resolution"), zero_uint2, 2);
      observed.runtime->enumerate_uniform_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_uniform_variable variable) {
        char source[64] {};
        if (runtime->get_annotation_string_from_uniform_variable(variable, "source", source) && std::strcmp(source, "bufready_depth") == 0) {
          observed.depth_ready_uniforms.push_back(variable);
        }
      });
      require(observed.depth_ready_uniforms.size() >= 2, "Original and Sunshine integration depth-readiness uniforms must be supplied");
      texture_named("texzBufferN_P", linear_depth, DXGI_FORMAT_R16G16_FLOAT, false);

      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = desc.ArraySize = 1;
      desc.SampleDesc.Count = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      checked(device->CreateTexture2D(&desc, nullptr, depth.put()), "Create synthetic native hardware depth");
      checked(device->CreateShaderResourceView(depth.p, nullptr, depth_view.put()), "Create actual DEPTH semantic view");
      observed.depth_view = api::resource_view {reinterpret_cast<std::uint64_t>(depth_view.p)};
      observed.inject_depth = observed.depth_ready = true;

      // Native stripes expose displacement, and diagonal detail exposes AA and
      // sharpening. Vertical asymmetry also catches upside-down depth mappings.
      backbuffer->GetDesc(&desc);
      desc.BindFlags = desc.MiscFlags = 0;
      const unsigned bytes = color == 2 ? 8 : 4;
      std::vector<std::uint8_t> pattern(size_t(width) * height * bytes);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const size_t pixel = size_t(y) * width + x;
          const bool stripe = ((x / 16) % 2) != (((x / 16) / 7 + (x / 16) / 13) % 2);
          if (color == 1) {
            const std::uint8_t values[4] {std::uint8_t(stripe ? 224 : 32), std::uint8_t(y < height / 3 ? 200 : 60), std::uint8_t(((x + 2 * y) / 31) % 2 ? 170 : 80), 255};
            std::memcpy(pattern.data() + pixel * 4, values, 4);
          } else if (color == 2) {
            const std::uint16_t values[4] {std::uint16_t(stripe ? 0x4000 : 0x3400), std::uint16_t(y < height / 3 ? 0x3c00 : 0x3800), std::uint16_t(((x + 2 * y) / 31) % 2 ? 0x3a00 : 0xb000), 0x3c00};
            std::memcpy(pattern.data() + pixel * 8, values, 8);
          } else {
            const std::uint32_t value = (stripe ? 700u : 256u) | ((y < height / 3 ? 600u : 300u) << 10) | ((((x + 2 * y) / 31) % 2 ? 500u : 200u) << 20) | (3u << 30);
            std::memcpy(pattern.data() + pixel * 4, &value, 4);
          }
        }
      }
      const D3D11_SUBRESOURCE_DATA data {pattern.data(), width * bytes, 0};
      checked(device->CreateTexture2D(&desc, &data, source_pattern.put()), "Create asymmetric native game source pattern");
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
              const float expected = a + (b - a) * blend, actual = exported_channel(stereo, eye * width + x, y, 0);
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
        std::printf("MEASURE original shift setting=%s eye=%u source_pixels=%.3f normalized_error=%.9g\n", setting, eye, best_shift, best_error);
        require(std::abs(best_shift) < limit - 1, "Original displacement fit reached its search boundary");
        require(best_error < .015, "Actual export does not correlate with a translated constant-plane source");
        result[eye] = best_shift;
      }
      return result;
    }

    void diagnose_flat_eyes(unsigned x, unsigned y, unsigned channel) {
      for (const char *name : {"SunshineEyeLeft", "SunshineEyeRight", "SunshineSharpLeft", "SunshineSharpRight"}) {
        // Only the original effect retains separate sharpening intermediates.
        if (std::strcmp(effect_file, "SunshineGame3D.fx") == 0 && std::strncmp(name, "SunshineSharp", 13) == 0) continue;
        com_ptr<ID3D11Texture2D> texture;
        texture_named(name, texture, DXGI_FORMAT_R16G16B16A16_FLOAT);
        const auto pixels = read(texture.p);
        std::printf("DIAGNOSTIC %s xy=%u,%u channel=%u neighboring_pixels=", name, x, y, channel);
        for (int offset = -3; offset <= 3; ++offset) {
          const unsigned px = unsigned(std::clamp(int(x) + offset, 0, int(width) - 1));
          const size_t sample = (size_t(y) * width + px) * 4 + channel;
          std::printf(" %.9g", half_float(reinterpret_cast<const std::uint16_t *>(pixels.data())[sample]));
        }
        std::putchar('\n');
      }
    }

    void check_original_only() {
      observed.runtime->enumerate_uniform_variables(effect_file, [](api::effect_runtime *runtime, api::effect_uniform_variable variable) {
        char name[256] {};
        runtime->get_uniform_variable_name(variable, name);
        require(std::strstr(name, "HostSBS_") == nullptr, "Removed Sunshine warp control remains in the compiled effect");
      });
      observed.runtime->enumerate_texture_variables(effect_file, [](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char name[256] {};
        runtime->get_texture_variable_name(variable, name);
        require(std::strstr(name, "HostSBS_") == nullptr, "Removed Sunshine warp resource remains in the compiled effect");
      });
      require(uniform("Sunshine_DepthReady", false).handle != 0, "Sunshine integration readiness uniform is missing");
      std::puts("PASS compiled effect contains original Depth3D and Sunshine export integration without alternate warp controls/resources");
    }

    std::vector<float> linear_samples(const char *plane_label = nullptr) {
      const auto bytes = read(linear_depth.p);
      require(!bytes.empty() && bytes.size() % 4 == 0, "Production linear depth is not packed RG16F");
      D3D11_TEXTURE2D_DESC desc {};
      linear_depth->GetDesc(&desc);
      require(bytes.size() == size_t(desc.Width) * desc.Height * 4, "Production linear depth readback size differs");
      std::vector<float> result;
      result.reserve(bytes.size() / 4);
      float worst_error = 0;
      size_t worst_pixel = 0, bad_pixels = 0;
      unsigned low_x = unsigned(desc.Width), low_y = desc.Height, high_x = 0, high_y = 0;
      for (size_t i = 0; i < bytes.size() / 4; ++i) {
        // Mod_Z uses four corner patches of this production texture for
        // temporal/weapon metadata. Only the remaining green texels are scene
        // depth. At 75% depth resolution the second texel center lies exactly
        // on the two-native-pixel boundary. Float raster interpolation can put
        // that center inside the shader's strict comparison, so include the
        // equality boundary in the metadata region. Integer center arithmetic
        // avoids a different rounding error from subtracting UV from one.
        const std::uint64_t cx = 2 * (i % desc.Width) + 1;
        const std::uint64_t cy = 2 * (i / desc.Width) + 1;
        if (std::min(cx, 2 * std::uint64_t(desc.Width) - cx) * width <= 4 * std::uint64_t(desc.Width) &&
            std::min(cy, 2 * std::uint64_t(desc.Height) - cy) * height <= 4 * std::uint64_t(desc.Height)) continue;
        std::uint16_t green = 0;
        std::memcpy(&green, bytes.data() + 4 * i + 2, sizeof(green));
        const float value = half_float(green);
        require(std::isfinite(value), "Production linear depth contains a non-finite value");
        if (plane_label) {
          const float error = std::abs(value - 1.f / (1.f + 319.f * .01f));
          const unsigned x = unsigned(i % desc.Width), y = unsigned(i / desc.Width);
          if (error > worst_error) { worst_error = error; worst_pixel = i; }
          if (error > .002f) {
            if (bad_pixels < 16) std::printf("MEASURE linear plane %s outlier xy=%u,%u value=%.9g error=%.9g\n", plane_label, x, y, value, error);
            ++bad_pixels;
            low_x = std::min(low_x, x); high_x = std::max(high_x, x);
            low_y = std::min(low_y, y); high_y = std::max(high_y, y);
          }
        }
        result.push_back(value);
      }
      if (plane_label && bad_pixels) std::printf("MEASURE linear plane %s texture=%ux%u native=%ux%u bad_count=%zu bounds=%u,%u-%u,%u worst_xy=%zu,%zu worst_error=%.9g\n",
          plane_label, unsigned(desc.Width), desc.Height, width, height, bad_pixels, low_x, low_y, high_x, high_y, worst_pixel % size_t(desc.Width), worst_pixel / size_t(desc.Width), worst_error);
      require(!result.empty(), "Production linear depth contains no scene texels");
      return result;
    }

    void check_original_pipeline() {
      prepare_depth();
      std::vector<float> hardware(size_t(width) * height, .99f);
      context->UpdateSubresource(depth.p, 0, nullptr, hardware.data(), width * sizeof(float), 0);
      set_float("Depth_Adjustment", 0);
      measured_frames();
      const auto flat = read(exported.p);
      require(std::abs(exported_channel(flat, 8, height / 2, 0) - exported_channel(flat, 24, height / 2, 0)) > .05f, "Flat export lost the nonconstant native source");

      float maximum_eye_error = 0;
      unsigned error_x = 0, error_y = 0, error_channel = 0;
      for (unsigned y = 0; y < height; y += std::max(1u, height / 360)) {
        for (unsigned x = 0; x < width; ++x) {
          for (unsigned c = 0; c < 3; ++c) {
            const float a = exported_channel(flat, x, y, c), b = exported_channel(flat, width + x, y, c);
            require(std::isfinite(a) && std::isfinite(b), "Zero divergence export contains invalid pixels");
            if (std::abs(a - b) > maximum_eye_error) {
              maximum_eye_error = std::abs(a - b);
              error_x = x;
              error_y = y;
              error_channel = c;
            }
          }
        }
      }
      std::printf("MEASURE zero-divergence eye_error=%.9g xy=%u,%u channel=%u left=%.9g right=%.9g\n", maximum_eye_error, error_x, error_y, error_channel, exported_channel(flat, error_x, error_y, error_channel), exported_channel(flat, width + error_x, error_y, error_channel));
      if (maximum_eye_error >= .003f) {
        diagnose_flat_eyes(error_x, error_y, error_channel);
      }
      require(maximum_eye_error < .003f, "Zero divergence does not present two complete identical source eyes");

      // Original POM has a compatibility offset even at tiny nonzero strength.
      // Measure that offset from exported pixels, then quantify each eye's
      // response to two nonzero strengths and a ZPD crossing on the same plane.
      set_float("Compatibility_Power", 1);
      std::array<std::array<float, 2>, 4> shifts {};
      const std::array<float, 4> strengths {.001f, 15.f, 65.f, 65.f};
      const std::array<float, 4> zpds {.05f, .05f, .05f, .25f};
      const char *settings[] {"tiny-positive baseline", "divergence15", "divergence65", "ZPD.25"};
      for (unsigned setting = 0; setting < strengths.size(); ++setting) {
        set_float("Zero_Parallax_Distance", zpds[setting]);
        set_float("Depth_Adjustment", strengths[setting]);
        measured_frames();
        shifts[setting] = measure_eye_shifts(flat, read(exported.p), settings[setting]);
      }
      std::array<std::array<float, 2>, 3> corrected {};
      for (unsigned setting = 1; setting < strengths.size(); ++setting) {
        for (unsigned eye = 0; eye < 2; ++eye) {
          const float actual = shifts[setting][eye] - shifts[0][eye];
          corrected[setting - 1][eye] = actual;
          const float direction = (setting == 3 ? -1.f : 1.f) * (eye == 0 ? 1.f : -1.f);
          std::printf("MEASURE original compatibility-adjusted shift setting=%s eye=%u actual=%.3f baseline=%.3f\n",
                      settings[setting], eye, actual, shifts[0][eye]);
          require(actual * direction > 0, "Original eye displacement has the wrong popout/ZPD direction");
        }
        require(std::abs(corrected[setting - 1][0] + corrected[setting - 1][1]) <= 1.f,
                "Original constant-plane eyes do not move equally in opposite directions");
      }
      for (unsigned eye = 0; eye < 2; ++eye) {
        const float weak = std::abs(corrected[0][eye]), strong = std::abs(corrected[1][eye]);
        require(weak > .5f && strong > weak + 1.f && strong > weak * 2,
                "Original export plateaus across nonzero depth strengths");
        require(corrected[1][eye] * corrected[2][eye] < 0,
                "Moving ZPD across the plane does not reverse original geometric eye displacement");
      }
      std::puts("PASS original exported eye displacement increases quantitatively at strengths15/65, moves opposite eyes, and reverses across ZPD after measured compatibility-bias subtraction");
      set_float("Depth_Adjustment", 50);
      set_float("Zero_Parallax_Distance", .05f);
      // Reversed hardware codes describe the same plane. The comparison is of
      // real shared preparation, without a retired physical-camera formula.
      set_int("Depth_Map", 0);
      measured_frames();
      const auto normal = linear_samples("normal");
      std::fill(hardware.begin(), hardware.end(), .01f);
      context->UpdateSubresource(depth.p, 0, nullptr, hardware.data(), width * sizeof(float), 0);
      set_int("Depth_Map", 1);
      measured_frames();
      const auto reversed = linear_samples("reversed");
      float reversal_error = 0, projection_error = 0;
      // DMA=40 and near=.125/far=1 give this normalized linear depth for
      // equivalent .99 normal-Z and .01 reversed-Z hardware depth.
      constexpr float expected_plane = 1.f / (1.f + 319.f * .01f);
      require(normal.size() == reversed.size(), "Linear-depth size changed between projection modes");
      for (size_t i = 0; i < normal.size(); ++i) {
        const float a = normal[i], b = reversed[i];
        require(std::isfinite(a) && std::isfinite(b), "Shared prepared depth contains non-finite values");
        reversal_error = std::max(reversal_error, std::abs(a - b));
        projection_error = std::max(projection_error, std::max(std::abs(a - expected_plane), std::abs(b - expected_plane)));
      }
      std::printf("MEASURE production linear plane normal=%.9g reversed=%.9g expected=%.9g reversal_error=%.9g projection_error=%.9g\n", normal[normal.size() / 2], reversed[reversed.size() / 2], expected_plane, reversal_error, projection_error);
      require(reversal_error < .002f, "Normal/reversed representations disagree in shared preparation");
      require(projection_error < .002f, "Production linear depth disagrees with the independent projection oracle");
      std::printf("PASS real normal/reversed DEPTH shares production linear map; maximum error=%.9g\n", reversal_error);

      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          hardware[size_t(y) * width + x] = .001f + .045f * float(x) / float(width - 1) + .003f * float(y) / float(height - 1);
        }
      }
      context->UpdateSubresource(depth.p, 0, nullptr, hardware.data(), width * sizeof(float), 0);
      set_int("Depth_Map_View", 2);
      {
        set_float("Depth_Adjustment", 50);
        set_float("Zero_Parallax_Distance", .05f);
        measured_frames();
        const auto pixels = read(exported.p);

        // Normal Depth describes the input scene, so artistic stereo controls
        // must not change its pixels. This rejects a display of signed,
        // convergence-adjusted depth: saturating that field hides near geometry.
        const std::array<float, 4> diagnostic_strengths {0.f, 100.f, 50.f, 50.f};
        const std::array<float, 4> diagnostic_zpds {.05f, .05f, 0.f, .8f};
        for (unsigned setting = 0; setting < diagnostic_strengths.size(); ++setting) {
          set_float("Depth_Adjustment", diagnostic_strengths[setting]);
          set_float("Zero_Parallax_Distance", diagnostic_zpds[setting]);
          measured_frames();
          const auto changed = read(exported.p);
          const float difference = image_difference(pixels, changed);
          std::printf("MEASURE normal-depth invariance strength=%.3f ZPD=%.3f difference=%.9g identical=%d\n", diagnostic_strengths[setting], diagnostic_zpds[setting], difference, changed == pixels);
          require(changed == pixels, "Normal depth diagnostic changes with Depth Adjustment or ZPD instead of showing unchanged scene depth");
        }
        set_float("Depth_Adjustment", 50);
        set_float("Zero_Parallax_Distance", .05f);
        measured_frames();
        require(read(exported.p) == pixels, "Restoring stereo controls does not restore the unchanged normal-depth diagnostic");

        float lo = INFINITY, hi = -INFINITY, eye_error = 0, stripe_error = 0, monotonic_error = 0, expected_error = 0, grayscale_error = 0;
        for (unsigned y = height / 8; y < height * 7 / 8; y += std::max(1u, height / 90)) {
          for (unsigned x = 0; x < width; ++x) {
            const float left = exported_channel(pixels, x, y, 0), right = exported_channel(pixels, width + x, y, 0);
            require(std::isfinite(left) && std::isfinite(right), "Normal depth diagnostic contains non-finite pixels");
            lo = std::min(lo, left);
            hi = std::max(hi, left);
            eye_error = std::max(eye_error, std::abs(left - right));
            for (unsigned eye = 0; eye < 2; ++eye) {
              for (unsigned component = 1; component < 3; ++component) {
                const float value = exported_channel(pixels, eye * width + x, y, component);
                require(std::isfinite(value), "Normal depth diagnostic contains a non-finite color channel");
                grayscale_error = std::max(grayscale_error, std::abs(value - left));
              }
            }
            // The bound synthetic R32F DEPTH is a reversed-Z ramp. With the
            // fixture's known near-plane adjustment 40 and range boost off,
            // its linear depth is 1 / (1 + (40/.125 - 1) * raw). This oracle
            // depends on the input projection, not on the warp's output field.
            const float expected = 1.f / (1.f + 319.f * hardware[size_t(y) * width + x]);
            expected_error = std::max(expected_error, std::abs(left - expected));
            if (x) {
              monotonic_error = std::max(monotonic_error, left - exported_channel(pixels, x - 1, y, 0));
            }
            // A smooth monotonic synthetic ramp must not acquire isolated white
            // columns, including row-chunk boundaries and either eye's edges.
            if (x > 2 && x + 3 < width) {
              const float before = exported_channel(pixels, x - 1, y, 0), after = exported_channel(pixels, x + 1, y, 0);
              stripe_error = std::max(stripe_error, left - std::max(before, after));
            }
          }
        }
        std::printf("MEASURE normal-depth eye_error=%.9g range=%.9g isolated-column=%.9g monotonic-error=%.9g expected-error=%.9g grayscale-error=%.9g\n", eye_error, hi - lo, stripe_error, monotonic_error, expected_error, grayscale_error);
        require(hi - lo > .01f && eye_error < .002f && stripe_error < .01f, "Shared normal-depth diagnostic is blank, splits eyes, or adds vertical stripes");
        require(monotonic_error < .002f && expected_error < .002f && grayscale_error < .002f, "Normal depth diagnostic differs from the finite monotonic linear scene-depth oracle");
      }
      std::puts("PASS shared normal-depth diagnostic matches linear scene depth, is invariant to stereo controls, and repeats a complete finite monotonic map without stripes in both eyes");
      set_int("Depth_Map_View", 0);

      // Diagonal blue edges and vertical red stripes expose both AA and
      // sharpening. Compare actual exported pixels, then disable to prove the
      // postprocessing response is reversible with the same source frame.
      if (sunshine_depth3d_fixture::sharpening_available(*this)) {
        set_float("Depth_Adjustment", 0);
        set_float("Sharpen_Power", 0);
        set_int("USE_AA", 0);
        measured_frames();
        const auto unprocessed = read(exported.p);
        set_float("Sharpen_Power", 1);
        measured_frames();
        const float sharp_response = image_difference(unprocessed, read(exported.p));
        set_float("Sharpen_Power", 0);
        measured_frames();
        require(image_difference(unprocessed, read(exported.p)) < .003f, "Disabling sharpening does not restore original export");
        require(sharp_response > .001f, "Shared sharpening does not reach exported DoubleTex");
        if (final_aa_available()) {
          set_int("USE_AA", 1);
          measured_frames();
          const float aa_response = image_difference(unprocessed, read(exported.p));
          set_int("USE_AA", 0);
          measured_frames();
          require(image_difference(unprocessed, read(exported.p)) < .003f, "Disabling AA does not restore original export");
          require(aa_response > .001f, "Original/legacy AA does not reach exported DoubleTex");
          std::printf("MEASURE exported post sharpen=%.9g AA=%.9g\n", sharp_response, aa_response);
          std::puts("PASS original/legacy AA and sharpen alter final exported pixels, reversible; physical mono byte-exact");
        } else {
          std::printf("MEASURE exported post sharpen=%.9g final_AA=absent\n", sharp_response);
          std::puts("PASS frozen legacy Game3D final-AA absent with reversible sharpening; physical mono byte-exact");
        }
      }

      set_float("Depth_Adjustment", 65);
      observed.depth_ready = false;
      {
        measured_frames();
        const auto fallback = read(exported.p);
        const auto fallback_error = image_difference(flat, fallback);
        std::printf("MEASURE missing-depth versus zero-strength color difference=%.9g source=%u compatible=%u\n", fallback_error, color, compatible);
        const float excess = sunshine_color_test::fallback_excess(width, height, color,
          [&](unsigned x, unsigned y, unsigned c) { return exported_channel(flat, x, y, c); },
          [&](unsigned x, unsigned y, unsigned c) { return exported_channel(fallback, x, y, c); });
        require(excess == 0.f, "Missing DEPTH differs from flat source eyes beyond export quantization");
      }
      std::puts("PASS missing real DEPTH/readiness returns flat duplicate source eyes");
    }

    void run() {
      discover();
      std::printf("MEASURE sharpening contract available=%u\n", unsigned(sunshine_depth3d_fixture::sharpening_available(*this)));
      std::printf("MEASURE final-AA contract available=%u legacy_opt_in=%s\n", unsigned(final_aa_available()),
        std::getenv("SUNSHINE_GAME3D_LEGACY_FINAL_AA") ? std::getenv("SUNSHINE_GAME3D_LEGACY_FINAL_AA") : "0");
      {
        if (color == 1) {
          check_colors({0.25f, 0.5f, 0.75f, 1.f}, {64.f / 255.f, 128.f / 255.f, 191.f / 255.f});
          check_colors({1, 0, 1, 1}, {1, 0, 1});
        } else if (color == 2) {
          check_colors({4, .5f, -.125f, 1}, {4, .5f, -.125f});
          check_colors({125, 12.5f, 1, 1}, {125, 12.5f, 1});
        } else {
          // Exact native 10-bit endpoints avoid introducing a CPU PQ decoder.
          // Rec.2020 red yields scRGB gamut negatives and highlights above one.
          check_colors({1, 1, 1, 1}, {125, 125, 125});
          check_colors({1, 0, 0, 1}, {207.5613753f, -15.5688093f, -2.2688454f});
        }
      }
      check_original_pipeline();
      std::printf("PASS actual official ReShade 6.8 full %s runtime source=%u compatibility=%u original preparation, divergence/ZPD, diagnostic and exported postprocessing; %u natural technique executions. No game-specific depth-selection or performance claim.\n", effect_file, color, compatible, observed.renders);
    }
  };
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 6) {
    std::fprintf(stderr, "usage: test_depth3d_runtime <official-ReShade64.dll> <external-Depth3D-Shaders> <isolated-ignored-output-directory> <srgb|scrgb|pq> <HDR_Compatible_Mode:0|1>\n");
    return 2;
  }
  std::thread([] {
    Sleep(70000);
    std::fputs("FAIL full Depth3D runtime watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 3);
  }).detach();
  try {
    const unsigned color = std::strcmp(argv[4], "srgb") == 0 ? 1 : std::strcmp(argv[4], "scrgb") == 0 ? 2 :
                                                                 std::strcmp(argv[4], "pq") == 0      ? 3 :
                                                                                                        0;
    require(color && (std::strcmp(argv[5], "0") == 0 || std::strcmp(argv[5], "1") == 0), "Invalid fixture color or compatibility argument");
    fixture_t fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fs::absolute(argv[3]), color, unsigned(argv[5][0] - '0'));
    fixture.run();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
