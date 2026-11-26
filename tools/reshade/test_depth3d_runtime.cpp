// SPDX-License-Identifier: GPL-3.0-only
// The D3D11 base fixture of the native runtime test: the official ReShade DLL, a
// hidden game window and swapchain with the separately supplied frozen effect,
// and readback helpers. No shader source is distributed here.
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
    com_ptr<ID3D11Texture2D> depth, source_pattern;
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
  };
}  // namespace
