// SPDX-License-Identifier: GPL-3.0-only
// Opt-in actual ReShade 6.8 D3D12 full-effect depth/geometry fixture. Loads only
// separately supplied Depth3D files into an isolated personal probe directory.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <reshade.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>
#include "test_depth3d_effect.h"
#include "test_color_precision.h"
#include "test_stereo_parity.h"
#include "test_depth3d_color.h"
#include "test_game3d_camera.h"
#include "test_game3d_jitter.h"

namespace {
  namespace api = reshade::api;
  namespace fs = std::filesystem;
  using sunshine_depth3d_fixture::effect_file;
  using sunshine_depth3d_fixture::technique_name;
  unsigned width = 640, height = 360;

  template<class T>
  struct com_ptr {
    T *p = nullptr;
    com_ptr() = default;
    com_ptr(const com_ptr &) = delete;
    com_ptr &operator=(const com_ptr &) = delete;

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

  void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void checked(HRESULT hr, const char *message) {
    if (FAILED(hr)) {
      char error[400];
      std::snprintf(error, sizeof(error), "%s (0x%08lx)", message, static_cast<unsigned long>(hr));
      throw std::runtime_error(error);
    }
  }

  bool named(const char *name, const char *expected) {
    const std::string value(name), tail(expected);
    return value == tail || (value.size() > tail.size() && value.compare(value.size() - tail.size(), tail.size(), tail) == 0 && value[value.size() - tail.size() - 1] == ':');
  }

  float half_float(std::uint16_t value) {
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

  unsigned bytes_per_pixel(DXGI_FORMAT format) {
    if (format == DXGI_FORMAT_R32G32B32A32_FLOAT) return 16;
    if (format == DXGI_FORMAT_R16_FLOAT) return 2;
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4;
  }

  void transition(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    commands->ResourceBarrier(1, &barrier);
  }

  D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES properties {};
    properties.Type = type;
    properties.CreationNodeMask = properties.VisibleNodeMask = 1;
    return properties;
  }

  struct observations_t {
    api::effect_runtime *runtime = nullptr;
    api::color_space color = api::color_space::unknown;
    api::resource_view depth_view {}, bound_depth_view {};
    ID3D12Resource *mono = nullptr;
    std::vector<api::effect_uniform_variable> ready_uniforms;
    unsigned reloads = 0, renders = 0, captures = 0;
    bool inject = false, ready = false, capture = false;
  } observed;

  void on_init(api::effect_runtime *runtime) {
    observed.runtime = runtime;
  }

  void on_reload(api::effect_runtime *) {
    ++observed.reloads;
    sunshine_parity::timeline_reload();
  }

  void on_present(api::command_queue *, api::swapchain *swapchain, const api::rect *, const api::rect *, std::uint32_t, const api::rect *) {
    observed.color = swapchain->get_color_space();
  }

  void on_begin_effects(api::effect_runtime *runtime, api::command_list *commands, api::resource_view, api::resource_view) {
    sunshine_parity::timeline_before_effects(runtime, effect_file, technique_name);
    if (!observed.inject) {
      sunshine_parity::begin(commands);
      return;
    }
    const auto view = observed.ready ? observed.depth_view : api::resource_view {};
    // The official generic-depth add-on updates this semantic when its selected
    // resource changes. D3D12 update_texture_bindings waits for descriptor users;
    // do not turn that into an unnecessary per-frame wait in this fixture.
    if (observed.bound_depth_view.handle != view.handle) {
      runtime->update_texture_bindings("DEPTH", view, view);
      observed.bound_depth_view = view;
    }
    for (const auto uniform : observed.ready_uniforms) {
      runtime->set_uniform_value_bool(uniform, observed.ready);
    }
    sunshine_parity::begin(commands);
  }

  void on_technique(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *commands, api::resource_view rtv, api::resource_view) {
    char name[256] {};
    runtime->get_technique_name(technique, name);
    if (!named(name, technique_name)) {
      return;
    }
    ++observed.renders;
    sunshine_parity::timeline_after_technique(runtime);
    sunshine_parity::end(commands);
    if (observed.capture && observed.mono && rtv.handle) {
      const auto source = runtime->get_device()->get_resource_from_view(rtv);
      const api::resource destination {reinterpret_cast<std::uint64_t>(observed.mono)};
      commands->barrier(source, api::resource_usage::render_target, api::resource_usage::copy_source);
      commands->copy_resource(source, destination);
      commands->barrier(source, api::resource_usage::copy_source, api::resource_usage::render_target);
      ++observed.captures;
    }
  }

  struct fixture_t {
    HMODULE runtime_module = nullptr;
    HWND window = nullptr;
    com_ptr<ID3D12Device> game;
    com_ptr<ID3D12CommandQueue> queue;
    com_ptr<ID3D12CommandAllocator> allocator;
    com_ptr<ID3D12GraphicsCommandList> commands;
    com_ptr<IDXGISwapChain3> swapchain;
    com_ptr<ID3D12Resource> backbuffers[2], mono, source_upload, depth, exported, linear_depth;
    com_ptr<ID3D12Fence> completion;
    HANDLE completion_event = nullptr;
    std::uint64_t fence_value = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT source_footprint {};
    api::resource_view depth_view {};
    unsigned color = 0, compatible = 0;
    DXGI_FORMAT source_format = DXGI_FORMAT_UNKNOWN;
    std::vector<std::uint8_t> source_bytes;
    fs::path native_oracle_output;
    std::function<void()> render_tracked_depth;

    ~fixture_t() {
      observed.capture = observed.inject = false;
      observed.mono = nullptr;
      observed.ready_uniforms.clear();
      if (observed.runtime && depth_view.handle) {
        observed.runtime->update_texture_bindings("DEPTH", {}, {});
        observed.runtime->get_device()->destroy_resource_view(depth_view);
      }
      if (runtime_module) {
        reshade::unregister_addon(GetModuleHandleW(nullptr), runtime_module);
      }
      exported.reset();
      linear_depth.reset();
      depth.reset();
      mono.reset();
      source_upload.reset();
      backbuffers[0].reset();
      backbuffers[1].reset();
      swapchain.reset();
      commands.reset();
      allocator.reset();
      completion.reset();
      queue.reset();
      game.reset();
      if (completion_event) {
        CloseHandle(completion_event);
      }
      if (window) {
        DestroyWindow(window);
      }
      // As in the natural runtime fixtures, leave the DLL loaded until process
      // exit, after the hooked native graphics objects have been released.
    }

    void buffer(com_ptr<ID3D12Resource> &resource, UINT64 size, D3D12_HEAP_TYPE type) {
      const auto heap = heap_properties(type);
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      desc.Width = size;
      desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(resource.put())), "Create fixture upload/readback buffer");
    }

    void texture(com_ptr<ID3D12Resource> &resource, DXGI_FORMAT format, D3D12_RESOURCE_STATES state,
        unsigned texture_width = 0, unsigned texture_height = 0) {
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = texture_width ? texture_width : width;
      desc.Height = texture_height ? texture_height : height;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = format;
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(resource.put())), "Create fixture texture");
    }

    void begin_commands() {
      checked(allocator->Reset(), "Reset fixture allocator after completion");
      checked(commands->Reset(allocator.p, nullptr), "Reset fixture command list");
    }

    void submit(bool present = false) {
      checked(commands->Close(), "Close fixture command list");
      ID3D12CommandList *lists[] {commands.p};
      queue->ExecuteCommandLists(1, lists);
      if (present) {
        checked(swapchain->Present(0, 0), "Actual natural D3D12 game Present");
      }
      checked(queue->Signal(completion.p, ++fence_value), "Fence fixture and natural ReShade work");
      checked(completion->SetEventOnCompletion(fence_value, completion_event), "Observe fixture completion");
      require(WaitForSingleObject(completion_event, 3000) == WAIT_OBJECT_0, "D3D12 fixture GPU work exceeded three seconds");
    }

    void fill_upload(com_ptr<ID3D12Resource> &upload, const D3D12_RESOURCE_DESC &desc, const void *data, D3D12_PLACED_SUBRESOURCE_FOOTPRINT &footprint) {
      UINT64 total = 0;
      game->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
      buffer(upload, total, D3D12_HEAP_TYPE_UPLOAD);
      void *mapped = nullptr;
      const D3D12_RANGE no_read {0, 0};
      checked(upload->Map(0, &no_read, &mapped), "Map fixture upload");
      const auto row = size_t(desc.Width) * bytes_per_pixel(desc.Format);
      for (UINT y = 0; y < desc.Height; ++y) {
        std::memcpy(static_cast<std::uint8_t *>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, static_cast<const std::uint8_t *>(data) + size_t(y) * row, row);
      }
      upload->Unmap(0, nullptr);
    }

    void upload_depth(const std::vector<float> &values) {
      const auto desc = depth->GetDesc();
      require(values.size() == size_t(desc.Width) * desc.Height, "Wrong synthetic hardware-depth size");
      com_ptr<ID3D12Resource> upload;
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
      fill_upload(upload, depth->GetDesc(), values.data(), footprint);
      begin_commands();
      transition(commands.p, depth.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
      D3D12_TEXTURE_COPY_LOCATION from {}, to {};
      from.pResource = upload.p;
      from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      from.PlacedFootprint = footprint;
      to.pResource = depth.p;
      to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(commands.p, depth.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      submit();
    }

    void initialize(const fs::path &runtime, const fs::path &shader_root, const fs::path &directory, unsigned source_color, unsigned hdr_compatible, const fs::path &depth_addon = {}) {
      sunshine_depth3d_fixture::select_effect();
      sunshine_parity::configure_timeline();
#ifdef SUNSHINE_DEPTH_SELECTION_RUNTIME
      require(!sunshine_parity::deterministic, "Deterministic renderer time must not alter selector, raw-policy or lifecycle fixtures");
#endif
      require(!sunshine_parity::deterministic || depth_addon.empty(), "Deterministic renderer parity cannot load a production depth/camera add-on");
      color = source_color;
      compatible = hdr_compatible;
      require(fs::is_regular_file(shader_root / effect_file), "Missing separately supplied Depth3D shader");
      const char *addon_filename = "SunshineSBS.addon64";
      bool automatic_performance = false;
#ifdef SUNSHINE_DEPTH_SELECTION_RUNTIME
      if (sunshine_camera_fixture::flag("SUNSHINE_DEPTH_MANUAL_RECOVERY_TEST")) addon_filename = "SunshineSBSTest.addon64";
#endif
#ifdef SUNSHINE_RAW_SCENE_RUNTIME
      automatic_performance = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_PERFORMANCE");
      if (sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST")) addon_filename = "SunshineSBSTest.addon64";
#endif
      fs::create_directories(directory / "effects");
      fs::create_directories(directory / "addons");
      for (const auto &entry : fs::directory_iterator(directory / "addons")) {
        const auto extension = entry.path().extension();
        if (extension == ".addon" || extension == ".addon64")
          require(!depth_addon.empty() && entry.path().filename() == addon_filename, "Fixture addon directory contains a legacy or unrelated addon; use a fresh isolated output");
      }
      if (!depth_addon.empty())
        fs::copy_file(depth_addon, directory / "addons" / addon_filename, fs::copy_options::overwrite_existing);
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
        config << "[ADDON]\nAddonPath=.\\addons\n";
        if (!depth_addon.empty())
#ifdef SUNSHINE_COMMAND_ASSOCIATION_RUNTIME
          config << "DisabledAddons=Generic Depth\n[DEPTH]\nDepthCopyBeforeClears=2\nUseAspectRatioHeuristics=1\n"
                    "[SUNSHINE_DEPTH]\nStreamlineCameraProbe=1\n";
#else
          config << "DisabledAddons=Generic Depth\n[DEPTH]\nDepthCopyBeforeClears="
                 << (sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST") ? 2 : 1)
                 << "\nUseAspectRatioHeuristics=1\n";
#endif
        #ifdef SUNSHINE_RAW_SCENE_RUNTIME
        config << "[SUNSHINE_DEPTH]\nAutoSelectSceneDepth=1\n";
        if (sunshine_camera_fixture::flag("SUNSHINE_DEPTH_GENERIC_ONLY_TEST"))
          config << "StreamlineDepthSource=0\nNGXDepthSource=0\nStreamlineCameraProbe=0\nUpscalerCallTrace=0\n";
        if (!sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC")) config << "RawSceneAutomation=1\n";
        #endif
        config << "[GENERAL]\nEffectSearchPaths=.\\effects\nPresetPath=.\\preset.ini\n"
                  "PerformanceMode=" << (automatic_performance ? 1 : 0) << "\nSkipLoadingDisabledEffects=0\nIntermediateCachePath=.\\cache\n"
                  "PreprocessorDefinitions=HDR_Compatible_Mode="
               << compatible << sunshine_depth3d_fixture::configuration_definitions() << sunshine_camera_fixture::definitions()
               << "\n[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n";
        std::ofstream preset(directory / "preset.ini");
        preset << "Techniques=" << technique_name << "@" << effect_file << "\nTechniqueSorting=" << technique_name << "@" << effect_file << "\n";
#ifdef SUNSHINE_RAW_SCENE_RUNTIME
        if (automatic_performance) {
          // Pinned Performance Mode specializes artistic uniforms at compile
          // time. Seed them before the runtime exists, never with test setters.
          preset << '[' << effect_file << "]\nDepth_Adjustment=100\nDepth_Map_View=0\n";
          if (sunshine_depth3d_fixture::expects_sharpening()) preset << "Sharpen_Power=0\n";
          if (std::strcmp(effect_file, "SunshineGame3D.fx"))
            preset << "USE_AA=0\nWP=1\nAuto_Scaler_Adjust=1\nRange_Boost=4\nAuto_Depth_Adjust=0.8\nZPD_OverShoot=0.75\nAR_Side_Shrink=0.2\n"
              "Depth_Map_Flip=1\nDB_AutoFit=1\nDLSS_FSR_Offset=0.02,-0.01\nDepth_Map_Adjust=40\n"
              "Compatibility_Power=1\nEye_Swap=0\nFlip_HV_Scale=0\nFlip_Opengl_Depth=0\nInficolor_3D_Emulator=0\n";
        }
#endif
        require(config.good() && preset.good(), "Could not write isolated fixture configuration");
      }
      sunshine_parity::provenance(runtime, shader_root, directory);
      sunshine_depth3d_color::initialize(runtime, shader_root, directory);
      sunshine_camera_fixture::initialize(runtime, shader_root, directory);
      sunshine_jitter_fixture::initialize(runtime, shader_root, directory);
      const bool mono_fallback = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_MONO_FALLBACK_TEST");
      const bool packing = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_PACKING_TEST");
      if (mono_fallback || packing) {
        require(mono_fallback != packing && !sunshine_camera_fixture::requested() && !sunshine_jitter_fixture::requested() &&
          !sunshine_depth3d_color::requested() && !sunshine_parity::requested(), "Select exactly one native shader oracle");
        native_oracle_output = directory / "native-oracle";
        require(!fs::exists(native_oracle_output), "Native shader oracle needs a fresh output directory");
        fs::create_directories(native_oracle_output);
        std::ofstream manifest(native_oracle_output / "provenance.txt");
        wchar_t executable[32768] {};
        require(GetModuleFileNameW(nullptr, executable, 32768) != 0, "Cannot identify native shader oracle executable");
        sunshine_parity::manifest_file(manifest, executable);
        sunshine_parity::manifest_file(manifest, runtime);
        for (const auto &entry : fs::recursive_directory_iterator(shader_root))
          if (entry.is_regular_file() && (entry.path().extension() == ".fx" || entry.path().extension() == ".fxh"))
            sunshine_parity::manifest_file(manifest, entry.path());
        require(manifest.good(), "Cannot save native shader oracle provenance");
      }
      SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", directory.c_str());
      wchar_t system[MAX_PATH] {};
      GetSystemDirectoryW(system, MAX_PATH);
      require(LoadLibraryW((fs::path(system) / "d3d12.dll").c_str()) != nullptr, "Could not preload native D3D12");
      runtime_module = LoadLibraryW((directory / "dxgi.dll").c_str());
      require(runtime_module != nullptr, "Could not load official ReShade 6.8 DLL");
      require(reshade::register_addon(GetModuleHandleW(nullptr), runtime_module), "Could not register actual runtime observer");
      reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init);
      reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(on_reload);
      reshade::register_event<reshade::addon_event::present>(on_present);
      reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
      reshade::register_event<reshade::addon_event::reshade_render_technique>(on_technique);
      WNDCLASSW wc {};
      wc.lpfnWndProc = DefWindowProcW;
      wc.hInstance = GetModuleHandleW(nullptr);
      wc.lpszClassName = L"Depth3DRealDepthD3D12Fixture";
      RegisterClassW(&wc);
      RECT bounds {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
      require(AdjustWindowRectEx(&bounds, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW), "Could not size hidden fixture window");
      window = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, wc.lpszClassName, L"Hidden Depth3D D3D12 depth probe", WS_OVERLAPPEDWINDOW, 0, 0, bounds.right - bounds.left, bounds.bottom - bounds.top, nullptr, nullptr, wc.hInstance, nullptr);
      require(window && !IsWindowVisible(window), "Could not create hidden fixture window");
      const auto create = reinterpret_cast<decltype(&D3D12CreateDevice)>(GetProcAddress(runtime_module, "D3D12CreateDevice"));
      const auto create_factory = reinterpret_cast<decltype(&CreateDXGIFactory2)>(GetProcAddress(runtime_module, "CreateDXGIFactory2"));
      require(create && create_factory, "Official runtime lacks D3D12 proxy exports");
      checked(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(game.put())), "Create wrapped native D3D12 game device");
      com_ptr<IDXGIFactory4> factory;
      checked(create_factory(0, IID_PPV_ARGS(factory.put())), "Create wrapped DXGI factory");
      D3D12_COMMAND_QUEUE_DESC queue_desc {};
      queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      checked(game->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue.put())), "Create game command queue");
      source_format = color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : color == 3 ? DXGI_FORMAT_R10G10B10A2_UNORM :
                                                                                 DXGI_FORMAT_R8G8B8A8_UNORM;
      DXGI_SWAP_CHAIN_DESC1 desc {};
      desc.Width = width;
      desc.Height = height;
      desc.Format = source_format;
      desc.SampleDesc.Count = 1;
      desc.BufferCount = 2;
      desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      com_ptr<IDXGISwapChain1> chain1;
      checked(factory->CreateSwapChainForHwnd(queue.p, window, &desc, nullptr, nullptr, chain1.put()), "Create natural ReShade D3D12 swapchain");
      checked(chain1->QueryInterface(IID_PPV_ARGS(swapchain.put())), "Query game color-space interface");
      checked(swapchain->SetColorSpace1(color == 2 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : color == 3 ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 :
                                                                                                            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),
              "Set real game source transfer");
      checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.put())), "Create fixture allocator");
      checked(game->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(commands.put())), "Create fixture commands");
      checked(commands->Close(), "Close initial fixture commands");
      checked(game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(completion.put())), "Create bounded fixture fence");
      completion_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      require(completion_event != nullptr, "Could not create fixture completion event");
      for (unsigned i = 0; i < 2; ++i) {
        checked(swapchain->GetBuffer(i, IID_PPV_ARGS(backbuffers[i].put())), "Get native game backbuffer");
      }
      texture(mono, source_format, D3D12_RESOURCE_STATE_COPY_DEST);
      observed.mono = mono.p;
      source_bytes.resize(size_t(width) * height * bytes_per_pixel(source_format));
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const size_t pixel = size_t(y) * width + x;
          const bool stripe = ((x / 16) % 2) != (((x / 16) / 7 + (x / 16) / 13) % 2);
          if (color == 1) {
            const std::uint8_t values[] {std::uint8_t(stripe ? 224 : 32), std::uint8_t(y < height / 3 ? 200 : 60), std::uint8_t(((x + 2 * y) / 31) % 2 ? 170 : 80), 255};
            std::memcpy(source_bytes.data() + pixel * 4, values, 4);
          } else if (color == 2) {
            const std::uint16_t values[] {std::uint16_t(stripe ? 0x4000 : 0x3400), std::uint16_t(y < height / 3 ? 0x3c00 : 0x3800), std::uint16_t(((x + 2 * y) / 31) % 2 ? 0x3a00 : 0xb000), 0x3c00};
            std::memcpy(source_bytes.data() + pixel * 8, values, 8);
          } else {
            const std::uint32_t value = (stripe ? 700u : 256u) | ((y < height / 3 ? 600u : 300u) << 10) | ((((x + 2 * y) / 31) % 2 ? 500u : 200u) << 20) | (3u << 30);
            std::memcpy(source_bytes.data() + pixel * 4, &value, 4);
          }
        }
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
    }

    void step() {
      const bool deterministic_step = sunshine_parity::timeline_running;
      const auto before_tick = sunshine_parity::timeline_tick;
      MSG message;
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      require(IsWindow(window) && !IsWindowVisible(window), "Fixture window became visible or disappeared");
      begin_commands();
      if (render_tracked_depth) render_tracked_depth();
      auto *backbuffer = backbuffers[swapchain->GetCurrentBackBufferIndex()].p;
      transition(commands.p, backbuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
      D3D12_TEXTURE_COPY_LOCATION from {}, to {};
      from.pResource = source_upload.p;
      from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      from.PlacedFootprint = source_footprint;
      to.pResource = backbuffer;
      to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(commands.p, backbuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
      submit(true);
      if (deterministic_step)
        require(sunshine_parity::timeline_tick == before_tick + 1 && !sunshine_parity::timeline_pending,
          "Deterministic input did not produce exactly one complete stereo execution");
      require(unsigned(observed.color) == color, "Actual runtime transfer differs from the native game swapchain");
      Sleep(3);
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
        throw std::runtime_error(std::string("Missing actual effect uniform ") + name);
      }
      return result;
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

    bool skip_retired_ray_control(const char *name) {
      if (!sunshine_parity::retired_ray_control(name) || !sunshine_parity::sole_host_warp(observed.runtime, effect_file)) return false;
      require(!uniform(name, false).handle, "Sole Host warp still exposes a retired ray control");
      std::printf("MEASURE retired ray control not applied: %s (shader capability v2 has no consumer)\n", name);
      return true;
    }

    void set_int(const char *name, int value) {
      // This one retired control has an explicit effect/version contract.
      // Enabled-AA requests still fail; no other missing uniform is tolerated.
      if (std::strcmp(name, "USE_AA") == 0 && !final_aa_available()) {
        require(value == 0, "AA1 was requested from Game3D after final-AA removal");
        return;
      }
      if (skip_retired_ray_control(name)) return;
      observed.runtime->set_uniform_value_int(uniform(name), &value, 1);
    }

    void set_float(const char *name, float value) {
      if (skip_retired_ray_control(name)) return;
      observed.runtime->set_uniform_value_float(uniform(name), &value, 1);
    }

    void find_texture(const char *name, com_ptr<ID3D12Resource> &resource, UINT expected_width, DXGI_FORMAT expected_format) {
      observed.runtime->enumerate_texture_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char candidate[256] {};
        runtime->get_texture_variable_name(variable, candidate);
        if (!named(candidate, name)) {
          return;
        }
        api::resource_view view {};
        runtime->get_texture_binding(variable, &view, nullptr);
        require(view.handle != 0, "Actual texture binding is missing");
        const auto actual = runtime->get_device()->get_resource_from_view(view);
        require(actual.handle != 0, "Adapted actual D3D12 texture getter returned null");
        auto *native = reinterpret_cast<ID3D12Resource *>(actual.handle);
        native->AddRef();
        resource.reset();
        resource.p = native;
        if (std::strcmp(name, "DoubleTex") == 0) {
          int marker = 0;
          unsigned w = 0, h = 0, source = 0;
          char transfer[32] {}, layout[32] {};
          require(runtime->get_annotation_int_from_texture_variable(variable, "sunshine_sbs_export", &marker, 1) && marker == 1, "Actual export marker missing");
          require(runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_width", &w, 1) && w == width && runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_height", &h, 1) && h == height, "Actual export source geometry wrong");
          require(runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_color_space", &source, 1) && source == color, "Actual source transfer permutation stale");
          require(runtime->get_annotation_string_from_texture_variable(variable, "sunshine_sbs_color_space", transfer) && std::strcmp(transfer, color == 1 ? "srgb" : "scrgb") == 0, "Actual export transfer wrong");
          require(runtime->get_annotation_string_from_texture_variable(variable, "sunshine_sbs_layout", layout) && std::strcmp(layout, "sbs_lr") == 0, "Actual export layout wrong");
        }
      });
      require(resource.p != nullptr, "Actual full effect did not expose expected texture");
      const auto desc = resource->GetDesc();
      const bool geometry = expected_width ? desc.Width == expected_width && desc.Height == height : desc.Width > 0 && desc.Width <= width && desc.Height > 0 && desc.Height <= height;
      require(desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && geometry && desc.Format == expected_format && desc.SampleDesc.Count == 1, "Actual intermediate/export texture geometry or format differs");
    }

    std::vector<std::uint8_t> read(ID3D12Resource *texture, D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE) {
      const auto desc = texture->GetDesc();
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
      UINT64 total = 0, row_bytes = 0;
      UINT row_count = 0;
      game->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &row_count, &row_bytes, &total);
      const size_t row = size_t(desc.Width) * bytes_per_pixel(desc.Format);
      require(row == row_bytes && row_count == desc.Height && row <= footprint.Footprint.RowPitch &&
        footprint.Offset + UINT64(desc.Height - 1) * footprint.Footprint.RowPitch + row <= total,
        "Native readback row size does not match the actual texture footprint");
      com_ptr<ID3D12Resource> staging;
      buffer(staging, total, D3D12_HEAP_TYPE_READBACK);
      begin_commands();
      transition(commands.p, texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
      D3D12_TEXTURE_COPY_LOCATION from {}, to {};
      from.pResource = texture;
      from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      to.pResource = staging.p;
      to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      to.PlacedFootprint = footprint;
      commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(commands.p, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
      submit();
      void *mapped = nullptr;
      const D3D12_RANGE range {0, static_cast<SIZE_T>(total)};
      checked(staging->Map(0, &range, &mapped), "Map completed native D3D12 readback");
      std::vector<std::uint8_t> bytes(row * desc.Height);
      for (UINT y = 0; y < desc.Height; ++y) {
        std::memcpy(bytes.data() + row * y, static_cast<const std::uint8_t *>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, row);
      }
      const D3D12_RANGE no_write {0, 0};
      staging->Unmap(0, &no_write);
      return bytes;
    }

    void capture_parity_preparation(const fs::path &base) {
      // Preserve every channel and corner of both actual Mod_Z outputs. These
      // readbacks happen between frames and never execute an additional effect.
      std::ostringstream layout;
      layout << "schema=parity-preparation-1\nsubresource=0\nformat=RG16F\n";
      for (const char *name : {"texzBufferN_P", "texzBufferN_L"}) {
        com_ptr<ID3D12Resource> texture;
        find_texture(name, texture, 0, DXGI_FORMAT_R16G16_FLOAT);
        const auto desc = texture->GetDesc();
        const auto bytes = read(texture.p);
        require(bytes.size() == size_t(desc.Width) * desc.Height * 4, "Parity preparation size differs");
        const auto path = fs::path(base.string() + "." + name + ".rg16f");
        sunshine_parity::write_bytes(path, bytes.data(), bytes.size());
        // Do not excuse undefined metadata by masking corners or replacing NaNs.
        for (size_t offset = 0; offset < bytes.size(); offset += 2) {
          std::uint16_t value {};
          std::memcpy(&value, bytes.data() + offset, sizeof(value));
          require(std::isfinite(half_float(value)), "Parity preparation contains nonfinite values");
        }
        layout << name << ' ' << desc.Width << ' ' << desc.Height << ' ' << bytes.size() << '\n';
      }
      const auto text = layout.str();
      sunshine_parity::write_bytes(fs::path(base.string() + ".preparation.txt"), text.data(), text.size());
    }

    void capture_host_warp(const fs::path &base) {
      for (const char *name : {"SunshineHostCandidate", "SunshineHostVerticalConditioned", "SunshineHostFinal"}) {
        com_ptr<ID3D12Resource> texture;
        find_texture(name, texture, 0, DXGI_FORMAT_R32_FLOAT);
        const auto desc = texture->GetDesc();
        require(desc.Width == width && desc.Height == height, "Host warp field is not full source resolution");
        const auto bytes = read(texture.p);
        require(bytes.size() == size_t(width) * height * sizeof(float), "Host warp readback size differs");
        for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
          float value;
          std::memcpy(&value, bytes.data() + offset, sizeof(value));
          require(std::isfinite(value) && std::abs(value) <= .040001f, "Host warp field is invalid or exceeds container");
        }
        sunshine_parity::write_bytes(fs::path(base.string() + "." + name + ".f32"), bytes.data(), bytes.size());
      }
    }

    void discover() {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
      const auto discovered = [&] { return sunshine_parity::deterministic ? sunshine_parity::timeline_main_seen : observed.renders != 0; };
      while (!discovered() && std::chrono::steady_clock::now() < deadline) {
        step();
        if (observed.runtime && observed.reloads && !discovered()) {
          observed.runtime->enumerate_techniques(effect_file, [](api::effect_runtime *runtime, api::effect_technique technique) {
            char name[256] {};
            runtime->get_technique_name(technique, name);
            if (named(name, technique_name)) {
              runtime->set_technique_state(technique, true);
            }
          });
        }
      }
      require(observed.runtime && discovered(), "Actual full Depth3D failed to compile/initialize through D3D12; inspect isolated ReShade.log");
      if (sunshine_parity::deterministic)
        require(observed.renders == 0, "Deterministic parity acquired uncontrolled initial stereo history");
      require(observed.runtime->get_device()->get_api() == api::device_api::d3d12, "Actual runtime did not use D3D12");
      find_texture("DoubleTex", exported, width * 2, color == 1 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT);
      if (sunshine_parity::sole_host_warp(observed.runtime, effect_file)) {
        check_sole_host_resources();
      } else {
        find_texture("texzBufferN_P", linear_depth, 0, DXGI_FORMAT_R16G16_FLOAT);
      }
      check_original_only();
      set_int("Depth_Map_View", 0);
      if (uniform("Depth_Map", false).handle) {
      set_int("Depth_Map", 0);
      set_float("Depth_Adjustment", 0.f);
      set_float("Zero_Parallax_Distance", .05f);
      set_float("Depth_Map_Adjust", 40.f);
      set_float("Auto_Depth_Adjust", 0.f);
      set_float("ZPD_Balance", 0.f);
      set_int("ZPD_Boundary", 0);
      set_float("PopOut_Target", 0.f);
      set_int("Range_Boost", 0);
      }
      set_float("Depth_Adjustment", 0.f);
      if (sunshine_depth3d_fixture::sharpening_available(*this)) set_float("Sharpen_Power", 0.f);
      set_int("USE_AA", 0);
      if (uniform("View_Mode", false).handle) set_int("View_Mode", 1);
      if (uniform("Auto_Scaler_Adjust", false).handle) set_int("Auto_Scaler_Adjust", 0);
      if (uniform("AR_Side_Shrink", false).handle) set_float("AR_Side_Shrink", 0);
      for (const auto name : {"Depth_Map_Flip", "Eye_Swap", "DB_AutoFit", "Flip_HV_Scale"}) {
        const auto control = uniform(name, false);
        if (control.handle) {
          observed.runtime->set_uniform_value_bool(control, false);
        }
      }
      const float zero[2] {0, 0}, one[2] {1, 1};
      for (const auto name : {"DLSS_FSR_Offset", "Image_Position_Adjust"}) {
        const auto control = uniform(name, false);
        if (control.handle) {
          observed.runtime->set_uniform_value_float(control, zero, 2);
        }
      }
      for (const auto name : {"Horizontal_and_Vertical", "Horizontal_and_Vertical_TL"}) {
        const auto control = uniform(name, false);
        if (control.handle) {
          observed.runtime->set_uniform_value_float(control, one, 2);
        }
      }
      const unsigned zero_uint[2] {0, 0};
      const auto starting = uniform("Starting_Resolution", false);
      if (starting.handle) {
        observed.runtime->set_uniform_value_uint(starting, zero_uint, 2);
      }
      observed.runtime->enumerate_uniform_variables(effect_file, [](api::effect_runtime *runtime, api::effect_uniform_variable variable) {
        char source[64] {};
        if (runtime->get_annotation_string_from_uniform_variable(variable, "source", source) && std::strcmp(source, "bufready_depth") == 0) {
          observed.ready_uniforms.push_back(variable);
        }
      });
      if (sunshine_parity::sole_host_warp(observed.runtime, effect_file)) {
        require(observed.ready_uniforms.size() == 1 && observed.ready_uniforms.front().handle == uniform("Sunshine_DepthReady").handle,
          "Sole Host warp requires exactly its source-owned depth-readiness uniform");
        uniform("Sunshine_CameraDepthReady");
      } else {
        require(observed.ready_uniforms.size() >= 2, "Original and Sunshine integration depth-readiness uniforms must exist");
      }
      replace_depth(width, height);
      observed.inject = observed.ready = true;
      std::printf("PASS actual D3D12 full effect compiled with export enabled by default; native 2W x H export, color=%u compatibility=%u sole_host=%u\n", color, compatible,
        unsigned(sunshine_parity::sole_host_warp(observed.runtime, effect_file)));
    }

    void replace_depth(unsigned depth_width, unsigned depth_height) {
      // Detach descriptor users before releasing the previous view/resource.
      // The actual runtime waits for its outstanding descriptor users here.
      if (depth_view.handle) {
        observed.runtime->update_texture_bindings("DEPTH", {}, {});
        observed.runtime->get_device()->destroy_resource_view(depth_view);
        depth_view = {};
        observed.bound_depth_view = {};
      }
      texture(depth, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, depth_width, depth_height);
      api::resource_view_desc view(api::format::r32_float, 0, 1, 0, 1);
      require(observed.runtime->get_device()->create_resource_view({reinterpret_cast<std::uint64_t>(depth.p)}, api::resource_usage::shader_resource, view, &depth_view), "Create tracked native ReShade DEPTH view");
      const auto actual_depth = observed.runtime->get_device()->get_resource_from_view(depth_view);
      require(actual_depth.handle == reinterpret_cast<std::uint64_t>(depth.p), "Adapted native DEPTH view resource mismatch");
      observed.depth_view = depth_view;
    }

    void measured_frames() {
      observed.capture = true;
      const auto before = observed.captures;
      for (unsigned i = 0; i < 3; ++i) {
        step();
      }
      observed.capture = false;
      require(observed.captures >= before + 3, "Actual full Depth3D did not run during all measured D3D12 frames");
      require(read(mono.p, D3D12_RESOURCE_STATE_COPY_DEST) == source_bytes, "Actual nonzero-depth effect changed the native mono image");
    }

    float channel(const std::vector<std::uint8_t> &data, unsigned x, unsigned y, unsigned c) const {
      const size_t pixel = size_t(y) * width * 2 + x;
      return color == 1 ? float((reinterpret_cast<const std::uint32_t *>(data.data())[pixel] >> (10 * c)) & 1023) / 1023.f : half_float(reinterpret_cast<const std::uint16_t *>(data.data())[4 * pixel + c]);
    }

    float image_difference(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b) const {
      float maximum = 0;
      for (unsigned y = height / 8; y < height * 7 / 8; y += std::max(1u, height / 360)) {
        for (unsigned x = width / 8; x < width * 7 / 8; x += std::max(1u, width / 640)) {
          for (unsigned eye = 0; eye < 2; ++eye) {
            for (unsigned c = 0; c < 3; ++c) {
              const float av = channel(a, eye * width + x, y, c), bv = channel(b, eye * width + x, y, c);
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
              const float a = channel(flat, sx, y, 0), b = channel(flat, sx + 1, y, 0);
              const float expected = a + (b - a) * blend, actual = channel(stereo, eye * width + x, y, 0);
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
        com_ptr<ID3D12Resource> texture;
        find_texture(name, texture, width, DXGI_FORMAT_R16G16B16A16_FLOAT);
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

    void check_sole_host_resources() {
      for (const auto name : {"Sunshine_WarpMethod", "Compatibility_Power", "Performance_Level", "View_Mode", "View_Mode_Warping",
             "Warping_Masking", "Weapon_Near_Halo_Reduction", "Reconstruction_Size", "De_Artifacting", "Extended_Smoothing"})
        require(!uniform(name, false).handle, "Sole Host warp retains a retired ray control");
      observed.runtime->enumerate_texture_variables(effect_file, [](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char name[256] {};
        runtime->get_texture_variable_name(variable, name);
        for (const auto retired : {"texDMN", "texCN", "texMiniReconBuffer", "texzBufferN_P", "texzBufferN_L", "texzBufferN_M",
               "texzBufferBlurN", "texzBufferBlurEx", "texReconBuffer", "texSmooth", "TAABuffer", "AccBuffer", "Info_Tex", "texAvrN"})
          require(!named(name, retired), "Sole Host warp retains an obsolete depth/history texture");
      });
      bool supported = false;
      require(observed.runtime->get_annotation_bool_from_uniform_variable(uniform("Depth_Adjustment"), "sunshine_host_warp_available", &supported, 1),
        "Sole Host warp availability annotation is missing");
      const bool fallback_test = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_MONO_FALLBACK_TEST");
      require(supported != fallback_test, "Native shader oracle does not match reflected Host warp availability");
      if (supported) {
        for (const auto name : {"SunshineHostCandidate", "SunshineHostVerticalMajorant", "SunshineHostVerticalConditioned", "SunshineHostFinal"}) {
          com_ptr<ID3D12Resource> field;
          find_texture(name, field, width, DXGI_FORMAT_R32_FLOAT);
        }
      } else {
        observed.runtime->enumerate_texture_variables(effect_file, [](api::effect_runtime *runtime, api::effect_texture_variable variable) {
          char name[256] {};
          runtime->get_texture_variable_name(variable, name);
          for (const auto field : {"SunshineHostCandidate", "SunshineHostVerticalMajorant", "SunshineHostVerticalConditioned", "SunshineHostFinal"})
            require(!named(name, field), "Unsupported Host warp still allocated field resources");
        });
      }
      std::printf("PASS sole Host warp capability v2: supported=%u; no retired ray controls or depth/history textures\n", unsigned(supported));
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
      std::puts("PASS compiled effect preserves Sunshine export readiness and excludes the deleted HostSBS iterative warp");
    }

    std::vector<float> linear_samples(const char *plane_label = nullptr) {
      require(linear_depth.p != nullptr, "Legacy preparation sampling requires the original depth pipeline");
      const auto bytes = read(linear_depth.p);
      require(!bytes.empty() && bytes.size() % 4 == 0, "Production linear depth is not packed RG16F");
      const auto desc = linear_depth->GetDesc();
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
      std::vector<float> hardware(size_t(width) * height, .99f);
      upload_depth(hardware);
      set_float("Depth_Adjustment", 0);
      measured_frames();
      const auto flat = read(exported.p);
      require(std::abs(channel(flat, 8, height / 2, 0) - channel(flat, 24, height / 2, 0)) > .05f, "Flat export lost the nonconstant native source");

      float maximum_eye_error = 0;
      unsigned error_x = 0, error_y = 0, error_channel = 0;
      for (unsigned y = 0; y < height; y += std::max(1u, height / 360)) {
        for (unsigned x = 0; x < width; ++x) {
          for (unsigned c = 0; c < 3; ++c) {
            const float a = channel(flat, x, y, c), b = channel(flat, width + x, y, c);
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
      std::printf("MEASURE zero-divergence eye_error=%.9g xy=%u,%u channel=%u left=%.9g right=%.9g\n", maximum_eye_error, error_x, error_y, error_channel, channel(flat, error_x, error_y, error_channel), channel(flat, width + error_x, error_y, error_channel));
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
      upload_depth(hardware);
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
      upload_depth(hardware);
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
            const float left = channel(pixels, x, y, 0), right = channel(pixels, width + x, y, 0);
            require(std::isfinite(left) && std::isfinite(right), "Normal depth diagnostic contains non-finite pixels");
            lo = std::min(lo, left);
            hi = std::max(hi, left);
            eye_error = std::max(eye_error, std::abs(left - right));
            for (unsigned eye = 0; eye < 2; ++eye) {
              for (unsigned component = 1; component < 3; ++component) {
                const float value = channel(pixels, eye * width + x, y, component);
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
              monotonic_error = std::max(monotonic_error, left - channel(pixels, x - 1, y, 0));
            }
            // A smooth monotonic synthetic ramp must not acquire isolated white
            // columns, including row-chunk boundaries and either eye's edges.
            if (x > 2 && x + 3 < width) {
              const float before = channel(pixels, x - 1, y, 0), after = channel(pixels, x + 1, y, 0);
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
      observed.ready = false;
      {
        measured_frames();
        const auto fallback = read(exported.p);
        const float excess = sunshine_color_test::fallback_excess(width, height, color,
          [&](unsigned x, unsigned y, unsigned c) { return channel(flat, x, y, c); },
          [&](unsigned x, unsigned y, unsigned c) { return channel(fallback, x, y, c); });
        std::printf("MEASURE missing-depth versus zero-strength color difference=%.9g quantization_excess=%.9g\n", image_difference(flat, fallback), excess);
        require(excess == 0.f, "Missing DEPTH differs from flat source eyes beyond export quantization");
      }
      std::puts("PASS missing real DEPTH/readiness returns flat duplicate source eyes");
    }

    void check_generic_weapon_policy() {
      const char *generic = std::getenv("SUNSHINE_GAME3D_GENERIC_CONFIG");
      if (!generic || std::strcmp(generic, "1") != 0) {
        return;
      }
      observed.ready = true;
      set_int("Depth_Map", 1);
      set_int("Depth_Map_View", 0);
      set_int("USE_AA", 0);
      set_float("Depth_Map_Adjust", 40);
      set_float("Depth_Adjustment", 35);
      set_float("Zero_Parallax_Distance", .05f);
      set_float("ZPD_OverShoot", 0);
      set_float("Auto_Depth_Adjust", 0);
      const float weapon[] {40.f, 8.f, 0.f, .25f};
      observed.runtime->set_uniform_value_float(uniform("Weapon_Adjust"), weapon, 4);
      std::vector<float> hardware(size_t(width) * height);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          hardware[size_t(y) * width + x] = x < width / 2 ? .02f : .005f;
        }
      }
      upload_depth(hardware);
      set_int("WP", 0);
      measured_frames();
      const auto off = read(exported.p);
      for (int old_profile : {2, 37, -1}) {
        set_int("WP", old_profile);
        measured_frames();
        const auto stale = read(exported.p);
        std::printf("MEASURE generic weapon saved-profile=%d off_difference=%.9g\n", old_profile, image_difference(off, stale));
        require(stale == off, "Unsupported saved weapon profile changed generic output instead of behaving as Off");
        int preserved = 0;
        observed.runtime->get_uniform_value_int(uniform("WP"), &preserved, 1);
        require(preserved == old_profile, "Generic rendering rewrote the user's stored weapon selection");
      }
      set_int("WP", 1);
      measured_frames();
      const float custom_response = image_difference(off, read(exported.p));
      std::printf("MEASURE generic custom weapon response=%.9g\n", custom_response);
      require(custom_response > .003f, "Custom weapon controls are inert in generic mode");
      set_int("WP", 0);
      measured_frames();
      require(read(exported.p) == off, "Disabling custom weapon processing did not restore generic Off pixels");
      std::puts("PASS generic weapon selection: stale numbered/negative values preserve preset and render Off; explicit Custom changes output reversibly");
    }

    void check_host_recovery(bool sole_host) {
      int method = sole_host ? 1 : 0;
      float strength = 0;
      if (!sole_host) observed.runtime->get_uniform_value_int(uniform("Sunshine_WarpMethod"), &method, 1);
      observed.runtime->get_uniform_value_float(uniform("Depth_Adjustment"), &strength, 1);
      require(method == 1 && strength > 0 && observed.ready, "Host recovery test needs calibrated, ready, nonzero Host warp");
      const auto capture = [&](const char *name) {
        measured_frames();
        const auto pixels = read(exported.p);
        require(pixels.size() == size_t(width) * height * 2 * bytes_per_pixel(exported->GetDesc().Format), "Host recovery export size differs");
        if (color != 1) {
          for (size_t i = 0; i < pixels.size(); i += 2) {
            std::uint16_t value;
            std::memcpy(&value, pixels.data() + i, sizeof(value));
            require(std::isfinite(half_float(value)), "Host recovery contains nonfinite RGBA");
          }
        }
        sunshine_parity::write_bytes(sunshine_parity::output_directory / (std::string("recovery-") + name + ".export.bin"), pixels.data(), pixels.size());
        return pixels;
      };
      set_int("Depth_Map_View", 0);
      const auto host = capture("host");
      set_int("Depth_Map_View", 2);
      const auto normal = capture("normal-depth");
      set_float("Depth_Adjustment", 0);
      require(capture("normal-depth-zero-strength") == normal, "Normal Depth diagnostic depends on strength");
      set_int("Depth_Map_View", 0);
      const auto flat = capture("zero-strength");
      require(image_difference(host, flat) > .002f, "Host warp strength has no visible effect");
      set_float("Depth_Adjustment", strength);
      observed.ready = false;
      require(capture("depth-lost") == flat, "Lost depth differs from the zero-strength current-color endpoint");
      observed.ready = true;
      require(capture("depth-recovered") == host, "Depth recovery did not exactly restore current Host frame");
      set_int("Depth_Map_View", 1);
      capture("stereo-depth");
      set_int("Depth_Map_View", 0);
      require(capture("diagnostic-return") == host, "Leaving depth diagnostics changed the current Host frame");
      std::puts("PASS Host warp diagnostic strength independence, exact zero/depth-loss endpoint, readiness recovery and diagnostic return; native mono retained");
    }

    void check_native_shader_oracle() {
      require(sunshine_parity::sole_host_warp(observed.runtime, effect_file), "Native shader oracle requires capability v2");
      const bool fallback = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_MONO_FALLBACK_TEST");
      require(!native_oracle_output.empty(), "Native shader oracle output missing");
      std::ofstream report(native_oracle_output / "measurements.txt");
      set_int("Depth_Map_View", 0);
      set_int("Sunshine_CameraCoordinateBasis", 1);
      set_float("Sunshine_CameraDepthScale", 128);
      set_float("Sunshine_CameraStrengthBlend", 1);
      const float projection[] {0, 1}, convergence[] {.05f, 1.f / 128}, rect[] {0, 0, 1, 1}, range[] {0, 1};
      observed.runtime->set_uniform_value_float(uniform("Sunshine_CameraProjection"), projection, 2);
      observed.runtime->set_uniform_value_float(uniform("Sunshine_CameraConvergence"), convergence, 2);
      observed.runtime->set_uniform_value_float(uniform("Sunshine_CameraDepthRect"), rect, 4);
      observed.runtime->set_uniform_value_float(uniform("Sunshine_CameraRawDepthRange"), range, 2);
      observed.runtime->set_uniform_value_bool(uniform("Sunshine_CameraDepthReady"), true);
      upload_depth(std::vector<float>(size_t(width) * height, .02f));
      observed.ready = true;
      set_float("Depth_Adjustment", fallback ? 100 : 50);
      if (sunshine_depth3d_fixture::sharpening_available(*this)) set_float("Sharpen_Power", 0);
      const auto capture = [&](const char *name) {
        measured_frames();
        const auto pixels = read(exported.p);
        sunshine_parity::write_bytes(native_oracle_output / (std::string(name) + ".export.bin"), pixels.data(), pixels.size());
        if (color != 1) for (size_t i = 0; i < pixels.size(); i += 2) {
          std::uint16_t component;
          std::memcpy(&component, pixels.data() + i, 2);
          require(std::isfinite(half_float(component)), "Native shader oracle has nonfinite RGBA");
        }
        return pixels;
      };
      const auto base = capture("base");
      const auto source_error = [&](const std::vector<std::uint8_t> &pixels) {
        double maximum = 0;
        for (unsigned y = 0; y < height; y += std::max(1u, height / 360)) for (unsigned x = 0; x < width; ++x) {
          const size_t pixel = size_t(y) * width + x;
          sunshine_depth3d_color::rgb expected {};
          for (unsigned c = 0; c < 3; ++c) {
            if (color == 1) expected[c] = source_bytes[pixel * 4 + c] / 255.0;
            else if (color == 2) {
              std::uint16_t component;
              std::memcpy(&component, source_bytes.data() + pixel * 8 + c * 2, 2);
              expected[c] = half_float(component);
            } else {
              std::uint32_t packed;
              std::memcpy(&packed, source_bytes.data() + pixel * 4, 4);
              expected[c] = ((packed >> (c * 10)) & 1023) / 1023.0;
            }
          }
          if (color == 3) expected = sunshine_depth3d_color::decoded(expected, 3);
          for (unsigned eye = 0; eye < 2; ++eye) for (unsigned c = 0; c < 3; ++c) {
            const double actual = channel(pixels, eye * width + x, y, c);
            maximum = std::max(maximum, std::abs(actual - expected[c]) / std::max(1.0, std::abs(expected[c])));
          }
        }
        return maximum;
      };
      if (fallback) {
        const double maximum = source_error(base);
        require(maximum < .004, "Unsupported warp resolution changed source color instead of showing mono");
        for (int view : {0, 1, 2}) {
          set_int("Depth_Map_View", view);
          require(capture(("view" + std::to_string(view)).c_str()) == base, "Unsupported warp exposed a stale depth diagnostic");
        }
        set_float("Depth_Adjustment", 0);
        require(capture("zero") == base, "Unsupported warp mono depends on strength");
        observed.ready = false;
        require(capture("depth-lost") == base, "Unsupported warp mono depends on depth readiness");
        report << "mono_fallback=true dimensions=" << width << 'x' << height << " maximum_relative_color_error=" << maximum << '\n';
        std::puts("PASS unsupported warp size: annotated unavailable, no field allocation, correct mono SDR/HDR and invariant diagnostic/strength/readiness states");
      } else {
        set_float("Depth_Adjustment", 0);
        const auto flat = capture("zero-strength");
        const double maximum = source_error(flat);
        require(maximum < .004, "Native packing changed source color at zero strength");
        const float difference = image_difference(base, flat);
        require(difference > .0001f, "Native packing oracle did not exercise stereo rendering");
        set_float("Depth_Adjustment", 50);
        observed.ready = false;
        require(capture("depth-lost") == flat, "Lost depth changed flat packed source color");
        observed.ready = true;
        require(capture("recovered") == base, "Depth recovery did not exactly restore stereo output");
        report << "packing=true dimensions=" << width << 'x' << height << " color=" << color
          << " maximum_relative_color_error=" << maximum << " stereo_response=" << difference
          << " restored_exact=true legacy_sharpening=" << sunshine_depth3d_fixture::expects_sharpening() << '\n';
        std::puts("PASS native packing: finite SDR/HDR, direct source color at zero strength, visible stereo, exact depth recovery and untouched native game image");
      }
      require(report.good(), "Cannot write native shader oracle measurements");
    }

    void run() {
      discover();
      std::printf("MEASURE sharpening contract available=%u\n", unsigned(sunshine_depth3d_fixture::sharpening_available(*this)));
      const bool sole_host = sunshine_parity::sole_host_warp(observed.runtime, effect_file);
      if (const char *method = std::getenv("SUNSHINE_GAME3D_WARP_METHOD"); method && *method) {
        require(!std::strcmp(method, "0") || !std::strcmp(method, "1"), "Warp method test must be 0 or 1");
        if (sole_host) {
          require(method[0] == '1', "Sole Host warp cannot select the retired ray renderer");
          std::puts("MEASURE Host method 1 selected by shader capability v2 (no legacy selector)");
        } else {
          set_int("Sunshine_WarpMethod", method[0] - '0');
        }
      }
      std::printf("MEASURE final-AA contract available=%u legacy_opt_in=%s\n", unsigned(final_aa_available()),
        std::getenv("SUNSHINE_GAME3D_LEGACY_FINAL_AA") ? std::getenv("SUNSHINE_GAME3D_LEGACY_FINAL_AA") : "0");
      if (!native_oracle_output.empty()) { check_native_shader_oracle(); return; }
      if (sunshine_jitter_fixture::requested()) {
        sunshine_jitter_fixture::run(*this, width, height, observed.runtime, observed.ready);
        return;
      }
      if (sunshine_camera_fixture::requested()) {
        sunshine_camera_fixture::run(*this, width, height, observed.runtime, observed.ready);
        return;
      }
      if (sunshine_depth3d_color::requested()) {
        sunshine_depth3d_color::run(*this, width, height, observed.runtime, effect_file);
        return;
      }
      if (sunshine_parity::requested()) {
        sunshine_parity::run(*this, width, height, false, observed.runtime, observed.ready, observed.capture, effect_file);
        if (const char *toggle = std::getenv("SUNSHINE_GAME3D_WARP_SWITCH_TEST"); toggle && !std::strcmp(toggle, "1")) {
          require(!sole_host, "Warp switching is unsupported after removing the legacy renderer; use HOST_RECOVERY_TEST");
          const auto capture = [&] { measured_frames(); return read(exported.p); };
          int method = 0;
          float strength = 0;
          observed.runtime->get_uniform_value_int(uniform("Sunshine_WarpMethod"), &method, 1);
          observed.runtime->get_uniform_value_float(uniform("Depth_Adjustment"), &strength, 1);
          require(method == 1 && strength > 0 && observed.ready, "Switch test needs active Host warp");
          const auto host = capture();
          set_int("Sunshine_WarpMethod", 0);
          const auto legacy = capture();
          require(image_difference(host, legacy) > .002f, "Live warp switch did not change edge rendering");
          set_int("Sunshine_WarpMethod", 1);
          require(capture() == host, "Returning to Host warp did not restore its exact current frame");
          set_float("Depth_Adjustment", 0);
          const auto flat = capture();
          require(image_difference(host, flat) > .002f, "Host warp strength has no visible effect");
          set_float("Depth_Adjustment", strength);
          observed.ready = false;
          require(image_difference(capture(), flat) < .002f, "Lost depth did not return flat current color");
          observed.ready = true;
          require(capture() == host, "Depth recovery did not restore the current Host frame");
          std::puts("PASS live warp switch, unchanged strength, exact return, zero strength, depth loss and recovery");
        }
        if (const char *recovery = std::getenv("SUNSHINE_GAME3D_HOST_RECOVERY_TEST"); recovery && *recovery) {
          require(!std::strcmp(recovery, "0") || !std::strcmp(recovery, "1"), "HOST_RECOVERY_TEST must be 0 or 1");
          if (!std::strcmp(recovery, "1")) check_host_recovery(sole_host);
        }
        return;
      }
      require(!sole_host, "Sole Host warp requires the parity, color, camera or jitter fixture; legacy preparation tests are inapplicable");
      check_original_pipeline();
      check_generic_weapon_policy();
      std::printf("PASS actual official ReShade6.8 D3D12 full %s shared pipeline source=%u compatibility=%u dimensions=%ux%u; %u natural technique executions. No game-specific depth-selection or performance claim.\n", effect_file, color, compatible, width, height, observed.renders);
    }
  };
}  // namespace

#ifndef SUNSHINE_DEPTH_SELECTION_RUNTIME
int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 6 && argc != 8) {
    std::fprintf(stderr, "usage: test_depth3d_runtime_d3d12 <official-ReShade64.dll> <external-Depth3D-Shaders> <isolated-ignored-output-directory> <srgb|scrgb|pq> <HDR_Compatible_Mode:0|1> [source-width source-height]\n");
    return 2;
  }
  std::thread([] {
    Sleep(180000);
    std::fputs("FAIL actual D3D12 full Depth3D fixture watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    const unsigned color = std::strcmp(argv[4], "srgb") == 0 ? 1 : std::strcmp(argv[4], "scrgb") == 0 ? 2 :
                                                                 std::strcmp(argv[4], "pq") == 0      ? 3 :
                                                                                                        0;
    require(color && (std::strcmp(argv[5], "0") == 0 || std::strcmp(argv[5], "1") == 0), "Invalid fixture color or compatibility argument");
    if (argc == 8) {
      size_t used_width = 0, used_height = 0;
      const auto requested_width = std::stoul(argv[6], &used_width);
      const auto requested_height = std::stoul(argv[7], &used_height);
      const bool fallback = sunshine_camera_fixture::flag("SUNSHINE_GAME3D_MONO_FALLBACK_TEST");
      require(used_width == std::strlen(argv[6]) && used_height == std::strlen(argv[7]) && requested_width >= 640 && requested_height >= 360 &&
        requested_width <= (fallback ? 4800 : 3840) && requested_height <= (fallback ? 2700 : 2160) && requested_width % 2 == 0 && requested_height % 2 == 0,
        "Fixture dimensions must be even and <=3840x2160; explicit mono-fallback oracle permits <=4800x2700");
      require(!fallback || requested_width > 3840, "Mono fallback oracle requires an unsupported Host warp width");
      width = static_cast<unsigned>(requested_width);
      height = static_cast<unsigned>(requested_height);
    }
    fixture_t fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fs::absolute(argv[3]), color, unsigned(argv[5][0] - '0'));
    fixture.run();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
#endif
