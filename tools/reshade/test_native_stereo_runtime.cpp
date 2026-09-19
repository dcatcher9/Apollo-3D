// SPDX-License-Identifier: GPL-3.0-only
// Opt-in actual ReShade 6.8 D3D12 independent stereo fixture. Loads only
// the independent Sunshine shader into an isolated personal probe directory.
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
#include <limits>
#include <reshade.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>
#include "test_stereo_parity.h"

namespace {
  namespace api = reshade::api;
  namespace fs = std::filesystem;
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
    ID3D12QueryHeap *timing_queries = nullptr;
    ID3D12Resource *timing_readback = nullptr;
    bool timing_armed = false, timing_started = false, timing_finished = false, timing_invalid = false;
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

  void on_begin_effects(api::effect_runtime *runtime, api::command_list *commands, api::resource_view, api::resource_view) {
    if (!observed.inject) {
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
    if (observed.timing_armed) {
      if (observed.timing_started || !observed.timing_queries || !observed.timing_readback) {
        observed.timing_invalid = true;
        return;
      }
      auto *native = reinterpret_cast<ID3D12GraphicsCommandList *>(commands->get_native());
      native->EndQuery(observed.timing_queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
      observed.timing_started = true;
    }
  }

  void on_technique(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *commands, api::resource_view rtv, api::resource_view) {
    char name[256] {};
    runtime->get_technique_name(technique, name);
    if (!named(name, "SunshineDepth3D")) {
      return;
    }
    ++observed.renders;
    sunshine_parity::end(commands);
    // ReShade 6.8 documents this event as AFTER the named technique rendered.
    // Stop before the fixture's optional mono copy or ReShade overlay work.
    if (observed.timing_armed) {
      if (!observed.timing_started || observed.timing_finished) {
        observed.timing_invalid = true;
      } else {
        auto *native = reinterpret_cast<ID3D12GraphicsCommandList *>(commands->get_native());
        native->EndQuery(observed.timing_queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        native->ResolveQueryData(observed.timing_queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, observed.timing_readback, 0);
        observed.timing_finished = true;
      }
    }
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
    com_ptr<ID3D12Resource> backbuffers[2], mono, source_upload, depth, exported;
    com_ptr<ID3D12Fence> completion;
    HANDLE completion_event = nullptr;
    std::uint64_t fence_value = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT source_footprint {};
    api::resource_view depth_view {};
    unsigned color = 0, performance_mode = 0;
    DXGI_FORMAT source_format = DXGI_FORMAT_UNKNOWN;
    std::vector<std::uint8_t> source_bytes;
    std::function<void()> render_tracked_depth;

    ~fixture_t() {
      observed.capture = observed.inject = false;
      observed.timing_armed = false;
      observed.timing_queries = nullptr;
      observed.timing_readback = nullptr;
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

    void texture(com_ptr<ID3D12Resource> &resource, DXGI_FORMAT format, D3D12_RESOURCE_STATES state) {
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width;
      desc.Height = height;
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
      require(values.size() == size_t(width) * height, "Wrong synthetic hardware-depth size");
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

    void initialize(const fs::path &runtime, const fs::path &shader_root, const fs::path &directory, unsigned source_color, unsigned requested_performance_mode, const fs::path &depth_addon = {}, bool automatic_depth = true) {
      color = source_color;
      performance_mode = requested_performance_mode;
      require(fs::is_regular_file(shader_root / "SunshineDepth3D.fx"), "Missing packaged SunshineDepth3D shader");
      fs::create_directories(directory / "effects");
      fs::create_directories(directory / "addons");
      for (const auto &entry : fs::directory_iterator(directory / "addons")) {
        const auto extension = entry.path().extension();
        if (extension == ".addon" || extension == ".addon64")
          require(!depth_addon.empty() && entry.path().filename() == "SunshineSBS.addon64", "Fixture addon directory contains a legacy or unrelated addon; use a fresh isolated output");
      }
      if (!depth_addon.empty())
        fs::copy_file(depth_addon, directory / "addons" / "SunshineSBS.addon64", fs::copy_options::overwrite_existing);
      for (const auto &entry : fs::recursive_directory_iterator(shader_root)) {
        if (!entry.is_regular_file() || (entry.path().extension() != ".fxh" && entry.path() != shader_root / "SunshineDepth3D.fx")) {
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
          config << "DisabledAddons=Generic Depth\n[DEPTH]\nDepthCopyBeforeClears=1\nUseAspectRatioHeuristics=1\n"
                    "[SUNSHINE_DEPTH]\nAutoSelectSceneDepth=" << (automatic_depth ? 1 : 0) << "\n";
        config << "[GENERAL]\nEffectSearchPaths=.\\effects\nPresetPath=.\\preset.ini\n"
                  "PerformanceMode=" << performance_mode << "\nSkipLoadingDisabledEffects=0\nIntermediateCachePath=.\\cache\n"
                  "[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n";
        std::ofstream preset(directory / "preset.ini");
        preset << "Techniques=SunshineDepth3D@SunshineDepth3D.fx\nTechniqueSorting=SunshineDepth3D@SunshineDepth3D.fx\n";
        require(config.good() && preset.good(), "Could not write isolated fixture configuration");
      }
      sunshine_parity::provenance(runtime, shader_root, directory);
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
      require(unsigned(observed.color) == color, "Actual runtime transfer differs from the native game swapchain");
      Sleep(3);
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
        throw std::runtime_error(std::string("Missing actual effect uniform ") + name);
      }
      return result;
    }

    void set_int(const char *name, int value) {
      observed.runtime->set_uniform_value_int(uniform(name), &value, 1);
    }

    void set_float(const char *name, float value) {
      observed.runtime->set_uniform_value_float(uniform(name), &value, 1);
    }

    void find_texture(const char *name, com_ptr<ID3D12Resource> &resource, UINT expected_width, DXGI_FORMAT expected_format) {
      observed.runtime->enumerate_texture_variables("SunshineDepth3D.fx", [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
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
      UINT64 total = 0;
      game->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
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
      const size_t row = size_t(desc.Width) * bytes_per_pixel(desc.Format);
      std::vector<std::uint8_t> bytes(row * desc.Height);
      for (UINT y = 0; y < desc.Height; ++y) {
        std::memcpy(bytes.data() + row * y, static_cast<const std::uint8_t *>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, row);
      }
      const D3D12_RANGE no_write {0, 0};
      staging->Unmap(0, &no_write);
      return bytes;
    }

    void discover() {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
      while (!observed.renders && std::chrono::steady_clock::now() < deadline) step();
      require(observed.runtime && observed.renders, "Independent shader did not compile/execute; inspect ReShade.log");
      find_texture("DoubleTex", exported, width * 2,
        color == 1 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT);
      texture(depth, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      api::resource_view_desc view(api::format::r32_float, 0, 1, 0, 1);
      require(observed.runtime->get_device()->create_resource_view({reinterpret_cast<std::uint64_t>(depth.p)}, api::resource_usage::shader_resource, view, &depth_view), "Create native depth view");
      observed.depth_view = depth_view;
      observed.ready_uniforms.push_back(uniform("Sunshine_DepthReady"));
      observed.ready_uniforms.push_back(uniform("Sunshine_Calibrated"));
      observed.inject = observed.ready = true;
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      const float rect[] {1, 1, 0, 0};
      observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), rect, 4);
      if (!performance_mode) {
        set_bool("EdgeAntialias", false);
        set_bool("DepthView", false);
      }
      std::puts("PASS independent shader compiled; full SBS/HDR annotations verified");
    }

    void set_bool(const char *name, bool value) {
      observed.runtime->set_uniform_value_bool(uniform(name), value);
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
      return color == 1 ? float((reinterpret_cast<const std::uint32_t *>(data.data())[pixel] >> (10 * c)) & (c == 3 ? 3u : 1023u)) / (c == 3 ? 3.f : 1023.f) : half_float(reinterpret_cast<const std::uint16_t *>(data.data())[4 * pixel + c]);
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
        std::printf("MEASURE native shift setting=%s eye=%u source_pixels=%.3f normalized_error=%.9g\n", setting, eye, best_shift, best_error);
        require(std::abs(best_shift) < limit - 1, "Native displacement fit reached its search boundary");
        require(best_error < .015, "Actual export does not correlate with a translated constant-plane source");
        result[eye] = best_shift;
      }
      return result;
    }



    void check_invalid_calibration(const std::vector<std::uint8_t> &flat, std::vector<float> &hardware) {
      std::fill(hardware.begin(), hardware.end(), .75f);
      upload_depth(hardware);
      observed.ready = true;
      set_float("Strength", 1.f);
      set_float("ScreenPlane", 0.f);
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      set_bool("DepthView", false);
      set_bool("EdgeAntialias", false);
      const std::array<float, 4> full_rect {1, 1, 0, 0};
      auto set_rect = [&](const std::array<float, 4> &rect) {
        observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), rect.data(), 4);
      };
      set_rect(full_rect);
      measured_frames();
      const auto valid = read(exported.p);
      require(image_difference(valid, flat) > .01f, "Invalid-calibration fixture never established nonzero stereo");
      auto expect_flat = [&](const char *setting) {
        measured_frames();
        if (read(exported.p) != flat) {
          throw std::runtime_error(std::string("Invalid calibration retained stereo or changed mono: ") + setting);
        }
      };
      const float nan = std::numeric_limits<float>::quiet_NaN();
      for (const float gain : {0.f, nan, std::numeric_limits<float>::infinity()}) {
        set_float("Sunshine_RawGain", gain);
        expect_flat("zero/nonfinite gain");
      }
      set_float("Sunshine_RawGain", 2.f);
      set_float("Sunshine_RawAnchor", nan);
      expect_flat("nonfinite anchor");
      set_float("Sunshine_RawAnchor", .5f);
      for (const auto &rect : {
             std::array<float, 4> {0, 1, 0, 0},
             std::array<float, 4> {1, -1, 0, 0},
             std::array<float, 4> {.5f, .5f, -.1f, 0},
             std::array<float, 4> {.75f, .5f, .5f, 0},
             std::array<float, 4> {nan, 1, 0, 0},
             std::array<float, 4> {.5f, .5f, 0, std::numeric_limits<float>::infinity()}}) {
        set_rect(rect);
        expect_flat("empty/negative/out-of-bounds/nonfinite depth rectangle");
      }
      set_rect(full_rect);
      measured_frames();
      require(read(exported.p) == valid, "Valid calibration did not recover after invalid dynamic fields");
      std::puts("PASS zero/nonfinite gain, nonfinite anchor and invalid depth rectangles fail flat and recover");
    }

    void check_padded_depth(std::vector<float> &hardware) {
      // The lower-right quarter is an active half-resolution viewport. Poison
      // every other texel, including the immediately adjacent top/left border.
      // Use the entire output, not an interior crop, to exercise all UV edges.
      const unsigned active_width = width / 2, active_height = height / 2;
      const unsigned left = width - active_width, top = height - active_height;
      const std::array<float, 4> full_rect {1, 1, 0, 0}, padded_rect {.5f, .5f, .5f, .5f};
      auto set_rect = [&](const std::array<float, 4> &rect) {
        observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), rect.data(), 4);
      };
      set_rect(full_rect);
      std::fill(hardware.begin(), hardware.end(), .75f);
      upload_depth(hardware);
      measured_frames();
      const auto full_plane = read(exported.p);
      std::fill(hardware.begin(), hardware.end(), std::numeric_limits<float>::quiet_NaN());
      auto active_value = [&](unsigned x, unsigned y) {
        return .2f + .3f * float(x) / float(active_width - 1) + .2f * float(y) / float(active_height - 1);
      };
      for (unsigned y = 0; y < active_height; ++y)
        for (unsigned x = 0; x < active_width; ++x)
          hardware[size_t(y + top) * width + x + left] = active_value(x, y);
      upload_depth(hardware);
      set_rect(padded_rect);
      set_bool("DepthView", true);
      measured_frames();
      const auto padded = read(exported.p);
      float maximum_error = 0;
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          // Each active raw texel covers exactly 2x2 output texels. With anchor
          // .5 and gain 2 the normalized diagnostic equals the raw input value.
          const float expected = active_value(x / 2, y / 2);
          for (unsigned eye = 0; eye < 2; ++eye)
            for (unsigned c = 0; c < 3; ++c) {
              const float actual = channel(padded, eye * width + x, y, c);
              require(std::isfinite(actual), "Padded depth diagnostic contains nonfinite color");
              maximum_error = std::max(maximum_error, std::abs(actual - expected));
            }
        }
      }
      std::printf("MEASURE padded half-resolution depth full-frame maximum error=%.9g\n", maximum_error);
      require(maximum_error < .002f, "Active depth rectangle sampled poisoned padding or mapped the wrong raw texel");
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
          if (x < left || y < top) hardware[size_t(y) * width + x] = 1.f;
      upload_depth(hardware);
      measured_frames();
      require(read(exported.p) == padded, "Changing inactive depth padding changed the output");
      for (unsigned y = top; y < height; ++y)
        for (unsigned x = left; x < width; ++x)
          hardware[size_t(y) * width + x] = .75f;
      upload_depth(hardware);
      set_bool("DepthView", false);
      measured_frames();
      require(read(exported.p) == full_plane, "Padded half-resolution constant-plane geometry differs from native depth");
      set_rect(full_rect);
      std::fill(hardware.begin(), hardware.end(), .75f);
      upload_depth(hardware);
      std::puts("PASS poisoned padded depth viewport, per-pixel diagnostic mapping and native stereo equivalence");
    }

    void check_surface_ownership(std::vector<float> &hardware) {
      // A red foreground half-plane overlaps blue background in the left eye
      // and reveals a source-less gap in the right. Integer pixel displacement
      // keeps this ownership oracle independent of filtering/AA tolerances.
      const auto saved_source = source_bytes;
      const unsigned boundary = width / 2;
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const bool foreground = x < boundary;
          const size_t pixel = size_t(y) * width + x;
          hardware[pixel] = foreground ? .75f : .25f;
          if (color == 1) {
            const std::uint8_t values[] {std::uint8_t(foreground ? 224 : 32), 32, std::uint8_t(foreground ? 32 : 224), 255};
            std::memcpy(source_bytes.data() + pixel * 4, values, 4);
          } else if (color == 2) {
            const std::uint16_t values[] {std::uint16_t(foreground ? 0x3c00 : 0x3400), 0x3400, std::uint16_t(foreground ? 0x3400 : 0x3c00), 0x3c00};
            std::memcpy(source_bytes.data() + pixel * 8, values, 8);
          } else {
            const std::uint32_t value = (foreground ? 600u : 256u) | (256u << 10) | ((foreground ? 256u : 600u) << 20) | (3u << 30);
            std::memcpy(source_bytes.data() + pixel * 4, &value, 4);
          }
        }
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      upload_depth(hardware);
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      set_bool("EdgeAntialias", false);
      set_float("Strength", 0.f);
      measured_frames();
      const auto flat = read(exported.p);
      const unsigned shift = std::max(2u, unsigned(std::floor(.0025f * width)));
      set_float("Strength", float(shift) / (.0025f * width));
      measured_frames();
      const auto stereo = read(exported.p);
      size_t overlap_samples = 0, gap_samples = 0;
      for (const unsigned y : {height / 4, height / 2, height * 3 / 4}) {
        for (unsigned x = boundary - shift + 1; x + 1 < boundary + shift; ++x) {
          for (unsigned c = 0; c < 3; ++c) {
            const float foreground = channel(flat, boundary / 2, y, c);
            const float background = channel(flat, boundary + boundary / 2, y, c);
            require(std::abs(channel(stereo, x, y, c) - foreground) <= .003f + .002f * std::abs(foreground), "Background defeated foreground at an overlapping depth edge");
            require(std::abs(channel(stereo, width + x, y, c) - background) <= .003f + .002f * std::abs(background), "Disocclusion gap fill leaked foreground color");
          }
          require(channel(stereo, x, y, 3) == 1.f, "Observed overlapping foreground lost confidence");
          require(channel(stereo, width + x, y, 3) == 0.f, "Filled disocclusion was marked as observed source content");
          ++overlap_samples;
          ++gap_samples;
        }
      }
      require(overlap_samples && gap_samples, "Ownership fixture did not inspect overlapping and uncovered pixels");
      source_bytes = saved_source;
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      std::printf("PASS foreground overlap ownership and background-only confidence-zero gap fill (%zu samples each)\n", gap_samples);
    }

    void check_mixed_edge_gap_fill(std::vector<float> &hardware) {
      // Upscaled/TAA color can retain foreground contribution in a background
      // depth texel. Repeating that boundary color across an entire disocclusion
      // creates a horizontal fringe. The missing area's known background is
      // constant here, so its interior has an independent, exact color oracle.
      const auto saved_source = source_bytes;
      // Even source coordinates align the half-resolution texels exactly; their
      // storage is offset into the lower-right quarter of the padded texture.
      const unsigned boundary = 2 * (width / 4), thin_column = 2 * (3 * width / 8);
      const unsigned shift = std::max(3u, unsigned(std::floor(.0025f * width)));
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      set_float("ScreenPlane", 0.f);
      set_bool("DepthView", false);

      struct edge_case {
        bool mirrored, half_depth;
        unsigned fringe_width;
        bool closer_donor = false;
      };

      for (const auto scenario : {
             edge_case {false, false, 1},
             edge_case {false, false, 2},
             edge_case {true, false, 1},
             edge_case {true, false, 2},
             edge_case {false, true, 3},
             edge_case {true, true, 3},
             edge_case {false, false, 0, true}
           }) {
        const unsigned thin_width = scenario.half_depth ? 2u : 1u;
        const unsigned depth_width = scenario.half_depth ? width / 2 : width;
        const unsigned depth_height = scenario.half_depth ? height / 2 : height;
        const unsigned depth_left = width - depth_width, depth_top = height - depth_height;
        const std::array<float, 4> rect {float(depth_width) / width, float(depth_height) / height, float(depth_left) / width, float(depth_top) / height};
        observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), rect.data(), 4);
        const auto foreground_at = [&](unsigned x) {
          return x < boundary || (x >= thin_column && x < thin_column + thin_width);
        };
        const auto closer_donor_at = [&](unsigned x) {
          return scenario.closer_donor && x == boundary + 2;
        };
        std::fill(hardware.begin(), hardware.end(), std::numeric_limits<float>::quiet_NaN());
        for (unsigned y = 0; y < depth_height; ++y) {
          for (unsigned x = 0; x < depth_width; ++x) {
            const unsigned canonical_x = scenario.mirrored ? depth_width - 1 - x : x;
            const unsigned native_x = scenario.half_depth ? canonical_x * 2 : canonical_x;
            // The donor trap is only 0.8 disparity pixels nearer than background
            // at this strength. A symmetric +/-1-pixel depth tolerance would
            // accept its bright color as a background donor; clean background
            // immediately before it is available without crossing the strip.
            const float raw = foreground_at(native_x) ? .75f : closer_donor_at(native_x) ? .25f + .4f / (2.f * shift) :
                                                                                           .25f;
            hardware[size_t(y + depth_top) * width + x + depth_left] = raw;
          }
        }
        for (unsigned y = 0; y < height; ++y) {
          for (unsigned x = 0; x < width; ++x) {
            const unsigned canonical_x = scenario.mirrored ? width - 1 - x : x;
            const bool foreground = foreground_at(canonical_x) || closer_donor_at(canonical_x);
            const bool mixed = !foreground && canonical_x >= boundary && canonical_x < boundary + scenario.fringe_width;
            const size_t pixel = size_t(y) * width + x;
            if (color == 1) {
              const std::uint8_t values[] {std::uint8_t(foreground ? 224 : mixed ? 128 :
                                                                                   32),
                                           32,
                                           std::uint8_t(foreground ? 32 : mixed ? 128 :
                                                                                  224),
                                           255};
              std::memcpy(source_bytes.data() + pixel * 4, values, 4);
            } else if (color == 2) {
              const std::uint16_t values[] {std::uint16_t(foreground ? 0x3c00 : mixed ? 0x3900 :
                                                                                        0x3400),
                                            0x3400,
                                            std::uint16_t(foreground ? 0x3400 : mixed ? 0x3900 :
                                                                                        0x3c00),
                                            0x3c00};
              std::memcpy(source_bytes.data() + pixel * 8, values, 8);
            } else {
              const std::uint32_t value = (foreground ? 600u : mixed ? 428u :
                                                                       256u) |
                                          (256u << 10) |
                                          ((foreground ? 256u : mixed ? 428u :
                                                                        600u)
                                           << 20) |
                                          (3u << 30);
              std::memcpy(source_bytes.data() + pixel * 4, &value, 4);
            }
          }
        }
        fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
        upload_depth(hardware);
        if (scenario.half_depth) {
          // Verify the actual padded semantic mapping independently of the gap
          // oracle. One selected raw texel covers exactly two native columns,
          // including the two-column foreground object and mirrored boundary.
          set_bool("DepthView", true);
          measured_frames();
          const auto diagnostic = read(exported.p);
          for (const unsigned y : {height / 4, height / 2, height * 3 / 4}) {
            for (unsigned x = 0; x < width; ++x) {
              const unsigned canonical_x = scenario.mirrored ? width - 1 - x : x;
              const float expected = foreground_at(canonical_x) ? .75f : .25f;
              for (unsigned eye = 0; eye < 2; ++eye) {
                require(std::abs(channel(diagnostic, eye * width + x, y, 0) - expected) < .002f, "Mixed-edge reduced DEPTH does not map its padded viewport to the expected native silhouette");
              }
            }
          }
          set_bool("DepthView", false);
        }
        set_float("Strength", 0.f);
        measured_frames();
        const auto flat = read(exported.p);
        set_float("Strength", float(shift) / (.0025f * width));
        for (const bool antialias : {false, true}) {
          set_bool("EdgeAntialias", antialias);
          measured_frames();
          const auto stereo = read(exported.p);
          float worst_gap_error = 0;
          size_t gap_samples = 0;
          const unsigned projected_boundary = scenario.mirrored ? width - boundary : boundary;
          const unsigned gap_eye = scenario.mirrored ? 0u : 1u;
          const unsigned background_reference = scenario.mirrored ? width - 1 - (boundary + width / 8) : boundary + width / 8;
          const unsigned thin_reference = scenario.mirrored ? width - 1 - thin_column : thin_column;
          for (const unsigned y : {height / 4, height / 2, height * 3 / 4}) {
            // Stay two output pixels away from both projected silhouettes. This
            // checks background reconstruction, not removal of the original TAA
            // fringe at an observed surface, and permits local edge footprints.
            for (unsigned x = projected_boundary - shift + 2; x + 2 < projected_boundary + shift; ++x) {
              for (unsigned c = 0; c < 3; ++c) {
                const float background = channel(flat, background_reference, y, c);
                const float actual = channel(stereo, gap_eye * width + x, y, c);
                require(std::isfinite(actual), "Mixed-edge gap contains nonfinite color");
                const float normalized_error = std::abs(actual - background) / (1.f + std::abs(background));
                worst_gap_error = std::max(worst_gap_error, normalized_error);
              }
              require(channel(stereo, gap_eye * width + x, y, 3) == 0.f, "Mixed-edge gap fill was marked as observed source content");
              ++gap_samples;
            }
            // Keep the native one-column oracle. Half-resolution cases instead
            // contain one real depth texel spanning two native color columns.
            // Broad erosion/blur cannot pass by flattening either true object.
            const float background = channel(flat, background_reference, y, 0);
            const float contrast = channel(flat, thin_reference, y, 0) - background;
            require(contrast > .05f, "Mixed-edge fixture has no foreground contrast");
            for (unsigned eye = 0; eye < 2; ++eye) {
              const float canonical_center = float(thin_column) + .5f * float(thin_width - 1);
              const float expected = (scenario.mirrored ? float(width - 1) - canonical_center : canonical_center) +
                                     (eye ? -float(shift) : float(shift));
              float mass = 0, weighted_x = 0, peak = 0;
              for (int x = int(std::floor(expected)) - 3; x <= int(std::ceil(expected)) + 3; ++x) {
                const float actual = channel(stereo, eye * width + unsigned(x), y, 0);
                require(std::isfinite(actual), "Mixed-edge thin foreground contains nonfinite color");
                const float weight = std::max(0.f, (actual - background) / contrast);
                mass += weight;
                weighted_x += weight * x;
                peak = std::max(peak, weight);
              }
              require(mass > .5f * thin_width && mass < thin_width + 1.f && peak > .5f, "Gap repair erased or spread the separate real foreground object");
              require(std::abs(weighted_x / mass - expected) < .5f, "Gap repair shifted the separate foreground away from its depth projection");
            }
          }
          require(gap_samples != 0, "Mixed-edge fixture never inspected the disocclusion interior");
          std::printf("MEASURE mixed foreground fringe=%u native pixels depth=%ux%u offset=%u,%u mirror=%u AA=%u closer-donor=%u gap-eye=%u samples=%zu maximum normalized background error=%.9g\n", scenario.fringe_width, depth_width, depth_height, depth_left, depth_top, unsigned(scenario.mirrored), unsigned(antialias), unsigned(scenario.closer_donor), gap_eye, gap_samples, worst_gap_error);
          require(worst_gap_error < .005f, "Boundary foreground mixture stretched across the background disocclusion gap");
        }
      }
      const float full_rect[] {1, 1, 0, 0};
      observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), full_rect, 4);
      set_bool("EdgeAntialias", false);
      source_bytes = saved_source;
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      std::puts("PASS both edge directions and AA modes avoid mixed-color gap fringes at native/half depth; real thin foreground survives");
    }

    void check_analytic_inverse(std::vector<float> &hardware) {
      // A tilted perspective plane has affine inverse depth in screen X. Its
      // projected coordinate is p = (1 + eye*M*a)*s + eye*M*b, so this fixture
      // has a closed-form inverse and does not reproduce the shader's search.
      const auto saved_source = source_bytes;
      const unsigned pixel_bytes = bytes_per_pixel(source_format);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const unsigned reference = (x / 2) % 2 ? 16u : 0u;
          std::memcpy(source_bytes.data() + (size_t(y) * width + x) * pixel_bytes, saved_source.data() + (size_t(y) * width + reference) * pixel_bytes, pixel_bytes);
        }
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      set_float("ScreenPlane", 0.f);
      set_bool("DepthView", false);
      set_bool("EdgeAntialias", false);
      const float rect[] {1, 1, 0, 0};
      observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), rect, 4);
      set_float("Strength", 0.f);
      measured_frames();
      const auto flat = read(exported.p);
      const auto linear = [&](float value) -> double {
        if (color != 1) {
          return value;  // scRGB/PQ export is already linear scRGB.
        }
        return value <= .04045f ? value / 12.92 : std::pow((value + .055) / 1.055, 2.4);
      };
      const double maximum_shift = .01 * width, center = .5 * width;
      const double ramp_width = std::max(32., double(width) / 32.);
      const double bias = .07;
      set_float("Strength", 2.f);
      for (const double direction : {-1., 1.}) {
        const double slope = direction * 1.6 / ramp_width;
        for (unsigned y = 0; y < height; ++y) {
          for (unsigned x = 0; x < width; ++x) {
            const double nearness = std::clamp(slope * (x + .5 - center) + bias, -.9, .9);
            hardware[size_t(y) * width + x] = float(.5 + .5 * nearness);
          }
        }
        upload_depth(hardware);
        measured_frames();
        const auto stereo = read(exported.p);
        double maximum_error = 0, squared_error = 0;
        size_t samples = 0;
        for (unsigned eye = 0; eye < 2; ++eye) {
          const double sign = eye ? -1. : 1.;
          for (unsigned x = 0; x < width; ++x) {
            const double source = (x + .5 - sign * maximum_shift * (bias - slope * center)) /
                                  (1. + sign * maximum_shift * slope);
            if (std::abs(source - center) > .35 * ramp_width) {
              continue;
            }
            const double coordinate = source - .5;
            const unsigned left = unsigned(std::floor(coordinate));
            const double fraction = coordinate - left;
            // Only contrasting fractional texel pairs constrain refinement.
            // Exclude the ramp's clamps/kinks and solid-color texel pairs.
            if (left + 1 >= width || fraction < .1 || fraction > .9) {
              continue;
            }
            for (const unsigned y : {height / 4, height / 2, height * 3 / 4}) {
              for (unsigned c = 0; c < 3; ++c) {
                const double a = linear(channel(flat, left, y, c));
                const double b = linear(channel(flat, left + 1, y, c));
                if (std::abs(b - a) < .02 * (1. + std::max(std::abs(a), std::abs(b)))) {
                  continue;
                }
                const double actual = linear(channel(stereo, eye * width + x, y, c));
                require(std::isfinite(actual), "Affine inverse surface produced nonfinite color");
                const double error = std::abs((actual - a) / (b - a) - fraction);
                maximum_error = std::max(maximum_error, error);
                squared_error += error * error;
                ++samples;
              }
              require(channel(stereo, eye * width + x, y, 3) == 1.f, "Continuous affine inverse surface was treated as a disocclusion");
            }
          }
        }
        require(samples >= 12, "Affine inverse fixture lacks contrasting fractional samples");
        const double rms = std::sqrt(squared_error / samples);
        std::printf("MEASURE affine inverse slope=%+.9g source-coordinate error max=%.9g RMS=%.9g samples=%zu\n", slope, maximum_error, rms, samples);
        require(maximum_error < .12 && rms < .05, "Refined inverse coordinates disagree with the independent tilted-plane solution");
      }
      source_bytes = saved_source;
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      std::puts("PASS both tilted-plane slopes and eyes match closed-form fractional inverse geometry");
    }

    void check_multilayer_inverse(std::vector<float> &hardware) {
      // Three disjoint source slabs all project over one destination pixel. The
      // middle slab has zero disparity, making it a wrong but exact fixed point
      // for a solver seeded only at the destination. The nearest slab must win.
      const auto saved_source = source_bytes;
      const unsigned target = width / 2;
      const std::array<float, 3> nearness {.75f, 0.f, -.75f};
      const double maximum_shift = .01 * width;
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      set_float("ScreenPlane", 0.f);
      set_bool("DepthView", false);
      set_bool("EdgeAntialias", false);
      for (unsigned tested_eye = 0; tested_eye < 2; ++tested_eye) {
        const double sign = tested_eye ? -1. : 1.;
        std::array<int, 3> centers {};
        for (unsigned layer = 0; layer < 3; ++layer) {
          centers[layer] = int(std::lround(target - sign * maximum_shift * nearness[layer]));
        }
        require(std::abs(centers[0] - centers[1]) >= 3 && std::abs(centers[1] - centers[2]) >= 3, "Multilayer source slabs overlap before projection");
        for (unsigned y = 0; y < height; ++y) {
          for (unsigned x = 0; x < width; ++x) {
            int layer = -1;
            for (unsigned candidate = 0; candidate < 3; ++candidate) {
              if (std::abs(int(x) - centers[candidate]) <= 1) {
                layer = int(candidate);
              }
            }
            const size_t pixel = size_t(y) * width + x;
            hardware[pixel] = layer < 0 ? .05f : .5f + .5f * nearness[unsigned(layer)];
            if (color == 1) {
              const std::uint8_t values[] {std::uint8_t(layer == 0 ? 224 : 32), std::uint8_t(layer == 1 ? 224 : 32), std::uint8_t(layer == 2 ? 224 : 32), 255};
              std::memcpy(source_bytes.data() + pixel * 4, values, 4);
            } else if (color == 2) {
              const std::uint16_t values[] {std::uint16_t(layer == 0 ? 0x3c00 : 0x3400), std::uint16_t(layer == 1 ? 0x3c00 : 0x3400), std::uint16_t(layer == 2 ? 0x3c00 : 0x3400), 0x3c00};
              std::memcpy(source_bytes.data() + pixel * 8, values, 8);
            } else {
              const std::uint32_t value = (layer == 0 ? 600u : 256u) | ((layer == 1 ? 600u : 256u) << 10) |
                                          ((layer == 2 ? 600u : 256u) << 20) | (3u << 30);
              std::memcpy(source_bytes.data() + pixel * 4, &value, 4);
            }
          }
        }
        fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
        upload_depth(hardware);
        set_float("Strength", 0.f);
        measured_frames();
        const auto flat = read(exported.p);
        set_float("Strength", 2.f);
        measured_frames();
        const auto stereo = read(exported.p);
        unsigned eligible = 0;
        for (unsigned layer = 0; layer < 3; ++layer) {
          // Each inverse lies between same-color source texel centers, away
          // from silhouette footprints and color filtering across an edge.
          const double source_index = target - sign * maximum_shift * nearness[layer];
          if (source_index >= centers[layer] - 1 && source_index <= centers[layer] + 1) {
            ++eligible;
          }
        }
        require(eligible == 3, "Multilayer fixture lacks three independent projected intersections");
        for (const unsigned y : {height / 4, height / 2, height * 3 / 4}) {
          for (unsigned c = 0; c < 3; ++c) {
            const float expected = channel(flat, unsigned(centers[0]), y, c);
            const float actual = channel(stereo, tested_eye * width + target, y, c);
            require(std::isfinite(actual) && std::abs(actual - expected) < .003f + .002f * std::abs(expected), "Inverse search chose a farther fixed point instead of the first visible foreground layer");
          }
          require(channel(stereo, tested_eye * width + target, y, 3) == 1.f, "Observed nearest multilayer intersection lost confidence");
        }
        std::printf("PASS eye=%u selects nearest of three analytic slab intersections; source centers=%d,%d,%d\n", tested_eye, centers[0], centers[1], centers[2]);
      }
      source_bytes = saved_source;
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
    }

    void check_saturated_neighbor(std::vector<float> &hardware) {
      // Adjacent endpoint depths can be far apart in raw nearness while their
      // bounded disparities differ by less than one pixel. The projected cell
      // is defined by those bounded endpoints; interpolating raw nearness and
      // then clipping would introduce an unsupported knot inside the cell.
      const auto saved_source = source_bytes;
      const unsigned boundary = width / 2, left_index = boundary - 1;
      const unsigned pixel_bytes = bytes_per_pixel(source_format);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const unsigned reference = x < boundary ? 0u : 16u;
          std::memcpy(source_bytes.data() + (size_t(y) * width + x) * pixel_bytes, saved_source.data() + (size_t(y) * width + reference) * pixel_bytes, pixel_bytes);
        }
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 200.f);
      set_float("ScreenPlane", 0.f);
      set_bool("DepthView", false);
      set_bool("EdgeAntialias", false);
      set_float("Strength", 0.f);
      measured_frames();
      const auto flat = read(exported.p);
      const auto linear = [&](float value) -> double {
        if (color != 1) {
          return value;
        }
        return value <= .04045f ? value / 12.92 : std::pow((value + .055) / 1.055, 2.4);
      };
      // A fractional maximum offset puts actual destination pixel centers
      // inside the projected cell in both mirrored/eye arrangements.
      const float strength = float((std::floor(.005 * width) + .3) / (.005 * width));
      const double maximum_shift = .005 * width * strength;
      const float raw_low = float(.5 + (1. - .8 / maximum_shift) / 200.);
      const double low_nearness = (double(raw_low) - .5) * 200.;
      const double low_disparity = maximum_shift * low_nearness;
      require(low_nearness > 0 && low_nearness < 1 && maximum_shift - low_disparity < 1., "Saturated-neighbor fixture must have a subpixel bounded disparity difference");
      set_float("Strength", strength);
      for (unsigned tested_eye = 0; tested_eye < 2; ++tested_eye) {
        for (unsigned y = 0; y < height; ++y) {
          for (unsigned x = 0; x < width; ++x) {
            const bool saturated = tested_eye ? x < boundary : x >= boundary;
            hardware[size_t(y) * width + x] = saturated ? 1.f : raw_low;  // raw1 -> nearness100.
          }
        }
        upload_depth(hardware);
        measured_frames();
        const auto stereo = read(exported.p);
        const double sign = tested_eye ? -1. : 1.;
        const double a = left_index + .5 + sign * (tested_eye ? maximum_shift : low_disparity);
        const double b = left_index + 1.5 + sign * (tested_eye ? low_disparity : maximum_shift);
        require(b > a, "Saturated-neighbor analytic cell is not ordered");
        double worst_error = 0;
        unsigned samples = 0;
        for (unsigned x = 0; x < width; ++x) {
          const double destination = x + .5;
          if (destination <= a + .1 || destination >= b - .1) {
            continue;
          }
          const double fraction = (destination - a) / (b - a);
          for (const unsigned y : {height / 4, height / 2, height * 3 / 4}) {
            const double first = linear(channel(flat, left_index, y, 0));
            const double second = linear(channel(flat, left_index + 1, y, 0));
            const double actual = linear(channel(stereo, tested_eye * width + x, y, 0));
            require(std::abs(second - first) > .05 && std::isfinite(actual), "Saturated-neighbor fixture lacks finite color contrast");
            worst_error = std::max(worst_error, std::abs((actual - first) / (second - first) - fraction));
            require(channel(stereo, tested_eye * width + x, y, 3) == 1.f, "Continuous bounded-disparity cell was incorrectly filled as a hole");
            ++samples;
          }
        }
        require(samples >= 3, "Saturated-neighbor fixture contains no interior destination samples");
        std::printf("MEASURE saturated-neighbor eye=%u raw-low=%.9g near-high=100 bounded-delta=%.9g fractional-error=%.9g samples=%u\n", tested_eye, raw_low, maximum_shift - low_disparity, worst_error, samples);
        require(worst_error < .05, "Inverse refinement interpolated raw nearness through the disparity saturation boundary");
      }
      source_bytes = saved_source;
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      std::puts("PASS saturated raw-nearness neighbors preserve the analytic bounded-disparity cell in both eyes");
    }

    void benchmark_gpu(std::vector<float> &hardware, bool stress = false) {
      require(!performance_mode, "GPU timing requires PerformanceMode0 so benchmark controls are explicit");
      unsigned enabled_techniques = 0;
      observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *runtime, api::effect_technique technique) {
        if (runtime->get_technique_state(technique)) {
          ++enabled_techniques;
        }
      });
      require(enabled_techniques == 1, "GPU timing requires only the SunshineDepth3D technique enabled");
      observed.ready = true;
      observed.capture = false;
      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2.f);
      const float strength = stress ? 2.f : 1.f;
      set_float("Strength", strength);
      set_float("ScreenPlane", 0.f);
      set_bool("DepthView", false);
      set_bool("EdgeAntialias", true);
      const unsigned active_width = stress ? unsigned(std::lround(double(width) * 2228. / 3840.)) : width;
      const unsigned active_height = stress ? unsigned(std::lround(double(height) * 1256. / 2160.)) : height;
      const unsigned depth_left = width - active_width, depth_top = height - active_height;
      const float rect[] {float(active_width) / width, float(active_height) / height, float(depth_left) / width, float(depth_top) / height};
      observed.runtime->set_uniform_value_float(uniform("Sunshine_DepthRect"), rect, 4);
      // The original deterministic native color pattern is restored by every
      // earlier test. This fixed scene has a sloped background and foreground
      // slab, exercising smooth surface searches and both occlusion boundaries.
      std::fill(hardware.begin(), hardware.end(), 0.f);
      for (unsigned y = 0; y < active_height; ++y) {
        for (unsigned x = 0; x < active_width; ++x) {
          const float u = (float(x) + .5f) / active_width, v = (float(y) + .5f) / active_height;
          // The original =1 scene/formulas remain unchanged. Stress adds one
          // coherent fence: four depth columns per 24-column period, over the
          // same sloped background, with a padded ~58% active depth viewport.
          const bool foreground = stress ? x % 24u < 4u && v > .08f && v < .92f :
                                           u > .4f + .12f * v && u < .65f + .05f * v && v > .15f && v < .85f;
          hardware[size_t(y + depth_top) * width + x + depth_left] = foreground ? .75f - .1f * v : .15f + .2f * u + .15f * v;
        }
      }
      upload_depth(hardware);
      const auto fingerprint = [](const void *data, size_t bytes) {
        std::uint64_t value = 14695981039346656037ull;
        const auto *input = static_cast<const std::uint8_t *>(data);
        for (size_t i = 0; i < bytes; ++i) {
          value ^= input[i];
          value *= 1099511628211ull;
        }
        return value;
      };
      std::printf("GPU_TIMING_CONFIG scope=effects_begin_to_SunshineDepth3D_end workload=%s source=%ux%u color=%u strength=%.0f screen_plane=0 AA=1 depth-storage=%ux%u depth-active=%ux%u depth-offset=%u,%u rect=%.9g,%.9g,%.9g,%.9g raw_anchor=.5 raw_gain=2 source_fnv1a64=%016llx depth_fnv1a64=%016llx warmup=4 samples=8\n", stress ? "dense-slats-reduced-depth" : "sloped-background-slab", width, height, color, strength, width, height, active_width, active_height, depth_left, depth_top, rect[0], rect[1], rect[2], rect[3], static_cast<unsigned long long>(fingerprint(source_bytes.data(), source_bytes.size())), static_cast<unsigned long long>(fingerprint(hardware.data(), hardware.size() * sizeof(float))));
      // These are GPU timestamps on the actual effect submission queue, not
      // CPU Present duration. The interval includes ReShade's effect setup,
      // excludes the game's upload/copy, and ends before overlay/mono readback.
      auto *effect_queue = reinterpret_cast<ID3D12CommandQueue *>(observed.runtime->get_command_queue()->get_native());
      UINT64 frequency = 0;
      checked(effect_queue->GetTimestampFrequency(&frequency), "Get actual effect queue timestamp frequency");
      require(frequency != 0, "Effect queue returned a zero timestamp frequency");
      com_ptr<ID3D12QueryHeap> queries;
      D3D12_QUERY_HEAP_DESC query_desc {};
      query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
      query_desc.Count = 2;
      checked(game->CreateQueryHeap(&query_desc, IID_PPV_ARGS(queries.put())), "Create bounded effect timestamp queries");
      com_ptr<ID3D12Resource> timestamps;
      buffer(timestamps, 2 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK);

      struct timing_scope {
        ~timing_scope() {
          observed.timing_armed = false;
          observed.timing_queries = nullptr;
          observed.timing_readback = nullptr;
        }
      } timing_cleanup;

      observed.timing_queries = queries.p;
      observed.timing_readback = timestamps.p;
      const unsigned before_warmup = observed.renders;
      for (unsigned warmup = 0; warmup < 4; ++warmup) {
        step();
      }
      require(observed.renders == before_warmup + 4, "GPU timing warmup did not render all four effect frames");
      bool ready = false, calibrated = false;
      observed.runtime->get_uniform_value_bool(uniform("Sunshine_DepthReady"), &ready, 1);
      observed.runtime->get_uniform_value_bool(uniform("Sunshine_Calibrated"), &calibrated, 1);
      require(ready && calibrated && observed.bound_depth_view.handle == depth_view.handle, "GPU timing requires currently bound depth and active injected calibration");
      // This preflight is deliberately outside the measured GPU intervals. It
      // prevents a timing-only iteration from silently measuring flat fallback;
      // it does not replace the complete geometry/color correctness suite.
      const auto preview = read(exported.p);
      float minimum = INFINITY, maximum = -INFINITY, eye_difference = 0;
      for (const unsigned y : {height / 4, height / 2, height * 3 / 4}) {
        for (unsigned x = 0; x < width; ++x) {
          for (unsigned c = 0; c < 4; ++c) {
            const float left = channel(preview, x, y, c), right = channel(preview, width + x, y, c);
            require(std::isfinite(left) && std::isfinite(right), "GPU timing preflight encountered nonfinite export pixels");
            if (c < 3) {
              minimum = std::min(minimum, std::min(left, right));
              maximum = std::max(maximum, std::max(left, right));
              eye_difference = std::max(eye_difference, std::abs(left - right));
            }
          }
        }
      }
      require(maximum - minimum > .05f && eye_difference > .001f, "GPU timing preflight is uniform or identical-eye fallback instead of active stereo");
      std::printf("GPU_TIMING_PREFLIGHT active_calibration=1 sampled_pixels_finite=1 eye_difference=%.9g range=%.9g\n", eye_difference, maximum - minimum);
      std::array<double, 8> milliseconds {};
      for (unsigned sample = 0; sample < milliseconds.size(); ++sample) {
        observed.timing_started = observed.timing_finished = observed.timing_invalid = false;
        observed.timing_armed = true;
        const auto before_renders = observed.renders;
        step();  // Existing GPU fence waits for query resolution before reuse.
        observed.timing_armed = false;
        require(observed.timing_started && observed.timing_finished && !observed.timing_invalid && observed.renders == before_renders + 1, "GPU timing did not observe exactly one complete Sunshine effect interval");
        void *mapped = nullptr;
        const D3D12_RANGE read_range {0, 2 * sizeof(UINT64)};
        checked(timestamps->Map(0, &read_range, &mapped), "Read completed effect timestamps");
        std::array<UINT64, 2> ticks {};
        std::memcpy(ticks.data(), mapped, sizeof(ticks));
        const D3D12_RANGE no_write {0, 0};
        timestamps->Unmap(0, &no_write);
        require(ticks[0] != 0 && ticks[1] > ticks[0], "Effect GPU timestamps are missing or nonmonotonic");
        milliseconds[sample] = 1000. * double(ticks[1] - ticks[0]) / double(frequency);
        std::printf("GPU_TIMING_SAMPLE index=%u begin=%llu end=%llu frequency=%llu gpu_ms=%.6f\n", sample, static_cast<unsigned long long>(ticks[0]), static_cast<unsigned long long>(ticks[1]), static_cast<unsigned long long>(frequency), milliseconds[sample]);
      }
      double total = 0;
      for (double value : milliseconds) {
        total += value;
      }
      std::sort(milliseconds.begin(), milliseconds.end());
      std::printf("GPU_TIMING_RESULT scope=effects_begin_to_SunshineDepth3D_end workload=%s samples=8 median_ms=%.6f mean_ms=%.6f min_ms=%.6f max_ms=%.6f\n", stress ? "dense-slats-reduced-depth" : "sloped-background-slab", .5 * (milliseconds[3] + milliseconds[4]), total / milliseconds.size(), milliseconds.front(), milliseconds.back());
    }

    void run() {
      discover();
      if (sunshine_parity::requested()) {
        require(!performance_mode, "Parity requires PerformanceMode0 and live artistic controls");
        sunshine_parity::run(*this, width, height, true, observed.runtime, observed.ready, observed.capture);
        return;
      }
      std::vector<float> hardware(size_t(width) * height, .75f);
      upload_depth(hardware);
      const char *timing = std::getenv("SUNSHINE_NATIVE_GPU_TIMING");
      const bool timing_stress = timing && std::strcmp(timing, "stress") == 0;
      const bool timing_requested = timing_stress || (timing && std::strcmp(timing, "1") == 0);
      require(!timing || !*timing || std::strcmp(timing, "0") == 0 || timing_requested, "SUNSHINE_NATIVE_GPU_TIMING must be 0, 1 or stress");
      require(!timing_requested || !performance_mode, "GPU timing requires PerformanceMode0 so benchmark controls are explicit");
      const char *only = std::getenv("SUNSHINE_NATIVE_GPU_TIMING_ONLY");
      const bool timing_only = only && std::strcmp(only, "1") == 0;
      require(!timing_only || timing_requested, "GPU timing-only mode requires a selected GPU timing workload");
      if (timing_only) {
        std::puts("GPU_TIMING_ONLY: correctness suite skipped; running bounded benchmark with readiness/export preflight");
        benchmark_gpu(hardware, timing_stress);
        return;
      }
      if (performance_mode) {
        // The REAL runtime is in PerformanceMode=1. UI strength/plane values
        // may specialize; add-on-owned calibration must remain mutable uniforms.
        observed.ready = false;
        measured_frames();
        const auto flat = read(exported.p);
        observed.ready = true;
        measured_frames();
        const auto forward = measure_eye_shifts(flat, read(exported.p), "performance live calibration");
        set_float("Sunshine_RawGain", -2.f);
        measured_frames();
        const auto reverse = measure_eye_shifts(flat, read(exported.p), "performance reversed calibration");
        require(forward[0] < -.5f && forward[1] > .5f && reverse[0] > .5f && reverse[1] < -.5f, "Performance mode froze dynamic calibration");
        observed.ready = false;
        measured_frames();
        require(read(exported.p) == flat, "Performance mode froze readiness");
        std::puts("PASS actual ReShade PerformanceMode1 retains dynamic calibration and readiness");
        return;
      }
      set_float("Strength", 0);
      measured_frames();
      const auto flat = read(exported.p);
      for (unsigned y = 0; y < height; y += std::max(1u, height / 80))
        for (unsigned x = 0; x < width; ++x)
          for (unsigned c = 0; c < 3; ++c)
            require(channel(flat, x, y, c) == channel(flat, x + width, y, c), "Zero strength differs between eyes");

      float previous = 0;
      for (const float strength : {.5f, 1.f, 2.f}) {
        set_float("Strength", strength);
        measured_frames();
        const auto stereo = read(exported.p);
        const auto shifts = measure_eye_shifts(flat, stereo, std::to_string(strength).c_str());
        const float expected = .0025f * strength * width;
        require(shifts[0] < 0 && shifts[1] > 0, "Foreground stereo directions incorrect");
        require(std::abs(std::abs(shifts[0]) - expected) < .4f && std::abs(std::abs(shifts[1]) - expected) < .4f, "Strength does not match analytical constant-plane shift");
        require(std::abs(shifts[0]) > previous + .25f, "Positive strength slider has a fixed effect");
        previous = std::abs(shifts[0]);
      }
      set_float("Strength", 1);
      set_float("ScreenPlane", .75f);
      measured_frames();
      const auto behind = measure_eye_shifts(flat, read(exported.p), "behind screen");
      require(behind[0] > 0 && behind[1] < 0, "Screen plane does not reverse depth across zero parallax");
      set_float("ScreenPlane", 0);

      // Same scene in two hardware depth conventions must have matching geometry.
      measured_frames();
      const auto reversed = read(exported.p);
      std::fill(hardware.begin(), hardware.end(), .25f);
      upload_depth(hardware);
      set_float("Sunshine_RawGain", -2);
      measured_frames();
      require(read(exported.p) == reversed, "Equivalent normal/reversed-Z inputs produce different stereo");

      // Diagnostic is raw relative scene depth, independent of artistic controls.
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
          hardware[size_t(y) * width + x] = .2f + .6f * (float(x) / float(width - 1));
      upload_depth(hardware);
      set_float("Sunshine_RawGain", 2);
      set_bool("DepthView", true);
      measured_frames();
      const auto diagnostic = read(exported.p);
      set_float("Strength", 0);
      set_float("ScreenPlane", .8f);
      measured_frames();
      require(read(exported.p) == diagnostic, "Diagnostic changes with strength or screen plane");
      require(std::abs(channel(diagnostic, width / 8, height / 2, 0) - channel(diagnostic, width * 7 / 8, height / 2, 0)) > .2f, "Depth diagnostic is flat");
      for (unsigned x = 0; x < width; ++x) {
        require(channel(diagnostic, x, height / 2, 0) == channel(diagnostic, x + width, height / 2, 0), "Diagnostic eyes differ");
        if (x) require(channel(diagnostic, x, height / 2, 0) + .002f >= channel(diagnostic, x - 1, height / 2, 0), "Depth diagnostic is not monotonic");
      }
      set_bool("DepthView", false);
      set_float("ScreenPlane", 0);
      set_float("Strength", 1);
      observed.ready = false;
      measured_frames();
      require(read(exported.p) == flat, "Missing source/calibration does not duplicate current mono");

      check_invalid_calibration(flat, hardware);
      check_padded_depth(hardware);

      // Source FP32 near1: an early FP16 conversion would collapse both planes.
      observed.ready = true;
      std::fill(hardware.begin(), hardware.end(), .999975f);
      upload_depth(hardware);
      set_float("Sunshine_RawAnchor", .99995f);
      set_float("Sunshine_RawGain", -20000.f);
      measured_frames();
      const auto precise = measure_eye_shifts(flat, read(exported.p), "near-one raw FP32");
      require(precise[0] > .5f && precise[1] < -.5f, "Native depth precision was lost near1");

      set_float("Sunshine_RawAnchor", .5f);
      set_float("Sunshine_RawGain", 2);
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
          hardware[size_t(y) * width + x] = x < width / 2 + y / 4 ? .8f : .2f;
      upload_depth(hardware);
      // Place the projected step inside a destination pixel at every test
      // resolution, rather than accidentally aligning a 4K edge between rays.
      set_float("Strength", (std::floor(.003f * width) + .3f) / (.003f * width));
      set_bool("EdgeAntialias", false);
      measured_frames();
      const auto noaa = read(exported.p);
      set_bool("EdgeAntialias", true);
      measured_frames();
      const auto aa = read(exported.p);
      std::printf("MEASURE warp-edge AA response=%.9g\n", image_difference(noaa, aa));
      require(image_difference(noaa, aa) > .0001f, "Subpixel boundary fixture did not exercise stereo-edge AA");
      set_bool("EdgeAntialias", false);
      measured_frames();
      require(read(exported.p) == noaa, "Edge AA changes are not reversible");
      for (unsigned y = 0; y < height; y += std::max(1u, height / 80))
        for (unsigned x = 0; x < width * 2; ++x)
          for (unsigned c = 0; c < 3; ++c)
            require(std::isfinite(channel(aa, x, y, c)), "Warp-edge rendering contains nonfinite color");
      check_surface_ownership(hardware);
      check_mixed_edge_gap_fill(hardware);
      check_analytic_inverse(hardware);
      check_multilayer_inverse(hardware);
      check_saturated_neighbor(hardware);
      if (timing_requested)
        benchmark_gpu(hardware, timing_stress);
      std::printf("PASS independent native stereo DX12 %ux%u color=%u: strength, screen plane, reverse-Z, raw precision, diagnostic, fallback, padded viewport, mono preservation, AA and ownership\n", width, height, color);
    }
  };
}  // namespace

#ifndef SUNSHINE_NATIVE_STEREO_RUNTIME_HELPER
int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 6 && argc != 8) {
    std::fprintf(stderr, "usage: test_native_stereo_runtime <official-ReShade64.dll> <Sunshine-Shaders> <isolated-ignored-output-directory> <srgb|scrgb|pq> <PerformanceMode:0|1> [source-width source-height]\n");
    return 2;
  }
  std::thread([] {
    Sleep(180000);
    std::fputs("FAIL actual D3D12 native stereo fixture watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    const unsigned color = std::strcmp(argv[4], "srgb") == 0 ? 1 : std::strcmp(argv[4], "scrgb") == 0 ? 2 :
                                                                 std::strcmp(argv[4], "pq") == 0      ? 3 :
                                                                                                        0;
    require(color && (std::strcmp(argv[5], "0") == 0 || std::strcmp(argv[5], "1") == 0), "Invalid fixture color or performance-mode argument");
    if (argc == 8) {
      size_t used_width = 0, used_height = 0;
      const auto requested_width = std::stoul(argv[6], &used_width);
      const auto requested_height = std::stoul(argv[7], &used_height);
      require(used_width == std::strlen(argv[6]) && used_height == std::strlen(argv[7]) && requested_width >= 640 && requested_height >= 360 && requested_width <= 3840 && requested_height <= 2160 && requested_width % 2 == 0 && requested_height % 2 == 0, "Fixture dimensions must be even, from 640x360 up to 3840x2160");
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
