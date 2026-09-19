// SPDX-License-Identifier: GPL-3.0-only
// Opt-in natural D3D12 Present fixture with actual ReShade and a test-only focus observer.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "src/reshade_bridge_protocol.h"
#include "test_overlay_pixels.h"
#include "test_receiver_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <reshade.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <windows.h>

namespace {
  namespace api = reshade::api;
  namespace wire = reshade_bridge;
  namespace fs = std::filesystem;
  using create_device_t = decltype(&D3D11CreateDeviceAndSwapChain);
  constexpr unsigned source_width = 960, source_height = 540;
#if defined(SUNSHINE_GAME_EXPORT_FIXTURE)
  constexpr const char *fixture_effect_name = "SunshineGame3D.fx";
  constexpr const char *fixture_technique_name = "SunshineGame3D";
#elif defined(SUNSHINE_NATIVE_EXPORT_FIXTURE)
  constexpr const char *fixture_effect_name = "SunshineDepth3D.fx";
  constexpr const char *fixture_technique_name = "SunshineDepth3D";
#else
  constexpr const char *fixture_effect_name = "SuperDepth3D.fx";
  constexpr const char *fixture_technique_name = "SuperDepth3D";
#endif

  template<class T>
  struct com_ptr {
    T *p = nullptr;

    ~com_ptr() {
      reset();
    }

    T *operator->() const {
      return p;
    }

    T **put() {
      reset();
      return &p;
    }

    void reset() {
      if (p) {
        p->Release();
        p = nullptr;
      }
    }
  };

  void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void checked(HRESULT hr, const char *message) {
    if (FAILED(hr)) {
      char detail[320];
      std::snprintf(detail, sizeof(detail), "%s (0x%08lx)", message, static_cast<unsigned long>(hr));
      throw std::runtime_error(detail);
    }
  }

  std::uint64_t read64(std::uint64_t &value) {
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&value), 0, 0));
  }

  bool snapshot(wire::shared_state_t &shared, wire::metadata_t &metadata) {
    const auto before = InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&shared.metadata_sequence), 0, 0);
    if (before & 1) {
      return false;
    }
    metadata = shared.metadata;
    MemoryBarrier();
    return before == InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&shared.metadata_sequence), 0, 0);
  }

  struct observation_t {
    api::effect_runtime *runtime = nullptr;
    api::color_space color = api::color_space::unknown;
    unsigned stage = 0, presents = 0, techniques = 0, finishes = 0, reloads = 0, overlay_opens = 0, overlay_closes = 0;
    bool order_failed = false;
  } observation;

  void on_init(api::effect_runtime *runtime) {
    observation.runtime = runtime;
  }

  void on_reload(api::effect_runtime *) {
    ++observation.reloads;
  }

  bool on_overlay(api::effect_runtime *, bool open, api::input_source) {
    if (open) {
      ++observation.overlay_opens;
    } else {
      ++observation.overlay_closes;
    }
    return false;
  }

  void on_begin(api::command_queue *, api::swapchain *swapchain, const api::rect *, const api::rect *, std::uint32_t, const api::rect *) {
    if (observation.stage != 0 && observation.stage != 4) {
      observation.order_failed = true;
    }
    observation.stage = 1;
    ++observation.presents;
    observation.color = swapchain->get_color_space();
  }

  void on_technique(api::effect_runtime *, api::effect_technique, api::command_list *, api::resource_view, api::resource_view) {
    if (observation.stage != 1 && observation.stage != 2) {
      observation.order_failed = true;
    }
    observation.stage = 2;
    ++observation.techniques;
  }

  void on_effect_present(api::effect_runtime *) {
    if (observation.stage != 1 && observation.stage != 2) {
      observation.order_failed = true;
    }
    observation.stage = 3;
  }

  void on_finish(api::command_queue *, api::swapchain *) {
    if (observation.stage != 3) {
      observation.order_failed = true;
    }
    observation.stage = 4;
    ++observation.finishes;
  }

  const char *fixture_effect = R"FX(
#if BUFFER_COLOR_SPACE == 1
#define EXPORT_FORMAT RGB10A2
#define EXPORT_TRANSFER "srgb"
#else
#define EXPORT_FORMAT RGBA16F
#define EXPORT_TRANSFER "scrgb"
#endif
texture DoubleTex <
  sunshine_sbs_export = 1;
  sunshine_sbs_layout = "sbs_lr";
  sunshine_sbs_color_space = EXPORT_TRANSFER;
  sunshine_sbs_source_color_space = BUFFER_COLOR_SPACE;
  sunshine_sbs_source_width = BUFFER_WIDTH;
  sunshine_sbs_source_height = BUFFER_HEIGHT;
> { Width = BUFFER_WIDTH * 2; Height = BUFFER_HEIGHT; Format = EXPORT_FORMAT; };
texture GameBackBuffer : COLOR;
sampler GameSampler { Texture = GameBackBuffer; };
uniform float StereoGain < ui_type = "slider"; ui_min = 0.25; ui_max = 1.0; > = 1.0;
void VS(uint vertex : SV_VertexID, out float4 position : SV_Position, out float2 uv : TEXCOORD0) {
  uv = float2((vertex << 1) & 2, vertex & 2);
  position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 Stereo(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
#if BUFFER_COLOR_SPACE == 1
  return float4((uv.x < 0.5 ? float3(1, 0.25, 0) : float3(0, 0.5, 1)) * StereoGain, 1);
#else
  return float4((uv.x < 0.5 ? float3(4, 0.5, -0.125) : float3(0.125, 2, 8)) * StereoGain, 1);
#endif
}
float4 Mono(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
  return tex2D(GameSampler, uv);
}
technique SuperDepth3D {
  pass DoubleOut { VertexShader = VS; PixelShader = Stereo; RenderTarget = DoubleTex; }
  pass StereoOut { VertexShader = VS; PixelShader = Mono; }
}
)FX";

  std::string fixture_shader_source() {
    std::string source(fixture_effect);
#if defined(SUNSHINE_NATIVE_EXPORT_FIXTURE) || defined(SUNSHINE_GAME_EXPORT_FIXTURE)
    // Transport-only controlled paint: exercise the selected effect identity,
    // overlay and shared-fence path. Full shader rendering is verified separately.
    const std::string old_technique = "technique SuperDepth3D {";
    const auto position = source.find(old_technique);
    require(position != std::string::npos, "Controlled transport technique marker is missing");
    source.replace(position, old_technique.size(), std::string("technique ") + fixture_technique_name + " {");
#endif
#ifdef SUNSHINE_NATIVE_EXPORT_FIXTURE
    source = R"FX(
// These are runtime-owned calibration inputs, intentionally without initializers.
uniform bool Sunshine_DepthReady < source = "bufready_depth"; hidden = true; >;
uniform bool Sunshine_Calibrated < hidden = true; >;
uniform float Sunshine_RawAnchor < hidden = true; >;
uniform float Sunshine_RawGain < hidden = true; >;
uniform float4 Sunshine_DepthRect < hidden = true; >;
uniform int DepthDirection < hidden = true; >;
)FX" + source;
#endif
    return source;
  }

  void write_file(const fs::path &path, const std::string &text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
    require(output.good(), "Could not write isolated fixture input");
  }

  float half_float(std::uint16_t value) {
    const bool negative = (value & 0x8000) != 0;
    const unsigned exponent = (value >> 10) & 31, fraction = value & 1023;
    const float result = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24) :
                                         std::ldexp(1.0f + fraction / 1024.0f, static_cast<int>(exponent) - 15);
    return negative ? -result : result;
  }

  struct fixture_t {
    HWND window = nullptr;
    HMODULE reshade_module = nullptr;
    HANDLE mapping = nullptr;
    wire::shared_state_t *shared = nullptr;
    com_ptr<ID3D11Device> receiver;
    com_ptr<ID3D11Device5> receiver5;
    com_ptr<ID3D11DeviceContext> receiver_context;
    com_ptr<ID3D12Device> game;
    com_ptr<ID3D12CommandQueue> queue;
    com_ptr<ID3D12CommandAllocator> allocator;
    com_ptr<ID3D12GraphicsCommandList> commands;
    com_ptr<ID3D12DescriptorHeap> targets;
    com_ptr<ID3D12Resource> backbuffers[2];
    com_ptr<ID3D12Fence> completion;
    com_ptr<IDXGISwapChain3> swapchain;
    HANDLE completion_event = nullptr;
    UINT target_stride = 0;
    std::uint64_t completion_value = 0;
    using set_foreground_t = void (*)(HWND);
    set_foreground_t set_foreground = nullptr;
    void (*set_overlay_patch)(BOOL) = nullptr;
#ifdef SUNSHINE_GAME_EXPORT_FIXTURE
    void (*set_production_panel)(unsigned) = nullptr;
    unsigned (*production_panel_draws)() = nullptr;
    BOOL (*query_control)(api::effect_runtime *, unsigned, unsigned *, float *) = nullptr;
    BOOL (*edit_control_float)(api::effect_runtime *, unsigned, float) = nullptr;
    BOOL (*edit_control_int)(api::effect_runtime *, unsigned, int) = nullptr;
    BOOL (*set_game_enabled)(api::effect_runtime *, BOOL) = nullptr;
    BOOL (*query_automatic)(api::effect_runtime *, unsigned *) = nullptr;
    BOOL (*recalibrate)(api::effect_runtime *) = nullptr;
#endif
    bool native_only = false, migrated_settings = false;
    fs::path output_directory;
    bool expect_overlay = false;
    float expected_gain = 1.0f;
    std::uint64_t last_sequence = 0, last_timestamp = 0;
    unsigned expected_color = 1;
    HMODULE receiver_module = nullptr;
    void *receiver_state = nullptr;
    sunshine_receiver_test_create receiver_create = nullptr;
    sunshine_receiver_test_focus receiver_focus = nullptr;
    sunshine_receiver_test_poll receiver_poll = nullptr;
    sunshine_receiver_test_destroy receiver_destroy = nullptr;
    sunshine_receiver_test_frame last_received;
    RECT source_rect {};

    ~fixture_t() {
      if (receiver_state) {
        receiver_destroy(receiver_state);
      }
      if (receiver_module) {
        FreeLibrary(receiver_module);
      }
      if (shared) {
        UnmapViewOfFile(shared);
      }
      if (mapping) {
        CloseHandle(mapping);
      }
      if (reshade_module) {
        reshade::unregister_addon(GetModuleHandleW(nullptr), reshade_module);
      }
      if (set_foreground) {
        set_foreground(nullptr);
      }
      commands.reset();
      allocator.reset();
      targets.reset();
      for (auto &backbuffer : backbuffers) {
        backbuffer.reset();
      }
      swapchain.reset();
      queue.reset();
      completion.reset();
      if (completion_event) {
        CloseHandle(completion_event);
      }
      game.reset();
      if (window) {
        DestroyWindow(window);
      }
      // Process exit unloads the official runtime after all COM resources are gone.
    }

    void initialize(const fs::path &runtime, const fs::path &addon, const fs::path &directory, unsigned color, const fs::path &receiver_dll = {}) {
      expected_color = color;
      output_directory = directory;
      const auto flag = [](const char *name) {
        const char *value = std::getenv(name);
        require(!value || !*value || !std::strcmp(value, "0") || !std::strcmp(value, "1"), "Native fixture flags must be 0 or 1");
        return value && !std::strcmp(value, "1");
      };
      native_only = flag("SUNSHINE_GAME3D_NATIVE_ONLY");
      migrated_settings = flag("SUNSHINE_GAME3D_NATIVE_MIGRATED");
#ifndef SUNSHINE_GAME_EXPORT_FIXTURE
      require(!native_only && !migrated_settings, "Native controls/no-FX tests require the Game 3D fixture executable");
#endif
      if (!receiver_dll.empty()) {
        receiver_module = LoadLibraryW(receiver_dll.c_str());
        require(receiver_module != nullptr, "Could not load production-receiver test facade");
        receiver_create = reinterpret_cast<sunshine_receiver_test_create>(GetProcAddress(receiver_module, "SunshineReceiverTestCreate"));
        receiver_focus = reinterpret_cast<sunshine_receiver_test_focus>(GetProcAddress(receiver_module, "SunshineReceiverTestSetFocus"));
        receiver_poll = reinterpret_cast<sunshine_receiver_test_poll>(GetProcAddress(receiver_module, "SunshineReceiverTestPoll"));
        receiver_destroy = reinterpret_cast<sunshine_receiver_test_destroy>(GetProcAddress(receiver_module, "SunshineReceiverTestDestroy"));
        require(receiver_create && receiver_focus && receiver_poll && receiver_destroy, "Receiver test facade is missing C API exports");
      }
      fs::create_directories(directory / "effects");
      fs::create_directories(directory / "addons");
      fs::copy_file(runtime, directory / "dxgi.dll", fs::copy_options::overwrite_existing);
      fs::copy_file(addon, directory / "addons" / "SunshineSBSTest.addon64", fs::copy_options::overwrite_existing);
      if (native_only)
        require(fs::is_empty(directory / "effects"), "Native no-FX fixture requires an empty effects directory");
      else
        write_file(directory / "effects" / fixture_effect_name, fixture_shader_source());
      const std::string technique = std::string(fixture_technique_name) + "@" + fixture_effect_name;
      std::string preset_definitions, global_definitions;
#ifdef SUNSHINE_GAME_EXPORT_FIXTURE
      preset_definitions = "PreprocessorDefinitions=SUNSHINE_PANEL_PRESET_SENTINEL=27\n";
      global_definitions = "PreprocessorDefinitions=SUNSHINE_PANEL_GLOBAL_SENTINEL=19\n";
#endif
      const std::string selected = native_only ? "" : technique;
      write_file(directory / "preset.ini", preset_definitions + "Techniques=" + selected + "\nTechniqueSorting=" + selected +
        "\n[SunshineGame3D.fx]\nDepth_Adjustment=137.125\nDepth_Map_View=2\n[Other.fx]\nUntouchedPreference=42.125\n");
      const std::string native_config = std::string("[SUNSHINE_GAME3D]\nEnabled=") + (native_only ? "1\n" : "0\n") +
        (migrated_settings ? "Strength=37.25\nDepthView=2\n" : "");
      write_file(directory / "ReShade.ini", "[ADDON]\nAddonPath=.\\addons\nDisabledAddons=Generic Depth\n[GENERAL]\n" + global_definitions + "EffectSearchPaths=.\\effects\nPresetPath=.\\preset.ini\nPresetTransitionDuration=250\nPerformanceMode=0\nSkipLoadingDisabledEffects=0\nEffectCachePath=.\\cache\n[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n[STYLE]\nHdrOverlayBrightness=203\n" + native_config);
      SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", directory.c_str());

      // Create the receiver before loading ReShade so it is a separate native D3D11 device.
      wchar_t system[MAX_PATH];
      GetSystemDirectoryW(system, MAX_PATH);
      const HMODULE native_d3d11 = LoadLibraryW((fs::path(system) / "d3d11.dll").c_str());
      require(native_d3d11 != nullptr, "Could not load system D3D11");
      // Directly calling ReShade's export requires its native trampoline to exist.
      // Merely using decltype(D3D12CreateDevice) does not retain a linker import.
      require(LoadLibraryW((fs::path(system) / "d3d12.dll").c_str()) != nullptr, "Could not load system D3D12");
      const auto native_create = reinterpret_cast<create_device_t>(GetProcAddress(native_d3d11, "D3D11CreateDeviceAndSwapChain"));
      checked(native_create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, nullptr, nullptr, receiver.put(), nullptr, receiver_context.put()), "Native receiver creation");
      checked(receiver->QueryInterface(IID_PPV_ARGS(receiver5.put())), "Native receiver shared-fence interface");

      reshade_module = LoadLibraryW((directory / "dxgi.dll").c_str());
      require(reshade_module != nullptr, "Could not load official ReShade DLL");
      require(reshade::register_addon(GetModuleHandleW(nullptr), reshade_module), "Could not register runtime observer");
      reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init);
      reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(on_reload);
      reshade::register_event<reshade::addon_event::reshade_open_overlay>(on_overlay);
      reshade::register_event<reshade::addon_event::present>(on_begin);
      reshade::register_event<reshade::addon_event::reshade_render_technique>(on_technique);
      reshade::register_event<reshade::addon_event::reshade_present>(on_effect_present);
      reshade::register_event<reshade::addon_event::finish_present>(on_finish);

      WNDCLASSW window_class {};
      window_class.lpfnWndProc = DefWindowProcW;
      window_class.hInstance = GetModuleHandleW(nullptr);
      window_class.lpszClassName = L"SunshineReShadeD3D12PresentFixture";
      RegisterClassW(&window_class);
      RECT bounds {0, 0, source_width, source_height};
      require(AdjustWindowRectEx(&bounds, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW), "Could not size the hidden fixture client area");
      window = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, window_class.lpszClassName, L"Sunshine hidden D3D12 Present fixture", WS_OVERLAPPEDWINDOW, 0, 0, bounds.right - bounds.left, bounds.bottom - bounds.top, nullptr, nullptr, window_class.hInstance, nullptr);
      require(window != nullptr, "Could not create fixture window");
      using create_d3d12_t = decltype(&D3D12CreateDevice);
      using create_factory_t = decltype(&CreateDXGIFactory2);
      const auto wrapped_create = reinterpret_cast<create_d3d12_t>(GetProcAddress(reshade_module, "D3D12CreateDevice"));
      const auto wrapped_factory = reinterpret_cast<create_factory_t>(GetProcAddress(reshade_module, "CreateDXGIFactory2"));
      require(wrapped_create && wrapped_factory, "Official runtime is missing D3D12/DXGI proxy exports");
      checked(wrapped_create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(game.put())), "ReShade wrapped D3D12 device creation");
      com_ptr<IDXGIFactory4> factory;
      checked(wrapped_factory(0, IID_PPV_ARGS(factory.put())), "ReShade wrapped DXGI factory creation");
      D3D12_COMMAND_QUEUE_DESC queue_desc {};
      queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      checked(game->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue.put())), "Create wrapped D3D12 queue");

      const HMODULE addon_module = GetModuleHandleW((directory / "addons" / "SunshineSBSTest.addon64").c_str());
      require(addon_module != nullptr, "ReShade did not load the test-only exporter");
      set_foreground = reinterpret_cast<set_foreground_t>(GetProcAddress(addon_module, "SunshineSbsTestSetForeground"));
      set_overlay_patch = reinterpret_cast<void (*)(BOOL)>(GetProcAddress(addon_module, "SunshineSbsTestSetOverlayPatch"));
      require(set_overlay_patch != nullptr, "Runtime test add-on is missing its overlay patch hook");
      require(set_foreground != nullptr, "Test requires SunshineSBSTest.addon64 with its controlled observer");
#ifdef SUNSHINE_GAME_EXPORT_FIXTURE
      set_production_panel = reinterpret_cast<decltype(set_production_panel)>(GetProcAddress(addon_module, "SunshineSbsTestSetProductionPanel"));
      production_panel_draws = reinterpret_cast<decltype(production_panel_draws)>(GetProcAddress(addon_module, "SunshineSbsTestProductionPanelDraws"));
      require(set_production_panel && production_panel_draws, "Test add-on is missing the actual production panel draw hooks");
      query_control = reinterpret_cast<decltype(query_control)>(GetProcAddress(addon_module, "SunshineGame3DTestQuery"));
      edit_control_float = reinterpret_cast<decltype(edit_control_float)>(GetProcAddress(addon_module, "SunshineGame3DTestEditFloat"));
      edit_control_int = reinterpret_cast<decltype(edit_control_int)>(GetProcAddress(addon_module, "SunshineGame3DTestEditInt"));
      set_game_enabled = reinterpret_cast<decltype(set_game_enabled)>(GetProcAddress(addon_module, "SunshineGame3DTestSetEnabled"));
      query_automatic = reinterpret_cast<decltype(query_automatic)>(GetProcAddress(addon_module, "SunshineGame3DTestQueryAutomatic"));
      recalibrate = reinterpret_cast<decltype(recalibrate)>(GetProcAddress(addon_module, "SunshineGame3DTestRecalibrate"));
      require(query_control && edit_control_float && edit_control_int && set_game_enabled && query_automatic && recalibrate,
        "Test add-on is missing compact-controls/automatic adapter hooks");
#endif
      set_foreground(window);

      DXGI_SWAP_CHAIN_DESC1 desc {};
      desc.Width = source_width;
      desc.Height = source_height;
      desc.Format = color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : color == 3 ? DXGI_FORMAT_R10G10B10A2_UNORM :
                                                                               DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count = 1;
      desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      desc.BufferCount = 2;
      desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      com_ptr<IDXGISwapChain1> swapchain1;
      checked(factory->CreateSwapChainForHwnd(queue.p, window, &desc, nullptr, nullptr, swapchain1.put()), "ReShade wrapped D3D12 swapchain creation");
      checked(swapchain1->QueryInterface(IID_PPV_ARGS(swapchain.put())), "Swapchain color-space interface");
      checked(swapchain->SetColorSpace1(color == 2 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : color == 3 ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 :
                                                                                                            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),
              "Set game color space");
      checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.put())), "Create game command allocator");
      checked(game->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(commands.put())), "Create game command list");
      checked(commands->Close(), "Close initial game command list");
      D3D12_DESCRIPTOR_HEAP_DESC target_desc {};
      target_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
      target_desc.NumDescriptors = 2;
      checked(game->CreateDescriptorHeap(&target_desc, IID_PPV_ARGS(targets.put())), "Create backbuffer RTV heap");
      target_stride = game->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
      auto target_handle = targets->GetCPUDescriptorHandleForHeapStart();
      for (unsigned index = 0; index < 2; ++index) {
        checked(swapchain->GetBuffer(index, IID_PPV_ARGS(backbuffers[index].put())), "Get D3D12 game backbuffer");
        game->CreateRenderTargetView(backbuffers[index].p, nullptr, target_handle);
        target_handle.ptr += target_stride;
      }
      checked(game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(completion.put())), "Create game submission fence");
      completion_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      require(completion_event != nullptr, "Create game completion event failed");
      const auto mapping_name = std::wstring(wire::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, mapping_name.c_str());
      require(mapping != nullptr, "Test exporter did not create its mapping");
      shared = static_cast<wire::shared_state_t *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(wire::shared_state_t)));
      require(shared != nullptr, "Could not map actual exporter publication");
      if (receiver_module) {
        require(GetClientRect(window, &source_rect), "Could not query fixture client rectangle");
        POINT origin {};
        require(ClientToScreen(window, &origin), "Could not query fixture client origin");
        OffsetRect(&source_rect, origin.x, origin.y);
        require(source_rect.right - source_rect.left == source_width && source_rect.bottom - source_rect.top == source_height, "Fixture client area differs from its logical source size");
        receiver_state = receiver_create(receiver.p, receiver_context.p, window, GetCurrentProcessId(), &source_rect);
        require(receiver_state != nullptr, "Could not initialize production receiver");
      } else {
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&shared->consumer_nonce), 0x12345678);
      }
      require(!IsWindowVisible(window), "Fixture unexpectedly became visible");
      std::puts("Actual D3D12 game runtime uses hidden window and test-only foreground observer");
    }

    void step() {
      MSG message;
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      require(IsWindow(window) && !IsWindowVisible(window), "Fixture window disappeared or became visible");
      checked(allocator->Reset(), "Reset game allocator");
      checked(commands->Reset(allocator.p, nullptr), "Reset game command list");
      const UINT index = swapchain->GetCurrentBackBufferIndex();
      D3D12_RESOURCE_BARRIER barrier {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Transition.pResource = backbuffers[index].p;
      barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
      commands->ResourceBarrier(1, &barrier);
      auto target = targets->GetCPUDescriptorHandleForHeapStart();
      target.ptr += index * target_stride;
      const auto background = native_background();
      commands->ClearRenderTargetView(target, background.data(), 0, nullptr);
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
      commands->ResourceBarrier(1, &barrier);
      checked(commands->Close(), "Close game command list");
      ID3D12CommandList *submitted[] {commands.p};
      queue->ExecuteCommandLists(1, submitted);
      checked(swapchain->Present(0, 0), "Natural DXGI Present");
      checked(queue->Signal(completion.p, ++completion_value), "Signal completed game Present");
      checked(completion->SetEventOnCompletion(completion_value, completion_event), "Observe game Present completion");
      require(WaitForSingleObject(completion_event, 2000) == WAIT_OBJECT_0, "D3D12 game/Present work exceeded two seconds");
      require(!observation.order_failed && observation.stage == 4, "Actual presentation event ordering differed from exporter assumptions");
      require(static_cast<unsigned>(observation.color) == expected_color, "Actual ReShade game color did not match requested transfer");
      Sleep(3);
    }

    std::uint64_t wait_pixels(std::uint64_t after_generation = 0, std::uint64_t same_generation = 0, std::uint64_t after_sequence = 0) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
      while (std::chrono::steady_clock::now() < deadline) {
        step();
        sunshine_receiver_test_frame received;
        int received_status = 0;
        if (receiver_state) {
          // Production code owns the handshake, slots, fence and private GPU copy.
          received_status = receiver_poll(receiver_state, &received);
          require(received_status >= 0, "Production receiver poll failed");
        }
        wire::metadata_t metadata;
        if (!snapshot(*shared, metadata) || !wire::valid_metadata(metadata) || metadata.generation <= after_generation) {
          continue;
        }
        require(!same_generation || metadata.generation == same_generation, "Overlay or uniform edit replaced the resource generation");
        require(metadata.source_width == source_width && metadata.source_height == source_height && metadata.window == reinterpret_cast<std::uint64_t>(window), "Exporter source identity/dimensions incorrect");
        require(metadata.color_transfer == (expected_color == 1 ? wire::transfer::srgb : wire::transfer::scrgb), "Exporter declared wrong transfer");
        require(metadata.dxgi_format == (expected_color == 1 ? 24u : 10u), "Exporter declared wrong resource format");
        if (receiver_state) {
          if (metadata.accepted_consumer_nonce != read64(shared->consumer_nonce)) {
            // Reattachment just requested a new producer ring. It needs another Present.
            continue;
          }
          std::uint64_t newest_sequence = 0;
          for (auto &slot : shared->slots) {
            if (wire::control_generation(read64(slot.control)) == metadata.generation) {
              newest_sequence = std::max(newest_sequence, read64(slot.sequence));
            }
          }
          if (!newest_sequence || newest_sequence <= after_sequence) {
            continue;
          }
          // Finish the already-submitted private copy without generating another Present.
          // A newer pending copy may legitimately replace an older cached frame here.
          const auto copy_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
          while ((!received_status || received.sequence != newest_sequence) && std::chrono::steady_clock::now() < copy_deadline) {
            received_status = receiver_poll(receiver_state, &received);
            require(received_status >= 0, "Production receiver completion poll failed");
            Sleep(1);
          }
          require(received_status == 1 && received.sequence == newest_sequence && received.texture && received.timestamp_ns, "Production receiver did not retire the latest D3D12 publication");
          require(received.linear == (expected_color != 1), "Production receiver lost the declared source transfer");
          if (!inspect_texture(receiver.p, receiver_context.p, received.texture, metadata.dxgi_format)) {
            continue;
          }
          last_sequence = received.sequence;
          last_timestamp = received.timestamp_ns;
          last_received = received;
          std::printf("production receiver pixels PASS color=%u generation=%llu sequence=%llu\n", expected_color, static_cast<unsigned long long>(metadata.generation), static_cast<unsigned long long>(received.sequence));
          return metadata.generation;
        }
        com_ptr<ID3D11Fence> fence;
        checked(receiver5->OpenSharedFence(reinterpret_cast<HANDLE>(metadata.ready_fence_handle), IID_PPV_ARGS(fence.put())), "Import exporter ready fence");
        for (unsigned index = 0; index < wire::slot_count; ++index) {
          auto &slot = shared->slots[index];
          const auto expected = wire::slot_control(metadata.generation, wire::slot_state::ready);
          if (read64(slot.control) != expected || read64(slot.sequence) <= after_sequence || fence->GetCompletedValue() < read64(slot.sequence)) {
            continue;
          }
          if (static_cast<std::uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), wire::slot_control(metadata.generation, wire::slot_state::reading), expected)) != expected) {
            continue;
          }
          require(slot.sequence && slot.qpc, "Exported frame missing sequence/source timestamp");
          const bool pixels_match = inspect_pixels(metadata, index);
          InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), wire::slot_control(metadata.generation, wire::slot_state::free), wire::slot_control(metadata.generation, wire::slot_state::reading));
          if (!pixels_match) {
            continue;
          }
          last_sequence = slot.sequence;
          last_timestamp = slot.qpc;
          std::printf("pixels PASS color=%u generation=%llu sequence=%llu\n", expected_color, static_cast<unsigned long long>(metadata.generation), static_cast<unsigned long long>(slot.sequence));
          return metadata.generation;
        }
      }
      throw std::runtime_error("Timed out waiting for the actual runtime/exporter to publish pixels");
    }

    void require_receiver_inactive(const char *message) {
      if (receiver_state) {
        sunshine_receiver_test_frame frame;
        require(receiver_poll(receiver_state, &frame) == 0, message);
      }
    }

    bool inspect_pixels(const wire::metadata_t &metadata, unsigned index) {
      com_ptr<ID3D11Texture2D> texture;
      checked(receiver5->OpenSharedResource1(reinterpret_cast<HANDLE>(metadata.texture_handles[index]), IID_PPV_ARGS(texture.put())), "Import exporter texture");
      return inspect_texture(receiver.p, receiver_context.p, texture.p, metadata.dxgi_format);
    }

    bool inspect_texture(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Texture2D *texture, unsigned format) {
      com_ptr<ID3D11Texture2D> staging;
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      require(desc.Width == source_width * 2 && desc.Height == source_height && static_cast<unsigned>(desc.Format) == format, "Actual runtime texture dimensions/format differ from export contract");
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      checked(device->CreateTexture2D(&desc, nullptr, staging.put()), "Create test readback");
      com_ptr<ID3D11Query> query;
      D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0};
      checked(device->CreateQuery(&query_desc, query.put()), "Create test completion query");
      context->CopyResource(staging.p, texture);
      context->End(query.p);
      context->Flush();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      HRESULT result;
      while ((result = context->GetData(query.p, nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE && std::chrono::steady_clock::now() < deadline) {
        Sleep(1);
      }
      require(result == S_OK, "Test readback did not finish within the bound");
      D3D11_MAPPED_SUBRESOURCE mapped {};
      checked(context->Map(staging.p, 0, D3D11_MAP_READ, 0, &mapped), "Map completed test readback");
      const bool correct = native_only ? inspect_native_pixels(mapped.pData, mapped.RowPitch, format) :
        sunshine_sbs_overlay_pixels::check(mapped.pData, mapped.RowPitch, format, source_width, source_height, expect_overlay, expect_overlay && set_overlay_patch, expected_gain);
      context->Unmap(staging.p, 0);
      return correct;
    }

    void set_gain(float value) {
      if (native_only) { expected_gain = value; return; }
      api::effect_uniform_variable gain {};
      observation.runtime->enumerate_uniform_variables(fixture_effect_name, [&](api::effect_runtime *runtime, api::effect_uniform_variable variable) {
        char name[128] {};
        runtime->get_uniform_variable_name(variable, name);
        if (std::strcmp(name, "StereoGain") == 0) {
          gain = variable;
        }
      });
      require(gain.handle != 0, "Actual effect is missing its live StereoGain control");
      observation.runtime->set_uniform_value_float(gain, &value, 1);
      expected_gain = value;
    }

    std::array<float, 4> native_background() const {
      std::array<float, 4> color = native_only && expected_color == 2 ?
        std::array<float, 4>{4.f, .5f, -.125f, 1.f} : std::array<float, 4>{.125f, .25f, .375f, 1.f};
      if (native_only) for (unsigned channel = 0; channel < 3; ++channel) color[channel] *= expected_gain;
      return color;
    }

    bool inspect_native_pixels(const void *data, unsigned pitch, unsigned format) const {
      auto expected = native_background();
      if (expected_color == 1) {
        for (unsigned c = 0; c < 3; ++c) expected[c] = std::round(expected[c] * 255.f) / 255.f;
      } else if (expected_color == 3) {
        std::array<double, 3> linear {};
        for (unsigned c = 0; c < 3; ++c) {
          const double code = std::round(expected[c] * 1023.0) / 1023.0;
          const double p = std::pow(code, 32.0 / 2523.0);
          linear[c] = 125.0 * std::pow(std::max(p - 3424.0 / 4096.0, 0.0) /
            (2413.0 / 128.0 - 2392.0 / 128.0 * p), 16384.0 / 2610.0);
        }
        expected[0] = float(1.6604910021 * linear[0] - .5876411388 * linear[1] - .0728498633 * linear[2]);
        expected[1] = float(-.1245504745 * linear[0] + 1.1328998971 * linear[1] - .0083494226 * linear[2]);
        expected[2] = float(-.0181507634 * linear[0] - .1005788980 * linear[1] + 1.1187296614 * linear[2]);
      }
      const auto pixel = [&](unsigned x, unsigned y, unsigned c) {
        const auto row = static_cast<const std::uint8_t *>(data) + y * pitch;
        return format == 24 ? float((reinterpret_cast<const std::uint32_t *>(row)[x] >> (10 * c)) & 1023) / 1023.f :
          sunshine_sbs_overlay_pixels::half(reinterpret_cast<const std::uint16_t *>(row)[x * 4 + c]);
      };
      const float tolerance = format == 24 ? .004f : .012f;
      const float white = format == 24 ? 1.f : 203.f / 80.f;
      const unsigned probes[][2] {{source_width - 200, source_height - 64}, {source_width - 96, source_height - 144},
        {source_width - 96, source_height - 80}};
      for (unsigned eye = 0; eye < 2; ++eye) {
        for (unsigned probe = 0; probe < 3; ++probe) {
          const float alpha = !expect_overlay || !set_overlay_patch || probe == 0 ? 0.f : probe == 1 ? 1.f : 128.f / 255.f;
          for (unsigned c = 0; c < 3; ++c) {
            const float wanted = format == 24 && alpha > 0 ?
              sunshine_sbs_overlay_pixels::encode_srgb(sunshine_sbs_overlay_pixels::decode_srgb(expected[c]) * (1 - alpha) + alpha) :
              expected[c] * (1 - alpha) + white * alpha;
            const float actual = pixel(eye * source_width + probes[probe][0], probes[probe][1], c);
            if (!std::isfinite(actual) || std::abs(actual - wanted) > tolerance) {
              std::printf("native no-FX pixel mismatch eye=%u probe=%u channel=%u actual=%.6g expected=%.6g overlay=%u\n",
                eye, probe, c, actual, wanted, unsigned(expect_overlay));
              return false;
            }
          }
        }
        unsigned changed = 0;
        for (unsigned y = 96; y < source_height - 32; y += 24) for (unsigned x = 24; x < source_width / 3 - 24; x += 24) {
          bool different = false;
          for (unsigned c = 0; c < 3; ++c) {
            const float actual = pixel(eye * source_width + x, y, c);
            if (!std::isfinite(actual)) return false;
            different |= std::abs(actual - expected[c]) > .05f;
          }
          changed += different;
        }
        if ((expect_overlay && changed < 20) || (!expect_overlay && changed != 0)) return false;
      }
      return true;
    }

    bool set_test_focus(bool focused) {
      set_foreground(focused ? window : nullptr);
      if (receiver_state) {
        receiver_focus(receiver_state, focused);
      }
      return true;
    }

#ifdef SUNSHINE_GAME_EXPORT_FIXTURE
    std::uint64_t exercise_game_controls() {
      auto *runtime = observation.runtime;
      reshade::set_config_value(runtime, "OVERLAY", "AutoSavePreset", false);
      const auto query = [&](unsigned index) {
        unsigned flags = 0;
        float value = NAN;
        require(query_control(runtime, index, &flags, &value) && flags == 3 && std::isfinite(value),
          "Native controls must remain available and editable without shader/preset state");
        return value;
      };
      const auto read_file = [](const fs::path &path) {
        std::ifstream in(path, std::ios::binary);
        require(in.good(), "Could not read fixture configuration");
        return std::string(std::istreambuf_iterator<char>(in), {});
      };
      const auto ini_value = [](const std::string &text, const std::string &section, const std::string &key) {
        std::istringstream input(text);
        std::string line, current_section;
        while (std::getline(input, line)) {
          if (!line.empty() && line.back() == '\r') line.pop_back();
          if (!line.empty() && line.front() == '[' && line.back() == ']') current_section = line.substr(1, line.size() - 2);
          else if (current_section == section) {
            const auto equals = line.find('=');
            if (equals != std::string::npos && line.substr(0, equals) == key) return line.substr(equals + 1);
          }
        }
        return std::string{};
      };
      const auto config_value = [&](const char *key) {
        return ini_value(read_file(output_directory / "ReShade.ini"), "SUNSHINE_GAME3D", key);
      };
      const auto wait_ini = [&](const auto &matches, const char *message) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        do { step(); if (matches()) return; } while (std::chrono::steady_clock::now() < deadline);
        require(false, message);
      };
      const auto reload = [&] {
        const auto previous = observation.reloads;
        runtime->reload_effect_next_frame(nullptr);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
        do { step(); } while (observation.reloads == previous && std::chrono::steady_clock::now() < deadline);
        require(observation.reloads > previous, "ReShade did not finish the controls-independence reload");
      };
      const float initial_strength = migrated_settings ? 37.25f : 50.f;
      const int initial_view = migrated_settings ? 2 : 0;
      require(query(0) == initial_strength && query(1) == initial_view,
        "Native defaults/migrated config were replaced by obsolete effect preset values");
      unsigned flags = 0;
      float value = NAN;
      require(!query_control(runtime, 2, &flags, &value) && !edit_control_float(runtime, 2, 1.f) &&
        !edit_control_int(runtime, 2, 1) && !edit_control_float(runtime, 1, .5f) && !edit_control_int(runtime, 0, 1) &&
        !edit_control_float(runtime, 0, NAN) && !edit_control_int(runtime, 1, 3),
        "Invalid, removed or mismatched native controls are accessible");
      const auto preset = output_directory / "preset.ini";
      const auto original_preset = read_file(preset);
      for (unsigned i = 0; i < 4; ++i) step();
      require(query(0) == initial_strength && query(1) == initial_view && read_file(preset) == original_preset,
        "Passive native controls modified saved settings");
      require(edit_control_float(runtime, 0, 37.250123f), "Native strength edit failed");
      const int changed_view = initial_view == 2 ? 1 : 2;
      require(edit_control_int(runtime, 1, changed_view), "Native preview edit failed");
      require(query(0) == 37.250123f && query(1) == changed_view, "Native scalar edits lost precision or changed another field");
      wait_ini([&] {
        const auto strength = config_value("Strength"), view = config_value("DepthView");
        return !strength.empty() && std::stof(strength) == 37.250123f && view == std::to_string(changed_view);
      }, "Native edits did not save automatically while shader Auto Save was off");
      require(read_file(preset) == original_preset, "Native config edit rewrote a shader preset");

      const std::string technique = native_only ? "" : std::string(fixture_technique_name) + "@" + fixture_effect_name;
      const auto target = output_directory / "unrelated-preset.ini";
      const std::string target_text = "PreprocessorDefinitions=UNRELATED_PRESET=13\nTechniques=" + technique +
        "\n[SunshineGame3D.fx]\nDepth_Adjustment=91\nDepth_Map_View=0\nStereoGain=1\n[Other.fx]\nUntouchedPreference=99\n";
      write_file(target, target_text);
      runtime->set_current_preset_path(target.u8string().c_str());
      require(edit_control_float(runtime, 0, 38.25f) && query(1) == changed_view,
        "Shader preset transition blocked native controls or replaced native preview");
      for (unsigned i = 0; i < 4; ++i) step();
      runtime->set_current_preset_path(preset.u8string().c_str());
      require(query(0) == 38.25f && query(1) == changed_view,
        "Native settings changed when switching shader presets");
      reshade::set_config_value(runtime, "GENERAL", "PerformanceMode", true);
      reload();
      require(edit_control_float(runtime, 0, 44.5f) && query(0) == 44.5f && query(1) == changed_view,
        "Shader Performance Mode or reload invalidated native controls");
      reshade::set_config_value(runtime, "GENERAL", "PerformanceMode", false);
      reload();
      require(query(0) == 44.5f && query(1) == changed_view, "Effect reload lost native settings");
      require(read_file(target) == target_text, "Native controls rewrote another shader preset");

      unsigned automatic_flags = 0;
      require(query_automatic(runtime, &automatic_flags) && bool(automatic_flags & 1) == native_only &&
        !(automatic_flags & 4), "No-depth fixture claims stereo readiness or incorrect enabled state");
      require(set_game_enabled(runtime, native_only ? FALSE : TRUE), "Native enabled toggle failed");
      require(query_automatic(runtime, &automatic_flags) && bool(automatic_flags & 1) != native_only,
        "Native enabled toggle did not reach the settings used by rendering");
      if (native_only) {
        step();
        wire::metadata_t metadata;
        require(snapshot(*shared, metadata) && metadata.generation == 0, "Native disable retained a stale SBS publication");
        require_receiver_inactive("Receiver retained native stereo while native rendering was disabled");
      }
      require(set_game_enabled(runtime, native_only ? TRUE : FALSE) &&
        !set_game_enabled(runtime, native_only ? TRUE : FALSE), "Native enabled reset failed or repeated a no-op save");
      require(edit_control_float(runtime, 0, 50.f) && !edit_control_float(runtime, 0, 50.f), "Native strength reset failed or rewrote its default");
      require(query(1) == changed_view && edit_control_int(runtime, 1, 0) && !edit_control_int(runtime, 1, 0) &&
        query(0) == 50.f, "Native reset changed another control or rewrote an unchanged default");
      wait_ini([&] {
        const auto strength = config_value("Strength");
        return !strength.empty() && std::stof(strength) == 50.f && config_value("DepthView") == "0" &&
          config_value("Enabled") == (native_only ? "1" : "0");
      }, "Native independent resets/enabled state did not reach the configuration file");
      require(ini_value(read_file(preset), "Other.fx", "UntouchedPreference") == "42.125" &&
        ini_value(read_file(output_directory / "ReShade.ini"), "GENERAL", "PreprocessorDefinitions") == "SUNSHINE_PANEL_GLOBAL_SENTINEL=19",
        "Native settings changed unrelated preset or global configuration");
      std::puts("PASS native Game 3D controls: defaults/migration, exact config edits, independent resets, enabled state, automatic saving, preset/Performance Mode/reload independence");
      return wait_pixels();
    }

    void exercise_production_panel(std::uint64_t generation) {
      for (unsigned panel_mode : {1u, 2u}) {
        const unsigned previous_draws = production_panel_draws();
        set_production_panel(panel_mode);
        for (unsigned frame = 0; frame < 3; ++frame) step();
        require(production_panel_draws() >= previous_draws + 2, "Actual production Game 3D panel did not complete multiple draws");
        wait_pixels(0, generation, last_sequence);
      }
      set_production_panel(0);
      std::puts("PASS production Game 3D panel: collapsed and expanded native controls, depth actions and visible calibration status");
    }
#endif

    std::uint64_t exercise_overlay(std::uint64_t generation) {
      if (set_overlay_patch) {
        set_overlay_patch(TRUE);
      } else {
        std::puts("SKIP deterministic GUI patch checks: they require the distinct runtime-test add-on");
      }
      expect_overlay = true;
      require(observation.runtime->open_overlay(true, api::input_source::keyboard), "Official ReShade overlay did not open");
      auto prior_sequence = last_sequence;
      auto prior_timestamp = last_timestamp;
      wait_pixels(0, generation, prior_sequence);
      require(last_timestamp > prior_timestamp, "Opening the overlay did not produce a fresh source frame");
#ifdef SUNSHINE_GAME_EXPORT_FIXTURE
      generation = exercise_game_controls();
      // Home-tab rendering and scalar control adapters do not exercise the
      // add-on's actual layout calls across the MinGW/MSVC ImGui ABI boundary.
      // Draw the production callback with both collapsed and expanded sections.
      exercise_production_panel(generation);
#endif

      // The real uniform is also a visible native ReShade slider. Changing it while the
      // menu is open must update both authored eyes without changing the IPC generation.
      prior_sequence = last_sequence;
      prior_timestamp = last_timestamp;
      set_gain(0.5f);
      wait_pixels(0, generation, prior_sequence);
      require(last_timestamp > prior_timestamp, "Live overlay adjustment reused the old source timestamp");

      // GUI activity never replaces proof that the authored technique actually executed.
      observation.runtime->set_effects_state(false);
      step();
      wire::metadata_t metadata;
      require(snapshot(*shared, metadata) && !metadata.generation, "Visible GUI kept export active after effects disable");
      require_receiver_inactive("Receiver retained stereo with GUI visible but effects disabled");
      observation.runtime->set_effects_state(true);
      generation = wait_pixels(generation);
      prior_sequence = last_sequence;
      set_gain(1.0f);
      wait_pixels(0, generation, prior_sequence);

      if (set_test_focus(false)) {
        step();
        require(snapshot(*shared, metadata) && !metadata.generation, "Open GUI bypassed the publisher focus gate");
        require_receiver_inactive("Receiver retained stereo after focus loss with GUI open");
        set_test_focus(true);
        generation = wait_pixels(generation);
      }

      const auto previous_reloads = observation.reloads;
      const auto previous_techniques = observation.techniques;
      observation.runtime->reload_effect_next_frame(nullptr);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
      do {
        step();
        if (observation.techniques == previous_techniques) {
          require(snapshot(*shared, metadata) && !metadata.generation, "Reload retained an old stereo publication before a new technique executed");
          require_receiver_inactive("Receiver retained stale stereo during overlay reload");
        }
      } while ((observation.reloads == previous_reloads || observation.techniques == previous_techniques) && std::chrono::steady_clock::now() < deadline);
      require(observation.reloads > previous_reloads && observation.techniques > previous_techniques, "Official runtime did not complete effect reload");
      generation = wait_pixels(generation);

      prior_sequence = last_sequence;
      if (set_overlay_patch) {
        set_overlay_patch(FALSE);
      }
      expect_overlay = false;
      require(observation.runtime->open_overlay(false, api::input_source::keyboard), "Official ReShade overlay did not close");
      require(observation.overlay_opens == 1 && observation.overlay_closes == 1, "Actual overlay transition callbacks were not delivered");
      wait_pixels(0, generation, prior_sequence);
      std::puts(set_overlay_patch ? "PASS actual GUI pixels in both eyes: native controls, opaque/translucent patches, live gain, close cleanup and overlay reload" : "PASS actual GUI pixels: native controls, distinct stereo backgrounds, live gain, close cleanup and overlay reload; deterministic patch checks skipped");
      return generation;
    }

    void inspect_runtime_texture() {
      api::effect_texture_variable variable {};
      observation.runtime->enumerate_texture_variables(fixture_effect_name, [&](api::effect_runtime *runtime, api::effect_texture_variable texture) {
        int marker = 0;
        if (runtime->get_annotation_int_from_texture_variable(texture, "sunshine_sbs_export", &marker, 1) && marker == 1) {
          variable = texture;
        }
      });
      require(variable.handle != 0, "Actual runtime did not expose annotated export texture");
      unsigned source_color = 0;
      require(observation.runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_color_space", &source_color, 1) && source_color == expected_color, "Actual FX annotation does not match native swapchain color");
      unsigned width = 0, height = 0;
      char layout[32] {}, transfer[32] {};
      require(observation.runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_width", &width, 1) && width == source_width && observation.runtime->get_annotation_uint_from_texture_variable(variable, "sunshine_sbs_source_height", &height, 1) && height == source_height && observation.runtime->get_annotation_string_from_texture_variable(variable, "sunshine_sbs_layout", layout) && std::strcmp(layout, "sbs_lr") == 0 && observation.runtime->get_annotation_string_from_texture_variable(variable, "sunshine_sbs_color_space", transfer) && std::strcmp(transfer, expected_color == 1 ? "srgb" : "scrgb") == 0, "Actual FX export layout, dimensions or transfer annotation differs from the contract");
      api::resource_view view {};
      observation.runtime->get_texture_binding(variable, &view, nullptr);
      require(view.handle != 0, "Actual runtime export texture has no shader resource view");
      // The generated pinned SDK adapter supplies MSVC's hidden result pointer.
      const auto sdk_resource = observation.runtime->get_device()->get_resource_from_view(view);
      require(sdk_resource.handle != 0, "Actual runtime did not expose the D3D12 export resource");
      const auto desc = reinterpret_cast<ID3D12Resource *>(sdk_resource.handle)->GetDesc();
      require(desc.Width == source_width * 2 && desc.Height == source_height && static_cast<unsigned>(desc.Format) == (expected_color == 1 ? 24u : 10u), "Actual D3D12 FX resource differs from the export contract");
      std::puts("Actual D3D12 FX annotations and native texture geometry PASS");
    }

#ifdef SUNSHINE_NATIVE_EXPORT_FIXTURE
    api::effect_uniform_variable native_uniform(const char *expected) {
      api::effect_uniform_variable result {};
      observation.runtime->enumerate_uniform_variables(fixture_effect_name, [&](api::effect_runtime *runtime, api::effect_uniform_variable variable) {
        char name[128] {};
        runtime->get_uniform_variable_name(variable, name);
        if (std::strcmp(name, expected) == 0) result = variable;
      });
      require(result.handle != 0, "Native transport fixture is missing a calibration ABI uniform");
      return result;
    }

    void exercise_native_preparation(std::uint64_t generation) {
      const auto ready = native_uniform("Sunshine_DepthReady"), calibrated = native_uniform("Sunshine_Calibrated");
      const auto anchor = native_uniform("Sunshine_RawAnchor"), gain = native_uniform("Sunshine_RawGain");
      const auto rect = native_uniform("Sunshine_DepthRect"), direction = native_uniform("DepthDirection");
      observation.runtime->set_uniform_value_bool(ready, true);
      observation.runtime->set_uniform_value_bool(calibrated, true);
      const float poison = 123.f, invalid_rect[] {-1.f, -1.f, -1.f, -1.f};
      observation.runtime->set_uniform_value_float(anchor, &poison, 1);
      observation.runtime->set_uniform_value_float(gain, &poison, 1);
      observation.runtime->set_uniform_value_float(rect, invalid_rect, 4);
      const auto prior_sequence = last_sequence;
      wait_pixels(0, generation, prior_sequence);

      // No depth buffer is rendered by this transport fixture. A fresh real
      // publisher callback must overwrite every poisoned value before publication.
      bool actual_ready = true, actual_calibrated = true;
      float actual_anchor = poison, actual_gain = poison, actual_rect[4] {};
      int actual_direction = -1;
      observation.runtime->get_uniform_value_bool(ready, &actual_ready, 1);
      observation.runtime->get_uniform_value_bool(calibrated, &actual_calibrated, 1);
      observation.runtime->get_uniform_value_float(anchor, &actual_anchor, 1);
      observation.runtime->get_uniform_value_float(gain, &actual_gain, 1);
      observation.runtime->get_uniform_value_float(rect, actual_rect, 4);
      observation.runtime->get_uniform_value_int(direction, &actual_direction, 1);
      require(!actual_ready && !actual_calibrated && actual_anchor == 0.f && actual_gain == 0.f &&
                actual_rect[0] == 1.f && actual_rect[1] == 1.f && actual_rect[2] == 0.f && actual_rect[3] == 0.f && actual_direction == 0,
              "Native publisher did not prepare fresh, complete no-depth state before controlled transport publication");
      std::puts("PASS native publisher refreshes every calibration input before publication; controlled paint is transport-only, not production geometry");
    }
#endif

#ifdef SUNSHINE_GAME_EXPORT_FIXTURE
    void run_native_only() {
      auto generation = wait_pixels();
      require(observation.runtime && observation.techniques == 0 && fs::is_empty(output_directory / "effects"),
        "Native no-FX renderer depended on an installed effect");
      unsigned techniques = 0;
      observation.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++techniques; });
      require(techniques == 0, "Native no-FX fixture unexpectedly loaded a technique");
      generation = exercise_game_controls();

      // A global effect toggle must not turn off add-on-owned rendering.
      observation.runtime->set_effects_state(false);
      wait_pixels(0, generation, last_sequence);
      observation.runtime->set_effects_state(true);
      require(edit_control_float(observation.runtime, 0, 0.f), "Native zero-strength control failed");
      wait_pixels(0, generation, last_sequence);
      require(edit_control_float(observation.runtime, 0, 50.f), "Native strength restoration failed");

      set_overlay_patch(TRUE);
      expect_overlay = true;
      require(observation.runtime->open_overlay(true, api::input_source::keyboard), "Native no-FX overlay failed to open");
      wait_pixels(0, generation, last_sequence);
      exercise_production_panel(generation);
      set_gain(.5f);
      wait_pixels(0, generation, last_sequence);
      set_gain(1.f);
      wait_pixels(0, generation, last_sequence);
      const auto previous_reloads = observation.reloads;
      observation.runtime->reload_effect_next_frame(nullptr);
      const auto reload_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
      do { step(); } while (observation.reloads == previous_reloads && std::chrono::steady_clock::now() < reload_deadline);
      require(observation.reloads > previous_reloads && observation.techniques == 0,
        "No-FX reload failed or introduced a shader technique");
      generation = wait_pixels(0, 0, last_sequence);
      set_overlay_patch(FALSE);
      expect_overlay = false;
      require(observation.runtime->open_overlay(false, api::input_source::keyboard), "Native no-FX overlay failed to close");
      wait_pixels(0, generation, last_sequence);

      set_test_focus(false);
      step();
      wire::metadata_t metadata;
      require(snapshot(*shared, metadata) && !metadata.generation, "Native no-FX output bypassed focus loss");
      require_receiver_inactive("Receiver retained native output after focus loss");
      set_test_focus(true);
      generation = wait_pixels(generation);
      if (receiver_state) {
        const auto previous_nonce = read64(shared->consumer_nonce);
        receiver_destroy(receiver_state);
        receiver_state = receiver_create(receiver.p, receiver_context.p, window, GetCurrentProcessId(), &source_rect);
        require(receiver_state != nullptr, "Could not restart native production receiver");
        wait_pixels(generation);
        require(read64(shared->consumer_nonce) != previous_nonce, "Native receiver restart reused its old nonce");
      }
      require(observation.presents == observation.finishes && observation.techniques == 0,
        "Native no-FX presentation lifecycle is incomplete");
      std::printf("PASS native add-on-only D3D12 color=%u: no FX files or techniques; native source/SDR/HDR pixels, controls/config, zero strength, enable/focus recovery, actual panel/overlay, reload%s\n",
        expected_color, receiver_state ? ", production receiver restart" : "");
    }
#endif

    void run() {
#ifdef SUNSHINE_GAME_EXPORT_FIXTURE
      if (native_only) { run_native_only(); return; }
#endif
      auto generation = wait_pixels();
      if (receiver_state) {
        sunshine_receiver_test_frame retained;
        require(receiver_poll(receiver_state, &retained) == 1 && retained.sequence == last_received.sequence && retained.timestamp_ns == last_received.timestamp_ns, "A retained frame changed its source sequence or timestamp without another game Present");
      }
      require(observation.runtime && observation.techniques, "Official runtime did not execute the fixture technique");
      inspect_runtime_texture();
#ifdef SUNSHINE_NATIVE_EXPORT_FIXTURE
      exercise_native_preparation(generation);
#endif
      observation.runtime->set_effects_state(false);
      step();
      wire::metadata_t metadata;
      require(snapshot(*shared, metadata) && metadata.generation == 0, "Effect disable retained a stale publication");
      require_receiver_inactive("Production receiver retained stereo after effects disable");
      observation.runtime->set_effects_state(true);
      generation = wait_pixels(generation);
      generation = exercise_overlay(generation);
      if (receiver_state) {
        const auto previous_nonce = read64(shared->consumer_nonce);
        receiver_destroy(receiver_state);
        receiver_state = receiver_create(receiver.p, receiver_context.p, window, GetCurrentProcessId(), &source_rect);
        require(receiver_state != nullptr, "Could not restart production receiver");
        wait_pixels(generation);
        require(read64(shared->consumer_nonce) != previous_nonce, "Receiver restart reused the old consumer nonce");
      }
      require(observation.presents == observation.finishes, "D3D12 Present and finish callback counts differ");
      std::printf("PASS actual ReShade6.8 D3D12 color=%u: present=%u technique=%u finish=%u reload=%u; actual exporter pixels/fence, effect disable, overlay/reload, controlled focus loss/recovery%s\n", expected_color, observation.presents, observation.techniques, observation.finishes, observation.reloads, receiver_state ? "; production receiver private pixels, cached timestamps and restart" : "");
    }
  };
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5 && argc != 6) {
    std::fprintf(stderr, "usage: reshade_runtime_present_d3d12_test <official-ReShade64.dll> <SunshineSBSTest.addon64> <isolated-output-directory> <srgb|scrgb|pq> [production-receiver-test.dll]\n");
    return 2;
  }
  // A broken presentation hook must never leave the synthetic window running indefinitely.
  std::thread([] {
    Sleep(45000);
    std::fputs("FAIL runtime fixture watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 3);
  }).detach();
  try {
    const unsigned color = std::strcmp(argv[4], "srgb") == 0 ? 1 : std::strcmp(argv[4], "scrgb") == 0 ? 2 :
                                                                 std::strcmp(argv[4], "pq") == 0      ? 3 :
                                                                                                        0;
    require(color != 0, "Unknown game color-space argument");
    fixture_t fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fs::absolute(argv[3]), color, argc == 6 ? fs::absolute(argv[5]) : fs::path {});
    fixture.run();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
