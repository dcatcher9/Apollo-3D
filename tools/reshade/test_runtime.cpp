// SPDX-License-Identifier: GPL-3.0-only
// Opt-in integration fixture: real official ReShade DLL, real exporter, synthetic FX input.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "src/reshade_bridge_protocol.h"
#include "test_overlay_pixels.h"
#include "test_receiver_api.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <d3d11_4.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <reshade.hpp>
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
    com_ptr<ID3D11Device> receiver, game;
    com_ptr<ID3D11Device5> receiver5;
    com_ptr<ID3D11DeviceContext> receiver_context, game_context;
    com_ptr<IDXGISwapChain> swapchain;
    com_ptr<ID3D11RenderTargetView> target;
    unsigned expected_color = 1;
    bool interactive = false;
    bool controlled = false;
    HMODULE receiver_module = nullptr;
    void *receiver_state = nullptr;
    sunshine_receiver_test_create receiver_create = nullptr;
    sunshine_receiver_test_focus receiver_focus = nullptr;
    sunshine_receiver_test_poll receiver_poll = nullptr;
    sunshine_receiver_test_destroy receiver_destroy = nullptr;
    void (*set_exporter_foreground)(HWND) = nullptr;
    void (*set_overlay_patch)(BOOL) = nullptr;
    bool expect_overlay = false;
    float expected_gain = 1.0f;
    std::uint64_t last_sequence = 0, last_timestamp = 0;
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
      target.reset();
      swapchain.reset();
      game_context.reset();
      game.reset();
      if (window) {
        DestroyWindow(window);
      }
      // Process exit unloads the official runtime after all COM resources are gone.
    }

    void initialize(const fs::path &runtime, const fs::path &addon, const fs::path &directory, unsigned color, const fs::path &receiver_dll = {}) {
      expected_color = color;
      controlled = !receiver_dll.empty();
      if (controlled) {
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
      fs::copy_file(addon, directory / "addons" / "SunshineSBS.addon64", fs::copy_options::overwrite_existing);
      write_file(directory / "effects" / "SuperDepth3D.fx", fixture_effect);
      write_file(directory / "preset.ini", "Techniques=SuperDepth3D@SuperDepth3D.fx\nTechniqueSorting=SuperDepth3D@SuperDepth3D.fx\n");
      write_file(directory / "ReShade.ini", "[ADDON]\nAddonPath=.\\addons\nDisabledAddons=Generic Depth\n[GENERAL]\nEffectSearchPaths=.\\effects\nPresetPath=.\\preset.ini\nPerformanceMode=0\nSkipLoadingDisabledEffects=0\nEffectCachePath=.\\cache\n[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n[STYLE]\nHdrOverlayBrightness=203\n");
      SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", directory.c_str());

      // Create the receiver before loading ReShade so it is a separate native D3D11 device.
      wchar_t system[MAX_PATH];
      GetSystemDirectoryW(system, MAX_PATH);
      const HMODULE native_d3d11 = LoadLibraryW((fs::path(system) / "d3d11.dll").c_str());
      require(native_d3d11 != nullptr, "Could not load system D3D11");
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
      window_class.lpszClassName = L"SunshineReShadeIntegrationFixture";
      RegisterClassW(&window_class);
      RECT bounds {0, 0, source_width, source_height};
      AdjustWindowRect(&bounds, WS_OVERLAPPEDWINDOW, FALSE);
      window = CreateWindowW(window_class.lpszClassName, L"Sunshine ReShade integration test (closes automatically)", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, bounds.right - bounds.left, bounds.bottom - bounds.top, nullptr, nullptr, window_class.hInstance, nullptr);
      require(window != nullptr, "Could not create fixture window");
      if (!controlled) {
        ShowWindow(window, SW_SHOW);
        SetForegroundWindow(window);
      }

      DXGI_SWAP_CHAIN_DESC desc {};
      desc.BufferDesc.Width = source_width;
      desc.BufferDesc.Height = source_height;
      desc.BufferDesc.Format = color == 2 ? DXGI_FORMAT_R16G16B16A16_FLOAT : color == 3 ? DXGI_FORMAT_R10G10B10A2_UNORM :
                                                                                          DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count = 1;
      desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      desc.BufferCount = 2;
      desc.OutputWindow = window;
      desc.Windowed = TRUE;
      desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      const auto wrapped_create = reinterpret_cast<create_device_t>(GetProcAddress(reshade_module, "D3D11CreateDeviceAndSwapChain"));
      require(wrapped_create != nullptr, "Official runtime is missing D3D11 proxy export");
      checked(wrapped_create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &desc, swapchain.put(), game.put(), nullptr, game_context.put()), "ReShade wrapped game creation");
      com_ptr<IDXGISwapChain3> swapchain3;
      checked(swapchain->QueryInterface(IID_PPV_ARGS(swapchain3.put())), "Swapchain color-space interface");
      checked(swapchain3->SetColorSpace1(color == 2 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : color == 3 ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 :
                                                                                                             DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),
              "Set game color space");
      com_ptr<ID3D11Texture2D> buffer;
      checked(swapchain->GetBuffer(0, IID_PPV_ARGS(buffer.put())), "Game backbuffer");
      checked(game->CreateRenderTargetView(buffer.p, nullptr, target.put()), "Game backbuffer view");
      const auto mapping_name = std::wstring(wire::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, mapping_name.c_str());
      require(mapping != nullptr, "Production exporter did not create its mapping");
      shared = static_cast<wire::shared_state_t *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(wire::shared_state_t)));
      require(shared != nullptr, "Could not map production publication");
      if (controlled) {
        const auto test_addon = GetModuleHandleW((directory / "addons" / "SunshineSBS.addon64").c_str());
        require(test_addon != nullptr, "Controlled test add-on was not loaded");
        set_exporter_foreground = reinterpret_cast<void (*)(HWND)>(GetProcAddress(test_addon, "SunshineSbsTestSetForeground"));
        set_overlay_patch = reinterpret_cast<void (*)(BOOL)>(GetProcAddress(test_addon, "SunshineSbsTestSetOverlayPatch"));
        require(set_exporter_foreground != nullptr, "Controlled mode requires the distinct SunshineSBSTest.addon64, never the production add-on");
        require(set_overlay_patch != nullptr, "Controlled mode requires the runtime-test overlay patch hook");
        require(GetClientRect(window, &source_rect), "Could not query fixture client rectangle");
        POINT origin {};
        require(ClientToScreen(window, &origin), "Could not query fixture client origin");
        OffsetRect(&source_rect, origin.x, origin.y);
        receiver_state = receiver_create(receiver.p, receiver_context.p, window, GetCurrentProcessId(), &source_rect);
        require(receiver_state != nullptr, "Could not initialize production receiver");
        set_exporter_foreground(window);
        std::puts("CONTROLLED TEST OBSERVATIONS: hidden game window, test-only exporter, unchanged production receiver; Windows foreground is not changed");
        return;
      }
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&shared->consumer_nonce), 0x12345678);
      // A hidden parent STARTUPINFO can override the first ShowWindow call. Activate only
      // this fixture window, after runtime initialization has finished.
      ShowWindow(window, SW_SHOWNORMAL);
      UpdateWindow(window);
      SetActiveWindow(window);
      const BOOL activated = SetForegroundWindow(window);
      std::printf("fixture visible=%d activated=%d own=%p foreground=%p\n", IsWindowVisible(window), activated, window, GetForegroundWindow());
      DWORD session = 0, bytes = 0;
      ProcessIdToSessionId(GetCurrentProcessId(), &session);
      wchar_t station_name[256] {}, desktop_name[256] {};
      GetUserObjectInformationW(GetProcessWindowStation(), UOI_NAME, station_name, sizeof(station_name), &bytes);
      GetUserObjectInformationW(GetThreadDesktop(GetCurrentThreadId()), UOI_NAME, desktop_name, sizeof(desktop_name), &bytes);
      std::printf("fixture session=%lu station=%ls desktop=%ls active=%p focus=%p\n", static_cast<unsigned long>(session), station_name, desktop_name, GetActiveWindow(), GetFocus());
      interactive = GetForegroundWindow() == window;
    }

    void step() {
      MSG message;
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      require(IsWindow(window) && (!interactive || GetForegroundWindow() == window), "Fixture lost foreground; test cannot validate exporter focus gate");
      const float background[4] {0.125f, 0.25f, 0.375f, 1};
      game_context->OMSetRenderTargets(1, &target.p, nullptr);
      game_context->ClearRenderTargetView(target.p, background);
      checked(swapchain->Present(0, 0), "Natural DXGI Present");
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
          // The real receiver performs nonce negotiation, handle import, fence checks, slot
          // ownership and private-copy retirement. This fixture never edits its control words.
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
            // A new consumer needs another Present to receive its replacement ring.
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
          // Drain this Present before testing retained timestamps: a pending copy may
          // legitimately replace an older cached image without another game Present.
          const auto copy_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
          while ((!received_status || received.sequence != newest_sequence) && std::chrono::steady_clock::now() < copy_deadline) {
            received_status = receiver_poll(receiver_state, &received);
            require(received_status >= 0, "Production receiver completion poll failed");
            Sleep(1);
          }
          require(received_status == 1 && received.sequence == newest_sequence && received.texture && received.timestamp_ns, "Production receiver did not retire the latest D3D11 publication");
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
      const bool correct = sunshine_sbs_overlay_pixels::check(mapped.pData, mapped.RowPitch, format, source_width, source_height, expect_overlay, expect_overlay && set_overlay_patch, expected_gain);
      context->Unmap(staging.p, 0);
      return correct;
    }

    void set_gain(float value) {
      api::effect_uniform_variable gain {};
      observation.runtime->enumerate_uniform_variables("SuperDepth3D.fx", [&](api::effect_runtime *runtime, api::effect_uniform_variable variable) {
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

    bool set_test_focus(bool focused) {
      if (!set_exporter_foreground) {
        return false;
      }
      set_exporter_foreground(focused ? window : nullptr);
      if (receiver_state) {
        receiver_focus(receiver_state, focused);
      }
      return true;
    }

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
      observation.runtime->enumerate_texture_variables("SuperDepth3D.fx", [&](api::effect_runtime *runtime, api::effect_texture_variable texture) {
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
      // ReShade's C++ handle-return getter is not MSVC/MinGW ABI compatible, even though
      // the handle occupies only eight bytes. Native COM provides a real output parameter.
      com_ptr<ID3D11Resource> resource;
      reinterpret_cast<ID3D11View *>(view.handle)->GetResource(resource.put());
      com_ptr<ID3D11Texture2D> texture;
      require(resource.p != nullptr, "Actual runtime did not expose export texture resource");
      checked(resource->QueryInterface(IID_PPV_ARGS(texture.put())), "Export resource is not a texture");
      const auto sdk_resource = observation.runtime->get_device()->get_resource_from_view(view);
      require(sdk_resource.handle == reinterpret_cast<std::uint64_t>(resource.p), "Adapted SDK getter differs from native COM resource");
      require(inspect_texture(game.p, game_context.p, texture.p, expected_color == 1 ? 24 : 10), "Actual FX texture pixels differ from the shader output");
      std::puts("actual runtime texture pixels and SDK/native COM resource identity PASS");
    }

    void run_background() {
      std::puts("SKIP foreground publication checks: fixture window is not foreground; validating real runtime/background rejection only");
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
      do {
        step();
      } while (!observation.techniques && std::chrono::steady_clock::now() < deadline);
      require(observation.runtime && observation.techniques, "Actual runtime did not compile/execute the fixture");
      inspect_runtime_texture();
      wire::metadata_t metadata;
      require(snapshot(*shared, metadata) && !metadata.generation, "Background game bypassed production foreground gate");
      const auto before_disable = observation.techniques;
      observation.runtime->set_effects_state(false);
      step();
      require(observation.techniques == before_disable, "Actual runtime ignored effects disable");
      observation.runtime->set_effects_state(true);
      step();
      require(observation.techniques > before_disable, "Actual runtime ignored effects enable");
      require(observation.runtime->open_overlay(true, api::input_source::keyboard), "Actual runtime overlay did not open");
      const auto before_reload = observation.reloads;
      const auto before_reload_techniques = observation.techniques;
      observation.runtime->reload_effect_next_frame(nullptr);
      const auto reload_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
      do {
        step();
      } while ((observation.reloads == before_reload || observation.techniques == before_reload_techniques) && std::chrono::steady_clock::now() < reload_deadline);
      require(observation.reloads > before_reload && observation.techniques > before_reload_techniques, "Actual runtime did not reload effects");
      require(observation.runtime->open_overlay(false, api::input_source::keyboard), "Actual runtime overlay did not close");
      require(observation.overlay_opens == 1 && observation.overlay_closes == 1, "Actual overlay transition callbacks were not delivered");
      step();
      inspect_runtime_texture();
      require(snapshot(*shared, metadata) && !metadata.generation, "Background reload/overlay bypassed foreground gate");
      std::printf("PASS partial actual runtime color=%u: present=%u technique=%u finish=%u reload=%u; FX pixels/ABI/color/events/background rejection. Full shared publication SKIPPED.\n", expected_color, observation.presents, observation.techniques, observation.finishes, observation.reloads);
    }

    void run() {
      if (!interactive && !controlled) {
        run_background();
        return;
      }
      auto generation = wait_pixels();
      if (receiver_state) {
        sunshine_receiver_test_frame retained;
        require(receiver_poll(receiver_state, &retained) == 1 && retained.sequence == last_received.sequence && retained.timestamp_ns == last_received.timestamp_ns, "A retained frame changed its source sequence or timestamp without another game Present");
      }
      require(observation.runtime && observation.techniques, "Official runtime did not execute the fixture technique");
      inspect_runtime_texture();
      observation.runtime->set_effects_state(false);
      step();
      wire::metadata_t metadata;
      require(snapshot(*shared, metadata) && metadata.generation == 0, "Effect disable retained a stale publication");
      require_receiver_inactive("Production receiver retained stereo after effect disable");
      observation.runtime->set_effects_state(true);
      generation = wait_pixels(generation);
      generation = exercise_overlay(generation);
      if (controlled) {
        const auto previous_nonce = read64(shared->consumer_nonce);
        receiver_destroy(receiver_state);
        receiver_state = receiver_create(receiver.p, receiver_context.p, window, GetCurrentProcessId(), &source_rect);
        require(receiver_state != nullptr, "Could not restart production receiver");
        wait_pixels(generation);
        require(read64(shared->consumer_nonce) != previous_nonce, "Receiver restart reused the old consumer nonce");
      }
      std::printf("PASS actual ReShade6.8 DX11 color=%u: present=%u technique=%u finish=%u reload=%u; pixels, effect disable, overlay/reload/close%s\n", expected_color, observation.presents, observation.techniques, observation.finishes, observation.reloads, controlled ? "; controlled focus, production receiver and receiver restart" : "");
    }
  };
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5 && argc != 6) {
    std::fprintf(stderr, "usage: reshade_runtime_test <official-ReShade64.dll> <SunshineSBS.addon64> <isolated-output-directory> <srgb|scrgb|pq> [production-receiver-test.dll]\nThe optional receiver facade requires the distinct SunshineSBSTest.addon64 and supplies controlled foreground observations.\n");
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
    return fixture.interactive || fixture.controlled ? 0 : 77;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
