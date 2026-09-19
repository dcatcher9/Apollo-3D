// SPDX-License-Identifier: GPL-3.0-only
// Opt-in ABI fixture: an actual MSVC-built ReShade DLL and native D3D12 objects.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <cstdio>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <fstream>
#include <reshade.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <windows.h>

namespace {
  namespace api = reshade::api;
  namespace fs = std::filesystem;

  void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void checked(HRESULT result, const char *message) {
    if (FAILED(result)) {
      char detail[320];
      std::snprintf(detail, sizeof(detail), "%s (0x%08lx)", message, static_cast<unsigned long>(result));
      throw std::runtime_error(detail);
    }
  }

  void stage(const char *message) {
    std::printf("%s\n", message);
    std::fflush(stdout);
  }

  template<class T>
  struct com_ptr {
    T *p = nullptr;

    com_ptr() = default;
    com_ptr(const com_ptr &) = delete;
    com_ptr &operator=(const com_ptr &) = delete;

    ~com_ptr() {
      if (p) {
        p->Release();
      }
    }

    T *operator->() const {
      return p;
    }
  };

  // Also bounds driver/runtime calls and teardown, independently of CTest.
  struct deadline_t {
    HANDLE completed = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    std::thread watchdog;

    deadline_t() {
      require(completed != nullptr, "CreateEventW for deadline failed");
      watchdog = std::thread([this] {
        if (WaitForSingleObject(completed, 30000) != WAIT_OBJECT_0) {
          std::fputs("FAIL: D3D12 ReShade ABI probe exceeded 30 seconds\n", stderr);
          std::fflush(stderr);
          TerminateProcess(GetCurrentProcess(), 124);
        }
      });
    }

    ~deadline_t() {
      SetEvent(completed);
      watchdog.join();
      CloseHandle(completed);
    }
  };

  struct module_t {
    HMODULE handle = nullptr;

    ~module_t() {
      if (handle) {
        FreeLibrary(handle);
      }
    }
  };

  struct window_t {
    HINSTANCE instance = GetModuleHandleW(nullptr);
    const wchar_t *class_name = L"SunshineReShadeD3D12AbiFixture";
    ATOM window_class = 0;
    HWND handle = nullptr;

    window_t() {
      WNDCLASSW description {};
      description.lpfnWndProc = DefWindowProcW;
      description.hInstance = instance;
      description.lpszClassName = class_name;
      window_class = RegisterClassW(&description);
      require(window_class != 0, "RegisterClassW failed");
      // A real, hidden top-level window satisfies DXGI without changing focus.
      handle = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, class_name, L"Sunshine D3D12 ABI probe", WS_OVERLAPPED, 0, 0, 640, 360, nullptr, nullptr, instance, nullptr);
      if (!handle) {
        UnregisterClassW(class_name, instance);
        throw std::runtime_error("CreateWindowExW failed");
      }
    }

    ~window_t() {
      DestroyWindow(handle);
      UnregisterClassW(class_name, instance);
    }
  };

  using create_runtime_t = bool (*)(api::device_api, void *, void *, void *, const char *, api::effect_runtime **);
  using destroy_runtime_t = void (*)(api::effect_runtime *);

  struct runtime_t {
    api::effect_runtime *p = nullptr;
    destroy_runtime_t destroy = nullptr;

    ~runtime_t() {
      if (p) {
        destroy(p);
      }
    }
  };

  struct view_t {
    api::device *device;
    api::resource_view value {};

    ~view_t() {
      if (value.handle) {
        device->destroy_resource_view(value);
      }
    }
  };

  void run(const fs::path &dll_path) {
    require(fs::is_regular_file(dll_path), "The supplied ReShade DLL does not exist");
    // Keep the DLL loaded until every object backed by it has been destroyed.
    module_t module;
    window_t window;
    com_ptr<IDXGIFactory4> factory;
    checked(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory.p)), "CreateDXGIFactory2 failed");
    com_ptr<ID3D12Device> native_device;
    checked(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&native_device.p)), "D3D12CreateDevice failed");
    com_ptr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_description {};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    checked(native_device->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue.p)), "CreateCommandQueue failed");

    DXGI_SWAP_CHAIN_DESC1 swapchain_description {};
    // ReShade intentionally ignores swapchains smaller than 160x120.
    swapchain_description.Width = 640;
    swapchain_description.Height = 360;
    swapchain_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapchain_description.SampleDesc.Count = 1;
    swapchain_description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchain_description.BufferCount = 2;
    swapchain_description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    com_ptr<IDXGISwapChain1> swapchain;
    checked(factory->CreateSwapChainForHwnd(queue.p, window.handle, &swapchain_description, nullptr, nullptr, &swapchain.p), "CreateSwapChainForHwnd failed");

    constexpr UINT texture_width = 1280, texture_height = 360;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC texture_description {};
    texture_description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture_description.Width = texture_width;
    texture_description.Height = texture_height;
    texture_description.DepthOrArraySize = 1;
    texture_description.MipLevels = 1;
    texture_description.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    texture_description.SampleDesc.Count = 1;
    com_ptr<ID3D12Resource> texture;
    checked(native_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture_description, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&texture.p)), "CreateCommittedResource(FP16) failed");
    require(!IsWindowVisible(window.handle), "Fixture window unexpectedly became visible");

    // An isolated empty effects directory prevents discovery of game effects.
    const fs::path fixture_directory = fs::temp_directory_path() / ("SunshineReShadeD3D12Abi-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));
    fs::create_directories(fixture_directory / "empty-effects");
    fs::create_directories(fixture_directory / "empty-addons");
    const fs::path config_path = fixture_directory / "ReShade.ini";
    {
      std::ofstream config(config_path, std::ios::binary | std::ios::trunc);
      config << "[ADDON]\nAddonPath=" << (fixture_directory / "empty-addons").u8string()
             << "\n[GENERAL]\nEffectSearchPaths=" << (fixture_directory / "empty-effects").u8string()
             << "\nTextureSearchPaths=" << (fixture_directory / "empty-effects").u8string()
             << "\nSkipLoadingDisabledEffects=1\nNoReloadOnInit=1\n[INPUT]\nKeyOverlay=0,0,0,0\n";
      require(config.good(), "Writing isolated ReShade config failed");
    }
    require(SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", fixture_directory.c_str()), "Setting isolated ReShade base path failed");

    // Load after creating native objects so this explicitly exercises the embedded
    // runtime API, rather than depending on graphics entry-point interception.
    // ReShade's direct-load policy requires the config to exist before DllMain.
    stage("Loading supplied ReShade DLL for embedded D3D12 runtime");
    module.handle = LoadLibraryExW(dll_path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module.handle) {
      throw std::runtime_error("LoadLibraryExW(ReShade) failed, Win32 error " + std::to_string(GetLastError()));
    }
    const auto create_runtime = reinterpret_cast<create_runtime_t>(GetProcAddress(module.handle, "ReShadeCreateEffectRuntime"));
    const auto destroy_runtime = reinterpret_cast<destroy_runtime_t>(GetProcAddress(module.handle, "ReShadeDestroyEffectRuntime"));
    require(create_runtime && destroy_runtime, "The supplied DLL does not export the embedded runtime API");

    runtime_t runtime;
    runtime.destroy = destroy_runtime;
    stage("Creating actual ReShade D3D12 effect runtime");
    require(create_runtime(api::device_api::d3d12, native_device.p, queue.p, swapchain.p, config_path.u8string().c_str(), &runtime.p) && runtime.p, "ReShadeCreateEffectRuntime(D3D12) failed");
    api::device *const device = runtime.p->get_device();
    require(device && device->get_api() == api::device_api::d3d12, "Runtime did not expose a D3D12 device");
    require(device->get_native() == reinterpret_cast<std::uint64_t>(native_device.p), "Runtime native device identity mismatch");

    view_t view {device};
    api::resource_view_desc view_description {};
    view_description.type = api::resource_view_type::texture_2d;
    view_description.format = api::format::r16g16b16a16_float;
    view_description.texture.levels = 1;
    view_description.texture.layers = 1;
    const api::resource known_resource {reinterpret_cast<std::uint64_t>(texture.p)};
    require(device->create_resource_view(known_resource, api::resource_usage::shader_resource, view_description, &view.value) && view.value.handle, "Real ReShade create_resource_view failed");

    stage("Calling patched SDK get_resource_from_view on actual MSVC D3D12 interface");
    const api::resource recovered = device->get_resource_from_view(view.value);
    require(recovered.handle == known_resource.handle, "SDK aggregate-return ABI corrupted the native resource pointer");
    const auto actual_description = reinterpret_cast<ID3D12Resource *>(recovered.handle)->GetDesc();
    require(actual_description.Width == texture_width && actual_description.Height == texture_height && actual_description.Format == DXGI_FORMAT_R16G16B16A16_FLOAT && actual_description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D, "Recovered D3D12 resource has incorrect FP16 texture description");
    require(!IsWindowVisible(window.handle), "Runtime unexpectedly made the fixture window visible");
    std::printf("Verified exact native pointer %p, FP16 texture %ux%u; no Present or foreground activation\n", static_cast<void *>(texture.p), texture_width, texture_height);
    stage("Destroying owned view, runtime, native objects, and DLL");
  }
}  // namespace

int main(int argc, char **argv) {
  try {
    deadline_t deadline;
    require(argc == 2, "Usage: reshade_runtime_d3d12_test <official ReShade64.dll>");
    run(fs::absolute(fs::u8path(argv[1])));
    stage("PASS: actual ReShade D3D12 get_resource_from_view ABI and native GetDesc");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
