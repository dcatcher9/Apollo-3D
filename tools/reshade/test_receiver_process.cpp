// SPDX-License-Identifier: GPL-3.0-only
// TEST ONLY: the helper alone loads the unchanged production receiver. CPU pixels
// returned by this facade are inspection instrumentation after its private GPU copy.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "test_receiver_process.h"

#include "test_receiver_api.h"

#include <array>
#include <bcrypt.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dxgi1_2.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
  namespace ipc = receiver_process_test;
  namespace fs = std::filesystem;

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  void checked(HRESULT result, const char *message) {
    if (FAILED(result)) {
      char text[320];
      std::snprintf(text, sizeof(text), "%s (0x%08lx)", message, static_cast<unsigned long>(result));
      throw std::runtime_error(text);
    }
  }

  struct handle_t {
    HANDLE value = nullptr;
    handle_t() = default;

    explicit handle_t(HANDLE handle):
        value(handle) {}

    handle_t(const handle_t &) = delete;
    handle_t &operator=(const handle_t &) = delete;

    ~handle_t() {
      if (value && value != INVALID_HANDLE_VALUE) {
        CloseHandle(value);
      }
    }
  };

  template<class T>
  struct com_ptr {
    T *value = nullptr;
    com_ptr() = default;
    com_ptr(const com_ptr &) = delete;
    com_ptr &operator=(const com_ptr &) = delete;

    ~com_ptr() {
      reset();
    }

    void reset() {
      if (value) {
        value->Release();
        value = nullptr;
      }
    }

    T **put() {
      reset();
      return &value;
    }

    T *operator->() const {
      return value;
    }
  };

  struct mapping_view_t {
    ipc::mailbox *value = nullptr;

    ~mapping_view_t() {
      if (value) {
        UnmapViewOfFile(value);
      }
    }
  };

  struct module_t {
    HMODULE value = nullptr;

    ~module_t() {
      if (value) {
        FreeLibrary(value);
      }
    }
  };

  std::uint64_t creation_time(HANDLE process) {
    FILETIME created {}, exited {}, kernel {}, user {};
    require(GetProcessTimes(process, &created, &exited, &kernel, &user), "Could not read process creation identity");
    return (std::uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime;
  }

  std::uint64_t luid_value(LUID luid) {
    std::uint64_t result;
    std::memcpy(&result, &luid, sizeof(result));
    return result;
  }

  std::size_t pixel_capacity(RECT source) {
    const auto width = std::int64_t(source.right) - source.left;
    const auto height = std::int64_t(source.bottom) - source.top;
    require(width > 0 && height > 0 && width <= ipc::max_source_dimension && height <= ipc::max_source_dimension, "Process fixture only accepts bounded source dimensions up to 1024x1024");
    return std::size_t(width) * 2 * std::size_t(height) * 8;
  }

  fs::path module_directory(HMODULE module) {
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(module, path.data(), DWORD(path.size()));
    require(length && length < path.size(), "Could not resolve fixture module path");
    path.resize(length);
    return fs::path(path).parent_path();
  }

  unsigned bytes_per_pixel(unsigned format) {
    if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
      return 8;
    }
    if (format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM) {
      return 4;
    }
    throw std::runtime_error("Process fixture received an unsupported texture format");
  }

#ifdef SUNSHINE_RECEIVER_PROCESS_HELPER
  struct child_t {
    handle_t mapping, request, reply, parent;
    mapping_view_t view;
    module_t facade;
    com_ptr<ID3D11Device> device;
    com_ptr<ID3D11DeviceContext> context;
    com_ptr<ID3D11Texture2D> staging;
    com_ptr<ID3D11Query> query;
    sunshine_receiver_test_create create = nullptr;
    sunshine_receiver_test_focus focus = nullptr;
    sunshine_receiver_test_poll poll = nullptr;
    sunshine_receiver_test_destroy destroy = nullptr;
    void *receiver = nullptr;

    ~child_t() {
      if (receiver) {
        destroy(receiver);
      }
    }

    void initialize() {
      auto &box = *view.value;
      require(box.parent_pid != GetCurrentProcessId() && GetProcessId(parent.value) == box.parent_pid && creation_time(parent.value) == box.parent_creation_time, "Inherited parent process identity does not match IPC");
      DWORD window_pid = 0;
      GetWindowThreadProcessId(reinterpret_cast<HWND>(box.window), &window_pid);
      require(window_pid == box.parent_pid, "Fixture game window is not owned by its parent process");
      com_ptr<IDXGIFactory1> factory;
      checked(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())), "Create child DXGI factory");
      com_ptr<IDXGIAdapter1> adapter;
      for (UINT index = 0;; ++index) {
        const HRESULT result = factory->EnumAdapters1(index, adapter.put());
        require(result != DXGI_ERROR_NOT_FOUND, "Parent adapter LUID not found in child");
        checked(result, "Enumerate child adapter");
        DXGI_ADAPTER_DESC1 description {};
        checked(adapter->GetDesc1(&description), "Read child adapter LUID");
        if (luid_value(description.AdapterLuid) == box.adapter_luid) {
          break;
        }
      }
      checked(D3D11CreateDevice(adapter.value, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, device.put(), nullptr, context.put()), "Create separate-process D3D11 device");
      facade.value = LoadLibraryExW((module_directory(nullptr) / L"reshade_receiver_test.dll").c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      require(facade.value != nullptr, "Child could not load sibling production receiver facade");
      create = reinterpret_cast<sunshine_receiver_test_create>(GetProcAddress(facade.value, "SunshineReceiverTestCreate"));
      focus = reinterpret_cast<sunshine_receiver_test_focus>(GetProcAddress(facade.value, "SunshineReceiverTestSetFocus"));
      poll = reinterpret_cast<sunshine_receiver_test_poll>(GetProcAddress(facade.value, "SunshineReceiverTestPoll"));
      destroy = reinterpret_cast<sunshine_receiver_test_destroy>(GetProcAddress(facade.value, "SunshineReceiverTestDestroy"));
      require(create && focus && poll && destroy, "Child receiver facade exports are missing");
      receiver = create(device.value, context.value, reinterpret_cast<HWND>(box.window), box.parent_pid, &box.source);
      require(receiver != nullptr, "Child production receiver creation failed");
    }

    int receive_frame() {
      auto &box = *view.value;
      sunshine_receiver_test_frame frame;
      const int result = poll(receiver, &frame);
      if (result != 1) {
        return result;
      }
      require(frame.texture && frame.sequence && frame.timestamp_ns, "Child receiver returned incomplete frame identity");
      D3D11_TEXTURE2D_DESC description {};
      frame.texture->GetDesc(&description);
      const unsigned pixel_size = bytes_per_pixel(description.Format);
      require(description.Width == 2 * unsigned(box.source.right - box.source.left) && description.Height == unsigned(box.source.bottom - box.source.top) && description.MipLevels == 1 && description.ArraySize == 1 && description.SampleDesc.Count == 1, "Child private texture dimensions differ from fixture source");
      D3D11_TEXTURE2D_DESC existing {};
      if (staging.value) {
        staging->GetDesc(&existing);
      }
      if (!staging.value || existing.Width != description.Width || existing.Height != description.Height || existing.Format != description.Format) {
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = description.MiscFlags = 0;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        checked(device->CreateTexture2D(&description, nullptr, staging.put()), "Create child inspection staging texture");
      }
      if (!query.value) {
        const D3D11_QUERY_DESC query_description {D3D11_QUERY_EVENT, 0};
        checked(device->CreateQuery(&query_description, query.put()), "Create child inspection completion query");
      }
      // This wait/readback is fixture instrumentation after the production receive.
      context->CopyResource(staging.value, frame.texture);
      context->End(query.value);
      context->Flush();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      HRESULT completed;
      while ((completed = context->GetData(query.value, nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE && std::chrono::steady_clock::now() < deadline) {
        Sleep(1);
      }
      require(completed == S_OK, "Child inspection GPU copy exceeded its two-second bound");
      D3D11_MAPPED_SUBRESOURCE mapped {};
      checked(context->Map(staging.value, 0, D3D11_MAP_READ, 0, &mapped), "Map completed child private-frame inspection");
      const std::size_t pitch = std::size_t(description.Width) * pixel_size;
      auto *pixels = reinterpret_cast<std::uint8_t *>(view.value) + ipc::pixel_offset;
      for (unsigned row = 0; row < description.Height; ++row) {
        std::memcpy(pixels + row * pitch, static_cast<const std::uint8_t *>(mapped.pData) + row * mapped.RowPitch, pitch);
      }
      context->Unmap(staging.value, 0);
      box.width = description.Width;
      box.height = description.Height;
      box.format = description.Format;
      box.row_pitch = unsigned(pitch);
      box.linear = frame.linear;
      box.sequence = frame.sequence;
      box.timestamp_ns = frame.timestamp_ns;
      return 1;
    }
  };

  std::uint64_t parse_number(const char *text) {
    char *end = nullptr;
    const auto value = std::strtoull(text, &end, 16);
    require(text[0] && end && !*end && value, "Invalid inherited capability argument");
    return value;
  }

  int run_child(int argc, char **argv) {
    require(argc == 7, "Receiver helper requires four inherited handles and its launch nonce");
    child_t child;
    child.mapping.value = reinterpret_cast<HANDLE>(parse_number(argv[1]));
    child.request.value = reinterpret_cast<HANDLE>(parse_number(argv[2]));
    child.reply.value = reinterpret_cast<HANDLE>(parse_number(argv[3]));
    child.parent.value = reinterpret_cast<HANDLE>(parse_number(argv[4]));
    const std::array<std::uint64_t, 2> nonce {parse_number(argv[5]), parse_number(argv[6])};
    child.view.value = static_cast<ipc::mailbox *>(MapViewOfFile(child.mapping.value, FILE_MAP_ALL_ACCESS, 0, 0, 0));
    require(child.view.value, "Could not map inherited fixture IPC");
    auto &box = *child.view.value;
    MEMORY_BASIC_INFORMATION region {};
    require(VirtualQuery(child.view.value, &region, sizeof(region)) == sizeof(region), "Could not inspect fixture mapping bounds");
    require(box.signature == ipc::magic && box.protocol_version == ipc::version && box.nonce[0] == nonce[0] && box.nonce[1] == nonce[1] && box.mapping_bytes == ipc::pixel_offset + pixel_capacity(box.source) && region.RegionSize >= box.mapping_bytes, "IPC identity, nonce or mapping bounds mismatch");
    // An orphan or wedged fixture is bounded even if its parent disappears mid-call.
    HANDLE watchdog_parent = nullptr;
    require(DuplicateHandle(GetCurrentProcess(), child.parent.value, GetCurrentProcess(), &watchdog_parent, SYNCHRONIZE, FALSE, 0), "Could not retain parent liveness for watchdog");
    std::thread([watchdog_parent] {
      WaitForSingleObject(watchdog_parent, 60000);
      CloseHandle(watchdog_parent);
      TerminateProcess(GetCurrentProcess(), 124);
    }).detach();
    box.child_pid = GetCurrentProcessId();
    try {
      child.initialize();
      box.result = 0;
    } catch (const std::exception &error) {
      std::snprintf(box.error, sizeof(box.error), "%s", error.what());
      box.result = -1;
      MemoryBarrier();
      SetEvent(child.reply.value);
      return 1;
    }
    MemoryBarrier();
    require(SetEvent(child.reply.value), "Signal child initialization");
    std::uint64_t previous_request = 0;
    const HANDLE events[] {child.request.value, child.parent.value};
    for (;;) {
      const DWORD ready = WaitForMultipleObjects(2, events, FALSE, 30000);
      if (ready != WAIT_OBJECT_0) {
        return ready == WAIT_OBJECT_0 + 1 ? 0 : 1;
      }
      MemoryBarrier();
      require(box.nonce[0] == nonce[0] && box.nonce[1] == nonce[1] && box.request_id == previous_request + 1, "Fixture request identity changed");
      previous_request = box.request_id;
      const auto operation = box.operation;
      try {
        if (operation == ipc::command::poll) {
          box.result = child.receive_frame();
        } else if (operation == ipc::command::focus) {
          child.focus(child.receiver, box.focused);
          box.result = 0;
        } else if (operation == ipc::command::quit) {
          child.destroy(child.receiver);
          child.receiver = nullptr;
          box.result = 0;
        } else {
          throw std::runtime_error("Unknown fixture IPC command");
        }
      } catch (const std::exception &error) {
        std::snprintf(box.error, sizeof(box.error), "%s", error.what());
        box.result = -1;
      }
      box.response_id = previous_request;
      MemoryBarrier();
      require(SetEvent(child.reply.value), "Signal fixture command completion");
      if (operation == ipc::command::quit || box.result < 0) {
        return box.result < 0 ? 1 : 0;
      }
    }
  }
#else
  std::wstring quote_argument(const std::wstring &argument) {
    std::wstring result = L"\"";
    std::size_t slashes = 0;
    for (wchar_t character : argument) {
      if (character == L'\\') {
        ++slashes;
        continue;
      }
      result.append(character == L'\"' ? slashes * 2 + 1 : slashes, L'\\');
      slashes = 0;
      result += character;
    }
    result.append(slashes * 2, L'\\');
    return result + L'\"';
  }

  std::wstring hex_number(std::uint64_t value) {
    wchar_t text[32];
    std::swprintf(text, std::size(text), L"%llx", static_cast<unsigned long long>(value));
    return text;
  }

  struct process_proxy_t {
    handle_t mapping, request, reply, parent, process;
    mapping_view_t view;
    com_ptr<ID3D11Device> device;
    com_ptr<ID3D11DeviceContext> context;
    com_ptr<ID3D11Texture2D> texture;
    std::array<std::uint64_t, 2> nonce {};
    DWORD child_pid = 0;
    std::uint64_t next_request = 0;
    std::mutex mutex;
    bool initialized = false;
    bool failed = false;

    ~process_proxy_t() {
      if (!process.value) {
        return;
      }
      if (initialized && !failed && WaitForSingleObject(process.value, 0) == WAIT_TIMEOUT) {
        try {
          transact(ipc::command::quit);
        } catch (...) {
          failed = true;
        }
      }
      if (WaitForSingleObject(process.value, ipc::command_timeout_ms) != WAIT_OBJECT_0) {
        TerminateProcess(process.value, 125);
        WaitForSingleObject(process.value, 1000);
        std::fprintf(stderr, "Separate receiver child %lu required bounded termination\n", static_cast<unsigned long>(child_pid));
      } else {
        DWORD exit = 1;
        GetExitCodeProcess(process.value, &exit);
        std::printf("Separate receiver child %lu exited %lu\n", static_cast<unsigned long>(child_pid), static_cast<unsigned long>(exit));
      }
    }

    void wait_response(DWORD timeout) {
      const HANDLE events[] {reply.value, process.value};
      const DWORD ready = WaitForMultipleObjects(2, events, FALSE, timeout);
      if (ready != WAIT_OBJECT_0) {
        failed = true;
        throw std::runtime_error(ready == WAIT_TIMEOUT ? "Separate receiver command exceeded its time bound" : "Separate receiver exited before replying");
      }
      MemoryBarrier();
      const auto &box = *view.value;
      require(box.nonce[0] == nonce[0] && box.nonce[1] == nonce[1] && box.child_pid == child_pid && box.response_id == next_request, "Separate receiver response identity mismatch");
      if (box.result < 0) {
        failed = true;
        throw std::runtime_error(box.error[0] ? box.error : "Separate receiver command failed");
      }
    }

    int transact(ipc::command operation, BOOL focused = TRUE) {
      require(initialized && !failed, "Separate receiver is not active");
      auto &box = *view.value;
      box.operation = operation;
      box.focused = focused;
      box.request_id = ++next_request;
      MemoryBarrier();
      require(SetEvent(request.value), "Could not send receiver command");
      wait_response(ipc::command_timeout_ms);
      return box.result;
    }

    void initialize(ID3D11Device *source_device, ID3D11DeviceContext *source_context, HWND window, DWORD pid, RECT source) {
      require(source_device && source_context && pid == GetCurrentProcessId(), "Process facade only accepts its own fixture process");
      DWORD window_pid = 0;
      GetWindowThreadProcessId(window, &window_pid);
      require(window_pid == pid, "Process facade window must belong to its fixture");
      const std::size_t capacity = pixel_capacity(source);
      device.value = source_device;
      device->AddRef();
      context.value = source_context;
      context->AddRef();
      com_ptr<IDXGIDevice> dxgi_device;
      checked(device->QueryInterface(IID_PPV_ARGS(dxgi_device.put())), "Read parent DXGI device");
      com_ptr<IDXGIAdapter> adapter;
      checked(dxgi_device->GetAdapter(adapter.put()), "Read parent GPU adapter");
      DXGI_ADAPTER_DESC adapter_description {};
      checked(adapter->GetDesc(&adapter_description), "Read parent adapter LUID");
      require(BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(nonce.data()), ULONG(sizeof(nonce)), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0 && nonce[0] && nonce[1], "Could not generate fixture launch nonce");
      SECURITY_ATTRIBUTES security {sizeof(security), nullptr, TRUE};
      mapping.value = CreateFileMappingW(INVALID_HANDLE_VALUE, &security, PAGE_READWRITE, 0, DWORD(ipc::pixel_offset + capacity), nullptr);
      request.value = CreateEventW(&security, FALSE, FALSE, nullptr);
      reply.value = CreateEventW(&security, FALSE, FALSE, nullptr);
      parent.value = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, TRUE, pid);
      require(mapping.value && request.value && reply.value && parent.value, "Could not create private inherited fixture capabilities");
      view.value = static_cast<ipc::mailbox *>(MapViewOfFile(mapping.value, FILE_MAP_ALL_ACCESS, 0, 0, ipc::pixel_offset + capacity));
      require(view.value, "Could not map parent fixture IPC");
      auto &box = *new (view.value) ipc::mailbox {};
      box.parent_pid = pid;
      box.parent_creation_time = creation_time(parent.value);
      box.nonce[0] = nonce[0];
      box.nonce[1] = nonce[1];
      box.window = reinterpret_cast<std::uint64_t>(window);
      box.adapter_luid = luid_value(adapter_description.AdapterLuid);
      box.mapping_bytes = ipc::pixel_offset + capacity;
      box.source = source;
      HMODULE own_module = nullptr;
      require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&quote_argument), &own_module), "Could not resolve process facade module");
      const auto directory = module_directory(own_module);
      const auto helper = directory / L"reshade_receiver_process_helper.exe";
      require(fs::is_regular_file(helper), "Sibling receiver process helper executable is missing");
      std::wstring command_line = quote_argument(helper.wstring());
      const HANDLE inherited[] {mapping.value, request.value, reply.value, parent.value};
      for (HANDLE handle : inherited) {
        command_line += L" " + hex_number(reinterpret_cast<std::uint64_t>(handle));
      }
      for (auto value : nonce) {
        command_line += L" " + hex_number(value);
      }
      SIZE_T attribute_bytes = 0;
      InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
      std::vector<std::uint8_t> storage(attribute_bytes);
      auto *attributes = reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
      require(InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes), "Initialize inherited handle allowlist");

      struct cleanup_t {
        PPROC_THREAD_ATTRIBUTE_LIST value;

        ~cleanup_t() {
          DeleteProcThreadAttributeList(value);
        }
      } cleanup {attributes};

      require(UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, const_cast<HANDLE *>(inherited), sizeof(inherited), nullptr, nullptr), "Set inherited handle allowlist");
      STARTUPINFOEXW startup {};
      startup.StartupInfo.cb = sizeof(startup);
      startup.lpAttributeList = attributes;
      PROCESS_INFORMATION child {};
      require(CreateProcessW(helper.c_str(), command_line.data(), nullptr, nullptr, TRUE, EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr, directory.c_str(), &startup.StartupInfo, &child), "Create own receiver helper process");
      process.value = child.hProcess;
      handle_t thread(child.hThread);
      child_pid = child.dwProcessId;
      for (HANDLE handle : inherited) {
        SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
      }
      require(child_pid && child_pid != pid && GetProcessId(process.value) == child_pid, "Receiver helper is not a distinct child process");
      wait_response(5000);
      initialized = true;
      std::printf("SEPARATE PROCESS receiver: game PID %lu -> child PID %lu, adapter LUID %llx; production receiver runs only in child; CPU return is test inspection\n", static_cast<unsigned long>(pid), static_cast<unsigned long>(child_pid), static_cast<unsigned long long>(box.adapter_luid));
    }

    int poll(sunshine_receiver_test_frame &output) {
      output = {};
      const int status = transact(ipc::command::poll);
      if (status != 1) {
        return status;
      }
      const auto &box = *view.value;
      const unsigned pixel_size = bytes_per_pixel(box.format);
      require(box.width == 2 * unsigned(box.source.right - box.source.left) && box.height == unsigned(box.source.bottom - box.source.top) && box.row_pitch == box.width * pixel_size && std::uint64_t(box.row_pitch) * box.height <= box.mapping_bytes - ipc::pixel_offset && box.sequence && box.timestamp_ns, "Child inspection frame exceeds declared bounds");
      D3D11_TEXTURE2D_DESC description {};
      if (texture.value) {
        texture->GetDesc(&description);
      }
      if (!texture.value || description.Width != box.width || description.Height != box.height || description.Format != box.format) {
        description = {};
        description.Width = box.width;
        description.Height = box.height;
        description.MipLevels = description.ArraySize = 1;
        description.Format = DXGI_FORMAT(box.format);
        description.SampleDesc.Count = 1;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        checked(device->CreateTexture2D(&description, nullptr, texture.put()), "Create parent test inspection texture");
      }
      context->UpdateSubresource(texture.value, 0, nullptr, reinterpret_cast<const std::uint8_t *>(view.value) + ipc::pixel_offset, box.row_pitch, 0);
      output.texture = texture.value;
      output.linear = box.linear;
      output.sequence = box.sequence;
      output.timestamp_ns = box.timestamp_ns;
      return 1;
    }
  };
#endif
}  // namespace

#ifdef SUNSHINE_RECEIVER_PROCESS_HELPER
int main(int argc, char **argv) {
  try {
    return run_child(argc, argv);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Receiver child failed: %s\n", error.what());
    return 1;
  }
}
#else
extern "C" __declspec(dllexport) void *SunshineReceiverTestCreate(ID3D11Device *device, ID3D11DeviceContext *context, HWND window, DWORD pid, const RECT *source) {
  try {
    require(source, "Missing receiver source rectangle");
    auto proxy = std::make_unique<process_proxy_t>();
    proxy->initialize(device, context, window, pid, *source);
    return proxy.release();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Process receiver creation failed: %s\n", error.what());
    return nullptr;
  }
}

extern "C" __declspec(dllexport) void SunshineReceiverTestSetFocus(void *opaque, BOOL focused) {
  auto *proxy = static_cast<process_proxy_t *>(opaque);
  if (!proxy) {
    return;
  }
  std::lock_guard<std::mutex> lock(proxy->mutex);
  try {
    proxy->transact(ipc::command::focus, focused);
  } catch (const std::exception &error) {
    proxy->failed = true;
    std::fprintf(stderr, "Process receiver focus failed: %s\n", error.what());
  }
}

extern "C" __declspec(dllexport) int SunshineReceiverTestPoll(void *opaque, sunshine_receiver_test_frame *output) {
  auto *proxy = static_cast<process_proxy_t *>(opaque);
  if (!proxy || !output) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(proxy->mutex);
  try {
    return proxy->poll(*output);
  } catch (const std::exception &error) {
    proxy->failed = true;
    *output = {};
    std::fprintf(stderr, "Process receiver poll failed: %s\n", error.what());
    return -1;
  }
}

extern "C" __declspec(dllexport) void SunshineReceiverTestDestroy(void *opaque) {
  delete static_cast<process_proxy_t *>(opaque);
}
#endif
