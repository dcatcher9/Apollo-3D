// SPDX-License-Identifier: GPL-3.0-only
#include "addon_lifetime.h"
// Original Sunshine exporter. ReShade's SDK headers are BSD-3-Clause OR MIT.
#include "overlay_compositor.h"
#include "depth_addon.h"
#include "depth_jitter.h"
#include "streamline_camera_probe.h"
#include "streamline_depth_provider.h"
#include "streamline_depth_capture.h"
#include "upscaler_call_trace.h"
#include "projection_depth_controller.h"
#include "raw_scene_pool.h"
#include "provided_raw_scene.h"
#include "automatic_scene_transition.h"
#include "game3d_controls.h"
#include "game3d_renderer.h"
#include "diagnostic_log_gate.h"
#include "src/reshade_bridge_protocol.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <imgui.h>
#include <memory>
#include <mutex>
#include <reshade.hpp>
#include "depth_content_sampler.h"
#include <string>
#include <unordered_map>
#include <utility>
#include <windows.h>

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  #include "test_overlay_patch.h"

  #include <atomic>
#endif

static_assert(RESHADE_API_VERSION == 20, "Build against the pinned ReShade 6.8.0 SDK");
static_assert(sizeof(void *) == 8, "Only 64-bit games are supported");

namespace {
  namespace api = reshade::api;
  namespace wire = reshade_bridge;
  std::atomic<std::uint64_t> next_raw_basis {1};

  struct automatic_ui_entry {
    sunshine_game3d::automatic_status status {};
    bool recalibrate = false;
  };
  class publisher_t;
  struct addon_session_t {
    addon_session_t();
    ~addon_session_t();
    std::mutex ui_mutex;
    std::unordered_map<api::effect_runtime *, automatic_ui_entry> ui;
    std::unique_ptr<publisher_t> publisher;
    enum class phase { inactive, active, stopping };
    std::atomic<phase> lifecycle{phase::inactive};
    bool begin() {
      auto expected = phase::inactive;
      return lifecycle.compare_exchange_strong(expected, phase::active, std::memory_order_acq_rel);
    }
    template<class Cleanup> bool end(Cleanup &&cleanup) {
      auto expected = phase::active;
      if (!lifecycle.compare_exchange_strong(expected, phase::stopping, std::memory_order_acq_rel)) return false;
      struct finish {
        std::atomic<phase> &state;
        ~finish() { state.store(phase::inactive, std::memory_order_release); }
      } finish_teardown{lifecycle};
      // Remove the public owner before invoking anything that can reenter.
      // This local owner alone disposes the publisher after dependent shutdown.
      auto owned_publisher = std::move(publisher);
      std::forward<Cleanup>(cleanup)();
      owned_publisher.reset();
      return true;
    }
  };
  addon_session_t &addon_session();
  // No ReShade/GPU calls while holding this mutex. UI readers and requests never
  // acquire the publisher mutex, including synchronous uniform-set callbacks.
  auto &automatic_ui_mutex = addon_session().ui_mutex;
  auto &automatic_ui = addon_session().ui;

  void clear_automatic_ui(api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    automatic_ui.erase(runtime);
  }

  void publish_automatic_ui(api::effect_runtime *runtime, sunshine_game3d::automatic_status status) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    auto &entry = automatic_ui[runtime];
    if (!entry.recalibrate) {
      if (status.phase != sunshine_game3d::automatic_phase::ready) status.scale.active = false;
      status.scale = sunshine_game3d::retain_automatic_scale(entry.status.scale, status.scale);
      entry.status = status;
    }
  }

  bool take_recalibration(api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    if (found == automatic_ui.end() || !found->second.recalibrate) return false;
    found->second.recalibrate = false;
    return true;
  }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // Only the distinct test DLL can substitute this observation. Null fails closed
  // until its fixture explicitly selects a real window owned by this process.
  std::atomic<HWND> test_foreground_window {nullptr};
#endif

  HWND observed_foreground_window() {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    return test_foreground_window.load(std::memory_order_acquire);
#else
    return GetForegroundWindow();
#endif
  }

  template<class T>
  class com_ptr {
  public:
    com_ptr() = default;

    ~com_ptr() {
      reset();
    }

    com_ptr(const com_ptr &) = delete;
    com_ptr &operator=(const com_ptr &) = delete;

    com_ptr(com_ptr &&other) noexcept:
        ptr_(other.release()) {}

    com_ptr &operator=(com_ptr &&other) noexcept {
      if (this != &other) {
        reset();
        ptr_ = other.release();
      }
      return *this;
    }

    T *get() const {
      return ptr_;
    }

    T *operator->() const {
      return ptr_;
    }

    explicit operator bool() const {
      return ptr_ != nullptr;
    }

    T **put() {
      reset();
      return &ptr_;
    }

    T *release() {
      return std::exchange(ptr_, nullptr);
    }

    void reset() {
      if (auto *old = release()) {
        old->Release();
      }
    }

  private:
    T *ptr_ = nullptr;
  };

  class handle_t {
  public:
    handle_t() = default;

    ~handle_t() {
      if (value_) {
        CloseHandle(value_);
      }
    }

    handle_t(const handle_t &) = delete;
    handle_t &operator=(const handle_t &) = delete;

    HANDLE get() const {
      return value_;
    }

    HANDLE *put() {
      return &value_;
    }

  private:
    HANDLE value_ = nullptr;
  };

  void log(reshade::log::level level, const char *message) {
#ifdef SUNSHINE_SBS_TEST
    (void) level;
    std::puts(message);
#else
    reshade::log::message(level, message);
#endif
  }

  void log_hr(const char *operation, HRESULT hr) {
    char message[256];
    std::snprintf(message, sizeof(message), "Sunshine SBS: %s failed (0x%08lx)", operation, static_cast<unsigned long>(hr));
    log(reshade::log::level::error, message);
  }

  bool exchange_state(wire::slot_t &slot, std::uint64_t generation, wire::slot_state desired, wire::slot_state expected) {
    const auto prior = wire::slot_control(generation, expected);
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(wire::slot_control(generation, desired)), static_cast<LONG64>(prior))) == prior;
  }

  void store_state(wire::slot_t &slot, std::uint64_t generation, wire::slot_state state) {
    InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(wire::slot_control(generation, state)));
  }

  std::uint64_t read_nonce(wire::shared_state_t &shared) {
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(
      reinterpret_cast<volatile LONG64 *>(&shared.consumer_nonce),
      0,
      0
    ));
  }

  DXGI_FORMAT typed_format(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_R8G8B8A8_TYPELESS:
      case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
      case DXGI_FORMAT_B8G8R8A8_TYPELESS:
      case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
      case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
      case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
      default:
        return format;
    }
  }

  struct export_color_t {
    api::color_space input = api::color_space::unknown;
    wire::transfer output = wire::transfer::srgb;
  };

  bool decode_export_color(const char *output, std::uint32_t input, export_color_t &color) {
    if (input == static_cast<std::uint32_t>(api::color_space::srgb) && std::strcmp(output, "srgb") == 0) {
      color = {api::color_space::srgb, wire::transfer::srgb};
      return true;
    }
    if ((input == static_cast<std::uint32_t>(api::color_space::scrgb) || input == static_cast<std::uint32_t>(api::color_space::hdr10_pq)) && std::strcmp(output, "scrgb") == 0) {
      color = {static_cast<api::color_space>(input), wire::transfer::scrgb};
      return true;
    }
    return false;
  }

  struct source_t {
    api::resource resource {};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    export_color_t color;
    std::uint64_t adapter = 0;
    HWND window = nullptr;
  };

  // Own native objects rather than ReShade wrappers, which disappear during runtime teardown.
  struct generation_t {
    api::device_api backend = api::device_api::d3d11;
    api::effect_runtime *runtime = nullptr;
    api::effect_runtime *owner_runtime = nullptr;
    bool owner_destroyed = false;
    std::uint64_t nonce = 0;
    std::uint64_t id = 0;
    std::uint64_t last_submitted = 0;
    std::uint64_t native_swapchain = 0;
    std::uint64_t pending_qpc = 0;
    std::uint32_t pending_slot = wire::slot_count;
    bool signal_failed = false;
    bool pending_overlay = false;
    // Diagnostic disposition travels with this copy until its slot is actually
    // published. A later effects callback must not relabel an earlier copy.
    bool pending_fg_output = false, pending_depth_ready = false, pending_reused_depth = false;
    std::array<std::unique_ptr<sunshine::overlay::compositor_t>, wire::slot_count> overlays;
    // A D3D12 command list does not retain the resource referenced by a recorded copy.
    // Hold the source across effect reload/destruction until submission is completed.
    std::array<com_ptr<IUnknown>, wire::slot_count> submitted_sources;
    source_t source;
    std::array<handle_t, wire::slot_count> texture_handles;
    handle_t fence_handle;
    std::array<com_ptr<ID3D11Texture2D>, wire::slot_count> textures11;
    com_ptr<ID3D11Device5> device11;
    com_ptr<ID3D11DeviceContext4> context11;
    com_ptr<ID3D11Fence> fence11;
    std::array<com_ptr<ID3D12Resource>, wire::slot_count> textures12;
    com_ptr<ID3D12Device> device12;
    com_ptr<ID3D12CommandQueue> queue12;
    com_ptr<ID3D12Fence> fence12;

    std::uint64_t completed() const {
      return fence11 ? fence11->GetCompletedValue() : fence12 ? fence12->GetCompletedValue() :
                                                                0;
    }

    bool finished() const {
      const auto value = completed();
      return value == UINT64_MAX || (pending_slot == wire::slot_count && !signal_failed && value >= last_submitted);
    }

    api::resource texture(std::uint32_t index) const {
      return {reinterpret_cast<std::uint64_t>(backend == api::device_api::d3d11 ? static_cast<void *>(textures11[index].get()) : static_cast<void *>(textures12[index].get()))};
    }

    // Unloading an add-on cannot run a background reaper after its DLL is gone. Preserve only
    // outstanding native allocations until process exit rather than wait on the game's GPU.
    void preserve_inflight_on_unload() {
      if (finished()) {
        return;
      }
      for (auto &texture : textures11) {
        texture.release();
      }
      for (auto &texture : textures12) {
        texture.release();
      }
      fence11.release();
      fence12.release();
      for (auto &source : submitted_sources) source.release();
      for (auto &overlay : overlays) overlay.release();
      log(reshade::log::level::warning, "Sunshine SBS: preserving one in-flight generation until process exit during add-on unload");
    }

    bool create(api::effect_runtime *owner, const source_t &input, std::uint64_t consumer, std::uint64_t generation_id) {
      runtime = owner;
      owner_runtime = owner;
      source = input;
      nonce = consumer;
      id = generation_id;
      native_swapchain = owner->get_native();
      return create_native(owner->get_device()->get_api(), owner->get_device()->get_native(), owner->get_command_queue()->get_native());
    }

    bool create_native(api::device_api renderer, std::uint64_t native_device_handle, std::uint64_t native_queue_handle) {
      backend = renderer;
      HRESULT hr = E_FAIL;
      if (backend == api::device_api::d3d11) {
        auto *native_device = reinterpret_cast<ID3D11Device *>(native_device_handle);
        auto *native_context = reinterpret_cast<ID3D11DeviceContext *>(native_queue_handle);
        if (FAILED(hr = native_device->QueryInterface(IID_PPV_ARGS(device11.put()))) || FAILED(hr = native_context->QueryInterface(IID_PPV_ARGS(context11.put())))) {
          log_hr("D3D11 shared fence interfaces", hr);
          return false;
        }
        if (FAILED(hr = device11->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence11.put()))) || FAILED(hr = fence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, fence_handle.put()))) {
          log_hr("D3D11 shared fence", hr);
          return false;
        }
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = source.width * 2;
        desc.Height = source.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = source.format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        // Modern D3D11 NT sharing with a shared fence. No keyed-mutex waits are involved.
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
          com_ptr<IDXGIResource1> shared;
          if (FAILED(hr = device11->CreateTexture2D(&desc, nullptr, textures11[index].put())) || FAILED(hr = textures11[index]->QueryInterface(IID_PPV_ARGS(shared.put()))) || FAILED(hr = shared->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, texture_handles[index].put()))) {
            log_hr("D3D11 shared texture", hr);
            return false;
          }
        }
      } else if (backend == api::device_api::d3d12) {
        auto *native_device = reinterpret_cast<ID3D12Device *>(native_device_handle);
        auto *native_queue = reinterpret_cast<ID3D12CommandQueue *>(native_queue_handle);
        if (FAILED(hr = native_device->QueryInterface(IID_PPV_ARGS(device12.put()))) || FAILED(hr = native_queue->QueryInterface(IID_PPV_ARGS(queue12.put()))) || FAILED(hr = device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence12.put()))) || FAILED(hr = device12->CreateSharedHandle(fence12.get(), nullptr, GENERIC_ALL, nullptr, fence_handle.put()))) {
          log_hr("D3D12 shared fence", hr);
          return false;
        }
        D3D12_HEAP_PROPERTIES heap {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = source.width * 2;
        desc.Height = source.height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = source.format;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
          if (FAILED(hr = device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(textures12[index].put()))) || FAILED(hr = device12->CreateSharedHandle(textures12[index].get(), nullptr, GENERIC_ALL, nullptr, texture_handles[index].put()))) {
            log_hr("D3D12 shared texture", hr);
            return false;
          }
        }
      } else {
        return false;
      }
      return true;
    }

    bool submit(api::command_list *commands, api::resource input, std::uint32_t index, std::uint64_t sequence) {
      auto &submitted_source = submitted_sources[index];
      submitted_source.reset();
      auto *source_object = reinterpret_cast<IUnknown *>(input.handle);
      source_object->AddRef();
      *submitted_source.put() = source_object;
      const auto destination = texture(index);
      // Set this before appending commands: an exception cannot make recorded work appear idle.
      last_submitted = sequence;
      commands->barrier(input, api::resource_usage::shader_resource, api::resource_usage::copy_source);
      if (backend == api::device_api::d3d12) {
        commands->barrier(destination, api::resource_usage::general, api::resource_usage::copy_dest);
      }
      commands->copy_resource(input, destination);
      if (backend == api::device_api::d3d12) {
        commands->barrier(destination, api::resource_usage::copy_dest, api::resource_usage::general);
      }
      commands->barrier(input, api::resource_usage::copy_source, api::resource_usage::shader_resource);
      // The native overlay is drawn after techniques. Its composition must be included in
      // this same bounded submission before signalling or exposing the slot to Sunshine.
      return true;
    }
  };

  // Value-only decision for one effects pass. Calibration resolves these values
  // first; shader uniforms, UI and export consume the same completed decision.
  // No borrowed textures or independently mutable copies of readiness live here.
  struct scene_parameters_t {
    bool owned = false, ready = false;
    int basis = 0;
    float scale = 0.f, blend = 0.f;
    std::array<float, 2> projection{}, zero{}, raw_range{0.f, 1.f};
    std::array<float, 4> rect{0.f, 0.f, 1.f, 1.f};
    sunshine_game3d::automatic_status ui;
  };
  struct frame_decision_t {
    bool prepared = false, depth_ready = false, fg_active = false, reused_depth = false;
    // Identify the real capture for which scene parameters were resolved. A
    // later generated presentation may reuse that scene, never another source's.
    std::array<std::uint64_t, 4> scene_source{}; // epoch, source, sequence, viewport
    scene_parameters_t scene;
    sunshine_depth_jitter::correction jitter;
  };

  struct runtime_t {
    bool addon_native = false;
    std::uint64_t swapchain = 0;
    std::unique_ptr<sunshine_game3d::renderer> renderer;
    api::resource native_output{};
    api::resource_view borrowed_depth{};
    api::effect_texture_variable texture {};
    std::string effect_name;
    api::effect_technique game_technique {}, native_technique {};
    api::effect_uniform_variable depth_ready {}, calibrated {}, raw_anchor {}, raw_gain {}, depth_rect {}, direction {};
    bool native_uniforms_checked = false;
    bool native_depth_prepared = false;
    api::effect_uniform_variable camera_ready {}, camera_basis {}, camera_projection {}, camera_scale {}, camera_zero {}, camera_blend {}, camera_rect {}, camera_raw_range {};
    api::effect_uniform_variable depth_jitter {};
    bool raw_supported = false;
    std::uint64_t raw_basis_epoch = 0;
    sunshine_raw_scene::pool raw_policy;
    sunshine_raw_scene::policy provided_raw_policy;
    sunshine_projection_depth::controller projection_policy;
    sunshine_projection_depth::domain projection_domain;
    sunshine_diagnostics::log_gate projection_log;
    bool projection_ready = false;
    sunshine_raw_scene::reentry_transition raw_reentry;
    sunshine_raw_scene::status raw_last_status = sunshine_raw_scene::status::uninitialized;
    sunshine_diagnostics::log_gate raw_log;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    export_color_t color;
    bool proof_checked = false;
    bool rendered_since_present = false;
    frame_decision_t frame;
    struct {
      std::uint64_t published_fresh_depth = 0, published_reused_depth = 0;
      std::uint64_t published_depth_missing = 0, next_log = 0;
    } fg_output;
  };

  class publisher_t {
  public:
    ~publisher_t() {
      if (!sunshine_addon_lifetime::stopping()) {
        {
          // The observer may keep this module resident across add-on unload.
          // A new publisher must not inherit status/actions at reused addresses.
          std::lock_guard<std::mutex> lock(automatic_ui_mutex);
          automatic_ui.clear();
        }
        if (shared_) {
          publish(identity_);
        }
        if (generation_) {
          generation_->preserve_inflight_on_unload();
        }
        if (retired_) {
          retired_->preserve_inflight_on_unload();
        }
      }
      if (shared_) {
        UnmapViewOfFile(shared_);
      }
    }

    bool init() {
      FILETIME created {}, exited {}, kernel {}, user {};
      if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        return false;
      }
      identity_.producer_pid = GetCurrentProcessId();
      identity_.producer_creation_time = (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) | created.dwLowDateTime;
      const auto name = std::wstring(wire::mapping_prefix) + std::to_wstring(identity_.producer_pid);
      *mapping_.put() = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(wire::shared_state_t), name.c_str());
      if (!mapping_.get() || GetLastError() == ERROR_ALREADY_EXISTS) {
        log(reshade::log::level::error, "Sunshine SBS: cannot create a unique per-process mapping");
        return false;
      }
      shared_ = static_cast<wire::shared_state_t *>(MapViewOfFile(mapping_.get(), FILE_MAP_ALL_ACCESS, 0, 0, sizeof(wire::shared_state_t)));
      if (!shared_) {
        return false;
      }
      // New paging-file mappings are zeroed; publish the versioned layout before callbacks start.
      new (shared_) wire::shared_state_t {};
      publish(identity_);
      log(reshade::log::level::info, "Sunshine SBS: exporter ready; waiting for native Game 3D or a reference stereo effect and consumer");
      return true;
    }

    void track_runtime(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      runtimes_[runtime].swapchain = runtime->get_native();
    }

    bool native_enabled(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = runtimes_.find(runtime);
      return found != runtimes_.end() && found->second.addon_native;
    }

    void render_present(api::swapchain *swapchain) {
      api::effect_runtime *runtime = nullptr;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[owner, proof] : runtimes_) if (proof.swapchain == swapchain->get_native()) { runtime = owner; break; }
      }
      if (!runtime) return;
      const auto settings = sunshine_game3d::query_render_settings(runtime);
      sunshine_depth::set_native_driver(runtime, settings.enabled);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        runtimes_[runtime].addon_native = settings.enabled;
      }
      if (!settings.enabled) {
        sunshine_depth::set_raw_scene_request(runtime, 0, false);
        return;
      }
      auto *device = runtime->get_device();
      const auto backend = device->get_api();
      auto *queue = runtime->get_command_queue();
      if (!queue || (backend != api::device_api::d3d11 && backend != api::device_api::d3d12)) return;
      auto *commands = queue->get_immediate_command_list();
      const auto index = runtime->get_current_back_buffer_index();
      auto *native_swapchain = reinterpret_cast<IDXGISwapChain *>(swapchain->get_native());
      com_ptr<ID3D11Texture2D> back11;
      com_ptr<ID3D12Resource> back12;
      api::resource backbuffer{};
      if (backend == api::device_api::d3d11) {
        if (FAILED(native_swapchain->GetBuffer(index, IID_PPV_ARGS(back11.put())))) return;
        backbuffer.handle = reinterpret_cast<uint64_t>(back11.get());
      } else {
        if (FAILED(native_swapchain->GetBuffer(index, IID_PPV_ARGS(back12.put())))) return;
        backbuffer.handle = reinterpret_cast<uint64_t>(back12.get());
      }
      sunshine_game3d::renderer *renderer = nullptr;
      api::resource_view rtv{};
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &proof = runtimes_[runtime];
        if (!proof.renderer) proof.renderer = std::make_unique<sunshine_game3d::renderer>();
        renderer = proof.renderer.get();
        if (!renderer->configure(runtime, backbuffer, swapchain->get_color_space())) {
          deactivate(runtime);
          sunshine_game3d::automatic_status unavailable;
          unavailable.phase = sunshine_game3d::automatic_phase::renderer_unavailable;
          publish_automatic_ui(runtime, unavailable);
          return;
        }
        rtv = renderer->native_rtv(backbuffer);
        const auto desc = device->get_resource_desc(backbuffer);
        proof.width = desc.texture.width; proof.height = desc.texture.height;
        proof.color = {swapchain->get_color_space(), swapchain->get_color_space() == api::color_space::srgb ? wire::transfer::srgb : wire::transfer::scrgb};
      }
      if (!rtv.handle) return;
      renderer->begin_frame_state();
      struct restore_game_state {
        sunshine_game3d::renderer *renderer;
        ~restore_game_state() { renderer->end_frame_state(); }
      } restore{renderer};
      // The owner brackets the same capture used by the FX reference path. No
      // raw resource pointer is retained once this lease closes.
      if (!sunshine_depth::begin_native_frame(runtime, commands)) return;
      struct finish_depth {
        api::effect_runtime *runtime; api::command_list *commands;
        ~finish_depth() { sunshine_depth::end_native_frame(runtime, commands); }
      } finish{runtime, commands};
      prepare_native_depth(runtime, commands, rtv);
      bool rendered = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &proof = runtimes_[runtime];
        const auto &scene = proof.frame.scene;
        sunshine_game3d::render_parameters p;
        p.strength = settings.strength; p.depth_view = settings.depth_view;
        p.depth_ready = proof.frame.depth_ready; p.camera_ready = scene.ready;
        p.coordinate_basis = scene.basis; p.depth_scale = scene.scale; p.strength_blend = scene.blend;
        p.projection = scene.projection; p.raw_depth_range = scene.raw_range;
        p.convergence = scene.zero; p.jitter = proof.frame.jitter.offset; p.depth_rect = scene.rect;
        rendered = proof.frame.prepared && renderer->render(commands, backbuffer, proof.borrowed_depth, p);
        proof.borrowed_depth = {};
        proof.native_output = rendered ? renderer->output() : api::resource{};
      }
      if (rendered) frame(runtime, {}, commands, rtv, true);
    }

    void invalidate(api::effect_runtime *runtime, bool destroy, std::uint64_t native_swapchain = 0) {
      // Terminal lifecycle events cannot be dropped: the runtime address may be reused.
      // The lock never encloses a CPU/GPU fence wait.
      std::lock_guard<std::mutex> lock(mutex_);
      const auto current = runtimes_.find(runtime);
      if (!destroy && current != runtimes_.end() && current->second.addon_native) {
        // Effects can disappear while the native owner and scene continue.
        // Any later return to a reference must rediscover all FX handles.
        auto &proof = current->second;
        proof.native_uniforms_checked = false;
        proof.game_technique = proof.native_technique = {};
        proof.depth_ready = proof.calibrated = proof.raw_anchor = proof.raw_gain = proof.depth_rect = proof.direction = {};
        proof.camera_ready = proof.camera_basis = proof.camera_projection = proof.camera_scale = proof.camera_zero = proof.camera_blend = proof.camera_rect = proof.camera_raw_range = proof.depth_jitter = {};
        proof.texture = {}; proof.proof_checked = false;
        return;
      }
      if (destroy && current != runtimes_.end() && current->second.renderer)
        current->second.renderer->reset_after_runtime_drain();
      sunshine_streamline::provider::invalidate_reused_depth(runtime);
      deactivate(runtime);
      if (destroy) runtimes_.erase(runtime);
      else {
        // FX reload invalidates FX handles, not the runtime identity or native
        // GPU owner. Native can be re-enabled even after a disabled reload.
        auto &proof = runtimes_[runtime];
        auto renderer = std::move(proof.renderer);
        const auto swapchain = proof.swapchain;
        proof = runtime_t{};
        proof.swapchain = swapchain;
        proof.renderer = std::move(renderer);
      }
      clear_automatic_ui(runtime);
      if (generation_ && generation_->owner_runtime == runtime) {
        generation_->owner_destroyed |= destroy;
        if (generation_->finished()) {
          generation_.reset();
        } else if (destroy) {
          retire_destroyed_generation();
        }
      }
      if (destroy) {
        colors_.erase(native_swapchain);
        overlays_.erase(runtime);
      }
    }

    void set_overlay_open(api::effect_runtime *runtime, bool open) {
      std::lock_guard<std::mutex> lock(mutex_);
      // Overlay visibility belongs to the runtime, independently of effect recompilation.
      overlays_[runtime] = open;
      log(reshade::log::level::info, open ? "Sunshine SBS: ReShade overlay opened; composing controls over both stereo eyes" : "Sunshine SBS: ReShade overlay closed");
    }

    void present(api::effect_runtime *runtime) {
      // Disable stale metadata when focus moves, including when no technique executes.
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = runtimes_.find(runtime);
      if (published_runtime_ == runtime && (observed_foreground_window() != reinterpret_cast<HWND>(published_metadata_.window) || found == runtimes_.end() || !found->second.rendered_since_present)) {
        sunshine_streamline::provider::invalidate_reused_depth(runtime);
        deactivate(runtime);
      }
      if (generation_ && generation_->owner_runtime == runtime && generation_->pending_slot != wire::slot_count) {
        if (generation_->pending_overlay) {
          generation_->pending_overlay = false;
          if (generation_->runtime && !generation_->overlays[generation_->pending_slot]->finish()) {
            deactivate(runtime);
            retry_at_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            log(reshade::log::level::warning, "Sunshine SBS: overlay capture unavailable; showing the captured desktop");
          }
        }
        if (generation_->backend == api::device_api::d3d11) {
          const auto index = generation_->pending_slot;
          generation_->pending_slot = wire::slot_count;
          const HRESULT hr = generation_->context11->Signal(generation_->fence11.get(), generation_->last_submitted);
          generation_->context11->Flush();
          if (FAILED(hr)) {
            generation_->signal_failed = true;
            log_hr("D3D11 publication fence signal", hr);
            deactivate(runtime);
          } else if (generation_->runtime && published_runtime_ == runtime && read_nonce(*shared_) == generation_->nonce) {
            complete_slot(index, generation_->pending_qpc);
          }
        }
      }
      if (found != runtimes_.end()) {
        found->second.rendered_since_present = false;
        found->second.native_depth_prepared = false;
        found->second.frame.prepared = false;
      }
    }

    void draw_overlay(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (generation_ && generation_->runtime == runtime && generation_->pending_overlay) {
        generation_->overlays[generation_->pending_slot]->append_capture_callback();
      }
    }

    void prepare_native_depth(api::effect_runtime *runtime, api::command_list *commands,
                              api::resource_view effects_rtv) {
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock.owns_lock()) return;
      auto &proof = runtimes_[runtime];
      const auto previous_frame = proof.frame;
      proof.native_depth_prepared = false;
      proof.frame = {};
      if (proof.addon_native) proof.raw_supported = true;
      if (!proof.addon_native && !proof.native_uniforms_checked) {
        proof.native_uniforms_checked = true;
        runtime->enumerate_techniques("SunshineGame3D.fx", [&](api::effect_runtime *owner, api::effect_technique technique) {
          char name[128] {};
          owner->get_technique_name(technique, name);
          if (std::strcmp(name, "SunshineGame3D") == 0) proof.game_technique = technique;
        });
        runtime->enumerate_techniques("SunshineDepth3D.fx", [&](api::effect_runtime *owner, api::effect_technique technique) {
          char name[128] {};
          owner->get_technique_name(technique, name);
          if (std::strcmp(name, "SunshineDepth3D") == 0) proof.native_technique = technique;
        });
        runtime->enumerate_uniform_variables("SunshineDepth3D.fx", [&](api::effect_runtime *owner, api::effect_uniform_variable variable) {
          char qualified[160] {};
          owner->get_uniform_variable_name(variable, qualified);
          const char *name = std::strrchr(qualified, ':');
          name = name ? name + 1 : qualified;
          if (std::strcmp(name, "Sunshine_DepthReady") == 0) proof.depth_ready = variable;
          else if (std::strcmp(name, "Sunshine_Calibrated") == 0) proof.calibrated = variable;
          else if (std::strcmp(name, "Sunshine_RawAnchor") == 0) proof.raw_anchor = variable;
          else if (std::strcmp(name, "Sunshine_RawGain") == 0) proof.raw_gain = variable;
          else if (std::strcmp(name, "Sunshine_DepthRect") == 0) proof.depth_rect = variable;
          else if (std::strcmp(name, "DepthDirection") == 0) proof.direction = variable;
        });
        runtime->enumerate_uniform_variables("SunshineGame3D.fx", [&](api::effect_runtime *owner, api::effect_uniform_variable variable) {
          char qualified[160] {};
          owner->get_uniform_variable_name(variable, qualified);
          const char *name = std::strrchr(qualified, ':');
          name = name ? name + 1 : qualified;
          if (std::strcmp(name, "Sunshine_CameraDepthReady") == 0) proof.camera_ready = variable;
          else if (std::strcmp(name, "Sunshine_CameraCoordinateBasis") == 0) proof.camera_basis = variable;
          else if (std::strcmp(name, "Sunshine_CameraProjection") == 0) proof.camera_projection = variable;
          else if (std::strcmp(name, "Sunshine_CameraDepthScale") == 0) proof.camera_scale = variable;
          else if (std::strcmp(name, "Sunshine_CameraConvergence") == 0) proof.camera_zero = variable;
          else if (std::strcmp(name, "Sunshine_CameraStrengthBlend") == 0) proof.camera_blend = variable;
          else if (std::strcmp(name, "Sunshine_CameraDepthRect") == 0) proof.camera_rect = variable;
          else if (std::strcmp(name, "Sunshine_DepthJitter") == 0) proof.depth_jitter = variable;
          else if (std::strcmp(name, "Sunshine_CameraRawDepthRange") == 0) proof.camera_raw_range = variable;
        });
        // Require the source-owned geometry interface; preset mode macros no
        // longer select an alternate renderer.
        proof.raw_supported = proof.camera_ready.handle && proof.camera_basis.handle &&
          proof.camera_projection.handle && proof.camera_scale.handle && proof.camera_zero.handle &&
          proof.camera_blend.handle && proof.camera_rect.handle;

      }
      // Fail closed before reading mutable capture/policy state. Publication
      // below is the only writer that can enable the resolved camera branch.
      if (!proof.addon_native && proof.camera_ready.handle) runtime->set_uniform_value_bool(proof.camera_ready, false);
      // The selector's begin-effects callback runs first. Its current capture is
      // shader-readable now; calibration history never substitutes for readiness.
      sunshine_depth::frame_depth depth;
      const bool game_enabled = proof.addon_native || (proof.game_technique.handle && runtime->get_technique_state(proof.game_technique));
      if (!game_enabled) sunshine_streamline::provider::invalidate_reused_depth(runtime);
      const bool independent_enabled = !game_enabled && proof.native_technique.handle && runtime->get_technique_state(proof.native_technique);
      if (independent_enabled && proof.depth_ready.handle && proof.calibrated.handle && proof.raw_anchor.handle &&
          proof.raw_gain.handle && proof.depth_rect.handle && proof.direction.handle) {
        int direction = 0;
        runtime->get_uniform_value_int(proof.direction, &direction, 1);
        sunshine_depth::get_frame_depth(runtime, depth, direction == 1 ? sunshine_depth::depth_orientation::normal :
          direction == 2 ? sunshine_depth::depth_orientation::reversed : sunshine_depth::depth_orientation::automatic);
        apply_native_depth(runtime, proof, depth);
      } else {
        // The original-derived effect owns its preparation and convergence.
        // Camera diagnostics may inspect capture identity without enabling the
        // independent renderer's percentile calibration or altering uniforms.
        sunshine_depth::get_frame_depth(runtime, depth, sunshine_depth::depth_orientation::automatic, false);
      }
      proof.borrowed_depth = proof.addon_native && depth.ready ? depth.shader_resource : api::resource_view{};
      // Shared capture observation also runs when every API source is disabled.
      // Deferred hook discovery separately serves lightweight source nomination.
      // Detailed camera/content observations remain diagnostic-only.
      if (sunshine_streamline::depth_capture::active() || sunshine_streamline::enabled() || sunshine_streamline::source_enabled() ||
          sunshine_upscaler_trace::enabled() || sunshine_upscaler_trace::capture_enabled()) {
        auto selected = sunshine_depth::camera_selection_snapshot(depth);
        // ReShade can resolve/copy the swapchain before effects. Observe the
        // actual effects input while its view is valid, never a guessed backbuffer.
        // This is identity/extent evidence, not proof that color and depth share UVs.
        if (sunshine_streamline::enabled() && effects_rtv.handle && commands) {
          auto *device = runtime->get_device();
          const auto resource = device->get_resource_from_view(effects_rtv);
          if (resource.handle) {
            const auto desc = device->get_resource_desc(resource);
            if (desc.type == api::resource_type::texture_2d || desc.type == api::resource_type::surface) {
              auto &input = selected.effects_input;
              input.runtime = reinterpret_cast<std::uint64_t>(runtime);
              input.device = device->get_native();
              input.command = commands->get_native();
              input.resource = resource.handle;
              input.width = desc.texture.width;
              input.height = desc.texture.height;
              input.format = static_cast<std::uint32_t>(desc.texture.format);
              input.ready = input.width && input.height;
            }
          }
        }
        sunshine_streamline::poll(selected);
        if (sunshine_streamline::enabled())
          sunshine_depth::report_camera_observations(runtime, selected);
      }
      const auto now = GetTickCount64();
      frame_decision_t frame;
      frame.depth_ready = depth.ready;
      if (depth.ready)
        frame.jitter = sunshine_depth_jitter::make(depth.provided.jitter, depth.width, depth.height,
          {depth.x, depth.y, depth.active_width, depth.active_height});
      frame.reused_depth = depth.reused_depth;
      frame.scene_source = {depth.provided.epoch, depth.provided.source_id,
        depth.provided.sequence, depth.provided.viewport};
      // Source choice and FG reuse are resolved once by the depth provider.
      // Another fallible metadata query here could contradict this same frame.
      frame.fg_active = game_enabled && depth.frame_generation_active;
      if (depth.reused_depth) {
        // The provider owns the short, invalidation-aware depth reuse window.
        // Render current color with those pixels and their previously resolved
        // geometry, without advancing calibration, samples or readiness ramps.
        // User strength remains a live shader uniform; an explicit recalibration
        // request is consumed when the next fresh real-depth frame arrives.
        frame.scene.owned = proof.raw_supported;
        if (frame.fg_active && previous_frame.scene.ready &&
            previous_frame.scene_source == frame.scene_source)
          frame.scene = previous_frame.scene;
        else if (proof.raw_supported) {
          frame.scene.ui = {sunshine_game3d::automatic_phase::waiting_for_depth, true};
          frame.depth_ready = false;
        }
      } else {
        const bool provided = sunshine_streamline::provider::selected(runtime);
        frame.scene = resolve_raw_scene(runtime, proof, depth, game_enabled, provided, now);
      }
      publish_frame(runtime, proof, std::move(frame));
    }

    scene_parameters_t resolve_raw_scene(api::effect_runtime *runtime, runtime_t &proof,
        const sunshine_depth::frame_depth &depth, bool game_enabled, bool api_selected, std::uint64_t now) {
      scene_parameters_t scene;
      scene.owned = proof.raw_supported;
      const bool enabled = proof.raw_supported && game_enabled;
      const bool provided = enabled && api_selected;
      const bool projection_provided = provided && depth.projection.supplied;
      const auto restart_reference = [&proof, now] {
        // Explicit recalibration invalidates queued samples. The raw controller
        // owns ordinary source changes and initializes each source only once.
        proof.raw_basis_epoch = next_raw_basis.fetch_add(1, std::memory_order_relaxed);
        proof.raw_policy.reset(proof.raw_basis_epoch, now);
        proof.provided_raw_policy.reset(proof.raw_basis_epoch, now);
        proof.raw_last_status = sunshine_raw_scene::status::uninitialized;
        proof.raw_log = {};
        proof.raw_reentry = {};
      };
      if (take_recalibration(runtime)) {
        if (projection_provided) {
          const sunshine_projection_depth::domain domain{depth.projection.epoch, depth.projection.viewport};
          if (!(domain == proof.projection_domain)) {
            proof.projection_domain = domain;
            proof.projection_policy.reset(domain, now);
          }
          proof.projection_policy.reset_reference(domain, now);
          proof.raw_reentry = {};
          log(reshade::log::level::info, "Sunshine 3D Streamline: user requested Recenter; acquiring a fresh screen plane and coupled stereo reference");
        } else {
          restart_reference();
          // latest_center may still describe a capture from the action's same
          // millisecond. Rewrapping it in the new raw epoch must not make it new.
          proof.raw_policy.scene_cut(now);
          proof.provided_raw_policy.scene_cut(now);
          log(reshade::log::level::info, "Sunshine 3D raw automation: user requested Recenter; acquiring a fresh screen plane and coupled stereo reference");
        }
      }
      if (enabled && !proof.raw_basis_epoch)
        restart_reference();
      sunshine_depth::set_raw_scene_request(runtime, proof.raw_basis_epoch, enabled && !provided);
      using phase = sunshine_game3d::automatic_phase;
      if (!proof.raw_supported) {
        return scene; // No competing camera-uniform writer.
      }
      const bool cropped = depth.ready && (depth.x || depth.y || depth.active_width != depth.width || depth.active_height != depth.height);
      // Native constants always carry the active rectangle. Only reference
      // effects need to prove that their older uniform interface supports it.
      if (proof.addon_native || proof.camera_rect.handle) {
        if (depth.ready && depth.width && depth.height) {
          scene.rect = {float(depth.x) / depth.width, float(depth.y) / depth.height,
            float(depth.active_width) / depth.width, float(depth.active_height) / depth.height};
        }
      } else if (cropped) {
        return scene; // An older shader must not sample allocation padding as scene depth.
      }
      if (projection_provided) {
        resolve_streamline_scene(runtime, proof, depth, now, scene);
        return scene;
      }
      proof.projection_policy.suspend();
      if (!enabled || !proof.raw_basis_epoch) {
        proof.raw_policy.scene_cut(now);
        proof.provided_raw_policy.scene_cut(now);
        proof.raw_reentry.update(false, now);
        scene.ui = {phase::waiting_for_depth, false};
        return scene;
      }

      sunshine_raw_scene::output state;
      bool source_changed = false;
      if (provided) {
        const auto current = sunshine_provided_raw::selected(depth.provided, proof.raw_basis_epoch, depth.ready);
        proof.provided_raw_policy.bind(current, now);
        sunshine_streamline::provider::center_sample measured;
        if (sunshine_streamline::provider::latest_center(runtime, measured)) {
          auto sample = sunshine_provided_raw::measured(measured.metadata, measured.id,
            proof.raw_basis_epoch, measured.raw);
          proof.provided_raw_policy.observe(sample, current.frame, now);
        }
        state = proof.provided_raw_policy.evaluate(current, now);
      } else {
        sunshine_raw_scene::selected_frame current;
        current.basis_epoch = proof.raw_basis_epoch;
        current.layout_epoch = depth.layout_epoch;
        current.source = {depth.source_resource.handle, depth.source_id, 0, depth.width, depth.height,
          depth.x, depth.y, depth.active_width, depth.active_height};
        current.frame = {depth.frame_index, depth.runtime_epoch};
        current.direction = depth.detected_orientation;
        current.depth_ready = depth.ready;
        current.aligned_viewport_assumed = depth.aligned_viewport_assumed;
        sunshine_depth::raw_source_roster roster;
        sunshine_depth::get_raw_source_roster(runtime, roster);
        // Capture slots are transient; only authoritative resource/basis changes
        // invalidate retained calibration. Validate all bounded histories at once.
        const auto history = proof.raw_policy.history_sources();
        proof.raw_policy.retain_history(sunshine_depth::validate_raw_history(runtime, history));
        source_changed = depth.source_id && !proof.raw_policy.contains(current);
        // Capture owns sample identity and source lifetime; calibration consumes
        // that evidence independently of which source can render this present.
        // A missing current copy never prevents a completed peer sample from
        // reaching its own controller. Membership cannot be adopted from a packet.
        if (proof.raw_policy.synchronize(roster, now)) {
          std::array<sunshine_raw_scene::sample, sunshine_raw_scene::maximum_sources> samples;
          const auto count = sunshine_depth::get_raw_samples(runtime, samples);
          for (unsigned i = 0; i < count; ++i) {
            proof.raw_policy.observe(samples[i], now);
          }
        }
        // Rendering separately requires this present's selected, usable copy.
        // The policy retains numeric history when this evaluation returns mono.
        state = proof.raw_policy.evaluate(current, now);
        }
      scene.blend = proof.raw_reentry.update(state.ready, now);
      {
        using reason = sunshine_raw_scene::status;
        phase display = phase::waiting_for_depth;
        if (state.ready) display = phase::ready;
        else if (state.reason == reason::calibrating)
          display = phase::calibrating;
        else if (state.reason == reason::unsupported_shader_domain)
          display = phase::suspended;
        scene.ui = {display, true, {
          depth.ready ? sunshine_game3d::automatic_scale_basis::relative_depth : sunshine_game3d::automatic_scale_basis::unknown,
          state.H, state.ready, static_cast<float>(state.target_H)}};
      }
      const bool status_changed = state.reason != proof.raw_last_status;
      // Rotation through already admitted members must not bypass the log gate
      // every frame. A newly admitted physical basis is still reported promptly.
      const bool suspended = status_changed && state.reason == sunshine_raw_scene::status::unsupported_shader_domain;
      if (proof.raw_log.due(now, status_changed, source_changed || suspended, state.ready)) {
        char message[448] {};
        std::snprintf(message, sizeof(message), "Sunshine 3D raw automation: %s; source=%llu; samples=%u stereo_scale=%.9g target_scale=%.9g t0=%.9g plane_state=%s; uncalibrated relative raw basis; depth_path=%s feedback_revision=%llu",
          sunshine_raw_scene::name(state.reason), static_cast<unsigned long long>(depth.source_id),
          state.calibration_samples, state.H, state.target_H, state.t0, sunshine_scene_gain::name(state.learning), provided ?
            (depth.provided.provider == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "SL") : "Generic",
          static_cast<unsigned long long>(provided ? depth.provided.feedback.revision : 0));
        log(reshade::log::level::info, message);
        proof.raw_last_status = state.reason;
      }
      scene.ready = state.ready;
      scene.basis = 1; // Explicitly not recovered camera coefficients.
      scene.projection = {state.shader_A, state.shader_inverseB};
      scene.zero = {state.referenceZPD, state.t0};
      scene.scale = state.H;
      return scene;
    }

    void resolve_streamline_scene(api::effect_runtime *runtime, runtime_t &proof,
        const sunshine_depth::frame_depth &depth, std::uint64_t now, scene_parameters_t &scene) {
      namespace projection = sunshine_projection_depth;
      using phase = sunshine_game3d::automatic_phase;
      bool ready = false;
      projection::coefficients coefficients;
      projection::center_output center;
      if (depth.ready && depth.projection.supplied) {
        const projection::domain domain{depth.projection.epoch, depth.projection.viewport};
        if (!(domain == proof.projection_domain)) {
          proof.projection_domain = domain;
          proof.projection_policy.reset(domain, now);
          proof.raw_reentry = {};
        }
        coefficients = projection::make(depth.projection.A, depth.projection.B,
          depth.projection.raw_scale, depth.projection.raw_bias);
        if (!proof.addon_native && !proof.camera_raw_range.handle &&
            (depth.projection.raw_scale != 1.0 || depth.projection.raw_bias != 0.0)) coefficients = {};
        proof.projection_policy.synchronize_feedback(depth.provided.feedback);
        sunshine_streamline::provider::center_sample measured;
        if (sunshine_streamline::provider::latest_center(runtime, measured)) {
          projection::sample sample;
          sample.id = measured.id; sample.capture_ms = measured.tick;
          sample.logical_domain = {measured.projection.epoch, measured.projection.viewport};
          sample.projection = projection::make(measured.projection.A, measured.projection.B,
            measured.projection.raw_scale, measured.projection.raw_bias);
          sample.raw = measured.raw;
          sample.feedback = measured.metadata.feedback;
          proof.projection_policy.observe(sample, now);
        }
        center = proof.projection_policy.evaluate(domain, coefficients, now);
        ready = center.ready;
      } else {
        proof.projection_policy.suspend();
      }
      scene.blend = proof.raw_reentry.update(ready, now, sunshine_raw_scene::stereo_basis::projection);
      const bool unsupported = center.reason == projection::center_status::unsupported_shader_domain ||
        center.reason == projection::center_status::invalid_projection;
      scene.ui = {
        ready ? phase::ready : !depth.ready ? phase::waiting_for_depth :
          unsupported ? phase::suspended : phase::calibrating, true,
        {sunshine_game3d::automatic_scale_basis::camera_matrix, center.K, ready, static_cast<float>(center.target_K)}};
      if (coefficients.valid()) {
        // Show the same affine conversion the shader applies, including any
        // packed raw-depth transform. Its units differ from the scene-estimated
        // stereo normalization K; neither display value drives rendering.
        scene.ui.scale.conversion_multiplier = coefficients.inverseB;
        scene.ui.scale.conversion_offset = -static_cast<double>(coefficients.shader_A) * coefficients.inverseB;
        scene.ui.scale.has_projection_conversion = true;
      }
      if (proof.projection_log.due(now, ready != proof.projection_ready, false, ready)) {
        char message[640]{};
        if (!depth.ready) {
          // No coefficients or center were evaluated on this pass. Reporting
          // their default zeros would falsely imply that calibration reset.
          std::snprintf(message, sizeof(message), "Sunshine 3D Streamline scale: waiting_for_depth; viewport=%u; retaining calibration state",
            depth.projection.viewport);
        } else {
          std::snprintf(message, sizeof(message), "Sunshine 3D Streamline scale: %s; viewport=%u A=%.9g B=%.9g near=%.9g stereo_scale=%.9g target_scale=%.9g q0=%.9g conversion_scale=%.9g conversion_offset=%.9g samples=%u plane_state=%s feedback_revision=%llu; projection depth, zero-plane stereo reference",
            ready ? "ready" : projection::name(center.reason), depth.projection.viewport,
            depth.projection.A, depth.projection.B, coefficients.near_plane, center.K, center.target_K, center.q0,
            scene.ui.scale.conversion_multiplier, scene.ui.scale.conversion_offset,
            center.calibration_samples, sunshine_scene_gain::name(center.learning),
            static_cast<unsigned long long>(depth.provided.feedback.revision));
        }
        log(reshade::log::level::info, message);
        proof.projection_ready = ready;
      }
      scene.ready = ready;
      scene.projection = {coefficients.shader_A, coefficients.inverseB};
      scene.zero = {projection::reference_zpd, center.q0};
      scene.raw_range = {coefficients.raw_min, coefficients.raw_max};
      scene.scale = center.K;
    }

    static void publish_frame(api::effect_runtime *runtime, runtime_t &proof, frame_decision_t frame) {
      // Publish from this capture, including on FG reuse. Missing
      // metadata clears the prior offset; it must not borrow the newest jitter.
      if (!proof.addon_native && proof.depth_jitter.handle) runtime->set_uniform_value_float(proof.depth_jitter, frame.jitter.offset.data(), 2);
      const auto &scene = frame.scene;
      if (scene.owned && !proof.addon_native) {
        if (proof.camera_rect.handle) runtime->set_uniform_value_float(proof.camera_rect, scene.rect.data(), 4);
        if (proof.camera_blend.handle) runtime->set_uniform_value_float(proof.camera_blend, scene.blend);
        if (scene.ready) {
          runtime->set_uniform_value_int(proof.camera_basis, scene.basis);
          runtime->set_uniform_value_float(proof.camera_projection, scene.projection.data(), 2);
          runtime->set_uniform_value_float(proof.camera_scale, scene.scale);
          runtime->set_uniform_value_float(proof.camera_zero, scene.zero.data(), 2);
          if (proof.camera_raw_range.handle) runtime->set_uniform_value_float(proof.camera_raw_range, scene.raw_range.data(), 2);
          runtime->set_uniform_value_bool(proof.camera_ready, true);
        }
      }
      auto ui = scene.ui;
      if (proof.addon_native && (proof.width > 3840 || proof.height > 3840))
        ui.phase = sunshine_game3d::automatic_phase::unsupported_resolution;
      publish_automatic_ui(runtime, ui);
      frame.prepared = true;
      proof.frame = std::move(frame);
    }

    void begin_present(std::uint64_t swapchain, api::color_space color) {
      std::lock_guard<std::mutex> lock(mutex_);
      // A preceding Present did not reach finish_present. Retain its copy until a later
      // matching submission can be fenced, but stop exposing the previous cached image.
      if (generation_ && generation_->pending_slot != wire::slot_count && generation_->native_swapchain == swapchain) {
        deactivate(generation_->runtime);
      }
      colors_[swapchain] = color;
    }

    void finish_present(std::uint64_t queue, std::uint64_t swapchain) {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto &[runtime, proof] : runtimes_)
        if (proof.swapchain == swapchain && proof.renderer) proof.renderer->finish_present();
      if (!generation_ || generation_->backend != api::device_api::d3d12 || generation_->pending_slot == wire::slot_count || swapchain != generation_->native_swapchain || queue != reinterpret_cast<std::uint64_t>(generation_->queue12.get())) {
        return;
      }
      const auto index = generation_->pending_slot;
      generation_->pending_slot = wire::slot_count;
      const HRESULT hr = generation_->queue12->Signal(generation_->fence12.get(), generation_->last_submitted);
      if (FAILED(hr)) {
        generation_->signal_failed = true;
        log_hr("D3D12 publication fence signal", hr);
        deactivate(generation_->runtime);
        return;
      }
      // Reload/focus/nonce changes may have retired this submission after it was recorded.
      // It still needs a fence, but must never republish its image.
      if (generation_->runtime && published_runtime_ == generation_->runtime && read_nonce(*shared_) == generation_->nonce) {
        complete_slot(index, generation_->pending_qpc);
      }
    }

    void frame(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *commands, api::resource_view native_rtv, bool addon_render = false) {
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock.owns_lock()) {
        return;
      }
      char effect[128] {}, name[128] {};
      if (!addon_render) {
        runtime->get_technique_effect_name(technique, effect);
        runtime->get_technique_name(technique, name);
      }
      const bool native = std::strcmp(effect, "SunshineDepth3D.fx") == 0 && std::strcmp(name, "SunshineDepth3D") == 0;
      const bool legacy = std::strcmp(effect, "SuperDepth3D.fx") == 0 && std::strcmp(name, "SuperDepth3D") == 0;
      const bool game = addon_render || (std::strcmp(effect, "SunshineGame3D.fx") == 0 && std::strcmp(name, "SunshineGame3D") == 0);
      if (!native && !legacy && !game) {
        return;
      }
      auto &proof = runtimes_[runtime];
      if (proof.addon_native && !addon_render) return;
      if (game && proof.raw_supported && !proof.frame.prepared) {
        deactivate(runtime);
        return;
      }
      // The renamed primary effect wins over accidentally enabled reference
      // effects. Never alternate producers between techniques in one present.
      if (!game && proof.game_technique.handle && runtime->get_technique_state(proof.game_technique)) return;
      if (native && !proof.native_depth_prepared) {
        deactivate(runtime);
        return;
      }
      // Preserve the older independent-over-reference priority when the renamed
      // primary effect is not enabled.
      if (legacy && proof.native_technique.handle && runtime->get_technique_state(proof.native_technique)) {
        return;
      }
      if (!addon_render && (!proof.proof_checked || proof.effect_name != effect)) {
        discover(runtime, proof, effect);
      }
      // Diagnostics also cover unfocused/consumer-free frames and failed export
      // validation. A successful publication is not evidence of useful depth.
      const HWND window = static_cast<HWND>(runtime->get_hwnd());
      DWORD window_pid = 0;
      GetWindowThreadProcessId(window, &window_pid);
      if (!window || window_pid != identity_.producer_pid || observed_foreground_window() != window) {
        sunshine_streamline::provider::invalidate_reused_depth(runtime);
        deactivate(runtime);
        return;
      }
      const auto color = colors_.find(runtime->get_native());
      if (color == colors_.end()) {
        deactivate(runtime);
        return;
      }
      // Add-ons may render techniques into deferred lists. Never signal an unrelated queue here.
      auto *queue = runtime->get_command_queue();
      if (!queue || commands != queue->get_immediate_command_list()) {
        deactivate(runtime);
        return;
      }
      const auto backend = runtime->get_device()->get_api();
      if (backend != api::device_api::d3d11 && backend != api::device_api::d3d12) {
        deactivate(runtime);
        return;
      }

      // The compiled input transfer must match the current game swapchain, including PQ to
      // scRGB switches that keep the exported FP16 format and transfer unchanged.
      if ((addon_render ? !proof.native_output.handle : !proof.texture.handle) || !source_color_matches(runtime, proof, color->second)) {
        deactivate(runtime);
        return;
      }
      source_t source;
      if (!describe(runtime, proof, source)) {
        deactivate(runtime);
        return;
      }
      proof.rendered_since_present = true;

      const auto nonce = read_nonce(*shared_);
      if (!nonce) {
        if (generation_) {
          deactivate(generation_->runtime);
        }
        wire::metadata_t metadata = identity_;
        set_source(metadata, source);
        // No consumer means no shared GPU allocations or copies.
        if (published_runtime_ != runtime || published_metadata_.generation || published_metadata_.window != metadata.window || published_metadata_.source_width != metadata.source_width || published_metadata_.source_height != metadata.source_height || published_metadata_.dxgi_format != metadata.dxgi_format || published_metadata_.adapter_luid != metadata.adapter_luid || published_metadata_.color_transfer != metadata.color_transfer) {
          publish(metadata, false, runtime);
        }
        return;
      }
      const bool replace = !generation_ || generation_->runtime != runtime || generation_->nonce != nonce ||
                           generation_->source.width != source.width || generation_->source.height != source.height ||
                           generation_->source.format != source.format || generation_->source.adapter != source.adapter ||
                           generation_->source.color.input != source.color.input || generation_->source.color.output != source.color.output ||
                           generation_->source.window != source.window;
      if (replace) {
        if (generation_) {
          deactivate(generation_->runtime);
        }
        retire_destroyed_generation();
        if (generation_ && !generation_->finished()) {
          return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < retry_at_) {
          return;
        }
        if (next_generation_ >= wire::max_generation) {
          deactivate(runtime);
          return;
        }
        auto next = std::make_unique<generation_t>();
        if (!next->create(runtime, source, nonce, ++next_generation_)) {
          deactivate(runtime);
          retry_at_ = now + std::chrono::seconds(2);
          return;
        }
        generation_ = std::move(next);
        wire::metadata_t metadata = identity_;
        set_source(metadata, source);
        metadata.generation = generation_->id;
        metadata.accepted_consumer_nonce = nonce;
        for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
          metadata.texture_handles[index] = reinterpret_cast<std::uint64_t>(generation_->texture_handles[index].get());
        }
        metadata.ready_fence_handle = reinterpret_cast<std::uint64_t>(generation_->fence_handle.get());
        publish(metadata, true, runtime);
        char message[256];
        std::snprintf(message, sizeof(message), "Sunshine SBS: generation %llu, %ux%u full SBS, DXGI %u, D3D%u, %s (source color %u)", static_cast<unsigned long long>(metadata.generation), metadata.packed_width, metadata.packed_height, metadata.dxgi_format, backend == api::device_api::d3d11 ? 11u : 12u, metadata.color_transfer == wire::transfer::scrgb ? "scRGB" : "sRGB", static_cast<unsigned>(source.color.input));
        log(reshade::log::level::info, message);
      }
      if (generation_->signal_failed) {
        return;
      }
      const auto completed = generation_->completed();
      if (completed == UINT64_MAX) {
        deactivate(runtime);
        return;
      }
      const auto index = acquire_slot(completed);
      if (index != wire::slot_count) {
        auto &slot = shared_->slots[index];
        // A nonce replacement races only on metadata. Do not issue new old-generation work.
        if (read_nonce(*shared_) != generation_->nonce) {
          store_state(slot, generation_->id, wire::slot_state::free);
          return;
        }
        const auto sequence = generation_->last_submitted + 1;
        LARGE_INTEGER timestamp {};
        QueryPerformanceCounter(&timestamp);
        // Mark ownership before recording any work, so reloads and callback failures keep
        // source, ring and overlay resources alive until that work can be fenced.
        generation_->pending_slot = index;
        generation_->pending_qpc = static_cast<std::uint64_t>(timestamp.QuadPart);
        generation_->pending_fg_output = game && proof.frame.fg_active;
        generation_->pending_depth_ready = proof.frame.depth_ready;
        generation_->pending_reused_depth = proof.frame.reused_depth;
        if (!generation_->submit(commands, source.resource, index, sequence)) {
          deactivate(runtime);
          return;
        }
        if (overlay_open(runtime)) {
          auto &overlay = generation_->overlays[index];
          if (!overlay) {
            overlay = std::make_unique<sunshine::overlay::compositor_t>();
          }
          if (!overlay->prepare(runtime, native_rtv, source.resource, generation_->texture(index), source.width, source.height, source.color.input, source.color.output == wire::transfer::scrgb ? api::color_space::scrgb : api::color_space::srgb)) {
            deactivate(runtime);
            retry_at_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            log(reshade::log::level::warning, "Sunshine SBS: cannot prepare stereo overlay; showing the captured desktop");
            return;
          }
          generation_->pending_overlay = true;
        }
        next_slot_ = (index + 1) % wire::slot_count;
        return;
      }
    }

  private:
    std::uint32_t acquire_slot(std::uint64_t completed) {
      // A recorded copy must be fenced before admitting the next. Submitted
      // work is bounded by the existing ring, not a global one-copy gate.
      if (!generation_ || generation_->signal_failed || completed == UINT64_MAX ||
          generation_->pending_slot != wire::slot_count) return wire::slot_count;
      for (std::uint32_t offset = 0; offset < wire::slot_count; ++offset) {
        const auto index = (next_slot_ + offset) % wire::slot_count;
        auto &slot = shared_->slots[index];
        // A consumer may discard a frame without reading it. Even a free slot
        // must retain its source/overlay until that slot's GPU work completes.
        if (completed < slot.sequence) continue;
        if (exchange_state(slot, generation_->id, wire::slot_state::writing, wire::slot_state::free) ||
            exchange_state(slot, generation_->id, wire::slot_state::writing, wire::slot_state::ready)) return index;
      }
      return wire::slot_count;
    }
#ifdef SUNSHINE_SBS_TEST
    friend struct publisher_tests;
#endif
    static void apply_native_depth(api::effect_runtime *runtime, runtime_t &proof, const sunshine_depth::frame_depth &depth) {
      const bool rectangle = depth.width && depth.height && depth.active_width && depth.active_height &&
        depth.x <= depth.width && depth.y <= depth.height && depth.active_width <= depth.width - depth.x &&
        depth.active_height <= depth.height - depth.y;
      const bool ready = depth.ready && rectangle && depth.shader_resource.handle;
      const bool calibrated = ready && depth.calibrated;
      const float rect[] {
        rectangle ? float(depth.active_width) / depth.width : 1.f,
        rectangle ? float(depth.active_height) / depth.height : 1.f,
        rectangle ? float(depth.x) / depth.width : 0.f,
        rectangle ? float(depth.y) / depth.height : 0.f};
      const float anchor = calibrated ? depth.raw_anchor : 0.f;
      const float gain = calibrated ? depth.raw_gain : 0.f;
      runtime->set_uniform_value_float(proof.raw_anchor, &anchor, 1);
      runtime->set_uniform_value_float(proof.raw_gain, &gain, 1);
      runtime->set_uniform_value_float(proof.depth_rect, rect, 4);
      runtime->set_uniform_value_bool(proof.calibrated, calibrated);
      sunshine_streamline::provider::set_depth_ready(runtime, ready);
      proof.native_depth_prepared = true;
    }

    bool source_color_matches(api::effect_runtime *runtime, const runtime_t &proof, api::color_space current) {
      if (proof.color.input == api::color_space::unknown || proof.color.input != current) {
        deactivate(runtime);
        return false;
      }
      return true;
    }

    bool overlay_open(api::effect_runtime *runtime) const {
      const auto found = overlays_.find(runtime);
      return found != overlays_.end() && found->second;
    }

    void retire_destroyed_generation() {
      if (retired_ && retired_->finished()) {
        retired_.reset();
      }
      if (!generation_ || !generation_->owner_destroyed) {
        return;
      }
      if (generation_->finished()) {
        generation_.reset();
        return;
      }
      if (!retired_) {
        retired_ = std::move(generation_);
        log(reshade::log::level::warning, "Sunshine SBS: retaining a destroyed runtime's pending generation; successor runtime may export");
      } else if (!retirement_limit_logged_) {
        retirement_limit_logged_ = true;
        log(reshade::log::level::error, "Sunshine SBS: two runtime generations have unconfirmed GPU work; restart the game to resume export");
      }
    }

    // One-shot invalidation also covers identity metadata published before a host connects.
    void deactivate(api::effect_runtime *runtime) {
      if (!runtime) {
        return;
      }
      if (published_runtime_ == runtime) {
        publish(identity_);
        log(reshade::log::level::info, "Sunshine SBS: export inactive; waiting for a valid focused technique");
      }
      if (generation_ && generation_->runtime == runtime) {
        for (auto &overlay : generation_->overlays) if (overlay) overlay->cancel();
        generation_->runtime = nullptr;
      }
    }

    void log_fg_output(api::effect_runtime *runtime, runtime_t &proof, std::uint64_t now) {
      auto &counts = proof.fg_output;
      if (!proof.frame.fg_active || now < counts.next_log) return;
      char text[384]{};
      std::snprintf(text, sizeof(text), "Sunshine SBS FG output: published_fresh_depth=%llu published_reused_depth=%llu published_depth_missing=%llu runtime=0x%llx generation=%llu; new color publications, reused depth is bounded and does not advance calibration",
        static_cast<unsigned long long>(counts.published_fresh_depth), static_cast<unsigned long long>(counts.published_reused_depth),
        static_cast<unsigned long long>(counts.published_depth_missing),
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(runtime)),
        static_cast<unsigned long long>(generation_ ? generation_->id : 0));
      log(reshade::log::level::info, text);
      counts.next_log = now + 5000;
    }

    void complete_slot(std::uint32_t index, std::uint64_t qpc) {
      const auto now = GetTickCount64();
      auto &slot = shared_->slots[index];
      slot.sequence = generation_->last_submitted;
      slot.qpc = qpc;
      store_state(slot, generation_->id, wire::slot_state::ready);
      if (generation_->pending_fg_output) {
        const auto found = runtimes_.find(generation_->runtime);
        if (found != runtimes_.end()) {
          auto &proof = found->second;
          if (generation_->pending_depth_ready) {
            if (generation_->pending_reused_depth) ++proof.fg_output.published_reused_depth;
            else ++proof.fg_output.published_fresh_depth;
          }
          else ++proof.fg_output.published_depth_missing;
          log_fg_output(generation_->runtime, proof, now);
        }
      }
      generation_->pending_fg_output = false;
    }

    void publish(const wire::metadata_t &metadata, bool reset_slots = false, api::effect_runtime *owner = nullptr) {
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared_->metadata_sequence));
      published_metadata_ = metadata;
      published_runtime_ = owner;
      MemoryBarrier();
      shared_->metadata = metadata;
      if (reset_slots) {
        for (auto &slot : shared_->slots) {
          slot.sequence = 0;
          slot.qpc = 0;
          store_state(slot, metadata.generation, wire::slot_state::free);
        }
      }
      MemoryBarrier();
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared_->metadata_sequence));
    }

    static void set_source(wire::metadata_t &metadata, const source_t &source) {
      metadata.window = reinterpret_cast<std::uint64_t>(source.window);
      metadata.adapter_luid = source.adapter;
      metadata.source_width = source.width;
      metadata.source_height = source.height;
      metadata.packed_width = source.width * 2;
      metadata.packed_height = source.height;
      metadata.dxgi_format = static_cast<std::uint32_t>(source.format);
      metadata.color_transfer = source.color.output;
    }

    static void discover(api::effect_runtime *runtime, runtime_t &proof, const char *effect = "SuperDepth3D.fx") {
      proof.texture = {};
      proof.effect_name = effect;
      proof.proof_checked = true;
      unsigned exports = 0;
      runtime->enumerate_texture_variables(effect, [&](api::effect_runtime *owner, api::effect_texture_variable texture) {
        int enabled = 0;
        unsigned width = 0, height = 0, input_color = 0;
        char layout[32] {}, color[32] {};
        export_color_t export_color;
        if (!owner->get_annotation_int_from_texture_variable(texture, "sunshine_sbs_export", &enabled, 1) || enabled != 1 || !owner->get_annotation_string_from_texture_variable(texture, "sunshine_sbs_layout", layout) || std::strcmp(layout, "sbs_lr") != 0 || !owner->get_annotation_string_from_texture_variable(texture, "sunshine_sbs_color_space", color) || !owner->get_annotation_uint_from_texture_variable(texture, "sunshine_sbs_source_color_space", &input_color, 1) || !decode_export_color(color, input_color, export_color) || !owner->get_annotation_uint_from_texture_variable(texture, "sunshine_sbs_source_width", &width, 1) || !owner->get_annotation_uint_from_texture_variable(texture, "sunshine_sbs_source_height", &height, 1)) {
          return;
        }
        if (!width || width > wire::max_source_width || width % 2 || !height || height > wire::max_source_height || height % 2) {
          return;
        }
        proof.texture = texture;
        ++exports;
        proof.width = width;
        proof.height = height;
        proof.color = export_color;
      });
      if (exports != 1) proof.texture = {};
      if (!proof.texture.handle) {
        log(reshade::log::level::warning, "Sunshine SBS: effect needs exactly one compatible stereo export texture; reinstall the matching Sunshine shader");
      }
    }

    static bool describe(api::effect_runtime *runtime, const runtime_t &proof, source_t &source) {
      std::uint32_t width = 0, height = 0;
      runtime->get_screenshot_width_and_height(&width, &height);
      if (width != proof.width || height != proof.height) {
        return false;
      }
      if (proof.addon_native) source.resource = proof.native_output;
      else {
        api::resource_view view {};
        runtime->get_texture_binding(proof.texture, &view, nullptr);
        if (!view.handle) return false;
        source.resource = runtime->get_device()->get_resource_from_view(view);
      }
      if (!source.resource.handle) {
        return false;
      }
      source.width = width;
      source.height = height;
      source.color = proof.color;
      source.window = static_cast<HWND>(runtime->get_hwnd());
      // ReShade's aggregate-return getters are not MSVC/MinGW ABI compatible. The build
      // adapts only get_resource_from_view; native COM supplies the other descriptions,
      // including MinGW's explicit aggregate-return wrapper for D3D12 GetDesc.
      if (runtime->get_device()->get_api() == api::device_api::d3d11) {
        D3D11_TEXTURE2D_DESC desc {};
        reinterpret_cast<ID3D11Texture2D *>(source.resource.handle)->GetDesc(&desc);
        if (desc.Width != width * 2 || desc.Height != height || desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1) {
          return false;
        }
        source.format = typed_format(desc.Format);
        com_ptr<IDXGIDevice> device;
        com_ptr<IDXGIAdapter> adapter;
        DXGI_ADAPTER_DESC adapter_desc {};
        auto *native = reinterpret_cast<ID3D11Device *>(runtime->get_device()->get_native());
        if (FAILED(native->QueryInterface(IID_PPV_ARGS(device.put()))) || FAILED(device->GetAdapter(adapter.put())) || FAILED(adapter->GetDesc(&adapter_desc))) {
          return false;
        }
        std::memcpy(&source.adapter, &adapter_desc.AdapterLuid, sizeof(source.adapter));
      } else {
        const auto desc = reinterpret_cast<ID3D12Resource *>(source.resource.handle)->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width != width * 2 || desc.Height != height || desc.MipLevels != 1 || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1) {
          return false;
        }
        source.format = typed_format(desc.Format);
        auto *native = reinterpret_cast<ID3D12Device *>(runtime->get_device()->get_native());
        const LUID luid = native->GetAdapterLuid();
        std::memcpy(&source.adapter, &luid, sizeof(source.adapter));
      }
      return wire::supported_format(static_cast<std::uint32_t>(source.format), source.color.output);
    }

    handle_t mapping_;
    wire::shared_state_t *shared_ = nullptr;
    wire::metadata_t identity_;
    wire::metadata_t published_metadata_;
    api::effect_runtime *published_runtime_ = nullptr;
    std::mutex mutex_;
    std::unordered_map<api::effect_runtime *, runtime_t> runtimes_;
    std::unordered_map<std::uint64_t, api::color_space> colors_;
    std::unordered_map<api::effect_runtime *, bool> overlays_;
    std::unique_ptr<generation_t> generation_;
    // A missing finish_present after runtime destruction must not wedge the next runtime.
    // Bound this fallback to one retired ring plus the active ring; never allocate indefinitely.
    std::unique_ptr<generation_t> retired_;
    bool retirement_limit_logged_ = false;
    std::uint64_t next_generation_ = 0;
    std::uint32_t next_slot_ = 0;
    std::chrono::steady_clock::time_point retry_at_ {};
  };

  addon_session_t::addon_session_t() = default;
  addon_session_t::~addon_session_t() = default;
  addon_session_t &addon_session() {
    // Native hooks pin this module and ReShade can notify unload after our CRT
    // detach. One explicit session owner, not a second CRT destructor, releases
    // publisher/UI resources on normal unload. The empty session remains valid
    // for reinitialization; final process reclamation belongs to Windows.
    static auto *const session = new addon_session_t;
    return *session;
  }
  auto &publisher = addon_session().publisher;

  void teardown_addon(HMODULE addon, HMODULE reshade) {
    if (sunshine_addon_lifetime::stopping()) return;
    // Claim teardown before any COM Release or foreign callback can reenter.
    // Observer metadata releases its source leases while capture/UI owners are
    // still alive; capture keeps any leases required by outstanding GPU work.
    addon_session().end([&] {
      sunshine_streamline::shutdown();
      sunshine_depth::shutdown();
      reshade::unregister_addon(addon, reshade);
    });
  }

  void on_technique(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *commands, api::resource_view rtv, api::resource_view) {
    try {
      if (publisher) {
        publisher->frame(runtime, technique, commands, rtv);
      }
    } catch (...) {
      if (publisher) {
        publisher->invalidate(runtime, false);
      }
      log(reshade::log::level::error, "Sunshine SBS: exporter callback failed; skipping frame");
    }
  }

  void on_reload(api::effect_runtime *runtime) {
    if (publisher) {
      publisher->invalidate(runtime, false);
    }
  }

  void on_init_runtime(api::effect_runtime *runtime) {
    if (publisher) publisher->track_runtime(runtime);
  }

  void on_begin_effects(api::effect_runtime *runtime, api::command_list *commands, api::resource_view rtv, api::resource_view) {
    try {
      if (publisher && !publisher->native_enabled(runtime)) publisher->prepare_native_depth(runtime, commands, rtv);
    } catch (...) {
      if (publisher) publisher->invalidate(runtime, false);
      log(reshade::log::level::error, "Sunshine SBS: depth calibration update failed; skipping source");
    }
  }

  void on_destroy(api::effect_runtime *runtime) {
    if (publisher) {
      publisher->invalidate(runtime, true, runtime->get_native());
    }
  }

  void on_present(api::effect_runtime *runtime) {
    try {
      if (publisher) {
        publisher->present(runtime);
      }
    } catch (...) {
      if (publisher) {
        publisher->invalidate(runtime, false);
      }
      log(reshade::log::level::error, "Sunshine SBS: overlay publication failed; skipping frame");
    }
  }

  void on_draw_overlay(api::effect_runtime *runtime) {
    try {
      if (publisher) {
        publisher->draw_overlay(runtime);
      }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      sunshine_sbs_test_overlay::draw(runtime);
#endif
    } catch (...) {
      if (publisher) {
        publisher->invalidate(runtime, false);
      }
      log(reshade::log::level::error, "Sunshine SBS: overlay capture callback failed; skipping frame");
    }
  }

  void on_begin_present(api::command_queue *, api::swapchain *swapchain, const api::rect *, const api::rect *, std::uint32_t, const api::rect *) {
    try {
      if (publisher) {
        publisher->begin_present(swapchain->get_native(), swapchain->get_color_space());
        publisher->render_present(swapchain);
      }
    } catch (...) {
      log(reshade::log::level::error, "Sunshine SBS: present metadata update failed");
    }
  }

  void on_finish_present(api::command_queue *queue, api::swapchain *swapchain) {
    if (publisher) {
      publisher->finish_present(queue->get_native(), swapchain->get_native());
    }
  }

  bool on_overlay(api::effect_runtime *runtime, bool open, api::input_source) {
    try {
      if (publisher) {
        publisher->set_overlay_open(runtime, open);
      }
    } catch (...) {
      log(reshade::log::level::error, "Sunshine SBS: overlay state update failed");
    }
    // Observe this advisory transition without cancelling ReShade's UI operation.
    return false;
  }
}  // namespace

namespace sunshine_game3d {
  automatic_status query_automatic(reshade::api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    return found == automatic_ui.end() ? automatic_status{} : found->second.status;
  }

  bool recalibrate_automatic(reshade::api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    if (found == automatic_ui.end() || !found->second.status.can_recalibrate || found->second.recalibrate)
      return false;
    found->second.recalibrate = true;
    found->second.status.phase = automatic_phase::calibrating;
    found->second.status.can_recalibrate = false;
    found->second.status.scale.active = false;
    return true;
  }
}

extern "C" {
  BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    // No locks, allocation, API unregistration or graphics calls under the
    // loader lock. Ordinary AddonUninit still performs its full cleanup.
    if (reason == DLL_PROCESS_DETACH) sunshine_addon_lifetime::begin_detach();
    return TRUE;
  }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  __declspec(dllexport) const char *NAME = "Sunshine SBS TEST ONLY";
  __declspec(dllexport) const char *DESCRIPTION = "TEST ONLY: Controlled-foreground runtime fixture. Never install this add-on into a game.";

  __declspec(dllexport) void SunshineSbsTestSetForeground(HWND window) {
    test_foreground_window.store(window, std::memory_order_release);
  }
#else
  __declspec(dllexport) const char *NAME = "Sunshine 3D";
  __declspec(dllexport) const char *DESCRIPTION = "Automatically selects and calibrates game depth for Sunshine stereo. Includes SDR/HDR sharing and stereo overlay controls.";
#endif

  __declspec(dllexport) bool AddonInit(HMODULE addon, HMODULE reshade) {
    if (sunshine_addon_lifetime::stopping()) return false;
    if (!addon_session().begin()) return false;
    if (!reshade::register_addon(addon, reshade)) {
      addon_session().lifecycle.store(addon_session_t::phase::inactive, std::memory_order_release);
      return false;
    }
    try {
      publisher = std::make_unique<publisher_t>();
      if (!publisher->init()) {
        teardown_addon(addon, reshade);
        return false;
      }
      // Generic Depth is registered before external add-ons. On a first manual
      // installation the module can request a restart while export stays available.
      sunshine_streamline::initialize(addon);
      sunshine_depth::initialize(addon);
      reshade::register_event<reshade::addon_event::init_effect_runtime>(sunshine_addon_lifetime::guarded<on_init_runtime>);
      reshade::register_event<reshade::addon_event::reshade_begin_effects>(sunshine_addon_lifetime::guarded<on_begin_effects>);
      reshade::register_event<reshade::addon_event::reshade_render_technique>(sunshine_addon_lifetime::guarded<on_technique>);
      reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(sunshine_addon_lifetime::guarded<on_reload>);
      reshade::register_event<reshade::addon_event::destroy_effect_runtime>(sunshine_addon_lifetime::guarded<on_destroy>);
      reshade::register_event<reshade::addon_event::reshade_present>(sunshine_addon_lifetime::guarded<on_present>);
      reshade::register_event<reshade::addon_event::present>(sunshine_addon_lifetime::guarded<on_begin_present>);
      reshade::register_event<reshade::addon_event::finish_present>(sunshine_addon_lifetime::guarded<on_finish_present>);
      reshade::register_event<reshade::addon_event::reshade_open_overlay>(sunshine_addon_lifetime::guarded<on_overlay>);
      reshade::register_event<reshade::addon_event::reshade_overlay>(sunshine_addon_lifetime::guarded<on_draw_overlay>);
      return true;
    } catch (...) {
      teardown_addon(addon, reshade);
      return false;
    }
  }

  __declspec(dllexport) void AddonUninit(HMODULE addon, HMODULE reshade) {
    // ReShade can notify us from its own DLL detach after our CRT has already
    // destroyed publisher/UI state. Do not reset those destroyed owners twice.
    if (sunshine_addon_lifetime::stopping()) return;
    teardown_addon(addon, reshade);
  }
}
