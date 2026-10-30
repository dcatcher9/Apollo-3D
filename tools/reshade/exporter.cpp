// SPDX-License-Identifier: GPL-3.0-only
#include "addon_lifetime.h"
// Original Sunshine exporter. ReShade's SDK headers are BSD-3-Clause OR MIT.
#include "overlay_compositor.h"
#include "depth_addon.h"
#include "depth_output.h"
#include "depth_jitter.h"
#include "projection_depth_controller.h"
#include "raw_scene_pool.h"
#include "provided_raw_scene.h"
#include "automatic_scene_transition.h"
#include "game3d_controls.h"
#include "game3d_display_scale.h"
#include "game3d_renderer.h"
#include "game3d_depth_input.h"
#include "hook_host.h"
#include "game3d_ui_input_provider.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_selection.h"
#include "game3d_debug_dump.h"
#include "game3d_test_render.h"
#include "diagnostic_log_gate.h"
#include "freshness.h"
#include "streamline_camera_probe.h"
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include "src/reshade_bridge_protocol.h"
#include "src/reshade_bridge_window.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <imgui.h>
#include <memory>
#include <mutex>
#include <reshade.hpp>
#include "async_log.h"
#include "depth_content_sampler.h"
#include <string>
#include <string_view>
#include <unordered_map>
#include <optional>
#include <utility>
#include <vector>
#include <windows.h>
#include "game3d_slow_step.h"
#include "game3d_ui_layer.h"
#include "game3d_vram.h"
#include "game3d_diagnostics.h"

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  #include "game3d_test_depth_events.h"
  #include "game3d_test_teardown.h"
  #include "test_overlay_patch.h"

  #include <atomic>
#endif

static_assert(RESHADE_API_VERSION == 20, "Build against the pinned ReShade 6.8.0 SDK");
static_assert(sizeof(void *) == 8, "Only 64-bit games are supported");

namespace {
  namespace api = reshade::api;
  namespace wire = reshade_bridge;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace depth_events = sunshine_game3d::test::depth_events;
#endif
  std::atomic<std::uint64_t> next_raw_basis {1};
  void log(reshade::log::level level, const char *message);

  struct automatic_ui_entry {
    sunshine_game3d::automatic_status status {};
    sunshine_game3d::source_alpha_ui_decision source_alpha;
    sunshine_game3d::source_alpha_ui_decision logged_source_alpha;
    sunshine_diagnostics::log_gate source_alpha_log;
    // The game session whose UI counters this runtime's lines carry, known
    // once a render acquired it; counters_logged once a line carried them, so
    // the runtime's reset or destruction ends the log with the session totals.
    sunshine_game3d::alpha_auto_policy *ui_session = nullptr;
    bool counters_logged = false;
#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
    // Explicit resets exist only for controlled regression fixtures.
    bool recalibrate = false;
#endif
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

  // Part declarations: UI status publication bodies live in exporter/ui_status.inl.
  sunshine_game3d::alpha_auto_policy *clear_automatic_ui(api::effect_runtime *runtime);
  void log_ui_counters(api::effect_runtime *runtime, sunshine_game3d::alpha_auto_policy &session);
  void publish_automatic_ui(api::effect_runtime *runtime, sunshine_game3d::automatic_status status);
  void publish_source_alpha_ui(api::effect_runtime *runtime, sunshine_game3d::source_alpha_ui_decision value,
      bool game3d_enabled = true, sunshine_game3d::alpha_auto_policy *ui_session = nullptr);
#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
  bool take_recalibration(api::effect_runtime *runtime);
#endif

#if defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON) || defined(SUNSHINE_SBS_TEST)
  // Only the distinct test DLL can substitute this observation. Null fails closed
  // until its fixture explicitly selects a real window owned by this process. The
  // unit tests substitute it only while set, and otherwise observe Windows.
  std::atomic<HWND> test_foreground_window {nullptr};
#endif

  HWND observed_foreground_window() {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    return test_foreground_window.load(std::memory_order_acquire);
#elif defined(SUNSHINE_SBS_TEST)
    const HWND window = test_foreground_window.load(std::memory_order_acquire);
    return window ? window : GetForegroundWindow();
#else
    return GetForegroundWindow();
#endif
  }

  // Whether a swapchain window that is not the foreground window is a child of it that owns its
  // client area (reshade_bridge::window_owner): compared with the root's own client area; the
  // host compares it with the captured display and requires the root to cover that exactly.
  reshade_bridge::window_owner::verdict child_window_verdict(HWND window, HWND foreground) {
    namespace owner = reshade_bridge::window_owner;
    if (!window) return owner::verdict::not_foreground;
    // No foreground window (the secure desktop, or briefly while another window activates): a
    // top-level swapchain window is still not a child, so the caller's top-level rule applies.
    if (!foreground) return GetAncestor(window, GA_ROOT) == window ? owner::verdict::not_child : owner::verdict::not_foreground;
    const auto chain = owner::query(window);
    const auto root = chain.count ? chain.links[chain.count - 1].client : owner::rect_t {};
    return owner::classify(chain, reinterpret_cast<std::uintptr_t>(foreground), GetCurrentProcessId(), root);
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
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      sunshine_game3d::test::teardown::handle_destroyed(this, value_ != nullptr);
#endif
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
    sunshine_log::message(level, message);
#endif
  }

  void log_hr(const char *operation, HRESULT hr) {
    char message[256];
    std::snprintf(message, sizeof(message), "Sunshine SBS: %s failed (0x%08lx)", operation, static_cast<unsigned long>(hr));
    log(reshade::log::level::error, message);
  }

#include "exporter/export_ring.inl"

  // A child swapchain window's verdicts (report_window_owner) that the player
  // or the game's own window changes explain, as a top-level window's
  // not_foreground pause is routine: the owner, a root out of focus, a hidden
  // child and one being resized. They are logged at info; a sibling overlap,
  // another process's window in the chain, an incomplete query, two
  // qualifying swapchains and any new verdict are warnings.
  constexpr std::array<std::string_view, 4> routine_window_verdicts{"owner", "not_foreground", "not_visible", "partial"};
  constexpr bool routine_window_verdict(std::string_view verdict) {
    for (const auto routine : routine_window_verdicts)
      if (routine == verdict) return true;
    return false;
  }

#include "exporter/scene_placement.inl"

  struct runtime_t {
    bool addon_native = false;
    sunshine_game3d::source_alpha_ui_policy source_alpha_policy;
    float frame_strength = sunshine_game3d::default_strength;
    std::uint64_t swapchain = 0;
    std::uint64_t presentation_ordinal = 0;
    std::unique_ptr<sunshine_game3d::renderer> renderer;
    api::resource native_output{};
    float native_ui_parallax_uv = 0.f;
    api::resource_view borrowed_depth{};
    // Populated only while a one-shot diagnostic is armed, and cleared before
    // the native depth lease ends. GPU handles here are never historical state.
    sunshine_depth::frame_depth diagnostic_depth;
    std::string diagnostic_ui_source;
    std::string diagnostic_ui_capture_attempt;
    bool diagnostic_armed = false;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // SunshineGame3DTestLastRender: this Present's prepared depth and the
    // last native render's inputs.
    sunshine_depth::frame_depth test_depth;
    sunshine_game3d::test::last_render test_render;
#endif
    api::effect_texture_variable texture {};
    std::string effect_name;
    api::effect_technique game_technique {}, native_technique {};
    api::effect_uniform_variable depth_ready {}, calibrated {}, raw_anchor {}, raw_gain {}, depth_rect {}, direction {};
    bool native_uniforms_checked = false;
    bool native_depth_prepared = false;
    api::effect_uniform_variable camera_ready {}, camera_basis {}, camera_projection {}, camera_scale {}, camera_zero {}, camera_blend {}, camera_rect {}, camera_raw_range {};
    api::effect_uniform_variable depth_jitter {}, effect_strength {};
    bool raw_supported = false;
    std::uint64_t raw_basis_epoch = 0;
    sunshine_raw_scene::pool raw_policy;
    // One retained reference per API-provided logical encoding, so FG switches
    // do not repeat the startup window when a source returns.
    sunshine_provided_raw::retained_policy provided_raw_policy;
    sunshine_projection_depth::controller projection_policy;
    sunshine_projection_depth::domain projection_domain;
    sunshine_diagnostics::log_gate projection_log;
    // API captures without their own camera that kept the projection scene.
    std::uint64_t camera_missing_holds = 0;
    bool projection_ready = false;
    sunshine_raw_scene::reentry_transition raw_reentry;
    // Whether raw_reentry heard from this runtime's current Present, and that
    // Present's tick (note_shown_present): a Present it never heard from was
    // shown without Game 3D stereo.
    std::uint64_t reentry_present_ms = 0;
    bool reentry_heard = true;
    sunshine_raw_scene::status raw_last_status = sunshine_raw_scene::status::uninitialized;
    sunshine_diagnostics::log_gate raw_log;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    export_color_t color;
    // The native render's PQ request (exports_pq for the consumer it read
    // before rendering): frame() exports that render only while the
    // consumer it then reads answers the same.
    bool pq_requested = false;
    // A runtime's device never changes adapter; resolve its LUID once.
    // Bounded timing of the add-on's own work inside the game's Present.
    // start: GetTickCount64 when the window began (its first Present, or the
    // previous timing line).
    struct {
      std::uint64_t presents = 0, next_log = 0, start = 0;
      double cpu_sum_ms = 0, cpu_max_ms = 0;
      std::array<double, 5> worst{}; // Setup, depth, UI, render, export of the slowest present.
    } timing;
    std::uint64_t adapter_luid = 0;
    bool adapter_known = false;
    // The adapter whose video memory this process's samples read
    // (game3d_vram.h), opened once from the device on an enabled Present.
    com_ptr<IDXGIAdapter3> vram_adapter;
    bool vram_adapter_checked = false;
    // Gaps between this runtime's Presents (present_gap_ms): the previous
    // Present's QueryPerformanceCounter, and this 10 s window's logged and
    // suppressed gap lines.
    std::int64_t last_present_qpc = 0;
    std::uint64_t gap_window_start = 0;
    unsigned gap_lines = 0, gap_suppressed = 0;
    bool proof_checked = false;
    bool rendered_since_present = false;
    frame_decision_t frame;
    // Every Game 3D publication, with or without frame generation.
    struct {
      std::uint64_t published = 0, fg = 0;
      std::uint64_t published_fresh_depth = 0, published_reused_depth = 0;
      std::uint64_t published_depth_missing = 0, next_log = 0;
      // published_depth_missing by depth_gap reason.
      std::uint64_t missing_unavailable = 0, missing_reuse_after_gap = 0, missing_reuse_other_source = 0;
      // Publications with depth that showed the colour frame or were fading in.
      std::uint64_t scene_flat = 0, scene_fading = 0;
      // Export slots (acquire_slot): Presents a consumer was attached to that
      // found no reusable slot, and packs written over a ready slot the
      // consumer never claimed (it took a newer one, or none yet).
      std::uint64_t dropped = 0, overwritten_unconsumed = 0;
      void add(const output_sample &sample) {
        ++published;
        if (sample.fg) ++fg;
        if (sample.scene_flat) ++scene_flat;
        if (sample.scene_fading) ++scene_fading;
        if (sample.depth_ready) {
          ++(sample.reused_depth ? published_reused_depth : published_fresh_depth);
          return;
        }
        ++published_depth_missing;
        ++(sample.gap == depth_gap::reuse_after_gap ? missing_reuse_after_gap :
          sample.gap == depth_gap::reuse_other_source ? missing_reuse_other_source : missing_unavailable);
      }
    } output;
  };

  // The depth view's content identity for the renderer's conditioning memo
  // (render_frame_input::depth_identity): nonzero only for an API capture,
  // whose borrowed view holds that capture's immutable pixels for its epoch,
  // source, sequence and viewport (a reused generated-frame presentation
  // repeats them exactly). Generic depth and missing depth give 0.
  std::uint64_t depth_identity(const runtime_t &proof) {
    return sunshine_game3d::depth_output::depth_identity(proof.frame.scene_source,
      proof.borrowed_depth.handle != 0, proof.frame.depth_ready);
  }

#include "exporter/export_ring_class.inl"

  class publisher_t {
  public:
    ~publisher_t() {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      // SunshineGame3DTestTeardownOrder: this body's steps, then (through
      // their own destructors) the members that record.
      namespace teardown = sunshine_game3d::test::teardown;
      teardown::note(teardown::step::publisher_begin, sunshine_addon_lifetime::stopping() ? 1 : 0);
#endif
      if (!sunshine_addon_lifetime::stopping()) {
        {
          // The observer may keep this module resident across add-on unload.
          // A new publisher must not inherit status/actions at reused addresses.
          std::lock_guard<std::mutex> lock(automatic_ui_mutex);
          automatic_ui.clear();
        }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::ui_cleared);
#endif
        ring_.unload();
      }
      ring_.unmap();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      ring_.note_rings();
      teardown::note_mapping(&mapping_);
      teardown::note(teardown::step::publisher_end);
#endif
    }

    bool init() {
      FILETIME created {}, exited {}, kernel {}, user {};
      if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        return false;
      }
      ring_.identity_.producer_pid = GetCurrentProcessId();
      ring_.identity_.producer_creation_time = (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) | created.dwLowDateTime;
      const auto name = std::wstring(wire::mapping_prefix) + std::to_wstring(ring_.identity_.producer_pid);
      *mapping_.put() = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(wire::shared_state_t), name.c_str());
      if (!mapping_.get() || GetLastError() == ERROR_ALREADY_EXISTS) {
        log(reshade::log::level::error, "Sunshine SBS: cannot create a unique per-process mapping");
        return false;
      }
      ring_.shared_ = static_cast<wire::shared_state_t *>(MapViewOfFile(mapping_.get(), FILE_MAP_ALL_ACCESS, 0, 0, sizeof(wire::shared_state_t)));
      if (!ring_.shared_) {
        return false;
      }
      // New paging-file mappings are zeroed; publish the versioned layout before callbacks start.
      new (ring_.shared_) wire::shared_state_t {};
      ring_.publish(ring_.identity_);
      if (!debug_dump_.initialize(ring_.identity_.producer_pid, ring_.identity_.producer_creation_time))
        log(reshade::log::level::warning, "Sunshine Game 3D: optional diagnostic mailbox unavailable");
      // The protocol is named so a pairing with an older host, which ignores this mapping
      // without a log line, can be told apart from a host that is not streaming.
      char ready[192];
      std::snprintf(ready, sizeof(ready), "Sunshine SBS: exporter ready (export protocol %u, %u export slots); waiting for native Game 3D or a reference stereo effect and consumer", wire::version, wire::slot_count);
      log(reshade::log::level::info, ready);
      return true;
    }

    // Whether `window`, the swapchain window of `runtime`, is the window the user sees in the
    // foreground (owns_foreground_locked).
    bool owns_foreground(api::effect_runtime *runtime, HWND window) {
      std::lock_guard<std::mutex> lock(mutex_);
      return owns_foreground_locked(runtime, window);
    }

    bool owns_foreground(api::swapchain *swapchain) {
      std::lock_guard<std::mutex> lock(mutex_);
      api::effect_runtime *runtime = nullptr;
      for (const auto &entry : runtimes_) if (entry.second.swapchain == swapchain->get_native()) { runtime = entry.first; break; }
      return owns_foreground_locked(runtime, static_cast<HWND>(swapchain->get_hwnd()));
    }

    void track_runtime(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &proof = runtimes_[runtime];
      proof.swapchain = runtime->get_native();
      // A runtime reinitialised after a reset (a resize) takes back its parked
      // renderer; any other runtime at that address starts fresh.
      const auto parked = std::find_if(parked_.begin(), parked_.end(), [&](const parked_t &p) { return p.runtime == runtime; });
      if (parked == parked_.end()) return;
      if (!proof.renderer && parked->native_swapchain == proof.swapchain && parked->device == runtime->get_device()) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        sunshine_game3d::test::teardown::note_parked(parked->renderer.get(), false);
#endif
        proof.renderer = std::move(parked->renderer);
      } else parked->renderer->reset_after_runtime_drain();
      parked_.erase(parked);
    }

    // Requires the lock: releases a renderer parked for runtime.
    void drop_parked(api::effect_runtime *runtime) {
      for (auto it = parked_.begin(); it != parked_.end();) {
        if (it->runtime == runtime) {
          it->renderer->reset_after_runtime_drain();
          it = parked_.erase(it);
        } else ++it;
      }
    }

    bool native_enabled(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = runtimes_.find(runtime);
      return found != runtimes_.end() && found->second.addon_native;
    }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    bool test_last_render(api::effect_runtime *runtime, sunshine_game3d::test::last_render &out) {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = runtimes_.find(runtime);
      if (found == runtimes_.end()) return false;
      out = found->second.test_render;
      return true;
    }
#endif

    // Present timing, present gaps, the VRAM adapter and the SBS output line are defined in
    // exporter/present_diagnostics.inl.
    void finish_present_timing(api::effect_runtime *runtime, const std::array<LARGE_INTEGER, 5> &marks, unsigned count);

    // A gap between two Presents of one runtime long enough to freeze the
    // picture: the host's first check of a stalled picture comes at 100 ms
    // too (nvenc_base.cpp). A game that freezes the whole GPU presents
    // nothing meanwhile, which the log report matches against the host's
    // upstream stalls. A loading screen below 10 fps logs at most this many
    // lines per 10 s and counts the rest.
    static constexpr double present_gap_ms = 100.0;
    static constexpr unsigned present_gap_lines_per_window = 10;
    struct present_gap_t {
      double ms = 0;
      bool fg = false, log = false;
      unsigned suppressed = 0;
    };
    present_gap_t measure_present_gap(runtime_t &proof, std::int64_t qpc, std::uint64_t now);
    void log_present_gap(api::effect_runtime *runtime, const present_gap_t &gap);
    IDXGIAdapter3 *vram_adapter(runtime_t &proof, api::device *device, IDXGISwapChain *swapchain);
    void close_timing_window(runtime_t &proof, std::uint64_t now, char (&message)[1024]);

    // Scene placement and reentry bodies are in exporter/scene_placement_impl.inl.
    float reentry_update(runtime_t &proof, bool ready, std::uint64_t now,
        sunshine_raw_scene::stereo_basis basis = sunshine_raw_scene::stereo_basis::raw);
    void note_shown_present(runtime_t &proof, std::uint64_t now);

    void render_present(api::swapchain *swapchain) {
      api::effect_runtime *runtime = nullptr;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[owner, proof] : runtimes_) if (proof.swapchain == swapchain->get_native()) { runtime = owner; break; }
      }
      if (!runtime) return;
      const bool diagnostic_owner = debug_dump_.awaiting_capture() &&
        owns_foreground(runtime, static_cast<HWND>(runtime->get_hwnd()));
      const auto settings = sunshine_game3d::query_render_settings(runtime);
      const auto presentation = sunshine_game3d::ui_input::observe();
      sunshine_game3d::source_alpha_ui_decision source_alpha;
      sunshine_depth::set_native_driver(runtime, settings.enabled);
      LARGE_INTEGER present_qpc {};
      QueryPerformanceCounter(&present_qpc);
      present_gap_t gap;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &proof = runtimes_[runtime];
        proof.addon_native = settings.enabled;
        source_alpha = sunshine_game3d::ui_input::resolve(proof.source_alpha_policy, settings, presentation);
        const auto tick = GetTickCount64();
        gap = measure_present_gap(proof, present_qpc.QuadPart, tick);
        note_shown_present(proof, tick);
      }
      struct publish_alpha_decision {
        api::effect_runtime *runtime;
        const sunshine_game3d::source_alpha_ui_decision &value;
        bool game3d_enabled;
        sunshine_game3d::alpha_auto_policy *ui_session = nullptr;
        ~publish_alpha_decision() { publish_source_alpha_ui(runtime, value, game3d_enabled, ui_session); }
      } publish_alpha{runtime, source_alpha, settings.enabled};
      if (!settings.enabled) {
        sunshine_game3d::ui_input::suspend(runtime);
        if (diagnostic_owner) debug_dump_.unavailable("The add-on Game 3D renderer is disabled for this foreground game.");
        sunshine_depth::set_raw_scene_request(runtime, 0, false);
        return;
      }
      // For the overlay's UI protection hints; published with the decision.
      source_alpha.sdr_output = swapchain->get_color_space() == api::color_space::srgb;
      // Everything below runs inside the game's Present, so it is latency.
      struct present_timer {
        publisher_t &self; api::effect_runtime *runtime;
        std::array<LARGE_INTEGER, 5> marks{}; // Start, then the end of setup, depth, UI and render.
        unsigned count = 0;
        void mark() { if (count < marks.size()) QueryPerformanceCounter(&marks[count++]); }
        ~present_timer() { self.finish_present_timing(runtime, marks, count); }
      } timer{*this, runtime};
      timer.mark();
      auto *device = runtime->get_device();
      const auto backend = device->get_api();
      sunshine_game3d::ui_input::configure(runtime, source_alpha, backend);
      auto *queue = runtime->get_command_queue();
      if (!queue || (backend != api::device_api::d3d11 && backend != api::device_api::d3d12)) {
        if (diagnostic_owner) debug_dump_.unavailable("The foreground runtime has no supported D3D11/D3D12 render queue.");
        return;
      }
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
      // The game's video memory (game3d_vram.h), at most every 250 ms and
      // queried outside the lock; before depth preparation, whose readiness
      // line, like a hitch and the gap line below, reports this sample.
      auto &memory = sunshine_game3d::vram::process();
      if (const auto tick = GetTickCount64(); memory.due(tick)) {
        IDXGIAdapter3 *adapter = nullptr;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          adapter = vram_adapter(runtimes_[runtime], device, native_swapchain);
          if (adapter) adapter->AddRef();
        }
        memory.record(query_vram(adapter), tick);
        if (adapter) adapter->Release();
      }
      if (gap.log) log_present_gap(runtime, gap);
      sunshine_game3d::ui_input::request(runtime, backbuffer, source_alpha, settings, swapchain->get_color_space());
      sunshine_game3d::renderer *renderer = nullptr;
      api::resource_view rtv{};
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &proof = runtimes_[runtime];
        if (!proof.renderer) proof.renderer = std::make_unique<sunshine_game3d::renderer>();
        renderer = proof.renderer.get();
        // Shaders compile, and the renderer's GPU objects are built, on the
        // thread pool; the game stays 2D until the renderer is ready.
        if (!renderer->configure(runtime, backbuffer, swapchain->get_color_space(), {}, true)) {
          const bool preparing = renderer->preparing();
          if (diagnostic_owner) debug_dump_.unavailable(preparing ? "The foreground Game 3D renderer is still being prepared." :
            "The foreground Game 3D renderer is unavailable for this swapchain format, extent or transition.");
          ring_.deactivate(runtime, "renderer_unavailable");
          sunshine_game3d::automatic_status unavailable;
          unavailable.phase = preparing ? sunshine_game3d::automatic_phase::renderer_preparing :
            sunshine_game3d::automatic_phase::renderer_unavailable;
          publish_automatic_ui(runtime, unavailable);
          return;
        }
        rtv = renderer->native_rtv(backbuffer);
        const auto desc = device->get_resource_desc(backbuffer);
        proof.width = desc.texture.width; proof.height = desc.texture.height;
        proof.frame_strength = settings.strength;
        // The export transfer (docs/reshade-sbs.md, PQ wire transfer): an
        // HDR10 swapchain exports 10-bit PQ to a consumer that accepts it,
        // and a native scRGB one to a consumer that encodes HDR10 PQ; any
        // other consumer gets FP16 scRGB (exports_pq).
        const auto space = swapchain->get_color_space();
        proof.pq_requested = exports_pq(space, read_capabilities(*ring_.shared_, read_nonce(*ring_.shared_)));
        const bool pq = renderer->set_pq_output(proof.pq_requested);
        proof.color = {space, space == api::color_space::srgb ? wire::transfer::srgb : pq ? wire::transfer::pq : wire::transfer::scrgb};
        // The overlay compositor of this export is compiled off the Present
        // before the overlay first opens (once per process and transfer).
        sunshine::overlay::warm(export_space(proof.color.output));
      }
      if (!rtv.handle) {
        if (diagnostic_owner) debug_dump_.unavailable("The foreground Game 3D renderer has no valid backbuffer view.");
        return;
      }
      renderer->begin_frame_state();
      struct restore_game_state {
        sunshine_game3d::renderer *renderer;
        ~restore_game_state() { renderer->end_frame_state(); }
      } restore{renderer};
      timer.mark();
      renderer->begin_gpu_profile(commands); // Depth and UI input copies follow.
      // The owner brackets the same capture used by the FX reference path. No
      // raw resource pointer is retained once this lease closes.
      if (![&] { const sunshine_game3d::slow_step step("native depth frame"); return sunshine_depth::begin_native_frame(runtime, commands); }()) {
        if (diagnostic_owner) debug_dump_.unavailable("The depth owner cannot open this runtime's native render lease.");
        return;
      }
      struct finish_depth {
        api::effect_runtime *runtime; api::command_list *commands;
        ~finish_depth() { sunshine_depth::end_native_frame(runtime, commands); }
      } finish{runtime, commands};
      {
        const sunshine_game3d::slow_step step("native depth preparation");
        prepare_native_depth(runtime, commands, rtv);
      }
      timer.mark();
      auto &ui_session = sunshine_game3d::source_alpha_session();
      publish_alpha.ui_session = &ui_session;
      auto ui_input = sunshine_game3d::ui_input::acquire(runtime, *renderer, source_alpha, ui_session, diagnostic_owner);
      // Snapshots leased by direct binding return to their resting state after
      // this Present's last read of them (the render, its owed pack in frame()
      // and a dump), before ReShade submits the immediate list at Present.
      struct end_leases {
        const bool &leased; api::command_list *commands;
        ~end_leases() {
          if (leased) sunshine_streamline::depth_capture::end_local_views(commands->get_native());
        }
      } leases{ui_input.leased, commands};
      timer.mark();
      // Publish what this render actually consumes, not the saved preference or
      // a later SDK observation. No RGB from the retained input is displayed.
      bool rendered = false;
      {
        // The timing line's render section: this lock, the UI input's match
        // and completion, and the render itself (named on its own below).
        const sunshine_game3d::slow_step section("stereo render section");
        std::lock_guard<std::mutex> lock(mutex_);
        auto &proof = runtimes_[runtime];
        const auto &scene = proof.frame.scene;
        sunshine_game3d::render_parameters p;
        p.strength = std::min(settings.strength, scene.admitted_strength); p.depth_view = settings.depth_view;
        p.depth_ready = proof.frame.depth_ready; p.camera_ready = scene.ready;
        p.coordinate_basis = scene.basis; p.depth_scale = scene.scale; p.strength_blend = scene.blend;
        p.projection = scene.projection; p.raw_depth_range = scene.raw_range;
        p.convergence = scene.zero; p.jitter = proof.frame.jitter.offset; p.depth_rect = scene.rect;
        proof.diagnostic_ui_source = std::move(ui_input.source_metadata);
        proof.diagnostic_ui_capture_attempt = std::move(ui_input.capture_metadata);
        const auto ui_observation = ui_input.match_scene(proof.frame.ui_source,
          proof.frame.prepared && proof.frame.depth_ready && scene.ready);
        // Reused depth belongs to an earlier frame: no hidden-scene evidence.
        sunshine_game3d::render_frame_input input{backbuffer, proof.borrowed_depth, p,
          ui_input.for_render(scene.ui_plane, ui_observation), !proof.frame.reused_depth};
        // A Dump 3D packs this Present with its full conditioning.
        input.diagnostic_armed = proof.diagnostic_armed;
        // Without a consumer (or an armed dump) nothing reads the adaptive
        // probe's conditioning: render() keeps running for the depth pool (E4
        // was rejected), but submits no probe.
        input.adaptive_probe = read_nonce(*ring_.shared_) != 0 || proof.diagnostic_armed;
        input.depth_identity = depth_identity(proof);
        {
          const sunshine_game3d::slow_step step("stereo render");
          rendered = proof.frame.prepared && renderer->render(commands, input, true);
        }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        ++proof.test_render.sequence;
        proof.test_render.rendered = rendered;
        proof.test_render.depth = proof.test_depth;
        proof.test_render.parameters = p;
        // The per-Present trace's frame decision and reentry ramp (game3d_test_render.h).
        proof.test_render.gap = static_cast<std::uint8_t>(proof.frame.gap);
        proof.test_render.scene_source = proof.frame.scene_source;
        proof.test_render.depth_identity = input.depth_identity;
        proof.test_render.reentry_heard = proof.reentry_heard;
        proof.test_render.reentry_phase = static_cast<std::uint8_t>(proof.raw_reentry.test_state());
        proof.test_render.reentry_basis = static_cast<std::uint8_t>(proof.raw_reentry.test_basis());
        proof.test_depth = {}; // Never describe a later Present without its own preparation.
#endif
        ui_input.complete(*renderer, rendered);
        source_alpha = ui_input.status;
        proof.frame.source_alpha = source_alpha;
        proof.native_output = rendered ? renderer->output() : api::resource{};
        proof.native_ui_parallax_uv = rendered ? sunshine_game3d::admitted_ui_parallax_uv(
          renderer->consumed_parameters(), renderer->consumed_ui_plane(), scene.ready && proof.frame.depth_ready,
          renderer->consumed_source_alpha_ui(), proof.width, proof.height) : 0.f;
      }
      timer.mark();
      // Export records the owed pack only when a slot receives this frame. Without
      // a consumer, a free slot or a dump, the SBS pack is skipped.
      if (rendered) {
        const sunshine_game3d::slow_step step("stereo export");
        frame(runtime, {}, commands, rtv, true);
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &proof = runtimes_[runtime];
        // A dump still helps when there is no admitted streaming slot. Such a
        // capture explicitly has no export generation/sequence association.
        if (diagnostic_owner && proof.diagnostic_armed && debug_dump_.requested()) {
          if (rendered) capture_diagnostic(runtime, proof, commands, 0, 0);
          else debug_dump_.unavailable("The foreground Game 3D render did not complete a current input/output pair.");
        }
        proof.borrowed_depth = {};
        proof.diagnostic_depth = {};
        proof.diagnostic_ui_source.clear();
        proof.diagnostic_ui_capture_attempt.clear();
        proof.diagnostic_armed = false;
      }
    }

    void invalidate(api::effect_runtime *runtime, bool destroy, std::uint64_t native_swapchain = 0) {
      // A runtime that logged the UI counters ends with the session totals,
      // written after the lock below is released.
      struct final_ui_counters {
        api::effect_runtime *runtime;
        sunshine_game3d::alpha_auto_policy *session = nullptr;
        ~final_ui_counters() {
          if (session) log_ui_counters(runtime, *session);
        }
      } final_counters{runtime};
      // Terminal lifecycle events cannot be dropped: the runtime address may be reused.
      // The lock never encloses a CPU/GPU fence wait.
      std::lock_guard<std::mutex> lock(mutex_);
      const auto current = runtimes_.find(runtime);
      if (destroy) sunshine_game3d::ui_input::invalidate(runtime);
      if (destroy) debug_dump_.retire_runtime(runtime);
      if (!destroy && current != runtimes_.end() && current->second.addon_native) {
        // Effects can disappear while the native owner and scene continue.
        // Any later return to a reference must rediscover all FX handles.
        auto &proof = current->second;
        proof.native_uniforms_checked = false;
        proof.game_technique = proof.native_technique = {};
        proof.depth_ready = proof.calibrated = proof.raw_anchor = proof.raw_gain = proof.depth_rect = proof.direction = {};
        proof.camera_ready = proof.camera_basis = proof.camera_projection = proof.camera_scale = proof.camera_zero = proof.camera_blend = proof.camera_rect = proof.camera_raw_range = proof.depth_jitter = proof.effect_strength = {};
        proof.texture = {}; proof.proof_checked = false;
        return;
      }
      // The reset discards the runtime's timing window: its partial window is
      // logged now (before the renderer, whose GPU timing it takes, is parked).
      if (current != runtimes_.end() && current->second.timing.presents) {
        char message[1024]{};
        close_timing_window(current->second, GetTickCount64(), message);
        log(reshade::log::level::info, message);
      }
      // destroy_effect_runtime also precedes every swapchain resize (an SDR
      // and HDR toggle): the renderer is parked for the runtime's
      // reinitialisation, and released when the swapchain itself is
      // destroyed (swapchain_destroyed) or its device goes.
      if (destroy && current != runtimes_.end() && current->second.renderer) {
        auto &renderer = current->second.renderer;
        drop_parked(runtime);
        if (auto *device = renderer->device(); device && native_swapchain) {
          renderer->park_after_runtime_drain();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
          sunshine_game3d::test::teardown::note_parked(renderer.get(), true);
#endif
          parked_.push_back({runtime, native_swapchain, device, std::move(renderer)});
        } else renderer->reset_after_runtime_drain();
      }
      sunshine_game3d::depth_input::invalidate_reuse(runtime);
      ring_.deactivate(runtime, "runtime_reset");
      if (destroy) runtimes_.erase(runtime);
      else {
        // FX reload invalidates FX handles, not the runtime identity or native
        // GPU owner. Native can be re-enabled even after a disabled reload.
        auto &proof = runtimes_[runtime];
        auto renderer = std::move(proof.renderer);
        const auto swapchain = proof.swapchain;
        const auto source_alpha_policy = proof.source_alpha_policy;
        proof = runtime_t{};
        proof.swapchain = swapchain;
        proof.renderer = std::move(renderer);
        proof.source_alpha_policy = source_alpha_policy;
      }
      final_counters.session = clear_automatic_ui(runtime);
      if (ring_.generation_ && ring_.generation_->owner_runtime == runtime) {
        if (destroy && ring_.generation_->finished() && ring_.generation_->completed() != UINT64_MAX) {
          // A runtime reset may be a resize: the finished, deactivated ring
          // stays for a toggle back (the export path caches or renews it) until
          // the swapchain is destroyed. It holds no back buffer meanwhile.
          ring_.generation_->release_sources();
        } else {
          ring_.generation_->owner_destroyed |= destroy;
          if (ring_.generation_->finished()) {
            ring_.release_ring(ring_.generation_, export_views{*this});
          } else if (destroy) {
            ring_.retire_destroyed_generation(export_views{*this});
          }
        }
      }
      if (destroy) {
        colors_.erase(native_swapchain);
        overlays_.erase(runtime);
        // The cached ring stays across a runtime reset (swapchain_destroyed
        // releases it with its swapchain); it holds no back buffer meanwhile.
        if (ring_.cached_ring_ && ring_.cached_ring_->owner_runtime == runtime) ring_.cached_ring_->release_sources();
      }
    }

    // ReShade's destroy_swapchain without resize: the swapchain itself is
    // destroyed. Its parked renderer and its rings are released (a destroyed
    // owner's ring is never reused; an unfinished one is retired).
    void swapchain_destroyed(std::uint64_t native_swapchain) {
      if (!native_swapchain) return;
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = parked_.begin(); it != parked_.end();) {
        if (it->native_swapchain == native_swapchain) {
          it->renderer->reset_after_runtime_drain();
          it = parked_.erase(it);
        } else ++it;
      }
      if (ring_.generation_ && ring_.generation_->native_swapchain == native_swapchain && !ring_.generation_->runtime) {
        ring_.generation_->owner_destroyed = true;
        if (ring_.generation_->finished()) ring_.release_ring(ring_.generation_, export_views{*this});
        else ring_.retire_destroyed_generation(export_views{*this});
      }
      if (ring_.cached_ring_ && ring_.cached_ring_->native_swapchain == native_swapchain) {
        ring_.cached_ring_->owner_destroyed = true;
        if (ring_.cached_ring_->finished()) ring_.release_ring(ring_.cached_ring_, export_views{*this});
      }
    }

    // A destroyed device takes any renderer still parked on it.
    void device_destroyed(api::device *device) {
      // First every background renderer build on it: this may wait for the
      // one device call in progress, so never under mutex_.
      sunshine_game3d::renderer::device_destroyed(device);
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = parked_.begin(); it != parked_.end();) {
        if (it->device == device) {
          it->renderer->reset_after_runtime_drain();
          it = parked_.erase(it);
        } else ++it;
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
      const bool foreground = ring_.published_runtime_ == runtime && owns_foreground_locked(runtime, reinterpret_cast<HWND>(ring_.published_metadata_.window));
      if (ring_.published_runtime_ == runtime && (!foreground || found == runtimes_.end() || !found->second.rendered_since_present)) {
        // The pause line first, then this stop's one reuse invalidation (see
        // the not_foreground stop in frame()).
        if (ring_.deactivate(runtime, !foreground ? "not_foreground" :
            found == runtimes_.end() ? "runtime_gone" : "present_without_render"))
          sunshine_game3d::depth_input::invalidate_reuse(runtime);
      }
      if (ring_.generation_ && ring_.generation_->owner_runtime == runtime && ring_.generation_->pending_slot != wire::slot_count) {
        if (ring_.generation_->pending_overlay) {
          ring_.generation_->pending_overlay = false;
          if (ring_.generation_->runtime && !ring_.generation_->overlays[ring_.generation_->pending_slot]->finish()) {
            ring_.deactivate(runtime, "overlay_capture_failed");
            ring_.retry_at_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            log(reshade::log::level::warning, "Sunshine SBS: overlay capture unavailable; showing the captured desktop");
          }
        }
        // The export is complete once its pack and overlay are recorded:
        // publish it now, before the game's Present. D3D11 records on the
        // immediate context, which signal_pending flushes. D3D12 submits
        // ReShade's immediate list first; ReShade 6.8 fires finish_present only
        // after the real Present returns, so signalling there queued the fence
        // behind the Present packet and the CPU's Present wait.
        if (ring_.generation_->backend == api::device_api::d3d11) {
          ring_.signal_pending(runtime, output_counts{*this});
        } else if (auto *queue = runtime->get_command_queue();
            queue && queue->get_native() == reinterpret_cast<std::uint64_t>(ring_.generation_->queue12.get())) {
          {
            const sunshine_streamline::native_observer::own_flush_scope own;
            queue->flush_immediate_command_list();
          }
          ring_.signal_pending(runtime, output_counts{*this});
        }
      }
      ring_.release_idle_rings(export_views{*this});
      if (found != runtimes_.end()) {
        found->second.rendered_since_present = false;
        found->second.native_depth_prepared = false;
        found->second.frame.prepared = false;
      }
    }

    void draw_overlay(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (ring_.generation_ && ring_.generation_->runtime == runtime && ring_.generation_->pending_overlay) {
        ring_.generation_->overlays[ring_.generation_->pending_slot]->append_capture_callback();
      }
    }

    void decide_scene_frame(api::effect_runtime *runtime, runtime_t &proof, frame_decision_t &frame,
        const frame_decision_t &previous_frame, const sunshine_depth::frame_depth &depth,
        bool game_enabled, std::uint64_t now);

    void prepare_native_depth(api::effect_runtime *runtime, api::command_list *,
                              api::resource_view) {
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
          else if (std::strcmp(name, "Depth_Adjustment") == 0) proof.effect_strength = variable;
        });
        // Require the source-owned geometry interface; preset mode macros no
        // longer select an alternate renderer.
        proof.raw_supported = proof.camera_ready.handle && proof.camera_basis.handle &&
          proof.camera_projection.handle && proof.camera_scale.handle && proof.camera_zero.handle &&
          proof.camera_blend.handle && proof.camera_rect.handle;

      }
      if (!proof.addon_native && proof.effect_strength.handle)
        runtime->get_uniform_value_float(proof.effect_strength, &proof.frame_strength, 1);
      proof.frame_strength = std::isfinite(proof.frame_strength) ? std::clamp(proof.frame_strength, 0.f, 100.f) : 0.f;
      // Fail closed before reading mutable capture/policy state. Publication
      // below is the only writer that can enable the resolved camera branch.
      if (!proof.addon_native && proof.camera_ready.handle) runtime->set_uniform_value_bool(proof.camera_ready, false);
      // The selector's begin-effects callback runs first. Its current capture is
      // shader-readable now; calibration history never substitutes for readiness.
      sunshine_depth::frame_depth depth;
      const bool game_enabled = proof.addon_native || (proof.game_technique.handle && runtime->get_technique_state(proof.game_technique));
      if (!game_enabled) sunshine_game3d::depth_input::invalidate_reuse(runtime);
      const bool independent_enabled = !game_enabled && proof.native_technique.handle && runtime->get_technique_state(proof.native_technique);
      if (independent_enabled && proof.depth_ready.handle && proof.calibrated.handle && proof.raw_anchor.handle &&
          proof.raw_gain.handle && proof.depth_rect.handle && proof.direction.handle) {
        int direction = 0;
        runtime->get_uniform_value_int(proof.direction, &direction, 1);
        sunshine_game3d::depth_input::query(runtime, depth, direction == 1 ? sunshine_depth::depth_orientation::normal :
          direction == 2 ? sunshine_depth::depth_orientation::reversed : sunshine_depth::depth_orientation::automatic, true);
        apply_native_depth(runtime, proof, depth);
      } else {
        // The original-derived effect owns its preparation and convergence.
        // Camera diagnostics may inspect capture identity without enabling the
        // independent renderer's percentile calibration or altering uniforms.
        const sunshine_game3d::slow_step step("depth input query");
        sunshine_game3d::depth_input::query(runtime, depth);
      }
      proof.borrowed_depth = proof.addon_native && depth.ready ? depth.shader_resource : api::resource_view{};
      proof.diagnostic_armed = proof.addon_native && debug_dump_.requested();
      proof.diagnostic_depth = proof.diagnostic_armed ? depth : sunshine_depth::frame_depth{};
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      proof.test_depth = depth;
#endif
      {
        const sunshine_game3d::slow_step step("depth input observation");
        sunshine_game3d::hook_host::poll();
      }
      const auto now = GetTickCount64();
      frame_decision_t frame;
      decide_scene_frame(runtime, proof, frame, previous_frame, depth, game_enabled, now);
      const sunshine_game3d::slow_step step("frame publication");
      publish_frame(runtime, proof, std::move(frame));
    }

    scene_parameters_t resolve_raw_scene(api::effect_runtime *runtime, runtime_t &proof,
        const sunshine_depth::frame_depth &depth, const sunshine_game3d::depth_output::shown_depth &shown,
        bool game_enabled, bool api_selected, std::uint64_t now);

    void resolve_projection_scene(api::effect_runtime *runtime, runtime_t &proof,
        const sunshine_depth::frame_depth &depth, std::uint64_t now,
        const sunshine_scene_gain::limits &budget, scene_parameters_t &scene);

    static void publish_frame(api::effect_runtime *runtime, runtime_t &proof, frame_decision_t frame) {
      // Publish from this capture, including on FG reuse. Missing
      // metadata clears the prior offset; it must not borrow the newest jitter.
      if (!proof.addon_native && proof.depth_jitter.handle) runtime->set_uniform_value_float(proof.depth_jitter, frame.jitter.offset.data(), 2);
      const auto &scene = frame.scene;
      if (scene.owned && !proof.addon_native) {
        const float strength_ratio = proof.frame_strength > scene.admitted_strength && proof.frame_strength > 0.f ?
          scene.admitted_strength / proof.frame_strength : 1.f;
        if (proof.camera_rect.handle) runtime->set_uniform_value_float(proof.camera_rect, scene.rect.data(), 4);
        if (proof.camera_blend.handle) runtime->set_uniform_value_float(proof.camera_blend, scene.blend * strength_ratio);
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
      bool diagnostic_frame = false;
      const bool awaiting_diagnostic = debug_dump_.awaiting_capture();
      for (auto &[runtime, proof] : runtimes_)
        if (proof.swapchain == swapchain) {
          ++proof.presentation_ordinal;
          // A missed finish_present must not leave native rendering wedged.
          if (proof.renderer) proof.renderer->begin_present();
          diagnostic_frame = awaiting_diagnostic && owns_foreground_locked(runtime, static_cast<HWND>(runtime->get_hwnd()));
          break;
        }
      debug_dump_.poll(diagnostic_frame);
      // A preceding Present signalled its export neither in reshade_present nor
      // in finish_present. Retain its recording until a later matching
      // submission can be fenced, but stop exposing the previous cached image.
      if (ring_.generation_ && ring_.generation_->pending_slot != wire::slot_count && ring_.generation_->native_swapchain == swapchain) {
        ring_.deactivate(ring_.generation_->runtime, "present_not_finished");
      }
      colors_[swapchain] = color;
    }

    void finish_present(std::uint64_t queue, std::uint64_t swapchain) {
      std::lock_guard<std::mutex> lock(mutex_);
      debug_dump_.finish_present(queue, swapchain);
      for (auto &[runtime, proof] : runtimes_)
        if (proof.swapchain == swapchain && proof.renderer) proof.renderer->finish_present();
      // A D3D12 export is normally signalled in reshade_present (present()).
      // One whose reshade_present did not run is fenced here, after the game's
      // Present, which ReShade preceded with a flush of its immediate list.
      if (!ring_.generation_ || ring_.generation_->backend != api::device_api::d3d12 || ring_.generation_->pending_slot == wire::slot_count || swapchain != ring_.generation_->native_swapchain || queue != reinterpret_cast<std::uint64_t>(ring_.generation_->queue12.get())) {
        return;
      }
      ring_.signal_pending(nullptr, output_counts{*this});
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
      ring_.release_idle_rings(export_views{*this});
      if (game && proof.raw_supported && !proof.frame.prepared) {
        ring_.deactivate(runtime, "frame_not_prepared");
        return;
      }
      // The renamed primary effect wins over accidentally enabled reference
      // effects. Never alternate producers between techniques in one present.
      if (!game && proof.game_technique.handle && runtime->get_technique_state(proof.game_technique)) return;
      if (native && !proof.native_depth_prepared) {
        ring_.deactivate(runtime, "native_depth_unprepared");
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
      if (!window || window_pid != ring_.identity_.producer_pid || !owns_foreground_locked(runtime, window, true)) {
        // A focus loss stops the export and ends the provider's bounded depth
        // reuse once, after the pause line. While the export stays inactive
        // the provider keeps its own bounded reuse (the renderer still runs):
        // invalidating on every background Present cleared each real frame's
        // copy before its generated frames could hold it, a readiness loss on
        // every generated Present (Assassin's Creed Shadows with FG on, 10-23).
        if (ring_.deactivate(runtime, "not_foreground")) sunshine_game3d::depth_input::invalidate_reuse(runtime);
        return;
      }
      const auto color = colors_.find(runtime->get_native());
      if (color == colors_.end()) {
        ring_.deactivate(runtime, "swapchain_unknown");
        return;
      }
      // Add-ons may render techniques into deferred lists. Never signal an unrelated queue here.
      auto *queue = runtime->get_command_queue();
      if (!queue || commands != queue->get_immediate_command_list()) {
        ring_.deactivate(runtime, "deferred_command_list");
        return;
      }
      const auto backend = runtime->get_device()->get_api();
      if (backend != api::device_api::d3d11 && backend != api::device_api::d3d12) {
        ring_.deactivate(runtime, "unsupported_api");
        return;
      }

      // The compiled input transfer must match the current game swapchain, including PQ to
      // scRGB switches that keep the exported FP16 format and transfer unchanged.
      if ((addon_render ? !proof.native_output.handle : !proof.texture.handle) || !source_color_matches(runtime, proof, color->second)) {
        ring_.deactivate(runtime, "output_unavailable");
        return;
      }
      source_t source;
      if (!describe(runtime, proof, source)) {
        ring_.deactivate(runtime, "source_unavailable");
        return;
      }
      proof.rendered_since_present = true;

      const auto nonce = read_nonce(*ring_.shared_);
      if (!nonce) {
        if (ring_.generation_) {
          ring_.deactivate(ring_.generation_->runtime, "no_consumer");
        }
        wire::metadata_t metadata = ring_.identity_;
        export_ring::set_source(metadata, source);
        // No consumer means no shared GPU allocations or copies.
        if (ring_.published_runtime_ != runtime || ring_.published_metadata_.generation || ring_.published_metadata_.window != metadata.window || ring_.published_metadata_.source_width != metadata.source_width || ring_.published_metadata_.source_height != metadata.source_height || ring_.published_metadata_.dxgi_format != metadata.dxgi_format || ring_.published_metadata_.adapter_luid != metadata.adapter_luid || ring_.published_metadata_.color_transfer != metadata.color_transfer) {
          ring_.publish(metadata, false, runtime);
        }
        return;
      }
      if (!ring_.consumer_speaks_protocol(runtime, nonce)) {
        return;
      }
      // C3 (transfer_current): a consumer attached, replaced or detached
      // between this Present's render and its export with another transfer
      // gets the next Present's export, rendered for it.
      if (!transfer_current(addon_render, proof.pq_requested, source.color.output,
            exports_pq(source.color.input, read_capabilities(*ring_.shared_, nonce)))) {
        ring_.deactivate(runtime, addon_render ? "consumer_transfer_changed" : "consumer_without_pq");
        return;
      }
      if (!ring_.serve(runtime, source, nonce, backend, export_views{*this})) return;
      if (ring_.generation_->signal_failed) {
        return;
      }
      const auto completed = ring_.generation_->completed();
      if (completed == UINT64_MAX) {
        ring_.deactivate(runtime, "device_removed");
        return;
      }
      // The owed native pack renders straight into the slot. The overlay and
      // dumps read the internal SBS image, so they pack it there first.
      auto *renderer = addon_render ? proof.renderer.get() : nullptr;
      // The dump mailbox is read once per exported Present.
      const bool dump = addon_render && proof.diagnostic_armed && debug_dump_.requested();
      const bool direct = renderer && !overlay_open(runtime) && !dump;
      bool overwrote = false;
      const auto index = ring_.acquire_slot(completed, &overwrote);
      if (index == wire::slot_count) {
        // Logged here too (rate limited), so a ring that drops every Present
        // still reports it with published frozen.
        ++proof.output.dropped;
        log_output(runtime, proof, GetTickCount64());
      } else if (overwrote) ++proof.output.overwritten_unconsumed;
      if (index != wire::slot_count) {
        auto &slot = ring_.shared_->slots[index];
        // A nonce replacement races only on metadata. Do not issue new old-generation work.
        if (read_nonce(*ring_.shared_) != ring_.generation_->nonce) {
          store_state(slot, ring_.generation_->id, wire::slot_state::free);
          return;
        }
        const auto sequence = ring_.generation_->last_submitted + 1;
        LARGE_INTEGER timestamp {};
        QueryPerformanceCounter(&timestamp);
        // Mark ownership before recording any work, so reloads and callback failures keep
        // source, ring and overlay resources alive until that work can be fenced.
        ring_.generation_->pending_slot = index;
        ring_.generation_->pending_qpc = static_cast<std::uint64_t>(timestamp.QuadPart);
        // The scalar belongs to this exact packed texture, including retained
        // exports. ReShade's own controls and cursor are at screen disparity.
        ring_.generation_->pending_ui_parallax_uv = addon_render && !overlay_open(runtime) ? proof.native_ui_parallax_uv : 0.f;
        ring_.generation_->pending_output = game ? std::optional(classify_output(proof.frame)) : std::nullopt;
        if (direct) ring_.generation_->adopt_rendered(index, sequence);
        if (direct && renderer->pack(commands, ring_.generation_->texture(index), ring_.generation_->id)) {
          if (!ring_.generation_->rendered_directly) {
            ring_.generation_->rendered_directly = true;
            log(reshade::log::level::info, "Sunshine SBS: rendering stereo directly into export slots (no copy)");
          }
        } else {
          if (renderer) renderer->pack(commands);
          // With the overlay open its composite writes the whole slot from the
          // source in reshade_present, so nothing is copied first; a dump, a
          // reference export or a failed direct pack copies the source.
          ring_.generation_->submit(commands, source.resource, index, sequence, !overlay_open(runtime));
        }
        if (dump) capture_diagnostic(runtime, proof, commands, ring_.generation_->id, sequence);
        if (overlay_open(runtime)) {
          auto &overlay = ring_.generation_->overlays[index];
          if (!overlay) {
            overlay = std::make_unique<sunshine::overlay::compositor_t>();
          }
          if (!overlay->prepare(runtime, native_rtv, source.resource, ring_.generation_->texture(index), source.width, source.height, source.color.input, export_space(source.color.output))) {
            ring_.deactivate(runtime, "overlay_prepare_failed");
            ring_.retry_at_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            log(reshade::log::level::warning, "Sunshine SBS: cannot prepare stereo overlay; showing the captured desktop");
            return;
          }
          ring_.generation_->pending_overlay = true;
        }
        ring_.next_slot_ = (index + 1) % wire::slot_count;
        return;
      }
    }

  private:
    void capture_diagnostic(api::effect_runtime *runtime, runtime_t &proof, api::command_list *commands,
        std::uint64_t generation, std::uint64_t sequence) {
      if (!proof.renderer) return;
      proof.renderer->pack(commands); // The dump reads the internal SBS image.
      LARGE_INTEGER qpc{}; QueryPerformanceCounter(&qpc);
      sunshine_game3d::diagnostic_frame frame;
      frame.parameters = proof.renderer->consumed_parameters();
      frame.ui_plane = proof.renderer->consumed_ui_plane();
      frame.ui_channel = proof.renderer->consumed_ui_channel();
      if (frame.ui_plane.mode == sunshine_game3d::ui_plane_mode::display_fraction)
        frame.ui_adaptive = proof.renderer->consumed_ui_adaptive();
      frame.source_alpha_ui = proof.renderer->consumed_source_alpha_ui();
      frame.ui_detection = proof.renderer->consumed_detection();
      frame.source_alpha_decision = proof.frame.source_alpha;
      frame.ui_source_metadata = proof.diagnostic_ui_source;
      frame.ui_capture_attempt_metadata = proof.diagnostic_ui_capture_attempt;
      frame.scene_status = proof.frame.scene.ui;
      frame.depth = proof.diagnostic_depth;
      frame.resources = proof.renderer->diagnostics();
      frame.color = proof.color.input;
      frame.presentation_ordinal = proof.presentation_ordinal;
      frame.backbuffer_index = runtime->get_current_back_buffer_index();
      frame.shader_source = proof.renderer->active_shader_source();
      frame.export_generation = generation; frame.export_sequence = sequence;
      frame.qpc = static_cast<std::uint64_t>(qpc.QuadPart);
      debug_dump_.capture(runtime, commands, frame);
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
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      depth_events::readiness(runtime, depth_events::site_id::native_ready, ready);
#endif
      sunshine_game3d::depth_input::set_ready(runtime, ready);
      proof.native_depth_prepared = true;
    }

    bool source_color_matches(api::effect_runtime *runtime, const runtime_t &proof, api::color_space current) {
      if (proof.color.input == api::color_space::unknown || proof.color.input != current) {
        ring_.deactivate(runtime, "color_space_changed");
        return false;
      }
      return true;
    }

    bool overlay_open(api::effect_runtime *runtime) const {
      const auto found = overlays_.find(runtime);
      return found != overlays_.end() && found->second;
    }
    // Requires mutex_. Whether `window`, the swapchain window of `runtime`, is the window the user
    // sees in the foreground: the foreground window itself, or a child of it that owns its client
    // area (reshade_bridge::window_owner). When another runtime's window qualifies too, the two
    // are ambiguous and neither owns the foreground. `report` logs the verdict of a child window
    // and an ambiguity; a top-level window that is not the foreground is not a child, not logged.
    bool owns_foreground_locked(api::effect_runtime *runtime, HWND window, bool report = false) {
      namespace owner = reshade_bridge::window_owner;
      const HWND foreground = observed_foreground_window();
      const char *reason = "owner";
      bool owns = window == foreground, named = false;
      if (!owns) {
        const auto verdict = child_window_verdict(window, foreground);
        owns = verdict == owner::verdict::owner;
        named = verdict != owner::verdict::not_child;
        reason = owner::verdict_name(verdict);
      }
      if (owns && runtimes_.size() > 1) {
        for (const auto &entry : runtimes_) {
          const HWND other = entry.first == runtime ? nullptr : static_cast<HWND>(entry.first->get_hwnd());
          if (other && other != window && (other == foreground || child_window_verdict(other, foreground) == owner::verdict::owner)) {
            owns = false;
            named = true;
            reason = "ambiguous";
            break;
          }
        }
      }
      if (report && named) {
        report_window_owner(window, foreground, reason);
      } else if (report && window == window_report_.window) {
        // A top-level window that is no longer named (its ambiguity cleared) forgets its last
        // verdict without a line, so a recurring ambiguity is a change and is logged again.
        window_report_.reason = "";
      }
      return owns;
    }

    // Names a child (or ambiguous) swapchain window's verdict when it changes, at most every 5 s
    // (a later call names the latest verdict); the line counts the changes it summarises. A
    // routine verdict (routine_window_verdict) is info, any other a warning.
    void report_window_owner(HWND window, HWND foreground, const char *reason) {
      if (window != window_report_.window || std::strcmp(reason, window_report_.reason) != 0) {
        window_report_.window = window;
        window_report_.reason = reason;
        ++window_report_.changes;
        window_report_.pending = true;
      }
      const auto now = GetTickCount64();
      if (!window_report_.pending || now < window_report_.next_log) return;
      window_report_.pending = false;
      window_report_.next_log = now + 5000;
      char message[256];
      std::snprintf(message, sizeof(message), "Sunshine SBS: swapchain window %p %s foreground window %p: owner=%s changes=%u",
        static_cast<void *>(window), window == foreground ? "is the" : "presents under the", static_cast<void *>(foreground), reason,
        std::exchange(window_report_.changes, 0u));
      log(routine_window_verdict(reason) ? reshade::log::level::info : reshade::log::level::warning, message);
    }

    void log_output(api::effect_runtime *runtime, runtime_t &proof, std::uint64_t now);

    // What export_ring needs of this publisher (export_ring_class.inl), called
    // under mutex_. export_views: renderers drop the views they cached of a
    // released ring's slot texture (renderer::pack). output_counts: a published
    // slot's sample joins its runtime's SBS output counters (log_output).
    struct export_views {
      publisher_t &owner;
      void operator()(api::resource texture) const {
        for (auto &entry : owner.runtimes_)
          if (entry.second.renderer) entry.second.renderer->release_export_view(texture);
        for (auto &parked : owner.parked_)
          if (parked.renderer) parked.renderer->release_export_view(texture);
      }
    };
    struct output_counts {
      publisher_t &owner;
      void operator()(api::effect_runtime *runtime, const output_sample &sample, std::uint64_t now) const {
        const auto found = owner.runtimes_.find(runtime);
        if (found != owner.runtimes_.end()) {
          found->second.output.add(sample);
          owner.log_output(runtime, found->second, now);
        }
      }
    };

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

    static bool describe(api::effect_runtime *runtime, runtime_t &proof, source_t &source) {
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
        if (!proof.adapter_known) {
          com_ptr<IDXGIDevice> device;
          com_ptr<IDXGIAdapter> adapter;
          DXGI_ADAPTER_DESC adapter_desc {};
          auto *native = reinterpret_cast<ID3D11Device *>(runtime->get_device()->get_native());
          if (FAILED(native->QueryInterface(IID_PPV_ARGS(device.put()))) || FAILED(device->GetAdapter(adapter.put())) || FAILED(adapter->GetDesc(&adapter_desc))) {
            return false;
          }
          std::memcpy(&proof.adapter_luid, &adapter_desc.AdapterLuid, sizeof(proof.adapter_luid));
          proof.adapter_known = true;
        }
      } else {
        const auto desc = reinterpret_cast<ID3D12Resource *>(source.resource.handle)->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width != width * 2 || desc.Height != height || desc.MipLevels != 1 || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1) {
          return false;
        }
        source.format = typed_format(desc.Format);
        if (!proof.adapter_known) {
          auto *native = reinterpret_cast<ID3D12Device *>(runtime->get_device()->get_native());
          const LUID luid = native->GetAdapterLuid();
          std::memcpy(&proof.adapter_luid, &luid, sizeof(proof.adapter_luid));
          proof.adapter_known = true;
        }
      }
      source.adapter = proof.adapter_luid;
      return wire::supported_format(static_cast<std::uint32_t>(source.format), source.color.output);
    }

    handle_t mapping_;
    sunshine_game3d::debug_dump debug_dump_;
    std::mutex mutex_;
    std::unordered_map<api::effect_runtime *, runtime_t> runtimes_;
    std::unordered_map<std::uint64_t, api::color_space> colors_;
    std::unordered_map<api::effect_runtime *, bool> overlays_;
    // The export slots, generations and published metadata (export_ring),
    // at the active generation's former position: its generations are
    // destroyed after parked_ and before overlays_, as before.
    export_ring ring_;
    // Renderers of runtimes reset by ReShade (destroy_effect_runtime), kept
    // for the runtime's reinitialisation after a resize (track_runtime) until
    // the swapchain or device is destroyed.
    struct parked_t {
      api::effect_runtime *runtime = nullptr;
      std::uint64_t native_swapchain = 0;
      api::device *device = nullptr;
      std::unique_ptr<sunshine_game3d::renderer> renderer;
    };
    std::vector<parked_t> parked_;
    // The last child or ambiguous swapchain window verdict (report_window_owner).
    struct {
      HWND window = nullptr;
      const char *reason = "";
      unsigned changes = 0;
      bool pending = false;
      std::uint64_t next_log = 0;
    } window_report_;
  };

  addon_session_t::addon_session_t() = default;
  addon_session_t::~addon_session_t() = default;
  addon_session_t &addon_session() {
    // AddonInit pins this module and ReShade can notify unload after our CRT
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
      sunshine_streamline::native_observer::set_submission_listener(nullptr);
      sunshine_game3d::ui_layer::unregister_events();
      sunshine_game3d::hook_host::shutdown();
      sunshine_depth::shutdown();
      reshade::unregister_addon(addon, reshade);
    });
  }

  void on_technique(api::effect_runtime *runtime, api::effect_technique technique, api::command_list *commands, api::resource_view rtv, api::resource_view) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    const depth_events::phase_scope phase(depth_events::handler::render_technique);
#endif
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
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    const depth_events::phase_scope phase(depth_events::handler::reload);
#endif
    if (publisher) {
      publisher->invalidate(runtime, false);
    }
  }

  void on_init_runtime(api::effect_runtime *runtime) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    const depth_events::phase_scope phase(depth_events::handler::init);
#endif
    if (publisher) publisher->track_runtime(runtime);
    // Windows display scaling can hold a DPI-unaware game below its display's
    // resolution (docs/reshade-sbs.md, display scaling); say so once per
    // swapchain (re)initialization. The panel shows the same warning.
    if (!runtime) return;
    uint32_t width = 0, height = 0;
    runtime->get_screenshot_width_and_height(&width, &height);
    if (const auto scale = sunshine_game3d::display_scale::query(runtime->get_hwnd(), width, height); scale.limited) {
      char message[320];
      std::snprintf(message, sizeof(message),
        "Sunshine Game 3D: display scaling limits the game: it renders at %ux%u on a %ux%u display because "
        "Windows display scaling is %u%%; set scaling to 100%% or this game's high-DPI override to Application",
        scale.game_width, scale.game_height, scale.display_width, scale.display_height, scale.scale_percent);
      log(reshade::log::level::warning, message);
    }
  }

  void on_begin_effects(api::effect_runtime *runtime, api::command_list *commands, api::resource_view rtv, api::resource_view) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    const depth_events::phase_scope phase(depth_events::handler::begin_effects);
#endif
    try {
      if (publisher && !publisher->native_enabled(runtime)) publisher->prepare_native_depth(runtime, commands, rtv);
    } catch (...) {
      if (publisher) publisher->invalidate(runtime, false);
      log(reshade::log::level::error, "Sunshine SBS: depth calibration update failed; skipping source");
    }
  }

  void on_destroy(api::effect_runtime *runtime) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    const depth_events::phase_scope phase(depth_events::handler::destroy);
#endif
    if (publisher) {
      publisher->invalidate(runtime, true, runtime->get_native());
    }
  }

  void on_destroy_swapchain(api::swapchain *swapchain, bool resize) {
    if (publisher && !resize && swapchain) publisher->swapchain_destroyed(swapchain->get_native());
  }

  void on_destroy_device(api::device *device) {
    if (publisher) publisher->device_destroyed(device);
  }

  void on_present(api::effect_runtime *runtime) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    const depth_events::phase_scope phase(depth_events::handler::reshade_present);
#endif
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
    // ReShade flushes its immediate list right after reshade_present, on the
    // same queue under the same queue lock. Flushing it here instead, as the
    // add-on's (own_flush_scope), leaves that flush nothing to submit, so
    // ReShade's Signal of its own fence there never waits on the queue-order
    // lock on the present thread either.
    if (auto *queue = runtime->get_command_queue(); queue && runtime->get_device()->get_api() == api::device_api::d3d12) {
      const sunshine_streamline::native_observer::own_flush_scope own;
      queue->flush_immediate_command_list();
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

  void on_begin_present(api::command_queue *queue, api::swapchain *swapchain, const api::rect *, const api::rect *, std::uint32_t, const api::rect *) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    const depth_events::phase_scope phase(depth_events::handler::present);
#endif
    try {
      if (publisher) {
        publisher->begin_present(swapchain->get_native(), swapchain->get_color_space());
        if (publisher->owns_foreground(swapchain))
          sunshine_game3d::ui_layer::observe_output(swapchain, queue);
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

  // What the last native Game 3D render consumed (game3d_test_render.h).
  __declspec(dllexport) BOOL SunshineGame3DTestLastRender(api::effect_runtime *runtime,
      sunshine_game3d::test::last_render *out) {
    return publisher && runtime && out && publisher->test_last_render(runtime, *out) ? TRUE : FALSE;
  }
#else
  __declspec(dllexport) const char *NAME = "Sunshine 3D";
  __declspec(dllexport) const char *DESCRIPTION = "Automatically selects and calibrates game depth for Sunshine stereo. Includes SDR/HDR sharing and stereo overlay controls.";
#endif

  __declspec(dllexport) bool AddonInit(HMODULE addon, HMODULE reshade) {
    if (sunshine_addon_lifetime::stopping()) return false;
    // Pin this module before anything below can start pool work, install a
    // hook or register a callback. ReShade unloads add-ons whenever a game
    // destroys its last device (Assassin's Creed Shadows does at startup),
    // while module discovery and shader compiles this add-on started on pool
    // threads may still be running its code and its imports (D3DCompiler).
    // Pinned, it stays mapped until process exit, and the next device
    // initializes it again through AddonInit.
    HMODULE self{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(&AddonInit), &self)) return false;
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
      sunshine_game3d::hook_host::initialize(addon);
      sunshine_depth::initialize(addon);
      reshade::register_event<reshade::addon_event::init_effect_runtime>(sunshine_addon_lifetime::guarded<on_init_runtime>);
      reshade::register_event<reshade::addon_event::reshade_begin_effects>(sunshine_addon_lifetime::guarded<on_begin_effects>);
      reshade::register_event<reshade::addon_event::reshade_render_technique>(sunshine_addon_lifetime::guarded<on_technique>);
      reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(sunshine_addon_lifetime::guarded<on_reload>);
      reshade::register_event<reshade::addon_event::destroy_effect_runtime>(sunshine_addon_lifetime::guarded<on_destroy>);
      reshade::register_event<reshade::addon_event::destroy_swapchain>(sunshine_addon_lifetime::guarded<on_destroy_swapchain>);
      reshade::register_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
      reshade::register_event<reshade::addon_event::reshade_present>(sunshine_addon_lifetime::guarded<on_present>);
      reshade::register_event<reshade::addon_event::present>(sunshine_addon_lifetime::guarded<on_begin_present>);
      reshade::register_event<reshade::addon_event::finish_present>(sunshine_addon_lifetime::guarded<on_finish_present>);
      reshade::register_event<reshade::addon_event::reshade_open_overlay>(sunshine_addon_lifetime::guarded<on_overlay>);
      reshade::register_event<reshade::addon_event::reshade_overlay>(sunshine_addon_lifetime::guarded<on_draw_overlay>);
      sunshine_game3d::ui_layer::register_events();
      // The layer's cross-queue fence is signalled right after each native
      // submission that ran a copy away from the presenting queue.
      sunshine_streamline::native_observer::set_submission_listener(sunshine_game3d::ui_layer::observed_submission);
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

#include "exporter/present_diagnostics.inl"

#include "exporter/ui_status.inl"

#include "exporter/scene_placement_impl.inl"
