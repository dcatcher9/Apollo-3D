// SPDX-License-Identifier: GPL-3.0-only
// The exporter's base declarations, which all of its parts share: its one
// namespace, the session and its UI status entries, logging, foreground
// observation and the handle owner. exporter_internal.h adds the export ring,
// the scene values and the publisher (tools/reshade/README.md, Source map).
#pragma once
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
#include "presentation_scope.h"
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
#include "com_ptr.h"
#include "game3d_slow_step.h"
#include "game3d_ui_layer.h"
#include "game3d_vram.h"
#include "game3d_diagnostics.h"
#include "game3d_clock.h"

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  #include "game3d_test_depth_events.h"
  #include "game3d_test_teardown.h"

  #include <atomic>
#endif

static_assert(RESHADE_API_VERSION == 20, "Build against the pinned ReShade 6.8.0 SDK");
static_assert(sizeof(void *) == 8, "Only 64-bit games are supported");

// The exporter's internal namespace, the same in every view (shipping, the test
// add-on and the SBS unit tests), so that its parts compile as translation units
// of their own. A top-level namespace looks names up as the anonymous namespace
// it replaced did (its own scope, then the global one), and this using-directive
// gives exporter.cpp's exports, the public UI queries and the lifecycle fixture
// its names unqualified, as that namespace's implicit one did.
namespace sunshine_exporter {}
using namespace sunshine_exporter;

namespace sunshine_exporter {
  namespace api = reshade::api;
  namespace wire = reshade_bridge;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace depth_events = sunshine_game3d::test::depth_events;
#endif

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
  extern std::atomic<std::uint64_t> next_raw_basis;
  void log(reshade::log::level level, const char *message);
  addon_session_t &addon_session();
  // No ReShade/GPU calls while holding this mutex. UI readers and requests never
  // acquire the publisher mutex, including synchronous uniform-set callbacks.
  extern decltype(addon_session_t::ui_mutex) &automatic_ui_mutex;
  extern decltype(addon_session_t::ui) &automatic_ui;

  // UI status publication, defined in exporter/ui_status.cpp.
  sunshine_game3d::alpha_auto_policy *clear_automatic_ui(api::effect_runtime *runtime);
  void log_ui_counters(api::effect_runtime *runtime, sunshine_game3d::alpha_auto_policy &session);
  void publish_automatic_ui(api::effect_runtime *runtime, sunshine_game3d::automatic_status status);
  void publish_source_alpha_ui(api::effect_runtime *runtime, sunshine_game3d::source_alpha_ui_decision value,
      bool game3d_enabled = true, sunshine_game3d::alpha_auto_policy *ui_session = nullptr);
#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
  bool take_recalibration(api::effect_runtime *runtime);
#endif

#if defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON) || defined(SUNSHINE_SBS_TEST)
  extern std::atomic<HWND> test_foreground_window;
#endif
  HWND observed_foreground_window();
  reshade_bridge::window_owner::verdict child_window_verdict(HWND window, HWND foreground);
  void log_hr(const char *operation, HRESULT hr);

  using sunshine_game3d::com_ptr;

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
}  // namespace sunshine_exporter
