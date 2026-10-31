// SPDX-License-Identifier: GPL-3.0-only
#define SUNSHINE_EXPORTER_IMPLEMENTATION
#define SUNSHINE_EXPORTER_TEST_ACCESS_STAGE 0
#include "exporter_test_access.h"
#undef SUNSHINE_EXPORTER_TEST_ACCESS_STAGE
namespace SUNSHINE_EXPORTER_INTERNAL {
  namespace api = reshade::api;
  namespace wire = reshade_bridge;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace depth_events = sunshine_game3d::test::depth_events;
#endif
  std::atomic<std::uint64_t> next_raw_basis {1};
  void log(reshade::log::level level, const char *message);

#define SUNSHINE_EXPORTER_TEST_ACCESS_STAGE 1
#include "exporter_test_access.h"
#undef SUNSHINE_EXPORTER_TEST_ACCESS_STAGE
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

#define SUNSHINE_EXPORTER_TEST_ACCESS_STAGE 2
#include "exporter_test_access.h"
#undef SUNSHINE_EXPORTER_TEST_ACCESS_STAGE

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

#define SUNSHINE_EXPORTER_TEST_ACCESS_STAGE 3
#include "exporter_test_access.h"
#undef SUNSHINE_EXPORTER_TEST_ACCESS_STAGE

  // A child swapchain window's verdicts (report_window_owner) that the player
  // or the game's own window changes explain, as a top-level window's
  // not_foreground pause is routine: the owner, a root out of focus, a hidden
  // child and one being resized. They are logged at info; a sibling overlap,
  // another process's window in the chain, an incomplete query, two
  // qualifying swapchains and any new verdict are warnings.
#define SUNSHINE_EXPORTER_TEST_ACCESS_STAGE 4
#include "exporter_test_access.h"
#undef SUNSHINE_EXPORTER_TEST_ACCESS_STAGE

  // The depth view's content identity for the renderer's conditioning memo
  // (render_frame_input::depth_identity): nonzero only for an API capture,
  // whose borrowed view holds that capture's immutable pixels for its epoch,
  // source, sequence and viewport (a reused generated-frame presentation
  // repeats them exactly). Generic depth and missing depth give 0.
  std::uint64_t depth_identity(const runtime_t &proof) {
    return sunshine_game3d::depth_output::depth_identity(proof.frame.scene_source,
      proof.borrowed_depth.handle != 0, proof.frame.depth_ready);
  }

#define SUNSHINE_EXPORTER_TEST_ACCESS_STAGE 5
#include "exporter_test_access.h"
#undef SUNSHINE_EXPORTER_TEST_ACCESS_STAGE

#define SUNSHINE_EXPORTER_TEST_ACCESS_STAGE 6
#include "exporter_test_access.h"
#undef SUNSHINE_EXPORTER_TEST_ACCESS_STAGE

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
#include "exporter/fx_reference.inl"

#undef SUNSHINE_EXPORTER_IMPLEMENTATION
