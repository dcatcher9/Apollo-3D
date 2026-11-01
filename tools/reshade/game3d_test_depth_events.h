// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// TEST add-on only (SunshineDepthTestBindingEvents): every DEPTH binding and
// readiness write of the add-on, in order, for the runtime fixtures'
// per-Present decision trace (refactor proof T8). Each record names the
// runtime, the Present it came with, the event handler it ran in (the phase,
// which phase_scope marks in the handlers that reach a DEPTH write), the call
// site, and whether the bound view was null or the readiness value written.
// The call sites record just before they call; a readiness write goes
// through the readiness uniform cache, which may find the value unchanged.
// One fixed ring (game3d_test_event_ring.h) and a fixed table of the threads
// inside a marked handler, under one lock no other code takes; nothing
// allocates, and no thread-local storage is added. The production add-on
// records and exports nothing.
#include "game3d_test_event_ring.h"

#include <cstdint>

namespace sunshine_game3d::test::depth_events {
  // The add-on event handler a write ran in. The values are stable record
  // ids.
  enum class handler : std::uint8_t {
    none = 0,  // Outside every marked handler.
    present = 1,  // addon_event::present: exporter on_begin_present (the native render).
    begin_effects = 2,  // reshade_begin_effects: depth_addon on_begin_render_effects, exporter on_begin_effects.
    render_technique = 3,  // reshade_render_technique: exporter on_technique.
    reload = 4,  // reshade_reloaded_effects: depth_addon on_reload_effect_runtime, exporter on_reload.
    init = 5,  // init_effect_runtime: exporter on_init_runtime.
    destroy = 6,  // destroy_effect_runtime: exporter on_destroy.
    reshade_present = 7,  // reshade_present: exporter on_present.
  };

  // The call site. The values are stable record ids, never renumbered.
  enum class site_id : std::uint8_t {
    none = 0,
    provider_bind = 1,  // streamline_depth_provider.cpp bind(): the binding of a changed view.
    provider_bind_ready = 2,  // bind(): its readiness.
    provider_reload_bind = 3,  // reload_effect_bindings(): the binding, while the API owns the pass.
    provider_reload_ready = 4,  // reload_effect_bindings(): its readiness.
    provider_reuse_ready = 5,  // invalidate_reused_depth(): false for a reused copy.
    generic_bind = 6,  // depth_addon.cpp update_effect_runtime(): the binding of a changed view.
    generic_ready = 7,  // update_effect_runtime(): its readiness.
    native_ready = 8,  // exporter.cpp apply_native_depth(): depth_input::set_ready.
    // streamline_depth_provider.cpp reload(): the runtime reload's reset of
    // the provider's binding and readiness caches (it calls neither), which
    // depth_addon.cpp on_reload_effect_runtime makes only with Game 3D off.
    provider_runtime_reload = 9,
    // 10 was provider_display_evict, prepare_display()'s record of an
    // evicted display and whether the effects named it (never: 15 of 15 in
    // round 0); it went on 11-01 with the unbind it guarded. Not reused.
  };

  // view or ready of a record that has none.
  inline constexpr std::uint8_t not_applicable = 2;

  // One record.
  struct event {
    std::uint64_t serial;  // 1 for the first event since the add-on loaded.
    std::uint64_t runtime;  // The effect runtime's address.
    // The exporter's present handlers entered so far in this process, the
    // current one included (counted as each enters, before its own writes).
    std::uint64_t present;
    handler phase;
    site_id site;
    // A binding: 1 a view, 0 null. A readiness write or a reset:
    // not_applicable.
    std::uint8_t view;
    // A readiness write: the value, 1 or 0. A binding or a reset:
    // not_applicable.
    std::uint8_t ready;
    std::uint8_t reserved[4];
  };
  static_assert(sizeof(event) == 32);

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace detail {
    inline constexpr unsigned capacity = 1024;
    inline constexpr unsigned thread_capacity = 16;

    struct thread_phase {
      DWORD thread;  // 0: free (no thread has id 0).
      handler phase;
    };

    struct state_t {
      event_ring<event, capacity> ring;
      std::uint64_t presents;
      thread_phase threads[thread_capacity];
    };

    inline state_t state;

    // Requires state.ring.lock.
    inline thread_phase *find_locked(DWORD thread) {
      for (auto &entry : state.threads) {
        if (entry.thread == thread) {
          return &entry;
        }
      }
      return nullptr;
    }

    inline void record(const void *runtime, site_id site, std::uint8_t view, std::uint8_t ready) {
      const auto self = GetCurrentThreadId();
      AcquireSRWLockExclusive(&state.ring.lock);
      const auto *current = find_locked(self);
      auto &value = state.ring.append_locked();
      value.runtime = reinterpret_cast<std::uintptr_t>(runtime);
      value.present = state.presents;
      value.phase = current ? current->phase : handler::none;
      value.site = site;
      value.view = view;
      value.ready = ready;
      ReleaseSRWLockExclusive(&state.ring.lock);
    }
  }  // namespace detail

  // update_texture_bindings for the DEPTH texture, about to be called.
  inline void binding(const void *runtime, site_id site, bool view) {
    detail::record(runtime, site, view ? 1 : 0, not_applicable);
  }

  // A readiness write, about to be made.
  inline void readiness(const void *runtime, site_id site, bool ready) {
    detail::record(runtime, site, not_applicable, ready ? 1 : 0);
  }

  // A reset of a binding or readiness cache that writes neither, about to be
  // made.
  inline void reset(const void *runtime, site_id site) {
    detail::record(runtime, site, not_applicable, not_applicable);
  }

  // Marks an event handler for the writes on its thread until it returns;
  // nested scopes restore the outer phase. A present scope counts a Present.
  class phase_scope {
  public:
    explicit phase_scope(handler phase) {
      const auto self = GetCurrentThreadId();
      AcquireSRWLockExclusive(&detail::state.ring.lock);
      auto *entry = detail::find_locked(self);
      if (!entry) {
        entry = detail::find_locked(0);
        if (entry) {
          entry->thread = self;
          entry->phase = handler::none;
        }
      }
      if (entry) {
        previous_ = entry->phase;
        entry->phase = phase;
      }
      if (phase == handler::present) {
        ++detail::state.presents;
      }
      ReleaseSRWLockExclusive(&detail::state.ring.lock);
    }

    ~phase_scope() {
      const auto self = GetCurrentThreadId();
      AcquireSRWLockExclusive(&detail::state.ring.lock);
      if (auto *entry = detail::find_locked(self)) {
        entry->phase = previous_;
        if (previous_ == handler::none) {
          entry->thread = 0;
        }
      }
      ReleaseSRWLockExclusive(&detail::state.ring.lock);
    }

    phase_scope(const phase_scope &) = delete;
    phase_scope &operator=(const phase_scope &) = delete;

  private:
    handler previous_ = handler::none;
  };

  // SunshineDepthTestBindingEvents (event_ring::read).
  inline unsigned read(std::uint64_t &cursor, event *out, unsigned count, std::uint64_t &skipped) {
    return detail::state.ring.read(cursor, out, count, skipped);
  }
#endif
}  // namespace sunshine_game3d::test::depth_events
