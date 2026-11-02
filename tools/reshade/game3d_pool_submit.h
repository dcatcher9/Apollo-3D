// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#ifdef SUNSHINE_GAME3D_TEST_CLOCK
#include <atomic>
#include <new>
#endif

// Every thread-pool callback the add-on starts goes through pool::submit,
// named by its family (README, Test builds). The shipping add-on submits
// directly; the test add-on (SUNSHINE_GAME3D_TEST_CLOCK) also counts each
// family's callbacks from submission until they return, so that a fixture
// can wait until the add-on's pool work is done (SunshineGame3DTestQuiesce,
// test_addon_lifetime_exports.cpp). A failed submission returns FALSE in
// both, and each caller keeps its own fallback for that.
namespace sunshine_game3d::pool {
  enum class family : unsigned {
    log,             // async_log.h: the log writer's drain
    storage_release, // depth_capture/pool.inl: capture storage released off the caller
    sampler_warm,    // depth_content_sampler.cpp: the content sampler's shader compile
    display_release, // depth_display.inl: provider display textures released off the present thread
    renderer_build,  // game3d_renderer.cpp: a background renderer build
    shader_prepare,  // game3d_shader_cache.h: one shader entry's compile
    ngx_discover,    // ngx_hooks.inl: NGX module discovery
    overlay_warm,    // overlay_compositor.cpp: the overlay's shader compile
    sl_install,      // sl_hooks.inl: Streamline hook installation
    test_held,       // TEST add-on only: SunshineSbsTestHoldPoolWork's held callback
    count
  };

  constexpr const char *family_name(family kind) {
    switch (kind) {
      case family::log: return "log";
      case family::storage_release: return "storage_release";
      case family::sampler_warm: return "sampler_warm";
      case family::display_release: return "display_release";
      case family::renderer_build: return "renderer_build";
      case family::shader_prepare: return "shader_prepare";
      case family::ngx_discover: return "ngx_discover";
      case family::overlay_warm: return "overlay_warm";
      case family::sl_install: return "sl_install";
      case family::test_held: return "test_held";
      case family::count: break;
    }
    return "unknown";
  }

  constexpr unsigned family_bit(family kind) { return 1u << unsigned(kind); }

#ifdef SUNSHINE_GAME3D_TEST_CLOCK
  namespace detail {
    struct counters_t {
      std::atomic<unsigned> total {0};
      std::atomic<unsigned> by_family[unsigned(family::count)] {};
    };
    inline counters_t counters;

    struct submission {
      PTP_SIMPLE_CALLBACK callback;
      void *context;
      family kind;
    };

    // The family first: once total reads 0, every family does.
    inline void finish(family kind) {
      --counters.by_family[unsigned(kind)];
      --counters.total;
    }

    inline void CALLBACK trampoline(PTP_CALLBACK_INSTANCE instance, void *context) {
      const auto work = *static_cast<const submission *>(context);
      delete static_cast<submission *>(context);
      work.callback(instance, work.context);
      finish(work.kind);
    }
  }

  inline BOOL submit(family kind, PTP_SIMPLE_CALLBACK callback, void *context) {
    // Without memory for the record, as without a pool thread.
    auto *work = new (std::nothrow) detail::submission {callback, context, kind};
    if (!work) return FALSE;
    ++detail::counters.total;
    ++detail::counters.by_family[unsigned(kind)];
    if (TrySubmitThreadpoolCallback(detail::trampoline, work, nullptr)) return TRUE;
    delete work;
    detail::finish(kind);
    return FALSE;
  }

  // Callbacks submitted and not yet returned, of every family.
  inline unsigned outstanding() { return detail::counters.total.load(); }

  // family_bit of each family with a callback submitted and not yet returned.
  inline unsigned outstanding_families() {
    unsigned families = 0;
    for (unsigned i = 0; i != unsigned(family::count); ++i)
      if (detail::counters.by_family[i].load()) families |= family_bit(family(i));
    return families;
  }
#else
  inline BOOL submit(family, PTP_SIMPLE_CALLBACK callback, void *context) {
    return TrySubmitThreadpoolCallback(callback, context, nullptr);
  }
#endif
}
