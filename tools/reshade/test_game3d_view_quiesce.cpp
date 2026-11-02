// SPDX-License-Identifier: GPL-3.0-only
// In-process fixtures (README, Test builds): an executable that links a view
// of the add-on's sources with the test clock's seam (the probe and renderer
// views) runs the add-on's code in its own process, so it exports the test
// add-on's clock exports itself (test_game3d_clock_exports.cpp, built into the
// same executable) and this barrier, over its own view's pool count and log
// queue. test_game3d_clock.h then binds the executable
// (bind(GetModuleHandleW(nullptr))) exactly as it binds SunshineSBSTest.
//
// The barrier is the test add-on's SunshineGame3DTestQuiesce
// (test_addon_lifetime_exports.cpp), which a view cannot link: it waits, at
// most bound_ms of real time, until every callback submitted through
// pool::submit has returned (game3d_pool_submit.h) and then the log writer's
// queue is empty with no drain under way (async_log.h; always so in a view
// that logs synchronously). The pool is read first: a callback that returned
// queued its lines before.
#ifndef SUNSHINE_GAME3D_TEST_CLOCK
  #error "Built only into an executable that links a view with the test clock's seam (SUNSHINE_GAME3D_TEST_CLOCK)"
#endif
#include "async_log.h"
#include "game3d_pool_submit.h"

#include <chrono>
#include <mutex>
#include <windows.h>

extern "C" __declspec(dllexport) BOOL SunshineGame3DTestQuiesce(unsigned bound_ms, unsigned *missed_families) {
  namespace pool = sunshine_game3d::pool;
  const auto log_idle = [] {
    auto &queue = sunshine_log::detail::queue();
    std::lock_guard<std::mutex> lock(queue.mutex);
    return queue.lines.empty() && !queue.draining;
  };
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    if (!pool::outstanding() && log_idle()) {
      if (missed_families) {
        *missed_families = 0;
      }
      return TRUE;
    }
    const auto waited = std::chrono::steady_clock::now() - start;
    if (waited >= std::chrono::milliseconds(bound_ms)) {
      if (missed_families) {
        *missed_families = pool::outstanding_families() | (log_idle() ? 0u : pool::family_bit(pool::family::log));
      }
      return FALSE;
    }
    // Short work ends within a few yields; longer work is polled.
    if (waited < std::chrono::milliseconds(2)) {
      SwitchToThread();
    } else {
      Sleep(1);
    }
  }
}
