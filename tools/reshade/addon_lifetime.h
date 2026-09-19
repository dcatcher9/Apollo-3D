// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <atomic>
#include <utility>

namespace sunshine_addon_lifetime {
  // DllMain sets this before the CRT destroys any C++ owners. Vendor detours
  // and ReShade callbacks can outlive those owners because the add-on is pinned.
  // The fence itself is constant-initialized and trivially destructible.
  inline std::atomic<bool> detaching{};
  static_assert(std::atomic<bool>::is_always_lock_free);
  inline bool stopping() noexcept { return detaching.load(std::memory_order_acquire); }
  inline void begin_detach() noexcept { detaching.store(true, std::memory_order_release); }

  template<auto Function> struct callback;
  template<class Result, class... Arguments, Result (*Function)(Arguments...)>
  struct callback<Function> {
    static Result invoke(Arguments... arguments) {
      if (!stopping()) return Function(std::forward<Arguments>(arguments)...);
      return Result(); // void(), or false for ReShade's cancel-operation callbacks.
    }
  };
  template<auto Function> inline constexpr auto guarded = &callback<Function>::invoke;
}
