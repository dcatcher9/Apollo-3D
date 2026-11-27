// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <Windows.h>
#include <array>
#include <cstdint>

namespace sunshine_upscaler_trace {
  // The default-off upscaler call-route diagnostic. The NGX hook host
  // (ngx_hooks.h), which owns the hooks and production NGX depth capture,
  // feeds it each hooked call and the Streamline evaluations it nests in; the
  // trace owns only its records and report lines. No destructible locks,
  // containers or workers: detours stay valid after the add-on's normal
  // shutdown.
  bool enabled();
  // Report actual SL evaluation-hook coverage, separately from call counts.
  void set_streamline_coverage(bool installed);

  // The hook host's feed (ngx_hooks.cpp); nothing else calls it.
  namespace feed {
    enum class operation : unsigned { create, evaluate, release, streamline };
    // The hook host's target bound: a hooked call names its target's index
    // (below target_limit); a Streamline evaluation names no_target.
    inline constexpr unsigned target_limit = 64, no_target = target_limit, stack_limit = 8;
    inline constexpr std::uint32_t unknown_feature = 0xffffffffu;
    struct invocation {
      std::uint64_t epoch{};
      std::uintptr_t caller{};
      bool inside_sl{};
      unsigned stack_count{};
      std::array<void *, stack_limit> stack{};
    };
    // The trace epoch: zero while the trace is off.
    std::uint64_t epoch();
    // Off, then its records cleared and, when enabled, a new epoch begun.
    void restart(bool enabled);
    void stop();
    // A hooked call's entry. sl_epoch, sl_depth: the calling thread's
    // Streamline evaluation nesting (its scope's epoch and depth).
    invocation begin(unsigned target, std::uintptr_t caller, std::uint64_t sl_epoch, unsigned sl_depth);
    // A call's outcome. owner, d3d11: the hooked export's module and API,
    // the namespace of its feature handles (a core and snippet may reuse
    // pointers).
    void record(unsigned target, HMODULE owner, bool d3d11, operation op, const invocation &call,
      std::uint32_t feature, const void *handle, bool success, std::uint32_t failure = 0);
    // The hook host's discovery and hooks, as one report reads them.
    struct target_coverage {
      const void *entry{};
      operation op{};
      bool d3d11{};
      unsigned state{}; // 1 created, 2 enabled, 3 rejected.
    };
    struct coverage {
      std::array<target_coverage, target_limit> targets{};
      unsigned count{}, rejected{}, modules{}, exports{}, skipped_frame_generation{}, truncations{};
      bool truncated{};
    };
    void report(const coverage &hooks);
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    struct counts {
      std::uint64_t epoch{}, ngx_create{}, ngx_evaluate{}, ngx_release{}, streamline{},
        inside_streamline{}, outside_streamline{}, succeeded{}, failed{},
        unknown_feature{}, dropped{}, stale{}, reports{};
      bool streamline_covered{};
    };
    counts test_counts();
#endif
  }
}
