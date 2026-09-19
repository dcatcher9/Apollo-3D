// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <Windows.h>
#include <cstdint>

namespace sunshine_upscaler_trace {
  // Shared native entry observation: diagnostics and NGX depth capture are
  // independent. Owner serializes initialize/poll/shutdown outside loader lock.
  // Hooks are additive and process-pinned, never removed while a game can be
  // inside an original. Shutdown makes them pure pass-through.
  void initialize(HMODULE addon, bool enabled, bool capture = false, bool calibration_probe = false);
  void shutdown();
  bool enabled();
  bool capture_enabled();
  void poll();
  // Report actual SL evaluation-hook coverage, separately from call counts.
  void set_streamline_coverage(bool installed);

  class streamline_scope {
  public:
    // Caller must be captured at the actual SL detour, not in a helper.
    streamline_scope(std::uint32_t feature, std::uintptr_t caller);
    ~streamline_scope();
    void finish(bool success);
    streamline_scope(const streamline_scope &) = delete;
    streamline_scope &operator=(const streamline_scope &) = delete;
  private:
    std::uint64_t epoch_{}, previous_epoch_{};
    std::uint32_t feature_{}, previous_depth_{};
    std::uintptr_t caller_{};
    bool finished_{};
  };

#ifdef SUNSHINE_UPSCALER_TRACE_TEST
  namespace testing {
    enum class operation : unsigned { create, evaluate, release };
    struct summary {
      std::uint64_t epoch{}, ngx_create{}, ngx_evaluate{}, ngx_release{}, streamline{},
        inside_streamline{}, outside_streamline{}, succeeded{}, failed{},
        unknown_feature{}, dropped{}, stale{}, reports{};
      unsigned installed{}, discovered{}, rejected{};
      bool enabled{}, streamline_covered{};
    };
    summary counts();
    // Same native validation/pinning/install route as discovered exports.
    bool install(void *target, operation op, bool d3d11 = false);
    void report_now();
  }
#endif
}
