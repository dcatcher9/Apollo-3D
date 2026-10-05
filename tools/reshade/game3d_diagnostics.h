// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <atomic>

namespace sunshine_game3d::diagnostics {
  // The add-on's Diagnostics switch (docs/reshade-sbs.md, Diagnostics switch):
  // ReShade.ini [SUNSHINE_GAME3D] Diagnostics=1 turns on diagnostic-only
  // per-frame work (per-pass GPU timestamps and per-Present diagnostic
  // bookkeeping that only feeds dumps and logs). Absent writes 0, so
  // the key is discoverable; 0 (or anything else) keeps it off. Nothing that
  // decides reads it: UI decisions, masks and exported pixels are the same
  // either way. One process-wide value, read on any thread (game threads
  // included) with a relaxed load; the controls set it at load and on edit.
  inline constexpr const char *key = "Diagnostics";
  inline constexpr const char *label = "Diagnostics";
  inline constexpr const char *tooltip =
    "Off (default): no per-pass GPU timing. On: records per-pass GPU timing and other per-frame evidence for logs and "
    "Dump 3D, at a small extra cost per frame. UI decisions and the 3D image are the same either way. Saved per game "
    "(Diagnostics).";

  inline std::atomic<bool> &state() noexcept {
    static std::atomic<bool> value{false};
    return value;
  }
  inline bool enabled() noexcept { return state().load(std::memory_order_relaxed); }
  inline void set_enabled(bool value) noexcept { state().store(value, std::memory_order_relaxed); }
}  // namespace sunshine_game3d::diagnostics
