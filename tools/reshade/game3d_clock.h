// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <cstdint>
#ifdef SUNSHINE_GAME3D_TEST_CLOCK
#include <atomic>
#endif

// The add-on's game clock, in milliseconds. Every read of game time is
// sunshine_game3d::clock::game_now(): source ages, holds and windows,
// cadences, idle releases, log throttles and the ticks diagnostics print, so
// a stamp and every read that ages it share one clock. Wall time keeps its
// literal OS call: release_device's fence-wait bound, the export ring's
// steady_clock lifetimes (the receiver process sees them), the renderer's
// present-thread time slice and the QueryPerformanceCounter measurements.
// check_clock_domains.cmake (the reshade_clock_domains test) holds every
// other clock read and pool submission to a stated domain (README, Test
// builds).
namespace sunshine_game3d::clock {
#ifdef SUNSHINE_GAME3D_TEST_CLOCK
  // TEST add-on only (SunshineSBSTest's view, test_game3d_clock_exports.cpp):
  // a virtual game clock that a fixture engages after its setup and then
  // moves forward between its add-on calls. 0 means not engaged, and game
  // time is then the real tick, as in the shipping build.
  inline std::atomic<std::uint64_t> virtual_ms{0};

  inline std::uint64_t game_now() {
    const auto value = virtual_ms.load(std::memory_order_acquire);
    return value ? value : GetTickCount64();
  }

  // Engages the virtual clock at the later of the real tick and its current
  // value, so every tick stamped before stays in the past: never 0, never
  // backwards, and there is no disengage. Returns the engaged value.
  inline std::uint64_t engage() {
    const std::uint64_t real = GetTickCount64();
    const std::uint64_t start = real ? real : 1;
    auto current = virtual_ms.load(std::memory_order_acquire);
    while (current < start &&
        !virtual_ms.compare_exchange_weak(current, start, std::memory_order_acq_rel, std::memory_order_acquire)) {}
    return current < start ? start : current;
  }

  // Moves the engaged clock forward by ms and returns its new value. Refused
  // (0, the clock unchanged) while it is not engaged, and when the sum would
  // wrap.
  inline std::uint64_t advance(std::uint64_t ms) {
    auto current = virtual_ms.load(std::memory_order_acquire);
    while (current && ms <= UINT64_MAX - current &&
        !virtual_ms.compare_exchange_weak(current, current + ms, std::memory_order_acq_rel, std::memory_order_acquire)) {}
    return current && ms <= UINT64_MAX - current ? current + ms : 0;
  }
#elif defined(__GNUC__)
  // The shipping game clock is the real tick: a second name for the
  // GetTickCount64 import (the same assembler symbol), so every read
  // compiles exactly as a direct GetTickCount64() call. An inline wrapper
  // would not: inlining it shifts register allocation in large callers.
  __declspec(dllimport) ULONGLONG WINAPI game_now() __asm__("GetTickCount64");
#else
  inline std::uint64_t game_now() { return GetTickCount64(); }
#endif
}  // namespace sunshine_game3d::clock
