// SPDX-License-Identifier: GPL-3.0-only
// TEST add-on only: the game clock's exports (game3d_clock.h). A fixture
// engages the virtual clock once its setup is done, reads it for its own
// game-time stamps and moves it forward between its add-on calls; until it
// engages, game time is the real tick. The production package excludes these
// exports.
#ifndef SUNSHINE_GAME3D_TEST_CLOCK
#error "Built only into the test add-on (SunshineSBSTest), whose view defines SUNSHINE_GAME3D_TEST_CLOCK"
#endif
#include "game3d_clock.h"

#include <cstdint>

// Engages the virtual clock lead_ms ahead of the real tick, or leaves it where
// it is when it is further ahead already (never 0, never back, no disengage),
// and returns it. The lead keeps a read that raced the engage on another
// thread, which returned the real tick, in the past (game3d_clock.h).
extern "C" __declspec(dllexport) std::uint64_t SunshineGame3DTestClockEngage(std::uint64_t lead_ms) {
  return sunshine_game3d::clock::engage(lead_ms);
}

// The game clock as the add-on reads it: the virtual value once engaged,
// before that the real tick.
extern "C" __declspec(dllexport) std::uint64_t SunshineGame3DTestClockNow() {
  return sunshine_game3d::clock::game_now();
}

// Moves the engaged clock forward by milliseconds and returns its new value;
// 0 (refused, unchanged) while it is not engaged.
extern "C" __declspec(dllexport) std::uint64_t SunshineGame3DTestClockAdvance(std::uint64_t milliseconds) {
  return sunshine_game3d::clock::advance(milliseconds);
}
