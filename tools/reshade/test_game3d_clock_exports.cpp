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

// Engages the virtual clock at the later of the real tick and its current
// value (never 0, never backwards, no disengage) and returns it.
extern "C" __declspec(dllexport) std::uint64_t SunshineGame3DTestClockEngage() {
  return sunshine_game3d::clock::engage();
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
