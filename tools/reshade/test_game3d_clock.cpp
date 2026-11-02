// SPDX-License-Identifier: GPL-3.0-only
// The test branch of the game clock (game3d_clock.h, compiled here with
// SUNSHINE_GAME3D_TEST_CLOCK as the test add-on's view compiles it): game time
// is the real tick until a fixture engages the virtual clock, and from then
// on it moves only when the fixture advances it, forward only.
#ifndef SUNSHINE_GAME3D_TEST_CLOCK
#error "Compiled with the test add-on's clock seam (SUNSHINE_GAME3D_TEST_CLOCK)"
#endif
#include "game3d_clock.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  namespace clock = sunshine_game3d::clock;
  try {
    // Not engaged: the real tick, and an advance is refused.
    const auto before = GetTickCount64();
    const auto unengaged = clock::game_now();
    const auto after = GetTickCount64();
    require(before <= unengaged && unengaged <= after, "An unengaged game clock did not read the real tick");
    require(clock::advance(16) == 0 && clock::virtual_ms.load() == 0, "An advance engaged the clock");

    // Engaged at the real tick at the latest: never 0, earlier stamps stay in
    // the past, and it stands still while real time passes.
    const auto stamped = clock::game_now();
    const auto engaged = clock::engage();
    require(engaged != 0 && engaged >= stamped && engaged <= GetTickCount64(),
      "Engaging did not start at the real tick");
    require(clock::game_now() == engaged, "The engaged clock did not read its virtual value");
    Sleep(40);
    require(clock::game_now() == engaged, "The engaged clock moved with real time");

    // Advance: forward by exactly the step, and 0 changes nothing.
    require(clock::advance(16) == engaged + 16 && clock::game_now() == engaged + 16, "An advance did not add its step");
    require(clock::advance(0) == engaged + 16 && clock::game_now() == engaged + 16, "A zero advance moved the clock");

    // Engaging again never moves the clock back.
    const auto ahead = clock::advance(1000000);
    require(ahead == engaged + 16 + 1000000, "A long advance did not add its step");
    require(clock::engage() == ahead && clock::game_now() == ahead, "Engaging again moved the clock back");

    // An advance that would wrap is refused and leaves the clock unchanged.
    require(clock::advance(UINT64_MAX) == 0 && clock::game_now() == ahead, "A wrapping advance was accepted");
    require(clock::advance(UINT64_MAX - ahead) == UINT64_MAX && clock::game_now() == UINT64_MAX,
      "An advance to the last value was refused");
    require(clock::advance(1) == 0 && clock::game_now() == UINT64_MAX, "An advance past the last value was accepted");

    std::cout << "The unengaged clock reads the real tick; the engaged clock starts at it, stands still and moves "
                 "only forward by its advances\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
