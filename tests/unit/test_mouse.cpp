/**
 * @file tests/unit/test_mouse.cpp
 * @brief Test the keyboard modifier handling in src/input.*.
 */
#include "../tests_common.h"

#include <src/input.h>

TEST(KeyboardModifierTest, SuppressesAltOnlyForSoleRemappedRightAlt) {
  constexpr uint16_t VKEY_LWIN = 0x5B;
  constexpr uint16_t VKEY_RWIN = 0x5C;
  constexpr uint16_t VKEY_RMENU = 0xA5;
  constexpr uint16_t VKEY_CONTROL = 0x11;

  EXPECT_TRUE(input::detail::suppress_synthetic_alt(VKEY_LWIN, false, true));
  EXPECT_TRUE(input::detail::suppress_synthetic_alt(VKEY_RWIN, false, true));

  EXPECT_FALSE(input::detail::suppress_synthetic_alt(VKEY_RMENU, false, true));
  EXPECT_FALSE(input::detail::suppress_synthetic_alt(VKEY_CONTROL, false, true));
  EXPECT_FALSE(input::detail::suppress_synthetic_alt(VKEY_LWIN, true, true));
  EXPECT_FALSE(input::detail::suppress_synthetic_alt(VKEY_LWIN, false, false));
}
