/**
 * @file tests/unit/platform/test_windows_misc.cpp
 * @brief Tests for Windows platform state helpers.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <src/platform/windows/misc.h>

namespace {
  MOUSEKEYS sample_mouse_keys_state() {
    MOUSEKEYS state {};
    state.cbSize = sizeof(state);
    state.dwFlags = MKF_MODIFIERS;
    state.iMaxSpeed = 37;
    state.iTimeToMaxSpeed = 1400;
    state.iCtrlSpeed = 3;
    return state;
  }

  void expect_same_mouse_keys_state(const MOUSEKEYS &actual, const MOUSEKEYS &expected) {
    EXPECT_EQ(actual.cbSize, expected.cbSize);
    EXPECT_EQ(actual.dwFlags, expected.dwFlags);
    EXPECT_EQ(actual.iMaxSpeed, expected.iMaxSpeed);
    EXPECT_EQ(actual.iTimeToMaxSpeed, expected.iTimeToMaxSpeed);
    EXPECT_EQ(actual.iCtrlSpeed, expected.iCtrlSpeed);
    EXPECT_EQ(actual.dwReserved1, expected.dwReserved1);
    EXPECT_EQ(actual.dwReserved2, expected.dwReserved2);
  }
}  // namespace

TEST(MouseKeysControllerTest, EnablesOnceAndRestoresTheExactPreviousState) {
  platf::detail::mouse_keys_controller_t controller;
  const auto original = sample_mouse_keys_state();
  MOUSEKEYS applied {};

  EXPECT_TRUE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      state = original;
      return true;
    },
    [&](MOUSEKEYS &state) {
      applied = state;
      return true;
    }
  ));
  EXPECT_TRUE(controller.enabled_by_host());
  EXPECT_EQ(applied.dwFlags & (MKF_MOUSEKEYSON | MKF_AVAILABLE), MKF_MOUSEKEYSON | MKF_AVAILABLE);
  EXPECT_EQ(applied.dwFlags & MKF_MODIFIERS, MKF_MODIFIERS);
  EXPECT_EQ(applied.iMaxSpeed, original.iMaxSpeed);
  EXPECT_EQ(applied.iTimeToMaxSpeed, original.iTimeToMaxSpeed);
  EXPECT_EQ(applied.iCtrlSpeed, original.iCtrlSpeed);

  EXPECT_FALSE(controller.refresh(
    false,
    [](MOUSEKEYS &) {
      ADD_FAILURE() << "The saved state must not be queried again";
      return false;
    },
    [](MOUSEKEYS &) {
      ADD_FAILURE() << "Mouse Keys must not be enabled twice";
      return false;
    }
  ));

  EXPECT_FALSE(controller.restore([](MOUSEKEYS &) {
    return false;
  }));
  EXPECT_TRUE(controller.enabled_by_host());

  MOUSEKEYS restored {};
  EXPECT_TRUE(controller.restore([&](MOUSEKEYS &state) {
    restored = state;
    return true;
  }));
  EXPECT_FALSE(controller.enabled_by_host());
  expect_same_mouse_keys_state(restored, original);
}

TEST(MouseKeysControllerTest, LeavesExistingOrUnavailableStateAlone) {
  platf::detail::mouse_keys_controller_t controller;
  int getter_calls = 0;
  int setter_calls = 0;

  EXPECT_FALSE(controller.refresh(
    true,
    [&](MOUSEKEYS &) {
      ++getter_calls;
      return true;
    },
    [&](MOUSEKEYS &) {
      ++setter_calls;
      return true;
    }
  ));
  EXPECT_EQ(getter_calls, 0);
  EXPECT_EQ(setter_calls, 0);

  EXPECT_FALSE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      ++getter_calls;
      state = sample_mouse_keys_state();
      state.dwFlags |= MKF_MOUSEKEYSON | MKF_AVAILABLE;
      return true;
    },
    [&](MOUSEKEYS &) {
      ++setter_calls;
      return true;
    }
  ));
  EXPECT_EQ(getter_calls, 1);
  EXPECT_EQ(setter_calls, 0);
  EXPECT_FALSE(controller.enabled_by_host());
}

TEST(MouseKeysControllerTest, RetriesAfterQueryOrEnableFailures) {
  platf::detail::mouse_keys_controller_t controller;
  const auto original = sample_mouse_keys_state();

  EXPECT_FALSE(controller.refresh(
    false,
    [](MOUSEKEYS &) {
      return false;
    },
    [](MOUSEKEYS &) {
      ADD_FAILURE() << "Setter must not run after a query failure";
      return false;
    }
  ));
  EXPECT_FALSE(controller.enabled_by_host());

  EXPECT_FALSE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      state = original;
      return true;
    },
    [](MOUSEKEYS &) {
      return false;
    }
  ));
  EXPECT_FALSE(controller.enabled_by_host());

  EXPECT_TRUE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      state = original;
      return true;
    },
    [](MOUSEKEYS &) {
      return true;
    }
  ));
  EXPECT_TRUE(controller.enabled_by_host());
}
#endif
