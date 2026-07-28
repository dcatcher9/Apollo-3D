/**
 * @file tests/unit/platform/test_windows_misc.cpp
 * @brief Tests for Windows platform state helpers.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <algorithm>
  #include <optional>
  #include <string>
  #include <vector>

  #include <src/platform/windows/misc.h>
  #include <ShlObj.h>

namespace {
  struct fake_explorer_restart_operations_t {
    std::optional<platf::explorer_restart_result_e> preflight_failure;
    bool terminate_succeeds = true;
    bool exit_succeeds = true;
    bool automatic_replacement = false;
    bool replacement_before_launch = false;
    bool launch_succeeds = true;
    bool fallback_replacement = true;
    std::vector<std::string> calls;

    std::optional<platf::explorer_restart_result_e> preflight() {
      calls.emplace_back("preflight");
      return preflight_failure;
    }

    bool terminate_exact_shell() {
      calls.emplace_back("terminate");
      return terminate_succeeds;
    }

    bool wait_for_old_shell_exit() {
      calls.emplace_back("wait-exit");
      return exit_succeeds;
    }

    bool wait_for_replacement(bool after_fallback_launch) {
      calls.emplace_back(after_fallback_launch ? "wait-fallback" : "wait-auto");
      return after_fallback_launch ? fallback_replacement : automatic_replacement;
    }

    bool replacement_present_now() {
      calls.emplace_back("check-before-launch");
      return replacement_before_launch;
    }

    bool launch_captured_shell() {
      calls.emplace_back("launch");
      return launch_succeeds;
    }
  };

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

TEST(ExplorerRestartControllerTest, AcceptsWindowsAutomaticRestartWithoutLaunchingAgain) {
  fake_explorer_restart_operations_t operations;
  operations.automatic_replacement = true;

  EXPECT_EQ(
    platf::detail::run_explorer_restart(operations),
    platf::explorer_restart_result_e::restarted_automatically
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "preflight",
      "terminate",
      "wait-exit",
      "wait-auto",
    })
  );
}

TEST(ExplorerRestartControllerTest, LaunchesExactlyOneFallbackAfterAutoRestartTimeout) {
  fake_explorer_restart_operations_t operations;

  EXPECT_EQ(
    platf::detail::run_explorer_restart(operations),
    platf::explorer_restart_result_e::relaunched
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "preflight",
      "terminate",
      "wait-exit",
      "wait-auto",
      "check-before-launch",
      "launch",
      "wait-fallback",
    })
  );
  EXPECT_EQ(
    std::ranges::count(operations.calls, "launch"),
    1
  );
}

TEST(ExplorerRestartControllerTest, CoalescesALateAutomaticRestartBeforeFallbackLaunch) {
  fake_explorer_restart_operations_t operations;
  operations.replacement_before_launch = true;

  EXPECT_EQ(
    platf::detail::run_explorer_restart(operations),
    platf::explorer_restart_result_e::restarted_automatically
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "preflight",
      "terminate",
      "wait-exit",
      "wait-auto",
      "check-before-launch",
      "wait-fallback",
    })
  );
  EXPECT_EQ(std::ranges::count(operations.calls, "launch"), 0);
}

TEST(ExplorerRestartControllerTest, NeverLaunchesWhenPreflightKillOrExitFails) {
  {
    fake_explorer_restart_operations_t operations;
    operations.preflight_failure =
      platf::explorer_restart_result_e::skipped_identity;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::skipped_identity
    );
    EXPECT_EQ(operations.calls, (std::vector<std::string> {"preflight"}));
  }
  {
    fake_explorer_restart_operations_t operations;
    operations.terminate_succeeds = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::terminate_failed
    );
    EXPECT_EQ(
      operations.calls,
      (std::vector<std::string> {"preflight", "terminate"})
    );
  }
  {
    fake_explorer_restart_operations_t operations;
    operations.exit_succeeds = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::exit_timeout
    );
    EXPECT_EQ(
      operations.calls,
      (std::vector<std::string> {"preflight", "terminate", "wait-exit"})
    );
  }
}

TEST(ExplorerRestartControllerTest, ReportsFallbackLaunchAndReplacementFailures) {
  {
    fake_explorer_restart_operations_t operations;
    operations.launch_succeeds = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::relaunch_failed
    );
    EXPECT_EQ(operations.calls.back(), "launch");
  }
  {
    fake_explorer_restart_operations_t operations;
    operations.fallback_replacement = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::replacement_timeout
    );
    EXPECT_EQ(operations.calls.back(), "wait-fallback");
    EXPECT_EQ(std::ranges::count(operations.calls, "launch"), 1);
  }
}

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

TEST(UserLocalAppDataTest, MatchesTheCurrentOrLinkedStandardUserProfile) {
  DWORD process_session = 0;
  ASSERT_TRUE(ProcessIdToSessionId(GetCurrentProcessId(), &process_session));
  if (process_session != WTSGetActiveConsoleSessionId()) {
    GTEST_SKIP() << "This contract requires an interactive test process";
  }

  HANDLE process_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(
    GetCurrentProcess(),
    TOKEN_QUERY,
    &process_token
  ));
  auto process_token_guard = util::fail_guard([&]() {
    CloseHandle(process_token);
  });

  TOKEN_ELEVATION_TYPE elevation_type {};
  DWORD returned = 0;
  ASSERT_TRUE(GetTokenInformation(
    process_token,
    TokenElevationType,
    &elevation_type,
    sizeof(elevation_type),
    &returned
  ));
  TOKEN_ELEVATION elevation {};
  ASSERT_TRUE(GetTokenInformation(
    process_token,
    TokenElevation,
    &elevation,
    sizeof(elevation),
    &returned
  ));
  if (
    elevation_type == TokenElevationTypeDefault &&
    elevation.TokenIsElevated != 0
  ) {
    GTEST_SKIP()
      << "A UAC-disabled administrator intentionally has no standard token";
  }

  HANDLE profile_token = nullptr;
  if (elevation_type == TokenElevationTypeFull) {
    TOKEN_LINKED_TOKEN linked {};
    ASSERT_TRUE(GetTokenInformation(
      process_token,
      TokenLinkedToken,
      &linked,
      sizeof(linked),
      &returned
    ));
    profile_token = linked.LinkedToken;
  }
  auto profile_token_guard = util::fail_guard([&]() {
    if (profile_token) {
      CloseHandle(profile_token);
    }
  });

  PWSTR expected_value = nullptr;
  ASSERT_TRUE(SUCCEEDED(SHGetKnownFolderPath(
    FOLDERID_LocalAppData,
    KF_FLAG_DEFAULT,
    profile_token,
    &expected_value
  )));
  ASSERT_NE(expected_value, nullptr);
  auto expected_value_guard = util::fail_guard([&]() {
    CoTaskMemFree(expected_value);
  });

  const auto actual = platf::user_local_appdata();
  ASSERT_FALSE(actual.empty());
  EXPECT_EQ(
    CompareStringOrdinal(
      actual.c_str(),
      -1,
      expected_value,
      -1,
      TRUE
    ),
    CSTR_EQUAL
  );
}

TEST(RunAsActiveUserTest, CanInspectTheUserProfileWithUsableImpersonation) {
  DWORD process_session = 0;
  ASSERT_TRUE(ProcessIdToSessionId(GetCurrentProcessId(), &process_session));
  if (process_session != WTSGetActiveConsoleSessionId()) {
    GTEST_SKIP() << "This contract requires an interactive test process";
  }

  HANDLE process_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token));
  auto process_token_guard = util::fail_guard([&]() {
    CloseHandle(process_token);
  });
  TOKEN_ELEVATION process_elevation {};
  DWORD returned = 0;
  ASSERT_TRUE(GetTokenInformation(
    process_token,
    TokenElevation,
    &process_elevation,
    sizeof(process_elevation),
    &returned
  ));
  if (process_elevation.TokenIsElevated == 0) {
    GTEST_SKIP() << "The impersonation regression requires an elevated tray process";
  }

  const auto expected_user = platf::active_user_id();
  if (!expected_user) {
    GTEST_SKIP() << "Windows did not expose a validated active desktop user";
  }
  const auto profile = platf::user_local_appdata();
  ASSERT_FALSE(profile.empty());

  bool callback_called = false;
  const auto result = platf::run_as_active_user(
    [&]() {
      callback_called = true;

      HANDLE thread_token = nullptr;
      ASSERT_TRUE(OpenThreadToken(
        GetCurrentThread(),
        TOKEN_QUERY,
        TRUE,
        &thread_token
      ));
      auto thread_token_guard = util::fail_guard([&]() {
        CloseHandle(thread_token);
      });
      SECURITY_IMPERSONATION_LEVEL level = SecurityAnonymous;
      DWORD returned = 0;
      ASSERT_TRUE(GetTokenInformation(
        thread_token,
        TokenImpersonationLevel,
        &level,
        sizeof(level),
        &returned
      ));
      EXPECT_GE(level, SecurityImpersonation);

      TOKEN_ELEVATION elevation {};
      ASSERT_TRUE(GetTokenInformation(
        thread_token,
        TokenElevation,
        &elevation,
        sizeof(elevation),
        &returned
      ));
      EXPECT_EQ(elevation.TokenIsElevated, 0u);

      HANDLE profile_handle = CreateFileW(
        profile.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        nullptr
      );
      ASSERT_NE(profile_handle, INVALID_HANDLE_VALUE)
        << "Win32 error " << GetLastError();
      CloseHandle(profile_handle);
    },
    expected_user
  );

  EXPECT_FALSE(result);
  EXPECT_TRUE(callback_called);
}
#endif
