/**
 * @file tests/unit/platform/test_windows_misc.cpp
 * @brief Tests for Windows platform state helpers.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <algorithm>
  #include <boost/filesystem/path.hpp>
  #include <boost/process/v1/child.hpp>
  #include <boost/process/v1/environment.hpp>
  #include <boost/process/v1/group.hpp>
  #include <cstdint>
  #include <cstdio>
  #include <filesystem>
  #include <optional>
  // display.h declares the small D3DKMT function table without depending on the WDK headers.
  // Match display_base.cpp's prerequisite after boost/process has selected its Windows headers.
  typedef long NTSTATUS;
  #include <src/platform/windows/display.h>
  #include <src/platform/windows/misc.h>
  #include <sddl.h>
  #include <ShlObj.h>
  #include <string>
  #include <vector>

namespace {
  struct fake_keyed_mutex_t {
    HRESULT acquire_result = S_OK;
    unsigned acquire_calls = 0;
    unsigned release_calls = 0;
    UINT64 acquired_key = 0;
    UINT64 released_key = 0;

    HRESULT AcquireSync(const UINT64 key, DWORD) noexcept {
      ++acquire_calls;
      acquired_key = key;
      return acquire_result;
    }

    HRESULT ReleaseSync(const UINT64 key) noexcept {
      ++release_calls;
      released_key = key;
      return S_OK;
    }
  };

  TEST(WindowsKeyedMutexLockTest, ReleasesSuccessfulAcquisitionOnEarlyReturn) {
    fake_keyed_mutex_t mutex;
    const auto conversion_with_early_return = [&]() {
      platf::dxgi::detail::keyed_mutex_lock_t lock {&mutex};
      if (lock.lock(3, INFINITE, 7) != S_OK) {
        return true;
      }
      return false;
    };

    EXPECT_FALSE(conversion_with_early_return());
    EXPECT_EQ(mutex.acquire_calls, 1u);
    EXPECT_EQ(mutex.release_calls, 1u);
    EXPECT_EQ(mutex.acquired_key, 3u);
    EXPECT_EQ(mutex.released_key, 7u);
  }

  TEST(WindowsKeyedMutexLockTest, DoesNotReleaseFailedAcquisition) {
    fake_keyed_mutex_t mutex;
    mutex.acquire_result = WAIT_TIMEOUT;
    {
      platf::dxgi::detail::keyed_mutex_lock_t lock {&mutex};
      EXPECT_EQ(lock.lock(), WAIT_TIMEOUT);
      EXPECT_FALSE(lock.owns_lock());
    }

    EXPECT_EQ(mutex.acquire_calls, 1u);
    EXPECT_EQ(mutex.release_calls, 0u);
  }

  TEST(WindowsDdupTimestampTest, CursorOnlyUpdateRetainsDesktopContentTimestamp) {
    using timestamp_t = std::int64_t;
    const auto initial_present = platf::dxgi::detail::select_ddup_timestamps(
      std::optional<timestamp_t> {100},
      std::optional<timestamp_t> {},
      std::optional<timestamp_t> {}
    );
    ASSERT_EQ(initial_present.presentation_timestamp, 100);
    ASSERT_EQ(initial_present.content_timestamp, 100);

    const auto cursor_only = platf::dxgi::detail::select_ddup_timestamps(
      std::optional<timestamp_t> {},
      std::optional<timestamp_t> {200},
      initial_present.content_timestamp
    );
    EXPECT_EQ(cursor_only.presentation_timestamp, 200);
    EXPECT_EQ(cursor_only.content_timestamp, 100);
  }

  TEST(WindowsDdupTimestampTest, NewerCursorCadenceDoesNotReplacePresentContentTime) {
    using timestamp_t = std::int64_t;
    const auto timestamps = platf::dxgi::detail::select_ddup_timestamps(
      std::optional<timestamp_t> {300},
      std::optional<timestamp_t> {350},
      std::optional<timestamp_t> {100}
    );

    EXPECT_EQ(timestamps.presentation_timestamp, 350);
    EXPECT_EQ(timestamps.content_timestamp, 300);
  }

  TEST(WindowsHostSbsEncoderInputStateTest, FirstRepeatStillInitializesPersistentInput) {
    platf::dxgi::detail::host_sbs_encoder_input_state_t state;

    EXPECT_TRUE(state.conversion_required(true, false));
    state.mark_converted();
    EXPECT_FALSE(state.conversion_required(true, false));
    EXPECT_TRUE(state.conversion_required(false, false));
  }

  TEST(WindowsHostSbsEncoderInputStateTest, ResetInvalidatesPersistentInput) {
    platf::dxgi::detail::host_sbs_encoder_input_state_t state;
    state.mark_converted();
    ASSERT_TRUE(state.initialized());

    state.reset();

    EXPECT_FALSE(state.initialized());
    EXPECT_TRUE(state.conversion_required(true, false));
  }

  TEST(WindowsHostSbsEncoderInputStateTest, LocalRgbBackbufferIsNeverReused) {
    platf::dxgi::detail::host_sbs_encoder_input_state_t state;
    state.mark_converted();

    EXPECT_TRUE(state.conversion_required(true, true));
  }

  TEST(WindowsHostSbsRecoveryTest, StartupAndDomainResetStayAsynchronous) {
    EXPECT_FALSE(
      platf::dxgi::detail::host_sbs_accepted_frame_needs_synchronous_recovery(
        false,
        false
      )
    );
  }

  TEST(WindowsHostSbsRecoveryTest, TrulyStalePriorSourceUsesBoundedRecovery) {
    EXPECT_TRUE(
      platf::dxgi::detail::host_sbs_accepted_frame_needs_synchronous_recovery(
        true,
        false
      )
    );
    EXPECT_TRUE(
      platf::dxgi::detail::host_sbs_accepted_frame_needs_synchronous_recovery(
        false,
        true
      )
    );
  }

  TEST(WindowsHostSbsAuthorityTest, SamplingRequiresAnEstimatorConsumer) {
    EXPECT_FALSE(
      platf::dxgi::detail::host_sbs_window_authority_observation_needed(
        false,
        true
      )
    );
    EXPECT_FALSE(
      platf::dxgi::detail::host_sbs_window_authority_observation_needed(
        true,
        false
      )
    );
    EXPECT_TRUE(
      platf::dxgi::detail::host_sbs_window_authority_observation_needed(
        true,
        true
      )
    );
  }

  TEST(WindowsUploadedValueStateTest, CommitsOnlyExplicitlyAcceptedValue) {
    platf::dxgi::detail::uploaded_value_state_t<std::array<int, 3>> state;
    constexpr std::array first {1, 2, 3};
    constexpr std::array second {1, 2, 4};

    EXPECT_FALSE(state.is_current(first));
    state.commit(first);
    EXPECT_TRUE(state.is_current(first));
    EXPECT_FALSE(state.is_current(second));
    state.reset();
    EXPECT_FALSE(state.is_current(first));
  }

  struct fake_desktop_retry_operations_t {
    std::vector<bool> attempt_results;
    bool synchronization_result = true;
    std::size_t attempt_count = 0;
    std::size_t synchronization_count = 0;

    bool attempt() {
      const auto result = attempt_results.at(attempt_count);
      ++attempt_count;
      return result;
    }

    bool synchronize() {
      ++synchronization_count;
      return synchronization_result;
    }
  };

  struct fake_desktop_handle_operations_t {
    std::uintptr_t candidate = 2;
    bool set_succeeds = true;
    std::vector<std::string> calls;

    std::uintptr_t open_input_desktop() {
      calls.emplace_back("open:" + std::to_string(candidate));
      return candidate;
    }

    bool set_thread_desktop(std::uintptr_t desktop) {
      calls.emplace_back("set:" + std::to_string(desktop));
      return set_succeeds;
    }

    void close_desktop(std::uintptr_t desktop) {
      calls.emplace_back("close:" + std::to_string(desktop));
    }
  };

  TEST(WindowsQpcTimeTest, ConvertsPerformanceCounterTicksToNanoseconds) {
    LARGE_INTEGER frequency {};
    ASSERT_TRUE(QueryPerformanceFrequency(&frequency));
    ASSERT_GT(frequency.QuadPart, 0);

    EXPECT_NEAR(
      platf::qpc_time_difference(frequency.QuadPart, 0).count(),
      1'000'000'000LL,
      1LL
    );
    EXPECT_NEAR(
      platf::qpc_time_difference(0, frequency.QuadPart).count(),
      -1'000'000'000LL,
      1LL
    );
  }

  struct fake_token_job_launch_operations_t {
    bool assignment_succeeds = true;
    bool resume_succeeds = true;
    DWORD failure = ERROR_ACCESS_DENIED;
    std::vector<std::string> calls;

    bool assign_to_job() {
      calls.emplace_back("assign");
      return assignment_succeeds;
    }

    bool resume_initial_thread() {
      calls.emplace_back("resume");
      return resume_succeeds;
    }

    DWORD last_error() {
      calls.emplace_back("last-error");
      return failure;
    }

    void terminate_process(const DWORD exit_code) {
      calls.emplace_back("terminate:" + std::to_string(exit_code));
    }

    void close_initial_thread() {
      calls.emplace_back("close-thread");
    }

    void close_process() {
      calls.emplace_back("close-process");
    }
  };

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

  std::optional<std::string> token_sid(HANDLE token) {
    DWORD bytes = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &bytes);
    if (bytes == 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      return std::nullopt;
    }
    std::vector<std::byte> buffer(bytes);
    if (!GetTokenInformation(
          token,
          TokenUser,
          buffer.data(),
          static_cast<DWORD>(buffer.size()),
          &bytes
        )) {
      return std::nullopt;
    }
    const auto token_user = reinterpret_cast<const TOKEN_USER *>(buffer.data());
    LPWSTR sid_text = nullptr;
    if (!ConvertSidToStringSidW(token_user->User.Sid, &sid_text) || !sid_text) {
      return std::nullopt;
    }
    auto sid_text_guard = util::fail_guard([&]() {
      LocalFree(sid_text);
    });
    return platf::to_utf8(sid_text);
  }
}  // namespace

TEST(WindowsInputDesktopRetryTest, DoesNotSynchronizeAfterImmediateSuccess) {
  fake_desktop_retry_operations_t operations {{true}};

  EXPECT_TRUE(platf::detail::run_with_desktop_retry([&]() {
    return operations.attempt();
  },
                                                    [&]() {
                                                      return operations.synchronize();
                                                    }));
  EXPECT_EQ(operations.attempt_count, 1u);
  EXPECT_EQ(operations.synchronization_count, 0u);
}

TEST(WindowsInputDesktopRetryTest, RetriesExactlyOnceAfterSuccessfulSynchronization) {
  fake_desktop_retry_operations_t operations {{false, false}};

  EXPECT_FALSE(platf::detail::run_with_desktop_retry([&]() {
    return operations.attempt();
  },
                                                     [&]() {
                                                       return operations.synchronize();
                                                     }));
  EXPECT_EQ(operations.attempt_count, 2u);
  EXPECT_EQ(operations.synchronization_count, 1u);
}

TEST(WindowsInputDesktopRetryTest, DoesNotRetryWhenSynchronizationFails) {
  fake_desktop_retry_operations_t operations {{false}};
  operations.synchronization_result = false;

  EXPECT_FALSE(platf::detail::run_with_desktop_retry([&]() {
    return operations.attempt();
  },
                                                     [&]() {
                                                       return operations.synchronize();
                                                     }));
  EXPECT_EQ(operations.attempt_count, 1u);
  EXPECT_EQ(operations.synchronization_count, 1u);
}

TEST(WindowsInputDesktopRetryTest, RetainsSuccessfulDesktopAndClosesSupersededHandle) {
  std::uintptr_t active = 1;
  fake_desktop_handle_operations_t operations;

  EXPECT_TRUE(platf::detail::replace_thread_desktop_handle(active, operations));
  EXPECT_EQ(active, 2u);
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {"open:2", "set:2", "close:1"})
  );
}

TEST(WindowsInputDesktopRetryTest, ClosesFailedCandidateAndPreservesActiveHandle) {
  std::uintptr_t active = 1;
  fake_desktop_handle_operations_t operations;
  operations.set_succeeds = false;

  EXPECT_FALSE(platf::detail::replace_thread_desktop_handle(active, operations));
  EXPECT_EQ(active, 1u);
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {"open:2", "set:2", "close:2"})
  );
}

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

      TOKEN_STATISTICS outer_statistics {};
      ASSERT_TRUE(GetTokenInformation(
        thread_token,
        TokenStatistics,
        &outer_statistics,
        sizeof(outer_statistics),
        &returned
      ));
      bool nested_callback_called = false;
      const auto nested_result = platf::run_as_active_user(
        [&]() {
          nested_callback_called = true;
        },
        expected_user
      );
      EXPECT_FALSE(nested_result);
      EXPECT_TRUE(nested_callback_called);

      HANDLE restored_thread_token = nullptr;
      ASSERT_TRUE(OpenThreadToken(
        GetCurrentThread(),
        TOKEN_QUERY,
        TRUE,
        &restored_thread_token
      ));
      auto restored_thread_token_guard = util::fail_guard([&]() {
        CloseHandle(restored_thread_token);
      });
      TOKEN_STATISTICS restored_statistics {};
      ASSERT_TRUE(GetTokenInformation(
        restored_thread_token,
        TokenStatistics,
        &restored_statistics,
        sizeof(restored_statistics),
        &returned
      ));
      EXPECT_EQ(restored_statistics.TokenId.HighPart, outer_statistics.TokenId.HighPart);
      EXPECT_EQ(restored_statistics.TokenId.LowPart, outer_statistics.TokenId.LowPart);

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

TEST(RunCommandUnelevatedTest, ElevatedTrayLaunchesProductionShapeStandardUserChild) {
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
    GTEST_SKIP() << "The de-elevated launch regression requires an elevated tray process";
  }

  const auto expected_user = platf::active_user_id();
  if (!expected_user) {
    GTEST_SKIP() << "Windows did not expose a validated active desktop user";
  }

  std::vector<wchar_t> system_directory(MAX_PATH + 1, L'\0');
  const auto system_directory_size = GetSystemDirectoryW(
    system_directory.data(),
    static_cast<UINT>(system_directory.size())
  );
  ASSERT_GT(system_directory_size, 0u);
  ASSERT_LT(system_directory_size, system_directory.size());
  const std::filesystem::path ping =
    std::filesystem::path {system_directory.data()} / L"ping.exe";
  ASSERT_TRUE(std::filesystem::is_regular_file(ping));

  const auto command =
    std::string {"\""} + platf::to_utf8(ping.wstring()) +
    "\" 127.0.0.1 -n 6";
  boost::filesystem::path working_directory {
    platf::to_utf8(std::filesystem::path {system_directory.data()}.wstring())
  };
  auto environment = boost::this_process::environment();
  FILE *child_output = std::tmpfile();
  ASSERT_NE(child_output, nullptr);
  auto child_output_guard = util::fail_guard([&]() {
    std::fclose(child_output);
  });
  boost::process::v1::group child_group;
  std::error_code launch_error;
  auto child = platf::run_command_unelevated(
    false,
    command,
    working_directory,
    environment,
    child_output,
    launch_error,
    &child_group,
    expected_user
  );
  ASSERT_FALSE(launch_error) << launch_error.message();
  ASSERT_TRUE(child.valid());
  auto child_cleanup = util::fail_guard([&]() {
    if (child.valid() && child.running()) {
      std::error_code ignored;
      child.terminate(ignored);
    }
    if (child.valid()) {
      std::error_code ignored;
      child.wait(ignored);
    }
  });

  HANDLE child_process = OpenProcess(
    PROCESS_QUERY_LIMITED_INFORMATION,
    FALSE,
    static_cast<DWORD>(child.id())
  );
  ASSERT_NE(child_process, nullptr) << "Win32 error " << GetLastError();
  auto child_process_guard = util::fail_guard([&]() {
    CloseHandle(child_process);
  });
  HANDLE child_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(child_process, TOKEN_QUERY, &child_token))
    << "Win32 error " << GetLastError();
  auto child_token_guard = util::fail_guard([&]() {
    CloseHandle(child_token);
  });
  TOKEN_ELEVATION child_elevation {};
  ASSERT_TRUE(GetTokenInformation(
    child_token,
    TokenElevation,
    &child_elevation,
    sizeof(child_elevation),
    &returned
  ));
  EXPECT_EQ(child_elevation.TokenIsElevated, 0u);

  TOKEN_ELEVATION_TYPE child_elevation_type = TokenElevationTypeFull;
  ASSERT_TRUE(GetTokenInformation(
    child_token,
    TokenElevationType,
    &child_elevation_type,
    sizeof(child_elevation_type),
    &returned
  ));
  EXPECT_NE(child_elevation_type, TokenElevationTypeFull);

  const auto child_user = token_sid(child_token);
  ASSERT_TRUE(child_user);
  EXPECT_EQ(*child_user, *expected_user);

  DWORD child_session = 0xFFFFFFFF;
  ASSERT_TRUE(GetTokenInformation(
    child_token,
    TokenSessionId,
    &child_session,
    sizeof(child_session),
    &returned
  ));
  EXPECT_EQ(child_session, process_session);
  EXPECT_EQ(child_session, WTSGetActiveConsoleSessionId());

  BOOL child_is_in_job = FALSE;
  ASSERT_TRUE(IsProcessInJob(
    child_process,
    child_group.native_handle(),
    &child_is_in_job
  )) << "Win32 error "
     << GetLastError();
  EXPECT_TRUE(child_is_in_job);

  std::error_code wait_error;
  child.wait(wait_error);
  ASSERT_FALSE(wait_error) << wait_error.message();
  ASSERT_EQ(_fseeki64(child_output, 0, SEEK_END), 0);
  const auto output_size = _ftelli64(child_output);
  ASSERT_GE(output_size, 0);
  EXPECT_GT(output_size, 0) << "The child did not inherit the redirected output handle";
}

TEST(RunCommandUnelevatedTest, TokenLaunchAssignsJobBeforeResumingChild) {
  constexpr DWORD requested_flags =
    EXTENDED_STARTUPINFO_PRESENT |
    CREATE_UNICODE_ENVIRONMENT |
    CREATE_NO_WINDOW;

  constexpr auto isolated =
    platf::detail::create_process_with_token_plan(requested_flags, true);
  EXPECT_EQ(isolated.creation_flags & EXTENDED_STARTUPINFO_PRESENT, 0u);
  EXPECT_NE(isolated.creation_flags & CREATE_SUSPENDED, 0u);
  EXPECT_NE(isolated.creation_flags & CREATE_UNICODE_ENVIRONMENT, 0u);
  EXPECT_NE(isolated.creation_flags & CREATE_NO_WINDOW, 0u);
  EXPECT_EQ(isolated.startup_info_size, sizeof(STARTUPINFOW));
  EXPECT_TRUE(isolated.assign_job_after_create);

  constexpr auto ungrouped =
    platf::detail::create_process_with_token_plan(requested_flags, false);
  EXPECT_EQ(ungrouped.creation_flags & EXTENDED_STARTUPINFO_PRESENT, 0u);
  EXPECT_EQ(ungrouped.creation_flags & CREATE_SUSPENDED, 0u);
  EXPECT_EQ(ungrouped.startup_info_size, sizeof(STARTUPINFOW));
  EXPECT_FALSE(ungrouped.assign_job_after_create);
}

TEST(RunCommandUnelevatedTest, SuspendedJobLaunchAssignsBeforeResume) {
  fake_token_job_launch_operations_t operations;

  EXPECT_EQ(
    platf::detail::finish_suspended_job_launch(operations),
    ERROR_SUCCESS
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {"assign", "resume"})
  );
}

TEST(RunCommandUnelevatedTest, SuspendedJobLaunchCleansUpAssignmentFailure) {
  fake_token_job_launch_operations_t operations;
  operations.assignment_succeeds = false;
  operations.failure = ERROR_SUCCESS;

  EXPECT_EQ(
    platf::detail::finish_suspended_job_launch(operations),
    ERROR_PROCESS_ABORTED
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "assign",
      "last-error",
      "terminate:" + std::to_string(ERROR_PROCESS_ABORTED),
      "close-thread",
      "close-process",
    })
  );
}

TEST(RunCommandUnelevatedTest, SuspendedJobLaunchCleansUpResumeFailure) {
  fake_token_job_launch_operations_t operations;
  operations.resume_succeeds = false;
  operations.failure = ERROR_ACCESS_DENIED;

  EXPECT_EQ(
    platf::detail::finish_suspended_job_launch(operations),
    ERROR_ACCESS_DENIED
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "assign",
      "resume",
      "last-error",
      "terminate:" + std::to_string(ERROR_ACCESS_DENIED),
      "close-thread",
      "close-process",
    })
  );
}
#endif
