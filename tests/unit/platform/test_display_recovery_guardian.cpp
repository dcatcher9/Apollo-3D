/** Tests use a suspended fixture process and injected recovery; real displays are untouched. */
#include "../../tests_common.h"
#include "src/platform/windows/display_recovery_guardian.h"
#include "src/utility.h"

#include <atomic>
#include <future>
#include <shellapi.h>

using namespace std::chrono_literals;

TEST(DisplayRecoveryGuardianTest, CommandArgumentsRoundTripWithoutShellInterpretation) {
  for (const std::wstring value : {L"", L"E:\\Apollo Dev\\sunshine.conf", L"E:\\Unicode α\\", L"a\\\"b"}) {
    const auto command = L"guardian.exe " + platf::display_recovery_guardian::detail::quote_argument(value);
    int count = 0;
    const auto arguments = CommandLineToArgvW(command.c_str(), &count);
    ASSERT_NE(arguments, nullptr);
    const auto free_arguments = util::fail_guard([&]() {
      LocalFree(arguments);
    });
    ASSERT_EQ(count, 2);
    EXPECT_EQ(std::wstring(arguments[1]), value);
  }
  const auto handle = platf::display_recovery_guardian::detail::parse_handle(L"123456");
  ASSERT_TRUE(handle);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*handle), 123456u);
  for (const auto invalid : {L"", L"0", L"-1", L"12x", L" 12", L"18446744073709551616", L"18446744073709551615"}) {
    EXPECT_FALSE(platf::display_recovery_guardian::detail::parse_handle(invalid));
  }
}

TEST(DisplayRecoveryGuardianTest, RejectsNonProcessHandlesWithoutReadinessOrRecovery) {
  const auto event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  ASSERT_NE(event, nullptr);
  const auto close_event = util::fail_guard([&]() {
    CloseHandle(event);
  });
  bool recovered = false;
  const auto status = platf::display_recovery_guardian::detail::watch_parent(event, event, [&]() {
    recovered = true;
    return true;
  });
  EXPECT_EQ(status, ERROR_INVALID_PARAMETER);
  EXPECT_EQ(WaitForSingleObject(event, 0), WAIT_TIMEOUT);
  EXPECT_FALSE(recovered);
}

TEST(DisplayRecoveryGuardianTest, ReadyWorkerNeverRecoversUntilActualParentExitsThenRetries) {
  std::wstring executable(32768, L'\0');
  const auto length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  ASSERT_GT(length, 0u);
  ASSERT_LT(length, executable.size());
  executable.resize(length);
  auto command = platf::display_recovery_guardian::detail::quote_argument(executable);
  STARTUPINFOW startup {};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION fixture {};
  ASSERT_TRUE(CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr, nullptr, &startup, &fixture));
  const auto close_fixture = util::fail_guard([&]() {
    TerminateProcess(fixture.hProcess, ERROR_PROCESS_ABORTED);
    WaitForSingleObject(fixture.hProcess, 2000);
    CloseHandle(fixture.hThread);
    CloseHandle(fixture.hProcess);
  });
  const auto ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  ASSERT_NE(ready, nullptr);
  const auto close_ready = util::fail_guard([&]() {
    CloseHandle(ready);
  });
  std::atomic_uint recovery_calls = 0;
  auto watcher = std::async(std::launch::async, [&]() {
    return platf::display_recovery_guardian::detail::watch_parent(fixture.hProcess, ready, [&]() {
      return ++recovery_calls >= 3;
    },
                                                                  1);
  });
  // No fatal assertions until the suspended fixture has been terminated: std::future must
  // never block test unwinding while the process it observes remains alive.
  EXPECT_EQ(WaitForSingleObject(ready, 2000), WAIT_OBJECT_0);
  EXPECT_EQ(watcher.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(recovery_calls.load(), 0u);
  EXPECT_TRUE(TerminateProcess(fixture.hProcess, ERROR_PROCESS_ABORTED));
  EXPECT_EQ(watcher.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(watcher.get(), ERROR_SUCCESS);
  EXPECT_EQ(recovery_calls.load(), 3u);
}
