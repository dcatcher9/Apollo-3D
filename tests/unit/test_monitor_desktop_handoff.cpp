
#include "src/platform/windows/monitor_desktop_handoff.h"

#include <gtest/gtest.h>
#include <limits>

namespace {
  using namespace platf::monitor_desktop_detail;

  TEST(MonitorDesktopHandoff, KnownLockScreenClassIsExcludedEvenWhenItsProcessCannotBeQueried) {
    EXPECT_TRUE(is_security_surface(L"LockScreenInputOcclusionFrame", L"explorer.exe"));
    EXPECT_TRUE(is_security_surface(L"lockscreeninputocclusionframe", L"EXPLORER.EXE"));
    EXPECT_TRUE(is_security_surface(L"LockScreenInputOcclusionFrame", {}));
  }

  TEST(MonitorDesktopHandoff, LockAppOwnerIsExcludedForAnyWindowClass) {
    EXPECT_TRUE(is_security_surface(L"Windows.UI.Core.CoreWindow", L"LockApp.exe"));
    EXPECT_TRUE(is_security_surface(L"ApplicationFrameWindow", L"LOCKAPP.EXE"));
    EXPECT_TRUE(is_security_surface(L"Credential Dialog Xaml Host", LR"(C:\Windows\System32\consent.exe)"));
    EXPECT_TRUE(is_security_surface(L"Windows.UI.Core.CoreWindow", L"CREDENTIALUIBROKER.EXE"));
    EXPECT_TRUE(is_security_surface(L"", L"C:\\Windows\\SystemApps\\LockApp_cw5n1h2txyewy\\LockApp.exe"));
  }

  TEST(MonitorDesktopHandoff, OrdinaryExplorerUwpAndGamesAreNotBlanketExcluded) {
    EXPECT_FALSE(is_security_surface(L"CabinetWClass", L"explorer.exe"));
    EXPECT_FALSE(is_security_surface(L"Windows.UI.Core.CoreWindow", L"CalculatorApp.exe"));
    EXPECT_FALSE(is_security_surface(L"ApplicationFrameWindow", L"ApplicationFrameHost.exe"));
    EXPECT_FALSE(is_security_surface(L"UnrealWindow", L"ExampleGame.exe"));
    EXPECT_FALSE(is_security_surface(L"ApplicationFrameWindow", L"explorer.exe"));
    EXPECT_FALSE(is_security_surface(L"UnrealWindow", {})); // An unreadable process name is not a security identity.
    EXPECT_FALSE(is_security_surface(L"", L"MyLockApp.exe"));
    EXPECT_FALSE(is_security_surface(L"LockScreenInputOcclusionFrameHelper", L"explorer.exe"));
  }

  void expect_rect(const std::optional<RECT> &actual, RECT expected) {
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->left, expected.left);
    EXPECT_EQ(actual->top, expected.top);
    EXPECT_EQ(actual->right, expected.right);
    EXPECT_EQ(actual->bottom, expected.bottom);
  }

  TEST(MonitorDesktopHandoff, RoundTripRetainsWindowPlacementAcrossNegativeDesktopOrigins) {
    const RECT monitor {-2560, -200, 0, 1240};
    const RECT source {1920, 0, 4480, 1440};
    const RECT window {-2500, 100, -1500, 800};
    const auto moved = map_rect(window, monitor, source);
    expect_rect(moved, {1980, 300, 2980, 1000});
    ASSERT_TRUE(moved);
    expect_rect(map_rect(*moved, source, monitor), window);
  }

  TEST(MonitorDesktopHandoff, ClampsWindowsAndCursorWhenRestoredMonitorIsSmaller) {
    const RECT from {1920, 0, 5760, 2160};
    const RECT to {0, 0, 1920, 1080};
    expect_rect(map_rect({2000, 0, 5500, 2160}, from, to), to);
    expect_rect(map_rect({5500, 1800, 5750, 2100}, from, to), {1670, 780, 1920, 1080});
    const auto cursor = map_point({5759, 2159}, from, to);
    ASSERT_TRUE(cursor);
    EXPECT_EQ(cursor->x, 1919);
    EXPECT_EQ(cursor->y, 1079);
  }

  TEST(MonitorDesktopHandoff, ReplacingTheSameLogicalMonitorPreservesSpanningWindows) {
    const RECT selected {0, 0, 2560, 1440};
    const RECT spanning {-400, 100, 2900, 1100};
    expect_rect(map_rect(spanning, selected, selected), spanning);
  }

  TEST(MonitorDesktopHandoff, UsesHalfOpenOwnershipAndWideArithmeticForDesktopCoordinates) {
    const RECT monitor {-1920, 0, 0, 1080};
    EXPECT_TRUE(contains(monitor, {-1920, 0}));
    EXPECT_FALSE(contains(monitor, {0, 0}));
    EXPECT_TRUE(contains_center(monitor, {-2000, 0, -1000, 500}));
    EXPECT_FALSE(contains_center(monitor, {-200, 0, 200, 500}));
    const LONG maximum = std::numeric_limits<LONG>::max();
    EXPECT_TRUE(contains_center({maximum - 1000, maximum - 1000, maximum, maximum}, {maximum - 900, maximum - 900, maximum - 100, maximum - 100}));
  }

  TEST(MonitorDesktopHandoff, MinimizedWindowBelongsToItsNormalPlacementInsteadOfShellIconCoordinates) {
    const RECT sink {6000, -200, 8560, 1240};
    const RECT source {0, 0, 2560, 1440};
    const RECT icon {-32000, -32000, -31840, -31972};
    const RECT normal {6200, 100, 7000, 800};
    const auto minimized = window_ownership_rect(icon, normal, true);
    expect_rect(minimized, normal);
    ASSERT_TRUE(minimized);
    EXPECT_TRUE(contains_center(sink, *minimized));
    EXPECT_FALSE(contains_center(source, *minimized));
    expect_rect(window_ownership_rect({100, 100, 900, 700}, normal, false), {100, 100, 900, 700});
    EXPECT_FALSE(window_ownership_rect(icon, {0, 0, 0, 0}, true));
  }

  TEST(MonitorDesktopHandoff, MinimizedMoveRetainsRestoreToMaximizedAndExplicitIconPlacement) {
    WINDOWPLACEMENT original {sizeof(WINDOWPLACEMENT)};
    original.flags = WPF_RESTORETOMAXIMIZED | WPF_SETMINPOSITION;
    original.showCmd = SW_SHOWMINIMIZED;
    original.ptMinPosition = {4, 5};
    original.ptMaxPosition = {0, 0};
    original.rcNormalPosition = {100, 140, 900, 740};
    const auto moved = placement_for_move(original, {6100, 240, 6900, 840}, true);
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->showCmd, SW_SHOWMINIMIZED);
    EXPECT_EQ(moved->flags, WPF_RESTORETOMAXIMIZED | WPF_SETMINPOSITION | WPF_ASYNCWINDOWPLACEMENT);
    EXPECT_EQ(moved->ptMinPosition.x, 6004);
    EXPECT_EQ(moved->ptMinPosition.y, 105);
    EXPECT_EQ(moved->ptMaxPosition.x, 6000);
    EXPECT_EQ(moved->ptMaxPosition.y, 100);
    expect_rect(moved->rcNormalPosition, {6100, 240, 6900, 840});
    const auto restored = placement_for_move(*moved, original.rcNormalPosition, true);
    ASSERT_TRUE(restored);
    EXPECT_EQ(restored->showCmd, SW_SHOWMINIMIZED);
    EXPECT_TRUE(restored->flags & WPF_RESTORETOMAXIMIZED);
    EXPECT_EQ(restored->ptMinPosition.x, original.ptMinPosition.x);
    EXPECT_EQ(restored->ptMinPosition.y, original.ptMinPosition.y);
    EXPECT_EQ(restored->ptMaxPosition.x, original.ptMaxPosition.x);
    EXPECT_EQ(restored->ptMaxPosition.y, original.ptMaxPosition.y);
  }

  TEST(MonitorDesktopHandoff, PlacementPolicyKeepsMaximizedAndHiddenStatesWithoutActivatingNormalWindows) {
    WINDOWPLACEMENT original {sizeof(WINDOWPLACEMENT)};
    original.rcNormalPosition = {100, 100, 900, 700};
    original.ptMaxPosition = {-1, -1};
    original.showCmd = SW_SHOWMAXIMIZED;
    const auto maximized = placement_for_move(original, {2100, 100, 2900, 700}, false);
    ASSERT_TRUE(maximized);
    EXPECT_EQ(maximized->showCmd, SW_SHOWMAXIMIZED);
    EXPECT_EQ(maximized->ptMaxPosition.x, -1);
    EXPECT_EQ(maximized->ptMaxPosition.y, -1);
    original.showCmd = SW_SHOWNORMAL;
    const auto normal = placement_for_move(original, {2100, 100, 2900, 700}, false);
    ASSERT_TRUE(normal);
    EXPECT_EQ(normal->showCmd, SW_SHOWNOACTIVATE);
    const auto hidden = placement_for_move(original, {2100, 100, 2900, 700}, false, false);
    ASSERT_TRUE(hidden);
    EXPECT_EQ(hidden->showCmd, SW_HIDE);
    EXPECT_FALSE(placement_for_move(original, {0, 0, 0, 100}, true));
  }

  TEST(MonitorDesktopHandoff, PendingSinkMigrationCanRestoreBeforeItsAsyncMoveAcknowledges) {
    const RECT original {-2560, -200, 0, 1240};
    const RECT parked_sink {6000, 0, 8560, 1440};
    const RECT source = original;
    const RECT still_on_sink {6200, 300, 7000, 900};
    const auto frozen_in_source = map_rect(still_on_sink, parked_sink, source);
    expect_rect(frozen_in_source, {-2360, 100, -1560, 700});
    ASSERT_TRUE(frozen_in_source);
    expect_rect(map_rect(*frozen_in_source, source, original), {-2360, 100, -1560, 700});
  }

  TEST(MonitorDesktopHandoff, RejectsInvalidGeometryBeforeAnyDesktopOperation) {
    EXPECT_FALSE(map_rect({0, 0, 0, 1}, {0, 0, 100, 100}, {0, 0, 100, 100}));
    EXPECT_FALSE(map_rect({0, 0, 20, 20}, {0, 0, 0, 100}, {0, 0, 100, 100}));
    EXPECT_FALSE(map_point({1, 1}, {0, 0, 100, 100}, {10, 10, 0, 0}));
    platf::monitor_desktop_handoff_t handoff;
    EXPECT_FALSE(handoff.capture({0, 0, 0, 100}));
    EXPECT_FALSE(handoff.activate({0, 0, 100, 100}));
    EXPECT_FALSE(handoff.refresh({200, 0, 300, 100}, {0, 0, 100, 100}));
    EXPECT_TRUE(handoff.restore({0, 0, 100, 100}));
  }
}  // namespace
