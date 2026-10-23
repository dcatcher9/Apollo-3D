// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_detection.h"

#include <array>
#include <gtest/gtest.h>

namespace {
  using platf::game3d::detail::is_game_candidate;
  using platf::game3d::detected_game;
  using platf::game3d::detection_confirmation;
  namespace foreground = platf::foreground_window;
  using namespace std::chrono_literals;
  const RECT capture {0, 0, 1920, 1080};
  const auto start = std::chrono::steady_clock::time_point {100s};

  foreground::observation_t game_window() {
    return {
      .status = foreground::status_e::ok,
      .window = 1,
      .process_id = 42,
      .client_screen_rect = {0, 0, 1920, 1080},
    };
  }

  foreground::observation_t windowed_game_window() {
    auto window = game_window();
    window.client_screen_rect = {100, 100, 740, 580};
    return window;
  }

  detected_game generic_game() {
    return {
      .executable = LR"(D:\Games\Example\Example.exe)",
      .process_id = 42,
      .creation_time = 1000,
      .name = "Example",
      .window = 1,
    };
  }

  TEST(Game3dDetection, OneGeneralRuleWithoutStoreLibrariesOrEngineLayouts) {
    const std::array<std::filesystem::path, 1> graphics11 {L"d3d11.dll"};
    const std::array<std::filesystem::path, 1> graphics12 {L"D3D12.DLL"};
    const std::filesystem::path arbitrary = LR"(C:\Applications\Unknown Engine\Arbitrary.exe)";
    EXPECT_TRUE(is_game_candidate(arbitrary, graphics11, game_window(), capture));
    EXPECT_TRUE(is_game_candidate(arbitrary, graphics12, game_window(), capture));
    // Store SDKs, engine runtimes and the Unreal Shipping layout neither admit a window nor
    // replace Direct3D: every executable meets the same coverage and graphics rule.
    for (const auto sdk : {L"steam_api64.dll", L"EOSSDK-Win64-Shipping.dll", L"Galaxy64.dll", L"UnityPlayer.dll", L"GameAssembly.dll", L"uplay_r1_loader64.dll", L"upc_r1_loader64.dll", L"upc_r2_loader64.dll", L"uplay_r2_loader64.dll"}) {
      SCOPED_TRACE(std::filesystem::path {sdk}.string());
      const std::array<std::filesystem::path, 2> with_sdk {L"d3d12.dll", sdk};
      const std::array<std::filesystem::path, 1> sdk_only {sdk};
      EXPECT_FALSE(is_game_candidate(LR"(D:\Games\Example\Example.exe)", with_sdk, windowed_game_window(), capture));
      EXPECT_TRUE(is_game_candidate(LR"(D:\Games\Example\Example.exe)", with_sdk, game_window(), capture));
      EXPECT_FALSE(is_game_candidate(LR"(D:\Games\Example\Example.exe)", sdk_only, game_window(), capture));
    }
    EXPECT_FALSE(is_game_candidate(LR"(D:\Games\Example\Binaries\Win64\Example-Win64-Shipping.exe)", graphics12, windowed_game_window(), capture));
    EXPECT_TRUE(is_game_candidate(LR"(D:\Games\Example\Binaries\Win64\Example-Win64-Shipping.exe)", graphics12, game_window(), capture));
    // An unknown fullscreen Direct3D application qualifies as well; setup only offers it.
    EXPECT_TRUE(is_game_candidate(LR"(C:\Program Files\Media Center\MediaCenter.exe)", graphics11, game_window(), capture));
  }

  TEST(Game3dDetection, RequiresExactClientCoverageOfSelectedDisplay) {
    const std::filesystem::path executable = LR"(D:\Games\Example\Example.exe)";
    const std::array<std::filesystem::path, 1> graphics {L"d3d12.dll"};
    auto window = game_window();
    for (const auto client : {
           foreground::rect_t {100, 100, 740, 580},
           foreground::rect_t {-100, 100, 540, 580},
           foreground::rect_t {0, 0, 200, 100},
           foreground::rect_t {0, 30, 1920, 1040},
           foreground::rect_t {0, 0, 1920, 1079},
           foreground::rect_t {-1, 0, 1920, 1080},
           foreground::rect_t {-640, 0, 1920, 1080},
           foreground::rect_t {1920, 0, 3840, 1080},
         }) {
      window.client_screen_rect = client;
      window.frame_screen_rect = {0, 0, 1920, 1080};
      EXPECT_FALSE(is_game_candidate(executable, graphics, window, capture));
    }
    window.client_screen_rect = {-1920, -1080, 0, 0};
    EXPECT_TRUE(is_game_candidate(executable, graphics, window, RECT {-1920, -1080, 0, 0}));
    EXPECT_FALSE(is_game_candidate(executable, graphics, window, capture));
    EXPECT_FALSE(is_game_candidate(executable, graphics, game_window(), RECT {0, 0, 0, 1080}));
  }

  TEST(Game3dDetection, ExcludesBrowserLauncherPlayerOfficeEditorAndShellWindows) {
    const std::array<std::filesystem::path, 1> graphics {L"d3d11.dll"};
    for (const auto name : {L"CHROME.EXE", L"msedge.exe", L"msedgewebview2.exe", L"steam.exe", L"EpicGamesLauncher.exe", L"explorer.exe", L"Unity.exe", L"UnrealEditor.exe", L"sunshine.exe", L"kodi.exe", L"mpc-hc64.exe", L"PotPlayerMini64.exe", L"POWERPNT.EXE", L"Code.exe", L"vrcompositor.exe"}) {
      SCOPED_TRACE(std::filesystem::path {name}.string());
      EXPECT_FALSE(is_game_candidate(std::filesystem::path {LR"(C:\Applications)"} / name, graphics, game_window(), capture));
    }
  }

  TEST(Game3dDetection, RejectsInvalidForegroundAndExecutableEvidence) {
    const std::array<std::filesystem::path, 1> modules {L"d3d11.dll"};
    const std::filesystem::path executable = LR"(D:\Games\Example\Example.exe)";
    auto window = game_window();
    for (const auto status : {foreground::status_e::no_foreground, foreground::status_e::shell_surface, foreground::status_e::foreground_changed, foreground::status_e::cloaked, foreground::status_e::minimized, foreground::status_e::interactive_move_size}) {
      window.status = status;
      EXPECT_FALSE(is_game_candidate(executable, modules, window, capture));
    }
    window = game_window();
    window.process_id = 0;
    EXPECT_FALSE(is_game_candidate(executable, modules, window, capture));
    window = game_window();
    window.window = 0;
    EXPECT_FALSE(is_game_candidate(executable, modules, window, capture));
    EXPECT_FALSE(is_game_candidate(L"Example.exe", modules, game_window(), capture));
    EXPECT_FALSE(is_game_candidate(LR"(D:\Games\Example\Example.dll)", modules, game_window(), capture));
  }

  TEST(Game3dDetectionConfirmation, EveryCandidateNeedsTwoSecondsOfSustainedEvidence) {
    detection_confirmation confirmation;
    const auto candidate = generic_game();
    EXPECT_FALSE(confirmation.observe(candidate, capture, start));
    EXPECT_FALSE(confirmation.observe(candidate, capture, start + 1999ms));
    const auto confirmed = confirmation.observe(candidate, capture, start + 2s);
    ASSERT_TRUE(confirmed);
    EXPECT_EQ(confirmed->executable, candidate.executable);
    EXPECT_TRUE(confirmation.observe(candidate, capture, start + 4s));
  }

  TEST(Game3dDetectionConfirmation, ProcessWindowExecutableAndCaptureChangesRestartConfirmation) {
    for (int change = 0; change < 5; ++change) {
      SCOPED_TRACE(change);
      detection_confirmation confirmation;
      auto candidate = generic_game();
      auto selected_capture = capture;
      EXPECT_FALSE(confirmation.observe(candidate, capture, start));
      switch (change) {
        case 0:
          ++candidate.process_id;
          break;
        case 1:
          ++candidate.creation_time;
          break;
        case 2:
          candidate.executable = LR"(D:\Games\Example\SP\Example.exe)";
          break;
        case 3:
          ++candidate.window;
          break;
        case 4:
          selected_capture = {-1920, 0, 0, 1080};
          break;
      }
      EXPECT_FALSE(confirmation.observe(candidate, selected_capture, start + 2s));
      EXPECT_TRUE(confirmation.observe(candidate, selected_capture, start + 4s));
    }
  }

  TEST(Game3dDetectionConfirmation, MissingObservationAndExplicitResetBreakContinuity) {
    detection_confirmation confirmation;
    const auto candidate = generic_game();
    EXPECT_FALSE(confirmation.observe(candidate, capture, start));
    EXPECT_FALSE(confirmation.observe(std::nullopt, capture, start + 2s));
    EXPECT_FALSE(confirmation.observe(candidate, capture, start + 4s));
    EXPECT_TRUE(confirmation.observe(candidate, capture, start + 6s));
    confirmation.reset();
    EXPECT_FALSE(confirmation.observe(candidate, capture, start + 8s));
    EXPECT_TRUE(confirmation.observe(candidate, capture, start + 10s));
  }

  TEST(Game3dDetectionConfirmation, StaleOrReversedPollTimesCannotConfirmPastEvidence) {
    detection_confirmation confirmation;
    const auto candidate = generic_game();
    EXPECT_FALSE(confirmation.observe(candidate, capture, start));
    EXPECT_FALSE(confirmation.observe(candidate, capture, start + 6s));
    EXPECT_TRUE(confirmation.observe(candidate, capture, start + 8s));
    EXPECT_FALSE(confirmation.observe(candidate, capture, start + 7s));
    EXPECT_TRUE(confirmation.observe(candidate, capture, start + 9s));
  }

  TEST(Game3dDetectionConfirmation, EvidenceNeedsProcessAndDisplayIdentity) {
    for (int missing = 0; missing < 4; ++missing) {
      SCOPED_TRACE(missing);
      detection_confirmation confirmation;
      auto candidate = generic_game();
      auto selected_capture = capture;
      switch (missing) {
        case 0:
          candidate.process_id = 0;
          break;
        case 1:
          candidate.creation_time = 0;
          break;
        case 2:
          candidate.window = 0;
          break;
        case 3:
          selected_capture.right = selected_capture.left;
          break;
      }
      EXPECT_FALSE(confirmation.observe(candidate, selected_capture, start));
      EXPECT_FALSE(confirmation.observe(candidate, selected_capture, start + 2s));
    }
  }
}  // namespace
