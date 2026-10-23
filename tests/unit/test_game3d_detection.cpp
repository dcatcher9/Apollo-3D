// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_detection.h"

#include <array>
#include <gtest/gtest.h>

namespace {
  using platf::game3d::detail::classify_game_candidate;
  using platf::game3d::detail::is_game_candidate;
  using platf::game3d::detected_game;
  using platf::game3d::detection_confirmation;
  using platf::game3d::detection_evidence_e;
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
      .evidence = detection_evidence_e::fullscreen_direct3d,
      .window = 1,
    };
  }

  TEST(Game3dDetection, WindowedGamesRequireGraphicsAndIndependentGameEvidence) {
    const std::filesystem::path executable = LR"(D:\Games\Example\Example.exe)";
    const std::array<std::filesystem::path, 1> graphics {L"d3d12.dll"};
    const std::array<std::filesystem::path, 1> runtime {L"steam_api64.dll"};
    const std::array<std::filesystem::path, 2> both {L"d3d12.dll", L"steam_api64.dll"};
    EXPECT_FALSE(is_game_candidate(executable, graphics, windowed_game_window(), capture));
    EXPECT_FALSE(is_game_candidate(executable, runtime, windowed_game_window(), capture));
    EXPECT_EQ(classify_game_candidate(executable, both, windowed_game_window(), capture), detection_evidence_e::known_runtime);
    EXPECT_FALSE(is_game_candidate(executable, runtime, game_window(), capture));
  }

  TEST(Game3dDetection, FullscreenDirect3dNeedsNoKnownEngineStorefrontOrExecutableName) {
    const std::array<std::filesystem::path, 1> graphics11 {L"d3d11.dll"};
    const std::array<std::filesystem::path, 1> graphics12 {L"D3D12.DLL"};
    const std::array<std::filesystem::path, 2> newer_sdk {L"d3d12.dll", L"upc_r2_loader64.dll"};
    const std::filesystem::path arbitrary = LR"(C:\Applications\Unknown Engine\Arbitrary.exe)";
    EXPECT_EQ(classify_game_candidate(arbitrary, graphics11, game_window(), capture), detection_evidence_e::fullscreen_direct3d);
    EXPECT_EQ(classify_game_candidate(arbitrary, graphics12, game_window(), capture), detection_evidence_e::fullscreen_direct3d);
    EXPECT_EQ(classify_game_candidate(LR"(D:\Games\Assassins Creed Shadows\ACShadows.exe)", newer_sdk, game_window(), capture), detection_evidence_e::fullscreen_direct3d);
  }

  TEST(Game3dDetection, GenericRouteRequiresExactClientCoverageOfSelectedDisplay) {
    const std::filesystem::path executable = LR"(D:\Games\Example\Example.exe)";
    const std::array<std::filesystem::path, 1> graphics {L"d3d12.dll"};
    auto window = game_window();
    for (const auto client : {
           foreground::rect_t {100, 100, 740, 580},
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
    EXPECT_EQ(classify_game_candidate(executable, graphics, window, RECT {-1920, -1080, 0, 0}), detection_evidence_e::fullscreen_direct3d);
    EXPECT_FALSE(is_game_candidate(executable, graphics, window, capture));
  }

  TEST(Game3dDetection, RecognizesUnityAndUnrealWithoutDistributionSdk) {
    const std::array<std::filesystem::path, 2> unity {L"D3D11.DLL", L"UnityPlayer.dll"};
    const std::array<std::filesystem::path, 1> graphics {L"d3d12.dll"};
    EXPECT_EQ(classify_game_candidate(LR"(D:\Games\Example\Example.exe)", unity, windowed_game_window(), capture), detection_evidence_e::known_runtime);
    EXPECT_EQ(classify_game_candidate(LR"(D:\Games\Example\Binaries\Win64\Example-Win64-Shipping.exe)", graphics, windowed_game_window(), capture), detection_evidence_e::known_runtime);
    EXPECT_FALSE(is_game_candidate(LR"(D:\Games\Example\Example-Win64-Shipping.exe)", graphics, windowed_game_window(), capture));
    EXPECT_FALSE(is_game_candidate(LR"(D:\Games\Example\Binaries\Win64\Example.exe)", graphics, windowed_game_window(), capture));
  }

  TEST(Game3dDetection, ExcludesBrowserLauncherSystemAndEditorWindows) {
    const std::array<std::filesystem::path, 2> modules {L"d3d11.dll", L"steam_api64.dll"};
    const std::array<std::filesystem::path, 1> graphics {L"d3d12.dll"};
    for (const auto name : {L"CHROME.EXE", L"msedge.exe", L"steam.exe", L"EpicGamesLauncher.exe", L"explorer.exe", L"Unity.exe", L"UnrealEditor.exe", L"sunshine.exe"}) {
      EXPECT_FALSE(is_game_candidate(std::filesystem::path {LR"(C:\Applications)"} / name, modules, game_window(), capture));
      EXPECT_FALSE(is_game_candidate(std::filesystem::path {LR"(C:\Applications)"} / name, graphics, game_window(), capture));
    }
  }

  TEST(Game3dDetection, AllowsWindowedGamesAndRequiresCapturedDisplayOwnership) {
    const std::array<std::filesystem::path, 2> modules {L"d3d12.dll", L"GameAssembly.dll"};
    const std::filesystem::path executable = LR"(D:\Games\Example\Example.exe)";
    auto window = game_window();
    window.client_screen_rect = {100, 100, 740, 580};
    EXPECT_TRUE(is_game_candidate(executable, modules, window, capture));
    window.client_screen_rect = {1600, 100, 2240, 580};
    EXPECT_FALSE(is_game_candidate(executable, modules, window, capture));
    window.client_screen_rect = {-100, 100, 540, 580};
    EXPECT_TRUE(is_game_candidate(executable, modules, window, capture));
    window.client_screen_rect = {0, 0, 200, 100};
    EXPECT_FALSE(is_game_candidate(executable, modules, window, capture));
    EXPECT_FALSE(is_game_candidate(executable, modules, game_window(), RECT {0, 0, 0, 1080}));
  }

  TEST(Game3dDetection, RejectsInvalidForegroundAndExecutableEvidence) {
    const std::array<std::filesystem::path, 2> modules {L"d3d11.dll", L"Galaxy64.dll"};
    const std::filesystem::path executable = LR"(D:\Games\Example\Example.exe)";
    auto window = game_window();
    for (const auto status : {foreground::status_e::no_foreground, foreground::status_e::shell_surface, foreground::status_e::foreground_changed, foreground::status_e::cloaked, foreground::status_e::minimized, foreground::status_e::interactive_move_size}) {
      window.status = status;
      EXPECT_FALSE(is_game_candidate(executable, modules, window, capture));
    }
    window = game_window();
    window.process_id = 0;
    EXPECT_FALSE(is_game_candidate(executable, modules, window, capture));
    EXPECT_FALSE(is_game_candidate(L"Example.exe", modules, game_window(), capture));
    EXPECT_FALSE(is_game_candidate(LR"(D:\Games\Example\Example.dll)", modules, game_window(), capture));
  }

  TEST(Game3dDetectionConfirmation, RequiresSustainedFullscreenEvidenceAcrossPolls) {
    detection_confirmation confirmation;
    const auto candidate = generic_game();
    EXPECT_FALSE(confirmation.observe(candidate, capture, start));
    EXPECT_FALSE(confirmation.observe(candidate, capture, start + 1999ms));
    const auto confirmed = confirmation.observe(candidate, capture, start + 2s);
    ASSERT_TRUE(confirmed);
    EXPECT_EQ(confirmed->executable, candidate.executable);
    EXPECT_EQ(confirmed->evidence, detection_evidence_e::fullscreen_direct3d);
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

  TEST(Game3dDetectionConfirmation, KnownRuntimeIsImmediateAndClearsProvisionalGenericEvidence) {
    detection_confirmation confirmation;
    auto candidate = generic_game();
    EXPECT_FALSE(confirmation.observe(candidate, capture, start));
    auto known_game = candidate;
    known_game.evidence = detection_evidence_e::known_runtime;
    EXPECT_TRUE(confirmation.observe(known_game, capture, start + 1s));
    EXPECT_FALSE(confirmation.observe(candidate, capture, start + 2s));
    EXPECT_TRUE(confirmation.observe(candidate, capture, start + 4s));
  }

  TEST(Game3dDetectionConfirmation, GenericEvidenceNeedsProcessAndDisplayIdentity) {
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
