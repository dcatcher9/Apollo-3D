// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_setup.h"

#include <algorithm>
#include <atomic>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <Windows.h>

namespace {
  namespace fs = std::filesystem;
  namespace game3d = platf::game3d;
  using decision = game3d::detail::setup_decision_e;
  using state = game3d::installation_state_e;

  void write(const fs::path &path, const std::string &value) {
    std::ofstream(path, std::ios::binary) << value;
  }

  std::string read(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  }

  class Game3DSetup: public testing::Test {
  protected:
    fs::path root;
    fs::path game;
    fs::path bundle;
    fs::path choices;
    std::vector<std::string> progress;

    void SetUp() override {
      static std::atomic<unsigned> serial {};
      root = fs::temp_directory_path() / ("sunshine-game3d-setup-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(serial++));
      // Four parents inside the fixture: the anti-cheat walk never reaches the real temp folder.
      game = root / "drive" / "library" / "common" / "Media Center" / "MediaCenter.exe";
      bundle = root / "host package";
      choices = root / "config" / "game3d_setup.json";
      fs::create_directories(game.parent_path());
      fs::create_directories(bundle);
      fs::create_directories(choices.parent_path());
      write(game, "unknown fullscreen Direct3D application");
      write(bundle / "SunshineSBS.addon64", "matching bundled addon");
      write(bundle / "SunshineSBS-LICENSES.txt", "matching bundled licenses");
      // Stand-ins for the bundled installers, so a test can prove whether setup ran them.
      write(bundle / "install-runtime.ps1", "param([string] $GameExecutable)\n$directory = Split-Path -Parent $GameExecutable\nCopy-Item -LiteralPath '" + fs::path(SUNSHINE_TEST_RESHADE_DLL).string() + "' -Destination (Join-Path $directory 'dxgi.dll')\n[IO.File]::WriteAllText((Join-Path $directory 'ReShade.ini'), \"[ADDON]`r`nAddonPath=.`r`n\")\n");
      addon_installer("Copy-Item -LiteralPath (Join-Path $AddonDirectory 'SunshineSBS.addon64') -Destination (Join-Path $directory 'SunshineSBS.addon64')");
    }

    void TearDown() override {
      std::error_code error;
      fs::remove_all(root, error);
    }

    void addon_installer(const std::string &body) {
      write(bundle / "install.ps1", "param([string] $GameExecutable, [string] $AddonDirectory, [switch] $AllowRunningGame)\n$directory = Split-Path -Parent $GameExecutable\n" + body + "\n");
    }

    std::vector<fs::path> game_files() const {
      std::vector<fs::path> result;
      for (const auto &entry : fs::recursive_directory_iterator(root / "drive")) {
        result.push_back(entry.path());
      }
      std::sort(result.begin(), result.end());
      return result;
    }

    game3d::detail::setup_outcome apply(game3d::detail::setup_decisions &decisions) {
      const auto record = [this](const std::string &detail) {
        progress.push_back(detail);
      };
      return game3d::detail::apply_setup(game, decisions, record, bundle);
    }
  };

  TEST_F(Game3DSetup, UnconfirmedFullscreenApplicationIsOfferedAndNeverWritten) {
    game3d::detail::setup_decisions decisions {choices};
    const auto before = game_files();
    const auto outcome = apply(decisions);
    EXPECT_EQ(outcome.state, "ready_to_install");
    EXPECT_EQ(outcome.installation.state, state::missing_reshade);
    EXPECT_EQ(outcome.identity, game3d::file_identity(game));
    EXPECT_TRUE(progress.empty());
    EXPECT_EQ(game_files(), before);
    // Offering records nothing either; only the user's choice is persisted.
    EXPECT_FALSE(fs::exists(choices));
  }

  TEST_F(Game3DSetup, DeclinedExecutableIsNeverWrittenEvenWhereReShadeExists) {
    game3d::detail::setup_decisions decisions {choices};
    ASSERT_TRUE(decisions.record(game, game3d::file_identity(game), decision::never));
    auto before = game_files();
    auto outcome = apply(decisions);
    EXPECT_EQ(outcome.state, "declined");
    EXPECT_EQ(game_files(), before);
    // A user-installed runtime does not override the choice: the add-on is not installed.
    fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, game.parent_path() / "dxgi.dll");
    write(game.parent_path() / "ReShade.ini", "[GENERAL]\n");
    before = game_files();
    outcome = apply(decisions);
    EXPECT_EQ(outcome.state, "declined");
    EXPECT_EQ(outcome.installation.state, state::missing_addon);
    EXPECT_EQ(game_files(), before);
  }

  TEST_F(Game3DSetup, ConfirmedExecutableInstallsOnceAndConsumesTheConfirmation) {
    game3d::detail::setup_decisions decisions {choices};
    ASSERT_TRUE(decisions.record(game, game3d::file_identity(game), decision::install));
    const auto outcome = apply(decisions);
    EXPECT_EQ(outcome.installation.state, state::installed) << outcome.installation.detail;
    EXPECT_EQ(outcome.state, "");
    EXPECT_EQ(progress.size(), 1U);
    EXPECT_TRUE(fs::exists(game.parent_path() / "dxgi.dll"));
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
    EXPECT_EQ(decisions.lookup(game, game3d::file_identity(game)), decision::none);
    // A later removal of ReShade is offered again rather than silently reinstalled.
    fs::remove(game.parent_path() / "dxgi.dll");
    fs::remove(game.parent_path() / "ReShade.ini");
    EXPECT_EQ(apply(decisions).state, "ready_to_install");
    EXPECT_FALSE(fs::exists(game.parent_path() / "dxgi.dll"));
  }

  TEST_F(Game3DSetup, ConfirmationAppliesOnlyToTheConfirmedExecutableFile) {
    game3d::detail::setup_decisions decisions {choices};
    ASSERT_TRUE(decisions.record(game, game3d::file_identity(game), decision::install));
    write(game, "an updated or replaced executable");
    const auto before = game_files();
    const auto outcome = apply(decisions);
    EXPECT_EQ(outcome.state, "ready_to_install");
    EXPECT_EQ(outcome.identity, game3d::file_identity(game));
    EXPECT_EQ(game_files(), before);
  }

  TEST_F(Game3DSetup, FailedConfirmedSetupRollsBackAndKeepsTheConfirmationForRetry) {
    game3d::detail::setup_decisions decisions {choices};
    const auto identity = game3d::file_identity(game);
    ASSERT_TRUE(decisions.record(game, identity, decision::install));
    addon_installer("Write-Error 'addon-step-failure'\nexit 1");
    const auto before = game_files();
    const auto outcome = apply(decisions);
    EXPECT_EQ(outcome.installation.state, state::unavailable);
    EXPECT_EQ(game_files(), before);
    EXPECT_EQ(decisions.lookup(game, identity), decision::install);
  }

  TEST_F(Game3DSetup, AntiCheatRefusesEvenAConfirmedExecutable) {
    game3d::detail::setup_decisions decisions {choices};
    ASSERT_TRUE(decisions.record(game, game3d::file_identity(game), decision::install));
    fs::create_directory(game.parent_path() / "EasyAntiCheat");
    const auto before = game_files();
    const auto outcome = apply(decisions);
    EXPECT_EQ(outcome.installation.state, state::anti_cheat);
    EXPECT_EQ(outcome.state, "");
    EXPECT_TRUE(outcome.identity.empty());
    EXPECT_EQ(game_files(), before);
  }

  TEST_F(Game3DSetup, AntiCheatDoesNotStopUpdatesWhereTheUserInstalledReShade) {
    game3d::detail::setup_decisions decisions {choices};
    fs::create_directory(game.parent_path().parent_path() / "EasyAntiCheat");
    fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, game.parent_path() / "dxgi.dll");
    write(game.parent_path() / "ReShade.ini", "[GENERAL]\n");
    write(game.parent_path() / "SunshineSBS.addon64", "old addon");
    const auto outcome = apply(decisions);
    EXPECT_EQ(outcome.installation.state, state::updated) << outcome.installation.detail;
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
  }

  TEST_F(Game3DSetup, ExistingReShadeFolderReceivesTheAddonOnlyAfterConfirmation) {
    game3d::detail::setup_decisions decisions {choices};
    // The user's own ReShade, for example for depth effects that use Generic Depth.
    fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, game.parent_path() / "dxgi.dll");
    write(game.parent_path() / "ReShade.ini", "[GENERAL]\n");
    const auto before = game_files();
    auto outcome = apply(decisions);
    EXPECT_EQ(outcome.state, "ready_to_install");
    EXPECT_EQ(outcome.installation.state, state::missing_addon);
    EXPECT_TRUE(outcome.installation.reshade_present);
    EXPECT_EQ(outcome.identity, game3d::file_identity(game));
    EXPECT_TRUE(progress.empty());
    EXPECT_EQ(game_files(), before);
    EXPECT_FALSE(fs::exists(choices));
    ASSERT_TRUE(decisions.record(game, outcome.identity, decision::install));
    outcome = apply(decisions);
    EXPECT_EQ(outcome.installation.state, state::installed) << outcome.installation.detail;
    EXPECT_EQ(progress.size(), 1U);
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
    EXPECT_EQ(decisions.lookup(game, game3d::file_identity(game)), decision::none);
  }

  TEST_F(Game3DSetup, InstalledAddonUpdatesWithoutAConfirmation) {
    game3d::detail::setup_decisions decisions {choices};
    fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, game.parent_path() / "dxgi.dll");
    write(game.parent_path() / "ReShade.ini", "[GENERAL]\n");
    write(game.parent_path() / "SunshineSBS.addon64", "old addon");
    const auto outcome = apply(decisions);
    EXPECT_EQ(outcome.installation.state, state::updated) << outcome.installation.detail;
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
    EXPECT_FALSE(fs::exists(choices));
  }

  TEST_F(Game3DSetup, ChoicesPersistAcrossRestartsByPathAndIdentity) {
    const auto identity = game3d::file_identity(game);
    const auto other = game.parent_path() / "Other.exe";
    {
      game3d::detail::setup_decisions decisions {choices};
      EXPECT_EQ(decisions.lookup(game, identity), decision::none);
      ASSERT_TRUE(decisions.record(game, identity, decision::install));
      ASSERT_TRUE(decisions.record(other, "other identity", decision::never));
      EXPECT_FALSE(decisions.record(game.filename(), identity, decision::never));
      EXPECT_FALSE(decisions.record(other, {}, decision::install));
    }
    {
      game3d::detail::setup_decisions restarted {choices};
      EXPECT_EQ(restarted.lookup(game, identity), decision::install);
      std::wstring upper = game.wstring();
      std::transform(upper.begin(), upper.end(), upper.begin(), towupper);
      EXPECT_EQ(restarted.lookup(upper, identity), decision::install);
      EXPECT_EQ(restarted.lookup(game, "changed identity"), decision::none);
      // "Never" follows the executable path, including across game updates.
      EXPECT_EQ(restarted.lookup(other, "updated identity"), decision::never);
      ASSERT_TRUE(restarted.record(game, {}, decision::none));
    }
    game3d::detail::setup_decisions again {choices};
    EXPECT_EQ(again.lookup(game, identity), decision::none);
    EXPECT_EQ(again.lookup(other, {}), decision::never);
  }

  TEST_F(Game3DSetup, UnreadableChoicesCauseAnOfferNotAWrite) {
    write(choices, "{\"version\": 1, \"executables\": [{\"path\": \"relative.exe\", \"decision\": \"install\", \"identity\": \"x\"}]}");
    game3d::detail::setup_decisions decisions {choices};
    EXPECT_EQ(decisions.lookup(L"relative.exe", "x"), decision::none);
    const auto before = game_files();
    EXPECT_EQ(apply(decisions).state, "ready_to_install");
    EXPECT_EQ(game_files(), before);
    write(choices, "not json");
    game3d::detail::setup_decisions corrupt {choices};
    EXPECT_EQ(apply(corrupt).state, "ready_to_install");
    EXPECT_EQ(game_files(), before);
  }
}  // namespace
