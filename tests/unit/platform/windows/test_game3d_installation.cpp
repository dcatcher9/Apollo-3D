// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_installation.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include <Windows.h>

namespace {
  namespace fs = std::filesystem;
  namespace game3d = platf::game3d;
  using state = game3d::installation_state_e;

  void write(const fs::path &path, const std::string &value) {
    std::ofstream(path, std::ios::binary) << value;
  }

  std::string read(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  }

  class Game3DInstallation: public testing::Test {
  protected:
    fs::path root;
    fs::path game;
    fs::path bundle;

    void SetUp() override {
      static std::atomic<unsigned> serial {};
      root = fs::temp_directory_path() / ("sunshine-game3d-install-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(serial++));
      // Four parents inside the fixture: the anti-cheat walk never reaches the real temp folder.
      game = root / "drive" / "library" / "common" / "game, with spaces" / "game.exe";
      bundle = root / "host package";
      fs::create_directories(game.parent_path());
      fs::create_directories(bundle);
      write(game, "test executable path");
      write(bundle / "SunshineSBS.addon64", "matching bundled addon");
      write(bundle / "SunshineSBS-LICENSES.txt", "matching bundled licenses");
      fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, game.parent_path() / "dxgi.dll");
    }

    void TearDown() override {
      std::error_code error;
      fs::remove_all(root, error);
    }

    void configure(const std::string &extra = {}) {
      write(game.parent_path() / "ReShade.ini", "[GENERAL]\nUnrelated=keep me\n" + extra);
    }

    void installed(const std::string &content) {
      configure();
      write(game.parent_path() / "SunshineSBS.addon64", content);
    }

    // Stand-ins for the bundled installers: the runtime step publishes a ReShade-identity
    // proxy and ReShade.ini like install-runtime.ps1; the add-on step runs `addon_body`.
    void fake_installers(const std::string &addon_body) {
      write(bundle / "install-runtime.ps1", "param([string] $GameExecutable)\n$directory = Split-Path -Parent $GameExecutable\nCopy-Item -LiteralPath '" + fs::path(SUNSHINE_TEST_RESHADE_DLL).string() + "' -Destination (Join-Path $directory 'dxgi.dll')\n[IO.File]::WriteAllText((Join-Path $directory 'ReShade.ini'), \"[ADDON]`r`nAddonPath=.`r`n\")\n");
      write(bundle / "install.ps1", "param([string] $GameExecutable, [string] $AddonDirectory, [switch] $AllowRunningGame)\n$directory = Split-Path -Parent $GameExecutable\n" + addon_body + "\n");
    }

    // Every file below the fixture root, so a test can prove that nothing was written.
    std::vector<fs::path> files() const {
      std::vector<fs::path> result;
      for (const auto &entry : fs::recursive_directory_iterator(root)) {
        result.push_back(entry.path());
      }
      std::sort(result.begin(), result.end());
      return result;
    }
  };

  TEST_F(Game3DInstallation, AntiCheatMarkersInTheGameFolderOrItsParentsRefuseAFreshRuntime) {
    const auto title = game.parent_path();
    const auto unreal_game = title / "Project" / "Binaries" / "Win64" / "Project-Win64-Shipping.exe";
    fs::create_directories(unreal_game.parent_path());
    write(unreal_game, "unreal shipping executable");
    fake_installers("[IO.File]::WriteAllText((Join-Path $directory 'SunshineSBS.addon64'), 'written')");
    const std::vector<std::pair<std::string, bool>> markers {
      {"EasyAntiCheat", true},
      {"EasyAntiCheat_EOS_Setup.exe", false},
      {"start_protected_game.exe", false},
      {"BattlEye", true},
      {"TslGame_BE.exe", false},
      {"EAAntiCheat.GameServiceLauncher.exe", false},
    };
    for (const auto &[name, directory] : markers) {
      SCOPED_TRACE(name);
      const auto marker = title / name;
      directory ? (void) fs::create_directory(marker) : write(marker, "marker");
      for (const auto &executable : {game, unreal_game}) {
        SCOPED_TRACE(executable.string());
        const auto before = files();
        EXPECT_EQ(game3d::inspect_installation(executable, bundle).state, state::anti_cheat);
        const auto status = game3d::install_reshade_and_addon_for_next_launch(executable, bundle);
        EXPECT_EQ(status.state, state::anti_cheat);
        EXPECT_NE(status.detail.find(name), std::string::npos) << status.detail;
        EXPECT_EQ(game3d::install_reshade_for_next_launch(executable, bundle).state, state::anti_cheat);
        EXPECT_EQ(files(), before);
      }
      fs::remove_all(marker);
      EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::missing_reshade);
    }
  }

  TEST_F(Game3DInstallation, AntiCheatMarkersDoNotRefuseAnExistingReShadeInstallation) {
    // Battlefield 6's campaign: the user installed ReShade beside SP\bf6.exe, and the game root
    // above it holds EA AntiCheat. Sunshine creates no proxy there, but keeps the add-on current.
    const auto campaign = game.parent_path() / "SP" / "bf6.exe";
    fs::create_directories(campaign.parent_path());
    write(campaign, "campaign executable");
    write(game.parent_path() / "EAAntiCheat.GameServiceLauncher.exe", "marker");
    fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, campaign.parent_path() / "dxgi.dll");
    write(campaign.parent_path() / "ReShade.ini", "[GENERAL]\n");
    fake_installers("Copy-Item -LiteralPath (Join-Path $AddonDirectory 'SunshineSBS.addon64') -Destination (Join-Path $directory 'SunshineSBS.addon64')");
    EXPECT_EQ(game3d::inspect_installation(campaign, bundle).state, state::missing_addon);
    const auto installed = game3d::install_addon_for_next_launch(campaign, bundle);
    EXPECT_EQ(installed.state, state::installed) << installed.detail;
    write(campaign.parent_path() / "SunshineSBS.addon64", "old addon");
    const auto updated = game3d::update_existing_addon(campaign, bundle);
    EXPECT_EQ(updated.state, state::updated) << updated.detail;
    EXPECT_EQ(read(campaign.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
  }

  TEST_F(Game3DInstallation, AntiCheatWalkStopsAtFourParentsAndNeverEntersSiblingGames) {
    // bin, then four parents: game, common, library and drive. The fixture root is fifth.
    const auto executable = game.parent_path() / "bin" / "game.exe";
    fs::create_directories(executable.parent_path());
    write(executable, "test executable path");
    // A sibling game's anti-cheat and a marker five levels up do not belong to this game.
    fs::create_directories(game.parent_path().parent_path() / "Online Game" / "EasyAntiCheat");
    write(game.parent_path().parent_path() / "Online Game" / "start_protected_game.exe", "sibling");
    fs::create_directory(root / "BattlEye");
    EXPECT_EQ(game3d::inspect_installation(executable, bundle).state, state::missing_reshade);
    write(root / "drive" / "Shooter_BE.exe", "marker");
    EXPECT_EQ(game3d::inspect_installation(executable, bundle).state, state::anti_cheat);
  }

  TEST_F(Game3DInstallation, EnvironmentVariableAddonPathNeedsManualSetup) {
    // The game expands %LOCALAPPDATA% for its own user; the service would resolve SYSTEM's.
    configure("[ADDON]\nAddonPath=%LOCALAPPDATA%\\ReShade\\Addons\n");
    fs::copy_file(fs::path(SUNSHINE_SOURCE_DIR) / "tools/reshade/install.ps1", bundle / "install.ps1");
    const auto before = files();
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::unavailable);
    EXPECT_EQ(game3d::install_addon_for_next_launch(game, bundle).state, state::unavailable);
    EXPECT_EQ(game3d::update_existing_addon(game, bundle).state, state::unavailable);
    EXPECT_EQ(files(), before);
    configure("[ADDON]\nAddonPath=" + (root / "absolute addons").string() + "\n");
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::missing_addon);
  }

  TEST_F(Game3DInstallation, FileIdentityFollowsTheExecutableFile) {
    const auto identity = game3d::file_identity(game);
    EXPECT_FALSE(identity.empty());
    EXPECT_EQ(game3d::file_identity(game), identity);
    write(game, "updated test executable");
    EXPECT_NE(game3d::file_identity(game), identity);
    EXPECT_TRUE(game3d::file_identity(game.parent_path()).empty());
    EXPECT_TRUE(game3d::file_identity(game.parent_path() / "missing.exe").empty());
  }

  TEST_F(Game3DInstallation, FailedAddonStepRemovesTheRuntimeItCreated) {
    fs::remove(game.parent_path() / "dxgi.dll");
    write(game.parent_path() / "ReShadePreset.ini", "Techniques=Curves@Curves.fx\n");
    fake_installers("Write-Error 'addon-step-failure'\nexit 1");
    const auto status = game3d::install_reshade_and_addon_for_next_launch(game, bundle);
    EXPECT_EQ(status.state, state::unavailable);
    EXPECT_NE(status.detail.find("removed again"), std::string::npos) << status.detail;
    EXPECT_NE(status.detail.find("addon-step-failure"), std::string::npos) << status.detail;
    EXPECT_FALSE(fs::exists(game.parent_path() / "dxgi.dll"));
    EXPECT_FALSE(fs::exists(game.parent_path() / "ReShade.ini"));
    EXPECT_EQ(read(game.parent_path() / "ReShadePreset.ini"), "Techniques=Curves@Curves.fx\n");
    // The folder is fresh again, so a later confirmed attempt can retry.
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::missing_reshade);
  }

  TEST_F(Game3DInstallation, RollbackKeepsRuntimeFilesChangedSinceTheRuntimeStep) {
    fs::remove(game.parent_path() / "dxgi.dll");
    fake_installers("[IO.File]::AppendAllText((Join-Path $directory 'ReShade.ini'), 'UserEdit=1')\nexit 1");
    const auto status = game3d::install_reshade_and_addon_for_next_launch(game, bundle);
    EXPECT_EQ(status.state, state::unavailable);
    EXPECT_NE(status.detail.find("could not be removed"), std::string::npos) << status.detail;
    // The proxy is removed first, so no runtime loads; the edited configuration is preserved.
    EXPECT_FALSE(fs::exists(game.parent_path() / "dxgi.dll"));
    EXPECT_NE(read(game.parent_path() / "ReShade.ini").find("UserEdit=1"), std::string::npos);
  }

  TEST_F(Game3DInstallation, CombinedInstallLeavesExistingFilesAndFinishesBothSteps) {
    fs::remove(game.parent_path() / "dxgi.dll");
    fake_installers("Copy-Item -LiteralPath (Join-Path $AddonDirectory 'SunshineSBS.addon64') -Destination (Join-Path $directory 'SunshineSBS.addon64')");
    const auto status = game3d::install_reshade_and_addon_for_next_launch(game, bundle);
    EXPECT_EQ(status.state, state::installed) << status.detail;
    EXPECT_TRUE(fs::exists(game.parent_path() / "dxgi.dll"));
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::current);
  }

  TEST_F(Game3DInstallation, RequiresRuntimeIdentityAndConfiguration) {
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::missing_reshade);
    configure();
    fs::remove(game.parent_path() / "dxgi.dll");
    write(game.parent_path() / "dxgi.dll", "some other graphics proxy");
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::missing_reshade);
  }

  TEST_F(Game3DInstallation, DoesNotInstallMissingAddonAsAnUpdate) {
    configure();
    EXPECT_EQ(game3d::update_existing_addon(game, bundle).state, state::missing_addon);
    EXPECT_FALSE(fs::exists(game.parent_path() / "SunshineSBS.addon64"));
  }

  TEST_F(Game3DInstallation, HonorsScalarEscapedCommaAddonPath) {
    configure("[ADDON]\nAddonPath=.\\addons,, personal\n");
    const auto directory = game.parent_path() / "addons, personal";
    fs::create_directory(directory);
    write(directory / "SunshineSBS.addon64", "matching bundled addon");
    auto status = game3d::inspect_installation(game, bundle);
    EXPECT_EQ(status.state, state::current);
    EXPECT_EQ(status.addon, directory / "SunshineSBS.addon64");
  }

  TEST_F(Game3DInstallation, RejectsAmbiguousConfiguredPath) {
    for (const auto &ini : {"[ADDON]\nAddonPath=one,two\n", "[ADDON]\nAddonPath=one\nAddonPath=two\n", "[ADDON]\n[ADDON]\n"}) {
      configure(ini);
      EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::unavailable);
    }
  }

  TEST_F(Game3DInstallation, MatchingAddonDoesNotRewriteFiles) {
    installed("matching bundled addon");
    EXPECT_EQ(game3d::update_existing_addon(game, bundle).state, state::current);
    EXPECT_FALSE(fs::exists(game.parent_path() / "SunshineSBS.addon64.previous"));
  }

  TEST_F(Game3DInstallation, ReplacesExistingAddonWithBackupAndPreservesPreferences) {
    installed("old addon");
    const auto ini = read(game.parent_path() / "ReShade.ini");
    EXPECT_EQ(game3d::update_existing_addon(game, bundle).state, state::updated);
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64.previous"), "old addon");
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS-LICENSES.txt"), "matching bundled licenses");
    EXPECT_EQ(read(game.parent_path() / "ReShade.ini"), ini);
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::current);
    write(bundle / "SunshineSBS.addon64", "next matching bundled addon");
    EXPECT_EQ(game3d::update_existing_addon(game, bundle).state, state::updated);
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64.previous"), "matching bundled addon");
  }

  TEST_F(Game3DInstallation, LeavesLockedAddonUntouched) {
    installed("old addon");
    const auto addon = game.parent_path() / "SunshineSBS.addon64";
    const auto locked = CreateFileW(addon.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    ASSERT_NE(locked, INVALID_HANDLE_VALUE);
    const auto status = game3d::update_existing_addon(game, bundle);
    CloseHandle(locked);
    EXPECT_EQ(status.state, state::update_pending);
    EXPECT_EQ(read(addon), "old addon");
    EXPECT_FALSE(fs::exists(game.parent_path() / "SunshineSBS.addon64.previous"));
  }

  TEST_F(Game3DInstallation, SameNamedExecutablesKeepIndependentInstallations) {
    installed("main game addon");
    const auto campaign_directory = game.parent_path() / "SP";
    fs::create_directory(campaign_directory);
    const auto campaign = campaign_directory / game.filename();
    write(campaign, "independent campaign executable");
    write(campaign_directory / "ReShade.ini", "[GENERAL]\nCampaign=keep me\n");
    fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, campaign_directory / "dxgi.dll");
    write(campaign_directory / "SunshineSBS.addon64", "campaign addon");

    const auto result = game3d::update_existing_addon(campaign, bundle);
    EXPECT_EQ(result.state, state::updated) << result.detail;
    EXPECT_EQ(result.executable, campaign);
    EXPECT_EQ(result.addon, campaign_directory / "SunshineSBS.addon64");
    EXPECT_EQ(read(campaign_directory / "SunshineSBS.addon64"), "matching bundled addon");
    EXPECT_EQ(read(campaign_directory / "SunshineSBS.addon64.previous"), "campaign addon");
    EXPECT_EQ(read(campaign_directory / "ReShade.ini"), "[GENERAL]\nCampaign=keep me\n");
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "main game addon");
    EXPECT_EQ(game3d::inspect_installation(game, bundle).state, state::update_available);
    EXPECT_EQ(game3d::inspect_installation(campaign, bundle).state, state::current);
  }

  TEST_F(Game3DInstallation, MissingHostPackageLeavesExistingAddonUntouched) {
    installed("old addon");
    fs::remove(bundle / "SunshineSBS-LICENSES.txt");
    EXPECT_EQ(game3d::update_existing_addon(game, bundle).state, state::unavailable);
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "old addon");
  }

  TEST_F(Game3DInstallation, InstallsMissingAddonWithBundledScriptAndKeepsUnrelatedSettings) {
    configure();
    fs::copy_file(fs::path(SUNSHINE_SOURCE_DIR) / "tools/reshade/install.ps1", bundle / "install.ps1");
    const auto status = game3d::install_addon_for_next_launch(game, bundle);
    EXPECT_EQ(status.state, state::installed) << status.detail;
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
    EXPECT_NE(read(game.parent_path() / "ReShade.ini").find("Unrelated=keep me"), std::string::npos);
    EXPECT_EQ(game3d::install_addon_for_next_launch(game, bundle).state, state::current);
  }

  TEST_F(Game3DInstallation, InvokesRuntimeInstallerWithoutShellWithQuotedGamePath) {
    write(bundle / "install-runtime.ps1", "param([string] $GameExecutable)\n[IO.File]::WriteAllText((Join-Path (Split-Path -Parent $GameExecutable) 'ReShade.ini'), '[GENERAL]')\n");
    const auto status = game3d::install_reshade_for_next_launch(game, bundle);
    EXPECT_EQ(status.state, state::installed) << status.detail;
    EXPECT_TRUE(status.reshade_present);
    EXPECT_FALSE(status.addon_present);
  }

  TEST_F(Game3DInstallation, CapturesInstallerErrorWrittenImmediatelyBeforeExit) {
    write(bundle / "install-runtime.ps1", "param([string] $GameExecutable)\nWrite-Error 'installation-specific-error'\nexit 1\n");
    const auto status = game3d::install_reshade_for_next_launch(game, bundle);
    EXPECT_EQ(status.state, state::unavailable);
    EXPECT_NE(status.detail.find("installation-specific-error"), std::string::npos) << status.detail;
  }

  TEST_F(Game3DInstallation, RefusesUpdatesWhileMatchingExecutableIsRunning) {
    installed("old addon");
    std::wstring own_path(32768, L'\0');
    own_path.resize(GetModuleFileNameW(nullptr, own_path.data(), own_path.size()));
    fs::copy_file(own_path, game, fs::copy_options::overwrite_existing);
    // The full host test executable imports DXGI. Its loader must use Windows'
    // real DLL, rather than this suite's resource-only ReShade identity fixture.
    // Restore the identity fixture only after the child finishes loading.
    fs::remove(game.parent_path() / "dxgi.dll");
    const auto event_name = L"Local\\SunshineGame3DInstallation-" + std::to_wstring(GetCurrentProcessId());

    struct child_fixture_t {
      HANDLE release {};
      HANDLE ready {};
      PROCESS_INFORMATION child {};

      ~child_fixture_t() {
        if (release) {
          SetEvent(release);
        }
        if (child.hProcess) {
          if (WaitForSingleObject(child.hProcess, 10000) != WAIT_OBJECT_0) {
            TerminateProcess(child.hProcess, 1);
          }
          CloseHandle(child.hProcess);
          CloseHandle(child.hThread);
        }
        if (ready) {
          CloseHandle(ready);
        }
        if (release) {
          CloseHandle(release);
        }
      }
    } fixture;

    fixture.release = CreateEventW(nullptr, TRUE, FALSE, (event_name + L"-release").c_str());
    fixture.ready = CreateEventW(nullptr, TRUE, FALSE, (event_name + L"-ready").c_str());
    ASSERT_NE(fixture.release, nullptr);
    ASSERT_NE(fixture.ready, nullptr);
    std::wstring command = L"\"" + game.wstring() + L"\" --gtest_filter=Game3DInstallationChild.WaitForRelease --game3d-event=" + event_name;
    STARTUPINFOW startup {};
    startup.cb = sizeof(startup);
    const auto created = CreateProcessW(game.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, game.parent_path().c_str(), &startup, &fixture.child);
    ASSERT_TRUE(created);
    // CreateProcess returns before the child's loader and test entry point have
    // initialized. Wait for an explicit ready signal before inspecting processes.
    const HANDLE wait_handles[] {fixture.ready, fixture.child.hProcess};
    const auto ready = WaitForMultipleObjects(2, wait_handles, FALSE, 10000);
    DWORD exit_code = 0;
    ASSERT_TRUE(GetExitCodeProcess(fixture.child.hProcess, &exit_code));
    ASSERT_EQ(ready, WAIT_OBJECT_0) << "Fixture did not become ready; child exit code=" << exit_code;
    ASSERT_EQ(exit_code, STILL_ACTIVE);
    fs::copy_file(SUNSHINE_TEST_RESHADE_DLL, game.parent_path() / "dxgi.dll");
    const auto status = game3d::update_existing_addon(game, bundle);
    EXPECT_EQ(status.state, state::update_pending) << status.detail;
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "old addon");
    fs::remove(game.parent_path() / "SunshineSBS.addon64");
    fs::copy_file(fs::path(SUNSHINE_SOURCE_DIR) / "tools/reshade/install.ps1", bundle / "install.ps1");
    const auto fresh = game3d::install_addon_for_next_launch(game, bundle);
    SetEvent(fixture.release);
    ASSERT_EQ(WaitForSingleObject(fixture.child.hProcess, 10000), WAIT_OBJECT_0);
    ASSERT_TRUE(GetExitCodeProcess(fixture.child.hProcess, &exit_code));
    EXPECT_EQ(exit_code, 0U);
    EXPECT_EQ(fresh.state, state::installed) << fresh.detail;
    EXPECT_EQ(read(game.parent_path() / "SunshineSBS.addon64"), "matching bundled addon");
  }
}  // namespace

TEST(Game3DInstallationChild, WaitForRelease) {
  const std::wstring command = GetCommandLineW();
  constexpr std::wstring_view marker = L"--game3d-event=";
  const auto begin = command.find(marker);
  if (begin == std::wstring::npos) {
    GTEST_SKIP();
  }
  const auto name = command.substr(begin + marker.size());
  const auto release = OpenEventW(SYNCHRONIZE, FALSE, (name + L"-release").c_str());
  ASSERT_NE(release, nullptr);
  const auto ready = OpenEventW(EVENT_MODIFY_STATE, FALSE, (name + L"-ready").c_str());
  ASSERT_NE(ready, nullptr);
  EXPECT_TRUE(SetEvent(ready));
  EXPECT_EQ(WaitForSingleObject(release, 60000), WAIT_OBJECT_0);
  CloseHandle(ready);
  CloseHandle(release);
}
