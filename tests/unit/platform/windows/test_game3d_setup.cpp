// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_setup.h"

#include <algorithm>
#include <atomic>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
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

  TEST_F(Game3DSetup, ForeignProxyIsNeitherOfferedNorRetriedAfterAnEarlierConfirmation) {
    game3d::detail::setup_decisions decisions {choices};
    // Confirmed while the folder was fresh; another injector's proxy arrived afterwards.
    ASSERT_TRUE(decisions.record(game, game3d::file_identity(game), decision::install));
    write(game.parent_path() / "dxgi.dll", "an upscaler mod's proxy");
    const auto before = game_files();
    for (int launch = 0; launch < 2; ++launch) {
      const auto outcome = apply(decisions);
      EXPECT_EQ(outcome.installation.state, state::unavailable);
      EXPECT_EQ(outcome.state, "");
      EXPECT_NE(outcome.installation.detail.find("dxgi.dll"), std::string::npos) << outcome.installation.detail;
    }
    // No installer ran and nothing was written or replaced.
    EXPECT_TRUE(progress.empty());
    EXPECT_EQ(game_files(), before);
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

  // An application its owner chose where no rule offered it is found again by its path alone,
  // since nothing else makes the host look at it; its file is still confirmed like any other.
  TEST_F(Game3DSetup, AChosenApplicationStaysChosenByItsPathUntilItIsStopped) {
    const auto identity = game3d::file_identity(game);
    const auto other = game.parent_path() / "Other.exe";
    {
      game3d::detail::setup_decisions decisions {choices};
      EXPECT_FALSE(decisions.chosen(game));
      ASSERT_TRUE(decisions.record(game, identity, decision::install, true));
      // An offer a rule made and the owner confirmed is no choice of this kind.
      ASSERT_TRUE(decisions.record(other, "other identity", decision::install));
      EXPECT_TRUE(decisions.chosen(game));
      EXPECT_FALSE(decisions.chosen(other));
    }
    {
      game3d::detail::setup_decisions restarted {choices};
      EXPECT_TRUE(restarted.chosen(game));
      EXPECT_FALSE(restarted.chosen(other));
      std::wstring upper = game.wstring();
      std::transform(upper.begin(), upper.end(), upper.begin(), towupper);
      EXPECT_TRUE(restarted.chosen(upper));
      // After a game update it is still found, and the new file is not yet confirmed.
      EXPECT_EQ(restarted.lookup(game, "changed identity"), decision::none);
      EXPECT_EQ(restarted.lookup(game, identity), decision::install);
      // Confirming the new file keeps it chosen.
      ASSERT_TRUE(restarted.record(game, "changed identity", decision::install, true));
      EXPECT_TRUE(restarted.chosen(game));
      // Stopped: it is no game to the host any more, and the refusal carries no mark.
      ASSERT_TRUE(restarted.record(game, {}, decision::never, true));
      EXPECT_FALSE(restarted.chosen(game));
    }
    game3d::detail::setup_decisions again {choices};
    EXPECT_FALSE(again.chosen(game));
    EXPECT_EQ(again.lookup(game, identity), decision::never);
    // A file written before this mark existed holds none.
    write(choices, R"({"version": 1, "executables": [{"path": ")" + game.generic_string() + R"(", "identity": "x", "decision": "install"}]})");
    game3d::detail::setup_decisions older {choices};
    EXPECT_EQ(older.lookup(game, "x"), decision::install);
    EXPECT_FALSE(older.chosen(game));
  }

  TEST_F(Game3DSetup, TheApplicationsToChooseFromAreTheLatestInFrontOnePerExecutable) {
    using list_t = game3d::detail::choosable_applications;
    list_t list;
    const auto application = [&](const wchar_t *name, const std::uint32_t process) {
      game3d::detected_game result;
      result.executable = game.parent_path() / name;
      result.process_id = process;
      result.creation_time = process * 10u;
      result.name = fs::path {name}.stem().string();
      result.kind = game3d::game_kind_e::text_hook;
      result.chosen = true;
      return result;
    };
    const auto observe = [&](const game3d::detected_game &seen, const std::string &identity) {
      list.observe(seen, seen.executable.string(), identity);
    };
    observe(application(L"One.exe", 1u), "identity one");
    observe(application(L"Two.exe", 2u), "identity two");
    // A file that cannot be identified cannot be confirmed either: it is not listed.
    observe(application(L"Unknown.exe", 3u), {});
    ASSERT_EQ(list.entries().size(), 2u);
    EXPECT_EQ(list.entries()[0].game.name, "Two");
    // In front again, as a new process: one entry, the latest, first.
    observe(application(L"ONE.exe", 4u), "identity one, updated");
    ASSERT_EQ(list.entries().size(), 2u);
    EXPECT_EQ(list.entries()[0].game.process_id, 4u);
    EXPECT_EQ(list.entries()[1].game.name, "Two");
    // Only the path and identity the list shows name an entry.
    const auto one = (game.parent_path() / L"ONE.exe").string();
    EXPECT_TRUE(list.find(one, "identity one, updated"));
    EXPECT_FALSE(list.find(one, "identity one"));
    EXPECT_FALSE(list.find((game.parent_path() / "One.exe").string(), "identity one, updated"));
    EXPECT_FALSE(list.find((game.parent_path() / "Three.exe").string(), "identity one, updated"));
    // A list to pick from, not a history.
    for (std::uint32_t process = 10u; process < 10u + list_t::most; ++process) {
      observe(application((L"More" + std::to_wstring(process) + L".exe").c_str(), process), "identity");
    }
    ASSERT_EQ(list.entries().size(), list_t::most);
    EXPECT_EQ(list.entries()[0].game.process_id, 10u + list_t::most - 1u);
    EXPECT_FALSE(list.find(one, "identity one, updated"));
    // Closed applications go, and so does one that has been chosen.
    list.keep_if([](const game3d::detected_game &seen) {
      return seen.process_id != 10u;
    });
    EXPECT_EQ(list.entries().size(), list_t::most - 1u);
    list.forget(game.parent_path() / L"more11.EXE");
    EXPECT_EQ(list.entries().size(), list_t::most - 2u);
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

  TEST_F(Game3DSetup, UnreadableChoicesAreReadAgainAndNeverOverwritten) {
    const auto identity = game3d::file_identity(game);
    const auto other = game.parent_path() / "Other.exe";
    {
      game3d::detail::setup_decisions decisions {choices};
      ASSERT_TRUE(decisions.record(other, "other identity", decision::never));
    }
    const auto saved = read(choices);
    {
      // Another program holds the file open without sharing it, for example a backup tool.
      game3d::detail::setup_decisions decisions {choices};
      const auto locked = CreateFileW(choices.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);
      ASSERT_NE(locked, INVALID_HANDLE_VALUE);
      EXPECT_EQ(decisions.lookup(other, {}), decision::none);
      // A new choice cannot replace the file it could not read.
      EXPECT_FALSE(decisions.record(game, identity, decision::install));
      CloseHandle(locked);
      EXPECT_EQ(read(choices), saved);
      // Read again once it is available: the earlier choice is still there.
      EXPECT_EQ(decisions.lookup(other, {}), decision::never);
      ASSERT_TRUE(decisions.record(game, identity, decision::install));
      EXPECT_EQ(decisions.lookup(other, {}), decision::never);
    }
    // A torn or damaged file is kept aside for recovery before the next choice is saved.
    const std::string torn = "{\"version\": 1, \"executables\": [{\"path\": ";
    write(choices, torn);
    {
      game3d::detail::setup_decisions decisions {choices};
      EXPECT_EQ(decisions.lookup(other, {}), decision::none);
      ASSERT_TRUE(decisions.record(game, identity, decision::never));
    }
    std::vector<fs::path> aside;
    for (const auto &entry : fs::directory_iterator(choices.parent_path())) {
      if (entry.path().filename().string().starts_with("game3d_setup.json.bad-")) {
        aside.push_back(entry.path());
      }
    }
    ASSERT_EQ(aside.size(), 1U);
    EXPECT_EQ(read(aside.front()), torn);
    game3d::detail::setup_decisions restarted {choices};
    EXPECT_EQ(restarted.lookup(game, identity), decision::never);
    EXPECT_EQ(restarted.lookup(other, {}), decision::none);
    EXPECT_FALSE(fs::exists(choices.parent_path() / "game3d_setup.json.tmp"));
  }
  // A text game: nothing is written, the helper is loaded only after a confirmation, and the
  // confirmation stands for later starts until the user withdraws it.
  TEST_F(Game3DSetup, TextGameLoadsItsHelperOnlyAfterAStandingConfirmation) {
    game3d::detail::setup_decisions decisions(choices);
    int loads = 0;
    const auto no_anti_cheat = [](const fs::path &) {
      return std::string {};
    };
    const auto load = [&] {
      ++loads;
      return game3d::text_hook::load_result {game3d::text_hook::load_state_e::loaded, "running"};
    };
    const auto before = read(game);

    auto offered = game3d::detail::apply_text_hook_setup(game, decisions, no_anti_cheat, load);
    EXPECT_EQ(offered.state, "ready_to_install");
    EXPECT_FALSE(offered.identity.empty());
    EXPECT_FALSE(offered.convert);
    EXPECT_EQ(loads, 0);

    ASSERT_TRUE(decisions.record(game, offered.identity, decision::install));
    for (int start = 1; start <= 2; ++start) {
      // Every start of the game loads the helper again without asking again.
      const auto active = game3d::detail::apply_text_hook_setup(game, decisions, no_anti_cheat, load);
      EXPECT_TRUE(active.state.empty());
      EXPECT_EQ(active.installation.state, state::current);
      EXPECT_TRUE(active.convert);
      EXPECT_TRUE(active.installation.addon_present);
      EXPECT_FALSE(active.retry_load);
      EXPECT_EQ(loads, start);
      EXPECT_EQ(decisions.lookup(game, offered.identity), decision::install);
    }

    ASSERT_TRUE(decisions.record(game, offered.identity, decision::never));
    const auto declined = game3d::detail::apply_text_hook_setup(game, decisions, no_anti_cheat, load);
    EXPECT_EQ(declined.state, "declined");
    EXPECT_FALSE(declined.convert);
    EXPECT_EQ(loads, 2);
    // Nothing was written beside the game at any point.
    EXPECT_EQ(read(game), before);
    EXPECT_EQ(std::distance(fs::directory_iterator(game.parent_path()), fs::directory_iterator {}), 1);
  }

  TEST_F(Game3DSetup, TextGameIsRefusedWithAntiCheatAndConvertedWithoutAHelperThatFails) {
    using load_state = game3d::text_hook::load_state_e;
    game3d::detail::setup_decisions decisions(choices);
    int loads = 0;
    auto outcome_of_load = load_state::failed;
    const auto load = [&] {
      ++loads;
      return game3d::text_hook::load_result {outcome_of_load, "refused."};
    };
    const auto clear = [](const fs::path &) {
      return std::string {};
    };
    const auto identity = game3d::file_identity(game);
    ASSERT_TRUE(decisions.record(game, identity, decision::install));
    // A confirmation never overrides the refusal, and the reason given is the one shown.
    const auto refused = game3d::detail::apply_text_hook_setup(game, decisions, [](const fs::path &) {
      return std::string {"anti-cheat here"};
    }, load);
    EXPECT_EQ(refused.installation.state, state::anti_cheat);
    EXPECT_EQ(refused.installation.detail, "anti-cheat here");
    EXPECT_FALSE(refused.convert);
    EXPECT_EQ(loads, 0);
    // The approval is for converting the game: a helper that does not come up is tried again,
    // and the game is converted meanwhile with the text the picture shows.
    const auto failed = game3d::detail::apply_text_hook_setup(game, decisions, clear, load);
    EXPECT_EQ(failed.installation.state, state::current);
    EXPECT_TRUE(failed.convert);
    EXPECT_TRUE(failed.retry_load);
    EXPECT_FALSE(failed.installation.addon_present);
    EXPECT_EQ(failed.installation.detail.rfind("refused.", 0), 0u);
    EXPECT_EQ(loads, 1);
    // A build without a helper for this kind of game converts it too, with nothing to retry.
    outcome_of_load = load_state::unavailable;
    const auto without = game3d::detail::apply_text_hook_setup(game, decisions, clear, load);
    EXPECT_EQ(without.installation.state, state::current);
    EXPECT_TRUE(without.convert);
    EXPECT_FALSE(without.retry_load);
    // A game that was closed before its approval is converted from its next start.
    outcome_of_load = load_state::not_running;
    const auto closed = game3d::detail::apply_text_hook_setup(game, decisions, clear, load);
    EXPECT_EQ(closed.installation.state, state::installed);
    EXPECT_FALSE(closed.convert);
    EXPECT_FALSE(closed.retry_load);
    EXPECT_EQ(loads, 3);
    // A changed executable is a different file: its confirmation no longer applies.
    write(game, "a different build of the game, with a different size");
    const auto changed = game3d::detail::apply_text_hook_setup(game, decisions, clear, load);
    EXPECT_EQ(changed.state, "ready_to_install");
    EXPECT_FALSE(changed.convert);
    EXPECT_EQ(loads, 3);
  }

  // A game that composes its text itself has nothing the helper could watch: approving it
  // converts it, and nothing is loaded into it, so the anti-cheat refusal has nothing to refuse.
  TEST_F(Game3DSetup, TextGameWithoutAHelperIsConvertedWithNothingLoadedOrRefused) {
    using load_state = game3d::text_hook::load_state_e;
    game3d::detail::setup_decisions decisions(choices);
    int refusals = 0;
    auto running = load_state::unavailable;
    const auto refusal = [&](const fs::path &) {
      ++refusals;
      return std::string {"anti-cheat here"};
    };
    const auto state_of = [&] {
      return game3d::text_hook::load_result {running, "composes its text itself."};
    };
    const auto offered = game3d::detail::apply_text_hook_setup(game, decisions, refusal, state_of, false);
    EXPECT_EQ(offered.state, "ready_to_install");
    EXPECT_FALSE(offered.convert);
    // The offer says what will happen to this game: no helper is mentioned as being loaded.
    EXPECT_NE(offered.installation.detail.find("Nothing is loaded into the game"), std::string::npos);

    ASSERT_TRUE(decisions.record(game, offered.identity, decision::install));
    const auto active = game3d::detail::apply_text_hook_setup(game, decisions, refusal, state_of, false);
    EXPECT_EQ(active.installation.state, state::current);
    EXPECT_TRUE(active.convert);
    EXPECT_FALSE(active.retry_load);
    EXPECT_FALSE(active.installation.addon_present);
    EXPECT_EQ(active.installation.detail, "composes its text itself. Its text is found in the picture alone.");
    // Once the game has closed it is no longer converted, and the approval stands.
    running = load_state::not_running;
    const auto closed = game3d::detail::apply_text_hook_setup(game, decisions, refusal, state_of, false);
    EXPECT_EQ(closed.installation.state, state::installed);
    EXPECT_FALSE(closed.convert);
    EXPECT_EQ(decisions.lookup(game, offered.identity), decision::install);
    EXPECT_EQ(refusals, 0);
  }

  TEST_F(Game3DSetup, NothingIsLoadedIntoAGameWithAntiCheatWhateverElseIsInItsFolder) {
    EXPECT_TRUE(game3d::injection_refusal(game).empty());
    // An existing ReShade installation lifts the refusal for an add-on installation, which only
    // adds a file beside it. It lifts nothing for loading a helper into the running game.
    write(game.parent_path() / "ReShade.ini", "[ADDON]\n");
    write(game.parent_path() / "EasyAntiCheat_EOS_Setup.exe", "marker");
    EXPECT_FALSE(game3d::injection_refusal(game).empty());
  }
}  // namespace
