// SPDX-License-Identifier: GPL-3.0-only
#include <gtest/gtest.h>

#include <src/platform/windows/host3d_subtitle_apps.h>

#include <string>

namespace {
  using platf::host3d_subtitle_apps::selection_t;

  TEST(Host3dSubtitleAppsTest, NothingIsProtectedUntilTheOwnerEnablesThatExecutable) {
    selection_t selection;
    EXPECT_EQ(selection.serialize(), "[]");
    EXPECT_FALSE(selection.observe("C:/Games/Example/game.exe"));
    EXPECT_FALSE(selection.observe("C:/Windows/explorer.exe"));
    ASSERT_TRUE(selection.set_enabled("C:/Games/Example/game.exe", true));
    EXPECT_TRUE(selection.observe("C:/Games/Example/game.exe"));
    // Enabling one application grants nothing to the desktop or any other application.
    EXPECT_FALSE(selection.observe("C:/Windows/explorer.exe"));
    EXPECT_FALSE(selection.observe("C:/Games/Other/game.exe"));
    ASSERT_TRUE(selection.set_enabled("C:/Games/Example/game.exe", false));
    EXPECT_FALSE(selection.observe("C:/Games/Example/game.exe"));
    EXPECT_EQ(selection.serialize(), "[]");
  }

  TEST(Host3dSubtitleAppsTest, OneExecutableHasOneSpelling) {
    selection_t selection;
    ASSERT_TRUE(selection.set_enabled("C:\\Games\\Example\\sub\\..\\Game.EXE", true));
    EXPECT_EQ(selection.serialize(), "[\"C:/Games/Example/Game.EXE\"]");
    EXPECT_TRUE(selection.enabled("c:/games/example/game.exe"));
    EXPECT_TRUE(selection.observe("C:\\GAMES\\EXAMPLE\\GAME.EXE"));
    ASSERT_TRUE(selection.set_enabled("c:/games/example/game.exe", true));
    EXPECT_EQ(selection.list().size(), 1u);
    ASSERT_TRUE(selection.set_enabled("C:/GAMES/example/game.exe", false));
    EXPECT_EQ(selection.serialize(), "[]");
  }

  TEST(Host3dSubtitleAppsTest, NonAsciiPathsRoundTrip) {
    const std::string path = "C:/\xE3\x82\xB2\xE3\x83\xBC\xE3\x83\xA0/\xE5\x92\xB2\xE5\xA4\x9C.exe";
    selection_t selection;
    ASSERT_TRUE(selection.set_enabled(path, true));
    selection_t reloaded;
    reloaded.load(selection.serialize());
    EXPECT_TRUE(reloaded.observe(path));
    ASSERT_EQ(reloaded.list().size(), 1u);
    EXPECT_EQ(reloaded.list().front().name, "\xE5\x92\xB2\xE5\xA4\x9C.exe");
  }

  TEST(Host3dSubtitleAppsTest, InvalidPathsAndConfigurationGrantNothing) {
    selection_t selection;
    for (const auto *invalid : {"", "game.exe", "..\\game.exe", "C:/Games/Example/readme.txt", "C:/Games/Example/"}) {
      EXPECT_FALSE(selection.set_enabled(invalid, true)) << invalid;
      EXPECT_FALSE(selection.observe(invalid)) << invalid;
    }
    EXPECT_FALSE(selection.set_enabled(std::string("C:/Games/a\0b.exe", 17u), true));
    EXPECT_FALSE(selection.set_enabled("C:/Games/\xFF\xFE.exe", true));
    for (const auto *malformed : {"enabled", "{\"a\":1}", "[1, true, null]", "[\"relative.exe\"]", "[\"C:/a.exe\""}) {
      selection.load(malformed);
      EXPECT_EQ(selection.serialize(), "[]") << malformed;
    }
    selection.load("[\"C:/a.exe\", 7, \"C:/b.txt\", \"C:/a.exe\", \"C:/c.exe\"]");
    EXPECT_EQ(selection.serialize(), "[\"C:/a.exe\",\"C:/c.exe\"]");
    // Loading replaces the set instead of merging into it.
    selection.load("[\"C:/d.exe\"]");
    EXPECT_EQ(selection.serialize(), "[\"C:/d.exe\"]");
    selection.load("");
    EXPECT_EQ(selection.serialize(), "[]");
  }

  TEST(Host3dSubtitleAppsTest, ListShowsEnabledFirstThenMostRecentAndIsBounded) {
    selection_t selection;
    ASSERT_TRUE(selection.set_enabled("C:/enabled.exe", true));
    for (std::size_t index = 0u; index < selection_t::max_recent + 3u; ++index) {
      selection.observe("C:/recent" + std::to_string(index) + ".exe");
    }
    selection.observe("C:/recent5.exe");  // Seen again: moves to the front without a duplicate.
    const auto list = selection.list();
    ASSERT_EQ(list.size(), 1u + selection_t::max_recent);
    EXPECT_EQ(list[0].executable, "C:/enabled.exe");
    EXPECT_TRUE(list[0].enabled);
    EXPECT_FALSE(list[0].recent);
    EXPECT_EQ(list[1].executable, "C:/recent5.exe");
    EXPECT_FALSE(list[1].enabled);
    EXPECT_TRUE(list[1].recent);
    EXPECT_EQ(list[2].name, "recent" + std::to_string(selection_t::max_recent + 2u) + ".exe");
    // The oldest observations fell out of the bounded recent list.
    for (const auto &app : list) {
      EXPECT_NE(app.executable, "C:/recent0.exe");
    }
    selection.observe("C:/enabled.exe");
    EXPECT_TRUE(selection.list()[0].recent);
    EXPECT_EQ(selection.list().size(), 1u + selection_t::max_recent - 1u);
  }

  TEST(Host3dSubtitleAppsTest, EnabledSetIsBounded) {
    selection_t selection;
    for (std::size_t index = 0u; index < selection_t::max_enabled; ++index) {
      ASSERT_TRUE(selection.set_enabled("C:/app" + std::to_string(index) + ".exe", true));
    }
    EXPECT_FALSE(selection.set_enabled("C:/one-too-many.exe", true));
    EXPECT_TRUE(selection.set_enabled("C:/app0.exe", true));  // Already enabled.
    EXPECT_TRUE(selection.set_enabled("C:/app0.exe", false));
    EXPECT_TRUE(selection.set_enabled("C:/one-too-many.exe", true));
  }
}  // namespace
