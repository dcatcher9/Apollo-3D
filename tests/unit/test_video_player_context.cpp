// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/video_player_context.h"

#include <gtest/gtest.h>

namespace {
  namespace player = platf::video_player::detail;

  TEST(VideoPlayerContext, ArbitraryMediaClientNameCanAuthorizeVideoCapability) {
    const player::registration_t registration {
      true,
      L"Any vendor's movie application",
      {{L".custom-video", L"video/x-vendor", {}, LR"("D:\Movie Tools\viewer.exe" "%1")"}},
    };
    const auto executables = player::registered_video_executables(registration);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables.front(), LR"(D:\Movie Tools\viewer.exe)");
  }

  TEST(VideoPlayerContext, FileHandlerWithoutMediaCategoryCannotAuthorizeAnApplication) {
    player::registration_t registration {
      false,
      L"A browser or editor",
      {{L".mp4", L"video/mp4", L"video", LR"("C:\Application\app.exe" "%1")"}},
    };
    EXPECT_TRUE(player::registered_video_executables(registration).empty());
    registration.registered_media_client = true;
    registration.name.clear();
    EXPECT_TRUE(player::registered_video_executables(registration).empty());
  }

  TEST(VideoPlayerContext, VideoMimeOrPerceivedTypeQualifiesButAudioDoesNot) {
    EXPECT_TRUE(player::is_video_type({L".mp4", L"VIDEO/MP4", {}, {}}));
    EXPECT_TRUE(player::is_video_type({L".mkv", {}, L"Video", {}}));
    EXPECT_FALSE(player::is_video_type({L".mp3", L"audio/mpeg", L"audio", {}}));
    EXPECT_FALSE(player::is_video_type({L".mp4", L"video/", {}, {}}));
    EXPECT_FALSE(player::is_video_type({L".mp4", L"application/video", {}, {}}));
    EXPECT_FALSE(player::is_video_type({L".mp4", {}, {}, {}}));
    EXPECT_FALSE(player::is_video_type({L"mp4", L"video/mp4", {}, {}}));
    EXPECT_FALSE(player::is_video_type({L".mp4\\child", L"video/mp4", {}, {}}));
    const player::registration_t audio {true, L"Audio client", {{L".mp3", L"audio/mpeg", {}, LR"(C:\audio.exe "%1")"}}};
    EXPECT_TRUE(player::registered_video_executables(audio).empty());
  }

  TEST(VideoPlayerContext, ParsesQuotedExecutableWithArgumentsWithoutLaunchingIt) {
    const auto quoted = player::command_executable(LR"("C:\Media Tools\viewer.exe" /new "%1")");
    ASSERT_TRUE(quoted);
    EXPECT_EQ(*quoted, LR"(C:\Media Tools\viewer.exe)");
    const auto unquoted = player::command_executable(LR"(C:\Media\viewer.exe "%1")");
    ASSERT_TRUE(unquoted);
    EXPECT_EQ(*unquoted, LR"(C:\Media\viewer.exe)");
    EXPECT_FALSE(player::command_executable(LR"(viewer.exe "%1")"));
    EXPECT_FALSE(player::command_executable(LR"(C:viewer.exe "%1")"));
    EXPECT_FALSE(player::command_executable(LR"(C:\Media Tools\viewer.exe "%1")"));
    EXPECT_FALSE(player::command_executable(LR"(C:\Media\viewer.cmd "%1")"));
    EXPECT_FALSE(player::command_executable(LR"("" "%1")"));
    EXPECT_FALSE(player::command_executable({}));
  }

  TEST(VideoPlayerContext, ComparesFullNormalizedPathsRatherThanBasenames) {
    EXPECT_TRUE(player::same_executable_path(LR"(C:\Media\viewer.exe)", L"c:/media/./viewer.exe"));
    EXPECT_TRUE(player::same_executable_path(LR"(C:\Media\Old\..\viewer.exe)", LR"(C:\Media\viewer.exe)"));
    EXPECT_FALSE(player::same_executable_path(LR"(C:\Media\viewer.exe)", LR"(D:\Other\viewer.exe)"));
    EXPECT_FALSE(player::same_executable_path(LR"(C:\Media\viewer.exe)", L"viewer.exe"));
    EXPECT_FALSE(player::same_executable_path({}, {}));
  }
}  // namespace
