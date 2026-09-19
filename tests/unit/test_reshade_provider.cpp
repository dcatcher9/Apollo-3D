#include "src/game_source_tracker.h"
#include "src/host_sbs_provider.h"
#include "src/video.h"

#include <gtest/gtest.h>
#include <limits>

TEST(ReShadeProviderDimensions, PreservesTheLogicalDesktopAndBothNativeEyes) {
  EXPECT_TRUE(video::external_sbs_dimensions_match(1920, 1080, 3840, 1080));
  EXPECT_TRUE(video::external_sbs_dimensions_match(3840, 2160, 7680, 2160));
  EXPECT_TRUE(video::external_sbs_dimensions_match(1080, 1920, 2160, 1920));
}

TEST(ReShadeProviderDimensions, RejectsPackedOrHalfSbsSourcesAndUnsafeRasterShapes) {
  EXPECT_FALSE(video::external_sbs_dimensions_match(3840, 1080, 3840, 1080));
  EXPECT_FALSE(video::external_sbs_dimensions_match(1920, 1080, 1920, 1080));
  EXPECT_FALSE(video::external_sbs_dimensions_match(1919, 1080, 3838, 1080));
  EXPECT_FALSE(video::external_sbs_dimensions_match(1920, 1081, 3840, 1081));
  EXPECT_FALSE(video::external_sbs_dimensions_match(0, 1080, 0, 1080));
  EXPECT_FALSE(video::external_sbs_dimensions_match(std::numeric_limits<int>::max(), 1080, -2, 1080));
}

TEST(ReShadeProviderDimensions, RejectsCodecAndConfiguredScalingInsteadOfChangingAuthoredStereo) {
  const auto hevc = video::host_sbs_output_dimensions(3840, 2160, 1, 8192, 8192, 8192);
  EXPECT_TRUE(video::external_sbs_dimensions_match(3840, 2160, hevc.width, hevc.height));

  const auto h264 = video::host_sbs_output_dimensions(3840, 2160, 0, 8192, 8192, 8192);
  EXPECT_FALSE(video::external_sbs_dimensions_match(3840, 2160, h264.width, h264.height));

  const auto restricted = video::host_sbs_output_dimensions(1920, 1080, 1, 1920, 8192, 8192);
  EXPECT_FALSE(video::external_sbs_dimensions_match(1920, 1080, restricted.width, restricted.height));

  const auto short_runtime = video::host_sbs_output_dimensions(1920, 1080, 1, 8192, 8192, 720);
  EXPECT_FALSE(video::external_sbs_dimensions_match(1920, 1080, short_runtime.width, short_runtime.height));
}

TEST(GameSourceStatus, SeparatesProviderDiscoveryFromPackedOutputAndAi) {
  EXPECT_TRUE(video::is_game_mode(video::SBS_GAME_MONO));
  EXPECT_TRUE(video::is_game_mode(video::SBS_GAME_SBS));
  EXPECT_FALSE(video::is_game_mode(video::SBS_AI));
  EXPECT_FALSE(video::is_game_mode(video::SBS_OFF));
  EXPECT_FALSE(video::is_packed_mode(video::SBS_GAME_MONO));
  EXPECT_TRUE(video::is_packed_mode(video::SBS_GAME_SBS));
  EXPECT_TRUE(video::is_packed_mode(video::SBS_AI));
}

TEST(GameSourceStatus, PublishesEdgesAndStableHeartbeatsWithinAppliedGeneration) {
  video::game_source_tracker_t tracker;
  video::game_source_state_t status {};
  status.presentation_generation = 7;
  status.source_width = 1920;
  status.source_height = 1080;
  status.packed_width = 3840;
  status.packed_height = 1080;
  const auto now = video::game_source_tracker_t::clock_t::time_point {};
  const video::game_source_identity_t source {42, 1234, 1};
  auto waiting = tracker.observe(status, {}, now);
  ASSERT_TRUE(waiting);
  EXPECT_EQ(waiting->source_revision, 1u);
  EXPECT_FALSE(tracker.observe(status, {}, now + std::chrono::milliseconds(10)));
  status.state = video::GAME_SOURCE_READY;
  status.provider = video::GAME_PROVIDER_RESHADE;
  auto ready = tracker.observe(status, source, now + std::chrono::milliseconds(20));
  ASSERT_TRUE(ready);
  EXPECT_EQ(ready->source_revision, 2u);
  EXPECT_FALSE(tracker.observe(status, source, now + std::chrono::milliseconds(900)));
  auto heartbeat = tracker.observe(status, source, now + std::chrono::milliseconds(1020));
  ASSERT_TRUE(heartbeat);
  EXPECT_EQ(heartbeat->source_revision, ready->source_revision);
  auto replacement = tracker.observe(status, {42, 1234, 2}, now + std::chrono::milliseconds(1030));
  ASSERT_TRUE(replacement);
  EXPECT_EQ(replacement->source_revision, 3u);
  status.state = video::GAME_SOURCE_WAITING;
  status.provider = video::GAME_PROVIDER_NONE;
  auto lost = tracker.observe(status, {}, now + std::chrono::milliseconds(1040));
  ASSERT_TRUE(lost);
  EXPECT_EQ(lost->source_revision, 4u);
}

TEST(GameSourceStatus, RepublishAfterBitrateGenerationChangeWithoutAnotherSourceEdge) {
  video::game_source_tracker_t tracker;
  video::game_source_state_t status {};
  status.state = video::GAME_SOURCE_READY;
  status.provider = video::GAME_PROVIDER_RESHADE;
  status.presentation_generation = 7;
  status.source_width = 3840;
  status.source_height = 2160;
  status.packed_width = 7680;
  status.packed_height = 2160;
  const video::game_source_identity_t source {42, 1234, 1};
  const auto now = video::game_source_tracker_t::clock_t::time_point {};
  ASSERT_TRUE(tracker.observe(status, source, now));
  status.presentation_generation = 8;
  auto renewed = tracker.observe(status, source, now + std::chrono::milliseconds(1));
  ASSERT_TRUE(renewed);
  EXPECT_EQ(renewed->presentation_generation, 8u);
  EXPECT_EQ(renewed->source_revision, 1u);
  status.presentation_generation = 0;
  EXPECT_FALSE(tracker.observe(status, source, now));
  status.presentation_generation = 9;
  status.packed_width = 3840;
  EXPECT_FALSE(tracker.observe(status, source, now));
}
