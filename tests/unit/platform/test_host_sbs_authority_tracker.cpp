/**
 * @file tests/unit/platform/test_host_sbs_authority_tracker.cpp
 * @brief The Host 3D authority tracker decides from its own state and reports depth effects.
 */
#include "../../tests_common.h"

#include <src/platform/windows/host_sbs_authority_tracker.h>

namespace {
  using platf::dxgi::host_sbs_authority_tracker_t;
  using platf::dxgi::host_sbs_region_attribution_t;
  namespace subtitle = models::host_subtitle_context;

  D3D11_TEXTURE2D_DESC source(const UINT width, const UINT height) {
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    return desc;
  }

  bool any(const host_sbs_authority_tracker_t::effects_t &effects) {
    return effects.depth_reprocess || effects.window_changed || effects.browser_changed ||
           effects.subtitle_context_changed;
  }
}  // namespace

TEST(HostSbsAuthorityTrackerTest, FixtureObservationClaimsOnlyItsSyntheticContext) {
  host_sbs_authority_tracker_t tracker;
  const platf::dxgi::converter_test_observer::context_t context {.width = 64, .height = 32, .instance = 7};
  tracker.set_test_context(&context);

  const auto matched = tracker.observe(source(64, 32), std::nullopt, {});
  EXPECT_FALSE(any(matched.effects));
  EXPECT_EQ(tracker.subtitle_context().kind(), subtitle::kind_e::registered_video_player);
  const auto generation = tracker.subtitle_context().generation();
  EXPECT_NE(generation, 0u);
  EXPECT_EQ(tracker.presentation_authority(), subtitle::presentation_authority_t {});

  // The same fixture source keeps its context; another raster loses it.
  (void) tracker.observe(source(64, 32), std::nullopt, {});
  EXPECT_EQ(tracker.subtitle_context().generation(), generation);
  (void) tracker.observe(source(65, 32), std::nullopt, {});
  EXPECT_FALSE(tracker.subtitle_context().eligible());
  EXPECT_NE(tracker.subtitle_context().generation(), generation);
}

TEST(HostSbsAuthorityTrackerTest, FixtureRenderAuthorityFollowsTheContextInstance) {
  host_sbs_authority_tracker_t tracker;
  const platf::dxgi::converter_test_observer::context_t context {.width = 64, .height = 32, .instance = 7};
  tracker.set_test_context(&context);
  const auto desc = source(64, 32);
  (void) tracker.observe(desc, std::nullopt, {});

  host_sbs_region_attribution_t slot;
  tracker.select_depth_input_region(slot, desc);
  tracker.attribute_subtitles(slot, desc);
  EXPECT_EQ(slot.test_instance, 7u);
  EXPECT_TRUE(slot.subtitle_context_eligible);
  EXPECT_EQ(slot.subtitle_context_generation, tracker.subtitle_context().generation());
  EXPECT_TRUE(tracker.window_region_authorized_for_render(slot));

  slot.test_instance = 8u;
  EXPECT_FALSE(tracker.window_region_authorized_for_render(slot));
  slot.test_instance = 7u;
  tracker.revoke_subtitle_context();
  EXPECT_FALSE(tracker.window_region_authorized_for_render(slot));
}

TEST(HostSbsAuthorityTrackerTest, NoWindowRegionSelectsTheCompleteSource) {
  host_sbs_authority_tracker_t tracker;
  host_sbs_region_attribution_t slot;
  tracker.select_depth_input_region(slot, source(1920, 1080));
  const auto &region = slot.depth_input_region;
  EXPECT_TRUE(region.valid());
  EXPECT_FALSE(region.is_video_region());
  EXPECT_EQ(region.authority, models::depth_analysis_authority_e::full_source);
  EXPECT_EQ(region.right, 1920u);
  EXPECT_EQ(region.bottom, 1080u);
  EXPECT_EQ(region.analysis_generation, 0u);
  EXPECT_FALSE(slot.depth_video_plan.has_value());
  EXPECT_FALSE(slot.window_region.has_value());
}

TEST(HostSbsAuthorityTrackerTest, ResetForgetsTheSubtitleContext) {
  host_sbs_authority_tracker_t tracker;
  const platf::dxgi::converter_test_observer::context_t context {.width = 64, .height = 32, .instance = 7};
  tracker.set_test_context(&context);
  (void) tracker.observe(source(64, 32), std::nullopt, {});
  ASSERT_TRUE(tracker.subtitle_context().eligible());
  tracker.reset();
  EXPECT_FALSE(tracker.subtitle_context().eligible());
  EXPECT_EQ(tracker.subtitle_context().generation(), subtitle::tracker_t {}.generation());
  EXPECT_FALSE(tracker.registry_ready_observed());
  EXPECT_EQ(tracker.presentation_authority(), subtitle::presentation_authority_t {});
}
