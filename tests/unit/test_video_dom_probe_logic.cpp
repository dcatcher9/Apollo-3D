/**
 * @file tests/unit/test_video_dom_probe_logic.cpp
 * @brief Tests for the diagnostics-only Chromium video DOM probe logic.
 */

#include "tools/video_dom_probe/probe_logic.h"

#include <array>
#include <gtest/gtest.h>

namespace {
  namespace probe = video_dom_probe;

  constexpr probe::rect_t browser {0, 0, 1920, 1080};

  [[nodiscard]] probe::video_candidate_t candidate(
    long id,
    probe::rect_t rect,
    bool available = true,
    bool contained = true
  ) {
    return {
      .unique_id = id,
      .element_rect = rect,
      .visible_rect = rect,
      .available = available,
      .fully_contained = contained,
      .credible_size = true,
    };
  }
}  // namespace

TEST(VideoDomProbeLogic, ParsesOnlyAnExactUnescapedVideoTag) {
  const auto ordinary = probe::match_video_tag(L"display:block;tag:video;class:player;");
  EXPECT_TRUE(ordinary.valid);
  EXPECT_TRUE(ordinary.is_video);

  const auto folded = probe::match_video_tag(L"TAG:VIDEO;");
  EXPECT_TRUE(folded.valid);
  EXPECT_TRUE(folded.is_video);

  const auto false_value = probe::match_video_tag(L"tag:videography;");
  EXPECT_TRUE(false_value.valid);
  EXPECT_FALSE(false_value.is_video);

  const auto embedded = probe::match_video_tag(L"description:tag\\:video;tag:div;");
  EXPECT_TRUE(embedded.valid);
  EXPECT_FALSE(embedded.is_video);
}

TEST(VideoDomProbeLogic, RejectsMalformedOrConflictingAttributes) {
  EXPECT_FALSE(probe::match_video_tag(L"").valid);
  EXPECT_FALSE(probe::match_video_tag(L"tag").valid);
  EXPECT_FALSE(probe::match_video_tag(L"tag:video\\").valid);
  EXPECT_FALSE(probe::match_video_tag(L"t\\ag:video;").valid);
  EXPECT_FALSE(probe::match_video_tag(L"tag:v\\ideo;").valid);
  EXPECT_FALSE(probe::match_video_tag(L"tag:video;tag:div;").valid);
  EXPECT_FALSE(probe::match_video_tag(std::wstring_view(L"tag:video\0tag:div", 17)).valid);
}

TEST(VideoDomProbeLogic, RectangleMathHandlesPartialAndNegativeCoordinates) {
  constexpr probe::rect_t monitor {-1920, 0, 0, 1080};
  constexpr probe::rect_t partial {-2000, 100, -100, 900};
  const auto clipped = probe::rect_intersection(monitor, partial);
  ASSERT_TRUE(clipped);
  EXPECT_EQ(*clipped, (probe::rect_t {-1920, 100, -100, 900}));
  EXPECT_FALSE(probe::rect_contains(monitor, partial));
  EXPECT_EQ(probe::rect_area(*clipped), 1820ULL * 800ULL);
}

TEST(VideoDomProbeLogic, ClipsOnlyOnePixelOfEndpointRounding) {
  constexpr probe::rect_t outer {0, 0, 1920, 1080};
  const auto clipped = probe::clip_if_within_tolerance(
    outer,
    probe::rect_t {-1, 20, 1921, 1060}
  );
  ASSERT_TRUE(clipped);
  EXPECT_EQ(*clipped, (probe::rect_t {0, 20, 1920, 1060}));
  EXPECT_FALSE(probe::clip_if_within_tolerance(
    outer,
    probe::rect_t {-2, 20, 1920, 1060}
  ));
}

TEST(VideoDomProbeLogic, SelectsOneAvailableFullyContainedVideo) {
  const std::array candidates {
    candidate(1, {200, 100, 1720, 955}),
    candidate(2, {0, 0, 0, 0}, false, false),
  };
  const auto selection = probe::select_candidate(candidates);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 0U);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::single);
}

TEST(VideoDomProbeLogic, SelectsTheUniqueLargestVideo) {
  const std::array candidates {
    candidate(10, {200, 100, 1720, 955}),
    candidate(20, {1500, 50, 1850, 250}),
  };
  const auto selection = probe::select_candidate(candidates);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 0U);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::largest);
}

TEST(VideoDomProbeLogic, EqualLargestVideosRemainAmbiguous) {
  const std::array candidates {
    candidate(10, {100, 100, 900, 700}),
    candidate(20, {1000, 100, 1800, 700}),
  };
  const auto selection = probe::select_candidate(candidates);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::ambiguous);
}

TEST(VideoDomProbeLogic, TinyVideoIsListedButNotSelected) {
  auto tiny = candidate(1, {10, 10, 20, 20});
  tiny.credible_size = false;
  const std::array candidates {tiny};
  const auto selection = probe::select_candidate(candidates);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::none);
}

TEST(VideoDomProbeLogic, RejectsOffscreenOrPartiallyClippedAuthority) {
  const std::array candidates {
    candidate(1, {-100, 100, 1000, 900}, true, false),
    candidate(2, {200, 100, 1200, 900}, false, true),
  };
  const auto selection = probe::select_candidate(candidates);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::ambiguous);
  EXPECT_TRUE(probe::rect_contains(browser, candidates[1].element_rect));
}

TEST(VideoDomProbeLogic, PartiallyClippedMainVideoCannotPromoteASmallerAd) {
  auto clipped_main = candidate(1, {-100, 100, 1400, 950}, true, false);
  clipped_main.visible_rect = {0, 100, 1400, 950};
  const std::array candidates {
    clipped_main,
    candidate(2, {1500, 100, 1850, 300}),
  };
  const auto selection = probe::select_candidate(candidates);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::ambiguous);
}

TEST(VideoDomProbeLogic, UnavailablePartiallyClippedMainStillPoisonsSmallerAd) {
  auto clipped_main = candidate(1, {-100, 100, 1400, 950}, false, false);
  clipped_main.visible_rect = {0, 100, 1400, 950};
  const std::array candidates {
    clipped_main,
    candidate(2, {1500, 100, 1850, 300}),
  };
  const auto selection = probe::select_candidate(candidates);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::ambiguous);
}
