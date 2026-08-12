/**
 * @file tests/unit/test_video_dom_probe_logic.cpp
 * @brief Tests for Chromium video DOM probe parsing, selection, and machine-mode policy.
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
      .fullscreen_available = available,
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

TEST(VideoDomProbeLogic, ObjectChurnBetweenScansPreservesStableSemanticsAndRequestsAudit) {
  const auto policy = probe::machine_event_policy(
    {.foreground = 4, .object = 100},
    {.foreground = 4, .object = 103}
  );

  EXPECT_FALSE(policy.discard_completed_scan);
  EXPECT_FALSE(policy.invalidate_semantic_state);
  EXPECT_TRUE(policy.request_follow_up_audit);
}

TEST(VideoDomProbeLogic, ObjectChurnDuringCompleteScanDoesNotDiscardObservation) {
  const auto policy = probe::machine_event_policy(
    {.foreground = 9, .object = 200},
    {.foreground = 9, .object = 201}
  );

  EXPECT_FALSE(policy.discard_completed_scan);
  EXPECT_FALSE(policy.invalidate_semantic_state);
  EXPECT_TRUE(policy.request_follow_up_audit);
}

TEST(VideoDomProbeLogic, ForegroundChangeDiscardsObservationAndSemanticStability) {
  const auto policy = probe::machine_event_policy(
    {.foreground = 12, .object = 300},
    {.foreground = 13, .object = 300}
  );

  EXPECT_TRUE(policy.discard_completed_scan);
  EXPECT_TRUE(policy.invalidate_semantic_state);
  EXPECT_TRUE(policy.request_follow_up_audit);
}

TEST(VideoDomProbeLogic, NoMachineEventsNeedNoFollowUpAudit) {
  const auto policy = probe::machine_event_policy(
    {.foreground = 20, .object = 400},
    {.foreground = 20, .object = 400}
  );

  EXPECT_FALSE(policy.discard_completed_scan);
  EXPECT_FALSE(policy.invalidate_semantic_state);
  EXPECT_FALSE(policy.request_follow_up_audit);
}

TEST(VideoDomProbeLogic, CompleteUniqueCensusStagesUnpublishedProvisionalSelection) {
  const probe::selection_t selection {
    .index = 0,
    .reason = probe::selection_reason_e::single,
  };

  const auto policy = probe::complete_census_policy(selection, false);
  EXPECT_TRUE(policy.stage_selection);
  EXPECT_FALSE(policy.revoke_cached_selection);
  EXPECT_EQ(policy.phase, probe::cached_selection_phase_e::provisional);
  EXPECT_FALSE(policy.publish_ok);
}

TEST(VideoDomProbeLogic, MatchingEstablishedSelectionRetainsEstablishedAuthority) {
  const probe::selection_t selection {
    .index = 1,
    .reason = probe::selection_reason_e::largest,
  };

  const auto policy = probe::complete_census_policy(selection, true);
  EXPECT_TRUE(policy.stage_selection);
  EXPECT_FALSE(policy.revoke_cached_selection);
  EXPECT_EQ(policy.phase, probe::cached_selection_phase_e::established);
  EXPECT_TRUE(policy.publish_ok);
}

TEST(VideoDomProbeLogic, CompleteNegativeOrAmbiguousCensusRevokesCachedSelection) {
  for (const auto selection : std::array {
         probe::selection_t {},
         probe::selection_t {.reason = probe::selection_reason_e::ambiguous},
       }) {
    const auto policy = probe::complete_census_policy(selection, false);
    EXPECT_FALSE(policy.stage_selection);
    EXPECT_TRUE(policy.revoke_cached_selection);
    EXPECT_FALSE(policy.publish_ok);
  }
}

TEST(VideoDomProbeLogic, NextTickExactRefreshPromotesProvisionalSelection) {
  const auto policy = probe::cached_refresh_policy(
    probe::cached_selection_phase_e::provisional,
    true,
    probe::machine_event_policy({.foreground = 3, .object = 8},
                                {.foreground = 3, .object = 8})
  );

  EXPECT_TRUE(policy.retain_selection);
  EXPECT_EQ(policy.next_phase, probe::cached_selection_phase_e::established);
  EXPECT_TRUE(policy.publish_ok);
  EXPECT_FALSE(policy.request_follow_up_audit);
}

TEST(VideoDomProbeLogic, FailedExactRefreshDropsProvisionalSelection) {
  const auto policy = probe::cached_refresh_policy(
    probe::cached_selection_phase_e::provisional,
    false,
    {}
  );

  EXPECT_FALSE(policy.retain_selection);
  EXPECT_FALSE(policy.publish_ok);
}

TEST(VideoDomProbeLogic, ObjectChurnDoesNotBlockProvisionalPromotion) {
  const auto events = probe::machine_event_policy(
    {.foreground = 5, .object = 10},
    {.foreground = 5, .object = 11}
  );
  const auto policy = probe::cached_refresh_policy(
    probe::cached_selection_phase_e::provisional,
    true,
    events
  );

  EXPECT_TRUE(policy.retain_selection);
  EXPECT_EQ(policy.next_phase, probe::cached_selection_phase_e::established);
  EXPECT_TRUE(policy.publish_ok);
  EXPECT_TRUE(policy.request_follow_up_audit);
}

TEST(VideoDomProbeLogic, ForegroundChangeVetoesProvisionalPromotion) {
  const auto events = probe::machine_event_policy(
    {.foreground = 5, .object = 10},
    {.foreground = 6, .object = 10}
  );
  const auto policy = probe::cached_refresh_policy(
    probe::cached_selection_phase_e::provisional,
    true,
    events
  );

  EXPECT_FALSE(policy.retain_selection);
  EXPECT_FALSE(policy.publish_ok);
  EXPECT_TRUE(policy.request_follow_up_audit);
}

TEST(VideoDomProbeLogic, EstablishedSelectionStaysEstablishedAfterExactRefresh) {
  const auto policy = probe::cached_refresh_policy(
    probe::cached_selection_phase_e::established,
    true,
    {}
  );

  EXPECT_TRUE(policy.retain_selection);
  EXPECT_EQ(policy.next_phase, probe::cached_selection_phase_e::established);
  EXPECT_TRUE(policy.publish_ok);
}

TEST(VideoDomProbeLogic, UncachedIncompleteRetriesFastButAccessibilityBacksOff) {
  EXPECT_EQ(
    probe::uncached_scan_retry_delay(
      probe::uncached_scan_outcome_e::traversal_incomplete
    ),
    std::chrono::milliseconds {1000}
  );
  EXPECT_EQ(
    probe::uncached_scan_retry_delay(
      probe::uncached_scan_outcome_e::accessibility_unavailable
    ),
    std::chrono::milliseconds {15000}
  );
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

TEST(VideoDomProbeLogic, DuplicateFullClientVideosAuthorizeFullscreen) {
  const std::array candidates {
    candidate(20, {0, 0, 1920, 1080}),
    candidate(10, {0, 0, 1920, 1080}),
  };

  const auto selection = probe::select_authority_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 1U);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::fullscreen);
  EXPECT_EQ(
    probe::selection_authority_rect(candidates, selection, browser),
    std::optional<probe::rect_t> {browser}
  );
}

TEST(VideoDomProbeLogic, FullClientSelectionIsIndependentOfTraversalOrder) {
  const std::array forward {
    candidate(30, {0, 0, 1920, 1080}),
    candidate(10, {-20, -10, 1940, 1090}, true, false),
  };
  const std::array reverse {forward[1], forward[0]};

  const auto forward_selection = probe::select_authority_candidate(forward, browser);
  const auto reverse_selection = probe::select_authority_candidate(reverse, browser);
  ASSERT_TRUE(forward_selection.index);
  ASSERT_TRUE(reverse_selection.index);
  EXPECT_EQ(forward[*forward_selection.index].unique_id, 10);
  EXPECT_EQ(reverse[*reverse_selection.index].unique_id, 10);
  EXPECT_EQ(forward_selection.reason, probe::selection_reason_e::fullscreen);
  EXPECT_EQ(reverse_selection.reason, probe::selection_reason_e::fullscreen);
  EXPECT_EQ(
    probe::selection_authority_rect(forward, forward_selection, browser),
    probe::selection_authority_rect(reverse, reverse_selection, browser)
  );
}

TEST(VideoDomProbeLogic, FullClientVideoBeatsUnrelatedPartialCandidate) {
  auto partial = candidate(1, {-100, 100, 1400, 950}, true, false);
  partial.visible_rect = {0, 100, 1400, 950};
  const std::array candidates {
    partial,
    candidate(2, {0, 0, 1920, 1080}),
  };

  const auto selection = probe::select_authority_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 1U);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::fullscreen);
}

TEST(VideoDomProbeLogic, DocumentClippingDoesNotDisableFullClientVideo) {
  auto document_clipped = candidate(7, {-30, -20, 1950, 1100}, false, false);
  document_clipped.visible_rect = {100, 80, 1820, 1000};
  document_clipped.fullscreen_available =
    probe::fullscreen_semantic_available(true, true, true);
  const std::array candidates {document_clipped};

  const auto selection = probe::select_authority_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::fullscreen);
  EXPECT_EQ(
    probe::selection_authority_rect(candidates, selection, browser),
    std::optional<probe::rect_t> {browser}
  );
}

TEST(VideoDomProbeLogic, UnavailableDocumentCannotAuthorizeFullClientVideo) {
  auto unavailable_document = candidate(7, {0, 0, 1920, 1080}, false, true);
  // The element itself may be geometrically full-client, but census collection leaves this false
  // when its owning document has an unavailable IA2 state.
  unavailable_document.fullscreen_available =
    probe::fullscreen_semantic_available(true, true, false);
  const std::array candidates {unavailable_document};

  const auto selection = probe::select_authority_candidate(candidates, browser);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::none);
}

TEST(VideoDomProbeLogic, UnavailableVideoCannotAuthorizeFullClientVideo) {
  auto unavailable_video = candidate(8, {0, 0, 1920, 1080}, false, true);
  unavailable_video.fullscreen_available =
    probe::fullscreen_semantic_available(false, true, true);
  const std::array candidates {unavailable_video};

  const auto selection = probe::select_authority_candidate(candidates, browser);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::none);
}

TEST(VideoDomProbeLogic, MaximizedBrowserDoesNotPromoteNonCoveringPageVideo) {
  const std::array candidates {
    candidate(9, {0, 120, 1920, 1080}),
  };

  const auto selection = probe::select_authority_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::single);
  const std::optional<probe::rect_t> expected {
    probe::rect_t {0, 120, 1920, 1080}
  };
  EXPECT_EQ(
    probe::selection_authority_rect(candidates, selection, browser),
    expected
  );
}

TEST(VideoDomProbeLogic, EqualWindowedRoiCandidatesRemainAmbiguous) {
  const std::array candidates {
    candidate(10, {100, 100, 900, 700}),
    candidate(20, {1000, 100, 1800, 700}),
  };

  const auto selection = probe::select_authority_candidate(candidates, browser);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::ambiguous);
}

TEST(VideoDomProbeLogic, CompleteFullscreenCensusCanStageAuthority) {
  const probe::selection_t selection {
    .index = 0,
    .reason = probe::selection_reason_e::fullscreen,
  };

  const auto policy = probe::complete_census_policy(selection, false);
  EXPECT_TRUE(policy.stage_selection);
  EXPECT_FALSE(policy.revoke_cached_selection);
  EXPECT_EQ(policy.phase, probe::cached_selection_phase_e::provisional);
  EXPECT_FALSE(policy.publish_ok);
}

TEST(VideoDomProbeLogic, FullscreenCacheRefreshAllowsDocumentAndOverscanChanges) {
  // Document geometry is intentionally absent from this policy. Only the retained video
  // identity, stable client authority, and continued full-client coverage are relevant.
  EXPECT_TRUE(probe::fullscreen_cache_refresh_matches(
    42,
    42,
    browser,
    browser,
    {-40, -25, 1960, 1110}
  ));
  EXPECT_TRUE(probe::fullscreen_cache_refresh_matches(
    42,
    42,
    browser,
    browser,
    {0, 0, 1920, 1080}
  ));
}

TEST(VideoDomProbeLogic, FullscreenCacheRefreshRejectsIdentityOrAuthorityChange) {
  EXPECT_FALSE(probe::fullscreen_cache_refresh_matches(
    42,
    43,
    browser,
    browser,
    {0, 0, 1920, 1080}
  ));
  EXPECT_FALSE(probe::fullscreen_cache_refresh_matches(
    42,
    42,
    browser,
    {10, 10, 1930, 1090},
    {0, 0, 1930, 1090}
  ));
  EXPECT_FALSE(probe::fullscreen_cache_refresh_matches(
    42,
    42,
    browser,
    browser,
    {2, 0, 1920, 1080}
  ));
}
