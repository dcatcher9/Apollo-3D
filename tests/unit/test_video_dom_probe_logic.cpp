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

TEST(VideoDomProbeLogic, LongCensusRetainsProgressWhileServicingFreshHeartbeats) {
  using namespace std::chrono_literals;
  std::chrono::steady_clock::duration elapsed {};
  std::chrono::steady_clock::duration last_service {};
  std::chrono::steady_clock::duration last_heartbeat {};
  std::size_t checkpoints = 0;
  // This complete walk takes ten seconds. Restarting at the former two-second budget
  // would never reach its final candidate; a service checkpoint must preserve its progress.
  for (std::size_t visited = 0; visited < 5000; ++visited) {
    elapsed += 2ms;
    const auto action = probe::census_checkpoint_policy(
      elapsed, elapsed - last_service, visited, 20000, 15s, 100ms
    );
    ASSERT_NE(action, probe::census_checkpoint_e::exhausted);
    if (action == probe::census_checkpoint_e::service) {
      ++checkpoints;
      last_service = elapsed;
      if (elapsed - last_heartbeat >= 1s) {
        last_heartbeat = elapsed;
      }
    }
    EXPECT_LT(elapsed - last_heartbeat, 2500ms);
  }
  EXPECT_EQ(elapsed, 10s);
  EXPECT_EQ(checkpoints, 100u);
  EXPECT_EQ(last_heartbeat, elapsed);
  // Completion still stages only an unpublished selection. Servicing was not authority.
  const auto census = probe::complete_census_policy(
    {.index = 0, .reason = probe::selection_reason_e::single}, false
  );
  EXPECT_TRUE(census.stage_selection);
  EXPECT_FALSE(census.publish_ok);
  EXPECT_FALSE(probe::cached_refresh_policy(census.phase, false, {}).publish_ok);
  EXPECT_TRUE(probe::cached_refresh_policy(census.phase, true, {}).publish_ok);
}

TEST(VideoDomProbeLogic, CensusLifetimeAndNodeLimitsTakePrecedenceOverService) {
  using namespace std::chrono_literals;
  EXPECT_EQ(probe::census_checkpoint_policy(99ms, 99ms, 99, 20000, 15s, 100ms),
    probe::census_checkpoint_e::continue_walk);
  EXPECT_EQ(probe::census_checkpoint_policy(100ms, 100ms, 100, 20000, 15s, 100ms),
    probe::census_checkpoint_e::service);
  EXPECT_EQ(probe::census_checkpoint_policy(14999ms, 99ms, 19999, 20000, 15s, 100ms),
    probe::census_checkpoint_e::continue_walk);
  EXPECT_EQ(probe::census_checkpoint_policy(15s, 100ms, 1000, 20000, 15s, 100ms),
    probe::census_checkpoint_e::exhausted);
  EXPECT_EQ(probe::census_checkpoint_policy(1s, 100ms, 20000, 20000, 15s, 100ms),
    probe::census_checkpoint_e::exhausted);
}

TEST(VideoDomProbeLogic, ForegroundAbaDuringLongCensusCannotRefreshOrPublishAuthority) {
  using namespace std::chrono_literals;
  ASSERT_EQ(probe::census_checkpoint_policy(3s, 100ms, 1500, 20000, 15s, 100ms),
    probe::census_checkpoint_e::service);
  // The root may have returned to the same HWND/client. The pumped foreground-event
  // generation still proves the intervening authority loss and vetoes the unfinished walk.
  const auto events = probe::machine_event_policy(
    {.foreground = 5, .object = 10}, {.foreground = 7, .object = 12}
  );
  EXPECT_TRUE(events.discard_completed_scan);
  EXPECT_TRUE(events.invalidate_semantic_state);
  for (const auto phase : {probe::cached_selection_phase_e::provisional,
                          probe::cached_selection_phase_e::established}) {
    const auto refreshed = probe::cached_refresh_policy(phase, true, events);
    EXPECT_FALSE(refreshed.retain_selection);
    EXPECT_FALSE(refreshed.publish_ok);
  }
}

TEST(VideoDomProbeLogic, CensusServiceRevalidatesAfterComRefreshBeforePublishing) {
  for (const bool foreground_aba : {false, true}) {
    const probe::machine_event_generations_t before {.foreground = 5, .object = 10};
    auto after = before;
    bool foreground_matches = true;
    bool refreshed = false;
    bool published = false;
    const bool serviced = probe::service_census_checkpoint([&]() {
      // Exact COM refresh succeeds, but a foreground transition or client change occurs
      // while it is executing. The tentative positive must not become a heartbeat.
      refreshed = true;
      if (foreground_aba) after.foreground += 2;
      else foreground_matches = false;
      return probe::cached_refresh_policy(probe::cached_selection_phase_e::established, true, {});
    }, [&]() {
      EXPECT_TRUE(refreshed);
      return foreground_matches && !probe::machine_event_policy(before, after).invalidate_semantic_state;
    }, [&](const auto &) {
      published = true;
    });
    EXPECT_TRUE(refreshed);
    EXPECT_FALSE(serviced);
    EXPECT_FALSE(published);
  }
}

TEST(VideoDomProbeLogic, CensusServicePublishesOnlyAfterSuccessfulValidation) {
  bool refreshed = false;
  bool validated = false;
  bool published = false;
  EXPECT_TRUE(probe::service_census_checkpoint([&]() {
    refreshed = true;
    return probe::cached_refresh_policy(probe::cached_selection_phase_e::established, true, {});
  }, [&]() {
    EXPECT_TRUE(refreshed);
    validated = true;
    return true;
  }, [&](const auto &record) {
    EXPECT_TRUE(validated);
    EXPECT_TRUE(record.publish_ok);
    published = true;
  }));
  EXPECT_TRUE(published);
}

TEST(VideoDomProbeLogic, ResizeReacquiresAuthorityOnlyAfterNewCensusAndExactRefresh) {
  constexpr probe::rect_t old_client {1449, 295, 3568, 2032};
  constexpr probe::rect_t old_video {1626, 704, 3367, 1680};
  constexpr probe::rect_t new_client {1922, 0, 3838, 2086};
  constexpr probe::rect_t new_document {1922, 100, 3838, 2086};
  constexpr probe::rect_t new_video {2000, 400, 3760, 1390};
  ASSERT_TRUE(probe::windowed_cache_refresh_matches(
    old_video, old_video, old_client, old_client, old_video
  ));

  const bool old_cache_matches = probe::windowed_cache_refresh_matches(
    old_video, old_video, new_client, new_document, new_video
  );
  const auto dropped = probe::cached_refresh_policy(
    probe::cached_selection_phase_e::established, old_cache_matches, {}
  );
  EXPECT_FALSE(dropped.retain_selection);
  EXPECT_FALSE(dropped.publish_ok);

  // The same semantic video can be selected again at its new position, but a complete
  // census alone must not publish the replacement geometry as established authority.
  const std::array candidates {candidate(-262467, new_video)};
  const auto selection = probe::select_authority_candidate(candidates, new_client);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::single);
  const auto authority = probe::selection_authority_rect(candidates, selection, new_client);
  ASSERT_TRUE(authority);
  EXPECT_EQ(*authority, new_video);
  const auto census = probe::complete_census_policy(selection, false);
  ASSERT_TRUE(census.stage_selection);
  EXPECT_FALSE(census.revoke_cached_selection);
  EXPECT_EQ(census.phase, probe::cached_selection_phase_e::provisional);
  EXPECT_FALSE(census.publish_ok);

  // A stale retained object still reporting the pre-resize rectangle cannot promote it.
  const auto stale = probe::cached_refresh_policy(
    census.phase,
    probe::windowed_cache_refresh_matches(new_video, *authority, new_client, new_document, old_video),
    {}
  );
  EXPECT_FALSE(stale.retain_selection);
  EXPECT_FALSE(stale.publish_ok);

  const auto repeated_census = probe::complete_census_policy(selection, false);
  ASSERT_TRUE(repeated_census.stage_selection);
  EXPECT_FALSE(repeated_census.publish_ok);
  const auto refreshed = probe::cached_refresh_policy(
    repeated_census.phase,
    probe::windowed_cache_refresh_matches(new_video, *authority, new_client, new_document, new_video),
    probe::machine_event_policy({.foreground = 5, .object = 10},
                                {.foreground = 5, .object = 11})
  );
  EXPECT_TRUE(refreshed.retain_selection);
  EXPECT_EQ(refreshed.next_phase, probe::cached_selection_phase_e::established);
  EXPECT_TRUE(refreshed.publish_ok);
  EXPECT_TRUE(refreshed.request_follow_up_audit);
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
  const auto selection = probe::select_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 0U);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::single);
}

TEST(VideoDomProbeLogic, SelectsTheUniqueLargestVideo) {
  const std::array candidates {
    candidate(10, {200, 100, 1720, 955}),
    candidate(20, {1500, 50, 1850, 250}),
  };
  const auto selection = probe::select_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 0U);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::largest);
}

TEST(VideoDomProbeLogic, EqualLargestVideosRemainAmbiguous) {
  const std::array candidates {
    candidate(10, {100, 100, 900, 700}),
    candidate(20, {1000, 100, 1800, 700}),
  };
  const auto selection = probe::select_candidate(candidates, browser);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::ambiguous);
}

TEST(VideoDomProbeLogic, TinyVideoIsListedButNotSelected) {
  auto tiny = candidate(1, {10, 10, 20, 20});
  tiny.credible_size = false;
  const std::array candidates {tiny};
  const auto selection = probe::select_candidate(candidates, browser);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::none);
}

TEST(VideoDomProbeLogic, RejectsUnavailableOrFullyOccludedWindowedVideo) {
  const std::array candidates {
    candidate(1, {0, 0, 0, 0}, true, false),
    candidate(2, {200, 100, 1200, 900}, false, true),
  };
  const auto selection = probe::select_candidate(candidates, browser);
  EXPECT_FALSE(selection.index);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::none);
  EXPECT_TRUE(probe::rect_contains(browser, candidates[1].element_rect));
}

TEST(VideoDomProbeLogic, PartiallyClippedMainVideoCannotPromoteASmallerAd) {
  auto clipped_main = candidate(1, {-1300, 100, 200, 950}, true, false);
  clipped_main.visible_rect = {0, 100, 200, 950};
  const std::array candidates {
    clipped_main,
    candidate(2, {1000, 200, 1600, 550}),
  };
  const auto selection = probe::select_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 0u);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::largest);
  EXPECT_LT(probe::rect_area(clipped_main.visible_rect), probe::rect_area(candidates[1].visible_rect));
}

TEST(VideoDomProbeLogic, UnavailableVideoCannotAuthorizeOrPoisonAvailableVideo) {
  auto clipped_main = candidate(1, {-100, 100, 1400, 950}, false, false);
  clipped_main.visible_rect = {0, 100, 1400, 950};
  const std::array candidates {
    clipped_main,
    candidate(2, {1400, 100, 1800, 400}),
  };
  const auto selection = probe::select_candidate(candidates, browser);
  ASSERT_TRUE(selection.index);
  EXPECT_EQ(*selection.index, 1u);
  EXPECT_EQ(selection.reason, probe::selection_reason_e::single);
}

TEST(VideoDomProbeLogic, BrowserWidthVideoPublishesItsExactDocumentClippedViewport) {
  constexpr probe::rect_t client {0, 0, 3840, 2088};
  constexpr probe::rect_t document {0, 132, 3840, 2088};
  constexpr probe::rect_t element {277, 85, 2813, 1512};
  constexpr probe::rect_t expected {277, 132, 2813, 1512};
  const auto visible = probe::windowed_video_viewport(client, document, element);
  ASSERT_TRUE(visible);
  EXPECT_EQ(*visible, expected);
  auto video = candidate(-764704, element, true, false);
  video.visible_rect = *visible;
  const std::array candidates {video};
  const auto selected = probe::select_authority_candidate(candidates, client);
  ASSERT_TRUE(selected.index);
  EXPECT_EQ(selected.reason, probe::selection_reason_e::single);
  EXPECT_EQ(probe::selection_authority_rect(candidates, selected, client), visible);
  EXPECT_TRUE(probe::windowed_cache_refresh_matches(element, expected, client, document, element));
  // The document may overscan the client without changing these exact visible video pixels.
  EXPECT_TRUE(probe::windowed_cache_refresh_matches(element, expected, client, {-500, 132, 4500, 2200}, element));
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(element, expected, client, {0, 133, 3840, 2088}, element));
}

TEST(VideoDomProbeLogic, EqualFullElementAreasRemainAmbiguousWhenClippingDiffers) {
  auto first = candidate(1, {-100, 0, 1100, 900}, true, false);
  first.visible_rect = {0, 0, 1100, 900};
  auto second = candidate(2, {1000, 0, 2200, 900}, true, false);
  second.visible_rect = {1000, 0, 1920, 900};
  const std::array candidates {first, second};
  const auto selected = probe::select_authority_candidate(candidates, browser);
  EXPECT_FALSE(selected.index);
  EXPECT_EQ(selected.reason, probe::selection_reason_e::ambiguous);
}

TEST(VideoDomProbeLogic, TinyVisibleSliverDoesNotPromoteSmallerAdvert) {
  auto main = candidate(1, {-1900, 100, 20, 1000}, true, false);
  main.visible_rect = {0, 100, 20, 1000};
  const std::array candidates {main, candidate(2, {1000, 100, 1500, 500})};
  const auto selected = probe::select_authority_candidate(candidates, browser);
  EXPECT_FALSE(selected.index);
  EXPECT_EQ(selected.reason, probe::selection_reason_e::none);
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(main.element_rect, main.visible_rect, browser, browser, main.element_rect));
}

TEST(VideoDomProbeLogic, VisibleViewportMustMeetExistingFivePercentThreshold) {
  constexpr probe::rect_t client {0, 0, 1000, 1000};
  EXPECT_TRUE(probe::credible_video_extent({0, 0, 1000, 50}, client));
  EXPECT_FALSE(probe::credible_video_extent({0, 0, 1000, 49}, client));
  EXPECT_FALSE(probe::windowed_video_viewport(client, {0, 0, 0, 1000}, client));
  EXPECT_FALSE(probe::windowed_video_viewport(client, client, {1200, 0, 1500, 1000}));
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

TEST(VideoDomProbeLogic, WindowedCacheRefreshKeepsStableVideoThroughDocumentLayoutChanges) {
  constexpr probe::rect_t video {200, 150, 1480, 870};
  EXPECT_TRUE(probe::windowed_cache_refresh_matches(video, video, browser, browser, video));
  // A document header/footer can move while the same video retains its exact screen authority.
  EXPECT_TRUE(probe::windowed_cache_refresh_matches(
    video, video, browser, {0, 40, 1920, 1040}, video
  ));
  EXPECT_TRUE(probe::windowed_cache_refresh_matches(video, video, browser, {20, 0, 1900, 1080}, video));
}

TEST(VideoDomProbeLogic, WindowedCacheRefreshRejectsVideoMotionOrVisibleViewportChanges) {
  constexpr probe::rect_t video {200, 150, 1480, 870};
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(video, video, browser, browser, {201, 150, 1481, 870}));
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(video, video, browser, {0, 151, 1920, 1080}, video));
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(video, video, browser, {0, 152, 1920, 1080}, video));
  EXPECT_TRUE(probe::windowed_cache_refresh_matches(video, video, browser, {-2, 0, 1920, 1080}, video));
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(video, video, browser, {0, 0, 0, 1080}, video));
  constexpr probe::rect_t tiny {200, 150, 210, 160};
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(tiny, tiny, browser, browser, tiny));
}

TEST(VideoDomProbeLogic, WindowedCacheRefreshRequiresExactVisibleIntersection) {
  constexpr probe::rect_t edge_video {-1, 100, 1000, 800};
  constexpr probe::rect_t visible {0, 100, 1000, 800};
  EXPECT_TRUE(probe::windowed_cache_refresh_matches(edge_video, visible, browser, browser, edge_video));
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(edge_video, edge_video, browser, browser, edge_video));
  EXPECT_FALSE(probe::windowed_cache_refresh_matches(edge_video, visible, browser, {1, 0, 1920, 1080}, edge_video));
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
