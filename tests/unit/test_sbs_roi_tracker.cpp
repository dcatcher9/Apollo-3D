/**
 * @file tests/unit/test_sbs_roi_tracker.cpp
 * @brief Tests for deterministic Host SBS ROI candidate tracking.
 */
#include "../tests_common.h"

#include <cstdint>
#include <limits>
#include <src/sbs_roi_tracker.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
  constexpr std::uint64_t detector_period_us = 100000;

  sbs_roi::tracker_config_t fast_config() {
    sbs_roi::tracker_config_t config;
    config.acquire_updates = 2;
    config.challenger_updates = 3;
    config.release_updates = 4;
    config.scroll_enter_updates = 2;
    config.scroll_exit_updates = 2;
    config.scroll_reacquire_timeout_updates = 4;
    config.invalid_input_timeout_updates = 3;
    config.max_continuity_gap_us = 250000;
    config.acquire_duration_us = detector_period_us;
    config.challenger_duration_us = 2 * detector_period_us;
    config.release_duration_us = 3 * detector_period_us;
    config.scroll_enter_duration_us = detector_period_us;
    config.scroll_exit_duration_us = detector_period_us;
    config.scroll_reacquire_timeout_us = 3 * detector_period_us;
    return config;
  }

  sbs_roi::candidate_t video_candidate(
    sbs_roi::normalized_rect_t rect,
    float motion = 0.7f,
    float density = 0.7f
  ) {
    sbs_roi::candidate_t candidate;
    candidate.kind = sbs_roi::roi_kind_e::video;
    candidate.rect = rect;
    candidate.temporal_occupancy = motion;
    candidate.photographic_density = density;
    candidate.primary_column_support = 0.9f;
    candidate.gutter_confidence = 0.9f;
    candidate.inside_primary_column = true;
    return candidate;
  }

  sbs_roi::candidate_t content_candidate(
    sbs_roi::normalized_rect_t rect,
    float density = 0.7f
  ) {
    auto candidate = video_candidate(rect, 0.0f, density);
    candidate.kind = sbs_roi::roi_kind_e::content;
    return candidate;
  }

  sbs_roi::observation_t observation(
    std::uint64_t id,
    std::vector<sbs_roi::candidate_t> candidates,
    std::uint64_t timestamp_us = 0,
    std::uint64_t arrival_timestamp_us = 0
  ) {
    sbs_roi::observation_t result;
    result.id = id;
    result.timestamp_us =
      timestamp_us == 0 ? id * detector_period_us : timestamp_us;
    result.arrival_timestamp_us =
      arrival_timestamp_us == 0 ?
        result.timestamp_us :
        arrival_timestamp_us;
    result.candidates = std::move(candidates);
    return result;
  }

  sbs_roi::tracker_output_t acquire(
    sbs_roi::tracker_t &tracker,
    const sbs_roi::candidate_t &candidate,
    std::uint64_t first_id = 1
  ) {
    tracker.update(observation(first_id, {candidate}));
    return tracker.update(observation(first_id + 1, {candidate}));
  }
}  // namespace

TEST(SbsRoiTracker, PrimaryVideoBeatsFasterBrighterSidebarAd) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto main =
    video_candidate({0.10f, 0.12f, 0.74f, 0.88f}, 0.35f, 0.65f);
  auto sidebar_ad =
    video_candidate({0.80f, 0.12f, 0.98f, 0.62f}, 1.0f, 1.0f);
  sidebar_ad.inside_primary_column = false;
  sidebar_ad.primary_column_support = 0.1f;

  tracker.update(observation(1, {sidebar_ad, main}));
  const auto result = tracker.update(observation(2, {main, sidebar_ad}));

  EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::video);
  EXPECT_FLOAT_EQ(result.rect.left, main.rect.left);
  EXPECT_FLOAT_EQ(result.rect.right, main.rect.right);
  EXPECT_EQ(result.generation, 1u);
  EXPECT_TRUE(result.committed_this_update);
}

TEST(SbsRoiTracker, StrongerContentBeatsEligibleAutoplayVideoAcrossKinds) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto main_content =
    content_candidate({0.06f, 0.08f, 0.76f, 0.92f}, 0.82f);
  const auto autoplay =
    video_candidate({0.72f, 0.15f, 0.98f, 0.75f}, 1.0f, 1.0f);

  tracker.update(observation(1, {autoplay, main_content}));
  const auto result =
    tracker.update(observation(2, {main_content, autoplay}));

  EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::content);
  EXPECT_FLOAT_EQ(result.rect.left, main_content.rect.left);
  EXPECT_FLOAT_EQ(result.rect.right, main_content.rect.right);
}

TEST(SbsRoiTracker, StaticOuterRailIsNotAcquiredButCentralCollageIs) {
  const auto static_ad =
    content_candidate({0.70f, 0.04f, 0.99f, 0.94f}, 1.0f);
  sbs_roi::tracker_t ad_only(fast_config());
  for (std::uint64_t id = 1; id <= 8; ++id) {
    ad_only.update(observation(id, {static_ad}));
  }
  EXPECT_EQ(ad_only.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(ad_only.snapshot().generation, 0u);

  const auto collage =
    content_candidate({0.05f, 0.08f, 0.82f, 0.92f}, 0.82f);
  sbs_roi::tracker_t central(fast_config());
  const auto acquired = acquire(central, collage);
  EXPECT_EQ(acquired.kind, sbs_roi::roi_kind_e::content);
  EXPECT_FLOAT_EQ(acquired.rect.left, collage.rect.left);
  EXPECT_FLOAT_EQ(acquired.rect.right, collage.rect.right);
}

TEST(SbsRoiTracker, RecentInteractionCanSelectAStaticOuterRail) {
  auto selected =
    content_candidate({0.70f, 0.04f, 0.99f, 0.94f}, 1.0f);
  selected.recent_interaction = true;
  sbs_roi::tracker_t tracker(fast_config());

  const auto acquired = acquire(tracker, selected);

  EXPECT_EQ(acquired.kind, sbs_roi::roi_kind_e::content);
  EXPECT_FLOAT_EQ(acquired.rect.left, selected.rect.left);
}

TEST(SbsRoiTracker, EqualVideosRemainAmbiguousRegardlessOfCandidateOrder) {
  const auto left = video_candidate({0.03f, 0.15f, 0.48f, 0.85f});
  const auto right = video_candidate({0.52f, 0.15f, 0.97f, 0.85f});
  sbs_roi::tracker_t forward(fast_config());
  sbs_roi::tracker_t reverse(fast_config());

  for (std::uint64_t id = 1; id <= 8; ++id) {
    forward.update(observation(id, {left, right}));
    reverse.update(observation(id, {right, left}));
  }

  EXPECT_EQ(forward.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(reverse.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(forward.snapshot().generation, 0u);
  EXPECT_EQ(reverse.snapshot().generation, 0u);
}

TEST(SbsRoiTracker, DistinctOverlappingVideosAreNotDeduplicated) {
  const auto left =
    video_candidate({0.05f, 0.10f, 0.65f, 0.90f});
  const auto right =
    video_candidate({0.195f, 0.10f, 0.795f, 0.90f});
  sbs_roi::tracker_t forward(fast_config());
  sbs_roi::tracker_t reverse(fast_config());

  for (std::uint64_t id = 1; id <= 4; ++id) {
    forward.update(observation(id, {left, right}));
    reverse.update(observation(id, {right, left}));
  }

  EXPECT_EQ(forward.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(reverse.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
}

TEST(SbsRoiTracker, DuplicateProposalsDoNotCompeteWithTheirContentTarget) {
  const auto left = video_candidate({0.08f, 0.20f, 0.44f, 0.78f});
  const auto right = video_candidate({0.48f, 0.20f, 0.84f, 0.78f});
  const auto envelope = content_candidate({0.05f, 0.15f, 0.88f, 0.84f});
  const auto envelope_variant =
    content_candidate({0.06f, 0.14f, 0.89f, 0.83f}, 0.68f);
  sbs_roi::tracker_t tracker(fast_config());

  tracker.update(observation(
    1,
    {left, envelope_variant, right, envelope}
  ));
  const auto result = tracker.update(observation(
    2,
    {right, envelope, left, envelope_variant}
  ));

  EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::content);
  EXPECT_NEAR(result.rect.left, envelope.rect.left, 0.02f);
  EXPECT_NEAR(result.rect.right, envelope.rect.right, 0.02f);
}

TEST(SbsRoiTracker, OneRecentInteractionBreaksOnlyAnEligibleCloseTie) {
  auto left = video_candidate({0.03f, 0.15f, 0.48f, 0.85f});
  const auto right = video_candidate({0.52f, 0.15f, 0.97f, 0.85f});
  left.recent_interaction = true;
  sbs_roi::tracker_t tracker(fast_config());

  tracker.update(observation(1, {right, left}));
  const auto result = tracker.update(observation(2, {left, right}));

  EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::video);
  EXPECT_FLOAT_EQ(result.rect.left, left.rect.left);
}

TEST(SbsRoiTracker, MultipleOrIneligibleInteractionsRemainAmbiguous) {
  auto left = video_candidate({0.03f, 0.15f, 0.48f, 0.85f});
  auto right = video_candidate({0.52f, 0.15f, 0.97f, 0.85f});
  left.recent_interaction = true;
  right.recent_interaction = true;
  auto sidebar =
    video_candidate({0.80f, 0.02f, 0.99f, 0.72f}, 1.0f, 1.0f);
  sidebar.recent_interaction = true;
  sidebar.inside_primary_column = false;
  sidebar.primary_column_support = 0.0f;
  sbs_roi::tracker_t tracker(fast_config());

  for (std::uint64_t id = 1; id <= 4; ++id) {
    tracker.update(observation(id, {sidebar, right, left}));
  }

  EXPECT_EQ(tracker.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(tracker.snapshot().generation, 0u);
}

TEST(SbsRoiTracker, QuietReclassifiedIncumbentSurvivesNoisyGutterEvidence) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto main = video_candidate({0.08f, 0.12f, 0.72f, 0.88f});
  acquire(tracker, main);
  const auto generation = tracker.snapshot().generation;

  auto retained = content_candidate(main.rect, 0.55f);
  retained.inside_primary_column = false;
  retained.crosses_stable_gutter = true;
  retained.primary_column_support = 0.0f;
  retained.gutter_confidence = 0.0f;
  auto sidebar_ad =
    video_candidate({0.77f, 0.15f, 0.98f, 0.55f}, 1.0f, 1.0f);
  sidebar_ad.inside_primary_column = false;
  sidebar_ad.primary_column_support = 0.0f;

  for (std::uint64_t id = 3; id <= 30; ++id) {
    const auto result =
      tracker.update(observation(id, {sidebar_ad, retained}));
    EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::video);
    EXPECT_FLOAT_EQ(result.rect.left, main.rect.left);
    EXPECT_EQ(result.generation, generation);
  }
}

TEST(SbsRoiTracker, MissingIncumbentRequiresBothCountAndDurationToRelease) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto main = video_candidate({0.08f, 0.12f, 0.72f, 0.88f});
  acquire(tracker, main);

  for (std::uint64_t id = 3; id <= 5; ++id) {
    EXPECT_EQ(
      tracker.update(observation(id, {})).kind,
      sbs_roi::roi_kind_e::video
    );
  }
  const auto released = tracker.update(observation(6, {}));
  EXPECT_EQ(released.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(released.generation, 2u);
  EXPECT_TRUE(released.committed_this_update);
}

TEST(SbsRoiTracker, ProvenReplacementTakesOverWithoutWaitingForRelease) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto initial =
    video_candidate({0.04f, 0.18f, 0.38f, 0.82f});
  const auto replacement =
    video_candidate({0.46f, 0.10f, 0.96f, 0.90f});
  acquire(tracker, initial);

  tracker.update(observation(3, {replacement}));
  tracker.update(observation(4, {replacement}));
  const auto result =
    tracker.update(observation(5, {replacement}));

  EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::video);
  EXPECT_FLOAT_EQ(result.rect.left, replacement.rect.left);
  EXPECT_EQ(result.generation, 2u);
  EXPECT_TRUE(result.committed_this_update);
}

TEST(SbsRoiTracker, IneligibleReplacementCannotRetainForever) {
  auto config = fast_config();
  config.weak_retention_grace_updates = 3;
  config.weak_retention_grace_us = 2 * detector_period_us;
  sbs_roi::tracker_t tracker(config);
  const auto initial =
    video_candidate({0.08f, 0.12f, 0.72f, 0.88f});
  acquire(tracker, initial);
  auto replacement = content_candidate(initial.rect, 0.8f);
  replacement.inside_primary_column = false;
  replacement.crosses_stable_gutter = true;
  replacement.primary_column_support = 0.0f;
  replacement.gutter_confidence = 0.0f;

  for (std::uint64_t id = 3; id <= 7; ++id) {
    EXPECT_EQ(
      tracker.update(observation(id, {replacement})).kind,
      sbs_roi::roi_kind_e::video
    );
  }
  const auto released =
    tracker.update(observation(8, {replacement}));
  EXPECT_EQ(released.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(released.generation, 2u);
}

TEST(SbsRoiTracker, PausedVideoOutlivesWeakGraceUntilLongReleaseContract) {
  auto config = fast_config();
  config.weak_retention_grace_updates = 3;
  config.weak_retention_grace_us = 2 * detector_period_us;
  config.release_updates = 40;
  config.release_duration_us = 39 * detector_period_us;
  sbs_roi::tracker_t tracker(config);
  const auto playing =
    video_candidate({0.10f, 0.12f, 0.72f, 0.88f});
  acquire(tracker, playing);
  const auto generation = tracker.snapshot().generation;

  auto paused = playing;
  paused.temporal_occupancy = 0.0f;
  paused.photographic_density = 0.0f;
  for (std::uint64_t id = 3; id < 44; ++id) {
    const auto result = tracker.update(observation(id, {paused}));
    EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::video);
    EXPECT_EQ(result.generation, generation);
  }

  const auto released = tracker.update(observation(44, {paused}));
  EXPECT_EQ(released.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(released.generation, generation + 1);
}

TEST(SbsRoiTracker, DecisiveChallengerCanReplacePausedVideo) {
  auto config = fast_config();
  config.release_updates = 100;
  config.release_duration_us = 99 * detector_period_us;
  sbs_roi::tracker_t tracker(config);
  const auto playing =
    video_candidate({0.10f, 0.15f, 0.50f, 0.85f});
  acquire(tracker, playing);

  auto paused = playing;
  paused.temporal_occupancy = 0.0f;
  const auto challenger =
    video_candidate({0.60f, 0.10f, 0.98f, 0.90f}, 1.0f, 1.0f);

  tracker.update(observation(3, {paused, challenger}));
  tracker.update(observation(4, {challenger, paused}));
  const auto replaced =
    tracker.update(observation(5, {paused, challenger}));

  EXPECT_EQ(replaced.kind, sbs_roi::roi_kind_e::video);
  EXPECT_FLOAT_EQ(replaced.rect.left, challenger.rect.left);
  EXPECT_EQ(replaced.generation, 2u);
}

TEST(SbsRoiTracker, EquivalentContentProposalDoesNotRelabelPausedVideo) {
  auto config = fast_config();
  config.release_updates = 100;
  config.release_duration_us = 99 * detector_period_us;
  sbs_roi::tracker_t tracker(config);
  const auto playing =
    video_candidate({0.10f, 0.12f, 0.72f, 0.88f});
  acquire(tracker, playing);
  const auto generation = tracker.snapshot().generation;

  const auto paused_content =
    content_candidate(playing.rect, 1.0f);
  for (std::uint64_t id = 3; id <= 30; ++id) {
    const auto result =
      tracker.update(observation(id, {paused_content}));
    EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::video);
    EXPECT_EQ(result.generation, generation);
    EXPECT_FLOAT_EQ(result.rect.left, playing.rect.left);
  }
}

TEST(SbsRoiTracker, StrongQuietIncumbentBeatsHigherScoringWeakOverlap) {
  auto config = fast_config();
  config.weak_retention_grace_updates = 3;
  config.weak_retention_grace_us = 2 * detector_period_us;
  config.release_updates = 100;
  config.release_duration_us = 99 * detector_period_us;
  sbs_roi::tracker_t tracker(config);
  const auto initial =
    video_candidate({0.10f, 0.15f, 0.50f, 0.85f});
  acquire(tracker, initial);
  const auto generation = tracker.snapshot().generation;

  auto quiet_incumbent = initial;
  quiet_incumbent.temporal_occupancy = 0.0f;
  auto weak_overlap =
    content_candidate({0.08f, 0.08f, 0.55f, 0.92f}, 0.9f);
  weak_overlap.inside_primary_column = false;
  weak_overlap.crosses_stable_gutter = true;
  weak_overlap.primary_column_support = 0.0f;
  weak_overlap.gutter_confidence = 0.0f;
  const auto challenger =
    video_candidate({0.62f, 0.18f, 0.94f, 0.78f}, 1.0f, 1.0f);

  for (std::uint64_t id = 3; id <= 12; ++id) {
    const auto result = tracker.update(observation(
      id,
      {weak_overlap, challenger, quiet_incumbent}
    ));
    EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::video);
    EXPECT_EQ(result.generation, generation);
    EXPECT_FLOAT_EQ(result.rect.left, initial.rect.left);
  }
}

TEST(SbsRoiTracker, AlternatingChallengersDoNotPoolDwell) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto incumbent =
    video_candidate({0.05f, 0.20f, 0.38f, 0.80f}, 0.2f, 0.3f);
  acquire(tracker, incumbent);
  const auto generation = tracker.snapshot().generation;
  const auto challenger_a =
    video_candidate({0.42f, 0.12f, 0.92f, 0.88f}, 1.0f, 1.0f);
  const auto challenger_b =
    video_candidate({0.12f, 0.02f, 0.88f, 0.35f}, 1.0f, 1.0f);

  for (std::uint64_t id = 3; id <= 12; ++id) {
    const auto &challenger = id % 2 == 0 ? challenger_a : challenger_b;
    const auto result =
      tracker.update(observation(id, {incumbent, challenger}));
    EXPECT_EQ(result.generation, generation);
    EXPECT_FLOAT_EQ(result.rect.left, incumbent.rect.left);
  }
}

TEST(SbsRoiTracker, SustainedDecisiveChallengerTakesOverOnce) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto incumbent =
    video_candidate({0.05f, 0.20f, 0.38f, 0.80f}, 0.2f, 0.3f);
  const auto challenger =
    video_candidate({0.42f, 0.10f, 0.96f, 0.90f}, 1.0f, 1.0f);
  acquire(tracker, incumbent);

  tracker.update(observation(3, {challenger, incumbent}));
  tracker.update(observation(4, {incumbent, challenger}));
  const auto result =
    tracker.update(observation(5, {challenger, incumbent}));

  EXPECT_FLOAT_EQ(result.rect.left, challenger.rect.left);
  EXPECT_EQ(result.generation, 2u);
  EXPECT_TRUE(result.committed_this_update);
  EXPECT_FALSE(tracker.snapshot().committed_this_update);
}

TEST(SbsRoiTracker, SameTargetGeometryUpdatesAfterHysteresis) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto initial =
    video_candidate({0.08f, 0.15f, 0.68f, 0.85f});
  const auto resized =
    video_candidate({0.08f, 0.12f, 0.78f, 0.88f});
  acquire(tracker, initial);

  tracker.update(observation(3, {resized}));
  tracker.update(observation(4, {resized}));
  const auto result = tracker.update(observation(5, {resized}));

  EXPECT_EQ(result.generation, 2u);
  EXPECT_FLOAT_EQ(result.rect.right, resized.rect.right);
  EXPECT_TRUE(result.committed_this_update);
}

TEST(SbsRoiTracker, StableGeometryJitterDoesNotAdvanceGeneration) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto initial =
    video_candidate({0.08f, 0.15f, 0.68f, 0.85f});
  const auto jittered =
    video_candidate({0.085f, 0.145f, 0.685f, 0.855f});
  acquire(tracker, initial);
  const auto generation = tracker.snapshot().generation;

  for (std::uint64_t id = 3; id <= 20; ++id) {
    const auto result = tracker.update(observation(id, {jittered}));
    EXPECT_EQ(result.generation, generation);
    EXPECT_FLOAT_EQ(result.rect.left, initial.rect.left);
  }
}

TEST(SbsRoiTracker, AlternatingGeometryCannotSatisfyResizeDwell) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto initial =
    video_candidate({0.10f, 0.10f, 0.70f, 0.90f});
  const auto left =
    video_candidate({0.06f, 0.10f, 0.66f, 0.90f});
  const auto right =
    video_candidate({0.14f, 0.10f, 0.74f, 0.90f});
  acquire(tracker, initial);
  const auto generation = tracker.snapshot().generation;

  for (std::uint64_t id = 3; id <= 14; ++id) {
    const auto &candidate = id % 2 == 0 ? left : right;
    const auto result = tracker.update(observation(id, {candidate}));
    EXPECT_EQ(result.generation, generation);
    EXPECT_FLOAT_EQ(result.rect.left, initial.rect.left);
  }
}

TEST(SbsRoiTracker, CandidateIdentityCannotWalkByChainedIou) {
  auto config = fast_config();
  config.acquire_updates = 3;
  config.acquire_duration_us = 2 * detector_period_us;
  sbs_roi::tracker_t tracker(config);
  const auto first =
    video_candidate({0.00f, 0.10f, 0.50f, 0.90f});
  const auto middle =
    video_candidate({0.12f, 0.10f, 0.62f, 0.90f});
  const auto last =
    video_candidate({0.24f, 0.10f, 0.74f, 0.90f});

  tracker.update(observation(1, {first}));
  tracker.update(observation(2, {middle}));
  EXPECT_EQ(
    tracker.update(observation(3, {last})).kind,
    sbs_roi::roi_kind_e::full_frame
  );
  EXPECT_EQ(
    tracker.update(observation(4, {last})).kind,
    sbs_roi::roi_kind_e::full_frame
  );
  const auto acquired = tracker.update(observation(5, {last}));
  EXPECT_EQ(acquired.kind, sbs_roi::roi_kind_e::video);
  EXPECT_FLOAT_EQ(acquired.rect.left, last.rect.left);
}

TEST(SbsRoiTracker, AlternatingOverlappingCandidatesCannotPoolAcquisition) {
  auto config = fast_config();
  config.acquire_updates = 3;
  config.acquire_duration_us = 2 * detector_period_us;
  sbs_roi::tracker_t tracker(config);
  const auto left =
    video_candidate({0.05f, 0.10f, 0.65f, 0.90f});
  const auto right =
    video_candidate({0.195f, 0.10f, 0.795f, 0.90f});

  for (std::uint64_t id = 1; id <= 10; ++id) {
    const auto &candidate = id % 2 == 0 ? left : right;
    EXPECT_EQ(
      tracker.update(observation(id, {candidate})).kind,
      sbs_roi::roi_kind_e::full_frame
    );
  }
  EXPECT_EQ(tracker.snapshot().generation, 0u);
}

TEST(SbsRoiTracker, OneScrollSampleBreaksPreScrollAcquisition) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  tracker.update(observation(1, {target}));

  auto scrolling = observation(2, {target});
  scrolling.broad_page_scroll = true;
  EXPECT_EQ(
    tracker.update(scrolling).kind,
    sbs_roi::roi_kind_e::full_frame
  );
  EXPECT_EQ(
    tracker.update(observation(3, {target})).kind,
    sbs_roi::roi_kind_e::full_frame
  );
  EXPECT_EQ(
    tracker.update(observation(4, {target})).kind,
    sbs_roi::roi_kind_e::video
  );
}

TEST(SbsRoiTracker, StaleScrollFlagCannotClearAcquisitionOrEnterHold) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  tracker.update(observation(1, {target}));

  auto stale_scroll =
    observation(1, {target}, detector_period_us, 150000);
  stale_scroll.broad_page_scroll = true;
  const auto ignored = tracker.update(stale_scroll);
  EXPECT_EQ(ignored.status, sbs_roi::update_status_e::ignored_stale);
  EXPECT_EQ(ignored.kind, sbs_roi::roi_kind_e::full_frame);

  const auto acquired = tracker.update(
    observation(2, {target}, 2 * detector_period_us, 250000)
  );
  EXPECT_EQ(acquired.kind, sbs_roi::roi_kind_e::video);
  EXPECT_EQ(acquired.generation, 1u);
}

TEST(SbsRoiTracker, MalformedScrollFlagCannotAdvanceScrollDwell) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  acquire(tracker, target);
  const auto generation = tracker.snapshot().generation;

  auto first_scroll = observation(3, {target});
  first_scroll.broad_page_scroll = true;
  EXPECT_EQ(
    tracker.update(first_scroll).kind,
    sbs_roi::roi_kind_e::video
  );

  auto malformed_target = target;
  malformed_target.photographic_density =
    std::numeric_limits<float>::quiet_NaN();
  auto malformed_scroll = observation(4, {malformed_target});
  malformed_scroll.broad_page_scroll = true;
  EXPECT_EQ(
    tracker.update(malformed_scroll).status,
    sbs_roi::update_status_e::ignored_invalid
  );

  auto valid_scroll = observation(5, {target});
  valid_scroll.broad_page_scroll = true;
  const auto first_accepted_scroll = tracker.update(valid_scroll);
  EXPECT_EQ(first_accepted_scroll.kind, sbs_roi::roi_kind_e::video);
  EXPECT_EQ(first_accepted_scroll.generation, generation);
}

TEST(SbsRoiTracker, ScrollHoldReacquiresWithOneGenerationChange) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto before =
    content_candidate({0.08f, 0.12f, 0.82f, 0.88f});
  const auto after =
    content_candidate({0.18f, 0.10f, 0.94f, 0.86f});
  acquire(tracker, before);
  const auto locked_generation = tracker.snapshot().generation;

  auto first_scroll = observation(3, {after});
  first_scroll.broad_page_scroll = true;
  EXPECT_EQ(
    tracker.update(first_scroll).kind,
    sbs_roi::roi_kind_e::content
  );
  auto second_scroll = observation(4, {after});
  second_scroll.broad_page_scroll = true;
  EXPECT_EQ(
    tracker.update(second_scroll).kind,
    sbs_roi::roi_kind_e::scroll_hold
  );

  EXPECT_EQ(
    tracker.update(observation(5, {after})).kind,
    sbs_roi::roi_kind_e::scroll_hold
  );
  EXPECT_EQ(
    tracker.update(observation(6, {after})).kind,
    sbs_roi::roi_kind_e::scroll_hold
  );
  const auto reacquired = tracker.update(observation(7, {after}));
  EXPECT_EQ(reacquired.kind, sbs_roi::roi_kind_e::content);
  EXPECT_EQ(reacquired.generation, locked_generation + 1);
  EXPECT_TRUE(reacquired.committed_this_update);
}

TEST(SbsRoiTracker, ScrollInvalidatesAnAlreadyFullFrameGenerationOnce) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto fullscreen =
    video_candidate({0.02f, 0.02f, 0.98f, 0.98f});

  auto first_scroll = observation(1, {fullscreen});
  first_scroll.broad_page_scroll = true;
  tracker.update(first_scroll);
  auto second_scroll = observation(2, {fullscreen});
  second_scroll.broad_page_scroll = true;
  EXPECT_EQ(
    tracker.update(second_scroll).kind,
    sbs_roi::roi_kind_e::scroll_hold
  );

  tracker.update(observation(3, {fullscreen}));
  tracker.update(observation(4, {fullscreen}));
  const auto reacquired =
    tracker.update(observation(5, {fullscreen}));
  EXPECT_EQ(reacquired.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(reacquired.generation, 1u);
  EXPECT_TRUE(reacquired.committed_this_update);
}

TEST(SbsRoiTracker, LongObservationGapResetsPersistence) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});

  tracker.update(observation(1, {target}, 100000));
  EXPECT_EQ(
    tracker.update(observation(1000, {target}, 1000000)).kind,
    sbs_roi::roi_kind_e::full_frame
  );
  EXPECT_EQ(
    tracker.update(observation(1001, {target}, 1100000)).kind,
    sbs_roi::roi_kind_e::video
  );
}

TEST(SbsRoiTracker, MalformedInputCannotBridgeAcquisitionDwell) {
  sbs_roi::tracker_t tracker(fast_config());
  auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  tracker.update(observation(1, {target}));

  target.photographic_density =
    std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(
    tracker.update(observation(2, {target})).status,
    sbs_roi::update_status_e::ignored_invalid
  );

  target.photographic_density = 0.7f;
  EXPECT_EQ(
    tracker.update(observation(3, {target})).kind,
    sbs_roi::roi_kind_e::full_frame
  );
  EXPECT_EQ(
    tracker.update(observation(4, {target})).kind,
    sbs_roi::roi_kind_e::video
  );
}

TEST(SbsRoiTracker, LayoutInvalidationOverridesMalformedCandidatePayload) {
  sbs_roi::tracker_t tracker(fast_config());
  auto main = video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  acquire(tracker, main);
  const auto previous_generation = tracker.snapshot().generation;

  main.photographic_density =
    std::numeric_limits<float>::quiet_NaN();
  auto changed = observation(3, {main});
  changed.invalidation_epoch = 1;
  const auto result = tracker.update(changed);

  EXPECT_EQ(result.status, sbs_roi::update_status_e::accepted);
  EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(result.generation, previous_generation + 1);
  EXPECT_TRUE(result.committed_this_update);
}

TEST(SbsRoiTracker, ConsecutiveMalformedInputFailsSafeExactlyOnce) {
  sbs_roi::tracker_t tracker(fast_config());
  auto main = video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  acquire(tracker, main);
  const auto previous_generation = tracker.snapshot().generation;
  main.photographic_density =
    std::numeric_limits<float>::infinity();

  EXPECT_EQ(
    tracker.update(observation(3, {main})).status,
    sbs_roi::update_status_e::ignored_invalid
  );
  EXPECT_EQ(
    tracker.update(observation(4, {main})).status,
    sbs_roi::update_status_e::ignored_invalid
  );
  const auto failed_safe = tracker.update(observation(5, {main}));
  EXPECT_EQ(
    failed_safe.status,
    sbs_roi::update_status_e::invalidated_invalid_input
  );
  EXPECT_EQ(failed_safe.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(failed_safe.generation, previous_generation + 1);

  const auto repeated = tracker.update(observation(6, {main}));
  EXPECT_EQ(repeated.status, sbs_roi::update_status_e::ignored_invalid);
  EXPECT_EQ(repeated.generation, previous_generation + 1);
}

TEST(SbsRoiTracker, FullFrameInvalidationEpochDistinguishesAdjacentEvents) {
  sbs_roi::tracker_t tracker(fast_config());

  auto changed = observation(1, {});
  changed.invalidation_epoch = 1;
  const auto first = tracker.update(changed);
  EXPECT_EQ(first.generation, 1u);
  EXPECT_TRUE(first.committed_this_update);

  changed = observation(2, {});
  changed.invalidation_epoch = 1;
  const auto held_high = tracker.update(changed);
  EXPECT_EQ(held_high.generation, 1u);
  EXPECT_FALSE(held_high.committed_this_update);

  changed = observation(3, {});
  changed.invalidation_epoch = 2;
  const auto second_edge = tracker.update(changed);
  EXPECT_EQ(second_edge.generation, 2u);
  EXPECT_TRUE(second_edge.committed_this_update);
}

TEST(SbsRoiTracker, ExplicitDetectorUnavailableUsesTheSameWatchdog) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  acquire(tracker, target);

  EXPECT_EQ(
    tracker.detector_unavailable(3 * detector_period_us).status,
    sbs_roi::update_status_e::ignored_invalid
  );
  EXPECT_EQ(
    tracker.detector_unavailable(4 * detector_period_us).status,
    sbs_roi::update_status_e::ignored_invalid
  );
  const auto failed_safe =
    tracker.detector_unavailable(5 * detector_period_us);
  EXPECT_EQ(
    failed_safe.status,
    sbs_roi::update_status_e::invalidated_invalid_input
  );
  EXPECT_EQ(failed_safe.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(failed_safe.generation, 2u);
}

TEST(SbsRoiTracker, BrokenMetadataUsesTheArrivalWatchdog) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  acquire(tracker, target);

  for (std::uint64_t index = 3; index <= 4; ++index) {
    auto broken = observation(
      0,
      {},
      0,
      index * detector_period_us
    );
    broken.timestamp_us = 0;
    EXPECT_EQ(
      tracker.update(broken).status,
      sbs_roi::update_status_e::ignored_invalid
    );
  }
  auto broken = observation(
    0,
    {},
    0,
    5 * detector_period_us
  );
  broken.timestamp_us = 0;
  const auto failed_safe = tracker.update(broken);
  EXPECT_EQ(
    failed_safe.status,
    sbs_roi::update_status_e::invalidated_invalid_input
  );
  EXPECT_EQ(failed_safe.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(failed_safe.generation, 2u);
}

TEST(SbsRoiTracker, CommitEventExistsOnlyOnTheUpdateReturn) {
  sbs_roi::tracker_t tracker(fast_config());
  const auto target =
    video_candidate({0.08f, 0.12f, 0.76f, 0.88f});
  tracker.update(observation(1, {target}));
  const auto committed = tracker.update(observation(2, {target}));

  EXPECT_TRUE(committed.committed_this_update);
  EXPECT_FALSE(tracker.snapshot().committed_this_update);
  EXPECT_FALSE(tracker.snapshot().committed_this_update);
  EXPECT_FALSE(
    tracker.update(observation(2, {target})).committed_this_update
  );
}

TEST(SbsRoiTracker, NearFullscreenTargetUsesIdentityGeometry) {
  sbs_roi::tracker_t initial(fast_config());
  const auto fullscreen =
    video_candidate({0.02f, 0.02f, 0.98f, 0.98f});
  initial.update(observation(1, {fullscreen}));
  initial.update(observation(2, {fullscreen}));
  EXPECT_EQ(initial.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(initial.snapshot().generation, 0u);

  sbs_roi::tracker_t expanded(fast_config());
  const auto embedded =
    video_candidate({0.05f, 0.05f, 0.85f, 0.95f});
  acquire(expanded, embedded);
  expanded.update(observation(3, {fullscreen}));
  expanded.update(observation(4, {fullscreen}));
  const auto result = expanded.update(observation(5, {fullscreen}));
  EXPECT_EQ(result.kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(result.generation, 2u);
}

TEST(SbsRoiTracker, RejectsUnsafeIdentityAndTieConfiguration) {
  auto config = fast_config();
  config.association_iou = 0.0f;
  EXPECT_THROW(sbs_roi::tracker_t {config}, std::invalid_argument);

  config = fast_config();
  config.video_winner_ratio = 1.0f;
  EXPECT_THROW(sbs_roi::tracker_t {config}, std::invalid_argument);

  config = fast_config();
  config.interaction_min_score_ratio = 0.0f;
  EXPECT_THROW(sbs_roi::tracker_t {config}, std::invalid_argument);

  config = fast_config();
  config.max_candidates = 1025;
  EXPECT_THROW(sbs_roi::tracker_t {config}, std::invalid_argument);
}
