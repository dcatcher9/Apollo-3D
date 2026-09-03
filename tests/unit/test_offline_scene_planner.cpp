/**
 * @file tests/unit/test_offline_scene_planner.cpp
 * @brief Tests for causal scene epochs derived from online cut state.
 */
#include "../tests_common.h"

#include <src/offline_scene_planner.h>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace {
  offline_sbs::scene_frame_t causal_sample(
    const std::uint64_t sequence,
    const bool pulse,
    const std::uint32_t count
  ) {
    offline_sbs::scene_frame_t frame;
    frame.sequence = sequence;
    frame.frame_id = std::to_string(sequence);
    frame.depth_updated = true;
    frame.hard_cut_pulse = pulse;
    frame.hard_cut_count = count;
    frame.depth_change_fraction = pulse ? 0.75f : 0.05f;
    frame.raw_rgb_change_fraction = pulse ? 0.8f : 0.02f;
    frame.structural_change_fraction = pulse ? 0.4f : 0.01f;
    frame.current_structural_support_fraction = 0.5f;
    frame.previous_structural_support_fraction = 0.5f;
    frame.common_structural_support_fraction = 0.5f;
    frame.pts_seconds = static_cast<double>(sequence - 1u) / 30.0;
    frame.duration_seconds = 1.0 / 30.0;
    return frame;
  }
}  // namespace

TEST(OfflineCausalSceneTracker, CutFrameStartsTheNewOnlineEpochImmediately) {
  offline_sbs::causal_scene_tracker_t tracker;
  EXPECT_TRUE(tracker.feed(causal_sample(1, false, 0)).empty());
  EXPECT_TRUE(tracker.feed(causal_sample(2, false, 0)).empty());

  auto cut = causal_sample(3, true, 1);
  cut.analysis_flags =
    offline_sbs::analysis_appearance_proposal |
    offline_sbs::analysis_relative_geometry_spike;
  const auto first = tracker.feed(std::move(cut));
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(first.front().start_sequence, 1u);
  EXPECT_EQ(first.front().end_sequence_exclusive, 3u);
  EXPECT_EQ(first.front().semantic_scene_id, 1u);
  EXPECT_EQ(first.front().cut_state_semantics, "causal-production-exact");
  EXPECT_EQ(first.front().boundary.final_sequence, 3u);
  EXPECT_EQ(
    first.front().boundary.decision,
    offline_sbs::boundary_decision_e::confirmed
  );
  EXPECT_EQ(first.front().boundary.revision_depth_updates, 0);
  EXPECT_EQ(first.front().boundary.revision_source_frames, 0);
  EXPECT_EQ(first.front().boundary.candidate_count, 1u);
  EXPECT_EQ(first.front().boundary.evidence_window_first_sequence, 3u);
  EXPECT_EQ(first.front().boundary.evidence_window_last_sequence, 3u);

  EXPECT_TRUE(tracker.feed(causal_sample(4, false, 1)).empty());
  const auto tail = tracker.finish();
  ASSERT_EQ(tail.size(), 1u);
  EXPECT_EQ(tail.front().start_sequence, 3u);
  EXPECT_EQ(tail.front().end_sequence_exclusive, 5u);
  EXPECT_EQ(tail.front().semantic_scene_id, 2u);
  EXPECT_EQ(
    tail.front().boundary.decision,
    offline_sbs::boundary_decision_e::end_of_stream
  );
  ASSERT_EQ(tracker.boundary_audit().size(), 1u);
  EXPECT_EQ(tracker.boundary_audit().front().final_sequence, 3u);
}

TEST(OfflineCausalSceneTracker, ConsecutiveCutsEmitWithoutDelay) {
  offline_sbs::causal_scene_tracker_t tracker;
  EXPECT_TRUE(tracker.feed(causal_sample(1, false, 0)).empty());

  const auto first = tracker.feed(causal_sample(2, true, 1));
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(first.front().start_sequence, 1u);
  EXPECT_EQ(first.front().end_sequence_exclusive, 2u);
  EXPECT_EQ(first.front().semantic_scene_id, 1u);

  const auto second = tracker.feed(causal_sample(3, true, 2));
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(second.front().start_sequence, 2u);
  EXPECT_EQ(second.front().end_sequence_exclusive, 3u);
  EXPECT_EQ(second.front().semantic_scene_id, 2u);

  const auto tail = tracker.finish();
  ASSERT_EQ(tail.size(), 1u);
  EXPECT_EQ(tail.front().start_sequence, 3u);
  EXPECT_EQ(tail.front().end_sequence_exclusive, 4u);
  EXPECT_EQ(tail.front().semantic_scene_id, 3u);
}

TEST(OfflineCausalSceneTracker, RejectsAnyPulseCounterDivergence) {
  offline_sbs::causal_scene_tracker_t missing_increment;
  EXPECT_TRUE(missing_increment.feed(causal_sample(1, false, 0)).empty());
  EXPECT_THROW(
    missing_increment.feed(causal_sample(2, true, 0)),
    offline_sbs::scene_plan_error
  );

  offline_sbs::causal_scene_tracker_t hidden_increment;
  EXPECT_TRUE(hidden_increment.feed(causal_sample(1, false, 0)).empty());
  EXPECT_THROW(
    hidden_increment.feed(causal_sample(2, false, 1)),
    offline_sbs::scene_plan_error
  );

  offline_sbs::causal_scene_tracker_t skipped_generation;
  EXPECT_TRUE(skipped_generation.feed(causal_sample(1, false, 0)).empty());
  EXPECT_THROW(
    skipped_generation.feed(causal_sample(2, true, 2)),
    offline_sbs::scene_plan_error
  );

  offline_sbs::causal_scene_tracker_t invalid_start;
  EXPECT_THROW(
    invalid_start.feed(causal_sample(1, true, 1)),
    offline_sbs::scene_plan_error
  );
}

TEST(OfflineCausalSceneTracker, AggregatesOnlyTheCurrentCausalEpoch) {
  offline_sbs::causal_scene_tracker_t tracker;
  auto first = causal_sample(1, false, 0);
  first.depth_change_fraction = 0.2f;
  first.analysis_flags = offline_sbs::analysis_appearance_veto;
  EXPECT_TRUE(tracker.feed(std::move(first)).empty());

  auto second = causal_sample(2, false, 0);
  second.depth_updated = false;
  second.depth_change_fraction = 0.4f;
  EXPECT_TRUE(tracker.feed(std::move(second)).empty());

  const auto scene = tracker.feed(causal_sample(3, true, 1));
  ASSERT_EQ(scene.size(), 1u);
  EXPECT_EQ(scene.front().frame_count, 2u);
  EXPECT_EQ(scene.front().evidence.source_frame_count, 2u);
  EXPECT_EQ(scene.front().evidence.depth_update_count, 1u);
  EXPECT_EQ(scene.front().evidence.appearance_veto_count, 1u);
  ASSERT_TRUE(scene.front().evidence.depth_change_max.has_value());
  EXPECT_FLOAT_EQ(*scene.front().evidence.depth_change_max, 0.4f);
  ASSERT_TRUE(scene.front().start_pts_seconds.has_value());
  ASSERT_TRUE(scene.front().end_pts_seconds_exclusive.has_value());
  EXPECT_DOUBLE_EQ(*scene.front().start_pts_seconds, 0.0);
  EXPECT_DOUBLE_EQ(*scene.front().end_pts_seconds_exclusive, 2.0 / 30.0);
}

TEST(OfflineCausalSceneTracker, ValidatesFrameAndTimelineContract) {
  offline_sbs::causal_scene_tracker_t empty;
  EXPECT_THROW(empty.finish(), offline_sbs::scene_plan_error);

  offline_sbs::causal_scene_tracker_t wrong_sequence;
  EXPECT_THROW(
    wrong_sequence.feed(causal_sample(2, false, 0)),
    offline_sbs::scene_plan_error
  );

  offline_sbs::causal_scene_tracker_t missing_id;
  auto blank = causal_sample(1, false, 0);
  blank.frame_id.clear();
  EXPECT_THROW(
    missing_id.feed(std::move(blank)),
    offline_sbs::scene_plan_error
  );

  offline_sbs::causal_scene_tracker_t invalid_fraction;
  auto fraction = causal_sample(1, false, 0);
  fraction.depth_change_fraction = 1.01f;
  EXPECT_THROW(
    invalid_fraction.feed(std::move(fraction)),
    offline_sbs::scene_plan_error
  );

  offline_sbs::causal_scene_tracker_t invalid_time;
  auto time = causal_sample(1, false, 0);
  time.duration_seconds = std::numeric_limits<double>::infinity();
  EXPECT_THROW(
    invalid_time.feed(std::move(time)),
    offline_sbs::scene_plan_error
  );
}

TEST(OfflineCausalSceneTracker, FinalizationClosesTheTracker) {
  offline_sbs::causal_scene_tracker_t tracker;
  EXPECT_TRUE(tracker.feed(causal_sample(1, false, 0)).empty());
  ASSERT_EQ(tracker.finish().size(), 1u);
  EXPECT_THROW(tracker.finish(), offline_sbs::scene_plan_error);
  EXPECT_THROW(
    tracker.feed(causal_sample(2, false, 0)),
    offline_sbs::scene_plan_error
  );
}
