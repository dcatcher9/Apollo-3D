/**
 * @file tests/unit/test_offline_scene_planner.cpp
 * @brief Tests for the native offline scene look-ahead policy.
 */
#include "../tests_common.h"

#include <src/offline_scene_planner.h>

#include <cmath>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

namespace {
  offline_sbs::scene_planner_config_t planner_config() {
    offline_sbs::scene_planner_config_t config;
    config.lookbehind_depth_updates = 2;
    config.lookahead_depth_updates = 2;
    config.duplicate_pulse_distance_updates = 1;
    config.minimum_scene_frames = 2;
    config.max_open_cache_bytes = 1024 * 1024;
    return config;
  }

  offline_sbs::scene_frame_t sample(
    std::uint64_t sequence,
    bool pulse = false,
    float depth = 0.05f,
    float raw = 0.02f,
    float structural = 0.01f
  ) {
    offline_sbs::scene_frame_t frame;
    frame.sequence = sequence;
    frame.frame_id = std::to_string(sequence);
    frame.depth_updated = true;
    frame.hard_cut_pulse = pulse;
    frame.depth_change_fraction = depth;
    frame.raw_rgb_change_fraction = raw;
    frame.structural_change_fraction = structural;
    frame.current_structural_support_fraction = 0.5f;
    frame.previous_structural_support_fraction = 0.5f;
    frame.common_structural_support_fraction = 0.5f;
    return frame;
  }
}  // namespace

TEST(OfflineScenePlanner, WaitsForFutureEvidenceBeforeCommitting) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 7; ++sequence) {
    auto emitted = planner.feed(sample(
      sequence,
      sequence == 4,
      sequence == 4 ? 0.65f : 0.05f
    ));
    EXPECT_TRUE(emitted.empty());
  }
  scenes = planner.feed(sample(8));
  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().start_sequence, 1u);
  EXPECT_EQ(scenes.front().end_sequence_exclusive, 4u);
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::confirmed
  );
  EXPECT_EQ(scenes.front().boundary.final_sequence, 4u);
}

TEST(OfflineScenePlanner, HeldDepthPulseIsNotReproposed) {
  auto config = planner_config();
  config.lookbehind_depth_updates = 0;
  config.lookahead_depth_updates = 1;
  offline_sbs::scene_planner_t planner(config);
  EXPECT_TRUE(planner.feed(sample(1)).empty());
  EXPECT_TRUE(planner.feed(sample(2)).empty());
  EXPECT_TRUE(planner.feed(sample(3, true, 0.70f)).empty());

  auto held = sample(4, true, 0.70f);
  held.depth_updated = false;
  EXPECT_TRUE(planner.feed(std::move(held)).empty());
  EXPECT_EQ(planner.pending_proposal_count(), 1u);

  const auto scenes = planner.feed(sample(5));
  ASSERT_EQ(scenes.size(), 1u);
  ASSERT_EQ(scenes.front().boundary.proposal_sequences.size(), 1u);
  EXPECT_EQ(scenes.front().boundary.proposal_sequences.front(), 3u);
}

TEST(OfflineScenePlanner, StrongerLookbehindGeometryCanMoveBoundary) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) {
    auto frame = sample(sequence);
    if (sequence == 3) {
      frame.depth_change_fraction = 0.90f;
    } else if (sequence == 4) {
      frame.hard_cut_pulse = true;
      frame.depth_change_fraction = 0.61f;
    }
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }
  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().end_sequence_exclusive, 3u);
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::moved_to_correlated_evidence
  );
  EXPECT_EQ(scenes.front().boundary.revision_depth_updates, -1);
  EXPECT_EQ(scenes.front().boundary.revision_source_frames, -1);
}

TEST(OfflineScenePlanner, StructuralFloorKeepsV61CutAndRejectsV60FalsePairs) {
  auto config = planner_config();
  config.lookahead_depth_updates = 4;
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 70; ++sequence) {
    auto frame = sample(sequence);
    switch (sequence) {
      case 19:
        frame.depth_change_fraction = 0.759617f;
        frame.structural_change_fraction = 0.028952f;
        break;
      case 20:
        frame.hard_cut_pulse = true;
        frame.depth_change_fraction = 0.802154f;
        frame.structural_change_fraction = 0.028183f;
        break;
      case 24:
        // This later, larger depth spike moved the v61 offline boundary before
        // structural-floor parity. It is ordinary content, below the live
        // floor, and must not displace the causal f19/f20 cut.
        frame.depth_change_fraction = 0.888000f;
        frame.structural_change_fraction = 0.003320f;
        frame.analysis_flags =
          offline_sbs::analysis_relative_geometry_spike;
        break;
      case 35:
        frame.depth_change_fraction = 0.771218f;
        frame.structural_change_fraction = 0.001680f;
        frame.analysis_flags =
          offline_sbs::analysis_relative_geometry_spike;
        break;
      case 36:
        frame.hard_cut_pulse = true;
        frame.depth_change_fraction = 0.674158f;
        frame.structural_change_fraction = 0.001721f;
        frame.analysis_flags =
          offline_sbs::analysis_relative_geometry_spike;
        break;
      case 53:
        frame.depth_change_fraction = 0.999555f;
        frame.structural_change_fraction = 0.003705f;
        frame.analysis_flags =
          offline_sbs::analysis_relative_geometry_spike;
        break;
      case 54:
        frame.hard_cut_pulse = true;
        frame.depth_change_fraction = 0.999980f;
        // Missing unrelated appearance telemetry must not override a measured
        // ordinary-geometry structural failure.
        frame.raw_rgb_change_fraction.reset();
        frame.structural_change_fraction = 0.003968f;
        frame.analysis_flags =
          offline_sbs::analysis_relative_geometry_spike;
        break;
      case 62:
        frame.depth_change_fraction = 0.984168f;
        frame.structural_change_fraction = 0.000587f;
        frame.analysis_flags =
          offline_sbs::analysis_relative_geometry_spike;
        break;
      case 63:
        frame.hard_cut_pulse = true;
        frame.depth_change_fraction = 0.994027f;
        frame.structural_change_fraction = 0.000931f;
        frame.analysis_flags =
          offline_sbs::analysis_relative_geometry_spike;
        break;
      default:
        break;
    }
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }
  auto tail = planner.finish();
  scenes.insert(
    scenes.end(),
    std::make_move_iterator(tail.begin()),
    std::make_move_iterator(tail.end())
  );

  ASSERT_EQ(scenes.size(), 2u);
  EXPECT_EQ(scenes.front().end_sequence_exclusive, 20u);
  EXPECT_EQ(scenes.back().start_sequence, 20u);

  const auto &audit = planner.boundary_audit();
  ASSERT_EQ(audit.size(), 4u);
  ASSERT_TRUE(audit[0].final_sequence);
  EXPECT_EQ(*audit[0].final_sequence, 20u);
  EXPECT_EQ(audit[0].decision, offline_sbs::boundary_decision_e::confirmed);
  EXPECT_TRUE(audit[0].selected_geometry_qualified);
  ASSERT_TRUE(audit[0].selected_structural_change_fraction);
  EXPECT_NEAR(
    *audit[0].selected_structural_change_fraction,
    0.028183f,
    1e-6f
  );

  const std::uint64_t rejected_proposals[] {36u, 54u, 63u};
  for (std::size_t index = 0; index < std::size(rejected_proposals); ++index) {
    const auto &rejected = audit[index + 1u];
    EXPECT_EQ(
      rejected.proposal_sequences,
      (std::vector<std::uint64_t> {rejected_proposals[index]})
    );
    EXPECT_FALSE(rejected.accepted);
    EXPECT_FALSE(rejected.final_sequence);
    EXPECT_EQ(rejected.candidate_count, 0u);
    EXPECT_EQ(
      rejected.decision,
      offline_sbs::boundary_decision_e::rejected_unsupported_proposal
    );
  }
}

TEST(OfflineScenePlanner, PersistentStructurelessPulseKeepsHeldEndpoint) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) {
    auto frame = sample(sequence);
    if (sequence == 3) {
      frame.depth_change_fraction = 0.90f;
      frame.structural_change_fraction = 0.0f;
      frame.analysis_flags = offline_sbs::analysis_structureless_transition;
    } else if (sequence == 4) {
      frame.hard_cut_pulse = true;
      frame.depth_change_fraction = 0.70f;
      frame.structural_change_fraction = 0.0f;
      frame.analysis_flags =
        offline_sbs::analysis_structureless_transition |
        offline_sbs::analysis_geometry_confirmation_candidate;
    }
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }

  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().end_sequence_exclusive, 4u);
  EXPECT_EQ(scenes.front().boundary.candidate_count, 1u);
  EXPECT_TRUE(scenes.front().boundary.selected_geometry_qualified);
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::confirmed
  );
}

TEST(OfflineScenePlanner, MissingStructuralTraceCannotRelocateCausalProposal) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) {
    auto frame = sample(sequence);
    if (sequence == 3) {
      frame.depth_change_fraction = 0.95f;
      frame.structural_change_fraction.reset();
    } else if (sequence == 4) {
      frame.hard_cut_pulse = true;
      frame.depth_change_fraction = 0.70f;
      frame.structural_change_fraction.reset();
    }
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }

  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().end_sequence_exclusive, 4u);
  EXPECT_EQ(scenes.front().boundary.candidate_count, 1u);
  EXPECT_FALSE(scenes.front().boundary.selected_geometry_qualified);
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::confirmed_causal_fallback
  );
  EXPECT_NE(
    scenes.front().boundary.reason.find("complete live-detector parity"),
    std::string::npos
  );
}

TEST(OfflineScenePlanner, MissingRawRgbDoesNotDowngradeCompleteGeometryEvidence) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) {
    auto frame = sample(sequence);
    if (sequence == 3) {
      frame.depth_change_fraction = 0.95f;
      frame.raw_rgb_change_fraction.reset();
      frame.structural_change_fraction = 0.02f;
    } else if (sequence == 4) {
      frame.hard_cut_pulse = true;
      frame.depth_change_fraction = 0.70f;
      frame.raw_rgb_change_fraction.reset();
      frame.structural_change_fraction = 0.02f;
    }
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }

  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().end_sequence_exclusive, 3u);
  EXPECT_TRUE(scenes.front().boundary.selected_geometry_qualified);
  EXPECT_TRUE(scenes.front().boundary.selected_evidence_score);
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::moved_to_correlated_evidence
  );
}

TEST(OfflineScenePlanner, ProducerLocalizedAppearanceCanMoveBoundary) {
  const auto run = [](bool producer_qualified) {
    auto config = planner_config();
    offline_sbs::scene_planner_t planner(config);
    std::vector<offline_sbs::scene_plan_t> scenes;
    for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) {
      auto frame = sample(sequence);
      if (sequence == 3) {
        frame.depth_change_fraction = 0.3864f;
        frame.raw_rgb_change_fraction = 0.2019f;
        frame.structural_change_fraction = 0.0396f;
        if (producer_qualified) {
          frame.analysis_flags = offline_sbs::analysis_appearance_proposal;
        }
      } else if (sequence == 4) {
        frame.hard_cut_pulse = true;
        frame.depth_change_fraction = 0.61f;
      }
      auto emitted = planner.feed(std::move(frame));
      scenes.insert(
        scenes.end(),
        std::make_move_iterator(emitted.begin()),
        std::make_move_iterator(emitted.end())
      );
    }
    return scenes;
  };

  const auto qualified = run(true);
  ASSERT_EQ(qualified.size(), 1u);
  EXPECT_EQ(qualified.front().end_sequence_exclusive, 3u);
  EXPECT_TRUE(qualified.front().boundary.selected_appearance_qualified);
  EXPECT_EQ(
    qualified.front().boundary.decision,
    offline_sbs::boundary_decision_e::moved_to_correlated_evidence
  );

  const auto unqualified = run(false);
  ASSERT_EQ(unqualified.size(), 1u);
  EXPECT_EQ(unqualified.front().end_sequence_exclusive, 4u);
  EXPECT_FALSE(unqualified.front().boundary.selected_appearance_qualified);
}

TEST(OfflineScenePlanner, OverlappingProposalWindowsMergeBeforeCommit) {
  auto config = planner_config();
  config.duplicate_pulse_distance_updates = 0;
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 12; ++sequence) {
    const auto pulse = sequence == 4 || sequence == 8;
    auto emitted = planner.feed(sample(
      sequence,
      pulse,
      pulse ? 0.70f : 0.05f
    ));
    if (sequence < 12) {
      EXPECT_TRUE(emitted.empty());
    }
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }
  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(
    scenes.front().boundary.proposal_sequences,
    (std::vector<std::uint64_t> {4, 8})
  );
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::merged_duplicate_proposals
  );
}

TEST(OfflineScenePlanner, SupportedFlashReturnRejectsProvisionalCut) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  for (std::uint64_t sequence = 1; sequence <= 7; ++sequence) {
    auto frame = sample(
      sequence,
      sequence == 4,
      sequence == 4 ? 0.70f : 0.05f,
      sequence == 4 ? 0.95f : 0.0f,
      0.0f
    );
    if (sequence == 4) {
      frame.analysis_flags =
        offline_sbs::analysis_appearance_proposal |
        offline_sbs::analysis_exposure_like |
        offline_sbs::analysis_appearance_veto;
    } else if (sequence == 6) {
      frame.analysis_flags = offline_sbs::analysis_same_scene_return;
    }
    EXPECT_TRUE(planner.feed(std::move(frame)).empty());
  }
  const auto scenes = planner.finish();
  ASSERT_EQ(scenes.size(), 1u);
  ASSERT_EQ(planner.boundary_audit().size(), 1u);
  EXPECT_EQ(
    planner.boundary_audit().front().decision,
    offline_sbs::boundary_decision_e::rejected_supported_flash_return
  );
}

TEST(OfflineScenePlanner, FlashLookaheadUsesSourceTimeAcrossFrameRates) {
  for (const double frames_per_second : {24.0, 60.0, 120.0}) {
    offline_sbs::scene_planner_config_t config;
    config.max_open_cache_bytes = 1024 * 1024;
    offline_sbs::scene_planner_t planner(config);
    const auto frame_count =
      static_cast<std::uint64_t>(std::llround(frames_per_second));
    bool proposed = false;
    bool returned = false;
    for (std::uint64_t index = 0; index <= frame_count; ++index) {
      const auto pts = static_cast<double>(index) / frames_per_second;
      const bool pulse = !proposed && pts >= 0.5;
      const bool same_scene_return = !returned && pts >= 0.6;
      auto frame = sample(
        index + 1,
        pulse,
        pulse ? 0.70f : 0.05f,
        pulse ? 0.95f : 0.0f,
        0.0f
      );
      frame.pts_seconds = pts;
      frame.duration_seconds = 1.0 / frames_per_second;
      if (pulse) {
        frame.analysis_flags =
          offline_sbs::analysis_appearance_proposal |
          offline_sbs::analysis_exposure_like |
          offline_sbs::analysis_appearance_veto;
        proposed = true;
      } else if (same_scene_return) {
        frame.analysis_flags = offline_sbs::analysis_same_scene_return;
        returned = true;
      }
      planner.feed(std::move(frame));
    }
    const auto scenes = planner.finish();
    ASSERT_FALSE(scenes.empty()) << frames_per_second;
    ASSERT_FALSE(planner.boundary_audit().empty()) << frames_per_second;
    EXPECT_EQ(
      planner.boundary_audit().front().decision,
      offline_sbs::boundary_decision_e::rejected_supported_flash_return
    ) << frames_per_second;
  }
}

TEST(OfflineScenePlanner, VariableRateLookaheadKeepsBoundaryTime) {
  const auto run = [](std::vector<double> timestamps) {
    auto config = planner_config();
    offline_sbs::scene_planner_t planner(config);
    std::vector<offline_sbs::scene_plan_t> scenes;
    for (std::size_t index = 0; index < timestamps.size(); ++index) {
      const auto pts = timestamps[index];
      const bool proposal = std::abs(pts - 0.5) < 1e-9;
      const bool stronger = std::abs(pts - (14.0 / 30.0)) < 1e-9;
      auto frame = sample(
        index + 1,
        proposal,
        stronger ? 0.90f : (proposal ? 0.61f : 0.05f)
      );
      frame.frame_id = stronger ? "correlated-transition" :
                       (proposal ? "causal-pulse" : std::to_string(index + 1));
      frame.pts_seconds = pts;
      frame.duration_seconds =
        index + 1 < timestamps.size() ?
          timestamps[index + 1] - pts :
          1.0 / 30.0;
      auto emitted = planner.feed(std::move(frame));
      scenes.insert(
        scenes.end(),
        std::make_move_iterator(emitted.begin()),
        std::make_move_iterator(emitted.end())
      );
    }
    auto tail = planner.finish();
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(tail.begin()),
      std::make_move_iterator(tail.end())
    );
    return scenes;
  };

  std::vector<double> constant_rate;
  std::vector<double> variable_rate;
  for (int ordinal = 0; ordinal <= 24; ++ordinal) {
    const auto timestamp = static_cast<double>(ordinal) / 30.0;
    constant_rate.push_back(timestamp);
    variable_rate.push_back(timestamp);
    // Insert irregular source frames without changing any reference-clock
    // evidence timestamp. This exercises true VFR/source-sequence divergence.
    if (ordinal < 24 && ordinal % 3 != 1) {
      variable_rate.push_back(timestamp + (ordinal % 2 == 0 ? 0.011 : 0.019));
    }
  }

  const auto constant = run(std::move(constant_rate));
  const auto variable = run(std::move(variable_rate));
  ASSERT_EQ(constant.size(), 2u);
  ASSERT_EQ(variable.size(), 2u);
  for (const auto *scenes : {&constant, &variable}) {
    ASSERT_TRUE((*scenes)[0].end_pts_seconds_exclusive);
    EXPECT_NEAR(*(*scenes)[0].end_pts_seconds_exclusive, 14.0 / 30.0, 1e-9);
    EXPECT_EQ(
      (*scenes)[0].boundary.decision,
      offline_sbs::boundary_decision_e::moved_to_correlated_evidence
    );
    ASSERT_TRUE((*scenes)[0].boundary.selected_frame_id);
    EXPECT_EQ(
      *(*scenes)[0].boundary.selected_frame_id,
      "correlated-transition"
    );
  }
}

TEST(OfflineScenePlanner, FailBudgetPolicyStillFailsWhileProposalIsUnresolved) {
  auto config = planner_config();
  config.lookbehind_depth_updates = 0;
  config.lookahead_depth_updates = 5;
  config.max_open_cache_bytes = 100;
  config.allow_administrative_split = false;
  offline_sbs::scene_planner_t planner(config);

  auto first = sample(1, true, 0.70f);
  first.cache_bytes = 60;
  EXPECT_TRUE(planner.feed(std::move(first)).empty());
  auto second = sample(2);
  second.cache_bytes = 60;
  EXPECT_THROW(
    planner.feed(std::move(second)),
    offline_sbs::scene_cache_budget_error
  );
  EXPECT_EQ(planner.pending_proposal_count(), 1u);
}

TEST(OfflineScenePlanner, SplitBudgetClosesPersistentProposalTrainWithoutGaps) {
  auto config = planner_config();
  config.minimum_scene_frames = 1;
  config.lookbehind_depth_updates = 2;
  config.lookahead_depth_updates = 2;
  config.duplicate_pulse_distance_updates = 1;
  config.max_open_cache_bytes = 100;
  config.allow_administrative_split = true;
  offline_sbs::scene_planner_t planner(config);

  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) {
    auto frame = sample(sequence, true, 0.70f);
    frame.cache_bytes = 60;
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
    EXPECT_LE(planner.open_cache_bytes(), config.max_open_cache_bytes);
  }
  auto tail = planner.finish();
  scenes.insert(
    scenes.end(),
    std::make_move_iterator(tail.begin()),
    std::make_move_iterator(tail.end())
  );

  ASSERT_FALSE(scenes.empty());
  std::uint64_t expected_start = 1;
  bool saw_budget_truncation = false;
  for (const auto &scene : scenes) {
    EXPECT_EQ(scene.start_sequence, expected_start);
    EXPECT_GT(scene.end_sequence_exclusive, scene.start_sequence);
    expected_start = scene.end_sequence_exclusive;
    EXPECT_LE(scene.cache_bytes, config.max_open_cache_bytes);
    saw_budget_truncation =
      saw_budget_truncation ||
      (scene.boundary.truncated && scene.boundary.budget_forced);
  }
  EXPECT_EQ(expected_start, 9u);
  EXPECT_TRUE(saw_budget_truncation);
  EXPECT_EQ(planner.open_cache_bytes(), 0u);
}

TEST(OfflineScenePlanner, SplitBudgetAuditsRejectedPendingEvidenceBeforeFallback) {
  auto config = planner_config();
  config.minimum_scene_frames = 1;
  config.lookbehind_depth_updates = 0;
  config.lookahead_depth_updates = 5;
  config.max_open_cache_bytes = 100;
  config.allow_administrative_split = true;
  offline_sbs::scene_planner_t planner(config);

  auto first = sample(1);
  first.cache_bytes = 40;
  EXPECT_TRUE(planner.feed(std::move(first)).empty());
  auto proposal = sample(2, true);
  proposal.cache_bytes = 40;
  EXPECT_TRUE(planner.feed(std::move(proposal)).empty());
  auto third = sample(3);
  third.cache_bytes = 40;
  const auto prefix = planner.feed(std::move(third));

  ASSERT_EQ(prefix.size(), 1u);
  EXPECT_EQ(prefix.front().start_sequence, 1u);
  EXPECT_EQ(prefix.front().end_sequence_exclusive, 3u);
  EXPECT_EQ(
    prefix.front().boundary.decision,
    offline_sbs::boundary_decision_e::administrative_cache_split
  );
  EXPECT_TRUE(prefix.front().boundary.budget_forced);
  EXPECT_LE(planner.open_cache_bytes(), config.max_open_cache_bytes);

  ASSERT_EQ(planner.boundary_audit().size(), 2u);
  EXPECT_EQ(
    planner.boundary_audit()[0].decision,
    offline_sbs::boundary_decision_e::rejected_unsupported_proposal
  );
  EXPECT_TRUE(planner.boundary_audit()[0].truncated);
  EXPECT_TRUE(planner.boundary_audit()[0].budget_forced);
  EXPECT_EQ(
    planner.boundary_audit()[1].decision,
    offline_sbs::boundary_decision_e::administrative_cache_split
  );

  const auto tail = planner.finish();
  ASSERT_EQ(tail.size(), 1u);
  EXPECT_EQ(tail.front().start_sequence, 3u);
  EXPECT_EQ(tail.front().end_sequence_exclusive, 4u);
}

TEST(OfflineScenePlanner, PersistentProposalsFormBoundedDuplicateClusters) {
  auto config = planner_config();
  config.minimum_scene_frames = 1;
  config.max_open_cache_bytes = 0;
  offline_sbs::scene_planner_t planner(config);

  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 12; ++sequence) {
    auto emitted = planner.feed(sample(sequence, true, 0.70f));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }

  ASSERT_FALSE(scenes.empty());
  EXPECT_LE(
    scenes.front().boundary.proposal_sequences.size(),
    config.lookbehind_depth_updates + config.lookahead_depth_updates + 1
  );
  EXPECT_GT(planner.pending_proposal_count(), 0u);
}

TEST(OfflineScenePlanner, ConfigurationAndFrameContractFailClosed) {
  auto bad = planner_config();
  bad.minimum_scene_frames = 0;
  EXPECT_THROW(
    offline_sbs::scene_planner_t {bad},
    offline_sbs::scene_plan_error
  );

  offline_sbs::scene_planner_t planner(planner_config());
  EXPECT_THROW(planner.feed(sample(2)), offline_sbs::scene_plan_error);
  EXPECT_EQ(planner.open_cache_bytes(), 0u);
}

TEST(OfflineScenePlanner, CacheFreeSceneMetadataFailsBeforeExceedingFrameCap) {
  auto config = planner_config();
  config.max_open_frames = 2;
  offline_sbs::scene_planner_t planner(config);

  EXPECT_TRUE(planner.feed(sample(1)).empty());
  EXPECT_TRUE(planner.feed(sample(2)).empty());
  EXPECT_EQ(planner.open_frame_count(), 2u);
  EXPECT_EQ(planner.open_cache_bytes(), 0u);

  try {
    static_cast<void>(planner.feed(sample(3)));
    FAIL() << "expected the analysis metadata cap to fail closed";
  } catch (const offline_sbs::scene_metadata_budget_error &error) {
    EXPECT_EQ(error.limit_frames, 2u);
    EXPECT_EQ(error.attempted_frames, 3u);
    EXPECT_EQ(error.open_start_sequence, 1u);
    EXPECT_EQ(error.current_sequence, 3u);
  }
  EXPECT_EQ(planner.open_frame_count(), 2u);
  EXPECT_EQ(planner.open_cache_bytes(), 0u);
  EXPECT_EQ(planner.pending_proposal_count(), 0u);
}

TEST(OfflineScenePlanner, ConfirmedBoundaryReleasesFrameMetadataBudget) {
  auto config = planner_config();
  config.max_open_frames = 8;
  offline_sbs::scene_planner_t planner(config);

  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) {
    auto emitted = planner.feed(sample(
      sequence,
      sequence == 4,
      sequence == 4 ? 0.65f : 0.05f
    ));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }
  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().end_sequence_exclusive, 4u);
  EXPECT_EQ(planner.open_frame_count(), 5u);

  EXPECT_TRUE(planner.feed(sample(9)).empty());
  EXPECT_TRUE(planner.feed(sample(10)).empty());
  EXPECT_TRUE(planner.feed(sample(11)).empty());
  EXPECT_EQ(planner.open_frame_count(), config.max_open_frames);

  auto tail = planner.finish();
  ASSERT_EQ(tail.size(), 1u);
  EXPECT_EQ(tail.front().start_sequence, 4u);
  EXPECT_EQ(tail.front().end_sequence_exclusive, 12u);
  EXPECT_EQ(planner.open_frame_count(), 0u);
}

TEST(OfflineScenePlanner, OversizedFrameIdentifierCannotBypassMetadataBound) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  auto oversized = sample(1);
  oversized.frame_id.assign(
    offline_sbs::max_scene_frame_id_bytes + 1,
    'x'
  );

  EXPECT_THROW(
    planner.feed(std::move(oversized)),
    offline_sbs::scene_plan_error
  );
  EXPECT_EQ(planner.open_frame_count(), 0u);
  EXPECT_EQ(planner.open_cache_bytes(), 0u);
}

TEST(OfflineScenePlanner, CompleteUnsupportedProposalIsRejected) {
  auto config = planner_config();
  config.lookbehind_depth_updates = 0;
  config.lookahead_depth_updates = 1;
  offline_sbs::scene_planner_t planner(config);
  for (std::uint64_t sequence = 1; sequence <= 5; ++sequence) {
    planner.feed(sample(sequence, sequence == 3));
  }
  const auto scenes = planner.finish();
  ASSERT_EQ(scenes.size(), 1u);
  ASSERT_EQ(planner.boundary_audit().size(), 1u);
  EXPECT_EQ(
    planner.boundary_audit().front().decision,
    offline_sbs::boundary_decision_e::rejected_unsupported_proposal
  );
}

TEST(OfflineScenePlanner, EofAcceptsTruncatedCutButRejectsUndersizedTail) {
  auto config = planner_config();
  config.lookahead_depth_updates = 5;
  offline_sbs::scene_planner_t accepted(config);
  for (std::uint64_t sequence = 1; sequence <= 6; ++sequence) {
    accepted.feed(sample(
      sequence,
      sequence == 5,
      sequence == 5 ? 0.70f : 0.05f
    ));
  }
  const auto accepted_scenes = accepted.finish();
  ASSERT_EQ(accepted_scenes.size(), 2u);
  EXPECT_TRUE(accepted_scenes.front().boundary.truncated);
  EXPECT_EQ(accepted_scenes.front().end_sequence_exclusive, 5u);

  offline_sbs::scene_planner_t rejected(config);
  for (std::uint64_t sequence = 1; sequence <= 6; ++sequence) {
    rejected.feed(sample(
      sequence,
      sequence == 6,
      sequence == 6 ? 0.70f : 0.05f
    ));
  }
  const auto rejected_scenes = rejected.finish();
  ASSERT_EQ(rejected_scenes.size(), 1u);
  EXPECT_EQ(
    rejected.boundary_audit().front().decision,
    offline_sbs::boundary_decision_e::rejected_minimum_scene_length
  );
}

TEST(OfflineScenePlanner, AdministrativeSplitCreatesANewCacheSegmentAndHonorsCap) {
  auto config = planner_config();
  config.minimum_scene_frames = 1;
  config.max_open_cache_bytes = 100;
  config.allow_administrative_split = true;
  offline_sbs::scene_planner_t planner(config);
  auto first = sample(1);
  first.cache_bytes = 60;
  EXPECT_TRUE(planner.feed(std::move(first)).empty());
  auto second = sample(2);
  second.cache_bytes = 60;
  const auto prefix = planner.feed(std::move(second));
  ASSERT_EQ(prefix.size(), 1u);
  EXPECT_TRUE(prefix.front().boundary.budget_forced);
  EXPECT_EQ(prefix.front().semantic_scene_id, 1u);
  EXPECT_LE(planner.open_cache_bytes(), 100u);
  const auto tail = planner.finish();
  ASSERT_EQ(tail.size(), 1u);
  EXPECT_EQ(tail.front().semantic_scene_id, 1u);

  offline_sbs::scene_planner_t oversized(config);
  auto huge = sample(1);
  huge.cache_bytes = 101;
  EXPECT_THROW(
    oversized.feed(std::move(huge)),
    offline_sbs::scene_cache_budget_error
  );
  EXPECT_EQ(oversized.open_cache_bytes(), 0u);
}

TEST(OfflineScenePlanner, FlashReturnDoesNotHideLaterGeometry) {
  auto config = planner_config();
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 9; ++sequence) {
    auto frame = sample(sequence);
    if (sequence == 4) {
      frame.hard_cut_pulse = true;
      frame.depth_change_fraction = 0.70f;
      frame.raw_rgb_change_fraction = 0.95f;
      frame.structural_change_fraction = 0.0f;
      frame.analysis_flags =
        offline_sbs::analysis_appearance_proposal |
        offline_sbs::analysis_exposure_like |
        offline_sbs::analysis_appearance_veto;
    } else if (sequence == 5) {
      frame.analysis_flags = offline_sbs::analysis_same_scene_return;
    } else if (sequence == 6) {
      frame.depth_change_fraction = 0.90f;
    }
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }
  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().boundary.final_sequence, 6u);
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::moved_to_correlated_evidence
  );
}

TEST(OfflineScenePlanner, MixedClusterFallsBackOnlyToIncompleteNonVetoedPulse) {
  auto config = planner_config();
  config.duplicate_pulse_distance_updates = 0;
  offline_sbs::scene_planner_t planner(config);
  std::vector<offline_sbs::scene_plan_t> scenes;
  for (std::uint64_t sequence = 1; sequence <= 12; ++sequence) {
    auto frame = sample(sequence, sequence == 4 || sequence == 8);
    if (sequence == 8) {
      frame.depth_change_fraction = 0.70f;
      frame.raw_rgb_change_fraction.reset();
      frame.structural_change_fraction.reset();
    }
    auto emitted = planner.feed(std::move(frame));
    scenes.insert(
      scenes.end(),
      std::make_move_iterator(emitted.begin()),
      std::make_move_iterator(emitted.end())
    );
  }
  ASSERT_EQ(scenes.size(), 1u);
  EXPECT_EQ(scenes.front().boundary.final_sequence, 8u);
  EXPECT_EQ(
    scenes.front().boundary.decision,
    offline_sbs::boundary_decision_e::confirmed_causal_fallback
  );
  EXPECT_FALSE(scenes.front().boundary.selected_evidence_score);
}

TEST(OfflineScenePlanner, RejectsOutOfRangeFractionsAndRetainsBoundaryEvidence) {
  offline_sbs::scene_planner_t invalid(planner_config());
  auto bad = sample(1);
  bad.depth_change_fraction = 1.01f;
  EXPECT_THROW(invalid.feed(std::move(bad)), offline_sbs::scene_plan_error);
  EXPECT_EQ(invalid.open_cache_bytes(), 0u);

  offline_sbs::scene_planner_t planner(planner_config());
  for (std::uint64_t sequence = 1; sequence <= 5; ++sequence) {
    auto frame = sample(sequence);
    planner.feed(std::move(frame));
  }
  const auto scene = planner.finish().front();
  EXPECT_EQ(scene.evidence.source_frame_count, 5u);
  EXPECT_EQ(scene.evidence.depth_update_count, 5u);
  EXPECT_TRUE(scene.evidence.depth_change_max);
  EXPECT_FALSE(scene.ground_truth);
  EXPECT_FALSE(scene.known_limit.empty());
}
