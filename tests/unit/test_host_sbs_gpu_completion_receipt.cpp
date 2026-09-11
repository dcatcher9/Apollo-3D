#include <gtest/gtest.h>

#include <tests/fixtures/host_sbs_gpu_completion_receipt.h>

#include <limits>

namespace {
  namespace fixture = host_sbs_gpu_completion_receipt_fixture;
  namespace receipt = models::host_sbs_gpu_completion_receipt;
  namespace graph = cuda_conditional_graph;
  namespace v2 = models::depth_coordinate_v2;

  TEST(HostSbsGpuCompletionReceipt, ReuseKeepsOneDepthAndSubtitleOwner) {
    const auto expected = fixture::expected();
    const auto decoded = receipt::decode(fixture::snapshot(expected), expected);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->depth, receipt::depth_disposition_e::reuse);
    EXPECT_TRUE(decoded->depth_cache_authorized());
    EXPECT_EQ(decoded->depth_owner_frame_id, expected.baseline_frame_id);
    EXPECT_NE(decoded->depth_owner_frame_id, expected.frame_id);
    EXPECT_EQ(decoded->subtitle, receipt::subtitle_disposition_e::held_with_depth);
    EXPECT_FALSE(decoded->subtitle_is_current());
    EXPECT_EQ(decoded->subtitle_frame_id, decoded->depth_owner_frame_id);
    EXPECT_TRUE(receipt::matches_publication(*decoded, expected));
  }

  TEST(HostSbsGpuCompletionReceipt, JointReuseDoesNotExpireItsAuthenticatedAnalysisOwner) {
    constexpr std::uint64_t owner_timestamp_us = 1'000'000u;
    for (const auto work : {graph::work_flag_e::optional_ocr, graph::work_flag_e::subtitle_observation}) {
      for (const auto gap : std::array<std::uint64_t, 3u> {3u, 65'536u, std::uint64_t {1u} << 40u}) {
        SCOPED_TRACE(gap);
        auto expected = fixture::expected(work);
        expected.frame_id = expected.baseline_frame_id + gap;
        expected.publication_sequence = gap;
        expected.observation_timestamp_us = owner_timestamp_us + gap * 60'000u;
        auto words = fixture::snapshot(expected);
        fixture::put_u64(words, receipt::history_owner_begin + 8u, owner_timestamp_us);
        const auto held = receipt::decode(words, expected);
        ASSERT_TRUE(held);
        EXPECT_TRUE(held->depth_cache_authorized());
        EXPECT_EQ(held->depth, receipt::depth_disposition_e::reuse);
        EXPECT_EQ(held->depth_owner_frame_id, expected.baseline_frame_id);
        EXPECT_EQ(held->depth_owner_timestamp_us, owner_timestamp_us);
        EXPECT_EQ(held->subtitle, receipt::subtitle_disposition_e::held_with_depth);
        EXPECT_EQ(held->subtitle_frame_id, held->depth_owner_frame_id);
        EXPECT_FALSE(held->subtitle_is_current());
      }
    }
  }

  TEST(HostSbsGpuCompletionReceipt, InferWithoutHistoryAdvanceCannotCreateAnOwner) {
    const auto expected = fixture::expected(graph::work_flag_e::optional_ocr);
    auto words = fixture::snapshot(expected, graph::branch_e::infer);
    words[receipt::history_owner_begin] = 0u;
    words[receipt::depth_frame_state_begin + 3u] = std::bit_cast<std::uint32_t>(0.0f);
    const auto decoded = receipt::decode(words, expected);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->depth, receipt::depth_disposition_e::infer);
    EXPECT_FALSE(decoded->depth_owner_valid);
    EXPECT_FALSE(decoded->depth_cache_authorized());
    EXPECT_EQ(decoded->depth_owner_frame_id, 0u);
    EXPECT_TRUE(decoded->subtitle_is_current());
  }

  TEST(HostSbsGpuCompletionReceipt, BothBranchesKeepDepthAndSubtitleOwnershipTogether) {
    for (const auto work : {graph::work_flag_e::optional_ocr, graph::work_flag_e::subtitle_observation}) {
      const auto expected = fixture::expected(work);
      const auto held = receipt::decode(fixture::snapshot(expected), expected);
      ASSERT_TRUE(held);
      EXPECT_EQ(held->subtitle, receipt::subtitle_disposition_e::held_with_depth);
      EXPECT_EQ(held->subtitle_frame_id, held->depth_owner_frame_id);
      const auto refreshed = receipt::decode(fixture::snapshot(expected, graph::branch_e::infer, false), expected);
      ASSERT_TRUE(refreshed);
      EXPECT_EQ(refreshed->subtitle, receipt::subtitle_disposition_e::abstention);
      EXPECT_TRUE(refreshed->subtitle_is_current());
      EXPECT_EQ(refreshed->subtitle_frame_id, refreshed->depth_owner_frame_id);
    }
  }

  TEST(HostSbsGpuCompletionReceipt, ReuseRejectsMixedHistoricalOwnersAndCurrentSubtitlePublication) {
    auto expected = fixture::expected();
    expected.baseline_frame_id = 7u;
    const auto good = fixture::snapshot(expected);
    ASSERT_TRUE(receipt::decode(good, expected));
    for (const auto subtitle_frame : {6u, 8u, 11u, 12u}) {
      auto words = good;
      fixture::put_u64(words, receipt::subtitle_locator_begin + 22u, subtitle_frame);
      EXPECT_FALSE(receipt::decode(words, expected));
    }
    for (const auto retired_work : {8u, 16u}) {
      const auto invalid = fixture::expected(static_cast<graph::work_flag_e>(retired_work));
      EXPECT_FALSE(receipt::valid_expected(invalid));
      EXPECT_FALSE(receipt::decode(fixture::snapshot(invalid), invalid));
    }
  }

  TEST(HostSbsGpuCompletionReceipt, ForcePublicationWithoutSourceTimeDoesNotInventHistoryAuthority) {
    auto expected = fixture::expected(graph::work_flag_e::none, receipt::submission_class_e::force_infer);
    expected.observation_timestamp_us = 0u;
    auto words = fixture::snapshot(expected, graph::branch_e::infer);
    words[receipt::history_owner_begin] = 0u;
    const auto decoded = receipt::decode(words, expected);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->depth, receipt::depth_disposition_e::infer);
    EXPECT_TRUE(decoded->geometry_valid);
    EXPECT_FALSE(decoded->depth_owner_valid);
    EXPECT_FALSE(decoded->depth_cache_authorized());
    EXPECT_EQ(decoded->depth_owner_timestamp_us, 0u);

    expected.submission_class = receipt::submission_class_e::gpu_undecided;
    expected.baseline_frame_id = expected.frame_id - 1u;
    EXPECT_FALSE(receipt::decode(words, expected));
  }

  TEST(HostSbsGpuCompletionReceipt, SuppressionDoesNotInventASubtitleObservation) {
    const auto expected = fixture::expected(graph::work_flag_e::none);
    const auto decoded = receipt::decode(fixture::snapshot(expected), expected);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->subtitle, receipt::subtitle_disposition_e::suppressed);
    EXPECT_FALSE(decoded->subtitle_is_current());
    EXPECT_EQ(decoded->subtitle_frame_id, 0u);
    EXPECT_TRUE(decoded->depth_cache_authorized());
  }

  TEST(HostSbsGpuCompletionReceipt, TokenAndFinalizerVerdictsMustMatchTheCompletedUnit) {
    const auto expected = fixture::expected();
    const auto good = fixture::snapshot(expected);
    for (const auto corrupt_word : {2u, 6u, 8u, 12u, 20u, 40u, 44u, 48u, 52u, 60u}) {
      auto words = good;
      words[corrupt_word] ^= 1u;
      EXPECT_FALSE(receipt::decode(words, expected)) << corrupt_word;
    }
    auto other = expected;
    ++other.transaction_token;
    EXPECT_FALSE(receipt::decode(good, other));
  }

  TEST(HostSbsGpuCompletionReceipt, HistoryIdentityAndValidityAreRequiredIndependentlyOfBranch) {
    const auto expected = fixture::expected();
    const auto good = fixture::snapshot(expected);
    for (const auto offset : {0u, 1u, 4u, 6u, 7u}) {
      auto words = good;
      words[receipt::history_owner_begin + offset] ^= 1u;
      EXPECT_FALSE(receipt::decode(words, expected)) << offset;
    }
    auto words = good;
    fixture::put_u64(words, receipt::history_owner_begin + 2u, expected.frame_id);
    EXPECT_FALSE(receipt::decode(words, expected));
    words = good;
    fixture::put_u64(words, receipt::history_owner_begin + 8u, expected.observation_timestamp_us + 1u);
    EXPECT_FALSE(receipt::decode(words, expected));
    words = good;
    words[receipt::depth_frame_state_begin + 3u] = 0u;
    EXPECT_FALSE(receipt::decode(words, expected));
  }

  TEST(HostSbsGpuCompletionReceipt, MalformedParallaxOrMinmaxStateNeverAuthorizesCache) {
    const auto expected = fixture::expected();
    const auto good = fixture::snapshot(expected);
    for (const auto offset : {v2::contract_tag_bits, v2::camera_center_integrity_bits, v2::renderer_authorization_bits}) {
      auto words = good;
      words[receipt::parallax_state_begin + offset] ^= 1u;
      EXPECT_FALSE(receipt::decode(words, expected));
    }
    for (std::size_t offset = 0u; offset < receipt::depth_frame_state_word_count; ++offset) {
      auto words = good;
      words[receipt::depth_frame_state_begin + offset] = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN());
      EXPECT_FALSE(receipt::decode(words, expected));
    }
    auto words = good;
    words[receipt::depth_frame_state_begin + 3u] = std::bit_cast<std::uint32_t>(3.0f);
    EXPECT_FALSE(receipt::decode(words, expected));
    words = good;
    words[receipt::depth_frame_state_begin + 2u] = 0u;
    EXPECT_FALSE(receipt::decode(words, expected));
    words = good;
    words[receipt::depth_frame_state_begin] = std::bit_cast<std::uint32_t>(2.0f);
    EXPECT_FALSE(receipt::decode(words, expected));
  }

  TEST(HostSbsGpuCompletionReceipt, SubtitleUsesAnalysisGenerationAndRejectsStaleObservation) {
    auto expected = fixture::expected();
    expected.analysis_generation = 0u;  // Full-source domains legitimately use generation zero.
    auto words = fixture::snapshot(expected, graph::branch_e::infer);
    ASSERT_TRUE(receipt::decode(words, expected));
    fixture::put_u64(words, receipt::subtitle_locator_begin + 10u, expected.domain_tag);
    EXPECT_FALSE(receipt::decode(words, expected));
    words = fixture::snapshot(expected, graph::branch_e::infer);
    fixture::put_u64(words, receipt::subtitle_locator_begin + 22u, expected.frame_id - 1u);
    EXPECT_FALSE(receipt::decode(words, expected));
    words = fixture::snapshot(expected, graph::branch_e::infer);
    words[receipt::subtitle_locator_begin + 2u] |= 1u << 31u;
    EXPECT_FALSE(receipt::decode(words, expected));
  }

  TEST(HostSbsGpuCompletionReceipt, ConditionPublicationMustMatchCurrentOrProvisionalLocator) {
    const auto expected = fixture::expected();
    const auto locator = receipt::subtitle_locator_begin;
    const auto condition = receipt::subtitle_condition_begin;
    auto words = fixture::snapshot(expected);
    words[condition] = v2::subtitle_condition_param_schema;
    EXPECT_FALSE(receipt::decode(words, expected));  // An empty observation requires canonical zeros.

    words[locator + 20u] = 1u;
    words[locator + 24u] = 2u;
    words[locator + 18u] = std::bit_cast<std::uint32_t>(0.01f);
    words[condition + 1u] = v2::subtitle_condition_param_tag;
    words[condition + 2u] = 1u;
    words[condition + 4u] = 2u;
    words[condition + 5u] = words[locator + 18u];
    ASSERT_TRUE(receipt::decode(words, expected));
    words[condition + 5u] ^= 1u;
    EXPECT_FALSE(receipt::decode(words, expected));

    words[locator + 2u] = v2::subtitle_locator_provisional_current_flag;
    words[locator + v2::subtitle_locator_provisional_target_word] = std::bit_cast<std::uint32_t>(0.02f);
    words[locator + v2::subtitle_locator_provisional_fade_word] = 1u;
    words[condition + 4u] = 1u;
    words[condition + 5u] = words[locator + v2::subtitle_locator_provisional_target_word];
    EXPECT_TRUE(receipt::decode(words, expected));
  }

  TEST(HostSbsGpuCompletionReceipt, PublicationSequenceRejectsStaleBorrowedResourcesWithRepeatedFrameId) {
    const auto expected = fixture::expected();
    const auto decoded = receipt::decode(fixture::snapshot(expected), expected);
    ASSERT_TRUE(decoded);
    auto current = expected;
    ++current.publication_sequence;
    EXPECT_FALSE(receipt::matches_publication(*decoded, current));
    current = expected;
    ++current.estimator_generation;
    EXPECT_FALSE(receipt::matches_publication(*decoded, current));
    current = expected;
    ++current.analysis_generation;
    EXPECT_FALSE(receipt::matches_publication(*decoded, current));
  }

  TEST(HostSbsGpuCompletionReceipt, BoundedSlotsKeepPendingProofAcrossLateAndStaleCompletions) {
    std::array<receipt::slot_state_t, receipt::slot_count> slots;
    auto expected = fixture::expected();
    for (auto &slot : slots) {
      ASSERT_TRUE(slot.reserve(expected));
      ++expected.publication_sequence;
    }
    for (auto &slot : slots) {
      EXPECT_TRUE(slot.pending());
      EXPECT_FALSE(slot.reserve(expected));
    }
    // A late valid completion retires only its own slot and cannot match the new publication.
    const auto oldest = *slots[0u].expected();
    const auto decoded = slots[0u].retire(fixture::snapshot(oldest), oldest.estimator_generation);
    ASSERT_TRUE(decoded);
    EXPECT_FALSE(receipt::matches_publication(*decoded, expected));
    EXPECT_FALSE(slots[0u].pending());
    ASSERT_TRUE(slots[0u].reserve(expected));
    EXPECT_TRUE(slots[1u].pending());
    EXPECT_TRUE(slots[2u].pending());

    // Resetting an estimator cannot make an unsignalled old slot reusable. Once its event is
    // actually ready, the old generation is discarded instead of publishing stale authority.
    const auto old_generation = *slots[1u].expected();
    auto fresh = expected;
    ++fresh.estimator_generation;
    EXPECT_FALSE(slots[1u].reserve(fresh));
    EXPECT_FALSE(slots[1u].retire(fixture::snapshot(old_generation), fresh.estimator_generation));
    EXPECT_TRUE(slots[1u].reserve(fresh));
  }

  TEST(HostSbsGpuCompletionReceipt, InvalidReadyReceiptFreesItsSlotWithoutPublishingAuthority) {
    receipt::slot_state_t slot;
    const auto expected = fixture::expected();
    ASSERT_TRUE(slot.reserve(expected));
    auto words = fixture::snapshot(expected);
    words[6u] = 0u;
    EXPECT_FALSE(slot.retire(words, expected.estimator_generation));
    EXPECT_FALSE(slot.pending());
    EXPECT_FALSE(slot.retire(words, expected.estimator_generation));
    EXPECT_TRUE(slot.reserve(expected));
  }

  TEST(HostSbsGpuCompletionReceipt, DiagnosticsPreserveCaptureIdentityAndCumulativeCounts) {
    const auto expected = fixture::expected();
    const auto sampled_at = std::chrono::steady_clock::time_point {std::chrono::milliseconds {1234}};
    const auto diagnostics = receipt::decode_diagnostics(fixture::snapshot(expected), expected, true, true, sampled_at);
    EXPECT_FALSE(diagnostics.failed);
    EXPECT_EQ(diagnostics.expected, expected);
    EXPECT_EQ(diagnostics.sampled_at, sampled_at);
    ASSERT_TRUE(diagnostics.cut_state);
    ASSERT_TRUE(diagnostics.outcome_counts);
    EXPECT_EQ(*diagnostics.cut_state, sbs_adaptive_state::initial_words);
    EXPECT_EQ(diagnostics.outcome_counts->infer, 10u);
    EXPECT_EQ(diagnostics.outcome_counts->reuse, 7u);
    EXPECT_EQ(diagnostics.outcome_counts->invalid, 2u);
  }

  TEST(HostSbsGpuCompletionReceipt, MissingDiagnosticCopiesCannotExposePreviousSlotContents) {
    const auto expected = fixture::expected();
    auto words = fixture::snapshot(expected);
    auto diagnostic = receipt::decode_diagnostics(words, expected, false, false, {});
    EXPECT_FALSE(diagnostic.failed);
    EXPECT_FALSE(diagnostic.cut_state);
    EXPECT_FALSE(diagnostic.outcome_counts);
    words[receipt::cut_state_begin] ^= 1u;
    words[receipt::outcome_counts_begin] ^= 1u;
    diagnostic = receipt::decode_diagnostics(words, expected, false, false, {});
    EXPECT_FALSE(diagnostic.failed);
    EXPECT_FALSE(diagnostic.cut_state);
    EXPECT_FALSE(diagnostic.outcome_counts);
    EXPECT_TRUE(receipt::decode(words, expected));
  }

  TEST(HostSbsGpuCompletionReceipt, DiagnosticAndFunctionalFailuresRemainIndependent) {
    const auto expected = fixture::expected();
    const auto good = fixture::snapshot(expected);
    auto words = good;
    words[6u] = 0u;
    EXPECT_FALSE(receipt::decode(words, expected));
    auto diagnostic = receipt::decode_diagnostics(words, expected, true, true, {});
    EXPECT_FALSE(diagnostic.failed);
    EXPECT_TRUE(diagnostic.cut_state);
    ASSERT_TRUE(diagnostic.outcome_counts);
    EXPECT_EQ(diagnostic.outcome_counts->invalid, 2u);

    words = good;
    words[receipt::cut_state_begin] ^= 1u;
    EXPECT_TRUE(receipt::decode(words, expected));
    diagnostic = receipt::decode_diagnostics(words, expected, true, true, {});
    EXPECT_TRUE(diagnostic.failed);
    EXPECT_FALSE(diagnostic.cut_state);
    EXPECT_TRUE(diagnostic.outcome_counts);

    words = good;
    words[receipt::outcome_counts_begin + 1u] ^= 1u;
    EXPECT_TRUE(receipt::decode(words, expected));
    diagnostic = receipt::decode_diagnostics(words, expected, true, true, {});
    EXPECT_TRUE(diagnostic.failed);
    EXPECT_TRUE(diagnostic.cut_state);
    EXPECT_FALSE(diagnostic.outcome_counts);
  }

  TEST(HostSbsGpuCompletionReceipt, CutDiagnosticsUseTheExistingFiniteReservedAndCounterContract) {
    const auto expected = fixture::expected();
    const auto good = fixture::snapshot(expected);
    using sbs_adaptive_state::word_e;
    for (const auto field : {word_e::reserved_cut_bridge_2, word_e::hard_cut_pulse,
                            word_e::model_input_history_state, word_e::analysis_flags,
                            word_e::cut_flags, word_e::scene_age}) {
      auto words = good;
      words[receipt::cut_state_begin + sbs_adaptive_state::index(field)] = std::bit_cast<std::uint32_t>(-1.0f);
      const auto diagnostic = receipt::decode_diagnostics(words, expected, true, true, {});
      EXPECT_TRUE(diagnostic.failed);
      EXPECT_FALSE(diagnostic.cut_state);
      EXPECT_TRUE(diagnostic.outcome_counts);
      EXPECT_TRUE(receipt::decode(words, expected));
    }
    auto words = good;
    words[receipt::cut_state_begin + sbs_adaptive_state::index(word_e::current_depth_change_fraction)] =
      std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN());
    EXPECT_FALSE(receipt::decode_diagnostics(words, expected, true, false, {}).cut_state);
    words = good;
    words[receipt::cut_state_begin + sbs_adaptive_state::index(word_e::hard_cut_count)] =
      std::numeric_limits<std::uint32_t>::max();
    EXPECT_FALSE(receipt::decode_diagnostics(words, expected, true, false, {}).cut_state);
  }

  TEST(HostSbsGpuCompletionReceipt, CumulativeDiagnosticsRetainEventsAcrossSkippedReceipts) {
    const auto expected = fixture::expected();
    auto words = fixture::snapshot(expected);
    models::host_sbs_gpu_outcomes::delta_tracker_t tracker;
    auto first = receipt::decode_diagnostics(words, expected, false, true, {});
    ASSERT_TRUE(first.outcome_counts);
    EXPECT_FALSE(tracker.observe(*first.outcome_counts));
    const auto initial = tracker.take_delta();
    EXPECT_EQ(initial.infer, 10u);
    EXPECT_EQ(initial.reuse, 7u);
    EXPECT_EQ(initial.invalid, 2u);
    // The slot ring can skip intermediate snapshots without dropping their GPU-counted events.
    fixture::put_u64(words, receipt::outcome_counts_begin + 2u, 18u);
    fixture::put_u64(words, receipt::outcome_counts_begin + 4u, 23u);
    fixture::put_u64(words, receipt::outcome_counts_begin + 6u, 4u);
    const auto later = receipt::decode_diagnostics(words, expected, false, true, {});
    ASSERT_TRUE(later.outcome_counts);
    EXPECT_FALSE(tracker.observe(*later.outcome_counts));
    const auto delta = tracker.take_delta();
    EXPECT_EQ(delta.infer, 8u);
    EXPECT_EQ(delta.reuse, 16u);
    EXPECT_EQ(delta.invalid, 2u);
  }
}  // namespace
