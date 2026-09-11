#include <gtest/gtest.h>

#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>

#include <src/video_depth_estimator.h>
#include <src/host_sbs_adaptive_submission.h>
#include "../fixtures/host_sbs_gpu_completion_receipt.h"

namespace {
  namespace receipt = models::host_sbs_gpu_completion_receipt;
  namespace fixture = host_sbs_gpu_completion_receipt_fixture;
  namespace graph = cuda_conditional_graph;

  class HostSbsGpuCompletionReadbackTest : public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      ASSERT_EQ(D3D11CreateDevice(
                  nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u,
                  requested, 1u, D3D11_SDK_VERSION,
                  device.GetAddressOf(), &actual, context.GetAddressOf()), S_OK);
      ASSERT_GE(actual, D3D_FEATURE_LEVEL_11_0);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
  };

  TEST_F(HostSbsGpuCompletionReadbackTest, JointReuseSnapshotSurvivesSourceOverwriteAndFullSlots) {
    const auto expected = fixture::expected();
    const auto snapshot = fixture::snapshot(expected);
    ASSERT_TRUE(receipt::decode(snapshot, expected));
    const auto observed = models::detail::publication_receipt_readback_for_test(
      device.Get(), context.Get(), snapshot, expected
    );
    ASSERT_TRUE(observed.resources_created);
    ASSERT_TRUE(observed.completed);
    ASSERT_FALSE(observed.result.failed);
    ASSERT_EQ(observed.result.count, receipt::slot_count);
    EXPECT_EQ(observed.result.submitted, receipt::slot_count);
    EXPECT_EQ(observed.result.skipped, 1u);
    EXPECT_EQ(observed.result.decoded, receipt::slot_count);
    EXPECT_EQ(observed.result.decode_failed, 0u);
    std::array<bool, receipt::slot_count> seen {};
    for (std::size_t i = 0u; i < observed.result.count; ++i) {
      const auto &actual = observed.result.receipts[i];
      ASSERT_GE(actual.expected.publication_sequence, expected.publication_sequence);
      ASSERT_LT(actual.expected.publication_sequence, expected.publication_sequence + receipt::slot_count);
      const auto offset = actual.expected.publication_sequence - expected.publication_sequence;
      EXPECT_FALSE(seen[offset]);
      seen[offset] = true;
      EXPECT_EQ(actual.expected.transaction_token, expected.transaction_token);
      EXPECT_EQ(actual.depth, receipt::depth_disposition_e::reuse);
      EXPECT_TRUE(actual.depth_cache_authorized());
      EXPECT_EQ(actual.depth_owner_frame_id, expected.baseline_frame_id);
      EXPECT_EQ(actual.subtitle, receipt::subtitle_disposition_e::held_with_depth);
      EXPECT_EQ(actual.subtitle_frame_id, actual.depth_owner_frame_id);
      EXPECT_FALSE(actual.subtitle_is_current());
    }
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, OrdinaryReuseKeepsThePreviousSubtitleObservation) {
    const auto expected = fixture::expected(graph::work_flag_e::optional_ocr);
    const auto observed = models::detail::publication_receipt_readback_for_test(
      device.Get(), context.Get(), fixture::snapshot(expected), expected
    );
    ASSERT_TRUE(observed.completed);
    ASSERT_FALSE(observed.result.failed);
    ASSERT_EQ(observed.result.count, receipt::slot_count);
    for (std::size_t i = 0u; i < observed.result.count; ++i) {
      const auto &actual = observed.result.receipts[i];
      EXPECT_EQ(actual.depth_owner_frame_id, expected.baseline_frame_id);
      EXPECT_EQ(actual.subtitle, receipt::subtitle_disposition_e::held_with_depth);
      EXPECT_EQ(actual.subtitle_frame_id, expected.frame_id - 1u);
      EXPECT_FALSE(actual.subtitle_is_current());
    }
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, InferSnapshotIdentifiesTheNewDepthOwner) {
    const auto expected = fixture::expected(graph::work_flag_e::optional_ocr);
    const auto observed = models::detail::publication_receipt_readback_for_test(
      device.Get(), context.Get(), fixture::snapshot(expected, graph::branch_e::infer), expected
    );
    ASSERT_TRUE(observed.completed);
    ASSERT_FALSE(observed.result.failed);
    ASSERT_EQ(observed.result.count, receipt::slot_count);
    for (std::size_t i = 0u; i < observed.result.count; ++i) {
      const auto &actual = observed.result.receipts[i];
      EXPECT_EQ(actual.depth, receipt::depth_disposition_e::infer);
      EXPECT_EQ(actual.depth_owner_frame_id, expected.frame_id);
      EXPECT_EQ(actual.depth_owner_timestamp_us, expected.observation_timestamp_us);
    }
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, ReadySnapshotsFromAnOldEstimatorCannotAuthorizeItsSuccessor) {
    const auto expected = fixture::expected();
    const auto observed = models::detail::publication_receipt_readback_for_test(
      device.Get(), context.Get(), fixture::snapshot(expected), expected,
      expected.estimator_generation + 1u
    );
    ASSERT_TRUE(observed.resources_created);
    ASSERT_TRUE(observed.completed);
    EXPECT_EQ(observed.result.count, 0u);
    EXPECT_EQ(observed.result.diagnostic_count, 0u);
    EXPECT_EQ(observed.result.decoded, 0u);
    EXPECT_EQ(observed.result.decode_failed, receipt::slot_count);
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, CompletedReceiptAuthorizesActualOwnerAndRejectsOverwrittenPublication) {
    const auto expected = fixture::expected();
    const auto observed = models::detail::publication_receipt_readback_for_test(
      device.Get(), context.Get(), fixture::snapshot(expected), expected
    );
    ASSERT_TRUE(observed.completed);
    ASSERT_FALSE(observed.result.failed);
    ASSERT_GT(observed.result.count, 0u);
    const auto &completed = observed.result.receipts[0u];
    ASSERT_NE(completed.depth_owner_frame_id, completed.expected.frame_id);
    const auto next_frame = completed.expected.frame_id + 1u;
    const auto next_time = completed.expected.observation_timestamp_us + 1u;

    const auto missing = models::make_gpu_adaptive_request(
      next_frame, true, nullptr, completed.expected, next_time
    );
    EXPECT_FALSE(missing.authorize_gpu_undecided_reuse);

    const auto ready = models::make_gpu_adaptive_request(
      next_frame, true, &completed, completed.expected, next_time
    );
    EXPECT_TRUE(ready.authorize_gpu_undecided_reuse);
    EXPECT_EQ(ready.baseline_frame_id, completed.depth_owner_frame_id);
    EXPECT_NE(ready.baseline_frame_id, completed.expected.frame_id);

    auto overwritten = completed.expected;
    ++overwritten.publication_sequence;
    const auto stale = models::make_gpu_adaptive_request(
      next_frame, true, &completed, overwritten, next_time
    );
    EXPECT_FALSE(stale.authorize_gpu_undecided_reuse);
    EXPECT_EQ(stale.baseline_frame_id, 0u);
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, DiagnosticsShareTheBoundedSnapshotAndPreserveCopyTime) {
    const auto expected = fixture::expected();
    const auto before_copy = std::chrono::steady_clock::now();
    const auto observed = models::detail::publication_receipt_readback_for_test(
      device.Get(), context.Get(), fixture::snapshot(expected), expected
    );
    const auto after_poll = std::chrono::steady_clock::now();
    ASSERT_TRUE(observed.completed);
    ASSERT_EQ(observed.result.diagnostic_count, receipt::slot_count);
    EXPECT_EQ(observed.result.skipped, 1u);
    for (std::size_t i = 0u; i < observed.result.diagnostic_count; ++i) {
      const auto &diagnostic = observed.result.diagnostics[i];
      EXPECT_FALSE(diagnostic.failed);
      ASSERT_TRUE(diagnostic.cut_state);
      ASSERT_TRUE(diagnostic.outcome_counts);
      EXPECT_EQ(diagnostic.expected.publication_sequence, expected.publication_sequence + i);
      EXPECT_GE(diagnostic.sampled_at, before_copy);
      EXPECT_LE(diagnostic.sampled_at, after_poll);
      EXPECT_EQ(diagnostic.outcome_counts->infer, 10u);
      EXPECT_EQ(diagnostic.outcome_counts->reuse, 7u);
      EXPECT_EQ(diagnostic.outcome_counts->invalid, 2u);
    }
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, AbsentOptionalCopiesNeverExposeUninitializedStagingBytes) {
    const auto expected = fixture::expected();
    for (const auto &present : {std::pair {false, false}, std::pair {true, false}, std::pair {false, true}}) {
      const auto observed = models::detail::publication_receipt_readback_for_test(
        device.Get(), context.Get(), fixture::snapshot(expected), expected,
        0u, present.first, present.second
      );
      ASSERT_TRUE(observed.completed);
      EXPECT_FALSE(observed.result.failed);
      ASSERT_EQ(observed.result.count, receipt::slot_count);
      ASSERT_EQ(observed.result.diagnostic_count, receipt::slot_count);
      for (std::size_t i = 0u; i < observed.result.diagnostic_count; ++i) {
        const auto &diagnostic = observed.result.diagnostics[i];
        EXPECT_FALSE(diagnostic.failed);
        EXPECT_EQ(diagnostic.cut_state.has_value(), present.first);
        EXPECT_EQ(diagnostic.outcome_counts.has_value(), present.second);
      }
    }
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, InvalidRenderProofStillPublishesValidCumulativeDiagnostics) {
    const auto expected = fixture::expected();
    auto snapshot = fixture::snapshot(expected);
    snapshot[6u] = 0u;
    const auto observed = models::detail::publication_receipt_readback_for_test(
      device.Get(), context.Get(), snapshot, expected
    );
    ASSERT_TRUE(observed.completed);
    EXPECT_TRUE(observed.result.failed);
    EXPECT_FALSE(observed.result.transport_failed);
    EXPECT_EQ(observed.result.count, 0u);
    EXPECT_EQ(observed.result.decode_failed, receipt::slot_count);
    ASSERT_EQ(observed.result.diagnostic_count, receipt::slot_count);
    for (std::size_t i = 0u; i < observed.result.diagnostic_count; ++i) {
      const auto &diagnostic = observed.result.diagnostics[i];
      EXPECT_FALSE(diagnostic.failed);
      ASSERT_TRUE(diagnostic.cut_state);
      ASSERT_TRUE(diagnostic.outcome_counts);
      EXPECT_EQ(diagnostic.outcome_counts->invalid, 2u);
    }
  }

  TEST_F(HostSbsGpuCompletionReadbackTest, CorruptDiagnosticsDoNotRevokeRenderingOrTheOtherPayload) {
    const auto expected = fixture::expected();
    for (const auto corrupt_cut : {false, true}) {
      auto snapshot = fixture::snapshot(expected);
      snapshot[corrupt_cut ? receipt::cut_state_begin : receipt::outcome_counts_begin] ^= 1u;
      const auto observed = models::detail::publication_receipt_readback_for_test(
        device.Get(), context.Get(), snapshot, expected
      );
      ASSERT_TRUE(observed.completed);
      EXPECT_FALSE(observed.result.failed);
      EXPECT_FALSE(observed.result.transport_failed);
      EXPECT_EQ(observed.result.count, receipt::slot_count);
      EXPECT_EQ(observed.result.decode_failed, 0u);
      ASSERT_EQ(observed.result.diagnostic_count, receipt::slot_count);
      for (std::size_t i = 0u; i < observed.result.diagnostic_count; ++i) {
        const auto &diagnostic = observed.result.diagnostics[i];
        EXPECT_TRUE(diagnostic.failed);
        EXPECT_EQ(diagnostic.cut_state.has_value(), !corrupt_cut);
        EXPECT_EQ(diagnostic.outcome_counts.has_value(), corrupt_cut);
      }
    }
  }
}  // namespace
#endif
