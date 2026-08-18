/**
 * @file tests/unit/platform/test_windows_misc.cpp
 * @brief Tests for Windows platform state helpers.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <algorithm>
  #include <boost/filesystem/path.hpp>
  #include <boost/process/v1/child.hpp>
  #include <boost/process/v1/environment.hpp>
  #include <boost/process/v1/group.hpp>
  #include <cstdint>
  #include <cstdio>
  #include <filesystem>
  #include <optional>
  // display.h declares the small D3DKMT function table without depending on the WDK headers.
  // Match display_base.cpp's prerequisite after boost/process has selected its Windows headers.
  typedef long NTSTATUS;
  #include <src/host_sbs_resolution.h>
  #include <src/platform/windows/display.h>
  #include <src/platform/windows/misc.h>
  #include <sddl.h>
  #include <ShlObj.h>
  #include <string>
  #include <vector>

namespace {
  struct fake_keyed_mutex_t {
    HRESULT acquire_result = S_OK;
    unsigned acquire_calls = 0;
    unsigned release_calls = 0;
    UINT64 acquired_key = 0;
    UINT64 released_key = 0;

    HRESULT AcquireSync(const UINT64 key, DWORD) noexcept {
      ++acquire_calls;
      acquired_key = key;
      return acquire_result;
    }

    HRESULT ReleaseSync(const UINT64 key) noexcept {
      ++release_calls;
      released_key = key;
      return S_OK;
    }
  };

  TEST(WindowsKeyedMutexLockTest, ReleasesSuccessfulAcquisitionOnEarlyReturn) {
    fake_keyed_mutex_t mutex;
    const auto conversion_with_early_return = [&]() {
      platf::dxgi::detail::keyed_mutex_lock_t lock {&mutex};
      if (lock.lock(3, INFINITE, 7) != S_OK) {
        return true;
      }
      return false;
    };

    EXPECT_FALSE(conversion_with_early_return());
    EXPECT_EQ(mutex.acquire_calls, 1u);
    EXPECT_EQ(mutex.release_calls, 1u);
    EXPECT_EQ(mutex.acquired_key, 3u);
    EXPECT_EQ(mutex.released_key, 7u);
  }

  TEST(WindowsKeyedMutexLockTest, DoesNotReleaseFailedAcquisition) {
    fake_keyed_mutex_t mutex;
    mutex.acquire_result = WAIT_TIMEOUT;
    {
      platf::dxgi::detail::keyed_mutex_lock_t lock {&mutex};
      EXPECT_EQ(lock.lock(), WAIT_TIMEOUT);
      EXPECT_FALSE(lock.owns_lock());
    }

    EXPECT_EQ(mutex.acquire_calls, 1u);
    EXPECT_EQ(mutex.release_calls, 0u);
  }

  TEST(WindowsDdupTimestampTest, CursorOnlyUpdateRetainsDesktopContentTimestamp) {
    using timestamp_t = std::int64_t;
    const auto initial_present = platf::dxgi::detail::select_ddup_timestamps(
      std::optional<timestamp_t> {100},
      std::optional<timestamp_t> {},
      std::optional<timestamp_t> {}
    );
    ASSERT_EQ(initial_present.presentation_timestamp, 100);
    ASSERT_EQ(initial_present.content_timestamp, 100);

    const auto cursor_only = platf::dxgi::detail::select_ddup_timestamps(
      std::optional<timestamp_t> {},
      std::optional<timestamp_t> {200},
      initial_present.content_timestamp
    );
    EXPECT_EQ(cursor_only.presentation_timestamp, 200);
    EXPECT_EQ(cursor_only.content_timestamp, 100);
  }

  TEST(WindowsDdupTimestampTest, NewerCursorCadenceDoesNotReplacePresentContentTime) {
    using timestamp_t = std::int64_t;
    const auto timestamps = platf::dxgi::detail::select_ddup_timestamps(
      std::optional<timestamp_t> {300},
      std::optional<timestamp_t> {350},
      std::optional<timestamp_t> {100}
    );

    EXPECT_EQ(timestamps.presentation_timestamp, 350);
    EXPECT_EQ(timestamps.content_timestamp, 300);
  }

  TEST(WindowsDdupDamageTest, NormalizesDirtyAndBothSidesOfMoveRects) {
    const RECT dirty {10, 20, 30, 40};
    DXGI_OUTDUPL_MOVE_RECT move {};
    move.SourcePoint = {100, 120};
    move.DestinationRect = {200, 220, 240, 260};

    const auto update = platf::dxgi::detail::make_ddup_damage_update(
      std::span<const RECT> {&dirty, 1u},
      std::span<const DXGI_OUTDUPL_MOVE_RECT> {&move, 1u},
      640,
      480
    );

    ASSERT_TRUE(update.known);
    ASSERT_EQ(update.rects.size(), 3u);
    EXPECT_EQ(update.rects[0].left, 10);
    EXPECT_EQ(update.rects[1].left, 100);
    EXPECT_EQ(update.rects[1].top, 120);
    EXPECT_EQ(update.rects[1].right, 140);
    EXPECT_EQ(update.rects[1].bottom, 160);
    EXPECT_EQ(update.rects[2].left, 200);
    EXPECT_EQ(update.rects[2].top, 220);
  }

  TEST(WindowsDdupDamageTest, ExactRangeReportsChangedUnchangedAndCursorIdentity) {
    using enum platf::dxgi::detail::ddup_damage_intersection_e;
    auto history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto base = history->commit({true, {}});
    const auto outside = history->commit({true, {{300, 300, 340, 340}}});
    const auto inside = history->commit({true, {{40, 40, 80, 80}}});
    const RECT roi {0, 0, 100, 100};

    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(base, base, roi),
      unchanged
    );
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(base, outside, roi),
      unchanged
    );
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(outside, inside, roi),
      changed
    );
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(base, inside, roi),
      changed
    );
  }

  TEST(WindowsDdupDamageTest, ClassifiesFullContentAndExactRoiProofsSeparately) {
    using enum platf::dxgi::detail::host_sbs_ddup_reuse_proof_e;
    using clock_t = std::chrono::steady_clock;
    const auto first_content = clock_t::time_point {std::chrono::milliseconds {1}};
    const auto second_content = clock_t::time_point {std::chrono::milliseconds {2}};
    const RECT roi {0, 0, 100, 100};
    auto history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto baseline = history->commit({true, {}});
    const auto outside = history->commit({true, {{200, 200, 220, 220}}});
    const auto inside = history->commit({true, {{50, 50, 60, 60}}});

    EXPECT_EQ(
      platf::dxgi::detail::classify_host_sbs_ddup_reuse(
        first_content, std::nullopt, false, roi, first_content, std::nullopt
      ),
      content_clock
    );
    EXPECT_EQ(
      platf::dxgi::detail::classify_host_sbs_ddup_reuse(
        first_content, baseline, false, roi, second_content, outside
      ),
      none
    );
    EXPECT_EQ(
      platf::dxgi::detail::classify_host_sbs_ddup_reuse(
        first_content, baseline, true, roi, first_content, baseline
      ),
      content_clock
    );
    EXPECT_EQ(
      platf::dxgi::detail::classify_host_sbs_ddup_reuse(
        first_content, baseline, true, roi, second_content, outside
      ),
      roi_damage
    );
    EXPECT_EQ(
      platf::dxgi::detail::classify_host_sbs_ddup_reuse(
        first_content, baseline, true, roi, second_content, inside
      ),
      none
    );
  }

  TEST(WindowsDdupDamageTest, UnknownAndUnrelatedHistoriesFailOpen) {
    using enum platf::dxgi::detail::ddup_damage_intersection_e;
    auto first_history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    auto second_history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto base = first_history->commit({true, {}});
    const auto discontinuity = first_history->commit({false, {}});
    const auto unrelated = second_history->commit({true, {}});
    const RECT roi {0, 0, 100, 100};

    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(base, discontinuity, roi),
      unknown
    );
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(base, unrelated, roi),
      unknown
    );
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(std::nullopt, base, roi),
      unknown
    );
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(base, base, RECT {0, 0, 0, 100}),
      unknown
    );
  }

  TEST(WindowsDdupDamageTest, EvictedSequenceRangeFailsOpen) {
    using enum platf::dxgi::detail::ddup_damage_intersection_e;
    auto history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto base = history->commit({true, {}});
    platf::dxgi::detail::ddup_damage_snapshot_t current;
    for (
      std::size_t i = 0u;
      i <= platf::dxgi::detail::ddup_damage_history_frame_budget;
      ++i
    ) {
      current = history->commit({true, {}});
    }

    const RECT roi {0, 0, 100, 100};
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(base, current, roi),
      unknown
    );
    const auto recent = history->commit({true, {}});
    EXPECT_EQ(
      platf::dxgi::detail::query_ddup_damage_between(current, recent, roi),
      unchanged
    );
  }

  TEST(WindowsDdupDamageTest, InvalidAndOversizeMetadataBecomeUnknown) {
    const RECT invalid_dirty {-1, 0, 10, 10};
    EXPECT_FALSE(
      platf::dxgi::detail::make_ddup_damage_update(
        std::span<const RECT> {&invalid_dirty, 1u},
        {},
        640,
        480
      ).known
    );

    DXGI_OUTDUPL_MOVE_RECT invalid_move {};
    invalid_move.SourcePoint = {630, 470};
    invalid_move.DestinationRect = {10, 10, 30, 30};
    EXPECT_FALSE(
      platf::dxgi::detail::make_ddup_damage_update(
        {},
        std::span<const DXGI_OUTDUPL_MOVE_RECT> {&invalid_move, 1u},
        640,
        480
      ).known
    );

    std::vector<RECT> too_many(
      platf::dxgi::detail::ddup_damage_frame_rect_budget + 1u,
      RECT {0, 0, 1, 1}
    );
    EXPECT_FALSE(
      platf::dxgi::detail::make_ddup_damage_update(
        too_many,
        {},
        640,
        480
      ).known
    );
    EXPECT_FALSE(
      platf::dxgi::detail::make_ddup_damage_update({}, {}, 640, 480, false).known
    );
  }

  TEST(WindowsDdupDamageTest, ContentPresentWithoutUpdateMetadataFailsOpen) {
    DXGI_OUTDUPL_FRAME_INFO frame_info {};
    frame_info.LastPresentTime.QuadPart = 1;
    frame_info.AccumulatedFrames = 1u;

    EXPECT_FALSE(
      platf::dxgi::detail::ddup_frame_damage_metadata_available(frame_info)
    );
    frame_info.TotalMetadataBufferSize = static_cast<UINT>(sizeof(RECT));
    EXPECT_TRUE(
      platf::dxgi::detail::ddup_frame_damage_metadata_available(frame_info)
    );
  }

  TEST(WindowsDdupDamageTest, CoverageSumsOverlapsAndSaturatesAtRegionArea) {
    auto history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto base = history->commit({true, {}});
    const auto overlap_twice = history->commit({
      true,
      {
        {10, 10, 20, 20},
        {10, 10, 20, 20},
        {95, 95, 110, 110},
      },
    });
    const RECT roi {0, 0, 100, 100};

    const auto coverage =
      platf::dxgi::detail::query_ddup_damage_coverage_between(
        base,
        overlap_twice,
        roi,
        true
      );
    ASSERT_TRUE(coverage.known);
    EXPECT_EQ(coverage.region_area, 10000u);
    // The repeated 10x10 rectangle is counted twice on purpose; the clipped corner adds 5x5.
    EXPECT_EQ(coverage.potentially_changed_area, 225u);
    EXPECT_EQ(coverage.max_single_intersection_area, 100u);

    const auto saturated = history->commit({
      true,
      {
        {0, 0, 100, 100},
        {0, 0, 100, 100},
      },
    });
    const auto saturated_coverage =
      platf::dxgi::detail::query_ddup_damage_coverage_between(
        overlap_twice,
        saturated,
        roi,
        true
      );
    ASSERT_TRUE(saturated_coverage.known);
    EXPECT_EQ(saturated_coverage.potentially_changed_area, 10000u);
    EXPECT_EQ(saturated_coverage.max_single_intersection_area, 10000u);
  }

  TEST(WindowsDdupDamageTest, CoverageDiscontinuityAndCrossHistoryFailOpen) {
    auto first_history =
      std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    auto second_history =
      std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto base = first_history->commit({true, {}});
    const auto discontinuity = first_history->commit({false, {}});
    const auto unrelated = second_history->commit({true, {}});
    const RECT roi {0, 0, 100, 100};

    EXPECT_FALSE(
      platf::dxgi::detail::query_ddup_damage_coverage_between(
        base,
        discontinuity,
        roi
      ).known
    );
    EXPECT_FALSE(
      platf::dxgi::detail::query_ddup_damage_coverage_between(
        base,
        unrelated,
        roi
      ).known
    );
  }

  TEST(WindowsHostSbsEncoderInputStateTest, FirstRepeatStillInitializesPersistentInput) {
    platf::dxgi::detail::host_sbs_encoder_input_state_t state;

    EXPECT_TRUE(state.conversion_required(true, false));
    state.mark_converted();
    EXPECT_FALSE(state.conversion_required(true, false));
    EXPECT_TRUE(state.conversion_required(false, false));
  }

  TEST(WindowsHostSbsEncoderInputStateTest, ResetInvalidatesPersistentInput) {
    platf::dxgi::detail::host_sbs_encoder_input_state_t state;
    state.mark_converted();
    ASSERT_TRUE(state.initialized());

    state.reset();

    EXPECT_FALSE(state.initialized());
    EXPECT_TRUE(state.conversion_required(true, false));
  }

  TEST(WindowsHostSbsEncoderInputStateTest, LocalRgbBackbufferIsNeverReused) {
    platf::dxgi::detail::host_sbs_encoder_input_state_t state;
    state.mark_converted();

    EXPECT_TRUE(state.conversion_required(true, true));
  }

  TEST(WindowsHostSbsP010MrtPolicyTest, RequiresEveryOptionalFastPathAuthority) {
    using platf::dxgi::detail::host_sbs_p010_y_mrt_eligible;
    constexpr std::array all_true {true, true, true, true, true, true, true, true};
    EXPECT_TRUE(host_sbs_p010_y_mrt_eligible(
      true, true, true, true, true, false, true, true
    ));

    // Each positive authority is independently fail-open to the established RGB conversion.
    for (std::size_t missing = 0; missing < all_true.size(); ++missing) {
      auto values = all_true;
      values[missing] = false;
      const bool dump_requested = !values[5];
      EXPECT_FALSE(host_sbs_p010_y_mrt_eligible(
        values[0], values[1], values[2], values[3], values[4], dump_requested,
        values[6], values[7]
      )) << "missing authority " << missing;
    }
  }

  TEST(WindowsLocalPresenterRetryTest, BusyFinalSourceSurvivesCaptureTimeout) {
    platf::dxgi::detail::local_presenter_retry_state_t state;
    EXPECT_FALSE(state.should_process(false, false, false));
    EXPECT_FALSE(state.should_convert(false, false, false));

    state.observe_source();
    EXPECT_TRUE(state.conversion_pending());
    EXPECT_FALSE(state.presentation_pending());
    EXPECT_TRUE(state.should_process(true, false, false));
    EXPECT_TRUE(state.should_convert(true, false, false));

    // A busy frame-latency wait leaves source conversion pending. Once conversion succeeds, a
    // busy Present retries the backbuffer without converting/enqueuing those pixels again.
    EXPECT_TRUE(state.should_convert(true, false, false));
    state.record_converted();
    EXPECT_FALSE(state.conversion_pending());
    EXPECT_TRUE(state.presentation_pending());
    // A busy Present leaves the converted backbuffer as the only pending step.
    EXPECT_TRUE(state.should_process(true, false, false));
    EXPECT_FALSE(state.should_convert(true, false, false));
    state.record_presented();
    EXPECT_FALSE(state.presentation_pending());
    EXPECT_FALSE(state.should_process(true, false, false));
    EXPECT_FALSE(state.should_convert(true, false, false));
  }

  TEST(WindowsLocalPresenterRetryTest, NewestSourceSupersedesPendingBackbuffer) {
    platf::dxgi::detail::local_presenter_retry_state_t state;
    state.observe_source();
    state.record_converted();
    ASSERT_TRUE(state.presentation_pending());

    // A newer retained source invalidates the old backbuffer retry and must be converted first.
    state.observe_source();
    EXPECT_TRUE(state.conversion_pending());
    EXPECT_FALSE(state.presentation_pending());
    EXPECT_TRUE(state.should_process(true, false, false));
    EXPECT_TRUE(state.should_convert(true, false, false));

    // Re-observing while conversion is already pending is idempotent.
    state.observe_source();
    EXPECT_TRUE(state.conversion_pending());
    EXPECT_FALSE(state.presentation_pending());
    state.record_converted();
    EXPECT_FALSE(state.conversion_pending());
    EXPECT_TRUE(state.presentation_pending());
    state.record_presented();
    EXPECT_FALSE(state.should_process(true, false, false));
  }

  TEST(WindowsLocalPresenterRetryTest, AsyncDepthRearmsPresentedStaticSource) {
    platf::dxgi::detail::local_presenter_retry_state_t state;
    state.observe_source();
    state.record_converted();
    state.record_presented();

    EXPECT_FALSE(state.should_convert(true, false, false));
    EXPECT_TRUE(state.should_convert(true, true, false));
    EXPECT_TRUE(state.should_convert(true, false, true));
    EXPECT_FALSE(state.should_convert(false, true, true));

    // The ready notification is consumed before conversion. If Present is then busy, the
    // converted depth result itself remains pending even with no second notification or poll.
    state.record_converted();
    EXPECT_TRUE(state.presentation_pending());
    EXPECT_TRUE(state.should_process(true, false, false));
    EXPECT_FALSE(state.should_convert(true, false, false));
    state.record_presented();
    EXPECT_FALSE(state.should_process(true, false, false));
    EXPECT_FALSE(state.should_convert(true, false, false));
  }

  TEST(WindowsDdupDamageTest, AdaptiveCandidateUsesCompleteHistoryNotDamageShape) {
    using platf::dxgi::detail::ddup_damage_coverage_t;
    using platf::dxgi::detail::host_sbs_adaptive_motion_damage_candidate;

    auto history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto accepted = history->commit({true, {}});
    DXGI_OUTDUPL_MOVE_RECT move {};
    move.SourcePoint = {0, 0};
    move.DestinationRect = {0, 0, 40, 100};
    const auto overlapping_move = platf::dxgi::detail::make_ddup_damage_update(
      {},
      std::span<const DXGI_OUTDUPL_MOVE_RECT> {&move, 1u},
      100,
      100
    );
    ASSERT_TRUE(overlapping_move.known);
    ASSERT_EQ(overlapping_move.rects.size(), 2u);
    const auto moved = history->commit(overlapping_move);
    const RECT roi {0, 0, 100, 100};
    const auto overlap = platf::dxgi::detail::query_ddup_damage_coverage_between(
      accepted,
      moved,
      roi,
      true
    );
    ASSERT_TRUE(overlap.known);
    EXPECT_EQ(overlap.potentially_changed_area, 8000u);
    EXPECT_EQ(overlap.max_single_intersection_area, 4000u);
    EXPECT_TRUE(host_sbs_adaptive_motion_damage_candidate(overlap));

    const auto broad = history->commit({true, {{0, 0, 50, 100}}});
    const auto broad_coverage =
      platf::dxgi::detail::query_ddup_damage_coverage_between(
        moved,
        broad,
        roi,
        true
      );
    EXPECT_EQ(broad_coverage.max_single_intersection_area, 5000u);
    EXPECT_TRUE(host_sbs_adaptive_motion_damage_candidate(broad_coverage));
    EXPECT_FALSE(host_sbs_adaptive_motion_damage_candidate(
      ddup_damage_coverage_t {1u, 10000u, false}
    ));
    EXPECT_FALSE(host_sbs_adaptive_motion_damage_candidate(
      ddup_damage_coverage_t {0u, 0u, true}
    ));
  }

  TEST(WindowsDdupDamageTest, AdaptiveCandidateCoversBothTopHalfPhases) {
    auto history = std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();
    const auto baseline = history->commit({true, {}});
    const RECT top_half {0, 0, 770, 217};
    const auto first_phase = history->commit({true, {top_half}});
    const auto second_phase = history->commit({true, {top_half}});
    const RECT analysis_region {0, 0, 770, 434};

    const auto expect_broad = [&](const auto &from, const auto &to) {
      const auto coverage =
        platf::dxgi::detail::query_ddup_damage_coverage_between(
          from,
          to,
          analysis_region,
          true
        );
      ASSERT_TRUE(coverage.known);
      EXPECT_EQ(coverage.region_area, 770u * 434u);
      EXPECT_EQ(coverage.max_single_intersection_area, 770u * 217u);
      EXPECT_TRUE(
        platf::dxgi::detail::host_sbs_adaptive_motion_damage_candidate(
          coverage
        )
      );
    };

    expect_broad(baseline, first_phase);
    expect_broad(first_phase, second_phase);
  }

  TEST(WindowsHostSbsSameFramePollTest, CadenceSlackCapsAndFailClosesWaitBudget) {
    using namespace std::chrono_literals;
    using platf::dxgi::detail::host_sbs_same_frame_poll_plan;

    const auto now = std::chrono::steady_clock::time_point {1s};
    const auto ample = host_sbs_same_frame_poll_plan(
      true,
      false,
      now + 20ms,
      now
    );
    ASSERT_TRUE(ample.eligible);
    EXPECT_EQ(ample.budget, 8ms);
    EXPECT_EQ(ample.deadline, now + 8ms);
    EXPECT_TRUE(ample.limited_by_hard_cap);

    const auto cadence_limited_large = host_sbs_same_frame_poll_plan(
      true,
      false,
      now + 10ms,
      now
    );
    ASSERT_TRUE(cadence_limited_large.eligible);
    EXPECT_EQ(cadence_limited_large.budget, 7ms);
    EXPECT_EQ(cadence_limited_large.deadline, now + 7ms);
    EXPECT_FALSE(cadence_limited_large.limited_by_hard_cap);

    const auto cadence_limited = host_sbs_same_frame_poll_plan(
      true,
      false,
      now + 5500us,
      now
    );
    ASSERT_TRUE(cadence_limited.eligible);
    EXPECT_EQ(cadence_limited.budget, 2500us);
    EXPECT_EQ(cadence_limited.deadline, now + 2500us);
    EXPECT_FALSE(cadence_limited.limited_by_hard_cap);

    const auto exact_minimum = host_sbs_same_frame_poll_plan(
      true,
      false,
      now + 3250us,
      now
    );
    ASSERT_TRUE(exact_minimum.eligible);
    EXPECT_EQ(exact_minimum.budget, 250us);
    EXPECT_FALSE(exact_minimum.limited_by_hard_cap);

    const auto exact_hard_cap_boundary = host_sbs_same_frame_poll_plan(
      true,
      false,
      now + 11ms,
      now
    );
    ASSERT_TRUE(exact_hard_cap_boundary.eligible);
    EXPECT_EQ(exact_hard_cap_boundary.budget, 8ms);
    EXPECT_TRUE(exact_hard_cap_boundary.limited_by_hard_cap);

    EXPECT_FALSE(host_sbs_same_frame_poll_plan(
                   true, false, now + 3249us, now
                 )
                   .eligible);
    EXPECT_FALSE(host_sbs_same_frame_poll_plan(
                   true, false, now + 2ms, now
                 )
                   .eligible);
    EXPECT_FALSE(host_sbs_same_frame_poll_plan(
                   false, false, now + 10ms, now
                 )
                   .eligible);
    EXPECT_FALSE(host_sbs_same_frame_poll_plan(
                   true, true, now + 10ms, now
                 )
                   .eligible);
    EXPECT_FALSE(host_sbs_same_frame_poll_plan(
                   true, false, std::nullopt, now
                 )
                   .eligible);
  }

  TEST(WindowsHostSbsSameFramePollTest, HitWaitBucketsKeepExactIntervalBoundaries) {
    using namespace std::chrono_literals;
    using bucket_e =
      platf::dxgi::detail::host_sbs_same_frame_poll_hit_bucket_e;
    using platf::dxgi::detail::host_sbs_same_frame_poll_hit_bucket;

    EXPECT_EQ(host_sbs_same_frame_poll_hit_bucket(0us), bucket_e::within_2_ms);
    EXPECT_EQ(host_sbs_same_frame_poll_hit_bucket(2ms), bucket_e::within_2_ms);
    EXPECT_EQ(
      host_sbs_same_frame_poll_hit_bucket(2001us),
      bucket_e::between_2_and_2_5_ms
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_hit_bucket(2500us),
      bucket_e::between_2_and_2_5_ms
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_hit_bucket(2501us),
      bucket_e::between_2_5_and_3_ms
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_hit_bucket(3ms),
      bucket_e::between_2_5_and_3_ms
    );
    EXPECT_EQ(host_sbs_same_frame_poll_hit_bucket(3001us), bucket_e::over_3_ms);
  }


  TEST(WindowsHostSbsSameFramePollTest, ExactOwnerIsAdoptedOnlyAfterReadyMatch) {
    using decision_e =
      platf::dxgi::detail::host_sbs_same_frame_completion_e;
    using outcome_e =
      platf::dxgi::detail::host_sbs_same_frame_poll_outcome_e;
    using platf::dxgi::detail::host_sbs_same_frame_completion;
    using platf::dxgi::detail::host_sbs_same_frame_poll_outcome;

    EXPECT_EQ(
      host_sbs_same_frame_completion(false, false, 0u, 42u),
      decision_e::keep_pending
    );
    EXPECT_EQ(
      host_sbs_same_frame_completion(false, true, 42u, 42u),
      decision_e::keep_pending
    );
    EXPECT_EQ(
      host_sbs_same_frame_completion(true, true, 42u, 42u),
      decision_e::adopt_exact
    );
    EXPECT_EQ(
      host_sbs_same_frame_completion(true, true, 41u, 42u),
      decision_e::discard_ready
    );
    EXPECT_EQ(
      host_sbs_same_frame_completion(true, false, 42u, 42u),
      decision_e::discard_ready
    );

    EXPECT_EQ(
      host_sbs_same_frame_poll_outcome(true, false, false, decision_e::adopt_exact),
      outcome_e::immediate_hit
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_outcome(true, true, false, decision_e::adopt_exact),
      outcome_e::wait_hit
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_outcome(false, false, false, decision_e::keep_pending),
      outcome_e::cadence_ineligible_busy
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_outcome(true, true, true, decision_e::keep_pending),
      outcome_e::eligible_timeout
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_outcome(true, true, false, decision_e::discard_ready),
      outcome_e::ready_failure
    );
    EXPECT_EQ(
      host_sbs_same_frame_poll_outcome(true, false, false, decision_e::keep_pending),
      outcome_e::wait_unavailable_busy
    );
  }

  TEST(WindowsHostSbsAuthorityTest, SamplingRequiresAnEstimatorConsumer) {
    EXPECT_FALSE(
      platf::dxgi::detail::host_sbs_window_authority_observation_needed(
        false,
        true
      )
    );
    EXPECT_FALSE(
      platf::dxgi::detail::host_sbs_window_authority_observation_needed(
        true,
        false
      )
    );
    EXPECT_TRUE(
      platf::dxgi::detail::host_sbs_window_authority_observation_needed(
        true,
        true
      )
    );
  }

  TEST(WindowsHostSbsContentReuseTest, RenderRequiresEveryFailClosedGate) {
    using platf::dxgi::detail::host_sbs_cached_geometry_render_allowed;
    const auto authorized =
      platf::dxgi::detail::make_host_sbs_depth_reuse_authorization(
        platf::dxgi::detail::host_sbs_depth_reuse_kind_e::exact_content,
        10u,
        11u,
        true
      );
    const platf::dxgi::detail::host_sbs_depth_reuse_authorization_t denied;

    EXPECT_TRUE(host_sbs_cached_geometry_render_allowed(
      true, true, authorized, 10u, 11u, true
    ));
    EXPECT_FALSE(host_sbs_cached_geometry_render_allowed(
      false, true, authorized, 10u, 11u, true
    ));
    EXPECT_FALSE(host_sbs_cached_geometry_render_allowed(
      true, false, authorized, 10u, 11u, true
    ));
    EXPECT_FALSE(host_sbs_cached_geometry_render_allowed(
      true, true, denied, 10u, 11u, true
    ));
    EXPECT_FALSE(host_sbs_cached_geometry_render_allowed(
      true, true, authorized, 9u, 11u, true
    ));
    EXPECT_FALSE(host_sbs_cached_geometry_render_allowed(
      true, true, authorized, 10u, 12u, true
    ));
    EXPECT_FALSE(host_sbs_cached_geometry_render_allowed(
      true, true, authorized, 10u, 11u, false
    ));
  }

  TEST(WindowsHostSbsContentReuseTest, LatestLineageRetentionDoesNotAuthorizeRender) {
    using platf::dxgi::detail::host_sbs_cached_geometry_render_allowed;
    using platf::dxgi::detail::host_sbs_latest_v2_completion_retention_allowed;

    EXPECT_TRUE(host_sbs_latest_v2_completion_retention_allowed(true, true, true, true));
    EXPECT_FALSE(host_sbs_latest_v2_completion_retention_allowed(false, true, true, true));
    EXPECT_FALSE(host_sbs_latest_v2_completion_retention_allowed(true, false, true, true));
    EXPECT_FALSE(host_sbs_latest_v2_completion_retention_allowed(true, true, false, true));
    EXPECT_FALSE(host_sbs_latest_v2_completion_retention_allowed(
      true, true, true, false
    ));

    // Retained authenticated resources still fail closed without a current exact geometry match.
    EXPECT_FALSE(host_sbs_cached_geometry_render_allowed(
      true,
      true,
      {},
      10u,
      11u,
      true
    ));
  }

  TEST(WindowsHostSbsGpuAdmissionTest, BarrierBlocksCacheButAdmitsExplicitOpaqueFollowup) {
    using admission_e = platf::dxgi::detail::host_sbs_depth_admission_e;
    using platf::dxgi::detail::host_sbs_depth_admission;

    EXPECT_EQ(host_sbs_depth_admission(true, true, false), admission_e::reuse_cached);
    EXPECT_EQ(host_sbs_depth_admission(false, true, false), admission_e::gpu_undecided);
    EXPECT_EQ(host_sbs_depth_admission(false, false, false), admission_e::force_infer);
    EXPECT_EQ(host_sbs_depth_admission(true, true, true), admission_e::force_infer);
    EXPECT_EQ(
      host_sbs_depth_admission(true, true, true, true),
      admission_e::gpu_undecided
    );
    EXPECT_EQ(
      host_sbs_depth_admission(true, false, true, true),
      admission_e::force_infer
    );
  }

  TEST(WindowsHostSbsGpuAdmissionTest, OpaqueFollowupAuthorityUsesSharedStrictHundredMsBound) {
    using namespace std::chrono_literals;
    using platf::dxgi::detail::host_sbs_gpu_followup_fresh;

    const auto enqueued_at = std::chrono::steady_clock::time_point {1s};
    EXPECT_TRUE(host_sbs_gpu_followup_fresh(40u, 40u, enqueued_at, enqueued_at + 99ms));
    EXPECT_FALSE(host_sbs_gpu_followup_fresh(40u, 40u, enqueued_at, enqueued_at + 100ms));
    EXPECT_FALSE(host_sbs_gpu_followup_fresh(40u, 41u, enqueued_at, enqueued_at + 1ms));
    EXPECT_FALSE(host_sbs_gpu_followup_fresh(0u, 0u, enqueued_at, enqueued_at + 1ms));
    EXPECT_FALSE(host_sbs_gpu_followup_fresh(40u, 40u, {}, enqueued_at + 1ms));
    EXPECT_FALSE(host_sbs_gpu_followup_fresh(40u, 40u, enqueued_at, enqueued_at - 1ms));
  }

  TEST(WindowsHostSbsGpuAdmissionTest, BarrierClearsOnlyOnNewerAcceptedForceInferCompletion) {
    platf::dxgi::detail::host_sbs_gpu_observation_barrier_t barrier;

    EXPECT_FALSE(barrier.active());
    barrier.record_gpu_undecided_enqueue(40u);
    EXPECT_TRUE(barrier.active());
    EXPECT_EQ(barrier.conditional_frame_id(), 40u);
    EXPECT_FALSE(barrier.record_known_force_infer_completion(41u, false));
    EXPECT_FALSE(barrier.record_known_force_infer_completion(40u, true));
    EXPECT_FALSE(barrier.record_known_force_infer_completion(39u, true));
    EXPECT_TRUE(barrier.active());

    // A later conditional transaction advances the watermark rather than weakening the barrier.
    barrier.record_gpu_undecided_enqueue(45u);
    EXPECT_EQ(barrier.conditional_frame_id(), 45u);
    EXPECT_FALSE(barrier.record_known_force_infer_completion(44u, true));
    EXPECT_TRUE(barrier.record_known_force_infer_completion(46u, true));
    EXPECT_FALSE(barrier.active());

    barrier.record_gpu_undecided_enqueue(50u);
    barrier.reset();
    EXPECT_FALSE(barrier.active());
  }

  TEST(WindowsHostSbsGpuAdmissionTest, SharedPolicyOwnsInitialOpaqueAndForceCompletionChaining) {
    using models::gpu_adaptive_submission_class_e;
    platf::dxgi::detail::host_sbs_gpu_observation_barrier_t policy;

    const auto forced = policy.make_request(10u, false, false, 0u, 1000u);
    EXPECT_FALSE(forced.authorize_gpu_undecided_reuse);
    EXPECT_EQ(forced.observation_timestamp_us, 1000u);
    const auto initial = policy.make_request(11u, true, false, 10u, 2000u);
    EXPECT_TRUE(initial.authorize_gpu_undecided_reuse);
    EXPECT_FALSE(initial.opaque_followup);
    EXPECT_EQ(initial.baseline_frame_id, 10u);
    EXPECT_EQ(initial.gpu_reuse_decision_token, 11u);
    EXPECT_EQ(
      policy.record_submission(11u, initial, false, true),
      gpu_adaptive_submission_class_e::gpu_undecided
    );
    ASSERT_TRUE(policy.active());
    EXPECT_EQ(policy.conditional_frame_id(), 11u);

    // An active watermark can name only the immediately preceding opaque transaction.
    EXPECT_FALSE(policy.make_request(12u, true, false, 10u, 3000u)
                   .authorize_gpu_undecided_reuse);
    EXPECT_FALSE(policy.make_request(12u, true, true, 10u, 3000u)
                   .authorize_gpu_undecided_reuse);
    const auto followup = policy.make_request(12u, true, true, 11u, 3000u);
    EXPECT_TRUE(followup.authorize_gpu_undecided_reuse);
    EXPECT_TRUE(followup.opaque_followup);
    EXPECT_EQ(
      policy.record_submission(12u, followup, false, true),
      gpu_adaptive_submission_class_e::gpu_undecided
    );
    EXPECT_EQ(policy.conditional_frame_id(), 12u);

    // A runtime fallback is CPU-known force but cannot release the watermark until its exact
    // accepted completion arrives.
    const auto fallback = policy.make_request(13u, true, true, 12u, 4000u);
    EXPECT_EQ(
      policy.record_submission(13u, fallback, true, false),
      gpu_adaptive_submission_class_e::force_infer
    );
    EXPECT_TRUE(policy.active());
    EXPECT_FALSE(policy.record_known_force_infer_completion(13u, false));
    EXPECT_TRUE(policy.record_known_force_infer_completion(13u, true));
    EXPECT_FALSE(policy.active());

    EXPECT_EQ(
      policy.record_submission(14u, {}, true, true),
      gpu_adaptive_submission_class_e::invalid
    );
  }

  TEST(WindowsHostSbsContentReuseTest, LatestLineageResetCoversAuthorityAndAliasRevocation) {
    using platf::dxgi::detail::host_sbs_latest_v2_lineage_reset_required;

    EXPECT_FALSE(host_sbs_latest_v2_lineage_reset_required(
      false, false, true, true, true
    ));
    EXPECT_FALSE(host_sbs_latest_v2_lineage_reset_required(
      true, true, false, false, false
    ));
    EXPECT_TRUE(host_sbs_latest_v2_lineage_reset_required(
      true, false, false, false, false
    ));
    EXPECT_TRUE(host_sbs_latest_v2_lineage_reset_required(
      true, true, true, false, false
    ));
    EXPECT_TRUE(host_sbs_latest_v2_lineage_reset_required(
      true, true, false, true, false
    ));
    EXPECT_TRUE(host_sbs_latest_v2_lineage_reset_required(
      true, true, false, false, true
    ));
  }

  TEST(WindowsHostSbsContentReuseTest, SuccessfulEnqueueBoundsSkipsAndAge) {
    using namespace std::chrono_literals;
    platf::dxgi::detail::host_sbs_content_refresh_state_t state;
    const auto start = std::chrono::steady_clock::time_point {1s};

    EXPECT_TRUE(state.refresh_due(start));
    state.record_successful_enqueue(start);
    EXPECT_FALSE(state.refresh_due(start));
    for (unsigned i = 0;
         i + 1u < platf::dxgi::detail::host_sbs_full_source_reuse_max_skips;
         ++i) {
      state.record_reuse();
    }
    EXPECT_FALSE(state.refresh_due(start + 249ms));
    state.record_reuse();
    EXPECT_TRUE(state.refresh_due(start + 249ms));

    state.record_successful_enqueue(start);
    EXPECT_FALSE(state.refresh_due(start + 249ms));
    EXPECT_TRUE(state.refresh_due(start + 250ms));
  }

  TEST(WindowsHostSbsContentReuseTest, TypedAuthorizationUnifiesProofAndRefreshSemantics) {
    using kind_e = platf::dxgi::detail::host_sbs_depth_reuse_kind_e;
    using refresh_e = platf::dxgi::detail::host_sbs_depth_reuse_refresh_e;
    using proof_e = platf::dxgi::detail::host_sbs_ddup_reuse_proof_e;
    using platf::dxgi::detail::select_host_sbs_cached_depth_reuse;

    const auto exact = select_host_sbs_cached_depth_reuse(
      proof_e::content_clock, 40u, 42u
    );
    EXPECT_TRUE(exact.valid());
    EXPECT_EQ(exact.kind, kind_e::exact_content);
    EXPECT_EQ(exact.refresh, refresh_e::bounded_content);

    const auto roi = select_host_sbs_cached_depth_reuse(
      proof_e::roi_damage, 40u, 42u
    );
    EXPECT_TRUE(roi.valid());
    EXPECT_EQ(roi.kind, kind_e::exact_roi_damage);

    const auto denied = select_host_sbs_cached_depth_reuse(
      proof_e::none, 40u, 42u
    );
    EXPECT_FALSE(denied.valid());

    auto malformed = exact;
    malformed.ocr_safe = false;
    EXPECT_FALSE(malformed.valid());
    malformed = exact;
    malformed.current_frame_id = malformed.baseline_frame_id;
    EXPECT_FALSE(malformed.valid());
  }

  TEST(WindowsHostSbsContentReuseTest, HostOwnedApproximateProvidersCannotChain) {
    using provider_e = platf::dxgi::detail::host_sbs_approximate_reuse_provider_e;
    using platf::dxgi::detail::host_sbs_approximate_reuse_provider_allowed;

    EXPECT_TRUE(host_sbs_approximate_reuse_provider_allowed(
      provider_e::none, provider_e::gpu_undecided, false
    ));
    EXPECT_FALSE(host_sbs_approximate_reuse_provider_allowed(
      provider_e::gpu_undecided, provider_e::gpu_undecided, false
    ));
    EXPECT_FALSE(host_sbs_approximate_reuse_provider_allowed(
      provider_e::none, provider_e::gpu_undecided, true
    ));
    EXPECT_FALSE(host_sbs_approximate_reuse_provider_allowed(
      provider_e::none, provider_e::none, false
    ));
  }

  TEST(WindowsHostSbsContentReuseTest, GpuUndecidedCandidateRequiresFullOwnedTuple) {
    using candidate_t =
      platf::dxgi::detail::host_sbs_gpu_undecided_candidate_t;
    auto damage_history =
      std::make_shared<platf::dxgi::detail::ddup_damage_history_t>();

    const candidate_t valid {
      .baseline_frame_id = 40u,
      .current_frame_id = 42u,
      .damage_history = damage_history.get(),
      .baseline_damage_token = 10u,
      .current_damage_token = 11u,
      .damage_history_complete = true,
    };
    EXPECT_TRUE(valid.valid());
    auto opaque_followup = valid;
    opaque_followup.opaque_followup = true;
    EXPECT_TRUE(opaque_followup.valid());
    auto invalid = valid;
    invalid.baseline_frame_id = 0u;
    EXPECT_FALSE(invalid.valid());
    invalid = valid;
    invalid.current_frame_id = invalid.baseline_frame_id;
    EXPECT_FALSE(invalid.valid());
    invalid = valid;
    invalid.damage_history = nullptr;
    EXPECT_FALSE(invalid.valid());
    invalid = valid;
    invalid.current_damage_token = invalid.baseline_damage_token;
    EXPECT_FALSE(invalid.valid());
    invalid = valid;
    invalid.damage_history_complete = false;
    EXPECT_FALSE(invalid.valid());
  }

  TEST(WindowsHostSbsAdaptiveMotionTest, LiveRouteEpochChangesWithoutAnAuthenticatedCache) {
    using epoch_t = platf::dxgi::detail::host_sbs_adaptive_motion_route_epoch_t;
    platf::dxgi::detail::host_sbs_adaptive_motion_route_state_t state;
    const epoch_t base {
      .source_width = 1920u,
      .source_height = 1080u,
      .mip_levels = 1u,
      .array_size = 1u,
      .source_format = 87u,
      .sample_count = 1u,
      .sample_quality = 0u,
      .input_color_space = 0u,
      .root_authority_generation = 10u,
      .region_authority_generation = 20u,
      .browser_authority_epoch = 30u,
      .interactive_move_size = false,
    };

    EXPECT_FALSE(state.observe(base));
    EXPECT_FALSE(state.observe(base));
    // Resetting reuse cadence must not erase this independent live-route baseline.
    platf::dxgi::detail::host_sbs_adaptive_hold_cadence_t cadence;
    cadence.reset();
    auto changed = base;
    ++changed.root_authority_generation;
    EXPECT_TRUE(state.observe(changed));
    ++changed.region_authority_generation;
    EXPECT_TRUE(state.observe(changed));
    ++changed.browser_authority_epoch;
    EXPECT_TRUE(state.observe(changed));
    ++changed.input_color_space;
    EXPECT_TRUE(state.observe(changed));
    ++changed.source_width;
    EXPECT_TRUE(state.observe(changed));
    ++changed.source_format;
    EXPECT_TRUE(state.observe(changed));
    changed.interactive_move_size = true;
    EXPECT_TRUE(state.observe(changed));
    state.reset();
    EXPECT_FALSE(state.observe(changed));
  }

  TEST(WindowsHostSbsAdaptiveMotionTest, SimulatedHoldForcesOneRefreshBeforeRearming) {
    using namespace std::chrono_literals;
    using decision_e = platf::dxgi::detail::host_sbs_adaptive_hold_decision_e;
    platf::dxgi::detail::host_sbs_adaptive_hold_cadence_t cadence;
    const auto start = std::chrono::steady_clock::time_point {1s};
    const auto a = start;
    const auto b = start + 1ms;
    const auto c = start + 2ms;
    const auto d = start + 3ms;

    cadence.record_successful_enqueue(a, start);
    EXPECT_EQ(cadence.observe_changed(b, true, start + 99ms), decision_e::hold_candidate);
    EXPECT_TRUE(cadence.refresh_required());
    EXPECT_TRUE(cadence.hold_candidate_still_fresh(b, start + 99ms));
    EXPECT_FALSE(cadence.hold_candidate_still_fresh(c, start + 99ms));
    EXPECT_FALSE(cadence.hold_candidate_still_fresh(b, start + 100ms));
    // The cadence identifies the repeated held identity. GPU admission treats this decision as
    // mandatory inference rather than granting a second approximate hold.
    EXPECT_EQ(
      cadence.observe_changed(b, true, start + 99ms),
      decision_e::hold_same_identity
    );
    // A distinct successor forces the simulated real inference before another hold can arm.
    EXPECT_EQ(cadence.observe_changed(c, true, start + 99ms), decision_e::infer);
    cadence.record_successful_enqueue(c, start + 99ms);
    EXPECT_EQ(cadence.observe_changed(d, true, start + 198ms), decision_e::hold_candidate);

    cadence.record_successful_enqueue(a, start);
    EXPECT_EQ(cadence.observe_changed(b, true, start + 100ms), decision_e::infer);

    cadence.record_successful_enqueue(a, start);
    ASSERT_EQ(cadence.observe_changed(b, true, start + 99ms), decision_e::hold_candidate);
    EXPECT_EQ(cadence.observe_changed(b, true, start + 100ms), decision_e::infer);
  }

  TEST(WindowsUploadedValueStateTest, CommitsOnlyExplicitlyAcceptedValue) {
    platf::dxgi::detail::uploaded_value_state_t<std::array<int, 3>> state;
    constexpr std::array first {1, 2, 3};
    constexpr std::array second {1, 2, 4};

    EXPECT_FALSE(state.is_current(first));
    state.commit(first);
    EXPECT_TRUE(state.is_current(first));
    EXPECT_FALSE(state.is_current(second));
    state.reset();
    EXPECT_FALSE(state.is_current(first));
  }

  struct fake_desktop_retry_operations_t {
    std::vector<bool> attempt_results;
    bool synchronization_result = true;
    std::size_t attempt_count = 0;
    std::size_t synchronization_count = 0;

    bool attempt() {
      const auto result = attempt_results.at(attempt_count);
      ++attempt_count;
      return result;
    }

    bool synchronize() {
      ++synchronization_count;
      return synchronization_result;
    }
  };

  struct fake_desktop_handle_operations_t {
    std::uintptr_t candidate = 2;
    bool set_succeeds = true;
    std::vector<std::string> calls;

    std::uintptr_t open_input_desktop() {
      calls.emplace_back("open:" + std::to_string(candidate));
      return candidate;
    }

    bool set_thread_desktop(std::uintptr_t desktop) {
      calls.emplace_back("set:" + std::to_string(desktop));
      return set_succeeds;
    }

    void close_desktop(std::uintptr_t desktop) {
      calls.emplace_back("close:" + std::to_string(desktop));
    }
  };

  TEST(WindowsQpcTimeTest, ConvertsPerformanceCounterTicksToNanoseconds) {
    LARGE_INTEGER frequency {};
    ASSERT_TRUE(QueryPerformanceFrequency(&frequency));
    ASSERT_GT(frequency.QuadPart, 0);

    EXPECT_NEAR(
      platf::qpc_time_difference(frequency.QuadPart, 0).count(),
      1'000'000'000LL,
      1LL
    );
    EXPECT_NEAR(
      platf::qpc_time_difference(0, frequency.QuadPart).count(),
      -1'000'000'000LL,
      1LL
    );
  }

  struct fake_token_job_launch_operations_t {
    bool assignment_succeeds = true;
    bool resume_succeeds = true;
    DWORD failure = ERROR_ACCESS_DENIED;
    std::vector<std::string> calls;

    bool assign_to_job() {
      calls.emplace_back("assign");
      return assignment_succeeds;
    }

    bool resume_initial_thread() {
      calls.emplace_back("resume");
      return resume_succeeds;
    }

    DWORD last_error() {
      calls.emplace_back("last-error");
      return failure;
    }

    void terminate_process(const DWORD exit_code) {
      calls.emplace_back("terminate:" + std::to_string(exit_code));
    }

    void close_initial_thread() {
      calls.emplace_back("close-thread");
    }

    void close_process() {
      calls.emplace_back("close-process");
    }
  };

  struct fake_explorer_restart_operations_t {
    std::optional<platf::explorer_restart_result_e> preflight_failure;
    bool terminate_succeeds = true;
    bool exit_succeeds = true;
    bool automatic_replacement = false;
    bool replacement_before_launch = false;
    bool launch_succeeds = true;
    bool fallback_replacement = true;
    std::vector<std::string> calls;

    std::optional<platf::explorer_restart_result_e> preflight() {
      calls.emplace_back("preflight");
      return preflight_failure;
    }

    bool terminate_exact_shell() {
      calls.emplace_back("terminate");
      return terminate_succeeds;
    }

    bool wait_for_old_shell_exit() {
      calls.emplace_back("wait-exit");
      return exit_succeeds;
    }

    bool wait_for_replacement(bool after_fallback_launch) {
      calls.emplace_back(after_fallback_launch ? "wait-fallback" : "wait-auto");
      return after_fallback_launch ? fallback_replacement : automatic_replacement;
    }

    bool replacement_present_now() {
      calls.emplace_back("check-before-launch");
      return replacement_before_launch;
    }

    bool launch_captured_shell() {
      calls.emplace_back("launch");
      return launch_succeeds;
    }
  };

  MOUSEKEYS sample_mouse_keys_state() {
    MOUSEKEYS state {};
    state.cbSize = sizeof(state);
    state.dwFlags = MKF_MODIFIERS;
    state.iMaxSpeed = 37;
    state.iTimeToMaxSpeed = 1400;
    state.iCtrlSpeed = 3;
    return state;
  }

  void expect_same_mouse_keys_state(const MOUSEKEYS &actual, const MOUSEKEYS &expected) {
    EXPECT_EQ(actual.cbSize, expected.cbSize);
    EXPECT_EQ(actual.dwFlags, expected.dwFlags);
    EXPECT_EQ(actual.iMaxSpeed, expected.iMaxSpeed);
    EXPECT_EQ(actual.iTimeToMaxSpeed, expected.iTimeToMaxSpeed);
    EXPECT_EQ(actual.iCtrlSpeed, expected.iCtrlSpeed);
    EXPECT_EQ(actual.dwReserved1, expected.dwReserved1);
    EXPECT_EQ(actual.dwReserved2, expected.dwReserved2);
  }

  std::optional<std::string> token_sid(HANDLE token) {
    DWORD bytes = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &bytes);
    if (bytes == 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      return std::nullopt;
    }
    std::vector<std::byte> buffer(bytes);
    if (!GetTokenInformation(
          token,
          TokenUser,
          buffer.data(),
          static_cast<DWORD>(buffer.size()),
          &bytes
        )) {
      return std::nullopt;
    }
    const auto token_user = reinterpret_cast<const TOKEN_USER *>(buffer.data());
    LPWSTR sid_text = nullptr;
    if (!ConvertSidToStringSidW(token_user->User.Sid, &sid_text) || !sid_text) {
      return std::nullopt;
    }
    auto sid_text_guard = util::fail_guard([&]() {
      LocalFree(sid_text);
    });
    return platf::to_utf8(sid_text);
  }
}  // namespace

TEST(WindowsInputDesktopRetryTest, DoesNotSynchronizeAfterImmediateSuccess) {
  fake_desktop_retry_operations_t operations {{true}};

  EXPECT_TRUE(platf::detail::run_with_desktop_retry([&]() {
    return operations.attempt();
  },
                                                    [&]() {
                                                      return operations.synchronize();
                                                    }));
  EXPECT_EQ(operations.attempt_count, 1u);
  EXPECT_EQ(operations.synchronization_count, 0u);
}

TEST(WindowsInputDesktopRetryTest, RetriesExactlyOnceAfterSuccessfulSynchronization) {
  fake_desktop_retry_operations_t operations {{false, false}};

  EXPECT_FALSE(platf::detail::run_with_desktop_retry([&]() {
    return operations.attempt();
  },
                                                     [&]() {
                                                       return operations.synchronize();
                                                     }));
  EXPECT_EQ(operations.attempt_count, 2u);
  EXPECT_EQ(operations.synchronization_count, 1u);
}

TEST(WindowsInputDesktopRetryTest, DoesNotRetryWhenSynchronizationFails) {
  fake_desktop_retry_operations_t operations {{false}};
  operations.synchronization_result = false;

  EXPECT_FALSE(platf::detail::run_with_desktop_retry([&]() {
    return operations.attempt();
  },
                                                     [&]() {
                                                       return operations.synchronize();
                                                     }));
  EXPECT_EQ(operations.attempt_count, 1u);
  EXPECT_EQ(operations.synchronization_count, 1u);
}

TEST(WindowsInputDesktopRetryTest, RetainsSuccessfulDesktopAndClosesSupersededHandle) {
  std::uintptr_t active = 1;
  fake_desktop_handle_operations_t operations;

  EXPECT_TRUE(platf::detail::replace_thread_desktop_handle(active, operations));
  EXPECT_EQ(active, 2u);
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {"open:2", "set:2", "close:1"})
  );
}

TEST(WindowsInputDesktopRetryTest, ClosesFailedCandidateAndPreservesActiveHandle) {
  std::uintptr_t active = 1;
  fake_desktop_handle_operations_t operations;
  operations.set_succeeds = false;

  EXPECT_FALSE(platf::detail::replace_thread_desktop_handle(active, operations));
  EXPECT_EQ(active, 1u);
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {"open:2", "set:2", "close:2"})
  );
}

TEST(ExplorerRestartControllerTest, AcceptsWindowsAutomaticRestartWithoutLaunchingAgain) {
  fake_explorer_restart_operations_t operations;
  operations.automatic_replacement = true;

  EXPECT_EQ(
    platf::detail::run_explorer_restart(operations),
    platf::explorer_restart_result_e::restarted_automatically
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "preflight",
      "terminate",
      "wait-exit",
      "wait-auto",
    })
  );
}

TEST(ExplorerRestartControllerTest, LaunchesExactlyOneFallbackAfterAutoRestartTimeout) {
  fake_explorer_restart_operations_t operations;

  EXPECT_EQ(
    platf::detail::run_explorer_restart(operations),
    platf::explorer_restart_result_e::relaunched
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "preflight",
      "terminate",
      "wait-exit",
      "wait-auto",
      "check-before-launch",
      "launch",
      "wait-fallback",
    })
  );
  EXPECT_EQ(
    std::ranges::count(operations.calls, "launch"),
    1
  );
}

TEST(ExplorerRestartControllerTest, CoalescesALateAutomaticRestartBeforeFallbackLaunch) {
  fake_explorer_restart_operations_t operations;
  operations.replacement_before_launch = true;

  EXPECT_EQ(
    platf::detail::run_explorer_restart(operations),
    platf::explorer_restart_result_e::restarted_automatically
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "preflight",
      "terminate",
      "wait-exit",
      "wait-auto",
      "check-before-launch",
      "wait-fallback",
    })
  );
  EXPECT_EQ(std::ranges::count(operations.calls, "launch"), 0);
}

TEST(ExplorerRestartControllerTest, NeverLaunchesWhenPreflightKillOrExitFails) {
  {
    fake_explorer_restart_operations_t operations;
    operations.preflight_failure =
      platf::explorer_restart_result_e::skipped_identity;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::skipped_identity
    );
    EXPECT_EQ(operations.calls, (std::vector<std::string> {"preflight"}));
  }
  {
    fake_explorer_restart_operations_t operations;
    operations.terminate_succeeds = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::terminate_failed
    );
    EXPECT_EQ(
      operations.calls,
      (std::vector<std::string> {"preflight", "terminate"})
    );
  }
  {
    fake_explorer_restart_operations_t operations;
    operations.exit_succeeds = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::exit_timeout
    );
    EXPECT_EQ(
      operations.calls,
      (std::vector<std::string> {"preflight", "terminate", "wait-exit"})
    );
  }
}

TEST(ExplorerRestartControllerTest, ReportsFallbackLaunchAndReplacementFailures) {
  {
    fake_explorer_restart_operations_t operations;
    operations.launch_succeeds = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::relaunch_failed
    );
    EXPECT_EQ(operations.calls.back(), "launch");
  }
  {
    fake_explorer_restart_operations_t operations;
    operations.fallback_replacement = false;
    EXPECT_EQ(
      platf::detail::run_explorer_restart(operations),
      platf::explorer_restart_result_e::replacement_timeout
    );
    EXPECT_EQ(operations.calls.back(), "wait-fallback");
    EXPECT_EQ(std::ranges::count(operations.calls, "launch"), 1);
  }
}

TEST(MouseKeysControllerTest, EnablesOnceAndRestoresTheExactPreviousState) {
  platf::detail::mouse_keys_controller_t controller;
  const auto original = sample_mouse_keys_state();
  MOUSEKEYS applied {};

  EXPECT_TRUE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      state = original;
      return true;
    },
    [&](MOUSEKEYS &state) {
      applied = state;
      return true;
    }
  ));
  EXPECT_TRUE(controller.enabled_by_host());
  EXPECT_EQ(applied.dwFlags & (MKF_MOUSEKEYSON | MKF_AVAILABLE), MKF_MOUSEKEYSON | MKF_AVAILABLE);
  EXPECT_EQ(applied.dwFlags & MKF_MODIFIERS, MKF_MODIFIERS);
  EXPECT_EQ(applied.iMaxSpeed, original.iMaxSpeed);
  EXPECT_EQ(applied.iTimeToMaxSpeed, original.iTimeToMaxSpeed);
  EXPECT_EQ(applied.iCtrlSpeed, original.iCtrlSpeed);

  EXPECT_FALSE(controller.refresh(
    false,
    [](MOUSEKEYS &) {
      ADD_FAILURE() << "The saved state must not be queried again";
      return false;
    },
    [](MOUSEKEYS &) {
      ADD_FAILURE() << "Mouse Keys must not be enabled twice";
      return false;
    }
  ));

  EXPECT_FALSE(controller.restore([](MOUSEKEYS &) {
    return false;
  }));
  EXPECT_TRUE(controller.enabled_by_host());

  MOUSEKEYS restored {};
  EXPECT_TRUE(controller.restore([&](MOUSEKEYS &state) {
    restored = state;
    return true;
  }));
  EXPECT_FALSE(controller.enabled_by_host());
  expect_same_mouse_keys_state(restored, original);
}

TEST(MouseKeysControllerTest, LeavesExistingOrUnavailableStateAlone) {
  platf::detail::mouse_keys_controller_t controller;
  int getter_calls = 0;
  int setter_calls = 0;

  EXPECT_FALSE(controller.refresh(
    true,
    [&](MOUSEKEYS &) {
      ++getter_calls;
      return true;
    },
    [&](MOUSEKEYS &) {
      ++setter_calls;
      return true;
    }
  ));
  EXPECT_EQ(getter_calls, 0);
  EXPECT_EQ(setter_calls, 0);

  EXPECT_FALSE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      ++getter_calls;
      state = sample_mouse_keys_state();
      state.dwFlags |= MKF_MOUSEKEYSON | MKF_AVAILABLE;
      return true;
    },
    [&](MOUSEKEYS &) {
      ++setter_calls;
      return true;
    }
  ));
  EXPECT_EQ(getter_calls, 1);
  EXPECT_EQ(setter_calls, 0);
  EXPECT_FALSE(controller.enabled_by_host());
}

TEST(MouseKeysControllerTest, RetriesAfterQueryOrEnableFailures) {
  platf::detail::mouse_keys_controller_t controller;
  const auto original = sample_mouse_keys_state();

  EXPECT_FALSE(controller.refresh(
    false,
    [](MOUSEKEYS &) {
      return false;
    },
    [](MOUSEKEYS &) {
      ADD_FAILURE() << "Setter must not run after a query failure";
      return false;
    }
  ));
  EXPECT_FALSE(controller.enabled_by_host());

  EXPECT_FALSE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      state = original;
      return true;
    },
    [](MOUSEKEYS &) {
      return false;
    }
  ));
  EXPECT_FALSE(controller.enabled_by_host());

  EXPECT_TRUE(controller.refresh(
    false,
    [&](MOUSEKEYS &state) {
      state = original;
      return true;
    },
    [](MOUSEKEYS &) {
      return true;
    }
  ));
  EXPECT_TRUE(controller.enabled_by_host());
}

TEST(UserLocalAppDataTest, MatchesTheCurrentOrLinkedStandardUserProfile) {
  DWORD process_session = 0;
  ASSERT_TRUE(ProcessIdToSessionId(GetCurrentProcessId(), &process_session));
  if (process_session != WTSGetActiveConsoleSessionId()) {
    GTEST_SKIP() << "This contract requires an interactive test process";
  }

  HANDLE process_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(
    GetCurrentProcess(),
    TOKEN_QUERY,
    &process_token
  ));
  auto process_token_guard = util::fail_guard([&]() {
    CloseHandle(process_token);
  });

  TOKEN_ELEVATION_TYPE elevation_type {};
  DWORD returned = 0;
  ASSERT_TRUE(GetTokenInformation(
    process_token,
    TokenElevationType,
    &elevation_type,
    sizeof(elevation_type),
    &returned
  ));
  TOKEN_ELEVATION elevation {};
  ASSERT_TRUE(GetTokenInformation(
    process_token,
    TokenElevation,
    &elevation,
    sizeof(elevation),
    &returned
  ));
  if (
    elevation_type == TokenElevationTypeDefault &&
    elevation.TokenIsElevated != 0
  ) {
    GTEST_SKIP()
      << "A UAC-disabled administrator intentionally has no standard token";
  }

  HANDLE profile_token = nullptr;
  if (elevation_type == TokenElevationTypeFull) {
    TOKEN_LINKED_TOKEN linked {};
    ASSERT_TRUE(GetTokenInformation(
      process_token,
      TokenLinkedToken,
      &linked,
      sizeof(linked),
      &returned
    ));
    profile_token = linked.LinkedToken;
  }
  auto profile_token_guard = util::fail_guard([&]() {
    if (profile_token) {
      CloseHandle(profile_token);
    }
  });

  PWSTR expected_value = nullptr;
  ASSERT_TRUE(SUCCEEDED(SHGetKnownFolderPath(
    FOLDERID_LocalAppData,
    KF_FLAG_DEFAULT,
    profile_token,
    &expected_value
  )));
  ASSERT_NE(expected_value, nullptr);
  auto expected_value_guard = util::fail_guard([&]() {
    CoTaskMemFree(expected_value);
  });

  const auto actual = platf::user_local_appdata();
  ASSERT_FALSE(actual.empty());
  EXPECT_EQ(
    CompareStringOrdinal(
      actual.c_str(),
      -1,
      expected_value,
      -1,
      TRUE
    ),
    CSTR_EQUAL
  );
}

TEST(RunAsActiveUserTest, CanInspectTheUserProfileWithUsableImpersonation) {
  DWORD process_session = 0;
  ASSERT_TRUE(ProcessIdToSessionId(GetCurrentProcessId(), &process_session));
  if (process_session != WTSGetActiveConsoleSessionId()) {
    GTEST_SKIP() << "This contract requires an interactive test process";
  }

  HANDLE process_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token));
  auto process_token_guard = util::fail_guard([&]() {
    CloseHandle(process_token);
  });
  TOKEN_ELEVATION process_elevation {};
  DWORD returned = 0;
  ASSERT_TRUE(GetTokenInformation(
    process_token,
    TokenElevation,
    &process_elevation,
    sizeof(process_elevation),
    &returned
  ));
  if (process_elevation.TokenIsElevated == 0) {
    GTEST_SKIP() << "The impersonation regression requires an elevated tray process";
  }

  const auto expected_user = platf::active_user_id();
  if (!expected_user) {
    GTEST_SKIP() << "Windows did not expose a validated active desktop user";
  }
  const auto profile = platf::user_local_appdata();
  ASSERT_FALSE(profile.empty());

  bool callback_called = false;
  const auto result = platf::run_as_active_user(
    [&]() {
      callback_called = true;

      HANDLE thread_token = nullptr;
      ASSERT_TRUE(OpenThreadToken(
        GetCurrentThread(),
        TOKEN_QUERY,
        TRUE,
        &thread_token
      ));
      auto thread_token_guard = util::fail_guard([&]() {
        CloseHandle(thread_token);
      });
      SECURITY_IMPERSONATION_LEVEL level = SecurityAnonymous;
      DWORD returned = 0;
      ASSERT_TRUE(GetTokenInformation(
        thread_token,
        TokenImpersonationLevel,
        &level,
        sizeof(level),
        &returned
      ));
      EXPECT_GE(level, SecurityImpersonation);

      TOKEN_ELEVATION elevation {};
      ASSERT_TRUE(GetTokenInformation(
        thread_token,
        TokenElevation,
        &elevation,
        sizeof(elevation),
        &returned
      ));
      EXPECT_EQ(elevation.TokenIsElevated, 0u);

      TOKEN_STATISTICS outer_statistics {};
      ASSERT_TRUE(GetTokenInformation(
        thread_token,
        TokenStatistics,
        &outer_statistics,
        sizeof(outer_statistics),
        &returned
      ));
      bool nested_callback_called = false;
      const auto nested_result = platf::run_as_active_user(
        [&]() {
          nested_callback_called = true;
        },
        expected_user
      );
      EXPECT_FALSE(nested_result);
      EXPECT_TRUE(nested_callback_called);

      HANDLE restored_thread_token = nullptr;
      ASSERT_TRUE(OpenThreadToken(
        GetCurrentThread(),
        TOKEN_QUERY,
        TRUE,
        &restored_thread_token
      ));
      auto restored_thread_token_guard = util::fail_guard([&]() {
        CloseHandle(restored_thread_token);
      });
      TOKEN_STATISTICS restored_statistics {};
      ASSERT_TRUE(GetTokenInformation(
        restored_thread_token,
        TokenStatistics,
        &restored_statistics,
        sizeof(restored_statistics),
        &returned
      ));
      EXPECT_EQ(restored_statistics.TokenId.HighPart, outer_statistics.TokenId.HighPart);
      EXPECT_EQ(restored_statistics.TokenId.LowPart, outer_statistics.TokenId.LowPart);

      HANDLE profile_handle = CreateFileW(
        profile.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        nullptr
      );
      ASSERT_NE(profile_handle, INVALID_HANDLE_VALUE)
        << "Win32 error " << GetLastError();
      CloseHandle(profile_handle);
    },
    expected_user
  );

  EXPECT_FALSE(result);
  EXPECT_TRUE(callback_called);
}

TEST(RunCommandUnelevatedTest, ElevatedTrayLaunchesProductionShapeStandardUserChild) {
  DWORD process_session = 0;
  ASSERT_TRUE(ProcessIdToSessionId(GetCurrentProcessId(), &process_session));
  if (process_session != WTSGetActiveConsoleSessionId()) {
    GTEST_SKIP() << "This contract requires an interactive test process";
  }

  HANDLE process_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token));
  auto process_token_guard = util::fail_guard([&]() {
    CloseHandle(process_token);
  });
  TOKEN_ELEVATION process_elevation {};
  DWORD returned = 0;
  ASSERT_TRUE(GetTokenInformation(
    process_token,
    TokenElevation,
    &process_elevation,
    sizeof(process_elevation),
    &returned
  ));
  if (process_elevation.TokenIsElevated == 0) {
    GTEST_SKIP() << "The de-elevated launch regression requires an elevated tray process";
  }

  const auto expected_user = platf::active_user_id();
  if (!expected_user) {
    GTEST_SKIP() << "Windows did not expose a validated active desktop user";
  }

  std::vector<wchar_t> system_directory(MAX_PATH + 1, L'\0');
  const auto system_directory_size = GetSystemDirectoryW(
    system_directory.data(),
    static_cast<UINT>(system_directory.size())
  );
  ASSERT_GT(system_directory_size, 0u);
  ASSERT_LT(system_directory_size, system_directory.size());
  const std::filesystem::path ping =
    std::filesystem::path {system_directory.data()} / L"ping.exe";
  ASSERT_TRUE(std::filesystem::is_regular_file(ping));

  const auto command =
    std::string {"\""} + platf::to_utf8(ping.wstring()) +
    "\" 127.0.0.1 -n 6";
  boost::filesystem::path working_directory {
    platf::to_utf8(std::filesystem::path {system_directory.data()}.wstring())
  };
  auto environment = boost::this_process::environment();
  FILE *child_output = std::tmpfile();
  ASSERT_NE(child_output, nullptr);
  auto child_output_guard = util::fail_guard([&]() {
    std::fclose(child_output);
  });
  boost::process::v1::group child_group;
  std::error_code launch_error;
  auto child = platf::run_command_unelevated(
    false,
    command,
    working_directory,
    environment,
    child_output,
    launch_error,
    &child_group,
    expected_user
  );
  ASSERT_FALSE(launch_error) << launch_error.message();
  ASSERT_TRUE(child.valid());
  auto child_cleanup = util::fail_guard([&]() {
    if (child.valid() && child.running()) {
      std::error_code ignored;
      child.terminate(ignored);
    }
    if (child.valid()) {
      std::error_code ignored;
      child.wait(ignored);
    }
  });

  HANDLE child_process = OpenProcess(
    PROCESS_QUERY_LIMITED_INFORMATION,
    FALSE,
    static_cast<DWORD>(child.id())
  );
  ASSERT_NE(child_process, nullptr) << "Win32 error " << GetLastError();
  auto child_process_guard = util::fail_guard([&]() {
    CloseHandle(child_process);
  });
  HANDLE child_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(child_process, TOKEN_QUERY, &child_token))
    << "Win32 error " << GetLastError();
  auto child_token_guard = util::fail_guard([&]() {
    CloseHandle(child_token);
  });
  TOKEN_ELEVATION child_elevation {};
  ASSERT_TRUE(GetTokenInformation(
    child_token,
    TokenElevation,
    &child_elevation,
    sizeof(child_elevation),
    &returned
  ));
  EXPECT_EQ(child_elevation.TokenIsElevated, 0u);

  TOKEN_ELEVATION_TYPE child_elevation_type = TokenElevationTypeFull;
  ASSERT_TRUE(GetTokenInformation(
    child_token,
    TokenElevationType,
    &child_elevation_type,
    sizeof(child_elevation_type),
    &returned
  ));
  EXPECT_NE(child_elevation_type, TokenElevationTypeFull);

  const auto child_user = token_sid(child_token);
  ASSERT_TRUE(child_user);
  EXPECT_EQ(*child_user, *expected_user);

  DWORD child_session = 0xFFFFFFFF;
  ASSERT_TRUE(GetTokenInformation(
    child_token,
    TokenSessionId,
    &child_session,
    sizeof(child_session),
    &returned
  ));
  EXPECT_EQ(child_session, process_session);
  EXPECT_EQ(child_session, WTSGetActiveConsoleSessionId());

  BOOL child_is_in_job = FALSE;
  ASSERT_TRUE(IsProcessInJob(
    child_process,
    child_group.native_handle(),
    &child_is_in_job
  )) << "Win32 error "
     << GetLastError();
  EXPECT_TRUE(child_is_in_job);

  std::error_code wait_error;
  child.wait(wait_error);
  ASSERT_FALSE(wait_error) << wait_error.message();
  ASSERT_EQ(_fseeki64(child_output, 0, SEEK_END), 0);
  const auto output_size = _ftelli64(child_output);
  ASSERT_GE(output_size, 0);
  EXPECT_GT(output_size, 0) << "The child did not inherit the redirected output handle";
}

TEST(RunCommandUnelevatedTest, TokenLaunchAssignsJobBeforeResumingChild) {
  constexpr DWORD requested_flags =
    EXTENDED_STARTUPINFO_PRESENT |
    CREATE_UNICODE_ENVIRONMENT |
    CREATE_NO_WINDOW;

  constexpr auto isolated =
    platf::detail::create_process_with_token_plan(requested_flags, true);
  EXPECT_EQ(isolated.creation_flags & EXTENDED_STARTUPINFO_PRESENT, 0u);
  EXPECT_NE(isolated.creation_flags & CREATE_SUSPENDED, 0u);
  EXPECT_NE(isolated.creation_flags & CREATE_UNICODE_ENVIRONMENT, 0u);
  EXPECT_NE(isolated.creation_flags & CREATE_NO_WINDOW, 0u);
  EXPECT_EQ(isolated.startup_info_size, sizeof(STARTUPINFOW));
  EXPECT_TRUE(isolated.assign_job_after_create);

  constexpr auto ungrouped =
    platf::detail::create_process_with_token_plan(requested_flags, false);
  EXPECT_EQ(ungrouped.creation_flags & EXTENDED_STARTUPINFO_PRESENT, 0u);
  EXPECT_EQ(ungrouped.creation_flags & CREATE_SUSPENDED, 0u);
  EXPECT_EQ(ungrouped.startup_info_size, sizeof(STARTUPINFOW));
  EXPECT_FALSE(ungrouped.assign_job_after_create);
}

TEST(RunCommandUnelevatedTest, SuspendedJobLaunchAssignsBeforeResume) {
  fake_token_job_launch_operations_t operations;

  EXPECT_EQ(
    platf::detail::finish_suspended_job_launch(operations),
    ERROR_SUCCESS
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {"assign", "resume"})
  );
}

TEST(RunCommandUnelevatedTest, SuspendedJobLaunchCleansUpAssignmentFailure) {
  fake_token_job_launch_operations_t operations;
  operations.assignment_succeeds = false;
  operations.failure = ERROR_SUCCESS;

  EXPECT_EQ(
    platf::detail::finish_suspended_job_launch(operations),
    ERROR_PROCESS_ABORTED
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "assign",
      "last-error",
      "terminate:" + std::to_string(ERROR_PROCESS_ABORTED),
      "close-thread",
      "close-process",
    })
  );
}

TEST(RunCommandUnelevatedTest, SuspendedJobLaunchCleansUpResumeFailure) {
  fake_token_job_launch_operations_t operations;
  operations.resume_succeeds = false;
  operations.failure = ERROR_ACCESS_DENIED;

  EXPECT_EQ(
    platf::detail::finish_suspended_job_launch(operations),
    ERROR_ACCESS_DENIED
  );
  EXPECT_EQ(
    operations.calls,
    (std::vector<std::string> {
      "assign",
      "resume",
      "last-error",
      "terminate:" + std::to_string(ERROR_ACCESS_DENIED),
      "close-thread",
      "close-process",
    })
  );
}
#endif
