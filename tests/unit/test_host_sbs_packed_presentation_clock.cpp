#include <gtest/gtest.h>

#include "src/host_sbs_packed_presentation_clock.h"
#include "src/video_depth_estimator.h"

namespace {
  using presentation_clock_t = platf::dxgi::detail::host_sbs_packed_presentation_clock_t;
  using steady_clock_t = presentation_clock_t::clock_t;
  using output_e = presentation_clock_t::output_e;
  using event_e = presentation_clock_t::timeout_event_e;
  using namespace std::chrono_literals;

  constexpr auto origin = steady_clock_t::time_point {1s};
  constexpr auto source_a = steady_clock_t::time_point {100ms};
  constexpr auto source_b = steady_clock_t::time_point {200ms};
  constexpr auto source_c = steady_clock_t::time_point {300ms};

  bool may_repeat(
    const presentation_clock_t &clock,
    const steady_clock_t::time_point copied_at,
    const presentation_clock_t::source_stamp_t &packed_source,
    const presentation_clock_t::source_stamp_t &current_source,
    const steady_clock_t::time_point now,
    const bool valid = true,
    const bool route_matches = true
  ) {
    return models::host_sbs_should_repeat_packed_presentation(
      models::host_sbs_renderer_e::parallax_v2, false, valid, route_matches,
      clock.hold_age(now - copied_at, packed_source, current_source, now),
      packed_source && current_source && *packed_source == *current_source
    );
  }
}  // namespace

TEST(HostSbsPackedPresentationClockTest, LongStaticSourceDoesNotPreExpireItsFirstChangedDelivery) {
  presentation_clock_t clock;
  clock.observe_source(source_a, source_a, origin);
  const auto changed_at = origin + 1h;
  clock.observe_source(source_a, source_a, changed_at - 1ms);
  ASSERT_TRUE(may_repeat(clock, origin, source_a, source_a, changed_at - 1ms));

  // Production observes B before its possibly busy/ineligible completion poll. The old age-only
  // decision fails on the first such delivery, although the changed-source hold has just begun.
  clock.observe_source(source_a, source_b, changed_at);
  EXPECT_FALSE(models::host_sbs_should_repeat_packed_presentation(
    models::host_sbs_renderer_e::parallax_v2, false, true, true, changed_at - origin, false));
  EXPECT_EQ(clock.hold_age(changed_at - origin, source_a, source_b, changed_at), 0ms);
  EXPECT_TRUE(may_repeat(clock, origin, source_a, source_b, changed_at + 8ms));

  // Neither a retained-source retry nor another changed source/admission failure rearms it.
  clock.observe_source(source_a, source_b, changed_at + 100ms);
  clock.observe_source(source_a, source_c, changed_at + 200ms);
  EXPECT_EQ(clock.hold_age(changed_at + 250ms - origin, source_a, source_c, changed_at + 250ms), 250ms);
  EXPECT_TRUE(may_repeat(clock, origin, source_a, source_c, changed_at + 250ms));
  EXPECT_FALSE(may_repeat(clock, origin, source_a, source_c, changed_at + 251ms));
}

TEST(HostSbsPackedPresentationClockTest, CompletionWaitAndUnsubmittedChangedDeliveriesConsumeTheSameInterval) {
  presentation_clock_t clock;
  const auto changed_at = origin + 1s;
  clock.observe_source(source_a, source_b, changed_at);
  // No publication or admission occurs during these polls; observing C cannot move the origin.
  clock.observe_source(source_a, source_c, changed_at + 249ms);
  EXPECT_TRUE(may_repeat(clock, origin, source_a, source_c, changed_at + 250ms));
  EXPECT_FALSE(may_repeat(clock, origin, source_a, source_c, changed_at + 251ms));
}

TEST(HostSbsPackedPresentationClockTest, NewerMismatchedPublicationMakesProgressWithoutRestartingDivergence) {
  presentation_clock_t clock;
  const auto changed_at = origin + 1s;
  clock.observe_source(source_a, source_b, changed_at);
  // B completed while C was current. Its genuine newer capture retains the old source-age cap;
  // the unresolved divergence interval is not restarted just because an owner was published.
  const auto b_copied_at = changed_at + 200ms;
  clock.observe_source(source_b, source_c, changed_at + 205ms);
  EXPECT_EQ(clock.hold_age(100ms, source_b, source_c, changed_at + 300ms), 100ms);
  EXPECT_TRUE(may_repeat(clock, b_copied_at, source_b, source_c, changed_at + 300ms));
  EXPECT_FALSE(may_repeat(clock, b_copied_at, source_b, source_c, b_copied_at + 251ms));

  // A mismatched completion with an old copy time cannot buy a fresh interval.
  EXPECT_FALSE(may_repeat(clock, origin, source_b, source_c, changed_at + 300ms));
}

TEST(HostSbsPackedPresentationClockTest, MatchingCurrentPublicationRearmsAfterAnotherStaticInterval) {
  presentation_clock_t clock;
  const auto changed_at = origin + 1s;
  clock.observe_source(source_a, source_b, changed_at);
  EXPECT_FALSE(may_repeat(clock, origin, source_a, source_b, changed_at + 251ms));
  const auto b_copied_at = changed_at + 260ms;
  clock.observe_source(source_b, source_b, b_copied_at);
  EXPECT_TRUE(may_repeat(clock, b_copied_at, source_b, source_b, b_copied_at + 1h));
  const auto c_arrived_at = b_copied_at + 1h;
  clock.observe_source(source_b, source_c, c_arrived_at);
  EXPECT_TRUE(may_repeat(clock, b_copied_at, source_b, source_c, c_arrived_at + 12ms));
  EXPECT_FALSE(may_repeat(clock, b_copied_at, source_b, source_c, c_arrived_at + 251ms));
}

TEST(HostSbsPackedPresentationClockTest, MissingIdentityDoesNotGrantOrRearmAChangedSourceHold) {
  presentation_clock_t clock;
  const auto changed_at = origin + 1s;
  clock.observe_source(source_a, std::nullopt, changed_at);
  EXPECT_FALSE(may_repeat(clock, origin, source_a, std::nullopt, changed_at));
  clock.observe_source(std::nullopt, source_b, changed_at);
  EXPECT_FALSE(may_repeat(clock, origin, std::nullopt, source_b, changed_at));

  clock.observe_source(source_a, source_b, changed_at);
  clock.observe_source(source_a, std::nullopt, changed_at + 100ms);
  EXPECT_FALSE(may_repeat(clock, origin, source_a, std::nullopt, changed_at + 100ms));
  clock.observe_source(source_a, source_c, changed_at + 251ms);
  EXPECT_FALSE(may_repeat(clock, origin, source_a, source_c, changed_at + 251ms));
  EXPECT_EQ(clock.hold_age(steady_clock_t::duration::max(), source_a, source_c, changed_at + 1ms),
            steady_clock_t::duration::max());
}

TEST(HostSbsPackedPresentationClockTest, PackedHoldDoesNotExtendGeometryOrBypassInvalidationAndRouteGuards) {
  presentation_clock_t clock;
  const auto changed_at = origin + 1h;
  clock.observe_source(source_a, source_b, changed_at);
  EXPECT_TRUE(may_repeat(clock, origin, source_a, source_b, changed_at + 1ms));
  EXPECT_FALSE(models::host_sbs_matched_completion_is_current(false, changed_at + 1ms - origin));
  // The production pixel owner still supplies all context/domain/transfer/terminal route gates.
  EXPECT_FALSE(may_repeat(clock, origin, source_a, source_b, changed_at + 1ms, false));
  EXPECT_FALSE(may_repeat(clock, origin, source_a, source_b, changed_at + 1ms, true, false));
  clock.invalidate();
  EXPECT_EQ(clock.hold_age(1h, source_a, source_b, changed_at + 1ms), 1h);
}

TEST(HostSbsPackedPresentationClockTest, TimeoutReportsActualFlatOnceAndOnlyCompletedDrawRecovers) {
  presentation_clock_t clock;
  // Rejecting a stale completion does not mean flat was shown: a valid packed/queued route may
  // still have been selected. Those choices must not arm the timeout or emit a false recovery.
  EXPECT_EQ(clock.observe_timeout(true, output_e::packed), event_e::none);
  EXPECT_EQ(clock.observe_timeout(true, output_e::queued), event_e::none);
  EXPECT_EQ(clock.observe_timeout(true, output_e::flat), event_e::expired);
  EXPECT_EQ(clock.observe_timeout(true, output_e::flat), event_e::none);
  clock.invalidate();
  EXPECT_EQ(clock.observe_timeout(true, output_e::flat), event_e::none);
  EXPECT_EQ(clock.observe_timeout(false, output_e::packed), event_e::none);
  EXPECT_EQ(clock.observe_timeout(false, output_e::queued), event_e::none);
  EXPECT_EQ(clock.observe_timeout(false, output_e::completed), event_e::recovered);
  EXPECT_EQ(clock.observe_timeout(false, output_e::completed), event_e::none);

  EXPECT_EQ(clock.observe_timeout(true, output_e::flat), event_e::expired);
  clock.reset();
  EXPECT_EQ(clock.observe_timeout(true, output_e::flat), event_e::expired);
}

TEST(HostSbsPackedPresentationClockTest, BenchmarkGateAllowsTheRegressionHoldWithinItsIndependentSafetyBound) {
  models::joined_completion_gate_benchmark::configuration_t gate;
  EXPECT_EQ(gate.hold_after_callback_enter_ms, 120u);
  EXPECT_EQ(gate.callback_safety_timeout_ms, 500u);
  gate.enabled = true;
  gate.target_frame_id = 3u;
  gate.placement = models::joined_completion_gate_benchmark::placement_e::after_interop_unmaps;
  gate.hold_after_callback_enter_ms = 350u;
  EXPECT_TRUE(gate.valid());
  gate.hold_after_callback_enter_ms = 400u;
  EXPECT_TRUE(gate.valid());
  gate.hold_after_callback_enter_ms = 401u;
  EXPECT_FALSE(gate.valid());
  gate.hold_after_callback_enter_ms = 350u;
  gate.callback_safety_timeout_ms = 350u;
  EXPECT_FALSE(gate.valid());
  gate.callback_safety_timeout_ms = 501u;
  EXPECT_FALSE(gate.valid());
}
