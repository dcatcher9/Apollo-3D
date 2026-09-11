#include <array>
#include <chrono>
#include <deque>
#include <gtest/gtest.h>
#include <src/platform/windows/virtual_display_retirement.h>
#include <vector>

namespace {
  using namespace std::chrono_literals;
  using VDISPLAY::display_identity_state_e;
  using VDISPLAY::retirement_step_e;

  struct fake_clock_t {
    std::chrono::steady_clock::time_point current {};

    auto now() const {
      return current;
    }

    void sleep_for(std::chrono::milliseconds delay) {
      current += delay;
    }
  };

  struct retirement_probe_t {
    fake_clock_t clock;
    VDISPLAY::retirement_record_t record {true, {}};
    std::deque<display_identity_state_e> observations;
    display_identity_state_e default_observation = display_identity_state_e::absent;
    retirement_step_e before_result = retirement_step_e::observe;
    std::vector<std::chrono::milliseconds> query_times;
    unsigned before_calls = 0u;
    unsigned finish_calls = 0u;
    bool finish_result = true;

    display_identity_state_e query() {
      query_times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(clock.now().time_since_epoch()));
      if (observations.empty()) {
        return default_observation;
      }
      const auto result = observations.front();
      observations.pop_front();
      return result;
    }

    VDISPLAY::retirement_callbacks_t callbacks() {
      return {
        .before_observation = [&]() {
          ++before_calls;
          return before_result;
        },
        .query = [&]() {
          return query();
        },
        .finish = [&]() {
          ++finish_calls;
          return finish_result;
        },
      };
    }

    bool wait(std::chrono::milliseconds budget, VDISPLAY::retirement_timing_t timing = {}) {
      return record.wait_until(clock.now() + budget, timing, callbacks(), clock);
    }
  };

  TEST(VirtualDisplayRetirementTest, LocalAndRemoteTimingKeepThreeAbsencesAndFinalSettle) {
    for (const auto timing : std::array {
           VDISPLAY::retirement_timing_t {50ms, 100ms},
           VDISPLAY::retirement_timing_t {50ms, 250ms},
           VDISPLAY::retirement_timing_t {250ms, 250ms},
         }) {
      retirement_probe_t probe;
      const auto complete_at = 2 * timing.poll_interval + timing.settle_interval;
      ASSERT_TRUE(probe.wait(complete_at, timing));
      EXPECT_EQ(probe.query_times, (std::vector<std::chrono::milliseconds> {
                                     0ms,
                                     timing.poll_interval,
                                     2 * timing.poll_interval,
                                     complete_at,
                                   }));
      EXPECT_EQ(probe.before_calls, 3u);
      EXPECT_EQ(probe.finish_calls, 1u);
    }
  }

  TEST(VirtualDisplayRetirementTest, RemoveAcknowledgementCannotReplaceTopologyAbsence) {
    retirement_probe_t probe;
    probe.default_observation = display_identity_state_e::present;
    auto callbacks = probe.callbacks();
    unsigned accepted_removals = 0u;
    callbacks.before_observation = [&]() {
      ++accepted_removals;
      return retirement_step_e::observe;
    };
    EXPECT_FALSE(probe.record.wait_until(probe.clock.now() + 250ms, {}, callbacks, probe.clock));
    EXPECT_GT(accepted_removals, 0u);
    EXPECT_EQ(probe.finish_calls, 0u);
    EXPECT_EQ(probe.clock.now().time_since_epoch(), 250ms);
  }

  TEST(VirtualDisplayRetirementTest, ExternalAbsenceCanCompleteAfterRejectedRemove) {
    retirement_probe_t probe;
    unsigned rejected_removals = 0u;
    auto callbacks = probe.callbacks();
    callbacks.before_observation = [&]() {
      ++rejected_removals;
      // The adapter retains/retries its failed IOCTL without claiming that it is removal proof.
      return retirement_step_e::observe;
    };
    EXPECT_TRUE(probe.record.wait_until(probe.clock.now() + 200ms, {}, callbacks, probe.clock));
    EXPECT_EQ(rejected_removals, 3u);
    EXPECT_EQ(probe.finish_calls, 1u);
  }

  TEST(VirtualDisplayRetirementTest, IndeterminateObservationRestartsConsecutiveProof) {
    retirement_probe_t probe;
    probe.observations = {
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::indeterminate,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
    };
    EXPECT_TRUE(probe.wait(350ms));
    EXPECT_EQ(probe.query_times, (std::vector<std::chrono::milliseconds> {0ms, 50ms, 100ms, 150ms, 200ms, 250ms, 350ms}));
    EXPECT_EQ(probe.finish_calls, 1u);
  }

  TEST(VirtualDisplayRetirementTest, LateReappearanceRequiresAnotherCompleteProof) {
    retirement_probe_t probe;
    probe.observations = {
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::present,
    };
    EXPECT_TRUE(probe.wait(400ms));
    EXPECT_EQ(probe.query_times, (std::vector<std::chrono::milliseconds> {0ms, 50ms, 100ms, 200ms, 200ms, 250ms, 300ms, 400ms}));
    EXPECT_EQ(probe.finish_calls, 1u);
  }

  TEST(VirtualDisplayRetirementTest, UnpublishedIdentityRetainsQuarantineAcrossWaiterSlices) {
    retirement_probe_t probe;
    probe.record.was_published = false;
    EXPECT_FALSE(probe.wait(300ms));
    EXPECT_EQ(probe.finish_calls, 0u);
    EXPECT_TRUE(probe.wait(600ms));
    EXPECT_EQ(probe.clock.now().time_since_epoch(), 850ms);
    EXPECT_EQ(probe.query_times.back(), 850ms);
    EXPECT_EQ(probe.finish_calls, 1u);
  }

  TEST(VirtualDisplayRetirementTest, TimeoutDoesNotSkipOrReuseAnUnfinishedProof) {
    retirement_probe_t probe;
    EXPECT_FALSE(probe.wait(0ms));
    EXPECT_EQ(probe.query_times, (std::vector<std::chrono::milliseconds> {0ms}));
    EXPECT_EQ(probe.finish_calls, 0u);
    EXPECT_FALSE(probe.wait(150ms));
    EXPECT_EQ(probe.clock.now().time_since_epoch(), 100ms);
    EXPECT_EQ(probe.finish_calls, 0u);
    probe.query_times.clear();
    EXPECT_TRUE(probe.wait(200ms));
    EXPECT_EQ(probe.query_times, (std::vector<std::chrono::milliseconds> {100ms, 150ms, 200ms, 300ms}));
  }

  TEST(VirtualDisplayRetirementTest, CleanupFailureRetainsOwnershipAndSuccessfulRestoreIsNotReplayed) {
    retirement_probe_t probe;
    probe.finish_result = false;
    EXPECT_FALSE(probe.wait(200ms));
    EXPECT_EQ(probe.finish_calls, 1u);

    bool cleanup_latched = false;
    unsigned topology_restores = 0u;
    auto callbacks = probe.callbacks();
    callbacks.finish = [&]() {
      if (!cleanup_latched) {
        ++topology_restores;
        cleanup_latched = true;
      }
      return probe.query() == display_identity_state_e::absent;
    };
    // Local restoration can emit a notification that makes its final query indeterminate.
    probe.observations = {
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::indeterminate,
    };
    EXPECT_FALSE(probe.record.wait_until(probe.clock.now() + 200ms, {}, callbacks, probe.clock));
    EXPECT_TRUE(cleanup_latched);
    EXPECT_EQ(topology_restores, 1u);
    EXPECT_TRUE(probe.record.wait_until(probe.clock.now() + 200ms, {}, callbacks, probe.clock));
    EXPECT_EQ(topology_restores, 1u);
  }

  TEST(VirtualDisplayRetirementTest, SupersededAndAlreadyCompletedOwnersDoNotRunCleanup) {
    retirement_probe_t probe;
    probe.before_result = retirement_step_e::blocked;
    EXPECT_FALSE(probe.wait(1s));
    EXPECT_TRUE(probe.query_times.empty());
    probe.before_result = retirement_step_e::complete;
    EXPECT_TRUE(probe.wait(1s));
    EXPECT_TRUE(probe.query_times.empty());
    EXPECT_EQ(probe.finish_calls, 0u);
  }
}  // namespace
