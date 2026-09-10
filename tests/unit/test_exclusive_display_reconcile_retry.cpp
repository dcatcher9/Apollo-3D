/**
 * @file tests/unit/test_exclusive_display_reconcile_retry.cpp
 * @brief Bounded exclusive-display reconciliation retry state tests.
 */
#include "../tests_common.h"
#include "src/platform/windows/exclusive_display_reconcile_retry.h"

namespace {
  using namespace std::chrono_literals;
  using platf::primary_display::detail::exclusive_reconcile_followup_e;
  using platf::primary_display::detail::exclusive_reconcile_retry_t;
  using platf::primary_display::detail::reconcile_followup;
  using platf::primary_display::exclusive_reconcile_result_e;
}  // namespace

TEST(ExclusiveDisplayReconcileRetry, FollowupPolicySeparatesAutonomousAndEventDrivenRetries) {
  EXPECT_EQ(
    reconcile_followup(exclusive_reconcile_result_e::retry_active, false, true),
    exclusive_reconcile_followup_e::retry_delayed
  );
  EXPECT_EQ(
    reconcile_followup(exclusive_reconcile_result_e::retry_active, true, true),
    exclusive_reconcile_followup_e::retry_delayed
  );
  EXPECT_EQ(
    reconcile_followup(exclusive_reconcile_result_e::pending_recovery, true, true),
    exclusive_reconcile_followup_e::retry_delayed
  );
  EXPECT_EQ(
    reconcile_followup(exclusive_reconcile_result_e::settled, true, true),
    exclusive_reconcile_followup_e::retry_immediate
  );
  EXPECT_EQ(
    reconcile_followup(exclusive_reconcile_result_e::retry_active, false, false),
    exclusive_reconcile_followup_e::exhausted
  );
  EXPECT_EQ(
    reconcile_followup(exclusive_reconcile_result_e::pending_recovery, true, false),
    exclusive_reconcile_followup_e::exhausted
  );
  EXPECT_EQ(
    reconcile_followup(exclusive_reconcile_result_e::pending_recovery, false, true),
    exclusive_reconcile_followup_e::finish
  );
}

TEST(ExclusiveDisplayReconcileRetry, TransientFailureSchedulesRetryWithoutAnotherDisplayEvent) {
  exclusive_reconcile_retry_t retry {3, 1s};
  retry.notify();
  const auto first_worker = retry.start();
  ASSERT_TRUE(first_worker);

  ASSERT_TRUE(retry.begin_attempt(*first_worker, 17));
  const auto ticket = retry.schedule(*first_worker, 17, 1);
  ASSERT_TRUE(ticket);
  EXPECT_EQ(retry.attempts(), 1u);

  const auto retry_worker = retry.dispatch(*ticket, 17);
  ASSERT_TRUE(retry_worker);
  EXPECT_TRUE(retry.begin_attempt(*retry_worker, 17));
  EXPECT_EQ(retry.attempts(), 2u);
  retry.finish(*retry_worker, 1);
  EXPECT_EQ(retry.attempts(), 0u);
}

TEST(ExclusiveDisplayReconcileRetry, PersistentFailureStopsAtAttemptLimit) {
  exclusive_reconcile_retry_t retry {2, 1s};
  retry.notify();
  auto worker = retry.start();
  ASSERT_TRUE(worker);

  ASSERT_TRUE(retry.begin_attempt(*worker, 21));
  auto ticket = retry.schedule(*worker, 21, 1);
  ASSERT_TRUE(ticket);
  auto next_worker = retry.dispatch(*ticket, 21);
  ASSERT_TRUE(next_worker);
  worker = next_worker;

  ASSERT_TRUE(retry.begin_attempt(*worker, 21));
  EXPECT_FALSE(retry.can_retry(*worker, 21));
  EXPECT_FALSE(retry.schedule(*worker, 21, 1));
  EXPECT_FALSE(retry.begin_attempt(*worker, 21));

  // A later real notification starts a new bounded cycle after the old one is idle.
  retry.finish(*worker, 1);
  EXPECT_FALSE(retry.start());
  retry.notify();
  worker = retry.start();
  ASSERT_TRUE(worker);
  EXPECT_TRUE(retry.begin_attempt(*worker, 21));
  EXPECT_EQ(retry.attempts(), 1u);
}

TEST(ExclusiveDisplayReconcileRetry, InFlightNotificationCannotResetRetryBudget) {
  exclusive_reconcile_retry_t retry {3, 1s};
  retry.notify();
  const auto worker = retry.start();
  ASSERT_TRUE(worker);
  ASSERT_TRUE(retry.begin_attempt(*worker, 9));

  retry.notify();
  EXPECT_FALSE(retry.start());
  EXPECT_EQ(retry.scheduler_generation(), *worker);
  EXPECT_EQ(retry.attempts(), 1u);
  EXPECT_TRUE(retry.begin_attempt(*worker, 9));
  EXPECT_EQ(retry.attempts(), 2u);
}

TEST(ExclusiveDisplayReconcileRetry, DelayedAndDispatchedWorkKeepOwnershipOfTheRetryBudget) {
  exclusive_reconcile_retry_t retry {3, 1s};
  retry.notify();
  const auto first_worker = retry.start();
  ASSERT_TRUE(first_worker);
  ASSERT_TRUE(retry.begin_attempt(*first_worker, 13));
  const auto ticket = retry.schedule(*first_worker, 13, 1);
  ASSERT_TRUE(ticket);

  // A display notification while the timer is pending joins the existing bounded cycle.
  retry.notify();
  EXPECT_FALSE(retry.start());
  EXPECT_EQ(retry.attempts(), 1u);

  // Dispatch claims worker ownership before releasing the controller lock. A notification in the
  // callback-to-task gap cannot reset the cycle or invalidate the worker.
  const auto dispatched_worker = retry.dispatch(*ticket, 13);
  ASSERT_TRUE(dispatched_worker);
  EXPECT_FALSE(retry.start());
  EXPECT_EQ(retry.scheduler_generation(), *first_worker);
  EXPECT_TRUE(retry.begin_attempt(*dispatched_worker, 13));
  EXPECT_EQ(retry.attempts(), 2u);
}

TEST(ExclusiveDisplayReconcileRetry, SessionReplacementAndShutdownInvalidateDelayedCallbacks) {
  exclusive_reconcile_retry_t retry {3, 1s};
  retry.notify();
  const auto worker = retry.start();
  ASSERT_TRUE(worker);
  ASSERT_TRUE(retry.begin_attempt(*worker, 5));
  const auto replaced_ticket = retry.schedule(*worker, 5, 1);
  ASSERT_TRUE(replaced_ticket);
  retry.notify();
  EXPECT_FALSE(retry.dispatch(*replaced_ticket, 6));
  EXPECT_EQ(retry.attempts(), 0u);
  EXPECT_EQ(retry.handled_event_generation(), 1u);

  const auto shutdown_worker = retry.start();
  ASSERT_TRUE(shutdown_worker);
  ASSERT_TRUE(retry.begin_attempt(*shutdown_worker, 6));
  const auto shutdown_ticket = retry.schedule(*shutdown_worker, 6, 2);
  ASSERT_TRUE(shutdown_ticket);
  retry.stop();
  EXPECT_FALSE(retry.dispatch(*shutdown_ticket, 6));
  EXPECT_FALSE(retry.begin_attempt(retry.scheduler_generation(), 6));
  EXPECT_FALSE(retry.start());
}

TEST(ExclusiveDisplayReconcileRetry, StalePostedMessageCannotRestartCompletedCycle) {
  exclusive_reconcile_retry_t retry {1, 1s};
  retry.notify();
  const auto worker = retry.start();
  ASSERT_TRUE(worker);
  ASSERT_TRUE(retry.begin_attempt(*worker, 7));

  // An event received before the exhausted transition belongs to the bounded cycle. A queued
  // message for that same event is stale, while a notification after finish starts a fresh cycle.
  retry.notify();
  retry.finish(*worker, retry.event_generation());
  EXPECT_EQ(retry.handled_event_generation(), 2u);
  EXPECT_FALSE(retry.start());
  retry.notify();
  EXPECT_TRUE(retry.start());
}

TEST(ExclusiveDisplayReconcileRetry, ElapsedDeadlineBoundsDelayedRetries) {
  using clock_t = exclusive_reconcile_retry_t::clock_t;
  const auto started_at = clock_t::time_point {};
  exclusive_reconcile_retry_t retry {10, 100ms};
  retry.notify();
  const auto worker = retry.start();
  ASSERT_TRUE(worker);
  ASSERT_TRUE(retry.begin_attempt(*worker, 23, started_at));
  const auto ticket = retry.schedule(*worker, 23, 1, started_at + 10ms);
  ASSERT_TRUE(ticket);

  retry.notify();
  EXPECT_FALSE(retry.dispatch(*ticket, 23, started_at + 100ms));
  EXPECT_EQ(retry.attempts(), 0u);
  EXPECT_EQ(retry.handled_event_generation(), 2u);
  EXPECT_FALSE(retry.start());

  retry.notify();
  EXPECT_TRUE(retry.start());
}
