/**
 * @file tests/unit/test_thread_safe.cpp
 * @brief Tests for thread-safe events.
 */
#include "../tests_common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <src/thread_safe.h>

using namespace std::chrono_literals;

TEST(ThreadSafeEventTest, TryPopConsumesAvailableValueWithoutWaiting) {
  safe::event_t<int> event;

  EXPECT_FALSE(event.try_pop());

  event.raise(42);
  auto value = event.try_pop();

  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 42);
  EXPECT_FALSE(event.try_pop());
}

TEST(ThreadSafeEventTest, TryRaiseDoesNotOverwriteAvailableValue) {
  safe::event_t<std::shared_ptr<int>> event;
  auto retained_value = std::make_shared<int>(7);

  EXPECT_TRUE(event.try_raise(std::make_shared<int>(42)));
  EXPECT_FALSE(event.try_raise(std::move(retained_value)));
  EXPECT_TRUE(retained_value);

  auto value = event.try_pop();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 42);
}

TEST(ThreadSafeEventTest, TimedPopWakesForValue) {
  safe::event_t<int> event;
  std::thread producer {[&event] {
    std::this_thread::sleep_for(10ms);
    event.raise(7);
  }};

  auto value = event.pop(1s);
  producer.join();

  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 7);
}

TEST(ThreadSafeEventTest, TimedPopWakesWhenStopped) {
  safe::event_t<int> event;
  std::thread stopper {[&event] {
    std::this_thread::sleep_for(10ms);
    event.stop();
  }};

  auto value = event.pop(1s);
  stopper.join();

  EXPECT_FALSE(value);
  EXPECT_FALSE(event.peek());
  EXPECT_FALSE(event.running());
}

TEST(ThreadSafeEventTest, WakeEndsATimedPopWithoutAValue) {
  safe::event_t<int> event;
  std::thread waker {[&event] {
    std::this_thread::sleep_for(10ms);
    event.wake();
  }};

  const auto started = std::chrono::steady_clock::now();
  auto value = event.pop(5s);
  waker.join();

  EXPECT_FALSE(value);
  EXPECT_LT(std::chrono::steady_clock::now() - started, 4s);
  EXPECT_TRUE(event.running());
}

TEST(ThreadSafeEventTest, WakeBeforeTheWaitIsKeptOnceAndAValueConsumesIt) {
  safe::event_t<int> event;
  // A wake between a consumer's last check and its wait is not lost...
  event.wake();
  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(5s));
  EXPECT_LT(std::chrono::steady_clock::now() - started, 4s);
  // ...and is consumed: the next wait times out normally.
  EXPECT_FALSE(event.pop(10ms));

  // A stored value wins over a pending wake and consumes it too.
  event.wake();
  event.raise(3);
  auto value = event.pop(0ms);
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 3);
  const auto after_value = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(20ms));
  EXPECT_GE(std::chrono::steady_clock::now() - after_value, 10ms);  // A full wait, not a wake.
}

TEST(ThreadSafeEventTest, DiscardedWakeLeavesAFullWaitAndKeepsStoredValues) {
  safe::event_t<int> event;
  event.wake();
  event.discard_wake();
  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(20ms));
  EXPECT_GE(std::chrono::steady_clock::now() - started, 15ms);

  event.raise(4);
  event.wake();
  event.discard_wake();
  auto value = event.pop(0ms);
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 4);
}

// With this toolchain (winpthreads, no clock-based condition-variable wait) a timed wait that
// nothing notifies ends only at the next Windows scheduler tick (15.625 ms), even for a zero
// timeout, whatever the timer resolution. The encode loop polls its control queues with zero
// timeouts every iteration; each such poll on an empty queue held it until the tick, locking the
// loop to 64 iterations/s. A zero (or negative) timeout must check and return.
TEST(ThreadSafeEventTest, ZeroTimeoutPopAndViewOfAnEmptyEventDoNotWait) {
  safe::event_t<int> event;
  const auto started = std::chrono::steady_clock::now();
  for (int poll = 0; poll < 10; ++poll) {
    EXPECT_FALSE(event.pop(0ms));
    EXPECT_FALSE(event.view(0ms));
    EXPECT_FALSE(event.pop(-1ms));
  }
  EXPECT_LT(std::chrono::steady_clock::now() - started, 15ms);
}

TEST(ThreadSafeEventTest, ZeroTimeoutPopKeepsTheTimedPopContract) {
  safe::event_t<int> event;
  event.raise(5);
  auto value = event.pop(0ms);
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 5);
  EXPECT_FALSE(event.pop(0ms));

  // A zero-timeout pop consumes a pending wake like any timed pop, so the next wait is a full one.
  event.wake();
  EXPECT_FALSE(event.pop(0ms));
  const auto after_wake = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(20ms));
  EXPECT_GE(std::chrono::steady_clock::now() - after_wake, 15ms);

  // view() leaves the value in place; a stopped event returns nothing.
  event.raise(6);
  ASSERT_TRUE(event.view(0ms));
  EXPECT_EQ(*event.view(0ms), 6);
  EXPECT_TRUE(event.peek());
  event.stop();
  EXPECT_FALSE(event.pop(0ms));
  EXPECT_FALSE(event.view(0ms));
}

TEST(ThreadSafeQueueTest, ZeroTimeoutPopOfAnEmptyQueueDoesNotWait) {
  safe::queue_t<int> queue;
  const auto started = std::chrono::steady_clock::now();
  for (int poll = 0; poll < 20; ++poll) {
    EXPECT_FALSE(queue.pop(0ms));
  }
  EXPECT_LT(std::chrono::steady_clock::now() - started, 15ms);

  queue.raise(1);
  queue.raise(2);
  auto first = queue.pop(0ms);
  ASSERT_TRUE(first);
  EXPECT_EQ(*first, 1);
  queue.stop();
  EXPECT_FALSE(queue.pop(0ms));
}

namespace {
  struct overshoot_t {
    std::chrono::nanoseconds median {}, max {};
    bool early = false;
  };

  // `waits` consecutive bounded waits that nothing ends early: how far past its bound each ended.
  // The median, because a single wait preempted on a busy host (a live stream, a game) may end
  // several milliseconds late on any timer; a tick-bound wait is late by the same 4-15 ms each time.
  template<class Wait>
  overshoot_t bounded_wait_overshoot(std::chrono::nanoseconds bound, int waits, Wait &&wait) {
    overshoot_t result;
    std::vector<std::chrono::nanoseconds> overs;
    for (int i = 0; i < waits; ++i) {
      const auto started = std::chrono::steady_clock::now();
      EXPECT_FALSE(wait(bound));
      const auto over = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started - bound);
      result.early = result.early || over < 0ns;
      overs.push_back(over);
    }
    std::nth_element(overs.begin(), overs.begin() + overs.size() / 2, overs.end());
    result.median = overs[overs.size() / 2];
    result.max = *std::max_element(overs.begin(), overs.end());
    return result;
  }
}  // namespace

// For the same reason, a timed pop that nothing notifies ends at the first scheduler tick at or
// after its bound: 3 ms pops measured 12.4-12.5 ms late on average (each one starts just after the
// tick that ended the previous one). With a deadline waiter it ends at its bound.
TEST(ThreadSafeEventTest, BoundedPopThatNothingNotifiesEndsWithinAMillisecondOfItsBound) {
  const auto waiter = platf::create_deadline_waiter();
  ASSERT_TRUE(waiter);
  safe::event_t<int> event;
  const auto tick_bound = bounded_wait_overshoot(3ms, 5, [&](auto bound) {
    return event.pop(bound);
  });
  const auto overshoot = bounded_wait_overshoot(3ms, 13, [&](auto bound) {
    return event.pop(bound, *waiter);
  });
  std::printf(
    "[ MEASURE  ] 3 ms bounded pops: condition variable overshoot median %.3f max %.3f ms; deadline waiter median %.3f max %.3f ms\n",
    std::chrono::duration<double, std::milli>(tick_bound.median).count(),
    std::chrono::duration<double, std::milli>(tick_bound.max).count(),
    std::chrono::duration<double, std::milli>(overshoot.median).count(),
    std::chrono::duration<double, std::milli>(overshoot.max).count()
  );
  EXPECT_FALSE(overshoot.early);
  // A median of 0.04-0.6 ms measured. No gate on the maximum: one preempted wait must not fail the suite, and a
  // regression to tick-bound waits is late on every wait, so the median catches it.
  EXPECT_LT(overshoot.median, 2ms);
}

namespace {
  // Records what the event asks of its waiter; waits as a real one would, ignoring early ends.
  struct recording_waiter_t: safe::deadline_waiter_t {
    std::unique_ptr<safe::deadline_waiter_t> real = platf::create_deadline_waiter();
    std::atomic<int> notifies {0}, waits {0};
    bool can_wait = true;

    void notify() noexcept override {
      ++notifies;
      real->notify();
    }

    bool wait_until(std::chrono::steady_clock::time_point deadline) noexcept override {
      ++waits;
      return can_wait && real->wait_until(deadline);
    }
  };
}  // namespace

TEST(ThreadSafeEventTest, DeadlinePopKeepsTheTimedPopContract) {
  recording_waiter_t waiter;
  ASSERT_TRUE(waiter.real);
  safe::event_t<int> event;

  // A stored value or a pending wake ends it without waiting, and it consumes the wake.
  event.raise(1);
  auto value = event.pop(1s, waiter);
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 1);
  event.wake();
  EXPECT_FALSE(event.pop(1s, waiter));
  EXPECT_EQ(waiter.waits, 0);

  // A zero timeout only checks.
  EXPECT_FALSE(event.pop(0ms, waiter));
  EXPECT_EQ(waiter.waits, 0);

  // A value, a wake or stop() from another thread ends the wait as it arrives, not at its bound.
  const auto ended_by = [&](auto &&end) {
    waiter.waits = 0;
    std::thread other {[&] {
      while (waiter.waits == 0) {
        std::this_thread::yield();
      }
      std::this_thread::sleep_for(2ms);
      end();
    }};
    const auto started = std::chrono::steady_clock::now();
    auto result = event.pop(1s, waiter);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    other.join();
    EXPECT_LT(elapsed, 500ms);
    return result;
  };
  value = ended_by([&] {
    event.raise(2);
  });
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 2);
  EXPECT_FALSE(ended_by([&] {
    event.wake();
  }));
  // The wake was consumed: the next wait runs to its bound.
  const auto after_wake = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(5ms, waiter));
  EXPECT_GE(std::chrono::steady_clock::now() - after_wake, 5ms);

  // The event notifies the waiter only while a pop waits on it.
  const int notified = waiter.notifies;
  event.raise(3);
  event.wake();
  EXPECT_EQ(waiter.notifies, notified);
  EXPECT_EQ(*event.pop(0ms), 3);

  EXPECT_FALSE(ended_by([&] {
    event.stop();
  }));
  EXPECT_FALSE(event.pop(1s, waiter));
}

TEST(ThreadSafeEventTest, DeadlinePopNeverEndsEarlyForAStaleNotification) {
  recording_waiter_t waiter;
  ASSERT_TRUE(waiter.real);
  safe::event_t<int> event;

  // A notification left over from an earlier wait, as when a capture arrives just after its timer.
  waiter.real->notify();
  auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(5ms, waiter));
  EXPECT_GE(std::chrono::steady_clock::now() - started, 5ms);
  EXPECT_GE(waiter.waits, 2);

  // The timer of a wait that a value ended does not end the next wait early.
  waiter.waits = 0;
  std::thread raiser {[&] {
    while (waiter.waits == 0) {
      std::this_thread::yield();
    }
    event.raise(1);
  }};
  EXPECT_TRUE(event.pop(20ms, waiter));
  raiser.join();
  started = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(30ms, waiter));
  EXPECT_GE(std::chrono::steady_clock::now() - started, 30ms);
}

TEST(ThreadSafeEventTest, DeadlinePopFallsBackToTheConditionVariable) {
  // A waiter that cannot wait leaves the condition variable's deadline, a scheduler tick late.
  recording_waiter_t waiter;
  ASSERT_TRUE(waiter.real);
  waiter.can_wait = false;
  safe::event_t<int> event;
  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(event.pop(5ms, waiter));
  EXPECT_GE(std::chrono::steady_clock::now() - started, 5ms);
  EXPECT_EQ(waiter.waits, 1);

  // So does a second consumer while another waits on its waiter; both still get their values.
  waiter.can_wait = true;
  waiter.waits = 0;
  recording_waiter_t second;
  ASSERT_TRUE(second.real);
  std::optional<int> first_value;
  std::thread first {[&] {
    if (auto value = event.pop(1s, waiter)) {
      first_value = *value;
    }
  }};
  while (waiter.waits == 0) {
    std::this_thread::yield();
  }
  std::optional<int> second_value;
  std::thread other {[&] {
    if (auto value = event.pop(1s, second)) {
      second_value = *value;
    }
  }};
  std::this_thread::sleep_for(5ms);  // Time for the second consumer to wait too.
  event.raise(1);
  // The event holds one value: raise the second only once a consumer has taken the first.
  const auto taken_by = std::chrono::steady_clock::now() + 2s;
  while (event.peek() && std::chrono::steady_clock::now() < taken_by) {
    std::this_thread::yield();
  }
  event.raise(2);
  first.join();
  other.join();
  ASSERT_TRUE(first_value);
  ASSERT_TRUE(second_value);
  EXPECT_EQ(*first_value + *second_value, 3);
}

TEST(ThreadSafeEventTest, TimedViewWakesWhenStopped) {
  safe::event_t<int> event;
  std::thread stopper {[&event] {
    std::this_thread::sleep_for(10ms);
    event.stop();
  }};

  auto value = event.view(1s);
  stopper.join();

  EXPECT_FALSE(value);
  EXPECT_FALSE(event.running());
}

TEST(ThreadSafeQueueTest, AccessorsReflectStoppedState) {
  safe::queue_t<int> queue;
  queue.raise(42);

  EXPECT_TRUE(queue.peek());
  EXPECT_TRUE(queue.running());

  queue.stop();

  EXPECT_FALSE(queue.peek());
  EXPECT_FALSE(queue.running());
}

TEST(ThreadSafeQueueTest, CanAtomicallyDiscardOverflowAndRecoveryDependentNewestValue) {
  safe::queue_t<int> queue {2};
  queue.raise(1);
  queue.raise(2);

  const auto result = queue.raise_with_overflow_policy(false, 3);

  EXPECT_FALSE(result.queued);
  EXPECT_EQ(result.dropped, 2);
  EXPECT_FALSE(queue.peek());
}

TEST(ThreadSafeQueueTest, CanReplaceOverflowWithNewestRecoveryValue) {
  safe::queue_t<int> queue {2};
  queue.raise(1);
  queue.raise(2);

  const auto result = queue.raise_with_overflow_policy(true, 3);

  EXPECT_TRUE(result.queued);
  EXPECT_EQ(result.dropped, 2);
  ASSERT_TRUE(queue.peek());
  const auto value = queue.pop(0ms);
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 3);
  EXPECT_FALSE(queue.peek());
}

TEST(ThreadSafeQueueTest, TimedPopWakesWhenStopped) {
  safe::queue_t<int> queue;
  std::thread stopper {[&queue] {
    std::this_thread::sleep_for(10ms);
    queue.stop();
  }};

  auto value = queue.pop(1s);
  stopper.join();

  EXPECT_FALSE(value);
  EXPECT_FALSE(queue.running());
}

TEST(ThreadSafeSharedTest, FailedConstructionDestroysObjectAndCanRetry) {
  struct tracked_t {
    ~tracked_t() {
      if (destructions) {
        ++*destructions;
      }
    }

    int *destructions = nullptr;
  };

  int destructions = 0;
  int attempts = 0;
  auto shared = safe::make_shared<tracked_t>(
    [&](tracked_t &object) {
      object.destructions = &destructions;
      return attempts++ == 0 ? -1 : 0;
    },
    [](tracked_t &) {}
  );

  EXPECT_FALSE(shared.ref());
  EXPECT_EQ(destructions, 1);

  {
    auto ref = shared.ref();
    ASSERT_TRUE(ref);
    EXPECT_EQ(attempts, 2);
  }

  EXPECT_EQ(destructions, 2);
}

TEST(ThreadSafeSharedTest, CopyAssignmentBalancesReferencesAndHandlesEmptyPointers) {
  struct tracked_t {
    ~tracked_t() {
      if (destructions) {
        ++*destructions;
      }
    }

    int *destructions = nullptr;
  };

  int destructions = 0;
  int external_destructions = 0;
  auto shared = safe::make_shared<tracked_t>(
    [&](tracked_t &object) {
      object.destructions = &destructions;
      return 0;
    },
    [&](tracked_t &) {
      ++external_destructions;
    }
  );
  using ptr_t = decltype(shared)::ptr_t;

  ptr_t empty_a;
  ptr_t empty_b;
  empty_a = empty_b;
  EXPECT_FALSE(empty_a);

  auto first = shared.ref();
  ASSERT_TRUE(first);
  ptr_t second;
  second = first;
  ASSERT_TRUE(second);
  second = second;
  ASSERT_TRUE(second);

  first = empty_a;
  EXPECT_FALSE(first);
  EXPECT_EQ(destructions, 0);
  EXPECT_EQ(external_destructions, 0);

  second = empty_a;
  EXPECT_FALSE(second);
  EXPECT_EQ(destructions, 1);
  EXPECT_EQ(external_destructions, 1);
}

TEST(ThreadSafePostTest, EmptyMailDestructionIsSafe) {
  safe::post_t<safe::event_t<int>> post {safe::mail_t {}};
  EXPECT_FALSE(post.mail);
}
