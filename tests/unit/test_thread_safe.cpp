/**
 * @file tests/unit/test_thread_safe.cpp
 * @brief Tests for thread-safe events.
 */
#include "../tests_common.h"

#include <chrono>
#include <memory>
#include <thread>

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
