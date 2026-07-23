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
