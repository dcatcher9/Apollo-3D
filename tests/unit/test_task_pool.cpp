#include "src/task_pool.h"
#include "src/thread_pool.h"

#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

TEST(TaskPool, RunsMoveOnlyCallablesAndArgumentsInFifoOrder) {
  task_pool_util::TaskPool pool;
  std::vector<int> observed;
  auto argument = std::make_unique<int>(5);
  pool.push([owned = std::make_unique<int>(7), &observed](std::unique_ptr<int> value) {
    observed.push_back(*owned + *value);
  },
            std::move(argument));
  pool.push([&observed]() {
    observed.push_back(3);
  });
  EXPECT_FALSE(argument);
  EXPECT_TRUE(pool.ready());
  EXPECT_FALSE(pool.next());

  while (auto task = pool.pop()) {
    (*task)->run();
  }
  EXPECT_EQ(observed, (std::vector<int> {12, 3}));
  EXPECT_FALSE(pool.ready());
  EXPECT_FALSE(pool.pop());
}

TEST(TaskPool, CancelsPendingTimersAndExposesTheEarliestDeadline) {
  task_pool_util::TaskPool pool;
  std::vector<int> observed;
  const auto later = pool.pushDelayed([&observed]() {
    observed.push_back(2);
  },
                                      2h);
  const auto earlier = pool.pushDelayed([&observed]() {
    observed.push_back(1);
  },
                                        1h);
  ASSERT_TRUE(pool.next());
  const auto earliest_deadline = *pool.next();
  EXPECT_FALSE(pool.ready());
  EXPECT_FALSE(pool.pop());
  EXPECT_TRUE(pool.cancel(earlier));
  EXPECT_FALSE(pool.cancel(earlier));
  ASSERT_TRUE(pool.next());
  EXPECT_GT(*pool.next(), earliest_deadline);

  const auto due = pool.pushDelayed([&observed](std::unique_ptr<int> value) {
    observed.push_back(*value);
  },
                                    -1h,
                                    std::make_unique<int>(9));
  ASSERT_TRUE(pool.next());
  EXPECT_LT(*pool.next(), std::chrono::steady_clock::now());
  EXPECT_TRUE(pool.ready());
  auto task = pool.pop();
  ASSERT_TRUE(task);
  (*task)->run();
  EXPECT_FALSE(pool.cancel(due));
  EXPECT_EQ(observed, (std::vector<int> {9}));
  EXPECT_FALSE(pool.ready());
  EXPECT_TRUE(pool.cancel(later));
  EXPECT_FALSE(pool.next());
}

TEST(TaskPool, ImmediateTasksKeepPriorityOverDueTimers) {
  task_pool_util::TaskPool pool;
  std::vector<int> observed;
  pool.pushDelayed([&observed]() {
    observed.push_back(2);
  },
                   -1h);
  pool.push([&observed]() {
    observed.push_back(1);
  });
  while (auto task = pool.pop()) {
    (*task)->run();
  }
  EXPECT_EQ(observed, (std::vector<int> {1, 2}));
}

TEST(TaskPool, TaskExceptionsDoNotEscapeExecution) {
  task_pool_util::TaskPool pool;
  pool.push([]() {
    throw std::runtime_error("discarded task failure");
  });
  auto task = pool.pop();
  ASSERT_TRUE(task);
  EXPECT_NO_THROW((*task)->run());
}

TEST(ThreadPool, KeepsRunningAfterATaskThrows) {
  thread_pool_util::ThreadPool pool;
  std::promise<int> completed;
  auto result = completed.get_future();
  pool.push([]() {
    throw std::runtime_error("discarded worker task failure");
  });
  pool.push([&completed]() {
    completed.set_value(42);
  });
  pool.start(1);
  const auto status = result.wait_for(2s);
  pool.stop();
  pool.join();
  ASSERT_EQ(status, std::future_status::ready);
  EXPECT_EQ(result.get(), 42);
}

TEST(ThreadPool, StopDuringACallbackDrainsQueuedImmediateTasks) {
  thread_pool_util::ThreadPool pool;
  std::promise<void> entered;
  auto started = entered.get_future();
  std::promise<void> release;
  auto resume = release.get_future();
  std::promise<void> drained;
  auto completed = drained.get_future();
  pool.push([&entered, &resume]() {
    entered.set_value();
    resume.wait();
  });
  pool.push([&drained]() {
    drained.set_value();
  });
  pool.start(1);

  const auto started_status = started.wait_for(2s);
  pool.stop();
  // Release and join before asserting, including when the worker missed the deadline.
  release.set_value();
  pool.join();

  ASSERT_EQ(started_status, std::future_status::ready);
  EXPECT_EQ(completed.wait_for(0s), std::future_status::ready);
}
