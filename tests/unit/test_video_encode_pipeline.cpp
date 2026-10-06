#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <set>
#include <src/video_encode_pipeline.h>
#include <thread>
#include <vector>

namespace {
  using namespace std::chrono_literals;
  using pipeline_t = video::detail::encode_pipeline_t<int>;

  // Holds each retirement until the test releases that frame, and records what retired where.
  class retire_gate {
  public:
    bool retire(int frame) {
      std::unique_lock lock(mutex);
      started.push_back(frame);
      threads.push_back(std::this_thread::get_id());
      condition.notify_all();
      condition.wait(lock, [&] {
        return open || released.contains(frame);
      });
      retired.push_back(frame);
      condition.notify_all();
      return !failing.contains(frame);
    }

    void release(int frame) {
      std::lock_guard lock(mutex);
      released.insert(frame);
      condition.notify_all();
    }

    void open_all() {
      std::lock_guard lock(mutex);
      open = true;
      condition.notify_all();
    }

    bool await_started(std::size_t count) {
      std::unique_lock lock(mutex);
      return condition.wait_for(lock, 3s, [&] {
        return started.size() >= count;
      });
    }

    std::vector<int> retired_frames() {
      std::lock_guard lock(mutex);
      return retired;
    }

    std::vector<int> started_frames() {
      std::lock_guard lock(mutex);
      return started;
    }

    std::vector<std::thread::id> retiring_threads() {
      std::lock_guard lock(mutex);
      return threads;
    }

    std::set<int> failing;

  private:
    std::mutex mutex;
    std::condition_variable condition;
    std::set<int> released;
    bool open = false;
    std::vector<int> started, retired;
    std::vector<std::thread::id> threads;
  };
}  // namespace

TEST(EncodePipelineTest, RetiresEveryPictureInSubmissionOrderOnItsOwnThread) {
  retire_gate gate;
  gate.open_all();
  std::vector<int> expected;
  {
    pipeline_t pipeline(2, [&gate](int &frame) {
      return gate.retire(frame);
    });
    for (int frame = 1; frame <= 200; ++frame) {
      ASSERT_TRUE(pipeline.acquire());
      pipeline.push(frame);
      expected.push_back(frame);
      EXPECT_LE(pipeline.in_flight(), 2u);
    }
    EXPECT_TRUE(pipeline.drain());
    EXPECT_EQ(pipeline.in_flight(), 0u);
    const auto stats = pipeline.take_stats();
    EXPECT_EQ(stats.pushed, 200u);
    EXPECT_EQ(stats.retired, 200u);
    EXPECT_LE(stats.pushed_behind, 200u);
  }
  EXPECT_EQ(gate.retired_frames(), expected);
  for (const auto id : gate.retiring_threads()) {
    EXPECT_NE(id, std::this_thread::get_id());
  }
}

TEST(EncodePipelineTest, SubmissionWaitsOnlyWhileDepthPicturesAreInFlight) {
  retire_gate gate;
  pipeline_t pipeline(2, [&gate](int &frame) {
    return gate.retire(frame);
  });
  // The next frame converts and submits while the previous picture encodes.
  ASSERT_TRUE(pipeline.acquire());
  pipeline.push(1);
  ASSERT_TRUE(gate.await_started(1));
  ASSERT_TRUE(pipeline.acquire());
  pipeline.push(2);
  EXPECT_EQ(pipeline.in_flight(), 2u);

  // A third waits for the oldest picture, not for both.
  auto third = std::async(std::launch::async, [&] {
    return pipeline.acquire();
  });
  EXPECT_EQ(third.wait_for(50ms), std::future_status::timeout);
  gate.release(1);
  ASSERT_EQ(third.wait_for(3s), std::future_status::ready);
  EXPECT_TRUE(third.get());
  EXPECT_EQ(gate.retired_frames(), (std::vector<int> {1}));
  EXPECT_EQ(pipeline.take_stats().pushed_behind, 1u);
  gate.open_all();
}

TEST(EncodePipelineTest, LaterPictureNeverRetiresBeforeAnEarlierOne) {
  retire_gate gate;
  pipeline_t pipeline(2, [&gate](int &frame) {
    return gate.retire(frame);
  });
  ASSERT_TRUE(pipeline.acquire());
  pipeline.push(1);
  ASSERT_TRUE(pipeline.acquire());
  pipeline.push(2);
  ASSERT_TRUE(gate.await_started(1));
  gate.release(2);  // The later picture "completes" first: it still waits for the earlier one.
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(gate.started_frames(), (std::vector<int> {1}));
  EXPECT_TRUE(gate.retired_frames().empty());
  gate.release(1);
  EXPECT_TRUE(pipeline.drain());
  EXPECT_EQ(gate.retired_frames(), (std::vector<int> {1, 2}));
}

TEST(EncodePipelineTest, FailedRetirementStopsRetiringWakesTheOwnerAndReleasesItsWaits) {
  retire_gate gate;
  gate.failing = {1};
  std::atomic<int> wakes {0};
  pipeline_t pipeline(2, [&gate](int &frame) {
    return gate.retire(frame);
  },
                      [&wakes] {
                        ++wakes;
                      });
  ASSERT_TRUE(pipeline.acquire());
  pipeline.push(1);
  ASSERT_TRUE(pipeline.acquire());
  pipeline.push(2);
  auto blocked = std::async(std::launch::async, [&] {
    return pipeline.acquire();
  });
  gate.release(1);
  ASSERT_EQ(blocked.wait_for(3s), std::future_status::ready);
  EXPECT_FALSE(blocked.get());
  EXPECT_TRUE(pipeline.failed());
  EXPECT_FALSE(pipeline.drain());  // Returns at once: the encoder's teardown drains frame 2.
  EXPECT_FALSE(pipeline.acquire());
  for (int i = 0; i < 300 && wakes.load() == 0; ++i) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_EQ(wakes.load(), 1);
  gate.release(2);
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(gate.retired_frames(), (std::vector<int> {1}));  // Nothing retires after a failure.
}

TEST(EncodePipelineTest, DestructionWithPicturesInFlightDeliversThemFirst) {
  retire_gate gate;
  auto pipeline = std::make_unique<pipeline_t>(2, [&gate](int &frame) {
    return gate.retire(frame);
  });
  ASSERT_TRUE(pipeline->acquire());
  pipeline->push(1);
  ASSERT_TRUE(pipeline->acquire());
  pipeline->push(2);
  ASSERT_TRUE(gate.await_started(1));
  auto destroyed = std::async(std::launch::async, [&] {
    pipeline.reset();
  });
  EXPECT_EQ(destroyed.wait_for(50ms), std::future_status::timeout);
  gate.release(1);
  gate.release(2);
  ASSERT_EQ(destroyed.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(gate.retired_frames(), (std::vector<int> {1, 2}));
}

TEST(EncodePipelineTest, DepthOneWaitsForEachPictureBeforeTheNext) {
  retire_gate gate;
  pipeline_t pipeline(1, [&gate](int &frame) {
    return gate.retire(frame);
  });
  ASSERT_TRUE(pipeline.acquire());
  pipeline.push(1);
  auto next = std::async(std::launch::async, [&] {
    return pipeline.acquire();
  });
  EXPECT_EQ(next.wait_for(50ms), std::future_status::timeout);
  gate.release(1);
  ASSERT_EQ(next.wait_for(3s), std::future_status::ready);
  EXPECT_TRUE(next.get());
  gate.open_all();
}

TEST(EncodePipelineTest, StartHookRunsOnTheRetiringThreadBeforeAnyRetirement) {
  std::promise<std::thread::id> started;
  auto started_future = started.get_future();
  std::thread::id retired_on;
  {
    pipeline_t pipeline(2, [&](int &) {
      retired_on = std::this_thread::get_id();
      return true;
    },
                        {},
                        [&] {
                          started.set_value(std::this_thread::get_id());
                        });
    ASSERT_TRUE(pipeline.acquire());
    pipeline.push(1);
    EXPECT_TRUE(pipeline.drain());
  }
  ASSERT_EQ(started_future.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(started_future.get(), retired_on);
  EXPECT_NE(retired_on, std::this_thread::get_id());
}
