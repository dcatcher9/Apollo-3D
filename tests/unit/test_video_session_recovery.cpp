/**
 * @file tests/unit/test_video_session_recovery.cpp
 * @brief Bounded failed encoder-session cleanup without a GPU or driver.
 */
#include "../tests_common.h"
#include "src/video_session_recovery.h"

#include <memory>
#include <stdexcept>
#include <thread>

namespace {
  using namespace std::chrono_literals;
  using video::detail::encoder_recovery_t;
  using state_t = encoder_recovery_t::state_t;
  constexpr encoder_recovery_t::clock::time_point start_time {};
}  // namespace

TEST(VideoSessionRecovery, NoCleanupIsImmediatelyReady) {
  encoder_recovery_t recovery;
  EXPECT_EQ(recovery.poll(start_time), state_t::ready);
  recovery.start({}, start_time);
  EXPECT_EQ(recovery.poll(start_time + 20s), state_t::ready);
}

TEST(VideoSessionRecovery, DelayedCleanupAllowsRepeatedNonblockingPollsThenRecovery) {
  std::promise<void> completion;
  encoder_recovery_t recovery;
  recovery.start(completion.get_future(), start_time);

  EXPECT_EQ(recovery.poll(start_time), state_t::pending);
  EXPECT_EQ(recovery.poll(start_time + 20ms), state_t::pending);
  EXPECT_EQ(recovery.poll(start_time + 9s), state_t::pending);
  completion.set_value();
  EXPECT_EQ(recovery.poll(start_time + 9s + 20ms), state_t::ready);
  EXPECT_EQ(recovery.poll(start_time + 20s), state_t::ready);
}

TEST(VideoSessionRecovery, PollingDoesNotExtendTheAbsoluteDeadline) {
  std::promise<void> completion;
  encoder_recovery_t recovery;
  recovery.start(completion.get_future(), start_time);

  for (auto elapsed = 0ms; elapsed < 10s; elapsed += 20ms) {
    EXPECT_EQ(recovery.poll(start_time + elapsed), state_t::pending);
  }
  EXPECT_EQ(recovery.poll(start_time + 10s), state_t::timed_out);
}

TEST(VideoSessionRecovery, CompletedCleanupAtTheDeadlineCanRecover) {
  std::promise<void> completion;
  encoder_recovery_t recovery;
  recovery.start(completion.get_future(), start_time);
  completion.set_value();

  EXPECT_EQ(recovery.poll(start_time + 10s), state_t::ready);
}

TEST(VideoSessionRecovery, LateCompletionCannotReverseAnObservedTimeout) {
  std::promise<void> completion;
  encoder_recovery_t recovery;
  recovery.start(completion.get_future(), start_time);
  EXPECT_EQ(recovery.poll(start_time + 10s), state_t::timed_out);
  completion.set_value();

  EXPECT_EQ(recovery.poll(start_time + 11s), state_t::timed_out);
}

TEST(VideoSessionRecovery, CleanupExceptionRemainsFailedAfterTheFutureIsConsumed) {
  std::packaged_task<void()> cleanup([]() {
    throw std::runtime_error("encoder cleanup failed");
  });
  encoder_recovery_t recovery;
  recovery.start(cleanup.get_future(), start_time);
  cleanup();

  EXPECT_EQ(recovery.poll(start_time), state_t::failed);
  EXPECT_EQ(recovery.poll(start_time + 20s), state_t::failed);
}

TEST(VideoSessionRecovery, MissingCompletionProducerFailsInsteadOfRecovering) {
  encoder_recovery_t recovery;
  {
    std::promise<void> abandoned_cleanup;
    recovery.start(abandoned_cleanup.get_future(), start_time);
  }

  EXPECT_EQ(recovery.poll(start_time), state_t::failed);
}

TEST(VideoSessionRecovery, ExplicitNewRecoveryStartsWithANewDeadline) {
  encoder_recovery_t recovery;
  std::promise<void> previous_completion;
  recovery.start(previous_completion.get_future(), start_time);
  EXPECT_EQ(recovery.poll(start_time + 10s), state_t::timed_out);

  std::promise<void> next_completion;
  recovery.start(next_completion.get_future(), start_time + 12s);
  EXPECT_EQ(recovery.poll(start_time + 21s), state_t::pending);
  next_completion.set_value();
  EXPECT_EQ(recovery.poll(start_time + 21s), state_t::ready);
}

TEST(VideoSessionRecovery, AbandoningPendingOrTimedOutRecoveryDoesNotJoinOrReleaseTheWorker) {
  for (const bool expire : {false, true}) {
    SCOPED_TRACE(expire ? "timed out" : "cancelled while pending");
    std::promise<void> release_worker;
    auto resources = std::make_shared<int>(1);
    std::weak_ptr<int> worker_resources = resources;
    std::packaged_task<void()> cleanup([resources, release = release_worker.get_future()]() mutable {
      release.wait();
    });
    resources.reset();
    auto recovery = std::make_unique<encoder_recovery_t>();
    recovery->start(cleanup.get_future(), start_time);
    std::thread worker(std::move(cleanup));

    EXPECT_EQ(recovery->poll(start_time + (expire ? 10s : 0s)), expire ? state_t::timed_out : state_t::pending);

    // Dispose on a separate thread so a regressed blocking destructor fails without hanging tests.
    std::promise<void> disposed;
    auto disposal_complete = disposed.get_future();
    std::thread dispose([recovery = std::move(recovery), disposed = std::move(disposed)]() mutable {
      recovery.reset();
      disposed.set_value();
    });
    const auto disposal_status = disposal_complete.wait_for(1s);
    const bool retained_while_unresolved = !worker_resources.expired();

    release_worker.set_value();
    dispose.join();
    worker.join();

    EXPECT_EQ(disposal_status, std::future_status::ready);
    EXPECT_TRUE(retained_while_unresolved);
    EXPECT_TRUE(worker_resources.expired());
  }
}
