/**
 * @file tests/unit/platform/test_sbs_debug_dump_async.cpp
 * @brief Behavioral tests for Dump 3D publication lifetime and request ordering.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <atomic>
  #include <chrono>
  #include <future>
  #include <memory>

  #include <src/platform/windows/sbs_debug_dump_async.h>

namespace {
  using namespace std::chrono_literals;
  namespace dump_detail = platf::sbs_debug::detail;

  TEST(SbsDebugDumpAsyncTest, ReleasingSessionDoesNotWaitForPublication) {
    auto state = dump_detail::publication_state::create();
    std::weak_ptr<dump_detail::publication_state> weak_state = state;

    std::promise<void> started_promise;
    auto started = started_promise.get_future();
    std::promise<void> release_promise;
    auto release = release_promise.get_future().share();
    std::promise<void> finished_promise;
    auto finished = finished_promise.get_future();

    ASSERT_TRUE(state->enqueue([&started_promise, release, &finished_promise] {
      started_promise.set_value();
      release.wait();
      finished_promise.set_value();
    }));
    ASSERT_EQ(started.wait_for(2s), std::future_status::ready);

    const auto begin = std::chrono::steady_clock::now();
    state.reset();
    const auto release_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - begin
    ).count();
    EXPECT_LT(release_duration_ms, 250);
    EXPECT_FALSE(weak_state.expired());

    release_promise.set_value();
    ASSERT_EQ(finished.wait_for(2s), std::future_status::ready);
  }

  TEST(SbsDebugDumpAsyncTest, SingleFlightRecoversAfterCompletionAndException) {
    auto state = dump_detail::publication_state::create();
    std::promise<void> started_promise;
    auto started = started_promise.get_future();
    std::promise<void> release_promise;
    auto release = release_promise.get_future().share();

    ASSERT_TRUE(state->enqueue([&started_promise, release] {
      started_promise.set_value();
      release.wait();
    }));
    ASSERT_EQ(started.wait_for(2s), std::future_status::ready);
    EXPECT_TRUE(state->busy());
    EXPECT_FALSE(state->enqueue([] {}));

    release_promise.set_value();
    ASSERT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(state->busy());

    ASSERT_TRUE(state->enqueue([] { throw 7; }));
    EXPECT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(state->busy());
  }

  TEST(SbsDebugDumpAsyncTest, CancellationEpochRejectsAnOlderFailure) {
    auto state = dump_detail::publication_state::create();
    auto button = std::make_shared<std::atomic<bool>>(false);
    const auto old_token = state->allow_retries_and_token();

    std::promise<void> started_promise;
    auto started = started_promise.get_future();
    std::promise<void> release_promise;
    auto release = release_promise.get_future().share();
    std::promise<bool> rearmed_promise;
    auto rearmed = rearmed_promise.get_future();
    ASSERT_TRUE(state->enqueue(
      [state, button, old_token, &started_promise, release, &rearmed_promise] {
        started_promise.set_value();
        release.wait();
        rearmed_promise.set_value(
          state->record_publication_failure(old_token, true, true, button)
        );
      }
    ));
    ASSERT_EQ(started.wait_for(2s), std::future_status::ready);

    state->cancel_retries(button);
    const auto new_token = state->allow_retries_and_token();
    EXPECT_NE(new_token, old_token);
    release_promise.set_value();

    ASSERT_EQ(rearmed.wait_for(2s), std::future_status::ready);
    EXPECT_FALSE(rearmed.get());
    ASSERT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(button->load());
    EXPECT_FALSE(state->take_publication_failed());
    EXPECT_FALSE(state->take_file_retry_pending());
  }

  TEST(SbsDebugDumpAsyncTest, CurrentFailureRearmsExactlyOnceAndSuccessDoesNot) {
    auto state = dump_detail::publication_state::create();
    auto button = std::make_shared<std::atomic<bool>>(false);
    const auto token = state->allow_retries_and_token();

    ASSERT_TRUE(state->record_publication_failure(token, true, true, button));
    EXPECT_TRUE(button->exchange(false));
    EXPECT_TRUE(state->take_publication_failed());
    EXPECT_FALSE(state->take_publication_failed());
    EXPECT_TRUE(state->take_file_retry_pending());
    EXPECT_FALSE(state->take_file_retry_pending());

    std::promise<void> success_promise;
    auto success = success_promise.get_future();
    ASSERT_TRUE(state->enqueue([&success_promise] { success_promise.set_value(); }));
    ASSERT_EQ(success.wait_for(2s), std::future_status::ready);
    ASSERT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(button->load());
    EXPECT_FALSE(state->take_publication_failed());
    EXPECT_FALSE(state->take_file_retry_pending());
  }

  TEST(SbsDebugDumpAsyncTest, ButtonGuardPreservesALaterClick) {
    auto button = std::make_shared<std::atomic<bool>>(true);
    {
      dump_detail::button_request_guard request(button);
      ASSERT_TRUE(request.consumed());
      EXPECT_FALSE(button->load());
      button->store(true);
      request.commit();
    }
    EXPECT_TRUE(button->load());

    button->store(true);
    {
      dump_detail::button_request_guard request(button);
      ASSERT_TRUE(request.consumed());
      EXPECT_FALSE(button->load());
      // An uncommitted validation/capture/enqueue failure restores the consumed click.
    }
    EXPECT_TRUE(button->load());
  }
}  // namespace
#endif
