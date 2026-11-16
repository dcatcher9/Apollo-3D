#include "src/video_capture_test.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {
  using namespace std::chrono_literals;
  using worker_t = video::detail::capture_worker_test_t;
  using context_t = video::detail::capture_worker_test_context_t;

  class manual_event_t {
  public:
    void set() {
      std::lock_guard lock(mutex_);
      set_ = true;
      condition_.notify_all();
    }

    bool wait(std::chrono::milliseconds timeout = 3s) {
      std::unique_lock lock(mutex_);
      return condition_.wait_for(lock, timeout, [&] {
        return set_;
      });
    }

  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool set_ = false;
  };

  struct callback_state_t {
    void release() {
      inspect.set();
      exit.set();
    }

    bool hold(manual_event_t &event) {
      const bool released = event.wait(10s);
      if (!released) {
        timed_out = true;
      }
      return released;
    }

    manual_event_t entered, inspect, inspected, cancelled, exit, finished;
    std::atomic<bool> timed_out {false}, callback_failed {false};
    std::atomic<unsigned> wake_calls {0};
    std::optional<video::config_t> observed_config;
    video::img_event_t observed_images;
    std::thread::id worker_thread;
    bool cancellation_seen = false;
    bool running_at_entry = false;
    bool images_running_at_entry = false;
    bool running_after_cancel = true;
    bool images_running_after_cancel = true;
    bool wake_alive_after_cancel = false;
  };

  void inspect_after_caller_mutation(void *opaque, const context_t &context) noexcept {
    auto &state = *static_cast<callback_state_t *>(opaque);
    try {
      state.worker_thread = std::this_thread::get_id();
      state.running_at_entry = context.running();
      state.images_running_at_entry = context.images->running();
      state.entered.set();
      if (state.hold(state.inspect)) {
        state.observed_config = context.config;
        state.observed_images = context.images;
        context.config.encode_wake->wake();
        state.inspected.set();
        state.hold(state.exit);
      }
    } catch (...) {
      state.callback_failed = true;
    }
    state.finished.set();
  }

  void exit_immediately(void *opaque, const context_t &context) noexcept {
    auto &state = *static_cast<callback_state_t *>(opaque);
    try {
      state.worker_thread = std::this_thread::get_id();
      state.running_at_entry = context.running();
      state.images_running_at_entry = context.images->running();
      state.entered.set();
    } catch (...) {
      state.callback_failed = true;
    }
    state.finished.set();
  }

  void hold_after_cancellation(void *opaque, const context_t &context) noexcept {
    auto &state = *static_cast<callback_state_t *>(opaque);
    try {
      state.worker_thread = std::this_thread::get_id();
      state.running_at_entry = context.running();
      state.images_running_at_entry = context.images->running();
      state.entered.set();
      state.cancellation_seen = context.wait_for_cancel(10s);
      state.running_after_cancel = context.running();
      state.images_running_after_cancel = context.images->running();
      state.wake_alive_after_cancel = static_cast<bool>(context.config.encode_wake);
      state.cancelled.set();
      state.hold(state.exit);
    } catch (...) {
      state.callback_failed = true;
    }
    state.finished.set();
  }

  struct release_guard_t {
    callback_state_t &state;

    ~release_guard_t() {
      state.release();
    }
  };

  // A regressed synchronous stop/destructor must not prevent the test from releasing its worker.
  // On every failure exit release the callback before joining this caller.
  class scoped_operation_t {
  public:
    template<class Operation>
    scoped_operation_t(callback_state_t &state, Operation operation):
        state_(state) {
      std::packaged_task<void()> task(std::move(operation));
      completion_ = task.get_future();
      caller_ = std::thread(std::move(task));
    }

    ~scoped_operation_t() {
      state_.release();
      if (caller_.joinable()) {
        caller_.join();
      }
    }

    bool returned(std::chrono::milliseconds timeout = 3s) {
      return completion_.wait_for(timeout) == std::future_status::ready;
    }

    void finish() {
      caller_.join();
      completion_.get();
    }

  private:
    callback_state_t &state_;
    std::future<void> completion_;
    std::thread caller_;
  };

  void expect_callback_completed(callback_state_t &state) {
    EXPECT_TRUE(state.finished.wait(0ms));
    EXPECT_FALSE(state.timed_out.load());
    EXPECT_FALSE(state.callback_failed.load());
    EXPECT_NE(state.worker_thread, std::this_thread::get_id());
  }
}  // namespace

TEST(VideoCaptureLifecycleTest, CopiesConfigAndRetainsExactImagesAndWakeDespiteCallerMutation) {
  callback_state_t state;
  video::config_t config {1920, 1080, 60, 35'000, 2, 3, 4, 1, 0, 59'940};
  config.framerateX100 = 5994;
  config.sbs_mode = video::SBS_AI;
  config.sbs_config = {false, 1.25, true, 7680};
  config.encode_wake = std::make_shared<video::encode_wake_t>();
  config.encode_wake->install([&] {
    ++state.wake_calls;
  });
  const auto expected = config;
  auto images = std::make_shared<video::img_event_t::element_type>();
  const auto expected_images = images;
  worker_t worker(config, images, inspect_after_caller_mutation, &state);
  release_guard_t release {state};
  ASSERT_TRUE(state.entered.wait());

  config = video::config_t {640, 480, 30, 5'000, 1, 0, 0, 0, 0, 30'000};
  config.encode_wake = std::make_shared<video::encode_wake_t>();
  images = std::make_shared<video::img_event_t::element_type>();
  state.inspect.set();
  ASSERT_TRUE(state.inspected.wait());
  ASSERT_TRUE(state.observed_config.has_value());
  const auto &actual = *state.observed_config;
  EXPECT_EQ(actual.width, expected.width);
  EXPECT_EQ(actual.height, expected.height);
  EXPECT_EQ(actual.framerate, expected.framerate);
  EXPECT_EQ(actual.framerateX100, expected.framerateX100);
  EXPECT_EQ(actual.encodingFramerate, expected.encodingFramerate);
  EXPECT_EQ(actual.bitrate, expected.bitrate);
  EXPECT_EQ(actual.slicesPerFrame, expected.slicesPerFrame);
  EXPECT_EQ(actual.numRefFrames, expected.numRefFrames);
  EXPECT_EQ(actual.encoderCscMode, expected.encoderCscMode);
  EXPECT_EQ(actual.videoFormat, expected.videoFormat);
  EXPECT_EQ(actual.dynamicRange, expected.dynamicRange);
  EXPECT_EQ(actual.sbs_mode, expected.sbs_mode);
  EXPECT_EQ(actual.sbs_config.reshade, expected.sbs_config.reshade);
  EXPECT_EQ(actual.sbs_config.pop_strength, expected.sbs_config.pop_strength);
  EXPECT_EQ(actual.sbs_config.foreground_subtitles, expected.sbs_config.foreground_subtitles);
  EXPECT_EQ(actual.sbs_config.max_encode_width, expected.sbs_config.max_encode_width);
  EXPECT_EQ(actual.encode_wake, expected.encode_wake);
  EXPECT_EQ(state.observed_images, expected_images);
  EXPECT_EQ(state.wake_calls.load(), 1u);
  EXPECT_TRUE(state.running_at_entry);
  EXPECT_TRUE(state.images_running_at_entry);
  state.exit.set();
  worker.stop();
  EXPECT_FALSE(expected_images->running());
  EXPECT_TRUE(images->running());
  EXPECT_FALSE(worker.running());
  expect_callback_completed(state);
}

TEST(VideoCaptureLifecycleTest, EarlyWorkerExitStopsItsImagesAndOwner) {
  callback_state_t state;
  auto images = std::make_shared<video::img_event_t::element_type>();
  worker_t worker({}, images, exit_immediately, &state);
  release_guard_t release {state};
  ASSERT_TRUE(state.entered.wait());
  // This empty image event wakes when the actual worker-exit guard stops it, without caller stop.
  EXPECT_FALSE(images->pop(3s));
  EXPECT_FALSE(images->running());
  EXPECT_FALSE(worker.running());
  EXPECT_TRUE(state.running_at_entry);
  EXPECT_TRUE(state.images_running_at_entry);
  worker.stop();
  worker.stop();
  expect_callback_completed(state);
}

TEST(VideoCaptureLifecycleTest, StopClosesImagesBeforeCancellationAndWaitsForWorkerCompletion) {
  callback_state_t state;
  auto images = std::make_shared<video::img_event_t::element_type>();
  worker_t worker({}, images, hold_after_cancellation, &state);
  release_guard_t release {state};
  ASSERT_TRUE(state.entered.wait());
  scoped_operation_t stop(state, [&] {
    worker.stop();
  });
  ASSERT_TRUE(state.cancelled.wait());
  EXPECT_TRUE(state.cancellation_seen);
  EXPECT_FALSE(state.images_running_after_cancel);
  EXPECT_FALSE(state.running_after_cancel);
  EXPECT_FALSE(images->running());
  EXPECT_FALSE(stop.returned(0ms));
  state.exit.set();
  ASSERT_TRUE(stop.returned());
  stop.finish();
  EXPECT_FALSE(worker.running());
  worker.stop();
  expect_callback_completed(state);
}

TEST(VideoCaptureLifecycleTest, DestructorDuringUnwindJoinsBeforeReleasingWorkerState) {
  callback_state_t state;
  video::config_t config {};
  config.encode_wake = std::make_shared<video::encode_wake_t>();
  const std::weak_ptr<video::encode_wake_t> wake = config.encode_wake;
  auto images = std::make_shared<video::img_event_t::element_type>();
  auto worker = std::make_unique<worker_t>(config, images, hold_after_cancellation, &state);
  release_guard_t release {state};
  config.encode_wake.reset();
  ASSERT_TRUE(state.entered.wait());
  scoped_operation_t unwind(state, [owned_worker = std::move(worker)]() mutable {
    try {
      auto local = std::move(owned_worker);
      throw std::runtime_error("exercise capture-owner scope unwind");
    } catch (const std::runtime_error &) {
    }
  });
  ASSERT_TRUE(state.cancelled.wait());
  EXPECT_TRUE(state.cancellation_seen);
  EXPECT_FALSE(state.images_running_after_cancel);
  EXPECT_TRUE(state.wake_alive_after_cancel);
  EXPECT_FALSE(wake.expired());
  EXPECT_FALSE(unwind.returned(0ms));
  state.exit.set();
  ASSERT_TRUE(unwind.returned());
  unwind.finish();
  EXPECT_TRUE(wake.expired());
  expect_callback_completed(state);
}

TEST(VideoCaptureLifecycleTest, IndependentOwnersKeepCancellationAndLiveModesSeparate) {
  callback_state_t first_state, second_state;
  auto first_images = std::make_shared<video::img_event_t::element_type>();
  auto second_images = std::make_shared<video::img_event_t::element_type>();
  worker_t first({}, first_images, hold_after_cancellation, &first_state);
  release_guard_t release_first {first_state};
  worker_t second({}, second_images, hold_after_cancellation, &second_state);
  release_guard_t release_second {second_state};
  ASSERT_TRUE(first_state.entered.wait());
  ASSERT_TRUE(second_state.entered.wait());
  EXPECT_FALSE(first.live_video_mode().has_value());
  EXPECT_FALSE(second.live_video_mode().has_value());
  video::video_mode_change_t mode {2560, 1440, 120, 11'988, 119'880, 60'000, 7, 9};
  first.set_live_video_mode(mode);
  ASSERT_TRUE(first.live_video_mode().has_value());
  EXPECT_EQ(first.live_video_mode()->width, mode.width);
  EXPECT_EQ(first.live_video_mode()->framerateX100, mode.framerateX100);
  EXPECT_EQ(first.live_video_mode()->transaction_id, mode.transaction_id);
  EXPECT_FALSE(second.live_video_mode().has_value());
  scoped_operation_t stop_first(first_state, [&] {
    first.stop();
  });
  ASSERT_TRUE(first_state.cancelled.wait());
  EXPECT_TRUE(first_state.cancellation_seen);
  EXPECT_FALSE(first_images->running());
  EXPECT_TRUE(second_images->running());
  EXPECT_TRUE(second.running());
  EXPECT_FALSE(second_state.cancelled.wait(0ms));
  first_state.exit.set();
  ASSERT_TRUE(stop_first.returned());
  stop_first.finish();
  EXPECT_TRUE(second.running());
  scoped_operation_t stop_second(second_state, [&] {
    second.stop();
  });
  ASSERT_TRUE(second_state.cancelled.wait());
  EXPECT_TRUE(second_state.cancellation_seen);
  EXPECT_FALSE(second_state.images_running_after_cancel);
  second_state.exit.set();
  ASSERT_TRUE(stop_second.returned());
  stop_second.finish();
  expect_callback_completed(first_state);
  expect_callback_completed(second_state);
}

namespace {
  struct admission_state_t {
    callback_state_t worker;
    std::atomic<unsigned> admitted_calls {0};
  };

  void hold_before_admission(void *opaque, const context_t &context) noexcept {
    auto &state = *static_cast<admission_state_t *>(opaque);
    hold_after_cancellation(&state.worker, context);
  }

  void count_admitted_body(void *opaque, const context_t &) noexcept {
    ++static_cast<admission_state_t *>(opaque)->admitted_calls;
  }
}  // namespace

TEST(VideoCaptureLifecycleTest, StopBeforeAdmissionSkipsWorkerBody) {
  admission_state_t state;
  auto images = std::make_shared<video::img_event_t::element_type>();
  worker_t worker({}, images, count_admitted_body, &state, hold_before_admission);
  release_guard_t release {state.worker};
  ASSERT_TRUE(state.worker.entered.wait());
  scoped_operation_t stop(state.worker, [&] {
    worker.stop();
  });
  ASSERT_TRUE(state.worker.cancelled.wait());
  EXPECT_TRUE(state.worker.cancellation_seen);
  EXPECT_FALSE(state.worker.images_running_after_cancel);
  EXPECT_FALSE(state.worker.running_after_cancel);
  EXPECT_FALSE(images->running());
  EXPECT_FALSE(stop.returned(0ms));
  state.worker.exit.set();
  ASSERT_TRUE(stop.returned());
  stop.finish();
  EXPECT_EQ(state.admitted_calls.load(), 0u);
  EXPECT_FALSE(worker.running());
  expect_callback_completed(state.worker);
}
