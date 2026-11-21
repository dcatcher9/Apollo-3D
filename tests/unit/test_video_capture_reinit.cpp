/** Full original capture-worker reinitialization with inert, per-instance displays. */
#include "src/capture_display_retirement.h"
#include "src/video_capture_test.h"

#ifdef _WIN32
  #include "src/platform/windows/display.h"
  #include "src/tracked_async_worker.h"

  #include <array>
  #include <atomic>
  #include <chrono>
  #include <condition_variable>
  #include <functional>
  #include <future>
  #include <gtest/gtest.h>
  #include <limits>
  #include <memory>
  #include <mutex>
  #include <string>
  #include <thread>
  #include <utility>
  #include <vector>

namespace {
  using namespace std::chrono_literals;
  using worker_t = video::detail::capture_reinit_test_t;
  using native_owner_t = platf::dxgi::native_capture_retirement_test_t;

  class manual_event_t {
  public:
    void set() {
      std::lock_guard lock(mutex_);
      set_ = true;
      changed_.notify_all();
    }

    bool wait(std::chrono::milliseconds timeout = 3s) {
      std::unique_lock lock(mutex_);
      return changed_.wait_for(lock, timeout, [&] {
        return set_;
      });
    }

  private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool set_ = false;
  };

  struct plain_image_t final: platf::img_t {
    explicit plain_image_t(unsigned generation):
        generation(generation) {}

    unsigned generation;
  };

  struct generation_state_t {
    manual_event_t entered, emit, emitted, finish, destroyed;
    bool emit_image = false;
    platf::capture_e result = platf::capture_e::reinit;
    std::weak_ptr<platf::img_t> image;
    std::thread::id capture_thread, destruction_thread;
    std::uint64_t final_claim = 0;
  };

  class scripted_display_t;

  struct script_state_t {
    explicit script_state_t(unsigned count):
        count(count) {
      generations[0].emit_image = true;
      generations[count - 1].result = platf::capture_e::ok;
    }

    void release_all() {
      topology_release.set();
      for (auto &generation : generations) {
        generation.emit.set();
        generation.finish.set();
      }
    }

    void record(const std::string &event) {
      std::lock_guard lock(mutex);
      order.push_back(event);
    }

    bool wait_for_release_waits(unsigned count) {
      std::unique_lock lock(mutex);
      return changed.wait_for(lock, 3s, [&] {
        return release_waits >= count;
      });
    }

    unsigned waits() {
      std::lock_guard lock(mutex);
      return release_waits;
    }

    unsigned opens() {
      std::lock_guard lock(mutex);
      return offered_configs.size();
    }

    video::config_t offered(unsigned index) {
      std::lock_guard lock(mutex);
      return offered_configs.at(index);
    }

    std::size_t position(const std::string &event) {
      std::lock_guard lock(mutex);
      for (std::size_t index = 0; index < order.size(); ++index) {
        if (order[index] == event) {
          return index;
        }
      }
      return std::numeric_limits<std::size_t>::max();
    }

    unsigned count;
    std::array<generation_state_t, 3> generations;
    std::array<std::weak_ptr<scripted_display_t>, 3> displays;
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<video::config_t> offered_configs;
    std::vector<std::string> order;
    unsigned release_waits = 0;
    manual_event_t topology_entered, topology_release;
    bool hold_topology = false;
    std::function<platf::capture_e(const platf::display_t::pull_free_image_cb_t &)> capture_images;
    std::atomic<bool> timed_out {false}, unexpected_operation {false};
  };

  class scripted_display_t final: public platf::dxgi::display_vram_t {
  public:
    scripted_display_t(std::shared_ptr<script_state_t> state, unsigned generation, platf::capture_backend_e backend):
        state_(std::move(state)),
        generation_(generation),
        backend_(backend) {}

    ~scripted_display_t() override {
      auto &generation = state_->generations[generation_];
      generation.destruction_thread = std::this_thread::get_id();
      generation.final_claim = external_pixels_owner.load(std::memory_order_acquire);
      state_->record("destroy-" + std::to_string(generation_));
      generation.destroyed.set();
    }

    platf::capture_e capture(const push_captured_image_cb_t &push, const pull_free_image_cb_t &pull, bool *) override {
      auto &generation = state_->generations[generation_];
      generation.capture_thread = std::this_thread::get_id();
      generation.entered.set();
      if (!hold(generation.emit)) {
        return platf::capture_e::error;
      }
      if (state_->capture_images) {
        return state_->capture_images(pull);
      }
      if (generation.emit_image) {
        std::shared_ptr<platf::img_t> image;
        if (!pull(image)) {
          return platf::capture_e::ok;
        }
        generation.image = image;
        if (!push(std::move(image), true)) {
          return platf::capture_e::ok;
        }
      }
      generation.emitted.set();
      if (!hold(generation.finish)) {
        return platf::capture_e::error;
      }
      return generation.result;
    }

    platf::capture_backend_e capture_backend() const noexcept override {
      return backend_;
    }

    std::shared_ptr<platf::img_t> alloc_img() override {
      return std::make_shared<plain_image_t>(generation_);
    }

    int dummy_img(platf::img_t *) override {
      state_->unexpected_operation = true;
      return -1;
    }

    std::vector<DXGI_FORMAT> get_supported_capture_formats() override {
      return {};
    }

  protected:
    platf::capture_e snapshot(const pull_free_image_cb_t &, std::shared_ptr<platf::img_t> &, std::chrono::milliseconds, bool) override {
      state_->unexpected_operation = true;
      return platf::capture_e::error;
    }

    platf::capture_e release_snapshot() override {
      state_->unexpected_operation = true;
      return platf::capture_e::error;
    }

  private:
    bool hold(manual_event_t &gate) {
      const bool ready = gate.wait(10s);
      if (!ready) {
        state_->timed_out = true;
      }
      return ready;
    }

    std::shared_ptr<script_state_t> state_;
    unsigned generation_;
    platf::capture_backend_e backend_;
  };

  struct fixture_t {
    explicit fixture_t(unsigned count = 2):
        state(std::make_shared<script_state_t>(count)),
        images(std::make_shared<video::img_event_t::element_type>()) {}

    ~fixture_t() {
      state->release_all();
      if (worker) {
        worker->stop();
      }
      EXPECT_FALSE(state->timed_out.load());
      EXPECT_FALSE(state->unexpected_operation.load());
    }

    void start(video::config_t config = {}) {
      video::detail::capture_reinit_test_hooks_t hooks;
      hooks.preferred_display_name = [] {
        return "script-display";
      };
      hooks.set_display_name = [state = state](const std::string &name) {
        if (name != "script-display") {
          state->unexpected_operation = true;
        }
      };
      hooks.refresh_displays = [state = state](std::vector<std::string> &names, int &index, const std::string &) {
        state->record("enumerate");
        names = {"script-display"};
        index = 0;
      };
      hooks.open_display = [state = state](const std::string &name, const video::config_t &config, platf::capture_backend_e backend, std::weak_ptr<platf::capture_display_retirement_t> retirement) -> std::shared_ptr<platf::display_t> {
        unsigned generation;
        {
          std::lock_guard lock(state->mutex);
          generation = static_cast<unsigned>(state->offered_configs.size());
          if (generation >= state->count || name != "script-display") {
            state->unexpected_operation = true;
            return {};
          }
          state->offered_configs.push_back(config);
          state->order.push_back("open-" + std::to_string(generation));
        }
        auto display = platf::make_unarmed_capture_display<scripted_display_t>(state, generation, backend);
        if (!platf::arm_capture_display_retirement(display, std::move(retirement))) {
          state->unexpected_operation = true;
          return {};
        }
        // The fixture never retains a strong display outside an explicit test borrower.
        state->displays[generation] = display;
        return display;
      };
      hooks.adjust_thread_priority = [] {
      };
      hooks.display_change_epoch = [] {
        return std::uint64_t {42};
      };
      hooks.reconcile_topology = [state = state] {
        state->record("topology-" + std::to_string(state->opens() - 1));
        state->topology_entered.set();
        if (state->hold_topology && !state->topology_release.wait(10s)) {
          state->timed_out = true;
        }
        return true;
      };
      hooks.waited_for_display_release = [state = state] {
        std::lock_guard lock(state->mutex);
        ++state->release_waits;
        state->changed.notify_all();
      };
      worker = std::make_unique<worker_t>(std::move(config), images, std::move(hooks));
    }

    void emit(unsigned generation) {
      state->generations[generation].emit.set();
    }

    void finish(unsigned generation) {
      state->generations[generation].finish.set();
    }

    std::shared_ptr<script_state_t> state;
    video::img_event_t images;
    std::unique_ptr<worker_t> worker;
  };

  class scoped_operation_t {
  public:
    template<class Operation, class Cleanup>
    scoped_operation_t(Operation operation, Cleanup cleanup):
        cleanup_(std::move(cleanup)) {
      std::packaged_task<void()> task(std::move(operation));
      returned_ = task.get_future();
      thread_ = std::thread(std::move(task));
    }

    ~scoped_operation_t() {
      cleanup_();
      if (thread_.joinable()) {
        thread_.join();
      }
    }

    bool returned(std::chrono::milliseconds timeout = 3s) {
      return returned_.wait_for(timeout) == std::future_status::ready;
    }

    void finish() {
      thread_.join();
      returned_.get();
    }

  private:
    std::function<void()> cleanup_;
    std::future<void> returned_;
    std::thread thread_;
  };

  void expect_reopen_order(fixture_t &fixture, unsigned old_generation, unsigned new_generation) {
    const auto destroyed = fixture.state->position("destroy-" + std::to_string(old_generation));
    const auto topology = fixture.state->position("topology-" + std::to_string(old_generation));
    const auto opened = fixture.state->position("open-" + std::to_string(new_generation));
    ASSERT_NE(destroyed, std::numeric_limits<std::size_t>::max());
    ASSERT_NE(topology, std::numeric_limits<std::size_t>::max());
    ASSERT_NE(opened, std::numeric_limits<std::size_t>::max());
    EXPECT_LT(destroyed, topology);
    EXPECT_LT(topology, opened);
    EXPECT_EQ(fixture.state->generations[old_generation].destruction_thread, fixture.state->generations[old_generation].capture_thread);
  }
}  // namespace

TEST(VideoCaptureReinitTest, DrainsTheOldImageAndFoldsTheNewestModeAfterAllBorrowersRelease) {
  std::atomic<unsigned> wakes {0};
  fixture_t fixture;
  video::config_t config {};
  config.encode_wake = std::make_shared<video::encode_wake_t>();
  config.encode_wake->install([&] {
    ++wakes;
  });
  fixture.start(config);
  ASSERT_TRUE(fixture.state->generations[0].entered.wait());
  auto first_borrower = fixture.worker->current_display();
  auto second_borrower = first_borrower;
  ASSERT_TRUE(first_borrower);
  fixture.emit(0);
  ASSERT_TRUE(fixture.state->generations[0].emitted.wait());
  EXPECT_FALSE(fixture.state->generations[0].image.expired());
  fixture.finish(0);
  ASSERT_TRUE(fixture.state->wait_for_release_waits(1));
  EXPECT_TRUE(fixture.worker->reinit_pending());
  EXPECT_GE(wakes.load(), 1u);
  EXPECT_TRUE(fixture.state->generations[0].image.expired());
  EXPECT_FALSE(fixture.images->peek());
  EXPECT_EQ(fixture.state->opens(), 1u);
  first_borrower.reset();
  const auto waits = fixture.state->waits();
  ASSERT_TRUE(fixture.state->wait_for_release_waits(waits + 1));
  EXPECT_EQ(fixture.state->opens(), 1u);

  fixture.worker->set_live_video_mode(video::video_mode_change_t {1600, 900, 60, 6000, 60000, 30000, 1, 1});
  const video::video_mode_change_t newest {2560, 1440, 120, 11988, 119880, 65000, 2, 2};
  fixture.worker->set_live_video_mode(newest);
  second_borrower.reset();
  ASSERT_TRUE(fixture.state->generations[1].entered.wait());
  EXPECT_FALSE(fixture.worker->reinit_pending());
  expect_reopen_order(fixture, 0, 1);
  const auto reopened = fixture.state->offered(1);
  EXPECT_EQ(reopened.width, newest.width);
  EXPECT_EQ(reopened.height, newest.height);
  EXPECT_EQ(reopened.framerate, newest.framerate);
  EXPECT_EQ(reopened.framerateX100, newest.framerateX100);
  EXPECT_EQ(reopened.encodingFramerate, newest.encodingFramerate);
  EXPECT_EQ(reopened.bitrate, newest.bitrate);
}

TEST(VideoCaptureReinitTest, OldNativeCompletionCannotReleaseTheNextDisplaysBorrower) {
  video::detail::tracked_async_worker_pool_t workers;
  fixture_t fixture(3);
  fixture.state->generations[1].emit_image = true;
  fixture.start();
  ASSERT_TRUE(fixture.state->generations[0].entered.wait());
  auto old_display = std::static_pointer_cast<platf::dxgi::display_base_t>(fixture.worker->current_display());
  ASSERT_TRUE(old_display);
  native_owner_t old_native(old_display, workers);
  fixture.emit(0);
  ASSERT_TRUE(fixture.state->generations[0].emitted.wait());
  fixture.finish(0);
  ASSERT_TRUE(fixture.state->wait_for_release_waits(1));
  old_native.retire();
  ASSERT_TRUE(old_native.wait_after_capture_release(3s));
  EXPECT_FALSE(old_native.wait_retired(0ms));
  old_display.reset();
  ASSERT_TRUE(fixture.state->generations[1].entered.wait());
  expect_reopen_order(fixture, 0, 1);
  auto next_borrower = fixture.worker->current_display();
  ASSERT_TRUE(next_borrower);
  fixture.emit(1);
  ASSERT_TRUE(fixture.state->generations[1].emitted.wait());
  const auto waits = fixture.state->waits();
  fixture.finish(1);
  ASSERT_TRUE(fixture.state->wait_for_release_waits(waits + 1));
  old_native.release_native_teardown();
  ASSERT_TRUE(old_native.wait_retired(3s));
  const auto after_old_completion = fixture.state->waits();
  ASSERT_TRUE(fixture.state->wait_for_release_waits(after_old_completion + 1));
  EXPECT_EQ(fixture.state->opens(), 2u);
  EXPECT_TRUE(fixture.worker->reinit_pending());
  EXPECT_TRUE(fixture.state->generations[1].image.expired());
  next_borrower.reset();
  ASSERT_TRUE(fixture.state->generations[2].entered.wait());
  expect_reopen_order(fixture, 1, 2);
  EXPECT_FALSE(old_native.gate_timed_out());
}

TEST(VideoCaptureReinitTest, ActualForeignClaimAndDisplayOwnerSurviveTheOlderNativeRelease) {
  video::detail::tracked_async_worker_pool_t workers;
  fixture_t fixture;
  fixture.start();
  ASSERT_TRUE(fixture.state->generations[0].entered.wait());
  auto display = std::static_pointer_cast<platf::dxgi::display_base_t>(fixture.worker->current_display());
  ASSERT_TRUE(display);
  native_owner_t older(display, workers);
  const auto old_token = display->external_pixels_owner.load();
  native_owner_t foreign(display, workers);
  const auto foreign_token = display->external_pixels_owner.load();
  ASSERT_NE(old_token, 0u);
  ASSERT_NE(foreign_token, 0u);
  ASSERT_NE(old_token, foreign_token);
  fixture.emit(0);
  ASSERT_TRUE(fixture.state->generations[0].emitted.wait());
  fixture.finish(0);
  ASSERT_TRUE(fixture.state->wait_for_release_waits(1));
  older.retire();
  ASSERT_TRUE(older.wait_after_capture_release(3s));
  EXPECT_EQ(display->external_pixels_owner.load(), foreign_token);
  display.reset();
  const auto waits = fixture.state->waits();
  ASSERT_TRUE(fixture.state->wait_for_release_waits(waits + 1));
  EXPECT_EQ(fixture.state->opens(), 1u);
  foreign.retire();
  ASSERT_TRUE(foreign.wait_after_capture_release(3s));
  ASSERT_TRUE(fixture.state->generations[1].entered.wait());
  EXPECT_FALSE(older.wait_retired(0ms));
  EXPECT_FALSE(foreign.wait_retired(0ms));
  EXPECT_EQ(fixture.state->generations[0].final_claim, 0u);
  expect_reopen_order(fixture, 0, 1);
  older.release_native_teardown();
  foreign.release_native_teardown();
  EXPECT_TRUE(older.wait_retired(3s));
  EXPECT_TRUE(foreign.wait_retired(3s));
  EXPECT_FALSE(older.gate_timed_out());
  EXPECT_FALSE(foreign.gate_timed_out());
}

TEST(VideoCaptureReinitTest, StopJoinsWithoutWaitingForAHeldDisplayAndItsLateDeletionIsSafe) {
  fixture_t fixture;
  fixture.start();
  ASSERT_TRUE(fixture.state->generations[0].entered.wait());
  auto borrower = fixture.worker->current_display();
  ASSERT_TRUE(borrower);
  fixture.emit(0);
  ASSERT_TRUE(fixture.state->generations[0].emitted.wait());
  fixture.finish(0);
  ASSERT_TRUE(fixture.state->wait_for_release_waits(1));
  scoped_operation_t stop([&] {
    fixture.worker->stop();
  },
                          [&] {
                            borrower.reset();
                            fixture.state->release_all();
                          });
  ASSERT_TRUE(stop.returned());
  stop.finish();
  EXPECT_FALSE(fixture.images->running());
  EXPECT_FALSE(fixture.worker->running());
  EXPECT_FALSE(fixture.state->displays[0].expired());
  EXPECT_EQ(fixture.state->opens(), 1u);
  borrower.reset();
  ASSERT_TRUE(fixture.state->generations[0].destroyed.wait());
  EXPECT_EQ(fixture.state->generations[0].destruction_thread, std::this_thread::get_id());
  EXPECT_NE(fixture.state->generations[0].destruction_thread, fixture.state->generations[0].capture_thread);
}

TEST(VideoCaptureReinitTest, ExhaustedImagePoolReusesAReleasedSlotAndCancelsOnStop) {
  std::array<std::shared_ptr<platf::img_t>, 12> borrowed;
  manual_event_t exhausted, exhausted_again;
  bool reacquired = false, cancelled = false;
  platf::img_t *recycled = nullptr;
  fixture_t fixture {1};
  fixture.state->capture_images = [&](const auto &pull) {
    for (auto &image : borrowed) {
      if (!pull(image)) {
        return platf::capture_e::interrupted;
      }
    }
    exhausted.set();
    std::shared_ptr<platf::img_t> next;
    reacquired = pull(next);
    recycled = next.get();
    exhausted_again.set();
    if (reacquired) {
      std::shared_ptr<platf::img_t> extra;
      cancelled = !pull(extra);
    }
    return platf::capture_e::interrupted;
  };
  fixture.start();
  ASSERT_TRUE(fixture.state->generations[0].entered.wait());
  fixture.emit(0);
  ASSERT_TRUE(exhausted.wait());
  const auto *released = borrowed[5].get();
  borrowed[5].reset();
  ASSERT_TRUE(exhausted_again.wait());
  EXPECT_TRUE(reacquired);
  EXPECT_EQ(recycled, released);

  scoped_operation_t stop([&] { fixture.worker->stop(); }, [&] { borrowed[6].reset(); });
  ASSERT_TRUE(stop.returned());
  stop.finish();
  EXPECT_TRUE(cancelled);
  EXPECT_FALSE(fixture.images->running());
}

TEST(VideoCaptureReinitTest, NoBorrowerPathDiscardsTheOldImageBeforeTheNextCapture) {
  fixture_t fixture;
  fixture.start();
  ASSERT_TRUE(fixture.state->generations[0].entered.wait());
  // Deliberately never acquire display A: real Windows images own COM resources, not display_t.
  fixture.emit(0);
  ASSERT_TRUE(fixture.state->generations[0].emitted.wait());
  fixture.finish(0);
  ASSERT_TRUE(fixture.state->generations[1].entered.wait());
  EXPECT_EQ(fixture.state->waits(), 0u);
  EXPECT_TRUE(fixture.state->displays[0].expired());
  EXPECT_FALSE(fixture.worker->reinit_pending());
  EXPECT_FALSE(fixture.state->generations[1].emitted.wait(0ms));
  EXPECT_FALSE(fixture.images->try_pop());
  EXPECT_TRUE(fixture.state->generations[0].image.expired());
  expect_reopen_order(fixture, 0, 1);
}

TEST(VideoCaptureReinitTest, StopDuringTopologyReconciliationJoinsAfterTheHookReturnsWithoutReopening) {
  fixture_t fixture;
  fixture.state->hold_topology = true;
  fixture.state->generations[0].emit_image = false;
  fixture.start();
  ASSERT_TRUE(fixture.state->generations[0].entered.wait());
  fixture.emit(0);
  ASSERT_TRUE(fixture.state->generations[0].emitted.wait());
  fixture.finish(0);
  ASSERT_TRUE(fixture.state->topology_entered.wait());
  ASSERT_TRUE(fixture.state->generations[0].destroyed.wait(0ms));
  EXPECT_TRUE(fixture.state->displays[0].expired());

  scoped_operation_t stop([&] {
    fixture.worker->stop();
  },
                          [&] {
                            fixture.state->release_all();
                          });
  // The event is empty: this wait is released by the real stop(), not an authored frame.
  EXPECT_FALSE(fixture.images->pop(3s));
  EXPECT_FALSE(fixture.images->running());
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (fixture.worker->running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_FALSE(fixture.worker->running());
  EXPECT_FALSE(stop.returned(0ms));
  EXPECT_EQ(fixture.state->opens(), 1u);
  fixture.state->topology_release.set();
  ASSERT_TRUE(stop.returned());
  stop.finish();
  EXPECT_EQ(fixture.state->opens(), 1u);
  EXPECT_FALSE(fixture.state->generations[1].entered.wait(0ms));
}
#endif
