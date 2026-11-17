/** Actual demand-owner orchestration and admission. Every GPU operation is a CPU substitute. */
#include <gtest/gtest.h>

#if defined(_WIN32) && defined(SUNSHINE_TESTS)
  #include "src/config.h"
  #include "src/gpu_workload_arbiter.h"
  #include "src/live_model_preparation.h"
  #include "src/live_model_preparation_test.h"
  #include "src/video_depth_estimator.h"

  #include <array>
  #include <atomic>
  #include <chrono>
  #include <condition_variable>
  #include <functional>
  #include <future>
  #include <mutex>
  #include <stdexcept>
  #include <string>
  #include <system_error>
  #include <thread>
  #include <utility>
  #include <vector>

namespace {
  using namespace std::chrono_literals;
  using access_t = models::testing::live_model_preparation_test_access;
  using hooks_t = models::testing::preparation_hooks_t;
  using status_t = models::engine_build_status;
  using clock_t = std::chrono::steady_clock;

  struct gate_t {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned arrivals = 0;
    bool released = false;
    bool timed_out = false;

    void hold() {
      std::unique_lock lock(mutex);
      ++arrivals;
      changed.notify_all();
      timed_out = !changed.wait_for(lock, 10s, [&] {
        return released;
      }) || timed_out;
    }

    bool wait(unsigned count = 1) {
      std::unique_lock lock(mutex);
      return changed.wait_for(lock, 3s, [&] {
        return arrivals >= count;
      });
    }

    void release() {
      std::lock_guard lock(mutex);
      released = true;
      changed.notify_all();
    }
  };

  struct release_t {
    gate_t &gate;

    ~release_t() {
      gate.release();
    }
  };

  struct threads_t {
    std::function<void()> release;
    std::vector<std::thread> workers;

    ~threads_t() {
      release();
      join();
    }

    void join() {
      for (auto &worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
    }
  };

  struct calls_t {
    std::atomic<unsigned> cpu {0}, depth {0}, ocr {0}, launches {0};
  };

  config::depth_model_info unique_model() {
    static std::atomic<unsigned> serial {0};
    return {"cpu-only-demand-owner-" + std::to_string(++serial), ""};
  }

  hooks_t cpu_operations(calls_t &calls) {
    hooks_t hooks;
    hooks.prewarm_shaders = [&](const auto &) {
      ++calls.cpu;
      return true;
    };
    hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
      ++calls.depth;
      return true;
    };
    hooks.prepare_ocr = [&](const auto &, const auto &) {
      ++calls.ocr;
      return true;
    };
    hooks.before_gpu_thread_launch = [&] {
      ++calls.launches;
    };
    return hooks;
  }

  bool wait_unregistered(const models::live_model_preparation_t &owner) {
    const auto deadline = clock_t::now() + 3s;
    while (clock_t::now() < deadline) {
      if (!access_t::registered(owner)) {
        return true;
      }
      std::this_thread::sleep_for(1ms);
    }
    return !access_t::registered(owner);
  }
}  // namespace

TEST(LiveModelPreparationDemandTest, ConstructionOnlyPrewarmsCpuWithoutConsumingGpuAdmission) {
  const auto model = unique_model();
  calls_t calls;
  gate_t cpu;
  auto hooks = cpu_operations(calls);
  hooks.prewarm_shaders = [&](const auto &) {
    ++calls.cpu;
    cpu.hold();
    return true;
  };
  auto offline = gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
  ASSERT_TRUE(offline);
  auto owner = access_t::create(model, "immutable-adapter", std::move(hooks));
  release_t release {cpu};
  ASSERT_TRUE(cpu.wait());
  EXPECT_EQ(calls.cpu.load(), 1u);
  EXPECT_EQ(calls.depth.load(), 0u);
  EXPECT_EQ(calls.ocr.load(), 0u);
  EXPECT_EQ(calls.launches.load(), 0u);
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown);
  EXPECT_FALSE(gpu_workload::live_stream_active());
  cpu.release();
  access_t::join(*owner);
  EXPECT_FALSE(cpu.timed_out);
  EXPECT_EQ(calls.depth.load(), 0u);
  EXPECT_EQ(calls.ocr.load(), 0u);
}

TEST(LiveModelPreparationDemandTest, FailedCpuPrewarmDoesNotConsumeOrFailAnExplicitGpuDemand) {
  const auto model = unique_model();
  calls_t calls;
  auto hooks = cpu_operations(calls);
  hooks.prewarm_shaders = [&](const auto &) {
    ++calls.cpu;
    return false;
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  access_t::join(*owner);
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown);
  ASSERT_TRUE(models::request_live_model_preparation());
  access_t::join(*owner);
  EXPECT_EQ(calls.cpu.load(), 1u);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_EQ(calls.launches.load(), 1u);
  // Fake success proves only owner orchestration; no real TensorRT readiness is asserted.
  EXPECT_NE(models::tensorrt_model_prepare_status(model), status_t::failed);
}

TEST(LiveModelPreparationDemandTest, ConcurrentAndRepeatedExplicitRequestsStartExactlyOneWorker) {
  const auto model = unique_model();
  calls_t calls;
  gate_t start, depth;
  auto hooks = cpu_operations(calls);
  hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
    ++calls.depth;
    depth.hold();
    return true;
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  access_t::join(*owner);
  std::array<bool, 8> accepted {};
  threads_t threads {[&] {
    start.release();
    depth.release();
  }};
  for (std::size_t index = 0; index < accepted.size(); ++index) {
    threads.workers.emplace_back([&, index] {
      start.hold();
      accepted[index] = models::request_live_model_preparation();
    });
  }
  ASSERT_TRUE(start.wait(static_cast<unsigned>(accepted.size())));
  start.release();
  threads.join();
  ASSERT_TRUE(depth.wait());
  for (const auto result : accepted) {
    EXPECT_TRUE(result);
  }
  EXPECT_EQ(calls.launches.load(), 1u);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 0u);
  depth.release();
  access_t::join(*owner);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_TRUE(models::request_live_model_preparation());
  access_t::join(*owner);
  EXPECT_EQ(calls.launches.load(), 1u);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_FALSE(start.timed_out);
  EXPECT_FALSE(depth.timed_out);
}

TEST(LiveModelPreparationDemandTest, WorkerLeaseExcludesOfflineAfterTheOriginalSessionLeaseEnds) {
  const auto model = unique_model();
  calls_t calls;
  gate_t depth;
  auto hooks = cpu_operations(calls);
  hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
    ++calls.depth;
    depth.hold();
    return true;
  };
  auto original = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
  ASSERT_TRUE(original);
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  release_t release {depth};
  ASSERT_TRUE(models::request_live_model_preparation());
  ASSERT_TRUE(depth.wait());
  original.reset();
  EXPECT_TRUE(gpu_workload::live_stream_active());
  EXPECT_FALSE(gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs));
  depth.release();
  access_t::join(*owner);
  EXPECT_FALSE(gpu_workload::live_stream_active());
  EXPECT_TRUE(gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs));
  EXPECT_FALSE(depth.timed_out);
}

TEST(LiveModelPreparationDemandTest, BusyDenialLeavesTheAttemptUnconsumedForALaterExplicitRequest) {
  const auto model = unique_model();
  calls_t calls;
  auto owner = access_t::create(model, "adapter", cpu_operations(calls));
  access_t::join(*owner);
  auto offline = gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
  ASSERT_TRUE(offline);
  EXPECT_FALSE(models::request_live_model_preparation());
  EXPECT_EQ(calls.depth.load(), 0u);
  EXPECT_EQ(calls.ocr.load(), 0u);
  EXPECT_EQ(calls.launches.load(), 0u);
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown);
  // Status reads are pure and cannot retry or claim a GPU lease.
  for (unsigned read = 0; read < 8u; ++read) {
    EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown);
  }
  offline.reset();
  ASSERT_TRUE(models::request_live_model_preparation());
  access_t::join(*owner);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_EQ(calls.launches.load(), 1u);
}

TEST(LiveModelPreparationDemandTest, ClosingUnregistersBeforeJoiningCpuAndRejectsTheFirstGpuRequest) {
  EXPECT_FALSE(models::request_live_model_preparation());
  const auto model = unique_model();
  calls_t calls;
  gate_t cpu;
  auto hooks = cpu_operations(calls);
  hooks.prewarm_shaders = [&](const auto &) {
    ++calls.cpu;
    cpu.hold();
    return true;
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  release_t release {cpu};
  ASSERT_TRUE(cpu.wait());
  std::promise<void> close_started, close_finished;
  auto started = close_started.get_future();
  auto finished = close_finished.get_future();
  threads_t threads {[&] {
    cpu.release();
  }};
  threads.workers.emplace_back([&] {
    close_started.set_value();
    access_t::close(*owner);
    close_finished.set_value();
  });
  ASSERT_EQ(started.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(wait_unregistered(*owner));
  EXPECT_EQ(finished.wait_for(0ms), std::future_status::timeout);
  EXPECT_FALSE(models::request_live_model_preparation());
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown);
  EXPECT_EQ(calls.depth.load(), 0u);
  cpu.release();
  ASSERT_EQ(finished.wait_for(3s), std::future_status::ready);
  threads.join();
  access_t::close(*owner);
  EXPECT_FALSE(models::request_live_model_preparation());
  EXPECT_EQ(calls.launches.load(), 0u);
  EXPECT_FALSE(cpu.timed_out);
}

TEST(LiveModelPreparationDemandTest, ClosingRejectsNewRequestsWhileJoiningTheAcceptedWorker) {
  const auto model = unique_model();
  calls_t calls;
  gate_t depth;
  auto hooks = cpu_operations(calls);
  hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
    ++calls.depth;
    depth.hold();
    return true;
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  release_t release {depth};
  ASSERT_TRUE(models::request_live_model_preparation());
  ASSERT_TRUE(depth.wait());
  std::promise<void> close_finished;
  auto finished = close_finished.get_future();
  threads_t threads {[&] {
    depth.release();
  }};
  threads.workers.emplace_back([&] {
    access_t::close(*owner);
    close_finished.set_value();
  });
  ASSERT_TRUE(wait_unregistered(*owner));
  EXPECT_FALSE(models::request_live_model_preparation());
  EXPECT_EQ(finished.wait_for(0ms), std::future_status::timeout);
  EXPECT_TRUE(gpu_workload::live_stream_active());
  depth.release();
  ASSERT_EQ(finished.wait_for(3s), std::future_status::ready);
  threads.join();
  EXPECT_EQ(calls.launches.load(), 1u);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_FALSE(gpu_workload::live_stream_active());
}

TEST(LiveModelPreparationDemandTest, RequestPublicationAndShutdownCannotCreateALateUnjoinedWorker) {
  const auto model = unique_model();
  calls_t calls;
  gate_t launch, depth;
  auto hooks = cpu_operations(calls);
  hooks.before_gpu_thread_launch = [&] {
    ++calls.launches;
    launch.hold();
  };
  hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
    ++calls.depth;
    depth.hold();
    return true;
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  std::promise<bool> request_finished;
  auto requested = request_finished.get_future();
  std::promise<void> close_started, close_finished;
  auto closing = close_started.get_future();
  auto closed = close_finished.get_future();
  threads_t threads {[&] {
    launch.release();
    depth.release();
  }};
  threads.workers.emplace_back([&] {
    request_finished.set_value(models::request_live_model_preparation());
  });
  ASSERT_TRUE(launch.wait());
  threads.workers.emplace_back([&] {
    close_started.set_value();
    access_t::close(*owner);
    close_finished.set_value();
  });
  ASSERT_EQ(closing.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(closed.wait_for(0ms), std::future_status::timeout);
  launch.release();
  ASSERT_EQ(requested.wait_for(3s), std::future_status::ready);
  EXPECT_TRUE(requested.get());
  ASSERT_TRUE(depth.wait());
  ASSERT_TRUE(wait_unregistered(*owner));
  EXPECT_EQ(closed.wait_for(0ms), std::future_status::timeout);
  depth.release();
  ASSERT_EQ(closed.wait_for(3s), std::future_status::ready);
  threads.join();
  EXPECT_FALSE(models::request_live_model_preparation());
  EXPECT_EQ(calls.launches.load(), 1u);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_FALSE(launch.timed_out);
  EXPECT_FALSE(depth.timed_out);
  EXPECT_FALSE(gpu_workload::live_stream_active());
}

TEST(LiveModelPreparationDemandTest, LaunchFailurePublishesTerminalFailureAndConsumesTheSingleAttempt) {
  const auto model = unique_model();
  calls_t calls;
  auto hooks = cpu_operations(calls);
  hooks.before_gpu_thread_launch = [&] {
    ++calls.launches;
    throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again), "Injected GPU-thread creation edge");
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  EXPECT_TRUE(models::request_live_model_preparation());
  access_t::join(*owner);
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::failed);
  EXPECT_TRUE(models::request_live_model_preparation());
  EXPECT_EQ(calls.launches.load(), 1u);
  EXPECT_EQ(calls.depth.load(), 0u);
  EXPECT_EQ(calls.ocr.load(), 0u);
  EXPECT_FALSE(gpu_workload::live_stream_active());
}

TEST(LiveModelPreparationDemandTest, DepthExceptionPublishesFailureStillAttemptsOcrAndReleasesTheLease) {
  const auto model = unique_model();
  calls_t calls;
  auto hooks = cpu_operations(calls);
  hooks.prepare_depth = [&](const auto &, const auto &, const auto &) -> bool {
    ++calls.depth;
    throw std::runtime_error("Injected depth operation failure");
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  ASSERT_TRUE(models::request_live_model_preparation());
  access_t::join(*owner);
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::failed);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_FALSE(gpu_workload::live_stream_active());
}

TEST(LiveModelPreparationDemandTest, FailedDepthReturnPublishesFailureWithoutSkippingOptionalOcr) {
  const auto model = unique_model();
  calls_t calls;
  auto hooks = cpu_operations(calls);
  hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
    ++calls.depth;
    return false;
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  ASSERT_TRUE(models::request_live_model_preparation());
  access_t::join(*owner);
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::failed);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
}

TEST(LiveModelPreparationDemandTest, OcrExceptionCannotDemoteAnInjectedOwnerCompletionFromReady) {
  const auto model = unique_model();
  calls_t calls;
  auto hooks = cpu_operations(calls);
  hooks.prepare_depth = [&](const auto &, const auto &selected, const auto &) {
    ++calls.depth;
    // Status-map/owner exception policy only: this unique CPU-only key has no TensorRT engine.
    // The GPU fixtures never use this bridge and require real authenticated output instead.
    models::testing::publish_model_preparation_status_for_test(selected, status_t::ready);
    return true;
  };
  hooks.prepare_ocr = [&](const auto &, const auto &) -> bool {
    ++calls.ocr;
    throw std::runtime_error("Injected optional OCR failure after simulated completion");
  };
  auto owner = access_t::create(model, "adapter", std::move(hooks));
  ASSERT_TRUE(models::request_live_model_preparation());
  access_t::join(*owner);
  EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::ready);
  EXPECT_EQ(calls.depth.load(), 1u);
  EXPECT_EQ(calls.ocr.load(), 1u);
  EXPECT_TRUE(models::request_live_model_preparation());
  EXPECT_EQ(calls.launches.load(), 1u);
  EXPECT_FALSE(gpu_workload::live_stream_active());
}

#endif
