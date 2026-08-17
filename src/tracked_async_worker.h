/**
 * @file src/tracked_async_worker.h
 * @brief Process-lifetime ownership for GPU/driver teardown workers.
 */
#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace video::detail {

  /** Own every asynchronous teardown thread until an explicit process shutdown drain.
   *
   * Session teardown may hand off a driver-blocking destructor, but it may not detach it: those
   * destructors still use logging, CUDA, TensorRT, and D3D process state. `seal_and_drain()` is
   * deliberately a join boundary and must run while the application's forced-shutdown watchdog
   * is alive.
   */
  class tracked_async_worker_pool_t {
  public:
    using task_t = std::packaged_task<void()>;

    tracked_async_worker_pool_t() = default;
    tracked_async_worker_pool_t(const tracked_async_worker_pool_t &) = delete;
    tracked_async_worker_pool_t &operator=(const tracked_async_worker_pool_t &) = delete;

    ~tracked_async_worker_pool_t() {
      seal_and_drain();
    }

    /** Launch a tracked worker. Once sealed, execute inline as a fail-closed lifetime fallback. */
    void launch(task_t task) {
      auto owned_task = std::make_shared<task_t>(std::move(task));
      auto completed = std::make_shared<std::atomic<bool>>(false);
      bool run_inline = false;

      {
        std::lock_guard lock(mutex_);
        reap_completed_locked();
        if (sealed_) {
          run_inline = true;
        } else {
          workers_.push_back({
            std::thread([owned_task, completed]() mutable {
              (*owned_task)();
              completed->store(true, std::memory_order_release);
            }),
            std::move(completed),
          });
        }
      }

      if (run_inline) {
        (*owned_task)();
      }
    }

    /** Reject future workers and join every tracked owner. Idempotent. */
    void seal_and_drain() noexcept {
      std::vector<worker_t> workers;
      {
        std::lock_guard lock(mutex_);
        sealed_ = true;
        workers.swap(workers_);
      }
      for (auto &worker : workers) {
        if (worker.thread.joinable()) {
          worker.thread.join();
        }
      }
    }

    [[nodiscard]] bool sealed() const noexcept {
      std::lock_guard lock(mutex_);
      return sealed_;
    }

  private:
    struct worker_t {
      std::thread thread;
      std::shared_ptr<std::atomic<bool>> completed;
    };

    void reap_completed_locked() {
      for (auto worker = workers_.begin(); worker != workers_.end();) {
        if (!worker->completed->load(std::memory_order_acquire)) {
          ++worker;
          continue;
        }
        if (worker->thread.joinable()) {
          worker->thread.join();
        }
        worker = workers_.erase(worker);
      }
    }

    mutable std::mutex mutex_;
    std::vector<worker_t> workers_;
    bool sealed_ = false;
  };

}  // namespace video::detail
