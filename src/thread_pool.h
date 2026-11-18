/**
 * @file src/thread_pool.h
 * @brief Declarations for the thread pool system.
 */
#pragma once

// standard includes
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// local includes
#include "task_pool.h"

namespace thread_pool_util {
  /**
   * Allow threads to execute unhindered while keeping full control over the threads.
   */
  class ThreadPool: public task_pool_util::TaskPool {
  private:
    std::vector<std::thread> _thread;

    std::condition_variable _cv;
    std::mutex _lock;

    std::atomic_bool _continue;

  public:
    ThreadPool():
        _continue {false} {
    }

    ~ThreadPool() noexcept {
      if (!_continue) {
        return;
      }

      stop();
      join();
    }

    template<class Function, class... Args>
    void push(Function &&newTask, Args &&...args) {
      std::lock_guard lg(_lock);
      TaskPool::push(std::forward<Function>(newTask), std::forward<Args>(args)...);

      _cv.notify_one();
    }

    template<class Function, class X, class Y, class... Args>
    task_id_t pushDelayed(Function &&newTask, std::chrono::duration<X, Y> duration, Args &&...args) {
      std::lock_guard lg(_lock);
      auto task_id = TaskPool::pushDelayed(std::forward<Function>(newTask), duration, std::forward<Args>(args)...);

      // Update all timers for wait_until
      _cv.notify_all();
      return task_id;
    }

    void start(int threads) {
      _continue = true;

      _thread.resize(threads);

      for (auto &t : _thread) {
        t = std::thread(&ThreadPool::_main, this);
      }
    }

    void stop() {
      std::lock_guard lg(_lock);

      _continue = false;
      _cv.notify_all();
    }

    void join() {
      for (auto &t : _thread) {
        t.join();
      }
    }

  public:
    void _main() {
      while (_continue) {
        if (auto task = this->pop()) {
          (*task)->run();
        } else {
          std::unique_lock uniq_lock(_lock);

          if (ready()) {
            continue;
          }

          if (!_continue) {
            break;
          }

          if (auto tp = next()) {
            _cv.wait_until(uniq_lock, *tp);
          } else {
            _cv.wait(uniq_lock);
          }
        }
      }

      // Execute remaining tasks
      while (auto task = this->pop()) {
        (*task)->run();
      }
    }
  };
}  // namespace thread_pool_util
