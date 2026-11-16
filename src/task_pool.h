/**
 * @file src/task_pool.h
 * @brief Declarations for the task pool system.
 */
#pragma once

// standard includes
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace task_pool_util {

  class _ImplBase {
  public:
    inline virtual ~_ImplBase() = default;

    virtual void run() = 0;
  };

  template<class Function>
  class _Impl: public _ImplBase {
    Function _func;

  public:
    _Impl(Function &&f):
        _func(std::forward<Function>(f)) {
    }

    void run() override {
      try {
        _func();
      } catch (...) {
        // Tasks have no result consumer. Previously packaged_task kept failures in an
        // unobserved future; preserve that isolation so one failure cannot stop the worker.
      }
    }
  };

  class TaskPool {
  public:
    typedef std::unique_ptr<_ImplBase> __task;
    typedef _ImplBase *task_id_t;

    typedef std::chrono::steady_clock::time_point __time_point;

  protected:
    std::deque<__task> _tasks;
    std::vector<std::pair<__time_point, __task>> _timer_tasks;
    std::mutex _task_mutex;

  public:
    TaskPool() = default;

    template<class Function, class... Args>
    void push(Function &&newTask, Args &&...args) {
      static_assert(std::is_invocable_v<Function, Args &&...>, "arguments don't match the function");

      auto bind = [task = std::forward<Function>(newTask), tuple_args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        std::apply(task, std::move(tuple_args));
      };

      std::lock_guard<std::mutex> lg(_task_mutex);
      _tasks.emplace_back(toRunnable(std::move(bind)));
    }

    void pushDelayed(std::pair<__time_point, __task> &&task) {
      std::lock_guard lg(_task_mutex);

      auto it = _timer_tasks.cbegin();
      for (; it < _timer_tasks.cend(); ++it) {
        if (std::get<0>(*it) < task.first) {
          break;
        }
      }

      _timer_tasks.emplace(it, task.first, std::move(task.second));
    }

    /**
     * @return An id to cancel the task while it is pending.
     */
    template<class Function, class X, class Y, class... Args>
    task_id_t pushDelayed(Function &&newTask, std::chrono::duration<X, Y> duration, Args &&...args) {
      static_assert(std::is_invocable_v<Function, Args &&...>, "arguments don't match the function");

      __time_point time_point;
      if constexpr (std::is_floating_point_v<X>) {
        time_point = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
      } else {
        time_point = std::chrono::steady_clock::now() + duration;
      }

      auto bind = [task = std::forward<Function>(newTask), tuple_args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        std::apply(task, std::move(tuple_args));
      };

      auto runnable = toRunnable(std::move(bind));

      task_id_t task_id = &*runnable;

      pushDelayed(std::pair {time_point, std::move(runnable)});

      return task_id;
    }

    bool cancel(task_id_t task_id) {
      std::lock_guard lg(_task_mutex);

      auto it = _timer_tasks.begin();
      for (; it < _timer_tasks.cend(); ++it) {
        const __task &task = std::get<1>(*it);

        if (&*task == task_id) {
          _timer_tasks.erase(it);

          return true;
        }
      }

      return false;
    }

    std::optional<__task> pop() {
      std::lock_guard lg(_task_mutex);

      if (!_tasks.empty()) {
        __task task = std::move(_tasks.front());
        _tasks.pop_front();
        return task;
      }

      if (!_timer_tasks.empty() && std::get<0>(_timer_tasks.back()) <= std::chrono::steady_clock::now()) {
        __task task = std::move(std::get<1>(_timer_tasks.back()));
        _timer_tasks.pop_back();
        return task;
      }

      return std::nullopt;
    }

    bool ready() {
      std::lock_guard<std::mutex> lg(_task_mutex);

      return !_tasks.empty() || (!_timer_tasks.empty() && std::get<0>(_timer_tasks.back()) <= std::chrono::steady_clock::now());
    }

    std::optional<__time_point> next() {
      std::lock_guard<std::mutex> lg(_task_mutex);

      if (_timer_tasks.empty()) {
        return std::nullopt;
      }

      return std::get<0>(_timer_tasks.back());
    }

  private:
    template<class Function>
    std::unique_ptr<_ImplBase> toRunnable(Function &&f) {
      return std::make_unique<_Impl<Function>>(std::forward<Function &&>(f));
    }
  };
}  // namespace task_pool_util
