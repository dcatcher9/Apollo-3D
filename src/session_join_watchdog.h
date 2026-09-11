/**
 * @file src/session_join_watchdog.h
 * @brief Bounds streaming worker shutdown without timing application or display cleanup.
 */
#pragma once

#include "task_pool.h"

#include <atomic>
#include <functional>
#include <memory>
#include <utility>

namespace stream::detail {
  class session_join_watchdog_t {
  public:
    using task_id_t = task_pool_util::TaskPool::task_id_t;
    using schedule_t = std::function<task_id_t(std::function<void()>)>;
    using cancel_t = std::function<void(task_id_t)>;

    session_join_watchdog_t(schedule_t schedule, cancel_t cancel, std::function<void()> expired):
        armed_(std::make_shared<std::atomic_bool>(true)),
        cancel_(std::move(cancel)),
        task_id_(schedule([armed = armed_, expired = std::move(expired)]() {
          if (armed->exchange(false)) {
            expired();
          }
        })) {
    }

    session_join_watchdog_t(const session_join_watchdog_t &) = delete;
    session_join_watchdog_t &operator=(const session_join_watchdog_t &) = delete;

    ~session_join_watchdog_t() {
      // Cancellation cannot recall a task already dequeued by the pool. Its callback keeps
      // this shared latch alive and can claim expiration only before worker shutdown ends.
      armed_->store(false);
      cancel_(task_id_);
    }

  private:
    std::shared_ptr<std::atomic_bool> armed_;
    cancel_t cancel_;
    task_id_t task_id_;
  };

  template<class ArmWatchdog, class JoinWorkers, class Cleanup>
  void join_workers_before_session_cleanup(ArmWatchdog &&arm_watchdog, JoinWorkers &&join_workers, Cleanup &&cleanup) {
    {
      const auto watchdog = std::forward<ArmWatchdog>(arm_watchdog)();
      std::forward<JoinWorkers>(join_workers)();
    }
    std::forward<Cleanup>(cleanup)();
  }
}  // namespace stream::detail
