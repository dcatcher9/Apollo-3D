/**
 * @file src/platform/windows/sbs_debug_dump_async.h
 * @brief Small process-lifetime worker and session request state for Dump 3D publication.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace platf::sbs_debug::detail {

  /** A process-lifetime serial queue. Its destructor drains accepted work before exit. */
  class publication_queue {
  public:
    publication_queue():
        worker_([this](std::stop_token) { run(); }) {
    }

    ~publication_queue() {
      {
        std::lock_guard lock(mutex_);
        stopping_ = true;
      }
      wake_.notify_all();
      if (worker_.joinable()) {
        worker_.join();
      }
    }

    publication_queue(const publication_queue &) = delete;
    publication_queue &operator=(const publication_queue &) = delete;

    bool enqueue(std::function<void()> task) noexcept {
      if (!task) {
        return false;
      }
      try {
        {
          std::lock_guard lock(mutex_);
          if (stopping_) {
            return false;
          }
          pending_.push_back(std::move(task));
        }
        wake_.notify_one();
        return true;
      } catch (...) {
        return false;
      }
    }

  private:
    void run() noexcept {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock lock(mutex_);
          wake_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
          if (pending_.empty()) {
            return;
          }
          task = std::move(pending_.front());
          pending_.pop_front();
        }
        try {
          task();
        } catch (...) {
          // Each package transaction has detailed error handling. Never terminate the process
          // because a final diagnostic/logging allocation escaped that boundary.
        }
      }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::function<void()>> pending_;
    bool stopping_ = false;
    std::jthread worker_;
  };

  inline publication_queue &process_publication_queue() {
    // Constructed after normal logging startup and therefore drained before those older process
    // services are destroyed. Session teardown never owns or joins this worker.
    static publication_queue queue;
    return queue;
  }

  /**
   * One session's single-flight and retry state. Accepted work captures shared ownership, so
   * releasing the session handle is nonblocking and cannot invalidate a queued callback.
   */
  class publication_state:
      public std::enable_shared_from_this<publication_state> {
  public:
    static std::shared_ptr<publication_state> create() {
      return std::shared_ptr<publication_state>(new publication_state);
    }

    bool busy() const noexcept {
      std::lock_guard lock(mutex_);
      return busy_;
    }

    bool enqueue(std::function<void()> task) noexcept {
      if (!task) {
        return false;
      }
      std::shared_ptr<publication_state> self;
      {
        std::lock_guard lock(mutex_);
        if (busy_) {
          return false;
        }
        busy_ = true;
        try {
          self = shared_from_this();
        } catch (...) {
          busy_ = false;
          return false;
        }
      }

      bool accepted = false;
      try {
        accepted = process_publication_queue().enqueue(
          [self = std::move(self), task = std::move(task)]() mutable {
            try {
              task();
            } catch (...) {
            }
            {
              std::lock_guard lock(self->mutex_);
              self->busy_ = false;
            }
            self->idle_.notify_all();
          }
        );
      } catch (...) {
        accepted = false;
      }
      if (!accepted) {
        std::lock_guard lock(mutex_);
        busy_ = false;
        idle_.notify_all();
      }
      return accepted;
    }

    bool wait_idle_for(const std::chrono::milliseconds timeout) noexcept {
      try {
        std::unique_lock lock(mutex_);
        return idle_.wait_for(lock, timeout, [this] { return !busy_; });
      } catch (...) {
        return false;
      }
    }

    /** Enable retries for a real request and return its cancellation epoch. */
    std::uint64_t allow_retries_and_token() noexcept {
      std::lock_guard lock(mutex_);
      retry_allowed_ = true;
      return retry_epoch_;
    }

    /** Invalidate all older jobs before clearing their session latch. */
    void cancel_retries(
      const std::shared_ptr<std::atomic<bool>> &button
    ) noexcept {
      std::lock_guard lock(mutex_);
      retry_allowed_ = false;
      ++retry_epoch_;
      publication_failed_ = false;
      file_retry_pending_ = false;
      if (button) {
        button->store(false, std::memory_order_release);
      }
    }

    /** Atomically re-arm a failed request only if no cancellation superseded its token. */
    bool record_publication_failure(
      const std::uint64_t retry_token,
      const bool by_button,
      const bool by_file,
      const std::shared_ptr<std::atomic<bool>> &button
    ) noexcept {
      std::lock_guard lock(mutex_);
      if (!retry_allowed_ || retry_token != retry_epoch_) {
        return false;
      }
      if (by_button && button) {
        button->store(true, std::memory_order_release);
      }
      file_retry_pending_ = file_retry_pending_ || by_file;
      publication_failed_ = true;
      return true;
    }

    void record_trigger_remove_failure() noexcept {
      std::lock_guard lock(mutex_);
      trigger_remove_failed_ = true;
    }

    bool take_publication_failed() noexcept {
      std::lock_guard lock(mutex_);
      return std::exchange(publication_failed_, false);
    }

    bool take_file_retry_pending() noexcept {
      std::lock_guard lock(mutex_);
      return std::exchange(file_retry_pending_, false);
    }

    bool take_trigger_remove_failed() noexcept {
      std::lock_guard lock(mutex_);
      return std::exchange(trigger_remove_failed_, false);
    }

  private:
    publication_state() = default;

    mutable std::mutex mutex_;
    std::condition_variable idle_;
    bool busy_ = false;
    bool retry_allowed_ = true;
    std::uint64_t retry_epoch_ = 1;
    bool publication_failed_ = false;
    bool file_retry_pending_ = false;
    bool trigger_remove_failed_ = false;
  };

  /**
   * Consumes the request before capture. Failure/exception restores it; commit never writes the
   * latch again, so a later click that arrives during capture survives for the next package.
   */
  class button_request_guard {
  public:
    explicit button_request_guard(
      std::shared_ptr<std::atomic<bool>> button
    ) noexcept:
        button_(std::move(button)),
        consumed_(button_ && button_->exchange(false, std::memory_order_acq_rel)),
        restore_(consumed_) {
    }

    ~button_request_guard() {
      if (restore_ && button_) {
        button_->store(true, std::memory_order_release);
      }
    }

    button_request_guard(const button_request_guard &) = delete;
    button_request_guard &operator=(const button_request_guard &) = delete;

    bool consumed() const noexcept {
      return consumed_;
    }

    void commit() noexcept {
      restore_ = false;
    }

  private:
    std::shared_ptr<std::atomic<bool>> button_;
    bool consumed_ = false;
    bool restore_ = false;
  };

}  // namespace platf::sbs_debug::detail
