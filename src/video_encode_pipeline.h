/**
 * @file src/video_encode_pipeline.h
 * @brief Pictures in flight between the encode thread and a retrieving thread.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace video::detail {

  /**
   * Up to `depth` encoder pictures in flight, retired in submission order on a thread of their own.
   *
   * The encode thread converts and submits a picture, then hands its frame description to push()
   * and goes on to the next frame. The retiring thread waits for the oldest picture, retrieves and
   * publishes it (`retire`), so each packet leaves as soon as its picture completes while the
   * encode thread holds, waits or converts. The encode thread blocks only in acquire() while
   * `depth` pictures are in flight, and in drain().
   *
   * A failed retirement (`retire` returns false) stops retiring: acquire() and drain() return at
   * once and `on_failure` runs on the retiring thread (to wake the encode thread). The pictures
   * still in flight are then the encoder teardown's to drain.
   */
  template<class Frame>
  class encode_pipeline_t {
  public:
    using clock_t = std::chrono::steady_clock;
    using retire_t = std::function<bool(Frame &)>;

    /** Totals since the last take_stats(). */
    struct stats_t {
      std::uint64_t pushed = 0;
      std::uint64_t pushed_behind = 0;  ///< Pushed while an earlier picture was still in flight.
      std::uint64_t retired = 0;
      std::chrono::nanoseconds latency_total {};  ///< push() to the end of its retirement.
      std::chrono::nanoseconds latency_max {};
    };

    /** `on_start` runs first on the retiring thread (for example to raise its priority). */
    encode_pipeline_t(unsigned depth, retire_t retire, std::function<void()> on_failure = {}, std::function<void()> on_start = {}):
        depth_(std::max(depth, 1u)),
        retire_(std::move(retire)),
        on_failure_(std::move(on_failure)),
        on_start_(std::move(on_start)),
        worker_([this] {
          run();
        }) {
    }

    encode_pipeline_t(const encode_pipeline_t &) = delete;
    encode_pipeline_t &operator=(const encode_pipeline_t &) = delete;

    /** Delivers every picture in flight (unless retirement failed), then stops the thread. */
    ~encode_pipeline_t() {
      drain();
      {
        std::lock_guard lock(mutex_);
        stopping_ = true;
      }
      condition_.notify_all();
      worker_.join();
    }

    [[nodiscard]] unsigned depth() const noexcept {
      return depth_;
    }

    /** Encode thread: wait until another picture may be submitted. False once retirement failed. */
    bool acquire() {
      std::unique_lock lock(mutex_);
      condition_.wait(lock, [&] {
        return failed_ || in_flight_locked() < depth_;
      });
      return !failed_;
    }

    /** Encode thread: hand over a submitted picture (after acquire()). */
    void push(Frame frame) {
      {
        std::lock_guard lock(mutex_);
        ++stats_.pushed;
        if (in_flight_locked() > 0) {
          ++stats_.pushed_behind;
        }
        queued_.push_back({std::move(frame), clock_t::now()});
      }
      condition_.notify_all();
    }

    /** Encode thread: wait until every picture in flight has been retired, or retirement failed. */
    bool drain() {
      std::unique_lock lock(mutex_);
      condition_.wait(lock, [&] {
        return failed_ || in_flight_locked() == 0;
      });
      return !failed_;
    }

    [[nodiscard]] bool failed() const {
      std::lock_guard lock(mutex_);
      return failed_;
    }

    [[nodiscard]] unsigned in_flight() const {
      std::lock_guard lock(mutex_);
      return in_flight_locked();
    }

    stats_t take_stats() {
      std::lock_guard lock(mutex_);
      return std::exchange(stats_, {});
    }

  private:
    struct entry_t {
      Frame frame;
      clock_t::time_point pushed_at;
    };

    unsigned in_flight_locked() const {
      return static_cast<unsigned>(queued_.size()) + (retiring_ ? 1u : 0u);
    }

    void run() {
      if (on_start_) {
        on_start_();
      }
      std::unique_lock lock(mutex_);
      while (true) {
        condition_.wait(lock, [&] {
          return stopping_ || (!failed_ && !queued_.empty());
        });
        if (stopping_) {
          return;
        }
        auto entry = std::move(queued_.front());
        queued_.pop_front();
        retiring_ = true;
        lock.unlock();
        const bool retired = retire_(entry.frame);
        const auto latency = clock_t::now() - entry.pushed_at;
        lock.lock();
        retiring_ = false;
        if (retired) {
          ++stats_.retired;
          stats_.latency_total += latency;
          stats_.latency_max = std::max(stats_.latency_max, std::chrono::duration_cast<std::chrono::nanoseconds>(latency));
        } else {
          failed_ = true;
        }
        condition_.notify_all();
        if (!retired && on_failure_) {
          lock.unlock();
          on_failure_();
          lock.lock();
        }
      }
    }

    const unsigned depth_;
    retire_t retire_;
    std::function<void()> on_failure_;
    std::function<void()> on_start_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<entry_t> queued_;
    bool retiring_ = false;
    bool failed_ = false;
    bool stopping_ = false;
    stats_t stats_;
    std::thread worker_;  ///< Last: starts once every other member exists.
  };

}  // namespace video::detail
