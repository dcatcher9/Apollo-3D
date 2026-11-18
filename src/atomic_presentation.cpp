/**
 * @file src/atomic_presentation.cpp
 * @brief Bounded atomic-presentation admission and direct plain encoder completion.
 */
#include "atomic_presentation.h"
#include "logging.h"

#include <algorithm>
#include <utility>

namespace stream::detail {
  atomic_presentation_t::~atomic_presentation_t() {
    stop();
    clear_stopped();
  }

  atomic_presentation_t::submission_t atomic_presentation_t::submit(request_t request) {
    // Allocate before locking. All externally shared records become immutable after identity
    // assignment; no mutable alias escapes this function.
    auto next = std::make_shared<request_t>(std::move(request));
    submission_t result;
    {
      std::lock_guard lock {mutex_};
      if (stopping_) {
        result.rejected = std::move(next);
      } else {
        ++last_transaction_id_;
        if (last_transaction_id_ == 0) {
          ++last_transaction_id_;
        }
        next->change.transaction_id = last_transaction_id_;
        result.superseded = std::exchange(pending_, std::move(next));
      }
    }
    // Superseded/rejected dependencies belong to the caller and never die under mutex_.
    return result;
  }

  void atomic_presentation_t::notify_pending() {
    wake_.notify_one();
  }

  atomic_presentation_t::request_ptr atomic_presentation_t::wait_next() {
    std::unique_lock lock {mutex_};
    wake_.wait(lock, [&] {
      return stopping_ || (!active_ && pending_);
    });
    if (stopping_) {
      return {};
    }
    active_ = std::move(pending_);
    receipt_phase_ = receipt_phase_e::awaiting;
    return active_;
  }

  bool atomic_presentation_t::complete(video::video_mode_applied_t completion) {
    auto incoming = std::make_unique<video::video_mode_applied_t>(std::move(completion));
    bool accepted = false;
    {
      std::lock_guard lock {mutex_};
      if (!stopping_ && active_ && receipt_phase_ == receipt_phase_e::awaiting) {
        const auto matching = std::find_if(
          incoming->requests.begin(),
          incoming->requests.end(),
          [&](const auto &request) {
            return request.transaction_id == active_->change.transaction_id;
          }
        );
        if (matching != incoming->requests.end()) {
          completion_ = std::move(incoming);
          receipt_phase_ = receipt_phase_e::received;
          accepted = true;
        }
      }
    }
    if (accepted) {
      wake_.notify_one();
    } else {
      BOOST_LOG(warning) << "Ignoring a stale or duplicate live video-mode encoder completion.";
    }
    return accepted;
  }

  std::unique_ptr<video::video_mode_applied_t> atomic_presentation_t::wait_completion(std::uint64_t transaction_id) {
    std::unique_lock lock {mutex_};
    if (!active_ || active_->change.transaction_id != transaction_id) {
      return {};
    }
    wake_.wait(lock, [&] {
      return stopping_ || completion_;
    });
    if (stopping_) {
      return {};
    }
    return std::move(completion_);
  }

  bool atomic_presentation_t::finish(std::uint64_t transaction_id) {
    request_ptr finished;
    std::unique_ptr<video::video_mode_applied_t> unused_completion;
    {
      std::lock_guard lock {mutex_};
      if (!active_ || active_->change.transaction_id != transaction_id) {
        return false;
      }
      finished = std::move(active_);
      // A late duplicate cannot survive this active request and become the next one's result.
      unused_completion = std::move(completion_);
    }
    wake_.notify_one();
    return true;
  }

  bool atomic_presentation_t::stopping() const {
    std::lock_guard lock {mutex_};
    return stopping_;
  }

  void atomic_presentation_t::stop() {
    request_ptr pending;
    std::unique_ptr<video::video_mode_applied_t> completion;
    {
      std::lock_guard lock {mutex_};
      stopping_ = true;
      pending = std::move(pending_);
      completion = std::move(completion_);
    }
    wake_.notify_all();
  }

  void atomic_presentation_t::clear_stopped() {
    request_ptr pending;
    request_ptr active;
    std::unique_ptr<video::video_mode_applied_t> completion;
    {
      std::lock_guard lock {mutex_};
      if (!stopping_) {
        return;
      }
      pending = std::move(pending_);
      active = std::move(active_);
      completion = std::move(completion_);
    }
  }
}  // namespace stream::detail
