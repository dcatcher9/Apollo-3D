/**
 * @file src/video_capture_generation.cpp
 * @brief Original-controlblock capture admission with sticky per-generation revocation.
 */
#include "capture_display_retirement.h"
#include "video_capture_generation.h"

#include <utility>

namespace video::detail {
  capture_generation_handoff_t::generation_t::generation_t():
      retirement_(std::make_shared<platf::capture_display_retirement_t>()) {
  }

  const std::shared_ptr<platf::capture_display_retirement_t> &capture_generation_handoff_t::generation_t::retirement() const noexcept {
    return retirement_;
  }

  capture_generation_handoff_t::display_lease_t::display_lease_t(
    std::shared_ptr<platf::display_t> display,
    std::shared_ptr<generation_t> generation
  ):
      display_(std::move(display)),
      generation_(std::move(generation)) {
  }

  capture_generation_handoff_t::display_lease_t::operator bool() const noexcept {
    return static_cast<bool>(display_);
  }

  const std::shared_ptr<platf::display_t> &capture_generation_handoff_t::display_lease_t::display() const noexcept {
    return display_;
  }

  bool capture_generation_handoff_t::display_lease_t::revoked() const noexcept {
    return !generation_ || generation_->phase_.load(std::memory_order_acquire) == phase_t::revoked;
  }

  capture_generation_handoff_t::~capture_generation_handoff_t() {
    stop();
  }

  std::shared_ptr<capture_generation_handoff_t::generation_t> capture_generation_handoff_t::begin() {
    auto next = std::shared_ptr<generation_t>(new generation_t());
    std::shared_ptr<generation_t> previous;
    {
      std::lock_guard lock {mutex_};
      if (stopped_ || (current_ && current_->phase_.load(std::memory_order_relaxed) != phase_t::revoked)) {
        return {};
      }
      next->phase_.store(current_ ? phase_t::reopening : phase_t::opening, std::memory_order_relaxed);
      previous = std::exchange(current_, next);
    }
    // A previous state can own returned raw storage. Never destroy it under admission's mutex.
    return next;
  }

  bool capture_generation_handoff_t::publish(
    const std::shared_ptr<generation_t> &generation,
    const std::shared_ptr<platf::display_t> &display
  ) {
    std::lock_guard lock {mutex_};
    if (stopped_ || !generation || generation != current_ || !display) {
      return false;
    }
    const auto phase = generation->phase_.load(std::memory_order_relaxed);
    if (phase != phase_t::opening && phase != phase_t::reopening) {
      return false;
    }
    generation->display_ = display;
    generation->phase_.store(phase_t::ready, std::memory_order_release);
    return true;
  }

  capture_generation_handoff_t::display_lease_t capture_generation_handoff_t::acquire() {
    std::lock_guard lock {mutex_};
    if (stopped_ || !current_ || current_->phase_.load(std::memory_order_relaxed) != phase_t::ready) {
      return {};
    }
    auto display = current_->display_.lock();
    return display ? display_lease_t {std::move(display), current_} : display_lease_t {};
  }

  bool capture_generation_handoff_t::revoke(const std::shared_ptr<generation_t> &generation) {
    std::lock_guard lock {mutex_};
    if (!generation || generation != current_) {
      return false;
    }
    generation->phase_.store(phase_t::revoked, std::memory_order_release);
    return true;
  }

  void capture_generation_handoff_t::stop() {
    std::shared_ptr<generation_t> current;
    {
      std::lock_guard lock {mutex_};
      stopped_ = true;
      current = current_;
      if (current) {
        current->phase_.store(phase_t::revoked, std::memory_order_release);
      }
    }
    if (current) {
      current->retirement_->cancel();
    }
  }

  bool capture_generation_handoff_t::reinit_pending() const {
    std::lock_guard lock {mutex_};
    if (!current_) {
      return false;
    }
    const auto phase = current_->phase_.load(std::memory_order_relaxed);
    return phase == phase_t::revoked || phase == phase_t::reopening;
  }
}  // namespace video::detail
