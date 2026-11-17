/**
 * @file src/capture_display_retirement.cpp
 * @brief Capture-thread ownership return for an original display control block.
 */
#include "capture_display_retirement.h"
#include "platform/common.h"

namespace platf {
  capture_display_retirement_t::~capture_display_retirement_t() {
    // In ordinary reinitialization the capture thread has already taken this object. During
    // cancellation, delete a returned object here without holding the slot mutex.
    auto returned = detach();
  }

  std::unique_ptr<display_t> capture_display_retirement_t::wait_for(const std::chrono::milliseconds timeout) {
    std::unique_lock lock {mutex_};
    if (!changed_.wait_for(lock, timeout, [&]() {
          return cancelled_ || returned_;
        }) ||
        cancelled_) {
      return {};
    }
    return std::unique_ptr<display_t>(std::exchange(returned_, nullptr));
  }

  void capture_display_retirement_t::cancel() noexcept {
    {
      std::lock_guard lock {mutex_};
      cancelled_ = true;
    }
    changed_.notify_all();
  }

  std::unique_ptr<display_t> capture_display_retirement_t::detach() noexcept {
    display_t *returned;
    {
      std::lock_guard lock {mutex_};
      attached_ = false;
      cancelled_ = true;
      returned = std::exchange(returned_, nullptr);
    }
    changed_.notify_all();
    return std::unique_ptr<display_t>(returned);
  }

  bool capture_display_retirement_t::arm(display_t *display) noexcept {
    std::lock_guard lock {mutex_};
    if (!attached_ || cancelled_ || bound_ || !display) {
      return false;
    }
    bound_ = display;
    return true;
  }

  void capture_display_retirement_t::return_display(display_t *display) noexcept {
    {
      std::lock_guard lock {mutex_};
      if (attached_ && bound_ == display && !returned_) {
        returned_ = display;
        changed_.notify_all();
        // The consumer may destroy the object immediately after this lock is released.
        return;
      }
    }
    delete display;
  }

  void capture_display_deleter_t::operator()(display_t *display) const noexcept {
    if (auto slot = retirement.lock()) {
      slot->return_display(display);
      return;
    }
    delete display;
  }

  bool arm_capture_display_retirement(
    const std::shared_ptr<display_t> &display,
    std::weak_ptr<capture_display_retirement_t> retirement
  ) noexcept {
    auto *deleter = std::get_deleter<capture_display_deleter_t>(display);
    const auto slot = retirement.lock();
    if (!display || !deleter || deleter->armed || !slot || !slot->arm(display.get())) {
      return false;
    }
    deleter->retirement = std::move(retirement);
    deleter->armed = true;
    return true;
  }
}  // namespace platf
