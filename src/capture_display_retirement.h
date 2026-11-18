/**
 * @file src/capture_display_retirement.h
 * @brief Return a revoked capture display to its capture thread for destruction.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace platf {
  class display_t;
  struct capture_display_deleter_t;

  /** One original display control block, one capture-thread consumer. */
  class capture_display_retirement_t {
  public:
    capture_display_retirement_t() = default;
    ~capture_display_retirement_t();

    capture_display_retirement_t(const capture_display_retirement_t &) = delete;
    capture_display_retirement_t &operator=(const capture_display_retirement_t &) = delete;

    /** Return null on timeout or sticky cancellation so capture can check both stop authorities. */
    [[nodiscard]] std::unique_ptr<display_t> wait_for(std::chrono::milliseconds timeout);

    /** Wake the consumer without destroying or discarding a returned display. */
    void cancel() noexcept;

    /** End capture-thread ownership and take any already returned display for deletion. */
    [[nodiscard]] std::unique_ptr<display_t> detach() noexcept;

  private:
    friend struct capture_display_deleter_t;
    friend bool arm_capture_display_retirement(
      const std::shared_ptr<display_t> &display,
      std::weak_ptr<capture_display_retirement_t> retirement
    ) noexcept;

    bool arm(display_t *display) noexcept;
    void return_display(display_t *display) noexcept;

    std::mutex mutex_;
    std::condition_variable changed_;
    display_t *bound_ = nullptr;
    display_t *returned_ = nullptr;
    bool attached_ = true;
    bool cancelled_ = false;
  };

  /** Stored in the ORIGINAL shared control block; its weak slot cannot form an ownership cycle. */
  struct capture_display_deleter_t {
    void operator()(display_t *display) const noexcept;

    std::weak_ptr<capture_display_retirement_t> retirement;
    bool armed = false;
  };

  /** Construct before initialization so inherited enable_shared_from_this uses this control block. */
  template<class Concrete, class... Args>
  [[nodiscard]] std::shared_ptr<Concrete> make_unarmed_capture_display(Args &&...args) {
    static_assert(std::is_base_of_v<display_t, Concrete>);
    return std::shared_ptr<Concrete>(new Concrete(std::forward<Args>(args)...), capture_display_deleter_t {});
  }

  /** Arm only the selected initialized candidate while its caller still owns a strong reference.
   * Arming is caller-serialized; it is not concurrent with another arming operation.
   */
  [[nodiscard]] bool arm_capture_display_retirement(
    const std::shared_ptr<display_t> &display,
    std::weak_ptr<capture_display_retirement_t> retirement
  ) noexcept;
}  // namespace platf
