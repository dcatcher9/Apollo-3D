#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

namespace platf::dxgi::detail {
  // Submission metadata only: tuple_group identifies a logical stereo-pair acquisition,
  // including when the GPU ring reuses the same captured texture. It is not a texture handle.
  struct shutter_queued_present_t {
    std::uint64_t epoch = 0;
    std::uint64_t tuple_group = 0;
    std::uint32_t present_id = 0;
    bool left_eye = true;
    std::int64_t present_begin_qpc = 0;
    std::int64_t present_end_qpc = 0;

    bool operator==(const shutter_queued_present_t &) const = default;
  };

  enum class shutter_pending_take_status_e {
    none,
    taken,
    rejected,
  };

  struct shutter_pending_take_result_t {
    shutter_pending_take_status_e status = shutter_pending_take_status_e::none;
    std::optional<shutter_queued_present_t> present;
  };

  // One owner may retain one already-submitted next image. This ledger never acquires GPU
  // resources, samples/advances a clock, or authorizes IR. A taken descriptor must still pass
  // the existing exact-display checks. The caller retains its keyed tuple through both eyes.
  class shutter_pending_present_ledger_t {
  public:
    explicit shutter_pending_present_ledger_t(std::uint64_t epoch = 1) noexcept:
        epoch_(epoch), accepting_(epoch != 0) {}

    shutter_pending_present_ledger_t(const shutter_pending_present_ledger_t &) = delete;
    shutter_pending_present_ledger_t &operator=(const shutter_pending_present_ledger_t &) = delete;

    bool commit(const shutter_queued_present_t &current, const shutter_queued_present_t &next) noexcept {
      if (!accepting_ || pending_ || current.epoch != epoch_ || next.epoch != epoch_ ||
          !valid(current) || !valid(next) || current.left_eye == next.left_eye ||
          next.present_id != static_cast<std::uint32_t>(current.present_id + 1u) ||
          next.present_begin_qpc < current.present_end_qpc) {
        return reject();
      }
      // L/R share a tuple; the following L begins exactly one new acquisition group.
      if (current.left_eye) {
        if (next.tuple_group != current.tuple_group) {
          return reject();
        }
      } else if (current.tuple_group == std::numeric_limits<std::uint64_t>::max() ||
                 next.tuple_group != current.tuple_group + 1u) {
        return reject();
      }
      pending_ = next;
      return true;
    }

    shutter_pending_take_result_t take(std::uint64_t expected_epoch, bool expected_left, std::uint64_t held_tuple_group) noexcept {
      if (!accepting_ || expected_epoch != epoch_) {
        reject();
        return {shutter_pending_take_status_e::rejected, std::nullopt};
      }
      if (!pending_) {
        return {shutter_pending_take_status_e::none, std::nullopt};
      }
      if (pending_->epoch != expected_epoch || pending_->left_eye != expected_left ||
          pending_->tuple_group != held_tuple_group) {
        reject();
        return {shutter_pending_take_status_e::rejected, std::nullopt};
      }
      const auto present = pending_;
      pending_.reset();
      return {shutter_pending_take_status_e::taken, present};
    }

    // Cancelling metadata cannot cancel an already-queued GPU image. Preserve the descriptor
    // for fault diagnostics while the caller stops submissions and verifies the emitter drain.
    void invalidate() noexcept {
      accepting_ = false;
    }

    // The caller must have completed its FIFO barrier/drain before calling this. This is not
    // proof of GPU completion and does not release the caller's current/pending resources.
    bool reset_after_quiet(std::uint64_t new_epoch) noexcept {
      if (accepting_ || new_epoch == 0 || new_epoch <= epoch_) {
        return reject();
      }
      pending_.reset();
      epoch_ = new_epoch;
      accepting_ = true;
      return true;
    }

    [[nodiscard]] const std::optional<shutter_queued_present_t> &pending() const noexcept { return pending_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
    [[nodiscard]] bool accepting() const noexcept { return accepting_; }

  private:
    static bool valid(const shutter_queued_present_t &present) noexcept {
      return present.epoch != 0 && present.tuple_group != 0 && present.present_begin_qpc > 0 &&
             present.present_end_qpc >= present.present_begin_qpc;
    }

    bool reject() noexcept {
      accepting_ = false;
      return false;
    }

    std::uint64_t epoch_;
    bool accepting_;
    std::optional<shutter_queued_present_t> pending_;
  };

  // A successful frame-latency Wait grants one admission independently of queued metadata.
  // Retain it when a cutoff expires before rendering, or when DO_NOT_WAIT confirms no frame
  // was accepted. Only an S_OK accepted Present may consume it; unknown queue errors are fatal.
  class shutter_frame_admission_t {
  public:
    shutter_frame_admission_t() = default;
    shutter_frame_admission_t(const shutter_frame_admission_t &) = delete;
    shutter_frame_admission_t &operator=(const shutter_frame_admission_t &) = delete;

    [[nodiscard]] bool held() const noexcept { return held_; }

    bool grant() noexcept {
      if (held_) {
        return false;
      }
      held_ = true;
      return true;
    }

    bool consume() noexcept {
      if (!held_) {
        return false;
      }
      held_ = false;
      return true;
    }

  private:
    bool held_ = false;
  };

  struct shutter_emit_phase_plan_t {
    std::int64_t target_qpc = 0;
    std::int64_t latest_write_qpc = 0;
  };

  // Split both factors around one second, so even extreme caller values cannot overflow an
  // intermediate product. Rounding the target up and the final write limit down never widens
  // the requested phase window. A zero phase belongs to the caller's unchanged immediate path.
  inline std::optional<shutter_emit_phase_plan_t> plan_shutter_emit_phase(
    std::int64_t boundary_qpc,
    std::int64_t qpc_frequency,
    std::int64_t phase_us,
    std::int64_t host_budget_us,
    std::int64_t allowed_lateness_us = 250
  ) noexcept {
    if (boundary_qpc <= 0 || qpc_frequency <= 0 || phase_us <= 0 || host_budget_us <= 0 ||
        allowed_lateness_us < 0 || phase_us > host_budget_us || allowed_lateness_us > host_budget_us - phase_us) {
      return std::nullopt;
    }
    const auto ticks_for = [qpc_frequency](std::int64_t microseconds, bool round_up) -> std::optional<std::int64_t> {
      constexpr std::int64_t scale = 1000000;
      constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
      const auto seconds = microseconds / scale;
      const auto fraction = microseconds % scale;
      if (seconds > maximum / qpc_frequency) {
        return std::nullopt;
      }
      auto ticks = seconds * qpc_frequency;
      const auto whole_fraction = fraction * (qpc_frequency / scale);
      const auto remainder_product = fraction * (qpc_frequency % scale);
      const auto fractional_ticks = whole_fraction + remainder_product / scale +
                                    (round_up && remainder_product % scale != 0 ? 1 : 0);
      if (ticks > maximum - fractional_ticks) {
        return std::nullopt;
      }
      return ticks + fractional_ticks;
    };
    const auto target = ticks_for(phase_us, true);
    const auto latest = ticks_for(phase_us + allowed_lateness_us, false);
    if (!target || !latest || *target > *latest || boundary_qpc > std::numeric_limits<std::int64_t>::max() - *latest) {
      return std::nullopt;
    }
    return shutter_emit_phase_plan_t {boundary_qpc + *target, boundary_qpc + *latest};
  }

  struct shutter_lookahead_admission_window_t {
    std::int64_t receipt_qpc = 0;
    std::int64_t cutoff_qpc = 0;
  };

  inline std::optional<shutter_lookahead_admission_window_t> plan_shutter_lookahead_admission(
    std::int64_t receipt_qpc,
    std::int64_t qpc_frequency,
    const shutter_emit_phase_plan_t &phase
  ) noexcept {
    if (receipt_qpc <= 0 || qpc_frequency <= 0 || phase.target_qpc <= 0 ||
        phase.latest_write_qpc < phase.target_qpc) {
      return std::nullopt;
    }
    // At most 500 us after receipt, and strictly before target minus a 250 us guard.
    // Floor/ceil only shrink the budget; division avoids overflowing either product.
    const auto horizon = qpc_frequency / 2000;
    const auto guard = qpc_frequency / 4000 + (qpc_frequency % 4000 != 0 ? 1 : 0);
    if (horizon <= 0 || receipt_qpc > std::numeric_limits<std::int64_t>::max() - horizon ||
        phase.target_qpc <= guard) {
      return std::nullopt;
    }
    return shutter_lookahead_admission_window_t {
      receipt_qpc, std::min(receipt_qpc + horizon, phase.target_qpc - guard),
    };
  }

  enum class shutter_lookahead_admission_action_e {
    retry,
    admit,
    expired,
    cancelled,
    invalid,
  };

  inline shutter_lookahead_admission_action_e shutter_lookahead_admission_action(
    const shutter_lookahead_admission_window_t &window,
    std::int64_t now_qpc,
    bool cancelled,
    bool ready
  ) noexcept {
    using action_e = shutter_lookahead_admission_action_e;
    if (cancelled) {
      return action_e::cancelled;
    }
    if (window.receipt_qpc <= 0 || window.cutoff_qpc <= 0 || now_qpc < window.receipt_qpc) {
      return action_e::invalid;
    }
    if (now_qpc >= window.cutoff_qpc) {
      return action_e::expired;
    }
    return ready ? action_e::admit : action_e::retry;
  }

  enum class shutter_emit_phase_wait_status_e {
    ready,
    cancelled,
    missed,
    invalid,
    wait_failed,
  };

  struct shutter_emit_phase_wait_io_t {
    using clock = std::chrono::steady_clock;
    std::function<std::int64_t()> now_qpc;
    std::function<clock::time_point()> now;
    std::function<bool()> cancelled;
    std::function<bool(clock::time_point)> wait_until;
    // A nonblocking processor hint only; the fine tail must not enter an OS sleep/yield.
    std::function<void()> relax;
  };

  // This authorizes no write itself. The caller must preserve the captured logical eye and
  // recheck latest_write_qpc at WriteFile, after transport checks. Already-late feedback never
  // catches up; lateness is reserved for a wait that began before its target.
  inline shutter_emit_phase_wait_status_e await_shutter_emit_phase(
    const shutter_emit_phase_wait_io_t &io,
    const shutter_emit_phase_plan_t &plan,
    std::int64_t qpc_frequency
  ) {
    using status_e = shutter_emit_phase_wait_status_e;
    using clock = shutter_emit_phase_wait_io_t::clock;
    if (!io.now_qpc || !io.now || !io.cancelled || !io.wait_until || !io.relax || qpc_frequency <= 0 ||
        plan.target_qpc <= 0 || plan.latest_write_qpc < plan.target_qpc) {
      return status_e::invalid;
    }
    if (io.cancelled()) {
      return status_e::cancelled;
    }
    auto qpc = io.now_qpc();
    auto now = io.now();
    if (io.cancelled()) {
      return status_e::cancelled;
    }
    if (qpc <= 0) {
      return status_e::invalid;
    }
    if (qpc > plan.target_qpc) {
      return status_e::missed;
    }
    if (qpc == plan.target_qpc) {
      return status_e::ready;
    }
    const auto duration_for = [qpc_frequency](std::int64_t ticks, bool round_up) -> std::optional<clock::duration> {
      const auto exact = static_cast<long double>(ticks) * clock::period::den /
                         (static_cast<long double>(qpc_frequency) * clock::period::num);
      const auto count = round_up ? std::ceil(exact) : std::floor(exact);
      if (!std::isfinite(count) || count < 0 || (round_up && count < 1) ||
          count >= static_cast<long double>(clock::duration::max().count())) {
        return std::nullopt;
      }
      return clock::duration {static_cast<clock::duration::rep>(count)};
    };
    const auto hard_wait = duration_for(plan.latest_write_qpc - qpc, true);
    if (!hard_wait || now.time_since_epoch().count() > clock::duration::max().count() - hard_wait->count()) {
      return status_e::invalid;
    }
    const auto deadline = now + *hard_wait;
    // Windows timer wakeups may overshoot sub-millisecond waits. Stop using the OS timer
    // before the final 750 us and poll the same clocks with a processor hint instead. This
    // does not widen either deadline. Separate limits bound broken timers and frozen clocks
    // without exhausting the timer retry count during legitimate fast QPC polling.
    constexpr auto fine_guard = std::chrono::duration_cast<clock::duration>(std::chrono::microseconds {750});
    constexpr auto coarse_poll = std::chrono::duration_cast<clock::duration>(std::chrono::microseconds {200});
    constexpr unsigned maximum_waits = 4096, maximum_spins = 131072;
    unsigned waits = 0, spins = 0;
    while (true) {
      if (io.cancelled()) {
        return status_e::cancelled;
      }
      if (now > deadline) {
        return status_e::missed;
      }
      if (qpc >= plan.target_qpc) {
        return qpc <= plan.latest_write_qpc ? status_e::ready : status_e::missed;
      }
      if (now >= deadline) {
        return status_e::missed;
      }
      const auto target_wait = duration_for(plan.target_qpc - qpc, false);
      if (!target_wait) {
        return status_e::invalid;
      }
      if (*target_wait <= fine_guard) {
        if (spins++ >= maximum_spins) {
          return status_e::wait_failed;
        }
        io.relax();
      } else {
        if (waits++ >= maximum_waits) {
          return status_e::wait_failed;
        }
        const auto delay = std::min({*target_wait - fine_guard, deadline - now, coarse_poll});
        const bool waited = io.wait_until(now + delay);
        if (io.cancelled()) {
          return status_e::cancelled;
        }
        if (!waited) {
          return status_e::wait_failed;
        }
      }
      const auto next_qpc = io.now_qpc();
      const auto next_now = io.now();
      if (io.cancelled()) {
        return status_e::cancelled;
      }
      if (next_qpc < qpc || next_now < now) {
        return status_e::invalid;
      }
      qpc = next_qpc;
      now = next_now;
    }
  }

  // A confirmed image can be held beyond its requested minimum dwell. Finite explicit eye
  // commands do not toggle during that hold. Bound the gap and retain raw QPC/counter agreement;
  // this never authorizes a shorter eye interval or infers an eye from refresh parity.
  constexpr bool held_refresh_delta_valid(unsigned repeats, std::uint32_t delta) noexcept {
    return repeats >= 1 && repeats <= 4 && delta >= repeats && delta <= 4 * repeats;
  }

  // A transient path change may be reacquired with the emitter quiet. Repeated losses must
  // still return to 2D instead of keeping the desktop indefinitely in a dark recovery loop.
  class shutter_recovery_budget_t {
  public:
    using clock = std::chrono::steady_clock;
    static constexpr auto acquisition_timeout = std::chrono::seconds {2};
    static constexpr auto retry_window = std::chrono::seconds {10};

    bool begin(clock::time_point now) noexcept {
      if (count_ && now < attempts_[count_ - 1]) {
        return false;
      }
      if (count_ == attempts_.size()) {
        if (now - attempts_.front() < retry_window) {
          return false;
        }
        for (std::size_t index = 1; index < attempts_.size(); ++index) {
          attempts_[index - 1] = attempts_[index];
        }
        --count_;
      }
      attempts_[count_++] = now;
      return true;
    }

    // A recovery holds the picture for a fraction of a second. Six in ten seconds is still a
    // visible disturbance, but a rare late timestamp no longer ends the whole session.
    static constexpr std::size_t maximum_attempts = 6;

  private:
    std::array<clock::time_point, maximum_attempts> attempts_ {};
    std::size_t count_ = 0;
  };

  enum class shutter_feedback_status_e {
    available,
    disjoint,
    unsupported,
  };

  struct shutter_feedback_t {
    shutter_feedback_status_e status = shutter_feedback_status_e::unsupported;
    std::uint32_t present_count = 0;
    std::uint32_t refresh_count = 0;
    std::uint32_t sync_refresh_count = 0;
    std::int64_t sync_qpc = 0;
    std::int32_t result_code = 0;
  };

  enum class shutter_feedback_tracker_status_e {
    acquiring,
    ready,
    accepted,
    rejected,
  };

  // Cold DWM feedback may stall or change counter domains before settling. Acquire a sequence
  // while shutters remain closed; after explicit arming, every sample must preserve the exact
  // submitted identity, physical cadence and caller-selected remaining emission window.
  class shutter_feedback_tracker_t {
  public:
    shutter_feedback_tracker_t(
      unsigned repeats,
      int refresh_millihz,
      std::int64_t qpc_frequency,
      std::int64_t host_age_budget_us,
      std::int64_t cadence_tolerance_us,
      unsigned required_samples = 13,
      // Once armed, the exact refresh-count difference already proves which refresh displayed
      // the image. The timestamp Windows attaches to that refresh can be several hundred
      // microseconds late under load, so the armed check may be wider than cold acquisition.
      // A negative value keeps the acquisition tolerance.
      std::int64_t armed_cadence_tolerance_us = -1
    ) noexcept:
        repeats_(repeats),
        qpc_frequency_(qpc_frequency),
        host_age_budget_us_(host_age_budget_us),
        cadence_tolerance_us_(cadence_tolerance_us),
        armed_cadence_tolerance_us_(armed_cadence_tolerance_us < 0 ? cadence_tolerance_us : armed_cadence_tolerance_us),
        required_samples_(required_samples) {
      failed_ = repeats < 1 || repeats > 4 || refresh_millihz <= 0 || qpc_frequency <= 0 ||
                host_age_budget_us < 0 || cadence_tolerance_us < 0 || required_samples == 0;
      if (!failed_) {
        refresh_ticks_ = static_cast<long double>(qpc_frequency) * 1000.0L / refresh_millihz;
        // All later integer offsets must fit before conversion/addition. A QPC frequency too
        // coarse to distinguish this eye interval cannot establish scanout continuity either.
        failed_ = refresh_ticks_ * repeats < 1.0L ||
                  refresh_ticks_ * repeats >= static_cast<long double>(std::numeric_limits<std::int64_t>::max());
      }
    }

    shutter_feedback_tracker_status_e observe(
      std::uint32_t requested_id,
      const shutter_feedback_t &feedback,
      std::int64_t query_qpc
    ) noexcept {
      using status_e = shutter_feedback_tracker_status_e;
      if (failed_) {
        return status_e::rejected;
      }
      const auto present_qpc = reconstruct(requested_id, feedback);
      if (!present_qpc || feedback.sync_qpc > query_qpc || !fresh(*present_qpc, query_qpc)) {
        return invalidate();
      }
      const auto refresh_delta = static_cast<std::uint32_t>(feedback.refresh_count - last_refresh_);
      const bool continuous = last_present_qpc_ &&
                              requested_id == static_cast<std::uint32_t>(last_present_id_ + 1u) &&
                              (armed_ ? held_refresh_delta_valid(repeats_, refresh_delta) : refresh_delta == repeats_) &&
                              cadence_matches(*present_qpc, refresh_delta);
      if (last_present_qpc_ && !continuous) {
        if (armed_) {
          return invalidate();
        }
        // This is a valid boundary in a new sequence, not evidence that the old eye continued.
        streak_ = 0;
      }
      last_present_id_ = requested_id;
      last_refresh_ = feedback.refresh_count;
      last_present_qpc_ = *present_qpc;
      if (streak_ < required_samples_) {
        ++streak_;
      }
      return armed_ ? status_e::accepted : streak_ >= required_samples_ ? status_e::ready : status_e::acquiring;
    }

    bool arm() noexcept {
      if (failed_ || armed_ || streak_ < required_samples_) {
        return false;
      }
      armed_ = true;
      return true;
    }

    // Check again immediately before the emitter write, using the last accepted PRESENT
    // boundary rather than a newer sync sample. This does not consume another eye/sample.
    bool fresh_for_emit(std::int64_t now_qpc) noexcept {
      if (!armed_ || failed_) {
        return false;
      }
      if (!last_present_qpc_ || !fresh(*last_present_qpc_, now_qpc)) {
        invalidate();
        return false;
      }
      return true;
    }

    // Reprove the same accepted image without consuming another observation or replacing its
    // phase anchor. A failed reproof is for the caller's fault path, not a clock repair.
    [[nodiscard]] bool confirmed_feedback_matches(
      std::uint32_t requested_id,
      const shutter_feedback_t &feedback,
      std::int64_t now_qpc
    ) const noexcept {
      if (!armed_ || failed_ || !last_present_qpc_ || requested_id != last_present_id_ ||
          feedback.refresh_count != last_refresh_ || feedback.sync_qpc > now_qpc ||
          !fresh(*last_present_qpc_, now_qpc)) {
        return false;
      }
      const auto present_qpc = reconstruct(requested_id, feedback);
      if (!present_qpc || *present_qpc > now_qpc) {
        return false;
      }
      // Both boundaries are positive, so their signed difference cannot overflow.
      const auto difference = *present_qpc - *last_present_qpc_;
      return difference >= -1 && difference <= 1;
    }

    [[nodiscard]] unsigned streak() const noexcept { return streak_; }
    [[nodiscard]] std::optional<std::int64_t> last_present_qpc() const noexcept { return last_present_qpc_; }
    [[nodiscard]] bool armed() const noexcept { return armed_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

  private:
    std::optional<std::int64_t> reconstruct(std::uint32_t requested_id, const shutter_feedback_t &feedback) const noexcept {
      if (feedback.status != shutter_feedback_status_e::available || feedback.present_count != requested_id || feedback.sync_qpc <= 0) {
        return std::nullopt;
      }
      const auto offset = static_cast<std::int32_t>(feedback.refresh_count - feedback.sync_refresh_count);
      if (offset < -static_cast<int>(repeats_) || offset > static_cast<int>(repeats_)) {
        return std::nullopt;
      }
      const auto rounded_shift = std::round(static_cast<long double>(offset) * refresh_ticks_);
      // Use floating point only for a bounded rational interval, then check signed integer
      // addition explicitly. Neither a corrupt timestamp nor an extreme QPC frequency may wrap.
      if (rounded_shift >= static_cast<long double>(std::numeric_limits<std::int64_t>::max()) ||
          rounded_shift <= static_cast<long double>(std::numeric_limits<std::int64_t>::min())) {
        return std::nullopt;
      }
      const auto shift = static_cast<std::int64_t>(rounded_shift);
      if ((shift > 0 && feedback.sync_qpc > std::numeric_limits<std::int64_t>::max() - shift) ||
          (shift < 0 && feedback.sync_qpc <= -shift)) {
        return std::nullopt;
      }
      return feedback.sync_qpc + shift;
    }

    bool fresh(std::int64_t present_qpc, std::int64_t now_qpc) const noexcept {
      return now_qpc >= present_qpc &&
             static_cast<long double>(now_qpc - present_qpc) * 1000000.0L / qpc_frequency_ <= host_age_budget_us_;
    }

    bool cadence_matches(std::int64_t present_qpc, std::uint32_t refresh_delta) const noexcept {
      if (present_qpc <= *last_present_qpc_) {
        return false;
      }
      const auto error_ticks = std::fabs(static_cast<long double>(present_qpc - *last_present_qpc_) - refresh_ticks_ * refresh_delta);
      return error_ticks * 1000000.0L / qpc_frequency_ <= (armed_ ? armed_cadence_tolerance_us_ : cadence_tolerance_us_);
    }

    shutter_feedback_tracker_status_e invalidate() noexcept {
      if (armed_) {
        failed_ = true;
        return shutter_feedback_tracker_status_e::rejected;
      }
      streak_ = 0;
      last_present_qpc_.reset();
      return shutter_feedback_tracker_status_e::acquiring;
    }

    unsigned repeats_;
    std::int64_t qpc_frequency_;
    std::int64_t host_age_budget_us_;
    std::int64_t cadence_tolerance_us_;
    std::int64_t armed_cadence_tolerance_us_;
    unsigned required_samples_;
    long double refresh_ticks_ = 0;
    unsigned streak_ = 0;
    std::uint32_t last_present_id_ = 0;
    std::uint32_t last_refresh_ = 0;
    std::optional<std::int64_t> last_present_qpc_;
    bool armed_ = false;
    bool failed_ = false;
  };

  enum class shutter_feedback_wait_status_e {
    displayed,
    cancelled,
    timeout,
    unsupported,
    flush_failed,
    wait_failed,
  };

  struct shutter_feedback_wait_result_t {
    shutter_feedback_wait_status_e status;
    shutter_feedback_t feedback {};
    std::uint32_t queries = 0;
    std::uint32_t flushes = 0;
    std::uint32_t waits = 0;
  };

  inline bool shutter_feedback_is_timing_interruption(const shutter_feedback_wait_result_t &result) noexcept {
    return result.status == shutter_feedback_wait_status_e::timeout ||
           (result.status == shutter_feedback_wait_status_e::unsupported &&
             result.feedback.status == shutter_feedback_status_e::disjoint);
  }

  struct shutter_feedback_wait_io_t {
    using clock = std::chrono::steady_clock;
    std::function<clock::time_point()> now;
    std::function<bool()> cancelled;
    std::function<shutter_feedback_t()> query;
    std::function<bool()> flush;
    std::function<bool(clock::time_point)> wait_until;
  };

  // GetFrameStatistics is asynchronous feedback. Waiting another whole vblank whenever it is
  // stale can itself miss the next eye deadline. DwmFlush can block past the current shutter
  // window, so permit it only during unsignalled acquisition. Armed frames poll raw metadata
  // with the shared cancellable high-resolution waiter, without guessing another present or
  // falling back to a blocking flush. Flush-dependent systems must time out and recover quietly.
  inline shutter_feedback_wait_result_t await_shutter_present(
    const shutter_feedback_wait_io_t &io,
    std::uint32_t requested_present,
    bool acquiring,
    std::chrono::milliseconds timeout = std::chrono::milliseconds {250}
  ) {
    using result_e = shutter_feedback_wait_status_e;
    const auto deadline = io.now() + timeout;
    constexpr auto poll_interval = std::chrono::microseconds {200};
    bool flushed = false;
    shutter_feedback_wait_result_t result {result_e::unsupported};
    const auto finish = [&](result_e status) {
      result.status = status;
      return result;
    };
    while (true) {
      if (io.cancelled()) {
        return finish(result_e::cancelled);
      }
      if (io.now() >= deadline) {
        return finish(result_e::timeout);
      }
      const auto feedback = io.query();
      result.feedback = feedback;
      ++result.queries;
      if (io.cancelled()) {
        return finish(result_e::cancelled);
      }
      if (io.now() >= deadline) {
        return finish(result_e::timeout);
      }
      if (feedback.status == shutter_feedback_status_e::available) {
        if (feedback.present_count == requested_present) {
          return finish(result_e::displayed);
        }
        if (static_cast<std::int32_t>(feedback.present_count - requested_present) > 0) {
          return finish(result_e::unsupported);
        }
      } else if (feedback.status != shutter_feedback_status_e::disjoint || !acquiring) {
        return finish(result_e::unsupported);
      }
      if (acquiring && !flushed) {
        flushed = true;
        ++result.flushes;
        const bool published = io.flush();
        if (io.cancelled()) {
          return finish(result_e::cancelled);
        }
        if (!published) {
          return finish(result_e::flush_failed);
        }
        continue;
      }
      const auto next_poll = std::min(deadline, io.now() + poll_interval);
      ++result.waits;
      if (!io.wait_until(next_poll)) {
        return finish(io.cancelled() ? result_e::cancelled : result_e::wait_failed);
      }
    }
  }

  // Eye phase advances once for an explicitly confirmed submitted image. A bounded extra hold
  // does not replay a command or advance phase by refresh parity; the feedback tracker must
  // independently prove exact identity, raw counter/QPC agreement and current-frame freshness.
  class shutter_presentation_clock_t {
  public:
    explicit shutter_presentation_clock_t(unsigned eye_repeats):
        repeats_(eye_repeats) {}

    [[nodiscard]] bool left_eye() const noexcept {
      return left_;
    }

    [[nodiscard]] bool starts_pair() const noexcept {
      return left_;
    }

    bool observe_displayed(std::uint32_t refresh_count) noexcept {
      if (repeats_ < 1 || repeats_ > 4 ||
          (last_refresh_ && !held_refresh_delta_valid(repeats_, static_cast<std::uint32_t>(refresh_count - *last_refresh_)))) {
        return false;
      }
      last_refresh_ = refresh_count;
      left_ = !left_;
      return true;
    }

  private:
    unsigned repeats_;
    bool left_ = true;
    std::optional<std::uint32_t> last_refresh_;
  };
}  // namespace platf::dxgi::detail
