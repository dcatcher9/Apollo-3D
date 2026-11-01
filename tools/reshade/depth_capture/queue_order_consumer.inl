// SPDX-License-Identifier: GPL-3.0-only
// The capture owner's Step 4 queue-order consumer, a part of
// streamline_depth_capture.cpp (included at its end, in the same translation
// unit): the only reader of the queue-order model's verdicts in the capture
// owner. It holds the ordered branch of readable_elsewhere, the producer
// positions a submission records, the device's forgetting of its order
// records, the reliance on an ordered read in consume_owned (with
// order_inflight_reads and the read_order 3/2 diagnostics), the late-conflict
// query and the queue order line. order_inflight_reads stays in the shell in
// its original order; the shell's part declarations name the functions the
// shell and depth_capture/queues.inl call.
// A later Step 4 deletion removes this part and still edits its call sites:
// readable_elsewhere's call of ordered_after_producer (queues.inl), the
// producer positions in submitted (queues.inl), forget_queue_order in
// release_device, and in consume_owned the relied verdict passed to
// readable_elsewhere and the read_order = 3 branch of rely_on_queue_order.
namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    // readable_elsewhere's queue-order branch (U-C1): the producer fence has
    // not completed, so another queue reads the snapshot only once the model
    // shows every call consumer submits from now on is ordered after its
    // producer position.
    bool ordered_after_producer(const slot &value, std::uint64_t consumer, sunshine_queue_order::verdict *relied) {
      if (!consumer || value.retirement_unknown || !value.producer_position) return false;
      const auto verdict = native_observer::queue_ordered(consumer, value.producer_position);
      if (relied) *relied = verdict;
      return verdict.ordered;
    }
    // This game submission's queue-order position, read before any list of
    // the add-on's own runs on this thread (copy_out takes the next one).
    native_observer::position game_submission_position() { return native_observer::last_execute_position(); }
    // The copy-out list's queue-order position, if copy_out ran one.
    native_observer::position copy_out_position(const std::array<bool, slot_limit> &copied) {
      return std::find(copied.begin(), copied.end(), true) != copied.end() ?
        native_observer::last_execute_position() : native_observer::position{};
    }
    // The position another queue's read must be ordered after (U-C1):
    // the copy-out list, else this submission (read only once its
    // recording retired). A replay moves it, as it moves the fence.
    void set_producer_position(slot &value, bool copied, bool signaled, const native_observer::position &copy_at,
        const native_observer::position &game_at) {
      value.producer_position = copied && signaled ? copy_at : game_at;
    }
    // The device's queue-order records go with it (never dereferenced).
    void forget_queue_order(ID3D12Device *device) {
      native_observer::queue_order_forget_device(reinterpret_cast<std::uintptr_t>(device));
    }
    // consume_owned's read on another queue, after readable_elsewhere gave
    // the verdict relied: false means the read does not stand
    // (capture_not_ready). A fence of the verdict poisoned or forgotten since
    // it was given (another thread's hold in between), or a busy order lock:
    // the read does not stand, and nothing is recorded yet (O1).
    bool rely_on_queue_order(const sunshine_queue_order::verdict &relied, const slot &entry,
        consumer_diagnostic *diagnostic) {
      if (relied.ordered) {
        if (!native_observer::queue_order_rely(relied, entry.id)) return false;
        order_inflight_reads.fetch_add(1, std::memory_order_relaxed);
      }
      if (diagnostic) {
        diagnostic->read_order = relied.ordered ? 3 : 2;
        diagnostic->producer_queue = entry.producer_position.queue;
        diagnostic->producer_ticket = entry.producer_position.ticket;
      }
      return true;
    }
  }
  bool read_conflicted(std::uint64_t capture) { return native_observer::queue_order_conflicted(capture); }
  bool queue_order_line(std::uint64_t now, char *out, std::size_t size) {
    static std::uint64_t logged_at{}, logged_total{UINT64_MAX}, checked_at{};
    static bool disabled_logged{};
    if (!native_observer::queue_order_enabled()) {
      if (disabled_logged) return false;
      disabled_logged = true;
      std::snprintf(out, size, "Sunshine 3D queue order: disabled (QueueFenceOrder=0)");
      return true;
    }
    disabled_logged = false;
    // A pending second-signaller report first, one per Present (each queue
    // once, at most 8 per process: native_observer::queue_order_signaller_line).
    if (native_observer::queue_order_signaller_line(out, size)) return true;
    // At most one model snapshot a second, from the present thread (each
    // Present until 10-27, waiting for the order lock if a game thread held
    // it); one the lock is busy for is taken at a later Present.
    if ((logged_at && now - logged_at < 10000) || (checked_at && now - checked_at < 1000)) return false;
    checked_at = now;
    const auto s = native_observer::queue_order_counts();
    if (s.busy) return false;
    const auto inflight = order_inflight_reads.load(std::memory_order_relaxed);
    std::uint64_t total = s.model.signals + s.model.waits + s.model.cpu_signals + s.ordered + s.resets + inflight +
      s.model.late_conflict + s.model.reset_conflicts + s.model.evicted_reads + s.signal_hooks + s.wait_hooks + s.fence_signal_hooks + s.pending + s.contention;
    for (const auto value : s.refused) total += value;
    for (const auto value : s.model.poisoned) total += value;
    if (total == logged_total || (logged_at && now - logged_at < 10000)) return false;
    logged_at = now; logged_total = total;
    using sunshine_queue_order::poison;
    using sunshine_queue_order::refusal;
    const auto p = [&](poison why) { return static_cast<unsigned long long>(s.model.poisoned[static_cast<unsigned>(why)]); };
    const auto r = [&](refusal why) { return static_cast<unsigned long long>(s.refused[static_cast<unsigned>(why)]); };
    std::snprintf(out, size,
      "Sunshine 3D queue order: hooks={signal=%u wait=%u fence_signal=%u pending=%u} queues=%llu fences=%llu settled=%llu "
      "signals=%llu waits=%llu cpu_signals=%llu poisoned={shared=%llu foreign=%llu cpu=%llu second=%llu monotonic=%llu "
      "unobserved=%llu removed=%llu unrecorded=%llu} refused={disabled=%llu pending_hooks=%llu uncovered=%llu untracked=%llu "
      "overlap=%llu unresolved=%llu baseline=%llu unsettled=%llu poisoned=%llu evicted=%llu cross_device=%llu} ordered=%llu "
      "proofs={inflight=%llu} evicted_waits=%llu resets=%llu cpu_unhooked=%llu late_conflict=%llu reset_conflicts=%llu "
      "evicted_reads=%llu contention=%llu own_signals=%llu",
      s.signal_hooks, s.wait_hooks, s.fence_signal_hooks, s.pending, static_cast<unsigned long long>(s.model.queues),
      static_cast<unsigned long long>(s.model.fences), static_cast<unsigned long long>(s.model.settled),
      static_cast<unsigned long long>(s.model.signals), static_cast<unsigned long long>(s.model.waits),
      static_cast<unsigned long long>(s.model.cpu_signals), p(poison::shared), p(poison::foreign_device), p(poison::cpu_signal),
      p(poison::second_signaller), p(poison::non_monotonic), p(poison::unobserved_signal), p(poison::removed),
      p(poison::unrecorded_signaller), r(refusal::disabled), r(refusal::pending_hooks), r(refusal::uncovered),
      r(refusal::untracked), r(refusal::overlap), r(refusal::unresolved), r(refusal::baseline), r(refusal::unsettled),
      r(refusal::poisoned), r(refusal::evicted), r(refusal::cross_device), static_cast<unsigned long long>(s.ordered),
      static_cast<unsigned long long>(inflight), static_cast<unsigned long long>(s.model.evicted_waits),
      static_cast<unsigned long long>(s.resets), static_cast<unsigned long long>(s.cpu_unhooked),
      static_cast<unsigned long long>(s.model.late_conflict), static_cast<unsigned long long>(s.model.reset_conflicts),
      static_cast<unsigned long long>(s.model.evicted_reads), static_cast<unsigned long long>(s.contention),
      static_cast<unsigned long long>(s.own_signals));
    return true;
  }
}
