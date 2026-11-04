// SPDX-License-Identifier: GPL-3.0-only
// Included inside depth_capture at the original test-block position.
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // The capture lock's counters (capture_lock_stats.h), cumulative; reset
  // clears them after the copy (the native provider fixture's contention mode).
  extern "C" __declspec(dllexport) BOOL SunshineDepthTestCaptureLockStats(lock_stats::values *out, BOOL reset) {
    if (out) *out = capture_lock_values();
    if (reset) reset_capture_lock_values();
    return TRUE;
  }
  // The executed-list proof's uncovered count (proven_counts().uncovered):
  // executions of reported, valid recordings whose barrier methods no hook
  // covered (the alternating fixture's --resume-pass case).
  extern "C" __declspec(dllexport) BOOL SunshineDepthTestUncoveredExecutions(std::uint64_t *out) {
    if (!out) return FALSE;
    *out = capture_observation.proven_counters.uncovered.load(std::memory_order_relaxed);
    return TRUE;
  }
  namespace {
    // One Present of an earlier frame landing inside the next nominated copy's
    // call, between its input's read and the copy (record_impl).
    std::atomic<bool> present_before_next_copy{};
  }
  extern "C" __declspec(dllexport) void SunshineStreamlineTestPresentBeforeNextCopy() {
    present_before_next_copy.store(true, std::memory_order_release);
  }

  // Withdraws that Present when no nominated copy took it (the session
  // timelines' Present inside an SR evaluation that nominated none); whether
  // it was still pending.
  extern "C" __declspec(dllexport) BOOL SunshineStreamlineTestCancelPresentBeforeNextCopy() {
    return present_before_next_copy.exchange(false, std::memory_order_acq_rel) ? TRUE : FALSE;
  }

  // Whether every consumer of a live capture reads its copy-out storage
  // (read_storage), not the snapshot its game recording may still replay
  // into: the NGX cross-queue regression's proof of what it admitted.
  extern "C" __declspec(dllexport) BOOL SunshineStreamlineTestCaptureReadsCopyOut(std::uint64_t capture, BOOL *copy_out) {
    if (!capture || !copy_out) return FALSE;
    std::lock_guard lock(mutex);
    for (const auto &value : capture_pool.slots)
      if (value.id == capture) { *copy_out = value.copied_out && value.published ? TRUE : FALSE; return TRUE; }
    return FALSE;
  }
  // While refuse is set, no copy-out is made (counted as ring_busy, every
  // list of the queue's ring in flight): another queue reads a snapshot only
  // once its recording retired (the Reset rule, immutable_for_other_queues).
  extern "C" __declspec(dllexport) void SunshineStreamlineTestRefuseCopyOuts(BOOL refuse) {
    refuse_copy_outs.store(refuse != FALSE, std::memory_order_release);
  }
  namespace {
    std::atomic<HANDLE> nomination_gate_entered{}, nomination_gate_release{};
    void wait_nomination_gate() {
      if (const auto entered = nomination_gate_entered.exchange(nullptr, std::memory_order_acq_rel)) {
        const auto release = nomination_gate_release.exchange(nullptr, std::memory_order_acq_rel);
        SetEvent(entered);
        if (release) WaitForSingleObject(release, 5000);
      }
    }
  }
  extern "C" __declspec(dllexport) void SunshineStreamlineTestNominationGate(HANDLE entered, HANDLE release) {
    nomination_gate_release.store(release, std::memory_order_relaxed);
    nomination_gate_entered.store(entered, std::memory_order_release);
  }
  namespace {
    // The newest nominated pixel copy published (record_impl): its capture
    // id, with the frame-generation-input flag in the top bit.
    std::atomic<std::uint64_t> newest_published_copy{};
  }
  // The capture id of the newest nominated pixel copy the add-on published,
  // and whether it is an FG scope's input (false: none yet). The session
  // timelines read it after each call that may copy, so their freshness
  // rule names each copy as frame_record::depth_capture does
  // (test_game3d_continuity.h).
  extern "C" __declspec(dllexport) BOOL SunshineStreamlineTestNewestCopy(std::uint64_t *id, BOOL *frame_generation_input) {
    const auto value = newest_published_copy.load(std::memory_order_acquire);
    if (id) *id = value & ~(1ull << 63);
    if (frame_generation_input) *frame_generation_input = (value >> 63) ? TRUE : FALSE;
    return value ? TRUE : FALSE;
  }
#endif

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    std::atomic<void (*)(void *)> allocation_hook{};
    std::atomic<void *> allocation_context{};
    void set_allocation_hook(void (*hook)(void *), void *context) {
      allocation_context.store(context);
      allocation_hook.store(hook);
    }
    std::atomic<unsigned> copy_out_allocation_failures{};
    void fail_copy_out_allocations(unsigned count) { copy_out_allocation_failures.store(count); }
    void settle_storage_releases() { wait_storage_releases(); }
    bool fail_copy_out_allocation() {
      for (auto count = copy_out_allocation_failures.load(); count;)
        if (copy_out_allocation_failures.compare_exchange_weak(count, count - 1)) return true;
      return false;
    }
  }
#endif
