// SPDX-License-Identifier: GPL-3.0-only
// Capture diagnostics and report text; functions-only part of streamline_depth_capture.cpp.
// Types, namespace-scope state and test variables stay in the shell.

namespace sunshine_streamline::depth_capture {
  namespace {
    // Requires mutex. Coverage: ReShade's lifecycle (or a registered runtime
    // list's submissions) reports the recording open and outside a render
    // pass. Bundles cannot record barriers, clears or copies, so executing
    // one changes no resource state and leaves coverage alone.
    list_view lifecycle_view(const command_state *owner) {
      if (!owner || !owner->reported) return list_view::unknown;
      return owner->closed ? list_view::closed : owner->render_pass ? list_view::pass : list_view::open;
    }

    void count_admission(list_view view, bool states_observed) {
      constexpr auto relaxed = std::memory_order_relaxed;
      auto &c = list_counts;
      if (view == list_view::open) (states_observed ? c.states_observed : c.states_declared).fetch_add(1, relaxed);
      (view == list_view::open ? c.covered : view == list_view::unknown ? c.unknown :
        view == list_view::closed ? c.closed : c.pass).fetch_add(1, relaxed);
    }
  }

  list_coverage_counts list_coverage() {
    list_coverage_counts out;
    const auto &c = list_counts;
    out.covered = c.covered.load(); out.unknown = c.unknown.load(); out.closed = c.closed.load();
    out.pass = c.pass.load();
    out.states_observed = c.states_observed.load(); out.states_declared = c.states_declared.load();
    return out;
  }

  bool capacity_report(char *out, std::size_t size) {
    const auto hits = capacity_hits.load(std::memory_order_relaxed);
    if (!(hits & ~capacity_reported.load(std::memory_order_relaxed))) return false;
    const auto fresh = hits & ~capacity_reported.fetch_or(hits, std::memory_order_relaxed);
    if (!fresh || !out || !size) return false;
    const auto named = [&](unsigned bit) { return (fresh & bit) != 0; };
    const auto bound = [&](unsigned bit, const char *text, unsigned limit) {
      if (!named(bit)) return;
      const auto used = std::strlen(out);
      if (used < size) std::snprintf(out + used, size - used, " %s=%u", text, limit);
    };
    std::snprintf(out, size, "Sunshine capture capacity: refused at a safety bound:");
    bound(queue_places, "queue_places", queue_limit);
    bound(source_table, "source_table", source_limit);
    bound(snapshot_pool, "depth_snapshots", pixel_slot_limit);
    bound(auxiliary_pool, "auxiliary_snapshots", diagnostic_slot_limit);
    const auto used = std::strlen(out);
    if (used < size)
      std::snprintf(out + used, size - used, "; every place or entry was in use (queue places: each queue names a "
        "capture, a provider or an unfinished fence), so that capture was refused as capacity; logged once per bound");
    return true;
  }

  bool list_coverage_report(std::uint64_t now_ms, char *out, std::size_t size) {
    static std::atomic<std::uint64_t> next_log{}, logged_admissions{};
    auto next = next_log.load(std::memory_order_relaxed);
    if (now_ms < next || !next_log.compare_exchange_strong(next, now_ms + 5000, std::memory_order_relaxed)) return false;
    const auto c = list_coverage();
    const auto admissions = c.covered + c.unknown + c.closed + c.pass;
    if (admissions == logged_admissions.exchange(admissions, std::memory_order_relaxed)) return false;
    const auto u = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    std::snprintf(out, size,
      "Sunshine list lifecycle: admissions covered=%llu (states observed=%llu declared=%llu) not_open={unknown=%llu closed=%llu pass=%llu}",
      u(c.covered), u(c.states_observed), u(c.states_declared), u(c.unknown), u(c.closed), u(c.pass));
    return true;
  }

  proven_state_counts proven_counts() {
    proven_state_counts out;
    {
      std::lock_guard lock(mutex);
      out.tracked = capture_observation.proven.size();
      out.latched = capture_observation.proven.latched_count();
      out.latch_lost = capture_observation.proven.latch_lost();
    }
    const auto &c = capture_observation.proven_counters;
    constexpr auto relaxed = std::memory_order_relaxed;
    out.unrecorded = c.unrecorded.load(relaxed); out.unreported = c.unreported.load(relaxed);
    out.invalidated = c.invalidated.load(relaxed); out.missed_reset = c.missed_reset.load(relaxed);
    out.uncovered = c.uncovered.load(relaxed); out.changed_copies = c.changed_copies.load(relaxed);
    out.earlier_copied = c.earlier_copied.load(relaxed); out.earlier_refused = c.earlier_refused.load(relaxed);
    out.copied_out = capture_queues.copy_out_counters.made.load(relaxed); out.copy_out_busy = capture_queues.copy_out_counters.ring_busy.load(relaxed);
    out.copy_out_failed = capture_queues.copy_out_counters.failed.load(relaxed);
    out.copy_out_predicted = capture_queues.copy_out_counters.predicted.load(relaxed); out.copy_out_taken = capture_queues.copy_out_counters.taken.load(relaxed);
    out.copy_out_released = capture_queues.copy_out_counters.released.load(relaxed);
    out.copy_out_unstored = capture_queues.copy_out_counters.unstored.load(relaxed);
    out.copy_out_skipped = capture_queues.copy_out_counters.skipped.load(relaxed);
    out.reader_moves = reader_moves.load(relaxed);
    out.reader_move_unreadable = reader_move_unreadable.load(relaxed);
    return out;
  }

  status last_status() { return reason.load(); }

  const char *name(status value) {
    switch (value) {
#define CASE(x) case status::x: return #x
    CASE(inactive); CASE(unavailable); CASE(malformed); CASE(unsupported_state); CASE(missing_state);
    CASE(conflicting_state); CASE(incomplete_state); CASE(unsupported_lifetime);
    CASE(unsupported_resource); CASE(unsupported_queue); CASE(exhausted); CASE(recorded); CASE(submitted);
    CASE(ready); CASE(stale); CASE(ambiguous); CASE(failed);
#undef CASE
    }
    return "unknown";
  }

  const char *name(capture_failure value) {
    switch (value) {
#define CASE(x) case capture_failure::x: return #x
    CASE(none); CASE(observer_loss); CASE(discarded_recording); CASE(close_failed); CASE(replay);
    CASE(producer_signal_failed); CASE(producer_queue_changed); CASE(consumer_signal_failed); CASE(consumer_queue_changed);
    CASE(evaluation_failed); CASE(queue_retired); CASE(consumer_capacity); CASE(source_retired);
    CASE(proven_state_changed);
#undef CASE
    }
    return "unknown";
  }

  const char *name(selection_reason value) {
    switch (value) {
#define CASE(x) case selection_reason::x: return #x
    CASE(not_attempted); CASE(no_current_nomination); CASE(inactive_views); CASE(ambiguous_views);
    CASE(pending_nomination); CASE(current_capture); CASE(current_already_consumed); CASE(no_admissible_capture);
    CASE(no_completed_snapshot); CASE(completed_before_gap); CASE(completed_not_older);
    CASE(completed_already_consumed); CASE(presentation_already_selected); CASE(format_changed);
    CASE(completed_not_readable); CASE(completed_snapshot); CASE(fg_scope_missing); CASE(fg_scope_mismatch);
    CASE(lineage_snapshot); CASE(completed_older_than_shown);
#undef CASE
    }
    return "unknown";
  }

  const char *name(state_basis value) {
    switch (value) {
#define CASE(x) case state_basis::x: return #x
    CASE(unknown); CASE(observed_legacy); CASE(observed_enhanced); CASE(declared); CASE(contract); CASE(proven);
#undef CASE
    }
    return "unknown";
  }

  const char *name(consumer_status value) {
    switch (value) {
#define CASE(x) case consumer_status::x: return #x
    CASE(not_attempted); CASE(ready); CASE(missing_input); CASE(unsupported_interface); CASE(observer_not_ready);
    CASE(missing_cookie); CASE(unsupported_command_type); CASE(missing_device); CASE(missing_device_identity);
    CASE(device_mismatch); CASE(recording_missing); CASE(recording_closed); CASE(recording_invalid);
    CASE(recording_render_pass); CASE(slot_missing); CASE(ownership_mismatch); CASE(consumer_capacity);
    CASE(capture_not_ready); CASE(invalid_destination);
#undef CASE
    }
    return "unknown";
  }

  const char *name(recording_loss value) {
    switch (value) {
#define CASE(x) case recording_loss::x: return #x
    CASE(none); CASE(global_observation_loss); CASE(source_identity_unavailable);
    CASE(source_state_capacity); CASE(close_failed);
#undef CASE
    }
    return "unknown";
  }

  const char *name(record_stage value) {
    switch (value) {
#define CASE(x) case record_stage::x: return #x
    CASE(not_attempted); CASE(inactive); CASE(malformed_input); CASE(missing_source);
    CASE(unsupported_lifetime); CASE(unsupported_proof); CASE(command_interface); CASE(source_identity);
    CASE(observer_coverage); CASE(command_type); CASE(command_device); CASE(device_identity); CASE(source_lifetime);
    CASE(recording_missing); CASE(recording_closed); CASE(recording_invalid); CASE(recording_render_pass);
    CASE(incomplete_state); CASE(missing_state); CASE(conflicting_state); CASE(unsupported_state); CASE(resource_region);
    CASE(capacity); CASE(texture_allocation); CASE(descriptor_allocation); CASE(recorded);
    CASE(source_interface); CASE(source_device); CASE(source_description); CASE(source_cookie); CASE(retained);
#undef CASE
    }
    return "unknown";
  }
}
