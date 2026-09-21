// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_alpha_auto.h"

#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {
  using namespace sunshine_game3d;
  constexpr std::uint64_t test_start_ms = 1000;
  constexpr std::uint64_t test_deadline_ms = test_start_ms + alpha_startup_window_ms;

  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }

  void observe(alpha_auto_policy &policy, std::uint64_t sequence, std::uint64_t tick_ms,
    std::uint32_t covered, std::uint32_t pixels = 1000, std::uint32_t invalid = 0, std::uint64_t source = 0) {
    policy.observe({sequence, tick_ms, covered, pixels, invalid}, tick_ms, source);
  }

  void expect(const alpha_auto_decision &decision, bool enabled, alpha_auto_state state,
    bool monitoring, const char *message) {
    require(decision.enabled == enabled && decision.state == state && decision.monitoring == monitoring &&
      ((decision.probe_interval_ms != 0) == monitoring), message);
  }

  void collecting(alpha_auto_policy &policy, std::uint64_t now, bool enabled, const char *message) {
    expect(policy.decision(now), enabled, alpha_auto_state::collecting, true, message);
  }

  void confirm(alpha_auto_policy &policy, std::uint64_t first = 1010) {
    observe(policy, 1, first, 200);
    observe(policy, 2, first + 500, 300);
  }

  void same_frozen(const alpha_auto_decision &a, const alpha_auto_decision &b, const char *message) {
    require(a.enabled == b.enabled && a.state == b.state && a.monitoring == b.monitoring &&
      a.covered == b.covered && a.pixels == b.pixels && a.sample_sequence == b.sample_sequence &&
      a.sample_tick_ms == b.sample_tick_ms && a.window_started == b.window_started &&
      a.window_start_ms == b.window_start_ms && a.accepted_samples == b.accepted_samples &&
      a.probe_interval_ms == b.probe_interval_ms, message);
  }

  void delayed_first_source_opens_one_complete_observation_window() {
    alpha_auto_policy policy;
    const auto waiting = policy.decision(25000);
    expect(waiting, false, alpha_auto_state::waiting_for_source, true, "Delayed game startup consumed the observation window");
    require(!waiting.window_started && !waiting.window_start_ms && !waiting.accepted_samples,
      "Waiting policy invented a window or observations");
    observe(policy, 1, 25000, 200);
    policy.reset_continuity();
    same_frozen(waiting, policy.decision(26000), "An observation implicitly armed the waiting policy");

    policy.begin_observation(30000);
    const auto started = policy.decision(30000);
    expect(started, false, alpha_auto_state::collecting, true, "Explicit first-source notification did not start detection");
    require(started.window_started && started.window_start_ms == 30000, "Window did not start at the first eligible dispatch");
    observe(policy, 1, 30000, 1000);
    policy.begin_observation(35000);
    policy.reset_continuity();
    policy.begin_observation(39000);
    collecting(policy, 30000 + alpha_startup_window_ms - 1, false, "First-source observation window ended early");
    const auto final = policy.decision(30000 + alpha_startup_window_ms);
    expect(final, false, alpha_auto_state::automatic_off, false, "First-source window did not end at its fixed deadline");
    require(final.window_start_ms == 30000 && final.accepted_samples == 1,
      "Repeated acquisition or source reset changed the window or session sample count");
    policy.begin_observation(31000 + alpha_startup_window_ms);
    same_frozen(final, policy.decision(32000 + alpha_startup_window_ms), "Acquisition restarted a finalized automatic decision");

    alpha_auto_policy delayed_selective;
    delayed_selective.decision(11000);
    delayed_selective.begin_observation(25000);
    confirm(delayed_selective, 25000);
    expect(delayed_selective.decision(25500), true, alpha_auto_state::automatic_on, false,
      "Selective alpha after slow startup could not confirm protection");

    alpha_auto_policy begin_at_zero;
    begin_at_zero.begin_observation(0);
    begin_at_zero.begin_observation(9000);
    require(begin_at_zero.decision(alpha_startup_window_ms - 1).window_started &&
      !begin_at_zero.decision(alpha_startup_window_ms - 1).window_start_ms,
      "A zero-time first source was treated as uninitialized");
    expect(begin_at_zero.decision(alpha_startup_window_ms), false, alpha_auto_state::automatic_off, false,
      "A zero-time first source restarted its deadline");
  }

  void manual_before_first_source_never_consumes_or_restarts_the_window() {
    for (bool enabled : {false, true}) {
      alpha_auto_policy policy;
      policy.set_manual(enabled);
      policy.begin_observation(25000);
      const auto manual = policy.decision(40000);
      expect(manual, enabled, enabled ? alpha_auto_state::manual_on : alpha_auto_state::manual_off, false,
        "First-source notification overrode a saved manual choice");
      require(!manual.window_started, "Manual mode started an observation window");
      policy.set_automatic(50000);
      expect(policy.decision(60000), false, alpha_auto_state::waiting_for_source, true,
        "Returning Auto before the first source failed to restore waiting");
      policy.begin_observation(65000);
      confirm(policy, 65000);
      const auto confirmed = policy.decision(65500);
      require(confirmed.window_start_ms == 65000 && confirmed.window_started, "Manual mode consumed the eventual source window");
      expect(confirmed, true, alpha_auto_state::automatic_on, false, "Late source did not confirm after manual mode ended");
      policy.set_manual(false);
      policy.set_automatic(66000);
      same_frozen(confirmed, policy.decision(66000), "Returning Auto lost confirmation or restarted the source window");
    }

    alpha_auto_policy resumed;
    resumed.begin_observation(25000);
    observe(resumed, 1, 25000, 200);
    resumed.set_manual(true);
    resumed.set_automatic(34000);
    resumed.begin_observation(34000);
    collecting(resumed, 25000 + alpha_startup_window_ms - 1, false, "Returning Auto did not preserve an already started source window");
    expect(resumed.decision(25000 + alpha_startup_window_ms), false, alpha_auto_state::automatic_off, false,
      "Returning Auto opened a second source window");
  }

  void first_retained_capture_uses_capture_time_independently_of_dispatch_time() {
    alpha_auto_policy policy;
    policy.begin_observation(25000);
    policy.observe({1, 24950, 200, 1000, 0}, 25000);
    collecting(policy, 25000, true, "Fresh retained input captured before the dispatch was rejected");
    observe(policy, 2, 25449, 200);
    collecting(policy, 25449, true, "Retained input confirmed before 500 ms of captured evidence");
    observe(policy, 3, 25450, 200);
    const auto final = policy.decision(25450);
    expect(final, true, alpha_auto_state::automatic_on, false, "Retained input lost its actual capture-time confirmation interval");
    require(final.window_start_ms == 25000 && final.accepted_samples == 3,
      "Retained input changed the observation deadline or accepted count");

    alpha_auto_policy resume_before_begin;
    resume_before_begin.set_manual(false);
    resume_before_begin.set_automatic(25000);
    resume_before_begin.begin_observation(25100);
    resume_before_begin.observe({100, 24999, 200, 1000, 0}, 25100);
    collecting(resume_before_begin, 25100, false, "First-source arming erased the manual-resume capture cutoff");
    resume_before_begin.observe({1, 25050, 200, 1000, 0}, 25100);
    observe(resume_before_begin, 2, 25550, 200);
    expect(resume_before_begin.decision(25550), true, alpha_auto_state::automatic_on, false,
      "Fresh post-resume retained captures could not confirm");
  }

  void accepted_sample_count_excludes_rejected_data_and_survives_continuity_resets() {
    alpha_auto_policy policy(1000);
    observe(policy, 1, 1010, 1000, 1000, 0, 11);
    observe(policy, 1, 1010, 0, 1000, 0, 22);
    policy.observe({1, 1100, 200, 1000, 0}, 1100, 11);
    policy.observe({2, 2000, 200, 1000, 0}, 1100, 11);
    policy.observe({2, 1100, 200, 1000, 1}, 1100, 11);
    require(policy.decision(1100).accepted_samples == 2, "Duplicate or invalid observations inflated the sample count");
    policy.reset_continuity(11);
    observe(policy, 1, 1200, 0, 1000, 0, 11);
    require(policy.decision(1200).accepted_samples == 3, "Reset erased the cumulative count or rejected the fresh source sequence");
    policy.set_manual(false);
    observe(policy, 2, 1300, 200, 1000, 0, 11);
    require(policy.decision(1300).accepted_samples == 3, "Manual mode accepted another observation");
    policy.set_automatic(1400);
    observe(policy, 1, 1400, 200);
    observe(policy, 2, 1900, 200);
    const auto final = policy.decision(1900);
    require(final.accepted_samples == 5, "Returning Auto erased cumulative observation diagnostics");
    observe(policy, 3, 2000, 200);
    same_frozen(final, policy.decision(2000), "Confirmation failed to freeze accepted sample diagnostics");
  }

  void no_selective_evidence_finishes_off_at_the_original_deadline() {
    alpha_auto_policy policy(1000);
    collecting(policy, 1000, false, "New startup invented alpha coverage");
    observe(policy, 1, 1010, 1000);
    collecting(policy, 1010, false, "Full alpha provisionally enabled UI protection");
    observe(policy, 2, 1110, 0);
    collecting(policy, 1110, false, "Empty alpha provisionally enabled UI protection");
    collecting(policy, test_deadline_ms - 1, false, "Startup ended before the five-minute deadline");
    const auto final = policy.decision(test_deadline_ms);
    expect(final, false, alpha_auto_state::automatic_off, false, "No selective evidence failed to finalize Off at five minutes");
    observe(policy, 3, test_deadline_ms + 100, 200);
    observe(policy, 4, test_deadline_ms + 600, 200);
    policy.reset_continuity();
    same_frozen(final, policy.decision(test_deadline_ms + 1000), "Late observations restarted finalized detection");

    alpha_auto_policy absent(1000);
    expect(absent.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "A no-sample startup stayed open forever");
    alpha_auto_policy late(1000);
    observe(late, 1, test_deadline_ms + 1000, 200);
    expect(late.decision(test_deadline_ms + 1000), false, alpha_auto_state::automatic_off, false, "A late first frame opened a new startup window");
    alpha_auto_policy zero_start(0);
    require(zero_start.decision(0).window_started && !zero_start.decision(0).window_start_ms,
      "Explicit zero constructor did not pre-arm the policy");
    expect(zero_start.decision(alpha_startup_window_ms), false, alpha_auto_state::automatic_off, false, "Zero start timestamp was treated as uninitialized");
  }

  void selective_evidence_enables_immediately_and_confirms_at_500ms() {
    alpha_auto_policy policy(1000);
    observe(policy, 1, 1010, 200);
    collecting(policy, 1010, true, "First selective observation did not provisionally enable protection");
    observe(policy, 2, 1509, 300);
    collecting(policy, 1509, true, "Selective coverage confirmed before 500 ms");
    observe(policy, 3, 1510, 400);
    const auto final = policy.decision(1510);
    expect(final, true, alpha_auto_state::automatic_on, false, "Stable selective alpha did not confirm and stop at exactly 500 ms");
    require(final.covered == 400 && final.pixels == 1000 && final.sample_sequence == 3 && final.sample_tick_ms == 1510,
      "Confirmation lost the accepted sample's identity or coverage");
    observe(policy, 4, 1600, 1000);
    observe(policy, 5, 1700, 0);
    observe(policy, 6, 1800, 100, 1000, 1);
    policy.reset_continuity();
    same_frozen(final, policy.decision(50000), "Confirmed Auto On kept monitoring or lost its frozen evidence");
  }

  void contradictory_or_invalid_evidence_cancels_only_provisional_protection() {
    for (unsigned kind = 0; kind < 5; ++kind) {
      alpha_auto_policy policy(1000);
      observe(policy, 1, 1010, 200);
      auto sample = alpha_coverage_sample {2, 1200, 200, 1000, 0};
      if (kind == 0) sample.invalid = 1;
      if (kind == 1) sample.pixels = 0;
      if (kind == 2) sample.covered = 1001;
      if (kind == 3) sample.covered = 0;
      if (kind == 4) sample.covered = 1000;
      policy.observe(sample, 1200);
      collecting(policy, 1200, false, "Invalid, empty or full alpha retained provisional protection");
      observe(policy, 3, 1300, 200);
      observe(policy, 4, 1799, 200);
      collecting(policy, 1799, true, "Selective recovery reused the interrupted confirmation interval");
      expect(policy.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false,
        "Unconfirmed recovery was promoted to On at the deadline");
    }
  }

  void polling_and_duplicates_do_not_confirm_and_stale_provisional_evidence_expires() {
    alpha_auto_policy policy(1000);
    observe(policy, 10, 1010, 200);
    collecting(policy, 1510, true, "Polling confirmed a single sample or expired it before its maximum age");
    policy.observe({10, 1510, 200, 1000, 0}, 1510);
    policy.observe({9, 1510, 1000, 1000, 0}, 1510);
    const auto unchanged = policy.decision(1510);
    require(unchanged.sample_sequence == 10 && unchanged.sample_tick_ms == 1010 && unchanged.covered == 200,
      "Duplicate or older sequence changed accepted evidence");
    collecting(policy, 1511, false, "Stale provisional evidence remained enabled");
    expect(policy.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "Polling or replayed samples earned confirmation");

    alpha_auto_policy repeated_tick(1000);
    observe(repeated_tick, 1, 1010, 200);
    repeated_tick.observe({2, 1010, 200, 1000, 0}, 1300);
    collecting(repeated_tick, 1300, true, "Fresh repeated capture time disabled otherwise usable provisional alpha");
    expect(repeated_tick.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false,
      "A new sequence without capture-time progression confirmed alpha");
  }

  void the_deadline_excludes_late_samples_even_if_captured_before_it() {
    alpha_auto_policy before(1000);
    observe(before, 1, test_deadline_ms - 501, 200);
    observe(before, 2, test_deadline_ms - 1, 200);
    expect(before.decision(test_deadline_ms - 1), true, alpha_auto_state::automatic_on, false, "Confirmation just before deadline was discarded");

    alpha_auto_policy at_deadline(1000);
    observe(at_deadline, 1, test_deadline_ms - 500, 200);
    observe(at_deadline, 2, test_deadline_ms, 200);
    expect(at_deadline.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "A sample at the exact deadline confirmed alpha");

    alpha_auto_policy delayed(1000);
    observe(delayed, 1, test_deadline_ms - 501, 200);
    delayed.observe({2, test_deadline_ms - 1, 200, 1000, 0}, test_deadline_ms);
    expect(delayed.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "Late completion changed the final decision");

    alpha_auto_policy predates(1000);
    predates.observe({1, 850, 200, 1000, 0}, 1000);
    collecting(predates, 1000, false, "Evidence captured before process startup enabled protection");
    observe(predates, 2, 1000, 200);
    expect(predates.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "Pre-startup evidence joined a confirmation interval");
  }

  void manual_edits_override_provisional_and_final_decisions() {
    for (bool initial : {false, true}) {
      alpha_auto_policy policy(1000);
      observe(policy, 1, 1010, 200);
      policy.set_manual(initial);
      const auto selected = policy.decision(1200);
      expect(selected, initial, initial ? alpha_auto_state::manual_on : alpha_auto_state::manual_off, false,
        "Manual selection did not override provisional detection immediately");
      observe(policy, 2, 1510, 200);
      policy.reset_continuity();
      same_frozen(selected, policy.decision(test_deadline_ms), "A pending observation or deadline overrode a manual choice");
      policy.set_manual(!initial);
      expect(policy.decision(test_deadline_ms + 100), !initial, initial ? alpha_auto_state::manual_off : alpha_auto_state::manual_on, false,
        "Manual choice could not be reversed after startup");
    }
    for (bool enabled : {false, true}) {
      alpha_auto_policy policy(1000);
      if (enabled) confirm(policy);
      policy.decision(test_deadline_ms);
      policy.set_manual(!enabled);
      expect(policy.decision(test_deadline_ms + 100), !enabled, enabled ? alpha_auto_state::manual_off : alpha_auto_state::manual_on, false,
        "Manual choice did not override a finalized automatic decision");
    }
    alpha_auto_policy immediate(1000);
    immediate.set_manual(true);
    expect(immediate.decision(1000), true, alpha_auto_state::manual_on, false, "Manual On waited for the first sample");
  }

  void return_to_auto_preserves_confirmation_and_the_original_deadline() {
    alpha_auto_policy confirmed(1000);
    confirm(confirmed);
    confirmed.set_manual(false);
    confirmed.set_automatic(1600);
    expect(confirmed.decision(1600), true, alpha_auto_state::automatic_on, false,
      "Returning to Auto before the deadline did not immediately restore confirmed On");
    confirmed.set_manual(false);
    confirmed.set_automatic(test_deadline_ms + 1000);
    expect(confirmed.decision(test_deadline_ms + 1000), true, alpha_auto_state::automatic_on, false, "Returning to Auto forgot confirmed On after deadline");

    alpha_auto_policy resumed(1000);
    observe(resumed, 1, 1010, 200);
    resumed.set_manual(true);
    resumed.set_automatic(9000);
    collecting(resumed, 9000, false, "Returning to Auto retained a provisional result from before manual mode");
    observe(resumed, 1, 9010, 200);
    collecting(resumed, 9010, true, "Returning to Auto did not accept a fresh restarted source sequence");
    expect(resumed.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "Returning to Auto restarted the deadline or reused old evidence");

    alpha_auto_policy remaining(1000);
    remaining.set_manual(false);
    remaining.set_automatic(9000);
    confirm(remaining, 9010);
    expect(remaining.decision(9510), true, alpha_auto_state::automatic_on, false, "Remaining startup time could not confirm fresh evidence");
    alpha_auto_policy expired(1000);
    expired.set_manual(true);
    expired.set_automatic(test_deadline_ms + 1000);
    expect(expired.decision(test_deadline_ms + 1000), false, alpha_auto_state::automatic_off, false, "Late Auto selection started a new detection interval");

    alpha_auto_policy pending_before_resume(1000);
    observe(pending_before_resume, 1, 1010, 200, 1000, 0, 11);
    pending_before_resume.set_manual(false);
    pending_before_resume.set_automatic(1400);
    pending_before_resume.observe({100, 1100, 200, 1000, 0}, 1400, 22);
    collecting(pending_before_resume, 1400, false,
      "A pending pre-resume GPU copy enabled protection after returning to Auto");
    observe(pending_before_resume, 1, 1400, 200, 1000, 0, 22);
    observe(pending_before_resume, 2, 1600, 200, 1000, 0, 22);
    collecting(pending_before_resume, 1600, true,
      "Pre-resume capture time shortened the fresh confirmation interval");
    observe(pending_before_resume, 3, 1899, 200, 1000, 0, 22);
    collecting(pending_before_resume, 1899, true,
      "Resumed Auto confirmed before 500 ms of newly captured evidence");
    observe(pending_before_resume, 4, 1900, 200, 1000, 0, 22);
    expect(pending_before_resume.decision(1900), true, alpha_auto_state::automatic_on, false,
      "Rejected pending evidence poisoned the sequence or fresh confirmation after Auto resumed");
  }

  void continuity_gaps_and_resets_require_a_fresh_confirmation_interval() {
    alpha_auto_policy policy(1000);
    observe(policy, 50, 1010, 200);
    policy.reset_continuity();
    collecting(policy, 1010, false, "Reset source retained provisional protection");
    observe(policy, 1, 1200, 200);
    collecting(policy, 1200, true, "Reset source could not begin a fresh provisional interval");
    observe(policy, 2, 1700, 200);
    expect(policy.decision(1700), true, alpha_auto_state::automatic_on, false, "Fresh source interval failed to confirm after reset");

    alpha_auto_policy gap(1000);
    observe(gap, 1, 1010, 200);
    observe(gap, 2, 1511, 200);
    collecting(gap, 1511, true, "Gap above 500 ms confirmed instead of restarting provisional evidence");
    expect(gap.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "Discontinuous samples confirmed at the deadline");
    alpha_auto_policy stale(1000);
    stale.observe({1, 1010, 200, 1000, 0}, 1511);
    collecting(stale, 1511, false, "An already stale readback enabled provisional protection");
    observe(stale, 2, 1550, 200);
    expect(stale.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "A stale readback contributed to confirmation");
  }

  void invalid_future_timestamps_do_not_poison_recovery() {
    alpha_auto_policy policy(1000);
    observe(policy, 1, 1010, 200);
    policy.observe({100, std::numeric_limits<std::uint64_t>::max(), 200, 1000, 0}, 1100);
    collecting(policy, 1100, false, "Future evidence left provisional protection enabled");
    observe(policy, 2, 1200, 200);
    collecting(policy, 1200, true, "Rejected future evidence poisoned valid sequence/timestamp recovery");
    observe(policy, 3, 1700, 200);
    expect(policy.decision(1700), true, alpha_auto_state::automatic_on, false, "Fresh recovery did not confirm after invalid future evidence");
    alpha_auto_policy startup(1000);
    startup.observe({100, std::numeric_limits<std::uint64_t>::max(), 200, 1000, 0}, 1000);
    confirm(startup);
    expect(startup.decision(1510), true, alpha_auto_state::automatic_on, false, "A future first sample poisoned later startup detection");
  }

  void coverage_threshold_is_precise_and_overflow_safe() {
    struct coverage_case { std::uint32_t covered, pixels; bool selective; };
    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    const coverage_case cases[] = {
      {0, 1000, false}, {1, 1000, true}, {998, 1000, true}, {999, 1000, false},
      {1000, 1000, false}, {1, 1, false}, {4290672327U, maximum, true}, {4290672328U, maximum, false}
    };
    for (const auto &sample : cases) {
      alpha_auto_policy policy(1000);
      observe(policy, 1, 1010, sample.covered, sample.pixels);
      collecting(policy, 1010, sample.selective, "Positive selective coverage or 99.9 percent threshold was misclassified provisionally");
      observe(policy, 2, 1510, sample.covered, sample.pixels);
      if (sample.selective)
        expect(policy.decision(1510), true, alpha_auto_state::automatic_on, false, "Selective threshold case did not confirm");
      else
        expect(policy.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "Empty or full threshold case was confirmed");
    }
  }

  void interleaved_sources_keep_independent_streaks_and_provisional_contributions() {
    alpha_auto_policy policy(1000);
    observe(policy, 100, 1010, 1000, 1000, 0, 11);
    observe(policy, 1, 1050, 200, 1000, 0, 22);
    observe(policy, 101, 1100, 1000, 1000, 0, 11);
    collecting(policy, 1100, true, "An opaque source canceled another source's provisional protection");
    policy.observe({102, std::numeric_limits<std::uint64_t>::max(), 200, 1000, 0}, 1200, 11);
    policy.reset_continuity(11);
    collecting(policy, 1200, true, "Invalid evidence or reset from another source canceled fresh selective protection");
    observe(policy, 2, 1550, 300, 1000, 0, 22);
    const auto final = policy.decision(1550);
    expect(final, true, alpha_auto_state::automatic_on, false, "Interleaved opaque sequence numbers suppressed selective confirmation");
    require(final.sample_sequence == 2 && final.sample_tick_ms == 1550 && final.covered == 300,
      "Latest accepted diagnostics came from another source's larger sequence");
    observe(policy, 1, 1600, 0, 1000, 0, 33);
    policy.reset_continuity(22);
    same_frozen(final, policy.decision(1700), "A new source changed confirmed protection");

    alpha_auto_policy separate(1000);
    observe(separate, 1, 1010, 200, 1000, 0, 11);
    observe(separate, 2, 1510, 200, 1000, 0, 22);
    collecting(separate, 1510, true, "Separate source samples combined into a confirmed interval");
    collecting(separate, 2011, false, "All stale source histories retained provisional protection");
    expect(separate.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "Separate single-sample sources confirmed Auto On");

    alpha_auto_policy reset_one(1000);
    observe(reset_one, 50, 1010, 200, 1000, 0, 11);
    observe(reset_one, 50, 1020, 200, 1000, 0, 22);
    const auto latest = reset_one.decision(1020);
    reset_one.reset_continuity(11);
    same_frozen(latest, reset_one.decision(1020), "Resetting one source removed another source's diagnostics or provisional state");
    observe(reset_one, 51, 1520, 200, 1000, 0, 22);
    expect(reset_one.decision(1520), true, alpha_auto_state::automatic_on, false, "Resetting one source broke another source's confirmation interval");

    alpha_auto_policy cleared(1000);
    observe(cleared, 50, 1010, 200, 1000, 0, 11);
    observe(cleared, 50, 1020, 200, 1000, 0, 22);
    cleared.set_manual(false);
    cleared.set_automatic(1300);
    observe(cleared, 1, 1510, 200, 1000, 0, 11);
    observe(cleared, 1, 1520, 200, 1000, 0, 22);
    collecting(cleared, 1520, true, "Returning Auto retained pre-manual per-source confirmation intervals");
    expect(cleared.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false, "A manual interval failed to clear all source histories");
  }

  void cadence_changes_at_one_minute_and_stops_at_five_minutes() {
    alpha_auto_policy waiting;
    require(waiting.decision(900000).probe_interval_ms == alpha_fast_probe_interval_ms,
      "Waiting for the first source prematurely selected sparse probing");
    waiting.begin_observation(test_start_ms);
    observe(waiting, 1, test_start_ms, 1000);
    require(waiting.decision(test_start_ms + alpha_initial_window_ms - 1).probe_interval_ms == alpha_fast_probe_interval_ms,
      "Frequent probing ended before the first minute elapsed");
    const auto sparse = waiting.decision(test_start_ms + alpha_initial_window_ms);
    expect(sparse, false, alpha_auto_state::collecting, true, "The first minute finalized detection instead of reducing frequency");
    require(sparse.probe_interval_ms == alpha_slow_probe_interval_ms,
      "Uniform alpha did not switch to once-per-second probing at one minute");
    require(waiting.decision(test_deadline_ms - 1).probe_interval_ms == alpha_slow_probe_interval_ms,
      "Sparse probing stopped before five minutes total");
    expect(waiting.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false,
      "Uniform alpha did not stop exactly five minutes after the first eligible probe");

    alpha_auto_policy crossing(test_start_ms);
    observe(crossing, 1, test_start_ms + alpha_initial_window_ms - 100, 200);
    require(crossing.decision(test_start_ms + alpha_initial_window_ms).probe_interval_ms == alpha_fast_probe_interval_ms,
      "The one-minute boundary interrupted a fresh candidate's confirmation burst");
    observe(crossing, 2, test_start_ms + alpha_initial_window_ms + 400, 200);
    expect(crossing.decision(test_start_ms + alpha_initial_window_ms + 400), true, alpha_auto_state::automatic_on, false,
      "A candidate straddling the cadence transition could not confirm");
  }

  void sparse_selective_probe_requests_a_frequent_confirmation_burst() {
    alpha_auto_policy policy(test_start_ms);
    const auto sparse_start = test_start_ms + alpha_initial_window_ms;
    observe(policy, 1, sparse_start, 1000);
    require(policy.decision(sparse_start).probe_interval_ms == alpha_slow_probe_interval_ms,
      "Opaque sparse-stage input incorrectly requested frequent probing");
    const auto candidate_start = sparse_start + alpha_slow_probe_interval_ms;
    observe(policy, 2, candidate_start, 200);
    const auto candidate = policy.decision(candidate_start);
    expect(candidate, true, alpha_auto_state::collecting, true, "Sparse selective alpha did not enable protection immediately");
    require(candidate.probe_interval_ms == alpha_fast_probe_interval_ms,
      "Sparse selective alpha did not immediately request a confirmation burst");
    for (std::uint64_t step = 1; step < 5; ++step) {
      const auto tick = candidate_start + step * alpha_fast_probe_interval_ms;
      observe(policy, step + 2, tick, 200);
      const auto confirming = policy.decision(tick);
      expect(confirming, true, alpha_auto_state::collecting, true, "Sparse candidate confirmed before 500 ms of fresh evidence");
      require(confirming.probe_interval_ms == alpha_fast_probe_interval_ms,
        "Fresh candidate dropped back to sparse probing before confirmation");
    }
    observe(policy, 7, candidate_start + alpha_selective_stability_ms, 200);
    expect(policy.decision(candidate_start + alpha_selective_stability_ms), true, alpha_auto_state::automatic_on, false,
      "A sparse candidate's 500 ms burst did not confirm and stop");
  }

  void failed_sparse_candidates_return_to_sparse_without_borrowing_old_evidence() {
    for (unsigned kind = 0; kind < 6; ++kind) {
      alpha_auto_policy policy(test_start_ms);
      const auto candidate_start = test_start_ms + alpha_initial_window_ms + 1000;
      observe(policy, 1, candidate_start, 200);
      auto failure_time = candidate_start + 100;
      if (kind < 3) {
        const auto covered = kind == 0 ? 1000U : kind == 1 ? 0U : 200U;
        observe(policy, 2, failure_time, covered, 1000, kind == 2 ? 1U : 0U);
      } else if (kind == 3) {
        failure_time = candidate_start + alpha_observation_max_age_ms + 1;
      } else if (kind == 4) {
        policy.reset_continuity();
      } else {
        failure_time = candidate_start + alpha_observation_max_age_ms + 1;
        policy.observe({1, failure_time, 200, 1000, 0}, failure_time);
      }
      const auto failed = policy.decision(failure_time);
      expect(failed, false, alpha_auto_state::collecting, true, "A failed sparse candidate left protection enabled");
      require(failed.probe_interval_ms == alpha_slow_probe_interval_ms,
        "A failed sparse candidate did not return to once-per-second probing");
      const auto retry = failure_time + alpha_slow_probe_interval_ms;
      observe(policy, 3, retry, 200);
      observe(policy, 4, retry + alpha_selective_stability_ms - 1, 200);
      expect(policy.decision(retry + alpha_selective_stability_ms - 1), true, alpha_auto_state::collecting, true,
        "A sparse retry borrowed time from an earlier failed candidate");
      observe(policy, 5, retry + alpha_selective_stability_ms, 200);
      expect(policy.decision(retry + alpha_selective_stability_ms), true, alpha_auto_state::automatic_on, false,
        "A fresh sparse retry could not confirm after a failed candidate");
    }

    alpha_auto_policy isolated(test_start_ms);
    const auto sparse_start = test_start_ms + alpha_initial_window_ms;
    for (std::uint64_t step = 0; step < 4; ++step) {
      const auto tick = sparse_start + step * alpha_slow_probe_interval_ms;
      observe(isolated, step + 1, tick, 200);
      expect(isolated.decision(tick), true, alpha_auto_state::collecting, true,
        "Isolated once-per-second selective samples accumulated confirmation");
    }
  }

  void sparse_bursts_keep_independent_sources_and_the_original_deadline() {
    alpha_auto_policy policy(test_start_ms);
    const auto candidate_start = test_start_ms + alpha_initial_window_ms + 1000;
    observe(policy, 50, candidate_start, 200, 1000, 0, 11);
    observe(policy, 1, candidate_start + 100, 1000, 1000, 0, 22);
    require(policy.decision(candidate_start + 100).probe_interval_ms == alpha_fast_probe_interval_ms,
      "An opaque source canceled another source's confirmation burst");
    observe(policy, 2, candidate_start + 200, 200, 1000, 0, 22);
    policy.reset_continuity(11);
    require(policy.decision(candidate_start + 200).probe_interval_ms == alpha_fast_probe_interval_ms,
      "Resetting one source canceled another fresh source's burst");
    observe(policy, 3, candidate_start + 500, 200, 1000, 0, 22);
    expect(policy.decision(candidate_start + 500), true, alpha_auto_state::collecting, true,
      "Sparse candidates from separate sources combined into a confirmed interval");
    observe(policy, 4, candidate_start + 700, 200, 1000, 0, 22);
    expect(policy.decision(candidate_start + 700), true, alpha_auto_state::automatic_on, false,
      "An independent sparse source could not finish its own confirmation burst");

    alpha_auto_policy deadline(test_start_ms);
    observe(deadline, 1, test_deadline_ms - 400, 200);
    require(deadline.decision(test_deadline_ms - 1).probe_interval_ms == alpha_fast_probe_interval_ms,
      "A fresh near-deadline candidate did not request frequent probing");
    expect(deadline.decision(test_deadline_ms), false, alpha_auto_state::automatic_off, false,
      "An unfinished confirmation burst extended the five-minute hard deadline");
    deadline.begin_observation(test_deadline_ms + 100);
    deadline.set_manual(true);
    expect(deadline.decision(test_deadline_ms + 100), true, alpha_auto_state::manual_on, false,
      "Manual On did not override an expired scan immediately");
    deadline.set_automatic(test_deadline_ms + 200);
    expect(deadline.decision(test_deadline_ms + 200), false, alpha_auto_state::automatic_off, false,
      "Returning Auto restarted the five-minute deadline after a failed burst");

    alpha_auto_policy resumed(test_start_ms);
    resumed.set_manual(false);
    resumed.set_automatic(test_start_ms + alpha_initial_window_ms + 1000);
    require(resumed.decision(test_start_ms + alpha_initial_window_ms + 1000).probe_interval_ms == alpha_slow_probe_interval_ms,
      "Returning Auto after one minute restarted the frequent stage");
  }
} // namespace

int main() {
  try {
    delayed_first_source_opens_one_complete_observation_window();
    manual_before_first_source_never_consumes_or_restarts_the_window();
    first_retained_capture_uses_capture_time_independently_of_dispatch_time();
    accepted_sample_count_excludes_rejected_data_and_survives_continuity_resets();
    no_selective_evidence_finishes_off_at_the_original_deadline();
    selective_evidence_enables_immediately_and_confirms_at_500ms();
    contradictory_or_invalid_evidence_cancels_only_provisional_protection();
    polling_and_duplicates_do_not_confirm_and_stale_provisional_evidence_expires();
    the_deadline_excludes_late_samples_even_if_captured_before_it();
    manual_edits_override_provisional_and_final_decisions();
    return_to_auto_preserves_confirmation_and_the_original_deadline();
    continuity_gaps_and_resets_require_a_fresh_confirmation_interval();
    invalid_future_timestamps_do_not_poison_recovery();
    coverage_threshold_is_precise_and_overflow_safe();
    interleaved_sources_keep_independent_streaks_and_provisional_contributions();
    cadence_changes_at_one_minute_and_stops_at_five_minutes();
    sparse_selective_probe_requests_a_frequent_confirmation_burst();
    failed_sparse_candidates_return_to_sparse_without_borrowing_old_evidence();
    sparse_bursts_keep_independent_sources_and_the_original_deadline();
    std::puts("Startup source alpha: 19 policy groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Startup source alpha failed: %s\n", error.what());
    return 1;
  }
}
