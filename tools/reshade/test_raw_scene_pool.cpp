// SPDX-License-Identifier: GPL-3.0-only
#include "raw_scene_pool.h"
#include "automatic_scene_transition.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_raw_scene;

  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }
  void close(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      std::fprintf(stderr, "%s: actual=%.17g expected=%.17g\n", message, actual, expected);
      throw std::runtime_error(message);
    }
  }
  sample capture(const selected_frame &current, std::uint64_t id, std::uint64_t time, float t) {
    sample value;
    value.id = id;
    value.capture_ms = time;
    value.metadata = current;
    value.readback_frame = current.frame;
    value.readback_source = current.source;
    value.readback_layout_epoch = current.layout_epoch;
    value.raw.fill(current.direction == orientation::normal ? 1.f - t : t);
    return value;
  }
  struct fixture {
    pool controller;
    source_roster roster;
    reentry_transition transition;
    std::array<sample, maximum_sources> latest{};
    std::array<output, maximum_sources> last{};
    std::array<std::uint64_t, maximum_sources> due{};
    std::array<float, maximum_sources> t{.125f, .25f, .0625f, .5f};
    std::uint64_t now{}, frame{}, id{};
    float blend{};

    explicit fixture(unsigned count = 3) {
      roster.basis_epoch = 7;
      roster.routing_epoch = 11;
      roster.count = count;
      for (unsigned i = 0; i < maximum_sources; ++i)
        roster.members[i] = {{0x100 + i, 20 + i, 0, 3840, 2160, 0, 0, 3840, 2160},
          3, 0, i == 1 ? orientation::normal : orientation::reversed};
      require(controller.reset(roster.basis_epoch, 0), "Initial reset rejected");
    }
    selected_frame current(unsigned index) {
      const auto &basis = roster.members[index];
      selected_frame value;
      value.basis_epoch = roster.basis_epoch;
      value.source = basis.source;
      value.layout_epoch = basis.layout_epoch;
      value.convention_epoch = basis.convention_epoch;
      value.direction = basis.direction;
      value.frame = {++frame, 99};
      roster.observed_frame = value.frame;
      value.depth_ready = value.aligned_viewport_assumed = true;
      return value;
    }
    output tick(std::uint64_t time, unsigned index, bool take_capture = false, bool use_cache = true) {
      now = time;
      auto selected = current(index);
      if (take_capture) latest[index] = capture(selected, ++id, time, t[index]);
      const auto result = controller.update(roster, selected,
        use_cache && latest[index].id ? &latest[index] : nullptr, time);
      if (result.ready)
        close(double(result.H) * result.t0, 1, 2e-7,
          "Ready rotating source did not use its current zero as the scale reference");
      blend = transition.update(result.ready, time);
      return last[index] = result;
    }
    output missing(std::uint64_t time) {
      now = time;
      roster.observed_frame = {++frame, 99};
      selected_frame absent;
      absent.basis_epoch = roster.basis_epoch;
      const auto result = controller.update(roster, absent, nullptr, time);
      blend = transition.update(result.ready, time);
      return result;
    }
    void rotate_until(std::uint64_t end, bool moving = false) {
      for (std::uint64_t time = 0; time <= end; time += 10) {
        const unsigned index = unsigned(time / 10 % roster.count);
        if (moving) t[index] = .25f + .002f * float(time * .001);
        const bool take = time >= due[index];
        if (take) due[index] = time + 250;
        tick(time, index, take);
      }
    }
  };

  void short_rotations_keep_independent_basis_and_continuous_reentry() {
    for (unsigned count : {2u, 3u, 4u}) {
      fixture value(count);
      value.rotate_until(1800);
      for (unsigned i = 0; i < count; ++i) {
        require(value.last[i].ready, "Short resource rotation repeatedly lost initialization");
        require(value.last[i].calibration_samples == 4, "Member shared or restarted the startup window");
        close(value.last[i].H, 1 / value.t[i], 0, "Resource inherited another member's H");
        close(value.last[i].t0, value.t[i], 0, "Resource inherited another member's zero plane");
      }
      close(value.blend, 1, 0, "Continuous ready rotation restarted global reentry");
    }
  }

  void real_gap_pauses_every_member_without_rendering_cached_depth() {
    fixture value;
    value.rotate_until(1800);
    require(!value.missing(1810).ready && value.blend == 0, "Missing current depth retained stereo");
    for (unsigned i = 0; i < 3; ++i) {
      const auto waiting = value.tick(1820 + i * 10, i);
      require(waiting.ready, "A brief missing present discarded fresh numerical evidence");
      close(waiting.H, 1 / value.t[i], 0, "A depth gap erased established numeric state");
    }
    for (unsigned i = 0; i < 3; ++i) {
      require(value.tick(1860 + i * 10, i, true).ready, "Fresh post-gap capture failed to recover member");
      close(value.last[i].H, 1 / value.t[i], 0, "Recovery recalibrated an established member");
    }
    require(value.blend == 1, "A brief missing present restarted established strength");

    fixture startup;
    startup.rotate_until(600);
    startup.missing(610);
    for (unsigned i = 0; i < 3; ++i) {
      const auto state = startup.tick(620 + i * 10, i, true);
      require(!state.ready && state.calibration_samples == 3, "A short gap discarded incomplete peer calibration");
    }
  }

  void capture_slot_omission_preserves_only_exact_history() {
    fixture value;
    value.rotate_until(1800);
    const auto old_roster = value.roster;
    // Pinning the same resource shrinks membership; it does not reset that key.
    value.roster.count = 1;
    ++value.roster.routing_epoch;
    require(value.tick(1810, 0).ready, "Same-source pin discarded its calibration");
    close(value.last[0].H, 8, 0, "Same-source pin changed H");
    auto omitted = value.current(1);
    auto unadmitted = capture(omitted, 1000000, 1820, .5f);
    require(value.controller.update(value.roster, omitted, &unadmitted, 1820).reason == status::source_mismatch,
      "Historical membership alone admitted an omitted source's pixels or samples");
    value.roster = old_roster;
    value.roster.routing_epoch += 2;
    require(value.tick(1830, 0).ready, "Same-source unpin discarded retained calibration");
    require(value.tick(1840, 1).ready && value.last[1].calibration_samples == 4,
      "Temporary capture-slot omission reinitialized a still-live exact source");
    close(value.last[1].H, 4, 0, "Returning physical source inherited another source's gain");
    close(value.last[1].t0, .25, 0, "Returning source spent inactive-time zero-plane credit");
    require(value.tick(1850, 1, true).ready, "Ignored high-ID omitted packet poisoned fresh reentry");

    for (unsigned change = 0; change < 7; ++change) {
      fixture changed(2);
      changed.rotate_until(1800);
      const auto original = changed.roster.members[0];
      auto &basis = changed.roster.members[0];
      if (change == 0) ++basis.source.lifetime; // Native address reused.
      if (change == 1) ++basis.layout_epoch;
      if (change == 2) { basis.source.left = 1; --basis.source.extent_width; }
      if (change == 3) basis.direction = orientation::normal;
      if (change == 4) ++basis.convention_epoch;
      if (change == 5) { ++basis.source.width; ++basis.source.extent_width; }
      if (change == 6) { ++basis.source.height; ++basis.source.extent_height; }
      require(changed.tick(1810, 1).ready, "Unrelated member lost state after roster replacement");
      const auto replacement = changed.tick(1820, 0);
      require(!replacement.ready && !replacement.calibrated && replacement.calibration_samples == 0,
        "Changed physical basis inherited numeric state or an old packet");
      basis = original;
      const auto restored = changed.tick(1830, 0);
      require(!restored.ready && !restored.calibrated && restored.calibration_samples == 0,
        "Returning to a superseded basis resurrected its discarded history");
    }
  }

  void omitted_history_holds_reference_after_expiry_but_requires_fresh_targets() {
    fixture value(2);
    value.rotate_until(1800);
    value.roster.count = 1;
    for (std::uint64_t time = 1810; time <= 4000; time += 10)
      value.tick(time, 0, time % 250 == 0);
    value.roster.count = 2;
    const auto held = value.tick(4010, 1);
    require(held.ready && held.reason == status::holding_reference && held.calibration_samples == 4,
      "Expired omitted history either restarted initialization or pretended its old target was fresh");
    close(held.H, 4, 0, "Omission changed the retained source's scale");
    close(held.t0, .25, 0, "Omission changed the retained source's zero");
    const auto duplicate = value.tick(4020, 1);
    require(duplicate.reason == status::holding_reference, "Duplicate packet refreshed an expired target");
    value.t[1] = .125f;
    const auto resumed = value.tick(4030, 1, true);
    require(resumed.ready && resumed.reason == status::ready && resumed.calibration_samples == 4,
      "Fresh target forced an initialized retained source through startup again");
    close(resumed.H, held.H, 0, "Fresh reentry spent omitted capture-time gain credit");
    close(resumed.t0, held.t0, 0, "Fresh reentry spent omitted motion-time credit");
    require(!value.missing(4040).ready, "Held numerical history fabricated missing current depth");
  }

  void authoritative_inventory_prunes_destroyed_history_before_reentry() {
    fixture value(2);
    value.rotate_until(1800);
    const auto history = value.controller.history_sources();
    std::uint32_t live_mask = 0;
    unsigned found = 0;
    for (std::size_t i = 0; i < history.size(); ++i) {
      if (history[i].source.native == value.roster.members[1].source.native) { ++found; continue; }
      if (history[i].source.native) live_mask |= std::uint32_t{1} << i;
    }
    require(found == 1, "Live inventory query did not expose one exact cached member");
    value.controller.retain_history(live_mask);
    require(!value.controller.evaluate(value.current(0), 1810).ready,
      "Applying an inventory mask left the prior synchronization open");
    require(value.tick(1810, 0).ready, "Pruning destroyed peer erased unrelated calibration");
    const auto readmitted = value.tick(1820, 1);
    require(!readmitted.ready && !readmitted.calibrated && readmitted.calibration_samples == 0,
      "Authoritatively removed source revived from cached state or pre-admission packet");
    require(value.tick(1830, 1, true).calibration_samples == 1,
      "Readmitted source failed to require new exact-source measurements");
    auto bad = value.roster;
    bad.members[1].source.native = bad.members[0].source.native;
    require(!value.controller.synchronize(bad, 1840), "Two simultaneous lifetimes of the same native resource were accepted");
  }

  void history_is_bounded_and_never_evicts_an_admitted_peer() {
    fixture value(2);
    value.rotate_until(1800);
    const auto original_peer = value.current(1);
    for (unsigned i = 0; i < maximum_history_sources; ++i) {
      value.roster.members[1].source.native = 0x200 + i;
      value.roster.members[1].source.lifetime = 500 + i;
      require(value.tick(1810 + i * 10, 0).ready, "Cache pressure evicted an admitted initialized source");
      close(value.last[0].H, 8, 0, "Cache pressure changed the admitted source's H");
    }
    unsigned retained = 0;
    for (const auto &basis : value.controller.history_sources()) retained += basis.source.native != 0;
    require(retained == maximum_history_sources && !value.controller.contains(original_peer),
      "History did not stay bounded by evicting its oldest inactive source");
    const auto first = value.roster.members[0];
    // Place the retained old member after a new allocation in the roster. All
    // current members must be protected before an LRU slot is chosen.
    value.roster.members[0] = value.roster.members[1];
    value.roster.members[0].source.native = 0x900;
    value.roster.members[0].source.lifetime = 900;
    value.roster.members[1] = first;
    require(value.tick(2000, 1, false, false).ready, "Roster order evicted the later admitted peer");
    close(value.last[1].H, 8, 0, "Roster order changed a retained source's independent basis");
  }

  void incomplete_history_expires_while_capture_slot_is_omitted() {
    fixture value(2);
    for (unsigned i = 0; i < 3; ++i) value.tick(i * 250, 1, true);
    require(value.last[1].calibration_samples == 3, "Incomplete startup control is invalid");
    value.roster.count = 1;
    for (unsigned time = 510; time <= 2100; time += 10) value.tick(time, 0);
    value.roster.count = 2;
    const auto returned = value.tick(2110, 1);
    require(!returned.calibrated && returned.calibration_samples == 0,
      "Omitted partial startup retained evidence beyond its expiry");
    const auto fresh = value.tick(2120, 1, true);
    require(!fresh.ready && fresh.calibration_samples == 1, "Incomplete history skipped fresh initialization after expiry");
  }

  void unknown_roster_member_does_not_block_valid_peer() {
    fixture value(2);
    value.roster.members[1].direction = orientation::automatic;
    for (unsigned i = 0; i < 4; ++i) value.tick(i * 250, 0, true);
    require(value.last[0].ready, "Unknown peer orientation blocked an independently valid current source");
    const auto unknown = value.tick(760, 1, true);
    require(!unknown.ready && unknown.reason == status::direction_unknown,
      "Unknown source orientation was fabricated from its peer");
    require(value.tick(770, 0).ready, "Unknown peer discarded the valid source's calibration");
    value.roster.members[1].direction = orientation::reversed;
    require(value.tick(780, 1).calibration_samples == 0, "New convention reused an unknown-orientation capture");
  }

  void epoch_cut_and_global_order_guards_survive_rotations() {
    fixture value;
    value.rotate_until(1800);
    value.controller.scene_cut(1810);
    for (unsigned i = 0; i < 3; ++i)
      require(!value.tick(1820 + i * 10, i).ready, "Explicit cut retained a peer's old target");
    for (unsigned i = 0; i < 3; ++i) value.tick(1860 + i * 10, i, true);
    auto backwards = value.current(1);
    backwards.frame.frame = 1;
    require(value.controller.update(value.roster, backwards, nullptr, 1900).reason == status::frame_mismatch,
      "Cross-member backward current-frame order was accepted");
    require(value.tick(1910, 0).ready, "Rejected presentation metadata erased another member's valid evidence");
    require(value.tick(1920, 0, true).ready, "Fresh capture did not recover from global order error");
    require(value.controller.update(value.roster, value.current(1), nullptr, 1919).reason == status::clock_went_backwards,
      "Cross-member clock rollback was accepted");
    require(!value.tick(1930, 0).ready, "Clock rollback preserved another member's live target");

    ++value.roster.basis_epoch;
    require(value.controller.reset(value.roster.basis_epoch, 1940), "Explicit recalibration epoch rejected");
    for (unsigned i = 0; i < 3; ++i) {
      const auto state = value.tick(1950 + i * 10, i);
      require(!state.ready && !state.calibrated && state.calibration_samples == 0,
        "Explicit recalibration accepted old epoch state or packet");
    }
  }

  void stale_wrong_basis_and_replayed_packets_cannot_publish() {
    fixture value(2);
    value.rotate_until(1800);
    auto current = value.current(0);
    const auto saved = value.last[0];
    auto wrong = value.latest[1];
    wrong.id = 1000000;
    require(value.controller.update(value.roster, current, &wrong, 1810).ready,
      "Wrong-source packet disturbed a useful current target");
    value.tick(1820, 0, true);
    close(value.last[0].H, saved.H, 0, "Ignored wrong-source packet poisoned valid sample ordering");
    const auto duplicate = value.latest[0];
    for (std::uint64_t now = 1830; now <= 3400; now += 10) {
      // Do not supply new packets for either member. Current color/depth remains
      // valid, but repeated captures cannot extend the numeric target lifetime.
      const unsigned index = unsigned(now / 10 % 2);
      value.tick(now, index);
    }
    require(value.last[0].reason == status::holding_reference && value.last[1].reason == status::holding_reference,
      "Duplicate packets kept expired targets fresh instead of holding only initialized references");
    require(value.controller.update(value.roster, value.current(0), &duplicate, 3410).reason == status::holding_reference,
      "Stale packet revived numerical adaptation after expiry");

    // A malformed matching-source readback is different from an unrelated packet.
    current = value.current(0);
    auto malformed = capture(current, ++value.id, 3420, .125f);
    ++malformed.readback_source.lifetime;
    require(value.controller.update(value.roster, current, &malformed, 3420).reason == status::source_mismatch,
      "Readback identity mismatch was ignored or rendered");
    require(value.tick(3430, 0, true).ready, "New exact-source packet failed to recover after malformed readback");
  }

  void long_member_absence_and_real_presentation_gap_require_fresh_evidence() {
    fixture value(2);
    value.rotate_until(1800);
    for (std::uint64_t time = 1810; time <= 2200; time += 10) value.tick(time, 0, time % 250 == 0);
    require(value.tick(2210, 1).ready, "A current fresh copy could not use unexpired numerical evidence");
    require(value.tick(2220, 1, true).ready, "Dormant established member lost its own H");
    require(value.tick(2600, 0).ready, "Short presentation absence discarded unexpired evidence");
    require(value.tick(2610, 1).ready, "Another member lost unexpired evidence");
    require(value.tick(3800, 0).reason == status::holding_reference && value.tick(3810, 1).reason == status::holding_reference,
      "Actual target expiry did not distinguish retained reference from fresh adaptation evidence");
  }

  void independent_observation_survives_missing_presents_and_offturn_sources() {
    fixture value(3);
    value.current(0);
    require(value.controller.synchronize(value.roster, 0), "Initial authoritative roster was rejected");
    selected_frame missing;
    missing.basis_epoch = value.roster.basis_epoch;
    for (unsigned i = 0; i < 4; ++i) {
      const std::uint64_t captured_at = 10 + i * 250;
      std::array<sample, 3> pending;
      for (unsigned member = 0; member < 3; ++member)
        pending[member] = capture(value.current(member), ++value.id, captured_at, value.t[member]);
      // The in-flight measurements were captured under an already-known roster.
      value.current(0);
      require(value.controller.synchronize(value.roster, captured_at + 10), "Missing-present roster synchronization failed");
      require(!value.controller.evaluate(missing, captured_at + 10).ready, "Missing pixels were made current");
      value.current(0);
      require(value.controller.synchronize(value.roster, captured_at + 20), "Readback-completion synchronization failed");
      for (const auto &packet : pending) value.controller.observe(packet, captured_at + 20);
      require(!value.controller.evaluate(missing, captured_at + 20).ready, "Completed numeric evidence fabricated current depth");
      for (unsigned member = 0; member < 3; ++member) {
        const auto current = value.current(member);
        value.controller.synchronize(value.roster, captured_at + 20);
        const auto state = value.controller.evaluate(current, captured_at + 20);
        require(state.calibration_samples == i + 1 && state.ready == (i == 3),
          "Missing selected depth discarded an exact off-turn asynchronous measurement");
        if (state.ready) close(state.H, 1 / value.t[member], 0, "Off-turn sources shared their raw coordinate scales");
      }
    }
    auto current = value.current(0);
    value.controller.synchronize(value.roster, 800);
    auto future = capture(current, 1000000, 800, .5f);
    future.metadata.frame.frame += 10;
    future.readback_frame = future.metadata.frame;
    value.controller.observe(future, 800);
    auto other_generation = future;
    other_generation.metadata.frame = current.frame;
    ++other_generation.metadata.frame.token_generation;
    other_generation.readback_frame = other_generation.metadata.frame;
    value.controller.observe(other_generation, 800);
    // A legitimate lower-ID sample proves rejected packets did not poison order.
    const auto real = capture(current, ++value.id, 800, .25f);
    value.controller.observe(real, 800);
    const auto refined = value.controller.evaluate(current, 800);
    require(refined.ready && refined.H < 8 && refined.H > 4 && refined.target_H == 4 && refined.t0 > .125f,
      "Future or wrong-generation packet poisoned ordinary sample admission");
    close(double(refined.H) * refined.t0, 1, 2e-7,
      "Off-turn observation left scale detached from the updated zero");

    auto malformed = capture(value.current(0), ++value.id, 850, .25f);
    ++malformed.readback_source.lifetime;
    value.controller.synchronize(value.roster, 850);
    value.controller.observe(malformed, 850);
    require(!value.controller.evaluate(malformed.metadata, 850).ready, "Malformed matching sample retained its target");
    const auto peer = value.current(1);
    value.controller.synchronize(value.roster, 850);
    require(value.controller.evaluate(peer, 850).ready, "Malformed one-source sample erased another source's evidence");
  }

  void failed_authority_sync_closes_admission_and_presentation_errors_do_not_erase_evidence() {
    fixture value(2);
    value.rotate_until(1800);
    auto current = value.current(0);
    value.controller.synchronize(value.roster, 1810);
    auto stale = current;
    --stale.frame.frame;
    require(!value.controller.evaluate(stale, 1810).ready, "Older-than-current capture watermark rendered stale depth");
    auto bogus = current;
    ++bogus.source.native;
    require(!value.controller.evaluate(bogus, 1810).ready, "Unrostered current source rendered");
    require(value.controller.evaluate(current, 1810).ready, "Bogus presentation selection erased observed evidence");
    auto bad_roster = value.roster;
    bad_roster.observed_frame.token_generation = 0;
    require(!value.controller.synchronize(bad_roster, 1820), "Zero authoritative token was accepted");
    const auto ignored = capture(current, 1000000, 1820, .5f);
    value.controller.observe(ignored, 1820);
    require(!value.controller.evaluate(current, 1820).ready, "Failed synchronization left an old roster open");
    require(value.controller.synchronize(value.roster, 1830), "Correct authority did not reopen admission");
    const auto fresh = capture(value.current(0), ++value.id, 1840, .25f);
    value.controller.synchronize(value.roster, 1840);
    value.controller.observe(fresh, 1840);
    require(value.controller.evaluate(fresh.metadata, 1840).ready, "Ignored high ID during failed sync poisoned recovery");
  }

  void same_coordinate_moving_scene_has_bounded_member_phase_difference() {
    fixture value;
    for (auto &basis : value.roster.members) basis.direction = orientation::reversed;
    output previous;
    for (std::uint64_t time = 0; time <= 5000; time += 10) {
      const unsigned index = unsigned(time / 10 % 3);
      value.t[index] = .25f + .002f * float(time * .001);
      const bool take = time >= value.due[index];
      if (take) value.due[index] = time + 250;
      const auto state = value.tick(time, index, take);
      if (time >= 1000) {
        require(state.ready, "A moving coherent scene lost readiness during rotation");
        // A 0.002 raw-units/s ramp has slightly different sample phases for each
        // member. These broad bounds catch cross-member reset/scale spikes,
        // without asserting that independently sampled controllers are identical.
        if (previous.ready) {
          require(std::abs(state.H / previous.H - 1) < .005f, "Member scale phase jump exceeded 0.5 percent");
          require(std::abs(state.t0 - previous.t0) < .002f, "Member zero-plane phase jump exceeded one second of motion");
        }
      }
      previous = state;
    }
    close(value.blend, 1, 0, "Moving ready scene repeatedly restarted reentry");
  }

  void rotating_members_follow_their_current_zero_through_room_wall_room() {
    fixture value;
    value.rotate_until(1800);
    for (const float center : {.0001f, .75f, .25f}) {
      const auto end = value.now + 12000;
      for (auto time = value.now + 10; time <= end; time += 10) {
        const unsigned index = unsigned(time / 10 % value.roster.count);
        value.t[index] = center;
        const bool take = time >= value.due[index];
        if (take) value.due[index] = time + 250;
        const auto state = value.tick(time, index, take);
        require(state.ready && state.calibration_samples == 4,
          "Room/wall changes recalibrated an established rotating source");
        close(double(state.H) * state.t0, 1, 2e-7,
          "Room/wall movement detached scale from the current zero plane");
        // Scene changes legitimately change separation of fixed raw depths.
        // Equal ratios to the CURRENT zero must still give equal warp fields.
        const double at_half_zero = state.referenceZPD * state.H * (state.t0 - .5 * state.t0);
        close(at_half_zero, .5 * state.referenceZPD, 1e-8,
          "Equal relative depth changed separation as the zero moved");
      }
      for (unsigned i = 0; i < value.roster.count; ++i) {
        close(value.last[i].t0, center, .001,
          "Rotating sources failed to converge to the fresh scene zero");
        close(double(value.last[i].H) * center, 1, .003,
          "Rotating source scale stayed at its startup value after zero converged");
      }
    }
    close(value.blend, 1, 0, "Room/wall rotation restarted the shared reentry");
  }
}

int main() {
  try {
    short_rotations_keep_independent_basis_and_continuous_reentry();
    real_gap_pauses_every_member_without_rendering_cached_depth();
    capture_slot_omission_preserves_only_exact_history();
    omitted_history_holds_reference_after_expiry_but_requires_fresh_targets();
    authoritative_inventory_prunes_destroyed_history_before_reentry();
    history_is_bounded_and_never_evicts_an_admitted_peer();
    incomplete_history_expires_while_capture_slot_is_omitted();
    unknown_roster_member_does_not_block_valid_peer();
    epoch_cut_and_global_order_guards_survive_rotations();
    stale_wrong_basis_and_replayed_packets_cannot_publish();
    long_member_absence_and_real_presentation_gap_require_fresh_evidence();
    independent_observation_survives_missing_presents_and_offturn_sources();
    failed_authority_sync_closes_admission_and_presentation_errors_do_not_erase_evidence();
    same_coordinate_moving_scene_has_bounded_member_phase_difference();
    rotating_members_follow_their_current_zero_through_room_wall_room();
    std::puts("Raw scene pool: 15 cases passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Raw scene pool failed: %s\n", error.what());
    return 1;
  }
}
