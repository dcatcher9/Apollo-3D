// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_decision.h"
#include "ui_evidence.h"

namespace sunshine_game3d {
  ui_decision::requested_mode ui_decision::request(const alpha_auto_source *automatic, bool has_detection) {
    const auto mode = automatic && automatic->session ? automatic->session->decision().state : alpha_auto_state::manual_off;
    const bool auto_mode = automatic && automatic->session && mode != alpha_auto_state::manual_on && mode != alpha_auto_state::manual_off;
    const bool manual_detection = automatic && automatic->session && mode == alpha_auto_state::manual_on && has_detection;
    return {mode, auto_mode, manual_detection};
  }
  ui_decision::detection_policy ui_decision::detection_request(bool auto_mode, bool manual_detection, bool hudless,
      const alpha_auto_source *automatic, alpha_auto_state mode, const ui_temporal::hold_decision &arbitration, uint32_t bits) {
    const bool needs_detection = auto_mode || manual_detection || hudless;
    const bool detection_requested = needs_detection && (!automatic || mode != alpha_auto_state::manual_off);
    const bool grace_only = arbitration.detect && !bits &&
      (arbitration.per_frame & ui_detection::per_frame_accepted_missing);
    const bool runs = arbitration.hold || (arbitration.detect && (bits || grace_only));
    return {needs_detection, detection_requested, runs};
  }
  uint32_t ui_decision::accepted(const alpha_auto_source *automatic, uint32_t bits, const candidate_signatures &signatures) {
    return automatic && automatic->session ? automatic->session->accepted(bits, signatures) : bits & ui_selection::candidate_bits;
  }
  ui_temporal::hold_decision ui_decision::arbitrate(bool hold_previous, const alpha_auto_source &observation, uint32_t bits) const {
    return temporal.arbitrate({hold_previous}, observation, bits);
  }
  void ui_decision::adopt(const ui_temporal::hold_decision &arbitration, uint32_t bits, uint32_t accepted) {
    if (arbitration.adopt) temporal.adopt(bits, accepted);
  }
  void ui_decision::resume_scope() {
    temporal = {};
    guard = {};
  }
  void ui_decision::begin_frame(bool detection_requested) {
    scene_bits = 0;
    scene_layer_proven = false;
    if (detection_requested) ++cpu_counts[ui_counter::auto_frames];
  }
  void ui_decision::scene_frame(const alpha_auto_source &observation, uint32_t bits,
      const candidate_signatures &signatures, const alpha_auto_source *automatic) {
    namespace candidate = ui_detection::candidate;
    using ui_selection::kind;
    scene_layer_proven = (bits & candidate::layer) && automatic && automatic->session &&
      automatic->session->pre_ui_proven(signatures.of(kind::ui_layer));
    scene_bits = guard.per_frame(observation.now_ms, bits, signatures.by_kind(), scene_layer_proven);
  }
  void ui_decision::held() {
    ++cpu_counts[ui_counter::held_generated];
    temporal.held();
  }
  void ui_decision::enter_scope(const alpha_auto_source &observation) {
    temporal.enter_scope(observation);
    guard.enter_scope(observation.epoch, observation.viewport);
  }
  uint32_t ui_decision::per_frame(bool hudless_reoffer, uint32_t bits, uint32_t unaligned_declared,
      const ui_temporal::hold_decision &arbitration, bool depth_current) const {
    namespace candidate = ui_detection::candidate;
    const bool reoffer = hudless_reoffer && (bits & candidate::hudless) && !(bits & candidate::exact);
    const uint32_t unaligned = (bits & candidate::exact) ?
      unaligned_declared & bits & (candidate::ui_alpha | candidate::ui_color) : 0u;
    const uint32_t per_frame = scene_bits | arbitration.per_frame |
      (!depth_current ? ui_detection::per_frame_depth_not_current : 0u) |
      (reoffer ? ui_detection::per_frame_reoffer : 0u) | (unaligned << ui_detection::per_frame_unaligned_shift);
    return per_frame;
  }
  void ui_decision::detected(const alpha_auto_source &observation) { temporal.detected(observation); }
  void ui_decision::unavailable() {
    ++cpu_counts[ui_counter::held_none];
    temporal.unavailable();
  }
  void ui_decision::inactive(bool detection_requested, std::size_t inactive_reason) {
    if (detection_requested) ++cpu_counts[inactive_reason];
    temporal.inactive();
  }
  bool ui_decision::sample_discarded(const alpha_auto_source &input) const {
    return ui_temporal::sample_discarded(input, detection_pending_source);
  }
  scene_guard::observation ui_decision::observe_sample(const uint32_t *counts, std::size_t count,
      uint64_t detection_submitted, alpha_auto_policy *session) {
    auto &latest = temporal.latest;
    latest = ui_temporal::decode_detection_sample(counts, count, detection_pending_source.now_ms, detection_submitted);
    temporal.latest_source = detection_pending_source;
    temporal.latest_key = detection_pending_status_key;
    // The scene guard reads the decoded sample before the acceptance ledger observes it.
    const auto observed = guard.observe(ui_temporal::guard_sample(latest), detection_pending_actionable,
      detection_pending_signatures.by_kind());
    if (session)
      session->observe(latest.evidence, latest.pixels, detection_pending_source.now_ms, detection_pending_signatures);
    return observed;
  }
  void ui_decision::commit_counters(const alpha_auto_decision &sample, alpha_auto_policy *session,
      const scene_guard::observation &observed, const std::array<uint32_t, ui_counter_word::count> &words) {
      const auto delta = ui_temporal::sample_counters(sample, pending_counts, committed_counts, words, committed_words,
        detection_pending_source.now_ms, observed);
      committed_counts = pending_counts;
      committed_words = words;
      if (session) session->add_counters(delta);
  }
  bool ui_decision::measure_scene(uint64_t now, bool proven_image) { return guard.measure(now, proven_image); }
  void ui_decision::update_alpha_auto(bool eligible, const alpha_auto_source *automatic,
    bool detection_active, bool &source_alpha_ui) {
    consumed_auto = {};
    if (!automatic || !automatic->session) {
      // Explicit offline/replay input is already a resolved choice. An
      // incomplete live observation never acquires that authority.
      source_alpha_ui = !automatic && eligible;
      consumed_auto.enabled = source_alpha_ui;
      consumed_auto.state = source_alpha_ui ? alpha_auto_state::manual_on : alpha_auto_state::manual_off;
      return;
    }
    const auto requested = automatic->session->decision().state;
    const bool manual_on = requested == alpha_auto_state::manual_on;
    if ((manual_on && !detection_active) || requested == alpha_auto_state::manual_off) {
      // Manual Off, or manual On through the explicit first filtered
      // candidate where detection cannot run.
      consumed_auto.state = requested;
      source_alpha_ui = eligible && manual_on;
      consumed_auto.enabled = source_alpha_ui;
      return;
    }
    // Current-frame GPU validation chooses the mask. CPU readback is only
    // a bounded status sample, never authority for a later input image;
    // poll_detection only updates which sources the session accepts.
    // Manual On through detection reports the same sample as manual_on.
    source_alpha_ui = eligible;
    consumed_auto = temporal.latest;
    // F1: the latest sample describes this frame while it is fresh, in this
    // scope and taken under the current winner.
    if (!temporal.status_fresh(*automatic)) {
      consumed_auto = {};
      consumed_auto.state = eligible ? alpha_auto_state::collecting : alpha_auto_state::waiting_for_source;
    }
    if (manual_on) {
      consumed_auto.state = alpha_auto_state::manual_on;
      consumed_auto.enabled = source_alpha_ui;
    }
    consumed_auto.scene_guard = {(scene_bits & ui_detection::per_frame_scene_hidden) != 0,
      (scene_bits & ui_detection::per_frame_pre_ui_visible) != 0, uint32_t(guard.refuted_count), scene_layer_proven};
  }

  bool ui_decision::generated_hold(bool capturing, bool available, uint64_t frame_sequence, uint64_t input_present,
      bool fg_active, uint32_t generated_frames) {
    return capturing && !available && ui_evidence::generated_without_input(frame_sequence, input_present, fg_active, generated_frames);
  }
  bool ui_decision::snapshot_hold(bool capturing, bool available, bool hold_previous,
      const ui_evidence::snapshot_inputs &offered_inputs, const ui_evidence::snapshot_inputs &detected_before,
      bool fg_active, bool own_input) {
    return capturing && available && !hold_previous && ui_evidence::same_snapshot_inputs(offered_inputs, detected_before, fg_active, own_input);
  }
}
