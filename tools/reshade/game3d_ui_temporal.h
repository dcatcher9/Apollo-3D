// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The CPU-side temporal state of live automatic UI detection: which Presents
// detect and which show the decision of the real frame they show (T1), which
// inputs a real frame adopts, when a completed status sample still describes
// the frame (F1), and the CPU-owned hidden-scene verdict of each route, plus
// the decode of a completed sample's decision texels and the counts it
// commits. docs/reshade-sbs.md (UI protection, hidden-scene evidence, UI
// counters, UI decision framework) owns the rules. The renderer keeps only the
// GPU and resource work; the sequence replay test (test_game3d_ui_sequence.cpp)
// drives the same calls without a GPU, in the same order:
//
//   render (one Present; Auto, or manual On through detection):
//     1. bits: the offered candidate bits (ui_detection::candidate), and the
//        candidate_signatures of the offered kinds; identity: the provider's
//        present_identity, {hold_previous, real_frame} (Present counting until
//        S3 stamps real frames: generated from the HUD-less pairing, or,
//        without one, a Present that offers nothing within the reported
//        generated count of the last one that offered a UI tag
//        (ui_mask::generated_without_input); the HUD-less tag's present
//        generation as the real-frame id, 0 without a HUD-less capture);
//     2. accepted = session.accepted(bits, signatures) (manual On: every
//        offered candidate, without the ledger);
//     3. t = state.arbitrate(identity, observation, bits);
//     4. flags are the layer's stored flags when bits has candidate::layer,
//        else 0; when t.adopt, state.adopt(bits, flags, accepted);
//     5. detection runs when it is requested, the frame fits and its resources
//        are prepared, and either t.hold, or t.detect with bits != 0 or
//        t.per_frame & per_frame_accepted_missing (a zero-offer real frame:
//        the reduce and mask passes only, no tiles pass and no sample). A
//        generated hold and a zero-offer frame apply the mask even while no
//        UI input is available:
//        - t.hold (kind generated): apply the detected mask as it is (the
//          decision of the real frame shown), count held.generated, then
//          state.held(identity); no poll and no sample;
//        - t.kind unavailable (a generated Present with no such decision): no
//          mask, count held.none, then state.unavailable();
//        - t.detect: state.enter_scope(observation), poll_detection,
//          scene_holds and measure_scene; then detection with this frame's
//          own bits, accepted and flags pushed (b2), flags | per_frame where
//          per_frame = scene hold bits | t.per_frame | depth_not_current,
//          and the hold store bound at u5 for the reduce; then
//          state.detected(observation, identity). A sample keeps the
//          signatures and pushed flags it was submitted with, and its status
//          key is state.status_key() at submission;
//        otherwise inactive.* and state.inactive().
//   poll_detection (a completed sample):
//     6. sample_discarded(input, pending source) drops it unread (another
//        scope, or stale on arrival); otherwise
//        sample = decode_detection_sample(words, count, tick, sequence, scene,
//        pushed flags),
//        state.latest = sample, state.latest_source = the pending source and
//        state.latest_key = the status key at submission;
//     7. state.observe_scene(sample, state.pending_scene_actionable);
//     8. session.observe(sample.evidence, sample.pixels, tick, submitted
//        signatures); a session in a manual mode ignores it;
//     9. sample_counters() commits the counts.
//   status (after render):
//    10. state.latest describes the frame only while
//        state.status_fresh(observation); otherwise the status is collecting.
//
// The GPU owns the T1 grace of a real frame without a decision of its own
// (ui_selection::decide, the hold store): this state only names the Presents
// that detect, the per-frame bits a detection pushes, and the Presents that
// show a real frame's decision. No ReShade dependency.
#include "game3d_alpha_auto.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sunshine_game3d::ui_temporal {
  // A completed status sample describes later frames for this long after its
  // tick; older, from the future or never taken, it describes nothing.
  inline constexpr std::uint64_t sample_fresh_ms = 500;
  constexpr bool sample_stale(std::uint64_t sample_tick_ms, std::uint64_t now_ms) {
    return !sample_tick_ms || now_ms < sample_tick_ms || now_ms - sample_tick_ms > sample_fresh_ms;
  }
  static_assert(sample_stale(0, 0) && !sample_stale(1000, 1500) && sample_stale(1000, 1501) && sample_stale(1000, 999));

  // The inputs that identify a detection's scope: a sample, a decision or a
  // held route from another epoch, revision or viewport is no evidence for
  // this one.
  inline bool scope_changed(const alpha_auto_source &a, const alpha_auto_source &b) {
    return a.epoch != b.epoch || a.revision != b.revision || a.viewport != b.viewport;
  }
  // The identity part of that scope (M1: a recreated swapchain or device, or
  // another viewport), which clears T1 holds. An observation revision alone
  // (a depth observation loss) is a missing input, not another identity: the
  // last real decision still describes the same output, so a generated
  // Present holds it and the next real frame without a decision of its own
  // reuses it once.
  inline bool hold_scope_changed(const alpha_auto_source &a, const alpha_auto_source &b) {
    return a.epoch != b.epoch || a.viewport != b.viewport;
  }
  // A completed sample is discarded unread only when its scope no longer
  // matches, or when it is stale on arrival (F1). A change of accepted or
  // offered candidates does not discard it: every sample in scope is
  // evidence for the ledger, and the status shows it only under the winner
  // it was taken for (detection_state::status_fresh).
  inline bool sample_discarded(const alpha_auto_source &input, const alpha_auto_source &pending) {
    return scope_changed(input, pending) || input.now_ms < pending.now_ms || input.now_ms - pending.now_ms > sample_fresh_ms;
  }

  // The status sample poll_detection reads from the decision texels `words`
  // (4 per texel, at least min_decision_texels; with scene, the evidence
  // texels 5 and 6 too; the offscreen UI layer's texel 7 when there are
  // layer_decision_texels; the one-way judgment counts, the refused candidate
  // and the frame reason of texels 8 and 9 when there are
  // judgment_decision_texels). sequence numbers the submitted samples, and
  // flags are the Sunshine_UIDetectionFlags the detection pushed (whether an
  // offered layer was the one-frame-late copy).
  inline alpha_auto_decision decode_detection_sample(const std::uint32_t *words, std::size_t count,
      std::uint64_t tick_ms, std::uint64_t sequence, bool scene, std::uint32_t flags = 0) {
    namespace word = ui_detection::decision_word;
    alpha_auto_decision sample;
    if (count < 4 * ui_detection::min_decision_texels ||
        (scene && count < 4 * ui_detection::scene_decision_texels)) return sample;
    sample.source_kind = words[word::source];
    sample.enabled = words[word::source] != 0;
    sample.state = words[word::source] ? alpha_auto_state::automatic_on : alpha_auto_state::automatic_off;
    sample.covered = words[word::covered]; sample.pixels = words[word::pixels];
    sample.sample_sequence = sequence;
    sample.sample_tick_ms = tick_ms;
    sample.accepted_samples = sequence;
    auto &evidence = sample.evidence;
    evidence.matching_tiles = words[word::matching_tiles];
    evidence.candidates = words[word::candidates]; evidence.hudless_changed = words[word::hudless_changed];
    evidence.hudless_unchanged = words[word::hudless_unchanged]; evidence.hudless_invalid = words[word::hudless_invalid];
    std::copy_n(words + word::alpha_covered, 4, evidence.alpha_covered.begin());
    std::copy_n(words + word::alpha_invalid, 4, evidence.alpha_invalid.begin());
    evidence.hudless_lit = words[word::hudless_lit];
    evidence.accepted = words[word::accepted];
    evidence.alpha_opaque = {words[word::alpha_opaque], words[word::alpha_opaque + 1]};
    evidence.late_layer = (evidence.candidates & ui_detection::candidate::layer) &&
      (flags & ui_detection::stored_late_layer);
    if (count >= 4 * ui_detection::layer_decision_texels) {
      evidence.layer_covered = words[word::layer_covered];
      evidence.layer_invalid = words[word::layer_invalid];
      evidence.layer_opaque = words[word::layer_opaque];
      evidence.valid_bits = words[word::valid_bits];
    }
    if (count >= 4 * ui_detection::judgment_decision_texels) {
      std::copy_n(words + word::strong, evidence.strong.size(), evidence.strong.begin());
      std::copy_n(words + word::contradicted, evidence.contradicted.size(), evidence.contradicted.begin());
      evidence.refused = words[word::refused];
      evidence.frame_reason = words[word::frame_reason] & ui_detection::frame_reason_reason_mask;
      evidence.reused = (words[word::frame_reason] & ui_detection::frame_reason_reused) != 0;
    }
    if (scene) {
      const auto decode = [&](std::size_t n, std::size_t d, std::size_t state) {
        alpha_auto_decision::scene_evidence result;
        result.n = words[n];
        std::memcpy(&result.d, &words[d], sizeof(result.d));
        result.valid = ui_detection::scene_state_valid(words[state]);
        result.ran = ui_detection::scene_state_ran(words[state]);
        result.verdict = ui_detection::scene_state_verdict(words[state]);
        return result;
      };
      evidence.scene = decode(word::scene_n, word::scene_d, word::scene_state);
      evidence.scene.decided = words[word::scene_decided];
      evidence.hudless_scene = decode(word::hudless_scene_n, word::hudless_scene_d, word::hudless_scene_state);
    }
    return sample;
  }

  // A sample that decided a whole-frame mask from alpha: a source 1-4 or the
  // offscreen UI layer (10) covering at least 99% of pixels (the GPU's
  // full_alpha counter word).
  inline bool full_alpha(const alpha_auto_decision &sample) {
    const bool alpha = (sample.source_kind >= 1 && sample.source_kind <= 4) || sample.source_kind == ui_detection::source_layer;
    return alpha && sample.pixels &&
      std::uint64_t(sample.covered) * 100 >= std::uint64_t(sample.pixels) * 99;
  }

  // The counts a completed sample commits (game3d_ui_counters.h): the CPU
  // counts snapshotted when it was submitted and the GPU words copied under
  // its fence, each as the change since the previous commit, and, when it
  // decided a full-frame route (6, 8, 9) or a whole-frame alpha, the
  // hidden-scene verdict that same sample measured.
  inline ui_counters sample_counters(const alpha_auto_decision &sample, const ui_counters &cpu,
      const ui_counters &committed_cpu, const std::array<std::uint32_t, ui_counter_word::count> &words,
      const std::array<std::uint32_t, ui_counter_word::count> &committed_words, std::uint64_t through_ms) {
    auto delta = cpu - committed_cpu;
    delta.add_gpu_delta(words, committed_words);
    const bool route = sample.source_kind == 6 || sample.source_kind == 8 || sample.source_kind == 9;
    if (route || full_alpha(sample)) {
      using ui_detection::scene_verdict;
      const auto &scene = sample.evidence.scene;
      const auto verdict = scene.valid ? scene.verdict : scene_verdict::none;
      const std::size_t hidden = route ? ui_counter::full_d_hidden : ui_counter::full_alpha_d_hidden;
      ++delta[hidden + (verdict == scene_verdict::hidden ? 0 : verdict == scene_verdict::ambiguous ? 1 :
        verdict == scene_verdict::visible ? 2 : 3)];
    }
    static_assert(ui_counter::full_d_invalid == ui_counter::full_d_hidden + 3 &&
      ui_counter::full_alpha_d_ambiguous == ui_counter::full_alpha_d_hidden + 1 &&
      ui_counter::full_alpha_d_visible == ui_counter::full_alpha_d_hidden + 2 &&
      ui_counter::full_alpha_d_invalid == ui_counter::full_alpha_d_hidden + 3);
    delta[ui_counter::samples] = 1;
    delta.through_ms = through_ms;
    return delta;
  }

  // Whether a sample frame runs the hidden-scene evidence passes, and whether
  // what they measure is actionable (may hold or release a route).
  struct scene_measurement {
    bool run{}, actionable{};
  };

  // T1 identity of one Present (M6), by Present counting until S3 stamps real
  // frames: whether the HUD-less pairing, or without one the count since the
  // last Present that offered a UI tag, classified it as generated
  // (hold_previous), and the real frame it shows, as the HUD-less tag's
  // present generation (0 without a HUD-less capture: no tag bound).
  struct present_identity {
    bool generated{};
    std::uint64_t real_frame{};
  };

  // What a Present does under T1: a real Present detects (none: no hold); a
  // generated Present shows the decision of the real frame it shows
  // (generated, counted held.generated) or, when no such decision exists in
  // its scope, has no mask (unavailable, counted held.none).
  enum class hold_kind : std::uint8_t { none, generated, unavailable };
  inline const char *name(hold_kind value) {
    switch (value) {
      case hold_kind::generated: return "generated";
      case hold_kind::unavailable: return "unavailable";
      default: return "none";
    }
  }
  struct hold_decision {
    hold_kind kind = hold_kind::none;
    // A real Present: it runs detection. A generated Present showing a real
    // frame's decision: it applies the detected mask without detecting.
    bool detect{}, hold{};
    // A real Present whose inputs become the adopted ones (the status key,
    // the hidden-scene route and the accepted-missing reference).
    bool adopt{};
    // The per-frame bits T1 pushes with a detection
    // (ui_detection::per_frame_hold_reset, per_frame_accepted_missing).
    std::uint32_t per_frame{};
  };

  // One renderer's detection state across frames. Candidate bits are
  // ui_detection::candidate's: 0x1 UIAlpha, 0x2 UI color tag, 0x4
  // Backbuffer, 0x8 current, 0x10 a paired HUD-less image, 0x20 an exact
  // pair, 0x40 the offscreen UI layer. Flags are the layer's stored
  // Sunshine_UIDetectionFlags; accepted the session's accepted candidates
  // (alpha_auto_policy::accepted), in the same bit positions.
  struct detection_state {
    // The inputs the last adopting real frame offered (adopt): they key the
    // status (status_key), the hidden-scene routes (scene_route_key, S2b) and
    // the accepted-missing reference.
    std::uint32_t bits{}, flags{}, accepted{};
    // T1 (M6): whether a real decision exists in this chain, its scope and
    // real-frame id, the first other real-frame id a generated Present
    // showed since (the next real frame; zero before), and whether the next
    // detection starts a new chain (per_frame_hold_reset).
    bool have_decision{};
    alpha_auto_source decision_scope;
    std::uint64_t decision_frame{}, next_frame{};
    bool reset_pending = true;
    // Consecutive generated Presents that applied the detected mask.
    std::uint32_t holds{};
    // The latest completed status sample, the inputs it was taken for and its
    // status key at submission (status_key()).
    alpha_auto_decision latest;
    alpha_auto_source latest_source;
    std::uint32_t latest_key{};
    // Hidden-scene verdicts the CPU holds: 0 layer route (source 8), 1
    // HUD-less route (source 9), each until this tick, zero when not held.
    std::array<std::uint64_t, 2> scene_hold_until{};
    // The latest sample's gate was open; the pending sample measured because a
    // gate was open or a hold active (not for the first-run shadow alone).
    bool scene_gate_open{}, pending_scene_actionable{};
    // The first sample tick of the current run of hidden samples without a
    // decided source; zero without a run.
    std::uint64_t shadow_run_start{};
    // The routes' inputs (ui_detection::scene_route_key), and the layer-route
    // inputs (1 UIAlpha, 2 UI layer) refuted as hiding the scene: nearly opaque
    // while valid evidence read the presented frame visible. A refuted slot
    // never holds the layer route until it is offered below that opacity.
    std::uint64_t scene_route{};
    std::uint32_t scene_refuted_slots{};

    // T1: a generated Present never detects. It shows the decision of the
    // real frame it shows: the last detection's (detected_mask) while it is
    // in the same identity scope (hold_scope_changed: an observation
    // revision alone keeps it) and the Present shows that real frame or the
    // next one
    // (its real-frame id equals the decision's, or is the first other id a
    // generated Present showed since; 0, no HUD-less capture, is no bound).
    // Otherwise it has no mask and the chain ends. No multiplier constant and
    // no time bound: a wrong frame-generation count fails safe. A real
    // Present always detects; its detection pushes per_frame_hold_reset when
    // no real decision exists in this chain (the first detection, an
    // identity scope change, an inactive frame or a Present without a mask
    // since), and
    // per_frame_accepted_missing when an accepted candidate the last adopting
    // real frame offered is missing now, in which case the GPU reuses the
    // previous real frame's decision once (ui_selection::decide) and the
    // frame adopts nothing. After an observation loss the previous
    // revision's captures are refused until the game tags again, so that
    // real frame is flagged too.
    hold_decision arbitrate(const present_identity &identity, const alpha_auto_source &scope, std::uint32_t offered) const {
      hold_decision result;
      const bool same_scope = have_decision && !hold_scope_changed(scope, decision_scope);
      if (identity.generated) {
        const bool shows = !identity.real_frame || identity.real_frame == decision_frame || !next_frame ||
          identity.real_frame == next_frame;
        result.hold = same_scope && shows;
        result.kind = result.hold ? hold_kind::generated : hold_kind::unavailable;
        return result;
      }
      result.detect = true;
      if (!have_decision || reset_pending || !same_scope) result.per_frame |= ui_detection::per_frame_hold_reset;
      else if (bits & accepted & ui_selection::candidate_bits & ~offered)
        result.per_frame |= ui_detection::per_frame_accepted_missing;
      result.adopt = !(result.per_frame & ui_detection::per_frame_accepted_missing);
      return result;
    }

    // Adopts a real frame's inputs. A change of them does not discard the
    // status sample (F1: status_fresh compares the winner). Hidden-scene holds
    // and refutations belong to their routes' inputs
    // (ui_detection::scene_route_key), not to a HUD-less pairing or to a UI
    // color tag.
    void adopt(std::uint32_t new_bits, std::uint32_t new_flags, std::uint32_t new_accepted) {
      if (const auto route = ui_detection::scene_route_key(new_bits, new_flags, new_accepted); route != scene_route) {
        scene_route = route;
        scene_refuted_slots = 0;
        clear_scene_holds();
      }
      bits = new_bits;
      flags = new_flags;
      accepted = new_accepted;
    }

    // F1: the status key, the winner of the adopted inputs: the first offered
    // and accepted candidate bit in draw order (alpha kinds, then HUD-less),
    // zero when none. The scope is compared apart (status_fresh).
    std::uint32_t status_key() const { return ui_selection::first_in_draw_order(bits & accepted & ui_selection::candidate_bits); }
    // The latest sample describes this frame: fresh, taken under the current
    // winner and in this scope.
    bool status_fresh(const alpha_auto_source &observation) const {
      return !sample_stale(latest.sample_tick_ms, observation.now_ms) && latest_key == status_key() &&
        !scope_changed(observation, latest_source);
    }

    // An active detection under another scope discards the sample and the
    // held routes.
    void enter_scope(const alpha_auto_source &observation) {
      if (scope_changed(observation, latest_source)) {
        latest = {};
        clear_scene_holds();
      }
    }

    // A generated Present that applied the detected mask: the first other
    // real-frame id it shows becomes the next real frame.
    void held(const present_identity &identity) {
      ++holds;
      if (identity.real_frame != decision_frame && !next_frame) next_frame = identity.real_frame;
    }
    // A real Present that detected: its decision is the chain's.
    void detected(const alpha_auto_source &scope, const present_identity &identity) {
      have_decision = true;
      decision_scope = scope;
      decision_frame = identity.real_frame;
      next_frame = 0;
      holds = 0;
      reset_pending = false;
    }
    // A generated Present without a mask ends the chain.
    void unavailable() {
      have_decision = false;
      reset_pending = true;
      holds = 0;
    }
    // A render without detection keeps no decision, verdict or refutation.
    void inactive() {
      unavailable();
      scene_refuted_slots = 0;
      clear_scene_holds();
    }

    // A sample still pending from before a clear cannot renew a hold.
    void clear_scene_holds() {
      scene_hold_until = {};
      scene_gate_open = pending_scene_actionable = false;
      shadow_run_start = 0;
    }

    // The CPU owns the hidden-scene verdict (docs/reshade-sbs.md, hidden-scene
    // evidence). A sample whose gate was open and whose presented evidence is
    // valid and hidden holds that route until its tick plus hold_ms; the
    // HUD-less route also needs its HUD-less image to read visible, and the
    // layer route a slot no visible verdict refuted. A valid visible verdict
    // releases both routes and refutes the slots that opened the layer route.
    // Invalid evidence renews nothing, and evidence the first-run shadow alone
    // measured (actionable false) changes nothing.
    void observe_scene(alpha_auto_decision &sample, bool actionable) {
      using ui_detection::scene_verdict;
      auto &evidence = sample.evidence;
      // As the shader decided (ui_selection::decide): the inputs are UIAlpha
      // and the offscreen UI layer, an invalid layer is no overlay, and the
      // accepted candidates are those pushed with the detection.
      const auto gates = ui_detection::scene_gates_of(sample.source_kind, evidence.candidates, evidence.accepted,
        sample.pixels, {evidence.alpha_invalid[0], evidence.layer_invalid}, {evidence.alpha_opaque[0], evidence.layer_opaque},
        evidence.hudless_changed);
      scene_gate_open = gates.layer || gates.hudless;
      // A slot offered below opaque is an overlay again.
      scene_refuted_slots &= ~gates.overlay_slots;
      const auto &scene = evidence.scene, &hudless = evidence.hudless_scene;
      const auto until = sample.sample_tick_ms + ui_detection::scene::hold_ms;
      if (actionable && scene.valid && scene.verdict == scene_verdict::visible) {
        scene_hold_until = {};
        scene_refuted_slots |= gates.layer_slots;
      } else if (actionable && scene.valid && scene.verdict == scene_verdict::hidden) {
        if (gates.layer && !(gates.layer_slots & scene_refuted_slots)) scene_hold_until[0] = until;
        if (gates.hudless && hudless.valid && ui_detection::scene_visible(hudless.d)) scene_hold_until[1] = until;
      }
      // An uncovered hidden scene: consecutive hidden samples while no source
      // decided. The first-run shadow measures it with the gates closed. A
      // sample with fewer decided comparisons than valid evidence needs edge
      // cells is blank (black, or a flat fade), with nothing to protect.
      if (scene.valid && scene.verdict == scene_verdict::hidden && !sample.source_kind &&
          scene.decided >= ui_detection::scene::min_edges) {
        if (!shadow_run_start || sample.sample_tick_ms < shadow_run_start) shadow_run_start = sample.sample_tick_ms;
        evidence.shadow_hidden_ms = sample.sample_tick_ms - shadow_run_start;
      } else shadow_run_start = 0;
    }

    // A sample frame measures hidden-scene evidence for the CPU when the latest
    // sample's gate was open or a route is held (actionable), for the
    // first-run shadow, and as a diagnostic after a sample that decided a
    // whole-frame alpha, so the counters can tell such a mask over a visible
    // scene from one over a hidden scene. Only actionable evidence changes a
    // hold; otherwise nothing runs.
    scene_measurement measure_scene(std::uint64_t now_ms, bool shadow) const {
      const bool actionable = scene_gate_open || scene_holds(now_ms) != 0;
      return {actionable || shadow || full_alpha(latest), actionable};
    }

    // This render's per-frame hold bits (ui_detection::per_frame_scene_hold*).
    std::uint32_t scene_holds(std::uint64_t now_ms) const {
      return (scene_hold_until[0] && now_ms <= scene_hold_until[0] ? ui_detection::per_frame_scene_hold : 0u) |
        (scene_hold_until[1] && now_ms <= scene_hold_until[1] ? ui_detection::per_frame_scene_hold_hudless : 0u);
    }
  };
}
