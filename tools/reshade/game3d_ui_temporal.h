// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The CPU-side temporal state of live automatic UI detection: which Presents
// detect and which show the decision of the real frame they show (T1), which
// inputs a real frame adopts, when a completed status sample still describes
// the frame (F1), plus the decode of a completed sample's decision texels and
// the counts it commits. The hidden-scene guard (M5, H1) is the depth path's
// own state, game3d_scene_guard.h, called in the order below.
// docs/reshade-sbs.md (UI protection, hidden-scene evidence, UI counters, UI
// decision framework) owns the rules. The renderer keeps only the GPU and
// resource work; the sequence replay test (test_game3d_ui_sequence.cpp)
// drives the same calls without a GPU, in the same order:
//
//   render (one Present; Auto, or manual On through detection):
//     1. bits: the offered candidate bits (ui_detection::candidate), and the
//        candidate_signatures of the offered kinds; identity: the provider's
//        present_identity {hold_previous} (Present counting: a Present that
//        offers nothing within the reported generated count of the last one
//        that offered a UI tag, ui_mask::generated_without_input);
//     2. accepted = session.accepted(bits, signatures) (manual On: every
//        offered candidate, without the ledger);
//     3. t = state.arbitrate(identity, observation, bits);
//     4. flags are the layer's stored flags when bits has candidate::layer,
//        else 0 (pushed only; nothing on the CPU reads them back); when
//        t.adopt, state.adopt(bits, accepted);
//     5. detection runs when it is requested, the frame fits and its resources
//        are prepared, and either t.hold, or t.detect with bits != 0 or
//        t.per_frame & per_frame_accepted_missing (a zero-offer real frame:
//        the reduce and mask passes only, no tiles pass and no sample). A
//        generated hold and a zero-offer frame apply the mask even while no
//        UI input is available:
//        - t.hold (kind generated): apply the detected mask as it is (the
//          last real decision), count held.generated, then state.held(); no
//          poll and no sample;
//        - t.kind unavailable (a generated Present with no such decision): no
//          mask, count held.none, then state.unavailable();
//        - t.detect: state.enter_scope(observation) and
//          scene_guard.enter_scope(epoch, viewport) (game3d_scene_guard.h:
//          only an identity change clears it), then poll_detection; then
//          per_frame = scene_guard.per_frame(now, bits, signatures) |
//          t.per_frame | depth_not_current | per_frame_reoffer (the
//          provider's hudless_reoffer of an inexact HUD-less snapshot
//          without a UIAlpha, UI color or Backbuffer tag) | the offered
//          declared tags outside the exact pair's tag batch shifted by
//          per_frame_unaligned_shift (the provider's
//          unaligned_declared), and detection
//          with this frame's own bits, accepted and flags pushed (b2), flags
//          | per_frame, plus per_frame_sample on a sample frame (at most one
//          every 100 ms while none is pending), and the hold store bound at
//          u5 for the reduce. A sample frame runs the
//          evidence passes when scene_guard.measure(now, proven_image) says
//          they are actionable, and keeps that with the pending sample. Then
//          state.detected(observation). A sample keeps the signatures and
//          pushed flags it was submitted with, and its status key is
//          state.status_key() at submission;
//        otherwise inactive.* and state.inactive() (the scene guard keeps
//        its state).
//   poll_detection (a completed sample):
//     6. sample_discarded(input, pending source) drops it unread (another
//        scope, or stale on arrival); otherwise
//        sample = decode_detection_sample(words, count, tick, sequence), the
//        one decode of the decision words, state.latest = sample,
//        state.latest_source = the pending source and state.latest_key = the
//        status key at submission;
//     7. obs = scene_guard.observe(guard_sample(sample), the pending
//        actionable, the submitted signatures by kind);
//     8. session.observe(sample.evidence, sample.pixels, tick, submitted
//        signatures); a session in a manual mode ignores it;
//     9. sample_counters(..., obs) commits the counts.
//   status (after render):
//    10. state.latest describes the frame only while
//        state.status_fresh(observation); otherwise the status is collecting.
//
// The GPU owns the T1 grace of a real frame without a decision of its own
// (ui_selection::decide, the hold store): this state only names the Presents
// that detect, the per-frame bits a detection pushes, and the Presents that
// show a real frame's decision. No ReShade dependency.
#include "game3d_alpha_auto.h"
#include "game3d_scene_guard.h"
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

  // The inputs that identify a detection's scope: a sample or a decision
  // from another epoch, revision or viewport is no evidence for this one.
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

  // The status sample poll_detection reads from the current revision's
  // decision texels `words` (4 per texel, ui_detection::decision_texels of
  // them; anything shorter, as from an older shader that only
  // ui_detection_replay pads, decodes as no sample): the decision, the
  // candidates and counts, the scene evidence texels 5 and 6 (zero when the
  // evidence passes did not run), the offscreen UI layer's texel 7, the
  // one-way judgment counts (texels 16, 8 and 9), the refused candidate and
  // the frame reason, the H1 texel 10 and the pre-UI pixel counts of texel
  // 11. sequence numbers the submitted samples.
  inline alpha_auto_decision decode_detection_sample(const std::uint32_t *words, std::size_t count,
      std::uint64_t tick_ms, std::uint64_t sequence) {
    namespace word = ui_detection::decision_word;
    alpha_auto_decision sample;
    if (count < 4 * ui_detection::decision_texels) return sample;
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
    evidence.layer_covered = words[word::layer_covered];
    evidence.layer_invalid = words[word::layer_invalid];
    evidence.layer_opaque = words[word::layer_opaque];
    evidence.valid_bits = words[word::valid_bits];
    evidence.strong = {words[word::strong_ui_alpha], words[word::strong_ui_color], words[word::strong_backbuffer],
      words[word::strong_current]};
    evidence.contradicted = {words[word::contradicted_ui_alpha], words[word::contradicted_ui_color],
      words[word::contradicted_backbuffer], words[word::contradicted_current]};
    evidence.refused = words[word::refused];
    evidence.frame_reason = words[word::frame_reason] & ui_detection::frame_reason_reason_mask;
    evidence.reused = (words[word::frame_reason] & ui_detection::frame_reason_reused) != 0;
    evidence.inferred_opaque = {words[word::opaque_backbuffer], words[word::opaque_current]};
    evidence.claims = words[word::claims];
    evidence.s1_source = words[word::h1] & ui_detection::h1_winner_mask;
    evidence.h1_applied = (words[word::h1] & ui_detection::h1_applied) != 0;
    evidence.pre_ui_match = words[word::pre_ui_match];
    evidence.pre_ui_image_lit = words[word::pre_ui_image_lit];
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
    // The pre-UI image's texel carries no verdict: the CPU reads it from D.
    evidence.pre_ui_scene = decode(word::pre_ui_scene_n, word::pre_ui_scene_d, word::pre_ui_scene_state);
    evidence.pre_ui_scene.verdict = evidence.pre_ui_scene.valid ?
      ui_detection::scene_verdict_of(evidence.pre_ui_scene.d) : ui_detection::scene_verdict::none;
    evidence.pre_ui_image = words[word::pre_ui_scene_image];
    return sample;
  }

  // The hidden-scene guard's view of a decoded status sample
  // (game3d_scene_guard.h) at its tick.
  inline scene_guard::sample guard_sample(const alpha_auto_decision &decoded) {
    const auto &evidence = decoded.evidence;
    scene_guard::sample s;
    s.tick = decoded.sample_tick_ms;
    s.pixels = decoded.pixels;
    s.offered = evidence.candidates;
    s.valid_bits = evidence.valid_bits;
    s.source = decoded.source_kind;
    s.claims = evidence.claims;
    using ui_selection::alpha_index;
    using ui_selection::kind;
    s.opaque[alpha_index(kind::ui_alpha)] = evidence.alpha_opaque[0];
    s.opaque[alpha_index(kind::ui_color)] = evidence.alpha_opaque[1];
    s.opaque[alpha_index(kind::ui_layer)] = evidence.layer_opaque;
    s.opaque[alpha_index(kind::backbuffer)] = evidence.inferred_opaque[0];
    s.opaque[alpha_index(kind::current)] = evidence.inferred_opaque[1];
    const auto image = [](const alpha_auto_decision::scene_evidence &e) {
      scene_guard::image_evidence result;
      result.n = e.n;
      result.d = e.d;
      result.valid = e.valid;
      result.ran = e.ran;
      result.verdict = e.verdict;
      result.decided = e.decided;
      return result;
    };
    s.presented = image(evidence.scene);
    s.pre_ui = image(evidence.pre_ui_scene);
    s.pre_ui_image = evidence.pre_ui_image;
    return s;
  }

  // The counts a completed sample commits (game3d_ui_counters.h): the CPU
  // counts snapshotted when it was submitted and the GPU words copied under
  // its fence, each as the change since the previous commit; when it decided
  // H1 (source 8), the hidden-scene verdict that same sample measured; and
  // what the hidden-scene guard's observation of it did (scene). The
  // full_alpha_d counters (accepted whole-frame decisions, which only a
  // removed diagnostic evidence run measured) stay zero, reserved.
  inline ui_counters sample_counters(const alpha_auto_decision &sample, const ui_counters &cpu,
      const ui_counters &committed_cpu, const std::array<std::uint32_t, ui_counter_word::count> &words,
      const std::array<std::uint32_t, ui_counter_word::count> &committed_words, std::uint64_t through_ms,
      const scene_guard::observation &observed = {}) {
    auto delta = cpu - committed_cpu;
    delta.add_gpu_delta(words, committed_words);
    if (sample.source_kind == 8) {
      using ui_detection::scene_verdict;
      const auto &scene = sample.evidence.scene;
      const auto verdict = scene.valid ? scene.verdict : scene_verdict::none;
      ++delta[ui_counter::full_d_hidden + (verdict == scene_verdict::hidden ? 0 : verdict == scene_verdict::ambiguous ? 1 :
        verdict == scene_verdict::visible ? 2 : 3)];
    }
    static_assert(ui_counter::full_d_ambiguous == ui_counter::full_d_hidden + 1 &&
      ui_counter::full_d_visible == ui_counter::full_d_hidden + 2 && ui_counter::full_d_invalid == ui_counter::full_d_hidden + 3);
    delta[ui_counter::scene_entered] = observed.entered;
    delta[ui_counter::scene_released] = observed.released;
    delta[ui_counter::scene_refuted] = observed.refuted;
    delta[ui_counter::samples] = 1;
    delta.through_ms = through_ms;
    return delta;
  }

  // T1 identity of one Present (M6), by Present counting: whether it offers
  // nothing within the reported generated count of the last Present that
  // offered a UI tag (ui_mask::generated_without_input, hold_previous). A
  // HUD-less pairing names no real frame: an offered image makes the Present
  // real.
  struct present_identity {
    bool generated{};
  };

  // What a Present does under T1: a real Present detects (none: no hold); a
  // generated Present shows the last real decision (generated, counted
  // held.generated) or, when no such decision exists in its scope, has no
  // mask (unavailable, counted held.none).
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
    // A real Present whose inputs become the adopted ones (the status key
    // and the accepted-missing reference).
    bool adopt{};
    // The per-frame bits T1 pushes with a detection
    // (ui_detection::per_frame_hold_reset, per_frame_accepted_missing).
    std::uint32_t per_frame{};
  };

  // One renderer's detection state across frames. Candidate bits are
  // ui_detection::candidate's: 0x1 UIAlpha, 0x2 UI color tag, 0x4
  // Backbuffer, 0x8 current, 0x10 a paired HUD-less image, 0x20 an exact
  // pair, 0x40 the offscreen UI layer; accepted the session's accepted
  // candidates (alpha_auto_policy::accepted), in the same bit positions. It
  // holds no hidden-scene state (game3d_scene_guard.h).
  struct detection_state {
    // The inputs the last adopting real frame offered (adopt): they key the
    // status (status_key) and the accepted-missing reference.
    std::uint32_t bits{}, accepted{};
    // T1 (M6): whether a real decision exists in this chain, its scope, and
    // whether the next detection starts a new chain (per_frame_hold_reset).
    bool have_decision{};
    alpha_auto_source decision_scope;
    bool reset_pending = true;
    // Consecutive generated Presents that applied the detected mask.
    std::uint32_t holds{};
    // The latest completed status sample, the inputs it was taken for and its
    // status key at submission (status_key()).
    alpha_auto_decision latest;
    alpha_auto_source latest_source;
    std::uint32_t latest_key{};

    // T1: a generated Present never detects. It shows the last real
    // decision (detected_mask) while that was made in the same identity
    // scope (hold_scope_changed: an observation revision alone keeps it);
    // otherwise it has no mask and the chain ends. No multiplier constant and
    // no time bound: Present counting (ui_mask::generated_without_input)
    // classifies only Presents that offer nothing, so a count reported too
    // high holds only those, and one too low leaves the later ones to the
    // grace below. A real
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
        result.hold = same_scope;
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
    // status sample (F1: status_fresh compares the winner), and never clears
    // the hidden-scene guard (identity only, game3d_scene_guard.h).
    void adopt(std::uint32_t new_bits, std::uint32_t new_accepted) {
      bits = new_bits;
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

    // An active detection under another scope discards the sample.
    void enter_scope(const alpha_auto_source &observation) {
      if (scope_changed(observation, latest_source)) latest = {};
    }

    // A generated Present that applied the detected mask.
    void held() { ++holds; }
    // A real Present that detected: its decision is the chain's.
    void detected(const alpha_auto_source &scope) {
      have_decision = true;
      decision_scope = scope;
      holds = 0;
      reset_pending = false;
    }
    // A generated Present without a mask ends the chain.
    void unavailable() {
      have_decision = false;
      reset_pending = true;
      holds = 0;
    }
    // A render without detection keeps no real decision (T1). The
    // hidden-scene guard keeps its state.
    void inactive() { unavailable(); }
  };
}
