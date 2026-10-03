// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The CPU-side temporal state of live automatic UI detection: which decision
// inputs a render adopts, when it holds the previous frame's mask instead of
// detecting, when a completed status sample still describes the frame, and
// the CPU-owned hidden-scene verdict of each route, plus the decode of a
// completed sample's decision texels and the counts it commits.
// docs/reshade-sbs.md (UI protection, hidden-scene evidence, UI counters) owns
// the rules. The renderer calls this state in a fixed order
// (game3d_renderer.cpp, render, detect_ui, poll_detection and
// update_alpha_auto) and keeps only the GPU and resource work; the sequence
// replay test drives the same calls without a GPU. No ReShade dependency.
#include "game3d_alpha_auto.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"

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

  // The inputs that identify a detection's scope: a sample or a held route
  // from another epoch, revision or viewport is no evidence for this one.
  inline bool scope_changed(const alpha_auto_source &a, const alpha_auto_source &b) {
    return a.epoch != b.epoch || a.revision != b.revision || a.viewport != b.viewport;
  }
  // A completed sample is discarded unread when its scope or decision key no
  // longer matches the current inputs, or when it is stale on arrival.
  inline bool sample_discarded(const alpha_auto_source &input, const alpha_auto_source &pending, std::uint64_t key,
      std::uint64_t pending_key) {
    return scope_changed(input, pending) || key != pending_key || input.now_ms < pending.now_ms ||
      input.now_ms - pending.now_ms > sample_fresh_ms;
  }

  // The status sample poll_detection reads from the decision texels `words`
  // (4 per texel, at least min_decision_texels; with scene, the evidence
  // texels 5 and 6 too). ui_layer: the UI color slot held the offscreen layer
  // when the sample was submitted. sequence numbers the submitted samples.
  inline alpha_auto_decision decode_detection_sample(const std::uint32_t *words, std::size_t count,
      std::uint64_t tick_ms, std::uint64_t sequence, bool ui_layer, bool scene) {
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
    evidence.trusted_alpha = words[word::trusted];
    evidence.ui_layer = ui_layer;
    evidence.alpha_opaque = {words[word::alpha_opaque], words[word::alpha_opaque + 1]};
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

  // A sample that decided a whole-frame mask from alpha: a source 1-4
  // covering at least 99% of pixels (the GPU's full_alpha counter word).
  inline bool full_alpha(const alpha_auto_decision &sample) {
    return sample.source_kind >= 1 && sample.source_kind <= 4 && sample.pixels &&
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

  // Why a render holds the previous frame's decision and mask. When several
  // reasons apply the first one names the hold: a generated Present, then a
  // real frame whose HUD-less pair is inexact right after an exact decision,
  // then a trusted alpha channel that decided the previous frame and is
  // missing from this one.
  enum class hold_kind : std::uint8_t { none, generated, inexact_after_exact, trusted_missing };
  inline const char *name(hold_kind value) {
    switch (value) {
      case hold_kind::generated: return "generated";
      case hold_kind::inexact_after_exact: return "inexact_after_exact";
      case hold_kind::trusted_missing: return "trusted_missing";
      default: return "none";
    }
  }
  struct hold_decision {
    bool hold{};
    hold_kind kind = hold_kind::none;
    // A hold was wanted (kind names why) but max_held_presents consecutive
    // Presents already reused the mask, so this one detects again.
    bool cap_reached{};
  };

  // One renderer's detection state across frames. Candidate bits: 1, 2, 4, 8
  // alpha slots (UI alpha, UI color or layer, Backbuffer, current), 16 a
  // paired HUD-less image, 32 an exact pair. Flags are the UI color slot's
  // stored Sunshine_UIDetectionFlags; trusted the session's trusted slots.
  struct detection_state {
    // The inputs the last render adopted, and whether its decided mask may be
    // held: a HUD-less image or a trusted alpha channel made it.
    std::uint32_t bits{}, flags{}, trusted{};
    bool mask_ready{};
    // Consecutive Presents that reused the mask.
    std::uint32_t holds{};
    // The last fresh decision used an exact HUD-less pair.
    bool exact{};
    // The latest completed status sample and the inputs it was taken for.
    alpha_auto_decision latest;
    alpha_auto_source latest_source;
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
    // slots (1 UIAlpha, 2 UI layer) refuted as hiding the scene: nearly opaque
    // while valid evidence read the presented frame visible. A refuted slot
    // never holds the layer route until it is offered below that opacity.
    std::uint64_t scene_route{};
    std::uint32_t scene_refuted_slots{};

    std::uint64_t key() const { return detection_decision_key(bits, flags, trusted); }

    // Whether the UI color slot holds the offscreen UI layer: this frame's
    // source when the slot is offered, else the one that filled it last.
    bool layer_slot(std::uint32_t offered, std::uint32_t color_alpha_flags) const {
      return (((offered & 2u) ? color_alpha_flags : flags) & ui_detection::stored_late_layer) != 0;
    }

    // A generated Present keeps the preceding real frame's decision and mask.
    // So does a real frame whose HUD-less pair is inexact (frame generation on,
    // outside the tag batch) right after an exact decision: detecting again
    // from that pair would flip a full-screen menu between flat and 3D. So does
    // a trusted alpha channel that decided the previous frame and is missing
    // from this one: an observation loss refuses the previous revision's
    // captures until the game tags again. A trusted alpha channel in this
    // frame decides by itself; nothing is held. A trusted layer in the UI
    // color slot that the latest sample set aside (a layer without alpha,
    // ui_detection::admitted_candidates) decides nothing by itself, so it does
    // not stop holding the mask another channel made. Holds are bounded to
    // max_held consecutive Presents.
    hold_decision arbitrate(std::uint32_t offered, std::uint32_t trusted_now, bool layer, bool hold_previous,
        std::uint32_t max_held) const {
      const bool inexact = (offered & 48u) == 16u;
      const bool trusted_missing = (bits & trusted_now & 15u & ~offered) != 0;
      const std::uint32_t admitted = ui_detection::admitted_candidates(15u,
        layer && latest.evidence.ui_layer ? ui_detection::stored_late_layer : 0u,
        latest.evidence.alpha_covered[1], latest.evidence.alpha_invalid[1], latest.pixels);
      hold_decision result;
      result.kind = hold_previous ? hold_kind::generated : inexact && exact ? hold_kind::inexact_after_exact :
        trusted_missing ? hold_kind::trusted_missing : hold_kind::none;
      const bool wanted = !(offered & trusted_now & admitted) && result.kind != hold_kind::none && mask_ready;
      result.hold = wanted && holds < max_held;
      result.cap_reached = wanted && !result.hold;
      if (!wanted) result.kind = hold_kind::none;
      return result;
    }

    // Adopts this render's decision inputs (the held ones when it holds). Only
    // a change in the inputs that decide the mask makes the status sample
    // stale. Beside a trusted alpha channel, a HUD-less pair that frame
    // generation pairs on some Presents only otherwise discarded nearly every
    // sample while the mask stayed applied (Resident Evil Requiem). Hidden-
    // scene holds and refutations belong to their route's inputs, not to such
    // a pairing.
    void adopt(std::uint32_t new_bits, std::uint32_t new_flags, std::uint32_t new_trusted) {
      if (key() != detection_decision_key(new_bits, new_flags, new_trusted)) latest = {};
      if (const auto route = ui_detection::scene_route_key(new_bits, new_flags, new_trusted); route != scene_route) {
        scene_route = route;
        scene_refuted_slots = 0;
        clear_scene_holds();
      }
      bits = new_bits;
      flags = new_flags;
      trusted = new_trusted;
    }

    // An active detection under another scope discards the sample and the
    // held routes.
    void enter_scope(const alpha_auto_source &observation) {
      if (scope_changed(observation, latest_source)) {
        latest = {};
        clear_scene_holds();
      }
    }

    // A render that reused the mask, and one that detected afresh from the
    // adopted inputs.
    void held() { ++holds; }
    void detected() {
      holds = 0;
      mask_ready = (bits & 16u) != 0 || (bits & trusted & 15u) != 0;
      exact = (bits & 48u) == 48u;
    }
    // A render without detection keeps no mask, verdict or refutation.
    void inactive() {
      mask_ready = exact = false;
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
      // As the shader decided: a layer without alpha is no layer, so it is no
      // overlay either.
      const auto layer_flags = evidence.ui_layer ? ui_detection::stored_late_layer : 0u;
      const auto candidates = ui_detection::admitted_candidates(evidence.candidates, layer_flags, evidence.alpha_covered[1],
        evidence.alpha_invalid[1], sample.pixels);
      const auto gates = ui_detection::scene_gates_of(sample.source_kind, candidates, evidence.trusted_alpha,
        layer_flags, sample.pixels, evidence.alpha_invalid, evidence.alpha_opaque, evidence.hudless_changed);
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
