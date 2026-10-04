// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The hidden-scene guard (UI decision framework M5, rule H1;
// docs/reshade-sbs.md, hidden-scene evidence), owned by the depth path: the
// CPU half of the decision "this frame's depth does not describe the image
// shown, so show it flat". SunshineUIDetectionReduceCS (ui_selection::decide)
// computes each frame's informative full claims and applies H1 from the bits
// this guard pushes; the guard holds the verdicts of the statistic D across
// samples and refutes claims per source signature. It reads only decision
// words, signatures and whether the offered layer is proven (from the
// acceptance ledger, below), never acceptance, slots, layer flags or the T1
// hold, so the single-frame replay (ui_detection_replay) and the sequence
// replay (test_game3d_ui_sequence) drive it exactly as the renderer does. It
// also holds the run of rule H2 (still screens without a UI source,
// game3d_still_screen.h), fed by the same samples and cleared by the same
// identity rule. No ReShade dependency.
//
// Call order, per renderer (one state each):
//   - on every render that requested detection: still_scope = Auto and
//     still_screen::sdr_output(swapchain colour space, format); without it,
//     still.leave(scope) ends any H2 run (the renderer logs and counts what
//     it ended), and so does still.leave(unmeasured) on a render in scope
//     that cannot be sampled (a zero-offer frame, an inactive or unavailable
//     detection);
//   - on every detecting frame, before anything else here: enter_scope(epoch,
//     viewport); only an identity change (another epoch or viewport, as
//     ui_temporal::hold_scope_changed) clears the state, and returns what it
//     ended of the H2 run (reason identity). An observation revision, an
//     inactive frame, acceptance changes and Forget never do;
//   - before detection: per_frame(now, offered, signatures, layer_proven) is
//     ORed into the pushed flags; still::flatten is pushed in b2 word 5 when
//     still_scope, the session enables it (UIFlattenStillScreens) and
//     still_flatten(); and measure(now, shadow, whole_frame, proven_image,
//     still_scope && still.wants_measure()) says whether this sample frame
//     runs the evidence passes and whether what they measure is actionable
//     (kept with the pending sample, as are still_scope and whether the
//     passes run for H2 alone, which keeps their evidence from the ledger:
//     ui_temporal::ledger_evidence). layer_proven: the
//     offered layer's signature is proven the pre-UI scene image
//     (alpha_auto_policy::pre_ui_proven of its signature); proven_image: that
//     layer is also the offer's pre-UI image;
//   - at poll, for a completed sample in scope: observe(sample_of(words, n,
//     tick, scene), actionable, the signatures it was submitted with, its
//     still_scope while this render is still in scope), before the
//     acceptance ledger observes it.
//
// Holds (run_hold): two valid hidden samples within hold_ms enter a hold,
// each further one renews it to its tick plus hold_ms, one valid visible
// sample releases it, an ambiguous one breaks an entry run, and invalid or
// non-actionable evidence changes nothing. The hidden hold is D on the
// presented frame reading hidden; the pre-UI hold is D on the pre-UI scene
// image reading visible on the samples that hit the hidden hold while the
// sample carries the pre-UI claim. A visible sample refutes the signature of
// every candidate whose full claim it carried, until an in-scope sample shows
// that candidate offered, valid and below 99% opaque (alpha), or as an exact
// pair without a full change set (HUD-less).
//
// Pre-UI proof (H1 d): the declared HUD-less image is the scene without its
// UI by contract; an offscreen layer holding colour without alpha is only
// inferred to be, and might be a scene buffer from before fog, grade, grain
// or vignette. Its claim (d) exists only while its signature is proven the
// pre-UI scene image, which the acceptance ledger owns (game3d_alpha_auto.h,
// key pre_ui:<format>:<space>: earned by samples whose layer equals the
// presented frame nearly everywhere, remembered across sessions, cleared by
// Forget only). The guard never gives or withdraws it and no identity change
// clears it; the caller passes the ledger's answer as layer_proven, which
// per_frame pushes as per_frame_pre_ui_proven. While a proven layer is the
// offer's pre-UI image every sample frame's evidence is actionable, so the
// first hidden samples after an identity change already count.
#include "game3d_still_screen.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sunshine_game3d::scene_guard {
  using ui_detection::scene_verdict;

  // One image's hidden-scene evidence in a sample (decision texel 5, the
  // presented frame, or 6, the pre-UI scene image, whose verdict the CPU
  // derives from D).
  struct image_evidence {
    std::uint32_t n{};
    float d{};
    bool valid{}, ran{};
    scene_verdict verdict = scene_verdict::none;
    // The presented image's decided comparisons (wins + losses).
    std::uint32_t decided{};
  };

  // What the guard reads of one completed detection sample.
  struct sample {
    std::uint64_t tick{};
    std::uint32_t pixels{}, offered{}, valid_bits{}, source{};
    // The raw informative claims before refutation (candidate bits |
    // ui_detection::claim_pre_ui); zero without texel 10.
    std::uint32_t claims{};
    // Pixels with alpha of at least 254/255, in ui_selection::alpha_index order.
    std::array<std::uint32_t, 5> opaque{};
    image_evidence presented, pre_ui;
    // The image texel 6 measured (ui_detection::pre_ui_image).
    std::uint32_t pre_ui_image{};
    // Texel 10 exists (selection revision 3), so every opaque count is known.
    bool complete{};
    // H2: the T1 grace reused the previous real frame's decision (texel 9),
    // and the still and compared cells of texel 12 (selection revision 5).
    bool reused{};
    std::uint32_t still_cells{}, still_compared{};
  };

  // Decodes a sample from the decision words (texel t, component c is word
  // 4 t + c): texels 0, 1, 4 and 7, the evidence texels 5 and 6 when scene
  // evidence ran (scene), the reused bit of texel 9 when there are
  // judgment_decision_texels, texel 10 when there are h1_decision_texels and
  // texel 12 (zero unless the evidence passes ran) when there are
  // still_decision_texels.
  inline sample sample_of(const std::uint32_t *words, std::size_t n, std::uint64_t tick, bool scene) {
    namespace word = ui_detection::decision_word;
    sample s;
    s.tick = tick;
    if (n < 4u * ui_detection::min_decision_texels) return s;
    s.source = words[word::source];
    s.pixels = words[word::pixels];
    s.offered = words[word::candidates];
    s.opaque[ui_selection::alpha_index(ui_selection::kind::ui_alpha)] = words[word::alpha_opaque];
    s.opaque[ui_selection::alpha_index(ui_selection::kind::ui_color)] = words[word::alpha_opaque + 1];
    if (n >= 4u * ui_detection::layer_decision_texels) {
      s.opaque[ui_selection::alpha_index(ui_selection::kind::ui_layer)] = words[word::layer_opaque];
      s.valid_bits = words[word::valid_bits];
    }
    if (n >= 4u * ui_detection::h1_decision_texels) {
      s.opaque[ui_selection::alpha_index(ui_selection::kind::backbuffer)] = words[word::opaque_backbuffer];
      s.opaque[ui_selection::alpha_index(ui_selection::kind::current)] = words[word::opaque_current];
      s.claims = words[word::claims];
      s.complete = true;
    }
    if (n >= 4u * ui_detection::judgment_decision_texels)
      s.reused = (words[word::frame_reason] & ui_detection::frame_reason_reused) != 0u;
    if (n >= 4u * ui_detection::still_decision_texels) {
      s.still_cells = words[word::still_cells];
      s.still_compared = words[word::still_compared];
    }
    if (scene && n >= 4u * ui_detection::scene_decision_texels) {
      const auto decode = [&](std::size_t first) {
        image_evidence result;
        result.n = words[first];
        std::memcpy(&result.d, &words[first + 1], sizeof(result.d));
        result.valid = ui_detection::scene_state_valid(words[first + 2]);
        result.ran = ui_detection::scene_state_ran(words[first + 2]);
        return result;
      };
      s.presented = decode(word::scene_n);
      s.presented.verdict = ui_detection::scene_state_verdict(words[word::scene_state]);
      s.presented.decided = words[word::scene_decided];
      s.pre_ui = decode(word::pre_ui_scene_n);
      s.pre_ui.verdict = s.pre_ui.valid ? ui_detection::scene_verdict_of(s.pre_ui.d) : scene_verdict::none;
      s.pre_ui_image = words[word::pre_ui_scene_image];
    }
    return s;
  }

  // A two-sample run hold over sample ticks (ms; 0 is no tick).
  struct run_hold {
    // The last hit of the current entry run, and the tick the hold lasts
    // through; zero when none.
    std::uint64_t last{}, until{};

    bool held(std::uint64_t now) const { return until && now <= until; }
    // A hit: renews a hold held at tick, else enters one when the previous
    // hit is within hold_ms. Returns whether it entered.
    bool hit(std::uint64_t tick) {
      bool entered = false;
      if (held(tick)) until = tick + ui_detection::scene::hold_ms;
      else if (last && tick >= last && tick - last <= ui_detection::scene::hold_ms) {
        until = tick + ui_detection::scene::hold_ms;
        entered = true;
      }
      last = tick;
      return entered;
    }
    // Breaks the entry run only; a held hold lasts until it expires.
    void miss() { last = 0; }
    void release() { last = until = 0; }
  };

  // What one observation did: the hidden hold entered, a hold was released
  // by a visible sample (or a sample that decided H1 read visible), how many
  // signatures it newly refuted, and the first-run shadow's run length; and
  // what it did to H2's run (still: it entered, an entered episode ended, or
  // a run ended before it entered, with its length).
  struct observation {
    bool entered{}, released{};
    std::uint32_t refuted{};
    std::uint64_t shadow_hidden_ms{};
    still_screen::observation still;
  };

  // Whether a sample frame runs the evidence passes, and whether what they
  // measure is actionable (may hold, release or refute).
  struct measurement {
    bool run{}, actionable{};
  };

  // The acceptance signature of every candidate kind, in ui_selection::kind
  // order; only offered kinds are read.
  using kind_signatures = std::array<ui_selection::signature, ui_selection::kind_names.size()>;

  struct state {
    // The hidden verdict of D on the presented frame, and D reading the
    // pre-UI scene image visible on the same samples.
    run_hold hidden, pre_ui;
    // Refuted signatures, a bounded FIFO (the oldest drops first).
    static constexpr std::size_t max_refuted = 8;
    std::array<ui_selection::signature, max_refuted> refutations{};
    std::size_t refuted_count{};
    // The latest sample had an acting-capable claim (a claim not refuted, or
    // the pre-UI claim).
    bool gate_open{};
    // The first sample tick of the current run of hidden samples without a
    // decided source (the first-run shadow); zero without a run.
    std::uint64_t shadow_run_start{};
    // H2's run of still samples that read the presented frame hidden while
    // no source decided (game3d_still_screen.h).
    still_screen::run still;
    // The identity this state belongs to.
    std::uint64_t epoch{};
    std::uint32_t viewport{};
    bool scoped{};

    // Only an identity change clears the guard; it returns what that ended
    // of H2's run (reason identity), empty otherwise.
    still_screen::observation enter_scope(std::uint64_t new_epoch, std::uint32_t new_viewport) {
      if (scoped && new_epoch == epoch && new_viewport == viewport) return {};
      const auto ended = still.leave(still_screen::end_reason::identity);
      *this = state{};
      scoped = true;
      epoch = new_epoch;
      viewport = new_viewport;
      return ended;
    }
    // H2 is active: the renderer pushes still::flatten when the session
    // enables it.
    bool still_flatten() const { return still.active(); }

    bool refuted(const ui_selection::signature &signature) const {
      for (std::size_t i = 0; i != refuted_count; ++i)
        if (refutations[i] == signature) return true;
      return false;
    }
    // The offered candidate bits whose signatures are refuted.
    std::uint32_t refuted_bits(std::uint32_t offered, const kind_signatures &signatures) const {
      std::uint32_t bits = 0;
      for (const auto k : ui_selection::draw_order)
        if ((offered & ui_selection::bit(k)) && refuted(signatures[std::size_t(k)])) bits |= ui_selection::bit(k);
      return bits;
    }

    // The per-frame bits of a detection at now: the held verdicts (the
    // pre-UI image's only while the offer's image is the HUD-less image or a
    // proven layer), the refuted offered candidates, and
    // per_frame_pre_ui_proven while the offered layer is proven
    // (layer_proven, the acceptance ledger's answer for its signature).
    std::uint32_t per_frame(std::uint64_t now, std::uint32_t offered, const kind_signatures &signatures,
        bool layer_proven) const {
      const bool layer_offered = (offered & ui_detection::candidate::layer) != 0u;
      const bool proven = layer_offered && layer_proven;
      const bool layer_image = ui_selection::pre_ui_image_of(offered) == ui_detection::pre_ui_image::layer;
      const bool pre_ui_visible = pre_ui.held(now) && (!layer_image || proven);
      return (hidden.held(now) ? ui_detection::per_frame_scene_hidden : 0u) |
        (pre_ui_visible ? ui_detection::per_frame_pre_ui_visible : 0u) |
        (refuted_bits(offered, signatures) << ui_detection::per_frame_refuted_shift) |
        (proven ? ui_detection::per_frame_pre_ui_proven : 0u);
    }

    // A sample frame runs the evidence passes when the latest sample had an
    // acting-capable claim, a hold is active or a proven layer is the offer's
    // pre-UI image (proven_image) (actionable), for the first-run shadow
    // (shadow), as a diagnostic after a whole-frame decision (whole_frame: an
    // accepted alpha covering 99% or source 6), and for H2 (still: in scope
    // and still.wants_measure()). Only actionable evidence holds, releases or
    // refutes; H2 reads every sample whatever its actionability.
    measurement measure(std::uint64_t now, bool shadow, bool whole_frame, bool proven_image, bool still_scope = false) const {
      const bool actionable = gate_open || hidden.held(now) || pre_ui.held(now) || proven_image;
      return {actionable || shadow || whole_frame || still_scope, actionable};
    }

    // still_scope: the sample was submitted in H2's scope (SDR Auto).
    observation observe(const sample &s, bool actionable, const kind_signatures &signatures, bool still_scope = false) {
      observation result;
      // A refuted candidate offered, valid and below 99% opaque (or an exact
      // pair without a full change set) is an overlay again.
      if (s.complete)
        for (const auto k : ui_selection::draw_order) {
          const auto b = ui_selection::bit(k);
          if (!(s.offered & s.valid_bits & b)) continue;
          const bool overlay = ui_selection::alpha_kind(k) ?
            !ui_selection::opaque_full(s.opaque[ui_selection::alpha_index(k)], s.pixels) :
            (s.offered & ui_detection::candidate::exact) && !(s.claims & b);
          if (overlay) restore(signatures[std::size_t(k)]);
        }
      const auto &presented = s.presented;
      if (actionable && presented.valid) {
        if (presented.verdict == scene_verdict::visible) {
          result.released = hidden.held(s.tick) || pre_ui.held(s.tick) || s.source == 8u;
          hidden.release();
          pre_ui.release();
          for (const auto k : ui_selection::draw_order)
            if ((s.claims & ui_selection::bit(k)) && refute(signatures[std::size_t(k)])) ++result.refuted;
        } else if (presented.verdict == scene_verdict::hidden) {
          result.entered = hidden.hit(s.tick);
          // The pre-UI image's own evidence, read only for the image the
          // claim names (a layer's claim exists only while it is proven);
          // invalid evidence changes nothing.
          if ((s.claims & ui_detection::claim_pre_ui) && s.pre_ui_image &&
              s.pre_ui_image == ui_selection::pre_ui_image_of(s.offered) && s.pre_ui.valid) {
            if (ui_detection::scene_visible(s.pre_ui.d)) pre_ui.hit(s.tick);
            else pre_ui.release();
          }
        } else {
          hidden.miss();
          pre_ui.miss();
        }
      }
      gate_open = (s.claims & ui_selection::candidate_bits & ~refuted_bits(s.offered, signatures)) != 0u ||
        (s.claims & ui_detection::claim_pre_ui) != 0u;
      // An uncovered hidden scene: consecutive hidden samples while no source
      // decided. The first-run shadow measures it. A sample with fewer decided
      // comparisons than valid evidence needs edge cells is blank (black, or a
      // flat fade), with nothing to protect.
      if (presented.valid && presented.verdict == scene_verdict::hidden && !s.source &&
          presented.decided >= ui_detection::scene::min_edges) {
        if (!shadow_run_start || s.tick < shadow_run_start) shadow_run_start = s.tick;
        result.shadow_hidden_ms = s.tick - shadow_run_start;
      } else shadow_run_start = 0;
      // H2: every completed sample extends or ends the run of still hidden
      // samples without a decided source.
      result.still = still.observe({s.tick, still_scope, s.source, s.reused, presented.ran, presented.n, presented.d,
        s.still_cells, s.still_compared});
      return result;
    }

  private:
    // Adds a refutation; false when it was already refuted.
    bool refute(const ui_selection::signature &signature) {
      if (refuted(signature)) return false;
      if (refuted_count == max_refuted) {
        for (std::size_t i = 1; i != max_refuted; ++i) refutations[i - 1] = refutations[i];
        --refuted_count;
      }
      refutations[refuted_count++] = signature;
      return true;
    }
    void restore(const ui_selection::signature &signature) {
      for (std::size_t i = 0; i != refuted_count; ++i)
        if (refutations[i] == signature) {
          for (std::size_t j = i + 1; j != refuted_count; ++j) refutations[j - 1] = refutations[j];
          --refuted_count;
          return;
        }
    }
  };
}
