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
// replay (test_game3d_ui_sequence) drive it exactly as the renderer does.
// Rule H2 (still screens without a UI source, fix 2), whose run it held, was
// removed. No ReShade dependency.
//
// Call order, per renderer (one state each):
//   - on every detecting frame, before anything else here: enter_scope(epoch,
//     viewport); only an identity change (another epoch or viewport, as
//     ui_temporal::hold_scope_changed) clears the state. An observation
//     revision, an inactive frame, acceptance changes and Forget never do;
//   - before detection: per_frame(now, offered, signatures, layer_proven) is
//     ORed into the pushed flags; and measure(now, proven_image) says whether
//     this sample frame runs the evidence passes, which is whether what they
//     measure is actionable (kept with the pending sample; the first-run
//     shadow and the whole-frame diagnostic that measured without acting were
//     removed). layer_proven: the offered layer's signature is proven the
//     pre-UI scene image (alpha_auto_policy::pre_ui_proven of its
//     signature); proven_image: that layer is also the offer's pre-UI image;
//   - at poll, for a completed sample in scope: observe(sample, actionable,
//     the signatures it was submitted with), before the acceptance ledger
//     observes it, with sample the guard's view of the decoded status sample
//     (ui_temporal::guard_sample of ui_temporal::decode_detection_sample, so
//     the decision words are decoded once).
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
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

#include <array>
#include <cstddef>
#include <cstdint>

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

  // What the guard reads of one completed detection sample (built from the
  // decoded sample by ui_temporal::guard_sample).
  struct sample {
    std::uint64_t tick{};
    std::uint32_t pixels{}, offered{}, valid_bits{}, source{};
    // The raw informative claims before refutation (candidate bits |
    // ui_detection::claim_pre_ui).
    std::uint32_t claims{};
    // Pixels with alpha of at least 254/255, in ui_selection::alpha_index order.
    std::array<std::uint32_t, 5> opaque{};
    image_evidence presented, pre_ui;
    // The image texel 6 measured (ui_detection::pre_ui_image).
    std::uint32_t pre_ui_image{};
  };

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
  // by a visible sample (or a sample that decided H1 read visible), and how
  // many signatures it newly refuted.
  struct observation {
    bool entered{}, released{};
    std::uint32_t refuted{};
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
    // The identity this state belongs to.
    std::uint64_t epoch{};
    std::uint32_t viewport{};
    bool scoped{};

    // Only an identity change clears the guard.
    void enter_scope(std::uint64_t new_epoch, std::uint32_t new_viewport) {
      if (scoped && new_epoch == epoch && new_viewport == viewport) return;
      *this = state{};
      scoped = true;
      epoch = new_epoch;
      viewport = new_viewport;
    }

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

    // A sample frame runs the evidence passes only when what they measure is
    // actionable: the latest sample had an acting-capable claim, a hold is
    // active or a proven layer is the offer's pre-UI image (proven_image).
    bool measure(std::uint64_t now, bool proven_image) const {
      return gate_open || hidden.held(now) || pre_ui.held(now) || proven_image;
    }

    observation observe(const sample &s, bool actionable, const kind_signatures &signatures) {
      observation result;
      // A refuted candidate offered, valid and below 99% opaque (or an exact
      // pair without a full change set) is an overlay again.
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
