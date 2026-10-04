// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The pre-UI change-set providers of the UI decision framework (fix 3;
// docs/reshade-sbs.md, UI decision framework): the contract the renderer,
// ui_detection_replay and test_game3d_ui_sequence share for the inferred
// provider, the offscreen layer proven the pre-UI scene image
// (ui_detection::candidate::pre_ui). The declared provider, a Streamline
// HUD-less image, needs none of this: its pair is the frame it belongs to.
//
// Pairing (a precondition, not something V2 validity can catch): the live
// tracker copies the layer before the first qualifying clear after a
// Present (game3d_ui_layer.h), so the copy holds the frame that Present
// showed, not the frame being rendered. presents_since_copy counts the
// Presents observed since the copy was recorded; the copy then pairs with
// the retained Present that many back (1 or 2, at t2 or t3), never with the
// current Present and never on a generated Present, which does not detect.
// Exactness is a property of the paired Presents, not of this render's mode:
// Present counting is exact only when the paired Present and every Present
// since, this one included, were real Presents with frame generation known
// off (the Streamline observer read the game's FG mode as off; a game that
// switches FG off in its menus qualifies, one that keeps it requested while
// suspended does not, since nothing marks its Presents real). An unknown
// mode (no observation yet, no Streamline FG observer, or FG another SDK
// generates) and the first Presents after FG turns off are late: the
// provider is never offered (its shadow still measures the late pair
// against the current Present). An engine that records a frame's clears
// before the previous Present would break the count; the shadow's pair
// verdict (the changed count against the Presents 0, 1 and 2 back, smallest
// at the paired offset) is the evidence that it holds, and needs no ground
// truth.
//
// The shadow (UIPinOnlyUI=0, the default): sample frames with an
// offered layer whose signature is proven the pre-UI scene image (b2 word 5
// ui_detection::change_set::shadow; a real UI layer never earns the proof
// and is never measured) measure the layer against its pair (decision
// texels 13-15)
// and measure_shadow reruns ui_selection::decide on that sample with the
// pre-UI change set offered and refine on, so would_source and would_refine
// are exactly what the switch would do. Nothing here feeds the acceptance
// ledger or any decision. No ReShade dependency.
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"
#include "game3d_ui_ticket.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace sunshine_game3d::change_set {
  // How a layer copy pairs with a Present: none without a layer; retained by
  // Present counting (exact, the only class offered for decisions); late
  // when frame generation was not known off on every Present since the one
  // it shows (against the current Present, inexact); unavailable when the
  // Present it shows is not retained (or the copy is of the current
  // interval). S3: token, the layer paired in token space with the
  // same-token Backbuffer tag (pair offset
  // ui_detection::change_set::pair_backbuffer), which the renderer chooses
  // only while identity is authoritative (game3d_ui_ticket.h).
  enum class pair_class : std::uint8_t { none, retained, late, unavailable, token };
  constexpr std::string_view name(pair_class value) {
    switch (value) {
      case pair_class::retained: return "retained";
      case pair_class::late: return "late";
      case pair_class::unavailable: return "unavailable";
      case pair_class::token: return "token";
      default: return "none";
    }
  }
  // The class and the Present offset (0 the current Present, 1 or 2 back)
  // that ui_detection::change_set::pair_mask carries.
  struct layer_pairing {
    pair_class kind = pair_class::none;
    std::uint32_t offset{};
    friend constexpr bool operator==(layer_pairing a, layer_pairing b) { return a.kind == b.kind && a.offset == b.offset; }
  };

  // The pairing of an offered layer copy: fg_off_presents is the run of
  // consecutive real Presents with frame generation known off, this one
  // included (zero when this one is not; next_fg_off_presents);
  // presents_since_copy (ui_layer::live_capture) and which of the Presents
  // one and two back are retained. Late unless the run covers the Present
  // the copy shows.
  constexpr layer_pairing pair_layer(bool layer_offered, std::uint32_t fg_off_presents, std::uint32_t presents_since_copy,
      bool retained_1, bool retained_2) {
    if (!layer_offered) return {};
    if (fg_off_presents <= presents_since_copy) return {pair_class::late, 0u};
    if ((presents_since_copy == 1u && retained_1) || (presents_since_copy == 2u && retained_2))
      return {pair_class::retained, presents_since_copy};
    return {pair_class::unavailable, 0u};
  }
  // S3: the token-space pairing (against the Backbuffer tag's colour).
  constexpr layer_pairing token_pairing() { return {pair_class::token, ui_detection::change_set::pair_backbuffer}; }
  // Whether a detection offers the pre-UI change set: in Auto, with the
  // session's switch on (UIPinOnlyUI), the layer's signature proven
  // the pre-UI scene image (the ledger's pre_ui key, which is also its
  // acceptance), no HUD-less image offered (a declared pair takes the slot)
  // and an exact retained pairing (S3, enabled only: or a token pairing).
  constexpr bool offered(bool auto_mode, bool enabled, bool layer_proven, bool hudless_offered, layer_pairing pairing) {
    return auto_mode && enabled && layer_proven && !hudless_offered &&
      (pairing.kind == pair_class::retained || pairing.kind == pair_class::token);
  }
  // b2 word 5 (Sunshine_UIRules): H2's flatten, refine
  // (ui_detection::rules::pin_only_ui, which the mask pass also reads for
  // rule P2's darkening, game3d_ui_darkening.h), the shadow (the
  // offered layer proven the pre-UI scene image), T1's change-set gap, the
  // pairing's Present offset and the bound retained Presents.
  constexpr std::uint32_t rule_bits(bool still_flatten, bool refine, bool shadow, bool gap, layer_pairing pairing,
      bool retained_1, bool retained_2) {
    namespace rule = ui_detection::change_set;
    return (still_flatten ? ui_detection::still::flatten : 0u) | (refine ? ui_detection::rules::pin_only_ui : 0u) |
      (shadow ? rule::shadow : 0u) | (gap ? rule::gap : 0u) | ((pairing.offset << rule::pair_shift) & rule::pair_mask) |
      (retained_1 ? rule::retained_1 : 0u) | (retained_2 ? rule::retained_2 : 0u);
  }
  // The run of real Presents with frame generation known off after one more
  // Present: one more while it is known off on a real Present, else zero.
  constexpr std::uint32_t next_fg_off_presents(std::uint32_t run, bool fg_known_off, bool generated) {
    return fg_known_off && !generated ? (run < 0xffffffffu ? run + 1u : run) : 0u;
  }
  static_assert(pair_layer(false, 5u, 1u, true, true) == layer_pairing{} &&
    pair_layer(true, 0u, 1u, true, true) == layer_pairing{pair_class::late, 0u} &&
    pair_layer(true, 1u, 1u, true, true) == layer_pairing{pair_class::late, 0u} &&
    pair_layer(true, 2u, 2u, true, true) == layer_pairing{pair_class::late, 0u} &&
    pair_layer(true, 2u, 1u, true, false) == layer_pairing{pair_class::retained, 1u} &&
    pair_layer(true, 3u, 2u, false, true) == layer_pairing{pair_class::retained, 2u} &&
    pair_layer(true, 3u, 2u, true, false) == layer_pairing{pair_class::unavailable, 0u} &&
    pair_layer(true, 1u, 0u, true, true) == layer_pairing{pair_class::unavailable, 0u} &&
    pair_layer(true, 4u, 3u, true, true) == layer_pairing{pair_class::unavailable, 0u});
  // S3 (game3d_ui_ticket.h, shadow): the present label the CPU proposes for an
  // offered layer copy, for the GPU to check against the copy's C_P read: the
  // label of the retained Present a retained pairing names (present_label,
  // this render's, minus its offset); 0, not proposed, for any other class,
  // without a label, or within the label's first Presents.
  constexpr std::uint32_t proposed_layer_label(layer_pairing pairing, std::uint32_t present_label) {
    return pairing.kind == pair_class::retained && pairing.offset && present_label > pairing.offset ?
      present_label - pairing.offset : 0u;
  }
  // The token label the CPU proposes for the layer copy with FG on, unknown
  // or suspended: the newest ready Backbuffer tag snapshot's token label
  // (the copy's C_T read must equal it), 0 when none is ready.
  constexpr std::uint32_t token_proposal(bool backbuffer_ready, std::uint32_t token) {
    return backbuffer_ready ? token : 0u;
  }
  static_assert(proposed_layer_label({pair_class::retained, 1u}, 10u) == 9u &&
    proposed_layer_label({pair_class::retained, 2u}, 10u) == 8u && !proposed_layer_label({pair_class::late, 0u}, 10u) &&
    !proposed_layer_label({pair_class::unavailable, 0u}, 10u) && !proposed_layer_label({pair_class::retained, 1u}, 1u) &&
    !proposed_layer_label({}, 10u) && token_proposal(true, 7u) == 7u && !token_proposal(false, 7u));
  static_assert(next_fg_off_presents(0u, true, false) == 1u && next_fg_off_presents(4u, true, false) == 5u &&
    next_fg_off_presents(4u, false, false) == 0u && next_fg_off_presents(4u, true, true) == 0u &&
    next_fg_off_presents(0xffffffffu, true, false) == 0xffffffffu);
  static_assert(offered(true, true, true, false, {pair_class::retained, 1u}) &&
    !offered(true, true, true, true, {pair_class::retained, 1u}) && !offered(true, true, true, false, {pair_class::late, 0u}) &&
    !offered(false, true, true, false, {pair_class::retained, 1u}) &&
    !offered(true, false, true, false, {pair_class::retained, 1u}) &&
    !offered(true, true, false, false, {pair_class::retained, 1u}) && offered(true, true, true, false, token_pairing()) &&
    ui_detection::change_set::pair_offset(rule_bits(false, true, true, false, token_pairing(), false, false)) == 3u);
  static_assert(rule_bits(true, true, false, false, {pair_class::retained, 2u}, true, true) == 0xe3u &&
    rule_bits(false, false, false, false, {pair_class::late, 0u}, false, false) == 0u &&
    rule_bits(false, false, true, false, {pair_class::retained, 1u}, true, false) == 0x54u &&
    rule_bits(false, true, false, true, {pair_class::unavailable, 0u}, false, false) == 0xau);

  // The pair verdict of a retained pairing, from the layer's changed counts
  // against the Presents 0, 1 and 2 back: verified when the paired offset's
  // is strictly below every other measured one, contradicted when another is
  // strictly below it, inconclusive otherwise; none without a retained
  // pairing. Since S3 it is the content cross-check of the stamp's identity
  // verdict (shadow_sample::identity), kept while S3 runs in shadow and
  // deleted when identity becomes authoritative.
  enum class pair_verdict : std::uint8_t { none, verified, contradicted, inconclusive };
  constexpr std::string_view name(pair_verdict value) {
    switch (value) {
      case pair_verdict::verified: return "verified";
      case pair_verdict::contradicted: return "contradicted";
      case pair_verdict::inconclusive: return "inconclusive";
      default: return "-";
    }
  }
  constexpr std::string_view judge_name(std::uint32_t kind) {
    return kind == ui_detection::change_set::judge::ui_alpha ? "ui_alpha" :
      kind == ui_detection::change_set::judge::ui_color ? "ui_color" : "none";
  }

  // One sample's change-set shadow.
  struct shadow_sample {
    layer_pairing pairing;
    // The session's switch (UIPinOnlyUI), and whether the pairing is
    // late (pair_layer's class without frame generation known off over the
    // pair).
    bool enabled{}, fg{};
    // Decision texels 13-15, the pixels, and the layer's lit (texel 11) and
    // covered (texel 7) pixels.
    ui_selection::change_set_shadow counts;
    std::uint32_t pixels{}, lit{}, layer_covered{};
    // The layer pair passes the pre-UI change set's validity (a partial
    // change set, lit on 1%, over a layer without coverage).
    bool valid{};
    // Changed pixels against the Presents 0, 1 and 2 back (offset 0: those
    // texel 11 does not match), measured or not.
    std::array<std::uint32_t, 3> offsets{};
    std::array<bool, 3> measured{};
    pair_verdict verdict = pair_verdict::none;
    // The judge's agreement: precision tp / filtered, recall tp / judge
    // pixels, IoU tp / (filtered + judge - tp); negative without a basis.
    double precision = -1., recall = -1., iou = -1.;
    // The sample's own decision: the S1 winner's source (decision word h1),
    // whether that winner was an alpha opaque on every pixel, and the applied
    // source; and what the pre-UI change set offered with
    // UIPinOnlyUI=1 would give (the sample's own decision when it
    // could not be offered): the applied source, whether it refined a
    // shapeless alpha, and whether S1 chose source 12 itself.
    std::uint32_t winner{}, applied_source{}, would_source{};
    bool shapeless{}, would_refine{}, would_decide{};
    // S3 (shadow): the GPU's identity verdict of the layer pair from the
    // change-set pass's verdict texel (exact, mismatch, unstamped or
    // unproposed; none where the shader writes none), and its stamp read
    // minus the proposed label.
    ui_ticket::gpu_verdict identity = ui_ticket::gpu_verdict::none;
    std::int64_t identity_delta{};
  };

  // The shadow of one completed sample: its counts (texels 0-15), the
  // offered and accepted candidates, flags and rules it was submitted with,
  // the pairing, whether the layer's signature was proven, a HUD-less image
  // offered, Auto and the switch on, and the hold store the sample found
  // (none when unknown, as in ui_detection_replay).
  inline shadow_sample measure_shadow(const ui_selection::counts &c, std::uint32_t offered_bits, std::uint32_t accepted,
      std::uint32_t flags, std::uint32_t rules, layer_pairing pairing, bool layer_proven, bool hudless_offered, bool auto_mode,
      bool enabled, const ui_selection::hold_state &previous = {}) {
    namespace candidate = ui_detection::candidate;
    shadow_sample s;
    s.pairing = pairing;
    s.enabled = enabled;
    s.fg = pairing.kind == pair_class::late;
    s.counts = c.shadow;
    s.pixels = c.pixels;
    s.lit = c.pre_ui_lit;
    s.layer_covered = c.covered[ui_selection::alpha_index(ui_selection::kind::ui_layer)];
    // The counterfactual slot: the layer pair's counts.
    ui_selection::counts pair = c;
    pair.changed = c.shadow.changed;
    pair.unchanged = c.shadow.unchanged;
    pair.nonfinite = c.shadow.nonfinite;
    pair.matching_tiles = c.shadow.matching_tiles;
    pair.lit = c.pre_ui_lit;
    s.valid = ui_selection::change_set_selective(pair) && !s.layer_covered;
    s.offsets = {c.pixels >= c.pre_ui_match ? c.pixels - c.pre_ui_match : 0u, c.shadow.changed_1, c.shadow.changed_2};
    s.measured = {true, (rules & ui_detection::change_set::retained_1) != 0u, (rules & ui_detection::change_set::retained_2) != 0u};
    if (pairing.kind == pair_class::retained && pairing.offset < s.offsets.size() && s.measured[pairing.offset]) {
      bool below = false, all_above = true;
      for (std::size_t i = 0; i != s.offsets.size(); ++i) {
        if (i == pairing.offset || !s.measured[i]) continue;
        if (s.offsets[i] < s.offsets[pairing.offset]) below = true;
        if (s.offsets[i] <= s.offsets[pairing.offset]) all_above = false;
      }
      s.verdict = below ? pair_verdict::contradicted : all_above ? pair_verdict::verified : pair_verdict::inconclusive;
    }
    if (c.shadow.judge_kind) {
      const double tp = c.shadow.judge_tp, filtered = c.shadow.filtered, judge = c.shadow.judge_pixels;
      if (filtered > 0.) s.precision = tp / filtered;
      if (judge > 0.) s.recall = tp / judge;
      if (filtered + judge - tp > 0.) s.iou = tp / (filtered + judge - tp);
    }
    const auto own = ui_selection::decide(c, offered_bits, accepted, flags, previous, rules);
    s.winner = own.s1_source;
    s.shapeless = own.shapeless;
    s.applied_source = own.source;
    if (offered(auto_mode, true, layer_proven, hudless_offered, pairing)) {
      const auto would = ui_selection::decide(pair, offered_bits | candidate::pre_ui, accepted | candidate::pre_ui, flags,
        previous, rules | ui_detection::rules::pin_only_ui);
      s.would_source = would.source;
      s.would_refine = would.refined;
      s.would_decide = would.s1_source == ui_detection::source_pre_ui && !would.refined;
    } else {
      s.would_source = own.source;
      s.would_refine = own.refined;
      s.would_decide = own.s1_source == ui_detection::source_pre_ui && !own.refined;
    }
    return s;
  }

  // S3 (shadow): the GPU's identity verdict of the sample's layer pair
  // (decision texel 12 .z/.w, ui_selection::identity_of_words) and its read
  // minus the proposal. The contract's verdict values are ui_ticket's.
  static_assert(std::uint32_t(ui_ticket::gpu_verdict::exact) == ui_detection::identity::exact &&
    std::uint32_t(ui_ticket::gpu_verdict::mismatch) == ui_detection::identity::mismatch &&
    std::uint32_t(ui_ticket::gpu_verdict::unstamped) == ui_detection::identity::unstamped &&
    std::uint32_t(ui_ticket::gpu_verdict::unproposed) == ui_detection::identity::unproposed &&
    std::uint32_t(ui_ticket::gpu_verdict::none) == ui_detection::identity::none);
  inline void add_identity(shadow_sample &s, const ui_selection::identity_verdict &v) {
    s.identity = v.layer < std::uint32_t(ui_ticket::gpu_verdict::count) ? ui_ticket::gpu_verdict(v.layer) :
      ui_ticket::gpu_verdict::none;
    s.identity_delta = v.layer_delta;
  }

  // The log text of a shadow sample (game3d_log_report.py parses it).
  inline std::string shadow_log_text(const shadow_sample &s) {
    const auto offset = [&](std::size_t i) { return s.measured[i] ? std::to_string(s.offsets[i]) : std::string("-"); };
    const auto ratio = [](double value) {
      if (value < 0.) return std::string("-");
      char text[16];
      std::snprintf(text, sizeof(text), "%.3f", value);
      return std::string(text);
    };
    const auto flag = [](bool value) { return value ? "1" : "0"; };
    std::string text = "Sunshine UI change set: pairing=";
    text.append(name(s.pairing.kind)).append(" offset=").append(std::to_string(s.pairing.offset));
    text.append(" fg=").append(flag(s.fg)).append(" UIPinOnlyUI=").append(flag(s.enabled));
    text.append(" changed=").append(std::to_string(s.counts.changed));
    text.append(" unchanged=").append(std::to_string(s.counts.unchanged));
    text.append(" nonfinite=").append(std::to_string(s.counts.nonfinite));
    text.append(" matching_tiles=").append(std::to_string(s.counts.matching_tiles));
    text.append(" lit=").append(std::to_string(s.lit));
    text.append(" layer_covered=").append(std::to_string(s.layer_covered));
    text.append(" valid=").append(flag(s.valid));
    text.append(" filtered=").append(std::to_string(s.counts.filtered));
    text.append(" offsets={0=").append(offset(0)).append(" 1=").append(offset(1)).append(" 2=").append(offset(2)).append("}");
    text.append(" pair=").append(name(s.verdict));
    text.append(" judge={kind=").append(judge_name(s.counts.judge_kind));
    text.append(" pixels=").append(std::to_string(s.counts.judge_pixels));
    text.append(" tp=").append(std::to_string(s.counts.judge_tp));
    text.append(" precision=").append(ratio(s.precision)).append(" recall=").append(ratio(s.recall));
    text.append(" iou=").append(ratio(s.iou)).append("}");
    text.append(" winner=").append(std::to_string(s.winner)).append(" shapeless=").append(flag(s.shapeless));
    text.append(" would_refine=").append(flag(s.would_refine)).append(" would_source=").append(std::to_string(s.would_source));
    text.append(" applied_source=").append(std::to_string(s.applied_source));
    text.append(" pixels=").append(std::to_string(s.pixels));
    // S3 (shadow): the layer pair's identity verdict, when the GPU wrote one.
    if (s.identity != ui_ticket::gpu_verdict::none)
      text.append(" identity=").append(ui_ticket::name(s.identity)).append(" identity_delta=")
        .append(std::to_string(s.identity_delta));
    return text;
  }

  // Log throttling: a line when the pairing, validity, would_refine or
  // would_source changes, else at most one per log_interval_ms.
  inline constexpr std::uint64_t log_interval_ms = 1000;
  struct shadow_log_state {
    bool emitted{};
    pair_class pairing = pair_class::none;
    bool valid{}, would_refine{};
    std::uint32_t would_source{};
    std::uint64_t last_ms{};
  };
  // Whether to log this sample at now_ms; records it when so.
  inline bool shadow_log_due(shadow_log_state &state, const shadow_sample &s, std::uint64_t now_ms) {
    const bool changed = !state.emitted || state.pairing != s.pairing.kind || state.valid != s.valid ||
      state.would_refine != s.would_refine || state.would_source != s.would_source;
    if (!changed && now_ms >= state.last_ms && now_ms - state.last_ms < log_interval_ms) return false;
    state = {true, s.pairing.kind, s.valid, s.would_refine, s.would_source, now_ms};
    return true;
  }

  // Adds one shadow sample to the session counters' change_set group.
  inline void add_shadow_counters(ui_counters &counters, const shadow_sample &s) {
    namespace n = ui_counter;
    ++counters[n::change_set_samples];
    if (s.pairing.kind == pair_class::retained) ++counters[n::change_set_retained];
    else if (s.pairing.kind == pair_class::late) ++counters[n::change_set_late];
    else if (s.pairing.kind == pair_class::unavailable) ++counters[n::change_set_unavailable];
    if (s.valid) ++counters[n::change_set_valid];
    if (s.would_refine) ++counters[n::change_set_would_refine];
    if (s.would_decide) ++counters[n::change_set_would_decide];
    if (s.verdict == pair_verdict::verified) ++counters[n::change_set_pair_verified];
    else if (s.verdict == pair_verdict::contradicted) ++counters[n::change_set_pair_contradicted];
  }
}
