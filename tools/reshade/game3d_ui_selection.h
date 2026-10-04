// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The selection side of the UI decision framework (docs/reshade-sbs.md, UI
// decision framework): candidate kinds, per-frame validity (V1 opacity, V2
// change sets), the S1 selection predicate, the one-way judgment counts and
// judges the acceptance ledger reads (A2), the T1 grace of a real frame
// without a decision of its own, the named no-mask reason and refused
// candidate (F1), pair comparability (V2) and the acceptance signature key
// (A1), the H1 override of a full-frame UI over a hidden scene (its CPU
// hold lives in game3d_scene_guard.h) and the H2 override of a still screen
// without a UI source (its CPU run lives in game3d_still_screen.h). decide()
// is the whole decision of
// SunshineUIDetectionReduceCS thread 0, the hold store included, which
// game3d_native.hlsl ports line for line with the same uint32 arithmetic:
// test_game3d_ui_selection_contract runs the real reduce against it,
// ui_detection_replay checks every replay case
// (mirror=match), and the shader's SUNSHINE_UI_SELECTION_REVISION must equal
// revision. Counts stay within the 3840 x 3840 detection domain, so no
// product below overflows 32 bits. No ReShade dependency.
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

namespace sunshine_game3d::ui_selection {
  // 2 (S2a): the T1 grace with the hold store, the one-way judgment counts
  // (texels 8-9), the refused candidate and the frame reason word. 3 (S2b):
  // H1 replaces the hidden-scene routes 8 and 9, with the informative claims,
  // the opaque Backbuffer and current counts and the h1 word (texel 10) and
  // the pre-UI scene image's evidence in texel 6. 4 (fix 1): the layer's claim
  // (d) needs a layer without alpha whose signature the ledger proved the
  // pre-UI scene image by pixels (per_frame_pre_ui_proven) instead of a
  // V1-invalid layer, with the layer-against-presented pixel counts of texel
  // 11 that prove it (b2 word 4, the layer's pair threshold). 5 (fix 2): H2,
  // a frame without a UI source decision shown flat as source 11 while the
  // CPU's run of still hidden samples pushes still::flatten (b2 word 5), with
  // the stillness counts of texel 12.
  inline constexpr std::uint32_t revision = 5;
  inline constexpr std::string_view revision_marker = "SUNSHINE_UI_SELECTION_REVISION";

  // Candidate kinds. Declared sources are the game's own UI contract (the
  // UIAlpha and UIColorAndAlpha tags, a HUD-less pair); inferred sources are
  // guesses (the offscreen UI layer, Backbuffer and current color alpha).
  // pre_ui is a ledger-only kind, never a candidate: the acceptance ledger's
  // proof that a layer signature holds the pre-UI scene image (H1 d, key
  // pre_ui:<layer format>:<space>, game3d_alpha_auto.h). It has no candidate
  // bit, source, slot or draw rank.
  enum class kind : std::uint8_t { ui_alpha, ui_color, ui_layer, backbuffer, current, hudless, pre_ui };
  // S1 draw order: opacity before change sets, declared before inferred.
  inline constexpr std::array<kind, 6> draw_order{kind::ui_alpha, kind::ui_color, kind::ui_layer, kind::backbuffer,
    kind::current, kind::hudless};

  namespace candidate = ui_detection::candidate;
  inline constexpr std::uint32_t alpha_bits = candidate::ui_alpha | candidate::ui_color | candidate::backbuffer |
    candidate::current | candidate::layer;
  inline constexpr std::uint32_t declared_alpha_bits = candidate::ui_alpha | candidate::ui_color;
  inline constexpr std::uint32_t inferred_alpha_bits = alpha_bits & ~declared_alpha_bits;
  // Every candidate bit; candidate::exact marks the HUD-less pair, not a candidate.
  inline constexpr std::uint32_t candidate_bits = alpha_bits | candidate::hudless;
  static_assert(alpha_bits == 0x4fu && inferred_alpha_bits == 0x4cu && candidate_bits == 0x5fu);

  constexpr std::uint32_t bit(kind k) {
    switch (k) {
      case kind::ui_alpha: return candidate::ui_alpha;
      case kind::ui_color: return candidate::ui_color;
      case kind::ui_layer: return candidate::layer;
      case kind::backbuffer: return candidate::backbuffer;
      case kind::current: return candidate::current;
      case kind::hudless: return candidate::hudless;
      default: return 0u; // pre_ui: not a candidate.
    }
  }
  // The decision source a kind decides as; a full change set from an exact
  // HUD-less pair decides the full-frame source 6.
  constexpr std::uint32_t source_id(kind k) {
    switch (k) {
      case kind::ui_alpha: return 1u;
      case kind::ui_color: return 2u;
      case kind::ui_layer: return ui_detection::source_layer;
      case kind::backbuffer: return 3u;
      case kind::current: return 4u;
      case kind::hudless: return 5u;
      default: return 0u; // pre_ui: decides nothing.
    }
  }
  constexpr bool declared(kind k) { return k == kind::ui_alpha || k == kind::ui_color || k == kind::hudless; }
  constexpr bool alpha_kind(kind k) { return k != kind::hudless && k != kind::pre_ui; }
  // Index in counts::covered and counts::invalid (alpha kinds only).
  constexpr std::size_t alpha_index(kind k) {
    switch (k) {
      case kind::ui_alpha: return 0;
      case kind::ui_color: return 1;
      case kind::ui_layer: return 2;
      case kind::backbuffer: return 3;
      default: return 4;
    }
  }
  inline constexpr std::array<std::string_view, 7> kind_names{"ui_alpha", "ui_color", "ui_layer", "backbuffer", "current",
    "hudless", "pre_ui"};
  constexpr std::string_view name(kind k) { return kind_names[std::size_t(k)]; }
  constexpr std::optional<kind> kind_named(std::string_view text) {
    for (std::size_t i = 0; i != kind_names.size(); ++i)
      if (kind_names[i] == text) return kind(i);
    return std::nullopt;
  }
  static_assert(bit(kind::ui_layer) == 0x40u && source_id(kind::ui_layer) == 10u && declared(kind::hudless) &&
    !declared(kind::ui_layer) && alpha_index(kind::current) == 4 && kind_named("ui_color") == kind::ui_color);
  static_assert(bit(kind::hudless) == 0x10u && source_id(kind::hudless) == 5u && !bit(kind::pre_ui) && !source_id(kind::pre_ui) &&
    !alpha_kind(kind::pre_ui) && !declared(kind::pre_ui) && kind_named("pre_ui") == kind::pre_ui &&
    std::size_t(kind::pre_ui) == draw_order.size() && kind_names.size() == draw_order.size() + 1);

  // The first candidate of a bit set in draw order, zero when none.
  constexpr std::uint32_t first_in_draw_order(std::uint32_t bits) {
    for (const auto k : draw_order)
      if (bits & bit(k)) return bit(k);
    return 0u;
  }
  // The kind name of one candidate bit (a refused candidate), "none" for zero
  // or anything else.
  constexpr std::string_view candidate_name(std::uint32_t candidate_bit) {
    for (const auto k : draw_order)
      if (candidate_bit == bit(k)) return name(k);
    return "none";
  }
  static_assert(first_in_draw_order(0x5cu) == 0x40u && first_in_draw_order(0x1cu) == 0x4u && !first_in_draw_order(0x20u) &&
    candidate_name(0x40u) == "ui_layer" && candidate_name(0u) == "none" && candidate_name(0x3u) == "none");

  // A2: the kinds the one-way test judges (inferred alpha), in the order of
  // counts::strong and counts::contradicted and of decision texels 8 and 9.
  inline constexpr std::array<kind, 3> judged_kinds{kind::ui_layer, kind::backbuffer, kind::current};
  // A2 judge (a): an exact change set, valid this frame (V2), partial or full;
  // acceptance is not required. offered and valid_bits in candidate bits.
  constexpr bool exact_judge(std::uint32_t offered, std::uint32_t valid_bits) {
    return (offered & (candidate::hudless | candidate::exact)) == (candidate::hudless | candidate::exact) &&
      (valid_bits & candidate::hudless) != 0u;
  }
  // A2, the one-way lit-pixel disagreement: at least a tenth of a judged
  // source's strong pixels (alpha of at least 1/2) lie where the exact
  // HUD-less image is lit and unchanged. Dims and tints over dark or changed
  // pixels never meet it.
  constexpr bool one_way_contradicted(std::uint32_t strong, std::uint32_t contradicted) {
    return strong && std::uint64_t(contradicted) * 10u >= strong;
  }
  // A2 judge (b): a declared alpha's coverage and an inferred alpha's differ
  // by at least a tenth of the frame.
  constexpr bool coverage_disagrees(std::uint32_t declared_covered, std::uint32_t inferred_covered, std::uint32_t pixels) {
    const std::uint64_t difference = declared_covered > inferred_covered ? declared_covered - inferred_covered :
      inferred_covered - declared_covered;
    return pixels && difference * 10u >= pixels;
  }
  static_assert(exact_judge(0x34u, 0x14u) && !exact_judge(0x14u, 0x14u) && !exact_judge(0x34u, 0x4u) &&
    one_way_contradicted(1000u, 100u) && !one_way_contradicted(1000u, 99u) && !one_way_contradicted(0u, 0u) &&
    coverage_disagrees(0u, 100u, 1000u) && !coverage_disagrees(150u, 51u, 1000u) && !coverage_disagrees(0u, 0u, 0u));

  // One detection's counts as the reduce sums them (decision texels 0-4, 7-10).
  struct counts {
    std::uint32_t pixels{};
    // Alpha kinds in alpha_index order: UIAlpha, UI color tag, UI layer,
    // Backbuffer, current.
    std::array<std::uint32_t, 5> covered{}, invalid{};
    // Pixels with alpha of at least 254/255.
    std::uint32_t opaque_ui_alpha{}, opaque_ui_color{}, opaque_layer{}, opaque_backbuffer{}, opaque_current{};
    // The HUD-less pair: changed, unchanged (within half the threshold) and
    // non-finite pixels, lit HUD-less pixels, tiles at least 99% matching.
    std::uint32_t changed{}, unchanged{}, nonfinite{}, lit{}, matching_tiles{};
    // A2, in judged_kinds order: pixels with alpha of at least 1/2, and those
    // of them where an offered exact pair's HUD-less image is lit and
    // unchanged (neither for the one-frame-late layer, E2).
    std::array<std::uint32_t, 3> strong{}, contradicted{};
    // Texel 11 (revision 4): the offscreen UI layer against the presented
    // frame at 8 times their pair threshold: matching pixels, lit layer
    // pixels, lit presented pixels, lit presented pixels that differ. No
    // decision reads them; the acceptance ledger proves the layer from them.
    std::uint32_t pre_ui_match{}, pre_ui_lit{}, presented_lit{}, presented_lit_differs{};
  };
  // Counts from decision words (texel t, component c is word 4 t + c); the
  // layer's (texel 7) read zero from fewer than 32 words, the one-way counts
  // (texels 8-9) from fewer than 40, the opaque Backbuffer and current counts
  // (texel 10) from fewer than 44, the pre-UI pixel counts (texel 11) from
  // fewer than 48.
  inline counts counts_from_words(const std::uint32_t *words, std::size_t n) {
    namespace word = ui_detection::decision_word;
    const auto at = [&](std::size_t i) { return i < n ? words[i] : 0u; };
    counts c;
    c.pixels = at(word::pixels);
    c.matching_tiles = at(word::matching_tiles);
    c.changed = at(word::hudless_changed);
    c.unchanged = at(word::hudless_unchanged);
    c.nonfinite = at(word::hudless_invalid);
    c.lit = at(word::hudless_lit);
    c.covered = {at(word::alpha_covered), at(word::alpha_covered + 1), at(word::layer_covered), at(word::alpha_covered + 2),
      at(word::alpha_covered + 3)};
    c.invalid = {at(word::alpha_invalid), at(word::alpha_invalid + 1), at(word::layer_invalid), at(word::alpha_invalid + 2),
      at(word::alpha_invalid + 3)};
    c.opaque_ui_alpha = at(word::alpha_opaque);
    c.opaque_ui_color = at(word::alpha_opaque + 1);
    c.opaque_layer = at(word::layer_opaque);
    if (n >= 4u * ui_detection::judgment_decision_texels)
      for (std::size_t i = 0; i != judged_kinds.size(); ++i) {
        c.strong[i] = words[word::strong + i];
        c.contradicted[i] = words[word::contradicted + i];
      }
    if (n >= 4u * ui_detection::h1_decision_texels) {
      c.opaque_backbuffer = words[word::opaque_backbuffer];
      c.opaque_current = words[word::opaque_current];
    }
    if (n >= 4u * ui_detection::pre_ui_decision_texels) {
      c.pre_ui_match = words[word::pre_ui_match];
      c.pre_ui_lit = words[word::pre_ui_image_lit];
      c.presented_lit = words[word::presented_lit];
      c.presented_lit_differs = words[word::presented_lit_differs];
    }
    return c;
  }
  // Pixels with alpha of at least 254/255 of one alpha kind.
  constexpr std::uint32_t opaque_of(const counts &c, kind k) {
    switch (k) {
      case kind::ui_alpha: return c.opaque_ui_alpha;
      case kind::ui_color: return c.opaque_ui_color;
      case kind::ui_layer: return c.opaque_layer;
      case kind::backbuffer: return c.opaque_backbuffer;
      default: return c.opaque_current;
    }
  }
  // H1: opaque-full, alpha of at least 254/255 on at least
  // scene::opaque_percent (99%) of the pixels.
  constexpr bool opaque_full(std::uint32_t opaque, std::uint32_t pixels) {
    return pixels && opaque * 100u >= pixels * ui_detection::scene::opaque_percent;
  }
  // H1 (d): the pre-UI scene image of an offer, whose claim the reduce
  // computes: the HUD-less image when one is offered, else the offscreen UI
  // layer's colour (ui_detection::pre_ui_image).
  constexpr std::uint32_t pre_ui_image_of(std::uint32_t offered) {
    return (offered & candidate::hudless) ? ui_detection::pre_ui_image::hudless :
      (offered & candidate::layer) ? ui_detection::pre_ui_image::layer : ui_detection::pre_ui_image::none;
  }
  static_assert(opaque_full(99u, 100u) && !opaque_full(98u, 100u) && !opaque_full(0u, 0u) && pre_ui_image_of(0x50u) == 1u &&
    pre_ui_image_of(0x48u) == 2u && pre_ui_image_of(0x2fu) == 0u);

  // V1: at most 1% invalid pixels (non-finite, outside [0, 1], or, for the
  // layer, beyond the premultiplied bound), whether or not accepted.
  constexpr bool alpha_valid(std::uint32_t invalid, std::uint32_t pixels) { return invalid * 100u <= pixels; }
  // Some coverage but less than 90% of the frame; full is 90% or more.
  constexpr bool selective(std::uint32_t covered, std::uint32_t pixels) { return covered && covered * 10u < pixels * 9u; }
  constexpr bool full(std::uint32_t covered, std::uint32_t pixels) { return covered * 10u >= pixels * 9u; }
  // H1 (d), fix 1: a sample proving an offscreen layer without alpha the
  // pre-UI scene image. Its colour equals the presented frame (8 times the
  // pair threshold) on a full share of the pixels (full: 90%) while it is lit
  // on at least half of them, as change_set_full requires of a HUD-less image
  // (a black image is no scene: a transparent black UI layer equals a black
  // presented frame everywhere). The caller also requires a layer without
  // coverage.
  constexpr bool pre_ui_match(std::uint32_t matched, std::uint32_t lit, std::uint32_t pixels) {
    return pixels && full(matched, pixels) && lit * 2u >= pixels;
  }
  static_assert(pre_ui_match(90u, 50u, 100u) && !pre_ui_match(89u, 100u, 100u) && !pre_ui_match(100u, 49u, 100u) &&
    !pre_ui_match(0u, 0u, 0u));
  // V2, a partial change set: changed pixels within broad unchanged scene
  // evidence in clean tiles (decides as source 5).
  constexpr bool change_set_selective(const counts &c) {
    return !c.nonfinite && c.changed && c.changed * 4u < c.pixels && c.unchanged * 100u >= c.pixels * 75u &&
      c.matching_tiles >= 128u;
  }
  // V2, a full change set: an exact pair changed nearly everywhere while the
  // HUD-less image is lit (decides as source 6).
  constexpr bool change_set_full(const counts &c, bool exact) {
    return exact && !c.nonfinite && c.changed * 100u >= c.pixels * 98u && c.lit * 2u >= c.pixels;
  }

  // T1: the GPU hold store of the last real detection (ui_detection::hold):
  // whether it decided on its own (own) or not (spent), and the decision it
  // applied. Default: none, as with nothing bound at u5.
  struct hold_state {
    std::uint32_t state{}, source{}, covered{};
  };

  struct decision {
    // The applied decision: the frame's own, or under the T1 grace (reused)
    // the previous real frame's from the hold store.
    std::uint32_t source{}, covered{};
    // The frame's own decision (S1).
    std::uint32_t own_source{}, own_covered{};
    // Offered candidates that passed V1/V2 this frame, in candidate-bit positions.
    std::uint32_t valid_bits{};
    // The own decision's reason when it decided no source (F1).
    std::size_t none_reason = ui_no_mask::other;
    // The own decision's refused candidate bit (F1): the first candidate in
    // draw order matching none_reason; zero when it decided or none matches.
    std::uint32_t refused{};
    // Decision word frame_reason: none_reason, or frame_reason_decided when
    // the own decision decided a source.
    std::uint32_t frame_reason{};
    // Judged kinds (layer, Backbuffer, current; offered and V1-valid) that a
    // valid exact pair contradicts in the one-way test this frame (A2).
    std::uint32_t contradicted_bits{};
    // T1: the frame had no decision of its own and applied the previous real
    // frame's own decision.
    bool reused{};
    // The hold store this detection writes.
    hold_state next{};
    // Counter words (game3d_ui_counters.h); untrusted_inferred and
    // presented_over_dedicated are invariants, zero by construction.
    // full_alpha is of the applied decision, the others of the own decision.
    bool untrusted_inferred{}, presented_over_dedicated{}, inexact_difference{}, contradicted{}, full_alpha{};
    // H1: the raw informative full claims before refutation (candidate bits |
    // ui_detection::claim_pre_ui), the S1 winner's source, and whether H1
    // overrode it with source 8.
    std::uint32_t claims{}, s1_source{};
    bool h1{};
    // H2: the applied decision had no source and was not reused, and the
    // CPU pushed still::flatten, so the frame is shown flat as source 11.
    // The hold store (next), the frame reason and the refused candidate keep
    // the decision before H2.
    bool still{};
  };
  // Decision word h1 of a decision (texel 10 .w).
  constexpr std::uint32_t h1_word(const decision &d) { return d.s1_source | (d.h1 ? ui_detection::h1_applied : 0u); }
  // Decision word frame_reason of a decision (texel 9 .w).
  constexpr std::uint32_t frame_reason_word(const decision &d) {
    return d.frame_reason | (d.reused ? ui_detection::frame_reason_reused : 0u);
  }
  // The frame reason's name: a ui_no_mask name, or "decided".
  inline std::string_view frame_reason_name(std::uint32_t word) {
    const auto reason = word & ui_detection::frame_reason_reason_mask;
    if (reason == ui_detection::frame_reason_decided) return "decided";
    return reason < ui_no_mask::count ? ui_no_mask::names[reason] : std::string_view("other");
  }

  // S1: among offered, accepted and valid candidates the first in draw order
  // decides, at any coverage (P1). An unaccepted or invalid candidate never
  // blocks another, except that an offered, accepted declared alpha (UIAlpha
  // or the UI color tag) keeps inferred alpha from deciding even while it is
  // invalid itself. A change set decides as source 5 when partial and as 6
  // when full from an exact pair.
  // H1 (M5): some valid candidate makes an informative full claim: (a) an
  // alpha S1 may select (eligible: accepted, and not an inferred alpha the
  // declared-alpha block keeps out) that is opaque-full, (b) the offscreen
  // UI layer, a target
  // proven cleared to transparent, V1-valid and opaque-full whether accepted
  // or not, (c) an exact full change set whether accepted or not, or (d) a
  // pre-UI scene image (claim_pre_ui): the HUD-less image changed on at least
  // 90% of pixels, or else an offered layer without coverage (alpha zero on
  // every pixel) whose signature the acceptance ledger proved the pre-UI
  // scene image by pixels (per_frame_pre_ui_proven, fix 1). An unaccepted
  // UIAlpha, UI color tag, Backbuffer or current alpha is never informative,
  // nor is an accepted inferred one the block keeps out. A claim acts unless
  // the CPU refuted its signature (per_frame_refuted_mask), and (d) only
  // while the CPU's held samples read the pre-UI image visible
  // (per_frame_pre_ui_visible). While the
  // CPU holds a hidden verdict (per_frame_scene_hidden) and the depth is
  // this frame's, an acting claim shows the frame flat as source 8, whatever
  // S1 selected, unless the S1 winner is already flat everywhere (source 6,
  // or an alpha winner opaque on every pixel).
  // T1: a real frame that decided no source while an accepted candidate is
  // missing (per_frame_accepted_missing) or offered but invalid has no
  // decision of its own; it applies the previous real frame's own decision
  // once (reused), unless per_frame_hold_reset says there is none, and then
  // no mask. previous is the hold store as the last detection wrote it.
  // H2 (M5, fix 2): an applied decision without a source that T1 did not
  // reuse is shown flat as source 11 while the CPU pushes still::flatten in
  // still (b2 word 5): its run of still samples that read the presented
  // frame hidden (game3d_still_screen.h). It runs after S1, H1 and T1, so an
  // accepted source deciding (even an empty mask) and H1 always win, and the
  // hold store keeps the decision before it.
  inline decision decide(const counts &c, std::uint32_t offered, std::uint32_t accepted, std::uint32_t flags,
      const hold_state &previous = {}, std::uint32_t still = 0u) {
    decision d;
    const std::uint32_t pixels = c.pixels;
    std::uint32_t valid = 0u, selective_bits = 0u;
    for (const auto k : draw_order) {
      if (!alpha_kind(k)) continue;
      const auto i = alpha_index(k);
      if (alpha_valid(c.invalid[i], pixels)) valid |= bit(k);
      if (selective(c.covered[i], pixels)) selective_bits |= bit(k);
    }
    const bool partial_set = change_set_selective(c), full_set = change_set_full(c, (offered & candidate::exact) != 0u);
    if (partial_set || full_set) valid |= candidate::hudless;
    if (partial_set) selective_bits |= candidate::hudless;
    valid &= offered & candidate_bits;
    d.valid_bits = valid;
    const bool block = (offered & accepted & declared_alpha_bits) != 0u;
    const std::uint32_t eligible = offered & accepted & valid & (block ? ~inferred_alpha_bits : ~0u);
    std::uint32_t source = 0u, covered = 0u, winner_opaque = 0u;
    for (const auto k : draw_order) {
      if (!alpha_kind(k) || !(eligible & bit(k))) continue;
      source = source_id(k);
      covered = c.covered[alpha_index(k)];
      winner_opaque = opaque_of(c, k);
      break;
    }
    if (!source && (eligible & candidate::hudless)) {
      if (partial_set) {
        source = 5u;
        covered = c.changed;
      } else {
        source = 6u;
        covered = pixels;
      }
    }
    // H1: the informative full claims of this frame, (a)-(d) above.
    const std::uint32_t s1_source = source;
    std::uint32_t claims = 0u;
    for (const auto k : draw_order) {
      if (!alpha_kind(k) || !(valid & bit(k)) || !opaque_full(opaque_of(c, k), pixels)) continue;
      if ((eligible & bit(k)) || k == kind::ui_layer) claims |= bit(k);
    }
    if (full_set && (valid & candidate::hudless)) claims |= candidate::hudless;
    const std::uint32_t image = pre_ui_image_of(offered);
    if (pixels && ((image == ui_detection::pre_ui_image::hudless &&
                     c.changed * 100u >= pixels * ui_detection::scene::hudless_changed_percent) ||
                    (image == ui_detection::pre_ui_image::layer && !c.covered[alpha_index(kind::ui_layer)] &&
                      (flags & ui_detection::per_frame_pre_ui_proven))))
      claims |= ui_detection::claim_pre_ui;
    const std::uint32_t refuted = (flags >> ui_detection::per_frame_refuted_shift) & candidate_bits;
    const std::uint32_t acting = (claims & candidate_bits & ~refuted) |
      ((flags & ui_detection::per_frame_pre_ui_visible) ? (claims & ui_detection::claim_pre_ui) : 0u);
    const bool alpha_winner = (source >= 1u && source <= 4u) || source == ui_detection::source_layer;
    const bool winner_full = source == 6u || (alpha_winner && pixels && winner_opaque == pixels);
    const bool h1 = acting && (flags & ui_detection::per_frame_scene_hidden) &&
      !(flags & ui_detection::per_frame_depth_not_current) && !winner_full;
    if (h1) {
      source = 8u;
      covered = pixels;
    }
    d.claims = claims;
    d.s1_source = s1_source;
    d.h1 = h1;
    const std::uint32_t unaccepted = offered & ~accepted;
    // F1: the reason, in priority order, and the refused candidate it names.
    // An acting claim that H1 did not apply (no held hidden verdict, or depth
    // that is not this frame's) refuses its first claimant in draw order, and
    // the pre-UI image alone its image (HUD-less when offered, else the layer).
    if (!source) {
      if (acting) {
        d.none_reason = ui_no_mask::gate_no_hold;
        d.refused = first_in_draw_order(acting & candidate_bits);
        if (!d.refused) d.refused = (offered & candidate::hudless) ? candidate::hudless : candidate::layer;
      } else if (block && (offered & accepted & valid & inferred_alpha_bits)) {
        d.none_reason = ui_no_mask::presented_blocked;
        d.refused = first_in_draw_order(offered & accepted & valid & inferred_alpha_bits);
      } else if (offered & accepted & alpha_bits & ~valid) {
        d.none_reason = ui_no_mask::trusted_invalid;
        d.refused = first_in_draw_order(offered & accepted & alpha_bits & ~valid);
      } else if ((offered & candidate::layer) && !(valid & candidate::layer)) {
        d.none_reason = ui_no_mask::layer_aside;
        d.refused = candidate::layer;
      } else if (unaccepted & valid & selective_bits) {
        d.none_reason = ui_no_mask::unaccepted;
        d.refused = first_in_draw_order(unaccepted & valid & selective_bits);
      } else if (offered & candidate::hudless) {
        d.none_reason = ui_no_mask::difference_failed;
        d.refused = candidate::hudless;
      } else if (unaccepted & valid & alpha_bits & ~selective_bits) {
        d.none_reason = ui_no_mask::ambiguous;
        d.refused = first_in_draw_order(unaccepted & valid & alpha_bits & ~selective_bits);
      } else {
        d.none_reason = !(offered & alpha_bits) ? ui_no_mask::no_candidate : ui_no_mask::other;
      }
    }
    d.frame_reason = source ? ui_detection::frame_reason_decided : std::uint32_t(d.none_reason);
    // A2: the judged kinds a valid exact pair contradicts this frame.
    if (exact_judge(offered, valid))
      for (std::size_t i = 0; i != judged_kinds.size(); ++i)
        if ((valid & bit(judged_kinds[i])) && one_way_contradicted(c.strong[i], c.contradicted[i]))
          d.contradicted_bits |= bit(judged_kinds[i]);
    // T1: no decision of its own, so the previous real frame's own decision
    // once; per_frame_hold_reset means there is none in this chain.
    const bool no_own = !source && ((flags & ui_detection::per_frame_accepted_missing) ||
      (offered & accepted & candidate_bits & ~valid));
    const std::uint32_t prior = (flags & ui_detection::per_frame_hold_reset) ? ui_detection::hold::none : previous.state;
    d.reused = no_own && prior == ui_detection::hold::own;
    d.own_source = source;
    d.own_covered = covered;
    d.source = d.reused ? previous.source : source;
    d.covered = d.reused ? previous.covered : covered;
    d.next = {no_own ? ui_detection::hold::spent : ui_detection::hold::own, d.source, d.covered};
    // H2: after T1, a frame applying no source of its own shows flat.
    if (!d.source && !d.reused && (still & ui_detection::still::flatten)) {
      d.source = ui_detection::source_still;
      d.covered = pixels;
      d.still = true;
    }
    // Counter words: the own decision's invariants and judgments, the applied
    // decision's whole-frame alpha.
    const bool inferred = source == 3u || source == 4u || source == ui_detection::source_layer;
    const std::uint32_t inferred_bit = source == 3u ? candidate::backbuffer : source == 4u ? candidate::current :
      candidate::layer;
    d.untrusted_inferred = inferred && !(accepted & inferred_bit);
    d.presented_over_dedicated = inferred && block;
    d.inexact_difference = source == 5u &&
      (offered & (candidate::hudless | candidate::exact)) == candidate::hudless;
    d.contradicted = inferred && (accepted & inferred_bit) && (d.contradicted_bits & inferred_bit);
    const bool alpha = (d.source >= 1u && d.source <= 4u) || d.source == ui_detection::source_layer;
    d.full_alpha = alpha && d.covered * 100u >= pixels * 99u;
    return d;
  }

  // The adds one detection frame makes to the GPU counter words: the applied
  // decision (detection frame, decided source, reused, whole-frame alpha,
  // depth), and the own decision's reason without a mask and judgments.
  inline std::array<std::uint32_t, ui_counter_word::count> counter_adds(const decision &d, std::uint32_t flags) {
    std::array<std::uint32_t, ui_counter_word::count> adds{};
    adds[ui_counter_word::detection_frames] = 1;
    if (d.source < ui_counter_word::decided_count) adds[ui_counter_word::decided + d.source] = 1;
    if (!d.source) adds[ui_counter_word::none + d.none_reason] = 1;
    adds[ui_counter_word::untrusted_inferred] = d.untrusted_inferred;
    adds[ui_counter_word::inexact_difference] = d.inexact_difference;
    adds[ui_counter_word::depth_not_current] = (flags & ui_detection::per_frame_depth_not_current) != 0u;
    adds[ui_counter_word::contradicted] = d.contradicted;
    adds[ui_counter_word::presented_over_dedicated] = d.presented_over_dedicated;
    adds[ui_counter_word::full_alpha] = d.full_alpha;
    adds[ui_counter_word::reused] = d.reused;
    return adds;
  }

  // V2 comparability of two snapshots from their own encodings: the typed
  // DXGI format and the swapchain color space (1 sRGB, 2 scRGB, 3 PQ, 4 HLG).
  struct encoding {
    std::uint32_t format{}, color_space{};
  };
  namespace detail {
    enum class family : std::uint8_t { unknown, float_color, unorm, srgb_typed };
    struct format_class {
      family kind = family::unknown;
      std::uint32_t bits{};
    };
    // Color formats a HUD-less pair can hold; typeless allocations read as
    // their default typed format (UNORM, or FLOAT for 16- and 32-bit floats).
    constexpr format_class classify(std::uint32_t dxgi) {
      switch (dxgi) {
        case 1: case 2: case 9: case 10: case 26: return {family::float_color, 16u};
        case 11: return {family::unorm, 16u};
        case 23: case 24: return {family::unorm, 10u};
        case 27: case 28: case 87: case 88: case 90: case 92: return {family::unorm, 8u};
        case 29: case 91: case 93: return {family::srgb_typed, 8u};
        default: return {};
      }
    }
  }
  // The difference threshold for a pair, or none when the two encodings do
  // not share a transfer: float color is linear and comparable only under
  // scRGB (the shader's relative scRGB tolerance); non-sRGB UNORM carries the
  // swapchain's transfer; *_SRGB views decode and compare only with each
  // other. Two 10-bit UNORM images use 4/1023, any other UNORM pair the
  // coarser 2/255, mixed 8/10-bit included.
  inline std::optional<float> comparable(encoding a, encoding b) {
    const auto x = detail::classify(a.format), y = detail::classify(b.format);
    if (a.color_space != b.color_space || a.color_space < 1u || a.color_space > 4u ||
        x.kind == detail::family::unknown || y.kind == detail::family::unknown || x.kind != y.kind)
      return std::nullopt;
    if (x.kind == detail::family::float_color) return a.color_space == 2u ? std::optional<float>(.005f) : std::nullopt;
    return x.bits == 10u && y.bits == 10u ? 4.f / 1023.f : 2.f / 255.f;
  }

  // A1: the acceptance key of one source, "<kind>:<dxgi decimal>:<space>".
  // FG mode is not part of it. The ledger-only kind pre_ui keys a layer
  // signature's pre-UI proof (pre_ui_key).
  inline constexpr std::array<std::string_view, 5> color_space_names{"unknown", "srgb", "scrgb", "pq", "hlg"};
  struct signature {
    kind source_kind{};
    std::uint32_t format{}, color_space{};

    std::string key() const {
      const auto space = color_space < color_space_names.size() ? color_space_names[color_space] : color_space_names[0];
      return std::string(name(source_kind)) + ':' + std::to_string(format) + ':' + std::string(space);
    }
    // A key as key() writes it, surrounding blanks ignored; anything else, a
    // legacy integer bitmask included, is none.
    static std::optional<signature> parse(std::string_view text) {
      while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
      while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' || text.back() == '\n'))
        text.remove_suffix(1);
      const auto first = text.find(':'), second = first == std::string_view::npos ? first : text.find(':', first + 1);
      if (second == std::string_view::npos || text.find(':', second + 1) != std::string_view::npos) return std::nullopt;
      const auto source = kind_named(text.substr(0, first));
      const auto number = text.substr(first + 1, second - first - 1), space = text.substr(second + 1);
      if (!source || number.empty() || number.size() > 9) return std::nullopt;
      signature result{*source, 0u, 0u};
      for (const char digit : number) {
        if (digit < '0' || digit > '9') return std::nullopt;
        result.format = result.format * 10u + std::uint32_t(digit - '0');
      }
      for (std::size_t i = 0; i != color_space_names.size(); ++i)
        if (color_space_names[i] == space) {
          result.color_space = std::uint32_t(i);
          return result;
        }
      return std::nullopt;
    }
    friend bool operator==(const signature &a, const signature &b) {
      return a.source_kind == b.source_kind && a.format == b.format && a.color_space == b.color_space;
    }
    friend bool operator!=(const signature &a, const signature &b) { return !(a == b); }
    friend bool operator<(const signature &a, const signature &b) {
      return std::tie(a.source_kind, a.format, a.color_space) < std::tie(b.source_kind, b.format, b.color_space);
    }
  };
  // H1 (d), fix 1: the ledger key of a layer signature's pre-UI proof,
  // "pre_ui:<layer format>:<space>": another format or colour space is
  // another key.
  constexpr signature pre_ui_key(const signature &layer) { return {kind::pre_ui, layer.format, layer.color_space}; }
  static_assert(pre_ui_key({kind::ui_layer, 87u, 1u}).source_kind == kind::pre_ui &&
    pre_ui_key({kind::ui_layer, 87u, 1u}).format == 87u && pre_ui_key({kind::ui_layer, 87u, 3u}).color_space == 3u);
}

template<> struct std::hash<sunshine_game3d::ui_selection::signature> {
  std::size_t operator()(const sunshine_game3d::ui_selection::signature &s) const noexcept {
    return std::hash<std::uint64_t>{}(std::uint64_t(s.source_kind) << 56 | std::uint64_t(s.color_space & 0xffu) << 48 |
      std::uint64_t(s.format));
  }
};
