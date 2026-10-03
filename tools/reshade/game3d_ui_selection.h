// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The selection side of the UI decision framework (docs/reshade-sbs.md, UI
// decision framework): candidate kinds, per-frame validity (V1 opacity, V2
// change sets), the S1 selection predicate, the one-way judgment counts and
// judges the acceptance ledger reads (A2), the T1 grace of a real frame
// without a decision of its own, the named no-mask reason and refused
// candidate (F1), pair comparability (V2) and the acceptance signature key
// (A1). decide() is the whole decision of SunshineUIDetectionReduceCS thread
// 0, the hold store included, which game3d_native.hlsl ports line for line
// with the same uint32 arithmetic: test_game3d_ui_selection_contract runs the
// real reduce against it, ui_detection_replay checks every replay case
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
  // (texels 8-9), the refused candidate and the frame reason word.
  inline constexpr std::uint32_t revision = 2;
  inline constexpr std::string_view revision_marker = "SUNSHINE_UI_SELECTION_REVISION";

  // Candidate kinds. Declared sources are the game's own UI contract (the
  // UIAlpha and UIColorAndAlpha tags, a HUD-less pair); inferred sources are
  // guesses (the offscreen UI layer, Backbuffer and current color alpha).
  enum class kind : std::uint8_t { ui_alpha, ui_color, ui_layer, backbuffer, current, hudless };
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
      default: return candidate::hudless;
    }
  }
  // The decision source a kind decides as; a HUD-less pair also decides the
  // full-frame source 6 and opens route 9.
  constexpr std::uint32_t source_id(kind k) {
    switch (k) {
      case kind::ui_alpha: return 1u;
      case kind::ui_color: return 2u;
      case kind::ui_layer: return ui_detection::source_layer;
      case kind::backbuffer: return 3u;
      case kind::current: return 4u;
      default: return 5u;
    }
  }
  constexpr bool declared(kind k) { return k == kind::ui_alpha || k == kind::ui_color || k == kind::hudless; }
  constexpr bool alpha_kind(kind k) { return k != kind::hudless; }
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
  inline constexpr std::array<std::string_view, 6> kind_names{"ui_alpha", "ui_color", "ui_layer", "backbuffer", "current",
    "hudless"};
  constexpr std::string_view name(kind k) { return kind_names[std::size_t(k)]; }
  constexpr std::optional<kind> kind_named(std::string_view text) {
    for (std::size_t i = 0; i != kind_names.size(); ++i)
      if (kind_names[i] == text) return kind(i);
    return std::nullopt;
  }
  static_assert(bit(kind::ui_layer) == 0x40u && source_id(kind::ui_layer) == 10u && declared(kind::hudless) &&
    !declared(kind::ui_layer) && alpha_index(kind::current) == 4 && kind_named("ui_color") == kind::ui_color);

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

  // One detection's counts as the reduce sums them (decision texels 0-4, 7-9).
  struct counts {
    std::uint32_t pixels{};
    // Alpha kinds in alpha_index order: UIAlpha, UI color tag, UI layer,
    // Backbuffer, current.
    std::array<std::uint32_t, 5> covered{}, invalid{};
    // Pixels with alpha of at least 254/255.
    std::uint32_t opaque_ui_alpha{}, opaque_ui_color{}, opaque_layer{};
    // The HUD-less pair: changed, unchanged (within half the threshold) and
    // non-finite pixels, lit HUD-less pixels, tiles at least 99% matching.
    std::uint32_t changed{}, unchanged{}, nonfinite{}, lit{}, matching_tiles{};
    // A2, in judged_kinds order: pixels with alpha of at least 1/2, and those
    // of them where an offered exact pair's HUD-less image is lit and
    // unchanged (neither for the one-frame-late layer, E2).
    std::array<std::uint32_t, 3> strong{}, contradicted{};
  };
  // Counts from decision words (texel t, component c is word 4 t + c); the
  // layer's (texel 7) read zero from fewer than 32 words, the one-way counts
  // (texels 8-9) from fewer than 40.
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
    return c;
  }

  // V1: at most 1% invalid pixels (non-finite, outside [0, 1], or, for the
  // layer, beyond the premultiplied bound), whether or not accepted.
  constexpr bool alpha_valid(std::uint32_t invalid, std::uint32_t pixels) { return invalid * 100u <= pixels; }
  // Some coverage but less than 90% of the frame; full is 90% or more.
  constexpr bool selective(std::uint32_t covered, std::uint32_t pixels) { return covered && covered * 10u < pixels * 9u; }
  constexpr bool full(std::uint32_t covered, std::uint32_t pixels) { return covered * 10u >= pixels * 9u; }
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
  };
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
  // when full from an exact pair. Otherwise the hidden-scene routes act on
  // unaccepted inputs under the CPU's held verdicts (8 layer, 9 HUD-less).
  // T1: a real frame that decided no source while an accepted candidate is
  // missing (per_frame_accepted_missing) or offered but invalid has no
  // decision of its own; it applies the previous real frame's own decision
  // once (reused), unless per_frame_hold_reset says there is none, and then
  // no mask. previous is the hold store as the last detection wrote it.
  inline decision decide(const counts &c, std::uint32_t offered, std::uint32_t accepted, std::uint32_t flags,
      const hold_state &previous = {}) {
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
    std::uint32_t source = 0u, covered = 0u;
    for (const auto k : draw_order) {
      if (!alpha_kind(k) || !(eligible & bit(k))) continue;
      source = source_id(k);
      covered = c.covered[alpha_index(k)];
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
    const auto gates = ui_detection::scene_gates_of(source, offered, accepted, pixels,
      {c.invalid[alpha_index(kind::ui_alpha)], c.invalid[alpha_index(kind::ui_layer)]}, {c.opaque_ui_alpha, c.opaque_layer},
      c.changed);
    if (!source && gates.layer && (flags & ui_detection::per_frame_scene_hold)) {
      source = 8u;
      covered = pixels;
    } else if (!source && gates.hudless && (flags & ui_detection::per_frame_scene_hold_hudless)) {
      source = 9u;
      covered = pixels;
    }
    const std::uint32_t unaccepted = offered & ~accepted;
    // F1: the reason, in priority order, and the refused candidate it names.
    if (!source) {
      if (gates.layer || gates.hudless) {
        d.none_reason = ui_no_mask::gate_no_hold;
        d.refused = (gates.layer_slots & 1u) ? candidate::ui_alpha : gates.layer ? candidate::layer : candidate::hudless;
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
    // Counter words: the own decision's invariants and judgments, the applied
    // decision's whole-frame alpha.
    const bool inferred = source == 3u || source == 4u || source == ui_detection::source_layer;
    const std::uint32_t inferred_bit = source == 3u ? candidate::backbuffer : source == 4u ? candidate::current :
      candidate::layer;
    d.untrusted_inferred = inferred && !(accepted & inferred_bit);
    d.presented_over_dedicated = inferred && block;
    d.inexact_difference = (source == 5u || source == 9u) &&
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
  // FG mode is not part of it.
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
}

template<> struct std::hash<sunshine_game3d::ui_selection::signature> {
  std::size_t operator()(const sunshine_game3d::ui_selection::signature &s) const noexcept {
    return std::hash<std::uint64_t>{}(std::uint64_t(s.source_kind) << 56 | std::uint64_t(s.color_space & 0xffu) << 48 |
      std::uint64_t(s.format));
  }
};
