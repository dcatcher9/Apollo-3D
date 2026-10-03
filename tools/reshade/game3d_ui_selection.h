// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The selection side of the UI decision framework (docs/reshade-sbs.md, UI
// decision framework): candidate kinds, per-frame validity (V1 opacity, V2
// change sets), the S1 selection predicate, pair comparability (V2) and the
// acceptance signature key (A1). decide() is the whole decision of
// SunshineUIDetectionReduceCS thread 0, which game3d_native.hlsl ports line
// for line with the same uint32 arithmetic: test_game3d_ui_selection_contract
// runs the real reduce against it, ui_detection_replay checks every replay
// case (mirror=match), and the shader's SUNSHINE_UI_SELECTION_REVISION must
// equal revision. Counts stay within the 3840 x 3840 detection domain, so no
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
  inline constexpr std::uint32_t revision = 1;
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

  // One detection's counts as the reduce sums them (decision texels 0-4, 7).
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
  };
  // Counts from decision words (texel t, component c is word 4 t + c); the
  // layer's (texel 7) read zero from fewer than 32 words.
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

  struct decision {
    std::uint32_t source{}, covered{};
    // Offered candidates that passed V1/V2 this frame, in candidate-bit positions.
    std::uint32_t valid_bits{};
    std::size_t none_reason = ui_no_mask::other; // Meaningful when source is 0.
    // Counter words (game3d_ui_counters.h); untrusted_inferred and
    // presented_over_dedicated are invariants, zero by construction.
    bool untrusted_inferred{}, presented_over_dedicated{}, inexact_difference{}, trusted_full{}, full_alpha{};
  };

  // S1: among offered, accepted and valid candidates the first in draw order
  // decides, at any coverage (P1). An unaccepted or invalid candidate never
  // blocks another, except that an offered, accepted declared alpha (UIAlpha
  // or the UI color tag) keeps inferred alpha from deciding even while it is
  // invalid itself. A change set decides as source 5 when partial and as 6
  // when full from an exact pair. Otherwise the hidden-scene routes act on
  // unaccepted inputs under the CPU's held verdicts (8 layer, 9 HUD-less).
  inline decision decide(const counts &c, std::uint32_t offered, std::uint32_t accepted, std::uint32_t flags) {
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
    for (const auto k : draw_order) {
      if (!alpha_kind(k) || !(eligible & bit(k))) continue;
      d.source = source_id(k);
      d.covered = c.covered[alpha_index(k)];
      break;
    }
    if (!d.source && (eligible & candidate::hudless)) {
      if (partial_set) {
        d.source = 5u;
        d.covered = c.changed;
      } else {
        d.source = 6u;
        d.covered = pixels;
      }
    }
    const auto gates = ui_detection::scene_gates_of(d.source, offered, accepted, pixels,
      {c.invalid[alpha_index(kind::ui_alpha)], c.invalid[alpha_index(kind::ui_layer)]}, {c.opaque_ui_alpha, c.opaque_layer},
      c.changed);
    if (!d.source && gates.layer && (flags & ui_detection::per_frame_scene_hold)) {
      d.source = 8u;
      d.covered = pixels;
    } else if (!d.source && gates.hudless && (flags & ui_detection::per_frame_scene_hold_hudless)) {
      d.source = 9u;
      d.covered = pixels;
    }
    const std::uint32_t unaccepted = offered & ~accepted;
    if (!d.source) {
      d.none_reason = gates.layer || gates.hudless ? ui_no_mask::gate_no_hold :
        block && (offered & accepted & valid & inferred_alpha_bits) ? ui_no_mask::presented_blocked :
        (offered & accepted & alpha_bits & ~valid) ? ui_no_mask::trusted_invalid :
        (offered & candidate::layer) && !(valid & candidate::layer) ? ui_no_mask::layer_aside :
        (unaccepted & valid & selective_bits) ? ui_no_mask::unaccepted :
        (offered & candidate::hudless) ? ui_no_mask::difference_failed :
        (unaccepted & valid & alpha_bits & ~selective_bits) ? ui_no_mask::ambiguous :
        !(offered & alpha_bits) ? ui_no_mask::no_candidate : ui_no_mask::other;
    }
    const bool inferred = d.source == 3u || d.source == 4u || d.source == ui_detection::source_layer;
    const bool alpha = (d.source >= 1u && d.source <= 4u) || d.source == ui_detection::source_layer;
    const std::uint32_t inferred_bit = d.source == 3u ? candidate::backbuffer : d.source == 4u ? candidate::current :
      candidate::layer;
    d.untrusted_inferred = inferred && !(accepted & inferred_bit);
    d.presented_over_dedicated = inferred && block;
    d.inexact_difference = (d.source == 5u || d.source == 9u) &&
      (offered & (candidate::hudless | candidate::exact)) == candidate::hudless;
    d.full_alpha = alpha && d.covered * 100u >= pixels * 99u;
    d.trusted_full = d.full_alpha && (offered & (candidate::hudless | candidate::exact)) == (candidate::hudless | candidate::exact) &&
      !c.nonfinite && c.unchanged * 2u >= pixels;
    return d;
  }

  // The adds one detection frame makes to the GPU counter words.
  inline std::array<std::uint32_t, ui_counter_word::count> counter_adds(const decision &d, std::uint32_t flags) {
    std::array<std::uint32_t, ui_counter_word::count> adds{};
    adds[ui_counter_word::detection_frames] = 1;
    if (d.source < ui_counter_word::decided_count) adds[ui_counter_word::decided + d.source] = 1;
    if (!d.source) adds[ui_counter_word::none + d.none_reason] = 1;
    adds[ui_counter_word::untrusted_inferred] = d.untrusted_inferred;
    adds[ui_counter_word::inexact_difference] = d.inexact_difference;
    adds[ui_counter_word::depth_not_current] = (flags & ui_detection::per_frame_depth_not_current) != 0u;
    adds[ui_counter_word::trusted_full] = d.trusted_full;
    adds[ui_counter_word::presented_over_dedicated] = d.presented_over_dedicated;
    adds[ui_counter_word::full_alpha] = d.full_alpha;
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
