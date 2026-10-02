// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Automatic UI detection contract shared by the renderer, Dump 3D and the
// offline replay tools. docs/reshade-sbs.md (UI detection flags and decision
// texels) owns the layout; game3d_native.hlsl mirrors the flag values as
// SUNSHINE_UI_STORED_* and SUNSHINE_UI_PER_FRAME_* defines, and
// test_game3d_ui_layer fails when the two disagree. No ReShade dependency.
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace sunshine_game3d {
  // Value of an integer shader capability marker "#define NAME value"; zero
  // when absent, as in older embedded replay shaders.
  inline std::uint32_t shader_marker(std::string_view source, std::string_view name) {
    const std::string key = "#define " + std::string(name) + ' ';
    const auto at = source.find(key);
    std::uint32_t value = 0;
    if (at != std::string_view::npos) std::from_chars(source.data() + at + key.size(), source.data() + source.size(), value);
    return value;
  }
}

namespace sunshine_game3d::ui_detection {
  // Sunshine_UIDetectionFlags, b2 word 3. Stored bits belong to the UI color
  // slot's source: the renderer keeps them between frames and they are part
  // of detection_decision_key.
  inline constexpr std::uint32_t stored_premultiplied = 0x1u; // Admit UI color+alpha only while premultiplied.
  inline constexpr std::uint32_t stored_hdr_headroom = 0x2u;  // With a float layer's HDR headroom.
  inline constexpr std::uint32_t stored_late_layer = 0x4u;    // The slot holds the one-frame-late offscreen UI layer.
  inline constexpr std::uint32_t stored_stage2 = 0x8u;        // Reserved.
  inline constexpr std::uint32_t stored_mask = 0xffffu;
  // Per-frame bits ride in the pushed flags word of one render only. They are
  // never stored in the renderer's detection flags and never key a decision.
  inline constexpr std::uint32_t per_frame_scene_hold = 0x10000u;
  inline constexpr std::uint32_t per_frame_sample = 0x20000u;
  inline constexpr std::uint32_t per_frame_depth_not_current = 0x40000u;
  inline constexpr std::uint32_t per_frame_mask = 0xffff0000u;
  static_assert((stored_mask & per_frame_mask) == 0 && (stored_mask | per_frame_mask) == 0xffffffffu);
  // The game3d_native.hlsl define mirroring each flag.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 7> hlsl_flag_defines{{
    {"SUNSHINE_UI_STORED_PREMULTIPLIED", stored_premultiplied},
    {"SUNSHINE_UI_STORED_HDR_HEADROOM", stored_hdr_headroom},
    {"SUNSHINE_UI_STORED_LATE_LAYER", stored_late_layer},
    {"SUNSHINE_UI_STORED_STAGE2", stored_stage2},
    {"SUNSHINE_UI_PER_FRAME_SCENE_HOLD", per_frame_scene_hold},
    {"SUNSHINE_UI_PER_FRAME_SAMPLE", per_frame_sample},
    {"SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT", per_frame_depth_not_current},
  }};

  // Flags for an offscreen UI layer copy in the UI color slot; a tagged
  // UIColorAndAlpha has none.
  constexpr std::uint32_t layer_detection_flags(bool float_layer) {
    return stored_premultiplied | stored_late_layer | (float_layer ? stored_hdr_headroom : 0u);
  }
  static_assert(layer_detection_flags(false) == 5u && layer_detection_flags(true) == 7u);
  // A layer of this DXGI format has HDR headroom: the whole R32G32B32A32
  // (1-4) or R16G16B16A16 (9-14) typeless family, UNORM included, as the
  // renderer names layer formats by their typeless format.
  constexpr bool float_layer_format(std::uint32_t dxgi_format) {
    return (dxgi_format >= 1u && dxgi_format <= 4u) || (dxgi_format >= 9u && dxgi_format <= 14u);
  }
  static_assert(float_layer_format(10u) && float_layer_format(11u) && float_layer_format(2u) &&
    !float_layer_format(28u) && !float_layer_format(87u) && !float_layer_format(24u));

  // Shader markers sizing the detection resources, and their values for
  // shaders without them.
  inline constexpr std::string_view decision_texels_marker = "SUNSHINE_UI_DECISION_TEXELS";
  inline constexpr std::string_view scene_evidence_images_marker = "SUNSHINE_UI_SCENE_EVIDENCE_IMAGES";
  inline constexpr std::uint32_t default_decision_texels = 5, default_scene_evidence_images = 0;
  // The CPU parses decision texels 0-4.
  inline constexpr std::uint32_t min_decision_texels = 5, max_decision_texels = 16, max_scene_evidence_images = 4;
  // Statistics rows: 64 rows of per-tile counts, 96 per evidence image and one
  // copy row when there is any image.
  constexpr std::uint32_t statistics_rows(std::uint32_t images) {
    return 64u + 96u * images + (images ? 1u : 0u);
  }
  static_assert(statistics_rows(0) == 64u && statistics_rows(2) == 257u && statistics_rows(3) == 353u);

  // Words of the decision readback (the decision texel table): texel t,
  // component c is word 4 * t + c.
  namespace decision_word {
    inline constexpr std::size_t source = 0, covered = 1, pixels = 2, matching_tiles = 3;
    inline constexpr std::size_t candidates = 4, hudless_changed = 5, hudless_unchanged = 6, hudless_invalid = 7;
    inline constexpr std::size_t alpha_covered = 8, alpha_invalid = 12; // Four words each, one per alpha candidate.
    inline constexpr std::size_t hudless_lit = 16, trusted = 17;
    // Texel 6, which no shader writes yet: the presented image's hidden-scene
    // statistic {n, asuint(kinterp), asuint(koct), valid | held << 1 | verdict << 2}.
    inline constexpr std::size_t scene_n = 24, scene_kinterp = 25, scene_koct = 26, scene_state = 27;
  }
  static_assert(decision_word::trusted < 4 * min_decision_texels);
  enum class scene_verdict : std::uint32_t { none = 0, hidden = 1, ambiguous = 2, visible = 3 };
  constexpr scene_verdict scene_state_verdict(std::uint32_t state) { return scene_verdict((state >> 2) & 3u); }
}
