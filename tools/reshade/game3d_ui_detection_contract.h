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
  inline constexpr std::uint32_t per_frame_scene_hold = 0x10000u;         // The CPU holds the layer route's hidden-scene verdict.
  inline constexpr std::uint32_t per_frame_sample = 0x20000u;             // Reserved: no render pushes it.
  inline constexpr std::uint32_t per_frame_depth_not_current = 0x40000u;  // The consumed depth is reused or generated.
  inline constexpr std::uint32_t per_frame_scene_hold_hudless = 0x80000u; // The CPU holds the HUD-less route's verdict.
  inline constexpr std::uint32_t per_frame_mask = 0xffff0000u;
  static_assert((stored_mask & per_frame_mask) == 0 && (stored_mask | per_frame_mask) == 0xffffffffu);
  // The game3d_native.hlsl define mirroring each flag.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 8> hlsl_flag_defines{{
    {"SUNSHINE_UI_STORED_PREMULTIPLIED", stored_premultiplied},
    {"SUNSHINE_UI_STORED_HDR_HEADROOM", stored_hdr_headroom},
    {"SUNSHINE_UI_STORED_LATE_LAYER", stored_late_layer},
    {"SUNSHINE_UI_STORED_STAGE2", stored_stage2},
    {"SUNSHINE_UI_PER_FRAME_SCENE_HOLD", per_frame_scene_hold},
    {"SUNSHINE_UI_PER_FRAME_SAMPLE", per_frame_sample},
    {"SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT", per_frame_depth_not_current},
    {"SUNSHINE_UI_PER_FRAME_SCENE_HOLD_HUDLESS", per_frame_scene_hold_hudless},
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

  // A layer without alpha is no layer (docs/reshade-sbs.md, UI detection):
  // premultiplied UI over transparent black cannot have color without alpha,
  // so an offscreen UI layer copy (stored_late_layer, never a tagged UI color)
  // with no alpha anywhere but color beyond its alpha on more than 1% of
  // pixels, the trusted-channel tolerance, is not this frame's UI layer.
  // Stellar Blade draws its scene into that cleared target in SDR. Detection
  // then proceeds as if no layer were offered: admitted_candidates clears the
  // UI color bit. game3d_native.hlsl (SunshineUIDetectionReduceCS) applies the
  // rule; the renderer's route, refutation and hold bookkeeping and the
  // overlay warning call this one definition.
  constexpr bool layer_without_alpha(std::uint32_t stored_flags, std::uint32_t covered, std::uint32_t invalid,
      std::uint64_t pixels) {
    return (stored_flags & stored_late_layer) && !covered && std::uint64_t(invalid) * 100u > pixels;
  }
  // The candidate bits detection acts on: the offered bits, without a layer
  // that has no alpha (slot 1's covered and invalid pixels).
  constexpr std::uint32_t admitted_candidates(std::uint32_t candidates, std::uint32_t stored_flags, std::uint32_t covered1,
      std::uint32_t invalid1, std::uint64_t pixels) {
    return layer_without_alpha(stored_flags, covered1, invalid1, pixels) ? candidates & ~2u : candidates;
  }
  static_assert(admitted_candidates(6u, layer_detection_flags(false), 0, 2, 100) == 4u &&
    admitted_candidates(6u, layer_detection_flags(false), 0, 1, 100) == 6u &&
    admitted_candidates(6u, layer_detection_flags(true), 1, 2, 100) == 6u &&
    admitted_candidates(6u, 0u, 0, 2, 100) == 6u && admitted_candidates(6u, layer_detection_flags(false), 0, 0, 0) == 6u);

  // Shader markers sizing the detection resources, and their values for
  // shaders without them.
  inline constexpr std::string_view decision_texels_marker = "SUNSHINE_UI_DECISION_TEXELS";
  inline constexpr std::string_view scene_evidence_images_marker = "SUNSHINE_UI_SCENE_EVIDENCE_IMAGES";
  inline constexpr std::uint32_t default_decision_texels = 5, default_scene_evidence_images = 0;
  // The CPU parses decision texels 0-4, and 5-6 when the shader writes them.
  // A statistics cell holds the luma of two images (presented, HUD-less).
  inline constexpr std::uint32_t min_decision_texels = 5, max_decision_texels = 16, max_scene_evidence_images = 2;
  inline constexpr std::uint32_t scene_decision_texels = 7;

  // Hidden-scene evidence D (docs/reshade-sbs.md, hidden-scene evidence),
  // mirrored by game3d_native.hlsl and inspect_game3d_dump.py: a grid of
  // cells over the frame; a depth edge is a cell-mean parallax step of at
  // least edge_px pixels per 2160 output rows; evidence needs min_edges edge
  // cells; D below hidden_percent / 100 is hidden, from visible_percent / 100
  // visible. Cell sums are in fixed point: luma_scale per unit perceptual
  // luma, parallax_scale per output pixel.
  namespace scene {
    inline constexpr std::uint32_t cells_x = 256, cells_y = 144, edge_px = 4, min_edges = 128;
    inline constexpr std::uint32_t hidden_percent = 15, visible_percent = 25;
    inline constexpr std::uint32_t luma_scale = 1u << 20, parallax_scale = 1u << 12;
    // The hidden-scene gates' opacity and difference shares, in percent, and
    // how long one valid hidden verdict holds its route.
    inline constexpr std::uint32_t opaque_percent = 99, hudless_changed_percent = 90;
    inline constexpr std::uint64_t hold_ms = 500;
  }
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 10> hlsl_scene_defines{{
    {"SUNSHINE_UI_SCENE_CELLS_X", scene::cells_x},
    {"SUNSHINE_UI_SCENE_CELLS_Y", scene::cells_y},
    {"SUNSHINE_UI_SCENE_EDGE_PX", scene::edge_px},
    {"SUNSHINE_UI_SCENE_MIN_EDGES", scene::min_edges},
    {"SUNSHINE_UI_SCENE_HIDDEN_PERCENT", scene::hidden_percent},
    {"SUNSHINE_UI_SCENE_VISIBLE_PERCENT", scene::visible_percent},
    {"SUNSHINE_UI_SCENE_LUMA_SCALE", scene::luma_scale},
    {"SUNSHINE_UI_SCENE_PARALLAX_SCALE", scene::parallax_scale},
    {"SUNSHINE_UI_SCENE_OPAQUE_PERCENT", scene::opaque_percent},
    {"SUNSHINE_UI_SCENE_HUDLESS_CHANGED_PERCENT", scene::hudless_changed_percent},
  }};

  // Statistics texture, 16 columns: 64 rows of per-tile counts, then, with
  // scene evidence, the sums of each 16x16-cell compare group, one row per 16
  // cell rows. The cells have a texture of their own, cells_x by cells_y.
  constexpr std::uint32_t statistics_rows(std::uint32_t images) { return 64u + (images ? scene::cells_y / 16u : 0u); }
  static_assert(scene::cells_x == 16u * 16u && scene::cells_y % 16u == 0u, "Compare groups must fill the 16 statistics columns");
  static_assert(statistics_rows(0) == 64u && statistics_rows(2) == 73u);

  // Words of the decision readback (the decision texel table): texel t,
  // component c is word 4 * t + c.
  namespace decision_word {
    inline constexpr std::size_t source = 0, covered = 1, pixels = 2, matching_tiles = 3;
    inline constexpr std::size_t candidates = 4, hudless_changed = 5, hudless_unchanged = 6, hudless_invalid = 7;
    inline constexpr std::size_t alpha_covered = 8, alpha_invalid = 12; // Four words each, one per alpha candidate.
    inline constexpr std::size_t hudless_lit = 16, trusted = 17;
    // Pixels with alpha of at least 254/255 in alpha candidates 0 and 1.
    inline constexpr std::size_t alpha_opaque = 18;
    // Texel 5, the presented image's hidden-scene evidence {n, asuint(D),
    // valid | ran << 1 | verdict << 2, decided comparisons (wins + losses)},
    // and texel 6, the HUD-less image's {n, asuint(D), valid | ran << 1, 0}.
    inline constexpr std::size_t scene_n = 20, scene_d = 21, scene_state = 22, scene_decided = 23;
    inline constexpr std::size_t hudless_scene_n = 24, hudless_scene_d = 25, hudless_scene_state = 26;
  }
  static_assert(decision_word::alpha_opaque + 1 < 4 * min_decision_texels &&
    decision_word::hudless_scene_state < 4 * scene_decision_texels);
  enum class scene_verdict : std::uint32_t { none = 0, hidden = 1, ambiguous = 2, visible = 3 };
  inline const char *name(scene_verdict value) {
    switch (value) {
      case scene_verdict::hidden: return "hidden";
      case scene_verdict::ambiguous: return "ambiguous";
      case scene_verdict::visible: return "visible";
      default: return "none";
    }
  }
  constexpr bool scene_state_valid(std::uint32_t state) { return (state & 1u) != 0; }
  constexpr bool scene_state_ran(std::uint32_t state) { return (state & 2u) != 0; }
  constexpr scene_verdict scene_state_verdict(std::uint32_t state) { return scene_verdict((state >> 2) & 3u); }
  // Whether a D value, valid evidence, reads visible: the HUD-less route's
  // test, which its texel carries no verdict for.
  constexpr bool scene_visible(float d) { return d * 100.f >= float(scene::visible_percent); }

  // The hidden-scene gates of one detection, from its counts, as
  // SunshineUIDetectionReduceCS opens them: no source 1-6 decided (sources 8
  // and 9 are these gates acting), and an untrusted UIAlpha or offscreen UI
  // layer (stored_late_layer, never a tagged UI color) whose alpha is at
  // least 254/255 on opaque_percent of pixels without an invalid pixel (layer
  // route), or a HUD-less image differing from the frame on
  // hudless_changed_percent of pixels (HUD-less route). Per slot (bit 1
  // UIAlpha, bit 2 UI layer), layer_slots are the slots that opened the layer
  // route, and overlay_slots those offered as its input below that opacity:
  // transparent somewhere, so an overlay rather than a scene buffer. Callers
  // pass admitted_candidates: a layer without alpha is no overlay.
  struct scene_gates {
    bool layer{}, hudless{};
    std::uint32_t layer_slots{}, overlay_slots{};
  };
  constexpr scene_gates scene_gates_of(std::uint32_t source, std::uint32_t candidates, std::uint32_t trusted,
      std::uint32_t stored_flags, std::uint64_t pixels, const std::array<std::uint32_t, 4> &invalid,
      const std::array<std::uint32_t, 2> &opaque, std::uint64_t hudless_changed) {
    if (!pixels) return {};
    const std::uint32_t inputs = (candidates & 1u) | ((stored_flags & stored_late_layer) ? candidates & 2u : 0u);
    std::uint32_t opaque_slots = 0;
    for (std::uint32_t slot = 0; slot != 2; ++slot)
      if (std::uint64_t(opaque[slot]) * 100u >= pixels * scene::opaque_percent) opaque_slots |= 1u << slot;
    scene_gates gates;
    gates.overlay_slots = inputs & ~opaque_slots;
    if (source && source != 8u && source != 9u) return gates;
    gates.layer_slots = inputs & ~trusted & opaque_slots & (invalid[0] ? ~1u : ~0u) & (invalid[1] ? ~2u : ~0u);
    gates.layer = gates.layer_slots != 0;
    gates.hudless = (candidates & 16u) && hudless_changed * 100u >= pixels * scene::hudless_changed_percent;
    return gates;
  }
  static_assert(scene_gates_of(0, 2, 0, stored_late_layer, 100, {}, {0, 99}, 0).layer &&
    !scene_gates_of(0, 2, 0, 0, 100, {}, {0, 100}, 0).layer && !scene_gates_of(0, 2, 2, stored_late_layer, 100, {}, {0, 100}, 0).layer &&
    !scene_gates_of(2, 2, 0, stored_late_layer, 100, {}, {0, 100}, 0).layer &&
    scene_gates_of(9, 16, 0, 0, 100, {}, {}, 90).hudless && !scene_gates_of(0, 16, 0, 0, 100, {}, {}, 89).hudless &&
    scene_gates_of(0, 3, 0, stored_late_layer, 100, {}, {100, 98}, 0).layer_slots == 1u &&
    scene_gates_of(2, 3, 3, stored_late_layer, 100, {}, {100, 98}, 0).overlay_slots == 2u &&
    scene_gates_of(0, 2, 0, 0, 100, {}, {0, 0}, 0).overlay_slots == 0u);

  // What identifies the hidden-scene routes' inputs: the layer route's alpha
  // slots 0 and 1, their trust and the UI color slot's stored flags. A held
  // route clears when they change. A HUD-less image that frame generation
  // pairs on some Presents only changes nothing here; each frame's own gate
  // still decides whether a held route acts on it.
  constexpr std::uint64_t scene_route_key(std::uint32_t candidates, std::uint32_t stored_flags, std::uint32_t trusted) {
    return std::uint64_t(stored_flags & stored_mask) << 32 | (trusted & 3u) << 2 | (candidates & 3u);
  }
  static_assert(scene_route_key(2u | 16u, 5u, 0u) == scene_route_key(2u | 48u, 5u, 0u) &&
    scene_route_key(2u, 5u, 0u) != scene_route_key(2u, 5u, 2u) && scene_route_key(2u, 5u, 0u) != scene_route_key(2u, 0u, 0u));
}
