// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Automatic UI detection contract shared by the renderer, Dump 3D and the
// offline replay tools. docs/reshade-sbs.md (UI detection flags and decision
// texels) owns the layout; game3d_native.hlsl mirrors the flag values as
// SUNSHINE_UI_STORED_* and SUNSHINE_UI_PER_FRAME_* defines, the T1 hold store
// as SUNSHINE_UI_HOLD_* and SUNSHINE_UI_FRAME_REASON_* defines and the
// candidate layout as SUNSHINE_UI_CANDIDATE_* and SUNSHINE_UI_SOURCE_* defines, and
// test_game3d_ui_layer fails when the two disagree. Retired flag values stay
// reserved so that dumps and replay cases keep their meaning.
// game3d_ui_selection.h owns the selection predicate over these counts. No
// ReShade dependency.
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
  // Candidate layout 2 (UI framework S1, E1): every candidate has its own
  // slot and bit in Sunshine_UICandidates (b2 word 0) and in the accepted mask
  // Sunshine_UIAcceptedCandidates (b2 word 2), so the tagged UIColorAndAlpha
  // and the offscreen UI layer never share a slot. Shaders without the layout
  // marker use layout 1: the layer in the UI color slot (t12, bit 0x2) with
  // stored_late_layer, and b2 word 2 the trusted slot indices.
  namespace candidate {
    inline constexpr std::uint32_t ui_alpha = 0x1u;   // t11 .r: a UIAlpha tag.
    inline constexpr std::uint32_t ui_color = 0x2u;   // t12 .a: a UIColorAndAlpha tag, never the layer.
    inline constexpr std::uint32_t backbuffer = 0x4u; // t13 .a: the Backbuffer tag.
    inline constexpr std::uint32_t current = 0x8u;    // t0 .a: the current (or paired) color.
    inline constexpr std::uint32_t hudless = 0x10u;   // t14: a HUD-less image paired with t0.
    inline constexpr std::uint32_t exact = 0x20u;     // The HUD-less pair is exact; not a candidate.
    inline constexpr std::uint32_t layer = 0x40u;     // t7 .a: the offscreen UI layer copy.
  }
  // Decision sources: 0 no mask, 1 UIAlpha, 2 UI color tag, 3 Backbuffer, 4
  // current, 5 HUD-less change set, 6 full change set, 8 layer route, 9
  // HUD-less route, 10 offscreen UI layer; 7 is retired and never reused.
  inline constexpr std::uint32_t source_layer = 10u, source_count = 11u;
  inline constexpr std::string_view candidate_layout_marker = "SUNSHINE_UI_CANDIDATE_LAYOUT";
  inline constexpr std::uint32_t candidate_layout = 2u, legacy_candidate_layout = 1u;
  // The game3d_native.hlsl define mirroring each candidate bit, the layer's
  // source id and the layout.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 9> hlsl_candidate_defines{{
    {"SUNSHINE_UI_CANDIDATE_LAYOUT", candidate_layout},
    {"SUNSHINE_UI_CANDIDATE_UI_ALPHA", candidate::ui_alpha},
    {"SUNSHINE_UI_CANDIDATE_UI_COLOR", candidate::ui_color},
    {"SUNSHINE_UI_CANDIDATE_BACKBUFFER", candidate::backbuffer},
    {"SUNSHINE_UI_CANDIDATE_CURRENT", candidate::current},
    {"SUNSHINE_UI_CANDIDATE_HUDLESS", candidate::hudless},
    {"SUNSHINE_UI_CANDIDATE_EXACT", candidate::exact},
    {"SUNSHINE_UI_CANDIDATE_LAYER", candidate::layer},
    {"SUNSHINE_UI_SOURCE_LAYER", source_layer},
  }};

  // Sunshine_UIDetectionFlags, b2 word 3. Stored bits describe the offscreen
  // UI layer slot (t7) and the renderer keeps them between frames. They key
  // no decision and no status (F1); only the hidden-scene route key reads
  // them (scene_route_key, until H1 lands in S2b).
  inline constexpr std::uint32_t stored_premultiplied = 0x1u; // The layer must pass the premultiplied bound (V1).
  inline constexpr std::uint32_t stored_hdr_headroom = 0x2u;  // With a float layer's HDR headroom.
  // The layer is the one-frame-late copy: inexact evidence (E2) that may
  // decide but never meets the one-way test (A2).
  inline constexpr std::uint32_t stored_late_layer = 0x4u;
  // 0x8u is reserved (a retired stage-2 bit) and never reused.
  inline constexpr std::uint32_t stored_mask = 0xffffu;
  // Per-frame bits ride in the pushed flags word of one render only. They are
  // never stored in the renderer's detection flags and never key a decision.
  inline constexpr std::uint32_t per_frame_scene_hold = 0x10000u;         // The CPU holds the layer route's hidden-scene verdict.
  // 0x20000u is reserved (a retired sample-frame bit the shader never read) and never reused.
  inline constexpr std::uint32_t per_frame_depth_not_current = 0x40000u;  // The consumed depth is reused or generated.
  inline constexpr std::uint32_t per_frame_scene_hold_hudless = 0x80000u; // The CPU holds the HUD-less route's verdict.
  // T1: an accepted candidate that the previously adopted real frame offered
  // is missing from this real frame, so this frame has no decision of its own.
  inline constexpr std::uint32_t per_frame_accepted_missing = 0x100000u;
  // T1: there is no previous real decision in this chain (first detection, a
  // scope change, an inactive frame or a Present without a mask since), so
  // the GPU hold store reads as hold::none.
  inline constexpr std::uint32_t per_frame_hold_reset = 0x200000u;
  inline constexpr std::uint32_t per_frame_mask = 0xffff0000u;
  static_assert((stored_mask & per_frame_mask) == 0 && (stored_mask | per_frame_mask) == 0xffffffffu);
  // The game3d_native.hlsl define mirroring each flag.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 8> hlsl_flag_defines{{
    {"SUNSHINE_UI_STORED_PREMULTIPLIED", stored_premultiplied},
    {"SUNSHINE_UI_STORED_HDR_HEADROOM", stored_hdr_headroom},
    {"SUNSHINE_UI_STORED_LATE_LAYER", stored_late_layer},
    {"SUNSHINE_UI_PER_FRAME_SCENE_HOLD", per_frame_scene_hold},
    {"SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT", per_frame_depth_not_current},
    {"SUNSHINE_UI_PER_FRAME_SCENE_HOLD_HUDLESS", per_frame_scene_hold_hudless},
    {"SUNSHINE_UI_PER_FRAME_ACCEPTED_MISSING", per_frame_accepted_missing},
    {"SUNSHINE_UI_PER_FRAME_HOLD_RESET", per_frame_hold_reset},
  }};

  // The T1 grace's GPU state (SunshineUIHoldStore, u5 of the detection reduce
  // only): store_texels R32_UINT texels holding {state, applied source,
  // applied covered} of the last real detection. own: that frame had a
  // decision of its own, which the next real frame without one reuses once;
  // spent: it had none (or reused one), so the next has no mask. An unbound
  // store, as in offline replay, reads none.
  namespace hold {
    inline constexpr std::uint32_t none = 0u, own = 1u, spent = 2u, store_texels = 3u;
    namespace word {
      inline constexpr std::uint32_t state = 0u, source = 1u, covered = 2u;
    }
  }
  // Decision word frame_reason: bits 0-7 the ui_no_mask index of the frame's
  // own decision, or frame_reason_decided when it decided a source; bit 16
  // set when the T1 grace reused the previous real frame's decision.
  inline constexpr std::uint32_t frame_reason_decided = 0xffu, frame_reason_reason_mask = 0xffu,
    frame_reason_reused = 0x10000u;
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 6> hlsl_hold_defines{{
    {"SUNSHINE_UI_HOLD_NONE", hold::none},
    {"SUNSHINE_UI_HOLD_OWN", hold::own},
    {"SUNSHINE_UI_HOLD_SPENT", hold::spent},
    {"SUNSHINE_UI_HOLD_STORE_TEXELS", hold::store_texels},
    {"SUNSHINE_UI_FRAME_REASON_DECIDED", frame_reason_decided},
    {"SUNSHINE_UI_FRAME_REASON_REUSED", frame_reason_reused},
  }};

  // Flags for an offscreen UI layer copy in the layer slot; the tags have none.
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
  // The CPU parses decision texels 0-4, 5-6 when the shader writes them, the
  // layer's texel 7 from candidate layout 2, and the one-way judgment and
  // frame reason texels 8-9 from selection revision 2. A statistics cell
  // holds the luma of two images (presented, HUD-less).
  inline constexpr std::uint32_t min_decision_texels = 5, max_decision_texels = 16, max_scene_evidence_images = 2;
  inline constexpr std::uint32_t scene_decision_texels = 7, layer_decision_texels = 8, judgment_decision_texels = 10;

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

  // Statistics texture, 16 columns: 112 rows of per-tile counts (rows 0-63
  // the alpha coverage, alpha invalid, HUD-less difference and lit/opaque
  // groups; rows 64-79 the offscreen UI layer's covered, invalid and opaque
  // pixels; from judgment_statistics_row the one-way judgment (A2) of the
  // layer, Backbuffer and current alpha: rows 80-95 their strong pixels,
  // alpha of at least 1/2, and rows 96-111 the strong pixels where an exact
  // pair's HUD-less image is lit and unchanged), then, with scene evidence,
  // from scene_partial_row the sums of each 16x16-cell compare group, one row
  // per 16 cell rows. The cells have a texture of their own, cells_x by
  // cells_y.
  inline constexpr std::uint32_t layer_statistics_row = 64u, judgment_statistics_row = 80u, scene_partial_row = 112u;
  constexpr std::uint32_t statistics_rows(std::uint32_t images) {
    return scene_partial_row + (images ? scene::cells_y / 16u : 0u);
  }
  static_assert(scene::cells_x == 16u * 16u && scene::cells_y % 16u == 0u, "Compare groups must fill the 16 statistics columns");
  static_assert(judgment_statistics_row + 32u == scene_partial_row && statistics_rows(0) == 112u && statistics_rows(2) == 121u);
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 12> hlsl_scene_defines{{
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
    {"SUNSHINE_UI_SCENE_PARTIAL_ROW", scene_partial_row},
    {"SUNSHINE_UI_JUDGMENT_ROW", judgment_statistics_row},
  }};

  // Words of the decision readback (the decision texel table): texel t,
  // component c is word 4 * t + c.
  namespace decision_word {
    inline constexpr std::size_t source = 0, covered = 1, pixels = 2, matching_tiles = 3;
    inline constexpr std::size_t candidates = 4, hudless_changed = 5, hudless_unchanged = 6, hudless_invalid = 7;
    // Four words each: UIAlpha, UI color tag, Backbuffer, current.
    inline constexpr std::size_t alpha_covered = 8, alpha_invalid = 12;
    // accepted: the pushed accepted mask in candidate-bit positions (layout 1:
    // the trusted slot indices).
    inline constexpr std::size_t hudless_lit = 16, accepted = 17;
    // Pixels with alpha of at least 254/255 in UIAlpha and the UI color tag.
    inline constexpr std::size_t alpha_opaque = 18;
    // Texel 5, the presented image's hidden-scene evidence {n, asuint(D),
    // valid | ran << 1 | verdict << 2, decided comparisons (wins + losses)},
    // and texel 6, the HUD-less image's {n, asuint(D), valid | ran << 1, 0}.
    inline constexpr std::size_t scene_n = 20, scene_d = 21, scene_state = 22, scene_decided = 23;
    inline constexpr std::size_t hudless_scene_n = 24, hudless_scene_d = 25, hudless_scene_state = 26;
    // Texel 7 (layout 2): the offscreen UI layer's covered, invalid (out of
    // range or beyond the premultiplied bound) and opaque pixels, and the
    // offered candidates that passed V1/V2, in candidate-bit positions.
    inline constexpr std::size_t layer_covered = 28, layer_invalid = 29, layer_opaque = 30, valid_bits = 31;
    // Texels 8 and 9 (selection revision 2): the one-way judgment counts (A2)
    // of the layer, Backbuffer and current alpha (strong: alpha of at least
    // 1/2; contradicted: strong where an offered exact pair's HUD-less image
    // is lit and unchanged; the late layer counts none), the refused
    // candidate bit of the own decision (F1, zero when it decided), and the
    // frame reason word (frame_reason_decided, frame_reason_reused).
    inline constexpr std::size_t strong = 32, refused = 35, contradicted = 36, frame_reason = 39;
  }
  static_assert(decision_word::alpha_opaque + 1 < 4 * min_decision_texels &&
    decision_word::hudless_scene_state < 4 * scene_decision_texels && decision_word::valid_bits < 4 * layer_decision_texels &&
    decision_word::frame_reason == 4 * judgment_decision_texels - 1);
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
  // SunshineUIDetectionReduceCS opens them: no source 1-6 or 10 decided
  // (sources 8 and 9 are these gates acting), and an unaccepted UIAlpha or
  // offscreen UI layer whose alpha is at least 254/255 on opaque_percent of
  // pixels without an invalid pixel (layer route), or a HUD-less image
  // differing from the frame on hudless_changed_percent of pixels (HUD-less
  // route). The inputs are the offered UIAlpha (candidate::ui_alpha) and layer
  // (candidate::layer); invalid and opaque hold their counts in that order.
  // Per slot (bit 1 UIAlpha, bit 2 UI layer), layer_slots are the slots that
  // opened the layer route, and overlay_slots the V1-valid inputs below that
  // opacity: transparent somewhere, so an overlay rather than a scene buffer.
  // An invalid layer, such as the SDR scene image Stellar Blade draws into its
  // cleared UI target, is neither.
  struct scene_gates {
    bool layer{}, hudless{};
    std::uint32_t layer_slots{}, overlay_slots{};
  };
  constexpr scene_gates scene_gates_of(std::uint32_t source, std::uint32_t offered, std::uint32_t accepted,
      std::uint64_t pixels, const std::array<std::uint32_t, 2> &invalid, const std::array<std::uint32_t, 2> &opaque,
      std::uint64_t hudless_changed) {
    if (!pixels) return {};
    const std::uint32_t inputs = ((offered & candidate::ui_alpha) ? 1u : 0u) | ((offered & candidate::layer) ? 2u : 0u);
    const std::uint32_t unaccepted = ((accepted & candidate::ui_alpha) ? 0u : 1u) | ((accepted & candidate::layer) ? 0u : 2u);
    std::uint32_t opaque_slots = 0, valid_slots = 0, clean_slots = 0;
    for (std::uint32_t slot = 0; slot != 2; ++slot) {
      if (std::uint64_t(opaque[slot]) * 100u >= pixels * scene::opaque_percent) opaque_slots |= 1u << slot;
      if (std::uint64_t(invalid[slot]) * 100u <= pixels) valid_slots |= 1u << slot;
      if (!invalid[slot]) clean_slots |= 1u << slot;
    }
    scene_gates gates;
    gates.overlay_slots = inputs & valid_slots & ~opaque_slots;
    if (source && source != 8u && source != 9u) return gates;
    gates.layer_slots = inputs & unaccepted & opaque_slots & clean_slots;
    gates.layer = gates.layer_slots != 0;
    gates.hudless = (offered & candidate::hudless) && hudless_changed * 100u >= pixels * scene::hudless_changed_percent;
    return gates;
  }
  static_assert(scene_gates_of(0, 0x40, 0, 100, {}, {0, 99}, 0).layer && !scene_gates_of(0, 2, 0, 100, {}, {0, 100}, 0).layer &&
    !scene_gates_of(0, 0x40, 0x40, 100, {}, {0, 100}, 0).layer && !scene_gates_of(10, 0x40, 0, 100, {}, {0, 100}, 0).layer &&
    !scene_gates_of(0, 0x40, 0, 100, {0, 1}, {0, 100}, 0).layer && scene_gates_of(0, 1, 0, 100, {}, {99, 0}, 0).layer &&
    scene_gates_of(9, 16, 0, 100, {}, {}, 90).hudless && !scene_gates_of(0, 16, 0, 100, {}, {}, 89).hudless &&
    scene_gates_of(0, 0x41, 0, 100, {}, {100, 98}, 0).layer_slots == 1u &&
    scene_gates_of(2, 0x41, 0x41, 100, {}, {100, 98}, 0).overlay_slots == 2u &&
    scene_gates_of(0, 0x40, 0, 100, {0, 2}, {0, 0}, 0).overlay_slots == 0u &&
    scene_gates_of(0, 0x40, 0, 100, {0, 1}, {0, 0}, 0).overlay_slots == 2u);

  // What identifies the hidden-scene routes' inputs: the offered UIAlpha and
  // layer candidates, their acceptance and the layer slot's stored flags. A
  // held route clears when they change. A UI color tag, or a HUD-less image
  // that frame generation pairs on some Presents only, changes nothing here;
  // each frame's own gate still decides whether a held route acts on it.
  constexpr std::uint64_t scene_route_key(std::uint32_t offered, std::uint32_t stored_flags, std::uint32_t accepted) {
    constexpr std::uint32_t inputs = candidate::ui_alpha | candidate::layer;
    return std::uint64_t(stored_flags & stored_mask) << 32 | std::uint64_t(accepted & inputs) << 8 | (offered & inputs);
  }
  static_assert(scene_route_key(0x40u | 16u, 5u, 0u) == scene_route_key(0x40u | 48u, 5u, 0u) &&
    scene_route_key(0x40u | 2u, 5u, 0u) == scene_route_key(0x40u, 5u, 0u) &&
    scene_route_key(0x40u, 5u, 0u) != scene_route_key(0x40u, 5u, 0x40u) &&
    scene_route_key(0x40u, 5u, 0u) != scene_route_key(0x40u, 0u, 0u) &&
    scene_route_key(1u, 0u, 0u) != scene_route_key(1u, 0u, 1u) && scene_route_key(1u, 0u, 2u) == scene_route_key(1u, 0u, 0u));
}
