// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Automatic UI detection contract shared by the renderer, Dump 3D and the
// offline replay tools. docs/reshade-sbs.md (UI detection flags and decision
// texels) owns the layout; game3d_native.hlsl mirrors the flag values as
// SUNSHINE_UI_STORED_* and SUNSHINE_UI_PER_FRAME_* defines, the T1 hold store
// as SUNSHINE_UI_HOLD_* and SUNSHINE_UI_FRAME_REASON_* defines, the H1 claim
// words as SUNSHINE_UI_CLAIM_PRE_UI, SUNSHINE_UI_H1_APPLIED and
// SUNSHINE_UI_PRE_UI_IMAGE_* defines and the candidate layout as
// SUNSHINE_UI_CANDIDATE_* and SUNSHINE_UI_SOURCE_* defines, and
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
  // current, 5 HUD-less change set, 6 full change set, 8 full-frame UI over a
  // hidden scene (H1), 10 offscreen UI layer; 7 and 9 (the HUD-less route
  // before S2b), 11 (fix 2's still screen without a UI source, H2) and 12
  // (fix 3's pre-UI change set), all removed, are retired and never reused.
  // Candidate bit 0x100 (fix 3's pre-UI change set) is reserved likewise.
  // source_count numbers the decided counter words, the retired ones
  // included.
  inline constexpr std::uint32_t source_layer = 10u, source_count = 12u;
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
  // UI layer slot (t7) and the renderer keeps them between frames. Only the
  // tiles pass reads them; they key no decision, no status (F1) and no
  // hidden-scene state, and nothing on the CPU reads them back.
  inline constexpr std::uint32_t stored_premultiplied = 0x1u; // The layer must pass the premultiplied bound (V1).
  inline constexpr std::uint32_t stored_hdr_headroom = 0x2u;  // With a float layer's HDR headroom.
  // The layer is the one-frame-late copy: inexact evidence (E2) that may
  // decide but never meets the one-way test (A2).
  inline constexpr std::uint32_t stored_late_layer = 0x4u;
  // 0x8u is reserved (a retired stage-2 bit) and never reused.
  inline constexpr std::uint32_t stored_mask = 0xffffu;
  // Per-frame bits ride in the pushed flags word of one render only. They are
  // never stored in the renderer's detection flags.
  // 0x10000u is reserved (the layer route's hold before S2b) and never reused.
  // 0x20000u is reserved (a retired sample-frame bit the shader never read) and never reused.
  inline constexpr std::uint32_t per_frame_depth_not_current = 0x40000u; // The consumed depth is reused or generated.
  // 0x80000u is reserved (the HUD-less route's hold before S2b) and never reused.
  // T1: an accepted candidate that the previously adopted real frame offered
  // is missing from this real frame, so this frame has no decision of its own.
  inline constexpr std::uint32_t per_frame_accepted_missing = 0x100000u;
  // T1: there is no previous real decision in this chain (first detection, a
  // scope change, an inactive frame or a Present without a mask since), so
  // the GPU hold store reads as hold::none.
  inline constexpr std::uint32_t per_frame_hold_reset = 0x200000u;
  // H1 (M5, game3d_scene_guard.h): the CPU holds a hidden verdict of D on
  // the presented frame, and its samples read the pre-UI scene image visible.
  inline constexpr std::uint32_t per_frame_scene_hidden = 0x400000u;
  inline constexpr std::uint32_t per_frame_pre_ui_visible = 0x800000u;
  // H1: the candidate bits whose source signature a visible verdict refuted,
  // shifted into bits 24-30; a refuted full claim is not acting.
  inline constexpr std::uint32_t per_frame_refuted_shift = 24u, per_frame_refuted_mask = 0x5f000000u;
  // 0x20000000u is reserved (bit 29, where the exact bit would shift to: the
  // evidence passes measured the layer beside a HUD-less image for the
  // layer's D proof before fix 1) and never reused.
  // H1 (d): the offered offscreen UI layer's signature is proven the pre-UI
  // scene image (the acceptance ledger's key pre_ui:<format>:<space>,
  // game3d_alpha_auto.h), so that its claim (d) may act.
  inline constexpr std::uint32_t per_frame_pre_ui_proven = 0x80000000u;
  inline constexpr std::uint32_t per_frame_mask = 0xffff0000u;
  static_assert((stored_mask & per_frame_mask) == 0 && (stored_mask | per_frame_mask) == 0xffffffffu);
  static_assert(((candidate::ui_alpha | candidate::ui_color | candidate::backbuffer | candidate::current | candidate::hudless |
    candidate::layer) << per_frame_refuted_shift) == per_frame_refuted_mask && (per_frame_refuted_mask & ~per_frame_mask) == 0 &&
    (per_frame_refuted_mask & (per_frame_scene_hidden | per_frame_pre_ui_visible | per_frame_hold_reset)) == 0 &&
    (per_frame_pre_ui_proven & (per_frame_refuted_mask | ~per_frame_mask | 0x20000000u)) == 0);
  // The game3d_native.hlsl define mirroring each flag.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 10> hlsl_flag_defines{{
    {"SUNSHINE_UI_STORED_PREMULTIPLIED", stored_premultiplied},
    {"SUNSHINE_UI_STORED_HDR_HEADROOM", stored_hdr_headroom},
    {"SUNSHINE_UI_STORED_LATE_LAYER", stored_late_layer},
    {"SUNSHINE_UI_PER_FRAME_DEPTH_NOT_CURRENT", per_frame_depth_not_current},
    {"SUNSHINE_UI_PER_FRAME_ACCEPTED_MISSING", per_frame_accepted_missing},
    {"SUNSHINE_UI_PER_FRAME_HOLD_RESET", per_frame_hold_reset},
    {"SUNSHINE_UI_PER_FRAME_SCENE_HIDDEN", per_frame_scene_hidden},
    {"SUNSHINE_UI_PER_FRAME_PRE_UI_VISIBLE", per_frame_pre_ui_visible},
    {"SUNSHINE_UI_PER_FRAME_REFUTED_SHIFT", per_frame_refuted_shift},
    {"SUNSHINE_UI_PER_FRAME_PRE_UI_PROVEN", per_frame_pre_ui_proven},
  }};

  // H1 (docs/reshade-sbs.md, hidden-scene evidence): the informative full
  // claims of one detection are candidate bits plus claim_pre_ui, a pre-UI
  // scene image (the HUD-less image when offered, else the offscreen UI
  // layer's colour) that the depth may describe instead of the presented
  // frame. Decision word h1: bits 0-7 the S1 winner's source, h1_applied
  // when H1 overrode it with source 8. Decision word pre_ui_scene_image: the
  // image texel 6 measured. Bit 0x200 of word h1 (fix 3's refined change
  // set, removed by user decision) is reserved and never reused.
  inline constexpr std::uint32_t claim_pre_ui = 0x80u, h1_applied = 0x100u, h1_winner_mask = 0xffu;
  static_assert((claim_pre_ui & (candidate::ui_alpha | candidate::ui_color | candidate::backbuffer | candidate::current |
    candidate::hudless | candidate::exact | candidate::layer)) == 0 && (h1_applied & h1_winner_mask) == 0);
  namespace pre_ui_image {
    inline constexpr std::uint32_t none = 0u, hudless = 1u, layer = 2u;
  }
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 4> hlsl_h1_defines{{
    {"SUNSHINE_UI_CLAIM_PRE_UI", claim_pre_ui},
    {"SUNSHINE_UI_H1_APPLIED", h1_applied},
    {"SUNSHINE_UI_PRE_UI_IMAGE_HUDLESS", pre_ui_image::hudless},
    {"SUNSHINE_UI_PRE_UI_IMAGE_LAYER", pre_ui_image::layer},
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
  // layer's texel 7 from candidate layout 2, the one-way judgment and frame
  // reason texels 8-9 from selection revision 2, and the H1 texel 10 from
  // selection revision 3, and the pre-UI pixel counts of texel 11 from
  // selection revision 4. A statistics cell holds the luma of two images
  // (presented, and the pre-UI scene image: HUD-less, else the UI layer).
  // Texel 12 (H2's stillness counts and S3's identity words in selection
  // revisions 5 to 8, both removed by revision 9) and texels 13-15 are
  // reserved; selection revision 9 writes 12 texels.
  inline constexpr std::uint32_t min_decision_texels = 5, max_decision_texels = 16, max_scene_evidence_images = 2;
  inline constexpr std::uint32_t scene_decision_texels = 7, layer_decision_texels = 8, judgment_decision_texels = 10,
    h1_decision_texels = 11, pre_ui_decision_texels = 12;

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
    // H1's opaque-full share (alpha of at least 254/255 on this percent of
    // pixels) and the HUD-less pre-UI claim's changed share, in percent, and
    // how long one hidden sample of a held verdict renews it (M5).
    inline constexpr std::uint32_t opaque_percent = 99, hudless_changed_percent = 90;
    inline constexpr std::uint64_t hold_ms = 500;
  }

  // b2 words the renderer pushes (SunshineUIDetectionConstants): 0 the
  // offered candidates, 1 the HUD-less pair's difference threshold, 2 the
  // accepted candidates, 3 the flags, 4 the layer's pre-UI threshold, and 5,
  // which is reserved and pushed as zero: H2's still-screen flag (fix 2,
  // 0x1) and fix 3 and fix 4's bits 0x2-0x200 used it, all removed. Words
  // 6-9 (S3's proposals and identity bits, removed with S3's frame identity)
  // are no longer pushed. Never reuse a reserved bit.
  inline constexpr std::uint32_t b2_words = 6u;
  inline constexpr std::uint32_t reserved_rule_bits = 0x3ffu;
  // Statistics texture, 16 columns: 112 rows of per-tile counts (rows 0-63
  // the alpha coverage, alpha invalid, HUD-less difference and lit/opaque
  // groups, row 48 .w the Backbuffer's opaque pixels; rows 64-79 the
  // offscreen UI layer's covered, invalid and opaque pixels and .w the
  // current alpha's opaque pixels; from judgment_statistics_row the one-way judgment (A2) of the
  // layer, Backbuffer and current alpha: rows 80-95 their strong pixels,
  // alpha of at least 1/2, and rows 96-111 the strong pixels where an exact
  // pair's HUD-less image is lit and unchanged), then, with scene evidence,
  // from scene_partial_row the sums of each 16x16-cell compare group, one row
  // per 16 cell rows (rows 112-120), and from selection revision 4 (decision
  // texels pre_ui_decision_texels) rows 128-143 from pre_ui_statistics_row:
  // per tile, the offscreen UI layer against the presented frame {matching
  // pixels, lit layer pixels, lit presented pixels, lit presented pixels that
  // differ} (decision texel 11), 144 rows in all. Rows 144-159 (fix 2's H2
  // stillness counts) and 160-207 (fix 3's change-set shadow and fix 4's
  // darkening), all removed, are reserved and never reused. The cells have a
  // texture of their own, cells_x by cells_y.
  inline constexpr std::uint32_t layer_statistics_row = 64u, judgment_statistics_row = 80u, scene_partial_row = 112u,
    pre_ui_statistics_row = 128u;
  // The per-tile rows (0-111 and the pre-UI rows 128-143) are counted in
  // tile_parts parts (SUNSHINE_UI_DETECTION_TILE_PARTS, the tiles pass's group
  // z): part k of tile (x, y) is at column x + 16k, and the reduce sums a
  // tile's parts (a 16-column texture reads zero beyond them). The scene
  // rows stay in the first 16 columns. A shader without the marker counts
  // each tile in one part.
  inline constexpr std::string_view tile_parts_marker = "SUNSHINE_UI_DETECTION_TILE_PARTS";
  inline constexpr std::uint32_t tile_parts = 4u, max_tile_parts = 16u;
  // The statistics texture's width for a shader's tile-parts marker (0: none).
  constexpr std::uint32_t statistics_columns(std::uint32_t parts) { return 16u * (parts ? parts : 1u); }
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 1> hlsl_tile_defines{{
    {"SUNSHINE_UI_DETECTION_TILE_PARTS", tile_parts},
  }};
  // The statistics rows of a shader with these markers (images: its
  // SUNSHINE_UI_SCENE_EVIDENCE_IMAGES, decision_texels: its
  // SUNSHINE_UI_DECISION_TEXELS).
  constexpr std::uint32_t statistics_rows(std::uint32_t images, std::uint32_t decision_texels = pre_ui_decision_texels) {
    return decision_texels >= pre_ui_decision_texels ? pre_ui_statistics_row + 16u :
      scene_partial_row + (images ? scene::cells_y / 16u : 0u);
  }
  static_assert(scene::cells_x == 16u * 16u && scene::cells_y % 16u == 0u, "Compare groups must fill the 16 statistics columns");
  static_assert(judgment_statistics_row + 32u == scene_partial_row && statistics_rows(0, h1_decision_texels) == 112u &&
    statistics_rows(2, h1_decision_texels) == 121u && scene_partial_row + scene::cells_y / 16u <= pre_ui_statistics_row &&
    statistics_rows(2, pre_ui_decision_texels) == 144u && statistics_rows(0, pre_ui_decision_texels) == 144u &&
    statistics_rows(2) == 144u && statistics_rows(0) == 144u);
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 13> hlsl_scene_defines{{
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
    {"SUNSHINE_UI_PRE_UI_ROW", pre_ui_statistics_row},
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
    // and texel 6, the pre-UI scene image's {n, asuint(D), valid | ran << 1,
    // image (pre_ui_image)}, all zero when no pre-UI image is offered.
    inline constexpr std::size_t scene_n = 20, scene_d = 21, scene_state = 22, scene_decided = 23;
    inline constexpr std::size_t pre_ui_scene_n = 24, pre_ui_scene_d = 25, pre_ui_scene_state = 26, pre_ui_scene_image = 27;
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
    // Texel 10 (selection revision 3, H1): pixels with alpha of at least
    // 254/255 in the Backbuffer and the current alpha, the raw informative
    // claims before refutation (candidate bits | claim_pre_ui), and the h1
    // word (the S1 winner's source | h1_applied).
    inline constexpr std::size_t opaque_backbuffer = 40, opaque_current = 41, claims = 42, h1 = 43;
    // Texel 11 (selection revision 4, fix 1): the offscreen UI layer against
    // the presented frame, whatever the offer's pre-UI image, at 8 times the
    // pair threshold (b2 word 4; all zero when that is zero): pixels whose
    // colours match, lit layer pixels, lit presented pixels, and lit
    // presented pixels that differ from the layer. The acceptance ledger
    // proves the layer the pre-UI scene image from the first two
    // (ui_selection::pre_ui_match); the last two are shadow statistics that
    // nothing acts on.
    inline constexpr std::size_t pre_ui_match = 44, pre_ui_image_lit = 45, presented_lit = 46, presented_lit_differs = 47;
    // Texel 12 (words 48-51) is reserved: from selection revision 5 H2's
    // still and compared cells (fix 2) and S3's identity verdicts and deltas
    // were written there, both removed. Texels 13-15 (words 52-63) are reserved
    // too: selection revision 6 (fix 3 and fix 4) used them for the removed
    // change-set shadow and darkening words. Never reuse them.
  }
  static_assert(decision_word::alpha_opaque + 1 < 4 * min_decision_texels &&
    decision_word::pre_ui_scene_image < 4 * scene_decision_texels && decision_word::valid_bits < 4 * layer_decision_texels &&
    decision_word::frame_reason == 4 * judgment_decision_texels - 1 && decision_word::h1 == 4 * h1_decision_texels - 1 &&
    decision_word::pre_ui_match == 4 * h1_decision_texels &&
    decision_word::presented_lit_differs == 4 * pre_ui_decision_texels - 1 && pre_ui_decision_texels <= max_decision_texels);
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
  // Whether a D value, valid evidence, reads visible: the test of the pre-UI
  // scene image, whose texel carries no verdict.
  constexpr bool scene_visible(float d) { return d * 100.f >= float(scene::visible_percent); }
  // The verdict of a D value of valid evidence, as SunshineSceneEvidenceCS
  // reads the presented image's.
  constexpr scene_verdict scene_verdict_of(float d) {
    return d * 100.f < float(scene::hidden_percent) ? scene_verdict::hidden :
      scene_visible(d) ? scene_verdict::visible : scene_verdict::ambiguous;
  }
  static_assert(scene_verdict_of(.149f) == scene_verdict::hidden && scene_verdict_of(.15f) == scene_verdict::ambiguous &&
    scene_verdict_of(.25f) == scene_verdict::visible && scene_verdict_of(-1.f) == scene_verdict::hidden);
}
