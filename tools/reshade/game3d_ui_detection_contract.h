// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Automatic UI detection contract shared by the renderer, Dump 3D and the
// offline replay tools. docs/reshade-sbs.md (UI detection flags and decision
// texels) owns the layout; game3d_native.hlsl mirrors the flag values as
// SUNSHINE_UI_STORED_* and SUNSHINE_UI_PER_FRAME_* defines, the T1 hold store
// as SUNSHINE_UI_HOLD_* and SUNSHINE_UI_FRAME_REASON_* defines, the H1 claim
// words as SUNSHINE_UI_CLAIM_PRE_UI, SUNSHINE_UI_H1_APPLIED, SUNSHINE_UI_H1_REFINED and
// SUNSHINE_UI_PRE_UI_IMAGE_* defines, the H2 still-screen words as
// SUNSHINE_UI_SOURCE_STILL and SUNSHINE_UI_STILL_* defines, the change-set
// rule bits and statistics row as SUNSHINE_UI_CHANGE_SET_* defines and the
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
    // Fix 3 (selection revision 6): the offscreen UI layer (t7) proven the
    // pre-UI scene image (the ledger's key pre_ui:<layer format>:<space>),
    // paired with the retained Present whose frame it holds (t2 or t3, b2
    // word 5's pair offset). An inferred change set, not an alpha: the same
    // difference math as the HUD-less pair (t14 against t0), at
    // change_set::inferred_scale times the pair threshold, partial only.
    inline constexpr std::uint32_t pre_ui = 0x100u;
  }
  // Decision sources: 0 no mask, 1 UIAlpha, 2 UI color tag, 3 Backbuffer, 4
  // current, 5 HUD-less change set, 6 full change set, 8 full-frame UI over a
  // hidden scene (H1), 10 offscreen UI layer, 11 a still screen without a UI
  // source (H2, fix 2: shown flat like 6 and 8), 12 the changed pixels of the
  // proven pre-UI layer against the retained Present it shows (fix 3, an
  // inferred partial change set); 7 and 9 (the HUD-less route before S2b) are
  // retired and never reused.
  inline constexpr std::uint32_t source_layer = 10u, source_still = 11u, source_pre_ui = 12u, source_count = 13u;
  inline constexpr std::string_view candidate_layout_marker = "SUNSHINE_UI_CANDIDATE_LAYOUT";
  inline constexpr std::uint32_t candidate_layout = 2u, legacy_candidate_layout = 1u;
  // The game3d_native.hlsl define mirroring each candidate bit, the layer's
  // and the pre-UI change set's source ids and the layout.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 11> hlsl_candidate_defines{{
    {"SUNSHINE_UI_CANDIDATE_LAYOUT", candidate_layout},
    {"SUNSHINE_UI_CANDIDATE_UI_ALPHA", candidate::ui_alpha},
    {"SUNSHINE_UI_CANDIDATE_UI_COLOR", candidate::ui_color},
    {"SUNSHINE_UI_CANDIDATE_BACKBUFFER", candidate::backbuffer},
    {"SUNSHINE_UI_CANDIDATE_CURRENT", candidate::current},
    {"SUNSHINE_UI_CANDIDATE_HUDLESS", candidate::hudless},
    {"SUNSHINE_UI_CANDIDATE_EXACT", candidate::exact},
    {"SUNSHINE_UI_CANDIDATE_LAYER", candidate::layer},
    {"SUNSHINE_UI_CANDIDATE_PRE_UI", candidate::pre_ui},
    {"SUNSHINE_UI_SOURCE_LAYER", source_layer},
    {"SUNSHINE_UI_SOURCE_PRE_UI", source_pre_ui},
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
  // shifted into bits 24-30; a refuted full claim is not acting. Only the
  // claimable candidates have a bit here: candidate::pre_ui never claims.
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
    (std::uint64_t(candidate::pre_ui) << per_frame_refuted_shift) > 0xffffffffu &&
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
  // when H1 overrode it with source 8, and (fix 3) h1_refined when a valid
  // exact selective change set replaced a shapeless whole-frame alpha winner
  // (the S1 winner's source is then the change set's, 5 or 12). Decision
  // word pre_ui_scene_image: the image texel 6 measured.
  inline constexpr std::uint32_t claim_pre_ui = 0x80u, h1_applied = 0x100u, h1_refined = 0x200u, h1_winner_mask = 0xffu;
  static_assert((claim_pre_ui & (candidate::ui_alpha | candidate::ui_color | candidate::backbuffer | candidate::current |
    candidate::hudless | candidate::exact | candidate::layer | candidate::pre_ui)) == 0 && (h1_applied & h1_winner_mask) == 0 &&
    (h1_refined & (h1_applied | h1_winner_mask)) == 0);
  namespace pre_ui_image {
    inline constexpr std::uint32_t none = 0u, hudless = 1u, layer = 2u;
  }
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 5> hlsl_h1_defines{{
    {"SUNSHINE_UI_CLAIM_PRE_UI", claim_pre_ui},
    {"SUNSHINE_UI_H1_APPLIED", h1_applied},
    {"SUNSHINE_UI_H1_REFINED", h1_refined},
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
  // From selection revision 5 (fix 2) texel 12 holds H2's stillness counts,
  // and from selection revision 6 (fix 3) texels 13-15 the change-set shadow
  // of the offscreen layer against its pair.
  inline constexpr std::uint32_t min_decision_texels = 5, max_decision_texels = 16, max_scene_evidence_images = 2;
  inline constexpr std::uint32_t scene_decision_texels = 7, layer_decision_texels = 8, judgment_decision_texels = 10,
    h1_decision_texels = 11, pre_ui_decision_texels = 12, still_decision_texels = 13, change_set_decision_texels = 16;

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

  // H2 (M5, fix 2; docs/reshade-sbs.md, still screens without a UI source):
  // in SDR Auto, a real frame without a UI source decision (applied source 0,
  // not reused by T1) is shown flat as source_still once consecutive samples
  // spanning run_ms read the presented frame's D at most d_percent / 100 and
  // keep at least still_percent of all the D grid's cells still: a cell's
  // presented-luma mean (scene::luma_scale fixed point) within tolerance,
  // 1/255 of code luma, of the previous measured sample's. The predicate,
  // the run and its release live on the CPU (game3d_still_screen.h); the GPU
  // counts still and compared cells (statistics rows from
  // still_statistics_row, decision texel 12) and applies the flag the CPU
  // pushes in b2 word 5 (Sunshine_UIRules): flatten. A run that has not
  // entered restarts when its next sample comes more than max_gap_ms after
  // its last, five of the renderer's 100 ms sample intervals: the 2 s rests
  // on samples about 100 ms apart (gameplay runs of at most 547 ms, real
  // screens of at least 4.2 s), so a stall must not bridge it.
  namespace still {
    inline constexpr std::uint32_t d_percent = 5, still_percent = 95;
    inline constexpr std::uint32_t tolerance = scene::luma_scale / 255u;
    inline constexpr std::uint64_t run_ms = 2000, max_gap_ms = 500;
    inline constexpr std::uint32_t flatten = 0x1u;
  }
  static_assert(still::tolerance == 4112u && still::d_percent < scene::hidden_percent && still::max_gap_ms < still::run_ms);

  // Change sets (fix 3; docs/reshade-sbs.md, UI decision framework): one
  // difference rule for every pre-UI image provider, the declared HUD-less
  // image (t14 against t0) and the inferred offscreen layer proven the
  // pre-UI scene image (candidate::pre_ui: t7 against the retained Present
  // whose frame it holds). A pixel is changed beyond k times the pair
  // threshold t (b2 word 1) and unchanged within k t / 2, where k is 1 for a
  // declared pair and inferred_scale for an inferred one; a partial change
  // set also needs its pre-UI image lit (beyond 8 t) on lit_percent of the
  // pixels, since a black image is no scene. The inferred change set's mask
  // keeps a changed pixel with at least min_neighbourhood changed pixels in
  // its 3x3 window, itself included (out-of-frame neighbours unchanged),
  // which removes specks and keeps one-pixel lines. b2 word 5
  // (Sunshine_UIRules) carries the rule bits beside H2's still::flatten:
  // refine (Auto with UIPinOnlyUI=1, rules::pin_only_ui) lets a valid exact selective
  // change set decide where S1's winner is an accepted alpha opaque on every
  // pixel; shadow that the offered layer's signature is proven the pre-UI
  // scene image, the only layer the change-set shadow measures; gap that the
  // previous real frame offered the pre-UI change set and this one does not
  // (T1's one-frame pairing gap, apart from the accepted-missing test); bits
  // 4-5 (pair_mask) the layer pair's Present offset (0 the current Present
  // t6, 1 the retained Present one back at t2, 2 two back at t3); retained_1
  // and retained_2 say that t2 and t3 are bound. The change-set shadow
  // (decision texels 13-15, statistics rows from change_set_statistics_row)
  // measures the layer against its pair at inferred_scale times b2 word 4 on
  // sample frames; judge names the declared alpha it is compared with.
  //
  // Fix 4 (rule P2, pin only UI; docs/reshade-sbs.md, UI decision framework)
  // shares the switch: rules::pin_only_ui is the refine bit, pushed in Auto
  // with UIPinOnlyUI=1 (fix 3's UIPinChangedPixels renamed), and
  // rules::darkening_measured says the darkening passes (namespace darkening)
  // run on this frame. The mask pass unpins pure darkening only with both.
  // rules::tag_linear says the UI color tag (t12) has a float format, whose
  // colour is linear (the offscreen layer's is stored_hdr_headroom, b2 word
  // 3), so that the opacity tolerance applies in linear light.
  namespace rules {
    inline constexpr std::uint32_t pin_only_ui = 0x2u, darkening_measured = 0x100u, tag_linear = 0x200u;
  }
  namespace change_set {
    inline constexpr std::uint32_t shadow = 0x4u, gap = 0x8u, pair_shift = 4u, pair_mask = 0x30u, retained_1 = 0x40u,
      retained_2 = 0x80u;
    // S3: pair offset 3 compares the layer with the Backbuffer tag's colour
    // (t13), the same-token final image of a token-space pair; the renderer
    // pushes it only while identity is authoritative.
    inline constexpr std::uint32_t pair_backbuffer = 3u;
    inline constexpr std::uint32_t inferred_scale = 8u, lit_percent = 1u, min_neighbourhood = 3u;
    namespace judge {
      inline constexpr std::uint32_t none = 0u, ui_alpha = 1u, ui_color = 2u;
    }
    // The layer pair's Present offset of a rules word.
    constexpr std::uint32_t pair_offset(std::uint32_t rules) { return (rules & pair_mask) >> pair_shift; }
    // The shadow's bit planes: a shader with planes_marker (planes) computes
    // the shadow in two passes of its own on shadow samples, the bits pass
    // (one bit per pixel and plane, 32 pixels per R32_UINT word, the planes
    // side by side in x: plane p of row y at texel (p * plane_words(width) +
    // word, y)) and the count pass (the statistics rows from
    // change_set_statistics_row), dispatched after the tiles pass exactly
    // when the shadow is on (shadow_dispatched). Fix 4 adds three planes the
    // darkening passes write after the reduce (darkening::dispatched): dark
    // (the decided source's darkening pixels), structure (those on a sharp
    // opacity step or touching raw colour UI) and tile (darkening::tile_rows
    // rows per row of 16x16 tiles, bit tx of word tx / 32: for a change set
    // row 3 ty a colour verdict in the tile's 3x3 ring, row 3 ty + 1 its own
    // darkening verdict, row 3 ty + 2 a darkening pixel in the tile, which
    // the region pass replaces by the tiles whose darkening region unpins;
    // for an opacity source bit 0 of row 8 gy, word gx, a colour pixel in the
    // bits pass's group (gx, gy), and bit 0 of row 1, word 0, a colour pixel
    // in the frame). A shader whose planes marker is planes has the
    // darkening passes.
    inline constexpr std::string_view planes_marker = "SUNSHINE_UI_CHANGE_SET_PLANES";
    inline constexpr std::uint32_t planes = 9u;
    namespace plane {
      inline constexpr std::uint32_t changed = 0u, unchanged = 1u, invalid = 2u, changed_1 = 3u, changed_2 = 4u, judge = 5u;
      inline constexpr std::uint32_t dark = 6u, structure = 7u, tile = 8u;
    }
    constexpr std::uint32_t plane_words(std::uint32_t width) { return (width + 31u) / 32u; }
    // The bits pass's groups (one word on each of 8 rows) and the count
    // pass's (one per 16x16 tile).
    inline constexpr std::uint32_t bits_group_rows = 8u;
    // Whether a detection run with these b2 words measures the shadow, the
    // shader's SunshineChangeSetShadow(): an offered layer, a pre-UI
    // threshold (a sample frame) and the shadow bit.
    constexpr bool shadow_dispatched(std::uint32_t candidates, float pre_ui_threshold, std::uint32_t rules) {
      return pre_ui_threshold > 0.f && (candidates & candidate::layer) != 0u && (rules & shadow) != 0u;
    }
  }
  static_assert(((rules::pin_only_ui | change_set::shadow | change_set::gap | change_set::pair_mask | change_set::retained_1 |
    change_set::retained_2) & still::flatten) == 0 && ((rules::pin_only_ui | change_set::shadow | change_set::gap) &
    (change_set::pair_mask | change_set::retained_1 | change_set::retained_2)) == 0 &&
    (rules::pin_only_ui & (change_set::shadow | change_set::gap)) == 0 && (change_set::shadow & change_set::gap) == 0 && (change_set::pair_mask >> change_set::pair_shift) == 3u && change_set::pair_offset(0x2fu) == 2u);
  static_assert(((rules::darkening_measured | rules::tag_linear) & (still::flatten | rules::pin_only_ui | change_set::shadow |
    change_set::gap | change_set::pair_mask | change_set::retained_1 | change_set::retained_2)) == 0 &&
    (rules::darkening_measured & rules::tag_linear) == 0);

  // S3 frame identity (docs/reshade-sbs.md, UI decision framework, "S3
  // snapshot ticket (shadow)"; game3d_ui_ticket.h owns the ticket and label
  // contract): the CPU proposes the label a paired snapshot must carry and
  // the GPU verifies it against the snapshot's stamp. The renderer's stamp
  // buffer (Buffer<uint4> SunshineUIStamps, t9 of the detection passes only,
  // one entry per ui_ticket::slot: x the C_P read, y the C_T read) holds the
  // stamp copied beside each candidate. b2 words 6-8 are the proposals, 0 not
  // proposed: the offscreen layer copy's present label (the label of the
  // retained Present it pairs with) and token label (the newest ready
  // Backbuffer tag's), and the HUD-less image's present label; b2 word 9
  // (Sunshine_UIIdentity) carries token_batch (the CPU proved the HUD-less
  // image and its Backbuffer reference one tag batch by token equality) and
  // the gates: gate_layer ANDs the layer pair's identity into the pre-UI
  // change set's V2 validity (source 12), gate_hudless the HUD-less pair's
  // into the HUD-less change set's (5 and 6). The renderer pushes a gate only
  // while ui_ticket::identity_authoritative (S3 enabled), so in the shadow
  // the verdict is measured and counted and decides nothing. A pair's
  // verdict: none (its candidate not offered), unproposed (no label
  // proposed), unstamped (the stamp read 0), exact or mismatch. The layer
  // pairs in present space when a present label is proposed, else in token
  // space; the HUD-less image in token space by token_batch, else in present
  // space against its stamp's C_P read + 1 (it is copied at its tag, the
  // frame after the Present it read). Decision texel 12 .z (word verdicts)
  // packs both verdicts, their label spaces and the pairs that are exact in
  // candidate-bit positions from ok_shift; .w (word deltas) the read minus
  // the proposal of each pair as two int16 halves (the layer's low; zero
  // unless both are nonzero). The reduce and the scene evidence pass both
  // write them. A shader with the identity marker has all of this; the
  // counter words ui_counter_word::identity_* count each offered pair's
  // verdict on every detection frame.
  namespace identity {
    inline constexpr std::string_view marker = "SUNSHINE_UI_IDENTITY";
    inline constexpr std::uint32_t version = 1u;
    inline constexpr std::uint32_t token_batch = 0x1u, gate_layer = 0x2u, gate_hudless = 0x4u;
    inline constexpr std::uint32_t stamp_layer = 4u, stamp_hudless = 5u, stamp_entries = 6u;
    inline constexpr std::uint32_t none = 0u, exact = 1u, mismatch = 2u, unstamped = 3u, unproposed = 4u;
    inline constexpr std::uint32_t space_none = 0u, space_token = 1u, space_present = 2u;
    inline constexpr std::uint32_t verdict_mask = 0xfu, space_shift = 4u, space_mask = 0x3u, hudless_shift = 8u, ok_shift = 16u;
    // The b2 words the renderer pushes: words 0-5 as before, then the three
    // proposals and Sunshine_UIIdentity.
    inline constexpr std::uint32_t b2_words = 10u;
  }
  static_assert(identity::stamp_entries == 6u && identity::unproposed < identity::verdict_mask &&
    ((identity::space_mask << identity::space_shift) & identity::verdict_mask) == 0u &&
    ((identity::verdict_mask | (identity::space_mask << identity::space_shift)) << identity::hudless_shift) <
      (1u << identity::ok_shift) && ((candidate::layer | candidate::hudless) << identity::ok_shift) <= 0xffffffffu &&
    (identity::token_batch & identity::gate_layer & identity::gate_hudless) == 0u);
  // The planes fit the 16384-texel D3D11 width at the shader's 8192 limit.
  static_assert(change_set::plane::judge + 1u == change_set::plane::dark && change_set::plane::tile + 1u == change_set::planes &&
    change_set::plane_words(254u) == 8u && change_set::plane_words(3840u) == 120u &&
    change_set::planes * change_set::plane_words(8192u) <= 16384u);

  // Fix 4, pin only UI (rule P2; docs/reshade-sbs.md, UI decision framework,
  // owns the thresholds and their measured justification;
  // game3d_ui_darkening.h is the CPU reference and aliases these, and
  // test_game3d_native_shader checks the shader's mirrors). A pure darkening
  // (the UI adds no colour of its own, P = s L) is not UI: on an eligible
  // source the mask pass zeroes the pin weight of a darkening pixel that
  // sharp structure does not keep and, for a change set, whose darkening
  // region (8-connected 16x16 tiles holding darkening pixels) has a proven
  // tile and no tile with a colour verdict in its 3x3 ring; an opacity
  // source must carry colour somewhere in the frame. The passes run after
  // the reduce exactly when b2 word 5 has rules::darkening_measured
  // (dispatched): the bits pass (one word on change_set::bits_group_rows
  // rows per group, as the change-set bits pass), the tiles pass (one group
  // per tile) and the region pass (one group); on sample frames the count
  // pass (one group per 16x16 detection tile, statistics rows from
  // statistics_row) and the finish pass (one group) write decision words
  // dk_unpinned and dk_kept. Integer mirrors: the opacity tolerance in
  // 1/255 of sRGB code (a float source compares its linear colour with the
  // sRGB EOTF of it), the change-set residual and tile effect in multiples
  // of the pair threshold (b2 word 1), the step as its inverse, the tile in
  // pixels and its minimum fitting pixels, and the minimum relative variance
  // in millionths. The region pass floods the tile grid in group-shared
  // memory, three sets of region_words words (a row of tiles padded to
  // whole words); a frame with more tiles (above about 6144x3456) unpins no
  // change-set darkening, nor does a frame whose regions do not settle
  // within the shader's iteration bound (two per tile row and 16 more).
  namespace darkening {
    inline constexpr std::uint32_t opacity_tolerance_255 = 4u, change_set_scale = 4u, step_inverse = 64u, tile = 16u,
      tile_min_pixels = 32u, tile_effect_scale = 4u, min_variance_ppm = 1u;
    inline constexpr std::uint32_t statistics_row = 192u, tile_rows = 3u, region_words = 2720u;
    // Word dk_kept's top bit: the decided opacity source carried no colour
    // on this frame, so it unpinned nothing (its darkening pixels are kept).
    inline constexpr std::uint32_t colourless = 0x80000000u;
    // The eligible sources: the UI color tag (2) and the offscreen UI layer
    // (source_layer) by opacity, the HUD-less difference (5) and the pre-UI
    // change set (source_pre_ui) by change set.
    constexpr bool opacity_source(std::uint32_t source) { return source == 2u || source == source_layer; }
    constexpr bool change_set_source(std::uint32_t source) { return source == 5u || source == source_pre_ui; }
    constexpr bool eligible(std::uint32_t source) { return opacity_source(source) || change_set_source(source); }
    // The tiles of an extent: the tiles pass's groups in x and y.
    constexpr std::uint32_t tiles(std::uint32_t pixels) { return (pixels + tile - 1u) / tile; }
    // Whether the region pass can flood a frame's tile grid.
    constexpr bool region_supported(std::uint32_t width, std::uint32_t height) {
      return std::uint64_t((tiles(width) + 31u) / 32u) * tiles(height) <= region_words;
    }
    // Whether a detection run with this b2 word 5 dispatches the darkening
    // passes (the shader's SUNSHINE_UI_DARKENING_MEASURED).
    constexpr bool dispatched(std::uint32_t rules) { return (rules & rules::darkening_measured) != 0u; }
  }
  // Whether a committed sample measured the darkening of its applied source
  // (words dk_unpinned and dk_kept): the passes dispatched, an eligible
  // source and a frame T1 did not reuse; and whether the mask pass applied
  // it (pin only UI on too).
  constexpr bool darkening_dispatched(std::uint32_t rules, std::uint32_t source, bool reused = false) {
    return darkening::dispatched(rules) && darkening::eligible(source) && !reused;
  }
  constexpr bool darkening_applied(std::uint32_t rules, std::uint32_t source, bool reused = false) {
    return darkening_dispatched(rules, source, reused) && (rules & rules::pin_only_ui) != 0u;
  }
  static_assert(darkening::eligible(2u) && darkening::eligible(5u) && darkening::eligible(10u) && darkening::eligible(12u) &&
    !darkening::eligible(0u) && !darkening::eligible(1u) && !darkening::eligible(3u) && !darkening::eligible(4u) &&
    !darkening::eligible(6u) && !darkening::eligible(8u) && !darkening::eligible(11u) && !darkening::eligible(13u) &&
    darkening::opacity_source(10u) && !darkening::opacity_source(5u) && darkening::change_set_source(12u) &&
    !darkening::change_set_source(2u) &&
    darkening_dispatched(rules::darkening_measured, 12u) && !darkening_dispatched(rules::darkening_measured, 12u, true) &&
    !darkening_applied(rules::darkening_measured, 10u) &&
    darkening_applied(rules::darkening_measured | rules::pin_only_ui, 10u) && !darkening_dispatched(rules::pin_only_ui, 10u) &&
    darkening::tiles(3840u) == 240u && darkening::tiles(2160u) == 135u && darkening::tiles(254u) == 16u &&
    // The tile plane's rows and words fit the plane from an extent of 3 (a
    // smaller one reads its missing rows as zero and unpins nothing), and
    // the three group-shared sets fit 32 KiB with the pass's flag words.
    3u * darkening::tiles(3u) <= 3u && 3u * darkening::tiles(17u) <= 17u && 3u * darkening::tiles(8192u) <= 8192u &&
    (darkening::tiles(8192u) + 31u) / 32u <= change_set::plane_words(8192u) &&
    (darkening::tiles(8u) + 31u) / 32u <= change_set::plane_words(8u) && 3u * darkening::region_words * 4u + 16u <= 32768u &&
    darkening::region_supported(3840u, 2160u) && darkening::region_supported(6144u, 3456u) &&
    !darkening::region_supported(7680u, 4320u) && darkening::colourless > 8192u * 8192u);

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
  // differ} (decision texel 11), and from selection revision 5 (decision
  // texels still_decision_texels) rows 144-152 from still_statistics_row:
  // per 16x16-cell compare group, H2's {still cells, compared cells, 0, 0}
  // (decision texel 12), 160 rows in all, and from selection revision 6
  // (decision texels change_set_decision_texels) rows 160-191 from
  // change_set_statistics_row: per tile, the layer against its pair's
  // Present {changed, unchanged, non-finite, changed and kept by the 3x3
  // rule}, then {changed against the Present one back, two back, judge
  // pixels, judge pixels the 3x3 rule kept} (decision texels 13-15), 192
  // rows in all, and from fix 4 (planes marker change_set::planes) rows
  // 192-207 from darkening_statistics_row: per tile, the darkening count
  // pass's {unpinned, kept, 0, 0} (decision words dk_unpinned and dk_kept),
  // 208 rows in all. The cells have a texture of their own, cells_x by
  // cells_y, and so do H2's previous presented-luma cell means (R32_UINT, u5
  // of the compare pass only).
  inline constexpr std::uint32_t layer_statistics_row = 64u, judgment_statistics_row = 80u, scene_partial_row = 112u,
    pre_ui_statistics_row = 128u, still_statistics_row = 144u, change_set_statistics_row = 160u,
    darkening_statistics_row = darkening::statistics_row;
  // The statistics rows of a shader with these markers (images: its
  // SUNSHINE_UI_SCENE_EVIDENCE_IMAGES, decision_texels: its
  // SUNSHINE_UI_DECISION_TEXELS, planes: its SUNSHINE_UI_CHANGE_SET_PLANES;
  // zero sizes no darkening rows).
  constexpr std::uint32_t statistics_rows(std::uint32_t images, std::uint32_t decision_texels = change_set_decision_texels,
      std::uint32_t planes = 0u) {
    return decision_texels >= change_set_decision_texels && planes >= change_set::planes ? darkening_statistics_row + 16u :
      decision_texels >= change_set_decision_texels ? change_set_statistics_row + 32u :
      decision_texels >= still_decision_texels ? still_statistics_row + 16u :
      decision_texels >= pre_ui_decision_texels ? pre_ui_statistics_row + 16u :
      scene_partial_row + (images ? scene::cells_y / 16u : 0u);
  }
  static_assert(scene::cells_x == 16u * 16u && scene::cells_y % 16u == 0u, "Compare groups must fill the 16 statistics columns");
  static_assert(judgment_statistics_row + 32u == scene_partial_row && statistics_rows(0, h1_decision_texels) == 112u &&
    statistics_rows(2, h1_decision_texels) == 121u && scene_partial_row + scene::cells_y / 16u <= pre_ui_statistics_row &&
    statistics_rows(2, pre_ui_decision_texels) == 144u && statistics_rows(0, pre_ui_decision_texels) == 144u &&
    pre_ui_statistics_row + 16u == still_statistics_row &&
    still_statistics_row + scene::cells_y / 16u <= statistics_rows(2, still_decision_texels) &&
    statistics_rows(2, still_decision_texels) == 160u && statistics_rows(0, still_decision_texels) == 160u &&
    still_statistics_row + 16u == change_set_statistics_row && statistics_rows(2) == 192u && statistics_rows(0) == 192u &&
    change_set_statistics_row + 32u == darkening_statistics_row && statistics_rows(2, change_set_decision_texels, 9u) == 208u &&
    statistics_rows(0, change_set_decision_texels, change_set::planes) == 208u &&
    statistics_rows(0, change_set_decision_texels, 6u) == 192u && statistics_rows(0, still_decision_texels, 9u) == 160u);
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
  // H2: the still-screen source, the flag of b2 word 5, the stillness
  // tolerance in luma fixed point and the first statistics row of the
  // compare groups' stillness counts.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 4> hlsl_still_defines{{
    {"SUNSHINE_UI_SOURCE_STILL", source_still},
    {"SUNSHINE_UI_STILL_FLATTEN", still::flatten},
    {"SUNSHINE_UI_STILL_TOLERANCE", still::tolerance},
    {"SUNSHINE_UI_STILL_ROW", still_statistics_row},
  }};
  // Fix 3: the change-set rule bits of b2 word 5, the inferred scale, lit
  // share and 3x3 bound, the judge kinds, the first statistics row of the
  // change-set shadow and its bit planes.
  // Fix 4: the pin-only-UI and darkening rule bits, the darkening planes and
  // statistics row, and the darkening rule's integer mirrors.
  // S3: the identity marker, Sunshine_UIIdentity's bits, the stamp entries,
  // the verdicts and label spaces and the verdict word's layout.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 54> hlsl_change_set_defines{{
    {"SUNSHINE_UI_IDENTITY", identity::version},
    {"SUNSHINE_UI_IDENTITY_TOKEN_BATCH", identity::token_batch},
    {"SUNSHINE_UI_IDENTITY_GATE_LAYER", identity::gate_layer},
    {"SUNSHINE_UI_IDENTITY_GATE_HUDLESS", identity::gate_hudless},
    {"SUNSHINE_UI_IDENTITY_STAMP_LAYER", identity::stamp_layer},
    {"SUNSHINE_UI_IDENTITY_STAMP_HUDLESS", identity::stamp_hudless},
    {"SUNSHINE_UI_IDENTITY_EXACT", identity::exact},
    {"SUNSHINE_UI_IDENTITY_MISMATCH", identity::mismatch},
    {"SUNSHINE_UI_IDENTITY_UNSTAMPED", identity::unstamped},
    {"SUNSHINE_UI_IDENTITY_UNPROPOSED", identity::unproposed},
    {"SUNSHINE_UI_IDENTITY_SPACE_TOKEN", identity::space_token},
    {"SUNSHINE_UI_IDENTITY_SPACE_PRESENT", identity::space_present},
    {"SUNSHINE_UI_IDENTITY_SPACE_SHIFT", identity::space_shift},
    {"SUNSHINE_UI_IDENTITY_HUDLESS_SHIFT", identity::hudless_shift},
    {"SUNSHINE_UI_IDENTITY_OK_SHIFT", identity::ok_shift},
    {"SUNSHINE_UI_CHANGE_SET_PAIR_BACKBUFFER", change_set::pair_backbuffer},
    {"SUNSHINE_UI_IDENTITY_STAMP_ENTRIES", identity::stamp_entries},
    {"SUNSHINE_UI_IDENTITY_SPACE_MASK", identity::space_mask},
    {"SUNSHINE_UI_PIN_ONLY_UI", rules::pin_only_ui},
    {"SUNSHINE_UI_DARKENING_MEASURED", rules::darkening_measured},
    {"SUNSHINE_UI_TAG_LINEAR", rules::tag_linear},
    {"SUNSHINE_UI_CHANGE_SET_SHADOW", change_set::shadow},
    {"SUNSHINE_UI_CHANGE_SET_GAP", change_set::gap},
    {"SUNSHINE_UI_CHANGE_SET_PAIR_SHIFT", change_set::pair_shift},
    {"SUNSHINE_UI_CHANGE_SET_PAIR_MASK", change_set::pair_mask},
    {"SUNSHINE_UI_CHANGE_SET_RETAINED_1", change_set::retained_1},
    {"SUNSHINE_UI_CHANGE_SET_RETAINED_2", change_set::retained_2},
    {"SUNSHINE_UI_CHANGE_SET_INFERRED_SCALE", change_set::inferred_scale},
    {"SUNSHINE_UI_CHANGE_SET_LIT_PERCENT", change_set::lit_percent},
    {"SUNSHINE_UI_CHANGE_SET_MIN_NEIGHBOURHOOD", change_set::min_neighbourhood},
    {"SUNSHINE_UI_CHANGE_SET_JUDGE_UI_ALPHA", change_set::judge::ui_alpha},
    {"SUNSHINE_UI_CHANGE_SET_JUDGE_UI_COLOR", change_set::judge::ui_color},
    {"SUNSHINE_UI_CHANGE_SET_ROW", change_set_statistics_row},
    {"SUNSHINE_UI_CHANGE_SET_PLANES", change_set::planes},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_CHANGED", change_set::plane::changed},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_UNCHANGED", change_set::plane::unchanged},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_INVALID", change_set::plane::invalid},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_CHANGED_1", change_set::plane::changed_1},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_CHANGED_2", change_set::plane::changed_2},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_JUDGE", change_set::plane::judge},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_DARK", change_set::plane::dark},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_STRUCT", change_set::plane::structure},
    {"SUNSHINE_UI_CHANGE_SET_PLANE_TILE", change_set::plane::tile},
    {"SUNSHINE_UI_DARKENING_ROW", darkening_statistics_row},
    {"SUNSHINE_UI_DARKENING_OPACITY_TOLERANCE", darkening::opacity_tolerance_255},
    {"SUNSHINE_UI_DARKENING_CHANGE_SET_SCALE", darkening::change_set_scale},
    {"SUNSHINE_UI_DARKENING_STEP_INVERSE", darkening::step_inverse},
    {"SUNSHINE_UI_DARKENING_TILE", darkening::tile},
    {"SUNSHINE_UI_DARKENING_TILE_MIN_PIXELS", darkening::tile_min_pixels},
    {"SUNSHINE_UI_DARKENING_TILE_EFFECT_SCALE", darkening::tile_effect_scale},
    {"SUNSHINE_UI_DARKENING_MIN_VARIANCE_PPM", darkening::min_variance_ppm},
    {"SUNSHINE_UI_DARKENING_TILE_ROWS", darkening::tile_rows},
    {"SUNSHINE_UI_DARKENING_REGION_WORDS", darkening::region_words},
    {"SUNSHINE_UI_DARKENING_COLOURLESS", darkening::colourless},
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
    // Texel 12 (selection revision 5, H2): of the D grid's cells, those whose
    // presented-luma mean stayed within still::tolerance of the previous
    // measured sample's, and those compared (a previous mean existed); zero
    // when the evidence passes did not run or the depth was inactive.
    inline constexpr std::size_t still_cells = 48, still_compared = 49;
    // Texel 12 .z/.w (S3, a shader with identity::marker): the identity
    // verdicts of the offered layer and HUD-less pairs and their deltas
    // (namespace identity), written by the reduce and rewritten unchanged by
    // the scene evidence pass; zero from a shader without the marker.
    inline constexpr std::size_t id_verdicts = 50, id_deltas = 51;
    // Texels 13-15 (selection revision 6, fix 3): the change-set shadow, the
    // offscreen layer against its pair's Present (b2 word 5's offset) at
    // change_set::inferred_scale times b2 word 4, on sample frames with an
    // offered layer (all zero otherwise): changed, unchanged and non-finite
    // pixels, tiles at least 99% unchanged; changed pixels the 3x3 rule
    // keeps, changed pixels against the Presents one and two back (zero
    // unless bound), the judge's pixels (the first offered and accepted
    // declared alpha above zero); judge pixels the 3x3 rule keeps and the
    // judge (change_set::judge). Nothing decides from them.
    inline constexpr std::size_t cs_changed = 52, cs_unchanged = 53, cs_nonfinite = 54, cs_matching_tiles = 55;
    inline constexpr std::size_t cs_filtered = 56, cs_changed_1 = 57, cs_changed_2 = 58, cs_judge_pixels = 59;
    inline constexpr std::size_t cs_judge_tp = 60, cs_judge_kind = 61;
    // Texel 15 .z/.w (fix 4, a shader whose planes marker is
    // change_set::planes): the pin-only-UI darkening of a sample whose
    // darkening passes ran (darkening_dispatched: rules::darkening_measured,
    // an eligible applied source, a frame T1 did not reuse), whether or not
    // the mask pass applied it: the source's darkening pixels it unpins and
    // those that stay pinned (sharp structure, a change set's unproven
    // region, or an opacity source without colour on the frame, which also
    // sets darkening::colourless in dk_kept); zero otherwise. Never reused
    // for anything else.
    inline constexpr std::size_t dk_unpinned = 62, dk_kept = 63;
  }
  static_assert(decision_word::alpha_opaque + 1 < 4 * min_decision_texels &&
    decision_word::pre_ui_scene_image < 4 * scene_decision_texels && decision_word::valid_bits < 4 * layer_decision_texels &&
    decision_word::frame_reason == 4 * judgment_decision_texels - 1 && decision_word::h1 == 4 * h1_decision_texels - 1 &&
    decision_word::pre_ui_match == 4 * h1_decision_texels &&
    decision_word::presented_lit_differs == 4 * pre_ui_decision_texels - 1 &&
    decision_word::still_cells == 4 * pre_ui_decision_texels && decision_word::still_compared == decision_word::still_cells + 1 &&
    decision_word::still_cells + 4 == 4 * still_decision_texels && decision_word::cs_changed == 4 * still_decision_texels &&
    decision_word::id_verdicts == decision_word::still_compared + 1 && decision_word::id_deltas + 1 == decision_word::cs_changed &&
    decision_word::cs_judge_kind + 3 == 4 * change_set_decision_texels && change_set_decision_texels <= max_decision_texels &&
    decision_word::dk_unpinned == decision_word::cs_judge_kind + 1 && decision_word::dk_kept + 1 == 4 * change_set_decision_texels);
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
