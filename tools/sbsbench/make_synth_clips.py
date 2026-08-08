#!/usr/bin/env python3
"""
make_synth_clips - generate the failure-mode clips the movie recordings don't cover
(tools/sbsbench/DATASETS.md core-suite table). Deterministic (fixed seed) and synthetic/spliced, so
they add no licensing surface. The generated frames are COMMITTED (clips/); rerun this only when
changing a clip's design, and regenerate baselines in the same commit.

  flat_page    a static desktop/document page: the depth model should output near-flat depth --
               diagnoses flat-content hallucination outside the SBS quality gate and measures
               pipeline shimmer on static input (flicker floor).
  fast_motion  a textured block crossing a textured background at a KNOWN 30 px/frame --
               the async-depth ghost scenario (and the anchor for a future ghost metric).
  scene_cut    a hard cut spliced from two committed clips (c841 bright/calm -> c647 dark/crowd)
               -- depth-normalization swim across cuts (A1 snap validation); expect the swim /
               flicker worst frame AT the cut.
  exposure_flash_strobe
               one static synthetic scene under exact full-frame RGB gain changes. Lossless PNG
               keeps this a provable exposure-only stimulus for the production shot-state trace.
  sustained_motion_scene_cut
               an initial cut latches shot state, broad persistent motion keeps both detector
               arms closed, then a second real cut must use the relative-geometry escape.
  structureless_history_bridge
               a lossless structured A -> black -> A flash followed by structured A -> black
               slate -> different structured B. The flash is ignored; persistent black and B cut.
  structureless_white_history_bridge
               the same authenticated history bridge using saturated white, whose lower raw-RGB
               replacement fraction exercises the support-loss-specific appearance gate.
  subtitle_cjk_dense
               movie-like moving imagery with dense, white, dark-outlined CJK subtitles.
  subtitle_bilingual_tall_stack
               an unusually tall four-line CJK/English subtitle stack near the bottom edge.
  subtitle_top_bottom_disjoint
               simultaneous translated-note and dialogue blocks at the top and bottom of frame.
  subtitle_cjk_highres_transitions
               a 2560x1440 thin-stroke CJK probe with empty/appear/replace/disappear transitions
               and one broad scene cut while the overlay changes.
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, "clips")
W, H, N = 854, 480, 24
EXPOSURE_GAINS_PERCENT = (
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    45, 165, 60, 150, 75, 135, 50, 160, 65, 145, 55, 155, 70,
)
SHOT_STATE_MONITOR_FROM_FRAME = 2
EXPOSURE_STABLE_FROM_FRAME = 10
SHOT_CUT_FRAME = N // 2 + 1
SUSTAINED_SETUP_CUT_FRAME = 11
SUSTAINED_TRUE_CUT_FRAME = 27
SUSTAINED_ESCAPE_PULSE_FRAME = SUSTAINED_TRUE_CUT_FRAME + 1
SUSTAINED_FRAME_COUNT = 36
SUSTAINED_ROLL_PX = 96
BRIDGE_FLASH_FRAME = 11
BRIDGE_FLASH_RETURN_FRAME = 12
BRIDGE_SLATE_FRAME = 17
BRIDGE_PERSISTENT_FRAME = BRIDGE_SLATE_FRAME + 1
BRIDGE_NEW_SCENE_FRAME = 21
BRIDGE_FRAME_COUNT = 30
BRIDGE_UNIFORM_RGB_BY_CLIP = {
    "structureless_history_bridge": (0, 0, 0),
    "structureless_white_history_bridge": (255, 255, 255),
}
BRIDGE_SCENE_A_BY_CLIP = {
    "structureless_history_bridge": ("c841", 1),
    # The bright page replaces only 28.6% of host model texels when changed to uniform white.
    # This prevents the conformance clip from accidentally depending on the ordinary 70% raw gate.
    "structureless_white_history_bridge": ("flat_page", 1),
}
BRIDGE_SCENE_B = ("c647", 13)
SUBTITLE_CUE_START_FRAMES = (1, 7, 13, 19)
SUBTITLE_STANDARD_CLIPS = (
    "subtitle_cjk_dense",
    "subtitle_bilingual_tall_stack",
    "subtitle_top_bottom_disjoint",
)
SUBTITLE_HIGHRES_CLIP = "subtitle_cjk_highres_transitions"
SUBTITLE_CLIPS = SUBTITLE_STANDARD_CLIPS + (SUBTITLE_HIGHRES_CLIP,)
SUBTITLE_HIGHRES_W = 2560
SUBTITLE_HIGHRES_H = 1440
SUBTITLE_DETECTOR_TARGET_W = 770
SUBTITLE_DETECTOR_TARGET_H = 434
SUBTITLE_HIGHRES_STROKE_PX = 3
SUBTITLE_HIGHRES_OUTLINE_PX = 2
SUBTITLE_HIGHRES_STATE_BY_FRAME = (
    ["empty"] * 4 +
    ["cue-a"] * 4 +
    ["cue-b"] * 4 +
    ["empty"] * 4 +
    ["cue-c"] * 4 +
    ["cue-d"] * 4
)
SUBTITLE_HIGHRES_SCENE_BY_FRAME = ["dusk-city"] * 20 + ["warm-interior"] * 4
SUBTITLE_HIGHRES_TEXT = {
    "cue-a": "危险在前不要回头",
    "cue-b": "我们快走等等不要走",
    "cue-c": "不要回头危险在前",
    "cue-d": "等等我们快走",
}
SUBTITLE_CJK_CUES = (
    "危险在前",
    "不要回头",
    "我们快走",
    "等等不要走",
)
SUBTITLE_ENGLISH_CUES = (
    "DANGER AHEAD",
    "DO NOT TURN",
    "STAY CLOSE",
    "WAIT FOR ME",
)

# These tiny stroke/bitmap fonts are authored as part of the fixture. Regeneration therefore does
# not depend on an installed Windows font, font fallback, locale, FreeType version, or antialiasing
# behavior. The CJK shapes intentionally retain dense interior strokes at production input scale.
CJK_STROKES = {
    "危": ((6, 1, 11, 1), (6, 1, 4, 4), (4, 4, 13, 4), (4, 4, 2, 9),
          (2, 9, 2, 14), (5, 7, 13, 7), (13, 7, 11, 10), (6, 9, 11, 9),
          (6, 9, 6, 14), (6, 14, 12, 14), (12, 14, 12, 12)),
    "险": ((1, 2, 5, 2), (5, 2, 3, 7), (3, 7, 5, 9), (5, 9, 3, 12),
          (3, 12, 3, 15), (10, 1, 6, 5), (10, 1, 14, 5), (7, 6, 13, 6),
          (7, 8, 8, 10), (13, 8, 12, 10), (6, 12, 14, 12), (8, 14, 12, 14)),
    "在": ((2, 4, 14, 4), (8, 1, 7, 5), (7, 5, 4, 10), (8, 6, 8, 14),
          (8, 8, 14, 8), (5, 14, 14, 14)),
    "前": ((5, 1, 7, 3), (11, 1, 9, 3), (3, 4, 13, 4), (4, 6, 9, 6),
          (4, 6, 4, 14), (4, 9, 9, 9), (4, 12, 9, 12), (9, 6, 9, 14),
          (12, 6, 12, 12), (14, 5, 14, 14), (14, 14, 11, 14)),
    "不": ((2, 3, 14, 3), (8, 3, 8, 14), (8, 6, 3, 11), (9, 7, 14, 12)),
    "要": ((2, 2, 14, 2), (4, 2, 4, 7), (8, 2, 8, 7), (12, 2, 12, 7),
          (3, 7, 13, 7), (7, 8, 5, 12), (5, 12, 13, 12), (10, 8, 9, 13),
          (9, 13, 5, 15), (8, 11, 14, 15)),
    "回": ((2, 2, 14, 2), (2, 2, 2, 14), (14, 2, 14, 14), (2, 14, 14, 14),
          (5, 5, 11, 5), (5, 5, 5, 11), (11, 5, 11, 11), (5, 11, 11, 11)),
    "头": ((6, 1, 8, 4), (3, 4, 5, 6), (2, 8, 14, 8), (8, 4, 8, 10),
          (8, 10, 4, 15), (9, 10, 14, 15)),
    "我": ((3, 3, 10, 2), (6, 2, 6, 13), (2, 7, 11, 6), (2, 11, 10, 9),
          (10, 1, 10, 8), (10, 8, 13, 14), (13, 14, 15, 11), (6, 13, 3, 15)),
    "们": ((4, 1, 1, 7), (3, 5, 3, 15), (7, 3, 7, 15), (7, 3, 14, 3),
          (14, 3, 14, 15), (10, 1, 12, 3), (9, 6, 9, 12), (11, 6, 11, 12)),
    "快": ((3, 3, 1, 7), (4, 1, 4, 15), (5, 5, 6, 8), (10, 1, 10, 12),
          (7, 5, 14, 5), (7, 10, 14, 10), (10, 10, 7, 15), (11, 10, 15, 15)),
    "走": ((8, 1, 8, 7), (4, 4, 12, 4), (3, 7, 13, 7), (8, 7, 6, 12),
          (6, 12, 3, 15), (8, 9, 11, 12), (11, 12, 15, 14), (7, 14, 15, 14)),
    "等": ((4, 1, 2, 5), (4, 2, 7, 2), (10, 1, 8, 5), (10, 2, 14, 2),
          (4, 6, 13, 6), (8, 5, 8, 10), (3, 10, 14, 10), (5, 12, 13, 12),
          (12, 10, 12, 15), (7, 12, 8, 14), (12, 15, 9, 15)),
}

LATIN_5X7 = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
}

# name + description written to each clip's meta.json (self-describing; the report labels by name).
DESC = {
    "flat_page": "Synthetic static document/desktop page: flat-content depth hallucination (A3).",
    "fast_motion": "Synthetic textured block crossing a textured background at 30 px/frame: async-depth ghost.",
    "scene_cut": (
        "Hard cut spliced kitchen-vlog -> washerwoman-pond: depth-normalization swim across "
        "cuts (A1). Shot-state expectations re-verified against the V2 cut-only analysis trace "
        "on 2026-08-04 (pulse schedule unchanged)."
    ),
    "flat_transition": (
        "Textured depth scene cutting to a static flat page: normalization recovery and "
        "false-stereo decay. Shot-state expectations re-verified against the V2 cut-only "
        "analysis trace on 2026-08-04 (pulse schedule unchanged)."
    ),
    "exposure_flash_strobe": (
        "Static synthetic depth scene under exact global RGB gain flashes/strobe: exposure must "
        "not reset the cut detector or valid depth history. Cut-state expectations were "
        "re-verified against the V2 cut-only analysis trace on 2026-08-04 (no pulses, "
        "unchanged)."
    ),
    "sustained_motion_scene_cut": (
        "A setup cut latches shot state; broad persistent horizontal motion prevents either "
        "proposal arm from rearming, then a real scene cut must survive one held-endpoint update "
        "before the relative-depth escape pulses. Shot-state expectations re-verified against "
        "the V2 cut-only analysis trace on 2026-08-04 (pulse schedule unchanged)."
    ),
    "structureless_history_bridge": (
        "A one-frame black flash returns to structured scene A without a cut; a later black "
        "slate must cut when low structure persists, then different structured scene B must cut "
        "on its visible return."
        " Pulse expectations re-derived 2026-08-04 for the V2 cut-only analysis: "
        "the structured return at frame 21 is accepted through the resolver's "
        "two-update geometry confirmation, pulsing at frame 22."
    ),
    "structureless_white_history_bridge": (
        "A one-frame white flash returns to structured scene A without a cut; a later white "
        "slate must cut when low structure persists, then different structured scene B must cut "
        "on its visible return."
        " Pulse expectations re-derived 2026-08-04 for the V2 cut-only analysis: "
        "the structured return at frame 21 is accepted through the resolver's "
        "two-update geometry confirmation, pulsing at frame 22."
    ),
    "subtitle_cjk_dense": (
        "Lossless synthetic movie imagery with dense white, dark-outlined CJK burned-in "
        "subtitles that change piecewise-statically every six frames."
    ),
    "subtitle_bilingual_tall_stack": (
        "Lossless synthetic movie imagery with an unusually tall four-line CJK/English "
        "burned-in subtitle stack near the bottom edge."
    ),
    "subtitle_top_bottom_disjoint": (
        "Lossless synthetic movie imagery with simultaneous disjoint top-note and bottom-dialogue "
        "burned-in subtitle regions."
    ),
    "subtitle_cjk_highres_transitions": (
        "Lossless 2560x1440 synthetic movie imagery with fine three-pixel CJK strokes and a "
        "two-pixel dark outline; authenticates empty, appear, subtitle-only replacement, "
        "disappear, and broad scene-cut transitions before 770x434 inference downscaling."
    ),
}


def write_meta(clip, **extra):
    os.makedirs(os.path.join(CLIPS, clip), exist_ok=True)
    provenance = (
        "Constructed probe: hard cut spliced from two committed clips by make_synth_clips.py."
        if clip == "scene_cut" else
        "Synthetic probe generated by make_synth_clips.py."
    )
    metadata = {
        "name": clip,
        "description": DESC.get(clip, ""),
        **extra,
        "content_type": "synthetic",
        "provenance_note": provenance,
    }
    with open(os.path.join(CLIPS, clip, "meta.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def save(clip, i, arr):
    d = os.path.join(CLIPS, clip)
    os.makedirs(d, exist_ok=True)
    Image.fromarray(arr).save(os.path.join(d, f"frame_{i + 1:05d}.jpg"), quality=90)


def save_gt_depth(clip, i, disparity):
    """16-bit normalized inverse-depth/disparity reference, aligned to the source frame."""
    d = os.path.join(CLIPS, clip, "gt_depth")
    os.makedirs(d, exist_ok=True)
    arr = np.round(np.clip(disparity, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(arr).save(os.path.join(d, f"frame_{i + 1:05d}.png"))


def _draw_stroke(canvas, stroke):
    """Draw one integer Bresenham segment into a tiny authored glyph cell."""
    x0, y0, x1, y1 = stroke
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        canvas[y0, x0] = True
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def _render_cjk_glyph(character, scale):
    if character not in CJK_STROKES:
        raise ValueError(f"missing authored CJK glyph {character!r}")
    base = np.zeros((16, 16), dtype=bool)
    for stroke in CJK_STROKES[character]:
        _draw_stroke(base, stroke)
    return np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)


def _render_latin_glyph(character, scale):
    if character not in LATIN_5X7:
        raise ValueError(f"missing authored Latin glyph {character!r}")
    base = np.array([
        [pixel == "1" for pixel in row]
        for row in LATIN_5X7[character]
    ], dtype=bool)
    return np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)


def _render_authored_text(text, alphabet, scale):
    glyphs = []
    for character in text:
        if character == " ":
            glyphs.append(np.zeros((7 * scale, 3 * scale), dtype=bool))
        elif alphabet == "cjk":
            glyphs.append(_render_cjk_glyph(character, scale))
        elif alphabet == "latin":
            glyphs.append(_render_latin_glyph(character, scale))
        else:
            raise ValueError(f"unknown subtitle alphabet {alphabet!r}")

    spacing = 2 * scale
    width = sum(glyph.shape[1] for glyph in glyphs) + spacing * (len(glyphs) - 1)
    height = max(glyph.shape[0] for glyph in glyphs)
    line = np.zeros((height, width), dtype=bool)
    x = 0
    for glyph in glyphs:
        y = (height - glyph.shape[0]) // 2
        line[y:y + glyph.shape[0], x:x + glyph.shape[1]] = glyph
        x += glyph.shape[1] + spacing
    return line


def _dilate_binary(mask, radius):
    """Square dilation on a compact text block; avoids platform-dependent image filtering."""
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    expanded = np.zeros_like(mask)
    for y_offset in range(2 * radius + 1):
        for x_offset in range(2 * radius + 1):
            expanded |= padded[
                y_offset:y_offset + mask.shape[0],
                x_offset:x_offset + mask.shape[1],
            ]
    return expanded


def _subtitle_layout(clip, cue_index):
    cjk = SUBTITLE_CJK_CUES[cue_index]
    english = SUBTITLE_ENGLISH_CUES[cue_index]
    white = (245, 245, 240)
    if clip == "subtitle_cjk_dense":
        return (("bottom", ((cjk, "cjk", 3, white),)),)
    if clip == "subtitle_bilingual_tall_stack":
        next_index = (cue_index + 1) % len(SUBTITLE_CJK_CUES)
        return (("bottom", (
            (cjk, "cjk", 3, white),
            (english, "latin", 4, white),
            (SUBTITLE_CJK_CUES[next_index], "cjk", 2, white),
            (SUBTITLE_ENGLISH_CUES[next_index], "latin", 3, white),
        )),)
    if clip == "subtitle_top_bottom_disjoint":
        top_index = (cue_index + 2) % len(SUBTITLE_CJK_CUES)
        return (
            ("top", ((SUBTITLE_CJK_CUES[top_index], "cjk", 2, white),)),
            ("bottom", (
                (cjk, "cjk", 3, white),
                (english, "latin", 3, white),
            )),
        )
    raise ValueError(f"unknown subtitle clip {clip}")


def _subtitle_layers(clip, frame_id):
    """Return authored RGB glyphs, glyph/outline masks, and loose binary region GT."""
    if frame_id < 1 or frame_id > N:
        raise ValueError(f"subtitle frame must be in [1, {N}]")
    cue_index = (frame_id - 1) // (N // len(SUBTITLE_CUE_START_FRAMES))
    glyph_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    glyph_mask = np.zeros((H, W), dtype=bool)
    outline_mask = np.zeros((H, W), dtype=bool)
    region_mask = np.zeros((H, W), dtype=np.uint8)

    for anchor, line_specs in _subtitle_layout(clip, cue_index):
        lines = [
            (_render_authored_text(text, alphabet, scale), color)
            for text, alphabet, scale, color in line_specs
        ]
        line_gap = 6
        block_width = max(line.shape[1] for line, _ in lines)
        block_height = sum(line.shape[0] for line, _ in lines) + line_gap * (len(lines) - 1)
        block_glyph = np.zeros((block_height, block_width), dtype=bool)
        block_rgb = np.zeros((block_height, block_width, 3), dtype=np.uint8)
        line_y = 0
        for line, color in lines:
            line_x = (block_width - line.shape[1]) // 2
            target = block_glyph[
                line_y:line_y + line.shape[0],
                line_x:line_x + line.shape[1],
            ]
            target |= line
            color_target = block_rgb[
                line_y:line_y + line.shape[0],
                line_x:line_x + line.shape[1],
            ]
            color_target[line] = color
            line_y += line.shape[0] + line_gap

        outline_radius = 3
        block_outline = _dilate_binary(block_glyph, outline_radius) & ~block_glyph
        x = (W - block_width) // 2
        if anchor == "top":
            y = 44
        elif anchor == "bottom":
            y = H - 30 - block_height
        else:
            raise ValueError(f"unknown subtitle anchor {anchor!r}")
        ys = slice(y, y + block_height)
        xs = slice(x, x + block_width)
        glyph_mask[ys, xs] |= block_glyph
        outline_mask[ys, xs] |= block_outline
        block_target = glyph_rgb[ys, xs]
        block_target[block_glyph] = block_rgb[block_glyph]

        # Ground truth deliberately covers one loose rectangle per visual block. It is not a
        # glyph-tight segmentation and therefore does not put a parallax cliff at every stroke.
        region_pad_x = 18
        region_pad_y = 12
        region_mask[
            max(0, y - region_pad_y):min(H, y + block_height + region_pad_y),
            max(0, x - region_pad_x):min(W, x + block_width + region_pad_x),
        ] = 255

    return glyph_rgb, glyph_mask, outline_mask, region_mask


def _fill_rect(frame, x0, y0, x1, y1, color):
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = color


def _movie_background(frame_id, variant):
    """Deterministic dusk-city motion plate with parallax, silhouettes, and moving traffic."""
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[..., 0] = 18 + yy * 32 // H
    frame[..., 1] = 24 + yy * 27 // H
    frame[..., 2] = 48 + yy * 38 // H
    phase = frame_id * 4 + variant * 53

    moon_x = 120 + variant * 230
    moon = (xx - moon_x) ** 2 + (yy - 86) ** 2 <= 31 ** 2
    frame[moon] = (174, 166, 141)

    hill_top = 205 + np.abs(((xx + phase) % 230) - 115) // 5
    hills = (yy >= hill_top) & (yy < 312)
    frame[hills] = (25, 36, 50)

    for building_index in range(13):
        width = 58 + (building_index * 19 + variant * 11) % 47
        raw_x = (building_index * 91 - phase * 2) % (W + 150) - 75
        top = 165 + (building_index * 37 + variant * 23) % 95
        color = (24 + building_index % 3 * 5, 30 + building_index % 4 * 4, 40 + building_index % 5 * 3)
        _fill_rect(frame, raw_x, top, raw_x + width, 320, color)
        for window_y in range(top + 14, 300, 22):
            for window_x in range(raw_x + 9, raw_x + width - 6, 18):
                lit = (window_x + window_y + frame_id + variant * 7) % 5 == 0
                if lit:
                    _fill_rect(frame, window_x, window_y, window_x + 5, window_y + 7, (151, 112, 61))

    road = yy >= 312
    road_shade = np.clip(58 - (yy - 312) * 33 // (H - 312), 20, 58).astype(np.uint8)
    frame[..., 0][road] = road_shade[road]
    frame[..., 1][road] = np.maximum(18, road_shade[road] - 8)
    frame[..., 2][road] = np.maximum(22, road_shade[road] - 3)

    for stripe_index in range(8):
        stripe_x = (stripe_index * 150 - frame_id * 19 + variant * 31) % (W + 180) - 90
        _fill_rect(frame, stripe_x, 402, stripe_x + 75, 409, (135, 116, 78))

    car_x = (frame_id * 41 + variant * 149) % (W + 260) - 130
    _fill_rect(frame, car_x, 337, car_x + 125, 380, (72 + variant * 17, 38, 42))
    _fill_rect(frame, car_x + 24, 322, car_x + 96, 342, (47, 58, 70))
    for wheel_x in (car_x + 26, car_x + 98):
        wheel = (xx - wheel_x) ** 2 + (yy - 381) ** 2 <= 12 ** 2
        frame[wheel] = (12, 14, 18)

    pedestrian_x = W - 80 - ((frame_id * 7 + variant * 29) % 180)
    head = (xx - pedestrian_x) ** 2 + (yy - 284) ** 2 <= 13 ** 2
    frame[head] = (13, 17, 22)
    _fill_rect(frame, pedestrian_x - 10, 297, pedestrian_x + 11, 368, (12, 16, 21))

    # A deterministic fine texture avoids an unrealistically sterile vector plate while remaining
    # bounded below subtitle white. It also makes the fixture useful for real depth inference.
    texture = ((xx * 3 + yy * 5 + frame_id * 7 + variant * 13) % 7).astype(np.uint16)
    frame = np.minimum(frame.astype(np.uint16) + texture[..., None], 210).astype(np.uint8)
    return frame


def _prepare_subtitle_output(clip):
    out = os.path.join(CLIPS, clip)
    region_out = os.path.join(out, "gt_subtitle_region")
    os.makedirs(region_out, exist_ok=True)
    for directory in (out, region_out):
        for filename in os.listdir(directory):
            if filename.startswith("frame_") and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(directory, filename))
    return out, region_out


def _subtitle_clip(clip, variant):
    out, region_out = _prepare_subtitle_output(clip)
    for frame_id in range(1, N + 1):
        frame = _movie_background(frame_id, variant)
        glyph_rgb, glyph_mask, outline_mask, region_mask = _subtitle_layers(clip, frame_id)
        frame[outline_mask] = (8, 9, 12)
        frame[glyph_mask] = glyph_rgb[glyph_mask]
        Image.fromarray(frame).save(
            os.path.join(out, f"frame_{frame_id:05d}.png"), compress_level=9)
        Image.fromarray(region_mask, mode="L").save(
            os.path.join(region_out, f"frame_{frame_id:05d}.png"), compress_level=9)


def subtitle_cjk_dense():
    _subtitle_clip("subtitle_cjk_dense", 0)


def subtitle_bilingual_tall_stack():
    _subtitle_clip("subtitle_bilingual_tall_stack", 1)


def subtitle_top_bottom_disjoint():
    _subtitle_clip("subtitle_top_bottom_disjoint", 2)


def _highres_subtitle_layers(frame_id):
    """Fine authored subtitle pixels and loose source-resolution region truth."""
    if frame_id < 1 or frame_id > N:
        raise ValueError(f"subtitle frame must be in [1, {N}]")
    glyph_rgb = np.zeros(
        (SUBTITLE_HIGHRES_H, SUBTITLE_HIGHRES_W, 3), dtype=np.uint8)
    glyph_mask = np.zeros((SUBTITLE_HIGHRES_H, SUBTITLE_HIGHRES_W), dtype=bool)
    outline_mask = np.zeros((SUBTITLE_HIGHRES_H, SUBTITLE_HIGHRES_W), dtype=bool)
    region_mask = np.zeros((SUBTITLE_HIGHRES_H, SUBTITLE_HIGHRES_W), dtype=np.uint8)
    state = SUBTITLE_HIGHRES_STATE_BY_FRAME[frame_id - 1]
    if state == "empty":
        return glyph_rgb, glyph_mask, outline_mask, region_mask

    line = _render_authored_text(
        SUBTITLE_HIGHRES_TEXT[state], "cjk", SUBTITLE_HIGHRES_STROKE_PX)
    block_glyph = np.pad(
        line, SUBTITLE_HIGHRES_OUTLINE_PX,
        mode="constant", constant_values=False)
    block_outline = (
        _dilate_binary(block_glyph, SUBTITLE_HIGHRES_OUTLINE_PX) & ~block_glyph)
    block_height, block_width = block_glyph.shape
    x = (SUBTITLE_HIGHRES_W - block_width) // 2
    y = SUBTITLE_HIGHRES_H - 88 - block_height
    ys = slice(y, y + block_height)
    xs = slice(x, x + block_width)
    glyph_mask[ys, xs] = block_glyph
    outline_mask[ys, xs] = block_outline
    glyph_rgb[ys, xs][block_glyph] = (248, 248, 244)

    # A loose rectangle remains much larger than the two/three-pixel authored edges. Empty frames
    # intentionally keep an all-zero sidecar rather than inventing a positional prior.
    region_pad_x = 36
    region_pad_y = 24
    region_mask[
        y - region_pad_y:y + block_height + region_pad_y,
        x - region_pad_x:x + block_width + region_pad_x,
    ] = 255
    return glyph_rgb, glyph_mask, outline_mask, region_mask


def _warm_interior_background(frame_id):
    """A low-frequency warm interior, deliberately distinct from the dusk-city scene."""
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[..., 0] = 154 + yy * 34 // H
    frame[..., 1] = 112 + yy * 28 // H
    frame[..., 2] = 78 + yy * 22 // H

    # A night window, curtains, table, lamp, and moving silhouette form a movie-like second shot.
    _fill_rect(frame, 58, 48, 512, 302, (103, 151, 185))
    _fill_rect(frame, 52, 42, 518, 50, (91, 56, 39))
    _fill_rect(frame, 52, 300, 518, 310, (91, 56, 39))
    _fill_rect(frame, 52, 42, 62, 310, (91, 56, 39))
    _fill_rect(frame, 508, 42, 518, 310, (91, 56, 39))
    _fill_rect(frame, 281, 48, 289, 302, (91, 56, 39))
    for light_index in range(16):
        light_x = 72 + (light_index * 71 + frame_id * 3) % 420
        light_y = 92 + (light_index * 37) % 174
        _fill_rect(frame, light_x, light_y, light_x + 4, light_y + 5, (226, 218, 173))

    _fill_rect(frame, 0, 349, W, H, (105, 73, 55))
    _fill_rect(frame, 420, 315, 805, 364, (134, 88, 56))
    _fill_rect(frame, 448, 364, 466, H, (82, 54, 43))
    _fill_rect(frame, 752, 364, 770, H, (82, 54, 43))
    lamp_x = 696
    lamp = (xx - lamp_x) ** 2 + (yy - 189) ** 2 <= 49 ** 2
    frame[lamp] = (197, 154, 94)
    _fill_rect(frame, lamp_x - 4, 234, lamp_x + 5, 322, (64, 42, 34))

    person_x = 182 + frame_id * 2
    head = (xx - person_x) ** 2 + (yy - 270) ** 2 <= 17 ** 2
    frame[head] = (29, 24, 25)
    _fill_rect(frame, person_x - 15, 287, person_x + 16, 391, (27, 23, 25))
    texture = ((xx * 2 + yy * 3 + frame_id * 5) % 5).astype(np.uint16)
    return np.minimum(frame.astype(np.uint16) + texture[..., None], 220).astype(np.uint8)


def _highres_movie_background(frame_id, scene):
    if scene == "dusk-city":
        low_resolution = _movie_background(frame_id, 0)
    elif scene == "warm-interior":
        low_resolution = _warm_interior_background(frame_id)
    else:
        raise ValueError(f"unknown high-resolution subtitle scene {scene!r}")
    # Exact integer enlargement keeps regeneration independent of an image-library resampler.
    enlarged = np.repeat(np.repeat(low_resolution, 3, axis=0), 3, axis=1)
    return enlarged[:, 1:1 + SUBTITLE_HIGHRES_W]


def subtitle_cjk_highres_transitions():
    out, region_out = _prepare_subtitle_output(SUBTITLE_HIGHRES_CLIP)
    for frame_id in range(1, N + 1):
        scene = SUBTITLE_HIGHRES_SCENE_BY_FRAME[frame_id - 1]
        frame = _highres_movie_background(frame_id, scene)
        glyph_rgb, glyph_mask, outline_mask, region_mask = (
            _highres_subtitle_layers(frame_id))
        frame[outline_mask] = (7, 8, 11)
        frame[glyph_mask] = glyph_rgb[glyph_mask]
        Image.fromarray(frame).save(
            os.path.join(out, f"frame_{frame_id:05d}.png"), compress_level=9)
        Image.fromarray(region_mask, mode="L").save(
            os.path.join(region_out, f"frame_{frame_id:05d}.png"), compress_level=9)


def flat_page():
    rng = np.random.default_rng(7)
    page = np.full((H, W, 3), 245, np.uint8)
    page[:28] = (60, 63, 68)  # window title bar
    y = 46
    while y < H - 20:  # text-like dark lines of varying length
        line_w = int(rng.uniform(0.35, 0.92) * (W - 80))
        page[y:y + 8, 40:40 + line_w] = int(rng.uniform(30, 90))
        y += 8 + int(rng.uniform(6, 14))
    for i in range(N):  # static: every frame identical
        save("flat_page", i, page)
        save_gt_depth("flat_page", i, np.full((H, W), 0.5, np.float32))


def fast_motion():
    rng = np.random.default_rng(11)
    # Textured background (soft large-scale noise) and a distinct textured foreground block.
    bg = rng.uniform(60, 190, (H // 8, W // 8, 3))
    bg = np.asarray(Image.fromarray(bg.astype(np.uint8)).resize((W, H), Image.BILINEAR))
    fw, fh = 120, 200
    fg = rng.uniform(40, 255, (fh // 4, fw // 4, 3))
    fg = np.asarray(Image.fromarray(fg.astype(np.uint8)).resize((fw, fh), Image.BILINEAR))
    y0 = (H - fh) // 2
    for i in range(N):
        x0 = 40 + i * 30  # known speed: 30 px/frame
        fr = bg.copy()
        fr[y0:y0 + fh, x0:x0 + fw] = fg
        save("fast_motion", i, fr)
        gt = np.full((H, W), 0.25, np.float32)
        gt[y0:y0 + fh, x0:x0 + fw] = 0.75
        save_gt_depth("fast_motion", i, gt)


def scene_cut():
    # Splice two committed clips of identical size; the cut lands at frame N/2.
    os.makedirs(os.path.join(CLIPS, "scene_cut"), exist_ok=True)
    for i in range(N):
        src_clip = "c841" if i < N // 2 else "c647"
        src = os.path.join(CLIPS, src_clip, f"frame_{i + 1:05d}.jpg")
        Image.open(src).save(os.path.join(CLIPS, "scene_cut", f"frame_{i + 1:05d}.jpg"), quality=90)


def flat_transition():
    """Give the slow-max range reference a textured history before entering flat content."""
    out = os.path.join(CLIPS, "flat_transition")
    gt_out = os.path.join(out, "gt_depth")
    os.makedirs(gt_out, exist_ok=True)
    for i in range(N):
        src_clip = "fast_motion" if i < N // 2 else "flat_page"
        src_i = i + 1 if i < N // 2 else i - N // 2 + 1
        name = f"frame_{src_i:05d}"
        shutil.copyfile(os.path.join(CLIPS, src_clip, name + ".jpg"),
                        os.path.join(out, f"frame_{i + 1:05d}.jpg"))
        shutil.copyfile(os.path.join(CLIPS, src_clip, "gt_depth", name + ".png"),
                        os.path.join(gt_out, f"frame_{i + 1:05d}.png"))


def exposure_flash_strobe():
    """Static geometry with an exact positive global gain applied equally to every RGB channel."""
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.empty((H, W, 3), np.uint8)
    base[..., 0] = 34 + (xx * 43 // W)
    base[..., 1] = 43 + (yy * 38 // H)
    base[..., 2] = 58 + ((xx + yy) * 31 // (W + H))

    # Fixed geometric layers and texture give the depth model real edges while staying below 155,
    # so even the 165% flash has no clipping. Only the global gain changes between frames.
    mid = (xx >= W // 5) & (xx < 4 * W // 5) & (yy >= H // 5) & (yy < 4 * H // 5)
    checker = ((xx // 32 + yy // 32) & 1).astype(bool)
    base[mid & checker] = (102, 91, 77)
    base[mid & ~checker] = (116, 104, 88)
    circle = (xx - 5 * W // 8) ** 2 + (yy - H // 2) ** 2 <= (H // 6) ** 2
    base[circle] = (148, 121, 92)
    near = (xx >= W // 12) & (xx < W // 4) & (yy >= H // 3) & (yy < 5 * H // 6)
    base[near] = np.where(checker[near][:, None], (139, 73, 62), (124, 61, 53))

    out = os.path.join(CLIPS, "exposure_flash_strobe")
    os.makedirs(out, exist_ok=True)
    for i, gain_percent in enumerate(EXPOSURE_GAINS_PERCENT):
        # Integer half-up quantization is recorded verbatim in metadata and is reproducible across
        # NumPy/Pillow versions. uint32 prevents overflow before the final uint8 clamp.
        scaled = np.minimum(
            (base.astype(np.uint32) * gain_percent + 50) // 100, 255).astype(np.uint8)
        Image.fromarray(scaled).save(os.path.join(out, f"frame_{i + 1:05d}.png"))


def sustained_motion_scene_cut():
    """Latch on one scene replacement, sustain broad motion, then replace the scene again."""
    with Image.open(os.path.join(CLIPS, "c841", "frame_00001.jpg")) as image:
        prelude = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(os.path.join(CLIPS, "c647", "frame_00013.jpg")) as image:
        moving = np.asarray(image.convert("RGB"), dtype=np.uint8)

    out = os.path.join(CLIPS, "sustained_motion_scene_cut")
    os.makedirs(out, exist_ok=True)
    for frame_id in range(1, SUSTAINED_FRAME_COUNT + 1):
        if frame_id < SUSTAINED_SETUP_CUT_FRAME:
            frame = prelude
        elif frame_id < SUSTAINED_TRUE_CUT_FRAME:
            shift = SUSTAINED_ROLL_PX if (
                frame_id - SUSTAINED_SETUP_CUT_FRAME) % 2 else 0
            frame = np.roll(moving, shift, axis=1)
        else:
            frame = prelude
        Image.fromarray(frame).save(
            os.path.join(out, f"frame_{frame_id:05d}.png"))


def _structureless_history_bridge(clip, uniform_rgb):
    """Bridge structureless intervals without confusing a clipped flash with a new scene."""
    scene_a_clip, scene_a_frame = BRIDGE_SCENE_A_BY_CLIP[clip]
    scene_b_clip, scene_b_frame = BRIDGE_SCENE_B
    with Image.open(os.path.join(
            CLIPS, scene_a_clip, f"frame_{scene_a_frame:05d}.jpg")) as image:
        scene_a = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(os.path.join(
            CLIPS, scene_b_clip, f"frame_{scene_b_frame:05d}.jpg")) as image:
        scene_b = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if scene_a.shape != scene_b.shape:
        raise ValueError("structureless bridge source anchors must have identical dimensions")
    uniform = np.empty_like(scene_a)
    uniform[...] = uniform_rgb

    out = os.path.join(CLIPS, clip)
    os.makedirs(out, exist_ok=True)
    for frame_id in range(1, BRIDGE_FRAME_COUNT + 1):
        if frame_id == BRIDGE_FLASH_FRAME:
            frame = uniform
        elif BRIDGE_SLATE_FRAME <= frame_id < BRIDGE_NEW_SCENE_FRAME:
            frame = uniform
        elif frame_id < BRIDGE_NEW_SCENE_FRAME:
            frame = scene_a
        else:
            frame = scene_b
        Image.fromarray(frame).save(
            os.path.join(out, f"frame_{frame_id:05d}.png"))


def structureless_history_bridge():
    _structureless_history_bridge(
        "structureless_history_bridge",
        BRIDGE_UNIFORM_RGB_BY_CLIP["structureless_history_bridge"])


def structureless_white_history_bridge():
    _structureless_history_bridge(
        "structureless_white_history_bridge",
        BRIDGE_UNIFORM_RGB_BY_CLIP["structureless_white_history_bridge"])


def clip_metadata(clip):
    if clip == "flat_page":
        return {"expected_flat": True, "gt_depth_kind": "disparity"}
    if clip == "fast_motion":
        return {"gt_depth_kind": "disparity"}
    if clip == "flat_transition":
        return {
            "gt_depth_kind": "disparity",
            "shot_state_contract": {
                "kind": "hard-cut",
                "monitor_from_frame": SHOT_STATE_MONITOR_FROM_FRAME,
                "expected_pulse_frames": [SHOT_CUT_FRAME],
            },
        }
    if clip == "scene_cut":
        return {
            "shot_state_contract": {
                "kind": "hard-cut",
                "monitor_from_frame": SHOT_STATE_MONITOR_FROM_FRAME,
                "expected_pulse_frames": [SHOT_CUT_FRAME],
            },
        }
    if clip == "exposure_flash_strobe":
        return {
            "evaluation_role": "conformance-only",
            "shot_state_contract": {
                "kind": "exposure-only",
                "monitor_from_frame": SHOT_STATE_MONITOR_FROM_FRAME,
                "expected_pulse_frames": [],
                "stable_from_frame": EXPOSURE_STABLE_FROM_FRAME,
                "base_frame": 1,
                "gain_percent_by_frame": list(EXPOSURE_GAINS_PERCENT),
                "rgb_transform": "min(255, (base_rgb * gain_percent + 50) // 100)",
            },
        }
    if clip == "sustained_motion_scene_cut":
        return {
            "evaluation_role": "conformance-only",
            "shot_state_contract": {
                "kind": "latched-motion-hard-cut",
                "monitor_from_frame": SHOT_STATE_MONITOR_FROM_FRAME,
                "expected_pulse_frames": [
                    SUSTAINED_SETUP_CUT_FRAME, SUSTAINED_ESCAPE_PULSE_FRAME],
                "setup_pulse_frame": SUSTAINED_SETUP_CUT_FRAME,
                "persistent_motion_frames": [
                    SUSTAINED_SETUP_CUT_FRAME, SUSTAINED_TRUE_CUT_FRAME],
                "escape_candidate_frame": SUSTAINED_TRUE_CUT_FRAME,
                "escape_pulse_frame": SUSTAINED_ESCAPE_PULSE_FRAME,
                "source_base_frame_by_frame": (
                    [1] * (SUSTAINED_SETUP_CUT_FRAME - 1) +
                    [SUSTAINED_SETUP_CUT_FRAME] * (
                        SUSTAINED_TRUE_CUT_FRAME - SUSTAINED_SETUP_CUT_FRAME) +
                    [1] * (SUSTAINED_FRAME_COUNT - SUSTAINED_TRUE_CUT_FRAME + 1)
                ),
                "horizontal_roll_px_by_frame": (
                    [0] * (SUSTAINED_SETUP_CUT_FRAME - 1) +
                    [
                        SUSTAINED_ROLL_PX if index % 2 else 0
                        for index in range(
                            SUSTAINED_TRUE_CUT_FRAME - SUSTAINED_SETUP_CUT_FRAME)
                    ] +
                    [0] * (SUSTAINED_FRAME_COUNT - SUSTAINED_TRUE_CUT_FRAME + 1)
                ),
                "rgb_transform": "np.roll(base_rgb, horizontal_shift_px, axis=1)",
            },
        }
    if clip in BRIDGE_UNIFORM_RGB_BY_CLIP:
        return {
            "evaluation_role": "conformance-only",
            "shot_state_contract": {
                "kind": "structureless-history-bridge",
                "monitor_from_frame": SHOT_STATE_MONITOR_FROM_FRAME,
                # The V2 cut resolver accepts a post-bridge structured return through its
                # two-update geometry confirmation (anti-transient by design), so the
                # new-scene pulse lands one frame after the content cut.
                "expected_pulse_frames": [
                    BRIDGE_PERSISTENT_FRAME,
                    BRIDGE_NEW_SCENE_FRAME + 1,
                ],
                "flash_frame": BRIDGE_FLASH_FRAME,
                "flash_return_frame": BRIDGE_FLASH_RETURN_FRAME,
                "slate_frame": BRIDGE_SLATE_FRAME,
                "persistent_frame": BRIDGE_PERSISTENT_FRAME,
                "new_scene_frame": BRIDGE_NEW_SCENE_FRAME,
                "uniform_rgb": list(BRIDGE_UNIFORM_RGB_BY_CLIP[clip]),
                "rgb_transform": (
                    "scene_a; one uniform flash; scene_a; uniform slate; scene_b"),
            },
        }
    if clip in SUBTITLE_STANDARD_CLIPS:
        return {
            "required_gt_subtitle_region": True,
            "subtitle_target_disparity_pct": 0.0,
            "subtitle_cue_start_frames": list(SUBTITLE_CUE_START_FRAMES),
            "source_artifacts": [
                "Authored burned-in subtitle pixels; region truth is intentionally loose rather "
                "than glyph-tight."
            ],
        }
    if clip == SUBTITLE_HIGHRES_CLIP:
        return {
            "required_gt_subtitle_region": True,
            "subtitle_target_disparity_pct": 0.0,
            "subtitle_transition_contract": {
                "kind": "highres-empty-appear-replace-disappear-hard-cut",
                "source_size_px": [SUBTITLE_HIGHRES_W, SUBTITLE_HIGHRES_H],
                "detector_target_size_px": [
                    SUBTITLE_DETECTOR_TARGET_W, SUBTITLE_DETECTOR_TARGET_H],
                "authored_glyph_stroke_width_px": SUBTITLE_HIGHRES_STROKE_PX,
                "authored_outline_radius_px": SUBTITLE_HIGHRES_OUTLINE_PX,
                "empty_frame_ranges": [[1, 4], [13, 16]],
                "appear_frames": [5, 17],
                "subtitle_only_replacement_frames": [9],
                "disappear_frames": [13],
                "broad_scene_cut_frames": [21],
                "overlay_replacement_at_scene_cut_frames": [21],
                "subtitle_state_by_frame": list(SUBTITLE_HIGHRES_STATE_BY_FRAME),
                "scene_state_by_frame": list(SUBTITLE_HIGHRES_SCENE_BY_FRAME),
            },
            "shot_state_contract": {
                "kind": "hard-cut",
                "monitor_from_frame": SHOT_STATE_MONITOR_FROM_FRAME,
                "expected_pulse_frames": [21],
            },
            "source_artifacts": [
                "Authored three-pixel CJK strokes with a two-pixel dark outline; empty frames "
                "have exact zero subtitle-region masks."
            ],
        }
    raise ValueError(f"unknown synthetic clip {clip}")


GENERATORS = {
    "flat_page": flat_page,
    "fast_motion": fast_motion,
    "scene_cut": scene_cut,
    "flat_transition": flat_transition,
    "exposure_flash_strobe": exposure_flash_strobe,
    "sustained_motion_scene_cut": sustained_motion_scene_cut,
    "structureless_history_bridge": structureless_history_bridge,
    "structureless_white_history_bridge": structureless_white_history_bridge,
    "subtitle_cjk_dense": subtitle_cjk_dense,
    "subtitle_bilingual_tall_stack": subtitle_bilingual_tall_stack,
    "subtitle_top_bottom_disjoint": subtitle_top_bottom_disjoint,
    "subtitle_cjk_highres_transitions": subtitle_cjk_highres_transitions,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "clips", nargs="*", choices=sorted(GENERATORS),
        help="specific synthetic clips to regenerate (default: all)")
    args = parser.parse_args()
    selected = args.clips or list(GENERATORS)
    for clip in selected:
        GENERATORS[clip]()
        write_meta(clip, **clip_metadata(clip))
        cdir = os.path.join(CLIPS, clip)
        n = len([f for f in os.listdir(cdir) if f.startswith("frame_")])
        print(f"{clip}: {n} frames")


if __name__ == "__main__":
    main()
