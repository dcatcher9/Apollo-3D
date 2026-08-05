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
