#!/usr/bin/env python3
"""
make_synth_clips - generate the failure-mode clips the movie recordings don't cover
(docs/sbs-benchmark-plan.md clip table). Deterministic (fixed seed) and synthetic/spliced, so
they add no licensing surface. The generated frames are COMMITTED (clips/); rerun this only when
changing a clip's design, and regenerate baselines in the same commit.

  flat_page    a static desktop/document page: the depth model should output near-flat depth --
               measures hallucinated depth (depth_spread/pop) + normalization amplification (A3)
               and pipeline shimmer on static input (flicker floor).
  fast_motion  a textured block crossing a textured background at a KNOWN 30 px/frame --
               the async-depth ghost scenario (and the anchor for a future ghost metric).
  scene_cut    a hard cut spliced from two committed clips (c841 bright/calm -> c647 dark/crowd)
               -- depth-normalization swim across cuts (A1 snap validation); expect the swim /
               flicker worst frame AT the cut.
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, "clips")
W, H, N = 854, 480, 24


def save(clip, i, arr):
    d = os.path.join(CLIPS, clip)
    os.makedirs(d, exist_ok=True)
    Image.fromarray(arr).save(os.path.join(d, f"frame_{i + 1:05d}.jpg"), quality=90)


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


def scene_cut():
    # Splice two committed clips of identical size; the cut lands at frame N/2.
    for i in range(N):
        src_clip = "c841" if i < N // 2 else "c647"
        src = os.path.join(CLIPS, src_clip, f"frame_{i + 1:05d}.jpg")
        Image.open(src).save(os.path.join(CLIPS, "scene_cut", f"frame_{i + 1:05d}.jpg"), quality=90)


if __name__ == "__main__":
    flat_page()
    fast_motion()
    os.makedirs(os.path.join(CLIPS, "scene_cut"), exist_ok=True)
    scene_cut()
    for c in ("flat_page", "fast_motion", "scene_cut"):
        n = len(os.listdir(os.path.join(CLIPS, c)))
        print(f"{c}: {n} frames")
