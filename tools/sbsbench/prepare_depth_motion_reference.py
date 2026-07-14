#!/usr/bin/env python3
"""Prepare an offline classical-flow reference for motion-compensated held depth.

The source run must contain fresh depth for every frame. For each frame that a --depth-every 2
live path would hold, this tool warps the last fresh depth into current-frame coordinates using
the evaluator's deterministic tile phase-correlation flow. The harness consumes these PNGs with
--depth-override-root while retaining the real subject state and production warp shaders.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

import sbsbench
import run_eval


def fail(message):
    raise SystemExit("prepare_depth_motion_reference: " + message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh-run", required=True,
                    help="current-once run_eval directory containing depth_*.png")
    ap.add_argument("--clips-root", required=True,
                    help="matching source clip suite")
    ap.add_argument("--out", required=True,
                    help="output root; one clip directory is created beneath it")
    ap.add_argument("--depth-every", type=int, default=2, choices=(2,),
                    help="prototype currently supports reuse-2 only")
    args = ap.parse_args()
    fresh_run = os.path.abspath(args.fresh_run)
    clips_root = os.path.abspath(args.clips_root)
    out_root = os.path.abspath(args.out)
    results_path = os.path.join(fresh_run, "results.json")
    if not os.path.exists(results_path):
        fail("fresh run lacks results.json")
    results = json.load(open(results_path, encoding="utf-8"))
    if results.get("meta", {}).get("depth_step") != "current-once":
        fail("source run must have depth_step=current-once")
    if results.get("meta", {}).get("depth_compensation", "none") != "none":
        fail("source run must have depth_compensation=none")

    manifest = {"schema": 3, "method": "classical-tile-phase-flow", "frame_policy": "held",
                "depth_every": args.depth_every,
                "fresh_run": fresh_run, "clips": {}}
    for clip in sorted(results.get("clips", {})):
        source_dir = os.path.join(clips_root, clip)
        run_dir = os.path.join(fresh_run, clip)
        src = sbsbench.indexed_files(os.path.join(source_dir, "frame_*.*"), "frame_")
        src = {i: p for i, p in src.items()
               if p.lower().endswith((".png", ".jpg", ".jpeg"))}
        depths = sbsbench.indexed_files(os.path.join(run_dir, "depth_*.png"), "depth_")
        if not src or set(src) != set(depths):
            fail(f"{clip}: source/fresh-depth identities differ")
        recorded_sha = results.get("meta", {}).get("clip_set_sha1", {}).get(clip)
        current_sha = run_eval.sha1_dir(source_dir)
        if not recorded_sha or recorded_sha != current_sha:
            fail(f"{clip}: clips root differs from the fresh run: "
                 f"recorded={recorded_sha}, current={current_sha}")
        ids = sorted(src)
        clip_out = os.path.join(out_root, clip)
        os.makedirs(clip_out, exist_ok=True)
        for stale in sbsbench.indexed_files(
                os.path.join(clip_out, "depth_*.png"), "depth_").values():
            os.remove(stale)
        coverages = []
        written = 0
        override_frame_ids = []
        for position, frame_id in enumerate(ids):
            if position % args.depth_every == 0:
                continue
            fresh_id = ids[position - (position % args.depth_every)]
            previous_depth = sbsbench.load_depth(depths[fresh_id])
            previous_src = sbsbench.load_gray(src[fresh_id])
            current_src = sbsbench.load_gray(src[frame_id])
            height, width = previous_depth.shape
            u, v, flow_valid = sbsbench.dense_source_flow(
                previous_src, current_src, width, height)
            warped_depth, depth_valid = sbsbench.warp_previous_nearest_with_flow(
                previous_depth, u, v)
            previous_small = sbsbench.resize_to(previous_src, width, height)
            current_small = sbsbench.resize_to(current_src, width, height)
            warped_source, source_valid = sbsbench.warp_previous_with_flow(
                previous_small, u, v)
            reliable = flow_valid & depth_valid & source_valid
            reliable &= np.abs(current_small - warped_source) <= 20.0 / 255.0
            compensated = np.where(reliable, warped_depth, previous_depth)
            output = np.round(np.clip(compensated, 0.0, 1.0) * 65535.0).astype(np.uint16)
            source_stem = os.path.splitext(os.path.basename(src[frame_id]))[0]
            source_id = source_stem.rsplit("_", 1)[-1]
            Image.fromarray(output).save(
                os.path.join(clip_out, f"depth_{source_id}.png"))
            coverages.append(float(reliable.mean()))
            written += 1
            override_frame_ids.append(frame_id)
        manifest["clips"][clip] = {
            "override_frames": written,
            "override_frame_ids": override_frame_ids,
            "clip_sha1": current_sha,
            "mean_reliable_coverage": float(np.mean(coverages)) if coverages else 0.0,
        }
        print(f"{clip}: {written} held frames, "
              f"{manifest['clips'][clip]['mean_reliable_coverage'] * 100.0:.1f}% reliable")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote", out_root)


if __name__ == "__main__":
    main()
