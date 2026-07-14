#!/usr/bin/env python3
"""Build an exact-flow oracle for motion-compensated depth EMA.

The source run must be current-frame depth with temporal pixel EMA disabled
(``--ema 1 --ema-edge-change 0``). The previous current depth is point-sampled into the current
frame with the clip's exact optical-flow sidecar. Reliable history receives the configured EMA;
invalid/disoccluded pixels and inconsistent depth edges snap to current depth. The output is an
all-frame treatment for ``run_eval --extra --depth-override-root ... --depth-override-all``.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

import run_eval
import sbsbench


def fail(message):
    raise SystemExit("prepare_flow_ema_reference: " + message)


def depth_gradient(depth):
    """Production-shaped four-neighbour maximum gradient."""
    padded = np.pad(depth, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    return np.maximum.reduce((
        np.abs(center - padded[1:-1, :-2]),
        np.abs(center - padded[1:-1, 2:]),
        np.abs(center - padded[:-2, 1:-1]),
        np.abs(center - padded[2:, 1:-1]),
    ))


def flow_aware_ema(current, previous_filtered, previous_source, current_source,
                   reference_flow, reference_valid, ema_alpha, edge_change,
                   edge_gradient, edge_strength):
    """Reproject filtered history and apply production-equivalent edge-selective EMA."""
    height, width = current.shape
    u, v, flow_valid = sbsbench.resize_forward_flow_to_current(
        reference_flow, reference_valid, width, height)
    warped_depth, depth_valid = sbsbench.warp_previous_nearest_with_flow(
        previous_filtered, u, v)
    previous_small = sbsbench.resize_to(previous_source, width, height)
    current_small = sbsbench.resize_to(current_source, width, height)
    warped_source, source_valid = sbsbench.warp_previous_with_flow(previous_small, u, v)
    reliable = flow_valid & depth_valid & source_valid
    reliable &= np.abs(current_small - warped_source) <= 10.0 / 255.0

    change = np.abs(current - warped_depth)
    edge = np.maximum(depth_gradient(current), depth_gradient(warped_depth))
    moving_edge = (change >= edge_change) & (edge >= edge_gradient)
    alpha = np.full(current.shape, ema_alpha, np.float32)
    alpha[moving_edge] += (1.0 - alpha[moving_edge]) * edge_strength
    alpha[~reliable] = 1.0
    filtered = warped_depth + alpha * (current - warped_depth)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32), reliable, moving_edge


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh-run", required=True,
                    help="current-once run generated with --ema 1 --ema-edge-change 0")
    ap.add_argument("--clips-root", required=True,
                    help="matching suite with exact gt_flow sidecars")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", type=float, default=0.5)
    ap.add_argument("--edge-change", type=float, default=0.05)
    ap.add_argument("--edge-gradient", type=float, default=0.02)
    ap.add_argument("--edge-strength", type=float, default=0.25)
    ap.add_argument("--recursive-history", action="store_true",
                    help="feed filtered history back recursively; default uses one-frame history")
    args = ap.parse_args()
    for name, value in (("ema", args.ema), ("edge-change", args.edge_change),
                        ("edge-gradient", args.edge_gradient),
                        ("edge-strength", args.edge_strength)):
        if not 0.0 <= value <= 1.0:
            fail(f"--{name} must be between 0 and 1")

    fresh_run = os.path.abspath(args.fresh_run)
    clips_root = os.path.abspath(args.clips_root)
    out_root = os.path.abspath(args.out)
    results_path = os.path.join(fresh_run, "results.json")
    if not os.path.exists(results_path):
        fail("fresh run lacks results.json")
    results = json.load(open(results_path, encoding="utf-8"))
    meta = results.get("meta", {})
    if meta.get("depth_step") != "current-once" or meta.get("depth_compensation") != "none":
        fail("source run must be uncompensated current-once depth")

    manifest = {
        "schema": 3,
        "method": "flow-aware-ema-oracle",
        "frame_policy": "all",
        "depth_every": 1,
        "fresh_run": fresh_run,
        "ema_alpha": args.ema,
        "edge_change": args.edge_change,
        "edge_gradient": args.edge_gradient,
        "edge_strength": args.edge_strength,
        "recursive_history": args.recursive_history,
        "clips": {},
    }
    for clip in sorted(results.get("clips", {})):
        source_dir = os.path.join(clips_root, clip)
        run_dir = os.path.join(fresh_run, clip)
        contract_path = os.path.join(run_dir, "contract.json")
        contract = json.load(open(contract_path, encoding="utf-8"))
        required_contract = {
            "schema": 14,
            "model": meta.get("model"),
            "profile": meta.get("profile"),
            "depth_step": "current-once",
            "depth_compensation": "none",
            "ema": 1.0,
            "ema_edge_change": 0.0,
        }
        mismatch = {key: (value, contract.get(key))
                    for key, value in required_contract.items()
                    if contract.get(key) != value}
        if mismatch:
            fail(f"{clip}: incompatible source contract: {mismatch}")
        sources = sbsbench.indexed_files(os.path.join(source_dir, "frame_*.*"), "frame_")
        sources = {i: p for i, p in sources.items()
                   if p.lower().endswith((".png", ".jpg", ".jpeg"))}
        depths = sbsbench.indexed_files(os.path.join(run_dir, "depth_*.png"), "depth_")
        flows = sbsbench.indexed_files(
            os.path.join(source_dir, "gt_flow", "frame_*.npz"), "frame_")
        ids = sorted(sources)
        if not ids or set(ids) != set(depths):
            fail(f"{clip}: source/fresh-depth identities differ")
        if set(flows) != set(ids[1:]):
            fail(f"{clip}: exact flow identities must match every frame after the first")
        clip_sha = run_eval.sha1_dir(source_dir)
        if meta.get("clip_set_sha1", {}).get(clip) != clip_sha:
            fail(f"{clip}: clips root differs from the source run")

        clip_out = os.path.join(out_root, clip)
        os.makedirs(clip_out, exist_ok=True)
        for stale in sbsbench.indexed_files(
                os.path.join(clip_out, "depth_*.png"), "depth_").values():
            os.remove(stale)
        previous_history = previous_source = None
        coverages = []
        edge_coverages = []
        for frame_id in ids:
            current = sbsbench.load_depth(depths[frame_id])
            current_source = sbsbench.load_gray(sources[frame_id])
            if previous_history is None:
                filtered = current
            else:
                with np.load(flows[frame_id], allow_pickle=False) as flow_data:
                    flow = np.asarray(flow_data["flow"], dtype=np.float32)
                    valid = (np.asarray(flow_data["valid"], dtype=bool)
                             if "valid" in flow_data else None)
                filtered, reliable, moving_edge = flow_aware_ema(
                    current, previous_history, previous_source, current_source,
                    flow, valid, args.ema, args.edge_change, args.edge_gradient,
                    args.edge_strength)
                coverages.append(float(reliable.mean()))
                edge_coverages.append(float(moving_edge.mean()))
            output = np.round(filtered * 65535.0).astype(np.uint16)
            source_id = os.path.splitext(os.path.basename(sources[frame_id]))[0].rsplit("_", 1)[-1]
            Image.fromarray(output).save(os.path.join(clip_out, f"depth_{source_id}.png"))
            previous_history = filtered if args.recursive_history else current
            previous_source = current_source

        manifest["clips"][clip] = {
            "override_frames": len(ids),
            "override_frame_ids": ids,
            "clip_sha1": clip_sha,
            "mean_reliable_coverage": float(np.mean(coverages)) if coverages else 1.0,
            "mean_moving_edge_coverage": float(np.mean(edge_coverages)) if edge_coverages else 0.0,
        }
        print(f"{clip}: {len(ids)} frames, "
              f"{manifest['clips'][clip]['mean_reliable_coverage'] * 100.0:.1f}% reliable")

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote", out_root)


if __name__ == "__main__":
    main()
