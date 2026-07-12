#!/usr/bin/env python3
"""Audit whether an A/B depth processor clips or crushes the saved pre-warp depth maps.

Usage: audit_depth_transform.py CONTROL_RUN TREATMENT_RUN [--out FILE]

The run directories must be compatible `run_eval.py` outputs. Every matched depth PNG is measured
in its native resolution. The audit reports p95-p5 depth spread, endpoint saturation, treatment /
control spread ratios, corresponding SBS stereo spread, and available GT depth metrics.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402


def frame_stats(path):
    a = sbsbench.load_depth(path)
    q = np.percentile(a, [1, 5, 50, 95, 99])
    eps = 1.5 / 65535.0
    return {
        "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]),
        "p95": float(q[3]), "p99": float(q[4]),
        "spread_p95_p05": float(q[3] - q[1]),
        "saturated_low_pct": float(np.mean(a <= eps) * 100.0),
        "saturated_high_pct": float(np.mean(a >= 1.0 - eps) * 100.0),
    }


def mean_stats(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("treatment")
    ap.add_argument("--out")
    args = ap.parse_args()
    ctrl_result = json.load(open(os.path.join(args.control, "results.json"), encoding="utf-8"))
    treat_result = json.load(open(os.path.join(args.treatment, "results.json"), encoding="utf-8"))
    if ctrl_result["meta"].get("clip_set_sha1") != treat_result["meta"].get("clip_set_sha1"):
        raise SystemExit("incompatible clip identities")
    clips = sorted(set(ctrl_result["clips"]) & set(treat_result["clips"]))
    output = {"schema": 1, "control": os.path.abspath(args.control),
              "treatment": os.path.abspath(args.treatment), "clips": {}}
    all_ratios = []
    for clip in clips:
        cfiles = {os.path.basename(p): p for p in glob.glob(os.path.join(args.control, clip, "depth_*.png"))}
        tfiles = {os.path.basename(p): p for p in glob.glob(os.path.join(args.treatment, clip, "depth_*.png"))}
        names = sorted(set(cfiles) & set(tfiles))
        if not names:
            continue
        crows, trows, ratios = [], [], []
        for name in names:
            cs, ts = frame_stats(cfiles[name]), frame_stats(tfiles[name])
            crows.append(cs)
            trows.append(ts)
            if cs["spread_p95_p05"] > 1e-6:
                ratios.append(ts["spread_p95_p05"] / cs["spread_p95_p05"])
        ca = ctrl_result["clips"][clip]["aggregate"]
        ta = treat_result["clips"][clip]["aggregate"]
        entry = {
            "frames": len(names), "control_depth": mean_stats(crows),
            "treatment_depth": mean_stats(trows),
            "spread_ratio_mean": float(np.mean(ratios)) if ratios else None,
            "spread_ratio_min": float(np.min(ratios)) if ratios else None,
            "frames_below_90pct_spread": int(sum(r < 0.9 for r in ratios)),
            "pop_spread_pct": {"control": ca.get("pop_spread_pct"),
                               "treatment": ta.get("pop_spread_pct")},
            "gt_depth_si_rmse": {"control": ca.get("depth_gt_si_rmse"),
                                 "treatment": ta.get("depth_gt_si_rmse")},
            "gt_depth_edge_f1": {"control": ca.get("depth_gt_edge_f1"),
                                 "treatment": ta.get("depth_gt_edge_f1")},
        }
        output["clips"][clip] = entry
        all_ratios.extend(ratios)
    output["summary"] = {
        "matched_clips": len(output["clips"]),
        "matched_frames": len(all_ratios),
        "spread_ratio_mean": float(np.mean(all_ratios)) if all_ratios else None,
        "spread_ratio_p05": float(np.percentile(all_ratios, 5)) if all_ratios else None,
        "spread_ratio_min": float(np.min(all_ratios)) if all_ratios else None,
        "frames_below_90pct_spread": int(sum(r < 0.9 for r in all_ratios)),
    }
    out = args.out or os.path.join(args.treatment, "depth_transform_audit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    print(json.dumps(output["summary"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
