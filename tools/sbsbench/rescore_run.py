#!/usr/bin/env python3
"""Recompute metric JSON from existing SBS/depth/source artifacts without rerunning the GPU.

Only comparison-only runs are accepted: committed baseline verdicts must be produced by run_eval,
not rewritten after the fact. Artifact identities remain unchanged; the metric contract hash and
derived aggregates/issues/worst frames are refreshed to the current scoring code.
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import run_eval  # noqa: E402
import sbsbench  # noqa: E402


def depth_compensation_from_meta(meta):
    """Preserve or derive the explicit schema-13 depth-compensation contract."""
    value = meta.get("depth_compensation")
    if value in ("none", "external-reference", "nvof-1x1"):
        return value
    extra_args = meta.get("extra_args") or []
    if "--depth-override-root" in extra_args:
        return "external-reference"
    if "--depth-motion-compensation" in extra_args:
        return "nvof-1x1"
    return "none"


def refresh_contract_metadata(data):
    """Stamp the same metric contract that fresh run_eval artifacts use."""
    data["meta"]["metric_sha256"] = run_eval.metric_contract_sha()
    data["meta"]["eval_schema"] = run_eval.EVAL_SCHEMA


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="sbs_eval run containing results.json and per-clip artifacts")
    ap.add_argument("--clips-root", default=os.path.join(SCRIPT_DIR, "clips"))
    ap.add_argument("--in-place", action="store_true",
                    help="replace results.json atomically (default writes results.rescored.json)")
    args = ap.parse_args()
    result_path = os.path.join(args.run_dir, "results.json")
    data = json.load(open(result_path, encoding="utf-8"))
    if data.get("verdict") not in ("comparison_only", "hard_failures"):
        raise SystemExit("refusing to rescore a committed-baseline verdict; rerun run_eval instead")
    thresholds = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8"))
    issues, hard_failures = [], []
    for clip, entry in data["clips"].items():
        clip_dir = os.path.join(args.clips_root, clip)
        expected_flat = bool(entry.get("meta", {}).get("expected_flat"))
        measured = sbsbench.measure_sequence(
            os.path.join(args.run_dir, clip), clip_dir, expected_flat=expected_flat)
        if not measured:
            raise SystemExit(f"{clip}: no measurable SBS artifacts")
        rows, agg = measured
        worst, clip_issues, clip_hard_failures = run_eval.score_clip_gates(
            rows, agg, thresholds, entry.get("meta", {}))
        issues.extend({"clip": clip, **item} for item in clip_issues)
        hard_failures.extend({"clip": clip, **item} for item in clip_hard_failures)
        entry["aggregate"] = agg
        entry["worst_frame"] = worst

    data["issues"] = issues
    data["hard_failures"] = hard_failures
    data["regressions"] = []
    data["verdict"] = "hard_failures" if hard_failures else "comparison_only"
    refresh_contract_metadata(data)
    depth_compensation = depth_compensation_from_meta(data.get("meta", {}))
    data["meta"]["depth_compensation"] = depth_compensation
    for entry in data["clips"].values():
        entry.setdefault("meta", {})["depth_compensation"] = depth_compensation
    data["meta"]["clip_set_sha1"] = {
        clip: run_eval.sha1_dir(os.path.join(args.clips_root, clip)) for clip in data["clips"]}
    out = result_path if args.in_place else os.path.join(args.run_dir, "results.rescored.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
