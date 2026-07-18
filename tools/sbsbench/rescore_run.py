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
    if value in ("none", "external-reference", "external-treatment", "nvof-1x1"):
        return value
    extra_args = meta.get("extra_args") or []
    if "--depth-override-root" in extra_args:
        return ("external-treatment" if "--depth-override-all" in extra_args else
                "external-reference")
    if "--depth-motion-compensation" in extra_args:
        return "nvof-1x1"
    return "none"


def refresh_contract_metadata(data):
    """Refresh only the Python metric contract; harness/evaluator schema is immutable."""
    data["meta"]["metric_sha256"] = run_eval.metric_contract_sha()


def validate_rescore_provenance(data):
    """Only current-schema, explicitly comparison-only artifacts are safe to rescore."""
    meta = data.get("meta", {})
    if meta.get("run_kind") != "comparison-only":
        raise SystemExit("refusing to rescore a run without comparison-only provenance")
    if meta.get("eval_schema") != run_eval.EVAL_SCHEMA:
        raise SystemExit(
            f"refusing evaluator schema {meta.get('eval_schema')!r}; rerun with current schema "
            f"{run_eval.EVAL_SCHEMA}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="sbs_eval run containing results.json and per-clip artifacts")
    ap.add_argument("--clips-root", help="source clips (default: recorded run clips_root)")
    ap.add_argument("--in-place", action="store_true",
                    help="replace results.json atomically (default writes results.rescored.json)")
    args = ap.parse_args()
    result_path = os.path.join(args.run_dir, "results.json")
    data = json.load(open(result_path, encoding="utf-8"))
    validate_rescore_provenance(data)
    clips_root = os.path.abspath(args.clips_root or data.get("meta", {}).get("clips_root")
                                 or os.path.join(SCRIPT_DIR, "clips"))
    current_clip_hashes = {
        clip: run_eval.sha1_dir(os.path.join(clips_root, clip)) for clip in data["clips"]}
    recorded_clip_hashes = data.get("meta", {}).get("clip_set_sha1", {})
    stale = {clip: (recorded_clip_hashes.get(clip), digest)
             for clip, digest in current_clip_hashes.items()
             if recorded_clip_hashes.get(clip) != digest}
    if stale:
        raise SystemExit(f"refusing changed source/GT evidence: {stale}")
    thresholds = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8"))
    issues, hard_failures, evidence_failures = [], [], []
    for clip, entry in data["clips"].items():
        clip_dir = os.path.join(clips_root, clip)
        expected_flat = bool(entry.get("meta", {}).get("expected_flat"))
        measured = sbsbench.measure_sequence(
            os.path.join(args.run_dir, clip), clip_dir, expected_flat=expected_flat)
        if not measured:
            raise SystemExit(f"{clip}: no measurable SBS artifacts")
        rows, agg = measured
        entry.setdefault("meta", {})["source_frame_count"] = len(
            sbsbench.indexed_files(os.path.join(clip_dir, "frame_*.*"), "frame_"))
        worst, clip_issues, clip_hard_failures = run_eval.score_clip_gates(
            rows, agg, thresholds, entry.get("meta", {}))
        issues.extend({"clip": clip, **item} for item in clip_issues)
        hard_failures.extend({"clip": clip, **item} for item in clip_hard_failures)
        evidence_failures.extend(run_eval.primary_evidence_failures(
            agg, thresholds, clip, entry["meta"], worst=worst))
        evidence_failures.extend(run_eval.perf_evidence_failures(
            None, entry.get("perf_ms", {}), thresholds, clip))
        entry["aggregate"] = agg
        entry["worst_frame"] = worst

    data["issues"] = issues
    data["hard_failures"] = hard_failures
    data["evidence_failures"] = evidence_failures
    data["regressions"] = []
    data["verdict"] = ("hard_failures" if hard_failures else
                       "evidence_failures" if evidence_failures else "comparison_only")
    refresh_contract_metadata(data)
    depth_compensation = depth_compensation_from_meta(data.get("meta", {}))
    data["meta"]["depth_compensation"] = depth_compensation
    for entry in data["clips"].values():
        entry.setdefault("meta", {})["depth_compensation"] = depth_compensation
    data["meta"]["clip_set_sha1"] = current_clip_hashes
    data["meta"]["clips_root"] = clips_root
    out = result_path if args.in_place else os.path.join(args.run_dir, "results.rescored.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
