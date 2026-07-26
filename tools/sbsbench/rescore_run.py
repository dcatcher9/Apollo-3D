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
import eval_parallel  # noqa: E402


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
    """Refresh Python metric/label semantics; harness/evaluator schema is immutable."""
    data["meta"]["metric_sha256"] = run_eval.metric_contract_sha()
    data["meta"]["label_contract_sha256"] = run_eval.label_contract_sha()
    data["meta"]["metric_runtime"] = run_eval.metric_runtime_provenance()


def validate_rescore_provenance(data):
    """Only current-schema, explicitly comparison-only artifacts are safe to rescore."""
    meta = data.get("meta", {})
    if meta.get("run_kind") != "comparison-only":
        raise SystemExit("refusing to rescore a run without comparison-only provenance")
    if meta.get("eval_schema") != run_eval.EVAL_SCHEMA:
        raise SystemExit(
            f"refusing evaluator schema {meta.get('eval_schema')!r}; rerun with current schema "
            f"{run_eval.EVAL_SCHEMA}")


def authoritative_clip_meta(
        data, clip, clips_root, run_dir, source_sha1=None, artifact_sha256=None):
    """Return a fresh scoring context, translating authentication errors for the CLI."""
    try:
        return run_eval.authoritative_remeasurement_clip_meta(
            data, clip, clips_root, run_dir,
            source_sha1=source_sha1, artifact_sha256=artifact_sha256)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{clip}: refusing unauthenticated scoring metadata: {exc}") from exc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="sbs_eval run containing results.json and per-clip artifacts")
    ap.add_argument("--clips-root", help="source clips (default: recorded run clips_root)")
    ap.add_argument("--in-place", action="store_true",
                    help="replace results.json atomically (default writes results.rescored.json)")
    ap.add_argument(
        "--jobs", type=run_eval.scoring_jobs_arg,
        default=min(8, os.cpu_count() or 1, eval_parallel.CLIP_SCORING_MAX_WORKERS),
        help="parallel CPU clip-scoring jobs (default: up to 8)")
    args = ap.parse_args()
    result_path = os.path.join(args.run_dir, "results.json")
    with open(result_path, encoding="utf-8") as results_file:
        data = json.load(results_file)
    validate_rescore_provenance(data)
    clips_root = os.path.abspath(args.clips_root or data.get("meta", {}).get("clips_root")
                                 or os.path.join(SCRIPT_DIR, "clips"))
    current_source_digests = {
        clip: run_eval.source_evidence_digests(os.path.join(clips_root, clip))
        for clip in data["clips"]}
    current_clip_hashes = {
        clip: digests[0] for clip, digests in current_source_digests.items()}
    recorded_clip_hashes = data.get("meta", {}).get("clip_set_sha1", {})
    stale = {clip: (recorded_clip_hashes.get(clip), digest)
             for clip, digest in current_clip_hashes.items()
             if recorded_clip_hashes.get(clip) != digest}
    if stale:
        raise SystemExit(f"refusing changed source/GT evidence: {stale}")
    with open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8") as threshold_file:
        thresholds = json.load(threshold_file)
    data.setdefault("meta", {})["training_labels"] = run_eval.training_label_manifest(thresholds)
    scoring_jobs = eval_parallel.clip_scoring_worker_count(args.jobs, len(data["clips"]))
    data["meta"]["scoring_jobs"] = scoring_jobs
    try:
        data["meta"]["scoring_pixel_budget_mpx"] = (
            run_eval.configured_scoring_pixel_budget())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    issues, hard_failures, evidence_failures = [], [], []
    artifact_hashes = {}
    numeric_hashes = {}
    prepared = {}
    recorded_artifact_hashes = data.get("meta", {}).get("scored_artifact_sha256")
    if (not isinstance(recorded_artifact_hashes, dict) or
            set(recorded_artifact_hashes) != set(data["clips"])):
        raise SystemExit(
            "refusing rescore without a complete recorded scored-artifact hash contract")
    for clip, entry in data["clips"].items():
        clip_dir = os.path.join(clips_root, clip)
        actual_artifact_hash, numeric_hash = run_eval.scored_artifact_digests(
            os.path.join(args.run_dir, clip))
        if numeric_hash is None:
            raise SystemExit(f"{clip}: no numeric metric inputs")
        clip_meta = authoritative_clip_meta(
            data, clip, clips_root, args.run_dir,
            source_sha1=current_clip_hashes[clip],
            artifact_sha256=actual_artifact_hash)
        # Never merge the cache into this object: even a single stale expected_flat/GT flag can
        # alter applicability, and a forged source count can make incomplete labels look valid.
        entry["meta"] = clip_meta
        artifact_hash = clip_meta["scored_artifact_sha256"]
        artifact_hashes[clip] = artifact_hash
        if actual_artifact_hash != artifact_hash:
            raise SystemExit(
                f"{clip}: scored artifact digest changed during rescore authentication")
        numeric_hashes[clip] = numeric_hash
        try:
            perf = run_eval.load_perf_metrics(
                os.path.join(args.run_dir, clip, "sbs_perf.json"))
        except ValueError as exc:
            raise SystemExit(f"{clip}: {exc}") from exc
        prepared[clip] = {
            "entry": entry,
            "clip_dir": clip_dir,
            "clip_meta": clip_meta,
            "artifact_dir": os.path.join(args.run_dir, clip),
            "perf": perf,
        }

    measurement_jobs = [
        (clip, prepared[clip]["artifact_dir"], prepared[clip]["clip_dir"])
        for clip in data["clips"]
    ]
    try:
        measured_clips = dict(eval_parallel.measure_clip_sequences(
            measurement_jobs, jobs=scoring_jobs))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for clip in data["clips"]:
        source_after = run_eval.source_evidence_digests(prepared[clip]["clip_dir"])
        artifact_after = run_eval.scored_artifact_digests(prepared[clip]["artifact_dir"])
        if source_after != current_source_digests[clip]:
            raise SystemExit(f"{clip}: source evidence changed during rescoring")
        if artifact_after != (artifact_hashes[clip], numeric_hashes[clip]):
            raise SystemExit(f"{clip}: scored artifacts changed during rescoring")

    for clip in data["clips"]:
        entry = prepared[clip]["entry"]
        clip_meta = prepared[clip]["clip_meta"]
        perf = prepared[clip]["perf"]
        rows, agg = measured_clips[clip]
        agg = sbsbench.filter_aggregate_by_evidence(
            rows, agg, thresholds["metrics"], clip_meta)
        worst, clip_issues, clip_hard_failures = run_eval.score_clip_gates(
            rows, agg, thresholds, clip_meta)
        issues.extend({"clip": clip, **item} for item in clip_issues)
        hard_failures.extend({"clip": clip, **item} for item in clip_hard_failures)
        evidence_failures.extend(run_eval.primary_evidence_failures(
            agg, thresholds, clip, clip_meta, worst=worst, rows=rows))
        evidence_failures.extend(run_eval.perf_evidence_failures(
            None, perf, thresholds, clip))
        entry["perf_ms"] = perf
        entry["aggregate"] = agg
        entry["worst_frame"] = worst
        frame_records = run_eval.build_frame_records(rows, thresholds, clip_meta)
        entry["frames"] = frame_records
        entry["label_summary"] = run_eval.summarize_frame_labels(frame_records, thresholds)

    data["issues"] = issues
    data["hard_failures"] = hard_failures
    data["evidence_failures"] = evidence_failures
    data["regressions"] = []
    data["verdict"] = ("hard_failures" if hard_failures else
                       "evidence_failures" if evidence_failures else "comparison_only")
    depth_compensation = depth_compensation_from_meta(data.get("meta", {}))
    data["meta"]["depth_compensation"] = depth_compensation
    for entry in data["clips"].values():
        entry.setdefault("meta", {})["depth_compensation"] = depth_compensation
    data["meta"]["clip_set_sha1"] = current_clip_hashes
    data["meta"]["clips_root"] = clips_root
    data["meta"]["scored_artifact_sha256"] = artifact_hashes
    refresh_contract_metadata(data)
    run_eval.bind_training_labels_to_evidence_gate(data, thresholds)
    out = result_path if args.in_place else os.path.join(args.run_dir, "results.rescored.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
