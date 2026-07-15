#!/usr/bin/env python3
"""Recompute metric JSON from existing SBS/depth/source artifacts without rerunning the GPU.

Only comparison-only runs are accepted: committed baseline verdicts must be produced by run_eval,
not rewritten after the fact. Artifact identities remain unchanged; the metric contract hash and
derived aggregates/issues/worst frames are refreshed to the current scoring code.
"""
import argparse
import json
import os
import re
import sys

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import run_eval  # noqa: E402
import sbsbench  # noqa: E402


HARNESS_SCHEMA = 24
EXACT_DISPARITY_SEMANTICS = (
    "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"
)
UNCLAMPED_DISPARITY_SEMANTICS = (
    "unclamped_full_binocular_normalized_at_artistic_scale_1_"
    "output_eye_raster_zero_bars"
)
ARTISTIC_DISPARITY_CONTRACT = (
    "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
    "times_aspect_scale_times_content_scale_x)"
)


def _artifact_ids(run_clip, pattern, prefix):
    return set(sbsbench.indexed_files(os.path.join(run_clip, pattern), prefix))


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_metric_contract_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value) is not None


def validate_current_artifact_contract(data, run_dir, clips_root):
    """Refuse to promote old/incomplete artifacts into the current evaluator schema.

    Rescoring may update metric arithmetic, but schema 29 also promises harness-24 output-eye
    disparity evidence. That promise can only come from the original GPU run; it cannot be created
    by rewriting results.json.
    """
    meta = data.get("meta")
    if not isinstance(meta, dict) or meta.get("eval_schema") != run_eval.EVAL_SCHEMA:
        raise RuntimeError(
            f"run uses eval schema {meta.get('eval_schema') if isinstance(meta, dict) else None}; "
            f"schema {run_eval.EVAL_SCHEMA} requires a fresh GPU run"
        )
    scoring_metric_hash = meta.get("metric_sha256")
    if not _is_metric_contract_hash(scoring_metric_hash):
        raise RuntimeError("run has no valid scoring metric hash")
    declared_artifact_metric_hash = meta.get("artifact_metric_sha256")
    if declared_artifact_metric_hash is not None and not _is_metric_contract_hash(
            declared_artifact_metric_hash):
        raise RuntimeError("run has an invalid artifact metric hash")
    artifact_metric_hash = declared_artifact_metric_hash
    policy_hash = meta.get("policy_warp_source_sha256")
    if not isinstance(policy_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", policy_hash):
        raise RuntimeError("run has no valid policy-warp source hash")
    clips = data.get("clips")
    if not isinstance(clips, dict) or not clips:
        raise RuntimeError("run contains no clip results")
    clip_hashes = meta.get("clip_set_sha1")
    if not isinstance(clip_hashes, dict) or set(clip_hashes) != set(clips):
        raise RuntimeError(
            "run has no exact original clip_set_sha1 identity for every result clip"
        )

    for clip, entry in clips.items():
        run_clip = os.path.join(run_dir, clip)
        source_clip = os.path.join(clips_root, clip)
        recorded_clip_hash = clip_hashes[clip]
        if (not isinstance(recorded_clip_hash, str) or
                not re.fullmatch(r"[0-9a-f]{12}", recorded_clip_hash)):
            raise RuntimeError(f"{clip}: invalid original clip source hash")
        supplied_clip_hash = run_eval.sha1_dir(source_clip)
        if supplied_clip_hash != recorded_clip_hash:
            raise RuntimeError(
                f"{clip}: supplied source clip differs from the original GPU run: "
                f"{supplied_clip_hash} != {recorded_clip_hash}"
            )
        contract_path = os.path.join(run_clip, "contract.json")
        try:
            with open(contract_path, encoding="utf-8") as fh:
                contract = json.load(fh)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{clip}: missing/invalid harness contract: {exc}") from exc
        expected_contract = {
            "schema": HARNESS_SCHEMA,
            "artifact_mode": "full",
            "warp_mask": {"red": "forward_disocclusion_before_fill"},
            "warp_disparity": EXACT_DISPARITY_SEMANTICS,
            "warp_unclamped_disparity": UNCLAMPED_DISPARITY_SEMANTICS,
            "artistic_disparity_contract": ARTISTIC_DISPARITY_CONTRACT,
            "policy_warp_source_sha256": policy_hash,
            "output_interval": meta.get("output_interval"),
            "output_gt_right_only": meta.get("output_gt_right_only"),
        }
        mismatch = {key: (expected, contract.get(key))
                    for key, expected in expected_contract.items()
                    if contract.get(key) != expected}
        if mismatch:
            raise RuntimeError(f"{clip}: stale/incompatible harness contract: {mismatch}")
        contract_metric_hash = contract.get("metric_sha256")
        if not _is_metric_contract_hash(contract_metric_hash):
            raise RuntimeError(f"{clip}: harness contract has no valid artifact metric hash")
        if artifact_metric_hash is None:
            artifact_metric_hash = contract_metric_hash
        elif artifact_metric_hash != contract_metric_hash:
            raise RuntimeError(
                f"{clip}: harness artifact metric hash differs from the run: "
                f"{contract_metric_hash} != {artifact_metric_hash}"
            )
        clip_meta = entry.get("meta", {}) if isinstance(entry, dict) else {}
        for field in (
                "artistic_policy_consumed", "artistic_policy_authorization",
                "model_onnx_sha256",
                "policy_metadata_sha256",
                "deployment_geometry_allowlist_sha256"):
            if (contract.get(field) != clip_meta.get(field) or
                    contract.get(field) != meta.get(field)):
                raise RuntimeError(
                    f"{clip}: {field} differs across harness/clip/run provenance"
                )
        consumed = contract.get("artistic_policy_consumed")
        authorization = contract.get("artistic_policy_authorization")
        if not isinstance(consumed, bool):
            raise RuntimeError(f"{clip}: invalid artistic-policy consumption state")
        expected_authorization = "candidate-evaluation" if consumed else "none"
        if authorization != expected_authorization:
            raise RuntimeError(f"{clip}: invalid artistic-policy authorization")
        for field in (
                "model_onnx_sha256", "policy_metadata_sha256",
                "deployment_geometry_allowlist_sha256"):
            value = contract.get(field)
            if consumed and not _is_sha256(value):
                raise RuntimeError(f"{clip}: consumed policy has invalid {field}")
            if not consumed and value != "":
                raise RuntimeError(f"{clip}: unconsumed policy unexpectedly records {field}")
        if clip_meta.get("policy_warp_source_sha256") != policy_hash:
            raise RuntimeError(f"{clip}: clip policy-warp source hash differs from the run")
        if clip_meta.get("clip_sha1") != recorded_clip_hash:
            raise RuntimeError(f"{clip}: clip result source hash differs from the run")
        clip_artifact_metric_hash = clip_meta.get("artifact_metric_sha256")
        if declared_artifact_metric_hash is None:
            if clip_artifact_metric_hash is not None:
                raise RuntimeError(f"{clip}: partial artifact metric provenance")
            if clip_meta.get("metric_sha256") != artifact_metric_hash:
                raise RuntimeError(f"{clip}: original clip metric hash differs from artifacts")
        else:
            if clip_artifact_metric_hash != artifact_metric_hash:
                raise RuntimeError(f"{clip}: clip artifact metric hash differs from the run")
            if clip_meta.get("metric_sha256") != scoring_metric_hash:
                raise RuntimeError(f"{clip}: clip scoring metric hash differs from the run")
        try:
            source_width = int(contract["source_width"])
            source_height = int(contract["source_height"])
            model_input_width = int(contract["model_input_width"])
            model_input_height = int(contract["model_input_height"])
            eye_width = int(contract["eye_width"])
            eye_height = int(contract["eye_height"])
            raster_width = int(contract["disparity_raster_width"])
            raster_height = int(contract["disparity_raster_height"])
            output_interval = int(contract["output_interval"])
            content_scale_x = float(contract["content_scale_x"])
            content_scale_y = float(contract["content_scale_y"])
            artistic_full_clamp_abs = float(contract["artistic_full_clamp_abs"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{clip}: incomplete harness geometry contract") from exc
        if min(source_width, source_height, model_input_width, model_input_height,
               eye_width, eye_height,
               raster_width, raster_height, output_interval) <= 0:
            raise RuntimeError(f"{clip}: invalid harness geometry contract")
        if raster_width != eye_width or raster_height != eye_height:
            raise RuntimeError(f"{clip}: disparity raster is not the complete output eye")
        if (not 0.0 < content_scale_x <= 1.0 or
                not 0.0 < content_scale_y <= 1.0 or
                not artistic_full_clamp_abs > 0.0):
            raise RuntimeError(f"{clip}: invalid scale/clamp geometry contract")
        if contract.get("color_mode") not in {
                "sdr-srgb-8bit", "linear-sdr-fp16", "hdr-scrgb-fp16"}:
            raise RuntimeError(f"{clip}: invalid color-mode contract")
        for field, expected in (
                ("source_width", source_width),
                ("source_height", source_height),
                ("model_input_width", model_input_width),
                ("model_input_height", model_input_height),
                ("eye_width", eye_width),
                ("eye_height", eye_height),
                ("disparity_raster_width", raster_width),
                ("disparity_raster_height", raster_height),
                ("content_scale_x", content_scale_x),
                ("content_scale_y", content_scale_y),
                ("artistic_full_clamp_abs", artistic_full_clamp_abs),
                ("color_mode", contract.get("color_mode"))):
            if clip_meta.get(field) != expected:
                raise RuntimeError(
                    f"{clip}: recorded dimensions/color {field} differ from harness contract"
                )

        source_files = sbsbench.indexed_files(
            os.path.join(source_clip, "frame_*.*"), "frame_")
        for frame_id, path in source_files.items():
            try:
                with Image.open(path) as image:
                    dimensions = image.size
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"{clip}: invalid source frame {frame_id}: {exc}") from exc
            if dimensions != (source_width, source_height):
                raise RuntimeError(
                    f"{clip}: source frame {frame_id} dimensions {dimensions} differ from "
                    f"the harness contract {(source_width, source_height)}"
                )
        source_ids = sorted(source_files)
        expected_ids = set(source_ids[::output_interval])
        if contract["output_gt_right_only"]:
            gt_ids = set(sbsbench.indexed_files(
                os.path.join(source_clip, "gt_right", "frame_*.*"), "frame_"))
            expected_ids &= gt_ids
        artifact_ids = {
            "sbs": _artifact_ids(run_clip, "sbs_*.png", "sbs_"),
            "depth": _artifact_ids(run_clip, "depth_*.png", "depth_"),
            "raw": _artifact_ids(run_clip, "raw_*.f32", "raw_"),
            "warp_mask": _artifact_ids(run_clip, "warp_mask_*.png", "warp_mask_"),
            "warp_disparity": _artifact_ids(
                run_clip, "warp_disparity_*.f32", "warp_disparity_"),
            "warp_unclamped_disparity": _artifact_ids(
                run_clip, "warp_unclamped_disparity_*.f32",
                "warp_unclamped_disparity_"),
        }
        mismatched_ids = {name: sorted(ids) for name, ids in artifact_ids.items()
                          if ids != expected_ids}
        if not expected_ids or mismatched_ids:
            raise RuntimeError(
                f"{clip}: incomplete exact artifact identities; "
                f"expected={sorted(expected_ids)}, actual={mismatched_ids}"
            )
    return artifact_metric_hash


def depth_compensation_from_meta(meta):
    """Preserve or derive the explicit depth-compensation contract."""
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


def refresh_contract_metadata(data, artifact_metric_sha256=None):
    """Record immutable artifact provenance and the current rescoring contract separately."""
    if data.get("meta", {}).get("eval_schema") != run_eval.EVAL_SCHEMA:
        raise RuntimeError("cannot rescore artifacts from an older evaluator schema")
    meta = data["meta"]
    if artifact_metric_sha256 is None:
        artifact_metric_sha256 = meta.get(
            "artifact_metric_sha256", meta.get("metric_sha256"))
    if not _is_metric_contract_hash(artifact_metric_sha256):
        raise RuntimeError("cannot preserve an invalid artifact metric hash")
    current_metric_sha256 = run_eval.metric_contract_sha()
    meta["artifact_metric_sha256"] = artifact_metric_sha256
    meta["metric_sha256"] = current_metric_sha256
    for entry in data.get("clips", {}).values():
        clip_meta = entry.setdefault("meta", {})
        clip_meta["artifact_metric_sha256"] = artifact_metric_sha256
        clip_meta["metric_sha256"] = current_metric_sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="sbs_eval run containing results.json and per-clip artifacts")
    ap.add_argument("--clips-root", default=None)
    ap.add_argument("--in-place", action="store_true",
                    help="replace results.json atomically (default writes results.rescored.json)")
    args = ap.parse_args()
    result_path = os.path.join(args.run_dir, "results.json")
    data = json.load(open(result_path, encoding="utf-8"))
    if data.get("meta", {}).get("run_kind") != "comparison_only":
        raise SystemExit("refusing to rescore a non-comparison run; rerun run_eval instead")
    clips_root = args.clips_root
    if clips_root is None:
        clips_root = data.get("meta", {}).get("clips_root")
    if not isinstance(clips_root, str) or not os.path.isdir(clips_root):
        raise SystemExit("cannot resolve the original clip root; pass --clips-root explicitly")
    try:
        artifact_metric_sha256 = validate_current_artifact_contract(
            data, args.run_dir, clips_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"refusing stale/incomplete rescore artifacts: {exc}") from exc
    thresholds = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json"), encoding="utf-8"))
    issues, hard_failures = [], []
    for clip, entry in data["clips"].items():
        clip_dir = os.path.join(clips_root, clip)
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
    refresh_contract_metadata(data, artifact_metric_sha256)
    depth_compensation = depth_compensation_from_meta(data.get("meta", {}))
    data["meta"]["depth_compensation"] = depth_compensation
    for entry in data["clips"].values():
        entry.setdefault("meta", {})["depth_compensation"] = depth_compensation
    out = result_path if args.in_place else os.path.join(args.run_dir, "results.rescored.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
