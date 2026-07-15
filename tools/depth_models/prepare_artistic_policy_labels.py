#!/usr/bin/env python3
"""Build artistic-policy labels with the deterministic stereo label fitter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from artistic_policy_contract import ART_SCALE_DELTA_MAX
from artistic_stereo_label_fitter import (
    LabelFitterConfig,
    finalize_shot,
    frame_analysis,
)


THIS_DIR = Path(__file__).resolve().parent
SBSBENCH_DIR = THIS_DIR.parent / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))
import sbsbench  # noqa: E402


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_paths(root: Path):
    return sorted(root.glob("frame_*.png"))


def load_float_texture(path: Path):
    header = np.fromfile(path, dtype="<u4", count=2)
    if header.size != 2 or np.any(header == 0):
        raise RuntimeError(f"invalid float-texture header: {path}")
    width, height = (int(header[0]), int(header[1]))
    values = np.fromfile(path, dtype="<f4", offset=8)
    if values.size != width * height or not np.all(np.isfinite(values)):
        raise RuntimeError(f"invalid float-texture payload: {path}")
    return values.reshape(height, width)


def run_contract(run: Path):
    results_path = run / "results.json"
    depth_manifest_path = run / "depth_run_manifest.json"
    if depth_manifest_path.is_file():
        payload = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
        return {
            "kind": "depth_run_manifest",
            "sha256": sha256(depth_manifest_path),
            "purpose": payload.get("purpose"),
            "model": payload.get("model"),
            "suite_manifest_sha256": payload.get("suite_manifest_sha256"),
            "executable_sha256": payload.get("executable_sha256"),
            "conf_sha256": payload.get("conf_sha256"),
            "metric_sha256": payload.get("metric_sha256"),
            "policy_warp_source_sha256": payload.get(
                "policy_warp_source_sha256"
            ),
            "frame_count": payload.get("frame_count"),
        }
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        return {
            "kind": "eval_results",
            "sha256": sha256(results_path),
            **{
                key: meta.get(key)
                for key in (
                    "model", "profile", "suite", "clip_set_sha1",
                    "eval_schema", "conf_sha256", "metric_sha256",
                    "policy_warp_source_sha256",
                )
            },
        }
    raise RuntimeError(f"run has no provenance manifest: {run}")


def clip_metadata(clip_root: Path):
    path = clip_root / "meta.json"
    return (json.loads(path.read_text(encoding="utf-8"))
            if path.is_file() else {})


def prepare_clip(clip_root: Path, run_clip: Path, config: LabelFitterConfig):
    metadata = clip_metadata(clip_root)
    harness_contract_path = run_clip / "contract.json"
    if not harness_contract_path.is_file():
        raise RuntimeError(f"missing harness contract: {harness_contract_path}")
    harness_contract = json.loads(
        harness_contract_path.read_text(encoding="utf-8")
    )
    warp_source_hash = harness_contract.get("policy_warp_source_sha256")
    metric_hash = harness_contract.get("metric_sha256")
    if (int(harness_contract.get("schema", 0)) != 24 or
            harness_contract.get("artistic_policy") is not False or
            harness_contract.get("artistic_policy_consumed") is not False or
            harness_contract.get("artistic_policy_authorization") != "none" or
            harness_contract.get("model_onnx_sha256") or
            harness_contract.get("policy_metadata_sha256") or
            harness_contract.get("deployment_geometry_allowlist_sha256") or
            float(harness_contract.get("artistic_scale_override", 0.0)) != 0.0 or
            int(harness_contract.get("source_width", 0)) <= 0 or
            int(harness_contract.get("source_height", 0)) <= 0 or
            int(harness_contract.get("model_input_width", 0)) <= 0 or
            int(harness_contract.get("model_input_height", 0)) <= 0 or
            int(harness_contract.get("eye_width", 0)) <= 0 or
            int(harness_contract.get("eye_height", 0)) <= 0 or
            float(harness_contract.get("content_scale_x", 0.0)) <= 0.0 or
            float(harness_contract.get("content_scale_y", 0.0)) <= 0.0 or
            int(harness_contract.get("disparity_raster_width", 0)) <= 0 or
            int(harness_contract.get("disparity_raster_height", 0)) <= 0 or
            int(harness_contract.get("disparity_raster_width", 0)) !=
            int(harness_contract.get("eye_width", 0)) or
            int(harness_contract.get("disparity_raster_height", 0)) !=
            int(harness_contract.get("eye_height", 0)) or
            harness_contract.get("color_mode") != "sdr-srgb-8bit" or
            harness_contract.get("warp_disparity") !=
            "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars" or
            harness_contract.get("warp_unclamped_disparity") !=
            "unclamped_full_binocular_normalized_at_artistic_scale_1_"
            "output_eye_raster_zero_bars" or
            not isinstance(warp_source_hash, str) or len(warp_source_hash) != 64 or
            not isinstance(metric_hash, str) or len(metric_hash) != 16 or
            float(harness_contract.get("artistic_full_clamp_abs", 0.0)) <= 0.0):
        raise RuntimeError(
            f"incompatible baseline-disparity contract: {harness_contract_path}"
        )
    if float(metadata.get("global_policy_weight", 1.0)) <= 0.0:
        attempted = len(list((clip_root / "gt_right").glob("frame_*.png")))
        return [], {
            "clip": clip_root.name,
            "attempted": attempted,
            "accepted": 0,
            "rejected": attempted,
            "reason": "dataset has no global-policy supervision role",
        }
    inputs = []
    analyses = []
    label_frames = [path for path in frame_paths(clip_root)
                    if (clip_root / "gt_right" / path.name).is_file()]
    for source_path in label_frames:
        suffix = source_path.stem.removeprefix("frame_")
        right_path = clip_root / "gt_right" / source_path.name
        reference_path = (
            clip_root / "gt_disparity" / f"frame_{suffix}.npz"
        )
        baseline_path = run_clip / f"baseline_disparity_{suffix}.f32"
        unclamped_path = (
            run_clip / f"baseline_unclamped_disparity_{suffix}.f32"
        )
        for path, description in (
            (right_path, "right eye"),
            (baseline_path, "Apollo baseline disparity"),
            (unclamped_path, "Apollo unclamped scale-1 disparity"),
        ):
            if not path.is_file():
                raise RuntimeError(
                    f"{clip_root.name}/{source_path.name}: missing {description}: {path}"
                )
        reference = None
        vertical = None
        if reference_path.is_file():
            with np.load(reference_path) as payload:
                reference = payload["disparity_px"]
                if "vertical_disparity_px" in payload:
                    vertical = payload["vertical_disparity_px"]
        source_gray = sbsbench.load_gray(str(source_path))
        if (source_gray.shape[1] != int(harness_contract["source_width"]) or
                source_gray.shape[0] != int(harness_contract["source_height"])):
            raise RuntimeError(
                f"{clip_root.name}/{source_path.name}: source geometry differs "
                "from the harness contract"
            )
        baseline_disparity = load_float_texture(baseline_path)
        unclamped_disparity = load_float_texture(unclamped_path)
        expected_disparity_shape = (
            int(harness_contract["disparity_raster_height"]),
            int(harness_contract["disparity_raster_width"]),
        )
        if (baseline_disparity.shape != expected_disparity_shape or
                unclamped_disparity.shape != expected_disparity_shape):
            raise RuntimeError(
                f"{clip_root.name}/{source_path.name}: disparity raster differs "
                "from the harness contract"
            )
        inputs.append((
            source_path, right_path, baseline_path, unclamped_path,
            reference_path, suffix
        ))
        analyses.append(frame_analysis(
            source_gray,
            sbsbench.load_gray(str(right_path)),
            baseline_disparity,
            config,
            reference,
            vertical,
        ))

    outputs = finalize_shot(
        analyses, config, ART_SCALE_DELTA_MAX
    )
    if outputs is None:
        return [], {
            "clip": clip_root.name,
            "attempted": len(inputs),
            "accepted": 0,
            "rejected": len(inputs),
            "reason": "no supported positive-polarity shot fit",
        }

    rows = []
    for paths, output in zip(inputs, outputs):
        (source_path, right_path, baseline_path, unclamped_path,
         reference_path, suffix) = paths
        if output is None:
            continue
        diagnostics = output["diagnostics"]
        row = {
            "label_schema": 7,
            "policy_contract": "stereo-fit-source-v2",
            "source": str(source_path.resolve()),
            "source_sha256": sha256(source_path),
            "right_eye": str(right_path.resolve()),
            "right_eye_sha256": sha256(right_path),
            "baseline_disparity": str(baseline_path.resolve()),
            "baseline_disparity_sha256": sha256(baseline_path),
            "baseline_unclamped_disparity": str(unclamped_path.resolve()),
            "baseline_unclamped_disparity_sha256": sha256(unclamped_path),
            "source_width": int(harness_contract["source_width"]),
            "source_height": int(harness_contract["source_height"]),
            "model_input_width": int(harness_contract["model_input_width"]),
            "model_input_height": int(harness_contract["model_input_height"]),
            "eye_width": int(harness_contract["eye_width"]),
            "eye_height": int(harness_contract["eye_height"]),
            "content_scale_x": float(harness_contract["content_scale_x"]),
            "content_scale_y": float(harness_contract["content_scale_y"]),
            "color_mode": harness_contract["color_mode"],
            "disparity_raster_width": int(
                harness_contract["disparity_raster_width"]
            ),
            "disparity_raster_height": int(
                harness_contract["disparity_raster_height"]
            ),
            "policy_warp_source_sha256": warp_source_hash,
            "metric_sha256": metric_hash,
            "artistic_full_clamp_abs": float(
                harness_contract["artistic_full_clamp_abs"]
            ),
            "harness_contract_sha256": sha256(harness_contract_path),
            "reference_disparity": (
                str(reference_path.resolve()) if reference_path.is_file()
                else None
            ),
            "reference_disparity_sha256": (
                sha256(reference_path) if reference_path.is_file() else None
            ),
            "stereo_fit_multiplier": output["baseline_multiplier"],
            "stereo_fit_confidence": output["confidence"],
            # Compatibility aliases for the source fitter only. Schema-8 replaces these
            # with the render-validated safe ceiling and its confidence.
            "baseline_multiplier": output["baseline_multiplier"],
            "confidence": output["confidence"],
            "clip": clip_root.name,
            "frame": int(suffix),
            "split": metadata.get("split"),
            "domain": metadata.get(
                "domain", metadata.get("dataset", "unknown")
            ),
            "dataset": metadata.get("dataset"),
            "film_id": metadata.get("film_id"),
            "license": metadata.get("license"),
            "source_split": metadata.get("source_split"),
            "projection": metadata.get("projection", "rectilinear"),
            "policy_role": metadata.get("policy_role", "legacy_bootstrap"),
            "global_policy_weight": float(
                metadata.get("global_policy_weight", 1.0)
            ),
            **diagnostics,
        }
        rows.append(row)
    confidences = [row["confidence"] for row in rows]
    return rows, {
        "clip": clip_root.name,
        "attempted": len(inputs),
        "accepted": len(rows),
        "rejected": len(inputs) - len(rows),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "comfort_scale": rows[0]["comfort_scale"] if rows else None,
        "shot_baseline_multiplier": (
            rows[0]["baseline_multiplier"] if rows else None
        ),
    }


def prepare(run: Path, clips: Path, output: Path,
            config=LabelFitterConfig(), overwrite=False):
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"output must be empty (or use --overwrite): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    clip_stats = []
    for clip_root in sorted(clips.iterdir()):
        if not clip_root.is_dir() or not (clip_root / "gt_right").is_dir():
            continue
        run_clip = run / clip_root.name
        if not run_clip.is_dir():
            raise RuntimeError(f"missing run output for clip: {clip_root.name}")
        clip_rows, stats = prepare_clip(
            clip_root, run_clip, config
        )
        rows.extend(clip_rows)
        clip_stats.append(stats)
        print(
            f"[{clip_root.name}] {stats['accepted']}/{stats['attempted']} labels",
            flush=True,
        )

    if not rows:
        raise RuntimeError("stereo label fitter accepted no labels")
    manifest = output / "labels.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    contract = {
        "schema": 7,
        "label_fitter": (
            "subpixel-stereo-source-fit-with-clamp-aware-apollo-disparity"
        ),
        "label_fitter_config": asdict(config),
        "code": {
            "label_fitter": {
                "path": str((THIS_DIR / "artistic_stereo_label_fitter.py").resolve()),
                "sha256": sha256(THIS_DIR / "artistic_stereo_label_fitter.py"),
            },
            "policy_contract": {
                "path": str((THIS_DIR / "artistic_policy_contract.py").resolve()),
                "sha256": sha256(THIS_DIR / "artistic_policy_contract.py"),
            },
            "label_preparation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "image_loader": {
                "path": str((SBSBENCH_DIR / "sbsbench.py").resolve()),
                "sha256": sha256(SBSBENCH_DIR / "sbsbench.py"),
            },
        },
        "model_limits": {
            "scale_delta_max": ART_SCALE_DELTA_MAX,
        },
        "run": str(run.resolve()),
        "clips": str(clips.resolve()),
        "run_contract": run_contract(run),
        "role_contract": {
            "cinematic_bootstrap": "global supervision",
            "cinematic_training": "global supervision",
            "local_geometry": "excluded from this milestone",
            "validation_only": "no training supervision",
            "vr_parallel": "excluded unless explicitly authored",
        },
    }
    contract_path = output / "label_fitter_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": 7,
        "accepted": len(rows),
        "rejected": sum(item["rejected"] for item in clip_stats),
        "labels_sha256": sha256(manifest),
        "label_fitter_contract_sha256": sha256(contract_path),
        "clip_stats": clip_stats,
        "clip_counts": {
            clip: sum(row["clip"] == clip for row in rows)
            for clip in sorted({row["clip"] for row in rows})
        },
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in sorted({row["domain"] for row in rows})
        },
        "policy_role_counts": {
            role: sum(row["policy_role"] == role for row in rows)
            for role in sorted({row["policy_role"] for row in rows})
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path,
                        help="Apollo run containing per-clip depth_*.png")
    parser.add_argument("--clips", required=True, type=Path,
                        help="Prepared clips containing frame_*.png and gt_right/")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analysis-width", type=int, default=512)
    parser.add_argument("--comfort-limit", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = LabelFitterConfig(
        analysis_width=args.analysis_width,
        comfort_limit=args.comfort_limit,
    )
    summary = prepare(
        args.run, args.clips, args.output, config, args.overwrite
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
