#!/usr/bin/env python3
"""Intersect schema-8 artistic labels across exact deployment geometries.

Each input is one complete selector bundle rendered at one destination geometry.
The output contains one schema-9 row per unique RGB image.  Its upward safe
ceiling is the intersection of every geometry's identity-connected frontier;
all geometry-specific raw disparity fields remain attached for worst-geometry
rendered-loss supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from artistic_geometry_contract import (
    COLOR_MODE_SDR,
    allowlist_sha256,
    geometry_tuple,
    tuple_key,
    validate_allowlist,
)


LABEL_SCHEMA = 9
SOURCE_LABEL_SCHEMA = 8
POLICY_CONTRACT = "safe-frontier-multistyle-apollo-v1"
OBJECTIVE = "multi-geometry-connected-safe-frontier-intersection-multistyle"
ACTION_EPSILON = 0.005


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("label_schema", 0)) != SOURCE_LABEL_SCHEMA:
                raise RuntimeError(f"{path}:{line_number}: expected schema-8 selector row")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"geometry label bundle is empty: {path}")
    return rows


def bundle_contract(labels_path):
    labels_path = Path(labels_path).resolve()
    summary_path = labels_path.parent / "summary.json"
    fitter_path = labels_path.parent / "label_fitter_contract.json"
    if not summary_path.is_file() or not fitter_path.is_file():
        raise RuntimeError(f"incomplete schema-8 bundle beside {labels_path}")
    summary = load_json(summary_path)
    fitter = load_json(fitter_path)
    if (int(summary.get("schema", 0)) != SOURCE_LABEL_SCHEMA or
            int(fitter.get("schema", 0)) != SOURCE_LABEL_SCHEMA):
        raise RuntimeError(f"obsolete geometry bundle beside {labels_path}")
    if summary.get("labels_sha256") != sha256(labels_path):
        raise RuntimeError(f"geometry label summary is stale: {labels_path}")
    if summary.get("label_fitter_contract_sha256") != sha256(fitter_path):
        raise RuntimeError(f"geometry label fitter contract is stale: {fitter_path}")
    if fitter.get("policy_contract") != POLICY_CONTRACT:
        raise RuntimeError("geometry bundle has incompatible policy contract")
    return {
        "labels": {"path": str(labels_path), "sha256": sha256(labels_path)},
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "fitter": {"path": str(fitter_path), "sha256": sha256(fitter_path)},
        "payload": fitter,
        "rows": load_rows(labels_path),
    }


def selector_semantics(fitter):
    config = fitter.get("label_fitter_config", {})
    return {
        "policy_contract": fitter.get("policy_contract"),
        "policy_baseline": fitter.get("policy_baseline"),
        "model_limits": fitter.get("model_limits"),
        "rendered_disparity_supervision": fitter.get(
            "rendered_disparity_supervision"
        ),
        "candidate_scales": config.get("candidate_scales"),
        "max_candidate_scale_step": config.get("max_candidate_scale_step"),
        "protected_primary_axes": config.get("protected_primary_axes"),
        "protected_metric_reduction": config.get("protected_metric_reduction"),
        "exact_pop_metric": config.get("exact_pop_metric"),
        "connected_frontier": config.get("connected_frontier"),
        "confidence_semantics": config.get("confidence_semantics"),
        "reliability_semantics": config.get("reliability_semantics"),
        "code_sha256": {
            role: identity.get("sha256")
            for role, identity in fitter.get("code", {}).items()
        },
        "thresholds_sha256": fitter.get("thresholds", {}).get("sha256"),
    }


def validate_source_image(row, origin):
    source = Path(row.get("source", ""))
    if not source.is_file() or sha256(source) != row.get("source_sha256"):
        raise RuntimeError(f"{origin}: source RGB is missing or changed")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"{origin}: source RGB cannot be decoded")
    if (image.shape[1], image.shape[0]) != (
            int(row["source_width"]), int(row["source_height"])):
        raise RuntimeError(f"{origin}: source dimensions differ from decoded RGB")


def validate_row_frontier(row, origin):
    if row.get("policy_contract") != POLICY_CONTRACT:
        raise RuntimeError(f"{origin}: incompatible policy contract")
    safe_min = float(row["safe_scale_min"])
    safe_max = float(row["safe_scale_max"])
    ceiling = float(row["safe_scale_ceiling"])
    if (not all(math.isfinite(value) for value in (safe_min, safe_max, ceiling)) or
            safe_min > 1.0 or safe_max < 1.0 or
            abs(safe_max - ceiling) > 1e-6):
        raise RuntimeError(f"{origin}: identity is not inside a connected safe frontier")
    if abs(float(row["style_targets"]["clean"]) - 1.0) > 1e-6:
        raise RuntimeError(f"{origin}: clean style is not identity")
    validate_source_image(row, origin)


def row_map(bundle):
    result = {}
    for index, row in enumerate(bundle["rows"], 1):
        origin = f"{bundle['labels']['path']}:{index}"
        validate_row_frontier(row, origin)
        identity = row["source_sha256"]
        if identity in result:
            previous = result[identity]
            supervision_fields = (
                "clip", "film_id", "split", "domain", "source_width",
                "source_height", "eye_width", "eye_height", "content_scale_x",
                "content_scale_y", "disparity_raster_width",
                "disparity_raster_height", "safe_scale_min", "safe_scale_max",
                "safe_scale_ceiling", "ceiling_confidence",
                "safety_margin_reliability", "style_targets",
                "baseline_disparity_sha256",
                "baseline_unclamped_disparity_sha256",
                "reference_disparity_sha256", "right_eye_sha256",
                "artistic_full_clamp_abs",
            )
            conflicts = {
                key: (previous.get(key), row.get(key))
                for key in supervision_fields
                if previous.get(key) != row.get(key)
            }
            if conflicts:
                raise RuntimeError(
                    "duplicate identical RGB frames have conflicting targets/context: "
                    f"{identity}: {conflicts}"
                )
            if int(row.get("frame", 0)) < int(previous.get("frame", 0)):
                result[identity] = row
            continue
        result[identity] = row
    return result


def load_manifest(path):
    payload = load_json(path)
    validate_allowlist(payload)
    return payload


def conservative_target(targets, scale):
    """Summarize guaranteed pop and worst comfort/clamp burden."""
    return {
        "scale": float(scale),
        "hlsl_full_clamp_abs": max(
            float(target["hlsl_full_clamp_abs"]) for target in targets
        ),
        "comfort_clamp_abs_pct": max(
            float(target["comfort_clamp_abs_pct"]) for target in targets
        ),
        "mean_abs_disparity_pct": max(
            float(target["mean_abs_disparity_pct"]) for target in targets
        ),
        "p95_abs_disparity_pct": max(
            float(target["p95_abs_disparity_pct"]) for target in targets
        ),
        "exact_pop_spread_pct": min(
            float(target["exact_pop_spread_pct"]) for target in targets
        ),
        "clamped_pixel_pct": max(
            float(target["clamped_pixel_pct"]) for target in targets
        ),
        "reduction": {
            "pop": "minimum across exact deployment geometries",
            "comfort_and_clamp_burden": "maximum across exact deployment geometries",
        },
    }


def load_float_texture(path):
    with Path(path).open("rb") as stream:
        header = np.frombuffer(stream.read(8), dtype="<u4")
        if header.size != 2:
            raise RuntimeError(f"invalid float-texture header: {path}")
        width, height = map(int, header)
        values = np.frombuffer(stream.read(), dtype="<f4")
    if width <= 0 or height <= 0 or values.size != width * height:
        raise RuntimeError(f"invalid float-texture payload: {path}")
    return values.reshape(height, width)


def content_values(raw, geometry):
    height, width = raw.shape
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) / np.float32(width)
    y = (np.arange(height, dtype=np.float32) + np.float32(0.5)) / np.float32(height)
    scale_x = np.float32(geometry["content_scale_x"])
    scale_y = np.float32(geometry["content_scale_y"])
    lo_x = np.float32(0.5) * np.float32(np.float32(1.0) - scale_x)
    lo_y = np.float32(0.5) * np.float32(np.float32(1.0) - scale_y)
    valid_x = (x >= lo_x) & (x <= np.float32(lo_x + scale_x))
    valid_y = (y >= lo_y) & (y <= np.float32(lo_y + scale_y))
    result = raw[valid_y][:, valid_x]
    if result.size == 0:
        raise RuntimeError("deployment geometry has no content-valid disparity pixels")
    return result


def render_target(raw, geometry, scale, clamp_abs):
    raw = content_values(raw, geometry)
    scaled = raw * float(scale)
    final = np.clip(scaled, -clamp_abs, clamp_abs)
    perceived = (
        (geometry["source_width"] / geometry["source_height"])
        / (5120.0 / 2160.0)
    )
    return {
        "scale": float(scale),
        "hlsl_full_clamp_abs": float(clamp_abs),
        "comfort_clamp_abs_pct": float(clamp_abs * perceived * 100.0),
        "mean_abs_disparity_pct": float(
            np.mean(np.abs(final)) * perceived * 100.0
        ),
        "p95_abs_disparity_pct": float(
            np.percentile(np.abs(final), 95) * perceived * 100.0
        ),
        "exact_pop_spread_pct": float(
            (np.percentile(final, 95) - np.percentile(final, 5))
            * perceived * 100.0
        ),
        "clamped_pixel_pct": float(np.mean(np.abs(scaled) > clamp_abs) * 100.0),
    }


def row_variant(row, geometry, ceiling, style_targets):
    raw = Path(row["baseline_unclamped_disparity"])
    expected = row["baseline_unclamped_disparity_sha256"]
    if not raw.is_file() or sha256(raw) != expected:
        raise RuntimeError(f"geometry disparity artifact is missing or changed: {raw}")
    raw_values = load_float_texture(raw)
    if tuple(raw_values.shape) != (
            geometry["disparity_raster_height"],
            geometry["disparity_raster_width"]):
        raise RuntimeError("geometry disparity artifact shape differs from its tuple")
    clamp_abs = float(row["artistic_full_clamp_abs"])
    return {
        "geometry": geometry,
        "baseline_unclamped_disparity": str(raw.resolve()),
        "baseline_unclamped_disparity_sha256": expected,
        "artistic_full_clamp_abs": clamp_abs,
        "safe_ceiling_render_target": render_target(
            raw_values, geometry, ceiling, clamp_abs
        ),
        "style_render_targets": {
            name: render_target(raw_values, geometry, scale, clamp_abs)
            for name, scale in style_targets.items()
        },
        "safe_scale_min": float(row["safe_scale_min"]),
        "safe_scale_max": float(row["safe_scale_max"]),
        "safety_margin_reliability": float(row["safety_margin_reliability"]),
    }


def merge_rows(rows, manifest, manifest_hash, color_mode,
               depth_short_side, depth_max_aspect):
    first = rows[0]
    stable_fields = (
        "source_sha256", "clip", "frame", "film_id", "split", "domain",
        "source_width", "source_height",
    )
    for row in rows[1:]:
        mismatches = {
            key: (first.get(key), row.get(key))
            for key in stable_fields if first.get(key) != row.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "identical RGB geometry variants have conflicting identity/context: "
                f"{mismatches}"
            )
    geometries = [
        geometry_tuple(
            row, color_mode,
            depth_short_side=depth_short_side,
            depth_max_aspect=depth_max_aspect,
        )
        for row in rows
    ]
    if len({tuple_key(value) for value in geometries}) != len(geometries):
        raise RuntimeError("duplicate exact geometry variant for one RGB")
    source_signature = (
        geometries[0]["source_width"], geometries[0]["source_height"],
        geometries[0]["model_input_width"], geometries[0]["model_input_height"],
        geometries[0]["depth_short_side"], geometries[0]["depth_max_aspect"],
        geometries[0]["color_mode"],
    )
    expected = {
        tuple_key(value) for value in manifest["tuples"]
        if (value["source_width"], value["source_height"],
            value["model_input_width"], value["model_input_height"],
            value["depth_short_side"], value["depth_max_aspect"],
            value["color_mode"]) == source_signature
    }
    actual = {tuple_key(value) for value in geometries}
    if not expected:
        raise RuntimeError(
            "deployment geometry manifest has no tuple for this source/model/color signature"
        )
    if actual != expected:
        raise RuntimeError(
            "RGB does not cover every exact geometry in its deployment allow-list group"
        )

    safe_min = max(float(row["safe_scale_min"]) for row in rows)
    ceiling = min(float(row["safe_scale_ceiling"]) for row in rows)
    if safe_min > 1.0 + 1e-6 or ceiling < 1.0 - 1e-6:
        raise RuntimeError("identity is not safe across every deployment geometry")
    actionable = ceiling - 1.0 >= ACTION_EPSILON
    reliability = (
        min(float(row["safety_margin_reliability"]) for row in rows)
        if actionable else 0.0
    )
    confidence = 1.0 if actionable else 0.0
    style_targets = {
        "clean": 1.0,
        "balanced": 1.0 + 0.5 * (ceiling - 1.0),
        "immersive": ceiling,
    }
    clamps = {float(row["artistic_full_clamp_abs"]) for row in rows}
    if len(clamps) != 1:
        raise RuntimeError(
            "identical RGB geometry variants disagree on the source-aspect comfort clamp"
        )
    variants = [
        row_variant(row, geometry, ceiling, style_targets)
        for row, geometry in zip(rows, geometries)
    ]
    # The selector summaries are exact for each geometry's own ceiling.  The
    # common ceiling cannot exceed any of those ceilings, so retaining every
    # raw artifact is authoritative for training; this aggregate is diagnostic.
    conservative = conservative_target(
        [variant["safe_ceiling_render_target"] for variant in variants], ceiling
    )
    merged = dict(first)
    merged.update({
        "label_schema": LABEL_SCHEMA,
        "safe_scale_min": safe_min,
        "safe_scale_max": ceiling,
        "safe_scale_ceiling": ceiling,
        "baseline_multiplier": ceiling,
        "ceiling_confidence": confidence,
        "confidence": confidence,
        "safety_margin_reliability": reliability,
        "render_evidence_confidence": reliability,
        "style_targets": style_targets,
        "style_render_targets": {
            name: conservative_target(
                [variant["style_render_targets"][name] for variant in variants],
                scale,
            )
            for name, scale in style_targets.items()
        },
        "safe_ceiling_render_target": conservative,
        "safe_ceiling_exact_pop_spread_pct": conservative[
            "exact_pop_spread_pct"
        ],
        "deployment_geometry_allowlist_sha256": manifest_hash,
        "deployment_geometry_variants": variants,
        "geometry_frontier_reduction": (
            "intersection of identity-connected safe frontiers; ceiling=min(max)"
        ),
        "artistic_full_clamp_abs": next(iter(clamps)),
    })
    # A merged row must not imply that its first artifact is the only evidence.
    # Keep the aliases for old diagnostics while the trainer consumes variants.
    merged["baseline_unclamped_disparity"] = variants[0][
        "baseline_unclamped_disparity"
    ]
    merged["baseline_unclamped_disparity_sha256"] = variants[0][
        "baseline_unclamped_disparity_sha256"
    ]
    return merged


def write_bundle(label_paths, manifest_path, output, color_mode=COLOR_MODE_SDR,
                 overwrite=False):
    if len(label_paths) < 2:
        raise RuntimeError("multi-geometry labels require at least two exact geometry bundles")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"output must be empty (or use --overwrite): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    manifest_hash = allowlist_sha256(manifest)
    bundles = [bundle_contract(path) for path in label_paths]
    semantics = {
        json.dumps(selector_semantics(bundle["payload"]), sort_keys=True)
        for bundle in bundles
    }
    if len(semantics) != 1:
        raise RuntimeError("geometry bundles use different selector semantics")
    baseline = bundles[0]["payload"].get("policy_baseline", {})
    try:
        depth_short_side = int(baseline["depth_short_side"])
        depth_max_aspect = float(baseline["depth_max_aspect"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "geometry selector baseline lacks exact depth preprocessing settings"
        ) from error
    maps = [row_map(bundle) for bundle in bundles]
    identities = set(maps[0])
    if any(set(rows) != identities for rows in maps[1:]):
        raise RuntimeError("geometry bundles do not contain the same unique RGB images")
    rows = [
        merge_rows(
            [row_map_[identity] for row_map_ in maps], manifest,
            manifest_hash, color_mode, depth_short_side, depth_max_aspect,
        )
        for identity in sorted(identities)
    ]
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("merged labels still contain duplicate RGB images")
    exercised = {
        tuple_key(variant["geometry"])
        for row in rows for variant in row["deployment_geometry_variants"]
    }
    approved = {tuple_key(value) for value in manifest["tuples"]}
    if exercised != approved:
        raise RuntimeError(
            "merged bundle does not exercise every tuple in the deployment geometry manifest"
        )

    labels_path = output / "labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    first_fitter = bundles[0]["payload"]
    code = dict(first_fitter["code"])
    code["geometry_merge"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
    }
    contract = {
        "schema": LABEL_SCHEMA,
        "label_fitter": "exact-apollo-multigeometry-connected-safe-frontier",
        "policy_contract": POLICY_CONTRACT,
        "policy_baseline": first_fitter["policy_baseline"],
        "label_fitter_config": {
            **first_fitter["label_fitter_config"],
            "objective": OBJECTIVE,
            "geometry_reduction": (
                "per-RGB intersection of all exact identity-connected safe frontiers"
            ),
            "artifact_reduction": (
                "retain every exact raw disparity; train on worst geometry loss"
            ),
        },
        "rendered_disparity_supervision": {
            **first_fitter["rendered_disparity_supervision"],
            "geometry_reduction": "maximum field and gradient loss per RGB",
        },
        "model_limits": first_fitter["model_limits"],
        "code": code,
        "thresholds": first_fitter["thresholds"],
        "control": first_fitter["control"],
        "deployment_geometry_allowlist": manifest,
        "deployment_geometry_allowlist_sha256": manifest_hash,
        "geometry_sources": [
            {key: bundle[key] for key in ("labels", "summary", "fitter")}
            for bundle in bundles
        ],
    }
    contract_path = output / "label_fitter_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": LABEL_SCHEMA,
        "accepted": len(rows),
        "rejected": 0,
        "labels_sha256": sha256(labels_path),
        "label_fitter_contract_sha256": sha256(contract_path),
        "unique_rgb_count": len(rows),
        "geometry_variant_count": len(bundles),
        "deployment_geometry_allowlist_sha256": manifest_hash,
        "clip_counts": dict(Counter(row["clip"] for row in rows)),
        "selected_scale_counts": dict(Counter(
            str(row["safe_scale_ceiling"]) for row in rows
        )),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-labels", action="append", required=True, type=Path)
    parser.add_argument("--deployment-geometry-manifest", required=True, type=Path)
    parser.add_argument("--color-mode", default=COLOR_MODE_SDR)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = write_bundle(
        args.geometry_labels, args.deployment_geometry_manifest,
        args.output, args.color_mode, args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
