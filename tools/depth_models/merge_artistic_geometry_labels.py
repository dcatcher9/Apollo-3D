#!/usr/bin/env python3
"""Intersect schema-8 labels per authenticated input condition and geometry.

Each input is one complete selector bundle rendered at one destination
geometry under one canonical ``depth_input_color`` variant.  The output
contains one schema-10 row per unique source image.  SDR-origin images carry
native-SDR plus the three Windows-HDR white conditions; native-PQ images carry
their native-PQ-in-Windows-HDR condition.  Each condition owns the intersection
of its two same-color deployment-geometry frontiers.  The all-condition
intersection remains diagnostic only; training uses the authenticated
per-condition targets and their exact disparity fields.
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

import depth_input_color as input_color
import native_hdr_capture
from artistic_geometry_contract import (
    COLOR_MODE_SDR,
    allowlist_sha256,
    geometry_tuple,
    tuple_key,
    validate_allowlist,
)


LABEL_SCHEMA = 10
SOURCE_LABEL_SCHEMA = 8
POLICY_CONTRACT = "safe-frontier-multistyle-apollo-v1"
OBJECTIVE = (
    "multi-geometry-input-domain-safe-frontier-or-hard-negative-multistyle"
)
ACTION_EPSILON = 0.005
INPUT_VARIANT_MANIFEST_SCHEMA = 1
INPUT_VARIANT_MANIFEST_CONTRACT = (
    "exact-artistic-policy-input-variants-v1"
)
CONDITION_TARGET_SCHEMA = 1
CONDITION_TARGET_CONTRACT = (
    "per-input-condition-two-geometry-safe-frontier-v1"
)
POLICY_HDR_RAW_WHITE_LEVELS = (1000, 2500, 6000)
SELECTED_LABEL_FITTER_CODE_ROLES = frozenset({
    "label_fitter",
    "policy_contract",
    "depth_input_color",
    "depth_input_color_contract",
    "label_preparation",
    "image_loader",
    "evaluator_runner",
})
MERGED_LABEL_FITTER_CODE_ROLES = frozenset({
    *SELECTED_LABEL_FITTER_CODE_ROLES,
    "geometry_merge",
})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label_fitter_code(code, schema, origin="label fitter contract"):
    """Authenticate the exact code roles recorded by one label stage.

    The selector publishes schema-8 bundles.  The geometry merger preserves
    those seven identities and adds itself as the eighth schema-10 identity.
    Keeping this check here lets generation resume, inspection, and training
    share one fail-closed definition instead of drifting independently.
    """
    if schema == SOURCE_LABEL_SCHEMA:
        expected_roles = SELECTED_LABEL_FITTER_CODE_ROLES
    elif schema == LABEL_SCHEMA:
        expected_roles = MERGED_LABEL_FITTER_CODE_ROLES
    else:
        raise RuntimeError(f"{origin}: unsupported label schema {schema}")
    if not isinstance(code, dict) or set(code) != expected_roles:
        raise RuntimeError(
            f"{origin}: label fitter code roles differ for schema {schema}"
        )
    authenticated = {}
    for role in sorted(expected_roles):
        identity = code[role]
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise RuntimeError(f"{origin}: malformed code identity for {role}")
        path = Path(identity.get("path", ""))
        expected = identity.get("sha256")
        if (not path.is_file() or not isinstance(expected, str) or
                sha256(path) != expected):
            raise RuntimeError(
                f"{origin}: code identity is missing or changed for {role}: {path}"
            )
        authenticated[role] = expected
    return authenticated


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_input_variant(value):
    """Validate and detach one canonical depth-input identity."""
    input_color.validate_input_variant(value)
    return json.loads(input_color.canonical_json_bytes(value).decode("utf-8"))


def build_input_variant_manifest(variants):
    """Build the strict, ordered input-domain allow-list for one merge."""
    unique = {}
    for value in variants:
        canonical = canonical_input_variant(value)
        unique[input_color.input_variant_sha256(canonical)] = canonical
    if not unique:
        raise RuntimeError("input-variant manifest is empty")
    return {
        "schema": INPUT_VARIANT_MANIFEST_SCHEMA,
        "contract": INPUT_VARIANT_MANIFEST_CONTRACT,
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        "variants": [unique[key] for key in sorted(unique)],
    }


def validate_input_variant_manifest(value):
    expected_keys = {
        "schema", "contract", "depth_input_color_contract_sha256", "variants",
    }
    if (not isinstance(value, dict) or set(value) != expected_keys or
            value.get("schema") != INPUT_VARIANT_MANIFEST_SCHEMA or
            value.get("contract") != INPUT_VARIANT_MANIFEST_CONTRACT or
            value.get("depth_input_color_contract_sha256") !=
            input_color.color_contract_sha256() or
            not isinstance(value.get("variants"), list)):
        raise RuntimeError("invalid input-variant manifest contract")
    canonical = build_input_variant_manifest(value["variants"])
    try:
        actual_bytes = input_color.canonical_json_bytes(value)
    except (TypeError, ValueError):
        actual_bytes = None
    if actual_bytes != input_color.canonical_json_bytes(canonical):
        raise RuntimeError("input-variant manifest is not canonical")
    return value


def input_variant_manifest_sha256(value):
    validate_input_variant_manifest(value)
    return input_color.canonical_sha256(value)


def policy_input_variants():
    """Return the exact image-derived runtime conditions used by the policy."""
    return [
        input_color.sdr_input_variant(),
        *(input_color.windows_hdr_input_variant(raw_white)
          for raw_white in POLICY_HDR_RAW_WHITE_LEVELS),
        input_color.native_pq_input_variant(),
    ]


def validate_policy_input_variant_manifest(value):
    """Require all five authenticated production input conditions."""
    validate_input_variant_manifest(value)
    expected = build_input_variant_manifest(policy_input_variants())
    if input_color.canonical_json_bytes(value) != (
            input_color.canonical_json_bytes(expected)):
        raise RuntimeError(
            "policy input-variant manifest must contain exactly native SDR and "
            "Windows-HDR raw-white 1000/2500/6000 plus native PQ in Windows HDR"
        )
    return value


def load_input_variant_manifest(path):
    value = load_json(path)
    validate_input_variant_manifest(value)
    return value


def row_input_variant(row, color_mode=None):
    """Resolve legacy SDR rows or validate an explicit canonical variant."""
    effective_color_mode = row.get("color_mode") or color_mode
    if effective_color_mode is None:
        raise RuntimeError(
            "schema-8 row lacks an explicit color_mode for automatic merge"
        )
    value = row.get("input_variant")
    if value is None:
        if effective_color_mode != input_color.COLOR_MODE_SDR:
            raise RuntimeError(
                "non-SDR schema-8 rows require an explicit input_variant"
            )
        value = input_color.sdr_input_variant()
    value = canonical_input_variant(value)
    if value["color_mode"] != effective_color_mode:
        raise RuntimeError("schema-8 row color mode differs from input_variant")
    identity = input_color.input_variant_sha256(value)
    recorded = row.get("input_variant_sha256")
    if recorded is not None and recorded != identity:
        raise RuntimeError("schema-8 row input_variant_sha256 is stale")
    color_hash = row.get("depth_input_color_contract_sha256")
    if (color_hash is not None and
            color_hash != input_color.color_contract_sha256()):
        raise RuntimeError("schema-8 row depth input color contract is stale")
    return value


def resolve_input_variant_manifest(path, bundles, color_mode=None):
    observed = {}
    for bundle in bundles:
        for row in bundle["rows"]:
            variant = row_input_variant(row, color_mode)
            observed[input_color.input_variant_sha256(variant)] = variant
    if path is None:
        sdr = input_color.sdr_input_variant()
        sdr_hash = input_color.input_variant_sha256(sdr)
        if set(observed) != {sdr_hash}:
            raise RuntimeError(
                "non-SDR or multi-input labels require an explicit "
                "input-variant manifest"
            )
        manifest = build_input_variant_manifest([sdr])
        source = None
    else:
        path = Path(path).resolve()
        manifest = load_input_variant_manifest(path)
        source = {"path": str(path), "sha256": sha256(path)}
    declared = {
        input_color.input_variant_sha256(value): value
        for value in manifest["variants"]
    }
    if not set(observed).issubset(declared):
        raise RuntimeError(
            "schema-8 rows contain an undeclared input variant"
        )
    observed_modes = {value["color_mode"] for value in observed.values()}
    declared_modes = {value["color_mode"] for value in declared.values()}
    if not observed_modes.issubset(declared_modes):
        raise RuntimeError(
            "input-variant manifest does not cover observed label color modes"
        )
    if color_mode is not None and observed_modes != {color_mode}:
        raise RuntimeError(
            "input-variant manifest differs from the requested legacy color mode"
        )
    return manifest, input_variant_manifest_sha256(manifest), source


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
    variant = row.get("input_variant")
    if (isinstance(variant, dict) and
            variant.get("kind") == input_color.INPUT_KIND_NATIVE_PQ):
        variant = row_input_variant(row)
        model_source = Path(row.get("model_source", ""))
        expected_size = (
            int(row["source_width"]) * int(row["source_height"]) * 4 * 2
        )
        if (row.get("model_source_encoding") !=
                native_hdr_capture.CAPTURE_ENCODING or
                not model_source.is_file() or
                model_source.stat().st_size != expected_size or
                sha256(model_source) != row.get("model_source_sha256")):
            raise RuntimeError(
                f"{origin}: native HDR FP16 model source is missing or changed"
            )


def validate_row_frontier(row, origin):
    if row.get("policy_contract") != POLICY_CONTRACT:
        raise RuntimeError(f"{origin}: incompatible policy contract")
    safe_min = float(row["safe_scale_min"])
    safe_max = float(row["safe_scale_max"])
    ceiling = float(row["safe_scale_ceiling"])
    identity_feasible = row.get("identity_feasible", True)
    if not isinstance(identity_feasible, bool):
        raise RuntimeError(f"{origin}: identity_feasible is not boolean")
    if not all(math.isfinite(value) for value in (safe_min, safe_max, ceiling)):
        raise RuntimeError(f"{origin}: frontier contains a non-finite value")
    if identity_feasible:
        if (safe_min > 1.0 or safe_max < 1.0 or
                abs(safe_max - ceiling) > 1e-6):
            raise RuntimeError(
                f"{origin}: identity is not inside a connected safe frontier"
            )
    else:
        violations = row.get("identity_violations")
        if (not isinstance(violations, list) or not violations or
                any(not isinstance(value, str) or not value.endswith(":hard")
                    for value in violations)):
            raise RuntimeError(
                f"{origin}: infeasible identity lacks measured hard-bound evidence"
            )
        if (any(abs(value - 1.0) > 1e-6
                for value in (safe_min, safe_max, ceiling)) or
                float(row.get("ceiling_confidence", -1.0)) != 0.0 or
                float(row.get("safety_margin_reliability", -1.0)) != 0.0):
            raise RuntimeError(
                f"{origin}: infeasible identity must be a confidence-zero no-op target"
            )
    if abs(float(row["style_targets"]["clean"]) - 1.0) > 1e-6:
        raise RuntimeError(f"{origin}: clean style is not identity")
    if (not identity_feasible and
            any(abs(float(value) - 1.0) > 1e-6
                for value in row["style_targets"].values())):
        raise RuntimeError(
            f"{origin}: infeasible identity cannot authorize a style multiplier"
        )
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
                "artistic_full_clamp_abs", "color_mode", "input_variant",
                "input_variant_sha256", "depth_input_color_contract_sha256",
                "identity_feasible", "identity_violations", "selection_reason",
                "source_kind", "model_source", "model_source_sha256",
                "model_source_encoding",
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
            "pop": "minimum across exact geometry/input-domain variants",
            "comfort_and_clamp_burden": (
                "maximum across exact geometry/input-domain variants"
            ),
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
        (geometry["eye_width"] / geometry["eye_height"])
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


def row_variant(row, geometry, input_variant, ceiling, style_targets):
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
    input_variant = canonical_input_variant(input_variant)
    return {
        "geometry": geometry,
        "input_variant": input_variant,
        "input_variant_sha256": input_color.input_variant_sha256(
            input_variant
        ),
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
        "identity_feasible": bool(row.get("identity_feasible", True)),
        "identity_violations": list(row.get("identity_violations", [])),
    }


def reduce_safe_frontier(geometry_rows):
    """Intersect identity-connected selector frontiers for one target scope."""
    identity_feasible = all(
        row.get("identity_feasible", True)
        for _geometry, _input_variant, row in geometry_rows
    )
    if identity_feasible:
        safe_min = max(
            float(row["safe_scale_min"])
            for _geometry, _input_variant, row in geometry_rows
        )
        ceiling = min(
            float(row["safe_scale_ceiling"])
            for _geometry, _input_variant, row in geometry_rows
        )
        if safe_min > 1.0 + 1e-6 or ceiling < 1.0 - 1e-6:
            raise RuntimeError(
                "identity is not safe across the target deployment geometries"
            )
    else:
        # A measured baseline failure makes this condition a confidence-zero
        # no-op.  It does not invalidate a different, image-distinguishable
        # input condition for the same frozen source frame.
        safe_min = 1.0
        ceiling = 1.0
    actionable = ceiling - 1.0 >= ACTION_EPSILON
    reliability = (
        min(
            float(row["safety_margin_reliability"])
            for _geometry, _input_variant, row in geometry_rows
        )
        if actionable else 0.0
    )
    confidence = 1.0 if actionable else 0.0
    styles = {
        "clean": 1.0,
        "balanced": 1.0 + 0.5 * (ceiling - 1.0),
        "immersive": ceiling,
    }
    return {
        "safe_scale_min": safe_min,
        "safe_scale_max": ceiling,
        "safe_scale_ceiling": ceiling,
        "baseline_multiplier": ceiling,
        "ceiling_confidence": confidence,
        "confidence": confidence,
        "safety_margin_reliability": reliability,
        "render_evidence_confidence": reliability,
        "identity_feasible": identity_feasible,
        "identity_infeasible_variants": [
            {
                "geometry": geometry,
                "input_variant_sha256":
                    input_color.input_variant_sha256(input_variant),
                "violations": row.get("identity_violations", []),
            }
            for geometry, input_variant, row in geometry_rows
            if not row.get("identity_feasible", True)
        ],
        "style_targets": styles,
    }


def conservative_variant_target(variants, scale):
    """Re-render one scale against every exact raw geometry artifact."""
    targets = []
    for variant in variants:
        raw = load_float_texture(variant["baseline_unclamped_disparity"])
        geometry = variant["geometry"]
        if tuple(raw.shape) != (
                geometry["disparity_raster_height"],
                geometry["disparity_raster_width"]):
            raise RuntimeError(
                "geometry disparity artifact shape differs from its tuple"
            )
        targets.append(render_target(
            raw, geometry, scale, variant["artistic_full_clamp_abs"]
        ))
    return conservative_target(targets, scale)


def build_condition_target(input_variant, geometry_rows, variants):
    """Authenticate one condition's two-geometry safety intersection."""
    if len(geometry_rows) != 2 or len(variants) != 2:
        raise RuntimeError(
            "each policy input condition requires exactly two deployment geometries"
        )
    input_variant = canonical_input_variant(input_variant)
    input_hash = input_color.input_variant_sha256(input_variant)
    if any(
            input_color.input_variant_sha256(value) != input_hash
            for _geometry, value, _row in geometry_rows):
        raise RuntimeError("condition target mixes authenticated input variants")
    geometry_keys = {tuple_key(geometry) for geometry, _value, _row in geometry_rows}
    if len(geometry_keys) != 2:
        raise RuntimeError("condition target repeats a deployment geometry")
    if any(
            geometry["color_mode"] != input_variant["color_mode"]
            for geometry, _value, _row in geometry_rows):
        raise RuntimeError("condition target includes a cross-color geometry")
    target = reduce_safe_frontier(geometry_rows)
    target.update({
        "schema": CONDITION_TARGET_SCHEMA,
        "contract": CONDITION_TARGET_CONTRACT,
        "input_variant": input_variant,
        "input_variant_sha256": input_hash,
        "deployment_geometry_variant_count": len(variants),
    })
    ceiling = target["safe_scale_ceiling"]
    target["safe_ceiling_render_target"] = conservative_variant_target(
        variants, ceiling
    )
    target["safe_ceiling_exact_pop_spread_pct"] = target[
        "safe_ceiling_render_target"
    ]["exact_pop_spread_pct"]
    target["style_render_targets"] = {
        name: conservative_variant_target(variants, scale)
        for name, scale in target["style_targets"].items()
    }
    return target


def merge_rows(rows, geometry_manifest, geometry_manifest_hash,
               input_variant_manifest, input_variant_manifest_hash,
               color_mode, depth_short_side, depth_max_aspect):
    first = rows[0]
    stable_fields = (
        "source_sha256", "clip", "frame", "film_id", "split", "domain",
        "source_width", "source_height", "source_kind", "model_source",
        "model_source_sha256", "model_source_encoding",
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
    geometry_and_input = []
    for row in rows:
        geometry = geometry_tuple(
            row, color_mode,
            depth_short_side=depth_short_side,
            depth_max_aspect=depth_max_aspect,
        )
        input_variant = row_input_variant(row, color_mode)
        geometry_and_input.append((geometry, input_variant, row))
    pair_keys = [
        (tuple_key(geometry), input_color.input_variant_sha256(input_variant))
        for geometry, input_variant, _row in geometry_and_input
    ]
    if len(pair_keys) != len(set(pair_keys)):
        raise RuntimeError(
            "duplicate exact geometry/input variant for one RGB"
        )
    geometries = [item[0] for item in geometry_and_input]
    source_signature = (
        geometries[0]["source_width"], geometries[0]["source_height"],
        geometries[0]["model_input_width"], geometries[0]["model_input_height"],
        geometries[0]["depth_short_side"], geometries[0]["depth_max_aspect"],
    )
    expected_geometries = {
        tuple_key(value): value for value in geometry_manifest["tuples"]
        if (value["source_width"], value["source_height"],
            value["model_input_width"], value["model_input_height"],
            value["depth_short_side"], value["depth_max_aspect"]) ==
        source_signature
    }
    if not expected_geometries:
        raise RuntimeError(
            "deployment geometry manifest has no tuple for this source/model signature"
        )
    declared_variants = {
        input_color.input_variant_sha256(value): value
        for value in input_variant_manifest["variants"]
    }
    # A geometry's color mode describes its capture/render contract.  Only
    # input variants from that same runtime regime are valid partners.  This
    # produces 2 SDR pairs + (2 geometries * 3 HDR-white variants), not the
    # nonsensical blind product that would combine SDR capture with HDR input.
    observed_variant_keys = {variant_key for _geometry_key, variant_key in pair_keys}
    native_key = input_color.input_variant_sha256(
        input_color.native_pq_input_variant()
    )
    sdr_origin_keys = {
        input_color.input_variant_sha256(input_color.sdr_input_variant()),
        *(input_color.input_variant_sha256(
            input_color.windows_hdr_input_variant(raw_white)
        ) for raw_white in POLICY_HDR_RAW_WHITE_LEVELS),
    }
    expected_family = {native_key} if native_key in observed_variant_keys else (
        sdr_origin_keys
    )
    if observed_variant_keys != expected_family:
        raise RuntimeError(
            "RGB does not cover one complete authenticated source-family variant set"
        )
    expected_pairs = {
        (geometry_key, variant_key)
        for geometry_key, geometry in expected_geometries.items()
        for variant_key, variant in declared_variants.items()
        if (variant_key in expected_family and
            geometry["color_mode"] == variant["color_mode"])
    }
    actual_pairs = set(pair_keys)
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            "RGB does not cover the exact deployment geometry x declared "
            "input-variant cross-product"
        )

    clamps = {float(row["artistic_full_clamp_abs"]) for row in rows}
    if len(clamps) != 1:
        raise RuntimeError(
            "identical RGB geometry variants disagree on the source-aspect comfort clamp"
        )
    by_input = {}
    for item in geometry_and_input:
        key = input_color.input_variant_sha256(item[1])
        by_input.setdefault(key, []).append(item)
    if set(by_input) != expected_family:
        raise RuntimeError("RGB condition targets differ from the declared input variants")

    condition_targets = []
    variants = []
    for input_hash in sorted(by_input):
        geometry_rows = sorted(
            by_input[input_hash], key=lambda item: tuple_key(item[0])
        )
        preliminary = reduce_safe_frontier(geometry_rows)
        condition_variants = [
            row_variant(
                row, geometry, input_variant,
                preliminary["safe_scale_ceiling"],
                preliminary["style_targets"],
            )
            for geometry, input_variant, row in geometry_rows
        ]
        condition_targets.append(build_condition_target(
            declared_variants[input_hash], geometry_rows, condition_variants
        ))
        variants.extend(condition_variants)

    # Retain the former all-condition intersection as a conservative diagnostic
    # only.  PolicyDataset binds each expanded image to its condition target.
    global_target = reduce_safe_frontier(geometry_and_input)
    ceiling = global_target["safe_scale_ceiling"]
    conservative = conservative_variant_target(variants, ceiling)
    global_target["safe_ceiling_render_target"] = conservative
    global_target["safe_ceiling_exact_pop_spread_pct"] = conservative[
        "exact_pop_spread_pct"
    ]
    global_target["style_render_targets"] = {
        name: conservative_variant_target(variants, scale)
        for name, scale in global_target["style_targets"].items()
    }
    merged = dict(first)
    merged.update({
        "label_schema": LABEL_SCHEMA,
        **global_target,
        "global_target_semantics": (
            "diagnostic all-condition intersection; never a PolicyDataset target"
        ),
        "condition_target_contract": CONDITION_TARGET_CONTRACT,
        "input_condition_targets": condition_targets,
        "deployment_geometry_allowlist_sha256": geometry_manifest_hash,
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        "input_variant_manifest": input_variant_manifest,
        "input_variant_manifest_sha256": input_variant_manifest_hash,
        "deployment_geometry_variants": variants,
        "geometry_frontier_reduction": (
            "per authenticated input condition: confidence-zero no-op when "
            "either same-color geometry has a measured hard failure; otherwise "
            "intersection of its two identity-connected safe frontiers; the "
            "all-condition intersection is diagnostic only"
        ),
        "artistic_full_clamp_abs": next(iter(clamps)),
    })
    merged.pop("color_mode", None)
    merged.pop("input_variant", None)
    merged.pop("input_variant_sha256", None)
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
                 overwrite=False, input_variant_manifest_path=None):
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
    input_manifest, input_manifest_hash, input_manifest_source = (
        resolve_input_variant_manifest(
            input_variant_manifest_path, bundles, color_mode
        )
    )
    validate_policy_input_variant_manifest(input_manifest)
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
            manifest_hash, input_manifest, input_manifest_hash,
            color_mode, depth_short_side, depth_max_aspect,
        )
        for identity in sorted(identities)
    ]
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("merged labels still contain duplicate RGB images")
    exercised = {
        (
            tuple_key(variant["geometry"]),
            variant["input_variant_sha256"],
        )
        for row in rows for variant in row["deployment_geometry_variants"]
    }
    approved_geometries = {
        tuple_key(value): value for value in manifest["tuples"]
    }
    approved_variants = {
        input_color.input_variant_sha256(value): value
        for value in input_manifest["variants"]
    }
    geometry_modes = {
        value["color_mode"] for value in approved_geometries.values()
    }
    declared_input_modes = {
        value["color_mode"] for value in approved_variants.values()
    }
    applicable_variant_keys = {
        target["input_variant_sha256"]
        for row in rows for target in row["input_condition_targets"]
    }
    input_modes = {
        approved_variants[key]["color_mode"]
        for key in applicable_variant_keys
    }
    if not input_modes.issubset(geometry_modes):
        raise RuntimeError(
            "deployment geometry manifest does not cover applicable input color modes"
        )
    approved = {
        (geometry_key, variant_key)
        for geometry_key, geometry in approved_geometries.items()
        for variant_key, variant in approved_variants.items()
        if (variant_key in applicable_variant_keys and
            geometry["color_mode"] == variant["color_mode"])
    }
    if exercised != approved:
        raise RuntimeError(
            "merged bundle does not exercise the complete deployment geometry "
            "x input-variant manifest"
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
            "condition_target_contract": CONDITION_TARGET_CONTRACT,
            "geometry_reduction": (
                "per-RGB and authenticated input condition: confidence-zero "
                "no-op if either exact same-color geometry has a measured hard "
                "failure; otherwise intersect that condition's two exact "
                "identity-connected geometry frontiers"
            ),
            "artifact_reduction": (
                "retain two exact raw disparities per applicable image condition; "
                "train on the worst of that condition's two geometry field losses"
            ),
        },
        "rendered_disparity_supervision": {
            **first_fitter["rendered_disparity_supervision"],
            "geometry_reduction": (
                "maximum field and gradient loss per expanded image condition "
                "across its two exact deployment geometries"
            ),
        },
        "model_limits": first_fitter["model_limits"],
        "code": code,
        "thresholds": first_fitter["thresholds"],
        "control": first_fitter["control"],
        "deployment_geometry_allowlist": manifest,
        "deployment_geometry_allowlist_sha256": manifest_hash,
        "input_variant_manifest": input_manifest,
        "input_variant_manifest_sha256": input_manifest_hash,
        "input_variant_manifest_source": input_manifest_source,
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        "condition_target_contract": CONDITION_TARGET_CONTRACT,
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
        "geometry_variant_count": len({key for key, _variant in approved}),
        "declared_geometry_variant_count": len(approved_geometries),
        "input_variant_count": len(applicable_variant_keys),
        "declared_input_variant_count": len(approved_variants),
        "geometry_input_variant_count": len(approved),
        "geometry_input_variant_count_by_color_mode": {
            color_mode: sum(
                1 for geometry_key, variant_key in approved
                if approved_geometries[geometry_key]["color_mode"] ==
                color_mode and
                approved_variants[variant_key]["color_mode"] == color_mode
            )
            for color_mode in sorted(input_modes)
        },
        "declared_input_color_modes": sorted(declared_input_modes),
        "deployment_geometry_allowlist_sha256": manifest_hash,
        "input_variant_manifest": input_manifest,
        "input_variant_manifest_sha256": input_manifest_hash,
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        "condition_target_contract": CONDITION_TARGET_CONTRACT,
        "condition_target_count_per_rgb": (
            len(rows[0]["input_condition_targets"])
            if rows else 0
        ),
        "clip_counts": dict(Counter(row["clip"] for row in rows)),
        "selected_scale_counts": dict(Counter(
            str(row["safe_scale_ceiling"]) for row in rows
        )),
        "condition_selected_scale_counts": dict(Counter(
            str(target["safe_scale_ceiling"])
            for row in rows for target in row["input_condition_targets"]
        )),
        "identity_feasible": sum(
            1 for row in rows if row["identity_feasible"]
        ),
        "identity_infeasible": sum(
            1 for row in rows if not row["identity_feasible"]
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-labels", action="append", required=True, type=Path)
    parser.add_argument("--deployment-geometry-manifest", required=True, type=Path)
    parser.add_argument("--input-variant-manifest", type=Path)
    parser.add_argument(
        "--color-mode",
        help=(
            "optional legacy single-mode fallback for rows without an explicit "
            "color_mode; omit for authenticated mixed SDR/HDR labels"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = write_bundle(
        args.geometry_labels, args.deployment_geometry_manifest,
        args.output, args.color_mode, args.overwrite,
        args.input_variant_manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
