#!/usr/bin/env python3
"""Authenticate and visualize the complete pretraining bootstrap dataset.

This tool is deliberately read-only with respect to both input workspaces and
never imports or invokes the trainer.  It audits the SDR-origin and native-PQ
branches, writes a machine-readable ``inspection.json`` plus a self-contained
``report.html``, and exits nonzero when any required contract fails.

Only training/development publications named by the two orchestration plans
are opened.  Sealed-test paths recorded in upstream split manifests are never
resolved, opened, enumerated, or hashed.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import datetime
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
SBSBENCH_DIR = THIS_DIR.parent / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))

import build_clip_hash_manifest as clip_hashes  # noqa: E402
import artistic_geometry_contract as geometry_contract  # noqa: E402
import depth_input_color as input_color  # noqa: E402
import generate_artistic_depth_run as depth_run  # noqa: E402
import merge_artistic_geometry_labels as label_merge  # noqa: E402
import native_hdr_capture  # noqa: E402
import prepare_chug_native_hdr_training as chug_prepare  # noqa: E402
from artistic_policy_contract import ART_SCALE_DELTA_MAX  # noqa: E402


SCHEMA = 1
CONTRACT = "apollo-artistic-bootstrap-pretraining-inspection-v1"
SDR_PLAN = "orchestration_plan.json"
SDR_PLAN_SCHEMA = 3
SDR_PLAN_CONTRACT = "apollo-public-mono-sdr-hdr-bootstrap-orchestration-v2"
SDR_BOOTSTRAP_CONTRACT = "apollo-public-mono-hdr-bootstrap-subset-v1"
NATIVE_PLAN = "native_hdr_label_plan.json"
NATIVE_PLAN_SCHEMA = 1
NATIVE_PLAN_CONTRACT = "apollo-chug-native-pq-label-orchestration-v1"
NATIVE_BOOTSTRAP_SCHEMA = chug_prepare.PREPARATION_SCHEMA
NATIVE_BOOTSTRAP_CONTRACT = chug_prepare.PREPARATION_CONTRACT
NATIVE_CONVERSION_CONTRACT = "pq-bt2020nc-to-windows-scrgb16-v1"
LABEL_SCHEMA = 10
POLICY_CONTRACT = "safe-frontier-multistyle-apollo-v1"
CONDITION_TARGET_CONTRACT = (
    "per-input-condition-two-geometry-safe-frontier-v1"
)
ACTION_EPSILON = 0.005
EXPECTED_CARDINALITY = {
    "sdr_origin": {"training": 240, "development": 80},
    "native_pq": {"training": 60, "development": 20},
    "combined": {"training": 300, "development": 100},
}
EXPECTED_SDR_RGB = {"training": 60, "development": 20}
EXPECTED_NATIVE_RGB = {"training": 60, "development": 20}
EXPECTED_REGIME_CARDINALITY = {
    "sdr": {"training": 60, "development": 20},
    "hdr": {"training": 180, "development": 60},
    "hdr-w1000": {"training": 60, "development": 20},
    "hdr-w2500": {"training": 60, "development": 20},
    "hdr-w6000": {"training": 60, "development": 20},
    "native-pq": {"training": 60, "development": 20},
}
EXPECTED_SDR_CONDITIONS = {
    input_color.INPUT_KIND_SDR,
    input_color.INPUT_KIND_WINDOWS_HDR,
}
SPLITS = ("training", "development")
REC709_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
FP16_MAX = float(np.finfo(np.float16).max)
NUMERIC_TOLERANCE = 1e-6


class InspectionError(RuntimeError):
    """One fail-closed inspection contract violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path, description: str):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InspectionError(f"cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise InspectionError(f"{description} is not an object: {path}")
    return payload


def load_jsonl(path: Path):
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise InspectionError(
                        f"invalid JSON row at {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise InspectionError(
                        f"label row is not an object at {path}:{line_number}"
                    )
                rows.append(row)
    except OSError as error:
        raise InspectionError(f"cannot read labels: {path}") from error
    if not rows:
        raise InspectionError(f"label bundle is empty: {path}")
    return rows


def assert_working_path(path: Path, description: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if "test" in {part.casefold() for part in resolved.parts}:
        raise InspectionError(
            f"sealed-test-looking {description} is forbidden: {resolved}"
        )
    return resolved


def require_hash(path: Path, expected, description: str) -> str:
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(value not in "0123456789abcdef" for value in expected)):
        raise InspectionError(f"{description} has an invalid SHA-256 identity")
    if not path.is_file():
        raise InspectionError(f"{description} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise InspectionError(
            f"{description} SHA-256 differs: {actual} != {expected}"
        )
    return actual


def finite_number(value, origin: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise InspectionError(f"{origin} is not numeric") from error
    if not math.isfinite(result):
        raise InspectionError(f"{origin} is non-finite")
    return result


def percentile_summary(values):
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise InspectionError("distribution contains non-finite values")
    percentiles = np.percentile(array, (0, 5, 25, 50, 75, 95, 100))
    return {
        "count": int(array.size),
        "min": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "max": float(percentiles[6]),
        "mean": float(np.mean(array)),
    }


def load_float_texture(path: Path, expected_sha=None, expected_shape=None):
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise InspectionError(f"cannot read geometry evidence: {path}") from error
    if expected_sha is not None and sha256_bytes(payload) != expected_sha:
        raise InspectionError(f"geometry evidence SHA-256 differs: {path}")
    if len(payload) < 8:
        raise InspectionError(f"geometry evidence header is truncated: {path}")
    header = np.frombuffer(payload[:8], dtype="<u4")
    width, height = map(int, header)
    values = np.frombuffer(payload[8:], dtype="<f4")
    if (width <= 0 or height <= 0 or values.size != width * height or
            not np.isfinite(values).all()):
        raise InspectionError(f"geometry evidence is invalid/non-finite: {path}")
    if (expected_shape is not None and
            (height, width) != tuple(expected_shape)):
        raise InspectionError(f"geometry evidence shape differs: {path}")
    return values.reshape(height, width)


def input_variant_key(variant) -> str:
    try:
        input_color.validate_input_variant(variant)
        return input_color.input_variant_sha256(variant)
    except RuntimeError as error:
        raise InspectionError(f"invalid input variant: {error}") from error


def condition_name(variant) -> str:
    kind = variant["kind"]
    if kind == input_color.INPUT_KIND_SDR:
        return "sdr"
    if kind == input_color.INPUT_KIND_WINDOWS_HDR:
        return f"hdr-w{variant['windows_sdr_white_level_raw']}"
    if kind == input_color.INPUT_KIND_NATIVE_PQ:
        return "native-pq"
    raise InspectionError(f"unsupported input condition: {kind}")


def geometry_identity(geometry) -> str:
    try:
        canonical = geometry_contract.canonical_geometry_tuple(geometry)
    except RuntimeError as error:
        raise InspectionError(f"invalid deployment geometry: {error}") from error
    return geometry_contract.tuple_key(canonical)


def validate_plan_contracts(plan, origin: str):
    geometry_manifest = plan.get("deployment_geometry_manifest")
    input_manifest = plan.get("input_variant_manifest")
    try:
        geometry_contract.validate_allowlist(geometry_manifest)
        geometry_hash = geometry_contract.allowlist_sha256(geometry_manifest)
        label_merge.validate_policy_input_variant_manifest(input_manifest)
        input_hash = label_merge.input_variant_manifest_sha256(input_manifest)
    except RuntimeError as error:
        raise InspectionError(f"{origin} plan contract differs: {error}") from error
    if plan.get("deployment_geometry_manifest_identity") != geometry_hash:
        raise InspectionError(f"{origin} geometry-manifest identity differs")
    if plan.get("input_variant_manifest_identity") != input_hash:
        raise InspectionError(f"{origin} input-manifest identity differs")
    if (plan.get("condition_target_contract") not in
            (None, CONDITION_TARGET_CONTRACT)):
        raise InspectionError(f"{origin} condition-target contract differs")
    return {
        "geometry_manifest": geometry_manifest,
        "geometry_hash": geometry_hash,
        "input_manifest": input_manifest,
        "input_hash": input_hash,
    }


def numeric_close(actual, expected, origin: str, tolerance=NUMERIC_TOLERANCE):
    left = finite_number(actual, origin)
    right = finite_number(expected, f"{origin} expected")
    if abs(left - right) > tolerance * max(1.0, abs(right)):
        raise InspectionError(f"{origin} differs: {left} != {right}")
    return left


RENDER_TARGET_FIELDS = (
    "scale", "hlsl_full_clamp_abs", "comfort_clamp_abs_pct",
    "mean_abs_disparity_pct", "p95_abs_disparity_pct",
    "exact_pop_spread_pct", "clamped_pixel_pct",
)


def validate_render_target(actual, expected, origin: str):
    if not isinstance(actual, dict):
        raise InspectionError(f"{origin} is missing")
    for field in RENDER_TARGET_FIELDS:
        numeric_close(actual.get(field), expected[field], f"{origin}.{field}")
    if finite_number(actual["hlsl_full_clamp_abs"], origin) <= 0.0:
        raise InspectionError(f"{origin} has a non-positive comfort clamp")
    for field in (
            "comfort_clamp_abs_pct", "mean_abs_disparity_pct",
            "p95_abs_disparity_pct", "exact_pop_spread_pct"):
        if finite_number(actual[field], f"{origin}.{field}") < 0.0:
            raise InspectionError(f"{origin}.{field} is negative")
    clamped = finite_number(
        actual["clamped_pixel_pct"], f"{origin}.clamped_pixel_pct"
    )
    if not 0.0 <= clamped <= 100.0:
        raise InspectionError(f"{origin}.clamped_pixel_pct is outside 0..100")


def expected_geometry_keys(plan_contract, input_variant, source_signature):
    return {
        geometry_contract.tuple_key(value)
        for value in plan_contract["geometry_manifest"]["tuples"]
        if value["color_mode"] == input_variant["color_mode"] and (
            value["source_width"], value["source_height"],
            value["model_input_width"], value["model_input_height"],
            value["depth_short_side"], value["depth_max_aspect"],
        ) == source_signature
    }


FLOW_SELECTION_KEYS = {
    "contract", "flow_support_contract", "flow_support_metric_sha256",
    "preferred_pair", "minimum_support", "search_radius_frames",
    "search_order", "nominal_source_label_frame_id",
    "selected_source_label_frame_id", "selected_offset_frames",
    "selected_previous_source_frame_id", "selected_pair_flow_support",
}


def validate_flow_selection(value, origin: str):
    if not isinstance(value, dict) or set(value) != FLOW_SELECTION_KEYS:
        raise InspectionError(f"{origin} flow-selection contract is incomplete")
    metric_hash = chug_prepare.flow_support_metric_sha256()
    expected = {
        "contract": chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT,
        "flow_support_contract": chug_prepare.FLOW_SUPPORT_CONTRACT,
        "flow_support_metric_sha256": metric_hash,
        "preferred_pair": "previous-source-frame-to-label-frame",
        "minimum_support": chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT,
        "search_radius_frames": chug_prepare.FLOW_SUPPORT_SEARCH_RADIUS_FRAMES,
        "search_order": "nominal-then-negative-positive-by-distance",
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise InspectionError(f"{origin}.{key} differs")
    integer_fields = (
        "nominal_source_label_frame_id", "selected_source_label_frame_id",
        "selected_offset_frames", "selected_previous_source_frame_id",
    )
    if any(type(value.get(key)) is not int for key in integer_fields):
        raise InspectionError(f"{origin} flow-selection frame IDs are invalid")
    nominal = value["nominal_source_label_frame_id"]
    selected = value["selected_source_label_frame_id"]
    if (value["selected_offset_frames"] != selected - nominal or
            value["selected_previous_source_frame_id"] != selected - 1 or
            abs(selected - nominal) >
            chug_prepare.FLOW_SUPPORT_SEARCH_RADIUS_FRAMES):
        raise InspectionError(f"{origin} flow-selection frame relation differs")
    support = finite_number(
        value.get("selected_pair_flow_support"),
        f"{origin}.selected_pair_flow_support",
    )
    if not chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT <= support <= 1.0:
        raise InspectionError(f"{origin} flow support is below the required floor")
    return value


class DistributionAccumulator:
    def __init__(self):
        self.scale = []
        self.confidence = []
        self.reliability = []
        self.pop = []
        self.clamped = []
        self.mean_abs_disparity = []
        self.p95_abs_disparity = []
        self.comfort_clamp = []
        self.actionable = 0
        self.identity = 0
        self.identity_feasible = 0
        self.identity_infeasible = 0
        self.scale_counts = Counter()
        self.condition_counts = Counter()
        self.violations = Counter()
        self.source_ids = set()
        self.shot_ids = set()

    def add(self, target, condition, source_id=None, shot_id=None):
        scale = finite_number(
            target.get("safe_scale_ceiling"), "safe scale ceiling"
        )
        confidence = finite_number(
            target.get("ceiling_confidence"), "ceiling confidence"
        )
        reliability = finite_number(
            target.get("safety_margin_reliability"),
            "safety margin reliability",
        )
        actionable = scale - 1.0 >= ACTION_EPSILON
        if confidence not in (0.0, 1.0) or confidence != float(actionable):
            raise InspectionError(
                "condition confidence is not the hard actionable target"
            )
        feasible = target.get("identity_feasible")
        if not isinstance(feasible, bool):
            raise InspectionError("condition identity feasibility is not boolean")
        if not feasible and (
                abs(scale - 1.0) > 1e-6 or confidence != 0.0 or
                abs(reliability) > 1e-6):
            raise InspectionError(
                "identity-infeasible condition is not a confidence-zero no-op"
            )
        render = target.get("safe_ceiling_render_target")
        if not isinstance(render, dict):
            raise InspectionError("condition lacks rendered-disparity safety evidence")
        render_scale = finite_number(render.get("scale"), "render target scale")
        if abs(render_scale - scale) > 1e-6:
            raise InspectionError("render target scale differs from safe ceiling")
        pop = finite_number(
            render.get("exact_pop_spread_pct"), "rendered pop spread"
        )
        clamped = finite_number(
            render.get("clamped_pixel_pct"), "rendered clamp burden"
        )
        mean_abs = finite_number(
            render.get("mean_abs_disparity_pct"), "rendered mean disparity"
        )
        p95_abs = finite_number(
            render.get("p95_abs_disparity_pct"), "rendered p95 disparity"
        )
        comfort = finite_number(
            render.get("comfort_clamp_abs_pct"), "comfort disparity limit"
        )
        if not 0.0 <= clamped <= 100.0:
            raise InspectionError("rendered clamp burden is outside 0..100 percent")
        if min(pop, mean_abs, p95_abs, comfort) < 0.0:
            raise InspectionError("rendered disparity summary is negative")
        if not 0.0 <= reliability <= 1.0:
            raise InspectionError("safety margin reliability is outside 0..1")
        if not actionable and abs(reliability) > NUMERIC_TOLERANCE:
            raise InspectionError("identity target has nonzero safety reliability")
        self.scale.append(scale)
        self.confidence.append(confidence)
        self.reliability.append(reliability)
        self.pop.append(pop)
        self.clamped.append(clamped)
        self.mean_abs_disparity.append(mean_abs)
        self.p95_abs_disparity.append(p95_abs)
        self.comfort_clamp.append(comfort)
        self.actionable += int(actionable)
        self.identity += int(not actionable)
        self.identity_feasible += int(feasible)
        self.identity_infeasible += int(not feasible)
        self.scale_counts[f"{scale:.3f}"] += 1
        self.condition_counts[condition] += 1
        for item in target.get("identity_infeasible_variants", []):
            for violation in item.get("violations", []):
                self.violations[str(violation)] += 1
        if source_id is not None:
            self.source_ids.add(tuple(source_id))
        if shot_id is not None:
            self.shot_ids.add(tuple(shot_id))

    def merge(self, other):
        for field in (
                "scale", "confidence", "reliability", "pop", "clamped",
                "mean_abs_disparity", "p95_abs_disparity", "comfort_clamp"):
            getattr(self, field).extend(getattr(other, field))
        for field in (
                "actionable", "identity", "identity_feasible",
                "identity_infeasible"):
            setattr(self, field, getattr(self, field) + getattr(other, field))
        self.scale_counts.update(other.scale_counts)
        self.condition_counts.update(other.condition_counts)
        self.violations.update(other.violations)
        self.source_ids.update(other.source_ids)
        self.shot_ids.update(other.shot_ids)
        return self

    def result(self):
        total = len(self.scale)
        return {
            "policy_samples": total,
            "sample_count": total,
            "source_frame_count": len(self.source_ids),
            "shot_count": len(self.shot_ids),
            "actionable": self.actionable,
            "identity": self.identity,
            "actionable_pct": (
                100.0 * self.actionable / total if total else 0.0
            ),
            "identity_feasible": self.identity_feasible,
            "identity_infeasible": self.identity_infeasible,
            "condition_counts": dict(sorted(self.condition_counts.items())),
            "scale_counts": dict(sorted(self.scale_counts.items())),
            "scale": percentile_summary(self.scale),
            "confidence": percentile_summary(self.confidence),
            "safety_margin_reliability": percentile_summary(self.reliability),
            "rendered_pop_spread_pct": percentile_summary(self.pop),
            "rendered_mean_abs_disparity_pct": percentile_summary(
                self.mean_abs_disparity
            ),
            "rendered_p95_abs_disparity_pct": percentile_summary(
                self.p95_abs_disparity
            ),
            "comfort_clamp_abs_pct": percentile_summary(self.comfort_clamp),
            "clamped_pixel_pct": percentile_summary(self.clamped),
            "identity_violations": dict(self.violations.most_common()),
        }


def validate_condition_safety(target, evidence_rows, origin: str):
    if len(evidence_rows) != 2:
        raise InspectionError(f"{origin} does not have exactly two geometries")
    lower = 1.0 - ART_SCALE_DELTA_MAX
    upper = 1.0 + ART_SCALE_DELTA_MAX
    identity_flags = []
    safe_mins = []
    safe_maxes = []
    reliabilities = []
    clamps = set()
    for index, item in enumerate(evidence_rows):
        evidence = item["evidence"]
        feasible = evidence.get("identity_feasible")
        if not isinstance(feasible, bool):
            raise InspectionError(
                f"{origin}:geometry[{index}] identity feasibility is not boolean"
            )
        identity_flags.append(feasible)
        safe_min = finite_number(
            evidence.get("safe_scale_min"),
            f"{origin}:geometry[{index}].safe_scale_min",
        )
        safe_max = finite_number(
            evidence.get("safe_scale_max"),
            f"{origin}:geometry[{index}].safe_scale_max",
        )
        reliability = finite_number(
            evidence.get("safety_margin_reliability"),
            f"{origin}:geometry[{index}].safety_margin_reliability",
        )
        clamp_abs = finite_number(
            evidence.get("artistic_full_clamp_abs"),
            f"{origin}:geometry[{index}].artistic_full_clamp_abs",
        )
        if (not lower <= safe_min <= upper or
                not lower <= safe_max <= upper or safe_min > safe_max):
            raise InspectionError(
                f"{origin}:geometry[{index}] safe frontier is outside model bounds"
            )
        if feasible and not (
                safe_min <= 1.0 + NUMERIC_TOLERANCE and
                safe_max >= 1.0 - NUMERIC_TOLERANCE):
            raise InspectionError(
                f"{origin}:geometry[{index}] feasible frontier excludes identity"
            )
        if not 0.0 <= reliability <= 1.0:
            raise InspectionError(
                f"{origin}:geometry[{index}] reliability is outside 0..1"
            )
        if clamp_abs <= 0.0:
            raise InspectionError(
                f"{origin}:geometry[{index}] comfort clamp is non-positive"
            )
        safe_mins.append(safe_min)
        safe_maxes.append(safe_max)
        reliabilities.append(reliability)
        clamps.add(clamp_abs)
    if len(clamps) != 1:
        raise InspectionError(f"{origin} geometries disagree on comfort clamp")

    identity_feasible = all(identity_flags)
    if identity_feasible:
        safe_min = max(safe_mins)
        ceiling = min(safe_maxes)
        if (safe_min > 1.0 + NUMERIC_TOLERANCE or
                ceiling < 1.0 - NUMERIC_TOLERANCE):
            raise InspectionError(f"{origin} two-geometry frontier excludes identity")
    else:
        safe_min = 1.0
        ceiling = 1.0
    actionable = ceiling - 1.0 >= ACTION_EPSILON
    reliability = min(reliabilities) if actionable else 0.0
    confidence = 1.0 if actionable else 0.0
    expected_styles = {
        "clean": 1.0,
        "balanced": 1.0 + 0.5 * (ceiling - 1.0),
        "immersive": ceiling,
    }
    for field, expected in (
            ("safe_scale_min", safe_min), ("safe_scale_max", ceiling),
            ("safe_scale_ceiling", ceiling),
            ("ceiling_confidence", confidence),
            ("safety_margin_reliability", reliability)):
        numeric_close(target.get(field), expected, f"{origin}.{field}")
    if target.get("identity_feasible") is not identity_feasible:
        raise InspectionError(f"{origin}.identity_feasible differs from evidence")
    styles = target.get("style_targets")
    if not isinstance(styles, dict) or set(styles) != set(expected_styles):
        raise InspectionError(f"{origin}.style_targets is incomplete")
    for name, expected in expected_styles.items():
        numeric_close(styles.get(name), expected, f"{origin}.style_targets.{name}")

    geometry_targets = []
    geometry_style_targets = {name: [] for name in expected_styles}
    for index, item in enumerate(evidence_rows):
        evidence = item["evidence"]
        raw = item["raw"]
        geometry = item["geometry"]
        clamp_abs = float(evidence["artistic_full_clamp_abs"])
        expected = label_merge.render_target(
            raw, geometry, ceiling, clamp_abs
        )
        validate_render_target(
            evidence.get("safe_ceiling_render_target"), expected,
            f"{origin}:geometry[{index}].safe_ceiling_render_target",
        )
        geometry_targets.append(expected)
        rendered_styles = evidence.get("style_render_targets")
        if (not isinstance(rendered_styles, dict) or
                set(rendered_styles) != set(expected_styles)):
            raise InspectionError(
                f"{origin}:geometry[{index}].style_render_targets is incomplete"
            )
        for name, scale in expected_styles.items():
            style_expected = label_merge.render_target(
                raw, geometry, scale, clamp_abs
            )
            validate_render_target(
                rendered_styles.get(name), style_expected,
                f"{origin}:geometry[{index}].style_render_targets.{name}",
            )
            geometry_style_targets[name].append(style_expected)

    conservative = label_merge.conservative_target(geometry_targets, ceiling)
    validate_render_target(
        target.get("safe_ceiling_render_target"), conservative,
        f"{origin}.safe_ceiling_render_target",
    )
    numeric_close(
        target.get("safe_ceiling_exact_pop_spread_pct"),
        conservative["exact_pop_spread_pct"],
        f"{origin}.safe_ceiling_exact_pop_spread_pct",
    )
    target_styles = target.get("style_render_targets")
    if not isinstance(target_styles, dict) or set(target_styles) != set(
            expected_styles):
        raise InspectionError(f"{origin}.style_render_targets is incomplete")
    for name, scale in expected_styles.items():
        expected = label_merge.conservative_target(
            geometry_style_targets[name], scale
        )
        validate_render_target(
            target_styles.get(name), expected,
            f"{origin}.style_render_targets.{name}",
        )
    return {
        "safe_scale_min": safe_min,
        "safe_scale_ceiling": ceiling,
        "ceiling_confidence": confidence,
        "safety_margin_reliability": reliability,
        "identity_feasible": identity_feasible,
        "geometry_targets": geometry_targets,
        "conservative_render_target": conservative,
    }


class HDRAccumulator:
    """Streaming exact proportions plus bounded luminance percentile samples."""

    def __init__(self):
        self.component_count = 0
        self.pixel_count = 0
        self.negative_components = 0
        self.superwhite_components = 0
        self.over_1000_nit_components = 0
        self.fp16_extreme_components = 0
        self.negative_luminance = 0
        self.superwhite_luminance = 0
        self.over_1000_nit_luminance = 0
        self.preview_pixels = 0
        self.preview_black = 0
        self.preview_saturated = 0
        self.samples = []
        self.frames = 0

    def add(self, model_path: Path, width: int, height: int,
            preview_path: Path):
        values = np.fromfile(model_path, dtype="<f2")
        expected = width * height * 4
        if values.size != expected:
            raise InspectionError(
                f"native FP16 element count differs: {model_path}"
            )
        if not np.isfinite(values).all():
            raise InspectionError(
                f"native FP16 contains non-finite values: {model_path}"
            )
        rgba = values.reshape(height, width, 4)
        rgb = rgba[..., :3].astype(np.float32)
        luminance = np.sum(rgb * REC709_LUMA, axis=2, dtype=np.float32)
        self.component_count += rgb.size
        self.pixel_count += luminance.size
        self.negative_components += int(np.count_nonzero(rgb < 0.0))
        self.superwhite_components += int(np.count_nonzero(rgb > 1.0))
        self.over_1000_nit_components += int(np.count_nonzero(rgb > 12.5))
        self.fp16_extreme_components += int(np.count_nonzero(
            np.abs(rgb) >= FP16_MAX
        ))
        self.negative_luminance += int(np.count_nonzero(luminance < 0.0))
        self.superwhite_luminance += int(np.count_nonzero(luminance > 1.0))
        self.over_1000_nit_luminance += int(np.count_nonzero(luminance > 12.5))
        flat = luminance.reshape(-1)
        stride = max(1, flat.size // 4096)
        self.samples.extend((flat[::stride][:4096] * 80.0).tolist())
        try:
            with Image.open(preview_path) as image:
                image.load()
                preview = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as error:
            raise InspectionError(
                f"cannot decode native HDR preview: {preview_path}"
            ) from error
        if preview.shape[:2] != (height, width):
            raise InspectionError(
                f"native HDR preview dimensions differ: {preview_path}"
            )
        peak = np.max(preview, axis=2)
        self.preview_pixels += peak.size
        self.preview_black += int(np.count_nonzero(peak == 0))
        self.preview_saturated += int(np.count_nonzero(peak == 255))
        self.frames += 1

    @staticmethod
    def _fraction(count, total):
        return float(count / total) if total else 0.0

    def result(self):
        luminance = percentile_summary(self.samples)
        result = {
            "frames": self.frames,
            "component_count": self.component_count,
            "pixel_count": self.pixel_count,
            "negative_component_fraction": self._fraction(
                self.negative_components, self.component_count
            ),
            "superwhite_component_fraction": self._fraction(
                self.superwhite_components, self.component_count
            ),
            "over_1000_nit_component_fraction": self._fraction(
                self.over_1000_nit_components, self.component_count
            ),
            "fp16_extreme_component_fraction": self._fraction(
                self.fp16_extreme_components, self.component_count
            ),
            "negative_luminance_fraction": self._fraction(
                self.negative_luminance, self.pixel_count
            ),
            "superwhite_luminance_fraction": self._fraction(
                self.superwhite_luminance, self.pixel_count
            ),
            "over_1000_nit_luminance_fraction": self._fraction(
                self.over_1000_nit_luminance, self.pixel_count
            ),
            "preview_black_fraction": self._fraction(
                self.preview_black, self.preview_pixels
            ),
            "preview_saturated_fraction": self._fraction(
                self.preview_saturated, self.preview_pixels
            ),
            "luminance_nits_sampled": luminance,
            "luminance_sampling": (
                "deterministic at most 4096 evenly spaced pixels per frame; "
                "all proportions above are exact"
            ),
        }
        if result["fp16_extreme_component_fraction"] != 0.0:
            raise InspectionError(
                "native FP16 reaches a representable clipping extreme"
            )
        if result["superwhite_luminance_fraction"] <= 0.0:
            raise InspectionError(
                "native branch has no luminance above the 80-nit scRGB SDR white"
            )
        return result


def dataset_chain(root: Path, dataset_row, expected_kind: str):
    split = dataset_row.get("split")
    if split not in SPLITS:
        raise InspectionError("dataset plan includes a sealed/unsupported split")
    root = assert_working_path(root, f"{split} dataset")
    manifest_path = root / "dataset_manifest.json"
    manifest_hash = require_hash(
        manifest_path, dataset_row.get("dataset_manifest_sha256"),
        f"{split} dataset manifest",
    )
    manifest = load_json(manifest_path, "dataset manifest")
    if (manifest.get("schema") != 2 or manifest.get("split") != split or
            manifest.get("source_kind") != expected_kind):
        raise InspectionError(f"{split} dataset contract differs")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise InspectionError(f"{split} dataset has no sequences")
    sequence_names = [
        row.get("clip") if isinstance(row, dict) else None for row in sequences
    ]
    planned = dataset_row.get("clips")
    if (not isinstance(planned, list) or sequence_names != planned or
            len(planned) != len(set(planned))):
        raise InspectionError(f"{split} dataset clip set/order differs")
    clip_manifest_path = root / clip_hashes.MANIFEST_NAME
    clip_manifest_hash = require_hash(
        clip_manifest_path, dataset_row.get("clip_hash_manifest_sha256"),
        f"{split} clip hash manifest",
    )
    try:
        clip_manifest = clip_hashes.load_manifest(clip_manifest_path)
        clip_hashes.verify_selected_clips(
            clip_manifest_path, root, planned, full=True
        )
    except clip_hashes.ClipHashManifestError as error:
        raise InspectionError(
            f"{split} clip hash chain failed: {error}"
        ) from error
    return {
        "root": root,
        "split": split,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "clip_hash_manifest": clip_manifest,
        "clip_hash_manifest_path": clip_manifest_path,
        "clip_hash_manifest_sha256": clip_manifest_hash,
        "clip_hash_semantic_content_sha256": clip_manifest[
            clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
        ],
        "sequences": sequences,
        "clips": planned,
    }


def validate_depth_publications(plan, datasets, workspace: Path, origin: str):
    """Authenticate every displayed depth PNG through its publication manifest."""
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise InspectionError(f"{origin} plan has no subprocess steps")
    depth_steps = [
        step for step in steps
        if isinstance(step, dict) and step.get("kind") == "depth"
    ]
    expected = {}
    for dataset_key, dataset in datasets.items():
        for variant in dataset["input_variants"]:
            expected[(dataset_key, input_variant_key(variant))] = (
                dataset, variant
            )
    observed = {}
    evidence_index = {}
    workspace = Path(workspace).resolve(strict=True)
    for step_index, step in enumerate(depth_steps):
        metadata = step.get("metadata")
        if not isinstance(metadata, dict):
            raise InspectionError(f"{origin} depth step {step_index} lacks metadata")
        dataset_key = metadata.get("dataset")
        variant = metadata.get("input_variant")
        variant_hash = input_variant_key(variant)
        key = (dataset_key, variant_hash)
        if key not in expected or key in observed:
            raise InspectionError(
                f"{origin} has an unexpected/duplicate depth publication {key}"
            )
        dataset, expected_variant = expected[key]
        if variant != expected_variant:
            raise InspectionError(f"{origin} depth-step input variant differs")
        output = Path(step.get("output", "")).resolve(strict=True)
        if not output.is_relative_to(workspace / "depth"):
            raise InspectionError(
                f"{origin} depth publication escapes the workspace: {output}"
            )
        manifest_path = output / "depth_run_manifest.json"
        manifest = load_json(manifest_path, "depth-run manifest")
        if (manifest.get("schema") != depth_run.DEPTH_RUN_MANIFEST_SCHEMA or
                manifest.get("purpose") != "artistic-policy depth supervision" or
                Path(manifest.get("suite", "")).resolve() != dataset["root"] or
                manifest.get("suite_manifest_sha256") !=
                dataset["manifest_sha256"] or
                manifest.get("input_variant") != variant or
                manifest.get("input_variant_sha256") != variant_hash or
                manifest.get("depth_input_color_contract_sha256") !=
                input_color.color_contract_sha256()):
            raise InspectionError(
                f"{origin} depth publication contract differs: {manifest_path}"
            )
        for field in (
                "dataset_manifest_sha256", "clip_hash_manifest_sha256"):
            expected_value = (
                dataset["manifest_sha256"] if field ==
                "dataset_manifest_sha256" else
                dataset["clip_hash_manifest_sha256"]
            )
            if metadata.get(field) != expected_value:
                raise InspectionError(
                    f"{origin} depth-step {field} differs for {dataset_key}"
                )
        if metadata.get("dataset_root") is not None and Path(
                metadata["dataset_root"]).resolve() != dataset["root"]:
            raise InspectionError(f"{origin} depth-step dataset root differs")
        rows = manifest.get("clips")
        if not isinstance(rows, list):
            raise InspectionError(f"{origin} depth publication has no clip rows")
        by_clip = {
            row.get("clip"): row for row in rows if isinstance(row, dict)
        }
        if (set(by_clip) != set(dataset["clips"]) or
                len(by_clip) != len(rows) or
                manifest.get("clip_count") != len(rows)):
            raise InspectionError(
                f"{origin} depth publication clip set differs for {dataset_key}"
            )
        for clip in dataset["clips"]:
            clip_root = output / clip
            row = by_clip[clip]
            try:
                current_identity = depth_run.depth_artifact_identity(clip_root)
            except (OSError, RuntimeError) as error:
                raise InspectionError(
                    f"{origin} cannot authenticate depth artifacts: {clip_root}"
                ) from error
            if not depth_run.artifact_identity_matches(row, current_identity):
                raise InspectionError(
                    f"{origin} depth artifact identity differs: {clip_root}"
                )
            records = {
                item.get("path"): item for item in row["artifact_files"]
                if isinstance(item, dict)
            }
            depth_records = {
                int(match.group(1)): (name, record)
                for name, record in records.items()
                for match in [re.fullmatch(r"depth_(\d+)\.png", name or "")]
                if match is not None
            }
            if len(depth_records) != int(row.get("frames", -1)):
                raise InspectionError(
                    f"{origin} depth frame count differs: {clip_root}"
                )
            for frame_id, (name, record) in depth_records.items():
                path = clip_root / name
                require_hash(
                    path, record.get("sha256"),
                    f"{origin} displayed depth {dataset_key}/{clip}/{frame_id}",
                )
                evidence_index[(dataset_key, variant_hash, clip, frame_id)] = {
                    "path": path.resolve(),
                    "sha256": record["sha256"],
                    "manifest": manifest_path.resolve(),
                    "manifest_sha256": sha256(manifest_path),
                    "artifact_content_sha256": row["artifact_content_sha256"],
                }
        observed[key] = str(manifest_path.resolve())
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise InspectionError(f"{origin} depth publications are incomplete: {missing}")
    return evidence_index


def validate_merged_bundle(workspace: Path, branch: str, dataset_key: str,
                           dataset, expected_rows: int,
                           expected_conditions: int, plan_contract,
                           depth_index):
    root = workspace / "merged" / dataset_key
    labels_path = root / "labels.jsonl"
    summary_path = root / "summary.json"
    contract_path = root / "label_fitter_contract.json"
    if not all(path.is_file() for path in (
            labels_path, summary_path, contract_path)):
        raise InspectionError(f"merged label bundle is incomplete: {root}")
    summary = load_json(summary_path, "merged label summary")
    contract = load_json(contract_path, "merged label contract")
    labels_hash = sha256(labels_path)
    contract_hash = sha256(contract_path)
    if (summary.get("schema") != LABEL_SCHEMA or
            contract.get("schema") != LABEL_SCHEMA or
            contract.get("policy_contract") != POLICY_CONTRACT or
            summary.get("labels_sha256") != labels_hash or
            summary.get("label_fitter_contract_sha256") != contract_hash or
            summary.get("condition_target_contract") !=
            CONDITION_TARGET_CONTRACT or
            contract.get("condition_target_contract") !=
            CONDITION_TARGET_CONTRACT):
        raise InspectionError(f"merged label hash/contract chain differs: {root}")
    try:
        label_merge.validate_label_fitter_code(
            contract.get("code"), LABEL_SCHEMA, str(contract_path)
        )
    except RuntimeError as error:
        raise InspectionError(str(error)) from error
    if (contract.get("deployment_geometry_allowlist") !=
            plan_contract["geometry_manifest"] or
            contract.get("deployment_geometry_allowlist_sha256") !=
            plan_contract["geometry_hash"] or
            contract.get("input_variant_manifest") !=
            plan_contract["input_manifest"] or
            contract.get("input_variant_manifest_sha256") !=
            plan_contract["input_hash"] or
            contract.get("depth_input_color_contract_sha256") !=
            input_color.color_contract_sha256()):
        raise InspectionError(
            f"merged label deployment/input contract differs: {root}"
        )
    rows = load_jsonl(labels_path)
    if len(rows) != expected_rows or summary.get("accepted") != expected_rows:
        raise InspectionError(
            f"{dataset_key} merged RGB cardinality {len(rows)} != {expected_rows}"
        )
    source_hashes = set()
    source_ids = set()
    film_ids = set()
    contacts = []
    distribution = DistributionAccumulator()
    condition_accumulators = {}
    artifact_paths = set()
    expected_variant_kinds = None
    for index, row in enumerate(rows, 1):
        origin = f"{labels_path}:{index}"
        if (row.get("label_schema") != LABEL_SCHEMA or
                row.get("policy_contract") != POLICY_CONTRACT or
                row.get("condition_target_contract") !=
                CONDITION_TARGET_CONTRACT or row.get("split") != dataset["split"]):
            raise InspectionError(f"{origin}: merged row contract differs")
        if (row.get("deployment_geometry_allowlist_sha256") !=
                plan_contract["geometry_hash"] or
                row.get("input_variant_manifest") !=
                plan_contract["input_manifest"] or
                row.get("input_variant_manifest_sha256") !=
                plan_contract["input_hash"] or
                row.get("depth_input_color_contract_sha256") !=
                input_color.color_contract_sha256()):
            raise InspectionError(f"{origin}: row deployment/input contract differs")
        if row.get("film_id") != dataset["manifest"].get("production_id"):
            raise InspectionError(f"{origin}: film identity differs from dataset")
        source = Path(row.get("source", ""))
        source_hash = row.get("source_sha256")
        if not source.is_file() or sha256(source) != source_hash:
            raise InspectionError(f"{origin}: source preview is missing or changed")
        if source_hash in source_hashes:
            raise InspectionError(f"{origin}: duplicate source SHA-256")
        source_hashes.add(source_hash)
        source_id = (
            str(row.get("film_id")), str(row.get("clip")), int(row.get("frame"))
        )
        if source_id in source_ids:
            raise InspectionError(f"{origin}: duplicate source ID")
        source_ids.add(source_id)
        film_ids.add(str(row["film_id"]))
        targets = row.get("input_condition_targets")
        if not isinstance(targets, list) or len(targets) != expected_conditions:
            raise InspectionError(
                f"{origin}: expected {expected_conditions} input conditions"
            )
        target_by_input = {}
        kinds = Counter()
        for target_index, target in enumerate(targets):
            if (not isinstance(target, dict) or target.get("schema") != 1 or
                    target.get("contract") != CONDITION_TARGET_CONTRACT):
                raise InspectionError(
                    f"{origin}: condition {target_index} contract differs"
                )
            variant = target.get("input_variant")
            variant_hash = input_variant_key(variant)
            if target.get("input_variant_sha256") != variant_hash:
                raise InspectionError(f"{origin}: stale input condition hash")
            if variant_hash in target_by_input:
                raise InspectionError(f"{origin}: duplicate input condition")
            if target.get("deployment_geometry_variant_count") != 2:
                raise InspectionError(
                    f"{origin}: condition does not declare two geometry artifacts"
                )
            target_by_input[variant_hash] = target
            kinds[variant["kind"]] += 1
        if branch == "sdr_origin":
            raw_levels = {
                target["input_variant"]["windows_sdr_white_level_raw"]
                for target in targets
                if target["input_variant"]["kind"] ==
                input_color.INPUT_KIND_WINDOWS_HDR
            }
            if (set(kinds) != EXPECTED_SDR_CONDITIONS or
                    kinds[input_color.INPUT_KIND_SDR] != 1 or
                    kinds[input_color.INPUT_KIND_WINDOWS_HDR] != 3 or
                    raw_levels != {1000, 2500, 6000}):
                raise InspectionError(f"{origin}: SDR-origin conditions differ")
        elif set(kinds) != {input_color.INPUT_KIND_NATIVE_PQ} or sum(
                kinds.values()) != 1:
            raise InspectionError(f"{origin}: native-PQ condition differs")
        signature = tuple(sorted(kinds.items()))
        if expected_variant_kinds is None:
            expected_variant_kinds = signature
        elif signature != expected_variant_kinds:
            raise InspectionError(f"{origin}: condition family changes within bundle")
        variants = row.get("deployment_geometry_variants")
        if (not isinstance(variants, list) or
                len(variants) != expected_conditions * 2):
            raise InspectionError(
                f"{origin}: geometry evidence cardinality differs"
            )
        by_input = {}
        for variant_index, evidence in enumerate(variants):
            if not isinstance(evidence, dict):
                raise InspectionError(f"{origin}: malformed geometry evidence")
            input_variant = evidence.get("input_variant")
            variant_hash = input_variant_key(input_variant)
            if evidence.get("input_variant_sha256") != variant_hash:
                raise InspectionError(f"{origin}: geometry input hash differs")
            geometry = evidence.get("geometry")
            geometry_hash = geometry_identity(geometry)
            path = Path(evidence.get("baseline_unclamped_disparity", ""))
            expected_hash = evidence.get("baseline_unclamped_disparity_sha256")
            if not path.is_file() or not isinstance(expected_hash, str):
                raise InspectionError(f"{origin}: geometry artifact is missing")
            shape = (
                int(geometry["disparity_raster_height"]),
                int(geometry["disparity_raster_width"]),
            )
            raw = load_float_texture(path, expected_hash, shape)
            resolved = str(path.resolve())
            if resolved in artifact_paths:
                raise InspectionError(
                    f"{origin}: geometry artifact is reused by another source"
                )
            artifact_paths.add(resolved)
            by_input.setdefault(variant_hash, []).append({
                "geometry_key": geometry_hash,
                "path": resolved,
                "evidence": evidence,
                "geometry": geometry,
                "raw": raw,
            })
        if set(by_input) != set(target_by_input):
            raise InspectionError(f"{origin}: conditions/evidence differ")
        for variant_hash, evidence in by_input.items():
            if (len(evidence) != 2 or
                    len({item["geometry_key"] for item in evidence}) != 2 or
                    len({item["path"] for item in evidence}) != 2):
                raise InspectionError(
                    f"{origin}: each condition needs two distinct geometry artifacts"
                )
            target = target_by_input[variant_hash]
            variant = target["input_variant"]
            first_geometry = evidence[0]["geometry"]
            source_signature = (
                first_geometry["source_width"], first_geometry["source_height"],
                first_geometry["model_input_width"],
                first_geometry["model_input_height"],
                first_geometry["depth_short_side"],
                first_geometry["depth_max_aspect"],
            )
            expected_geometries = expected_geometry_keys(
                plan_contract, variant, source_signature
            )
            actual_geometries = {
                item["geometry_key"] for item in evidence
            }
            if actual_geometries != expected_geometries or len(
                    expected_geometries) != 2:
                raise InspectionError(
                    f"{origin}: condition does not use the exact planned geometries"
                )
            safety = validate_condition_safety(
                target, evidence,
                f"{origin}:condition[{condition_name(variant)}]",
            )
            condition = condition_name(variant)
            distribution.add(
                target, condition, source_id,
                (str(row["film_id"]), str(row["clip"])),
            )
            accumulator = condition_accumulators.setdefault(
                condition, DistributionAccumulator()
            )
            accumulator.add(
                target, condition, source_id,
                (str(row["film_id"]), str(row["clip"])),
            )
            depth_key = (
                dataset_key, variant_hash, str(row["clip"]), int(row["frame"])
            )
            if depth_key not in depth_index:
                raise InspectionError(
                    f"{origin}: condition lacks authenticated depth evidence"
                )
            target["_inspection_geometry_targets"] = safety[
                "geometry_targets"
            ]
        if branch == "native_pq":
            model_source = Path(row.get("model_source", ""))
            if (row.get("model_source_encoding") !=
                    native_hdr_capture.CAPTURE_ENCODING or
                    not model_source.is_file() or
                    sha256(model_source) != row.get("model_source_sha256")):
                raise InspectionError(f"{origin}: native model source differs")
        contacts.append({
            "branch": branch,
            "split": dataset["split"],
            "dataset_key": dataset_key,
            "row": row,
            "depth_by_input": {
                variant_hash: depth_index[(
                    dataset_key, variant_hash, str(row["clip"]), int(row["frame"])
                )]
                for variant_hash in target_by_input
            },
        })
    return {
        "dataset_key": dataset_key,
        "split": dataset["split"],
        "labels": str(labels_path.resolve()),
        "labels_sha256": labels_hash,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract_hash,
        "rgb_rows": len(rows),
        "policy_samples": len(rows) * expected_conditions,
        "conditions_per_rgb": expected_conditions,
        "geometry_artifacts_per_condition": 2,
        "source_hashes": source_hashes,
        "source_ids": source_ids,
        "film_ids": film_ids,
        "distribution": distribution.result(),
        "condition_distributions": {
            name: accumulator.result()
            for name, accumulator in sorted(condition_accumulators.items())
        },
        "_condition_accumulators": condition_accumulators,
        "contacts": contacts,
    }


def validate_sdr_workspace(workspace: Path):
    workspace = Path(workspace).resolve(strict=True)
    plan_path = workspace / SDR_PLAN
    plan = load_json(plan_path, "SDR orchestration plan")
    if (plan.get("schema") != SDR_PLAN_SCHEMA or
            plan.get("contract") != SDR_PLAN_CONTRACT):
        raise InspectionError("unsupported SDR orchestration plan")
    plan_contract = validate_plan_contracts(plan, "SDR")
    if plan.get("workspace") is not None and Path(
            plan["workspace"]).resolve() != workspace:
        raise InspectionError("SDR plan workspace path differs")
    bootstrap_path = assert_working_path(
        Path(plan.get("bootstrap_manifest", "")), "SDR bootstrap manifest"
    )
    bootstrap_hash = require_hash(
        bootstrap_path, plan.get("bootstrap_manifest_sha256"),
        "SDR bootstrap manifest",
    )
    bootstrap = load_json(bootstrap_path, "SDR bootstrap manifest")
    if (bootstrap.get("schema") != 1 or
            bootstrap.get("preparation_contract") != SDR_BOOTSTRAP_CONTRACT):
        raise InspectionError("unsupported SDR bootstrap contract")
    datasets = plan.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 4:
        raise InspectionError("SDR plan must name four train/development datasets")
    bootstrap_rows = bootstrap.get("datasets")
    if not isinstance(bootstrap_rows, list):
        raise InspectionError("SDR bootstrap has no dataset rows")
    bootstrap_by_key = {
        (row.get("source"), row.get("split")): row for row in bootstrap_rows
        if isinstance(row, dict)
    }
    expected_keys = {
        ("reds", "training"), ("reds", "development"),
        ("spring", "training"), ("spring", "development"),
    }
    if set(bootstrap_by_key) != expected_keys:
        raise InspectionError("SDR bootstrap dataset set differs")
    sdr_variants = [
        input_color.sdr_input_variant(),
        *(input_color.windows_hdr_input_variant(value)
          for value in (1000, 2500, 6000)),
    ]
    dataset_chains = {}
    for row in datasets:
        source = row.get("source")
        split = row.get("split")
        key = (source, split)
        if key not in expected_keys:
            raise InspectionError("SDR plan contains an unexpected dataset")
        bootstrap_row = bootstrap_by_key[key]
        if (row.get("dataset_manifest_sha256") != bootstrap_row.get(
                "dataset_manifest_sha256") or row.get(
                "clip_hash_manifest_sha256") != bootstrap_row.get(
                "clip_hash_manifest_sha256")):
            raise InspectionError(
                f"SDR {source}/{split} plan/bootstrap hash differs"
            )
        dataset = dataset_chain(Path(row.get("root", "")), row, "mono-video")
        dataset["input_variants"] = sdr_variants
        dataset_chains[f"{source}-{split}"] = dataset
    if len(dataset_chains) != 4:
        raise InspectionError("SDR plan repeats a dataset")
    depth_index = validate_depth_publications(
        plan, dataset_chains, workspace, "SDR"
    )
    split_data = {
        split: {
            "context_source_shas": set(), "label_source_shas": set(),
            "film_ids": set(), "capture_group_ids": set(),
            "source_ids": set(),
        } for split in SPLITS
    }
    dataset_results = []
    merged_hashes = []
    contacts = []
    condition_accumulators = {split: {} for split in SPLITS}
    for row in datasets:
        source = row.get("source")
        split = row.get("split")
        key = (source, split)
        if key not in expected_keys:
            raise InspectionError("SDR plan contains an unexpected dataset")
        bootstrap_row = bootstrap_by_key[key]
        dataset_key = f"{source}-{split}"
        dataset = dataset_chains[dataset_key]
        sequence_name = dataset["manifest"].get("source_sequence_manifest")
        if (not isinstance(sequence_name, str) or
                Path(sequence_name).name != sequence_name):
            raise InspectionError("SDR source sequence manifest path is unsafe")
        sequence_path = dataset["root"] / sequence_name
        expected_sequence_hash = bootstrap_row.get(
            "source_sequence_manifest_sha256"
        )
        require_hash(
            sequence_path, expected_sequence_hash,
            f"SDR {source}/{split} source sequence manifest",
        )
        if dataset["manifest"].get("video_sha256") != expected_sequence_hash:
            raise InspectionError("SDR dataset/source sequence identity differs")
        sequence_manifest = load_json(
            sequence_path, "SDR source sequence manifest"
        )
        sequence_rows = sequence_manifest.get("sequences")
        if not isinstance(sequence_rows, list):
            raise InspectionError("SDR source sequence list is missing")
        by_clip = {
            item.get("clip"): item for item in sequence_rows
            if isinstance(item, dict)
        }
        if set(by_clip) != set(dataset["clips"]):
            raise InspectionError("SDR source sequence clip set differs")
        for clip in dataset["clips"]:
            frames = by_clip[clip].get("frames")
            if not isinstance(frames, list) or not frames:
                raise InspectionError(f"SDR source sequence {clip} has no frames")
            clip_records = {
                record["path"]: record
                for record in dataset["clip_hash_manifest"]["clips"][clip]["files"]
            }
            for frame in frames:
                output = frame.get("output")
                record = clip_records.get(output)
                if (not isinstance(record, dict) or
                        record.get("sha256") != frame.get("sha256")):
                    raise InspectionError(
                        f"SDR derived/source hash chain differs for {clip}/{output}"
                    )
                source_sha = frame.get("source_sha256")
                if not isinstance(source_sha, str) or len(source_sha) != 64:
                    raise InspectionError("SDR source frame SHA-256 is invalid")
                split_data[split]["context_source_shas"].add(source_sha)
        expected_rows = (
            40 if key == ("reds", "training") else
            20 if key == ("spring", "training") else 10
        )
        bundle = validate_merged_bundle(
            workspace, "sdr_origin", dataset_key, dataset,
            expected_rows, 4, plan_contract, depth_index,
        )
        dataset_results.append({
            key_: value for key_, value in bundle.items()
            if key_ not in {
                "source_hashes", "source_ids", "film_ids", "contacts",
                "_condition_accumulators",
            }
        })
        merged_hashes.append({
            "dataset": dataset_key, "split": split,
            "labels_sha256": bundle["labels_sha256"],
        })
        split_data[split]["label_source_shas"].update(bundle["source_hashes"])
        split_data[split]["source_ids"].update(bundle["source_ids"])
        split_data[split]["film_ids"].update(bundle["film_ids"])
        for name, accumulator in bundle["_condition_accumulators"].items():
            condition_accumulators[split].setdefault(
                name, DistributionAccumulator()
            ).merge(accumulator)
        contacts.extend(bundle["contacts"])
    rgb = Counter()
    policy = Counter()
    distributions = {}
    for item in dataset_results:
        rgb[item["split"]] += item["rgb_rows"]
        policy[item["split"]] += item["policy_samples"]
    for split in SPLITS:
        split_items = [
            item["distribution"] for item in dataset_results
            if item["split"] == split
        ]
        distributions[split] = combine_distributions(split_items)
    if dict(rgb) != EXPECTED_SDR_RGB or dict(policy) != EXPECTED_CARDINALITY[
            "sdr_origin"]:
        raise InspectionError(
            f"SDR-origin cardinality differs: rgb={dict(rgb)}, policy={dict(policy)}"
        )
    return {
        "branch": "sdr_origin",
        "workspace": str(workspace),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "bootstrap_manifest": str(bootstrap_path),
        "bootstrap_manifest_sha256": bootstrap_hash,
        "datasets": dataset_results,
        "rgb_cardinality": dict(rgb),
        "policy_cardinality": dict(policy),
        "merged_label_hashes": merged_hashes,
        "merged_label_set_sha256": canonical_sha256(merged_hashes),
        "distributions": distributions,
        "condition_distributions": {
            split: {
                name: accumulator.result()
                for name, accumulator in sorted(values.items())
            } for split, values in condition_accumulators.items()
        },
        "_split_data": split_data,
        "_condition_accumulators": condition_accumulators,
        "_contacts": contacts,
    }


def combine_distributions(items):
    if not items:
        return {"policy_samples": 0}
    result = {
        "policy_samples": sum(item["policy_samples"] for item in items),
        "actionable": sum(item["actionable"] for item in items),
        "identity": sum(item["identity"] for item in items),
        "identity_feasible": sum(item["identity_feasible"] for item in items),
        "identity_infeasible": sum(
            item["identity_infeasible"] for item in items
        ),
    }
    result["actionable_pct"] = (
        100.0 * result["actionable"] / result["policy_samples"]
        if result["policy_samples"] else 0.0
    )
    for key in ("condition_counts", "scale_counts", "identity_violations"):
        counter = Counter()
        for item in items:
            counter.update(item.get(key, {}))
        result[key] = dict(sorted(counter.items()))
    # Bundle summaries already expose exact per-bundle percentiles. Preserve
    # them rather than pretending percentiles can be algebraically merged.
    result["bundle_distributions"] = [{
        key: item[key] for key in (
            "scale", "confidence", "safety_margin_reliability",
            "rendered_pop_spread_pct", "rendered_mean_abs_disparity_pct",
            "rendered_p95_abs_disparity_pct", "comfort_clamp_abs_pct",
            "clamped_pixel_pct",
        )
    } for item in items]
    return result


PRIMARY_DISTRIBUTIONS = (
    "scale", "confidence", "safety_margin_reliability",
    "rendered_pop_spread_pct", "rendered_mean_abs_disparity_pct",
    "rendered_p95_abs_disparity_pct", "comfort_clamp_abs_pct",
    "clamped_pixel_pct",
)


def merged_accumulator(accumulators):
    result = DistributionAccumulator()
    for accumulator in accumulators:
        result.merge(accumulator)
    return result


def validate_regime_result(result, expected_count: int, expected_sources: int,
                           origin: str):
    if (result.get("sample_count") != expected_count or
            result.get("source_frame_count") != expected_sources or
            result.get("shot_count", 0) <= 0):
        raise InspectionError(
            f"{origin} cardinality differs: "
            f"{result.get('sample_count')}/{result.get('source_frame_count')} "
            f"!= {expected_count}/{expected_sources}"
        )
    for field in PRIMARY_DISTRIBUTIONS:
        if result.get(field, {}).get("count") != expected_count:
            raise InspectionError(f"{origin} lacks complete {field} evidence")
    return result


def build_runtime_regimes(branches):
    if not {"sdr_origin", "native_pq"}.issubset(branches):
        return {}, (
            "SDR-versus-HDR comparison is pending until both authenticated "
            "branches are present."
        )
    sdr_branch = branches["sdr_origin"]["_condition_accumulators"]
    native_branch = branches["native_pq"]["_condition_accumulators"]
    groups = {}

    def publish(name, by_split):
        expected = EXPECTED_REGIME_CARDINALITY[name]
        splits = {}
        total = DistributionAccumulator()
        for split in SPLITS:
            accumulator = by_split[split]
            expected_sources = (
                expected[split] // 3 if name == "hdr" else expected[split]
            )
            result = validate_regime_result(
                accumulator.result(), expected[split], expected_sources,
                f"{name}/{split}",
            )
            splits[split] = result
            total.merge(accumulator)
        groups[name] = {
            "status": "pass",
            "expected_sample_count": expected,
            "splits": splits,
            "combined": total.result(),
        }

    publish("sdr", {
        split: sdr_branch[split]["sdr"] for split in SPLITS
    })
    for condition in ("hdr-w1000", "hdr-w2500", "hdr-w6000"):
        publish(condition, {
            split: sdr_branch[split][condition] for split in SPLITS
        })
    publish("hdr", {
        split: merged_accumulator([
            sdr_branch[split][condition]
            for condition in ("hdr-w1000", "hdr-w2500", "hdr-w6000")
        ]) for split in SPLITS
    })
    publish("native-pq", {
        split: native_branch[split]["native-pq"] for split in SPLITS
    })

    sdr = groups["sdr"]["splits"]["development"]
    hdr = groups["hdr"]["splits"]["development"]
    native = groups["native-pq"]["splits"]["development"]
    conclusion = (
        "All runtime regimes pass independently. On development labels, native "
        f"SDR is {sdr['actionable_pct']:.1f}% actionable with median scale "
        f"{sdr['scale']['p50']:.3f}; simulated Windows HDR is "
        f"{hdr['actionable_pct']:.1f}% actionable with median scale "
        f"{hdr['scale']['p50']:.3f}; native PQ is "
        f"{native['actionable_pct']:.1f}% actionable with median scale "
        f"{native['scale']['p50']:.3f}. No aggregate is used to waive a "
        "regime-local failure."
    )
    return groups, conclusion


def validate_native_workspace(workspace: Path):
    workspace = Path(workspace).resolve(strict=True)
    plan_path = workspace / NATIVE_PLAN
    plan = load_json(plan_path, "native orchestration plan")
    if (plan.get("schema") != NATIVE_PLAN_SCHEMA or
            plan.get("contract") != NATIVE_PLAN_CONTRACT or
            plan.get("training_command_present") is not False):
        raise InspectionError("unsupported native-PQ orchestration plan")
    plan_contract = validate_plan_contracts(plan, "native-PQ")
    bootstrap_path = assert_working_path(
        Path(plan.get("bootstrap_manifest", "")), "native bootstrap manifest"
    )
    bootstrap_hash = require_hash(
        bootstrap_path, plan.get("bootstrap_manifest_sha256"),
        "native bootstrap manifest",
    )
    bootstrap = load_json(bootstrap_path, "native bootstrap manifest")
    if (bootstrap.get("schema") != NATIVE_BOOTSTRAP_SCHEMA or
            bootstrap.get("contract") != NATIVE_BOOTSTRAP_CONTRACT or
            bootstrap.get("sealed_test_policy") !=
            "CHUG test masters were not decoded or opened"):
        raise InspectionError("unsupported native bootstrap contract")
    expected_flow_hash = chug_prepare.flow_support_metric_sha256()
    if (bootstrap.get("temporal_evidence_selection_contract") !=
            chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT or
            bootstrap.get("source_flow_support_contract") !=
            chug_prepare.FLOW_SUPPORT_CONTRACT or
            bootstrap.get("source_flow_metric_sha256") != expected_flow_hash or
            bootstrap.get("source_flow_support_minimum") !=
            chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT):
        raise InspectionError("native bootstrap flow-support contract differs")
    conversion_path = assert_working_path(
        Path(bootstrap.get("conversion_contract", "")),
        "native PQ-to-scRGB conversion contract",
    )
    conversion_hash = require_hash(
        conversion_path,
        bootstrap.get("conversion_contract_sha256"),
        "native PQ-to-scRGB conversion contract",
    )
    conversion = load_json(
        conversion_path, "native PQ-to-scRGB conversion contract"
    )
    expected_source = {
        "codec": "hevc",
        "pixel_format": "yuv420p10+",
        "range": "limited",
        "primaries": "bt2020",
        "matrix": "bt2020nc",
        "transfer": "smpte2084",
    }
    if (conversion.get("schema") != 1 or
            conversion.get("contract") != NATIVE_CONVERSION_CONTRACT or
            conversion.get("source") != expected_source or
            conversion.get("scrgb_reference_white_nits") != 80.0 or
            conversion.get("depth_input_color_contract_sha256") !=
            input_color.color_contract_sha256()):
        raise InspectionError(
            "native branch is not authenticated HEVC 10-bit BT.2020/PQ "
            "converted to production Windows scRGB"
        )
    datasets = plan.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise InspectionError("native plan must name training and development")
    bootstrap_datasets = bootstrap.get("datasets")
    if not isinstance(bootstrap_datasets, dict):
        raise InspectionError("native bootstrap dataset map is missing")
    native_variant = input_color.native_pq_input_variant()
    dataset_chains = {}
    planned_native_splits = set()
    for row in datasets:
        split = row.get("split")
        if split not in SPLITS or split in planned_native_splits:
            raise InspectionError("native plan repeats/contains unsupported split")
        planned_native_splits.add(split)
        bootstrap_row = bootstrap_datasets.get(split)
        if not isinstance(bootstrap_row, dict):
            raise InspectionError(f"native bootstrap lacks {split}")
        if (row.get("dataset_manifest_sha256") !=
                bootstrap_row.get("dataset_manifest_sha256") or
                row.get("clip_hash_manifest_sha256") !=
                bootstrap_row.get("clip_hash_manifest", {}).get("sha256")):
            raise InspectionError(f"native {split} plan/bootstrap hash differs")
        dataset = dataset_chain(
            Path(row.get("root", "")), row, "native-hdr-video"
        )
        dataset["input_variants"] = [native_variant]
        dataset_chains[f"chug-native-pq-{split}"] = dataset
    if len(dataset_chains) != 2:
        raise InspectionError("native plan does not contain both working splits")
    depth_index = validate_depth_publications(
        plan, dataset_chains, workspace, "native-PQ"
    )
    split_data = {
        split: {
            "context_source_shas": set(), "label_source_shas": set(),
            "film_ids": set(), "capture_group_ids": set(),
            "source_ids": set(), "model_source_shas": set(),
        } for split in SPLITS
    }
    hdr = HDRAccumulator()
    dataset_results = []
    merged_hashes = []
    contacts = []
    condition_accumulators = {split: {} for split in SPLITS}
    for row in datasets:
        split = row.get("split")
        if split not in SPLITS:
            raise InspectionError("native plan includes a sealed/unsupported split")
        bootstrap_row = bootstrap_datasets.get(split)
        if not isinstance(bootstrap_row, dict):
            raise InspectionError(f"native bootstrap lacks {split}")
        dataset_key = f"chug-native-pq-{split}"
        dataset = dataset_chains[dataset_key]
        if dataset["manifest"].get(
                "conversion_contract_sha256") != conversion_hash:
            raise InspectionError(
                f"native {split} conversion-contract hash differs"
            )
        if (dataset["manifest"].get("preparation_contract") !=
                NATIVE_BOOTSTRAP_CONTRACT or
                dataset["manifest"].get(
                    "temporal_evidence_selection_contract") !=
                chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT or
                dataset["manifest"].get("source_flow_support_contract") !=
                chug_prepare.FLOW_SUPPORT_CONTRACT or
                dataset["manifest"].get("source_flow_metric_sha256") !=
                expected_flow_hash or
                dataset["manifest"].get("source_flow_support_minimum") !=
                chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT):
            raise InspectionError(f"native {split} flow-support contract differs")
        expected_labels = EXPECTED_NATIVE_RGB[split]
        if (row.get("label_frames") != expected_labels or
                bootstrap_row.get("label_frame_count") != expected_labels or
                dataset["manifest"].get("label_frame_count") != expected_labels):
            raise InspectionError(f"native {split} label cardinality differs")
        manifest_sequence_by_clip = {
            item.get("clip"): item for item in dataset["sequences"]
        }
        for clip in dataset["clips"]:
            clip_root = dataset["root"] / clip
            try:
                authentication = native_hdr_capture.validate_clip(
                    clip_root, full=True
                )
                payload, frames, _manifest_path = (
                    native_hdr_capture.load_manifest(clip_root)
                )
            except RuntimeError as error:
                raise InspectionError(
                    f"native {split}/{clip} authentication failed: {error}"
                ) from error
            ids = list(frames)
            if ids != list(range(len(ids))):
                raise InspectionError(f"native {split}/{clip} cadence differs")
            timestamps = [
                finite_number(frames[index]["timestamp_seconds"], "timestamp")
                for index in ids
            ]
            if any(right <= left for left, right in zip(
                    timestamps, timestamps[1:])):
                raise InspectionError(
                    f"native {split}/{clip} timestamps are not increasing"
                )
            previews = {
                int(path.stem.removeprefix("frame_")): path.resolve()
                for path in clip_root.glob("frame_*.png")
                if path.stem.removeprefix("frame_").isdigit()
            }
            if set(previews) != set(frames):
                raise InspectionError(
                    f"native {split}/{clip} preview/sidecar cadence differs"
                )
            source_video_sha = payload.get("source_video", {}).get("sha256")
            if not isinstance(source_video_sha, str) or len(source_video_sha) != 64:
                raise InspectionError("native source-video SHA-256 is invalid")
            source_video = payload["source_video"]
            if (source_video.get("dataset") != "CHUG" or
                    source_video.get("license") != "CC BY-NC-SA 4.0" or
                    payload.get("conversion", {}).get("contract_sha256") !=
                    conversion_hash):
                raise InspectionError(
                    f"native {split}/{clip} lacks genuine PQ source provenance"
                )
            split_data[split]["context_source_shas"].add(source_video_sha)
            sequence = manifest_sequence_by_clip.get(clip)
            if not isinstance(sequence, dict):
                raise InspectionError(f"native dataset lacks sequence {clip}")
            capture_group = sequence.get("capture_group_id")
            if not isinstance(capture_group, str) or not capture_group:
                raise InspectionError(f"native {clip} capture group is missing")
            split_data[split]["capture_group_ids"].add(capture_group)
            metadata = load_json(clip_root / "meta.json", "native clip metadata")
            if (metadata.get("capture_group_id") != capture_group or
                    metadata.get("split") != split or
                    metadata.get("source_kind") != "native-hdr-video"):
                raise InspectionError(f"native {clip} metadata differs")
            selection = validate_flow_selection(
                sequence.get("temporal_evidence_selection"),
                f"native {split}/{clip} sequence",
            )
            if (sequence.get("nominal_source_label_frame_id") !=
                    selection["nominal_source_label_frame_id"] or
                    sequence.get("source_label_frame_id") !=
                    selection["selected_source_label_frame_id"] or
                    finite_number(
                        sequence.get("selected_pair_flow_support"),
                        "sequence selected flow support",
                    ) != selection["selected_pair_flow_support"]):
                raise InspectionError(
                    f"native {split}/{clip} sequence flow selection differs"
                )
            metadata_selection = metadata.get("frame_selection", {}).get(
                "temporal_evidence_selection"
            )
            source_selection = source_video.get("temporal_evidence_selection")
            if (metadata_selection != selection or source_selection != selection):
                raise InspectionError(
                    f"native {split}/{clip} flow selection is not end-to-end bound"
                )
            frame_selection = metadata.get("frame_selection", {})
            if (frame_selection.get("contract") !=
                    chug_prepare.FRAME_SELECTION_CONTRACT or
                    frame_selection.get("source_label_frame_id") !=
                    selection["selected_source_label_frame_id"] or
                    source_video.get("frame_selection_contract") !=
                    chug_prepare.FRAME_SELECTION_CONTRACT or
                    source_video.get("source_label_frame_id") !=
                    selection["selected_source_label_frame_id"]):
                raise InspectionError(
                    f"native {split}/{clip} frame-selection contract differs"
                )
            for frame_id, frame in frames.items():
                hdr.add(
                    frame["model_path"], authentication["width"],
                    authentication["height"], frame["preview_path"],
                )
                split_data[split]["model_source_shas"].add(frame["sha256"])
        bundle = validate_merged_bundle(
            workspace, "native_pq", dataset_key, dataset,
            EXPECTED_NATIVE_RGB[split], 1, plan_contract, depth_index,
        )
        dataset_results.append({
            key: value for key, value in bundle.items()
            if key not in {
                "source_hashes", "source_ids", "film_ids", "contacts",
                "_condition_accumulators",
            }
        })
        merged_hashes.append({
            "dataset": dataset_key, "split": split,
            "labels_sha256": bundle["labels_sha256"],
        })
        split_data[split]["label_source_shas"].update(bundle["source_hashes"])
        split_data[split]["source_ids"].update(bundle["source_ids"])
        split_data[split]["film_ids"].update(bundle["film_ids"])
        for name, accumulator in bundle["_condition_accumulators"].items():
            condition_accumulators[split].setdefault(
                name, DistributionAccumulator()
            ).merge(accumulator)
        contacts.extend(bundle["contacts"])
    rgb = Counter()
    policy = Counter()
    distributions = {}
    for item in dataset_results:
        rgb[item["split"]] += item["rgb_rows"]
        policy[item["split"]] += item["policy_samples"]
    for split in SPLITS:
        distributions[split] = combine_distributions([
            item["distribution"] for item in dataset_results
            if item["split"] == split
        ])
    if dict(rgb) != EXPECTED_NATIVE_RGB or dict(policy) != EXPECTED_CARDINALITY[
            "native_pq"]:
        raise InspectionError(
            f"native-PQ cardinality differs: rgb={dict(rgb)}, policy={dict(policy)}"
        )
    return {
        "branch": "native_pq",
        "workspace": str(workspace),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "bootstrap_manifest": str(bootstrap_path),
        "bootstrap_manifest_sha256": bootstrap_hash,
        "conversion_contract": str(conversion_path),
        "conversion_contract_sha256": conversion_hash,
        "datasets": dataset_results,
        "rgb_cardinality": dict(rgb),
        "policy_cardinality": dict(policy),
        "merged_label_hashes": merged_hashes,
        "merged_label_set_sha256": canonical_sha256(merged_hashes),
        "native_hdr_statistics": hdr.result(),
        "distributions": distributions,
        "condition_distributions": {
            split: {
                name: accumulator.result()
                for name, accumulator in sorted(values.items())
            } for split, values in condition_accumulators.items()
        },
        "_split_data": split_data,
        "_condition_accumulators": condition_accumulators,
        "_contacts": contacts,
    }


def validate_leakage(branches):
    combined = {
        split: {
            "context_source_shas": set(), "label_source_shas": set(),
            "film_ids": set(), "capture_group_ids": set(),
            "source_ids": set(), "model_source_shas": set(),
        } for split in SPLITS
    }
    duplicate_source_ids = []
    seen_source_ids = set()
    for branch in branches.values():
        for split in SPLITS:
            source = branch["_split_data"][split]
            for key in combined[split]:
                values = set(source.get(key, set()))
                if key == "source_ids":
                    duplicates = values & seen_source_ids
                    duplicate_source_ids.extend(sorted(map(str, duplicates)))
                    seen_source_ids.update(values)
                combined[split][key].update(values)
    overlaps = {}
    for key in (
            "context_source_shas", "label_source_shas", "film_ids",
            "capture_group_ids", "model_source_shas"):
        overlap = combined["training"][key] & combined["development"][key]
        if overlap:
            overlaps[key] = sorted(map(str, overlap))[:20]
    label_duplicate = (
        combined["training"]["label_source_shas"] &
        combined["development"]["label_source_shas"]
    )
    if duplicate_source_ids or overlaps or label_duplicate:
        raise InspectionError(
            "train/development leakage or duplicate source IDs detected: "
            f"overlaps={overlaps}, duplicate_source_ids={duplicate_source_ids[:20]}"
        )
    return {
        "pass": True,
        "train_dev_context_source_sha_overlap": 0,
        "train_dev_label_source_sha_overlap": 0,
        "train_dev_film_overlap": 0,
        "train_dev_capture_group_overlap": 0,
        "train_dev_native_sidecar_sha_overlap": 0,
        "duplicate_source_ids": 0,
        "counts": {
            split: {
                key: len(value) for key, value in combined[split].items()
            } for split in SPLITS
        },
    }


def image_data_uri(image: Image.Image, width=320, height=180) -> str:
    converted = image.convert("RGB")
    converted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (8, 12, 20))
    left = (width - converted.width) // 2
    top = (height - converted.height) // 2
    canvas.paste(converted, (left, top))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def path_image_data_uri(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image.load()
            return image_data_uri(image)
    except (OSError, ValueError) as error:
        raise InspectionError(f"cannot render contact image: {path}") from error


def depth_image_data_uri(path: Path) -> str:
    """Render 16-bit depth without the saturating PIL I;16 -> RGB conversion."""
    try:
        with Image.open(path) as image:
            image.load()
            values = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError) as error:
        raise InspectionError(f"cannot render contact depth: {path}") from error

    finite = np.isfinite(values)
    if not finite.any():
        raise InspectionError(f"contact depth has no finite values: {path}")
    samples = values[finite]
    low, high = map(float, np.percentile(samples, (1.0, 99.0)))
    if high <= low:
        low, high = float(samples.min()), float(samples.max())
    if high <= low:
        stretched = np.full(values.shape, 0.5, dtype=np.float32)
    else:
        stretched = np.clip((values - low) / (high - low), 0.0, 1.0)
    stretched[~finite] = 0.0
    rendered = Image.fromarray(np.uint8(np.round(stretched * 255.0)))
    return image_data_uri(rendered)


def disparity_image(values, limit) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    if not math.isfinite(limit) or limit <= 1e-9:
        normalized = np.full(values.shape, 0.5, dtype=np.float32)
    else:
        normalized = np.clip(values / (2.0 * limit) + 0.5, 0.0, 1.0)
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    rgb = np.stack((red, green, blue), axis=2)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)), "RGB")


def disparity_comparison_data_uri(path: Path, geometry, scale, clamp_abs) -> str:
    raw = load_float_texture(
        path, expected_shape=(
            geometry["disparity_raster_height"],
            geometry["disparity_raster_width"],
        )
    )
    selected = np.clip(raw * float(scale), -float(clamp_abs), float(clamp_abs))
    limit = max(
        float(clamp_abs), float(np.percentile(np.abs(raw), 99.0)),
        float(np.percentile(np.abs(selected), 99.0)),
    )
    left = disparity_image(raw, limit)
    right = disparity_image(selected, limit)
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width + 2, height), (8, 12, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 2, 0))
    return image_data_uri(canvas)


def select_contacts(branches, per_split: int):
    selected = []
    for branch_name, branch in sorted(branches.items()):
        by_split = {split: [] for split in SPLITS}
        for contact in branch["_contacts"]:
            by_split[contact["split"]].append(contact)
        for split in SPLITS:
            values = sorted(
                by_split[split],
                key=lambda item: (
                    item["dataset_key"], item["row"]["clip"],
                    int(item["row"]["frame"]),
                ),
            )
            if not values:
                continue
            indices = np.linspace(
                0, len(values) - 1, min(per_split, len(values)), dtype=int
            )
            for index in sorted(set(map(int, indices))):
                item = values[index]
                row = item["row"]
                source_path = Path(row["source"])
                order = {
                    "sdr": 0, "hdr-w1000": 1, "hdr-w2500": 2,
                    "hdr-w6000": 3, "native-pq": 4,
                }
                targets = sorted(
                    row["input_condition_targets"],
                    key=lambda value: order[condition_name(value["input_variant"])],
                )
                for target in targets:
                    target_hash = target["input_variant_sha256"]
                    geometries = [
                        value for value in row["deployment_geometry_variants"]
                        if value["input_variant_sha256"] == target_hash
                    ]
                    if len(geometries) != 2:
                        raise InspectionError(
                            "contact row has no exact two-geometry evidence"
                        )
                    depth = item["depth_by_input"].get(target_hash)
                    if not isinstance(depth, dict):
                        raise InspectionError(
                            "contact row lacks authenticated depth evidence"
                        )
                    depth_path = Path(depth["path"])
                    require_hash(
                        depth_path, depth.get("sha256"),
                        "contact-sheet displayed depth",
                    )
                    condition = condition_name(target["input_variant"])
                    geometry_rows = []
                    image_rows = [
                        {
                            "label": "source preview",
                            "uri": path_image_data_uri(source_path),
                        },
                        {
                            "label": (
                                f"{condition} authenticated depth "
                                "(p1-p99 display stretch)"
                            ),
                            "uri": depth_image_data_uri(depth_path),
                        },
                    ]
                    for geometry_index, value in enumerate(geometries):
                        geometry = value["geometry"]
                        path = Path(value["baseline_unclamped_disparity"])
                        label = (
                            f"{geometry['eye_width']}x{geometry['eye_height']} eye "
                            f"· baseline → selected {target['safe_scale_ceiling']:.3f}"
                        )
                        image_rows.append({
                            "label": label,
                            "uri": disparity_comparison_data_uri(
                                path, geometry,
                                target["safe_scale_ceiling"],
                                value["artistic_full_clamp_abs"],
                            ),
                        })
                        geometry_rows.append({
                            "geometry": geometry,
                            "baseline_unclamped_disparity": str(path.resolve()),
                            "baseline_unclamped_disparity_sha256": sha256(path),
                            "selected_scale": float(target["safe_scale_ceiling"]),
                            "artistic_full_clamp_abs": float(
                                value["artistic_full_clamp_abs"]
                            ),
                            "selected_render_target": value[
                                "safe_ceiling_render_target"
                            ],
                            "visual_semantics": (
                                "left=scale-1 baseline; right=selected safe "
                                "scale after production comfort clamp"
                            ),
                            "index": geometry_index,
                        })
                    selected.append({
                        "branch": branch_name,
                        "split": split,
                        "dataset": item["dataset_key"],
                        "clip": row["clip"],
                        "frame": int(row["frame"]),
                        "condition": condition,
                        "scale": float(target["safe_scale_ceiling"]),
                        "confidence": float(target["ceiling_confidence"]),
                        "safety_margin_reliability": float(
                            target["safety_margin_reliability"]
                        ),
                        "safe_ceiling_render_target": target[
                            "safe_ceiling_render_target"
                        ],
                        "preview_path": str(source_path.resolve()),
                        "preview_sha256": sha256(source_path),
                        "depth_path": str(depth_path.resolve()),
                        "depth_sha256": depth["sha256"],
                        "depth_manifest": str(depth["manifest"]),
                        "depth_manifest_sha256": depth["manifest_sha256"],
                        "geometry_evidence": geometry_rows,
                        "images": image_rows,
                    })
    return selected


def public_branch(branch):
    return {
        key: value for key, value in branch.items()
        if not key.startswith("_")
    }


def build_html(payload, contacts):
    verdict = payload["verdict"]
    passed = verdict in {"pass", "pass_partial"}
    badge = "PASS" if verdict == "pass" else (
        "PARTIAL PASS" if verdict == "pass_partial" else "FAIL"
    )
    color = "#36d399" if passed else "#fb7185"
    branch_cards = []
    for name, branch in payload["branches"].items():
        if branch.get("status") != "pass":
            error = html.escape(branch.get("error", "failed"))
            branch_cards.append(
                f"<article><h3>{html.escape(name)}</h3>"
                f"<p class='bad'>{error}</p></article>"
            )
            continue
        policy = branch["policy_cardinality"]
        branch_cards.append(
            f"<article><h3>{html.escape(name)}</h3>"
            f"<p><b>{policy['training']}</b> training conditions &middot; "
            f"<b>{policy['development']}</b> development conditions</p>"
            f"<p class='mono'>{html.escape(branch['merged_label_set_sha256'])}</p>"
            "</article>"
        )
    error_rows = "".join(
        f"<li>{html.escape(value)}</li>" for value in payload["errors"]
    ) or "<li>None</li>"
    distribution_rows = []
    for branch_name, branch in payload["branches"].items():
        if branch.get("status") != "pass":
            continue
        for split, values in branch["distributions"].items():
            scale_counts = html.escape(json.dumps(
                values.get("scale_counts", {}), sort_keys=True
            ))
            distribution_rows.append(
                "<tr>"
                f"<td>{html.escape(branch_name)}</td>"
                f"<td>{html.escape(split)}</td>"
                f"<td>{values.get('policy_samples', 0)}</td>"
                f"<td>{values.get('actionable', 0)}</td>"
                f"<td>{values.get('identity', 0)}</td>"
                f"<td>{values.get('actionable_pct', 0.0):.1f}%</td>"
                f"<td class='mono'>{scale_counts}</td>"
                "</tr>"
            )
    regime_rows = []
    for name, group in payload.get("runtime_regimes", {}).items():
        for split, values in group.get("splits", {}).items():
            regime_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{html.escape(split)}</td>"
                f"<td>{values['sample_count']}</td>"
                f"<td>{values['shot_count']}</td>"
                f"<td>{values['scale']['p50']:.3f}</td>"
                f"<td>{values['confidence']['mean']:.3f}</td>"
                f"<td>{values['rendered_pop_spread_pct']['p50']:.3f}%</td>"
                f"<td>{values['clamped_pixel_pct']['p95']:.3f}%</td>"
                f"<td>{values['safety_margin_reliability']['mean']:.3f}</td>"
                "</tr>"
            )
    hdr_section = ""
    native = payload["branches"].get("native_pq", {})
    if native.get("status") == "pass":
        stats = native["native_hdr_statistics"]
        luminance = stats["luminance_nits_sampled"]
        superwhite = stats["superwhite_luminance_fraction"] * 100.0
        negative = stats["negative_component_fraction"] * 100.0
        extremes = stats["fp16_extreme_component_fraction"] * 100.0
        saturation = stats["preview_saturated_fraction"] * 100.0
        hdr_section = "".join((
            "<section><h2>Native HDR signal audit</h2>",
            "<div class='cards'>",
            "<article><h3>Superwhite</h3>",
            f"<p>{superwhite:.4f}% luminance pixels</p></article>",
            "<article><h3>Negative gamut</h3>",
            f"<p>{negative:.4f}% RGB components</p></article>",
            "<article><h3>Luminance</h3>",
            f"<p>p50 {luminance.get('p50', 0):.1f} nit &middot; ",
            f"p95 {luminance.get('p95', 0):.1f} nit &middot; ",
            f"max {luminance.get('max', 0):.1f} nit</p></article>",
            "<article><h3>Clipping</h3>",
            f"<p>FP16 extremes {extremes:.6f}% &middot; ",
            f"preview saturation {saturation:.3f}%</p></article>",
            "</div></section>",
        ))
    contact_html = []
    for item in contacts:
        images = "".join(
            f"<figure><img src='{image['uri']}' alt='{html.escape(image['label'])}'>"
            f"<figcaption>{html.escape(image['label'])}</figcaption></figure>"
            for image in item["images"]
        )
        contact_html.append(
            "<article class='contact'>"
            f"<h3>{html.escape(item['branch'])} &middot; "
            f"{html.escape(item['split'])} &middot; "
            f"{html.escape(str(item['clip']))}:{item['frame']}</h3>"
            f"<p>{html.escape(item['condition'])} &middot; "
            f"target {item['scale']:.3f} &middot; "
            f"confidence {item['confidence']:.1f}</p>"
            f"<div class='strip'>{images}</div></article>"
        )
    contact_body = "".join(contact_html)
    if not contact_body:
        contact_body = "<p>No authenticated contact samples available.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apollo artistic bootstrap inspection</title>
<style>
:root{{--bg:#07111d;--panel:#0d1b2a;--line:#1f3548;
--text:#e5edf5;--muted:#90a4b8;--accent:{color}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);
font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}
h1,h2,h3{{margin:.2em 0 .55em}} h2{{margin-top:32px}}
.hero{{border:1px solid var(--line);
background:linear-gradient(135deg,#0d1b2a,#11263b);
padding:28px;border-radius:18px}}
.badge{{display:inline-block;color:#051018;background:var(--accent);
font-weight:900;padding:7px 12px;border-radius:999px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}}
article{{background:var(--panel);border:1px solid var(--line);
border-radius:14px;padding:16px;overflow:hidden}}
.mono{{font:12px ui-monospace,monospace;word-break:break-all;
color:var(--muted)}} .bad{{color:#fb7185}}
table{{width:100%;border-collapse:collapse;background:var(--panel)}}
th,td{{padding:10px;border:1px solid var(--line);
text-align:left;vertical-align:top}}
.contact{{margin:14px 0}}
.strip{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));
gap:8px}}
figure{{margin:0;background:#050a10;border-radius:9px;overflow:hidden}}
img{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain}}
figcaption{{padding:6px 9px;color:var(--muted);font-size:12px}} code{{color:#93c5fd}}
@media(max-width:800px){{.strip{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<section class="hero"><span class="badge">{badge}</span>
<h1>Pretraining dataset inspection</h1>
<p>{html.escape(payload['conclusion'])}</p>
<p class="mono">Contract {CONTRACT} · sealed test accessed: false</p></section>
<section><h2>Branch authentication</h2><div class="cards">
{''.join(branch_cards)}</div></section>
<section><h2>Acceptance issues</h2><ul>{error_rows}</ul></section>
<section><h2>Runtime-regime gate</h2>
<p>{html.escape(payload.get('sdr_vs_hdr_conclusion', 'Pending.'))}</p>
<table><thead><tr><th>Regime</th><th>Split</th><th>Samples</th>
<th>Shots</th><th>Scale p50</th><th>Confidence mean</th>
<th>Rendered pop p50</th><th>Clamp p95</th><th>Safety mean</th>
</tr></thead><tbody>{''.join(regime_rows)}</tbody></table></section>
<section><h2>Policy target distributions</h2><table><thead><tr>
<th>Branch</th><th>Split</th><th>Samples</th><th>Action</th>
<th>Identity</th><th>Action rate</th><th>Scale counts</th>
</tr></thead><tbody>{''.join(distribution_rows)}</tbody></table></section>
{hdr_section}
<section><h2>Preview · depth · two-geometry evidence</h2>
<p>Every image is embedded. Each geometry panel shows the authenticated scale-1
baseline on the left and the selected safe scale after the production clamp on the right.</p>
{contact_body}
</section></main></body></html>"""


def write_atomic(path: Path, data: bytes):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_report(output: Path, payload, contacts, overwrite=False):
    output = Path(output).resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise InspectionError(f"output exists and is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "inspection.json"
    html_path = output / "report.html"
    if not overwrite and (json_path.exists() or html_path.exists()):
        raise InspectionError(
            f"inspection output exists; use --overwrite: {output}"
        )
    public_payload = json.loads(json.dumps(payload, allow_nan=False))
    write_atomic(
        json_path,
        (json.dumps(public_payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    report = build_html(public_payload, contacts)
    write_atomic(html_path, report.encode("utf-8"))
    return json_path, html_path


def inspect(sdr_workspace: Path, native_workspace: Path, output: Path,
            allow_partial=False, overwrite=False, contact_samples_per_split=3):
    roots = {
        "sdr_origin": Path(sdr_workspace),
        "native_pq": Path(native_workspace),
    }
    present = {name: root.is_dir() for name, root in roots.items()}
    errors = []
    branches = {}
    private_branches = {}
    if not any(present.values()):
        errors.append("neither SDR-origin nor native-PQ workspace exists")
    if not allow_partial and not all(present.values()):
        missing = [name for name, value in present.items() if not value]
        errors.append(f"required branch is absent: {', '.join(missing)}")
    validators = {
        "sdr_origin": validate_sdr_workspace,
        "native_pq": validate_native_workspace,
    }
    for name, root in roots.items():
        if not present[name]:
            branches[name] = {"status": "absent", "workspace": str(root)}
            continue
        try:
            private = validators[name](root)
            private_branches[name] = private
            branches[name] = {"status": "pass", **public_branch(private)}
        except (InspectionError, OSError, ValueError, TypeError, KeyError) as error:
            message = f"{name}: {error}"
            errors.append(message)
            branches[name] = {
                "status": "fail", "workspace": str(root.resolve()),
                "error": str(error),
            }
    leakage = None
    if private_branches:
        try:
            leakage = validate_leakage(private_branches)
        except InspectionError as error:
            errors.append(str(error))
            leakage = {"pass": False, "error": str(error)}
    actual = {split: 0 for split in SPLITS}
    for branch in private_branches.values():
        for split in SPLITS:
            actual[split] += int(branch["policy_cardinality"][split])
    if len(private_branches) == 2 and actual != EXPECTED_CARDINALITY["combined"]:
        errors.append(
            f"combined cardinality differs: {actual} != "
            f"{EXPECTED_CARDINALITY['combined']}"
        )
    runtime_regimes = {}
    sdr_vs_hdr_conclusion = (
        "SDR-versus-HDR comparison is unavailable because a required branch "
        "did not authenticate."
    )
    try:
        runtime_regimes, sdr_vs_hdr_conclusion = build_runtime_regimes(
            private_branches
        )
    except (InspectionError, KeyError, TypeError) as error:
        errors.append(f"runtime-regime gate: {error}")
    contacts = []
    if private_branches:
        try:
            contacts = select_contacts(
                private_branches, contact_samples_per_split
            )
        except (InspectionError, OSError, ValueError) as error:
            errors.append(f"contact sheet: {error}")
    complete = len(private_branches) == 2
    if errors:
        verdict = "fail"
        conclusion = (
            "Do not train. The bootstrap failed one or more authentication, "
            "leakage, cardinality, HDR-signal, or evidence checks."
        )
    elif complete:
        verdict = "pass"
        conclusion = (
            "Both SDR-origin and native-PQ branches are authenticated, split-clean, "
            "complete at 300/100 policy samples, and ready for a separate explicit "
            "training command. This inspection did not start training."
        )
    else:
        verdict = "pass_partial"
        conclusion = (
            "The present branch passes its complete branch-local contract. The "
            "combined 300/100 gate remains pending because --allow-partial admitted "
            "one absent branch. Training should remain paused."
        )
    payload = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        "verdict": verdict,
        "conclusion": conclusion,
        "training_started": False,
        "sealed_test_accessed": False,
        "allow_partial": bool(allow_partial),
        "expected_cardinality": EXPECTED_CARDINALITY,
        "actual_combined_policy_cardinality": actual,
        "branches": branches,
        "runtime_regimes": runtime_regimes,
        "sdr_vs_hdr_conclusion": sdr_vs_hdr_conclusion,
        "leakage": leakage,
        "contact_sheet_samples": [{
            key: value for key, value in item.items() if key != "images"
        } for item in contacts],
        "errors": errors,
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    output_root = Path(output).resolve(strict=False)
    payload["outputs"] = {
        "json": str(output_root / "inspection.json"),
        "html": str(output_root / "report.html"),
    }
    json_path, html_path = publish_report(
        output, payload, contacts, overwrite=overwrite
    )
    if payload["outputs"] != {
            "json": str(json_path), "html": str(html_path)}:
        raise InspectionError("published report paths differ")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdr-workspace", required=True, type=Path)
    parser.add_argument("--native-workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--contact-samples-per-split", type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.contact_samples_per_split <= 12:
        parser.error("--contact-samples-per-split must be between 1 and 12")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        result = inspect(
            args.sdr_workspace, args.native_workspace, args.output,
            allow_partial=args.allow_partial, overwrite=args.overwrite,
            contact_samples_per_split=args.contact_samples_per_split,
        )
    except (InspectionError, OSError, ValueError) as error:
        raise SystemExit(f"bootstrap inspection failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verdict"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
