#!/usr/bin/env python3
"""Train the global DA-V2 shared-feature stereo controller.

The trainable head predicts ``[safe_scale_ceiling, safe_ceiling_confidence]``.
Style is a deterministic runtime request clamped by this learned scene-safety cap.
The DA-V2 backbone and depth decoder remain frozen and behavior-neutral; Apollo
retains its existing convergence plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from artistic_policy_model import (
    ART_SCALE_DELTA_MAX,
    POLICY_CHECKPOINT_SCHEMA,
    POLICY_CONTRACT,
    POLICY_FEATURE_CONTRACT,
    POLICY_OUTPUT_SEMANTICS,
    ArtisticPolicyModel,
    load_depth_anything_small,
    policy_state_dict,
    use_dynamic_onnx_position_encoding,
)
from artistic_geometry_contract import (
    allowlist_sha256,
    tuple_key,
    validate_allowlist,
    validate_geometry_tuple,
)
from artistic_sources import schema_is
import audit_artistic_dataset_splits as split_audit
import depth_input_color as input_color
import merge_artistic_geometry_labels as label_merge
import native_hdr_capture


GEOMETRY_GROUP_FIELDS = (
    "source_width", "source_height", "model_input_width", "model_input_height",
    "depth_short_side", "depth_max_aspect",
)


def geometry_group_key(value):
    """Geometry dimensions the RGB/backbone observes, excluding destination-eye variants."""
    return tuple(value[field] for field in GEOMETRY_GROUP_FIELDS)


SBSBENCH_DIR = Path(__file__).resolve().parents[1] / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))
import sbsbench  # noqa: E402


MAX_WIDTH = 1008
MAX_HEIGHT = 1008
LABEL_SCHEMA = 10
TRAINING_SCHEMA = POLICY_CHECKPOINT_SCHEMA
SUPPORTED_STYLES = {"immersive", "balanced", "clean", "authored"}
ACTION_EPSILON = 0.005
MAX_CONSISTENT_FRONTIER_DELTA = 0.10
MAX_EQUIVALENT_DISPARITY_NRMSE = 0.01
CONDITION_TARGET_EFFECTIVE_FIELDS = (
    "safe_scale_min", "safe_scale_max", "safe_scale_ceiling",
    "baseline_multiplier", "ceiling_confidence", "confidence",
    "safety_margin_reliability", "render_evidence_confidence",
    "identity_feasible", "identity_infeasible_variants", "style_targets",
    "style_render_targets", "safe_ceiling_render_target",
    "safe_ceiling_exact_pop_spread_pct",
)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_file_hash(paths):
    """Match the evaluator's normalized semantic metric identity."""
    digest = hashlib.sha256()
    for path in map(Path, paths):
        digest.update(path.name.encode())
        data = path.read_bytes()
        if path.suffix.lower() in {".py", ".json", ".conf", ".md", ".hlsl"}:
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
    return digest.hexdigest()[:16]


def verified_identity(identity, description):
    if not isinstance(identity, dict):
        raise RuntimeError(f"label fitter lacks {description} identity")
    path = Path(identity.get("path", ""))
    if not path.is_file() or sha256(path) != identity.get("sha256"):
        raise RuntimeError(f"label fitter {description} is missing or changed: {path}")
    return path


def validate_condition_target(target, origin):
    """Validate one authenticated image condition's effective policy target."""
    if (not isinstance(target, dict) or
            target.get("schema") != label_merge.CONDITION_TARGET_SCHEMA or
            target.get("contract") != label_merge.CONDITION_TARGET_CONTRACT):
        raise RuntimeError(f"{origin}: incompatible condition target contract")
    for key in (
            "input_variant", "input_variant_sha256",
            "deployment_geometry_variant_count", *CONDITION_TARGET_EFFECTIVE_FIELDS):
        if target.get(key) is None:
            raise RuntimeError(f"{origin}: missing condition target {key}")
    variant = target["input_variant"]
    input_color.validate_input_variant(variant)
    variant_hash = input_color.input_variant_sha256(variant)
    if target["input_variant_sha256"] != variant_hash:
        raise RuntimeError(f"{origin}: stale condition input identity")
    if target["deployment_geometry_variant_count"] != 2:
        raise RuntimeError(
            f"{origin}: condition target must bind exactly two deployment geometries"
        )
    scale = float(target["safe_scale_ceiling"])
    safe_min = float(target["safe_scale_min"])
    safe_max = float(target["safe_scale_max"])
    confidence = float(target["ceiling_confidence"])
    reliability = float(target["safety_margin_reliability"])
    numeric = (
        scale, safe_min, safe_max, confidence, reliability,
        float(target["baseline_multiplier"]), float(target["confidence"]),
        float(target["render_evidence_confidence"]),
        float(target["safe_ceiling_exact_pop_spread_pct"]),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f"{origin}: non-finite condition target")
    if (abs(scale - safe_max) > 1e-6 or
            abs(scale - float(target["baseline_multiplier"])) > 1e-6):
        raise RuntimeError(f"{origin}: condition ceiling aliases disagree")
    action = 1.0 if is_actionable_scale(scale) else 0.0
    if (confidence not in (0.0, 1.0) or confidence != action or
            abs(confidence - float(target["confidence"])) > 1e-6):
        raise RuntimeError(f"{origin}: condition confidence is not the hard action")
    if (abs(reliability - float(target["render_evidence_confidence"])) > 1e-6 or
            (action > 0.5 and not 0.5 <= reliability <= 1.0) or
            (action < 0.5 and abs(reliability) > 1e-6)):
        raise RuntimeError(f"{origin}: inconsistent condition reliability")
    identity_feasible = target["identity_feasible"]
    if not isinstance(identity_feasible, bool):
        raise RuntimeError(f"{origin}: condition identity_feasible is not boolean")
    if identity_feasible:
        if safe_min > 1.0 + 1e-6 or safe_max < 1.0 - 1e-6:
            raise RuntimeError(f"{origin}: condition frontier excludes identity")
    else:
        violations = target["identity_infeasible_variants"]
        if (not isinstance(violations, list) or not violations or
                any(not isinstance(item, dict) or
                    not isinstance(item.get("violations"), list) or
                    not item["violations"] or
                    any(not isinstance(value, str) or not value.endswith(":hard")
                        for value in item["violations"])
                    for item in violations)):
            raise RuntimeError(
                f"{origin}: infeasible condition lacks hard-failure evidence"
            )
        if any(abs(value - 1.0) > 1e-6 for value in (safe_min, safe_max, scale)):
            raise RuntimeError(f"{origin}: infeasible condition is not a no-op")
    if set(target["style_targets"]) != SUPPORTED_STYLES - {"authored"}:
        raise RuntimeError(f"{origin}: condition style targets are incomplete")
    for style in target["style_targets"]:
        style_scale = style_target_scale(target, style)
        if not safe_min - 1e-6 <= style_scale <= scale + 1e-6:
            raise RuntimeError(
                f"{origin}: condition {style} target is outside its safe frontier"
            )
        render = target["style_render_targets"].get(style)
        render_scale = (
            float(render.get("scale", math.nan))
            if isinstance(render, dict) else math.nan
        )
        if (not math.isfinite(render_scale) or
                abs(render_scale - style_scale) > 1e-6):
            raise RuntimeError(f"{origin}: stale condition style render target")
    render = target["safe_ceiling_render_target"]
    render_scale = (
        float(render.get("scale", math.nan))
        if isinstance(render, dict) else math.nan
    )
    if (not math.isfinite(render_scale) or abs(render_scale - scale) > 1e-6):
        raise RuntimeError(f"{origin}: stale condition ceiling render target")
    return variant_hash


def input_condition_target_map(row, origin="policy row"):
    """Return one complete source-family target set keyed by input identity."""
    if row.get("condition_target_contract") != label_merge.CONDITION_TARGET_CONTRACT:
        raise RuntimeError(f"{origin}: incompatible condition target contract")
    manifest = row.get("input_variant_manifest")
    label_merge.validate_policy_input_variant_manifest(manifest)
    declared = {
        input_color.input_variant_sha256(value)
        for value in manifest["variants"]
    }
    targets = row.get("input_condition_targets")
    if not isinstance(targets, list) or len(targets) not in {1, 4}:
        raise RuntimeError(
            f"{origin}: policy row requires one native-PQ or four SDR-origin "
            "condition targets"
        )
    result = {}
    for index, target in enumerate(targets):
        key = validate_condition_target(target, f"{origin}:condition[{index}]")
        if key in result:
            raise RuntimeError(f"{origin}: duplicate condition target")
        result[key] = target
    native = {
        input_color.input_variant_sha256(
            input_color.native_pq_input_variant()
        )
    }
    sdr_origin = {
        input_color.input_variant_sha256(value)
        for value in label_merge.policy_input_variants()
        if value["kind"] != input_color.INPUT_KIND_NATIVE_PQ
    }
    observed = frozenset(result)
    if not observed.issubset(declared) or observed not in {
            frozenset(native), frozenset(sdr_origin)}:
        raise RuntimeError(
            f"{origin}: condition targets do not form one complete source family"
        )
    return result


def validate_row(row, origin="label row"):
    if not schema_is(row.get("label_schema"), LABEL_SCHEMA):
        raise RuntimeError(
            f"{origin}: expected label schema {LABEL_SCHEMA}; regenerate safe-frontier labels"
        )
    if row.get("policy_contract") != POLICY_CONTRACT:
        raise RuntimeError(f"{origin}: incompatible policy contract")
    for key in ("source", "source_sha256", "baseline_multiplier",
                "confidence", "baseline_disparity_mean_abs_pct", "clip",
                "frame", "split", "film_id", "style_targets", "style_render_targets",
                "safe_scale_ceiling", "ceiling_confidence",
                "safety_margin_reliability", "render_evidence_confidence",
                "safe_scale_min", "safe_scale_max",
                "safe_ceiling_exact_pop_spread_pct", "safe_ceiling_render_target",
                "source_width", "source_height", "artistic_full_clamp_abs",
                "render_grid_key",
                "deployment_geometry_allowlist_sha256",
                "condition_target_contract", "input_condition_targets",
                "deployment_geometry_variants",
                "baseline_unclamped_disparity",
                "baseline_unclamped_disparity_sha256",
                "baseline_unclamped_disparity_mean_abs_pct"):
        if row.get(key) is None:
            raise RuntimeError(f"{origin}: missing {key}")
    source = Path(row["source"])
    if not source.is_file() or sha256(source) != row["source_sha256"]:
        raise RuntimeError(f"{origin}: source file is missing or changed: {source}")
    targets = input_condition_target_map(row, origin)
    native_hash = input_color.input_variant_sha256(
        input_color.native_pq_input_variant()
    )
    if native_hash in targets:
        model_source = Path(row.get("model_source", ""))
        expected_bytes = int(row["source_width"]) * int(row["source_height"]) * 8
        if (row.get("model_source_encoding") !=
                native_hdr_capture.CAPTURE_ENCODING or
                not model_source.is_file() or
                model_source.stat().st_size != expected_bytes or
                sha256(model_source) != row.get("model_source_sha256")):
            raise RuntimeError(
                f"{origin}: native HDR model source is missing or changed"
            )
    for path_key, hash_key in (
        ("right_eye", "right_eye_sha256"),
        ("baseline_disparity", "baseline_disparity_sha256"),
        ("reference_disparity", "reference_disparity_sha256"),
        ("baseline_unclamped_disparity", "baseline_unclamped_disparity_sha256"),
    ):
        value = row.get(path_key)
        expected = row.get(hash_key)
        if value is None and expected is None:
            continue
        path = Path(value or "")
        if not expected or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{origin}: {path_key} is missing or changed: {path}")
    if row["split"] not in {"training", "development", "test"}:
        raise RuntimeError(f"{origin}: split must be training/development/test")
    if float(row.get("global_policy_weight", 1.0)) <= 0.0:
        raise RuntimeError(f"{origin}: global-only training cannot consume zero-weight rows")
    ceiling = float(row["safe_scale_ceiling"])
    ceiling_confidence = float(row["ceiling_confidence"])
    safety_reliability = float(row["safety_margin_reliability"])
    if abs(ceiling - float(row["baseline_multiplier"])) > 1e-6:
        raise RuntimeError(f"{origin}: baseline multiplier alias does not match safe ceiling")
    if abs(ceiling_confidence - float(row["confidence"])) > 1e-6:
        raise RuntimeError(f"{origin}: confidence alias does not match ceiling confidence")
    action_target = 1.0 if is_actionable_scale(ceiling) else 0.0
    if ceiling_confidence not in (0.0, 1.0) or ceiling_confidence != action_target:
        raise RuntimeError(f"{origin}: confidence is not the hard actionable target")
    if abs(safety_reliability - float(row["render_evidence_confidence"])) > 1e-6:
        raise RuntimeError(f"{origin}: safety reliability aliases disagree")
    if ((action_target > 0.5 and not 0.5 <= safety_reliability <= 1.0) or
            (action_target < 0.5 and abs(safety_reliability) > 1e-6)):
        raise RuntimeError(f"{origin}: safety-margin reliability is inconsistent")
    safe_min = float(row["safe_scale_min"])
    safe_max = float(row["safe_scale_max"])
    identity_feasible = row.get("identity_feasible", True)
    if not isinstance(identity_feasible, bool):
        raise RuntimeError(f"{origin}: identity_feasible is not boolean")
    if identity_feasible:
        if (safe_min > 1.0 + 1e-6 or safe_max < 1.0 - 1e-6 or
                abs(ceiling - safe_max) > 1e-6):
            raise RuntimeError(
                f"{origin}: ceiling does not match connected safe frontier"
            )
    else:
        violations = row.get("identity_infeasible_variants")
        if (not isinstance(violations, list) or not violations or
                any(not isinstance(item, dict) or
                    not isinstance(item.get("violations"), list) or
                    not item["violations"] or
                    any(not isinstance(value, str) or
                        not value.endswith(":hard")
                        for value in item["violations"])
                    for item in violations)):
            raise RuntimeError(
                f"{origin}: infeasible identity lacks exact hard-failure evidence"
            )
        if (any(abs(value - 1.0) > 1e-6
                for value in (safe_min, safe_max, ceiling)) or
                action_target != 0.0 or abs(safety_reliability) > 1e-6):
            raise RuntimeError(
                f"{origin}: infeasible identity must remain a confidence-zero no-op"
            )
    if set(row["style_targets"]) != SUPPORTED_STYLES - {"authored"}:
        raise RuntimeError(f"{origin}: style targets are incomplete")
    for style in row["style_targets"]:
        target_scale = style_target_scale(row, style)
        if not safe_min - 1e-6 <= target_scale <= ceiling + 1e-6:
            raise RuntimeError(f"{origin}: {style} target is outside safe frontier")
    source_width = int(row["source_width"])
    source_height = int(row["source_height"])
    clamp_abs = float(row["artistic_full_clamp_abs"])
    render_clamp = row["safe_ceiling_render_target"].get("hlsl_full_clamp_abs")
    if (source_width <= 0 or source_height <= 0 or
            not math.isfinite(clamp_abs) or clamp_abs <= 0.0 or
            render_clamp is None or
            not math.isfinite(float(render_clamp)) or
            abs(float(render_clamp) - clamp_abs) > 1e-8):
        raise RuntimeError(f"{origin}: missing or inconsistent exact HLSL comfort clamp")
    numeric = ("baseline_multiplier", "confidence",
               "baseline_disparity_mean_abs_pct", "safe_scale_min", "safe_scale_max",
               "safe_scale_ceiling", "ceiling_confidence",
               "safety_margin_reliability", "render_evidence_confidence",
               "safe_ceiling_exact_pop_spread_pct",
               "baseline_unclamped_disparity_mean_abs_pct")
    if any(not math.isfinite(float(row[key])) for key in numeric):
        raise RuntimeError(f"{origin}: non-finite policy target")
    geometry_hash = row["deployment_geometry_allowlist_sha256"]
    if (not isinstance(geometry_hash, str) or len(geometry_hash) != 64 or
            any(character not in "0123456789abcdef" for character in geometry_hash)):
        raise RuntimeError(f"{origin}: invalid deployment geometry identity")
    input_manifest = row.get("input_variant_manifest")
    label_merge.validate_policy_input_variant_manifest(input_manifest)
    input_manifest_hash = label_merge.input_variant_manifest_sha256(
        input_manifest
    )
    if row.get("input_variant_manifest_sha256") != input_manifest_hash:
        raise RuntimeError(f"{origin}: stale input-variant manifest identity")
    if row.get("depth_input_color_contract_sha256") != (
            input_color.color_contract_sha256()):
        raise RuntimeError(f"{origin}: stale depth input color contract")
    condition_targets = input_condition_target_map(row, origin)
    variants = row["deployment_geometry_variants"]
    expected_variant_count = sum(
        int(target["deployment_geometry_variant_count"])
        for target in condition_targets.values()
    )
    if (not isinstance(variants, list) or
            len(variants) != expected_variant_count):
        raise RuntimeError(
            f"{origin}: policy row requires {expected_variant_count} "
            "geometry/input evidence variants"
        )
    seen_render_variants = set()
    preprocessing_signatures = set()
    render_variants_by_input = {}
    allowed_input_variant_hashes = {
        input_color.input_variant_sha256(value)
        for value in input_manifest["variants"]
    }
    for variant in variants:
        if not isinstance(variant, dict):
            raise RuntimeError(f"{origin}: malformed deployment geometry variant")
        geometry = variant.get("geometry")
        validate_geometry_tuple(geometry)
        input_variant = variant.get(
            "input_variant", input_color.sdr_input_variant()
        )
        input_color.validate_input_variant(input_variant)
        input_variant_hash = input_color.input_variant_sha256(input_variant)
        if (variant.get("input_variant_sha256") != input_variant_hash or
                input_variant_hash not in allowed_input_variant_hashes):
            raise RuntimeError(
                f"{origin}: render variant has stale or undeclared input identity"
            )
        if geometry["color_mode"] != input_variant["color_mode"]:
            raise RuntimeError(
                f"{origin}: geometry and input variant color modes differ"
            )
        key = (
            tuple_key(geometry),
            input_variant_hash,
        )
        preprocessing_signatures.add((
            geometry["source_width"], geometry["source_height"],
            geometry["model_input_width"], geometry["model_input_height"],
            geometry["depth_short_side"], geometry["depth_max_aspect"],
        ))
        if key in seen_render_variants:
            raise RuntimeError(f"{origin}: duplicate deployment/input variant")
        seen_render_variants.add(key)
        render_variants_by_input.setdefault(input_variant_hash, []).append(variant)
        path = Path(variant.get("baseline_unclamped_disparity", ""))
        expected = variant.get("baseline_unclamped_disparity_sha256")
        if not expected or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(
                f"{origin}: geometry disparity artifact is missing or changed: {path}"
            )
        clamp = variant.get("artistic_full_clamp_abs")
        if (not isinstance(clamp, (int, float)) or isinstance(clamp, bool) or
                not math.isfinite(float(clamp)) or float(clamp) <= 0.0):
            raise RuntimeError(f"{origin}: geometry variant has invalid clamp")
        for field in ("safe_scale_min", "safe_scale_max",
                      "safety_margin_reliability"):
            if (not isinstance(variant.get(field), (int, float)) or
                    isinstance(variant.get(field), bool) or
                    not math.isfinite(float(variant[field]))):
                raise RuntimeError(
                    f"{origin}: geometry variant has invalid frontier evidence"
                )
        if not isinstance(variant.get("identity_feasible"), bool):
            raise RuntimeError(
                f"{origin}: geometry variant lacks identity-feasibility evidence"
            )
    if len(preprocessing_signatures) != 1:
        raise RuntimeError(
            f"{origin}: one RGB has conflicting model preprocessing geometries"
        )
    if set(render_variants_by_input) != set(condition_targets):
        raise RuntimeError(f"{origin}: condition targets and render evidence differ")
    for input_hash, target in condition_targets.items():
        evidence = render_variants_by_input[input_hash]
        if (len(evidence) != 2 or
                len({tuple_key(item["geometry"]) for item in evidence}) != 2):
            raise RuntimeError(
                f"{origin}: each condition requires two distinct deployment geometries"
            )
        identity_feasible = all(item["identity_feasible"] for item in evidence)
        if identity_feasible != target["identity_feasible"]:
            raise RuntimeError(
                f"{origin}: condition identity target differs from geometry evidence"
            )
        if identity_feasible:
            expected_min = max(float(item["safe_scale_min"]) for item in evidence)
            expected_ceiling = min(float(item["safe_scale_max"]) for item in evidence)
        else:
            expected_min = expected_ceiling = 1.0
        if (abs(float(target["safe_scale_min"]) - expected_min) > 1e-6 or
                abs(float(target["safe_scale_ceiling"]) - expected_ceiling) > 1e-6):
            raise RuntimeError(
                f"{origin}: condition target is not its two-geometry intersection"
            )
        expected_reliability = (
            min(float(item["safety_margin_reliability"]) for item in evidence)
            if is_actionable_scale(expected_ceiling) else 0.0
        )
        if abs(float(target["safety_margin_reliability"]) - expected_reliability) > 1e-6:
            raise RuntimeError(
                f"{origin}: condition reliability differs from geometry evidence"
            )
        for item in evidence:
            render_scale = float(
                item.get("safe_ceiling_render_target", {}).get(
                    "scale", math.nan
                )
            )
            if (not math.isfinite(render_scale) or
                    abs(render_scale - expected_ceiling) > 1e-6):
                raise RuntimeError(
                    f"{origin}: geometry render target differs from condition ceiling"
                )


def style_target_scale(row, style=None):
    style = style or "immersive"
    targets = row.get("style_targets", {})
    if style not in targets:
        raise RuntimeError(f"label row has no target for style {style}")
    value = targets[style]
    if isinstance(value, dict):
        for key in ("scale", "selected_scale", "baseline_multiplier"):
            if value.get(key) is not None:
                value = value[key]
                break
        else:
            raise RuntimeError(f"style target {style} has no scale")
    value = float(value)
    if not 1.0 - ART_SCALE_DELTA_MAX <= value <= 1.0 + ART_SCALE_DELTA_MAX:
        raise RuntimeError(f"style target {style} is outside the model bounds")
    return value


def safe_ceiling_confidence(row):
    return float(row.get("ceiling_confidence", row.get("confidence", 0.0)))


def is_actionable_scale(scale):
    return abs(float(scale) - 1.0) >= ACTION_EPSILON


def input_variant_runtime_regime(variant):
    """Derive the runtime regime from one authenticated input variant."""
    input_color.validate_input_variant(variant)
    if variant["kind"] == input_color.INPUT_KIND_SDR:
        return "sdr"
    if variant["kind"] in {
            input_color.INPUT_KIND_WINDOWS_HDR,
            input_color.INPUT_KIND_NATIVE_PQ}:
        return "hdr"
    raise RuntimeError("unsupported authenticated input runtime regime")


def row_runtime_regime(row):
    """Return derived in-memory regime metadata for an expanded policy sample."""
    variant = row.get("_input_variant")
    if variant is None:
        # Compatibility for focused unit-test rows. PolicyDataset always attaches
        # the authenticated variant before this helper reaches training.
        return "sdr"
    return input_variant_runtime_regime(variant)


def row_action(row):
    target = row.get("_condition_target", row)
    return is_actionable_scale(target.get("safe_scale_ceiling", 1.0))


def input_variant_acceptance_risk(prediction, target):
    """Order one RGB variant by runtime safety, then perceptual decision error."""
    scale, confidence = map(float, prediction[:2])
    target_scale, target_confidence = map(float, target[:2])
    if not all(math.isfinite(value) for value in (
            scale, confidence, target_scale, target_confidence)):
        raise RuntimeError("input-variant acceptance values must be finite")
    action = is_actionable_scale(target_scale)
    predicted_action = confidence >= 0.5
    effective = scale if predicted_action else 1.0
    target_effective = target_scale if action else 1.0
    return (
        max(effective - target_scale, 0.0),
        abs(effective - target_effective),
        (confidence - float(action)) ** 2,
    )


class PolicyDataset(Dataset):
    def __init__(self, rows):
        self.samples = []
        self.rows = []
        for row_index, row in enumerate(rows):
            condition_targets = input_condition_target_map(
                row, f"policy row {row_index}"
            )
            by_input = {}
            for render_variant in row["deployment_geometry_variants"]:
                variant = render_variant.get(
                    "input_variant", input_color.sdr_input_variant()
                )
                input_color.validate_input_variant(variant)
                key = input_color.input_variant_sha256(variant)
                group = by_input.setdefault(key, {
                    "input_variant": variant,
                    "render_variants": [],
                })
                if group["input_variant"] != variant:
                    raise RuntimeError("input-variant hash collision")
                group["render_variants"].append(render_variant)
            if not by_input:
                raise RuntimeError("policy row has no input variants")
            if set(by_input) != set(condition_targets):
                raise RuntimeError(
                    "policy row condition targets differ from render variants"
                )
            for key in sorted(by_input):
                group = by_input[key]
                if len(group["render_variants"]) != 2:
                    raise RuntimeError(
                        "each input variant must have exactly two deployment geometries"
                    )
                if len({
                        tuple_key(variant["geometry"])
                        for variant in group["render_variants"]
                }) != 2:
                    raise RuntimeError(
                        "each input variant must have two distinct deployment geometries"
                    )
                if any(
                        variant["geometry"]["color_mode"] !=
                        group["input_variant"]["color_mode"]
                        for variant in group["render_variants"]):
                    raise RuntimeError(
                        "input variant includes a cross-color deployment geometry"
                    )
                sample_row = dict(row)
                sample_row["_input_variant_sha256"] = key
                sample_row["_input_variant"] = group["input_variant"]
                sample_row["_render_variants"] = group["render_variants"]
                sample_row["_condition_target"] = condition_targets[key]
                # Downstream sampling, action coverage, shot latching, and
                # reporting operate on the expanded condition row.  Override
                # every effective target alias so none can accidentally fall
                # back to the diagnostic all-condition intersection.
                for field in CONDITION_TARGET_EFFECTIVE_FIELDS:
                    sample_row[field] = condition_targets[key][field]
                self.samples.append((
                    sample_row, group["input_variant"],
                    group["render_variants"], condition_targets[key],
                ))
                self.rows.append(sample_row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, input_variant, render_variants, condition_target = self.samples[index]
        bgr = cv2.imread(row["source"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["source"])
        # Production never upsamples the model tensor beyond the native capture raster:
        # video_depth_estimator.cpp passes min(1008, input dimension) as each profile bound.
        # Keep offline training/evaluation on that exact feature grid, especially for sources
        # whose short side is below the usual 434-pixel target.
        first_geometry = render_variants[0]["geometry"]
        if (bgr.shape[1], bgr.shape[0]) != (
                first_geometry["source_width"], first_geometry["source_height"]):
            raise RuntimeError("decoded RGB dimensions differ from deployment geometry")
        width = first_geometry["model_input_width"]
        height = first_geometry["model_input_height"]
        if input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ:
            model_source = Path(row["model_source"])
            scrgb = np.fromfile(model_source, dtype="<f2")
            expected = first_geometry["source_width"] * first_geometry[
                "source_height"
            ] * 4
            if scrgb.size != expected:
                raise RuntimeError("native HDR model-source payload size changed")
            scrgb = scrgb.reshape(
                first_geometry["source_height"],
                first_geometry["source_width"], 4,
            )
            image = input_color.preprocess_scrgb_f16_to_nchw(
                scrgb, width, height, input_variant
            )
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = input_color.preprocess_rgb8_to_nchw(
                rgb, width, height, input_variant
            )
        scale = float(condition_target["safe_scale_ceiling"])
        target = np.array([
            np.clip(scale,
                    1.0 - ART_SCALE_DELTA_MAX,
                    1.0 + ART_SCALE_DELTA_MAX),
            np.clip(safe_ceiling_confidence(condition_target), 0.0, 1.0),
            np.clip(condition_target["safe_scale_min"],
                    1.0 - ART_SCALE_DELTA_MAX, 1.0 + ART_SCALE_DELTA_MAX),
            np.clip(condition_target["safe_scale_max"],
                    1.0 - ART_SCALE_DELTA_MAX, 1.0 + ART_SCALE_DELTA_MAX),
            np.clip(condition_target["safety_margin_reliability"], 0.0, 1.0),
        ], dtype=np.float32)
        raw_disparity = []
        clamp_abs = []
        for variant in render_variants:
            field = sbsbench.load_float_texture(
                variant["baseline_unclamped_disparity"]
            ).astype(np.float32, copy=False)
            if field.size == 0 or not np.isfinite(field).all():
                raise RuntimeError("invalid exact unclamped geometry disparity artifact")
            geometry = variant["geometry"]
            if tuple(field.shape) != (
                    geometry["disparity_raster_height"],
                    geometry["disparity_raster_width"]):
                raise RuntimeError("geometry disparity shape differs from its exact tuple")
            scale_x = np.float32(geometry["content_scale_x"])
            scale_y = np.float32(geometry["content_scale_y"])
            field_height, field_width = field.shape
            x = ((np.arange(field_width, dtype=np.float32) + np.float32(0.5)) /
                 np.float32(field_width))
            y = ((np.arange(field_height, dtype=np.float32) + np.float32(0.5)) /
                 np.float32(field_height))
            lo_x = np.float32(0.5) * np.float32(np.float32(1.0) - scale_x)
            lo_y = np.float32(0.5) * np.float32(np.float32(1.0) - scale_y)
            valid_x = (x >= lo_x) & (x <= np.float32(lo_x + scale_x))
            valid_y = (y >= lo_y) & (y <= np.float32(lo_y + scale_y))
            field = field[valid_y][:, valid_x]
            if field.size == 0:
                raise RuntimeError("geometry disparity has no content-valid pixels")
            raw_disparity.append(torch.from_numpy(field.copy()))
            clamp_abs.append(float(variant["artistic_full_clamp_abs"]))
        return (torch.from_numpy(image.copy()), torch.from_numpy(target),
                raw_disparity, clamp_abs)


class CachedPolicyDataset(Dataset):
    def __init__(self, pooled, targets, raw_disparities, clamp_abs, rows):
        validate_expanded_variant_targets(rows, targets)
        validate_expanded_variant_frontiers(rows, raw_disparities)
        self.pooled = pooled
        self.targets = targets
        self.raw_disparities = raw_disparities
        self.clamp_abs = clamp_abs
        self.rows = rows
        self.peers = same_shot_peer_indices(rows)

    def __len__(self):
        return self.pooled.shape[0]

    def __getitem__(self, index):
        peer = self.peers[index]
        return (self.pooled[index], self.targets[index],
                self.pooled[peer], self.targets[peer],
                self.raw_disparities[index], self.clamp_abs[index])


def collate_policy_samples(batch):
    pooled, targets, peer_pooled, peer_targets, raw, clamp_abs = zip(*batch)
    return (torch.stack(pooled), torch.stack(targets), torch.stack(peer_pooled),
            torch.stack(peer_targets), list(raw), list(clamp_abs))


def validate_expanded_variant_targets(rows, targets):
    """Authenticate one complete source-family target set per source frame.

    Runtime-condition provenance never enters the model.  Distinct targets are
    nevertheless valid when the image-derived evidence is distinct; the
    equivalent-evidence audit below rejects discrepancies the model cannot infer.
    """
    if len(rows) != len(targets):
        raise RuntimeError("expanded policy targets do not match sample metadata")
    sdr_origin_variants = {
        input_color.input_variant_sha256(value)
        for value in label_merge.policy_input_variants()
        if value["kind"] != input_color.INPUT_KIND_NATIVE_PQ
    }
    native_variants = {
        input_color.input_variant_sha256(input_color.native_pq_input_variant())
    }
    groups = {}
    for index, row in enumerate(rows):
        source_identity = (
            row.get("film_id"), row.get("clip"), int(row.get("frame", index)),
            row.get("source_sha256"),
        )
        variant = row.get("_input_variant_sha256", "native-sdr")
        group = groups.setdefault(source_identity, {})
        if variant in group:
            raise RuntimeError("duplicate input variant for one source-frame target")
        group[variant] = index
    for source_identity, variants in groups.items():
        if frozenset(variants) not in {
                frozenset(sdr_origin_variants), frozenset(native_variants)}:
            raise RuntimeError(
                "source frame does not have one complete authenticated "
                f"source-family target set: {source_identity}"
            )
        for variant, index in variants.items():
            row = rows[index]
            condition = row.get("_condition_target")
            if (not isinstance(condition, dict) or
                    condition.get("input_variant_sha256") != variant):
                raise RuntimeError(
                    "expanded sample lacks its authenticated condition target"
                )
            expected = torch.tensor([
                float(condition["safe_scale_ceiling"]),
                float(condition["ceiling_confidence"]),
                float(condition["safe_scale_min"]),
                float(condition["safe_scale_max"]),
                float(condition["safety_margin_reliability"]),
            ], dtype=torch.float32)
            candidate = torch.as_tensor(targets[index], dtype=torch.float32)
            if (candidate.shape != expected.shape or
                    not torch.allclose(candidate, expected, rtol=0.0, atol=1e-7)):
                raise RuntimeError(
                    "expanded sample tensor differs from its condition target for "
                    f"source frame {source_identity}"
                )


def _frontier_geometry_key(geometry):
    return tuple(
        (key, geometry[key]) for key in sorted(geometry) if key != "color_mode"
    )


def _disparity_nrmse(first, second):
    first = torch.as_tensor(first, dtype=torch.float32)
    second = torch.as_tensor(second, dtype=torch.float32)
    if first.shape != second.shape or first.numel() == 0:
        return float("inf")
    difference = torch.sqrt(torch.mean((first - second) ** 2))
    magnitude = torch.maximum(
        torch.sqrt(torch.mean(first ** 2)),
        torch.sqrt(torch.mean(second ** 2)),
    ).clamp_min(1e-6)
    return float((difference / magnitude).cpu())


def validate_expanded_variant_frontiers(rows, raw_disparities):
    """Reject condition-target differences the image evidence cannot support.

    If two authenticated input conditions produce near-identical unclamped
    disparity at both output geometries but their condition ceilings differ by
    a full 0.1 grid step, that is evaluator/preprocessing inconsistency and must
    not be learned.
    """
    if len(rows) != len(raw_disparities):
        raise RuntimeError("expanded disparity evidence does not match samples")
    source_groups = {}
    for index, row in enumerate(rows):
        identity = (
            row.get("film_id"), row.get("clip"), int(row.get("frame", index)),
            row.get("source_sha256"),
        )
        render_variants = row.get("_render_variants")
        fields = raw_disparities[index]
        if (not isinstance(render_variants, list) or len(render_variants) != 2 or
                len(fields) != len(render_variants)):
            raise RuntimeError("frontier audit requires both deployment geometries")
        evidence = {}
        for variant, field in zip(render_variants, fields):
            geometry_key = _frontier_geometry_key(variant["geometry"])
            if geometry_key in evidence:
                raise RuntimeError("frontier audit has duplicate structural geometry")
            evidence[geometry_key] = field
        ceiling = float(row["safe_scale_ceiling"])
        if not math.isfinite(ceiling):
            raise RuntimeError("frontier audit has a non-finite condition ceiling")
        source_groups.setdefault(identity, []).append({
            "variant": row.get("_input_variant_sha256", "native-sdr"),
            "ceiling": ceiling,
            "evidence": evidence,
        })
    for identity, conditions in source_groups.items():
        for first_index, first in enumerate(conditions):
            for second in conditions[first_index + 1:]:
                if set(first["evidence"]) != set(second["evidence"]):
                    continue
                equivalent = all(
                    _disparity_nrmse(first["evidence"][key],
                                     second["evidence"][key]) <=
                    MAX_EQUIVALENT_DISPARITY_NRMSE
                    for key in first["evidence"]
                )
                if (equivalent and abs(first["ceiling"] - second["ceiling"]) >=
                        MAX_CONSISTENT_FRONTIER_DELTA - 1e-8):
                    raise RuntimeError(
                        "equivalent input-condition disparity has conflicting "
                        f"condition targets for source frame {identity}: "
                        f"{first['variant']}={first['ceiling']:.3f}, "
                        f"{second['variant']}={second['ceiling']:.3f}"
                    )


def same_shot_peer_indices(rows):
    """Pair adjacent frames in one complete shot without a last-to-first wrap."""
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((
            row.get("film_id"), row["clip"],
            row.get("_input_variant_sha256", "sdr"),
        ), []).append(index)
    peers = list(range(len(rows)))
    for indices in groups.values():
        indices.sort(key=lambda index: int(rows[index].get("frame", index)))
        if len(indices) > 1:
            for position, index in enumerate(indices):
                peers[index] = indices[position + 1] if position + 1 < len(indices) else indices[-2]
    return peers


def cache_policy_dataset(model, rows, device, batch_size=None):
    """Cache compact policy features plus full disparity fields for exact clamp-aware loss.

    The policy feature cache is O(frames), avoiding the former dense-token cost. Exact rendered
    supervision necessarily retains O(total disparity pixels) for every approved
    destination geometry.  There is still only one pooled RGB feature per labelled frame.
    """
    dataset = PolicyDataset(rows)
    pooled = []
    targets = []
    raw_disparities = []
    clamp_abs = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target, raw_disparity, disparity_clamp = dataset[index]
            image = image[None].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                feature = model.policy_features(image)
            pooled.append(feature[0].half().cpu())
            targets.append(target)
            raw_disparities.append(raw_disparity)
            clamp_abs.append(disparity_clamp)
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"cache {index + 1}/{len(dataset)}", flush=True)
    return CachedPolicyDataset(
        torch.stack(pooled), torch.stack(targets), raw_disparities,
        clamp_abs, dataset.rows
    )


def load_rows(paths, validate=False):
    if isinstance(paths, (str, Path)):
        paths = (Path(paths),)
    rows = []
    identities = set()
    rgb_rows = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                rgb_identity = row.get("source_sha256")
                if rgb_identity is not None and rgb_identity in rgb_rows:
                    if rgb_rows[rgb_identity] != row:
                        raise RuntimeError(
                            f"duplicate identical RGB has conflicting labels "
                            f"{rgb_identity} at {path}:{line_number}"
                        )
                    continue
                if rgb_identity is not None:
                    rgb_rows[rgb_identity] = row
                identity = (row.get("clip"), row.get("frame"))
                if identity in identities:
                    raise RuntimeError(f"duplicate artistic label {identity} at {path}:{line_number}")
                identities.add(identity)
                if validate:
                    validate_row(row, f"{path}:{line_number}")
                rows.append(row)
    return rows


def labels_contract(paths):
    sources = []
    fitter_contracts = []
    for path in paths:
        path = Path(path).resolve()
        summary_path = path.parent / "summary.json"
        fitter_path = path.parent / "label_fitter_contract.json"
        if not summary_path.is_file() or not fitter_path.is_file():
            raise RuntimeError(f"label bundle is incomplete beside {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fitter = json.loads(fitter_path.read_text(encoding="utf-8"))
        label_hash = sha256(path)
        fitter_hash = sha256(fitter_path)
        if (not isinstance(summary, dict) or not isinstance(fitter, dict) or
                not schema_is(summary.get("schema"), LABEL_SCHEMA) or
                not schema_is(fitter.get("schema"), LABEL_SCHEMA)):
            raise RuntimeError(f"obsolete label bundle beside {path}")
        fitter_config = fitter.get("label_fitter_config", {})
        if fitter_config.get("objective") != label_merge.OBJECTIVE:
            raise RuntimeError(f"incompatible safe-frontier objective beside {path}")
        confidence_semantics = str(fitter_config.get("confidence_semantics", ""))
        if ("action" not in confidence_semantics.lower() or
                "hard" not in confidence_semantics.lower()):
            raise RuntimeError(f"incompatible confidence contract beside {path}")
        reliability_semantics = str(
            fitter_config.get("reliability_semantics", "")
        )
        if "margin" not in reliability_semantics.lower():
            raise RuntimeError(f"incompatible reliability contract beside {path}")
        if (fitter_config.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                fitter.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                summary.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT):
            raise RuntimeError(
                f"incompatible condition target contract beside {path}"
            )
        policy_baseline = fitter.get("policy_baseline")
        if not isinstance(policy_baseline, dict) or not policy_baseline:
            raise RuntimeError(f"label bundle lacks policy baseline beside {path}")
        if summary.get("labels_sha256") != label_hash:
            raise RuntimeError(f"label bundle summary does not match {path}")
        if summary.get("label_fitter_contract_sha256") != fitter_hash:
            raise RuntimeError(f"label fitter contract does not match {path}")
        code = fitter.get("code", {})
        code_sha256 = label_merge.validate_label_fitter_code(
            code, LABEL_SCHEMA, f"label fitter contract beside {path}"
        )
        thresholds_path = verified_identity(
            fitter.get("thresholds"), "metric thresholds"
        )
        control_path = verified_identity(
            fitter.get("control"), "control results"
        )
        control = json.loads(control_path.read_text(encoding="utf-8"))
        metric_sha256 = control.get("meta", {}).get("metric_sha256")
        expected_metric = semantic_file_hash((
            Path(code["image_loader"]["path"]), thresholds_path,
            Path(code["evaluator_runner"]["path"]),
        ))
        if metric_sha256 != expected_metric:
            raise RuntimeError(
                f"label metric implementation changed: "
                f"{metric_sha256} != {expected_metric}"
            )
        geometry_allowlist = fitter.get("deployment_geometry_allowlist")
        validate_allowlist(geometry_allowlist)
        geometry_hash = allowlist_sha256(geometry_allowlist)
        if fitter.get("deployment_geometry_allowlist_sha256") != geometry_hash:
            raise RuntimeError(f"deployment geometry identity is stale beside {path}")
        input_manifest = fitter.get("input_variant_manifest")
        label_merge.validate_policy_input_variant_manifest(input_manifest)
        input_manifest_hash = label_merge.input_variant_manifest_sha256(
            input_manifest
        )
        if (fitter.get("input_variant_manifest_sha256") !=
                input_manifest_hash or
                fitter.get("depth_input_color_contract_sha256") !=
                input_color.color_contract_sha256()):
            raise RuntimeError(f"input color identity is stale beside {path}")
        allowed_input_variants = {
            input_color.input_variant_sha256(value): value
            for value in input_manifest["variants"]
        }
        allowed_by_key = {
            tuple_key(value): value for value in geometry_allowlist["tuples"]
        }
        allowed_tuples = set(allowed_by_key)
        seen_rgb = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("deployment_geometry_allowlist_sha256") != geometry_hash:
                    raise RuntimeError(
                        f"{path}:{line_number}: row has another deployment geometry identity"
                    )
                if (row.get("input_variant_manifest") != input_manifest or
                        row.get("input_variant_manifest_sha256") !=
                        input_manifest_hash or
                        row.get("depth_input_color_contract_sha256") !=
                        input_color.color_contract_sha256() or
                        row.get("condition_target_contract") !=
                        label_merge.CONDITION_TARGET_CONTRACT):
                    raise RuntimeError(
                        f"{path}:{line_number}: row has another input color identity"
                    )
                condition_targets = input_condition_target_map(
                    row, f"{path}:{line_number}"
                )
                rgb = row.get("source_sha256")
                if rgb in seen_rgb:
                    raise RuntimeError(
                        f"{path}:{line_number}: duplicate RGB survived geometry collapse"
                    )
                seen_rgb.add(rgb)
                actual_tuples = {
                    tuple_key(variant["geometry"])
                    for variant in row.get("deployment_geometry_variants", ())
                }
                if not actual_tuples or not actual_tuples <= allowed_tuples:
                    raise RuntimeError(
                        f"{path}:{line_number}: row uses an unapproved deployment geometry"
                    )
                groups = {
                    geometry_group_key(allowed_by_key[key]) for key in actual_tuples
                }
                if len(groups) != 1:
                    raise RuntimeError(
                        f"{path}:{line_number}: row mixes distinct RGB/model geometry groups"
                    )
                group = next(iter(groups))
                expected_tuples = {
                    key for key, value in allowed_by_key.items()
                    if (geometry_group_key(value) == group and
                        value["color_mode"] in {
                            allowed_input_variants[input_key]["color_mode"]
                            for input_key in condition_targets
                        })
                }
                if actual_tuples != expected_tuples:
                    raise RuntimeError(
                        f"{path}:{line_number}: row omits a matching deployment geometry variant"
                    )
                actual_pairs = {
                    (
                        tuple_key(variant["geometry"]),
                        variant.get("input_variant_sha256"),
                    )
                    for variant in row.get("deployment_geometry_variants", ())
                }
                expected_pairs = {
                    (geometry_key, input_key)
                    for geometry_key in expected_tuples
                    for input_key in condition_targets
                    for input_variant in (allowed_input_variants[input_key],)
                    if allowed_by_key[geometry_key]["color_mode"] ==
                    input_variant["color_mode"]
                }
                if actual_pairs != expected_pairs:
                    raise RuntimeError(
                        f"{path}:{line_number}: row omits a matching geometry/input variant"
                    )
        fitter_contracts.append({
            **{
                key: fitter.get(key)
                for key in ("schema", "label_fitter", "label_fitter_config",
                            "model_limits", "policy_contract",
                            "rendered_disparity_supervision", "policy_baseline",
                            "deployment_geometry_allowlist",
                            "deployment_geometry_allowlist_sha256",
                            "input_variant_manifest",
                            "input_variant_manifest_sha256",
                            "depth_input_color_contract_sha256",
                            "condition_target_contract")
            },
            "code_sha256": code_sha256,
            "metric_sha256": metric_sha256,
            "deployment_geometry_allowlist": geometry_allowlist,
            "deployment_geometry_allowlist_sha256": geometry_hash,
            "input_variant_manifest": input_manifest,
            "input_variant_manifest_sha256": input_manifest_hash,
            "depth_input_color_contract_sha256":
                input_color.color_contract_sha256(),
            "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
        })
        sources.append({
            "path": str(path),
            "sha256": label_hash,
            "summary_sha256": sha256(summary_path),
            "label_fitter_contract_sha256": fitter_hash,
            "policy_baseline": policy_baseline,
            "metric_sha256": metric_sha256,
            "deployment_geometry_allowlist": geometry_allowlist,
            "deployment_geometry_allowlist_sha256": geometry_hash,
            "input_variant_manifest": input_manifest,
            "input_variant_manifest_sha256": input_manifest_hash,
            "depth_input_color_contract_sha256":
                input_color.color_contract_sha256(),
            "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
        })
    metric_hashes = {source["metric_sha256"] for source in sources}
    if len(metric_hashes) != 1:
        raise RuntimeError("label bundles use different metric implementations")
    geometry_hashes = {
        source["deployment_geometry_allowlist_sha256"] for source in sources
    }
    if len(geometry_hashes) != 1:
        raise RuntimeError("label bundles use different deployment geometry allow-lists")
    input_hashes = {
        source["input_variant_manifest_sha256"] for source in sources
    }
    if len(input_hashes) != 1:
        raise RuntimeError("label bundles use different input-variant manifests")
    canonical = {
        json.dumps(contract, sort_keys=True) for contract in fitter_contracts
    }
    if len(canonical) != 1:
        raise RuntimeError("label bundles use different fitter contracts")
    fitter_identity = hashlib.sha256(next(iter(canonical)).encode()).hexdigest()
    for source in sources:
        source["label_fitter_identity_sha256"] = fitter_identity
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return sources, digest


def load_active_split(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or
            not schema_is(payload.get("schema"), 1)):
        raise RuntimeError(f"unsupported active split manifest: {path}")

    def require_hash(value, label):
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise RuntimeError(f"active split has invalid {label}")
        return value

    def referenced_path(value, label):
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"active split has no {label}")
        referenced = Path(value)
        if not referenced.is_absolute():
            referenced = path.parent / referenced
        referenced = referenced.resolve()
        if not referenced.is_file():
            raise RuntimeError(f"active split {label} is missing: {referenced}")
        return referenced

    catalog = referenced_path(payload.get("catalog"), "source catalog")
    expected_catalog_hash = require_hash(
        payload.get("catalog_sha256"), "catalog_sha256"
    )
    if sha256(catalog) != expected_catalog_hash:
        raise RuntimeError("active split source catalog hash is stale")

    split_productions = payload.get("split_productions", {})
    if not isinstance(split_productions, dict):
        raise RuntimeError("active split has invalid split_productions")
    assigned = {}
    for split in ("training", "development", "test"):
        productions = split_productions.get(split)
        if (not isinstance(productions, list) or not productions or
                any(not isinstance(value, str) or not value
                    for value in productions)):
            raise RuntimeError(f"active split has no {split} productions")
        if len(productions) != len(set(productions)):
            raise RuntimeError(f"active split repeats a {split} production")
        for production in productions:
            previous = assigned.setdefault(production, split)
            if previous != split:
                raise RuntimeError(
                    f"active split production {production!r} appears in both "
                    f"{previous} and {split}"
                )

    production_rows = payload.get("productions")
    if not isinstance(production_rows, list) or not production_rows:
        raise RuntimeError("active split has no production provenance")
    observed = {}
    videos = {}
    native_video_ids = {}
    native_capture_groups = {}
    dataset_manifests = set()
    for index, row in enumerate(production_rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"active split production {index} is invalid")
        production = row.get("production_id")
        split = row.get("split")
        if not isinstance(production, str) or not production or split not in assigned.values():
            raise RuntimeError(f"active split production {index} has invalid identity")
        if production in observed:
            raise RuntimeError(
                f"active split repeats production provenance for {production!r}"
            )
        if assigned.get(production) != split:
            raise RuntimeError(
                f"active split production provenance disagrees for {production!r}"
            )
        observed[production] = split

        dataset_manifest = referenced_path(
            row.get("dataset_manifest"),
            f"dataset manifest for {production}",
        )
        if dataset_manifest in dataset_manifests:
            raise RuntimeError("active split reuses one dataset manifest")
        dataset_manifests.add(dataset_manifest)
        expected_dataset_hash = require_hash(
            row.get("dataset_manifest_sha256"),
            f"dataset_manifest_sha256 for {production}",
        )
        if sha256(dataset_manifest) != expected_dataset_hash:
            raise RuntimeError(
                f"active split dataset manifest hash is stale for {production}"
            )
        try:
            dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"active split cannot read dataset manifest for {production}"
            ) from error
        dataset_schema = dataset.get("schema") if isinstance(dataset, dict) else None
        dataset_split = dataset.get("split") if isinstance(dataset, dict) else None
        if schema_is(dataset_schema, 1):
            dataset_production = dataset.get("film_id")
            dataset_kind = "authored-stereo"
        elif schema_is(dataset_schema, 2):
            dataset_production = dataset.get("production_id")
            dataset_kind = dataset.get("source_kind")
        else:
            dataset_production = dataset_kind = None
        recorded_schema = row.get("dataset_manifest_schema")
        recorded_kind = row.get("source_kind")
        if (dataset_production != production or dataset_split != split or
                (schema_is(dataset_schema, 2) and dataset_kind not in {
                    "mono-video", "native-hdr-video", "authored-stereo",
                    "gt-depth-flow"
                }) or
                (schema_is(dataset_schema, 2) and
                 not schema_is(recorded_schema, 2)) or
                (schema_is(dataset_schema, 2) and
                 recorded_kind != dataset_kind) or
                (recorded_schema is not None and
                 not schema_is(recorded_schema, dataset_schema)) or
                (recorded_kind is not None and recorded_kind != dataset_kind)):
            raise RuntimeError(
                f"active split dataset identity disagrees for {production}"
            )

        if dataset_kind == split_audit.NATIVE_HDR_SOURCE_KIND:
            identity = split_audit.native_hdr_source_identity(
                dataset, dataset_manifest, production, verify_media=False
            )
            if (row.get("source_identity_kind") !=
                    split_audit.NATIVE_HDR_COLLECTION_IDENTITY_KIND or
                    row.get("source_collection_sha256") !=
                    identity["source_collection_sha256"] or
                    row.get("source_videos") != identity["source_videos"] or
                    row.get("source_video_ids") != identity["source_video_ids"] or
                    row.get("source_capture_group_ids") !=
                    identity["source_capture_group_ids"] or
                    row.get("source_provenance") != identity["source_provenance"]):
                raise RuntimeError(
                    f"active split native HDR collection identity disagrees for "
                    f"{production}"
                )
            for record in identity["source_videos"]:
                for seen, label, value in (
                    (native_video_ids, "video id", record["video_id"]),
                    (videos, "video hash", record["video_sha256"]),
                    (native_capture_groups, "capture group", record["capture_group_id"]),
                ):
                    duplicate = seen.setdefault(value, production)
                    if duplicate != production:
                        raise RuntimeError(
                            "active split assigns the same native HDR source "
                            f"{label} to multiple productions: "
                            f"{duplicate!r}, {production!r}"
                        )
        else:
            video_hash = require_hash(
                row.get("video_sha256"), f"video_sha256 for {production}"
            )
            if dataset.get("video_sha256") != video_hash:
                raise RuntimeError(
                    f"active split dataset video identity disagrees for {production}"
                )
            duplicate = videos.setdefault(video_hash, production)
            if duplicate != production:
                raise RuntimeError(
                    "active split assigns the same source video to multiple productions: "
                    f"{duplicate!r}, {production!r}"
                )
    if observed != assigned:
        raise RuntimeError(
            "active split production provenance does not exactly match split_productions"
        )
    return payload, sha256(path)


def validate_rows_against_active_split(rows, active_split, allowed_splits):
    allowed_splits = set(allowed_splits)
    unexpected = sorted({row["split"] for row in rows} - allowed_splits)
    if unexpected:
        raise RuntimeError(
            "label bundle exposes disallowed splits: " + ", ".join(unexpected)
        )
    expected_by_split = active_split["split_productions"]
    for split in sorted(allowed_splits):
        actual = {row["film_id"] for row in rows if row["split"] == split}
        expected = set(expected_by_split[split])
        if actual != expected:
            raise RuntimeError(
                f"{split} label productions do not match the active split: "
                f"{sorted(actual)} != {sorted(expected)}"
            )


def resolve_split_clips(rows, split):
    clips = {row["clip"] for row in rows if row.get("split") == split}
    if not clips:
        raise RuntimeError(f"labels contain no {split} clips")
    return clips


def resolve_val_clips(rows, specification):
    """Compatibility helper used by tests; validation now means development."""
    if specification.strip().lower() == "auto":
        return resolve_split_clips(rows, "development")
    return {item.strip() for item in specification.split(",") if item.strip()}


def validate_global_film_split(first_rows, second_rows):
    first = {row.get("film_id") for row in first_rows
             if row.get("film_id")
             and float(row.get("global_policy_weight", 1.0)) > 0.0}
    second = {row.get("film_id") for row in second_rows
              if row.get("film_id")
              and float(row.get("global_policy_weight", 1.0)) > 0.0}
    overlap = first & second
    if overlap:
        raise RuntimeError("global-policy validation leaks complete films into training: "
                           + ", ".join(sorted(overlap)))


def balanced_sample_weights(rows):
    """Balance regime first, then action and image-derived conditions.

    Native SDR and aggregate HDR each receive half the sampling mass.  Action
    classes are balanced independently inside each regime because condition
    targets can differ.  The three HDR anchors split their regime/action cell,
    so retaining all anchors never gives HDR three times the influence.
    """
    regimes = set()
    actions = {}
    variants = {}
    domains = {}
    clips = {}
    frames = {}
    for row in rows:
        action = row_action(row)
        regime = row_runtime_regime(row)
        domain = row.get("domain") or "unknown"
        variant = row.get("_input_variant_sha256", "native-sdr")
        regimes.add(regime)
        actions.setdefault(regime, set()).add(action)
        variants.setdefault((regime, action), set()).add(variant)
        domains.setdefault((regime, action, variant), set()).add(domain)
        clips.setdefault((regime, action, variant, domain), set()).add(
            row["clip"]
        )
        key = (regime, action, variant, domain, row["clip"])
        frames[key] = frames.get(key, 0) + 1
    return [
        float(row.get("global_policy_weight", 1.0)) / (
            len(regimes)
            * len(actions[row_runtime_regime(row)])
            * len(variants[(row_runtime_regime(row), row_action(row))])
            * len(domains[(row_runtime_regime(row), row_action(row),
                           row.get("_input_variant_sha256", "native-sdr"))])
            * len(clips[(row_runtime_regime(row), row_action(row),
                         row.get("_input_variant_sha256", "native-sdr"),
                         row.get("domain") or "unknown")])
            * frames[(row_runtime_regime(row), row_action(row),
                      row.get("_input_variant_sha256", "native-sdr"),
                      row.get("domain") or "unknown", row["clip"])]
        )
        for row in rows
    ]


def exact_clamped_disparity_errors(predicted_scale, target_scale,
                                   raw_disparities, clamp_abs,
                                   return_gradient=False):
    """Exact differentiable field/edge error after Apollo's shipping clamp."""
    if raw_disparities is None or clamp_abs is None:
        raise ValueError("exact unclamped disparity fields are required")
    if len(raw_disparities) != predicted_scale.shape[0]:
        raise ValueError("exact disparity batch does not match predictions")
    predicted_scale = predicted_scale.float()
    target_scale = target_scale.float()
    errors = []
    gradient_errors = []
    for index, sample_fields in enumerate(raw_disparities):
        if isinstance(sample_fields, torch.Tensor):
            sample_fields = (sample_fields,)
        sample_limits = clamp_abs[index]
        if isinstance(sample_limits, torch.Tensor) and sample_limits.ndim == 0:
            sample_limits = (sample_limits,)
        elif isinstance(sample_limits, (int, float)):
            sample_limits = (sample_limits,)
        if len(sample_fields) != len(sample_limits) or not sample_fields:
            raise ValueError("geometry disparity fields and clamps do not match")
        geometry_errors = []
        geometry_gradient_errors = []
        for raw, raw_limit in zip(sample_fields, sample_limits):
            raw = raw.to(
                device=predicted_scale.device, dtype=torch.float32,
                non_blocking=True,
            )
            limit = torch.as_tensor(
                raw_limit, device=predicted_scale.device, dtype=torch.float32
            )
            predicted_field = torch.clamp(
                raw * predicted_scale[index], min=-limit, max=limit
            )
            target_field = torch.clamp(
                raw * target_scale[index], min=-limit, max=limit
            )
            # Normalize by each exact geometry's comfort magnitude, then use
            # the worst geometry so a low-resolution/easy raster cannot dilute
            # a deployment-specific artifact regression.
            field_error = (
                (predicted_field - target_field).abs().mean()
                / limit.clamp_min(1e-6)
            )
            geometry_errors.append(field_error)
            if raw.ndim >= 2:
                predicted_dx = predicted_field[..., 1:] - predicted_field[..., :-1]
                target_dx = target_field[..., 1:] - target_field[..., :-1]
                predicted_dy = predicted_field[..., 1:, :] - predicted_field[..., :-1, :]
                target_dy = target_field[..., 1:, :] - target_field[..., :-1, :]
                edge_terms = []
                if predicted_dx.numel():
                    edge_terms.append((predicted_dx - target_dx).abs().mean())
                if predicted_dy.numel():
                    edge_terms.append((predicted_dy - target_dy).abs().mean())
                geometry_gradient_errors.append(
                    torch.stack(edge_terms).mean() / limit.clamp_min(1e-6)
                    if edge_terms else field_error.new_zeros(())
                )
            else:
                geometry_gradient_errors.append(field_error.new_zeros(()))
        errors.append(torch.stack(geometry_errors).max())
        gradient_errors.append(torch.stack(geometry_gradient_errors).max())
    field_result = torch.stack(errors)
    if return_gradient:
        return field_result, torch.stack(gradient_errors)
    return field_result


def losses(predicted, target, paired_prediction=None,
           raw_disparities=None, clamp_abs=None):
    action = ((target[:, 0] - 1.0).abs() >= ACTION_EPSILON).to(target.dtype)
    # Identity is a real scale target, not an absence of supervision. Giving it
    # full weight prevents a confidence false positive from exposing an arbitrary
    # multiplier. The hard confidence target remains a calibrated action
    # probability; separate render evidence grades non-identity supervision.
    reliability = torch.where(action > 0.5, target[:, 4], 1.0)
    scale = F.smooth_l1_loss(
        (predicted[:, 0] - 1.0) / ART_SCALE_DELTA_MAX,
        (target[:, 0] - 1.0) / ART_SCALE_DELTA_MAX, reduction="none")
    style = (scale * reliability).sum() / reliability.sum().clamp_min(1e-6)
    # Do not factor this through mean(abs(D)): saturation makes the shipping
    # clamp nonlinear. Retain every raw disparity sample and clamp both rendered
    # fields exactly before measuring their mean difference.
    rendered, rendered_gradient = exact_clamped_disparity_errors(
        predicted[:, 0], target[:, 0], raw_disparities, clamp_abs,
        return_gradient=True,
    )
    rendered = (rendered * reliability).sum() / reliability.sum().clamp_min(1e-6)
    rendered_gradient = (
        rendered_gradient * reliability
    ).sum() / reliability.sum().clamp_min(1e-6)
    safety = (F.relu(target[:, 2] - predicted[:, 0]) +
              F.relu(predicted[:, 0] - target[:, 3]))
    safety = (safety * reliability).sum() / reliability.sum().clamp_min(1e-6)
    with torch.amp.autocast(device_type=predicted.device.type, enabled=False):
        confidence = F.binary_cross_entropy(predicted[:, 1].float(),
                                            target[:, 1].float())
    consistency = torch.zeros((), device=predicted.device, dtype=predicted.dtype)
    if paired_prediction is not None:
        scale_pair = F.smooth_l1_loss(
            (predicted[:, 0] - paired_prediction[:, 0]) / ART_SCALE_DELTA_MAX,
            torch.zeros_like(predicted[:, 0]))
        confidence_pair = F.mse_loss(predicted[:, 1], paired_prediction[:, 1])
        consistency = scale_pair + 0.2 * confidence_pair
    return (style + 0.5 * rendered + 0.2 * rendered_gradient +
            0.2 * confidence + 0.2 * safety
            + 0.1 * consistency), {
        "global_style": style.detach(), "rendered_disparity": rendered.detach(),
        "rendered_gradient": rendered_gradient.detach(),
        "global_conf": confidence.detach(), "safe_frontier": safety.detach(),
        "shot_consistency": consistency.detach()
    }


def run_epoch(model, loader, device, optimizer, scaler):
    training = optimizer is not None
    model.train(training)
    model.depth_model.eval()
    totals = {"loss": 0.0}
    batches = 0
    for (pooled, target, peer_pooled, _peer_target,
         raw_disparities, clamp_abs) in loader:
        pooled = pooled.to(device)
        target = target.to(device)
        peer_pooled = peer_pooled.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                predicted = model.forward_policy_features(pooled)
                peer_prediction = model.forward_policy_features(peer_pooled)
                loss, parts = losses(
                    predicted, target, peer_prediction,
                    raw_disparities, clamp_abs,
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        totals["loss"] += float(loss.detach())
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def calibration_error(probability, target, bins=10):
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = ((probability >= lower) &
                    (probability <= upper if upper >= 1.0 else probability < upper))
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean()))
    return error


def first_frame_indices(rows):
    """Return the earliest label for every complete shot and input condition."""
    first = {}
    for index, row in enumerate(rows):
        key = (
            row["film_id"], row["clip"],
            row.get("_input_variant_sha256", "single-input-variant"),
        )
        candidate = (int(row.get("frame", index)), index)
        if key not in first or candidate < first[key]:
            first[key] = candidate
    return [value[1] for _, value in sorted(first.items())]


def validate_action_coverage(rows, split):
    first = first_frame_indices(rows)
    actions = {}
    for index in first:
        row = rows[index]
        actions.setdefault(row_runtime_regime(row), set()).add(row_action(row))
    expected_regimes = {"sdr", "hdr"}
    if set(actions) != expected_regimes:
        raise RuntimeError(
            f"{split} action coverage requires both native SDR and HDR regimes"
        )
    for regime in sorted(expected_regimes):
        if True not in actions[regime]:
            raise RuntimeError(
                f"{split} {regime} has no actionable safe-ceiling shots; "
                "identity cannot select a model"
            )
        if False not in actions[regime]:
            raise RuntimeError(
                f"{split} {regime} has no identity safe-ceiling shots; "
                "confidence cannot be calibrated"
            )


def _mean_optional(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def film_balanced_acceptance(predicted, target, rows):
    """Evaluate each shot/input condition independently, then macro by film."""
    predicted = np.asarray(predicted, np.float64)
    target = np.asarray(target, np.float64)
    if (predicted.ndim != 2 or target.ndim != 2 or
            len(predicted) != len(rows) or len(target) != len(rows) or
            predicted.shape[1] != 2 or target.shape[1] < 2):
        raise ValueError("prediction metadata mismatch")
    action = np.abs(target[:, 0] - 1.0) >= ACTION_EPSILON
    predicted_action = predicted[:, 1] >= 0.5
    effective = np.where(predicted_action, predicted[:, 0], 1.0)
    target_effective = np.where(action, target[:, 0], 1.0)
    films = {}
    for index, row in enumerate(rows):
        variant = row.get(
            "_input_variant_sha256",
            row.get("input_variant_sha256", "single-input-variant"),
        )
        films.setdefault(row["film_id"], {}).setdefault(
            row["clip"], {}
        ).setdefault(variant, []).append(index)
    film_metrics = {}
    for film, clips in films.items():
        condition_clip_metrics = []
        condition_probabilities = {}
        condition_actions = {}
        for clip, variants in clips.items():
            reference_frames = None
            for variant, variant_indices in variants.items():
                ordered = sorted(
                    variant_indices,
                    key=lambda index: int(rows[index].get("frame", index)),
                )
                frames = [int(rows[index].get("frame", index)) for index in ordered]
                if len(frames) != len(set(frames)):
                    raise RuntimeError(
                        f"duplicate frames for {film}/{clip}/{variant}"
                    )
                if reference_frames is None:
                    reference_frames = frames
                elif frames != reference_frames:
                    raise RuntimeError(
                        f"input variants have different shot frames for {film}/{clip}"
                    )
                indices = np.asarray(ordered)
                first = indices[0]
                first_action = bool(action[first])
                first_prediction_action = bool(predicted_action[first])
                condition_probabilities.setdefault(variant, []).append(
                    float(predicted[first, 1])
                )
                condition_actions.setdefault(variant, []).append(
                    float(first_action)
                )
                condition_clip_metrics.append({
                    "first_frame_effective_scale_mae_pct": float(
                        abs(effective[first] - target_effective[first]) * 100.0
                    ),
                    "first_frame_raw_scale_mae_pct": float(
                        abs(predicted[first, 0] - target[first, 0]) * 100.0
                    ),
                    "first_frame_actionable_scale_mae_pct": (
                        float(abs(predicted[first, 0] - target[first, 0]) * 100.0)
                        if first_action else None
                    ),
                    "first_frame_action_brier": float(
                        (predicted[first, 1] - float(first_action)) ** 2
                    ),
                    "first_frame_action_recall_pct": (
                        100.0 if first_prediction_action else 0.0
                    ) if first_action else None,
                    "first_frame_identity_false_action_pct": (
                        100.0 if first_prediction_action else 0.0
                    ) if not first_action else None,
                    "within_shot_scale_std_pct": float(
                        np.std(predicted[indices, 0]) * 100.0
                    ),
                    "within_shot_confidence_std_pct": float(
                        np.std(predicted[indices, 1]) * 100.0
                    ),
                    "within_shot_action_flip_pct": float(
                        np.mean(
                            predicted_action[indices] != first_prediction_action
                        ) * 100.0
                    ),
                })
        if not condition_clip_metrics:
            raise RuntimeError(f"film {film} has no condition-specific shot metrics")
        film_metrics[film] = {
            key: _mean_optional([
                metrics[key] for metrics in condition_clip_metrics
            ])
            for key in condition_clip_metrics[0]
        }
        film_metrics[film]["action_ece"] = _mean_optional(
            calibration_error(
                np.asarray(condition_probabilities[variant]),
                np.asarray(condition_actions[variant]),
            )
            for variant in sorted(condition_probabilities)
        )
        recall = film_metrics[film]["first_frame_action_recall_pct"]
        false_action = film_metrics[film]["first_frame_identity_false_action_pct"]
        film_metrics[film]["first_frame_balanced_action_error_pct"] = (
            (100.0 - recall + false_action) * 0.5
            if recall is not None and false_action is not None else None
        )
    macro = {
        key: _mean_optional([metrics[key] for metrics in film_metrics.values()])
        for key in next(iter(film_metrics.values()))
    }
    macro_recall = macro["first_frame_action_recall_pct"]
    macro_false_action = macro["first_frame_identity_false_action_pct"]
    macro["first_frame_balanced_action_error_pct"] = (
        (100.0 - macro_recall + macro_false_action) * 0.5
        if macro_recall is not None and macro_false_action is not None else
        (100.0 - macro_recall if macro_recall is not None else None)
    )
    return {"macro": macro, "films": film_metrics}


def evaluate_acceptance(model, dataset, rows, device, batch_size=256):
    predictions = []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_policy_samples,
    )
    model.eval()
    with torch.inference_mode():
        for (pooled, _target, _peer_pooled, _peer_target,
             _raw_disparities, _clamp_abs) in loader:
            output = model.forward_policy_features(pooled.to(device))
            predictions.append(output.float().cpu())
    predicted = torch.cat(predictions).numpy()
    target = dataset.targets.float().numpy()
    return film_balanced_acceptance(predicted, target, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path, nargs="+")
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--depth-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.labels, validate=True)
    active_split, active_split_hash = load_active_split(args.split_manifest)
    validate_rows_against_active_split(
        rows, active_split, {"training", "development"}
    )
    train_rows = [row for row in rows if row["split"] == "training"]
    development_rows = [row for row in rows if row["split"] == "development"]
    if not train_rows or not development_rows:
        raise RuntimeError("training requires non-empty training and development splits")
    validate_global_film_split(train_rows, development_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtisticPolicyModel(load_depth_anything_small(
        args.depth_anything_root, args.depth_weights))
    # The same positional interpolation must be used in train, evaluation and export.
    use_dynamic_onnx_position_encoding(model)
    model.freeze_base()
    model.to(device)
    optimizer = torch.optim.AdamW(model.global_head.parameters(),
                                  lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print("Caching production-shaped pooled DA-V2 features...", flush=True)
    train_dataset = cache_policy_dataset(model, train_rows, device)
    dev_dataset = cache_policy_dataset(model, development_rows, device)
    validate_action_coverage(train_dataset.rows, "training")
    validate_action_coverage(dev_dataset.rows, "development")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        balanced_sample_weights(train_dataset.rows),
        len(train_dataset), replacement=True, generator=generator,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_policy_samples,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_policy_samples,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    label_sources, labels_digest = labels_contract(args.labels)
    contract = {
        "schema": TRAINING_SCHEMA, "policy_contract": POLICY_CONTRACT,
        "output_semantics": POLICY_OUTPUT_SEMANTICS,
        "policy_feature_contract": POLICY_FEATURE_CONTRACT,
        "policy_baseline": label_sources[0]["policy_baseline"],
        "metric_sha256": label_sources[0]["metric_sha256"],
        "deployment_geometry_allowlist": label_sources[0][
            "deployment_geometry_allowlist"
        ],
        "deployment_geometry_allowlist_sha256": label_sources[0][
            "deployment_geometry_allowlist_sha256"
        ],
        "input_variant_manifest": label_sources[0][
            "input_variant_manifest"
        ],
        "input_variant_manifest_sha256": label_sources[0][
            "input_variant_manifest_sha256"
        ],
        "depth_input_color_contract_sha256": label_sources[0][
            "depth_input_color_contract_sha256"
        ],
        "condition_target_contract": label_sources[0][
            "condition_target_contract"
        ],
        "labels": label_sources, "labels_sha256": labels_digest,
        "label_fitter_identity_sha256": (
            label_sources[0]["label_fitter_identity_sha256"]
        ),
        "active_split": str(args.split_manifest.resolve()),
        "active_split_sha256": active_split_hash,
        "depth_weights_sha256": sha256(args.depth_weights),
        "train_clips": sorted({row["clip"] for row in train_rows}),
        "development_clips": sorted({row["clip"] for row in development_rows}),
        "sealed_test_productions": active_split["split_productions"]["test"],
        "preprocessing": (
            "authenticated production SDR/HDR color transform, aspect-aligned "
            "linear sampling, and dynamic position encoding"
        ),
        "sampling": (
            "equal SDR/HDR runtime regimes; equal condition-specific action classes "
            "inside each regime; equal HDR white anchors, domains, and clips inside "
            "each regime/action cell; adjacent same-shot same-condition pairs"
        ),
        "objectives": ["per-input-condition safe-scale ceiling",
                       "condition-local worst-geometry post-clamp disparity and gradients",
                       "hard actionable probability",
                       "safety-margin reliability weighting",
                       "safe-bound containment", "same-shot consistency"],
        "checkpoint_selection": (
            "shot-first clip-then-film effective ceiling MAE, balanced action error, "
            "raw actionable ceiling MAE, Brier, ECE and shot variation"
        ),
        "seed": args.seed,
    }
    (args.output / "training_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    history = []
    best_key = (float("inf"),) * 6
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer, scaler)
        with torch.no_grad():
            development = run_epoch(model, dev_loader, device, None, scaler)
            acceptance = evaluate_acceptance(
                model, dev_dataset, dev_dataset.rows, device
            )
        development["acceptance"] = acceptance
        history.append({"epoch": epoch, "training": training,
                        "development": development})
        print(json.dumps(history[-1]), flush=True)
        macro = acceptance["macro"]
        if (macro["first_frame_actionable_scale_mae_pct"] is None or
                macro["first_frame_balanced_action_error_pct"] is None):
            raise RuntimeError("development acceptance lacks actionable shots")
        selection_key = (
            macro["first_frame_effective_scale_mae_pct"],
            macro["first_frame_balanced_action_error_pct"],
            macro["first_frame_actionable_scale_mae_pct"],
            macro["first_frame_action_brier"], macro["action_ece"],
            macro["within_shot_scale_std_pct"],
        )
        if selection_key < best_key:
            best_key = selection_key
            torch.save({
                "schema": TRAINING_SCHEMA,
                "policy_contract": contract["policy_contract"],
                "output_semantics": contract["output_semantics"],
                "policy_feature_contract": POLICY_FEATURE_CONTRACT,
                "policy_baseline": contract["policy_baseline"],
                "metric_sha256": contract["metric_sha256"],
                "deployment_geometry_allowlist": contract[
                    "deployment_geometry_allowlist"
                ],
                "deployment_geometry_allowlist_sha256": contract[
                    "deployment_geometry_allowlist_sha256"
                ],
                "input_variant_manifest": contract[
                    "input_variant_manifest"
                ],
                "input_variant_manifest_sha256": contract[
                    "input_variant_manifest_sha256"
                ],
                "depth_input_color_contract_sha256": contract[
                    "depth_input_color_contract_sha256"
                ],
                "condition_target_contract": contract[
                    "condition_target_contract"
                ],
                "policy_state": policy_state_dict(model), "epoch": epoch,
                "development_loss": development["loss"],
                "development_acceptance": acceptance,
                "checkpoint_selection_key": selection_key,
                "development_clips": contract["development_clips"],
                "sealed_test_productions": contract["sealed_test_productions"],
                "active_split_sha256": active_split_hash,
                "labels_sha256": labels_digest,
                "label_fitter_identity_sha256": (
                    contract["label_fitter_identity_sha256"]
                ),
                "depth_weights_sha256": contract["depth_weights_sha256"],
                "bounds": {"scale_delta_max": ART_SCALE_DELTA_MAX},
            }, args.output / "artistic_policy_best.pt")
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
