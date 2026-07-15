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


GEOMETRY_GROUP_FIELDS = (
    "source_width", "source_height", "model_input_width", "model_input_height",
    "depth_short_side", "depth_max_aspect", "color_mode",
)


def geometry_group_key(value):
    """Geometry dimensions the RGB/backbone observes, excluding destination-eye variants."""
    return tuple(value[field] for field in GEOMETRY_GROUP_FIELDS)


SBSBENCH_DIR = Path(__file__).resolve().parents[1] / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))
import sbsbench  # noqa: E402


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_WIDTH = 1008
MAX_HEIGHT = 1008
LABEL_SCHEMA = 9
TRAINING_SCHEMA = POLICY_CHECKPOINT_SCHEMA
SUPPORTED_STYLES = {"immersive", "balanced", "clean", "authored"}
ACTION_EPSILON = 0.005


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


def validate_row(row, origin="label row"):
    if int(row.get("label_schema", 0)) != LABEL_SCHEMA:
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
                "deployment_geometry_variants",
                "baseline_unclamped_disparity",
                "baseline_unclamped_disparity_sha256",
                "baseline_unclamped_disparity_mean_abs_pct"):
        if row.get(key) is None:
            raise RuntimeError(f"{origin}: missing {key}")
    source = Path(row["source"])
    if not source.is_file() or sha256(source) != row["source_sha256"]:
        raise RuntimeError(f"{origin}: source file is missing or changed: {source}")
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
    if (safe_min > 1.0 + 1e-6 or safe_max < 1.0 - 1e-6 or
            abs(ceiling - safe_max) > 1e-6):
        raise RuntimeError(f"{origin}: ceiling does not match connected safe frontier")
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
    variants = row["deployment_geometry_variants"]
    if not isinstance(variants, list) or len(variants) < 2:
        raise RuntimeError(f"{origin}: labels were not collapsed across multiple geometries")
    seen_geometries = set()
    preprocessing_signatures = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise RuntimeError(f"{origin}: malformed deployment geometry variant")
        geometry = variant.get("geometry")
        validate_geometry_tuple(geometry)
        key = tuple_key(geometry)
        preprocessing_signatures.add((
            geometry["source_width"], geometry["source_height"],
            geometry["model_input_width"], geometry["model_input_height"],
            geometry["depth_short_side"], geometry["depth_max_aspect"],
            geometry["color_mode"],
        ))
        if key in seen_geometries:
            raise RuntimeError(f"{origin}: duplicate deployment geometry variant")
        seen_geometries.add(key)
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
    if len(preprocessing_signatures) != 1:
        raise RuntimeError(
            f"{origin}: one RGB has conflicting model preprocessing geometries"
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


def row_action(row):
    return is_actionable_scale(row.get("safe_scale_ceiling", 1.0))


class PolicyDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        bgr = cv2.imread(row["source"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["source"])
        # Production never upsamples the model tensor beyond the native capture raster:
        # video_depth_estimator.cpp passes min(1008, input dimension) as each profile bound.
        # Keep offline training/evaluation on that exact feature grid, especially for sources
        # whose short side is below the usual 434-pixel target.
        first_geometry = row["deployment_geometry_variants"][0]["geometry"]
        if (bgr.shape[1], bgr.shape[0]) != (
                first_geometry["source_width"], first_geometry["source_height"]):
            raise RuntimeError("decoded RGB dimensions differ from deployment geometry")
        width = first_geometry["model_input_width"]
        height = first_geometry["model_input_height"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Production uses a linear texture sampler for the model-input resize.
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        image = rgb.astype(np.float32) / 255.0
        image = ((image - MEAN) / STD).transpose(2, 0, 1)
        scale = float(row["safe_scale_ceiling"])
        target = np.array([
            np.clip(scale,
                    1.0 - ART_SCALE_DELTA_MAX,
                    1.0 + ART_SCALE_DELTA_MAX),
            np.clip(safe_ceiling_confidence(row), 0.0, 1.0),
            np.clip(row["safe_scale_min"],
                    1.0 - ART_SCALE_DELTA_MAX, 1.0 + ART_SCALE_DELTA_MAX),
            np.clip(row["safe_scale_max"],
                    1.0 - ART_SCALE_DELTA_MAX, 1.0 + ART_SCALE_DELTA_MAX),
            np.clip(row["safety_margin_reliability"], 0.0, 1.0),
        ], dtype=np.float32)
        raw_disparity = []
        clamp_abs = []
        for variant in row["deployment_geometry_variants"]:
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


def same_shot_peer_indices(rows):
    """Pair adjacent frames in one complete shot without a last-to-first wrap."""
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row.get("film_id"), row["clip"]), []).append(index)
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
        clamp_abs, rows
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
        if (int(summary.get("schema", 0)) != LABEL_SCHEMA or
                int(fitter.get("schema", 0)) != LABEL_SCHEMA):
            raise RuntimeError(f"obsolete label bundle beside {path}")
        fitter_config = fitter.get("label_fitter_config", {})
        if fitter_config.get("objective") != (
                "multi-geometry-connected-safe-frontier-intersection-multistyle"):
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
        policy_baseline = fitter.get("policy_baseline")
        if not isinstance(policy_baseline, dict) or not policy_baseline:
            raise RuntimeError(f"label bundle lacks policy baseline beside {path}")
        if summary.get("labels_sha256") != label_hash:
            raise RuntimeError(f"label bundle summary does not match {path}")
        if summary.get("label_fitter_contract_sha256") != fitter_hash:
            raise RuntimeError(f"label fitter contract does not match {path}")
        code = fitter.get("code", {})
        required_code = {
            "label_fitter", "policy_contract", "label_preparation", "image_loader",
            "geometry_merge", "evaluator_runner",
        }
        if set(code) != required_code:
            raise RuntimeError(f"label fitter code contract is incomplete: {path}")
        for role, identity in code.items():
            code_path = Path(identity.get("path", ""))
            if (not code_path.is_file() or
                    sha256(code_path) != identity.get("sha256")):
                raise RuntimeError(
                    f"label fitter code changed for {role}: {code_path}"
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
                    if geometry_group_key(value) == group
                }
                if actual_tuples != expected_tuples:
                    raise RuntimeError(
                        f"{path}:{line_number}: row omits a matching deployment geometry variant"
                    )
        fitter_contracts.append({
            **{
                key: fitter.get(key)
                for key in ("schema", "label_fitter", "label_fitter_config",
                            "model_limits", "policy_contract",
                            "rendered_disparity_supervision", "policy_baseline",
                            "deployment_geometry_allowlist",
                            "deployment_geometry_allowlist_sha256")
            },
            "code_sha256": {
                role: identity["sha256"] for role, identity in code.items()
            },
            "metric_sha256": metric_sha256,
            "deployment_geometry_allowlist": geometry_allowlist,
            "deployment_geometry_allowlist_sha256": geometry_hash,
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
        })
    metric_hashes = {source["metric_sha256"] for source in sources}
    if len(metric_hashes) != 1:
        raise RuntimeError("label bundles use different metric implementations")
    geometry_hashes = {
        source["deployment_geometry_allowlist_sha256"] for source in sources
    }
    if len(geometry_hashes) != 1:
        raise RuntimeError("label bundles use different deployment geometry allow-lists")
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
    if not isinstance(payload, dict) or int(payload.get("schema", 0)) != 1:
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
        if (not isinstance(dataset, dict) or dataset.get("schema") != 1 or
                dataset.get("film_id") != production or
                dataset.get("split") != split):
            raise RuntimeError(
                f"active split dataset identity disagrees for {production}"
            )

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
    """Equalize ceiling-action class, domain, and clip before source weights."""
    domains = {}
    clips = {}
    frames = {}
    for row in rows:
        action = row_action(row)
        domain = row.get("domain") or "unknown"
        domains.setdefault(action, set()).add(domain)
        clips.setdefault((action, domain), set()).add(row["clip"])
        key = (action, domain, row["clip"])
        frames[key] = frames.get(key, 0) + 1
    actions = set(domains)
    return [
        float(row.get("global_policy_weight", 1.0)) / (
            len(actions)
            * len(domains[row_action(row)])
            * len(clips[(row_action(row), row.get("domain") or "unknown")])
            * frames[(row_action(row), row.get("domain") or "unknown", row["clip"])]
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
    """Return the earliest available authored label for every complete shot."""
    first = {}
    for index, row in enumerate(rows):
        key = (row["film_id"], row["clip"])
        candidate = (int(row.get("frame", index)), index)
        if key not in first or candidate < first[key]:
            first[key] = candidate
    return [value[1] for _, value in sorted(first.items())]


def validate_action_coverage(rows, split):
    first = first_frame_indices(rows)
    actions = {row_action(rows[index]) for index in first}
    if True not in actions:
        raise RuntimeError(
            f"{split} has no actionable safe-ceiling shots; identity cannot select a model"
        )


def _mean_optional(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def film_balanced_acceptance(predicted, target, rows):
    """Evaluate the shot-latched first-frame decision with film-balanced metrics."""
    predicted = np.asarray(predicted, np.float64)
    target = np.asarray(target, np.float64)
    if len(predicted) != len(rows) or predicted.shape[1] != 2:
        raise ValueError("prediction metadata mismatch")
    action = np.abs(target[:, 0] - 1.0) >= ACTION_EPSILON
    predicted_action = predicted[:, 1] >= 0.5
    effective = np.where(predicted_action, predicted[:, 0], 1.0)
    target_effective = np.where(action, target[:, 0], 1.0)
    films = {}
    for index, row in enumerate(rows):
        films.setdefault(row["film_id"], {}).setdefault(row["clip"], []).append(index)
    film_metrics = {}
    for film, clips in films.items():
        clip_metrics = []
        first_probabilities = []
        first_actions = []
        for indices in clips.values():
            indices = np.asarray(sorted(
                indices, key=lambda index: int(rows[index].get("frame", index))
            ))
            first = indices[0]
            first_action = bool(action[first])
            first_prediction_action = bool(predicted_action[first])
            first_probabilities.append(predicted[first, 1])
            first_actions.append(float(first_action))
            clip_metrics.append({
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
                    np.mean(predicted_action[indices] != first_prediction_action) * 100.0
                ),
            })
        film_metrics[film] = {
            key: _mean_optional([clip[key] for clip in clip_metrics])
            for key in clip_metrics[0]
        }
        film_metrics[film]["action_ece"] = calibration_error(
            np.asarray(first_probabilities), np.asarray(first_actions)
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
    validate_action_coverage(train_rows, "training")
    validate_action_coverage(development_rows, "development")

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
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(balanced_sample_weights(train_rows),
                                    len(train_rows), replacement=True,
                                    generator=generator)
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
        "preprocessing": "production aspect-aligned linear resize and dynamic position encoding",
        "sampling": (
            "equal ceiling-action classes, domains and clips, with adjacent same-shot pairs"
        ),
        "objectives": ["all-shot safe-scale ceiling",
                       "exact post-clamp rendered disparity and gradients",
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
                model, dev_dataset, development_rows, device
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
