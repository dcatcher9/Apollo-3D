#!/usr/bin/env python3
"""Evaluate a shallow DA-V2-statistics controller on held-out shots.

This is deliberately not a replacement training pipeline.  It asks a narrower
question: can eight scale-invariant statistics that DA-V2 already computes
separate shots where Apollo's scale-1.1 render is safe from shots where the
render-feasibility gates reject it?

Training labels come from an exact scale-1.0/1.1 render pair.  Development
labels come from the authenticated native-SDR condition in a separately
prepared safe-frontier bundle.  Features are read only from the first
monocular RGB frame of each shot; candidate SBS images, target metrics, later
frames, and authored right eyes are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import select_render_feasible_labels as selector
import depth_input_color as input_color
import merge_artistic_geometry_labels as label_merge
from artistic_policy_model import (
    POLICY_FEATURE_CONTRACT,
    POLICY_STAT_SIZE,
    ArtisticPolicyModel,
    load_depth_anything_small,
)


SCHEMA = 1
TARGET_SCALE = 1.1
SIMPLE_FEATURE_CONTRACT = "first-frame-dav2-depth-dpt-global-stats-v1"
MIN_USEFUL_SAFE_RECALL_PCT = 75.0
FEATURE_NAMES = (
    "depth_mean_over_rms",
    "depth_variation_over_rms",
    "depth_edge_x_over_rms",
    "depth_edge_y_over_rms",
    "dpt_mean_over_rms",
    "dpt_variation_over_rms",
    "dpt_edge_x_over_rms",
    "dpt_edge_y_over_rms",
)
DIAGNOSTIC_THRESHOLDS = (0.5, 0.75, 0.9, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_tree_sha256(root: Path) -> str:
    root = Path(root).resolve()
    paths = sorted(root.rglob("*.py"))
    if not paths:
        raise RuntimeError(f"source tree contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def git_state() -> dict:
    root = Path(__file__).resolve().parents[2]

    def command(*arguments):
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=True,
            text=True, capture_output=True,
        )
        return result.stdout.strip()

    return {
        "sha": command("rev-parse", "HEAD"),
        "dirty": bool(command("status", "--porcelain")),
    }


def first_frame(directory: Path) -> Path:
    frames = sorted(Path(directory).glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"shot has no source frames: {directory}")
    return frames[0].resolve()


def _shot_identity(clip: str, meta: dict) -> tuple[str, str]:
    film = str(meta.get("film_id") or meta.get("dataset") or clip.split("_shot_")[0])
    domain = str(meta.get("domain") or meta.get("dataset") or film)
    return film, domain


def load_training_shots(control_path: Path, candidate_path: Path,
                        thresholds_path: Path) -> tuple[list[dict], dict]:
    """Build safe-at-1.1 shot labels through the canonical selector gates."""
    control_path = Path(control_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    thresholds_path = Path(thresholds_path).resolve()
    control = load_json(control_path)
    candidate = load_json(candidate_path)

    selector.validate_metric_contract(control.get("meta", {}), thresholds_path)
    selector.validate_context(control, candidate, TARGET_SCALE, candidate_path)
    if not math.isclose(
            float(control.get("meta", {}).get("artistic_scale_override", 0.0)),
            1.0, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("training control is not exact artistic scale 1.0")

    control_clips = control.get("clips", {})
    candidate_clips = candidate.get("clips", {})
    if set(control_clips) != set(candidate_clips):
        raise RuntimeError("training control/candidate clip identities differ")
    clips_root = Path(control.get("meta", {}).get("clips_root", "")).resolve()
    if not clips_root.is_dir():
        raise RuntimeError(f"training clips root does not exist: {clips_root}")

    metric_specs = load_json(thresholds_path)["metrics"]
    samples = []
    violation_counts = {}
    for clip in sorted(control_clips):
        control_entry = control_clips[clip]
        candidate_entry = candidate_clips[clip]
        control_agg = selector.project_protected_worst_metrics(
            control_entry, metric_specs, f"control/{clip}"
        )
        candidate_agg = selector.project_protected_worst_metrics(
            candidate_entry, metric_specs, f"candidate/{clip}"
        )
        identity_violations = selector.feasibility_violations(
            control_agg, control_agg, metric_specs,
            control_entry.get("meta", {}),
        )
        if identity_violations:
            raise RuntimeError(
                f"{clip}: scale-1 identity render fails its own constraints: "
                f"{identity_violations}"
            )
        violations = selector.feasibility_violations(
            control_agg, candidate_agg, metric_specs,
            control_entry.get("meta", {}),
        )
        for violation in violations:
            metric = violation.split(":", 1)[0]
            violation_counts[metric] = violation_counts.get(metric, 0) + 1
        clip_meta = control_entry.get("meta", {})
        model_width = int(clip_meta.get("model_input_width", 0))
        model_height = int(clip_meta.get("model_input_height", 0))
        if model_width <= 0 or model_height <= 0:
            raise RuntimeError(f"{clip}: missing production model-input geometry")
        film, domain = _shot_identity(clip, clip_meta)
        samples.append({
            "split": "training",
            "clip": clip,
            "film_id": film,
            "domain": domain,
            "source": str(first_frame(clips_root / clip)),
            "model_input_width": model_width,
            "model_input_height": model_height,
            "safe": not violations,
            "safe_scale_ceiling": TARGET_SCALE if not violations else 1.0,
            "violations": violations,
        })
    provenance = {
        "control": {"path": str(control_path), "sha256": sha256(control_path)},
        "candidate": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        "thresholds": {"path": str(thresholds_path), "sha256": sha256(thresholds_path)},
        "metric_sha256": control["meta"]["metric_sha256"],
        "policy_warp_source_sha256": control["meta"]["policy_warp_source_sha256"],
        "violation_counts": violation_counts,
    }
    return samples, provenance


def native_sdr_condition_target(row: dict, origin: str) -> dict:
    """Select the target matching this diagnostic's native-SDR preprocessing.

    The schema-10 labels intentionally retain independent safety targets for
    native SDR and every authenticated Windows-HDR white anchor.  This shallow
    controller still measures only native-SDR DA-V2 statistics, so consuming the
    diagnostic all-condition alias (or an HDR target) would pair unlike evidence.
    """
    if row.get("condition_target_contract") != label_merge.CONDITION_TARGET_CONTRACT:
        raise RuntimeError(f"{origin}: incompatible condition-target contract")
    targets = row.get("input_condition_targets")
    if not isinstance(targets, list):
        raise RuntimeError(f"{origin}: missing input-condition safety targets")
    sdr = input_color.sdr_input_variant()
    sdr_hash = input_color.input_variant_sha256(sdr)
    matches = [
        target for target in targets
        if isinstance(target, dict)
        and target.get("input_variant_sha256") == sdr_hash
        and target.get("input_variant") == sdr
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{origin}: expected one authenticated native-SDR target")
    target = matches[0]
    if (target.get("schema") != label_merge.CONDITION_TARGET_SCHEMA or
            target.get("contract") != label_merge.CONDITION_TARGET_CONTRACT):
        raise RuntimeError(f"{origin}: malformed native-SDR condition target")
    return target


def load_development_shots(labels_path: Path, metric_sha256: str,
                           warp_sha256: str) -> tuple[list[dict], dict]:
    """Load first-frame native-SDR evidence and its own shot target per clip."""
    labels_path = Path(labels_path).resolve()
    rows_by_clip = {}
    with labels_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("label_schema", 0)) != label_merge.LABEL_SCHEMA:
                raise RuntimeError(
                    f"{labels_path}:{line_number}: expected current schema-"
                    f"{label_merge.LABEL_SCHEMA} label"
                )
            if row.get("policy_contract") != selector.POLICY_CONTRACT:
                raise RuntimeError(
                    f"{labels_path}:{line_number}: incompatible policy contract"
                )
            if row.get("metric_sha256") != metric_sha256:
                raise RuntimeError(
                    f"{labels_path}:{line_number}: metric contract differs from training"
                )
            if row.get("policy_warp_source_sha256") != warp_sha256:
                raise RuntimeError(
                    f"{labels_path}:{line_number}: warp contract differs from training"
                )
            if row.get("split") != "development":
                raise RuntimeError(
                    f"{labels_path}:{line_number}: row is not development data"
                )
            target = native_sdr_condition_target(
                row, f"{labels_path}:{line_number}"
            )
            rows_by_clip.setdefault(str(row["clip"]), []).append((row, target))
    if not rows_by_clip:
        raise RuntimeError("development label bundle is empty")

    geometry_hashes = {
        str(row.get("deployment_geometry_allowlist_sha256", ""))
        for rows in rows_by_clip.values() for row, _target in rows
    }
    if (len(geometry_hashes) != 1 or not next(iter(geometry_hashes)) or
            len(next(iter(geometry_hashes))) != 64):
        raise RuntimeError("development labels have no unique geometry allow-list")

    samples = []
    for clip, row_targets in sorted(rows_by_clip.items()):
        rows = [row for row, _target in row_targets]
        scales = {
            float(target["safe_scale_ceiling"])
            for _row, target in row_targets
        }
        films = {str(row["film_id"]) for row in rows}
        domains = {str(row["domain"]) for row in rows}
        geometries = {
            (int(row["model_input_width"]), int(row["model_input_height"]))
            for row in rows
        }
        if len(scales) != 1 or len(films) != 1 or len(domains) != 1:
            raise RuntimeError(f"{clip}: development shot metadata is inconsistent")
        if len(geometries) != 1:
            raise RuntimeError(f"{clip}: model-input geometry changes within the shot")
        source_directories = {Path(row["source"]).resolve().parent for row in rows}
        if len(source_directories) != 1:
            raise RuntimeError(f"{clip}: source directory changes within the shot")
        model_width, model_height = next(iter(geometries))
        ceiling = next(iter(scales))
        samples.append({
            "split": "development",
            "clip": clip,
            "film_id": next(iter(films)),
            "domain": next(iter(domains)),
            "source": str(first_frame(next(iter(source_directories)))),
            "model_input_width": model_width,
            "model_input_height": model_height,
            "safe": ceiling >= TARGET_SCALE - 1e-8,
            "safe_scale_ceiling": ceiling,
            "violations": [],
        })
    return samples, {
        "labels": {"path": str(labels_path), "sha256": sha256(labels_path)},
        "shot_count": len(samples),
        "input_condition": "native-sdr",
        "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
        "deployment_geometry_allowlist_sha256": next(iter(geometry_hashes)),
    }


def preprocess_source(sample: dict) -> torch.Tensor:
    bgr = cv2.imread(sample["source"], cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(sample["source"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = input_color.preprocess_rgb8_to_nchw(
        rgb,
        sample["model_input_width"],
        sample["model_input_height"],
        input_color.sdr_input_variant(),
    )
    return torch.from_numpy(image.copy())


def extract_statistics(samples: list[dict], depth_anything_root: Path,
                       depth_weights: Path, device_name: str) -> list[dict]:
    """Attach only the eight global depth/DPT statistics used by the policy head."""
    if POLICY_STAT_SIZE != len(FEATURE_NAMES):
        raise RuntimeError("policy statistic names are stale")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    depth_model = load_depth_anything_small(depth_anything_root, depth_weights)
    model = ArtisticPolicyModel(depth_model)
    model.freeze_base()
    model.eval().to(device)
    enriched = []
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            image = preprocess_source(sample)[None].to(device)
            with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda"):
                pooled = model.policy_features(image)
            values = pooled[0, -POLICY_STAT_SIZE:].float().cpu().numpy()
            if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
                raise RuntimeError(f"{sample['clip']}: invalid DA-V2 statistics")
            enriched.append({
                **sample,
                "source_sha256": sha256(Path(sample["source"])),
                "features": {
                    name: float(value)
                    for name, value in zip(FEATURE_NAMES, values)
                },
            })
            print(
                f"features {index + 1}/{len(samples)}: {sample['split']}/{sample['clip']}",
                flush=True,
            )
    return enriched


@dataclass
class TreeNode:
    sample_count: int
    safe_count: int
    feature: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def safe_probability(self) -> float:
        return self.safe_count / self.sample_count if self.sample_count else 0.0

    @property
    def leaf(self) -> bool:
        return self.feature is None


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    probability = float(np.mean(values))
    return 2.0 * probability * (1.0 - probability)


def fit_tree(features: np.ndarray, targets: np.ndarray, max_depth: int = 2,
             min_leaf: int = 8) -> TreeNode:
    """Fit a small deterministic CART classifier without an sklearn dependency."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.bool_)
    if features.ndim != 2 or targets.shape != (features.shape[0],):
        raise ValueError("tree features/targets have incompatible shapes")
    if features.shape[0] < 2 * min_leaf:
        raise ValueError("tree needs at least two minimum-size leaves")
    if not np.isfinite(features).all():
        raise ValueError("tree features must be finite")

    def build(indices: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode(
            sample_count=int(indices.size),
            safe_count=int(np.count_nonzero(targets[indices])),
        )
        if depth >= max_depth or node.safe_count in (0, node.sample_count):
            return node
        parent_impurity = _gini(targets[indices])
        best = None
        for feature_index in range(features.shape[1]):
            values = features[indices, feature_index]
            order = np.argsort(values, kind="stable")
            sorted_indices = indices[order]
            sorted_values = values[order]
            for split in range(min_leaf, indices.size - min_leaf + 1):
                if sorted_values[split - 1] == sorted_values[split]:
                    continue
                left_indices = sorted_indices[:split]
                right_indices = sorted_indices[split:]
                impurity = (
                    split * _gini(targets[left_indices]) +
                    (indices.size - split) * _gini(targets[right_indices])
                ) / indices.size
                gain = parent_impurity - impurity
                threshold = float(
                    (sorted_values[split - 1] + sorted_values[split]) * 0.5
                )
                candidate = (
                    gain, -feature_index, -threshold,
                    feature_index, threshold, left_indices, right_indices,
                )
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None or best[0] <= 1e-12:
            return node
        gain, _neg_feature, _neg_threshold, feature_index, threshold, left, right = best
        node.feature = int(feature_index)
        node.threshold = float(threshold)
        node.gain = float(gain)
        node.left = build(left, depth + 1)
        node.right = build(right, depth + 1)
        return node

    return build(np.arange(features.shape[0]), 0)


def predict_probabilities(tree: TreeNode, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    probabilities = []
    for row in features:
        node = tree
        while not node.leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        probabilities.append(node.safe_probability)
    return np.asarray(probabilities, dtype=np.float64)


def tree_payload(node: TreeNode):
    payload = {
        "sample_count": node.sample_count,
        "safe_count": node.safe_count,
        "safe_probability": node.safe_probability,
    }
    if not node.leaf:
        payload.update({
            "feature": FEATURE_NAMES[node.feature],
            "threshold": node.threshold,
            "gini_gain": node.gain,
            "left": tree_payload(node.left),
            "right": tree_payload(node.right),
        })
    return payload


def tree_rules(node: TreeNode, prefix: str = "") -> list[str]:
    if node.leaf:
        condition = prefix or "all shots"
        return [
            f"{condition} => training_safe_fraction={node.safe_probability:.3f} "
            f"({node.safe_count}/{node.sample_count})"
        ]
    feature = FEATURE_NAMES[node.feature]
    left_prefix = f"{prefix} and " if prefix else ""
    right_prefix = left_prefix
    return (
        tree_rules(node.left, f"{left_prefix}{feature} <= {node.threshold:.6g}") +
        tree_rules(node.right, f"{right_prefix}{feature} > {node.threshold:.6g}")
    )


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.bool_)
    predictions = np.asarray(predictions, dtype=np.bool_)
    tp = int(np.count_nonzero(targets & predictions))
    fp = int(np.count_nonzero(~targets & predictions))
    tn = int(np.count_nonzero(~targets & ~predictions))
    fn = int(np.count_nonzero(targets & ~predictions))

    def percentage(numerator, denominator):
        return 100.0 * numerator / denominator if denominator else None

    safe_count = tp + fn
    unsafe_count = tn + fp
    return {
        "shot_count": int(targets.size),
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "safe_recall_pct": percentage(tp, safe_count),
        "unsafe_recall_pct": percentage(tn, unsafe_count),
        "unsafe_exposure_pct": percentage(fp, unsafe_count),
        "safe_precision_pct": percentage(tp, tp + fp),
        "action_rate_pct": percentage(tp + fp, targets.size),
        "accuracy_pct": percentage(tp + tn, targets.size),
        "balanced_accuracy_pct": (
            0.5 * (percentage(tp, safe_count) + percentage(tn, unsafe_count))
            if safe_count and unsafe_count else None
        ),
    }


def captured_safe_headroom_pct(samples: list[dict], actions: np.ndarray):
    """Return captured labelled headroom for a binary TARGET_SCALE action.

    A shot labelled safe through 1.3 still captures only 0.1 when this binary
    controller selects 1.1. The denominator intentionally remains all known
    labelled headroom so the report also shows what this policy cannot exploit.
    """
    total = sum(
        max(float(sample["safe_scale_ceiling"]) - 1.0, 0.0)
        for sample in samples
    )
    captured = sum(
        max(min(TARGET_SCALE, float(sample["safe_scale_ceiling"])) - 1.0, 0.0)
        for sample, action in zip(samples, actions) if action
    )
    return 100.0 * captured / total if total > 0.0 else None


def feature_matrix(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([
        [sample["features"][name] for name in FEATURE_NAMES]
        for sample in samples
    ], dtype=np.float64)
    targets = np.asarray([sample["safe"] for sample in samples], dtype=np.bool_)
    return features, targets


def evaluate_thresholds(tree: TreeNode, samples: list[dict],
                        thresholds=DIAGNOSTIC_THRESHOLDS) -> tuple[dict, np.ndarray]:
    features, targets = feature_matrix(samples)
    probabilities = predict_probabilities(tree, features)
    results = {
        f"{threshold:g}": classification_metrics(
            targets, probabilities >= threshold
        )
        for threshold in thresholds
    }
    return results, probabilities


def leave_one_film_out(samples: list[dict], max_depth: int,
                       min_leaf: int, action_threshold: float) -> list[dict]:
    films = sorted({sample["film_id"] for sample in samples})
    rows = []
    for film in films:
        training = [sample for sample in samples if sample["film_id"] != film]
        validation = [sample for sample in samples if sample["film_id"] == film]
        if len(training) < 2 * min_leaf:
            rows.append({"film_id": film, "error": "insufficient training shots"})
            continue
        tree = fit_tree(*feature_matrix(training), max_depth=max_depth,
                        min_leaf=min_leaf)
        features, targets = feature_matrix(validation)
        probabilities = predict_probabilities(tree, features)
        rows.append({
            "film_id": film,
            "training_shots": len(training),
            "validation_shots": len(validation),
            "metrics": classification_metrics(
                targets, probabilities >= action_threshold
            ),
        })
    return rows


def _fmt(value, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.1f}{suffix}"


def write_html(path: Path, payload: dict):
    primary = payload["development"]["primary_metrics"]
    training_primary = payload["training"]["primary_metrics"]
    fixed = payload["development"]["baselines"]["fixed_1.1"]
    identity = payload["development"]["baselines"]["identity"]
    verdict = payload["decision"]["summary"]

    def score_card(name, metrics, css_class):
        recall = metrics["safe_recall_pct"] or 0.0
        exposure = metrics["unsafe_exposure_pct"] or 0.0
        return f"""
        <div class="card {css_class}">
          <h3>{html.escape(name)}</h3>
          <div class="number">{_fmt(recall, '%')}</div><div>safe-shot recall</div>
          <div class="bar"><i style="width:{recall:.2f}%"></i></div>
          <div class="number danger">{_fmt(exposure, '%')}</div><div>unsafe exposure</div>
          <div class="bar danger"><i style="width:{exposure:.2f}%"></i></div>
        </div>"""

    shot_rows = []
    threshold = payload["controller"]["action_threshold"]
    for sample in payload["development"]["shots"]:
        action = sample["safe_probability"] >= threshold
        outcome = "TP" if action and sample["safe"] else (
            "FP" if action else ("TN" if not sample["safe"] else "FN")
        )
        shot_rows.append(
            "<tr>"
            f"<td>{html.escape(sample['clip'])}</td>"
            f"<td>{sample['safe_scale_ceiling']:.2f}</td>"
            f"<td>{sample['safe_probability']:.3f}</td>"
            f"<td>{'1.1' if action else '1.0'}</td>"
            f"<td class=\"{outcome.lower()}\">{outcome}</td>"
            "</tr>"
        )
    rules = "".join(f"<li><code>{html.escape(rule)}</code></li>"
                    for rule in payload["controller"]["rules"])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Simple artistic controller</title>
<style>
body{{font:15px system-ui;margin:28px;max-width:1200px;color:#20242a;background:#f5f7fa}}
h1,h2,h3{{margin:.3em 0}} .conclusion{{padding:18px;border-left:6px solid #3867d6;background:white}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:14px;margin:18px 0}}
.card{{background:white;padding:16px;border-radius:10px;box-shadow:0 1px 4px #ccd2dc}}
.number{{font-size:28px;font-weight:700;margin-top:10px}} .danger{{color:#b42318}}
.bar{{height:10px;background:#e7ebf1;border-radius:5px;overflow:hidden;margin:5px 0 12px}}
.bar i{{display:block;height:100%;background:#277b52}} .bar.danger i{{background:#d92d20}}
details,section{{background:white;padding:16px;margin:16px 0;border-radius:10px}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}
code{{white-space:normal}} .tp,.tn{{color:#277b52;font-weight:700}} .fp,.fn{{color:#b42318;font-weight:700}}
small{{color:#667085}}
</style></head><body>
<h1>Simple DA-V2 statistics controller</h1>
<div class="conclusion"><h2>Conclusion</h2><p>{html.escape(verdict)}</p>
<p><strong>Fitting-shot safety check:</strong>
{training_primary['false_positive']} false accept(s),
{_fmt(training_primary['unsafe_exposure_pct'], '%')} unsafe exposure.</p>
<small>This is a held-out development screen, not production approval.</small></div>
<div class="cards">
{score_card('Development: conservative tree', primary, 'tree')}
{score_card('Development: always 1.1', fixed, 'fixed')}
{score_card('Development: always identity', identity, 'identity')}
</div>
<section><h2>Method</h2><p>A depth-{payload['controller']['max_depth']} tree was fit on
{payload['training']['shot_count']} training shots using only the first monocular frame and eight
scale-invariant DA-V2 depth/DPT statistics. The configured empirical leaf-fraction action threshold
is {threshold:g}; it is not a calibrated probability.</p></section>
<details open><summary><strong>Tree rules</strong></summary><ul>{rules}</ul></details>
<section><h2>Held-out shots</h2><table><thead><tr><th>Shot</th><th>Safe ceiling</th>
<th>Training leaf safe fraction</th><th>Selected scale</th><th>Outcome</th></tr></thead>
<tbody>{''.join(shot_rows)}</tbody></table></section>
<details open><summary><strong>Evidence limitations</strong></summary><ul>
{''.join(f'<li>{html.escape(item)}</li>' for item in payload['limitations'])}</ul></details>
<details><summary><strong>Metric definitions and provenance</strong></summary>
<p>Safe recall is the percentage of truly scale-1.1-safe shots where the controller uses 1.1.
Unsafe exposure is the percentage of unsafe shots where it incorrectly uses 1.1. False positives
are safety failures; false negatives only leave pop unused.</p>
<pre>{html.escape(json.dumps(payload['provenance'], indent=2))}</pre></details>
</body></html>"""
    path.write_text(document, encoding="utf-8", newline="\n")


def run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"output must be empty (or use --overwrite): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    training, training_provenance = load_training_shots(
        args.training_control, args.training_candidate, args.thresholds
    )
    development, development_provenance = load_development_shots(
        args.development_labels,
        training_provenance["metric_sha256"],
        training_provenance["policy_warp_source_sha256"],
    )
    all_samples = extract_statistics(
        training + development,
        Path(args.depth_anything_root).resolve(),
        Path(args.depth_weights).resolve(),
        args.device,
    )
    training = [sample for sample in all_samples if sample["split"] == "training"]
    development = [sample for sample in all_samples if sample["split"] == "development"]
    tree = fit_tree(*feature_matrix(training), max_depth=args.max_depth,
                    min_leaf=args.min_leaf)
    screen_thresholds = tuple(sorted(set(
        DIAGNOSTIC_THRESHOLDS + (args.action_threshold,)
    )))
    training_thresholds, training_probabilities = evaluate_thresholds(
        tree, training, screen_thresholds
    )
    development_thresholds, development_probabilities = evaluate_thresholds(
        tree, development, screen_thresholds
    )
    for sample, probability in zip(training, training_probabilities):
        sample["safe_probability"] = float(probability)
    for sample, probability in zip(development, development_probabilities):
        sample["safe_probability"] = float(probability)

    _development_features, development_targets = feature_matrix(development)
    primary_predictions = development_probabilities >= args.action_threshold
    primary_metrics = classification_metrics(development_targets, primary_predictions)
    threshold_key = f"{args.action_threshold:g}"
    training_primary_metrics = training_thresholds[threshold_key]
    development_brier = float(np.mean(
        (development_probabilities - development_targets.astype(np.float64)) ** 2
    ))
    train_prior = float(np.mean(feature_matrix(training)[1]))
    train_prior_brier = float(np.mean(
        (train_prior - development_targets.astype(np.float64)) ** 2
    ))
    identity_metrics = classification_metrics(
        development_targets, np.zeros_like(development_targets)
    )
    fixed_metrics = classification_metrics(
        development_targets, np.ones_like(development_targets)
    )
    if training_primary_metrics["false_positive"]:
        decision = (
            "The simple controller is already unsafe on its fitting shots: it applies scale "
            f"1.1 to {training_primary_metrics['false_positive']} known-unsafe shot(s). Its "
            "uncalibrated global-statistics leaves do not replace the learned semantic head."
        )
        status = "unsafe_on_training_shots"
    elif primary_metrics["false_positive"]:
        decision = (
            "The simple controller is not safe enough: it applies scale 1.1 to "
            f"{primary_metrics['false_positive']} held-out unsafe shot(s). Keep the learned "
            "semantic head or develop a stronger hybrid risk predictor."
        )
        status = "unsafe_false_positive"
    elif primary_metrics["true_positive"] == 0:
        decision = (
            "The conservative tree is safe only by behaving like identity and captures no "
            "held-out pop headroom. These statistics alone do not replace the learned head."
        )
        status = "no_gain_over_identity"
    elif primary_metrics["safe_recall_pct"] < MIN_USEFUL_SAFE_RECALL_PCT:
        decision = (
            "The conservative tree avoids held-out unsafe shots but captures only "
            f"{primary_metrics['true_positive']} of {primary_metrics['safe_count']} safe "
            "shots. Its global statistics do not demonstrate useful cross-film gain over "
            "identity, so they do not replace the learned semantic head."
        )
        status = "safe_but_insufficient_cross_film_gain"
    else:
        decision = (
            "The simple controller is promising: it exposes no held-out unsafe shot and "
            f"captures {primary_metrics['true_positive']} of "
            f"{primary_metrics['safe_count']} safe shots. More independent films are still "
            "required before replacing the learned head."
        )
        status = "promising_not_approved"

    payload = {
        "schema": SCHEMA,
        "experiment": "first-frame-dav2-statistics-safe-at-1.1",
        "target_scale": TARGET_SCALE,
        "feature_contract": SIMPLE_FEATURE_CONTRACT,
        "feature_names": list(FEATURE_NAMES),
        "controller": {
            "type": "deterministic-cart",
            "max_depth": args.max_depth,
            "min_leaf": args.min_leaf,
            "action_threshold": args.action_threshold,
            "leaf_score_semantics": (
                "uncalibrated safe fraction among fitting shots in the reached leaf"
            ),
            "tree": tree_payload(tree),
            "rules": tree_rules(tree),
        },
        "training": {
            "shot_count": len(training),
            "safe_count": sum(sample["safe"] for sample in training),
            "unsafe_count": sum(not sample["safe"] for sample in training),
            "threshold_screens": training_thresholds,
            "primary_metrics": training_primary_metrics,
            "leave_one_film_out": leave_one_film_out(
                training, args.max_depth, args.min_leaf, args.action_threshold
            ),
            "shots": training,
        },
        "development": {
            "shot_count": len(development),
            "safe_count": sum(sample["safe"] for sample in development),
            "unsafe_count": sum(not sample["safe"] for sample in development),
            "threshold_screens": development_thresholds,
            "primary_metrics": primary_metrics,
            "brier": development_brier,
            "train_prior_brier": train_prior_brier,
            "captured_safe_headroom_pct": captured_safe_headroom_pct(
                development, primary_predictions
            ),
            "baselines": {
                "identity": identity_metrics,
                "fixed_1.1": fixed_metrics,
            },
            "shots": development,
        },
        "decision": {
            "status": status,
            "production_approved": False,
            "summary": decision,
        },
        "limitations": [
            "training labels cover only the 1280x720 deployment geometry",
            "training positives are right-censored at safe ceiling >= 1.1",
            "development contains ten shots from one film and is not sealed test",
            "training is film-imbalanced and dominated by Big Buck Bunny",
            "tree leaf safe fractions are empirical and are not calibrated probabilities",
            "the eight DPT/depth statistics are internal to the model graph and are not a current C++ output",
        ],
        "provenance": {
            "training": training_provenance,
            "development": development_provenance,
            "depth_anything_root": str(Path(args.depth_anything_root).resolve()),
            "depth_weights": {
                "path": str(Path(args.depth_weights).resolve()),
                "sha256": sha256(Path(args.depth_weights).resolve()),
            },
            "source_policy_feature_contract": POLICY_FEATURE_CONTRACT,
            "git": git_state(),
            "code": {
                "controller": sha256(Path(__file__).resolve()),
                "policy_model": sha256(
                    Path(__file__).resolve().with_name("artistic_policy_model.py")
                ),
                "selector": sha256(
                    Path(__file__).resolve().with_name("select_render_feasible_labels.py")
                ),
            },
            "depth_anything_source_sha256": source_tree_sha256(
                Path(args.depth_anything_root).resolve() / "depth_anything_v2"
            ),
            "runtime": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": args.device,
                "device_name": (
                    torch.cuda.get_device_name(torch.device(args.device))
                    if torch.device(args.device).type == "cuda" else "cpu"
                ),
                "feature_autocast": "float16" if torch.device(args.device).type == "cuda" else "float32",
            },
            "feature_leakage_guard": (
                "first monocular RGB frame only; no candidate SBS, target metric, later frame, "
                "right eye, or target disparity is read by feature extraction"
            ),
        },
    }
    evaluation_path = output / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    write_html(output / "report.html", payload)
    print(json.dumps({
        "decision": payload["decision"],
        "development": primary_metrics,
        "evaluation": str(evaluation_path),
        "report": str(output / "report.html"),
    }, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-control", required=True, type=Path)
    parser.add_argument("--training-candidate", required=True, type=Path)
    parser.add_argument("--development-labels", required=True, type=Path)
    parser.add_argument(
        "--thresholds", type=Path,
        default=Path(__file__).resolve().parents[1] / "sbsbench" / "thresholds.json",
    )
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--depth-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-leaf", type=int, default=12)
    parser.add_argument("--action-threshold", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_depth < 1 or args.max_depth > 4:
        parser.error("--max-depth must be between 1 and 4")
    if args.min_leaf < 2:
        parser.error("--min-leaf must be at least 2")
    if not 0.5 <= args.action_threshold <= 1.0:
        parser.error("--action-threshold must be between 0.5 and 1.0")
    run(args)


if __name__ == "__main__":
    main()
