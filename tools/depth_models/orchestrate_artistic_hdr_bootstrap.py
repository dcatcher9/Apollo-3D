#!/usr/bin/env python3
"""Run the bounded public-mono SDR/HDR artistic-policy bootstrap end to end.

The orchestration consumes only the four derived training/development
productions published by ``prepare_artistic_bootstrap_subset.py``.  It passes
the frozen active-split manifest to the trainer and development evaluator, but
never constructs, opens, or hashes a sealed-test label or frame path.

The expensive render phase is identity-first and resumable.  Non-identity
render directories are compacted to their authenticated ``results.json`` as
soon as scoring finishes; identity runs retain only the contracts and exact
clamped/unclamped disparity rasters required by selection and training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import artistic_geometry_contract as geometry_contract
import artistic_policy_evaluation_contract as evaluation_contract
import depth_input_color as input_color
import generate_artistic_depth_run as depth_run
import merge_artistic_geometry_labels as label_merge
import select_render_feasible_labels as selector


SCHEMA = 3
CONTRACT = "apollo-public-mono-sdr-hdr-bootstrap-orchestration-v2"
BOOTSTRAP_CONTRACT = "apollo-public-mono-hdr-bootstrap-subset-v1"
WHITE_LEVELS = (1000, 2500, 6000)
INPUT_CONDITIONS = (("sdr", None),) + tuple(
    (f"w{raw_white}", raw_white) for raw_white in WHITE_LEVELS
)
DEPTH_MODEL = "depth_anything_v2_fp16"
SCALES = (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
GEOMETRIES = (
    ("uncapped-1280x720", 1280, 720),
    ("packed-width-capped-960x540", 960, 540),
)
EXPECTED_DATASETS = {
    ("reds", "training"),
    ("reds", "development"),
    ("spring", "training"),
    ("spring", "development"),
}
PHASES = (
    "depth", "sources", "identity", "render", "select", "merge",
    "train", "evaluate",
)
MAX_RENDER_WORKERS = 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read JSON: {path}") from error


def canonical_bytes(value):
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_relative_to(path, root):
    try:
        Path(path).resolve(strict=False).relative_to(
            Path(root).resolve(strict=False)
        )
        return True
    except ValueError:
        return False


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def slug(value):
    result = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    if not result:
        raise RuntimeError("empty orchestration slug")
    return result


def scale_slug(scale):
    return f"s{int(round(float(scale) * 100)):03d}"


@dataclass(frozen=True)
class Dataset:
    source: str
    split: str
    production_id: str
    root: Path
    clips: tuple[str, ...]
    context_frames: int
    label_frames: int
    dataset_manifest_sha256: str
    clip_hash_manifest_sha256: str

    @property
    def key(self):
        return f"{self.source}-{self.split}"


@dataclass(frozen=True)
class Geometry:
    name: str
    eye_width: int
    eye_height: int

    @property
    def key(self):
        return f"g{self.eye_width}x{self.eye_height}"


@dataclass(frozen=True)
class Step:
    key: str
    phase: str
    kind: str
    command: tuple[str, ...]
    output: Path
    metadata: dict

    def as_dict(self):
        return {
            "key": self.key,
            "phase": self.phase,
            "kind": self.kind,
            "output": str(self.output),
            "command": subprocess.list2cmdline(list(self.command)),
            "argv": list(self.command),
            "metadata": self.metadata,
        }


@dataclass
class Plan:
    repo: Path
    workspace: Path
    build_dir: Path
    conf: Path
    python: Path
    depth_anything_root: Path
    depth_weights: Path
    active_split: Path
    bootstrap_manifest: Path
    datasets: tuple[Dataset, ...]
    geometries: tuple[Geometry, ...]
    geometry_manifest: dict
    input_variant_manifest: dict
    geometry_manifest_path: Path
    input_variant_manifest_path: Path
    steps: tuple[Step, ...]
    estimates: dict

    def as_dict(self):
        counts = {}
        for step in self.steps:
            counts[step.kind] = counts.get(step.kind, 0) + 1
        return {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "sealed_test_policy": (
                "active-split identities/manifests only; never open or hash "
                "sealed-test labels or frames"
            ),
            "post_bootstrap_required": (
                "schema-28 makes older committed baselines stale; run fresh "
                "core and extended production baseline evaluations/reports after "
                "the bootstrap, never rescore old HDR-preview artifacts"
            ),
            "repo": str(self.repo),
            "workspace": str(self.workspace),
            "build_dir": str(self.build_dir),
            "conf": str(self.conf),
            "python": str(self.python),
            "depth_anything_root": str(self.depth_anything_root),
            "depth_weights": str(self.depth_weights),
            "bootstrap_manifest": str(self.bootstrap_manifest),
            "bootstrap_manifest_sha256": sha256(self.bootstrap_manifest),
            "active_split": str(self.active_split),
            "active_split_sha256": sha256(self.active_split),
            "deployment_geometry_manifest": self.geometry_manifest,
            "deployment_geometry_manifest_identity":
                geometry_contract.allowlist_sha256(self.geometry_manifest),
            "input_variant_manifest": self.input_variant_manifest,
            "input_variant_manifest_identity":
                label_merge.input_variant_manifest_sha256(
                    self.input_variant_manifest
                ),
            "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
            "datasets": [
                {
                    "source": item.source,
                    "split": item.split,
                    "production_id": item.production_id,
                    "root": str(item.root),
                    "clips": list(item.clips),
                    "context_frames": item.context_frames,
                    "label_frames": item.label_frames,
                    "dataset_manifest_sha256":
                        item.dataset_manifest_sha256,
                    "clip_hash_manifest_sha256":
                        item.clip_hash_manifest_sha256,
                }
                for item in self.datasets
            ],
            "counts": counts,
            "subprocess_steps": len(self.steps),
            "estimates": self.estimates,
            "steps": [step.as_dict() for step in self.steps],
        }


def _validate_bootstrap_manifest(workspace, datasets_root=None):
    datasets_root = (
        Path(datasets_root).resolve()
        if datasets_root is not None else (workspace / "datasets").resolve()
    )
    path = datasets_root / "bootstrap_manifest.json"
    payload = load_json(path)
    if (payload.get("schema") != 1 or
            payload.get("preparation_contract") != BOOTSTRAP_CONTRACT):
        raise RuntimeError("unsupported bootstrap manifest")
    normalization = payload.get("normalization", {})
    if (normalization.get("target_width"),
            normalization.get("target_height")) != (1280, 720):
        raise RuntimeError("bootstrap source geometry must be exactly 1280x720")
    rows = payload.get("datasets")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_DATASETS):
        raise RuntimeError("bootstrap must contain exactly four working datasets")
    observed = {(row.get("source"), row.get("split")) for row in rows}
    if observed != EXPECTED_DATASETS:
        raise RuntimeError(
            "bootstrap working datasets differ from REDS/Spring train/development"
        )
    datasets = []
    for row in sorted(rows, key=lambda value: (
            value["split"] != "training", value["source"])):
        if row["split"] not in {"training", "development"}:
            raise RuntimeError("sealed-test dataset entered the working set")
        root = Path(row["output_root"]).resolve()
        if not is_relative_to(root, datasets_root):
            raise RuntimeError("working dataset escapes the bootstrap workspace")
        manifest = root / "dataset_manifest.json"
        if (not manifest.is_file() or
                sha256(manifest) != row.get("dataset_manifest_sha256")):
            raise RuntimeError(f"working dataset manifest is stale: {manifest}")
        clip_hash_manifest = root / "clip_hash_manifest.json"
        if (not clip_hash_manifest.is_file() or
                sha256(clip_hash_manifest) !=
                row.get("clip_hash_manifest_sha256")):
            raise RuntimeError(
                f"working clip hash manifest is stale: {clip_hash_manifest}"
            )
        clips = row.get("clips")
        if (not isinstance(clips, list) or not clips or
                any(not isinstance(value, str) or not value for value in clips)):
            raise RuntimeError("bootstrap dataset has invalid clip identities")
        datasets.append(Dataset(
            source=row["source"], split=row["split"],
            production_id=row["production_id"], root=root,
            clips=tuple(clips), context_frames=int(row["context_frame_count"]),
            label_frames=int(row["label_frame_count"]),
            dataset_manifest_sha256=row["dataset_manifest_sha256"],
            clip_hash_manifest_sha256=row["clip_hash_manifest_sha256"],
        ))
    training = payload.get("training_contract", {})
    active_split = Path(training.get("active_split", "")).resolve()
    if (not active_split.is_file() or
            sha256(active_split) != training.get("active_split_sha256")):
        raise RuntimeError("bootstrap active-split identity is stale")
    active = load_json(active_split)
    split_productions = active.get("split_productions", {})
    for split in ("training", "development"):
        expected = {
            item.production_id for item in datasets if item.split == split
        }
        if set(split_productions.get(split, ())) != expected:
            raise RuntimeError(f"active split disagrees with bootstrap {split}")
    sealed = split_productions.get("test")
    if not isinstance(sealed, list) or len(sealed) < 2:
        raise RuntimeError("active split lacks independent sealed tests")
    working = {item.production_id for item in datasets}
    if working.intersection(sealed):
        raise RuntimeError("sealed-test production entered the working set")
    return path, tuple(datasets), active_split


def _geometry(source_width, source_height, name, eye_width, eye_height):
    del source_width, source_height
    return Geometry(
        name=name, eye_width=eye_width, eye_height=eye_height,
    )


def _geometry_value(source_width, source_height, geometry, color_mode):
    scale_x, scale_y = geometry_contract.source_content_scales(
        source_width, source_height, geometry.eye_width, geometry.eye_height
    )
    row = {
        "source_width": source_width,
        "source_height": source_height,
        "eye_width": geometry.eye_width,
        "eye_height": geometry.eye_height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
        "disparity_raster_width": geometry.eye_width,
        "disparity_raster_height": geometry.eye_height,
        "color_mode": color_mode,
    }
    return geometry_contract.geometry_tuple(row, color_mode)


def _input_variant(raw_white):
    if raw_white is None:
        return input_color.sdr_input_variant()
    return input_color.windows_hdr_input_variant(raw_white)


def _condition_color_mode(raw_white):
    return _input_variant(raw_white)["color_mode"]


def _condition_preview_encoding(raw_white):
    return selector.sbs_contract.expected_metric_preview_encoding(
        _condition_color_mode(raw_white)
    )


def _render_label(prefix, dataset, condition, geometry, scale):
    return "-".join((
        slug(prefix), slug(dataset.source), slug(dataset.split),
        condition, f"g{geometry.eye_width}", scale_slug(scale),
    ))


def _source_paths(workspace, dataset, condition):
    stem = f"{dataset.key}-{condition}"
    return (
        workspace / "depth" / stem,
        workspace / "sources" / stem,
    )


def _selection_path(workspace, dataset, condition, geometry):
    return (
        workspace / "selected" / dataset.key / condition /
        geometry.key
    )


def _build_estimates(datasets, geometries):
    context = sum(item.context_frames for item in datasets)
    labels = sum(item.label_frames for item in datasets)
    training_labels = sum(
        item.label_frames for item in datasets if item.split == "training"
    )
    development_labels = sum(
        item.label_frames for item in datasets if item.split == "development"
    )
    conditions = len(INPUT_CONDITIONS)
    render_runs = len(datasets) * conditions * len(geometries) * len(SCALES)
    selected_outputs = labels * 2 * conditions * len(geometries) * len(SCALES)
    f32_bytes = 0
    identity_f32_bytes = 0
    for geometry in geometries:
        bytes_per_output = 2 * geometry.eye_width * geometry.eye_height * 4
        geometry_outputs = labels * 2 * conditions * len(SCALES)
        f32_bytes += geometry_outputs * bytes_per_output
        identity_f32_bytes += (
            labels * 2 * conditions * bytes_per_output
        )
    gib = 1024.0 ** 3
    return {
        "input_conditions": [key for key, _raw in INPUT_CONDITIONS],
        "depth_generation_runs": len(datasets) * conditions,
        "depth_source_frames_scanned": context * conditions,
        "depth_sparse_output_upper_bound": labels * 2 * conditions,
        "training_rgb_labels": training_labels,
        "development_rgb_labels": development_labels,
        "training_policy_samples": training_labels * conditions,
        "development_policy_samples": development_labels * conditions,
        "safety_geometries_per_policy_sample": len(geometries),
        "render_runs": render_runs,
        "identity_render_runs": (
            len(datasets) * conditions * len(geometries)
        ),
        "candidate_render_runs": (
            render_runs - len(datasets) * conditions * len(geometries)
        ),
        "full_cadence_frame_visits": (
            context * conditions * len(geometries) * len(SCALES)
        ),
        "sparse_output_frame_artifact_sets": selected_outputs,
        "exact_two_disparity_rasters_written_gib": round(f32_bytes / gib, 2),
        "estimated_total_render_writes_gib": [
            round(1.5 * f32_bytes / gib, 1),
            round(2.5 * f32_bytes / gib, 1),
        ],
        "retained_identity_disparity_gib_after_compaction": round(
            identity_f32_bytes / gib, 2
        ),
        "compaction_note": (
            "non-identity runs retain results.json only; identity runs retain "
            "results, clip contracts, and exact disparity rasters"
        ),
    }


def build_plan(args):
    repo = Path(__file__).resolve().parents[2]
    workspace = args.workspace.resolve()
    build_dir = args.build_dir.resolve()
    executable = build_dir / "sunshine.exe"
    conf = args.conf.resolve()
    python = args.python.resolve()
    depth_anything_root = args.depth_anything_root.resolve()
    depth_weights = args.depth_weights.resolve()
    bootstrap_manifest, datasets, active_split = (
        _validate_bootstrap_manifest(workspace, args.datasets_root)
    )
    if not conf.is_file():
        raise RuntimeError(f"missing evaluator config: {conf}")
    if not executable.is_file():
        raise RuntimeError(f"missing benchmark executable: {executable}")
    if not python.is_file():
        raise RuntimeError(f"missing Python interpreter: {python}")
    if not depth_anything_root.is_dir():
        raise RuntimeError(
            f"missing Depth Anything V2 source: {depth_anything_root}"
        )
    if not depth_weights.is_file():
        raise RuntimeError(f"missing DA-V2 weights: {depth_weights}")
    geometries = tuple(
        _geometry(1280, 720, name, width, height)
        for name, width, height in GEOMETRIES
    )
    geometry_manifest = geometry_contract.build_allowlist([
        _geometry_value(1280, 720, geometry, color_mode)
        for geometry in geometries
        for color_mode in (
            input_color.COLOR_MODE_SDR, input_color.COLOR_MODE_HDR,
        )
    ])
    # This is the global policy allow-list.  The SDR-origin orchestration still
    # renders only its four applicable conditions; native PQ is generated by
    # the separately authenticated CHUG branch.
    variants = label_merge.policy_input_variants()
    input_manifest = label_merge.build_input_variant_manifest(variants)
    manifests_root = workspace / "manifests"
    geometry_manifest_path = manifests_root / "deployment-geometries.json"
    input_manifest_path = manifests_root / "input-variants.json"
    eval_root = build_dir / "sbs_eval"
    executable_sha = sha256(executable)
    conf_sha = depth_run.eval_semantic_file_hash(conf)
    model_asset_identity = depth_run.selected_depth_model_identity(
        executable, DEPTH_MODEL
    )
    model_asset_identity_sha = depth_run.clip_hashes.canonical_json_sha256(
        model_asset_identity
    )
    steps = []
    source_outputs = {}
    render_results = {}

    for dataset in datasets:
        for condition, raw_white in INPUT_CONDITIONS:
            variant = _input_variant(raw_white)
            color_mode = variant["color_mode"]
            preview_encoding = _condition_preview_encoding(raw_white)
            depth_root, source_root = _source_paths(
                workspace, dataset, condition
            )
            source_outputs[(dataset.key, condition)] = source_root
            depth_command = [
                str(python),
                str(repo / "tools" / "depth_models" /
                    "generate_artistic_depth_run.py"),
                "--suite", str(dataset.root),
                "--output", str(depth_root),
                "--build-dir", str(build_dir),
                "--conf", str(conf),
                "--model", DEPTH_MODEL,
            ]
            # The four conditions use the same frozen RGB files.  Authenticate
            # their content fully once on native SDR, then bind the other three
            # publications to the same manifest/stat identities.
            verification = "full" if condition == "sdr" else "stat"
            if verification == "full":
                depth_command.append("--verify-clip-hashes")
            if raw_white is not None:
                depth_command.extend((
                    "--simulate-hdr",
                    "--sdr-white-level-raw", str(raw_white),
                ))
            steps.append(Step(
                key=f"depth-{dataset.key}-{condition}",
                phase="depth", kind="depth",
                command=tuple(depth_command),
                output=depth_root,
                metadata={
                    "dataset": dataset.key,
                    "condition": condition,
                    "raw_white": raw_white,
                    "color_mode": color_mode,
                    "clips": list(dataset.clips),
                    "dataset_root": str(dataset.root),
                    "dataset_manifest_sha256":
                        dataset.dataset_manifest_sha256,
                    "clip_hash_manifest_sha256":
                        dataset.clip_hash_manifest_sha256,
                    "clip_hash_verification": verification,
                    "executable": str(executable),
                    "executable_sha256": executable_sha,
                    "conf": str(conf),
                    "conf_sha256": conf_sha,
                    "model": DEPTH_MODEL,
                    "model_asset_identity": model_asset_identity,
                    "model_asset_identity_sha256": model_asset_identity_sha,
                    "input_variant": variant,
                    "metric_preview_encoding": preview_encoding,
                },
            ))
            steps.append(Step(
                key=f"source-{dataset.key}-{condition}",
                phase="sources", kind="source",
                command=(
                    str(python),
                    str(repo / "tools" / "depth_models" /
                        "prepare_artistic_source_rows.py"),
                    "--run", str(depth_root),
                    "--clips", str(dataset.root),
                    "--output", str(source_root),
                ),
                output=source_root,
                metadata={
                    "dataset": dataset.key,
                    "condition": condition,
                    "raw_white": raw_white,
                    "color_mode": color_mode,
                    "input_variant": variant,
                    "metric_preview_encoding": preview_encoding,
                    "expected_labels": dataset.label_frames,
                    "dataset_root": str(dataset.root),
                    "depth_run": str(depth_root),
                    "dataset_manifest_sha256":
                        dataset.dataset_manifest_sha256,
                },
            ))

    def append_render_steps(scales, phase):
        for dataset in datasets:
            for condition, raw_white in INPUT_CONDITIONS:
                variant = _input_variant(raw_white)
                color_mode = variant["color_mode"]
                preview_encoding = _condition_preview_encoding(raw_white)
                for geometry in geometries:
                    for scale in scales:
                        label = _render_label(
                            args.run_prefix, dataset, condition,
                            geometry, scale,
                        )
                        output = eval_root / label
                        command = [
                            str(python),
                            str(repo / "tools" / "sbsbench" / "run_eval.py"),
                            "--build-dir", str(build_dir),
                            "--conf", str(conf),
                            "--clips-root", str(dataset.root),
                            "--label", label,
                            "--score-workers", str(args.score_workers),
                            "--comparison-only",
                            "--extra",
                            "--eye-w", str(geometry.eye_width),
                            "--eye-h", str(geometry.eye_height),
                            "--no-artistic-policy",
                            "--artistic-scale-override", f"{scale:.1f}",
                            "--output-label-frames",
                        ]
                        if raw_white is not None:
                            insert_at = command.index("--no-artistic-policy")
                            command[insert_at:insert_at] = [
                                "--simulate-hdr",
                                "--sdr-white-level-raw", str(raw_white),
                            ]
                        render_results[(
                            dataset.key, condition, geometry.key, scale,
                        )] = output / "results.json"
                        steps.append(Step(
                            key=(
                                f"render-{dataset.key}-{condition}-"
                                f"{geometry.key}-{scale_slug(scale)}"
                            ),
                            phase=phase, kind="render",
                            command=tuple(command),
                            output=output,
                            metadata={
                                "dataset": dataset.key,
                                "clips": list(dataset.clips),
                                "clips_root": str(dataset.root),
                                "clip_hash_manifest_sha256":
                                    dataset.clip_hash_manifest_sha256,
                                "condition": condition,
                                "raw_white": raw_white,
                                "color_mode": color_mode,
                                "input_variant": variant,
                                "metric_preview_encoding": preview_encoding,
                                "scale": scale,
                                "eye_width": geometry.eye_width,
                                "eye_height": geometry.eye_height,
                                "identity": math.isclose(scale, 1.0),
                            },
                        ))

    append_render_steps((1.0,), "identity")
    append_render_steps(
        tuple(scale for scale in SCALES if not math.isclose(scale, 1.0)),
        "render",
    )

    selected = {}
    for dataset in datasets:
        for condition, raw_white in INPUT_CONDITIONS:
            for geometry in geometries:
                output = _selection_path(
                    workspace, dataset, condition, geometry
                )
                source = source_outputs[(dataset.key, condition)]
                identity = render_results[(
                    dataset.key, condition, geometry.key, 1.0,
                )]
                command = [
                    str(python),
                    str(repo / "tools" / "depth_models" /
                        "select_render_feasible_labels.py"),
                    "--source-labels", str(source / "labels.jsonl"),
                    "--control", str(identity),
                ]
                for scale in SCALES:
                    command.extend((
                        "--candidate",
                        f"{scale:.1f}={render_results[(dataset.key, condition, geometry.key, scale)]}",
                    ))
                command.extend(("--output", str(output)))
                selected[(dataset.key, condition, geometry.key)] = output
                steps.append(Step(
                    key=(
                        f"select-{dataset.key}-{condition}-{geometry.key}"
                    ),
                    phase="select", kind="select",
                    command=tuple(command), output=output,
                    metadata={
                        "dataset": dataset.key,
                        "condition": condition,
                        "raw_white": raw_white,
                        "color_mode": _condition_color_mode(raw_white),
                        "geometry": geometry.key,
                        "expected_labels": dataset.label_frames,
                    },
                ))

    merged = {}
    for dataset in datasets:
        output = workspace / "merged" / dataset.key
        command = [
            str(python),
            str(repo / "tools" / "depth_models" /
                "merge_artistic_geometry_labels.py"),
        ]
        for condition, _raw_white in INPUT_CONDITIONS:
            for geometry in geometries:
                command.extend((
                    "--geometry-labels",
                    str(selected[(
                        dataset.key, condition, geometry.key,
                    )] / "labels.jsonl"),
                ))
        command.extend((
            "--deployment-geometry-manifest", str(geometry_manifest_path),
            "--input-variant-manifest", str(input_manifest_path),
            "--output", str(output),
        ))
        merged[dataset.key] = output
        steps.append(Step(
            key=f"merge-{dataset.key}", phase="merge", kind="merge",
            command=tuple(command), output=output,
            metadata={
                "dataset": dataset.key,
                "split": dataset.split,
                "expected_labels": dataset.label_frames,
                "expected_policy_samples": (
                    dataset.label_frames * len(INPUT_CONDITIONS)
                ),
                "condition_target_contract":
                    label_merge.CONDITION_TARGET_CONTRACT,
            },
        ))

    training = workspace / "training" / f"seed-{args.seed}"
    train_command = [
        str(python),
        str(repo / "tools" / "depth_models" / "train_artistic_policy.py"),
        "--labels",
    ]
    train_command.extend(
        str(merged[item.key] / "labels.jsonl") for item in datasets
    )
    train_command.extend((
        "--split-manifest", str(active_split),
        "--depth-anything-root", str(depth_anything_root),
        "--depth-weights", str(depth_weights),
        "--output", str(training),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
    ))
    steps.append(Step(
        key=f"train-seed-{args.seed}", phase="train", kind="train",
        command=tuple(train_command), output=training,
        metadata={
            "seed": args.seed, "epochs": args.epochs,
            "sealed_test_labels": "not supplied",
            "expected_training_rgb_labels": sum(
                item.label_frames for item in datasets
                if item.split == "training"
            ),
            "expected_development_rgb_labels": sum(
                item.label_frames for item in datasets
                if item.split == "development"
            ),
            "expected_training_policy_samples": sum(
                item.label_frames * len(INPUT_CONDITIONS)
                for item in datasets if item.split == "training"
            ),
            "expected_development_policy_samples": sum(
                item.label_frames * len(INPUT_CONDITIONS)
                for item in datasets if item.split == "development"
            ),
            "active_split_sha256": sha256(active_split),
            "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
        },
    ))

    evaluation = workspace / "evaluation" / f"development-seed-{args.seed}"
    eval_command = [
        str(python),
        str(repo / "tools" / "depth_models" / "evaluate_artistic_policy.py"),
        "--labels",
    ]
    development = [item for item in datasets if item.split == "development"]
    eval_command.extend(
        str(merged[item.key] / "labels.jsonl") for item in development
    )
    eval_command.extend((
        "--split-manifest", str(active_split),
        "--depth-anything-root", str(depth_anything_root),
        "--depth-weights", str(depth_weights),
        "--policy", str(training / "artistic_policy_best.pt"),
        "--output", str(evaluation),
        "--split", "development",
    ))
    steps.append(Step(
        key=f"evaluate-development-seed-{args.seed}",
        phase="evaluate", kind="evaluate", command=tuple(eval_command),
        output=evaluation,
        metadata={
            "split": "development",
            "sealed_test_labels": "not supplied",
            "expected_policy_samples": sum(
                item.label_frames * len(INPUT_CONDITIONS)
                for item in development
            ),
            "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
        },
    ))
    return Plan(
        repo=repo, workspace=workspace, build_dir=build_dir, conf=conf,
        python=python, depth_anything_root=depth_anything_root,
        depth_weights=depth_weights, active_split=active_split,
        bootstrap_manifest=bootstrap_manifest, datasets=datasets,
        geometries=geometries, geometry_manifest=geometry_manifest,
        input_variant_manifest=input_manifest,
        geometry_manifest_path=geometry_manifest_path,
        input_variant_manifest_path=input_manifest_path,
        steps=tuple(steps), estimates=_build_estimates(datasets, geometries),
    )


def _source_complete(step):
    labels = step.output / "labels.jsonl"
    summary_path = step.output / "summary.json"
    contract_path = step.output / "source_contract.json"
    if not all(path.is_file() for path in (labels, summary_path, contract_path)):
        return False
    try:
        summary = load_json(summary_path)
        contract = load_json(contract_path)
        depth_run = Path(step.metadata["depth_run"]).resolve()
        depth_manifest_path = depth_run / "depth_run_manifest.json"
        if not depth_manifest_path.is_file():
            return False
        depth_manifest_sha256 = sha256(depth_manifest_path)
        run_contract = contract.get("run_contract", {})
        depth_authentication = contract.get("depth_authentication", {})
        if (not isinstance(run_contract, dict) or
                not isinstance(depth_authentication, dict)):
            return False
        variant = contract["input_variant"]
        input_color.validate_input_variant(variant)
        return (
            summary.get("accepted") == step.metadata["expected_labels"] and
            summary.get("labels_sha256") == sha256(labels) and
            summary.get("source_contract_sha256") == sha256(contract_path) and
            variant == step.metadata["input_variant"] and
            contract.get("input_variant_sha256") ==
            input_color.input_variant_sha256(step.metadata["input_variant"]) and
            variant["windows_sdr_white_level_raw"] ==
            step.metadata["raw_white"] and
            Path(contract.get("clips", "")).resolve() ==
            Path(step.metadata["dataset_root"]).resolve() and
            Path(contract.get("run", "")).resolve() == depth_run and
            run_contract.get("kind") == "depth_run_manifest" and
            run_contract.get("sha256") == depth_manifest_sha256 and
            depth_authentication.get("manifest_sha256") ==
            depth_manifest_sha256 and
            contract.get("metric_preview_encoding") ==
            step.metadata["metric_preview_encoding"] and
            run_contract.get(
                "suite_manifest_sha256"
            ) == step.metadata["dataset_manifest_sha256"]
        )
    except (KeyError, RuntimeError, OSError, ValueError, TypeError):
        return False


def _depth_complete(step):
    """Authenticate one resumable dataset/input depth publication."""
    manifest_path = step.output / "depth_run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
        variant = step.metadata["input_variant"]
        input_color.validate_input_variant(variant)
        executable = Path(step.metadata["executable"])
        conf = Path(step.metadata["conf"])
        if (
            manifest.get("schema") != depth_run.DEPTH_RUN_MANIFEST_SCHEMA
            or manifest.get("harness_schema") !=
            selector.EXPECTED_HARNESS_SCHEMA
            or manifest.get("purpose") !=
            "artistic-policy depth supervision"
            or Path(manifest.get("suite", "")).resolve() !=
            Path(step.metadata["dataset_root"]).resolve()
            or manifest.get("suite_manifest_sha256") !=
            step.metadata["dataset_manifest_sha256"]
            or manifest.get("clip_hash_manifest_file_sha256") !=
            step.metadata["clip_hash_manifest_sha256"]
            or manifest.get("clip_hash_verification") !=
            step.metadata["clip_hash_verification"]
            or manifest.get("executable_sha256") !=
            step.metadata["executable_sha256"]
            or sha256(executable) != step.metadata["executable_sha256"]
            or manifest.get("conf_sha256") != step.metadata["conf_sha256"]
            or depth_run.eval_semantic_file_hash(conf) !=
            step.metadata["conf_sha256"]
            or manifest.get("model") != step.metadata["model"]
            or manifest.get("model_asset_identity") !=
            step.metadata["model_asset_identity"]
            or manifest.get("model_asset_identity_sha256") !=
            step.metadata["model_asset_identity_sha256"]
            or depth_run.selected_depth_model_identity(
                executable, step.metadata["model"]
            ) != step.metadata["model_asset_identity"]
            or manifest.get("input_variant") != variant
            or manifest.get("input_variant_sha256") !=
            input_color.input_variant_sha256(variant)
            or manifest.get("metric_preview_encoding") !=
            step.metadata["metric_preview_encoding"]
            or manifest.get("output_gt_right_only") is not False
        ):
            return False
        rows = manifest.get("clips")
        if (not isinstance(rows, list) or
                [row.get("clip") for row in rows if isinstance(row, dict)] !=
                step.metadata["clips"] or
                manifest.get("clip_count") != len(step.metadata["clips"])):
            return False
        identities = manifest.get("source_identities")
        if not isinstance(identities, dict):
            return False
        for row in rows:
            clip = row["clip"]
            if (row.get("metric_preview_encoding") !=
                    step.metadata["metric_preview_encoding"] or
                    not depth_run.valid_completed_clip(
                        Path(step.metadata["dataset_root"]) / clip,
                        step.output / clip,
                        step.metadata["model"],
                        step.metadata["executable_sha256"],
                        step.metadata["conf_sha256"],
                        identities.get(clip),
                        False,
                        variant,
                    )):
                return False
        return True
    except (KeyError, RuntimeError, OSError, ValueError, TypeError):
        return False


def _render_complete(step):
    results = step.output / "results.json"
    if not results.is_file():
        return False
    try:
        payload = load_json(results)
        meta = payload["meta"]
        if (payload.get("verdict") not in {"comparison_only", "hard_failures"} or
                meta.get("run_name") != step.output.name or
                not math.isclose(
                    float(meta["artistic_scale_override"]),
                    float(step.metadata["scale"]), abs_tol=1e-8,
                ) or meta.get("output_selection_mode") != "label-frames" or
                meta.get("artistic_policy") is not False or
                Path(meta.get("clips_root", "")).resolve() !=
                Path(step.metadata["clips_root"]).resolve() or
                meta.get("clip_hash_manifest_sha256") !=
                step.metadata["clip_hash_manifest_sha256"] or
                set(payload.get("clips", {})) != set(step.metadata["clips"])):
            return False
        for clip_name, entry in payload["clips"].items():
            clip = entry["meta"]
            if (clip.get("harness_schema") !=
                    selector.EXPECTED_HARNESS_SCHEMA or
                    clip.get("color_mode") != step.metadata["color_mode"] or
                    clip.get("hdr_source_kind") !=
                    selector.sbs_contract.input_variant_hdr_source_kind(
                        step.metadata["input_variant"]
                    ) or
                    clip.get("metric_preview_encoding") !=
                    step.metadata["metric_preview_encoding"] or
                    clip.get("sdr_white_level_raw") !=
                    (step.metadata["raw_white"] or 0) or
                    not math.isclose(
                        float(clip.get("hdr_input_scale")),
                        (step.metadata["raw_white"] or 0) / 1000.0,
                        abs_tol=1e-9,
                    ) or clip.get("eye_width") !=
                    step.metadata["eye_width"] or
                    clip.get("eye_height") !=
                    step.metadata["eye_height"] or
                    clip.get("depth_compensation") != "none" or
                    clip.get("output_selection_mode") != "label-frames"):
                return False
            if step.metadata["identity"]:
                clip_root = step.output / clip_name
                frame_ids = clip.get("output_selected_frame_ids")
                if (not (clip_root / "contract.json").is_file() or
                        not isinstance(frame_ids, list) or not frame_ids):
                    return False
                for frame_id in frame_ids:
                    if (not isinstance(frame_id, int) or
                            isinstance(frame_id, bool) or frame_id < 0 or
                            not (clip_root /
                                 f"warp_disparity_{frame_id:05d}.f32").is_file() or
                            not (clip_root /
                                 f"warp_unclamped_disparity_{frame_id:05d}.f32").is_file()):
                        return False
        return True
    except (KeyError, RuntimeError, OSError, ValueError, TypeError):
        return False


def _bundle_complete(step, schema):
    labels = step.output / "labels.jsonl"
    summary_path = step.output / "summary.json"
    contract_path = step.output / "label_fitter_contract.json"
    if not all(path.is_file() for path in (labels, summary_path, contract_path)):
        return False
    try:
        summary = load_json(summary_path)
        complete = (
            summary.get("schema") == schema and
            summary.get("labels_sha256") == sha256(labels) and
            summary.get("label_fitter_contract_sha256") ==
            sha256(contract_path) and
            (summary.get("accepted", summary.get("unique_rgb_count")) ==
             step.metadata["expected_labels"])
        )
        contract = load_json(contract_path)
        if not complete:
            return False
        label_merge.validate_label_fitter_code(
            contract.get("code"), schema, str(contract_path)
        )
        if schema != label_merge.LABEL_SCHEMA:
            return True
        expected_samples = step.metadata["expected_policy_samples"]
        expected_condition_count = step.metadata.get(
            "expected_condition_count", len(INPUT_CONDITIONS)
        )
        if (contract.get("schema") != label_merge.LABEL_SCHEMA or
                contract.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                summary.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                summary.get("condition_target_count_per_rgb") !=
                expected_condition_count or
                summary.get("unique_rgb_count") * expected_condition_count !=
                expected_samples):
            return False
        row_count = 0
        with labels.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                targets = row.get("input_condition_targets")
                if (row.get("label_schema") != label_merge.LABEL_SCHEMA or
                        row.get("condition_target_contract") !=
                        label_merge.CONDITION_TARGET_CONTRACT or
                        not isinstance(targets, list) or
                        len(targets) != expected_condition_count):
                    return False
                identities = set()
                for target in targets:
                    variant = target.get("input_variant")
                    input_color.validate_input_variant(variant)
                    identity = input_color.input_variant_sha256(variant)
                    if (target.get("contract") !=
                            label_merge.CONDITION_TARGET_CONTRACT or
                            target.get("input_variant_sha256") != identity or
                            target.get("deployment_geometry_variant_count") !=
                            len(GEOMETRIES) or identity in identities):
                        return False
                    identities.add(identity)
                row_count += 1
        return (
            row_count == step.metadata["expected_labels"] and
            row_count * expected_condition_count == expected_samples
        )
    except (KeyError, RuntimeError, OSError, ValueError, TypeError):
        return False


def _command_value(step, option):
    try:
        index = step.command.index(option)
        return Path(step.command[index + 1]).resolve()
    except (ValueError, IndexError):
        return None


def _training_complete(step):
    checkpoint_path = step.output / "artistic_policy_best.pt"
    history_path = step.output / "history.json"
    contract_path = step.output / "training_contract.json"
    if not all(path.is_file() for path in (
            checkpoint_path, history_path, contract_path)):
        return False
    try:
        import torch
        import artistic_policy_model as policy_model

        contract = load_json(contract_path)
        history = load_json(history_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (not isinstance(history, list) or not history or
                not isinstance(checkpoint, dict) or
                contract.get("schema") != policy_model.POLICY_CHECKPOINT_SCHEMA or
                checkpoint.get("schema") !=
                policy_model.POLICY_CHECKPOINT_SCHEMA or
                contract.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                checkpoint.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                contract.get("active_split_sha256") !=
                step.metadata["active_split_sha256"] or
                checkpoint.get("active_split_sha256") !=
                step.metadata["active_split_sha256"]):
            return False
        sources = contract.get("labels")
        if not isinstance(sources, list) or len(sources) != len(EXPECTED_DATASETS):
            return False
        rgb_counts = {"training": 0, "development": 0}
        sample_counts = {"training": 0, "development": 0}
        for source in sources:
            labels = Path(source.get("path", ""))
            if not labels.is_file() or sha256(labels) != source.get("sha256"):
                return False
            summary = load_json(labels.parent / "summary.json")
            # Every orchestration merge path includes its dataset key; map it
            # back to the fixed working split without opening any sealed data.
            matching = [
                value for value in ("training", "development")
                if f"-{value}" in labels.parent.name
            ]
            if len(matching) != 1:
                return False
            split = matching[0]
            count = summary.get("unique_rgb_count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                return False
            rgb_counts[split] += count
            sample_counts[split] += count * len(INPUT_CONDITIONS)
        return (
            rgb_counts["training"] ==
            step.metadata["expected_training_rgb_labels"] and
            rgb_counts["development"] ==
            step.metadata["expected_development_rgb_labels"] and
            sample_counts["training"] ==
            step.metadata["expected_training_policy_samples"] and
            sample_counts["development"] ==
            step.metadata["expected_development_policy_samples"]
        )
    except (ImportError, KeyError, RuntimeError, OSError, ValueError, TypeError):
        return False


def _evaluation_complete(step):
    evaluation_path = step.output / "evaluation.json"
    report_path = step.output / "report.html"
    if not evaluation_path.is_file() or not report_path.is_file():
        return False
    try:
        payload = load_json(evaluation_path)
        runtime = payload.get("runtime_regime_evaluation")
        decision = payload.get("decision")
        policy = _command_value(step, "--policy")
        if (payload.get("schema") != evaluation_contract.EVALUATION_SCHEMA or
                payload.get("split") != "development" or
                payload.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                not isinstance(runtime, dict) or not isinstance(decision, dict) or
                runtime.get("contract") !=
                evaluation_contract.RUNTIME_REGIME_ACCEPTANCE_CONTRACT or
                runtime.get("condition_target_contract") !=
                label_merge.CONDITION_TARGET_CONTRACT or
                runtime.get("hdr_aggregation_contract") !=
                evaluation_contract.HDR_AGGREGATION_CONTRACT or
                runtime.get("expected_hdr_white_levels_raw") !=
                list(WHITE_LEVELS) or runtime.get("missing_regimes") != [] or
                runtime.get("missing_hdr_white_levels_raw") != [] or
                runtime.get("unexpected_hdr_white_levels_raw") != [] or
                runtime.get("incomplete_source_frame_count") != 0 or
                runtime.get("source_condition_coverage_complete") is not True or
                set(runtime.get("regimes", {})) != {"sdr", "hdr"} or
                set(runtime.get("hdr_by_white_level_raw", {})) !=
                {str(value) for value in WHITE_LEVELS} or
                policy is None or not policy.is_file() or
                payload.get("checkpoint_sha256") != sha256(policy)):
            return False
        sdr = runtime["regimes"]["sdr"].get("primary", {})
        hdr = runtime["regimes"]["hdr"].get("primary", {})
        white_primary = [
            runtime["hdr_by_white_level_raw"][str(value)].get("primary", {})
            for value in WHITE_LEVELS
        ]
        expected = step.metadata["expected_policy_samples"]
        if (sdr.get("variant_sample_count", 0) +
                hdr.get("variant_sample_count", 0) != expected or
                sdr.get("variant_sample_count") !=
                sdr.get("unique_rgb_sample_count") or
                hdr.get("variant_sample_count") !=
                len(WHITE_LEVELS) * hdr.get("unique_rgb_sample_count", -1) or
                any(primary.get("variant_sample_count") !=
                    primary.get("unique_rgb_sample_count")
                    for primary in white_primary) or
                sum(primary.get("variant_sample_count", 0)
                    for primary in white_primary) !=
                hdr.get("variant_sample_count") or
                any(primary.get("unique_rgb_sample_count") !=
                    hdr.get("unique_rgb_sample_count")
                    for primary in white_primary) or
                any(primary.get("shot_count") != sdr.get("shot_count")
                    for primary in white_primary) or
                hdr.get("shot_count") != sdr.get("shot_count") or
                not isinstance(decision.get("runtime_regime_acceptance"), dict)):
            return False
        return report_path.stat().st_size > 0
    except (KeyError, RuntimeError, OSError, ValueError, TypeError):
        return False


def step_complete(step):
    if step.kind == "depth":
        return _depth_complete(step)
    if step.kind == "source":
        return _source_complete(step)
    if step.kind == "render":
        return _render_complete(step)
    if step.kind == "select":
        return _bundle_complete(step, 8)
    if step.kind == "merge":
        return _bundle_complete(step, label_merge.LABEL_SCHEMA)
    if step.kind == "train":
        return _training_complete(step)
    if step.kind == "evaluate":
        return _evaluation_complete(step)
    raise RuntimeError(f"unknown orchestration step kind: {step.kind}")


def _safe_remove(path, root):
    path = Path(path).resolve(strict=False)
    root = Path(root).resolve(strict=False)
    if path == root or not is_relative_to(path, root):
        raise RuntimeError(f"refusing to remove path outside stage root: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def compact_render(step, eval_root):
    """Retain the replayable minimum after an authenticated render result."""
    if not _render_complete(step):
        raise RuntimeError(f"refusing to compact incomplete render: {step.key}")
    output = step.output.resolve()
    eval_root = Path(eval_root).resolve()
    if not is_relative_to(output, eval_root) or output == eval_root:
        raise RuntimeError("render compaction escaped the evaluator output root")
    identity = bool(step.metadata["identity"])
    retained = {output / "results.json"}
    if identity:
        for clip in step.metadata["clips"]:
            clip_root = output / clip
            retained.add(clip_root / "contract.json")
            retained.update(clip_root.glob("warp_disparity_*.f32"))
            retained.update(clip_root.glob("warp_unclamped_disparity_*.f32"))
    retained = {path.resolve(strict=False) for path in retained}
    deleted_files = 0
    deleted_bytes = 0
    for path in sorted(output.rglob("*"), key=lambda item: len(item.parts),
                       reverse=True):
        resolved = path.resolve(strict=False)
        if path.is_file() and resolved not in retained:
            deleted_bytes += path.stat().st_size
            path.unlink()
            deleted_files += 1
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    marker = {
        "schema": 1,
        "contract": "artistic-bootstrap-render-compaction-v1",
        "identity": identity,
        "results_sha256": sha256(output / "results.json"),
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "retained": (
            "results plus clip contracts and clamped/unclamped disparity"
            if identity else "results only"
        ),
    }
    write_json_atomic(output / "bootstrap_compaction.json", marker)


def identity_screen(plan):
    """Prove action potential independently for every authenticated condition.

    A hard identity failure makes only that condition's two-geometry target a
    confidence-zero no-op.  It must never erase an otherwise actionable target
    for the same source frame under native SDR or another HDR white.
    """
    identity_steps = [
        step for step in plan.steps if step.phase == "identity"
    ]
    if not identity_steps or any(not _render_complete(step)
                                 for step in identity_steps):
        expected = len(plan.datasets) * len(INPUT_CONDITIONS) * len(plan.geometries)
        raise RuntimeError(
            f"identity screen requires all {expected} authenticated runs"
        )
    by_dataset = {}
    for dataset in plan.datasets:
        conditions = {}
        for condition, raw_white in INPUT_CONDITIONS:
            clip_failures = {clip: [] for clip in dataset.clips}
            matching = [
                step for step in identity_steps
                if (step.metadata["dataset"] == dataset.key and
                    step.metadata["condition"] == condition)
            ]
            if len(matching) != len(plan.geometries):
                raise RuntimeError(
                    f"identity screen lacks both geometries for "
                    f"{dataset.key}/{condition}"
                )
            for step in matching:
                payload = load_json(step.output / "results.json")
                by_clip = {}
                for failure in payload.get("hard_failures", []):
                    clip = failure.get("clip")
                    metric = failure.get("metric")
                    if clip in clip_failures and isinstance(metric, str):
                        by_clip.setdefault(clip, set()).add(metric)
                for clip, metrics in by_clip.items():
                    clip_failures[clip].append({
                        "eye_width": step.metadata["eye_width"],
                        "eye_height": step.metadata["eye_height"],
                        "metrics": sorted(metrics),
                    })
            feasible = sorted(
                clip for clip, failures in clip_failures.items() if not failures
            )
            conditions[condition] = {
                "runtime_regime": "sdr" if raw_white is None else "hdr",
                "raw_white": raw_white,
                "color_mode": _condition_color_mode(raw_white),
                "clips": len(dataset.clips),
                "identity_feasible_across_two_geometries": len(feasible),
                "potentially_actionable_clips": feasible,
                "identity_hard_failure_evidence": {
                    clip: failures for clip, failures in clip_failures.items()
                    if failures
                },
            }
        by_dataset[dataset.key] = {
            "split": dataset.split,
            "clips": len(dataset.clips),
            "conditions": conditions,
        }
    split_counts = {}
    for split in ("training", "development"):
        rows = [value for value in by_dataset.values()
                if value["split"] == split]
        conditions = {
            condition: {
                "runtime_regime": "sdr" if raw_white is None else "hdr",
                "raw_white": raw_white,
                "clips": sum(value["conditions"][condition]["clips"]
                             for value in rows),
                "identity_feasible_across_two_geometries": sum(
                    value["conditions"][condition][
                        "identity_feasible_across_two_geometries"
                    ] for value in rows
                ),
            }
            for condition, raw_white in INPUT_CONDITIONS
        }
        split_counts[split] = {
            "clips": sum(value["clips"] for value in rows),
            "conditions": conditions,
            "regimes": {
                "sdr": conditions["sdr"][
                    "identity_feasible_across_two_geometries"
                ],
                "hdr": sum(
                    conditions[condition][
                        "identity_feasible_across_two_geometries"
                    ] for condition, raw_white in INPUT_CONDITIONS
                    if raw_white is not None
                ),
            },
        }
    blocked_conditions = sorted(
        f"{split}:{condition}"
        for split, value in split_counts.items()
        for condition, evidence in value["conditions"].items()
        if evidence["identity_feasible_across_two_geometries"] == 0
    )
    blocked_regimes = sorted({
        f"{item.split(':', 1)[0]}:"
        f"{'sdr' if item.endswith(':sdr') else 'hdr'}"
        for item in blocked_conditions
    })
    blocked_splits = sorted({
        item.split(":", 1)[0] for item in blocked_conditions
    })
    return {
        "schema": 2,
        "contract": "artistic-bootstrap-condition-identity-admission-v2",
        "identity_semantics": (
            "a measured hard failure disconnects multiplier candidates only "
            "for that authenticated condition; missing evidence is never "
            "admitted as a negative"
        ),
        "required_cross_product": {
            "input_conditions": [
                {
                    "name": condition,
                    "color_mode": _condition_color_mode(raw_white),
                    "raw_white": raw_white,
                }
                for condition, raw_white in INPUT_CONDITIONS
            ],
            "geometries": [geometry.key for geometry in plan.geometries],
        },
        "datasets": by_dataset,
        "splits": split_counts,
        "blocked_conditions": blocked_conditions,
        "blocked_regimes": blocked_regimes,
        "blocked_splits": blocked_splits,
        "decision": (
            "stop-before-candidate-grid" if blocked_conditions else "proceed"
        ),
        "reason": (
            "one or more required conditions has zero possible actionable rows "
            "after its two-geometry intersection" if blocked_conditions else
            "identity evidence leaves candidates for every condition in both "
            "working splits"
        ),
    }


def target_coverage_screen(plan):
    """Verify condition-own action/identity classes before training starts."""
    merge_steps = [step for step in plan.steps if step.kind == "merge"]
    if (len(merge_steps) != len(plan.datasets) or
            any(not _bundle_complete(step, label_merge.LABEL_SCHEMA)
                for step in merge_steps)):
        raise RuntimeError("target coverage requires all authenticated merge bundles")
    first = {}
    sample_counts = {"training": 0, "development": 0}
    rgb_counts = {"training": 0, "development": 0}
    for step in merge_steps:
        split = step.metadata["split"]
        labels = step.output / "labels.jsonl"
        with labels.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rgb_counts[split] += 1
                for target in row["input_condition_targets"]:
                    variant = target["input_variant"]
                    identity = target["input_variant_sha256"]
                    condition = (
                        "sdr" if variant["kind"] == input_color.INPUT_KIND_SDR
                        else f"w{variant['windows_sdr_white_level_raw']}"
                    )
                    key = (split, row["film_id"], row["clip"], identity)
                    candidate = (int(row["frame"]), condition, target)
                    if key not in first or candidate[0] < first[key][0]:
                        first[key] = candidate
                    sample_counts[split] += 1
    coverage = {}
    blocked_conditions = []
    for split in ("training", "development"):
        conditions = {}
        for condition, raw_white in INPUT_CONDITIONS:
            selected = [
                target for (row_split, _film, _clip, _variant),
                (_frame, row_condition, target) in first.items()
                if row_split == split and row_condition == condition
            ]
            actionable = sum(
                abs(float(target["safe_scale_ceiling"]) - 1.0) >= 0.005
                for target in selected
            )
            identity = len(selected) - actionable
            conditions[condition] = {
                "runtime_regime": "sdr" if raw_white is None else "hdr",
                "raw_white": raw_white,
                "shot_condition_count": len(selected),
                "actionable_shot_condition_count": actionable,
                "identity_shot_condition_count": identity,
                "passed": actionable > 0 and identity > 0,
            }
            if not conditions[condition]["passed"]:
                blocked_conditions.append(f"{split}:{condition}")
        coverage[split] = {
            "rgb_label_count": rgb_counts[split],
            "policy_sample_count": sample_counts[split],
            "conditions": conditions,
            "regime_pass": {
                "sdr": conditions["sdr"]["passed"],
                "hdr": all(
                    conditions[condition]["passed"]
                    for condition, raw_white in INPUT_CONDITIONS
                    if raw_white is not None
                ),
            },
        }
    expected = plan.estimates
    cardinality_pass = (
        rgb_counts["training"] == expected["training_rgb_labels"] and
        rgb_counts["development"] == expected["development_rgb_labels"] and
        sample_counts["training"] == expected["training_policy_samples"] and
        sample_counts["development"] == expected["development_policy_samples"]
    )
    if not cardinality_pass:
        blocked_conditions.append("cardinality")
    return {
        "schema": 1,
        "contract": "artistic-bootstrap-condition-target-coverage-v1",
        "condition_target_contract": label_merge.CONDITION_TARGET_CONTRACT,
        "coverage": coverage,
        "expected_cardinality": {
            key: expected[key] for key in (
                "training_rgb_labels", "development_rgb_labels",
                "training_policy_samples", "development_policy_samples",
            )
        },
        "cardinality_pass": cardinality_pass,
        "blocked_conditions": sorted(blocked_conditions),
        "decision": "stop-before-training" if blocked_conditions else "proceed",
    }


def _run_step(step, plan, logs_root):
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"{step.key}.log"
    print(f"\n[{step.phase}] {step.key}", flush=True)
    print(subprocess.list2cmdline(list(step.command)), flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            step.command, cwd=plan.repo, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if step.kind == "render":
        # Comparison-only grids intentionally return 1 when a hard gate fails;
        # the authenticated result is negative evidence consumed by selection.
        if return_code not in {0, 1} or not _render_complete(step):
            raise RuntimeError(
                f"render step failed or published an invalid result: {step.key}; "
                f"see {log_path}"
            )
    elif return_code != 0 or not step_complete(step):
        raise RuntimeError(
            f"step failed or published an invalid result: {step.key}; "
            f"see {log_path}"
        )


class _ParallelRenderCancelled(RuntimeError):
    """Internal signal used to stop sibling render subprocesses promptly."""


def _validate_render_group(steps, plan):
    """Require one phase of independent evaluator outputs.

    This guard keeps the generic parallel runner from ever being reused for
    depth generation, selection, merging, training, or evaluation.
    """
    steps = tuple(steps)
    if not steps:
        return steps
    phases = {step.phase for step in steps}
    if (len(phases) != 1 or not phases.issubset({"identity", "render"}) or
            any(step.kind != "render" for step in steps)):
        raise RuntimeError(
            "parallel execution is restricted to one render phase"
        )
    expected_script = (
        Path(plan.repo) / "tools" / "sbsbench" / "run_eval.py"
    ).resolve(strict=False)
    if any(
            len(step.command) < 2 or
            Path(step.command[1]).resolve(strict=False) != expected_script
            for step in steps):
        raise RuntimeError(
            "parallel render group contains a non-run_eval command"
        )
    eval_root = (Path(plan.build_dir) / "sbs_eval").resolve(strict=False)
    outputs = [step.output.resolve(strict=False) for step in steps]
    if any(output == eval_root or not is_relative_to(output, eval_root)
           for output in outputs):
        raise RuntimeError("parallel render output escapes evaluator root")
    if len(outputs) != len(set(outputs)):
        raise RuntimeError("parallel render group has duplicate outputs")
    for index, output in enumerate(outputs):
        for other in outputs[index + 1:]:
            if is_relative_to(output, other) or is_relative_to(other, output):
                raise RuntimeError(
                    "parallel render outputs are not independent"
                )
    return steps


def _terminate_subprocess_tree(process):
    """Best-effort bounded teardown for a cancelled evaluator process tree."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def _run_render_step_buffered(step, plan, logs_root, cancel_event):
    """Run one evaluator with isolated logging and cooperative cancellation."""
    log_path = logs_root / f"{step.key}.log"
    popen_options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt" else {"start_new_session": True}
    )
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            step.command, cwd=plan.repo, stdout=log,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", **popen_options,
        )
        while process.poll() is None:
            if cancel_event.wait(0.1):
                _terminate_subprocess_tree(process)
                raise _ParallelRenderCancelled(
                    f"cancelled after a sibling render failed: {step.key}"
                )
        return_code = process.returncode
    if return_code not in {0, 1} or not _render_complete(step):
        raise RuntimeError(
            f"render step failed or published an invalid result: {step.key}; "
            f"see {log_path}"
        )
    print(f"[complete] {step.key}; log: {log_path}", flush=True)


def _run_render_batch(steps, plan, logs_root, workers):
    """Run one identity or candidate group with at most two subprocesses."""
    steps = _validate_render_group(steps, plan)
    if not steps:
        return
    if not 1 <= workers <= MAX_RENDER_WORKERS:
        raise RuntimeError(
            f"render workers must be between 1 and {MAX_RENDER_WORKERS}"
        )
    resolved_logs = Path(logs_root).resolve(strict=False)
    workspace = Path(plan.workspace).resolve(strict=False)
    if (resolved_logs == workspace or
            not is_relative_to(resolved_logs, workspace)):
        raise RuntimeError("parallel render logs escape workspace")
    logs_root.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        for step in steps:
            _run_step(step, plan, logs_root)
        return

    cancel_event = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    future_to_step = {}
    try:
        for step in steps:
            print(f"\n[{step.phase}] {step.key}", flush=True)
            print(subprocess.list2cmdline(list(step.command)), flush=True)
            future = executor.submit(
                _run_render_step_buffered, step, plan, logs_root,
                cancel_event,
            )
            future_to_step[future] = step
        done, pending = concurrent.futures.wait(
            future_to_step, return_when=concurrent.futures.FIRST_EXCEPTION,
        )
        failures = [future for future in done if future.exception() is not None]
        if failures:
            cancel_event.set()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            ordered_failures = []
            for future, step in future_to_step.items():
                if future.cancelled():
                    continue
                error = future.exception()
                if error is not None:
                    ordered_failures.append((step, error))
            primary = next(
                (item for item in ordered_failures
                 if not isinstance(item[1], _ParallelRenderCancelled)),
                ordered_failures[0],
            )
            raise RuntimeError(
                f"parallel render group failed at {primary[0].key}: "
                f"{primary[1]}"
            ) from primary[1]
        # FIRST_EXCEPTION returns only after every future when none failed.
        for future in future_to_step:
            future.result()
    finally:
        cancel_event.set()
        executor.shutdown(wait=True, cancel_futures=True)


def _prepare_step(step, plan, args):
    """Apply resume/restart/stale-output policy before launching a step."""
    complete = step_complete(step)
    if args.restart and not complete and step.output.exists():
        root = (
            plan.build_dir / "sbs_eval" if step.kind == "render"
            else plan.workspace
        )
        _safe_remove(step.output, root)
        complete = False
    if complete:
        print(f"[resume] {step.key}", flush=True)
        return False
    if step.output.exists() and any(step.output.iterdir()):
        raise RuntimeError(
            f"stale or partial output blocks {step.key}: {step.output}; "
            "use --restart to replace this experiment's outputs"
        )
    return True


def execute(plan, args):
    write_json_atomic(plan.geometry_manifest_path, plan.geometry_manifest)
    write_json_atomic(
        plan.input_variant_manifest_path, plan.input_variant_manifest
    )
    plan_path = plan.workspace / "orchestration_plan.json"
    write_json_atomic(plan_path, plan.as_dict())
    logs_root = plan.workspace / "orchestration_logs"
    state_path = plan.workspace / "orchestration_state.json"
    completed = []
    stop_index = PHASES.index(args.stop_after)
    screen = None
    coverage_screen = None
    eligible_steps = tuple(
        step for step in plan.steps
        if PHASES.index(step.phase) <= stop_index
    )

    def record_complete(step):
        completed.append(step.key)
        write_json_atomic(state_path, {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "plan_sha256": canonical_sha256(plan.as_dict()),
            "post_bootstrap_required": (
                "fresh schema-28 core+extended production baselines/reports; "
                "do not rescore schema-25 preview artifacts"
            ),
            "stop_after": args.stop_after,
            "completed": completed,
            "last_completed": step.key,
        })

    index = 0
    while index < len(eligible_steps):
        step = eligible_steps[index]
        if step.kind == "render" and step.phase in {"identity", "render"}:
            end = index + 1
            while (end < len(eligible_steps) and
                   eligible_steps[end].kind == "render" and
                   eligible_steps[end].phase == step.phase):
                end += 1
            group = eligible_steps[index:end]
            _validate_render_group(group, plan)
            if step.phase == "render" and screen is None:
                # The complete identity phase is a hard barrier.  No candidate
                # subprocess is launched before its authenticated admission.
                screen = identity_screen(plan)
                write_json_atomic(
                    plan.workspace / "identity_screen.json", screen
                )
                if screen["blocked_splits"]:
                    raise RuntimeError(
                        "identity admission blocks the candidate grid: " +
                        ", ".join(screen["blocked_conditions"]) +
                        "; inspect identity_screen.json before changing data "
                        "or metrics"
                    )
            pending = tuple(
                item for item in group if _prepare_step(item, plan, args)
            )
            _run_render_batch(
                pending, plan, logs_root, args.render_workers
            )
            if args.compact_renders:
                for item in group:
                    compact_render(item, plan.build_dir / "sbs_eval")
            # Commit state in immutable plan order, never completion order.
            for item in group:
                record_complete(item)
            index = end
            continue

        if (step.phase == "train" and coverage_screen is None and
                stop_index >= PHASES.index("train")):
            coverage_screen = target_coverage_screen(plan)
            write_json_atomic(
                plan.workspace / "target_coverage_screen.json",
                coverage_screen,
            )
            if coverage_screen["blocked_conditions"]:
                raise RuntimeError(
                    "condition-target coverage blocks training: " +
                    ", ".join(coverage_screen["blocked_conditions"]) +
                    "; inspect target_coverage_screen.json"
                )
        if _prepare_step(step, plan, args):
            _run_step(step, plan, logs_root)
        record_complete(step)
        index += 1
    if (stop_index >= PHASES.index("identity") and
            all(step_complete(step) for step in plan.steps
                if step.phase == "identity")):
        screen = identity_screen(plan)
        write_json_atomic(plan.workspace / "identity_screen.json", screen)
    return {
        "plan": str(plan_path),
        "state": str(state_path),
        "stop_after": args.stop_after,
        "completed_steps": len(completed),
        "last_completed": completed[-1] if completed else None,
        "identity_screen": screen,
        "target_coverage_screen": coverage_screen,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--datasets-root", type=Path,
        help=(
            "optional immutable prepared bootstrap root to reuse; defaults to "
            "WORKSPACE/datasets"
        ),
    )
    parser.add_argument(
        "--build-dir", type=Path,
        default=Path("cmake-build-relwithdebinfo"),
    )
    parser.add_argument(
        "--conf", type=Path, default=Path("tools/sbsbench/bench.conf")
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--depth-anything-root", required=True, type=Path
    )
    parser.add_argument("--depth-weights", required=True, type=Path)
    parser.add_argument("--run-prefix", default="artboot")
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument(
        "--render-workers", type=int, default=1,
        help="parallel run_eval processes per identity/candidate phase (max 2)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stop-after", choices=PHASES, default="evaluate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restart", action="store_true",
        help=(
            "replace stale/partial outputs owned by this experiment while "
            "preserving every authenticated complete step"
        ),
    )
    parser.add_argument(
        "--no-compact-renders", action="store_false",
        dest="compact_renders",
        help="retain full sparse artifacts for every non-identity scale",
    )
    parser.set_defaults(compact_renders=True)
    args = parser.parse_args(argv)
    if (args.score_workers < 1 or args.epochs < 1 or args.batch_size < 1 or
            not 1 <= args.render_workers <= MAX_RENDER_WORKERS):
        parser.error(
            "score workers, epochs, and batch size must be positive; "
            f"render workers must be 1..{MAX_RENDER_WORKERS}"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("learning rate must be positive and finite")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            result = plan.as_dict()
        else:
            result = execute(plan, args)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"orchestration failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
