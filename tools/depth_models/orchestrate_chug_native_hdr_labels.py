#!/usr/bin/env python3
"""Generate authenticated native-PQ CHUG safety labels and stop before training.

This orchestration consumes only ``native_hdr_bootstrap_manifest.json`` from
``prepare_chug_native_hdr_training.py``.  It executes the production FP16
scRGB depth path, the exact Apollo render grid at both deployment geometries,
and one per-source native-PQ safety target.  It deliberately has no training
phase or training command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import artistic_geometry_contract as geometry_contract
import depth_input_color as input_color
import generate_artistic_depth_run as depth_run
import merge_artistic_geometry_labels as label_merge
import native_hdr_capture
import orchestrate_artistic_hdr_bootstrap as common
import prepare_chug_native_hdr_training as chug_prepare
import select_render_feasible_labels as selector


SCHEMA = 1
CONTRACT = "apollo-chug-native-pq-label-orchestration-v1"
DEPTH_MODEL = "depth_anything_v2_fp16"
SCALES = common.SCALES
GEOMETRIES = common.GEOMETRIES
PHASES = ("depth", "sources", "identity", "render", "select", "merge")
EXPECTED_HARNESS_SCHEMA = 28
ALLOWED_STEP_SCRIPTS = {
    "depth": "generate_artistic_depth_run.py",
    "source": "prepare_artistic_source_rows.py",
    "render": "run_eval.py",
    "select": "select_render_feasible_labels.py",
    "merge": "merge_artistic_geometry_labels.py",
}
EXPECTED_SOURCE_CLIPS = {
    "training": chug_prepare.TRAINING_COUNT,
    "development": chug_prepare.DEVELOPMENT_COUNT,
}
EXPECTED_WINDOW_CLIPS = {
    split: count * chug_prepare.LABELS_PER_CLIP
    for split, count in EXPECTED_SOURCE_CLIPS.items()
}
WINDOW_FRAME_COUNT = 2 * chug_prepare.TEMPORAL_WINDOW_RADIUS + 1
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Dataset:
    split: str
    root: Path
    clips: tuple[str, ...]
    context_frames: int
    label_frames: int
    source_clips: int
    dataset_manifest_sha256: str
    clip_hash_manifest_sha256: str

    @property
    def key(self):
        return f"chug-native-pq-{self.split}"


@dataclass
class Plan:
    repo: Path
    workspace: Path
    build_dir: Path
    conf: Path
    python: Path
    datasets: tuple[Dataset, ...]
    geometries: tuple[common.Geometry, ...]
    geometry_manifest: dict
    input_variant_manifest: dict
    geometry_manifest_path: Path
    input_variant_manifest_path: Path
    steps: tuple[common.Step, ...]
    bootstrap_manifest: Path

    def as_dict(self):
        return {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "terminal_phase": "merge",
            "training_command_present": False,
            "expected_harness_schema": EXPECTED_HARNESS_SCHEMA,
            "sparse_window_layout": {
                "contract": chug_prepare.FRAME_SELECTION_CONTRACT,
                "training_window_clips": EXPECTED_WINDOW_CLIPS["training"],
                "development_window_clips":
                    EXPECTED_WINDOW_CLIPS["development"],
                "frames_per_window": WINDOW_FRAME_COUNT,
                "labels_per_window": 1,
            },
            "global_contract_cardinality": {
                "input_variants": 5,
                "deployment_geometry_tuples": 4,
            },
            "bootstrap_manifest": str(self.bootstrap_manifest),
            "bootstrap_manifest_sha256": common.sha256(
                self.bootstrap_manifest
            ),
            "deployment_geometry_manifest": self.geometry_manifest,
            "deployment_geometry_manifest_identity":
                geometry_contract.allowlist_sha256(self.geometry_manifest),
            "input_variant_manifest": self.input_variant_manifest,
            "input_variant_manifest_identity":
                label_merge.input_variant_manifest_sha256(
                    self.input_variant_manifest
                ),
            "datasets": [{
                "split": item.split,
                "root": str(item.root),
                "clips": list(item.clips),
                "context_frames": item.context_frames,
                "label_frames": item.label_frames,
                "source_clips": item.source_clips,
                "dataset_manifest_sha256": item.dataset_manifest_sha256,
                "clip_hash_manifest_sha256":
                    item.clip_hash_manifest_sha256,
            } for item in self.datasets],
            "steps": [step.as_dict() for step in self.steps],
        }


def _absolute_manifest_path(value, description):
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"native HDR bootstrap lacks {description}")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(
            f"native HDR bootstrap {description} is not absolute: {value!r}"
        )
    return path.resolve(strict=False)


def _safe_clip_name(value):
    if (not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value) or
            value in {".", ".."} or "/" in value or "\\" in value or
            Path(value).name != value):
        raise RuntimeError(f"unsafe native HDR window-clip name: {value!r}")
    return value


def _require_sha256(value, description):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise RuntimeError(f"native HDR {description} is not a SHA-256")
    return value


def _validate_temporal_evidence_selection(value, source_label, clip_root,
                                          origin):
    expected_keys = {
        "contract", "flow_support_contract", "preferred_pair",
        "flow_support_metric_sha256",
        "minimum_support", "search_radius_frames", "search_order",
        "nominal_source_label_frame_id", "selected_source_label_frame_id",
        "selected_offset_frames", "selected_previous_source_frame_id",
        "selected_pair_flow_support",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError(f"native HDR flow-selection fields differ: {origin}")
    nominal = value.get("nominal_source_label_frame_id")
    support = value.get("selected_pair_flow_support")
    if (value.get("contract") !=
            chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT or
            value.get("flow_support_contract") !=
            chug_prepare.FLOW_SUPPORT_CONTRACT or
            value.get("flow_support_metric_sha256") !=
            chug_prepare.flow_support_metric_sha256() or
            value.get("preferred_pair") !=
            "previous-source-frame-to-label-frame" or
            value.get("minimum_support") !=
            chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT or
            value.get("search_radius_frames") !=
            chug_prepare.FLOW_SUPPORT_SEARCH_RADIUS_FRAMES or
            value.get("search_order") !=
            "nominal-then-negative-positive-by-distance" or
            type(nominal) is not int or type(source_label) is not int or
            value.get("selected_source_label_frame_id") != source_label or
            value.get("selected_offset_frames") != source_label - nominal or
            abs(source_label - nominal) >
            chug_prepare.FLOW_SUPPORT_SEARCH_RADIUS_FRAMES or
            value.get("selected_previous_source_frame_id") != source_label - 1 or
            not isinstance(support, (int, float)) or isinstance(support, bool) or
            not math.isfinite(float(support)) or
            not chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT <=
            float(support) <= 1.0):
        raise RuntimeError(f"native HDR flow-selection contract differs: {origin}")
    previous = selector.sbsbench.load_gray(
        str(clip_root / "frame_00000.png")
    )
    current = selector.sbsbench.load_gray(
        str(clip_root / "frame_00001.png")
    )
    _temporal, _depth, measured = selector.sbsbench.flow_temporal_metrics(
        current, current, previous, previous, current, previous,
        min_support=1.1,
    )
    if (not math.isclose(measured, float(support), rel_tol=0.0, abs_tol=1e-12) or
            measured < chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT):
        raise RuntimeError(
            f"native HDR selected flow support differs: {origin}"
        )
    return value


def _validate_window_clip(clip_root, sequence, split, conversion_sha256):
    clip = _safe_clip_name(sequence.get("clip"))
    expected_keys = {
        "clip", "frames", "source_frames", "master_source_frames",
        "source_frame_rate", "label_frames", "split", "capture_group_id",
        "video_id", "window_index", "source_label_frame_id",
        "nominal_source_label_frame_id", "selected_pair_flow_support",
        "temporal_evidence_selection",
    }
    if set(sequence) != expected_keys:
        raise RuntimeError(f"native HDR window row fields differ: {clip}")
    window_index = sequence.get("window_index")
    video_id = sequence.get("video_id")
    capture_group = sequence.get("capture_group_id")
    source_label = sequence.get("source_label_frame_id")
    master_frames = sequence.get("master_source_frames")
    source_frame_rate = sequence.get("source_frame_rate")
    if (sequence.get("frames") != WINDOW_FRAME_COUNT or
            sequence.get("source_frames") != WINDOW_FRAME_COUNT or
            sequence.get("label_frames") != 1 or
            sequence.get("split") != split or
            type(window_index) is not int or
            window_index not in range(chug_prepare.LABELS_PER_CLIP) or
            type(source_label) is not int or
            source_label < chug_prepare.TEMPORAL_WINDOW_RADIUS or
            type(master_frames) is not int or
            master_frames <=
            source_label + chug_prepare.TEMPORAL_WINDOW_RADIUS or
            not isinstance(source_frame_rate, (int, float)) or
            isinstance(source_frame_rate, bool) or
            not math.isfinite(float(source_frame_rate)) or
            float(source_frame_rate) <= 0.0 or
            not isinstance(video_id, str) or not video_id or
            not isinstance(capture_group, str) or not capture_group or
            clip != chug_prepare._window_clip_name(video_id, window_index)):
        raise RuntimeError(f"native HDR sparse-window identity differs: {clip}")
    labels = common.load_json(clip_root / "label_frames.json")
    if labels != {
            "schema": 1,
            "frame_ids": [chug_prepare.TEMPORAL_WINDOW_RADIUS],
    }:
        raise RuntimeError(f"native HDR center label differs: {clip}")
    metadata = common.load_json(clip_root / "meta.json")
    selection = metadata.get("frame_selection")
    if (metadata.get("preparation_contract") !=
            chug_prepare.PREPARATION_CONTRACT or
            metadata.get("split") != split or
            metadata.get("capture_group_id") != capture_group or
            metadata.get("source_video_id") != video_id or
            metadata.get("source_kind") != "native-hdr-video" or
            metadata.get("native_hdr") is not True or
            not isinstance(selection, dict)):
        raise RuntimeError(f"native HDR window metadata differs: {clip}")
    expected_selection_keys = {
        "contract", "source_frame_count", "source_frame_rate",
        "window_index", "retained_frame_count", "temporal_window_radius",
        "label_frame_ids", "source_label_frame_id", "frames",
        "temporal_evidence_selection",
    }
    frame_rows = selection.get("frames")
    if (set(selection) != expected_selection_keys or
            selection.get("contract") != chug_prepare.FRAME_SELECTION_CONTRACT or
            selection.get("source_frame_count") != master_frames or
            selection.get("source_frame_rate") != source_frame_rate or
            selection.get("window_index") != window_index or
            selection.get("retained_frame_count") != WINDOW_FRAME_COUNT or
            selection.get("temporal_window_radius") !=
            chug_prepare.TEMPORAL_WINDOW_RADIUS or
            selection.get("label_frame_ids") !=
            [chug_prepare.TEMPORAL_WINDOW_RADIUS] or
            selection.get("source_label_frame_id") !=
            sequence.get("source_label_frame_id") or
            not isinstance(frame_rows, list) or
            len(frame_rows) != WINDOW_FRAME_COUNT):
        raise RuntimeError(f"native HDR frame-selection contract differs: {clip}")
    local_ids = []
    source_ids = []
    timestamps = []
    for row in frame_rows:
        if (not isinstance(row, dict) or
                set(row) != {
                    "frame", "source_frame", "source_timestamp_seconds",
                } or type(row.get("frame")) is not int or
                type(row.get("source_frame")) is not int or
                not isinstance(row.get("source_timestamp_seconds"),
                               (int, float)) or
                isinstance(row.get("source_timestamp_seconds"), bool) or
                not math.isfinite(float(row["source_timestamp_seconds"])) or
                float(row["source_timestamp_seconds"]) < 0.0):
            raise RuntimeError(f"native HDR frame map is invalid: {clip}")
        local_ids.append(row["frame"])
        source_ids.append(row["source_frame"])
        timestamps.append(float(row["source_timestamp_seconds"]))
    if (local_ids != list(range(WINDOW_FRAME_COUNT)) or
            source_ids != list(range(
                source_label - chug_prepare.TEMPORAL_WINDOW_RADIUS,
                source_label + chug_prepare.TEMPORAL_WINDOW_RADIUS + 1,
            )) or any(
                later <= earlier
                for earlier, later in zip(timestamps, timestamps[1:])
            ) or any(
                not math.isclose(
                    timestamp, source_frame / float(source_frame_rate),
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                for source_frame, timestamp in zip(source_ids, timestamps)
            )):
        raise RuntimeError(f"native HDR sparse-window cadence differs: {clip}")
    evidence = _validate_temporal_evidence_selection(
        selection.get("temporal_evidence_selection"), source_label,
        clip_root, clip,
    )
    if (sequence.get("nominal_source_label_frame_id") !=
            evidence["nominal_source_label_frame_id"] or
            sequence.get("selected_pair_flow_support") !=
            evidence["selected_pair_flow_support"] or
            sequence.get("temporal_evidence_selection") != evidence):
        raise RuntimeError(f"native HDR flow-selection provenance differs: {clip}")
    authentication = native_hdr_capture.validate_clip(clip_root, full=False)
    sidecar, sidecar_frames, _manifest_path = native_hdr_capture.load_manifest(
        clip_root
    )
    source_video = sidecar.get("source_video")
    expected_source_video = {
        "dataset": "CHUG",
        "video_id": video_id,
        "split": split,
        "capture_group_id": capture_group,
        "license": "CC BY-NC-SA 4.0",
        "source_frame_count": master_frames,
        "source_frame_rate": source_frame_rate,
        "frame_selection_contract": chug_prepare.FRAME_SELECTION_CONTRACT,
        "window_index": window_index,
        "source_window_frame_ids": source_ids,
        "source_label_frame_id": source_label,
        "source_window_timestamps_seconds": timestamps,
        "temporal_evidence_selection": evidence,
    }
    valid_sidecar_frames = (
        isinstance(sidecar_frames, dict) and
        sorted(sidecar_frames) == local_ids and
        all(
            isinstance(sidecar_frames[frame], dict) and
            isinstance(sidecar_frames[frame].get("timestamp_seconds"),
                       (int, float)) and
            not isinstance(sidecar_frames[frame].get("timestamp_seconds"), bool)
            for frame in local_ids
        )
    )
    if (authentication.get("frame_count") != WINDOW_FRAME_COUNT or
            authentication.get("width") != chug_prepare.TARGET_WIDTH or
            authentication.get("height") != chug_prepare.TARGET_HEIGHT or
            sidecar.get("conversion", {}).get("contract_sha256") !=
            conversion_sha256 or
            not isinstance(source_video, dict) or
            any(source_video.get(key) != value
                for key, value in expected_source_video.items()) or
            not valid_sidecar_frames or
            any(
                not math.isclose(
                    float(sidecar_frames[frame].get("timestamp_seconds")),
                    timestamps[frame], rel_tol=1e-12, abs_tol=1e-12,
                )
                for frame in local_ids
            )):
        raise RuntimeError(f"native HDR model-source sidecar differs: {clip}")
    return video_id, capture_group, window_index, master_frames


def _validate_dataset_publication(root, split, item, conversion_sha256):
    expected_windows = EXPECTED_WINDOW_CLIPS[split]
    expected_sources = EXPECTED_SOURCE_CLIPS[split]
    expected_context = expected_windows * WINDOW_FRAME_COUNT
    dataset_root = _absolute_manifest_path(item.get("root"), f"{split} root")
    expected_root = (root / split).resolve(strict=False)
    dataset_manifest = _absolute_manifest_path(
        item.get("dataset_manifest"), f"{split} dataset manifest"
    )
    clip_manifest = item.get("clip_hash_manifest")
    if not isinstance(clip_manifest, dict):
        raise RuntimeError(f"native HDR bootstrap lacks {split} clip manifest")
    clip_manifest_path = _absolute_manifest_path(
        clip_manifest.get("path"), f"{split} clip manifest"
    )
    clips_value = item.get("clips")
    if not isinstance(clips_value, list):
        raise RuntimeError(f"native HDR {split} clips are invalid")
    clips = tuple(_safe_clip_name(clip) for clip in clips_value)
    if (dataset_root != expected_root or not dataset_root.is_dir() or
            dataset_manifest != dataset_root / "dataset_manifest.json" or
            clip_manifest_path != dataset_root / "clip_hash_manifest.json" or
            not dataset_manifest.is_file() or
            not clip_manifest_path.is_file() or
            len(clips) != expected_windows or len(set(clips)) != len(clips) or
            list(clips) != sorted(clips)):
        raise RuntimeError(f"native HDR {split} publication layout differs")
    dataset_sha = _require_sha256(
        item.get("dataset_manifest_sha256"),
        f"{split} dataset-manifest digest",
    )
    clip_sha = _require_sha256(
        clip_manifest.get("sha256"), f"{split} clip-manifest digest"
    )
    semantic_sha = _require_sha256(
        clip_manifest.get("semantic_content_sha256"),
        f"{split} clip semantic digest",
    )
    if (common.sha256(dataset_manifest) != dataset_sha or
            common.sha256(clip_manifest_path) != clip_sha):
        raise RuntimeError(f"native HDR {split} publication digest differs")
    manifest = common.load_json(dataset_manifest)
    sequences = manifest.get("sequences")
    if (manifest.get("schema") != 2 or
            manifest.get("dataset") != "chug-native-pq-v1" or
            manifest.get("domain") != "native_hdr_cinematic" or
            manifest.get("source_kind") != "native-hdr-video" or
            manifest.get("split") != split or
            manifest.get("source_split") != split or
            manifest.get("preparation_contract") !=
            chug_prepare.PREPARATION_CONTRACT or
            manifest.get("temporal_evidence_selection_contract") !=
            chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT or
            manifest.get("source_flow_support_contract") !=
            chug_prepare.FLOW_SUPPORT_CONTRACT or
            manifest.get("source_flow_metric_sha256") !=
            chug_prepare.flow_support_metric_sha256() or
            manifest.get("source_flow_support_minimum") !=
            chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT or
            manifest.get("conversion_contract_sha256") != conversion_sha256 or
            not isinstance(sequences, list) or
            [row.get("clip") for row in sequences
             if isinstance(row, dict)] != list(clips) or
            manifest.get("frame_count") != expected_context or
            manifest.get("source_frame_count") != expected_context or
            manifest.get("label_frame_count") != expected_windows):
        raise RuntimeError(f"native HDR {split} dataset contract differs")
    actual_dirs = {
        path.name for path in dataset_root.iterdir() if path.is_dir()
    }
    if actual_dirs != set(clips):
        raise RuntimeError(f"native HDR {split} has stale/extra clip folders")
    by_video = {}
    by_group = {}
    master_by_video = {}
    for sequence in sequences:
        clip_root = (dataset_root / sequence["clip"]).resolve(strict=True)
        if not clip_root.is_relative_to(dataset_root):
            raise RuntimeError("native HDR window clip escapes dataset root")
        video_id, capture_group, window_index, master_frames = (
            _validate_window_clip(
                clip_root, sequence, split, conversion_sha256
            )
        )
        by_video.setdefault(video_id, set()).add(window_index)
        by_group.setdefault(capture_group, set()).add(video_id)
        master_by_video.setdefault(video_id, set()).add(master_frames)
    expected_indices = set(range(chug_prepare.LABELS_PER_CLIP))
    master_source_frames = sum(
        next(iter(values)) for values in master_by_video.values()
    )
    if (len(by_video) != expected_sources or len(by_group) != expected_sources or
            any(indices != expected_indices for indices in by_video.values()) or
            any(len(video_ids) != 1 for video_ids in by_group.values()) or
            any(len(values) != 1 for values in master_by_video.values()) or
            item.get("capture_group_ids") != sorted(by_group)):
        raise RuntimeError(f"native HDR {split} sparse-window grouping differs")
    if (item.get("context_frame_count") != expected_context or
            item.get("source_context_frame_count") != expected_context or
            item.get("master_source_frame_count") != master_source_frames or
            manifest.get("master_source_frame_count") != master_source_frames or
            item.get("label_frame_count") != expected_windows):
        raise RuntimeError(f"native HDR {split} bootstrap counts differ")
    clip_hash_payload = depth_run.clip_hashes.load_manifest(clip_manifest_path)
    if (set(clip_hash_payload.get("clips", {})) != set(clips) or
            clip_hash_payload.get(
                depth_run.clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
            ) != semantic_sha):
        raise RuntimeError(f"native HDR {split} clip identity set differs")
    depth_run.clip_hashes.verify_selected_clips(
        clip_manifest_path, dataset_root, clips, full=False
    )
    return Dataset(
        split=split, root=dataset_root, clips=clips,
        context_frames=expected_context, label_frames=expected_windows,
        source_clips=expected_sources,
        dataset_manifest_sha256=dataset_sha,
        clip_hash_manifest_sha256=clip_sha,
    )


def _validate_dataset_bootstrap(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"native HDR dataset root is not a directory: {root}")
    manifest_path = root / chug_prepare.BOOTSTRAP_MANIFEST
    payload = common.load_json(manifest_path)
    output_root = _absolute_manifest_path(
        payload.get("output_root"), "output root"
    )
    conversion_path = _absolute_manifest_path(
        payload.get("conversion_contract"), "conversion contract"
    )
    conversion_sha256 = _require_sha256(
        payload.get("conversion_contract_sha256"),
        "conversion-contract digest",
    )
    if (payload.get("schema") != chug_prepare.PREPARATION_SCHEMA or
            payload.get("contract") != chug_prepare.PREPARATION_CONTRACT or
            output_root != root or
            conversion_path != root / "conversion_contract.json" or
            not conversion_path.is_file() or
            payload.get("temporal_evidence_selection_contract") !=
            chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT or
            payload.get("source_flow_support_contract") !=
            chug_prepare.FLOW_SUPPORT_CONTRACT or
            payload.get("source_flow_metric_sha256") !=
            chug_prepare.flow_support_metric_sha256() or
            payload.get("source_flow_support_minimum") !=
            chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT or
            payload.get("sealed_test_policy") !=
            "CHUG test masters were not decoded or opened"):
        raise RuntimeError("native HDR bootstrap manifest is stale or invalid")
    conversion = common.load_json(conversion_path)
    if native_hdr_capture.canonical_sha256(conversion) != conversion_sha256:
        raise RuntimeError("native HDR conversion contract digest differs")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {
            "training", "development"}:
        raise RuntimeError("native HDR bootstrap dataset set differs")
    rows = []
    for split in ("training", "development"):
        item = datasets.get(split)
        if not isinstance(item, dict):
            raise RuntimeError(f"native HDR bootstrap lacks {split}")
        rows.append(_validate_dataset_publication(
            root, split, item, conversion_sha256
        ))
    training_groups = set(datasets["training"]["capture_group_ids"])
    development_groups = set(datasets["development"]["capture_group_ids"])
    if training_groups & development_groups:
        raise RuntimeError("native HDR capture group crosses train/development")
    retention = payload.get("retention")
    if (not isinstance(retention, dict) or
            retention.get("contract") !=
            chug_prepare.FRAME_SELECTION_CONTRACT or
            retention.get("temporal_window_radius") !=
            chug_prepare.TEMPORAL_WINDOW_RADIUS or
            retention.get("temporal_evidence_selection") != {
                "contract": chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT,
                "flow_support_contract":
                    chug_prepare.FLOW_SUPPORT_CONTRACT,
                "flow_support_metric_sha256":
                    chug_prepare.flow_support_metric_sha256(),
                "minimum_support":
                    chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT,
                "search_radius_frames":
                    chug_prepare.FLOW_SUPPORT_SEARCH_RADIUS_FRAMES,
                "search_order":
                    "nominal-then-negative-positive-by-distance",
                "preferred_pair": "previous-source-frame-to-label-frame",
            } or
            retention.get("stored_identity") !=
            "independent-contiguous-window-clip-with-source-frame-map"):
        raise RuntimeError("native HDR sparse retention contract differs")
    retention_splits = retention.get("splits")
    for split in ("training", "development"):
        item = retention_splits.get(split) if isinstance(
            retention_splits, dict
        ) else None
        expected_windows = EXPECTED_WINDOW_CLIPS[split]
        if (not isinstance(item, dict) or
                item.get("source_clips") != EXPECTED_SOURCE_CLIPS[split] or
                item.get("window_clips") != expected_windows or
                item.get("retained_frames") !=
                expected_windows * WINDOW_FRAME_COUNT or
                item.get("label_frames") != expected_windows):
            raise RuntimeError(f"native HDR {split} retention counts differ")
    summary = payload.get("summary")
    if summary != {
            "training_clips": EXPECTED_WINDOW_CLIPS["training"],
            "development_clips": EXPECTED_WINDOW_CLIPS["development"],
            "training_policy_samples": EXPECTED_WINDOW_CLIPS["training"],
            "development_policy_samples":
            EXPECTED_WINDOW_CLIPS["development"],
    }:
        raise RuntimeError("native HDR bootstrap summary differs")
    return manifest_path, tuple(rows)


def _geometry(source_width, source_height, name, eye_width, eye_height):
    return common._geometry(
        source_width, source_height, name, eye_width, eye_height
    )


def _validate_workspace_path(workspace):
    if not workspace.is_absolute() or workspace.parent == workspace:
        raise RuntimeError(f"unsafe native HDR workspace: {workspace}")


def _validate_global_manifests(geometry_manifest, input_manifest, geometries):
    geometry_contract.validate_allowlist(geometry_manifest)
    expected_geometry = geometry_contract.build_allowlist([
        common._geometry_value(1280, 720, geometry, color_mode)
        for geometry in geometries
        for color_mode in (
            input_color.COLOR_MODE_SDR, input_color.COLOR_MODE_HDR,
        )
    ])
    if (geometry_manifest != expected_geometry or
            len(geometry_manifest["tuples"]) != 4):
        raise RuntimeError(
            "native HDR orchestration requires exactly four global geometry "
            "tuples"
        )
    label_merge.validate_policy_input_variant_manifest(input_manifest)
    if len(input_manifest["variants"]) != 5:
        raise RuntimeError(
            "native HDR orchestration requires exactly five global inputs"
        )


def build_plan(args):
    repo = Path(__file__).resolve().parents[2]
    workspace = args.workspace.resolve()
    dataset_root = args.dataset_root.resolve()
    build_dir = args.build_dir.resolve()
    conf = args.conf.resolve()
    python = args.python.resolve()
    _validate_workspace_path(workspace)
    if args.run_prefix != common.slug(args.run_prefix):
        raise RuntimeError(
            "run prefix must already be a lowercase, path-safe slug"
        )
    executable = build_dir / "sunshine.exe"
    bootstrap_manifest, datasets = _validate_dataset_bootstrap(dataset_root)
    for path, description in (
            (executable, "benchmark executable"),
            (conf, "evaluator config"),
            (python, "Python interpreter")):
        if not path.is_file():
            raise RuntimeError(f"missing {description}: {path}")

    geometries = tuple(
        _geometry(1280, 720, name, width, height)
        for name, width, height in GEOMETRIES
    )
    # Use the same global geometry and input allow-lists as the SDR-origin
    # branch.  A native source exercises only its two HDR geometry tuples and
    # one native-PQ condition; the unused declarations remain authenticated.
    geometry_manifest = geometry_contract.build_allowlist([
        common._geometry_value(1280, 720, geometry, color_mode)
        for geometry in geometries
        for color_mode in (
            input_color.COLOR_MODE_SDR, input_color.COLOR_MODE_HDR,
        )
    ])
    input_manifest = label_merge.build_input_variant_manifest(
        label_merge.policy_input_variants()
    )
    _validate_global_manifests(geometry_manifest, input_manifest, geometries)
    manifests_root = workspace / "manifests"
    geometry_manifest_path = manifests_root / "deployment-geometries.json"
    input_manifest_path = manifests_root / "input-variants.json"
    native = input_color.native_pq_input_variant()
    preview = selector.sbs_contract.input_variant_metric_preview_encoding(native)
    executable_sha = common.sha256(executable)
    conf_sha = depth_run.eval_semantic_file_hash(conf)
    model_identity = depth_run.selected_depth_model_identity(
        executable, DEPTH_MODEL
    )
    model_identity_hash = depth_run.clip_hashes.canonical_json_sha256(
        model_identity
    )
    eval_root = build_dir / "sbs_eval"
    steps = []
    depth_steps = []
    source_steps = []
    identity_steps = []
    candidate_render_steps = []
    source_outputs = {}
    render_results = {}

    for dataset in datasets:
        depth_root = workspace / "depth" / dataset.key
        source_root = workspace / "sources" / dataset.key
        source_outputs[dataset.key] = source_root
        depth_steps.append(common.Step(
            key=f"depth-{dataset.key}", phase="depth", kind="depth",
            command=(
                str(python),
                str(repo / "tools" / "depth_models" /
                    "generate_artistic_depth_run.py"),
                "--suite", str(dataset.root), "--output", str(depth_root),
                "--build-dir", str(build_dir), "--conf", str(conf),
                "--model", DEPTH_MODEL, "--native-hdr-scrgb",
                "--verify-clip-hashes",
            ),
            output=depth_root,
            metadata={
                "dataset": dataset.key, "condition": "native-pq",
                "raw_white": None, "color_mode": native["color_mode"],
                "clips": list(dataset.clips),
                "dataset_root": str(dataset.root),
                "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
                "clip_hash_manifest_sha256":
                    dataset.clip_hash_manifest_sha256,
                "clip_hash_verification": "full",
                "executable": str(executable),
                "executable_sha256": executable_sha,
                "conf": str(conf), "conf_sha256": conf_sha,
                "model": DEPTH_MODEL,
                "model_asset_identity": model_identity,
                "model_asset_identity_sha256": model_identity_hash,
                "input_variant": native,
                "metric_preview_encoding": preview,
            },
        ))
        source_steps.append(common.Step(
            key=f"source-{dataset.key}", phase="sources", kind="source",
            command=(
                str(python),
                str(repo / "tools" / "depth_models" /
                    "prepare_artistic_source_rows.py"),
                "--run", str(depth_root), "--clips", str(dataset.root),
                "--output", str(source_root),
            ),
            output=source_root,
            metadata={
                "dataset": dataset.key, "condition": "native-pq",
                "raw_white": None, "color_mode": native["color_mode"],
                "input_variant": native,
                "metric_preview_encoding": preview,
                "expected_labels": dataset.label_frames,
                "dataset_root": str(dataset.root),
                "depth_run": str(depth_root),
                "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
            },
        ))

    steps.extend(depth_steps)
    steps.extend(source_steps)

    for dataset in datasets:
        for geometry in geometries:
            for scale in SCALES:
                scale_name = common.scale_slug(scale)
                label = (
                    f"{args.run_prefix}-{dataset.key}-native-pq-"
                    f"g{geometry.eye_width}-{scale_name}"
                )
                output = eval_root / label
                command = (
                    str(python),
                    str(repo / "tools" / "sbsbench" / "run_eval.py"),
                    "--build-dir", str(build_dir), "--conf", str(conf),
                    "--clips-root", str(dataset.root), "--label", label,
                    "--score-workers", str(args.score_workers),
                    "--comparison-only", "--extra", "--eye-w",
                    str(geometry.eye_width), "--eye-h", str(geometry.eye_height),
                    "--native-hdr-scrgb", "--no-artistic-policy",
                    "--artistic-scale-override", f"{scale:.1f}",
                    "--output-label-frames",
                )
                render_results[(dataset.key, geometry.key, scale)] = (
                    output / "results.json"
                )
                render_step = common.Step(
                    key=(f"render-{dataset.key}-native-pq-{geometry.key}-"
                         f"{scale_name}"),
                    phase="identity" if math.isclose(scale, 1.0) else "render",
                    kind="render", command=command, output=output,
                    metadata={
                        "dataset": dataset.key, "clips": list(dataset.clips),
                        "clips_root": str(dataset.root),
                        "clip_hash_manifest_sha256":
                            dataset.clip_hash_manifest_sha256,
                        "condition": "native-pq", "raw_white": None,
                        "color_mode": native["color_mode"],
                        "input_variant": native,
                        "metric_preview_encoding": preview,
                        "expected_harness_schema": EXPECTED_HARNESS_SCHEMA,
                        "hdr_source_kind":
                            selector.sbs_contract.input_variant_hdr_source_kind(
                                native
                            ),
                        "scale": scale, "eye_width": geometry.eye_width,
                        "eye_height": geometry.eye_height,
                        "identity": math.isclose(scale, 1.0),
                    },
                )
                if render_step.metadata["identity"]:
                    identity_steps.append(render_step)
                else:
                    candidate_render_steps.append(render_step)

    # Identity controls must be authenticated before any non-identity render.
    # This also makes --stop-after identity a complete, resumable checkpoint.
    steps.extend(identity_steps)
    steps.extend(candidate_render_steps)

    selected = {}
    for dataset in datasets:
        for geometry in geometries:
            output = workspace / "selected" / dataset.key / geometry.key
            command = [
                str(python),
                str(repo / "tools" / "depth_models" /
                    "select_render_feasible_labels.py"),
                "--source-labels",
                str(source_outputs[dataset.key] / "labels.jsonl"),
                "--control",
                str(render_results[(dataset.key, geometry.key, 1.0)]),
            ]
            for scale in SCALES:
                command.extend((
                    "--candidate",
                    f"{scale:.1f}="
                    f"{render_results[(dataset.key, geometry.key, scale)]}",
                ))
            command.extend(("--output", str(output)))
            selected[(dataset.key, geometry.key)] = output
            steps.append(common.Step(
                key=f"select-{dataset.key}-native-pq-{geometry.key}",
                phase="select", kind="select", command=tuple(command),
                output=output,
                metadata={
                    "dataset": dataset.key, "condition": "native-pq",
                    "raw_white": None, "color_mode": native["color_mode"],
                    "geometry": geometry.key,
                    "expected_labels": dataset.label_frames,
                },
            ))

    for dataset in datasets:
        output = workspace / "merged" / dataset.key
        command = [
            str(python),
            str(repo / "tools" / "depth_models" /
                "merge_artistic_geometry_labels.py"),
        ]
        for geometry in geometries:
            command.extend((
                "--geometry-labels",
                str(selected[(dataset.key, geometry.key)] / "labels.jsonl"),
            ))
        command.extend((
            "--deployment-geometry-manifest", str(geometry_manifest_path),
            "--input-variant-manifest", str(input_manifest_path),
            "--output", str(output),
        ))
        steps.append(common.Step(
            key=f"merge-{dataset.key}-native-pq", phase="merge", kind="merge",
            command=tuple(command), output=output,
            metadata={
                "dataset": dataset.key, "split": dataset.split,
                "expected_labels": dataset.label_frames,
                "expected_policy_samples": dataset.label_frames,
                "expected_condition_count": 1,
                "condition_target_contract":
                    label_merge.CONDITION_TARGET_CONTRACT,
            },
        ))

    plan = Plan(
        repo=repo, workspace=workspace, build_dir=build_dir, conf=conf,
        python=python, datasets=datasets, geometries=geometries,
        geometry_manifest=geometry_manifest,
        input_variant_manifest=input_manifest,
        geometry_manifest_path=geometry_manifest_path,
        input_variant_manifest_path=input_manifest_path,
        steps=tuple(steps), bootstrap_manifest=bootstrap_manifest,
    )
    _validate_label_only_plan(plan)
    return plan


def _validate_label_only_plan(plan):
    """Reject any plan that can escape label generation or its stage roots."""
    if (PHASES != (
            "depth", "sources", "identity", "render", "select", "merge") or
            selector.EXPECTED_HARNESS_SCHEMA != EXPECTED_HARNESS_SCHEMA):
        raise RuntimeError(
            "native HDR label orchestration contract is stale; training is "
            "not permitted"
        )
    _validate_workspace_path(plan.workspace)
    _validate_global_manifests(
        plan.geometry_manifest, plan.input_variant_manifest, plan.geometries
    )
    expected_dataset_counts = {
        item.split: (item.source_clips, len(item.clips), item.label_frames)
        for item in plan.datasets
    }
    if expected_dataset_counts != {
            split: (
                EXPECTED_SOURCE_CLIPS[split], EXPECTED_WINDOW_CLIPS[split],
                EXPECTED_WINDOW_CLIPS[split],
            )
            for split in ("training", "development")
    }:
        raise RuntimeError("native HDR plan does not bind the 80-window layout")
    expected_counts = {
        "depth": len(plan.datasets),
        "source": len(plan.datasets),
        "render": len(plan.datasets) * len(plan.geometries) * len(SCALES),
        "select": len(plan.datasets) * len(plan.geometries),
        "merge": len(plan.datasets),
    }
    counts = {
        kind: sum(step.kind == kind for step in plan.steps)
        for kind in expected_counts
    }
    if counts != expected_counts or len(plan.steps) != sum(expected_counts.values()):
        raise RuntimeError("native HDR label-only step cross-product differs")
    phase_counts = {
        phase: sum(step.phase == phase for step in plan.steps)
        for phase in PHASES
    }
    if (phase_counts["identity"] !=
            len(plan.datasets) * len(plan.geometries) or
            phase_counts["render"] !=
            len(plan.datasets) * len(plan.geometries) * (len(SCALES) - 1)):
        raise RuntimeError("native HDR identity/render phase split differs")
    phases = [PHASES.index(step.phase) for step in plan.steps]
    if phases != sorted(phases):
        raise RuntimeError("native HDR steps are not phase ordered")
    expected_scripts = {
        "depth": plan.repo / "tools" / "depth_models" /
        ALLOWED_STEP_SCRIPTS["depth"],
        "source": plan.repo / "tools" / "depth_models" /
        ALLOWED_STEP_SCRIPTS["source"],
        "render": plan.repo / "tools" / "sbsbench" /
        ALLOWED_STEP_SCRIPTS["render"],
        "select": plan.repo / "tools" / "depth_models" /
        ALLOWED_STEP_SCRIPTS["select"],
        "merge": plan.repo / "tools" / "depth_models" /
        ALLOWED_STEP_SCRIPTS["merge"],
    }
    expected_phase = {
        "depth": {"depth"},
        "source": {"sources"},
        "render": {"identity", "render"},
        "select": {"select"},
        "merge": {"merge"},
    }
    keys = set()
    outputs = set()
    eval_root = (plan.build_dir / "sbs_eval").resolve(strict=False)
    native = input_color.native_pq_input_variant()
    for step in plan.steps:
        if (step.kind not in expected_scripts or
                step.phase not in expected_phase[step.kind] or
                len(step.command) < 2 or
                Path(step.command[0]).resolve(strict=False) != plan.python or
                Path(step.command[1]).resolve(strict=False) !=
                expected_scripts[step.kind].resolve(strict=False)):
            raise RuntimeError(
                f"native HDR plan contains a non-label command: {step.key}"
            )
        output = step.output.resolve(strict=False)
        stage_root = eval_root if step.kind == "render" else plan.workspace
        if (not output.is_absolute() or output == stage_root or
                not output.is_relative_to(stage_root)):
            raise RuntimeError(
                f"native HDR step output escapes its stage root: {step.key}"
            )
        if step.key in keys or output in outputs:
            raise RuntimeError("native HDR plan has duplicate step identity")
        keys.add(step.key)
        outputs.add(output)
        if step.kind in {"depth", "source", "render"}:
            if (step.metadata.get("condition") != "native-pq" or
                    step.metadata.get("input_variant") != native or
                    step.metadata.get("color_mode") !=
                    input_color.COLOR_MODE_HDR):
                raise RuntimeError(
                    f"native HDR step input identity differs: {step.key}"
                )
        if step.kind == "render":
            if (step.metadata.get("expected_harness_schema") !=
                    EXPECTED_HARNESS_SCHEMA or
                    step.metadata.get("hdr_source_kind") !=
                    selector.sbs_contract.input_variant_hdr_source_kind(
                        native
                    ) or "--native-hdr-scrgb" not in step.command or
                    "--simulate-hdr" in step.command):
                raise RuntimeError(
                    f"native HDR render contract differs: {step.key}"
                )
    if (plan.geometry_manifest_path.resolve(strict=False) !=
            plan.workspace / "manifests" / "deployment-geometries.json" or
            plan.input_variant_manifest_path.resolve(strict=False) !=
            plan.workspace / "manifests" / "input-variants.json"):
        raise RuntimeError("native HDR global manifest path differs")
    return plan


def _safe_remove(step, plan):
    root = plan.build_dir / "sbs_eval" if step.kind == "render" else plan.workspace
    common._safe_remove(step.output, root)


def _write_contract_once(path, payload, description):
    path = Path(path)
    if path.exists():
        if not path.is_file():
            raise RuntimeError(f"existing {description} is not a file: {path}")
        existing = common.load_json(path)
        if common.canonical_sha256(existing) != common.canonical_sha256(payload):
            raise RuntimeError(
                f"existing {description} belongs to a different immutable "
                f"plan: {path}; use a fresh workspace"
            )
        return
    common.write_json_atomic(path, payload)


def _render_compaction_complete(step):
    marker = step.output / "bootstrap_compaction.json"
    results = step.output / "results.json"
    if not marker.is_file() or not results.is_file():
        return False
    try:
        payload = common.load_json(marker)
        identity = payload.get("identity")
        return (
            payload.get("schema") == 1 and
            payload.get("contract") ==
            "artistic-bootstrap-render-compaction-v1" and
            type(identity) is bool and
            identity is bool(step.metadata["identity"]) and
            payload.get("results_sha256") == common.sha256(results)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def identity_screen(plan):
    """Fail closed before native-PQ multiplier candidates are rendered."""
    identity_steps = tuple(
        step for step in plan.steps if step.phase == "identity"
    )
    expected = len(plan.datasets) * len(plan.geometries)
    if (len(identity_steps) != expected or
            any(not common._render_complete(step) for step in identity_steps)):
        raise RuntimeError(
            f"native HDR identity screen requires all {expected} "
            "authenticated runs"
        )
    datasets = {}
    blocked_splits = []
    for dataset in plan.datasets:
        matching = tuple(
            step for step in identity_steps
            if step.metadata["dataset"] == dataset.key
        )
        if len(matching) != len(plan.geometries):
            raise RuntimeError(
                f"native HDR identity screen lacks both geometries for "
                f"{dataset.key}"
            )
        failures = {clip: [] for clip in dataset.clips}
        for step in matching:
            payload = common.load_json(step.output / "results.json")
            by_clip = {}
            for failure in payload.get("hard_failures", []):
                clip = failure.get("clip")
                metric = failure.get("metric")
                if clip in failures and isinstance(metric, str):
                    by_clip.setdefault(clip, set()).add(metric)
            for clip, metrics in by_clip.items():
                failures[clip].append({
                    "eye_width": step.metadata["eye_width"],
                    "eye_height": step.metadata["eye_height"],
                    "metrics": sorted(metrics),
                })
        feasible = sorted(
            clip for clip, evidence in failures.items() if not evidence
        )
        if not feasible:
            blocked_splits.append(dataset.split)
        datasets[dataset.key] = {
            "split": dataset.split,
            "clips": len(dataset.clips),
            "identity_feasible_across_two_geometries": len(feasible),
            "potentially_actionable_clips": feasible,
            "identity_hard_failure_evidence": {
                clip: evidence for clip, evidence in failures.items()
                if evidence
            },
        }
    blocked_splits = sorted(set(blocked_splits))
    return {
        "schema": 1,
        "contract": "native-pq-identity-admission-v1",
        "condition": "native-pq",
        "required_geometries": [item.key for item in plan.geometries],
        "datasets": datasets,
        "blocked_splits": blocked_splits,
        "decision": (
            "stop-before-candidate-grid" if blocked_splits else "proceed"
        ),
    }


def _prepare_step(step, plan, args):
    complete = common.step_complete(step)
    if args.restart and not complete and step.output.exists():
        _safe_remove(step, plan)
        complete = False
    if complete:
        print(f"[resume] {step.key}", flush=True)
        return False
    if step.output.exists():
        if not step.output.is_dir() or any(step.output.iterdir()):
            raise RuntimeError(
                f"stale or partial output blocks {step.key}; use --restart"
            )
    return True


def execute(plan, args):
    _validate_label_only_plan(plan)
    plan_payload = plan.as_dict()
    plan_sha256 = common.canonical_sha256(plan_payload)
    plan_path = plan.workspace / "native_hdr_label_plan.json"
    state_path = plan.workspace / "native_hdr_label_state.json"
    if state_path.exists():
        state = common.load_json(state_path)
        if (state.get("schema") != SCHEMA or
                state.get("contract") != CONTRACT or
                state.get("plan_sha256") != plan_sha256):
            raise RuntimeError(
                "existing native HDR state belongs to a different immutable "
                "plan; use a fresh workspace"
            )
    _write_contract_once(
        plan.geometry_manifest_path, plan.geometry_manifest,
        "deployment-geometry manifest",
    )
    _write_contract_once(
        plan.input_variant_manifest_path, plan.input_variant_manifest,
        "input-variant manifest",
    )
    _write_contract_once(plan_path, plan_payload, "native HDR label plan")
    logs = plan.workspace / "orchestration_logs"
    completed = []
    stop_index = PHASES.index(args.stop_after)
    screen = None
    eligible_steps = tuple(
        step for step in plan.steps
        if PHASES.index(step.phase) <= stop_index
    )

    def record_complete(step):
        completed.append(step.key)
        common.write_json_atomic(state_path, {
            "schema": SCHEMA, "contract": CONTRACT,
            "plan_sha256": plan_sha256,
            "stop_after": args.stop_after, "completed": completed,
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
            common._validate_render_group(group, plan)
            if step.phase == "render" and screen is None:
                screen = identity_screen(plan)
                common.write_json_atomic(
                    plan.workspace / "identity_screen.json", screen
                )
                if screen["blocked_splits"]:
                    raise RuntimeError(
                        "native HDR identity admission blocks the candidate "
                        "grid: " + ", ".join(screen["blocked_splits"])
                    )
            pending = tuple(
                item for item in group if _prepare_step(item, plan, args)
            )
            common._run_render_batch(
                pending, plan, logs, args.render_workers
            )
            if args.compact_renders:
                for item in group:
                    if not _render_compaction_complete(item):
                        common.compact_render(
                            item, plan.build_dir / "sbs_eval"
                        )
            for item in group:
                record_complete(item)
            index = end
            continue

        if _prepare_step(step, plan, args):
            common._run_step(step, plan, logs)
        record_complete(step)
        index += 1

    if (stop_index >= PHASES.index("identity") and
            all(common.step_complete(step) for step in plan.steps
                if step.phase == "identity")):
        screen = identity_screen(plan)
        common.write_json_atomic(
            plan.workspace / "identity_screen.json", screen
        )
    return {
        "plan": str(plan_path), "stop_after": args.stop_after,
        "completed_steps": len(completed),
        "last_completed": completed[-1] if completed else None,
        "identity_screen": screen,
        "training_started": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--build-dir", type=Path, default=Path("cmake-build-relwithdebinfo")
    )
    parser.add_argument(
        "--conf", type=Path, default=Path("tools/sbsbench/bench.conf")
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-prefix", default="art-native-pq")
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument(
        "--render-workers", type=int, default=1,
        help="parallel run_eval processes per identity/candidate phase (max 2)",
    )
    parser.add_argument("--stop-after", choices=PHASES, default="merge")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-compact-renders", action="store_false",
        dest="compact_renders",
    )
    parser.set_defaults(compact_renders=True)
    args = parser.parse_args(argv)
    if (args.score_workers < 1 or
            not 1 <= args.render_workers <= common.MAX_RENDER_WORKERS):
        parser.error(
            "score workers must be positive; render workers must be "
            f"1..{common.MAX_RENDER_WORKERS}"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        result = plan.as_dict() if args.dry_run else execute(plan, args)
    except (OSError, RuntimeError, ValueError,
            subprocess.SubprocessError) as error:
        raise SystemExit(f"native HDR orchestration failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
