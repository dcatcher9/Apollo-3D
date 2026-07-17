#!/usr/bin/env python3
"""Prepare authenticated full-cadence native-PQ CHUG train/dev clips.

This is deliberately separate from ``prepare_chug_native_hdr_training.py``.
The older production keeps five small metric windows per source video; this
production keeps every frame of the same frozen train/dev source-video split so
the causal ordinal controller can be replayed at the source cadence.  CHUG test
masters are never resolved, stat'ed, hashed, probed, decoded, or copied.

The color path is the exact production mirror already audited by the sparse
preparer: limited-range BT.2020 NCL decode, explicit ST-2084 EOTF, linear
BT.2020-to-Rec.709 conversion, 80-nit Windows scRGB normalization, an FP16
capture boundary, and the production HDR depth-input preview.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import native_hdr_capture
import prepare_chug_native_hdr_training as sparse
import preprocessing_artifact_cache as artifact_cache
import chug_native_raw_contract as raw_native


PREPARATION_SCHEMA = 3
PREPARATION_CONTRACT = "apollo-chug-native-pq-full-cadence-v1"
DATASET_NAME = "chug-native-pq-full-cadence-v3"
PRODUCTION_PREFIX = "chug_native_pq_full_cadence_v3"
FULL_CADENCE_CONTRACT = "full-cadence-native-pq-video-v1"
TIMING_SCHEMA = 1
TIMING_CONTRACT = "apollo-source-video-timing-v1"
CUT_SCHEMA = 1
CUT_CONTRACT = "apollo-source-preview-cut-evidence-v1"
SOURCE_BOOTSTRAP_SCHEMA = 3
SOURCE_BOOTSTRAP_CONTRACT = "apollo-chug-native-pq-training-v3"
BOOTSTRAP_MANIFEST = "native_hdr_full_cadence_manifest.json"
TARGET_WIDTH = sparse.TARGET_WIDTH
TARGET_HEIGHT = sparse.TARGET_HEIGHT
TRAINING_SOURCE_VIDEOS = 12
DEVELOPMENT_SOURCE_VIDEOS = 4
EXPECTED_SOURCE_FRAMES = {"training": 3088, "development": 930}
EXPECTED_LABEL_FRAMES = {"training": 60, "development": 20}
LABELS_PER_SOURCE_VIDEO = 5
SCRGB_BYTES_PER_PIXEL = sparse.SCRGB_BYTES_PER_PIXEL
SPLITS = sparse.SPLITS
CUT_ANALYSIS_WIDTH = 160
CUT_ANALYSIS_HEIGHT = 90
CUT_THRESHOLD = 0.18
RAW_NATIVE_SCHEMA = raw_native.SCHEMA
RAW_NATIVE_CONTRACT = raw_native.CONTRACT
RAW_NATIVE_MANIFEST = raw_native.MANIFEST


def _resolve_reference(document_path: Path, value, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"missing {label}")
    path = Path(value)
    if not path.is_absolute():
        path = document_path.parent / path
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"missing {label}: {path}") from error


def _verified_reference(document_path: Path, row, path_key: str,
                        hash_key: str, label: str) -> Path:
    path = _resolve_reference(document_path, row.get(path_key), label)
    expected = row.get(hash_key)
    if (not isinstance(expected, str) or len(expected) != 64 or
            sparse.sha256(path) != expected):
        raise RuntimeError(f"stale {label}: {path}")
    return path


def _rational(value, label: str) -> Fraction:
    if not isinstance(value, str) or not value or value in {"0/0", "N/A"}:
        raise RuntimeError(f"invalid {label}: {value!r}")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"invalid {label}: {value!r}") from error
    if result <= 0:
        raise RuntimeError(f"non-positive {label}: {value!r}")
    return result


def _fraction_record(value: Fraction):
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "rational": f"{value.numerator}/{value.denominator}",
        "decimal": float(value),
    }


def _load_sparse_selection(source_bootstrap_path: Path):
    """Collapse five sparse windows into one authenticated row per video."""

    source_bootstrap_path = source_bootstrap_path.resolve(strict=True)
    bootstrap = sparse.read_json(source_bootstrap_path, "sparse CHUG bootstrap")
    if (bootstrap.get("schema") != SOURCE_BOOTSTRAP_SCHEMA or
            bootstrap.get("contract") != SOURCE_BOOTSTRAP_CONTRACT or
            bootstrap.get("sealed_test_policy") !=
            "CHUG test masters were not decoded or opened"):
        raise RuntimeError("unsupported sparse CHUG bootstrap contract")
    datasets = bootstrap.get("datasets")
    if not isinstance(datasets, dict):
        raise RuntimeError("sparse CHUG bootstrap has no datasets")

    result = {}
    dataset_references = {}
    for split in SPLITS:
        entry = datasets.get(split)
        if not isinstance(entry, dict):
            raise RuntimeError(f"sparse CHUG bootstrap lacks {split}")
        manifest_path = _verified_reference(
            source_bootstrap_path, entry, "dataset_manifest",
            "dataset_manifest_sha256", f"sparse CHUG {split} manifest",
        )
        manifest = sparse.read_json(manifest_path, f"sparse CHUG {split} manifest")
        if (manifest.get("schema") != 2 or
                manifest.get("dataset") != "chug-native-pq-v1" or
                manifest.get("production_id") != f"chug_native_pq_v1_{split}" or
                manifest.get("split") != split or
                manifest.get("preparation_contract") !=
                SOURCE_BOOTSTRAP_CONTRACT):
            raise RuntimeError(f"sparse CHUG {split} dataset contract differs")
        sequences = manifest.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise RuntimeError(f"sparse CHUG {split} sequences are missing")
        by_video = {}
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise RuntimeError("sparse CHUG sequence is invalid")
            video_id = sequence.get("video_id")
            group = sequence.get("capture_group_id")
            source_frames = sequence.get("master_source_frames")
            fps = sequence.get("source_frame_rate")
            label = sequence.get("source_label_frame_id")
            window_index = sequence.get("window_index")
            if (not isinstance(video_id, str) or len(video_id) != 32 or
                    not isinstance(group, str) or len(group) != 64 or
                    type(source_frames) is not int or source_frames <= 0 or
                    not isinstance(fps, (int, float)) or
                    isinstance(fps, bool) or not math.isfinite(float(fps)) or
                    float(fps) <= 0.0 or type(label) is not int or
                    label < 0 or label >= source_frames or
                    type(window_index) is not int or
                    window_index < 0 or window_index >= LABELS_PER_SOURCE_VIDEO or
                    sequence.get("split") != split):
                raise RuntimeError("sparse CHUG sequence identity/cadence is invalid")
            row = by_video.setdefault(video_id, {
                "video_id": video_id,
                "split": split,
                "capture_group_id": group,
                "source_frame_count": source_frames,
                "sparse_source_frame_rate": float(fps),
                "label_frame_ids": [],
                "window_indices": [],
                "temporal_evidence_selection": [],
            })
            expected = (
                row["capture_group_id"], row["source_frame_count"],
                row["sparse_source_frame_rate"], row["split"],
            )
            current = (group, source_frames, float(fps), split)
            if current != expected:
                raise RuntimeError("sparse CHUG video changes identity across windows")
            row["label_frame_ids"].append(label)
            row["window_indices"].append(window_index)
            evidence = sequence.get("temporal_evidence_selection")
            if not isinstance(evidence, dict):
                raise RuntimeError("sparse CHUG curated label lacks temporal evidence")
            row["temporal_evidence_selection"].append(evidence)

        expected_videos = (TRAINING_SOURCE_VIDEOS if split == "training" else
                           DEVELOPMENT_SOURCE_VIDEOS)
        if len(by_video) != expected_videos:
            raise RuntimeError(
                f"sparse CHUG {split} has {len(by_video)} source videos; "
                f"expected {expected_videos}"
            )
        for row in by_video.values():
            row["label_frame_ids"].sort()
            row["window_indices"].sort()
            if (row["window_indices"] != list(range(LABELS_PER_SOURCE_VIDEO)) or
                    len(row["label_frame_ids"]) != LABELS_PER_SOURCE_VIDEO or
                    len(set(row["label_frame_ids"])) != LABELS_PER_SOURCE_VIDEO):
                raise RuntimeError("sparse CHUG window/label coverage differs")
        frame_total = sum(row["source_frame_count"] for row in by_video.values())
        label_total = sum(len(row["label_frame_ids"]) for row in by_video.values())
        if (frame_total != EXPECTED_SOURCE_FRAMES[split] or
                label_total != EXPECTED_LABEL_FRAMES[split] or
                manifest.get("master_source_frame_count") != frame_total or
                manifest.get("label_frame_count") != label_total):
            raise RuntimeError(f"sparse CHUG {split} frozen cardinality differs")
        result[split] = sorted(by_video.values(), key=lambda item: item["video_id"])
        dataset_references[split] = {
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": sparse.sha256(manifest_path),
        }
    all_groups = [row["capture_group_id"] for values in result.values()
                  for row in values]
    all_videos = [row["video_id"] for values in result.values() for row in values]
    if len(all_groups) != len(set(all_groups)) or len(all_videos) != len(set(all_videos)):
        raise RuntimeError("sparse CHUG source identity crosses train/dev")
    return result, {
        "sparse_bootstrap_manifest": str(source_bootstrap_path),
        "sparse_bootstrap_manifest_sha256": sparse.sha256(source_bootstrap_path),
        "sparse_datasets": dataset_references,
    }


def _selected_source_rows(chug_root: Path, selected):
    """Bind only selected train/dev media; sealed test paths remain untouched."""

    chug_root = chug_root.resolve(strict=True)
    selection_path = chug_root / "selection_manifest.json"
    receipt_path = chug_root / "download_receipt.json"
    selection = sparse.read_json(selection_path, "CHUG selection manifest")
    receipt = sparse.read_json(receipt_path, "CHUG download receipt")
    if (selection.get("schema") != 1 or receipt.get("schema") != 1 or
            receipt.get("license") != "CC BY-NC-SA 4.0"):
        raise RuntimeError("unsupported CHUG source/usage contract")
    selection_rows = {
        row.get("video_id"): row for row in selection.get("clips", ())
        if isinstance(row, dict)
    }
    receipt_rows = {
        row.get("video_id"): row for row in receipt.get("accepted", ())
        if isinstance(row, dict)
    }
    videos_root = (chug_root / "videos").resolve(strict=True)
    expected_color = {
        "codec": "hevc", "color_range": "tv",
        "color_primaries": "bt2020", "color_space": "bt2020nc",
        "color_transfer": "smpte2084",
    }
    resolved = {}
    for split, rows in selected.items():
        bound = []
        for sparse_row in rows:
            video_id = sparse_row["video_id"]
            selection_row = selection_rows.get(video_id)
            receipt_row = receipt_rows.get(video_id)
            content_id = (receipt_row.get("content_id")
                          if isinstance(receipt_row, dict) else None)
            if (not isinstance(selection_row, dict) or
                    not isinstance(receipt_row, dict) or
                    selection_row.get("split") != split or
                    receipt_row.get("split") != split or
                    selection_row.get("capture_group_id") !=
                    sparse_row["capture_group_id"] or
                    receipt_row.get("capture_group_id") !=
                    sparse_row["capture_group_id"] or
                    selection_row.get("content_id") != content_id or
                    not isinstance(content_id, str) or not content_id):
                raise RuntimeError(f"{video_id}: CHUG provenance differs")
            audit = receipt_row.get("audit")
            download = receipt_row.get("download")
            if not isinstance(audit, dict) or not isinstance(download, dict):
                raise RuntimeError(f"{video_id}: CHUG audit/download is missing")
            mismatches = {
                key: (audit.get(key), value) for key, value in expected_color.items()
                if audit.get(key) != value
            }
            if (mismatches or
                    not str(audit.get("pixel_format", "")).startswith("yuv420p10") or
                    receipt_row.get("orientation") != "Landscape" or
                    int(audit.get("width", 0)) * 9 != int(audit.get("height", 0)) * 16):
                raise RuntimeError(f"{video_id}: native PQ audit differs: {mismatches}")
            candidate = videos_root / f"{video_id}.mp4"
            try:
                video_path = candidate.resolve(strict=True)
                video_path.relative_to(videos_root)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"{video_id}: selected source video is missing") from error
            # This is intentionally the first media operation, and it occurs
            # only after the selected IDs were derived from train/dev manifests.
            source_snapshot = artifact_cache.source_file_snapshot(video_path)
            if (download.get("sha256") != source_snapshot["sha256"] or
                    download.get("bytes") != source_snapshot["bytes"] or
                    download.get("video_id") != video_id):
                raise RuntimeError(f"{video_id}: selected source video changed")
            bound.append({
                **sparse_row,
                "video_path": video_path,
                "source_receipt": receipt_row,
                "source_selection": selection_row,
                "audit": audit,
                "download": download,
                "content_id": content_id,
                "_source_snapshot": source_snapshot,
            })
        resolved[split] = bound
    return resolved, {
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sparse.sha256(selection_path),
        "download_receipt": str(receipt_path.resolve()),
        "download_receipt_sha256": sparse.sha256(receipt_path),
    }


def _probe_source_timing(ffprobe: Path, row):
    video_id = row["video_id"]
    command = [
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_frames", "-show_entries",
        (
            "stream=codec_name,pix_fmt,color_range,color_space,color_transfer,"
            "color_primaries,width,height,avg_frame_rate,r_frame_rate,time_base,"
            "start_time,duration,nb_frames:frame=best_effort_timestamp,"
            "best_effort_timestamp_time,key_frame,pict_type"
        ),
        "-of", "json", str(row["video_path"]),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=180
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{video_id}: cannot probe source timing") from error
    streams = payload.get("streams")
    frames = payload.get("frames")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(frames, list):
        raise RuntimeError(f"{video_id}: FFprobe timing payload is incomplete")
    stream = streams[0]
    audit = row["audit"]
    expected = {
        "codec_name": "hevc", "color_range": "tv",
        "color_space": "bt2020nc", "color_transfer": "smpte2084",
        "color_primaries": "bt2020", "width": int(audit["width"]),
        "height": int(audit["height"]),
    }
    mismatches = {key: (stream.get(key), value) for key, value in expected.items()
                  if stream.get(key) != value}
    if (mismatches or
            not str(stream.get("pix_fmt", "")).startswith("yuv420p10")):
        raise RuntimeError(f"{video_id}: probed source contract differs: {mismatches}")
    frame_rate = _rational(stream.get("avg_frame_rate"), "average frame rate")
    nominal_rate = _rational(stream.get("r_frame_rate"), "nominal frame rate")
    time_base = _rational(stream.get("time_base"), "stream time base")
    if (abs(float(frame_rate) - row["sparse_source_frame_rate"]) > 1e-6 or
            abs(float(frame_rate) - float(row["source_receipt"][
                "probed_frame_rate"])) > 1e-6):
        raise RuntimeError(f"{video_id}: source frame rate differs from provenance")
    if len(frames) != row["source_frame_count"]:
        raise RuntimeError(
            f"{video_id}: FFprobe returned {len(frames)} frames; "
            f"expected {row['source_frame_count']}"
        )
    stream_nb_frames = stream.get("nb_frames")
    if stream_nb_frames not in (None, "N/A") and int(stream_nb_frames) != len(frames):
        raise RuntimeError(f"{video_id}: stream frame count differs")

    ticks = []
    timing_rows = []
    for frame_id, frame in enumerate(frames):
        try:
            timestamp_ticks = int(frame["best_effort_timestamp"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{video_id}: frame {frame_id} lacks a PTS") from error
        if timestamp_ticks < 0 or (ticks and timestamp_ticks <= ticks[-1]):
            raise RuntimeError(f"{video_id}: source PTS is negative/nonmonotonic")
        ticks.append(timestamp_ticks)
        timestamp = timestamp_ticks * time_base
        timing_rows.append({
            "frame": frame_id,
            "timestamp_ticks": timestamp_ticks,
            "timestamp_seconds_numerator": timestamp.numerator,
            "timestamp_seconds_denominator": timestamp.denominator,
            "timestamp_seconds": float(timestamp),
            "ffprobe_timestamp_seconds": frame.get("best_effort_timestamp_time"),
            "key_frame": bool(frame.get("key_frame")),
            "picture_type": frame.get("pict_type"),
        })
    deltas = [right - left for left, right in zip(ticks, ticks[1:])]
    nominal_duration = Fraction(1, 1) / frame_rate
    nominal_duration_ticks = nominal_duration / time_base
    if nominal_duration_ticks.denominator != 1:
        raise RuntimeError(f"{video_id}: nominal cadence is not integral in time base")
    final_delta = int(nominal_duration_ticks)
    durations = deltas + [final_delta]
    for timing_row, duration_ticks in zip(timing_rows, durations):
        duration = duration_ticks * time_base
        timing_row.update({
            "duration_ticks": duration_ticks,
            "duration_seconds_numerator": duration.numerator,
            "duration_seconds_denominator": duration.denominator,
            "duration_seconds": float(duration),
        })
    timing = {
        "schema": TIMING_SCHEMA,
        "contract": TIMING_CONTRACT,
        "video_id": video_id,
        "source_video_sha256": row["download"]["sha256"],
        "frame_count": len(timing_rows),
        "source_frame_rate": _fraction_record(frame_rate),
        "nominal_frame_rate": _fraction_record(nominal_rate),
        "time_base": _fraction_record(time_base),
        "stream_start_time_seconds": stream.get("start_time"),
        "stream_duration_seconds": stream.get("duration"),
        "constant_frame_rate": len(set(durations)) == 1,
        "unique_frame_duration_ticks": sorted(set(durations)),
        "timestamp_source": "ffprobe-best-effort-timestamp-in-stream-time-base",
        "frames": timing_rows,
    }
    timing["content_sha256"] = native_hdr_capture.canonical_sha256({
        key: value for key, value in timing.items() if key != "content_sha256"
    })
    return timing


def _clip_name(video_id: str) -> str:
    return f"chug_pq_full_{video_id}"


def _result(row, status: str):
    timing = row["timing"]
    return {
        "clip": _clip_name(row["video_id"]),
        "status": status,
        "frames": row["source_frame_count"],
        "source_frames": row["source_frame_count"],
        "label_frames": len(row["label_frame_ids"]),
        "label_frame_ids": row["label_frame_ids"],
        "split": row["split"],
        "capture_group_id": row["capture_group_id"],
        "video_id": row["video_id"],
        "source_frame_rate": timing["source_frame_rate"]["decimal"],
        "source_frame_rate_rational": timing["source_frame_rate"]["rational"],
        "source_time_base_rational": timing["time_base"]["rational"],
        "source_timing_content_sha256": timing["content_sha256"],
        "source_cut_candidate_count": row.get("source_cut_candidate_count", 0),
    }


def _semantic_content_hash(payload, label: str) -> str:
    expected = payload.get("content_sha256")
    semantic = {
        key: value for key, value in payload.items()
        if key != "content_sha256"
    }
    actual = native_hdr_capture.canonical_sha256(semantic)
    if not isinstance(expected, str) or expected != actual:
        raise RuntimeError(f"{label} semantic content hash differs")
    return expected


def _validate_clip_sidecar_bindings(
        *, metadata, labels, timing, cuts, frame_manifest,
        timing_path: Path, cut_path: Path, split: str, production_id: str,
        capture_group_id: str, video_id: str, content_id,
        source_video_bytes: int, source_video_sha256: str,
        source_frame_count: int, label_frame_ids,
        source_timing_content_sha256: str, conversion_hash: str,
        cut_threshold: float, expected_cut_candidate_count=None):
    """Bind one clip's sidecars to its current source and publication row."""

    timing_content_hash = _semantic_content_hash(timing, "source timing")
    cut_content_hash = _semantic_content_hash(cuts, "source cut evidence")
    timing_file_hash = sparse.sha256(timing_path)
    cut_file_hash = sparse.sha256(cut_path)
    source_video = frame_manifest.get("source_video")
    conversion = frame_manifest.get("conversion")
    cut_rows = cuts.get("frames")
    if not isinstance(cut_rows, list) or len(cut_rows) != source_frame_count:
        raise RuntimeError("source cut evidence frame coverage differs")
    for frame_id, cut_row in enumerate(cut_rows):
        if (not isinstance(cut_row, dict) or
                cut_row.get("frame") != frame_id or
                type(cut_row.get("cut_candidate")) is not bool):
            raise RuntimeError("source cut evidence row identity differs")
    cut_candidate_count = sum(
        1 for cut_row in cut_rows if cut_row["cut_candidate"]
    )
    declared_cut_candidate_count = cuts.get("cut_candidate_count")
    threshold = cuts.get("threshold")
    threshold_matches = (
        isinstance(threshold, (int, float)) and
        not isinstance(threshold, bool) and
        math.isfinite(float(threshold)) and
        math.isclose(
            float(threshold), float(cut_threshold),
            rel_tol=0.0, abs_tol=1e-12,
        )
    )
    if (type(declared_cut_candidate_count) is not int or
            declared_cut_candidate_count != cut_candidate_count or
            (expected_cut_candidate_count is not None and
             (type(expected_cut_candidate_count) is not int or
              cut_candidate_count != expected_cut_candidate_count))):
        raise RuntimeError("source cut evidence candidate count differs")
    if not threshold_matches:
        raise RuntimeError("source cut evidence threshold differs")

    expected_source_video = {
        "video_id": video_id,
        "split": split,
        "capture_group_id": capture_group_id,
        "content_id": content_id,
        "bytes": source_video_bytes,
        "sha256": source_video_sha256,
        "source_frame_count": source_frame_count,
        "source_timing_sha256": timing_file_hash,
        "source_timing_content_sha256": source_timing_content_sha256,
        "source_cut_evidence_sha256": cut_file_hash,
        "source_cut_evidence_content_sha256": cut_content_hash,
    }
    metadata_expected = {
        "production_id": production_id,
        "split": split,
        "capture_group_id": capture_group_id,
        "source_video_id": video_id,
        "frame_count": source_frame_count,
        "source_timing_sha256": timing_file_hash,
        "source_timing_content_sha256": source_timing_content_sha256,
        "source_cut_evidence_sha256": cut_file_hash,
        "source_cut_evidence_content_sha256": cut_content_hash,
    }
    if (metadata.get("preparation_contract") != PREPARATION_CONTRACT or
            metadata.get("full_cadence_contract") !=
            FULL_CADENCE_CONTRACT or
            any(metadata.get(key) != value
                for key, value in metadata_expected.items()) or
            metadata.get("curated_diagnostic_label_frame_ids") !=
            label_frame_ids):
        raise RuntimeError("full-cadence metadata identity differs")
    if labels != {"schema": 1, "frame_ids": label_frame_ids}:
        raise RuntimeError("full-cadence label-frame identity differs")
    if (timing.get("schema") != TIMING_SCHEMA or
            timing.get("contract") != TIMING_CONTRACT or
            timing.get("video_id") != video_id or
            timing.get("source_video_sha256") != source_video_sha256 or
            timing.get("frame_count") != source_frame_count or
            timing_content_hash != source_timing_content_sha256):
        raise RuntimeError("source timing identity differs")
    if (cuts.get("schema") != CUT_SCHEMA or
            cuts.get("contract") != CUT_CONTRACT or
            cuts.get("video_id") != video_id or
            cuts.get("source_video_sha256") != source_video_sha256 or
            cuts.get("frame_count") != source_frame_count):
        raise RuntimeError("source cut evidence identity differs")
    if (not isinstance(source_video, dict) or
            any(source_video.get(key) != value
                for key, value in expected_source_video.items())):
        raise RuntimeError("source-video sidecar identity differs")
    if (not isinstance(conversion, dict) or
            conversion.get("contract_sha256") != conversion_hash):
        raise RuntimeError("capture conversion identity differs")
    return {
        "timing_content_sha256": timing_content_hash,
        "cut_content_sha256": cut_content_hash,
        "cut_candidate_count": cut_candidate_count,
    }


def _reuse_clip(destination: Path, row, conversion_hash: str,
                cut_threshold: float):
    try:
        authentication = native_hdr_capture.validate_clip(destination, full=False)
        metadata = sparse.read_json(destination / "meta.json", "full-cadence metadata")
        labels = sparse.read_json(
            destination / "label_frames.json", "full-cadence label frames"
        )
        timing_path = destination / "source_timing.json"
        cut_path = destination / "source_cut_evidence.json"
        timing = sparse.read_json(timing_path, "full-cadence timing")
        cuts = sparse.read_json(cut_path, "full-cadence cut evidence")
        frame_manifest = sparse.read_json(
            destination / native_hdr_capture.MANIFEST_NAME,
            "native-HDR frame manifest",
        )
        bindings = _validate_clip_sidecar_bindings(
            metadata=metadata, labels=labels, timing=timing, cuts=cuts,
            frame_manifest=frame_manifest, timing_path=timing_path,
            cut_path=cut_path, split=row["split"],
            production_id=f"{PRODUCTION_PREFIX}_{row['split']}",
            capture_group_id=row["capture_group_id"],
            video_id=row["video_id"], content_id=row["content_id"],
            source_video_bytes=row["download"]["bytes"],
            source_video_sha256=row["download"]["sha256"],
            source_frame_count=row["source_frame_count"],
            label_frame_ids=row["label_frame_ids"],
            source_timing_content_sha256=row["timing"]["content_sha256"],
            conversion_hash=conversion_hash, cut_threshold=cut_threshold,
        )
        valid = (
            authentication["frame_count"] == row["source_frame_count"] and
            timing == row["timing"]
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError(
            f"existing full-cadence clip is stale; refusing to replace it: {destination}"
        )
    row["source_cut_candidate_count"] = bindings["cut_candidate_count"]
    return _result(row, "reused")


def _raw_timing_payload(timing):
    """Path/name-independent timing needed to rebuild publication sidecars."""
    return raw_native.timing_payload(timing)


def _final_code_paths():
    """Files that can change cheap labels, sidecars, or publication layout."""
    return {
        "full_cadence_preparer": Path(__file__).resolve(),
        "native_hdr_capture": Path(native_hdr_capture.__file__).resolve(),
        "sparse_preparer": Path(sparse.__file__).resolve(),
        "native_raw_contract": Path(raw_native.__file__).resolve(),
        "artifact_cache": Path(artifact_cache.__file__).resolve(),
    }


def _raw_native_identity(row, conversion_hash, width, height, *,
                         code_identity, runtime_identity_value=None):
    return raw_native.identity(
        row=row, conversion_hash=conversion_hash,
        width=width, height=height,
        cut_analysis_width=CUT_ANALYSIS_WIDTH,
        cut_analysis_height=CUT_ANALYSIS_HEIGHT,
        code_identity=code_identity,
        runtime_identity_value=runtime_identity_value,
    )


def _raw_native_rows(root):
    root = Path(root).resolve(strict=True)
    return [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": artifact_cache.sha256_file(path),
    } for path in sorted(root.rglob("*")) if path.is_file()]


def _raw_cache_summary(identity, receipt, status):
    rows = receipt["files"]
    return {
        "enabled": True,
        "contract": RAW_NATIVE_CONTRACT,
        "key_sha256": artifact_cache.DirectoryArtifactCache.key(identity),
        "status": status,
        "payload_bytes": sum(row["bytes"] for row in rows),
        "file_count": len(rows),
    }


def _disabled_raw_cache_summary():
    return {
        "enabled": False,
        "contract": RAW_NATIVE_CONTRACT,
        "key_sha256": None,
        "status": "disabled",
        "payload_bytes": 0,
        "file_count": 0,
    }


def _unseeded_raw_cache_summary(identity):
    return {
        "enabled": True,
        "contract": RAW_NATIVE_CONTRACT,
        "key_sha256": artifact_cache.DirectoryArtifactCache.key(identity),
        "status": "unseeded-existing",
        "payload_bytes": 0,
        "file_count": 0,
    }


def _observe_cache(observer, summary):
    if observer is not None:
        observer(dict(summary))


def _aggregate_cache_events(events):
    counts = {}
    by_key = {}
    for event in events:
        status = event["status"]
        counts[status] = counts.get(status, 0) + 1
        key = event.get("key_sha256")
        if key is not None:
            by_key[key] = event
    return {
        "enabled": any(event["enabled"] for event in events),
        "contract": RAW_NATIVE_CONTRACT,
        "clip_count": len(events),
        "status_counts": counts,
        "unique_key_count": len(by_key),
        "unique_payload_bytes": sum(
            event["payload_bytes"] for event in by_key.values()
        ),
    }


def _verify_row_source(row):
    snapshot = row.get("_source_snapshot")
    if snapshot is not None:
        artifact_cache.verify_source_file_snapshot(
            row["video_path"], snapshot
        )


def _link_or_copy(source, destination):
    """Avoid a third transient full-frame copy while seeding the raw CAS."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _validate_raw_native_packet(root: Path, identity, *,
                                authenticated_files=None):
    root = Path(root).resolve(strict=True)
    path = root / RAW_NATIVE_MANIFEST
    try:
        manifest_bytes = path.read_bytes()
        packet = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read raw native-HDR preprocessing packet") from error
    preprocessing = identity["preprocessing"]
    source = identity["source"]
    selection = identity["selection"]
    rows = packet.get("frames") if isinstance(packet, dict) else None
    if (not isinstance(packet, dict) or
            packet.get("schema") != RAW_NATIVE_SCHEMA or
            packet.get("contract") != RAW_NATIVE_CONTRACT or
            packet.get("identity_sha256") !=
            artifact_cache.DirectoryArtifactCache.key(identity) or
            packet.get("source") != source or
            packet.get("split") != selection["split"] or
            packet.get("source_frame_count") !=
            selection["source_frame_count"] or
            packet.get("timing") != _raw_timing_payload(packet.get("timing", {})) or
            native_hdr_capture.canonical_sha256(packet.get("timing")) !=
            selection["timing_sha256"] or
            packet.get("width") != preprocessing["width"] or
            packet.get("height") != preprocessing["height"] or
            packet.get("conversion_contract_sha256") !=
            preprocessing["conversion_contract_sha256"] or
            packet.get("cut_analysis") != preprocessing["cut_analysis"] or
            not isinstance(rows, list) or
            len(rows) != selection["source_frame_count"]):
        raise RuntimeError("raw native-HDR preprocessing identity differs")
    expected_files = {RAW_NATIVE_MANIFEST}
    expected_size = packet["width"] * packet["height"] * SCRGB_BYTES_PER_PIXEL
    for frame_id, row in enumerate(rows):
        suffix = f"{frame_id:05d}"
        expected_model = (
            f"{native_hdr_capture.MODEL_SOURCE_DIRECTORY}/frame_{suffix}.scrgb16"
        )
        expected_preview = f"frame_{suffix}.png"
        delta = row.get("preview_mean_absolute_delta") \
            if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("frame") != frame_id or
                row.get("path") != expected_model or
                row.get("preview") != expected_preview or
                row.get("size") != expected_size or
                not isinstance(row.get("sha256"), str) or
                len(row["sha256"]) != 64 or
                not isinstance(row.get("preview_sha256"), str) or
                len(row["preview_sha256"]) != 64 or
                not isinstance(row.get("timestamp_seconds"), (int, float)) or
                isinstance(row.get("timestamp_seconds"), bool) or
                not math.isfinite(float(row["timestamp_seconds"])) or
                not isinstance(row.get("stats"), dict) or
                (frame_id == 0 and delta is not None) or
                (frame_id > 0 and (
                    not isinstance(delta, (int, float)) or isinstance(delta, bool) or
                    not math.isfinite(float(delta)) or delta < 0.0))):
            raise RuntimeError("raw native-HDR preprocessing frame row differs")
        model_path = root / expected_model
        preview_path = root / expected_preview
        if (not model_path.is_file() or model_path.is_symlink() or
                model_path.stat().st_size != expected_size or
                not preview_path.is_file() or preview_path.is_symlink() or
                preview_path.stat().st_size <= 0):
            raise RuntimeError("raw native-HDR preprocessing frame is missing")
        expected_files.update((expected_model, expected_preview))
    observed_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*") if item.is_file()
    }
    if observed_files != expected_files:
        raise RuntimeError("raw native-HDR preprocessing packet files differ")
    if authenticated_files is not None:
        receipt = {
            item["path"]: item for item in authenticated_files
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        if set(receipt) != expected_files:
            raise RuntimeError("raw native-HDR cache receipt coverage differs")
        manifest_row = receipt[RAW_NATIVE_MANIFEST]
        if (manifest_row.get("bytes") != len(manifest_bytes) or
                manifest_row.get("sha256") !=
                hashlib.sha256(manifest_bytes).hexdigest()):
            raise RuntimeError("raw native-HDR cache manifest receipt differs")
        for row in rows:
            model = receipt[row["path"]]
            preview = receipt[row["preview"]]
            if (model.get("bytes") != row["size"] or
                    model.get("sha256") != row["sha256"] or
                    preview.get("sha256") != row["preview_sha256"]):
                raise RuntimeError(
                    "raw native-HDR cache receipt differs from frame manifest"
                )
    return packet


def _publish_raw_native_from_clip(cache, identity, destination, row,
                                  conversion_hash, split_root, code_identity):
    validated = cache.validated_payload_receipt(identity)
    if validated is not None:
        payload, receipt = validated
        _validate_raw_native_packet(
            payload, identity, authenticated_files=receipt["files"]
        )
        return payload, _raw_cache_summary(identity, receipt, "hit")
    frame_manifest = sparse.read_json(
        destination / native_hdr_capture.MANIFEST_NAME,
        "native-HDR frame manifest",
    )
    cuts = sparse.read_json(
        destination / "source_cut_evidence.json", "source cut evidence"
    )
    cut_by_id = {
        item["frame"]: item for item in cuts.get("frames", ())
        if isinstance(item, dict) and type(item.get("frame")) is int
    }
    records = frame_manifest.get("frames")
    if (not isinstance(records, list) or
            len(records) != row["source_frame_count"] or
            len(cut_by_id) != len(records)):
        raise RuntimeError("cannot derive complete raw native-HDR packet")
    split_root.mkdir(parents=True, exist_ok=True)
    staging = artifact_cache.inheriting_temporary_directory(
        split_root, ".raw-native-hdr-"
    )
    try:
        shutil.copytree(
            destination / native_hdr_capture.MODEL_SOURCE_DIRECTORY,
            staging / native_hdr_capture.MODEL_SOURCE_DIRECTORY,
            copy_function=_link_or_copy,
        )
        raw_rows = []
        for record in records:
            preview = record["preview"]
            _link_or_copy(destination / preview, staging / preview)
            raw_rows.append({
                "frame": record["frame"],
                "path": record["path"],
                "size": record["size"],
                "sha256": record["sha256"],
                "preview": preview,
                "preview_sha256": record["preview_sha256"],
                "timestamp_seconds": record["timestamp_seconds"],
                "stats": record["stats"],
                "preview_mean_absolute_delta":
                    cut_by_id[record["frame"]].get(
                        "preview_mean_absolute_delta"
                    ),
            })
        packet = {
            "schema": RAW_NATIVE_SCHEMA,
            "contract": RAW_NATIVE_CONTRACT,
            "identity_sha256": artifact_cache.DirectoryArtifactCache.key(identity),
            "source": identity["source"],
            "split": row["split"],
            "source_frame_count": row["source_frame_count"],
            "timing": _raw_timing_payload(row["timing"]),
            "width": identity["preprocessing"]["width"],
            "height": identity["preprocessing"]["height"],
            "conversion_contract_sha256": conversion_hash,
            "cut_analysis": identity["preprocessing"]["cut_analysis"],
            "frames": raw_rows,
        }
        sparse.write_json_atomic(staging / RAW_NATIVE_MANIFEST, packet)
        staging_rows = _raw_native_rows(staging)
        _validate_raw_native_packet(
            staging, identity, authenticated_files=staging_rows
        )
        raw_native.verify_code_identity(code_identity)
        cache.publish(identity, staging)
        validated = cache.validated_payload_receipt(identity)
        if validated is None:
            raise RuntimeError("raw native-HDR cache publication disappeared")
        payload, receipt = validated
        if receipt["files"] != staging_rows:
            raise RuntimeError(
                "raw native-HDR cache key produced different preprocessing bytes"
            )
        _validate_raw_native_packet(
            payload, identity, authenticated_files=receipt["files"]
        )
        return payload, _raw_cache_summary(identity, receipt, "published")
    finally:
        sparse.remove_path(staging)


def _materialize_native_from_raw(payload, identity, destination, row,
                                 conversion_hash, width, height,
                                 cut_threshold, split_root,
                                 final_code_paths, final_code_identity,
                                 receipt):
    packet = _validate_raw_native_packet(
        payload, identity, authenticated_files=receipt["files"]
    )
    timing = row["timing"]
    if _raw_timing_payload(timing) != packet["timing"]:
        raise RuntimeError("raw native-HDR timing differs from current source probe")
    split_root.mkdir(parents=True, exist_ok=True)
    staging = artifact_cache.inheriting_temporary_directory(
        split_root,
        f".{_clip_name(row['video_id'])}.raw-cache-partial-",
    )
    try:
        model_root = staging / native_hdr_capture.MODEL_SOURCE_DIRECTORY
        shutil.copytree(
            Path(payload) / native_hdr_capture.MODEL_SOURCE_DIRECTORY,
            model_root, copy_function=shutil.copy2, dirs_exist_ok=True,
        )
        records = []
        cut_rows = []
        for raw in packet["frames"]:
            frame_id = raw["frame"]
            preview_path = staging / raw["preview"]
            shutil.copy2(Path(payload) / raw["preview"], preview_path)
            model_path = staging / raw["path"]
            model_stat = model_path.stat()
            records.append({
                "frame": frame_id,
                "path": raw["path"],
                "size": model_stat.st_size,
                "mtime_ns": model_stat.st_mtime_ns,
                "sha256": raw["sha256"],
                "preview": raw["preview"],
                "preview_sha256": raw["preview_sha256"],
                "timestamp_seconds": raw["timestamp_seconds"],
                "stats": raw["stats"],
            })
            timing_row = timing["frames"][frame_id]
            score = raw["preview_mean_absolute_delta"]
            cut_rows.append({
                "frame": frame_id,
                "timestamp_ticks": timing_row["timestamp_ticks"],
                "timestamp_seconds": timing_row["timestamp_seconds"],
                "scene_start": frame_id == 0,
                "preview_mean_absolute_delta": score,
                "cut_candidate": score is not None and score >= cut_threshold,
            })
        timing_path = staging / "source_timing.json"
        sparse.write_json_atomic(timing_path, timing)
        cut_evidence = {
            "schema": CUT_SCHEMA,
            "contract": CUT_CONTRACT,
            "video_id": row["video_id"],
            "source_video_sha256": row["download"]["sha256"],
            "frame_count": len(cut_rows),
            "analysis_input": native_hdr_capture.PREVIEW_ENCODING,
            "analysis_geometry": {
                "width": CUT_ANALYSIS_WIDTH,
                "height": CUT_ANALYSIS_HEIGHT,
                "resize": "PIL-bilinear-grayscale",
            },
            "score": "mean-absolute-delta-of-normalized-preview-luma",
            "threshold": cut_threshold,
            "cut_candidate_count": sum(
                item["cut_candidate"] for item in cut_rows
            ),
            "role": (
                "diagnostic source-boundary candidates only; authoritative hard "
                "cuts come from completed-depth SubjectState runtime evidence"
            ),
            "clip_contract": (
                "one contiguous source video; candidates do not split or drop frames"
            ),
            "frames": cut_rows,
        }
        cut_evidence["content_sha256"] = native_hdr_capture.canonical_sha256({
            key: value for key, value in cut_evidence.items()
            if key != "content_sha256"
        })
        cut_path = staging / "source_cut_evidence.json"
        sparse.write_json_atomic(cut_path, cut_evidence)
        row["source_cut_candidate_count"] = cut_evidence["cut_candidate_count"]

        source_video = {
            "dataset": "CHUG",
            "video_id": row["video_id"],
            "path": str(row["video_path"].resolve()),
            "bytes": row["download"]["bytes"],
            "sha256": row["download"]["sha256"],
            "split": row["split"],
            "capture_group_id": row["capture_group_id"],
            "content_id": row["content_id"],
            "license": "CC BY-NC-SA 4.0",
            "source_frame_count": row["source_frame_count"],
            "source_frame_rate": timing["source_frame_rate"]["decimal"],
            "source_frame_rate_rational": timing["source_frame_rate"]["rational"],
            "source_time_base_rational": timing["time_base"]["rational"],
            "full_cadence_contract": FULL_CADENCE_CONTRACT,
            "source_timing_manifest": "source_timing.json",
            "source_timing_sha256": sparse.sha256(timing_path),
            "source_timing_content_sha256": timing["content_sha256"],
            "source_cut_evidence": "source_cut_evidence.json",
            "source_cut_evidence_sha256": sparse.sha256(cut_path),
            "source_cut_evidence_content_sha256": cut_evidence["content_sha256"],
        }
        conversion_identity = {
            "contract": sparse.CONVERSION_CONTRACT,
            "contract_sha256": conversion_hash,
        }
        semantic_rows = [{
            key: value for key, value in record.items()
            if key not in {"mtime_ns", "stats"}
        } for record in records]
        semantic = {
            "contract": native_hdr_capture.MANIFEST_CONTRACT,
            "capture_encoding": native_hdr_capture.CAPTURE_ENCODING,
            "preview_encoding": native_hdr_capture.PREVIEW_ENCODING,
            "width": width,
            "height": height,
            "row_pitch_bytes": width * SCRGB_BYTES_PER_PIXEL,
            "source_video": source_video,
            "conversion": conversion_identity,
            "frames": semantic_rows,
        }
        frame_manifest = {
            "schema": native_hdr_capture.MANIFEST_SCHEMA,
            "contract": native_hdr_capture.MANIFEST_CONTRACT,
            "capture_encoding": native_hdr_capture.CAPTURE_ENCODING,
            "preview_encoding": native_hdr_capture.PREVIEW_ENCODING,
            "width": width,
            "height": height,
            "row_pitch_bytes": width * SCRGB_BYTES_PER_PIXEL,
            "source_video": source_video,
            "conversion": conversion_identity,
            "frames": records,
            "frame_count": len(records),
            "content_sha256": native_hdr_capture.canonical_sha256(semantic),
        }
        sparse.write_json_atomic(
            staging / native_hdr_capture.MANIFEST_NAME, frame_manifest
        )
        sparse.write_json_atomic(staging / "label_frames.json", {
            "schema": 1,
            "frame_ids": row["label_frame_ids"],
        })
        sparse.write_json_atomic(staging / "meta.json", {
            "name": f"CHUG native PQ full cadence {row['video_id']}",
            "description": (
                "Authenticated complete CHUG BT.2020/PQ source video converted "
                "to FP16 Windows scRGB without cadence subsampling"
            ),
            "dataset": "chug",
            "production_id": f"{PRODUCTION_PREFIX}_{row['split']}",
            "split": row["split"],
            "source_kind": "native-hdr-video",
            "license": "CC BY-NC-SA 4.0",
            "capture_group_id": row["capture_group_id"],
            "source_video_id": row["video_id"],
            "native_hdr": True,
            "required_gt_depth": False,
            "required_gt_flow": False,
            "required_gt_stereo": False,
            "global_policy_weight": 1.0,
            "preparation_contract": PREPARATION_CONTRACT,
            "full_cadence_contract": FULL_CADENCE_CONTRACT,
            "frame_count": len(records),
            "source_frame_rate": timing["source_frame_rate"],
            "source_time_base": timing["time_base"],
            "source_timing": "source_timing.json",
            "source_timing_sha256": sparse.sha256(timing_path),
            "source_timing_content_sha256": timing["content_sha256"],
            "source_cut_evidence": "source_cut_evidence.json",
            "source_cut_evidence_sha256": sparse.sha256(cut_path),
            "source_cut_evidence_content_sha256": cut_evidence["content_sha256"],
            "runtime_cut_authority": (
                "SubjectState runtime-scene evidence emitted by the exact "
                "full-cadence render harness"
            ),
            "curated_diagnostic_label_frame_ids": row["label_frame_ids"],
        })
        native_hdr_capture.validate_clip(staging, full=True)
        artifact_cache.verify_code_identities(
            final_code_paths, final_code_identity
        )
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(
                f"destination appeared during raw-cache materialization: {destination}"
            )
        staging.replace(destination)
        return _result(row, "raw-cache-reused")
    finally:
        sparse.remove_path(staging)


def _prepare_clip(row, split_root: Path, ffmpeg: Path, conversion,
                  conversion_hash: str, width: int, height: int,
                  cut_threshold: float, preprocess_cache=None, *,
                  raw_code_identity=None, final_code_paths=None,
                  final_code_identity=None, raw_runtime_identity=None,
                  cache_observer=None):
    artifact_cache.require_working_split(row.get("split"))
    video_id = row["video_id"]
    destination = split_root / _clip_name(video_id)
    cache = None
    raw_identity = None
    effective_code_identity = raw_code_identity or raw_native.code_identity()
    effective_final_paths = final_code_paths or _final_code_paths()
    effective_final_identity = (
        final_code_identity or artifact_cache.code_identities(
            effective_final_paths
        )
    )
    effective_runtime_identity = (
        raw_runtime_identity or raw_native.runtime_identity(fresh=True)
    )
    cache_summary = _disabled_raw_cache_summary()
    _verify_row_source(row)
    if preprocess_cache is not None:
        cache = artifact_cache.DirectoryArtifactCache(preprocess_cache)
        raw_identity = _raw_native_identity(
            row, conversion_hash, width, height,
            code_identity=effective_code_identity,
            runtime_identity_value=effective_runtime_identity,
        )
    if destination.exists() or destination.is_symlink():
        result = _reuse_clip(
            destination, row, conversion_hash, cut_threshold
        )
        if cache is not None:
            validated = cache.validated_payload_receipt(raw_identity)
            if validated is None:
                # Legacy final clips do not bind the NumPy/Pillow runtime that
                # produced their raw bytes.  They remain valid final datasets,
                # but must not be mislabeled as output of the current runtime.
                cache_summary = _unseeded_raw_cache_summary(raw_identity)
            else:
                payload, receipt = validated
                _validate_raw_native_packet(
                    payload, raw_identity,
                    authenticated_files=receipt["files"],
                )
                cache_summary = _raw_cache_summary(
                    raw_identity, receipt, "hit"
                )
        _observe_cache(cache_observer, cache_summary)
        return result
    if cache is not None:
        validated = cache.validated_payload_receipt(raw_identity)
        if validated is not None:
            raw_payload, receipt = validated
            _validate_raw_native_packet(
                raw_payload, raw_identity,
                authenticated_files=receipt["files"],
            )
            result = _materialize_native_from_raw(
                raw_payload, raw_identity, destination, row,
                conversion_hash, width, height, cut_threshold, split_root,
                effective_final_paths, effective_final_identity, receipt,
            )
            raw_native.verify_code_identity(effective_code_identity)
            cache_summary = _raw_cache_summary(
                raw_identity, receipt, "hit"
            )
            _observe_cache(cache_observer, cache_summary)
            return result
    split_root.mkdir(parents=True, exist_ok=True)
    staging = artifact_cache.inheriting_temporary_directory(
        split_root, f".{_clip_name(video_id)}.partial-"
    )
    try:
        timing = row["timing"]
        timing_path = staging / "source_timing.json"
        sparse.write_json_atomic(timing_path, timing)

        source_width = int(row["audit"]["width"])
        source_height = int(row["audit"]["height"])
        records, cut_rows = raw_native.decode_frames(
            ffmpeg=ffmpeg,
            source=row["video_path"],
            filter_text=conversion["decoder"]["filter"],
            source_width=source_width,
            source_height=source_height,
            width=width,
            height=height,
            expected_frame_count=row["source_frame_count"],
            timing=timing,
            staging=staging,
            video_id=video_id,
            cut_analysis_width=CUT_ANALYSIS_WIDTH,
            cut_analysis_height=CUT_ANALYSIS_HEIGHT,
        )
        _verify_row_source(row)
        for cut_row in cut_rows:
            score = cut_row["preview_mean_absolute_delta"]
            cut_row["cut_candidate"] = (
                score is not None and score >= cut_threshold
            )

        cut_evidence = {
            "schema": CUT_SCHEMA,
            "contract": CUT_CONTRACT,
            "video_id": video_id,
            "source_video_sha256": row["download"]["sha256"],
            "frame_count": len(cut_rows),
            "analysis_input": native_hdr_capture.PREVIEW_ENCODING,
            "analysis_geometry": {
                "width": CUT_ANALYSIS_WIDTH,
                "height": CUT_ANALYSIS_HEIGHT,
                "resize": "PIL-bilinear-grayscale",
            },
            "score": "mean-absolute-delta-of-normalized-preview-luma",
            "threshold": cut_threshold,
            "cut_candidate_count": sum(item["cut_candidate"] for item in cut_rows),
            "role": (
                "diagnostic source-boundary candidates only; authoritative hard "
                "cuts come from completed-depth SubjectState runtime evidence"
            ),
            "clip_contract": (
                "one contiguous source video; candidates do not split or drop frames"
            ),
            "frames": cut_rows,
        }
        cut_evidence["content_sha256"] = native_hdr_capture.canonical_sha256({
            key: value for key, value in cut_evidence.items()
            if key != "content_sha256"
        })
        cut_path = staging / "source_cut_evidence.json"
        sparse.write_json_atomic(cut_path, cut_evidence)
        row["source_cut_candidate_count"] = cut_evidence["cut_candidate_count"]

        source_video = {
            "dataset": "CHUG",
            "video_id": video_id,
            "path": str(row["video_path"].resolve()),
            "bytes": row["download"]["bytes"],
            "sha256": row["download"]["sha256"],
            "split": row["split"],
            "capture_group_id": row["capture_group_id"],
            "content_id": row["content_id"],
            "license": "CC BY-NC-SA 4.0",
            "source_frame_count": row["source_frame_count"],
            "source_frame_rate": timing["source_frame_rate"]["decimal"],
            "source_frame_rate_rational": timing["source_frame_rate"]["rational"],
            "source_time_base_rational": timing["time_base"]["rational"],
            "full_cadence_contract": FULL_CADENCE_CONTRACT,
            "source_timing_manifest": "source_timing.json",
            "source_timing_sha256": sparse.sha256(timing_path),
            "source_timing_content_sha256": timing["content_sha256"],
            "source_cut_evidence": "source_cut_evidence.json",
            "source_cut_evidence_sha256": sparse.sha256(cut_path),
            "source_cut_evidence_content_sha256": cut_evidence["content_sha256"],
        }
        conversion_identity = {
            "contract": sparse.CONVERSION_CONTRACT,
            "contract_sha256": conversion_hash,
        }
        semantic_rows = [{
            key: value for key, value in record.items()
            if key not in {"mtime_ns", "stats"}
        } for record in records]
        semantic = {
            "contract": native_hdr_capture.MANIFEST_CONTRACT,
            "capture_encoding": native_hdr_capture.CAPTURE_ENCODING,
            "preview_encoding": native_hdr_capture.PREVIEW_ENCODING,
            "width": width,
            "height": height,
            "row_pitch_bytes": width * SCRGB_BYTES_PER_PIXEL,
            "source_video": source_video,
            "conversion": conversion_identity,
            "frames": semantic_rows,
        }
        frame_manifest = {
            "schema": native_hdr_capture.MANIFEST_SCHEMA,
            "contract": native_hdr_capture.MANIFEST_CONTRACT,
            "capture_encoding": native_hdr_capture.CAPTURE_ENCODING,
            "preview_encoding": native_hdr_capture.PREVIEW_ENCODING,
            "width": width,
            "height": height,
            "row_pitch_bytes": width * SCRGB_BYTES_PER_PIXEL,
            "source_video": source_video,
            "conversion": conversion_identity,
            "frames": records,
            "frame_count": len(records),
            "content_sha256": native_hdr_capture.canonical_sha256(semantic),
        }
        sparse.write_json_atomic(
            staging / native_hdr_capture.MANIFEST_NAME, frame_manifest
        )
        sparse.write_json_atomic(staging / "label_frames.json", {
            "schema": 1,
            "frame_ids": row["label_frame_ids"],
        })
        sparse.write_json_atomic(staging / "meta.json", {
            "name": f"CHUG native PQ full cadence {video_id}",
            "description": (
                "Authenticated complete CHUG BT.2020/PQ source video converted "
                "to FP16 Windows scRGB without cadence subsampling"
            ),
            "dataset": "chug",
            "production_id": f"{PRODUCTION_PREFIX}_{row['split']}",
            "split": row["split"],
            "source_kind": "native-hdr-video",
            "license": "CC BY-NC-SA 4.0",
            "capture_group_id": row["capture_group_id"],
            "source_video_id": video_id,
            "native_hdr": True,
            "required_gt_depth": False,
            "required_gt_flow": False,
            "required_gt_stereo": False,
            "global_policy_weight": 1.0,
            "preparation_contract": PREPARATION_CONTRACT,
            "full_cadence_contract": FULL_CADENCE_CONTRACT,
            "frame_count": len(records),
            "source_frame_rate": timing["source_frame_rate"],
            "source_time_base": timing["time_base"],
            "source_timing": "source_timing.json",
            "source_timing_sha256": sparse.sha256(timing_path),
            "source_timing_content_sha256": timing["content_sha256"],
            "source_cut_evidence": "source_cut_evidence.json",
            "source_cut_evidence_sha256": sparse.sha256(cut_path),
            "source_cut_evidence_content_sha256": cut_evidence["content_sha256"],
            "runtime_cut_authority": (
                "SubjectState runtime-scene evidence emitted by the exact "
                "full-cadence render harness"
            ),
            "curated_diagnostic_label_frame_ids": row["label_frame_ids"],
        })
        native_hdr_capture.validate_clip(staging, full=True)
        artifact_cache.verify_code_identities(
            effective_final_paths, effective_final_identity
        )
        raw_native.verify_runtime_identity(effective_runtime_identity)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"destination appeared during preparation: {destination}")
        staging.replace(destination)
        if cache is not None:
            raw_native.verify_code_identity(effective_code_identity)
            _payload, cache_summary = _publish_raw_native_from_clip(
                cache, raw_identity, destination, row, conversion_hash,
                split_root, effective_code_identity,
            )
        _observe_cache(cache_observer, cache_summary)
        return _result(row, "prepared")
    finally:
        sparse.remove_path(staging)


def _dataset_manifest(split: str, results, source_provenance,
                      sparse_provenance, conversion_hash: str,
                      cut_threshold: float):
    sequences = sorted(results, key=lambda item: item["clip"])
    expected_videos = (TRAINING_SOURCE_VIDEOS if split == "training" else
                       DEVELOPMENT_SOURCE_VIDEOS)
    if (len(sequences) != expected_videos or
            len({item["video_id"] for item in sequences}) != expected_videos or
            len({item["capture_group_id"] for item in sequences}) != expected_videos):
        raise RuntimeError(f"full-cadence {split} sequence identity differs")
    frame_count = sum(item["frames"] for item in sequences)
    label_count = sum(item["label_frames"] for item in sequences)
    if (frame_count != EXPECTED_SOURCE_FRAMES[split] or
            label_count != EXPECTED_LABEL_FRAMES[split]):
        raise RuntimeError(f"full-cadence {split} cardinality differs")
    return {
        "schema": 2,
        "dataset": DATASET_NAME,
        "domain": "native_hdr_cinematic",
        "production_id": f"{PRODUCTION_PREFIX}_{split}",
        "source_kind": "native-hdr-video",
        "split": split,
        "source_split": split,
        "projection": "rectilinear",
        "policy_role": "cinematic_training",
        "global_policy_weight": 1.0,
        "license": "CC BY-NC-SA 4.0",
        "preparation_contract": PREPARATION_CONTRACT,
        "full_cadence_contract": FULL_CADENCE_CONTRACT,
        "timing_contract": TIMING_CONTRACT,
        "source_cut_evidence_contract": CUT_CONTRACT,
        "source_cut_candidate_threshold": cut_threshold,
        "runtime_cut_authority": (
            "completed-depth SubjectState runtime-scene evidence; source preview "
            "candidates are diagnostic and never split clips"
        ),
        "conversion_contract_sha256": conversion_hash,
        "sequences": [{
            "clip": item["clip"],
            "frames": item["frames"],
            "source_frames": item["source_frames"],
            "label_frames": item["label_frames"],
            "label_frame_ids": item["label_frame_ids"],
            "split": split,
            "capture_group_id": item["capture_group_id"],
            "video_id": item["video_id"],
            "source_frame_rate": item["source_frame_rate"],
            "source_frame_rate_rational": item["source_frame_rate_rational"],
            "source_time_base_rational": item["source_time_base_rational"],
            "source_timing_content_sha256": item[
                "source_timing_content_sha256"
            ],
            "source_cut_candidate_count": item["source_cut_candidate_count"],
            "temporal_contract": (
                "one-source-video-one-contiguous-clip-no-cadence-subsampling"
            ),
        } for item in sequences],
        "frame_count": frame_count,
        "source_frame_count": frame_count,
        "source_video_count": expected_videos,
        "label_frame_count": label_count,
        "source_cut_candidate_count": sum(
            item["source_cut_candidate_count"] for item in sequences
        ),
        "source_frame_rates": sorted({
            item["source_frame_rate_rational"] for item in sequences
        }),
        "source_provenance": {
            **source_provenance,
            **sparse_provenance,
        },
    }


def _build_or_reuse_clip_hash_manifest(split_root: Path, workers: int):
    """Preserve an authenticated publication identity on no-op reruns."""

    path = split_root / "clip_hash_manifest.json"
    if path.is_file():
        import sys
        sbsbench = Path(__file__).resolve().parents[1] / "sbsbench"
        sys.path.insert(0, str(sbsbench))
        import build_clip_hash_manifest as clip_hashes  # noqa: E402
        manifest = clip_hashes.load_manifest(path)
        clips = sorted(
            item.name for item in split_root.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        )
        if set(manifest.get("clips", {})) != set(clips):
            raise RuntimeError("existing clip hash manifest coverage differs")
        clip_hashes.verify_selected_clips(
            path, split_root, clips, full=False
        )
        return {
            "path": str(path.resolve()),
            "sha256": sparse.sha256(path),
            "semantic_content_sha256": manifest[
                clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
            ],
        }
    return sparse._build_clip_hash_manifest(split_root, workers)


def prepare(args):
    raw_code_identity = raw_native.code_identity()
    raw_runtime_identity = (
        raw_native.runtime_identity(fresh=True)
        if getattr(args, "preprocess_cache", None) is not None else None
    )
    final_code_paths = _final_code_paths()
    final_code_identity = artifact_cache.code_identities(final_code_paths)
    chug_root = args.chug_root.resolve(strict=True)
    source_bootstrap = args.source_bootstrap_manifest.resolve(strict=True)
    source_root = source_bootstrap.parent.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    if getattr(args, "preprocess_cache", None) is not None:
        artifact_cache.require_disjoint_roots(
            args.preprocess_cache, output_root, chug_root, source_root
        )
    for protected, label in ((chug_root, "CHUG masters"),
                             (source_root, "sparse CHUG production")):
        if output_root == protected or output_root.is_relative_to(protected):
            raise RuntimeError(f"full-cadence output must not overlap {label}")

    selected, sparse_provenance = _load_sparse_selection(source_bootstrap)
    selected, source_provenance = _selected_source_rows(chug_root, selected)
    ffprobe = sparse._ffprobe_for_ffmpeg(ffmpeg)
    ffmpeg_identity = sparse.ffmpeg_version(ffmpeg)
    ffprobe_identity = sparse.ffprobe_version(ffprobe)
    conversion = sparse._conversion_contract(
        ffmpeg, ffmpeg_identity, ffprobe, ffprobe_identity,
        args.width, args.height,
    )
    conversion_hash = native_hdr_capture.canonical_sha256(conversion)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_probe_source_timing, ffprobe, row): row
            for values in selected.values() for row in values
        }
        for future in as_completed(futures):
            row = futures[future]
            row["timing"] = future.result()

    retention = {
        "contract": FULL_CADENCE_CONTRACT,
        "stored_identity": "one-contiguous-clip-per-selected-source-video",
        "no_cadence_subsampling": True,
        "splits": {
            split: {
                "source_videos": len(rows),
                "source_frames": sum(row["source_frame_count"] for row in rows),
                "retained_frames": sum(row["source_frame_count"] for row in rows),
                "label_frames": sum(len(row["label_frame_ids"]) for row in rows),
                "frame_rates": sorted({
                    row["timing"]["source_frame_rate"]["rational"] for row in rows
                }),
            }
            for split, rows in selected.items()
        },
    }
    retention["total"] = {
        "source_frames": sum(value["source_frames"]
                             for value in retention["splits"].values()),
        "retained_frames": sum(value["retained_frames"]
                               for value in retention["splits"].values()),
        "raw_scrgb16_bytes": sum(value["retained_frames"]
                                 for value in retention["splits"].values()) *
        args.width * args.height * SCRGB_BYTES_PER_PIXEL,
    }
    if args.dry_run:
        return {
            "schema": PREPARATION_SCHEMA,
            "contract": PREPARATION_CONTRACT,
            "dry_run": True,
            "output_root": str(output_root),
            "conversion_contract": conversion,
            "conversion_contract_sha256": conversion_hash,
            "source_provenance": source_provenance,
            "sparse_provenance": sparse_provenance,
            "selected": {
                split: [{
                    "video_id": row["video_id"],
                    "capture_group_id": row["capture_group_id"],
                    "source_frame_count": row["source_frame_count"],
                    "source_frame_rate": row["timing"]["source_frame_rate"],
                    "time_base": row["timing"]["time_base"],
                    "first_timestamp": row["timing"]["frames"][0],
                    "last_timestamp": row["timing"]["frames"][-1],
                    "label_frame_ids": row["label_frame_ids"],
                } for row in rows]
                for split, rows in selected.items()
            },
            "retention": retention,
            "sealed_test_policy": (
                "CHUG test masters were not resolved, stat'ed, hashed, probed, "
                "decoded, or copied"
            ),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    conversion_path = output_root / "conversion_contract.json"
    if conversion_path.exists():
        existing = sparse.read_json(conversion_path, "conversion contract")
        if existing != conversion:
            raise RuntimeError(
                f"existing conversion contract differs; refusing replacement: {conversion_path}"
            )
    else:
        sparse.write_json_atomic(conversion_path, conversion)

    prepared = {}
    cache_events = []
    for split in SPLITS:
        split_root = output_root / split
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _prepare_clip, row, split_root, ffmpeg, conversion,
                    conversion_hash, args.width, args.height, args.cut_threshold,
                    getattr(args, "preprocess_cache", None),
                    raw_code_identity=raw_code_identity,
                    raw_runtime_identity=raw_runtime_identity,
                    final_code_paths=final_code_paths,
                    final_code_identity=final_code_identity,
                    cache_observer=cache_events.append,
                ): row["video_id"]
                for row in selected[split]
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[{split}] {result['clip']} {result['status']} "
                    f"({result['frames']} frames @ "
                    f"{result['source_frame_rate_rational']} fps)",
                    flush=True,
                )
        manifest = _dataset_manifest(
            split, results, source_provenance, sparse_provenance,
            conversion_hash, args.cut_threshold,
        )
        manifest_path = split_root / "dataset_manifest.json"
        if manifest_path.exists():
            existing = sparse.read_json(manifest_path, f"{split} dataset manifest")
            if existing != manifest:
                raise RuntimeError(
                    f"existing dataset manifest differs; refusing replacement: {manifest_path}"
                )
        else:
            sparse.write_json_atomic(manifest_path, manifest)
        clip_hash = _build_or_reuse_clip_hash_manifest(
            split_root, args.workers
        )
        prepared[split] = {
            "root": str(split_root.resolve()),
            "dataset_manifest": str(manifest_path.resolve()),
            "dataset_manifest_sha256": sparse.sha256(manifest_path),
            "clip_hash_manifest": clip_hash,
            "clips": [item["clip"] for item in sorted(
                results, key=lambda value: value["clip"]
            )],
            "context_frame_count": manifest["frame_count"],
            "source_context_frame_count": manifest["source_frame_count"],
            "label_frame_count": manifest["label_frame_count"],
            "capture_group_ids": sorted({
                item["capture_group_id"] for item in results
            }),
        }
    cache_summary = _aggregate_cache_events(cache_events)
    print(
        "[preprocess-cache] " + json.dumps(cache_summary, sort_keys=True),
        flush=True,
    )
    payload = {
        "schema": PREPARATION_SCHEMA,
        "contract": PREPARATION_CONTRACT,
        "dataset": DATASET_NAME,
        "output_root": str(output_root),
        "source_provenance": source_provenance,
        "sparse_provenance": sparse_provenance,
        "conversion_contract": str(conversion_path.resolve()),
        "conversion_contract_sha256": conversion_hash,
        "full_cadence_contract": FULL_CADENCE_CONTRACT,
        "timing_contract": TIMING_CONTRACT,
        "source_cut_evidence_contract": CUT_CONTRACT,
        "sealed_test_policy": (
            "CHUG test masters were not resolved, stat'ed, hashed, probed, "
            "decoded, or copied"
        ),
        "retention": retention,
        "datasets": prepared,
        "summary": {
            "training_clips": len(prepared["training"]["clips"]),
            "development_clips": len(prepared["development"]["clips"]),
            "training_context_frames": prepared["training"][
                "context_frame_count"
            ],
            "development_context_frames": prepared["development"][
                "context_frame_count"
            ],
            "training_diagnostic_labels": prepared["training"][
                "label_frame_count"
            ],
            "development_diagnostic_labels": prepared["development"][
                "label_frame_count"
            ],
        },
    }
    manifest_path = output_root / BOOTSTRAP_MANIFEST
    if manifest_path.exists():
        existing = sparse.read_json(manifest_path, "full-cadence bootstrap manifest")
        if existing != payload:
            raise RuntimeError(
                f"existing bootstrap differs; refusing replacement: {manifest_path}"
            )
    else:
        sparse.write_json_atomic(manifest_path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chug-root", required=True, type=Path)
    parser.add_argument("--source-bootstrap-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cut-threshold", type=float, default=CUT_THRESHOLD)
    parser.add_argument(
        "--preprocess-cache", type=Path,
        help=(
            "optional authenticated content-addressed cache for converted "
            "training/development clips"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for value, name, maximum in (
            (args.width, "width", 8192),
            (args.height, "height", 8192),
            (args.workers, "workers", 4)):
        if type(value) is not int or value < 1 or value > maximum:
            parser.error(f"{name} must be between 1 and {maximum}")
    if (not math.isfinite(args.cut_threshold) or
            args.cut_threshold <= 0.0 or args.cut_threshold >= 1.0):
        parser.error("cut threshold must be finite and between zero and one")
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
