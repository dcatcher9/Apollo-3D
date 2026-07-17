#!/usr/bin/env python3
"""Authenticated native-HDR model-source sidecars used by Apollo tooling.

The evaluator-facing ``frame_*.png`` files are deliberately only perceptual
previews.  The depth model and harness consume the matching little-endian
RGBA16F scRGB files under ``model_source/``.  This module keeps that distinction
fail-closed and provides a cheap stat check for repeated render-grid passes plus
an explicit full-content audit before a source is first admitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


MANIFEST_NAME = "frame_model_sources.json"
MANIFEST_SCHEMA = 1
MANIFEST_CONTRACT = "apollo-native-pq-windows-scrgb-frames-v1"
CAPTURE_ENCODING = "linear-scrgb-rec709-float16-rgba-le"
PREVIEW_ENCODING = "perceptual-srgb-from-native-scrgb-reinhard-v1"
MODEL_SOURCE_DIRECTORY = "model_source"
FRAME_PATTERN = re.compile(r"^frame_([0-9]+)[.]scrgb16$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_relative_path(value, root: Path, origin: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{origin}: missing model-source path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{origin}: unsafe model-source path {value!r}")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(f"{origin}: model-source path escapes clip") from error
    return resolved


def load_manifest(clip_root: Path):
    clip_root = Path(clip_root).resolve(strict=True)
    path = clip_root / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read native-HDR frame manifest: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"native-HDR frame manifest is not an object: {path}")
    expected_top = {
        "schema", "contract", "capture_encoding", "preview_encoding",
        "width", "height", "row_pitch_bytes", "source_video",
        "conversion", "frames", "frame_count", "content_sha256",
    }
    if set(payload) != expected_top:
        raise RuntimeError(f"native-HDR frame manifest has unknown/missing fields: {path}")
    if (payload.get("schema") != MANIFEST_SCHEMA or
            payload.get("contract") != MANIFEST_CONTRACT or
            payload.get("capture_encoding") != CAPTURE_ENCODING or
            payload.get("preview_encoding") != PREVIEW_ENCODING):
        raise RuntimeError(f"unsupported native-HDR frame contract: {path}")
    width = payload.get("width")
    height = payload.get("height")
    row_pitch = payload.get("row_pitch_bytes")
    if (type(width) is not int or type(height) is not int or
            width <= 0 or height <= 0 or row_pitch != width * 8):
        raise RuntimeError(f"invalid native-HDR frame geometry: {path}")
    source_video = payload.get("source_video")
    if (not isinstance(source_video, dict) or
            not SHA256_PATTERN.fullmatch(source_video.get("sha256", ""))):
        raise RuntimeError(f"native-HDR source-video identity is invalid: {path}")
    conversion = payload.get("conversion")
    if (not isinstance(conversion, dict) or
            not SHA256_PATTERN.fullmatch(conversion.get("contract_sha256", ""))):
        raise RuntimeError(f"native-HDR conversion identity is invalid: {path}")
    frames = payload.get("frames")
    if (not isinstance(frames, list) or not frames or
            payload.get("frame_count") != len(frames)):
        raise RuntimeError(f"native-HDR frame list is invalid: {path}")
    semantic_rows = []
    by_id = {}
    expected_size = width * height * 8
    previous = -1
    for index, row in enumerate(frames):
        if not isinstance(row, dict):
            raise RuntimeError(f"native-HDR frame {index} is invalid: {path}")
        expected_fields = {
            "frame", "path", "size", "mtime_ns", "sha256",
            "preview", "preview_sha256", "timestamp_seconds", "stats",
        }
        if set(row) != expected_fields:
            raise RuntimeError(f"native-HDR frame {index} fields differ: {path}")
        frame_id = row.get("frame")
        if type(frame_id) is not int or frame_id != previous + 1:
            raise RuntimeError(f"native-HDR frame IDs are not zero-based cadence: {path}")
        previous = frame_id
        model_path = _safe_relative_path(row.get("path"), clip_root, str(path))
        match = FRAME_PATTERN.fullmatch(model_path.name)
        if (match is None or int(match.group(1)) != frame_id or
                model_path.parent.name != MODEL_SOURCE_DIRECTORY):
            raise RuntimeError(f"native-HDR model-source identity differs: {model_path}")
        preview_path = _safe_relative_path(
            row.get("preview"), clip_root, str(path)
        )
        if preview_path.name != f"frame_{frame_id:05d}.png":
            raise RuntimeError(f"native-HDR preview identity differs: {preview_path}")
        if (row.get("size") != expected_size or
                not SHA256_PATTERN.fullmatch(row.get("sha256", "")) or
                not SHA256_PATTERN.fullmatch(row.get("preview_sha256", "")) or
                not isinstance(row.get("mtime_ns"), int) or
                isinstance(row.get("mtime_ns"), bool)):
            raise RuntimeError(f"native-HDR frame identity is malformed: {model_path}")
        try:
            timestamp = float(row.get("timestamp_seconds"))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"native-HDR frame timestamp is invalid: {model_path}") from error
        if timestamp < 0.0:
            raise RuntimeError(f"native-HDR frame timestamp is negative: {model_path}")
        if not isinstance(row.get("stats"), dict):
            raise RuntimeError(f"native-HDR frame stats are missing: {model_path}")
        semantic_rows.append({
            "frame": frame_id,
            "path": row["path"],
            "size": row["size"],
            "sha256": row["sha256"],
            "preview": row["preview"],
            "preview_sha256": row["preview_sha256"],
            "timestamp_seconds": timestamp,
        })
        by_id[frame_id] = {
            **row,
            "model_path": model_path,
            "preview_path": preview_path,
        }
    semantic = {
        "contract": MANIFEST_CONTRACT,
        "capture_encoding": CAPTURE_ENCODING,
        "preview_encoding": PREVIEW_ENCODING,
        "width": width,
        "height": height,
        "row_pitch_bytes": row_pitch,
        "source_video": source_video,
        "conversion": conversion,
        "frames": semantic_rows,
    }
    if payload.get("content_sha256") != canonical_sha256(semantic):
        raise RuntimeError(f"native-HDR semantic content hash differs: {path}")
    return payload, by_id, path


def validate_clip(clip_root: Path, full: bool = False):
    """Validate one native-HDR clip and return authenticated frame records.

    ``full=False`` checks the manifest-bound size/mtime tuple for repeated grid
    passes. ``full=True`` rehashes every FP16 source and preview and is required
    at first admission.
    """

    payload, by_id, path = load_manifest(clip_root)
    for frame_id, row in by_id.items():
        model_path = row["model_path"]
        preview_path = row["preview_path"]
        try:
            stat = model_path.stat()
            preview_stat = preview_path.stat()
        except OSError as error:
            raise RuntimeError(
                f"native-HDR frame artifact is missing: frame {frame_id}"
            ) from error
        if (stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"] or
                preview_stat.st_size <= 0):
            raise RuntimeError(f"native-HDR frame stat identity differs: {model_path}")
        if full:
            if sha256(model_path) != row["sha256"]:
                raise RuntimeError(f"native-HDR model-source hash differs: {model_path}")
            if sha256(preview_path) != row["preview_sha256"]:
                raise RuntimeError(f"native-HDR preview hash differs: {preview_path}")
    return {
        "manifest": str(path),
        "manifest_sha256": sha256(path),
        "content_sha256": payload["content_sha256"],
        "width": payload["width"],
        "height": payload["height"],
        "frame_count": payload["frame_count"],
        "frames": by_id,
        "verification": "full" if full else "stat",
    }


def model_source_for_preview(preview_path: Path, full: bool = False):
    preview_path = Path(preview_path).resolve(strict=True)
    clip_root = preview_path.parent
    match = re.fullmatch(r"frame_([0-9]+)[.]png", preview_path.name)
    if match is None:
        raise RuntimeError(f"native-HDR preview name is invalid: {preview_path}")
    frame_id = int(match.group(1))
    authentication = validate_clip(clip_root, full=full)
    try:
        row = authentication["frames"][frame_id]
    except KeyError as error:
        raise RuntimeError(
            f"native-HDR preview has no model-source record: {preview_path}"
        ) from error
    if row["preview_path"] != preview_path:
        raise RuntimeError(f"native-HDR preview path differs: {preview_path}")
    return row
