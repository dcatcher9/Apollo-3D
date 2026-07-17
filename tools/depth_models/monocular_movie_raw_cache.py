#!/usr/bin/env python3
"""Stable raw-decode CAS producer for full-cadence SDR movie frames.

This module deliberately contains only the expensive decode, resize, PNG, and
frame-delta boundary.  Publication metadata, shot thresholds, label cadence,
and report-side derivation live in ``prepare_monocular_movie_training.py`` and
therefore do not invalidate these large frame objects.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np

import preprocessing_artifact_cache as artifact_cache
import native_runtime_identity as native_runtime


SCHEMA = 1
CONTRACT = "apollo-monocular-raw-full-cadence-v1"
MANIFEST = "raw_full_cadence.json"
FRAMES = "frames"


class BoundedPngWriter:
    def __init__(self, workers):
        if workers < 1:
            raise RuntimeError("write_workers must be at least 1")
        self.workers = workers
        self.max_pending = 2 * workers
        self.pending = deque()
        self.executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raw-png")
            if workers > 1 else None
        )

    @staticmethod
    def _write(path, image, params):
        if not cv2.imwrite(str(path), image, params):
            raise RuntimeError(f"cannot write {path}")

    def submit(self, path, image, params):
        if self.executor is None:
            self._write(path, image, params)
            return
        while len(self.pending) >= self.max_pending:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(
            self._write, path, image, params
        ))

    def close(self):
        while self.pending:
            self.pending.popleft().result()
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def abort(self):
        for future in self.pending:
            future.cancel()
        self.pending.clear()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


def runtime_identity(*, fresh=False):
    build = cv2.getBuildInformation()
    return {
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": hashlib.sha256(
            build.encode("utf-8")
        ).hexdigest(),
        "numpy_version": np.__version__,
        "native_binaries": {
            "opencv": native_runtime.module_native_identity(
                "opencv", cv2, fresh=fresh
            ),
            "numpy": native_runtime.module_native_identity(
                "numpy", np, fresh=fresh
            ),
        },
        "decode_backend": "opencv-video-capture",
        "png_writer": "opencv-imwrite",
    }


def verify_runtime_identity(expected):
    if runtime_identity(fresh=True) != expected:
        raise RuntimeError(
            "movie native preprocessing runtime changed during generation"
        )


def code_paths():
    return {
        "movie_raw_producer": Path(__file__).resolve(),
        "native_runtime_identity": Path(native_runtime.__file__).resolve(),
    }


def identity(*, video_size, video_sha256, split, start_seconds,
             end_margin_seconds, output_width, png_compression,
             color_probe, code_identity, runtime_identity_value=None):
    return artifact_cache.cache_identity(
        artifact_kind="apollo-monocular-raw-full-cadence-v1",
        source={"bytes": video_size, "sha256": video_sha256},
        selection={
            "split": split,
            "start_seconds": start_seconds,
            "end_margin_seconds": end_margin_seconds,
        },
        preprocessing={
            "contract": CONTRACT,
            "output_width": output_width,
            "png_compression": png_compression,
            "native_runtime": (
                runtime_identity() if runtime_identity_value is None else
                runtime_identity_value
            ),
        },
        color_contract=color_probe,
        code=code_identity,
    )


def analysis_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def change_score(previous, current):
    return float(np.mean(np.abs(
        previous.astype(np.float32) - current.astype(np.float32)
    )) / 255.0)


def resized_frame(frame, output_width):
    if output_width <= 0 or frame.shape[1] == output_width:
        return frame
    height = max(1, round(frame.shape[0] * output_width / frame.shape[1]))
    interpolation = (cv2.INTER_AREA if output_width < frame.shape[1]
                     else cv2.INTER_CUBIC)
    return cv2.resize(frame, (output_width, height), interpolation=interpolation)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate(root, identity_value, *, authenticated_files=None):
    root = Path(root).resolve(strict=True)
    try:
        manifest_bytes = (root / MANIFEST).read_bytes()
        value = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read raw movie preprocessing packet") from error
    rows = value.get("frames") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA or
            value.get("contract") != CONTRACT or
            value.get("identity_sha256") !=
            artifact_cache.DirectoryArtifactCache.key(identity_value) or
            not isinstance(value.get("source_fps"), (int, float)) or
            isinstance(value.get("source_fps"), bool) or
            not math.isfinite(float(value["source_fps"])) or
            float(value["source_fps"]) <= 0.0 or
            type(value.get("source_frames_reported")) is not int or
            type(value.get("start_frame")) is not int or
            not isinstance(rows, list) or not rows or
            value.get("decoded_frame_count") != len(rows)):
        raise RuntimeError("raw movie preprocessing packet identity differs")
    expected = {MANIFEST}
    previous_source = None
    for ordinal, row in enumerate(rows):
        name = f"frame_{ordinal:06d}.png"
        relative = f"{FRAMES}/{name}"
        score = row.get("change_score") if isinstance(row, dict) else None
        source = row.get("source_frame") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("ordinal") != ordinal or
                row.get("path") != relative or type(source) is not int or
                source < 0 or (previous_source is not None and
                               source != previous_source + 1) or
                type(row.get("width")) is not int or row["width"] <= 0 or
                type(row.get("height")) is not int or row["height"] <= 0 or
                not isinstance(score, (int, float)) or isinstance(score, bool) or
                not math.isfinite(float(score)) or score < 0.0 or
                not (root / relative).is_file()):
            raise RuntimeError("raw movie preprocessing frame row differs")
        expected.add(relative)
        previous_source = source
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if observed != expected:
        raise RuntimeError("raw movie preprocessing packet files differ")
    if authenticated_files is not None:
        receipt = {
            row.get("path"): row for row in authenticated_files
            if isinstance(row, dict) and isinstance(row.get("path"), str)
        }
        if set(receipt) != expected:
            raise RuntimeError("raw movie cache receipt coverage differs")
        manifest_row = receipt[MANIFEST]
        if (manifest_row.get("bytes") != len(manifest_bytes) or
                manifest_row.get("sha256") !=
                hashlib.sha256(manifest_bytes).hexdigest()):
            raise RuntimeError("raw movie cache manifest receipt differs")
    return value


def _rows(root):
    root = Path(root).resolve(strict=True)
    return [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": artifact_cache.sha256_file(path),
    } for path in sorted(root.rglob("*")) if path.is_file()]


def _summary(identity_value, receipt, status):
    rows = receipt["files"]
    return {
        "enabled": True,
        "contract": CONTRACT,
        "key_sha256": artifact_cache.DirectoryArtifactCache.key(identity_value),
        "status": status,
        "payload_bytes": sum(row["bytes"] for row in rows),
        "file_count": len(rows),
    }


def _decode(video, root, identity_value, output_width, start_seconds,
            end_margin_seconds, png_compression, write_workers):
    frames_root = Path(root) / FRAMES
    frames_root.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    writer = BoundedPngWriter(write_workers)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open monocular movie: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0.0 or not math.isfinite(fps):
            raise RuntimeError("movie has an invalid frame rate")
        start = max(0, round(start_seconds * fps))
        end = reported - max(0, round(end_margin_seconds * fps)) \
            if reported > 0 else None
        if end is not None and start >= end:
            raise RuntimeError("movie trim removes all frames")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        rows = []
        previous = None
        source = start
        while end is None or source < end:
            ok, frame = capture.read()
            if not ok:
                break
            gray = analysis_gray(frame)
            score = 0.0 if previous is None else change_score(previous, gray)
            prepared = resized_frame(frame, output_width)
            ordinal = len(rows)
            name = f"frame_{ordinal:06d}.png"
            writer.submit(
                frames_root / name, prepared,
                [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
            )
            rows.append({
                "ordinal": ordinal, "source_frame": source,
                "path": f"{FRAMES}/{name}",
                "width": int(prepared.shape[1]),
                "height": int(prepared.shape[0]),
                "change_score": score,
            })
            previous = gray
            source += 1
        writer.close()
        if not rows:
            raise RuntimeError("movie preprocessing decoded no frames")
        _write_json(Path(root) / MANIFEST, {
            "schema": SCHEMA, "contract": CONTRACT,
            "identity_sha256":
                artifact_cache.DirectoryArtifactCache.key(identity_value),
            "source_fps": fps, "source_frames_reported": reported,
            "start_frame": start, "decoded_frame_count": len(rows),
            "frames": rows,
        })
        return validate(root, identity_value)
    except BaseException:
        writer.abort()
        raise
    finally:
        capture.release()


def resolve(*, cache, identity_value, video, output_parent, output_width,
            start_seconds, end_margin_seconds, png_compression,
            write_workers, code_identity, source_snapshot,
            runtime_identity_value):
    validated = cache.validated_payload_receipt(identity_value)
    if validated is not None:
        payload, receipt = validated
        return (
            payload,
            validate(
                payload, identity_value,
                authenticated_files=receipt["files"],
            ),
            _summary(identity_value, receipt, "hit"),
            receipt,
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = artifact_cache.inheriting_temporary_directory(
        output_parent, ".raw-movie-decode-"
    )
    try:
        _decode(video, staging, identity_value, output_width, start_seconds,
                end_margin_seconds, png_compression, write_workers)
        artifact_cache.verify_source_file_snapshot(video, source_snapshot)
        artifact_cache.verify_code_identities(code_paths(), code_identity)
        verify_runtime_identity(runtime_identity_value)
        cache.publish(identity_value, staging)
        validated = cache.validated_payload_receipt(identity_value)
        if validated is None:
            raise RuntimeError("raw movie cache publication disappeared")
        payload, receipt = validated
        if receipt["files"] != _rows(staging):
            raise RuntimeError(
                "raw movie cache key produced different preprocessing bytes"
            )
        return (
            payload,
            validate(
                payload, identity_value,
                authenticated_files=receipt["files"],
            ),
            _summary(identity_value, receipt, "published"),
            receipt,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
