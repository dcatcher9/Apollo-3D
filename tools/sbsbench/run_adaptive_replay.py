#!/usr/bin/env python3
"""One-command, headset-free A/B for the production Host SBS adaptive estimator.

Each video is decoded exactly once to a lossless frame corpus.  The same corpus is then processed
serially by a force-infer control and by ``--device-conditional-replay``.  The C++ harness owns the
deep GPU-trace authentication; this runner independently binds that trace, both artifact sets, the
runtime identity, and deterministic infer/reuse invariants into one JSON/HTML verdict.

Exit codes match ``run_eval.py``: 0 pass, 1 comparison gate failure, 2 setup/run/evidence failure.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_eval  # noqa: E402
import split_video  # noqa: E402
import depth_coordinate_v2_contract  # noqa: E402
import depth_coordinate_v2_dump_contract  # noqa: E402
import host_sbs_shader_manifest  # noqa: E402

CONTROL_HARNESS_SCHEMA = 29
CONDITIONAL_HARNESS_SCHEMA = 28
CONDITIONAL_METADATA_SCHEMA = 3
TRACE_HEADER_WORDS = 16
TRACE_RING_SCHEMA = 3
TRACE_RING_TAG = 0x48525447
TRACE_RECORD_TAG = 0x31525447
TRACE_RECORD_WORDS = 176
TRACE_RECORD_FRAME = 4
TRACE_RECORD_SUBMISSION = 12
TRACE_RECORD_DEPTH = 13
TRACE_RECORD_EXPECTED_WORK = 14
TRACE_RECORD_SUBTITLE = 15
TRACE_RECORD_FLAGS = 16
TRACE_RECORD_HOST_SUBTITLE_OUTCOME = 17
TRACE_RECORD_ANALYSIS_GENERATION = 6
TRACE_RECORD_DOMAIN_TAG = 8
TRACE_RECORD_SOURCE_WIDTH = 18
TRACE_RECORD_SOURCE_HEIGHT = 19
TRACE_RECORD_FIELD_WIDTH = 20
TRACE_RECORD_FIELD_HEIGHT = 21
TRACE_RECORD_TRANSACTION_WORDS = 22
TRACE_RECORD_RESERVED0 = 23
TRACE_RECORD_TRANSACTION_BEGIN = 24
TRACE_TRANSACTION_WORDS = 64
TRACE_LOCATOR_WORDS = 80
TRACE_CONDITION_WORDS = 6
TRACE_RECORD_LOCATOR_BEGIN = TRACE_RECORD_TRANSACTION_BEGIN + TRACE_TRANSACTION_WORDS
TRACE_RECORD_CONDITION_BEGIN = TRACE_RECORD_LOCATOR_BEGIN + TRACE_LOCATOR_WORDS
TRACE_RECORD_OBSERVATION_TIMESTAMP = TRACE_RECORD_CONDITION_BEGIN + TRACE_CONDITION_WORDS
TRACE_SUBMISSION_FORCE = 1
TRACE_SUBMISSION_GPU_UNDECIDED = 2
TRACE_DEPTH_REUSE = 1
TRACE_DEPTH_INFER = 2
TRACE_SUBTITLE_OPTIONAL_OCR = 1
TRACE_SUBTITLE_ABSTENTION = 2
TRACE_SUBTITLE_SUPPRESSED = 0
TRACE_SUBTITLE_HELD_WITH_DEPTH = 5
TRACE_SUBTITLE_INVALID = 6
TRACE_HOST_SUBTITLE_SUPPRESSED = 0
TRACE_HOST_SUBTITLE_ORDINARY = 1
TRACE_FLAG_INPUT_DOMAIN_RESET = 1 << 0
TRACE_FLAG_OCR_RECORD_SUBMITTED = 1 << 2
TRACE_FLAG_SUBTITLE_SUPPRESSED = 1 << 3
TRACE_FLAG_CONDITION_EXECUTED = 1 << 4
TRACE_FLAG_SUBTITLE_BRANCH_GATED = 1 << 5
TRACE_KNOWN_FLAGS = ((1 << 6) - 1)
TRACE_LOCATOR_CUT_EPOCH_WORD = 26
TRACE_LOCATOR_FRAME_WORD = 22
WORK_NONE = 0
WORK_OPTIONAL_OCR = 1
WORK_SUBTITLE_OBSERVATION = 2
WORK_OPTIONAL_OCR_DUE = 8
WORK_SUBTITLE_OBSERVATION_DUE = 16
WORK_VALUES = {
    WORK_NONE, WORK_OPTIONAL_OCR, WORK_SUBTITLE_OBSERVATION,
    WORK_OPTIONAL_OCR_DUE, WORK_SUBTITLE_OBSERVATION_DUE,
}
OPTIONAL_OCR_RECEIPT_MAGIC = 0x52434F4F
PARALLAX_CONTAINER = np.float32(0.04)
MAX_TRACE_FRAMES = 300
MAX_REUSE_OWNER_AGE = 4
MAX_REUSE_OWNER_OBSERVATION_AGE_US = 100_000
UINT64_MAX = (1 << 64) - 1
OCR_MAX_OBSERVATION_AGE_US = 33_000
OCR_MAX_DIRTY_HOLDS = 2
ADAPTIVE_REQUEST_POLICY_SCHEMA = 2
FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
OBSERVATION_TIMELINE_MAGIC = b"SBSOTL1\0"
OBSERVATION_TIMELINE_SCHEMA = 1
OBSERVATION_TIMELINE_HEADER_BYTES = 24
CONTROL_SCOPE = "force-infer oracle for private adaptive replay"
TREATMENT_SCOPE = "shared estimator transaction/OCR cadence; offline full-frame admission"
TRACE_ROLE = "shared production estimator transaction and OCR cadence; offline ordered full-frame admission"
TRACE_FILENAME = "device_conditional_gpu_trace_ring.u32"
METADATA_FILENAME = "device_conditional_replay.json"
PER_FRAME_ARTIFACT_SCOPE = {
    "current_output": "sbs_*.png and authenticated final_parallax_*.f32",
    "branch_dependent": (
        "depth/raw/structure/ema artifacts are branch-dependent/frozen: current on infer and "
        "retained from the last infer on reuse"),
    "do_not_interpret_as": (
        "current-frame DAV2 inference evidence without the authenticated trace disposition"),
}
CONTRACT_COMPOSITE_PROVENANCE_KEYS = {
    "schema", "runtime", "model", "onnx_sha256", "embedded_dav2_onnx_sha256",
    "zipdepth_checkpoint_sha256", "guidance_preprocess_source_closure_sha256",
    "engine_recipe", "engine_artifact", "active_engine_manifest",
}
RUNTIME_COMPOSITE_PROVENANCE_KEYS = CONTRACT_COMPOSITE_PROVENANCE_KEYS | {
    "engine_sha256", "active_engine_manifest_sha256",
}


class EvidenceError(RuntimeError):
    """Invalid setup, subprocess output, or authenticated evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_artifact_provenance(build_dir: Path, logical_model: str,
                              asset_relative_path: str,
                              expected_onnx_sha256: str) -> dict:
    """Authenticate one manifest-selected engine against its contract-pinned ONNX asset."""
    assets = build_dir / "assets"
    manifest_path = assets / f"{logical_model}.active-engine.json"
    manifest = load_json(manifest_path)
    engine_name = manifest.get("engine")
    if (manifest.get("schema"), manifest.get("model"), manifest.get("onnx_sha256")) != (
            1, logical_model, expected_onnx_sha256):
        raise EvidenceError(f"invalid active engine manifest identity: {manifest_path}")
    if (not isinstance(engine_name, str) or not engine_name or
            Path(engine_name).name != engine_name):
        raise EvidenceError(f"unsafe engine name in {manifest_path}")
    onnx_relative = Path(asset_relative_path)
    if onnx_relative.is_absolute() or ".." in onnx_relative.parts:
        raise EvidenceError(f"unsafe ONNX asset path for {logical_model}")
    engine_path = assets / engine_name
    onnx_path = assets / onnx_relative
    if not engine_path.is_file() or not onnx_path.is_file():
        raise EvidenceError(
            f"missing engine or contract ONNX asset: {engine_path}, {onnx_path}")
    actual_onnx_sha256 = sha256_file(onnx_path)
    if actual_onnx_sha256 != expected_onnx_sha256:
        raise EvidenceError(
            f"contract ONNX digest mismatch for {logical_model}: {actual_onnx_sha256}")
    return {
        "logical_model": logical_model,
        "manifest_sha256": sha256_file(manifest_path),
        "engine_name": engine_name,
        "engine_sha256": sha256_file(engine_path),
        "onnx_asset": asset_relative_path,
        "onnx_sha256": actual_onnx_sha256,
    }


def adaptive_runtime_identity_snapshot(exe: Path, build_dir: Path,
                                       depth_model: str) -> dict:
    identity = dict(run_eval.runtime_identity_snapshot(
        str(exe), str(build_dir), depth_model))
    ocr = depth_coordinate_v2_contract.SUBTITLE_OCR
    identity["depth_coordinate_v2_contract_sha256"] = \
        depth_coordinate_v2_contract.CONTRACT_CANONICAL_SHA256
    identity["subtitle_ocr"] = model_artifact_provenance(
        build_dir, ocr.logical_model, ocr.asset_path, ocr.artifact_onnx_sha256)
    return identity


def require_adaptive_runtime_identity_unchanged(expected: dict, exe: Path,
                                                build_dir: Path,
                                                depth_model: str) -> dict:
    observed = adaptive_runtime_identity_snapshot(exe, build_dir, depth_model)
    if observed != expected:
        changed = sorted(
            key for key in set(expected) | set(observed)
            if expected.get(key) != observed.get(key))
        raise EvidenceError(
            "adaptive runtime identity changed during evaluation: " + ", ".join(changed))
    return observed


def indexed_files(directory: Path, prefix: str, suffix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$", re.IGNORECASE)
    for path in directory.iterdir():
        match = pattern.match(path.name) if path.is_file() else None
        if match:
            frame_id = int(match.group(1))
            if frame_id in result:
                raise EvidenceError(f"duplicate {prefix} frame id {frame_id} in {directory}")
            result[frame_id] = path
    return result


def source_frames(directory: Path, limit: int) -> list[Path]:
    indexed = []
    pattern = re.compile(r"^frame_(\d+)$", re.IGNORECASE)
    for path in directory.iterdir():
        match = pattern.match(path.stem) if path.is_file() else None
        if match and path.suffix.lower() in FRAME_SUFFIXES:
            indexed.append((int(match.group(1)), path.name.lower(), path))
    indexed.sort()
    if len({frame_id for frame_id, _, _ in indexed}) != len(indexed):
        raise EvidenceError(f"duplicate numeric frame id in {directory}")
    frames = [path for _, _, path in indexed]
    if not frames:
        raise EvidenceError(f"no frame_*.png/jpg/bmp images in {directory}")
    return frames[:limit]


def stage_prepared_corpus(source_dir: Path, frames_dir: Path, max_frames: int) -> list[Path]:
    """Copy the exact numeric subset once into private 1-based canonical names.

    This closes both ordering and identity gaps between Python provenance and the C++ harness.
    Both A/B legs see only this immutable directory, regardless of the source's starting id,
    digit width, extension, or additional frames beyond the trace capacity.
    """
    selected = source_frames(source_dir, max_frames)
    frames_dir.mkdir(parents=True, exist_ok=False)
    for sequence, source in enumerate(selected, 1):
        destination = frames_dir / f"frame_{sequence:06d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
    return source_frames(frames_dir, max_frames)


def corpus_manifest(frames: list[Path]) -> dict:
    records = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in frames
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": 1,
        "frame_count": len(records),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "frames": records,
    }


def corpus_frame_shape(frames: list[Path]) -> tuple[int, int]:
    """Return the one exact decoded source shape shared by the staged corpus."""

    if not frames:
        raise EvidenceError("staged corpus is empty")
    shapes: set[tuple[int, int]] = set()
    for path in frames:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise EvidenceError(f"cannot authenticate staged frame dimensions: {path}") from exc
        if width <= 0 or height <= 0:
            raise EvidenceError(f"staged frame has invalid dimensions: {path}")
        shapes.add((width, height))
    if len(shapes) != 1:
        rendered = ", ".join(f"{width}x{height}" for width, height in sorted(shapes))
        raise EvidenceError(
            f"staged corpus changes source dimensions across frames: {rendered}")
    return next(iter(shapes))


def safe_clip_name(path: Path, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._") or "clip"
    identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}-{identity}"
    if candidate in used:
        raise EvidenceError(f"duplicate clip identity: {path}")
    used.add(candidate)
    return candidate


def resolve_ffmpeg(explicit: str | None, build_dir: Path) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(build_dir / "tools" / "ffmpeg.exe")
    try:
        candidates.append(Path(split_video.resolve_ffmpeg()))
    except RuntimeError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise EvidenceError("no FFmpeg executable found; pass --ffmpeg")


def resolve_ffprobe(explicit: str | None, ffmpeg: Path, build_dir: Path) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        ffmpeg.with_name("ffprobe.exe"), ffmpeg.with_name("ffprobe"),
        build_dir / "tools" / "ffprobe.exe",
    ])
    discovered = shutil.which("ffprobe")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise EvidenceError("no FFprobe executable found; pass --ffprobe")


def video_observation_timestamps(video: Path, ffprobe: Path,
                                 max_frames: int) -> list[int]:
    command = [
        str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=time_base:frame=pts,best_effort_timestamp",
        "-of", "json=compact=1", str(video),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise EvidenceError(
            f"FFprobe failed for {video} (exit {result.returncode}):\n" +
            (result.stdout + result.stderr)[-2000:])
    try:
        value = json.loads(result.stdout)
        streams = value["streams"]
        frames = value["frames"][:max_frames]
        time_base = streams[0]["time_base"]
        numerator_text, denominator_text = time_base.split("/", 1)
        numerator, denominator = int(numerator_text), int(denominator_text)
        pts = []
        for frame in frames:
            timestamp = frame.get("pts")
            if timestamp is None:
                timestamp = frame.get("best_effort_timestamp")
            pts.append(int(timestamp))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise EvidenceError(f"invalid FFprobe frame timeline for {video}: {exc}") from exc
    if not pts or numerator <= 0 or denominator <= 0 or any(
            later <= earlier for earlier, later in zip(pts, pts[1:])):
        raise EvidenceError(
            f"FFprobe timeline is empty, non-increasing, or has invalid time base: {video}")
    first = pts[0]
    timestamps = [1 + ((value - first) * numerator * 1_000_000 // denominator)
                  for value in pts]
    if any(timestamp <= 0 or timestamp > (1 << 64) - 1 for timestamp in timestamps):
        raise EvidenceError(f"FFprobe timeline exceeds uint64 microseconds: {video}")
    return timestamps


def parse_positive_rational(value: str, option: str) -> tuple[int, int]:
    try:
        numerator_text, denominator_text = value.split("/", 1)
        numerator, denominator = int(numerator_text), int(denominator_text)
    except (ValueError, AttributeError) as exc:
        raise EvidenceError(f"{option} must be a positive NUM/DEN rational") from exc
    if numerator <= 0 or denominator <= 0:
        raise EvidenceError(f"{option} must be a positive NUM/DEN rational")
    return numerator, denominator


def prepared_observation_timestamps(frame_count: int, fps: str) -> list[int]:
    numerator, denominator = parse_positive_rational(fps, "--prepared-fps")
    return [1 + (index * 1_000_000 * denominator // numerator)
            for index in range(frame_count)]


def write_observation_timeline(path: Path, timestamps: list[int]) -> None:
    if (not timestamps or any(timestamp <= 0 for timestamp in timestamps) or
            any(later < earlier for earlier, later in zip(timestamps, timestamps[1:]))):
        raise EvidenceError("observation timeline is empty, zero, or regressed")
    payload = struct.pack(
        "<8sIIQ", OBSERVATION_TIMELINE_MAGIC, OBSERVATION_TIMELINE_SCHEMA,
        OBSERVATION_TIMELINE_HEADER_BYTES, len(timestamps))
    payload += struct.pack(f"<{len(timestamps)}Q", *timestamps)
    path.write_bytes(payload)


def read_observation_timeline(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) < OBSERVATION_TIMELINE_HEADER_BYTES:
        raise EvidenceError("observation timeline is shorter than its header")
    magic, schema, header_bytes, count = struct.unpack("<8sIIQ", data[:24])
    if (magic != OBSERVATION_TIMELINE_MAGIC or schema != OBSERVATION_TIMELINE_SCHEMA or
            header_bytes != OBSERVATION_TIMELINE_HEADER_BYTES or count == 0 or
            len(data) != header_bytes + count * 8):
        raise EvidenceError("observation timeline header or length is invalid")
    timestamps = list(struct.unpack(f"<{count}Q", data[header_bytes:]))
    if (any(timestamp == 0 for timestamp in timestamps) or
            any(later < earlier for earlier, later in zip(timestamps, timestamps[1:]))):
        raise EvidenceError("observation timeline timestamps are zero or regressed")
    return timestamps


def decode_video_once(video: Path, frames_dir: Path, ffmpeg: Path, max_frames: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(video), "-map", "0:v:0", "-vsync", "0", "-frames:v", str(max_frames),
        "-pix_fmt", "rgb24", str(frames_dir / "frame_%06d.png"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise EvidenceError(
            f"FFmpeg failed for {video} (exit {result.returncode}):\n" +
            (result.stdout + result.stderr)[-2000:])
    return source_frames(frames_dir, max_frames)


def run_harness(exe: Path, conf: Path, build_dir: Path, frames_dir: Path,
                observation_timeline: Path, out_dir: Path, frame_count: int,
                conditional: bool, timeout: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(exe), str(conf), "--sbs-bench", "--frames", str(frames_dir),
        "--out", str(out_dir), "--limit", str(frame_count),
        "--observation-timeline", str(observation_timeline),
    ]
    if conditional:
        command.append("--device-conditional-replay")
    else:
        command.append("--device-conditional-replay-control")
    result = subprocess.run(
        command, cwd=build_dir, capture_output=True, text=True, timeout=timeout,
        env=run_eval.production_subprocess_env())
    (out_dir / "harness.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise EvidenceError(
            f"{'conditional' if conditional else 'control'} harness failed "
            f"(exit {result.returncode}); see {out_dir / 'harness.log'}")


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return value


def common_contract_identity(contract: dict) -> dict:
    return {
        key: contract.get(key)
        for key in (
            "model", "pop_strength", "cuda_graph", "cuda_graph_captured",
            "raw_model_provenance", "parallax_v2_live", "parallax_v2_shadow",
            "parallax_v2_render", "cut_state", "warp_mapping",
            "adaptive_conditional", "final_parallax_field",
        )
    }


def _join_u64(low: int, high: int) -> int:
    return low | (high << 32)


def decode_trace(path: Path, metadata: dict, expected_frames: int) -> list[dict]:
    data = path.read_bytes()
    if len(data) % 4:
        raise EvidenceError("GPU trace length is not uint32-aligned")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if len(words) < TRACE_HEADER_WORDS:
        raise EvidenceError("GPU trace is shorter than its header")
    schema, tag, capacity, record_words = words[:4]
    next_sequence = words[4] | (words[5] << 32)
    next_slot, committed = words[6], words[7]
    ring = metadata.get("ring")
    if not isinstance(ring, dict):
        raise EvidenceError("conditional metadata is missing ring identity")
    expected_header = {
        "schema": schema,
        "tag": tag,
        "capacity": capacity,
        "record_words": record_words,
        "committed_count": committed,
        "next_sequence": next_sequence,
    }
    if any(ring.get(key) != value for key, value in expected_header.items()):
        raise EvidenceError(f"trace header disagrees with metadata: {expected_header} != {ring}")
    if (schema != TRACE_RING_SCHEMA or tag != TRACE_RING_TAG or
            capacity != MAX_TRACE_FRAMES or record_words != TRACE_RECORD_WORDS or
            committed != expected_frames):
        raise EvidenceError(
            f"unexpected trace identity/capacity/count: tag={tag:#x}, "
            f"schema={schema}, capacity={capacity}, record_words={record_words}, "
            f"committed={committed}")
    if len(words) != TRACE_HEADER_WORDS + capacity * record_words:
        raise EvidenceError("GPU trace byte size disagrees with capacity/record size")
    if any(words[8:TRACE_HEADER_WORDS]):
        raise EvidenceError("GPU trace header reserved words are nonzero")

    records = []
    oldest_slot = (next_slot + capacity - committed) % capacity
    first_sequence = next_sequence - committed
    for offset in range(committed):
        slot = (oldest_slot + offset) % capacity
        base = TRACE_HEADER_WORDS + slot * record_words
        row = words[base:base + record_words]
        sequence = _join_u64(row[2], row[3])
        frame_id = _join_u64(row[TRACE_RECORD_FRAME], row[TRACE_RECORD_FRAME + 1])
        if (row[0] != TRACE_RING_SCHEMA or row[1] != TRACE_RECORD_TAG or
                sequence != first_sequence + offset or
                row[TRACE_RECORD_TRANSACTION_WORDS] != TRACE_TRANSACTION_WORDS or
                row[TRACE_RECORD_RESERVED0] != 0 or
                row[TRACE_RECORD_FLAGS] & ~TRACE_KNOWN_FLAGS or
                row[TRACE_RECORD_EXPECTED_WORK] not in WORK_VALUES):
            raise EvidenceError(f"torn/out-of-order GPU trace record in slot {slot}")
        transaction = tuple(row[
            TRACE_RECORD_TRANSACTION_BEGIN:
            TRACE_RECORD_TRANSACTION_BEGIN + TRACE_TRANSACTION_WORDS])
        records.append({
            "frame_id": frame_id,
            "analysis_generation": _join_u64(
                row[TRACE_RECORD_ANALYSIS_GENERATION],
                row[TRACE_RECORD_ANALYSIS_GENERATION + 1]),
            "domain_tag": _join_u64(
                row[TRACE_RECORD_DOMAIN_TAG], row[TRACE_RECORD_DOMAIN_TAG + 1]),
            "submission": row[TRACE_RECORD_SUBMISSION],
            "depth": row[TRACE_RECORD_DEPTH],
            "expected_work": row[TRACE_RECORD_EXPECTED_WORK],
            "subtitle": row[TRACE_RECORD_SUBTITLE],
            "flags": row[TRACE_RECORD_FLAGS],
            "host_subtitle_outcome": row[TRACE_RECORD_HOST_SUBTITLE_OUTCOME],
            "source_width": row[TRACE_RECORD_SOURCE_WIDTH],
            "source_height": row[TRACE_RECORD_SOURCE_HEIGHT],
            "field_width": row[TRACE_RECORD_FIELD_WIDTH],
            "field_height": row[TRACE_RECORD_FIELD_HEIGHT],
            "transaction": transaction,
            "optional_ocr_executed": transaction[7] == OPTIONAL_OCR_RECEIPT_MAGIC,
            "locator": tuple(row[
                TRACE_RECORD_LOCATOR_BEGIN:
                TRACE_RECORD_LOCATOR_BEGIN + TRACE_LOCATOR_WORDS]),
            "condition": tuple(row[
                TRACE_RECORD_CONDITION_BEGIN:
                TRACE_RECORD_CONDITION_BEGIN + TRACE_CONDITION_WORDS]),
            "observation_timestamp_us": _join_u64(
                row[TRACE_RECORD_OBSERVATION_TIMESTAMP],
                row[TRACE_RECORD_OBSERVATION_TIMESTAMP + 1]),
        })
    if [record["frame_id"] for record in records] != list(range(1, expected_frames + 1)):
        raise EvidenceError("GPU trace frame identities are not the exact ordered corpus")
    return records


def _trace_subtitle_disposition(record: dict) -> int:
    work = record["expected_work"]
    depth = record["depth"]
    optional = record["optional_ocr_executed"]
    if work == WORK_NONE:
        if optional:
            raise EvidenceError("suppressed subtitle work carried an optional OCR receipt")
        return TRACE_SUBTITLE_SUPPRESSED
    if work in (WORK_SUBTITLE_OBSERVATION, WORK_SUBTITLE_OBSERVATION_DUE) and optional:
        raise EvidenceError("observation-only subtitle work carried an optional OCR receipt")
    if work == WORK_OPTIONAL_OCR and depth == TRACE_DEPTH_REUSE:
        if optional:
            raise EvidenceError("ordinary OCR executed on a depth-reuse branch")
        return TRACE_SUBTITLE_HELD_WITH_DEPTH
    if work == WORK_SUBTITLE_OBSERVATION and depth == TRACE_DEPTH_REUSE:
        return TRACE_SUBTITLE_HELD_WITH_DEPTH
    if optional:
        return TRACE_SUBTITLE_OPTIONAL_OCR
    return TRACE_SUBTITLE_ABSTENTION


def _validate_trace_transaction(record: dict) -> None:
    transaction = record["transaction"]
    expected_branch = 0 if record["depth"] == TRACE_DEPTH_REUSE else 1
    if (transaction[0] != expected_branch or
            transaction[13] != record["expected_work"] or
            transaction[7] not in (0, OPTIONAL_OCR_RECEIPT_MAGIC)):
        raise EvidenceError(
            f"frame {record['frame_id']} trace transaction disagrees with its disposition")


def _validate_subtitle_record(record: dict, previous: dict | None) -> None:
    frame_id = record["frame_id"]
    submission = record["submission"]
    flags = record["flags"]
    branch_gated = bool(flags & TRACE_FLAG_SUBTITLE_BRANCH_GATED)
    record_published = bool(flags & TRACE_FLAG_OCR_RECORD_SUBMITTED)
    conditioned = bool(flags & TRACE_FLAG_CONDITION_EXECUTED)
    suppressed = bool(flags & TRACE_FLAG_SUBTITLE_SUPPRESSED)
    work = record["expected_work"]
    expected_subtitle = _trace_subtitle_disposition(record)
    if record["subtitle"] != expected_subtitle:
        raise EvidenceError(
            f"frame {frame_id} subtitle disposition disagrees with authenticated work/receipt")
    if submission == TRACE_SUBMISSION_GPU_UNDECIDED:
        if (record["host_subtitle_outcome"] != TRACE_HOST_SUBTITLE_ORDINARY or
                not branch_gated or record_published or conditioned or suppressed):
            raise EvidenceError(
                f"frame {frame_id} opaque subtitle work has invalid host execution flags")
    elif submission == TRACE_SUBMISSION_FORCE:
        if branch_gated:
            raise EvidenceError(f"frame {frame_id} force work is incorrectly branch-gated")
        if work == WORK_NONE:
            if (record["host_subtitle_outcome"] != TRACE_HOST_SUBTITLE_SUPPRESSED or
                    not suppressed or record_published or conditioned):
                raise EvidenceError(f"frame {frame_id} suppression flags are inconsistent")
        elif (record["host_subtitle_outcome"] != TRACE_HOST_SUBTITLE_ORDINARY or
              suppressed or not record_published or not conditioned):
            raise EvidenceError(f"frame {frame_id} force subtitle publication is incomplete")
    else:
        raise EvidenceError(f"frame {frame_id} has an invalid submission class")

    locator_frame = _join_u64(
        record["locator"][TRACE_LOCATOR_FRAME_WORD],
        record["locator"][TRACE_LOCATOR_FRAME_WORD + 1])
    if expected_subtitle == TRACE_SUBTITLE_HELD_WITH_DEPTH:
        if (previous is None or
                (record["locator"], record["condition"]) !=
                (previous["locator"], previous["condition"])):
            raise EvidenceError("ordinary reuse did not bit-exactly hold SLR80 + condition6")
        if locator_frame == 0 or locator_frame >= frame_id:
            raise EvidenceError("held subtitle tuple does not retain an older frame identity")
    elif expected_subtitle in (TRACE_SUBTITLE_OPTIONAL_OCR, TRACE_SUBTITLE_ABSTENTION):
        if locator_frame != frame_id:
            raise EvidenceError("published subtitle tuple is not bound to the current frame")


def _validate_ocr_cadence(records: list[dict]) -> None:
    last_guaranteed = 0
    dirty_holds = 0
    for record in records:
        timestamp = record["observation_timestamp_us"]
        due = (last_guaranteed == 0 or timestamp < last_guaranteed or
               timestamp - last_guaranteed >= OCR_MAX_OBSERVATION_AGE_US or
               dirty_holds >= OCR_MAX_DIRTY_HOLDS)
        work = record["expected_work"]
        if work == WORK_NONE:
            raise EvidenceError("offline adaptive replay unexpectedly suppressed subtitle work")
        if due != (work in (WORK_OPTIONAL_OCR_DUE, WORK_SUBTITLE_OBSERVATION_DUE)):
            raise EvidenceError(
                f"frame {record['frame_id']} disagrees with the shared due-OCR cadence")
        if due or record["submission"] == TRACE_SUBMISSION_FORCE:
            last_guaranteed = timestamp
            dirty_holds = 0
        else:
            dirty_holds = min(dirty_holds + 1, OCR_MAX_DIRTY_HOLDS)


def _validate_contract_and_trace(control_dir: Path, treatment_dir: Path,
                                 expected_frames: int,
                                 observation_timeline: Path | None = None
                                 ) -> tuple[dict, list[dict]]:
    control = load_json(control_dir / "contract.json")
    treatment = load_json(treatment_dir / "contract.json")
    if (control.get("schema"), control.get("depth_step"),
            control.get("depth_reuse_interval")) != (
                CONTROL_HARNESS_SCHEMA, "force-current-adaptive-replay", 1):
        raise EvidenceError("control is not the private force-current adaptive replay oracle")
    control_descriptor = control.get("device_conditional_replay_control")
    if control_descriptor != {"enabled": True, "scope": CONTROL_SCOPE}:
        raise EvidenceError("control lacks private adaptive replay oracle authority")
    if (treatment.get("schema"), treatment.get("depth_step"),
            treatment.get("depth_reuse_interval")) != (
                CONDITIONAL_HARNESS_SCHEMA, "gpu-device-conditional", None):
        raise EvidenceError(
            f"treatment is not the device-conditional schema-{CONDITIONAL_HARNESS_SCHEMA} replay")
    control_identity = common_contract_identity(control)
    treatment_identity = common_contract_identity(treatment)
    if any(value is None for value in control_identity.values()):
        raise EvidenceError("control contract is missing shared runtime identity")
    if control_identity != treatment_identity:
        raise EvidenceError(
            "control/treatment producer, renderer, model, or shader identity differs")
    adaptive = control_identity.get("adaptive_conditional")
    if adaptive != {
            "request_policy_schema": ADAPTIVE_REQUEST_POLICY_SCHEMA,
            "near_identical_detector_source_closure_sha256":
                host_sbs_shader_manifest.NEAR_IDENTICAL_DETECTOR_GROUP.
                source_closure_sha256,
            }:
        raise EvidenceError("adaptive request policy or detector identity is stale")
    if observation_timeline is None:
        observation_timeline = control_dir.parent / "observation_timeline.sbsotl"
    timeline = read_observation_timeline(observation_timeline)
    if len(timeline) != expected_frames:
        raise EvidenceError("observation timeline does not cover the exact corpus")
    timeline_descriptor = {
        "schema": OBSERVATION_TIMELINE_SCHEMA,
        "timestamp_unit": "monotonic-source-us-plus-one",
        "count": expected_frames,
        "sha256": sha256_file(observation_timeline),
    }
    if (control.get("observation_timeline") != timeline_descriptor or
            treatment.get("observation_timeline") != timeline_descriptor):
        raise EvidenceError(
            "control/treatment contracts do not bind the exact observation timeline")
    final_descriptor = control.get("final_parallax_field")
    final_contract = depth_coordinate_v2_contract.FINAL_PARALLAX
    if final_descriptor != {
            "file_pattern": "final_parallax_<frame-id>.f32",
            "dtype": "float32-le",
            "layout": "row-major",
            "authority": final_contract.authority,
            "contract_schema": final_contract.schema,
            "publication_policy": final_contract.publication_policy,
            "reuse_policy": final_contract.reuse_policy,
            "invalid_policy": final_contract.invalid_policy,
            "current_rgb_policy": final_contract.current_rgb_policy,
            }:
        raise EvidenceError("adaptive final-parallax field contract is stale or incomplete")

    descriptor = treatment.get("device_conditional_replay")
    expected_descriptor_keys = {
        "enabled", "scope", "bootstrap", "followup", "raw_trace", "metadata",
        "force_submissions", "gpu_undecided_submissions",
    }
    if (not isinstance(descriptor, dict) or set(descriptor) != expected_descriptor_keys or
            descriptor.get("enabled") is not True or
            descriptor.get("scope") != TREATMENT_SCOPE or
            descriptor.get("bootstrap") != "force-infer" or
            descriptor.get("followup") != "gpu-owned-infer-or-reuse" or
            descriptor.get("raw_trace") != TRACE_FILENAME or
            descriptor.get("metadata") != METADATA_FILENAME):
        raise EvidenceError("treatment contract lacks device_conditional_replay authority")
    metadata_path = treatment_dir / METADATA_FILENAME
    trace_path = treatment_dir / TRACE_FILENAME
    metadata = load_json(metadata_path)
    expected_metadata_keys = {
        "schema", "role", "raw_trace", "ring", "capture_match", "submission_counts",
        "authenticated_device_dispositions", "authenticated_subtitle_dispositions",
        "per_frame_artifact_scope", "gpu_trace_source",
    }
    if (set(metadata) != expected_metadata_keys or
            metadata.get("schema") != CONDITIONAL_METADATA_SCHEMA or
            metadata.get("role") != TRACE_ROLE or
            metadata.get("raw_trace") != TRACE_FILENAME or
            metadata.get("per_frame_artifact_scope") != PER_FRAME_ARTIFACT_SCOPE):
        raise EvidenceError(
            f"conditional metadata schema must be {CONDITIONAL_METADATA_SCHEMA}")
    submissions = metadata.get("submission_counts")
    depth = metadata.get("authenticated_device_dispositions")
    subtitle = metadata.get("authenticated_subtitle_dispositions")
    if not all(isinstance(value, dict) for value in (submissions, depth, subtitle)):
        raise EvidenceError("conditional disposition summaries are missing")
    if submissions.get("force", 0) + submissions.get("gpu_undecided", 0) != expected_frames:
        raise EvidenceError("conditional submission counts do not cover the corpus")
    if depth.get("infer", 0) + depth.get("reuse", 0) != expected_frames:
        raise EvidenceError("authenticated depth dispositions do not cover the corpus")
    expected_subtitle_keys = {
        "suppressed", "optional_ocr", "abstention", "held_with_depth",
    }
    if set(subtitle) != expected_subtitle_keys or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in subtitle.values()):
        raise EvidenceError("authenticated subtitle disposition summary is malformed")
    if sum(subtitle.values()) != expected_frames:
        raise EvidenceError("authenticated subtitle dispositions do not cover the corpus")
    if descriptor.get("force_submissions") != submissions.get("force") or \
            descriptor.get("gpu_undecided_submissions") != submissions.get("gpu_undecided"):
        raise EvidenceError("contract and metadata submission counts disagree")
    provenance = metadata.get("gpu_trace_source")
    if provenance != {
            "closure_schema": host_sbs_shader_manifest.SOURCE_CLOSURE_SCHEMA,
            "compile_flags": host_sbs_shader_manifest.SHADER_COMPILE_FLAGS,
            "macro_count": host_sbs_shader_manifest.SOURCE_MACRO_COUNT,
            "closure_sha256": host_sbs_shader_manifest.GPU_TRACE_GROUP.source_closure_sha256,
            }:
        raise EvidenceError("GPU trace shader provenance is incomplete")

    records = decode_trace(trace_path, metadata, expected_frames)
    if [record["observation_timestamp_us"] for record in records] != timeline:
        raise EvidenceError("GPU trace observation timestamps disagree with the media timeline")
    capture = metadata.get("capture_match")
    expected_capture_keys = {
        "matched_frame_id", "analysis_generation", "source_width", "source_height",
        "field_width", "field_height", "domain_tag", "input_domain_reset",
    }
    if not isinstance(capture, dict) or set(capture) != expected_capture_keys:
        actual_keys = sorted(capture) if isinstance(capture, dict) else type(capture).__name__
        raise EvidenceError(
            f"conditional capture_match keys are incomplete: expected "
            f"{sorted(expected_capture_keys)}, got {actual_keys}")
    if (type(capture["matched_frame_id"]) is not int or
            capture["matched_frame_id"] != expected_frames):
        raise EvidenceError(
            f"conditional capture matched_frame_id must be {expected_frames}, got "
            f"{capture['matched_frame_id']!r}")
    analysis_generation = capture["analysis_generation"]
    if (type(analysis_generation) is not int or analysis_generation < 0 or
            analysis_generation > UINT64_MAX):
        raise EvidenceError(
            "conditional capture analysis_generation must be an unsigned 64-bit value; "
            "zero is the canonical full-frame generation")
    domain_tag = capture["domain_tag"]
    if type(domain_tag) is not int or domain_tag <= 0 or domain_tag > UINT64_MAX:
        raise EvidenceError("conditional capture domain_tag must be a nonzero unsigned 64-bit value")
    if type(capture["input_domain_reset"]) is not bool:
        raise EvidenceError("conditional capture input_domain_reset must be boolean")
    for record in records:
        if record["analysis_generation"] != analysis_generation:
            raise EvidenceError(
                f"GPU trace frame {record['frame_id']} analysis_generation "
                f"{record['analysis_generation']} disagrees with capture {analysis_generation}")
        if record["domain_tag"] != domain_tag:
            raise EvidenceError(
                f"GPU trace frame {record['frame_id']} domain_tag {record['domain_tag']} "
                f"disagrees with capture {domain_tag}")
    trace_reset = bool(records[-1]["flags"] & TRACE_FLAG_INPUT_DOMAIN_RESET)
    if trace_reset != capture["input_domain_reset"]:
        raise EvidenceError(
            f"latest GPU trace input-domain reset {trace_reset} disagrees with capture "
            f"{capture['input_domain_reset']}")
    _validate_ocr_cadence(records)
    reuse_count = 0
    infer_count = 0
    previous = None
    decoded_subtitles = {
        "suppressed": 0, "optional_ocr": 0, "abstention": 0,
        "held_with_depth": 0,
    }
    subtitle_names = {
        TRACE_SUBTITLE_SUPPRESSED: "suppressed",
        TRACE_SUBTITLE_OPTIONAL_OCR: "optional_ocr",
        TRACE_SUBTITLE_ABSTENTION: "abstention",
        TRACE_SUBTITLE_HELD_WITH_DEPTH: "held_with_depth",
    }
    for record in records:
        _validate_trace_transaction(record)
        _validate_subtitle_record(record, previous)
        decoded_subtitles[subtitle_names[record["subtitle"]]] += 1
        if record["depth"] == TRACE_DEPTH_REUSE:
            reuse_count += 1
            if record["submission"] != TRACE_SUBMISSION_GPU_UNDECIDED:
                raise EvidenceError("a reuse record was not a GPU-undecided submission")
        elif record["depth"] == TRACE_DEPTH_INFER:
            infer_count += 1
            if record["submission"] not in (
                    TRACE_SUBMISSION_FORCE, TRACE_SUBMISSION_GPU_UNDECIDED):
                raise EvidenceError("infer record has an invalid submission class")
        else:
            raise EvidenceError("GPU trace contains an invalid depth disposition")
        previous = record
    if (infer_count, reuse_count) != (depth.get("infer"), depth.get("reuse")):
        raise EvidenceError("decoded trace and disposition summary disagree")
    if decoded_subtitles != subtitle:
        raise EvidenceError("decoded trace and subtitle disposition summary disagree")
    _authenticated_reuse_owner_ages(records)
    return metadata, records


def _selected_runtime_composite(runtime_identity: dict) -> dict | None:
    """Return the exact contract projection of the preflight-selected fused runtime."""

    if not isinstance(runtime_identity, dict):
        raise EvidenceError("adaptive validation lacks its preflight runtime identity")
    for key in ("engine_name", "engine_sha256", "onnx_sha256",
                "preprocess_source_closure_sha256"):
        if not isinstance(runtime_identity.get(key), str) or not runtime_identity[key]:
            raise EvidenceError(f"adaptive preflight runtime identity lacks {key}")
    runtime_composite = runtime_identity.get("composite_runtime_provenance")
    if runtime_composite is None:
        if "composite_runtime_provenance" in runtime_identity:
            raise EvidenceError("adaptive preflight composite identity cannot be null")
        return None
    if (not isinstance(runtime_composite, dict) or
            set(runtime_composite) != RUNTIME_COMPOSITE_PROVENANCE_KEYS):
        raise EvidenceError("adaptive preflight fused runtime provenance is malformed")
    if (runtime_composite.get("engine_artifact") != runtime_identity["engine_name"] or
            runtime_composite.get("engine_sha256") != runtime_identity["engine_sha256"] or
            runtime_composite.get("embedded_dav2_onnx_sha256") !=
                runtime_identity["onnx_sha256"] or
            not re.fullmatch(r"[0-9a-f]{64}", runtime_composite["engine_sha256"]) or
            not re.fullmatch(
                r"[0-9a-f]{64}",
                str(runtime_composite.get("active_engine_manifest_sha256", "")))):
        raise EvidenceError("adaptive preflight fused engine/manifest identity is inconsistent")
    projected = {
        key: runtime_composite[key] for key in CONTRACT_COMPOSITE_PROVENANCE_KEYS
    }
    try:
        depth_coordinate_v2_dump_contract.validate_composite_runtime_provenance(
            projected,
            expected_dav2_onnx_sha256=runtime_identity["onnx_sha256"],
            expected_preprocess_source_closure_sha256=
                runtime_identity["preprocess_source_closure_sha256"],
        )
    except ValueError as exc:
        raise EvidenceError(
            f"adaptive preflight fused runtime provenance is unauthenticated: {exc}"
        ) from exc
    return projected


def _validated_adaptive_grid_identity(
        control_dir: Path, treatment_dir: Path, records: list[dict],
        expected_source_shape: tuple[int, int], runtime_identity: dict) -> tuple[int, int]:
    """Bind adaptive evidence to one exact source-derived coarse or fused profile."""

    source_width, source_height = expected_source_shape
    if (type(source_width) is not int or type(source_height) is not int or
            source_width <= 0 or source_height <= 0):
        raise EvidenceError("adaptive source shape must be a positive integer pair")
    selected_composite = _selected_runtime_composite(runtime_identity)
    contracts = [
        load_json(control_dir / "contract.json"),
        load_json(treatment_dir / "contract.json"),
    ]
    provenances = [contract.get("raw_model_provenance") for contract in contracts]
    expected_provenance_keys = {
        "schema", "model", "depth_model_url", "onnx_sha256", "preprocess_profile",
        "preprocess_source_closure_sha256", "raw_width", "raw_height",
    }
    if any(not isinstance(value, dict) or set(value) != expected_provenance_keys or
           value.get("schema") != 1 for value in provenances):
        raise EvidenceError("adaptive contracts lack exact raw model provenance")
    if provenances[0] != provenances[1]:
        raise EvidenceError("adaptive control/treatment raw model provenance differs")
    provenance = provenances[0]
    if (provenance["onnx_sha256"] != runtime_identity.get("onnx_sha256") or
            provenance["preprocess_source_closure_sha256"] !=
                runtime_identity.get("preprocess_source_closure_sha256")):
        raise EvidenceError("adaptive raw producer differs from the preflight runtime identity")
    calibrations = [
        calibration for calibration in depth_coordinate_v2_contract.MODEL_CALIBRATIONS
        if (calibration.depth_model == provenance["model"] and
            calibration.depth_model_url == provenance["depth_model_url"] and
            calibration.onnx_sha256 == provenance["onnx_sha256"] and
            calibration.preprocess.profile == provenance["preprocess_profile"] and
            calibration.preprocess.source_closure_sha256 ==
            provenance["preprocess_source_closure_sha256"])
    ]
    if len(calibrations) != 1:
        raise EvidenceError("adaptive raw producer has no unique calibrated identity")

    coarse = depth_coordinate_v2_dump_contract.expected_capture_grid_for_source(
        source_width, source_height, scale=1)
    high = depth_coordinate_v2_dump_contract.expected_capture_grid_for_source(
        source_width, source_height, scale=2)
    observed = provenance["raw_width"], provenance["raw_height"]
    if any("composite_runtime_provenance" not in contract for contract in contracts):
        raise EvidenceError("adaptive contracts omit explicit composite runtime provenance")
    composites = [contract["composite_runtime_provenance"] for contract in contracts]
    if selected_composite is not None:
        if observed != high or high is None:
            raise EvidenceError(
                "adaptive evidence downgraded the preflight-selected fused runtime")
        if composites != [selected_composite, selected_composite]:
            raise EvidenceError(
                "adaptive contracts do not bind the preflight-selected fused engine")
    elif observed == coarse and coarse is not None:
        if composites != [None, None]:
            raise EvidenceError("legacy adaptive grid unexpectedly claims fused provenance")
    else:
        raise EvidenceError(
            f"adaptive raw grid {observed[0]}x{observed[1]} is not the exact "
            f"source-derived profile for {source_width}x{source_height}")

    for directory in (control_dir, treatment_dir):
        shape = load_json(directory / "raw_shape.json")
        if shape != {
                "schema": 1,
                "width": observed[0],
                "height": observed[1],
                "dtype": "float32-le",
                "layout": "row-major",
                "stage": "raw model output before transform/normalization/EMA/curvature",
                }:
            raise EvidenceError("adaptive raw_shape.json disagrees with authenticated profile")
    for record in records:
        if ((record["source_width"], record["source_height"]) !=
                expected_source_shape or
                (record["field_width"], record["field_height"]) != observed):
            raise EvidenceError(
                f"frame {record['frame_id']} trace dimensions disagree with the exact "
                "source/profile binding")
    return observed


def validate_contract_and_trace(
        control_dir: Path, treatment_dir: Path, expected_frames: int,
        observation_timeline: Path, expected_source_shape: tuple[int, int],
        runtime_identity: dict
        ) -> tuple[dict, list[dict]]:
    """Validate production adaptive evidence with mandatory source/profile binding."""

    metadata, records = _validate_contract_and_trace(
        control_dir, treatment_dir, expected_frames, observation_timeline)
    observed = _validated_adaptive_grid_identity(
        control_dir, treatment_dir, records, expected_source_shape, runtime_identity)
    capture = metadata.get("capture_match")
    if (not isinstance(capture, dict) or
            (capture.get("source_width"), capture.get("source_height")) !=
            expected_source_shape or
            (capture.get("field_width"), capture.get("field_height")) != observed):
        raise EvidenceError(
            "adaptive trace metadata disagrees with the exact source/profile binding")
    return metadata, records


def _load_parallax_field(path: Path, expected_values: int) -> np.ndarray:
    data = path.read_bytes()
    expected_bytes = expected_values * np.dtype("<f4").itemsize
    if len(data) != expected_bytes:
        raise EvidenceError(
            f"misaligned parallax artifact {path}: {len(data)} bytes, "
            f"expected {expected_bytes}")
    values = np.frombuffer(data, dtype="<f4")
    if not np.all(np.isfinite(values)):
        raise EvidenceError(f"non-finite parallax artifact: {path}")
    if np.any(np.abs(values) > PARALLAX_CONTAINER):
        raise EvidenceError(f"out-of-container parallax artifact: {path}")
    return values


def _bit_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint32), right.view(np.uint32))


def _authenticated_reuse_owner_ages(records: list[dict]) -> dict[int, int]:
    """Derive GPU history-owner age from authenticated infer completions.

    The owner is the most recent authenticated infer-authorized completion in the current input
    domain.  It is deliberately not the host request's baseline delta.
    """
    max_age = MAX_REUSE_OWNER_AGE
    most_recent_infer_frame_id = None
    most_recent_infer_timestamp_us = None
    result = {}
    previous_frame_id = None
    for record in records:
        frame_id = record["frame_id"]
        if previous_frame_id is not None and frame_id != previous_frame_id + 1:
            raise EvidenceError("GPU trace is not frame-contiguous for owner-age replay")
        if record["depth"] == TRACE_DEPTH_INFER:
            most_recent_infer_frame_id = frame_id
            most_recent_infer_timestamp_us = record["observation_timestamp_us"]
        elif record["depth"] == TRACE_DEPTH_REUSE:
            if record["flags"] & (TRACE_FLAG_INPUT_DOMAIN_RESET |
                                   TRACE_FLAG_SUBTITLE_SUPPRESSED):
                raise EvidenceError("reuse crossed a CPU-known display snap boundary")
            if (most_recent_infer_frame_id is None or
                    most_recent_infer_timestamp_us is None):
                raise EvidenceError("reuse has no authenticated infer history owner")
            owner_age = frame_id - most_recent_infer_frame_id
            if not 1 <= owner_age <= max_age:
                raise EvidenceError(
                    "reuse has invalid authenticated GPU history-owner age")
            observation_age = (
                record["observation_timestamp_us"] - most_recent_infer_timestamp_us)
            if not 0 <= observation_age < MAX_REUSE_OWNER_OBSERVATION_AGE_US:
                raise EvidenceError(
                    "reuse exceeds the strict authenticated GPU history-owner time bound")
            result[frame_id] = owner_age
        else:
            raise EvidenceError("cannot derive owner age for invalid depth disposition")
        previous_frame_id = frame_id
    return result


def _transition_summary(rows: dict[str, list[float]]) -> dict:
    return {name: frame_summary(values) for name, values in rows.items()}


def validate_adaptive_artifacts(control_dir: Path, treatment_dir: Path,
                                records: list[dict]) -> dict:
    reuse_owner_ages = _authenticated_reuse_owner_ages(records)
    control = indexed_files(control_dir, "raw_", ".f32")
    treatment = indexed_files(treatment_dir, "raw_", ".f32")
    expected = {record["frame_id"] for record in records}
    if set(control) != expected or set(treatment) != expected:
        raise EvidenceError("raw depth artifacts do not cover the exact trace frame set")
    control_final = indexed_files(control_dir, "final_parallax_", ".f32")
    treatment_final = indexed_files(treatment_dir, "final_parallax_", ".f32")
    if set(control_final) != expected or set(treatment_final) != expected:
        raise EvidenceError("final-parallax artifacts do not cover the exact trace frame set")

    control_contract = load_json(control_dir / "contract.json")
    provenance = control_contract.get("raw_model_provenance")
    if not isinstance(provenance, dict):
        raise EvidenceError("control contract lacks raw field shape provenance")
    width, height = provenance.get("raw_width"), provenance.get("raw_height")
    if (not isinstance(width, int) or isinstance(width, bool) or width <= 0 or
            not isinstance(height, int) or isinstance(height, bool) or height <= 0):
        raise EvidenceError("control contract has invalid raw field dimensions")
    expected_values = width * height
    expected_bytes = expected_values * np.dtype("<f4").itemsize

    infer_current_equal = 0
    reuse_previous_equal = 0
    previous_treatment_hash = None
    previous_treatment_final = None
    held_previous_final_equal = 0
    reuse_subtitle_publications = 0
    transition_rows = {
        role: {
            "final_step_mae": [],
            "final_jerk_mae": [],
        }
        for role in ("control", "treatment")
    }
    transition_state = {
        role: {"final": None, "step": None}
        for role in ("control", "treatment")
    }

    def record_transitions(role: str, final: np.ndarray) -> None:
        state = transition_state[role]
        if state["final"] is not None:
            step = np.subtract(final, state["final"], dtype=np.float32)
            transition_rows[role]["final_step_mae"].append(
                float(np.mean(np.abs(step), dtype=np.float64)))
            if state["step"] is not None:
                jerk = np.subtract(step, state["step"], dtype=np.float32)
                transition_rows[role]["final_jerk_mae"].append(
                    float(np.mean(np.abs(jerk), dtype=np.float64)))
            state["step"] = step
        state["final"] = final

    hashes = {}
    for record in records:
        frame_id = record["frame_id"]
        for path in (control[frame_id], treatment[frame_id]):
            if path.stat().st_size != expected_bytes:
                raise EvidenceError(
                    f"raw depth artifact {path} is misaligned with {width}x{height}")
        control_hash = sha256_file(control[frame_id])
        treatment_hash = sha256_file(treatment[frame_id])
        control_final_values = _load_parallax_field(
            control_final[frame_id], expected_values)
        treatment_final_values = _load_parallax_field(
            treatment_final[frame_id], expected_values)

        if record["depth"] == TRACE_DEPTH_INFER:
            if treatment_hash != control_hash:
                raise EvidenceError(f"infer frame {frame_id} raw depth differs from force control")
            infer_current_equal += 1
        else:
            if previous_treatment_hash is None or treatment_hash != previous_treatment_hash:
                raise EvidenceError(f"reuse frame {frame_id} did not retain previous raw depth")
            reuse_previous_equal += 1
            if record["subtitle"] == TRACE_SUBTITLE_HELD_WITH_DEPTH:
                if (previous_treatment_final is None or
                        not _bit_exact(treatment_final_values, previous_treatment_final)):
                    raise EvidenceError(
                        f"held frame {frame_id} did not retain previous atomic final field")
                held_previous_final_equal += 1
            else:
                if record["subtitle"] not in (
                        TRACE_SUBTITLE_OPTIONAL_OCR, TRACE_SUBTITLE_ABSTENTION):
                    raise EvidenceError(
                        f"reuse frame {frame_id} has no authenticated subtitle publication")
                reuse_subtitle_publications += 1

        record_transitions("control", control_final_values)
        record_transitions("treatment", treatment_final_values)
        previous_treatment_hash = treatment_hash
        previous_treatment_final = treatment_final_values
        hashes[frame_id] = treatment_hash

    transitions = {
        role: _transition_summary(rows) for role, rows in transition_rows.items()
    }
    control_step = transitions["control"]["final_step_mae"]["p95"]
    treatment_step = transitions["treatment"]["final_step_mae"]["p95"]
    transitions["treatment_final_step_p95_vs_control"] = (
        None if not control_step else treatment_step / control_step)
    return {
        "field_shape": {"width": width, "height": height, "dtype": "float32-le"},
        "infer_current_raw_bit_exact_frames": infer_current_equal,
        "reuse_previous_raw_bit_exact_frames": reuse_previous_equal,
        "held_previous_final_parallax_bit_exact_frames": held_previous_final_equal,
        "reuse_subtitle_publication_frames": reuse_subtitle_publications,
        "authenticated_reuse_owner_ages": {
            str(frame_id): age for frame_id, age in reuse_owner_ages.items()
        },
        "transition_metrics": transitions,
        "treatment_raw_sha256": hashes,
    }


def frame_summary(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0, "nonzero_count": 0,
            "p50": None, "p95": None, "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "nonzero_count": int(np.count_nonzero(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def final_field_gate(artifact_checks: dict) -> dict:
    """Summarize the direct-render invariant already enforced while loading artifacts."""
    reuse_ages = artifact_checks["authenticated_reuse_owner_ages"]
    held = artifact_checks["held_previous_final_parallax_bit_exact_frames"]
    publications = artifact_checks["reuse_subtitle_publication_frames"]
    if held + publications != len(reuse_ages):
        raise EvidenceError("final-field proof does not cover every authenticated depth reuse")
    return {
        "status": "pass",
        "authenticated_reuse_frames": len(reuse_ages),
        "bit_exact_atomic_final_holds": held,
        "independent_subtitle_publications_on_reuse": publications,
    }


def comparison_metrics(control_dir: Path, treatment_dir: Path,
                       expected_frames: int, analysis_width: int = 960) -> dict:
    control = indexed_files(control_dir, "sbs_", ".png")
    treatment = indexed_files(treatment_dir, "sbs_", ".png")
    expected = set(range(1, expected_frames + 1))
    if set(control) != expected or set(treatment) != expected:
        raise EvidenceError("SBS output ids do not match the exact ordered corpus")
    rows = []
    previous_residual = None
    for frame_id in sorted(expected):
        with (Image.open(control[frame_id]) as left_image,
              Image.open(treatment[frame_id]) as right_image):
            if left_image.size != right_image.size:
                raise EvidenceError(f"SBS geometry differs at frame {frame_id}")
            width, height = left_image.size
            scale = min(1.0, analysis_width / width)
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            left = np.asarray(left_image.convert("RGB").resize(size, Image.Resampling.BOX),
                              dtype=np.int16)
            right = np.asarray(right_image.convert("RGB").resize(size, Image.Resampling.BOX),
                               dtype=np.int16)
        residual = right - left
        split = residual.shape[0] * 2 // 3
        row = {
            "frame_id": frame_id,
            "residual_mae": float(np.mean(np.abs(residual))),
            "scene_residual_mae": float(np.mean(np.abs(residual[:split]))),
            "subtitle_band_residual_mae": float(np.mean(np.abs(residual[split:]))),
        }
        if previous_residual is not None:
            delta = residual - previous_residual
            row.update({
                "residual_delta_mae": float(np.mean(np.abs(delta))),
                "scene_residual_delta_mae": float(np.mean(np.abs(delta[:split]))),
                "subtitle_band_residual_delta_mae": float(np.mean(np.abs(delta[split:]))),
            })
        previous_residual = residual
        rows.append(row)
    names = (
        "residual_mae", "scene_residual_mae", "subtitle_band_residual_mae",
        "residual_delta_mae", "scene_residual_delta_mae",
        "subtitle_band_residual_delta_mae",
    )
    return {
        "analysis_max_width": analysis_width,
        "units": "8-bit RGB absolute difference after deterministic BOX downsample",
        "summary": {
            name: frame_summary([row[name] for row in rows if name in row])
            for name in names
        },
        "frames": rows,
    }


def performance_comparison(control_dir: Path, treatment_dir: Path) -> dict:
    control = load_json(control_dir / "sbs_perf.json").get("stages")
    treatment = load_json(treatment_dir / "sbs_perf.json").get("stages")
    if (not isinstance(control, dict) or not control or
            not isinstance(treatment, dict) or set(control) != set(treatment)):
        raise EvidenceError("control/treatment performance stages are missing or misaligned")
    result = {}
    numeric_fields = ("min_ms", "p50_ms", "p95_ms", "max_ms", "mean_ms")
    for stage in sorted(control):
        left, right = control[stage], treatment[stage]
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise EvidenceError(f"invalid performance stage record: {stage}")
        if (any(not isinstance(row.get(name), (int, float))
                for row in (left, right) for name in numeric_fields) or
                left.get("n") != right.get("n") or left.get("total") != right.get("total") or
                not isinstance(left.get("n"), int) or left["n"] <= 0):
            raise EvidenceError(f"misaligned performance evidence: {stage}")

        def delta(field: str) -> tuple[float, float | None]:
            difference = float(right[field] - left[field])
            percent = None if left[field] == 0 else difference / left[field] * 100.0
            return difference, percent

        mean_delta, mean_percent = delta("mean_ms")
        p95_delta, p95_percent = delta("p95_ms")
        result[stage] = {
            "sample_count": left["n"],
            "control_mean_ms": left["mean_ms"],
            "treatment_mean_ms": right["mean_ms"],
            "mean_delta_ms": mean_delta,
            "mean_delta_percent": mean_percent,
            "control_p95_ms": left["p95_ms"],
            "treatment_p95_ms": right["p95_ms"],
            "p95_delta_ms": p95_delta,
            "p95_delta_percent": p95_percent,
        }
    return {
        "scope": "serial harness stage samples; nested stages must not be summed",
        "stages": result,
    }


def html_report(payload: dict) -> str:
    def number(value: float | None, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    rows = []
    for clip in payload["clips"]:
        summary = clip["comparison"]["summary"]
        checks = clip["adaptive_artifact_checks"]
        transitions = checks["transition_metrics"]["treatment"]
        final_gate = clip["final_field_gate"]
        transaction_perf = clip["performance"]["stages"][
            "depth_conditional_transaction"]

        rows.append(
            "<tr>" +
            f"<td>{html.escape(clip['name'])}</td>" +
            f"<td>{clip['frame_count']}</td>" +
            f"<td>{clip['dispositions']['infer']}</td>" +
            f"<td>{clip['dispositions']['reuse']}</td>" +
            f"<td>{final_gate['bit_exact_atomic_final_holds']}</td>" +
            f"<td>{number(transitions['final_step_mae']['p95'], 8)}</td>" +
            f"<td>{number(transitions['final_jerk_mae']['p95'], 8)}</td>" +
            f"<td>{number(transaction_perf['mean_delta_percent'], 2)}%</td>" +
            f"<td>{summary['scene_residual_delta_mae']['p95']:.4f}</td>" +
            f"<td>{summary['subtitle_band_residual_delta_mae']['p95']:.4f}</td>" +
            f"<td>{html.escape(str(clip['control_dir']))}</td>" +
            f"<td>{html.escape(str(clip['treatment_dir']))}</td>" +
            "</tr>")
    return """<!doctype html><meta charset="utf-8"><title>Host SBS adaptive replay A/B</title>
<style>
body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}
th,td{border:1px solid #bbb;padding:.4rem;text-align:left}code{white-space:pre-wrap}
</style>
<h1>Host SBS adaptive replay A/B</h1>
<p><b>Verdict:</b> %s</p>
<p>Every accepted depth reuse is bound to the four-frame/strict-100-ms GPU owner policy.
Ordinary reuse holds the atomic subtitle/final tuple exactly; cadence-due OCR may independently
publish a current subtitle tuple over the retained depth. The complete atomic final field is
sampled directly by the production renderer.
Image residuals are diagnostics; optional command-line bounds make them gates.</p>
<table><thead><tr>
<th>Clip</th><th>Frames</th><th>Infer</th><th>Reuse</th><th>Exact final holds</th>
<th>Final step p95</th><th>Final jerk p95</th>
<th>Transaction mean Δ</th>
<th>Scene residual Δ p95</th><th>Subtitle-band residual Δ p95</th>
<th>Control</th><th>Treatment</th></tr></thead><tbody>%s</tbody></table>
<h2>Failures</h2><code>%s</code>""" % (
        html.escape(payload["verdict"]), "".join(rows),
        html.escape("\n".join(payload["failures"]) or "none"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decode once and run force/production-conditional Host SBS A/B without a headset"))
    parser.add_argument("clips", nargs="+", help="video file(s) or prepared frame directories")
    parser.add_argument("--build-dir", default=str(REPO / "cmake-build-relwithdebinfo"))
    parser.add_argument("--conf", default=str(SCRIPT_DIR / "bench.conf"))
    parser.add_argument(
        "--out", help="new output directory (default: <build>/adaptive_replay/<time>)")
    parser.add_argument("--ffmpeg", help="FFmpeg executable for video inputs")
    parser.add_argument("--ffprobe", help="FFprobe executable for exact video timestamps")
    parser.add_argument(
        "--prepared-fps", help="required NUM/DEN source cadence for prepared frame directories")
    parser.add_argument("--max-frames", type=int, default=MAX_TRACE_FRAMES)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per harness run")
    parser.add_argument("--allow-zero-reuse", action="store_true",
                        help="accept a valid treatment that did not exercise reuse")
    parser.add_argument("--max-scene-residual-delta-p95", type=float)
    parser.add_argument("--max-subtitle-residual-delta-p95", type=float)
    args = parser.parse_args(argv)

    try:
        if not 2 <= args.max_frames <= MAX_TRACE_FRAMES:
            raise EvidenceError(f"--max-frames must be in [2,{MAX_TRACE_FRAMES}]")
        build_dir = Path(args.build_dir).resolve()
        conf = Path(args.conf).resolve()
        exe = build_dir / "sunshine.exe"
        if not exe.is_file() or not conf.is_file():
            raise EvidenceError(f"missing sunshine.exe or config: {exe}, {conf}")
        run_eval.require_current_build(str(build_dir))
        clip_paths = [Path(value).resolve() for value in args.clips]
        missing = [str(path) for path in clip_paths if not path.exists()]
        if missing:
            raise EvidenceError(f"missing clip input(s): {missing}")
        expected_model = run_eval.expected_depth_model()
        config_sha256 = sha256_file(conf)
        runtime_identity = None
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(args.out).resolve() if args.out else \
            build_dir / "adaptive_replay" / timestamp
        if out_root.exists():
            raise EvidenceError(f"output already exists; choose a new --out: {out_root}")
        out_root.mkdir(parents=True)
        ffmpeg = None
        ffprobe = None
        used_names: set[str] = set()
        payload = {
            "schema": 1,
            "verdict": "pass",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "runtime": {
                "python_executable": sys.executable,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "sunshine_executable": str(exe),
                "sunshine_sha256": sha256_file(exe),
                "config": str(conf),
                "config_sha256": config_sha256,
            },
            "failures": [],
            "clips": [],
        }
        for clip_path in clip_paths:
            name = safe_clip_name(clip_path, used_names)
            clip_root = out_root / name
            clip_root.mkdir()
            if clip_path.is_dir():
                if not args.prepared_fps:
                    raise EvidenceError(
                        "--prepared-fps NUM/DEN is required for prepared frame directories")
                frames_dir = clip_root / "frames"
                frames = stage_prepared_corpus(clip_path, frames_dir, args.max_frames)
                observation_timestamps = prepared_observation_timestamps(
                    len(frames), args.prepared_fps)
                source_identity = {
                    "kind": "prepared-frames-staged-once", "path": str(clip_path),
                    "fps": args.prepared_fps,
                }
            else:
                if ffmpeg is None:
                    ffmpeg = resolve_ffmpeg(args.ffmpeg, build_dir)
                    ffprobe = resolve_ffprobe(args.ffprobe, ffmpeg, build_dir)
                    payload["runtime"]["ffmpeg"] = str(ffmpeg)
                    payload["runtime"]["ffmpeg_sha256"] = sha256_file(ffmpeg)
                    payload["runtime"]["ffprobe"] = str(ffprobe)
                    payload["runtime"]["ffprobe_sha256"] = sha256_file(ffprobe)
                source_video_sha256 = sha256_file(clip_path)
                frames_dir = clip_root / "frames"
                observation_timestamps = video_observation_timestamps(
                    clip_path, ffprobe, args.max_frames)
                frames = decode_video_once(
                    clip_path, frames_dir, ffmpeg, args.max_frames)
                if sha256_file(clip_path) != source_video_sha256:
                    raise EvidenceError(f"source video changed during probe/decode: {clip_path}")
                if len(observation_timestamps) != len(frames):
                    raise EvidenceError(
                        f"decoded frame/timeline count mismatch for {clip_path}: "
                        f"{len(frames)} != {len(observation_timestamps)}")
                source_identity = {
                    "kind": "video-decoded-once", "path": str(clip_path),
                    "sha256": source_video_sha256,
                }
            observation_timeline = clip_root / "observation_timeline.sbsotl"
            write_observation_timeline(observation_timeline, observation_timestamps)
            source_identity["observation_timeline"] = {
                "schema": OBSERVATION_TIMELINE_SCHEMA,
                "count": len(observation_timestamps),
                "sha256": sha256_file(observation_timeline),
            }
            observation_timeline_sha256 = source_identity["observation_timeline"]["sha256"]
            before = corpus_manifest(frames)
            source_shape = corpus_frame_shape(frames)
            (clip_root / "corpus_manifest.json").write_text(
                json.dumps(before, indent=2), encoding="utf-8")
            if runtime_identity is None:
                print("untimed exact-engine preflight...", flush=True)
                run_eval.run_engine_preflight(
                    str(exe), str(conf), str(build_dir), str(frames_dir), expected_model)
                runtime_identity = adaptive_runtime_identity_snapshot(
                    exe, build_dir, expected_model)
                payload["runtime"]["runtime_identity"] = runtime_identity
            control_dir = clip_root / "control"
            treatment_dir = clip_root / "treatment"
            print(f"[{name}] force control ({len(frames)} frames)...", flush=True)
            run_harness(exe, conf, build_dir, frames_dir, observation_timeline, control_dir,
                        len(frames), False, args.timeout)
            require_adaptive_runtime_identity_unchanged(
                runtime_identity, exe, build_dir, expected_model)
            if sha256_file(conf) != config_sha256:
                raise EvidenceError(f"configuration changed during {name} control")
            if corpus_manifest(frames) != before:
                raise EvidenceError(f"source corpus changed during {name} control")
            if sha256_file(observation_timeline) != observation_timeline_sha256:
                raise EvidenceError(f"observation timeline changed during {name} control")
            if (not clip_path.is_dir() and
                    sha256_file(clip_path) != source_identity["sha256"]):
                raise EvidenceError(f"source video changed during {name} control")
            print(f"[{name}] shared production conditional treatment...", flush=True)
            run_harness(exe, conf, build_dir, frames_dir, observation_timeline, treatment_dir,
                        len(frames), True, args.timeout)
            require_adaptive_runtime_identity_unchanged(
                runtime_identity, exe, build_dir, expected_model)
            if sha256_file(conf) != config_sha256:
                raise EvidenceError(f"configuration changed during {name} treatment")
            if corpus_manifest(frames) != before:
                raise EvidenceError(f"source corpus changed during {name} treatment")
            if sha256_file(observation_timeline) != observation_timeline_sha256:
                raise EvidenceError(f"observation timeline changed during {name} treatment")
            if (not clip_path.is_dir() and
                    sha256_file(clip_path) != source_identity["sha256"]):
                raise EvidenceError(f"source video changed during {name} treatment")
            metadata, records = validate_contract_and_trace(
                control_dir, treatment_dir, len(frames), observation_timeline,
                source_shape, runtime_identity)
            artifact_checks = validate_adaptive_artifacts(
                control_dir, treatment_dir, records)
            final_gate = final_field_gate(artifact_checks)
            comparison = comparison_metrics(control_dir, treatment_dir, len(frames))
            performance = performance_comparison(control_dir, treatment_dir)
            dispositions = metadata["authenticated_device_dispositions"]
            clip_failures = []
            if not args.allow_zero_reuse and dispositions["reuse"] == 0:
                clip_failures.append("conditional replay exercised zero reuse frames")
            scene_p95 = comparison["summary"]["scene_residual_delta_mae"]["p95"]
            subtitle_p95 = comparison["summary"][
                "subtitle_band_residual_delta_mae"]["p95"]
            if (args.max_scene_residual_delta_p95 is not None and
                    scene_p95 > args.max_scene_residual_delta_p95):
                clip_failures.append(
                    f"scene residual delta p95 {scene_p95:.6f} > "
                    f"{args.max_scene_residual_delta_p95:.6f}")
            if (args.max_subtitle_residual_delta_p95 is not None and
                    subtitle_p95 > args.max_subtitle_residual_delta_p95):
                clip_failures.append(
                    f"subtitle residual delta p95 {subtitle_p95:.6f} > "
                    f"{args.max_subtitle_residual_delta_p95:.6f}")
            payload["failures"].extend(f"{name}: {failure}" for failure in clip_failures)
            payload["clips"].append({
                "name": name,
                "source": source_identity,
                "corpus_sha256": before["sha256"],
                "frame_count": len(frames),
                "control_dir": str(control_dir),
                "treatment_dir": str(treatment_dir),
                "dispositions": dispositions,
                "subtitle_dispositions": metadata["authenticated_subtitle_dispositions"],
                "adaptive_artifact_checks": artifact_checks,
                "final_field_gate": final_gate,
                "comparison": comparison,
                "performance": performance,
                "failures": clip_failures,
            })
        if payload["failures"]:
            payload["verdict"] = "comparison-failed"
        report_json = out_root / "adaptive_replay_report.json"
        report_html = out_root / "adaptive_replay_report.html"
        report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_html.write_text(html_report(payload), encoding="utf-8")
        print(f"\n=== {payload['verdict'].upper()} ===")
        print(f"JSON: {report_json}")
        print(f"HTML: {report_html}")
        for failure in payload["failures"]:
            print(f"  FAIL {failure}")
        return 1 if payload["failures"] else 0
    except (EvidenceError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"run_adaptive_replay: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
