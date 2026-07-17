#!/usr/bin/env python3
"""Prepare a small canonical-resolution public-mono bootstrap dataset.

This tool derives a bounded training/development subset from the already
authenticated REDS and Spring prepared roots.  It never reads a sealed-test
root and never mutates an input.  Selection is the first N clip names in
Unicode code-point order from each source dataset manifest.

RGB normalization is deterministic and deliberately remains display-referred
sRGB.  An RGB8 PNG that is already at the canonical size is copied byte for
byte.  Otherwise Pillow decodes it without color management, resizes the
gamma-encoded RGB samples with ``Image.Resampling.BILINEAR`` and
``reducing_gap=None``, then writes an RGB PNG with ``compress_level=6`` and
``optimize=False``.  The later HDR experiment, not this preparation step,
converts these SDR samples to simulated linear scRGB.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import uuid
import zlib

import PIL
import PIL._imaging as pillow_imaging
from PIL import Image, PngImagePlugin

import artistic_sources
import audit_artistic_dataset_splits as split_audit
import preprocessing_artifact_cache as artifact_cache


SBSBENCH_DIR = Path(__file__).resolve().parents[1] / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))
import build_clip_hash_manifest as clip_hashes  # noqa: E402


BOOTSTRAP_SCHEMA = 1
PREPARATION_CONTRACT = "apollo-public-mono-hdr-bootstrap-subset-v1"
NORMALIZATION_CONTRACT = "srgb8-canonical-bilinear-pillow-v1"
CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720
FRAME_PATTERN = re.compile(r"^frame_([0-9]{5})[.]png$")
SOURCE_CATALOG_NAME = "artistic_sources_bootstrap.json"
ACTIVE_SPLIT_NAME = "active_artistic_split_bootstrap.json"
SOURCE_LAYOUT = {
    "reds": "reds-mono-v1",
    "spring": "spring-mono-v1",
}
SOURCE_GROUPS = {
    "reds": "reds_gopro_capture",
    "spring": "spring_blender_movie",
}
SELECTION_COUNTS = {
    ("reds", "training"): 8,
    ("spring", "training"): 4,
    ("reds", "development"): 2,
    ("spring", "development"): 2,
}
SPLITS = ("training", "development")
SOURCES = ("reds", "spring")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_file(path: Path):
    path = Path(path).resolve(strict=True)
    return {
        "name": path.name.casefold(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pillow_runtime_identity():
    """Bind native/Python components that can change exact normalized PNG bytes."""
    module_paths = {
        "pillow_package": Path(PIL.__file__),
        "pillow_image": Path(Image.__file__),
        "pillow_png": Path(PngImagePlugin.__file__),
        "pillow_native_imaging": Path(pillow_imaging.__file__),
        "python_executable": Path(sys.executable),
    }
    if getattr(zlib, "__file__", None):
        module_paths["python_zlib"] = Path(zlib.__file__)
    value = {
        "contract": "apollo-bootstrap-pillow-runtime-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "pillow_version": PIL.__version__,
        "zlib": {
            "compile_version": zlib.ZLIB_VERSION,
            "runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        },
        "files": {
            role: _runtime_file(path) for role, path in sorted(module_paths.items())
        },
    }
    artifact_cache.canonical_bytes(value)
    return value


def verify_pillow_runtime_identity(expected):
    observed = pillow_runtime_identity()
    if observed != expected:
        raise RuntimeError("Pillow/Python runtime changed during bootstrap preparation")
    return observed


def _bootstrap_code_paths():
    return {
        "bootstrap_tool": Path(__file__).resolve(),
        "artifact_cache": Path(artifact_cache.__file__).resolve(),
    }


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path, description: str):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object: {path}")
    return value


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return (left == right or left.is_relative_to(right) or
            right.is_relative_to(left))


def validate_roots(prepared_root: Path, output_root: Path) -> None:
    prepared = prepared_root.resolve(strict=True)
    output = output_root.resolve(strict=False)
    if not prepared.is_dir():
        raise RuntimeError(f"public prepared root is not a directory: {prepared}")
    if paths_overlap(prepared, output):
        raise RuntimeError("bootstrap output and public prepared roots overlap")
    if output.parent == output:
        raise RuntimeError("bootstrap output cannot be a filesystem root")


def normalization_contract(width: int, height: int):
    return {
        "contract": NORMALIZATION_CONTRACT,
        "target_width": width,
        "target_height": height,
        "input_encoding": "display-referred gamma-encoded sRGB RGB8 PNG",
        "resize_domain": "gamma-encoded sRGB samples; no EOTF or color management",
        "identity_rule": "byte-exact copy when source dimensions already match",
        "resize_filter": "PIL.Image.Resampling.BILINEAR",
        "reducing_gap": None,
        "output_mode": "RGB",
        "output_format": "PNG",
        "png_compress_level": 6,
        "png_optimize": False,
        "pillow_version": PIL.__version__,
        "hdr_note": (
            "HDR is not baked into these files; the bounded experiment later "
            "applies its authenticated SDR-to-linear-scRGB simulation"
        ),
    }


def _safe_clip_name(value, origin: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."} or
            Path(value).name != value or "/" in value or "\\" in value):
        raise RuntimeError(f"{origin}: unsafe clip name {value!r}")
    return value


def _source_dataset(prepared_root: Path, source: str, split: str,
                    count: int, verify_source_hashes: bool):
    if source not in SOURCES or split not in SPLITS:
        raise RuntimeError("bootstrap source/split is outside the admitted contract")
    if type(count) is not int or count <= 0:
        raise RuntimeError("bootstrap selection count must be a positive integer")
    root = (prepared_root / SOURCE_LAYOUT[source] / split).resolve(strict=True)
    if "test" in {part.casefold() for part in root.parts}:
        raise RuntimeError("sealed-test roots are forbidden")
    manifest_path = root / "dataset_manifest.json"
    manifest = read_json(manifest_path, "source dataset manifest")
    expected_production = f"{source}_mono_v1_{split}"
    if (manifest.get("schema") != 2 or
            manifest.get("source_kind") != "mono-video" or
            manifest.get("split") != split or
            manifest.get("production_id") != expected_production):
        raise RuntimeError(f"source dataset contract differs: {manifest_path}")
    sequence_manifest_name = manifest.get("source_sequence_manifest")
    if (not isinstance(sequence_manifest_name, str) or
            Path(sequence_manifest_name).name != sequence_manifest_name):
        raise RuntimeError("source dataset has an unsafe sequence manifest path")
    sequence_manifest_path = root / sequence_manifest_name
    if sha256(sequence_manifest_path) != manifest.get("video_sha256"):
        raise RuntimeError("source dataset video identity is stale")
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source dataset has no sequence rows")
    by_clip = {}
    for index, row in enumerate(rows):
        clip = _safe_clip_name(
            row.get("clip") if isinstance(row, dict) else None,
            f"source sequence {index}",
        )
        if clip in by_clip:
            raise RuntimeError(f"source dataset repeats clip {clip}")
        if row.get("split") != split:
            raise RuntimeError(f"source sequence {clip} has the wrong split")
        by_clip[clip] = row
    selected = sorted(by_clip)[:count]
    if len(selected) != count:
        raise RuntimeError(
            f"source dataset has only {len(selected)} clips; {count} required"
        )
    source_hash_path = root / clip_hashes.MANIFEST_NAME
    source_hash_manifest = clip_hashes.load_manifest(source_hash_path)
    identities = clip_hashes.verify_selected_clips(
        source_hash_path, root, selected, full=verify_source_hashes
    )
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "sequence_manifest_path": sequence_manifest_path,
        "sequence_manifest_sha256": sha256(sequence_manifest_path),
        "clip_hash_manifest_path": source_hash_path,
        "clip_hash_manifest_sha256": sha256(source_hash_path),
        "clip_hash_semantic_content_sha256": source_hash_manifest[
            clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
        ],
        "selected": selected,
        "rows": by_clip,
        "identities": identities,
    }


def numeric_frames(clip_root: Path):
    frames = {}
    for path in clip_root.iterdir():
        if not path.is_file():
            continue
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id in frames:
            raise RuntimeError(f"duplicate frame ID in {clip_root}: {frame_id}")
        frames[frame_id] = path
    ordered = sorted(frames)
    if not ordered or ordered[0] != 0 or any(
            right != left + 1 for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError(f"source clip is not zero-based full cadence: {clip_root}")
    return frames


def _validate_label_frame_contract(value, path: Path, available_ids):
    frame_ids = value.get("frame_ids")
    if (set(value) != {"schema", "frame_ids"} or value.get("schema") != 1 or
            not isinstance(frame_ids, list) or not frame_ids or
            any(type(item) is not int or item < 0 for item in frame_ids) or
            frame_ids != sorted(set(frame_ids)) or
            not set(frame_ids).issubset(available_ids)):
        raise RuntimeError(f"invalid label-frame manifest: {path}")
    return value


def _stable_read(path: Path) -> bytes:
    before = path.stat()
    value = path.read_bytes()
    after = path.stat()
    if (before.st_size != after.st_size or
            before.st_mtime_ns != after.st_mtime_ns or
            before.st_dev != after.st_dev or before.st_ino != after.st_ino):
        raise RuntimeError(f"source changed while reading: {path}")
    return value


def _stat_receipt(path: Path):
    value = Path(path).stat()
    return {
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "device": value.st_dev,
        "inode": value.st_ino,
    }


def _stable_file_snapshot(path: Path):
    before = _stat_receipt(path)
    value = Path(path).read_bytes()
    after = _stat_receipt(path)
    if before != after or len(value) != after["bytes"]:
        raise RuntimeError(f"source changed while reading: {path}")
    return ({
        "bytes": len(value),
        "sha256": sha256_bytes(value),
    }, after, value)


def _verify_input_receipts(receipts):
    for path, expected in receipts.items():
        if _stat_receipt(path) != expected:
            raise RuntimeError(f"bootstrap source changed during generation: {path}")


def _stable_json(path: Path, description: str):
    _identity, receipt, data = _stable_file_snapshot(path)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object: {path}")
    return value, data, receipt


def normalize_frame(source: Path, destination: Path,
                    width: int, height: int):
    data = _stable_read(source)
    source_hash = sha256_bytes(data)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise RuntimeError(f"source frame is not an RGB8 PNG: {source}")
            image.load()
            source_width, source_height = image.size
            if source_width * height != source_height * width:
                raise RuntimeError(
                    f"source aspect would be stretched by canonical resize: {source}"
                )
            if image.size == (width, height):
                output = data
                operation = "identity-byte-copy"
            else:
                resized = image.resize(
                    (width, height), Image.Resampling.BILINEAR,
                    reducing_gap=None,
                )
                encoded = io.BytesIO()
                resized.save(
                    encoded, format="PNG", compress_level=6, optimize=False
                )
                output = encoded.getvalue()
                operation = "srgb-bilinear-resize"
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot decode source frame: {source}") from error
    destination.write_bytes(output)
    return {
        "output": destination.name,
        "source": str(source.resolve()),
        "source_bytes": len(data),
        "source_sha256": source_hash,
        "source_width": source_width,
        "source_height": source_height,
        "bytes": len(output),
        "sha256": sha256_bytes(output),
        "width": width,
        "height": height,
        "operation": operation,
    }


def _derived_production(source: str, split: str) -> str:
    return f"{source}_mono_hdr_bootstrap_v1_{split}"


def _dataset_contract(manifest_path: Path, source: str, split: str,
                      derived: bool):
    manifest_path = manifest_path.resolve(strict=True)
    payload = read_json(manifest_path, "artistic dataset manifest")
    production = (
        _derived_production(source, split) if derived else
        f"{source}_mono_v1_{split}"
    )
    expected_container = (
        "derived-public-image-sequences" if derived else
        "image-sequence-archives"
    )
    if (payload.get("schema") != 2 or
            payload.get("production_id") != production or
            payload.get("source_kind") != "mono-video" or
            payload.get("source_container") != expected_container or
            payload.get("split") != split):
        raise RuntimeError(
            f"artistic dataset contract differs: {manifest_path}"
        )
    relative_name = payload.get("source_sequence_manifest")
    if not isinstance(relative_name, str) or not relative_name.strip():
        raise RuntimeError(
            f"{production}: source sequence manifest is missing"
        )
    relative_path = Path(relative_name)
    if relative_path.is_absolute():
        raise RuntimeError(
            f"{production}: source sequence manifest must be relative"
        )
    dataset_root = manifest_path.parent
    sequence_path = (dataset_root / relative_path).resolve()
    try:
        sequence_path.relative_to(dataset_root)
    except ValueError as error:
        raise RuntimeError(
            f"{production}: source sequence manifest escapes dataset root"
        ) from error
    expected_identity = payload.get("video_sha256")
    if (not sequence_path.is_file() or
            not isinstance(expected_identity, str) or
            len(expected_identity) != 64 or
            any(character not in "0123456789abcdef"
                for character in expected_identity)):
        raise RuntimeError(
            f"{production}: source sequence manifest identity is missing"
        )
    current_identity = sha256(sequence_path)
    if current_identity != expected_identity:
        raise RuntimeError(
            f"{production}: source sequence manifest identity changed"
        )
    weight = payload.get("global_policy_weight")
    try:
        weight = float(weight)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{production}: invalid policy weight") from error
    if (not math.isfinite(weight) or weight <= 0.0 or
            type(payload.get("context_frame_count")) is not int or
            payload["context_frame_count"] <= 0 or
            type(payload.get("label_frame_count")) is not int or
            payload["label_frame_count"] <= 0):
        raise RuntimeError(f"{production}: unusable artistic dataset")
    for field in ("dataset", "homepage", "license", "license_url"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise RuntimeError(f"{production}: missing {field}")
    return payload, sequence_path


def _catalog_row(source: str, split: str, manifest_path: Path,
                 payload, derived: bool):
    production = payload["production_id"]
    if derived:
        upstream = payload.get("upstream")
        if not isinstance(upstream, dict):
            raise RuntimeError(f"{production}: upstream lineage is missing")
        retrieval = {
            "kind": "derived-public-image-sequences",
            "preparation_contract": PREPARATION_CONTRACT,
            "upstream_dataset_manifest": upstream.get("dataset_manifest"),
            "upstream_dataset_manifest_sha256":
                upstream.get("dataset_manifest_sha256"),
        }
    else:
        retrieval = {
            "kind": "pinned-public-image-sequence-archives",
            "sealed_test": True,
            "bootstrap_publication_access": (
                "dataset and source-sequence manifests only; frames unopened"
            ),
        }
    return {
        "id": production,
        "production_id": production,
        "source_kind": "mono-video",
        "source_group": SOURCE_GROUPS[source],
        "split": split,
        "admission": "global_policy",
        "complete_production": True,
        "global_policy_weight": payload["global_policy_weight"],
        "dataset": payload["dataset"],
        "domain": payload.get("domain"),
        "policy_role": payload.get("policy_role"),
        "experiment_role": payload.get("experiment_role"),
        "homepage": payload["homepage"],
        "license": payload["license"],
        "license_url": payload["license_url"],
        "dataset_manifest": str(manifest_path.resolve()),
        "retrieval": retrieval,
    }


def prepare_clip(source: str, split: str, source_root: Path,
                 source_row, source_identity: str, destination: Path,
                 executor: ThreadPoolExecutor, width: int, height: int,
                 tool_sha256: str, code_snapshot, runtime_snapshot,
                 preprocess_cache=None):
    artifact_cache.require_working_split(split)
    if code_snapshot.get("bootstrap_tool") != tool_sha256:
        raise RuntimeError("bootstrap tool identity snapshot differs")
    clip = source_row["clip"]
    source_clip = source_root / clip
    if not source_clip.is_dir():
        raise RuntimeError(f"source clip is missing: {source_clip}")
    frames = numeric_frames(source_clip)
    if len(frames) != source_row.get("context_frames"):
        raise RuntimeError(f"source frame count differs from manifest: {source_clip}")
    label_path = source_clip / "label_frames.json"
    label_value, label_bytes, label_receipt = _stable_json(
        label_path, "label-frame manifest",
    )
    label_value = _validate_label_frame_contract(
        label_value, label_path, set(frames),
    )
    if len(label_value["frame_ids"]) != source_row.get("label_frames"):
        raise RuntimeError(f"source label count differs from manifest: {source_clip}")
    source_meta_path = source_clip / "meta.json"
    source_record_path = source_clip / "source_sequence_record.json"
    source_meta, source_meta_bytes, source_meta_receipt = _stable_json(
        source_meta_path, "source clip metadata",
    )
    source_record, source_record_bytes, source_record_receipt = _stable_json(
        source_record_path, "source sequence record",
    )
    if (source_meta.get("split") != split or
            source_meta.get("production_id") !=
            f"{source}_mono_v1_{split}" or
            source_record.get("clip") != clip or
            source_record.get("source_frame_count") != len(frames)):
        raise RuntimeError(f"source clip provenance differs: {source_clip}")

    cache = None
    cache_identity = None
    source_files = []
    input_receipts = {}
    if preprocess_cache is not None:
        input_receipts = {
            label_path: label_receipt,
            source_meta_path: source_meta_receipt,
            source_record_path: source_record_receipt,
        }
        for frame_id, path in sorted(frames.items()):
            file_identity, receipt, _data = _stable_file_snapshot(path)
            source_files.append({
                "frame": frame_id,
                "path": str(path.resolve()),
                **file_identity,
            })
            input_receipts[path] = receipt
        cache = artifact_cache.DirectoryArtifactCache(preprocess_cache)
        cache_identity = artifact_cache.cache_identity(
            artifact_kind="apollo-public-mono-bootstrap-clip-v1",
            source={
                "source": source,
                "clip": clip,
                "clip_root": str(source_clip.resolve()),
                "clip_manifest_identity": source_identity,
                "frames": source_files,
                "meta_sha256": sha256_bytes(source_meta_bytes),
                "sequence_record_sha256": sha256_bytes(source_record_bytes),
                "label_frames_sha256": sha256_bytes(label_bytes),
            },
            selection={
                "split": split,
                "source_row": source_row,
                "label_frame_ids": label_value["frame_ids"],
            },
            preprocessing=normalization_contract(width, height),
            color_contract={
                "source": source_meta.get("color_contract"),
                "normalized":
                    "rgb8-png-canonical-display-referred-srgb",
            },
            code={
                "bootstrap_tool_sha256": code_snapshot["bootstrap_tool"],
                "artifact_cache_sha256": code_snapshot["artifact_cache"],
                "pillow_python_runtime": runtime_snapshot,
            },
        )
        if cache.materialize(cache_identity, destination):
            artifact_cache.verify_code_identities(
                _bootstrap_code_paths(), code_snapshot,
            )
            verify_pillow_runtime_identity(runtime_snapshot)
            _verify_input_receipts(input_receipts)
            record = read_json(
                destination / "source_sequence_record.json",
                "cached source sequence record",
            )
            if (record.get("clip") != clip or
                    record.get("source_clip_sha1") != source_identity or
                    record.get("width") != width or
                    record.get("height") != height):
                raise RuntimeError("cached source sequence identity differs")
            return record

    destination.mkdir()
    futures = {
        frame_id: executor.submit(
            normalize_frame, path, destination / path.name, width, height
        )
        for frame_id, path in sorted(frames.items())
    }
    frame_rows = []
    for frame_id in sorted(futures):
        row = futures[frame_id].result()
        row["local_frame"] = frame_id
        frame_rows.append(row)
    (destination / "label_frames.json").write_bytes(label_bytes)

    production = _derived_production(source, split)
    derived_meta = dict(source_meta)
    derived_meta.update({
        "schema": 2,
        "name": clip,
        "film_id": production,
        "production_id": production,
        "split": split,
        "source_width": width,
        "source_height": height,
        "source_color_contract": (
            "rgb8-png-canonical-display-referred-srgb"
        ),
        "normalization_contract": NORMALIZATION_CONTRACT,
        "experiment_role": "bounded-hdr-bootstrap",
        "auxiliary_disparity": None,
        "excluded_auxiliary_data": (
            "Spring disparity remains at its original raster and is not copied "
            "into this normalized RGB derivative"
            if source == "spring" else None
        ),
        "source_provenance": {
            "clip_root": str(source_clip.resolve()),
            "clip_sha1": source_identity,
            "meta_path": str(source_meta_path.resolve()),
            "meta_sha256": sha256_bytes(source_meta_bytes),
            "sequence_record_path": str(source_record_path.resolve()),
            "sequence_record_sha256": sha256_bytes(source_record_bytes),
            "label_frames_path": str(label_path.resolve()),
            "label_frames_sha256": sha256_bytes(label_bytes),
        },
    })
    write_json_atomic(destination / "meta.json", derived_meta)
    record = {
        "schema": BOOTSTRAP_SCHEMA,
        "preparation_contract": PREPARATION_CONTRACT,
        "normalization_contract": NORMALIZATION_CONTRACT,
        "clip": clip,
        "source_sequence": source_row.get("source_sequence"),
        "source_first_frame": source_row.get("source_start_frame"),
        "source_last_frame": source_row.get("source_end_frame"),
        "source_frame_count": len(frame_rows),
        "label_frame_ids": label_value["frame_ids"],
        "source_clip_root": str(source_clip.resolve()),
        "source_clip_sha1": source_identity,
        "source_meta_sha256": sha256_bytes(source_meta_bytes),
        "source_sequence_record_sha256": sha256_bytes(source_record_bytes),
        "source_label_frames_sha256": sha256_bytes(label_bytes),
        "width": width,
        "height": height,
        "frames": frame_rows,
        "excluded_auxiliary_data": derived_meta["excluded_auxiliary_data"],
    }
    write_json_atomic(destination / "source_sequence_record.json", record)
    if cache is not None:
        expected_by_frame = {row["frame"]: row for row in source_files}
        for row in frame_rows:
            expected = expected_by_frame[row["local_frame"]]
            if (row.get("source_bytes") != expected["bytes"] or
                    row.get("source_sha256") != expected["sha256"]):
                raise RuntimeError(
                    "bootstrap source bytes changed between cache keying and resize"
                )
        artifact_cache.verify_code_identities(
            _bootstrap_code_paths(), code_snapshot,
        )
        verify_pillow_runtime_identity(runtime_snapshot)
        _verify_input_receipts(input_receipts)
        cache.publish(cache_identity, destination)
    return record


def _dataset_output(stage: Path, source: str, split: str) -> Path:
    return stage / f"{source}-mono-hdr-bootstrap-v1" / split


def prepare_dataset(stage: Path, source: str, split: str, source_data,
                    executor: ThreadPoolExecutor, width: int, height: int,
                    tool_sha256: str, code_snapshot, runtime_snapshot,
                    preprocess_cache=None):
    output = _dataset_output(stage, source, split)
    output.mkdir(parents=True)
    records = []
    for index, clip in enumerate(source_data["selected"], 1):
        records.append(prepare_clip(
            source, split, source_data["root"], source_data["rows"][clip],
            source_data["identities"][clip], output / clip, executor,
            width, height, tool_sha256, code_snapshot, runtime_snapshot,
            preprocess_cache,
        ))
        print(
            f"[{source}/{split}] {index}/{len(source_data['selected'])} {clip}",
            flush=True,
        )
    source_manifest = source_data["manifest"]
    selection = {
        "schema": BOOTSTRAP_SCHEMA,
        "preparation_contract": PREPARATION_CONTRACT,
        "selection_rule": (
            "first N Unicode-code-point-sorted clip names from the authenticated "
            "source dataset manifest"
        ),
        "source": source,
        "split": split,
        "source_production_id": source_manifest["production_id"],
        "production_id": _derived_production(source, split),
        "source_dataset_manifest": str(
            source_data["manifest_path"].resolve()
        ),
        "source_dataset_manifest_sha256": source_data["manifest_sha256"],
        "source_sequence_manifest": str(
            source_data["sequence_manifest_path"].resolve()
        ),
        "source_sequence_manifest_sha256": source_data[
            "sequence_manifest_sha256"
        ],
        "source_clip_hash_manifest": str(
            source_data["clip_hash_manifest_path"].resolve()
        ),
        "source_clip_hash_manifest_sha256": source_data[
            "clip_hash_manifest_sha256"
        ],
        "source_clip_hash_semantic_content_sha256": source_data[
            "clip_hash_semantic_content_sha256"
        ],
        "selected_clip_count": len(records),
        "selected_clips": source_data["selected"],
        "normalization": normalization_contract(width, height),
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": tool_sha256,
        "sequences": records,
        "context_frame_count": sum(
            record["source_frame_count"] for record in records
        ),
        "label_frame_count": sum(
            len(record["label_frame_ids"]) for record in records
        ),
    }
    write_json_atomic(output / "source_sequence_manifest.json", selection)
    video_identity = sha256(output / "source_sequence_manifest.json")
    rows = []
    for record in records:
        source_row = source_data["rows"][record["clip"]]
        rows.append({
            "clip": record["clip"],
            "source_sequence": record["source_sequence"],
            "source_start_frame": source_row.get("source_start_frame"),
            "source_end_frame": source_row.get("source_end_frame"),
            "context_frames": record["source_frame_count"],
            "label_frames": len(record["label_frame_ids"]),
            "split": split,
        })
    dataset = {
        "schema": 2,
        "dataset": source_manifest["dataset"],
        "domain": source_manifest["domain"],
        "production_id": _derived_production(source, split),
        "source_kind": "mono-video",
        "source_container": "derived-public-image-sequences",
        "source_sequence_manifest": "source_sequence_manifest.json",
        "temporal_contract": "full-cadence-shot",
        "policy_role": source_manifest.get("policy_role", "cinematic_training"),
        "experiment_role": "bounded-hdr-bootstrap",
        "homepage": source_manifest.get("homepage"),
        "license": source_manifest.get("license"),
        "license_url": source_manifest.get("license_url"),
        "split": split,
        "context_fps": source_manifest["context_fps"],
        "global_policy_weight": source_manifest["global_policy_weight"],
        "color_contract": "decoded-sdr-bgr8",
        "source_color_contract": (
            "rgb8-png-canonical-display-referred-srgb"
        ),
        "normalization_contract": NORMALIZATION_CONTRACT,
        "canonical_source_width": width,
        "canonical_source_height": height,
        "video_sha256": video_identity,
        "sequences": rows,
        "shot_count": len(rows),
        "context_frame_count": selection["context_frame_count"],
        "label_frame_count": selection["label_frame_count"],
        "split_rule": "bounded bootstrap subset; sealed test is never read",
        "selection_rule": selection["selection_rule"],
        "upstream": {
            "production_id": source_manifest["production_id"],
            "dataset_manifest": str(source_data["manifest_path"].resolve()),
            "dataset_manifest_sha256": source_data["manifest_sha256"],
            "video_sha256": source_manifest.get("video_sha256"),
        },
    }
    write_json_atomic(output / "dataset_manifest.json", dataset)
    write_json_atomic(output / "preparation_contract.json", {
        "schema": BOOTSTRAP_SCHEMA,
        "preparation_contract": PREPARATION_CONTRACT,
        "normalization": normalization_contract(width, height),
        "selection_rule": selection["selection_rule"],
        "source": source,
        "split": split,
        "selected_clips": source_data["selected"],
        "tool_sha256": tool_sha256,
    })
    return {
        "source": source,
        "split": split,
        "relative_root": str(output.relative_to(stage)).replace("\\", "/"),
        "production_id": dataset["production_id"],
        "clips": source_data["selected"],
        "shot_count": dataset["shot_count"],
        "context_frame_count": dataset["context_frame_count"],
        "label_frame_count": dataset["label_frame_count"],
        "dataset_manifest_sha256": sha256(output / "dataset_manifest.json"),
        "source_sequence_manifest_sha256": video_identity,
    }


def _safe_remove_tree(path: Path, expected_parent: Path) -> None:
    path = path.resolve(strict=False)
    expected_parent = expected_parent.resolve(strict=True)
    if path.parent != expected_parent or path == expected_parent:
        raise RuntimeError(f"refusing to remove unsafe path: {path}")
    if path.exists():
        shutil.rmtree(path)


def _finalize_manifests(output_root: Path, datasets, workers: int):
    for row in datasets:
        root = output_root / Path(row["relative_root"])
        manifest, manifest_path = clip_hashes.build_and_write(
            root, workers=workers
        )
        clip_hashes.verify_selected_clips(
            manifest_path, root, sorted(manifest["clips"]), full=True
        )
        row["clip_hash_manifest_sha256"] = sha256(manifest_path)
        row["clip_hash_semantic_content_sha256"] = manifest[
            clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
        ]
        row["output_root"] = str(root.resolve())
    return datasets


def publish_training_contract(prepared_root: Path, output_root: Path,
                              datasets):
    expected = {(source, split) for source in SOURCES for split in SPLITS}
    indexed = {
        (row["source"], row["split"]): row
        for row in datasets
        if isinstance(row, dict)
    }
    if set(indexed) != expected:
        raise RuntimeError(
            "bootstrap datasets do not cover the four training productions"
        )

    catalog_rows = []
    manifest_paths = []
    sequence_identities = {}
    for source in SOURCES:
        for split in SPLITS:
            manifest_path = (
                output_root / indexed[(source, split)]["relative_root"] /
                "dataset_manifest.json"
            )
            payload, sequence_path = _dataset_contract(
                manifest_path, source, split, derived=True
            )
            catalog_rows.append(_catalog_row(
                source, split, manifest_path, payload, derived=True
            ))
            manifest_paths.append(manifest_path)
            sequence_identities[payload["production_id"]] = {
                "path": str(sequence_path),
                "sha256": sha256(sequence_path),
            }

    sealed_tests = []
    for source in SOURCES:
        manifest_path = (
            prepared_root / SOURCE_LAYOUT[source] / "test" /
            "dataset_manifest.json"
        )
        payload, sequence_path = _dataset_contract(
            manifest_path, source, "test", derived=False
        )
        catalog_rows.append(_catalog_row(
            source, "test", manifest_path, payload, derived=False
        ))
        manifest_paths.append(manifest_path)
        sequence_identities[payload["production_id"]] = {
            "path": str(sequence_path),
            "sha256": sha256(sequence_path),
        }
        sealed_tests.append({
            "source": source,
            "production_id": payload["production_id"],
            "source_group": SOURCE_GROUPS[source],
            "dataset_manifest": str(manifest_path.resolve()),
            "dataset_manifest_sha256": sha256(manifest_path),
            "source_sequence_manifest": str(sequence_path),
            "source_sequence_manifest_sha256": sha256(sequence_path),
            "frame_access": "none",
        })

    catalog = {
        "schema": 2,
        "purpose": (
            "Bounded canonical public-mono HDR bootstrap training and "
            "development productions with original independent REDS and "
            "Spring sealed tests"
        ),
        "preparation_contract": PREPARATION_CONTRACT,
        "sealed_test_policy": (
            "Reference original test dataset/source-sequence manifests only; "
            "do not enumerate, decode, hash, or copy sealed-test frames"
        ),
        "sources": catalog_rows,
    }
    artistic_sources.validate_catalog(
        catalog, "generated bootstrap artistic source catalog"
    )
    catalog_path = output_root / SOURCE_CATALOG_NAME
    write_json_atomic(catalog_path, catalog)
    active = split_audit.audit(catalog_path, manifest_paths)
    active_path = output_root / ACTIVE_SPLIT_NAME
    write_json_atomic(active_path, active)
    return {
        "source_catalog": str(catalog_path.resolve()),
        "source_catalog_sha256": sha256(catalog_path),
        "active_split": str(active_path.resolve()),
        "active_split_sha256": sha256(active_path),
        "split_productions": active["split_productions"],
        "active_totals": active["totals"],
        "source_sequence_identities": sequence_identities,
        "sealed_tests": sealed_tests,
    }


def prepare_bootstrap(prepared_root: Path, output_root: Path, workers=4,
                      overwrite=False, verify_source_hashes=False,
                      width=CANONICAL_WIDTH, height=CANONICAL_HEIGHT,
                      selection_counts=None, preprocess_cache=None):
    prepared_root = Path(prepared_root)
    output_root = Path(output_root).resolve(strict=False)
    validate_roots(prepared_root, output_root)
    if preprocess_cache is not None:
        cache_root = Path(preprocess_cache).resolve(strict=False)
        if (paths_overlap(cache_root, prepared_root) or
                paths_overlap(cache_root, output_root)):
            raise RuntimeError(
                "preprocessing cache must not overlap source or output roots"
            )
    if (type(workers) is not int or workers < 1 or
            type(width) is not int or width <= 0 or
            type(height) is not int or height <= 0):
        raise RuntimeError("workers and canonical dimensions must be positive integers")
    counts = dict(SELECTION_COUNTS if selection_counts is None else
                  selection_counts)
    if set(counts) != set(SELECTION_COUNTS):
        raise RuntimeError("selection counts must cover the four admitted source splits")
    if output_root.exists() and not overwrite:
        raise RuntimeError(f"output already exists; use --overwrite: {output_root}")

    source_data = {
        (source, split): _source_dataset(
            prepared_root, source, split, counts[(source, split)],
            verify_source_hashes,
        )
        for source in SOURCES for split in SPLITS
    }
    tool_path = Path(__file__).resolve()
    code_paths = _bootstrap_code_paths()
    code_snapshot = artifact_cache.code_identities(code_paths)
    tool_hash = code_snapshot["bootstrap_tool"]
    runtime_snapshot = pillow_runtime_identity()
    parent = output_root.resolve(strict=False).parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{output_root.name}.partial-{uuid.uuid4().hex}"
    backup = parent / f".{output_root.name}.backup-{uuid.uuid4().hex}"
    published = False
    had_backup = False
    try:
        stage.mkdir()
        datasets = []
        with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="artistic-bootstrap-resize") as executor:
            for source in SOURCES:
                for split in SPLITS:
                    datasets.append(prepare_dataset(
                        stage, source, split, source_data[(source, split)],
                        executor, width, height, tool_hash,
                        code_snapshot, runtime_snapshot,
                        preprocess_cache,
                    ))
        artifact_cache.verify_code_identities(code_paths, code_snapshot)
        verify_pillow_runtime_identity(runtime_snapshot)
        if output_root.exists():
            if not overwrite:
                raise RuntimeError(
                    f"output already exists; use --overwrite: {output_root}"
                )
            output_root.replace(backup)
            had_backup = True
        stage.replace(output_root)
        published = True
        datasets = _finalize_manifests(output_root, datasets, workers)
        training_contract = publish_training_contract(
            prepared_root.resolve(strict=True), output_root, datasets
        )
        summary = {
            "schema": BOOTSTRAP_SCHEMA,
            "preparation_contract": PREPARATION_CONTRACT,
            "output_root": str(output_root.resolve()),
            "selection_counts": {
                f"{source}_{split}": counts[(source, split)]
                for source in SOURCES for split in SPLITS
            },
            "normalization": normalization_contract(width, height),
            "tool": str(tool_path),
            "tool_sha256": tool_hash,
            "source_verification": (
                "full-content" if verify_source_hashes else
                "manifest-stat plus per-frame content hashing"
            ),
            "sealed_test_access": "forbidden; training/development roots only",
            "datasets": datasets,
            "training_contract": training_contract,
            "totals": {
                "productions": len(datasets),
                "shots": sum(row["shot_count"] for row in datasets),
                "context_frames": sum(
                    row["context_frame_count"] for row in datasets
                ),
                "label_frames": sum(
                    row["label_frame_count"] for row in datasets
                ),
            },
        }
        write_json_atomic(output_root / "bootstrap_manifest.json", summary)
        summary["bootstrap_manifest_sha256"] = sha256(
            output_root / "bootstrap_manifest.json"
        )
        if had_backup:
            _safe_remove_tree(backup, parent)
        return summary
    except BaseException:
        if published and output_root.exists():
            _safe_remove_tree(output_root, parent)
        if had_backup and backup.exists():
            backup.replace(output_root)
        raise
    finally:
        if stage.exists():
            _safe_remove_tree(stage, parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-prepared-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-source-hashes", action="store_true",
        help="fully re-hash selected source clip manifests before preparation",
    )
    parser.add_argument(
        "--preprocess-cache", type=Path,
        help=(
            "optional authenticated content-addressed cache for normalized "
            "training/development clips"
        ),
    )
    args = parser.parse_args()
    try:
        result = prepare_bootstrap(
            args.public_prepared_root, args.output_root, args.workers,
            args.overwrite, args.verify_source_hashes,
            preprocess_cache=args.preprocess_cache,
        )
    except (RuntimeError, OSError, clip_hashes.ClipHashManifestError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
