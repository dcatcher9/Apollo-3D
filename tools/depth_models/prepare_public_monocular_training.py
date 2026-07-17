#!/usr/bin/env python3
"""Download and prepare the pinned REDS/Spring monocular image sequences.

The preparation is deliberately lossless: every admitted RGB8 PNG is copied
byte-for-byte at full cadence.  Five evenly spaced frames per source sequence
are selected for expensive artistic-policy labels.  Spring training disparity
is retained only for those sparse label frames.

Each aggregate split is staged under a stable ``.partial`` directory so an
interrupted multi-hour extraction can resume at completed sequence boundaries.
Only a fully validated split is atomically published.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import urllib.error
import urllib.request
import uuid
import zipfile

from PIL import Image

import artistic_sources
import audit_artistic_dataset_splits as split_audit


DEFAULT_SOURCES = Path(__file__).with_name("public_monocular_sources.json")
PREPARATION_CONTRACT = "public-monocular-image-sequences-v1"
SOURCE_SEQUENCE_MANIFEST = "source_sequence_manifest.json"
DATASET_MANIFEST = "dataset_manifest.json"
PREPARATION_MANIFEST = "preparation_contract.json"
LABEL_MANIFEST = "label_frames.json"
SEQUENCE_RECORD = "source_sequence_record.json"
SPLITS = ("training", "development", "test")
CHUNK_SIZE = 4 * 1024 * 1024

# The public source file intentionally contains only upstream facts.  Expected
# archive cardinalities are a local admission rule, not upstream metadata.
OFFICIAL_SEQUENCE_COUNTS = {
    "reds": {"train_rgb": 240, "test_rgb": 30},
    "spring": {"train_rgb": 37, "test_rgb": 10},
}


def _schema_is(value, expected):
    return type(value) is int and value == expected


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) +
            "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with staging.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(path)
    finally:
        if staging.exists():
            staging.unlink()


def _write_json(path: Path, value):
    _atomic_write(path, _json_bytes(value))


def hash_file(path: Path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value, description):
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"{description} is not a safe relative path")
    posix = PurePosixPath(value)
    if (posix.is_absolute() or any(part in {"", ".", ".."}
                                   for part in posix.parts)):
        raise RuntimeError(f"{description} is not a safe relative path")
    return Path(*posix.parts)


def _safe_component(value, description):
    if (not isinstance(value, str) or not value or value != value.strip() or
            Path(value).name != value or value in {".", ".."} or
            any(character in '<>:"/\\|?*' or ord(character) < 32
                for character in value)):
        raise RuntimeError(f"{description} is not a safe path component")
    return value


def paths_overlap(left: Path, right: Path):
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return (left == right or left.is_relative_to(right) or
            right.is_relative_to(left))


def _validate_roots(sources_path: Path, download_root: Path,
                    prepared_root: Path, catalog_output: Path | None,
                    active_output: Path | None):
    source = sources_path.resolve(strict=True)
    download = download_root.resolve(strict=False)
    prepared = prepared_root.resolve(strict=False)
    if paths_overlap(download, prepared):
        raise RuntimeError("download and prepared roots overlap")
    if source.is_relative_to(download) or source.is_relative_to(prepared):
        raise RuntimeError("source manifest must be outside mutable output roots")
    outputs = [path.resolve(strict=False) for path in
               (catalog_output, active_output) if path is not None]
    if len(outputs) != len(set(outputs)):
        raise RuntimeError("catalog and active-split outputs must be distinct")
    if any(path.is_relative_to(download) for path in outputs):
        raise RuntimeError("metadata output must be outside the download root")
    if any(path.is_relative_to(prepared) for path in outputs):
        raise RuntimeError("metadata output must be outside the prepared root")


def _validate_archive_spec(role, spec):
    if not isinstance(spec, dict):
        raise RuntimeError(f"archive {role!r} is not an object")
    relative = _safe_relative_path(spec.get("relative_path"),
                                   f"archive {role} relative_path")
    url = spec.get("url")
    expected_bytes = spec.get("bytes")
    algorithm = spec.get("hash_algorithm")
    expected_hash = spec.get("hash")
    pattern_text = spec.get("member_pattern")
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"archive {role}: missing URL")
    if (type(expected_bytes) is not int or expected_bytes <= 0 or
            algorithm not in {"sha256", "md5"} or
            not isinstance(expected_hash, str) or
            len(expected_hash) != hashlib.new(algorithm).digest_size * 2 or
            any(char not in "0123456789abcdef" for char in expected_hash)):
        raise RuntimeError(f"archive {role}: invalid byte/hash contract")
    try:
        pattern = re.compile(pattern_text)
    except (TypeError, re.error) as error:
        raise RuntimeError(f"archive {role}: invalid member pattern") from error
    if not {"sequence", "frame"}.issubset(pattern.groupindex):
        raise RuntimeError(
            f"archive {role}: member pattern needs sequence/frame groups"
        )
    normalized = dict(spec)
    normalized.update({
        "role": role,
        "relative": relative,
        "pattern": pattern,
    })
    return normalized


def load_sources(path: Path):
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read public source manifest: {path}") from error
    if not isinstance(payload, dict) or not _schema_is(payload.get("schema"), 1):
        raise RuntimeError("unsupported public source manifest")
    source_rows = payload.get("sources")
    if not isinstance(source_rows, dict) or set(source_rows) != {"reds", "spring"}:
        raise RuntimeError("public source manifest must contain REDS and Spring")
    normalized = {}
    relative_owners = {}
    for key, source in source_rows.items():
        if not isinstance(source, dict):
            raise RuntimeError(f"source {key!r} is not an object")
        for field in ("version", "dataset", "domain", "source_group",
                      "homepage", "license", "license_url", "split_salt"):
            if not isinstance(source.get(field), str) or not source[field]:
                raise RuntimeError(f"source {key}: missing {field}")
        fps = source.get("context_fps")
        weight = source.get("global_policy_weight")
        development_count = source.get("development_count")
        if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or
                not math.isfinite(float(fps)) or fps <= 0 or
                isinstance(weight, bool) or not isinstance(weight, (int, float)) or
                not math.isfinite(float(weight)) or weight <= 0 or
                type(development_count) is not int or development_count <= 0):
            raise RuntimeError(f"source {key}: invalid fps/weight/split contract")
        archives = source.get("archives")
        required = ({"train_rgb", "test_rgb"} if key == "reds" else
                    {"train_rgb", "train_disparity", "test_rgb"})
        if not isinstance(archives, dict) or set(archives) != required:
            raise RuntimeError(f"source {key}: archive roles differ from contract")
        archive_rows = {}
        for role, spec in archives.items():
            archive = _validate_archive_spec(role, spec)
            relative_key = archive["relative"].as_posix().casefold()
            owner = relative_owners.setdefault(relative_key, (key, role))
            if owner != (key, role):
                raise RuntimeError("archive relative paths collide")
            archive_rows[role] = archive
        row = dict(source)
        row["key"] = key
        row["archives"] = archive_rows
        normalized[key] = row
    return normalized, hashlib.sha256(raw).hexdigest()


def _verify_file(path: Path, spec, description="archive"):
    try:
        size = path.stat().st_size
    except OSError as error:
        raise RuntimeError(f"cannot stat {description}: {path}") from error
    if size != spec["bytes"]:
        raise RuntimeError(
            f"{description} byte count differs: {path} ({size} != {spec['bytes']})"
        )
    actual = hash_file(path, spec["hash_algorithm"])
    if actual != spec["hash"]:
        raise RuntimeError(f"{description} hash differs: {path}")


def _response_status(response):
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return status


def download_archive(download_root: Path, spec, opener=None):
    """Verify one pinned archive, resuming an interrupted HTTP transfer."""
    opener = opener or urllib.request.urlopen
    destination = download_root / spec["relative"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists():
        current_size = destination.stat().st_size
        if current_size == spec["bytes"]:
            print(f"[verify] {spec['relative'].as_posix()}", flush=True)
            _verify_file(destination, spec)
            print(f"[ready] {spec['relative'].as_posix()}", flush=True)
            return destination
        if current_size < spec["bytes"] and not partial.exists():
            # Some download tools leave the incomplete payload at its final
            # name.  Adopt it as our stable resume file without discarding a
            # potentially multi-gigabyte prefix.
            destination.replace(partial)
        else:
            raise RuntimeError(
                f"archive byte count differs and cannot be resumed: {destination}"
            )
    if partial.exists() and partial.stat().st_size > spec["bytes"]:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == spec["bytes"]:
        try:
            _verify_file(partial, spec, "partial archive")
        except RuntimeError:
            partial.unlink()
            offset = 0
        else:
            partial.replace(destination)
            return destination

    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(spec["url"], headers=headers)
    print(
        f"[download] {spec['relative'].as_posix()} from byte {offset}",
        flush=True,
    )
    try:
        response = opener(request)
        with response:
            status = _response_status(response)
            append = bool(offset and status == 206)
            if offset and status not in {200, 206}:
                raise RuntimeError(
                    f"server rejected archive resume at byte {offset}: HTTP {status}"
                )
            if append:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError("archive resume returned the wrong byte range")
            elif offset:
                offset = 0  # A 200 response ignored Range; restart safely.
            mode = "ab" if append else "wb"
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"cannot download archive: {spec['url']}") from error

    if partial.stat().st_size != spec["bytes"]:
        raise RuntimeError(
            f"archive download is incomplete: {partial.stat().st_size} / "
            f"{spec['bytes']} bytes"
        )
    try:
        _verify_file(partial, spec, "downloaded archive")
    except RuntimeError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination


def download_all(download_root: Path, sources, workers=2):
    if type(workers) is not int or workers < 1:
        raise RuntimeError("download_workers must be a positive integer")
    unique = [
        (source_key, role, spec)
        for source_key, source in sources.items()
        for role, spec in source["archives"].items()
    ]
    paths = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(unique)),
                            thread_name_prefix="public-mono-download") as executor:
        futures = {
            executor.submit(download_archive, download_root, spec):
            (source_key, role)
            for source_key, role, spec in unique
        }
        for future in as_completed(futures):
            source_key, role = futures[future]
            path = future.result()
            paths[(source_key, role)] = path
    return paths


def _safe_zip_member(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        return False
    path = PurePosixPath(name)
    return (not path.is_absolute() and
            all(part not in {"", ".", ".."} for part in path.parts))


def index_archive(path: Path, spec):
    """Return sequence -> numeric frame -> exact ZIP member name."""
    rows = {}
    names = set()
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not _safe_zip_member(info.filename):
                    raise RuntimeError(f"unsafe ZIP member: {info.filename!r}")
                folded = info.filename.casefold()
                if folded in names:
                    raise RuntimeError(f"duplicate ZIP member: {info.filename}")
                names.add(folded)
                if info.is_dir():
                    continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise RuntimeError(f"ZIP symlink is not allowed: {info.filename}")
                if info.flag_bits & 0x1:
                    raise RuntimeError(f"encrypted ZIP member: {info.filename}")
                match = spec["pattern"].fullmatch(info.filename)
                if match is None:
                    raise RuntimeError(
                        f"unexpected file in {spec['role']} archive: {info.filename}"
                    )
                sequence = match.group("sequence")
                frame_text = match.group("frame")
                frame = int(frame_text)
                sequence_rows = rows.setdefault(sequence, {})
                if frame in sequence_rows:
                    raise RuntimeError(
                        f"duplicate source frame {sequence}/{frame_text}"
                    )
                if info.file_size <= 0:
                    raise RuntimeError(f"empty ZIP member: {info.filename}")
                sequence_rows[frame] = info.filename
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise RuntimeError(f"cannot index ZIP archive: {path}") from error
    if not rows:
        raise RuntimeError(f"archive has no admitted members: {path}")
    for sequence, frames in rows.items():
        ordered = sorted(frames)
        if any(right != left + 1 for left, right in zip(ordered, ordered[1:])):
            raise RuntimeError(f"{spec['role']} sequence {sequence} has frame gaps")
    return rows


def deterministic_development(sequences, count, salt):
    if count >= len(sequences):
        raise RuntimeError("development split would consume the training archive")
    ranked = sorted(
        sequences,
        key=lambda sequence: (
            hashlib.sha256(
                (salt + "\0" + sequence).encode("utf-8")
            ).digest(),
            sequence,
        ),
    )
    return set(ranked[:count])


def label_frame_ids(frame_count):
    if type(frame_count) is not int or frame_count < 5:
        raise RuntimeError("a source sequence needs at least five frames")
    # Integer half-up interpolation avoids platform/NumPy rounding behavior.
    ids = [(index * (frame_count - 1) + 2) // 4 for index in range(5)]
    if len(set(ids)) != 5 or ids[0] != 0 or ids[-1] != frame_count - 1:
        raise RuntimeError("cannot choose five unique evenly spaced labels")
    return ids


def _png_contract(data, decode=True):
    if (len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or
            data[12:16] != b"IHDR"):
        raise RuntimeError("source member is not a valid PNG")
    length = struct.unpack(">I", data[8:12])[0]
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    if length != 13 or width <= 0 or height <= 0:
        raise RuntimeError("source PNG has an invalid IHDR")
    if bit_depth != 8 or color_type != 2:
        raise RuntimeError("source PNG must be truecolor RGB8 without alpha")
    if decode:
        try:
            with Image.open(io.BytesIO(data)) as image:
                if (image.format != "PNG" or image.mode != "RGB" or
                        image.size != (width, height)):
                    raise RuntimeError("source PNG does not decode as RGB8")
                image.load()
        except (OSError, ValueError) as error:
            raise RuntimeError("source PNG does not decode as RGB8") from error
    return int(width), int(height)


def _copy_png(path: Path, data: bytes):
    width, height = _png_contract(data)
    with path.open("xb") as stream:
        if stream.write(data) != len(data):
            raise RuntimeError(f"short write while copying source PNG: {path}")
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
    }


class _BoundedPngCopier:
    """Parallel PNG validation/writes with a strict pending-memory bound."""

    def __init__(self, workers):
        if type(workers) is not int or workers < 1:
            raise RuntimeError("workers must be a positive integer")
        self.workers = workers
        self.max_pending = workers * 2
        self.pending = deque()
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="public-mono-copy"
        )

    def submit(self, path, data, metadata):
        completed = []
        while len(self.pending) >= self.max_pending:
            old_meta, future = self.pending.popleft()
            completed.append({**old_meta, **future.result()})
        future = self.executor.submit(_copy_png, path, data)
        self.pending.append((metadata, future))
        return completed

    def wait(self):
        completed = []
        while self.pending:
            metadata, future = self.pending.popleft()
            completed.append({**metadata, **future.result()})
        return completed

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def _archive_identity(spec):
    return {
        "role": spec["role"],
        "relative_path": spec["relative"].as_posix(),
        "url": spec["url"],
        "bytes": spec["bytes"],
        "hash_algorithm": spec["hash_algorithm"],
        "hash": spec["hash"],
    }


def _production_id(source_key, split):
    return f"{source_key}_mono_v1_{split}"


def _clip_name(source_key, split, sequence):
    return f"{source_key}_{split}_{_safe_component(sequence, 'sequence id')}"


def _split_contract(source, split, sequences, config_sha256, rgb_role,
                    disparity_role=None):
    archives = [_archive_identity(source["archives"][rgb_role])]
    if disparity_role:
        archives.append(_archive_identity(source["archives"][disparity_role]))
    return {
        "schema": 1,
        "preparation_contract": PREPARATION_CONTRACT,
        "public_sources_sha256": config_sha256,
        "source_key": source["key"],
        "source_version": source["version"],
        "split": split,
        "production_id": _production_id(source["key"], split),
        "split_salt": source["split_salt"],
        "development_count": source["development_count"],
        "split_assignment": (
            "rank sha256(split_salt + NUL + source_sequence); first "
            "development_count train-archive sequences are development"
        ),
        "archives": archives,
        "source_sequences": sorted(sequences),
        "label_rule": "five-evenly-spaced-including-endpoints",
        "rgb_copy": "exact-zip-member-bytes-rgb8-png",
        "disparity_admission": (
            "five-label-frames-only" if disparity_role else "none"
        ),
    }


def _read_json(path: Path, description):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid {description}: {path}")
    return value


def validate_sequence_directory(clip_root: Path, record, expected_sequence,
                                expected_frame_ids, expect_disparity,
                                expected_clip=None, verify_contents=True):
    expected_clip = expected_clip or clip_root.name
    if (record.get("schema") != 1 or
            record.get("preparation_contract") != PREPARATION_CONTRACT or
            record.get("source_sequence") != expected_sequence or
            record.get("clip") != expected_clip):
        raise RuntimeError(f"stale sequence record: {clip_root}")
    frames = record.get("rgb_frames")
    if not isinstance(frames, list) or len(frames) != len(expected_frame_ids):
        raise RuntimeError(f"sequence frame record differs: {clip_root}")
    if [row.get("source_frame") for row in frames] != expected_frame_ids:
        raise RuntimeError(f"sequence source-frame IDs differ: {clip_root}")
    expected_files = {"meta.json", LABEL_MANIFEST, SEQUENCE_RECORD}
    dimensions = set()
    for local_id, row in enumerate(frames):
        expected_name = f"frame_{local_id:05d}.png"
        if row.get("local_frame") != local_id or row.get("output") != expected_name:
            raise RuntimeError(f"sequence local-frame mapping differs: {clip_root}")
        path = clip_root / expected_name
        if not path.is_file():
            raise RuntimeError(f"RGB frame is missing: {path}")
        if path.stat().st_size != row.get("bytes"):
            raise RuntimeError(f"RGB frame byte count differs: {path}")
        if verify_contents:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != row.get("sha256"):
                raise RuntimeError(f"RGB frame identity differs: {path}")
            width, height = _png_contract(data, decode=False)
        else:
            width, height = row.get("width"), row.get("height")
            if (type(width) is not int or type(height) is not int or
                    width <= 0 or height <= 0):
                raise RuntimeError(f"RGB frame dimensions are invalid: {path}")
        dimensions.add((width, height))
        expected_files.add(expected_name)
    if len(dimensions) != 1:
        raise RuntimeError(f"sequence resolution changed: {clip_root}")
    labels = label_frame_ids(len(frames))
    label_payload = _read_json(clip_root / LABEL_MANIFEST, "label manifest")
    if label_payload != {"schema": 1, "frame_ids": labels}:
        raise RuntimeError(f"label manifest differs: {clip_root}")
    disparity = record.get("disparity_frames", [])
    disparity_root = clip_root / "gt_disparity"
    if expect_disparity:
        if not isinstance(disparity, list) or len(disparity) != 5:
            raise RuntimeError(f"sparse disparity record differs: {clip_root}")
        expected_disparity = set()
        for local_id, row in zip(labels, disparity):
            name = f"frame_{local_id:05d}.dsp5"
            if row.get("local_frame") != local_id or row.get("output") != name:
                raise RuntimeError(f"sparse disparity mapping differs: {clip_root}")
            disparity_path = disparity_root / name
            if (not disparity_path.is_file() or
                    disparity_path.stat().st_size != row.get("bytes")):
                raise RuntimeError(
                    f"disparity byte count differs: {disparity_path}"
                )
            if verify_contents and hash_file(disparity_path) != row.get("sha256"):
                raise RuntimeError(
                    f"disparity identity differs: {disparity_path}"
                )
            expected_disparity.add(name)
        actual = {path.name for path in disparity_root.iterdir() if path.is_file()}
        if actual != expected_disparity:
            raise RuntimeError(f"sparse disparity file set differs: {clip_root}")
        expected_files.add("gt_disparity")
    elif disparity or disparity_root.exists():
        raise RuntimeError(f"unexpected disparity sidecars: {clip_root}")
    actual_files = {path.name for path in clip_root.iterdir()}
    if actual_files != expected_files:
        raise RuntimeError(f"sequence output file set differs: {clip_root}")
    return next(iter(dimensions))


def _prepare_sequence(stage: Path, source, split, sequence, frame_members,
                      rgb_archive: Path, rgb_spec, copier,
                      disparity_members=None, disparity_archive=None,
                      disparity_spec=None):
    clip = _clip_name(source["key"], split, sequence)
    destination = stage / clip
    ordered_ids = sorted(frame_members)
    if destination.is_dir():
        record = _read_json(destination / SEQUENCE_RECORD, "sequence record")
        validate_sequence_directory(
            destination, record, sequence, ordered_ids,
            disparity_members is not None,
        )
        return record
    partial = stage / f".{clip}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    frame_rows = []
    labels = label_frame_ids(len(ordered_ids))
    try:
        with zipfile.ZipFile(rgb_archive) as archive:
            for local_id, source_id in enumerate(ordered_ids):
                member = frame_members[source_id]
                data = archive.read(member)
                metadata = {
                    "local_frame": local_id,
                    "source_frame": source_id,
                    "member": member,
                    "output": f"frame_{local_id:05d}.png",
                }
                frame_rows.extend(copier.submit(
                    partial / metadata["output"], data, metadata
                ))
            frame_rows.extend(copier.wait())
        frame_rows.sort(key=lambda row: row["local_frame"])
        dimensions = {(row["width"], row["height"]) for row in frame_rows}
        if len(dimensions) != 1:
            raise RuntimeError(f"source resolution changed in sequence {sequence}")
        width, height = next(iter(dimensions))
        disparity_rows = []
        if disparity_members is not None:
            disparity_root = partial / "gt_disparity"
            disparity_root.mkdir()
            with zipfile.ZipFile(disparity_archive) as archive:
                for local_id in labels:
                    source_id = ordered_ids[local_id]
                    member = disparity_members.get(source_id)
                    if member is None:
                        raise RuntimeError(
                            f"Spring disparity missing label {sequence}/{source_id}"
                        )
                    data = archive.read(member)
                    if not data:
                        raise RuntimeError("Spring disparity sidecar is empty")
                    output = f"frame_{local_id:05d}.dsp5"
                    output_path = disparity_root / output
                    with output_path.open("xb") as stream:
                        if stream.write(data) != len(data):
                            raise RuntimeError(
                                f"short write while copying disparity: {output_path}"
                            )
                    disparity_rows.append({
                        "local_frame": local_id,
                        "source_frame": source_id,
                        "member": member,
                        "output": output,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    })
        production = _production_id(source["key"], split)
        record = {
            "schema": 1,
            "preparation_contract": PREPARATION_CONTRACT,
            "clip": clip,
            "source_sequence": sequence,
            "source_archive": _archive_identity(rgb_spec),
            "source_frame_count": len(frame_rows),
            "source_first_frame": ordered_ids[0],
            "source_last_frame": ordered_ids[-1],
            "width": width,
            "height": height,
            "label_frame_ids": labels,
            "source_label_frame_ids": [ordered_ids[index] for index in labels],
            "rgb_frames": frame_rows,
            "disparity_archive": (
                _archive_identity(disparity_spec) if disparity_spec else None
            ),
            "disparity_frames": disparity_rows,
        }
        meta = {
            "schema": 2,
            "name": clip,
            "dataset": source["dataset"],
            "domain": source["domain"],
            "production_id": production,
            "film_id": production,
            "homepage": source["homepage"],
            "license": source["license"],
            "purpose": "artistic-policy monocular render-feasibility supervision",
            "policy_role": "cinematic_training",
            "global_policy_weight": source["global_policy_weight"],
            "source_kind": "mono-video",
            "source_container": "image-sequence-archives",
            "temporal_contract": "full-cadence-shot",
            "source_sequence": sequence,
            "source_start_frame": ordered_ids[0],
            "source_end_frame": ordered_ids[-1],
            "context_fps": source["context_fps"],
            "context_frame_count": len(frame_rows),
            "label_sampling": "five-evenly-spaced-including-endpoints",
            "label_frame_count": 5,
            "source_width": width,
            "source_height": height,
            "split": split,
            "required_gt_stereo": False,
            "required_temporal_evidence": True,
            "color_contract": "decoded-sdr-bgr8",
            "source_color_contract": "rgb8-png-byte-exact",
            "auxiliary_disparity": (
                "spring-dsp5-label-frames-only" if disparity_rows else None
            ),
        }
        _write_json(partial / LABEL_MANIFEST,
                    {"schema": 1, "frame_ids": labels})
        _write_json(partial / "meta.json", meta)
        _write_json(partial / SEQUENCE_RECORD, record)
        validate_sequence_directory(
            partial, record, sequence, ordered_ids,
            disparity_members is not None, expected_clip=clip,
            verify_contents=False,
        )
        partial.replace(destination)
        return record
    except BaseException:
        # Keep the split staging root and completed sibling sequences.  This
        # incomplete sequence is discarded and will be retried on resume.
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        raise


def _dataset_sequence_row(record, split):
    return {
        "clip": record["clip"],
        "source_sequence": record["source_sequence"],
        "source_start_frame": record["source_first_frame"],
        "source_end_frame": record["source_last_frame"],
        "context_frames": record["source_frame_count"],
        "label_frames": len(record["label_frame_ids"]),
        "split": split,
    }


def _validate_published_split(root: Path, contract, frame_index,
                              expect_disparity, verify_contents=True):
    if not root.is_dir():
        raise RuntimeError(f"prepared split is missing: {root}")
    current_contract = _read_json(root / PREPARATION_MANIFEST,
                                  "preparation contract")
    if current_contract != contract:
        raise RuntimeError(f"prepared split contract is stale: {root}")
    source_manifest = _read_json(root / SOURCE_SEQUENCE_MANIFEST,
                                 "source sequence manifest")
    dataset = _read_json(root / DATASET_MANIFEST, "dataset manifest")
    if dataset.get("video_sha256") != hash_file(root / SOURCE_SEQUENCE_MANIFEST):
        raise RuntimeError(f"prepared source identity differs: {root}")
    records = source_manifest.get("sequences")
    if not isinstance(records, list):
        raise RuntimeError(f"prepared source sequence list is invalid: {root}")
    by_source = {row.get("source_sequence"): row for row in records
                 if isinstance(row, dict)}
    if set(by_source) != set(contract["source_sequences"]):
        raise RuntimeError(f"prepared source sequence set differs: {root}")
    clip_names = set()
    for sequence in contract["source_sequences"]:
        record = by_source[sequence]
        clip_root = root / record.get("clip", "")
        disk_record = _read_json(clip_root / SEQUENCE_RECORD, "sequence record")
        if disk_record != record:
            raise RuntimeError(f"aggregate sequence record differs: {clip_root}")
        validate_sequence_directory(
            clip_root, record, sequence, sorted(frame_index[sequence]),
            expect_disparity, verify_contents=verify_contents,
        )
        clip_names.add(clip_root.name)
    expected_root = clip_names | {
        PREPARATION_MANIFEST, SOURCE_SEQUENCE_MANIFEST, DATASET_MANIFEST,
    }
    if {path.name for path in root.iterdir()} != expected_root:
        raise RuntimeError(f"prepared split contains unexpected entries: {root}")
    if dataset.get("sequences") != [
            _dataset_sequence_row(by_source[sequence], contract["split"])
            for sequence in contract["source_sequences"]]:
        raise RuntimeError(f"dataset sequence summary differs: {root}")
    return dataset


def _publish_stage(stage: Path, destination: Path, overwrite_stale):
    if not destination.exists():
        stage.replace(destination)
        return
    if not overwrite_stale:
        raise RuntimeError(f"prepared destination is stale: {destination}")
    backup = destination.with_name(
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    destination.replace(backup)
    try:
        stage.replace(destination)
    except BaseException:
        if not destination.exists():
            backup.replace(destination)
        raise
    shutil.rmtree(backup)


def prepare_split(prepared_root: Path, source, split, sequences,
                  frame_index, rgb_archive: Path, rgb_spec,
                  config_sha256, workers=4, disparity_index=None,
                  disparity_archive=None, disparity_spec=None,
                  overwrite_stale=False):
    dataset_family = prepared_root / f"{source['key']}-mono-v1"
    dataset_family.mkdir(parents=True, exist_ok=True)
    destination = dataset_family / split
    sequences = sorted(sequences)
    contract = _split_contract(
        source, split, sequences, config_sha256, rgb_spec["role"],
        disparity_spec["role"] if disparity_spec else None,
    )
    if destination.exists():
        try:
            return _validate_published_split(
                destination, contract, frame_index,
                disparity_index is not None,
            )
        except RuntimeError:
            if not overwrite_stale:
                raise
    stage = dataset_family / f".{split}.partial"
    if stage.exists():
        try:
            if (_read_json(stage / PREPARATION_MANIFEST,
                           "staged preparation contract") != contract):
                raise RuntimeError("staged preparation contract is stale")
        except RuntimeError:
            if not overwrite_stale:
                raise
            shutil.rmtree(stage)
    if not stage.exists():
        stage.mkdir()
        _write_json(stage / PREPARATION_MANIFEST, contract)
    for partial in stage.glob(".*.partial"):
        if partial.is_dir():
            shutil.rmtree(partial)

    copier = _BoundedPngCopier(workers)
    records = []
    try:
        for sequence in sequences:
            if sequence not in frame_index:
                raise RuntimeError(f"missing indexed RGB sequence: {sequence}")
            disparity_members = None
            if disparity_index is not None:
                disparity_members = disparity_index.get(sequence)
                if disparity_members is None:
                    raise RuntimeError(
                        f"Spring disparity sequence missing: {sequence}"
                    )
            records.append(_prepare_sequence(
                stage, source, split, sequence, frame_index[sequence],
                rgb_archive, rgb_spec, copier, disparity_members,
                disparity_archive, disparity_spec,
            ))
            print(
                f"[{source['key']}/{split}] {len(records)}/{len(sequences)} "
                f"source sequence {sequence}",
                flush=True,
            )
    finally:
        copier.close()

    sequence_manifest = {
        "schema": 1,
        "preparation_contract": PREPARATION_CONTRACT,
        "source_container": "image-sequence-archives",
        "public_sources_sha256": config_sha256,
        "source_key": source["key"],
        "source_version": source["version"],
        "dataset": source["dataset"],
        "domain": source["domain"],
        "production_id": contract["production_id"],
        "split": split,
        "split_salt": contract["split_salt"],
        "split_assignment": contract["split_assignment"],
        "archives": contract["archives"],
        "context_fps": source["context_fps"],
        "label_rule": contract["label_rule"],
        "sequences": records,
        "context_frame_count": sum(row["source_frame_count"] for row in records),
        "label_frame_count": 5 * len(records),
    }
    _write_json(stage / SOURCE_SEQUENCE_MANIFEST, sequence_manifest)
    source_identity = hash_file(stage / SOURCE_SEQUENCE_MANIFEST)
    dataset_rows = [_dataset_sequence_row(record, split) for record in records]
    dataset_manifest = {
        "schema": 2,
        "dataset": source["dataset"],
        "domain": source["domain"],
        "production_id": contract["production_id"],
        "source_kind": "mono-video",
        "source_container": "image-sequence-archives",
        "source_sequence_manifest": SOURCE_SEQUENCE_MANIFEST,
        "temporal_contract": "full-cadence-shot",
        "policy_role": "cinematic_training",
        "homepage": source["homepage"],
        "license": source["license"],
        "license_url": source["license_url"],
        "split": split,
        "context_fps": source["context_fps"],
        "global_policy_weight": source["global_policy_weight"],
        "color_contract": "decoded-sdr-bgr8",
        "source_color_contract": "rgb8-png-byte-exact",
        "video_sha256": source_identity,
        "sequences": dataset_rows,
        "shot_count": len(dataset_rows),
        "context_frame_count": sequence_manifest["context_frame_count"],
        "label_frame_count": sequence_manifest["label_frame_count"],
        "split_rule": "source sequences are assigned once by salted identity",
    }
    _write_json(stage / DATASET_MANIFEST, dataset_manifest)
    _validate_published_split(
        stage, contract, frame_index, disparity_index is not None,
        verify_contents=False,
    )
    _publish_stage(stage, destination, overwrite_stale)
    return dataset_manifest


def _expected_counts(source):
    override = source.get("expected_sequence_counts")
    if override is None:
        return OFFICIAL_SEQUENCE_COUNTS[source["key"]]
    if (not isinstance(override, dict) or
            set(override) != {"train_rgb", "test_rgb"} or
            any(type(value) is not int or value <= 0
                for value in override.values())):
        raise RuntimeError("invalid expected_sequence_counts test contract")
    return override


def _catalog_row(source, split, manifest_path, sources_path):
    production = _production_id(source["key"], split)
    return {
        "id": production,
        "production_id": production,
        "source_kind": "mono-video",
        "source_group": source["source_group"],
        "split": split,
        "admission": "global_policy",
        "complete_production": True,
        "global_policy_weight": source["global_policy_weight"],
        "dataset": source["dataset"],
        "homepage": source["homepage"],
        "license": source["license"],
        "license_url": source["license_url"],
        "dataset_manifest": str(manifest_path.resolve()),
        "retrieval": {
            "kind": "pinned-public-image-sequence-archives",
            "source_manifest": str(sources_path.resolve()),
        },
    }


def prepare_public_sources(sources_path: Path, download_root: Path,
                           prepared_root: Path, catalog_output: Path,
                           active_output: Path, workers=4,
                           download_workers=2, overwrite_stale=False,
                           selected_sources=None, generate_metadata=True):
    _validate_roots(
        sources_path, download_root, prepared_root,
        catalog_output, active_output,
    )
    all_sources, config_sha256 = load_sources(sources_path)
    selected = (set(all_sources) if selected_sources is None else
                set(selected_sources))
    if not selected or not selected.issubset(all_sources):
        raise RuntimeError("selected_sources must contain REDS and/or Spring")
    if generate_metadata and selected != set(all_sources):
        raise RuntimeError(
            "the six-row catalog requires both REDS and Spring; use prepare-only"
        )
    if generate_metadata and (catalog_output is None or active_output is None):
        raise RuntimeError("catalog and active-split outputs are required")
    sources = {key: all_sources[key] for key in all_sources if key in selected}
    archive_paths = download_all(download_root, sources, download_workers)
    indexes = {}
    for source_key, source in sources.items():
        for role, spec in source["archives"].items():
            indexes[(source_key, role)] = index_archive(
                archive_paths[(source_key, role)], spec
            )
        expected = _expected_counts(source)
        for role, count in expected.items():
            actual = len(indexes[(source_key, role)])
            if actual != count:
                raise RuntimeError(
                    f"{source_key}/{role} sequence count differs: "
                    f"{actual} != {count}"
                )
        if source_key == "spring":
            rgb = indexes[(source_key, "train_rgb")]
            disparity = indexes[(source_key, "train_disparity")]
            if set(rgb) != set(disparity):
                raise RuntimeError(
                    "Spring training RGB/disparity sequence sets differ"
                )
            for sequence in rgb:
                if set(rgb[sequence]) != set(disparity[sequence]):
                    raise RuntimeError(
                        f"Spring training RGB/disparity frames differ: {sequence}"
                    )

    manifests = []
    for source_key, source in sources.items():
        train_index = indexes[(source_key, "train_rgb")]
        test_index = indexes[(source_key, "test_rgb")]
        development = deterministic_development(
            train_index, source["development_count"], source["split_salt"]
        )
        training = set(train_index) - development
        assignments = {
            "training": (training, train_index, "train_rgb"),
            "development": (development, train_index, "train_rgb"),
            "test": (set(test_index), test_index, "test_rgb"),
        }
        for split in SPLITS:
            sequences, frame_index, role = assignments[split]
            disparity_index = None
            disparity_path = None
            disparity_spec = None
            if source_key == "spring" and split != "test":
                disparity_index = indexes[(source_key, "train_disparity")]
                disparity_path = archive_paths[(source_key, "train_disparity")]
                disparity_spec = source["archives"]["train_disparity"]
            prepare_split(
                prepared_root, source, split, sequences, frame_index,
                archive_paths[(source_key, role)], source["archives"][role],
                config_sha256, workers, disparity_index, disparity_path,
                disparity_spec, overwrite_stale,
            )
            manifests.append(
                prepared_root / f"{source_key}-mono-v1" / split /
                DATASET_MANIFEST
            )

    if not generate_metadata:
        datasets = [
            _read_json(path, "dataset manifest") for path in manifests
        ]
        return {
            "catalog": None,
            "active_split": None,
            "dataset_manifests": [str(path.resolve()) for path in manifests],
            "context_frames": sum(
                row["context_frame_count"] for row in datasets
            ),
            "label_frames": sum(row["label_frame_count"] for row in datasets),
        }

    catalog = {
        "schema": 2,
        "purpose": (
            "Active public monocular full-cadence sources for artistic-policy "
            "training, development, and sealed testing"
        ),
        "sources": [
            _catalog_row(source, split,
                         prepared_root / f"{source_key}-mono-v1" / split /
                         DATASET_MANIFEST, sources_path)
            for source_key, source in all_sources.items()
            for split in SPLITS
        ],
    }
    artistic_sources.validate_catalog(catalog, "generated public mono catalog")
    _write_json(catalog_output, catalog)
    active = split_audit.audit(catalog_output, manifests)
    _write_json(active_output, active)
    return {
        "catalog": str(catalog_output.resolve()),
        "active_split": str(active_output.resolve()),
        "dataset_manifests": [str(path.resolve()) for path in manifests],
        "context_frames": active["totals"]["context_frames"],
        "label_frames": active["totals"]["label_frames"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--catalog-output", type=Path)
    parser.add_argument("--active-split-output", type=Path)
    parser.add_argument(
        "--source", action="append", choices=("reds", "spring"),
        help="prepare only selected source(s); default prepares both",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="prepare selected archives without freezing the six-row catalog",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--overwrite-stale", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and (
            args.catalog_output is None or args.active_split_output is None):
        parser.error(
            "--catalog-output and --active-split-output are required unless "
            "--prepare-only is used"
        )
    result = prepare_public_sources(
        args.sources, args.download_root, args.prepared_root,
        args.catalog_output, args.active_split_output,
        args.workers, args.download_workers, args.overwrite_stale,
        args.source, not args.prepare_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
