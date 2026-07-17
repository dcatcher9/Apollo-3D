#!/usr/bin/env python3
"""Build and verify frozen SBS clip-identity manifests.

The evaluator's historical ``sha1_dir`` identity reads every source and
reference frame.  Prepared suites are tens of gigabytes, so repeating that
work for every experiment is wasteful.  This module computes the historical
identity and a stronger per-file manifest once, then permits a cheap
path/size/mtime verification on later runs.  ``--verify`` re-reads content and
checks every SHA-256 when a full audit is wanted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent


def canonical_json_sha256(value):
    """Hash one strict, stable JSON representation for semantic identities."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


MANIFEST_NAME = "clip_hash_manifest.json"
MANIFEST_SCHEMA = 2
MANIFEST_CONTRACT = "apollo-sbs-clip-hash-manifest-v2"
MANIFEST_CONTENT_CONTRACT = "apollo-sbs-clip-semantic-content-v1"
MANIFEST_CONTENT_SHA256_FIELD = "semantic_content_sha256"
CONTENT_PATTERNS = (
    "frame_*",
    "gt_depth/frame_*",
    "gt_flow/frame_*",
    "gt_right/frame_*",
    "label_frames.json",
)
SEMANTIC_META_FIELDS = (
    "expected_flat",
    "gt_depth_kind",
    "dataset",
    "required_gt_depth",
    "required_gt_flow",
    "required_gt_stereo",
)
SEMANTIC_CONTRACT = {
    "identity": "legacy-sha1-dir-12",
    "content_patterns": list(CONTENT_PATTERNS),
    "semantic_meta_fields": list(SEMANTIC_META_FIELDS),
    "relative_path_separator": "/",
    "semantic_json": "json.dumps(sort_keys=True)",
}
SEMANTIC_CONTRACT_SHA256 = canonical_json_sha256(SEMANTIC_CONTRACT)
SHA1_RE = re.compile(r"^[0-9a-f]{12}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ClipHashManifestError(RuntimeError):
    pass


def _absolute(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _real(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _directory_identity(path):
    stat = os.stat(path)
    return {
        "path": _absolute(path),
        "real_path": _real(path),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _validate_clip_name(name):
    if (not isinstance(name, str) or not name or name in {".", ".."} or
            os.path.basename(name) != name or "/" in name or "\\" in name):
        raise ClipHashManifestError(f"invalid clip name: {name!r}")
    return name


def selected_clip_names(clips_root, clips=None):
    root = Path(clips_root)
    if not root.is_dir():
        raise ClipHashManifestError(f"clips root is missing: {root}")
    names = list(clips) if clips is not None else sorted(
        item.name for item in root.iterdir() if item.is_dir()
    )
    if not names:
        raise ClipHashManifestError("clip selection is empty")
    if len(names) != len(set(names)):
        raise ClipHashManifestError("clip selection contains duplicates")
    for name in names:
        _validate_clip_name(name)
        if not (root / name).is_dir():
            raise ClipHashManifestError(f"clip directory is missing: {name}")
    return sorted(names)


def semantic_file_paths(clip_dir):
    """Return the exact ordered file set consumed by historical ``sha1_dir``."""

    root = os.fspath(clip_dir)
    paths = []
    for pattern in CONTENT_PATTERNS:
        paths.extend(glob.glob(os.path.join(root, *pattern.split("/"))))
    return sorted(paths)


def _semantic_meta(meta_path):
    try:
        with open(meta_path, encoding="utf-8") as stream:
            payload = json.load(stream)
        return {
            key: payload[key] for key in SEMANTIC_META_FIELDS if key in payload
        }, "valid"
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError):
        return None, "invalid"


def sha1_dir(path):
    """Compute the exact legacy evaluator clip identity."""

    root = os.fspath(path)
    digest = hashlib.sha1()
    for file_path in semantic_file_paths(root):
        relative = os.path.relpath(file_path, root).replace("\\", "/")
        digest.update(relative.encode())
        with open(file_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    semantic, state = _semantic_meta(os.path.join(root, "meta.json"))
    if state == "valid":
        digest.update(json.dumps(semantic, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def _hash_file(file_path, clip_root, legacy_digest=None, capture=False):
    before = os.stat(file_path)
    before_real = _real(file_path)
    sha256 = hashlib.sha256()
    captured = [] if capture else None
    with open(file_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            if legacy_digest is not None:
                legacy_digest.update(chunk)
            if captured is not None:
                captured.append(chunk)
    after = os.stat(file_path)
    after_real = _real(file_path)
    stable = (
        before.st_size == after.st_size and
        before.st_mtime_ns == after.st_mtime_ns and
        before.st_dev == after.st_dev and
        before.st_ino == after.st_ino and
        before_real == after_real
    )
    if not stable:
        raise ClipHashManifestError(f"file changed while hashing: {file_path}")
    record = {
        "path": os.path.relpath(file_path, clip_root).replace("\\", "/"),
        "real_path": before_real,
        "size": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "sha256": sha256.hexdigest(),
    }
    return record, (b"".join(captured) if captured is not None else None)


def build_clip_entry(clips_root, clip_name):
    root = _absolute(clips_root)
    clip_name = _validate_clip_name(clip_name)
    clip_root = os.path.join(root, clip_name)
    if not os.path.isdir(clip_root):
        raise ClipHashManifestError(f"clip directory is missing: {clip_name}")

    legacy = hashlib.sha1()
    records = []
    for file_path in semantic_file_paths(clip_root):
        relative = os.path.relpath(file_path, clip_root).replace("\\", "/")
        legacy.update(relative.encode())
        record, _captured = _hash_file(file_path, clip_root, legacy)
        records.append(record)

    meta_path = os.path.join(clip_root, "meta.json")
    semantic = None
    meta_state = "missing"
    meta_record = None
    if os.path.isfile(meta_path):
        meta_record, meta_bytes = _hash_file(meta_path, clip_root, capture=True)
        try:
            payload = json.loads(meta_bytes.decode("utf-8"))
            semantic = {
                key: payload[key] for key in SEMANTIC_META_FIELDS
                if key in payload
            }
            meta_state = "valid"
        except (UnicodeError, ValueError):
            meta_state = "invalid"
    if meta_state == "valid":
        legacy.update(json.dumps(semantic, sort_keys=True).encode())

    return {
        "clip_sha1": legacy.hexdigest()[:12],
        "clip_path": _absolute(clip_root),
        "clip_real_path": _real(clip_root),
        "files": records,
        "meta_file": meta_record,
        "meta_state": meta_state,
        "semantic_meta": semantic,
    }


def _build_clip_task(arguments):
    return arguments[1], build_clip_entry(*arguments)


def semantic_content_payload(manifest):
    """Return the path/time-independent semantic and content identity.

    The on-disk manifest deliberately carries creation time, source paths and
    stat data for provenance and cheap change detection.  None of those values
    identify the evaluator inputs.  This projection includes the exact
    semantic file paths/content plus the metadata fields consumed by the
    evaluator, so rebuilding or relocating an unchanged manifest preserves its
    cache identity.
    """

    entries = manifest.get("clips")
    if not isinstance(entries, dict) or not entries:
        raise ClipHashManifestError("clip hash manifest has no clips")
    clips = {}
    for name in sorted(entries):
        _validate_clip_name(name)
        entry = entries[name]
        if not isinstance(entry, dict):
            raise ClipHashManifestError(f"invalid clip manifest entry: {name}")
        files = entry.get("files")
        if not isinstance(files, list):
            raise ClipHashManifestError(
                f"clip manifest has no semantic file records: {name}"
            )
        content_records = []
        seen = set()
        for record in files:
            if not isinstance(record, dict):
                raise ClipHashManifestError(
                    f"invalid semantic file record: {name}"
                )
            relative = record.get("path")
            size = record.get("size")
            digest = record.get("sha256")
            if (not isinstance(relative, str) or not relative or
                    relative in seen or
                    not isinstance(size, int) or isinstance(size, bool) or
                    size < 0 or not isinstance(digest, str) or
                    not SHA256_RE.fullmatch(digest)):
                raise ClipHashManifestError(
                    f"invalid semantic file identity: {name}"
                )
            seen.add(relative)
            content_records.append({
                "path": relative,
                "size": size,
                "sha256": digest,
            })
        identity = entry.get("clip_sha1")
        if not isinstance(identity, str) or not SHA1_RE.fullmatch(identity):
            raise ClipHashManifestError(f"invalid clip SHA-1 for {name}")
        clips[name] = {
            "clip_sha1": identity,
            "files": sorted(content_records, key=lambda item: item["path"]),
            "meta_state": entry.get("meta_state"),
            "semantic_meta": entry.get("semantic_meta"),
        }
    return {
        "contract": MANIFEST_CONTENT_CONTRACT,
        "semantic_contract": manifest.get("semantic_contract"),
        "semantic_contract_sha256": manifest.get("semantic_contract_sha256"),
        "clips": clips,
    }


def semantic_content_sha256(manifest):
    try:
        return canonical_json_sha256(semantic_content_payload(manifest))
    except (TypeError, ValueError) as error:
        raise ClipHashManifestError(
            f"clip hash manifest semantic content is not canonical: {error}"
        ) from error


def build_manifest(clips_root, clips=None, workers=None):
    root = _absolute(clips_root)
    names = selected_clip_names(root, clips)
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if isinstance(workers, bool) or workers < 1:
        raise ClipHashManifestError("workers must be at least 1")

    tasks = [(root, name) for name in names]
    if workers == 1 or len(tasks) == 1:
        pairs = [_build_clip_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            pairs = list(pool.map(_build_clip_task, tasks))
    entries = {name: entry for name, entry in sorted(pairs)}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract": MANIFEST_CONTRACT,
        "semantic_contract": SEMANTIC_CONTRACT,
        "semantic_contract_sha256": SEMANTIC_CONTRACT_SHA256,
        "clips_root": _directory_identity(root),
        "clips": entries,
        "clip_count": len(entries),
    }
    manifest[MANIFEST_CONTENT_SHA256_FIELD] = semantic_content_sha256(manifest)
    manifest["created_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(timespec="seconds")
    return manifest


def write_manifest_atomic(manifest, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=output.parent,
                prefix=f".{output.name}.", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_and_write(clips_root, clips=None, workers=None, output=None):
    root = _absolute(clips_root)
    output = Path(output) if output else Path(root) / MANIFEST_NAME
    manifest = build_manifest(root, clips, workers)
    write_manifest_atomic(manifest, output)
    return manifest, output


def load_manifest(path):
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as error:
        raise ClipHashManifestError(
            f"cannot read clip hash manifest {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ClipHashManifestError("clip hash manifest is not an object")
    if (payload.get("schema") != MANIFEST_SCHEMA or
            payload.get("contract") != MANIFEST_CONTRACT or
            payload.get("semantic_contract") != SEMANTIC_CONTRACT or
            payload.get("semantic_contract_sha256") !=
            SEMANTIC_CONTRACT_SHA256):
        raise ClipHashManifestError("clip hash manifest contract is stale")
    expected_digest = payload.get(MANIFEST_CONTENT_SHA256_FIELD)
    if (not isinstance(expected_digest, str) or
            not SHA256_RE.fullmatch(expected_digest) or
            expected_digest != semantic_content_sha256(payload)):
        raise ClipHashManifestError(
            "clip hash manifest semantic content digest is invalid"
        )
    return payload


def _entry_records(entry):
    records = entry.get("files")
    if not isinstance(records, list):
        raise ClipHashManifestError("clip manifest has no file records")
    meta = entry.get("meta_file")
    if meta is not None:
        records = records + [meta]
    paths = [record.get("path") for record in records if isinstance(record, dict)]
    if len(paths) != len(records) or len(paths) != len(set(paths)):
        raise ClipHashManifestError("clip manifest has invalid file records")
    return {record["path"]: record for record in records}


def _current_relative_files(clip_root):
    paths = {
        os.path.relpath(path, clip_root).replace("\\", "/")
        for path in semantic_file_paths(clip_root)
    }
    meta = os.path.join(clip_root, "meta.json")
    if os.path.isfile(meta):
        paths.add("meta.json")
    return paths


def _safe_file_path(clip_root, relative):
    if (not isinstance(relative, str) or not relative or
            relative.startswith("/") or re.match(r"^[A-Za-z]:", relative)):
        raise ClipHashManifestError(f"unsafe manifest path: {relative!r}")
    candidate = os.path.abspath(os.path.join(clip_root, *relative.split("/")))
    if os.path.commonpath([_absolute(clip_root), candidate]) != _absolute(clip_root):
        raise ClipHashManifestError(f"unsafe manifest path: {relative!r}")
    return candidate


def _verify_record(clip_root, relative, record):
    if not isinstance(record, dict):
        raise ClipHashManifestError(f"invalid file record: {relative}")
    path = _safe_file_path(clip_root, relative)
    try:
        stat = os.stat(path)
    except OSError as error:
        raise ClipHashManifestError(f"manifest file is missing: {path}") from error
    expected = {
        "real_path": _real(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ClipHashManifestError(
                f"clip hash manifest {field} changed: {path}"
            )
    digest = record.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ClipHashManifestError(f"invalid SHA-256 in manifest: {path}")


def _verify_full_entry(root, name, expected, identity):
    current = build_clip_entry(root, name)
    if (current["meta_state"] != expected.get("meta_state") or
            current["semantic_meta"] != expected.get("semantic_meta")):
        raise ClipHashManifestError(f"semantic metadata changed: {name}")
    expected_records = _entry_records(expected)
    current_records = _entry_records(current)
    if set(expected_records) != set(current_records):
        raise ClipHashManifestError(
            f"clip hash manifest file set changed: {name}"
        )
    fields = (
        "real_path", "size", "mtime_ns", "device", "inode", "sha256"
    )
    for relative, expected_record in expected_records.items():
        current_record = current_records[relative]
        for field in fields:
            if current_record.get(field) != expected_record.get(field):
                label = "content hash" if field == "sha256" else field
                raise ClipHashManifestError(
                    f"clip hash manifest {label} changed: "
                    f"{os.path.join(root, name, relative)}"
                )
    if current["clip_sha1"] != identity:
        raise ClipHashManifestError(f"legacy clip identity changed: {name}")


def verify_selected_clips(manifest_path, clips_root, clips, full=False):
    """Validate selected entries and return their legacy clip hashes.

    Cheap verification performs no content reads: it verifies the logical and
    resolved roots, exact semantic file set, and every file's size, mtime,
    device and inode.  Full verification additionally checks per-file SHA-256
    and recomputes the legacy SHA-1 identity.
    """

    payload = load_manifest(manifest_path)
    root = _absolute(clips_root)
    names = selected_clip_names(root, clips)
    current_root = _directory_identity(root)
    if payload.get("clips_root") != current_root:
        raise ClipHashManifestError("clip hash manifest clips root changed")
    entries = payload.get("clips")
    if not isinstance(entries, dict):
        raise ClipHashManifestError("clip hash manifest has no clips")

    identities = {}
    for name in names:
        entry = entries.get(name)
        if not isinstance(entry, dict):
            raise ClipHashManifestError(
                f"clip hash manifest has no exact entry for {name}"
            )
        clip_root = os.path.join(root, name)
        if (entry.get("clip_path") != _absolute(clip_root) or
                entry.get("clip_real_path") != _real(clip_root)):
            raise ClipHashManifestError(
                f"clip hash manifest path changed: {name}"
            )
        identity = entry.get("clip_sha1")
        if not isinstance(identity, str) or not SHA1_RE.fullmatch(identity):
            raise ClipHashManifestError(f"invalid clip SHA-1 for {name}")

        if full:
            _verify_full_entry(root, name, entry, identity)
        else:
            records = _entry_records(entry)
            current_paths = _current_relative_files(clip_root)
            if set(records) != current_paths:
                raise ClipHashManifestError(
                    f"clip hash manifest file set changed: {name}"
                )
            for relative, record in records.items():
                _verify_record(clip_root, relative, record)
        identities[name] = identity
    return identities


def main():
    parser = argparse.ArgumentParser(
        description="Build a frozen prepared-suite clip hash manifest"
    )
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", help="clip names (default: all)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify", action="store_true",
        help="fully verify the manifest after writing it",
    )
    args = parser.parse_args()
    try:
        manifest, output = build_and_write(
            args.clips_root, args.clips, args.workers, args.output
        )
        if args.verify:
            verify_selected_clips(
                output, args.clips_root, sorted(manifest["clips"]), full=True
            )
    except ClipHashManifestError as error:
        parser.error(str(error))
    print(f"wrote {output} ({len(manifest['clips'])} clips)")


if __name__ == "__main__":
    main()
