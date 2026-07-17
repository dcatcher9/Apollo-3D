#!/usr/bin/env python3
"""Authenticated content-addressed cache for prepared training media.

The cache stores immutable directory artifacts produced from working
train/development sources.  Cache hits are accepted only after every payload
byte is rehashed, then materialized through a sibling staging directory and an
atomic rename.  A corrupt or stale object fails closed instead of being hidden
by an automatic rebuild.

This module deliberately knows nothing about sealed-test manifests.  Callers
must invoke :func:`require_working_split` before resolving, stat'ing, hashing,
probing, or decoding a source path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat


CACHE_SCHEMA = 1
CACHE_CONTRACT = "apollo-preprocessing-content-cache-v1"
KEY_SCHEMA = 1
KEY_CONTRACT = "apollo-preprocessing-content-key-v1"
MANIFEST_NAME = "cache_manifest.json"
PAYLOAD_DIRECTORY = "payload"
WORKING_SPLITS = frozenset({"training", "development"})


def is_link_or_junction(path):
    path = Path(path)
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    if os.name == "nt":
        try:
            attributes = path.lstat().st_file_attributes
        except (AttributeError, OSError):
            return False
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def _plain_path_chain(root, path):
    """Return false when an existing component below root is a reparse link."""
    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    if current.exists() and is_link_or_junction(current):
        return False
    for part in relative.parts:
        if part in {"", ".", ".."}:
            return False
        current /= part
        if current.exists() and is_link_or_junction(current):
            return False
    return True


def canonical_bytes(value):
    return (json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_snapshot(path):
    """Authenticate one source file and bind its stable filesystem identity.

    The stat tuple is sampled on both sides of the full byte hash so ordinary
    replacement/truncation during hashing fails closed.  Callers that decode a
    source later must verify this snapshot again after the decode and before
    publishing any artifact derived from it.
    """
    path = Path(path).resolve(strict=True)
    if not path.is_file() or is_link_or_junction(path):
        raise RuntimeError(f"preprocessing source is not a plain file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    fields = ("st_size", "st_mtime_ns", "st_dev", "st_ino")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError(f"preprocessing source changed while hashing: {path}")
    return {
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "device": after.st_dev,
        "inode": after.st_ino,
        "sha256": digest,
    }


def verify_source_file_snapshot(path, expected):
    """Fail closed if a decoded source no longer matches its first snapshot."""
    observed = source_file_snapshot(path)
    if observed != expected:
        raise RuntimeError(
            f"preprocessing source changed during generation: {Path(path)}"
        )
    return observed


def code_identities(paths):
    """Snapshot the loaded toolchain's on-disk Python source identities.

    Long-running decoders must use the same snapshot in their cache key and
    verify it again immediately before publication.  Otherwise an editor can
    change a live script halfway through a run and the old, already-imported
    implementation would be published under the new file hash.
    """
    rows = {
        str(role): sha256_file(Path(path).resolve(strict=True))
        for role, path in sorted(paths.items())
    }
    canonical_bytes(rows)
    return rows


def verify_code_identities(paths, expected):
    """Fail closed if any source file changed after identity snapshotting."""
    observed = code_identities(paths)
    if observed != expected:
        raise RuntimeError(
            "preprocessing implementation changed during generation"
        )


def require_disjoint_roots(cache_root, *other_roots):
    """Reject cache/source/output overlap before expensive source access."""
    cache = Path(cache_root).resolve(strict=False)
    for value in other_roots:
        if value is None:
            continue
        other = Path(value).resolve(strict=False)
        if (cache == other or cache.is_relative_to(other) or
                other.is_relative_to(cache)):
            raise RuntimeError(
                "preprocessing cache must not overlap source or output roots"
            )
    return cache


def require_working_split(split):
    """Reject sealed/unknown splits before a caller touches source media."""
    if split not in WORKING_SPLITS:
        raise RuntimeError(
            "preprocessing cache is train/development only; refusing source "
            f"split {split!r}"
        )
    return split


def cache_identity(*, artifact_kind, source, selection, preprocessing,
                   color_contract, code):
    """Return the complete semantic identity used as the object key."""
    if not isinstance(artifact_kind, str) or not artifact_kind:
        raise RuntimeError("preprocessing cache artifact kind is invalid")
    value = {
        "schema": KEY_SCHEMA,
        "contract": KEY_CONTRACT,
        "artifact_kind": artifact_kind,
        "source": source,
        "selection": selection,
        "preprocessing": preprocessing,
        "color_contract": color_contract,
        "code": code,
    }
    # This both checks JSON serializability and rejects NaN/Infinity.
    canonical_bytes(value)
    return value


def _payload_rows(root):
    root = Path(root)
    if is_link_or_junction(root):
        raise RuntimeError(f"cache payload is not a plain directory: {root}")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"cache payload is not a plain directory: {root}")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if is_link_or_junction(path):
            raise RuntimeError(f"cache payload contains a link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"cache payload contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not rows:
        raise RuntimeError("preprocessing cache payload is empty")
    return rows


def _validate_rows(root, expected):
    if not isinstance(expected, list) or not expected:
        raise RuntimeError("preprocessing cache file manifest is invalid")
    observed = _payload_rows(root)
    if observed != expected:
        raise RuntimeError("preprocessing cache payload bytes differ")
    return observed


def _remove(path):
    path = Path(path)
    if is_link_or_junction(path):
        if getattr(path, "is_junction", lambda: False)():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)
    elif path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _copy_tree(source, destination):
    source = Path(source)
    if is_link_or_junction(source):
        raise RuntimeError(f"cache source is not a plain directory: {source}")
    source = source.resolve(strict=True)
    destination = Path(destination)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def inheriting_temporary_directory(parent, prefix):
    """Create an unpredictable sibling directory that inherits parent ACLs.

    Python's Windows ``tempfile.mkdtemp`` may install a creator-only DACL.
    Renaming such a directory into a shared persistent cache makes it unreadable
    to later service/admin/sandbox identities.  ``Path.mkdir`` uses normal
    parent inheritance while the random name still prevents collisions.
    """
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(100):
        candidate = parent / f"{prefix}{secrets.token_hex(12)}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate cache staging directory under {parent}")


class DirectoryArtifactCache:
    """Immutable, byte-validated directory object store."""

    def __init__(self, root):
        root = Path(root)
        if root.exists() and is_link_or_junction(root):
            raise RuntimeError(
                f"preprocessing cache root is not a plain directory: {root}"
            )
        self.root = root.resolve(strict=False)

    @staticmethod
    def key(identity):
        if (not isinstance(identity, dict) or
                identity.get("schema") != KEY_SCHEMA or
                identity.get("contract") != KEY_CONTRACT):
            raise RuntimeError("preprocessing cache key identity is invalid")
        return canonical_sha256(identity)

    def _entry(self, key):
        return self.root / "objects" / key[:2] / key

    def _manifest(self, entry, identity, rows):
        return {
            "schema": CACHE_SCHEMA,
            "contract": CACHE_CONTRACT,
            "key_sha256": self.key(identity),
            "identity": identity,
            "file_count": len(rows),
            "payload_bytes": sum(row["bytes"] for row in rows),
            "payload_manifest_sha256": canonical_sha256(rows),
            "files": rows,
        }

    def _validate_entry(self, identity):
        key = self.key(identity)
        entry = self._entry(key)
        manifest_path = entry / MANIFEST_NAME
        payload = entry / PAYLOAD_DIRECTORY
        if not entry.exists():
            return None
        if (not _plain_path_chain(self.root, entry) or
                is_link_or_junction(entry) or not entry.is_dir()):
            raise RuntimeError(f"preprocessing cache entry is invalid: {entry}")
        children = {path.name: path for path in entry.iterdir()}
        if (set(children) != {MANIFEST_NAME, PAYLOAD_DIRECTORY} or
                is_link_or_junction(manifest_path) or
                not manifest_path.is_file() or
                not _plain_path_chain(self.root, payload)):
            raise RuntimeError(
                f"preprocessing cache entry layout is invalid: {entry}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"cannot read preprocessing cache manifest: {manifest_path}"
            ) from error
        if (not isinstance(manifest, dict) or
                manifest.get("schema") != CACHE_SCHEMA or
                manifest.get("contract") != CACHE_CONTRACT or
                manifest.get("key_sha256") != key or
                manifest.get("identity") != identity):
            raise RuntimeError("preprocessing cache identity differs")
        rows = _validate_rows(payload, manifest.get("files"))
        if (manifest.get("file_count") != len(rows) or
                manifest.get("payload_bytes") !=
                sum(row["bytes"] for row in rows) or
                manifest.get("payload_manifest_sha256") !=
                canonical_sha256(rows)):
            raise RuntimeError("preprocessing cache manifest is stale")
        return entry, manifest

    def materialize(self, identity, destination):
        """Materialize a verified hit atomically; return ``False`` on miss."""
        validated = self._validate_entry(identity)
        if validated is None:
            return False
        entry, manifest = validated
        destination = Path(destination).resolve(strict=False)
        if destination.exists() or is_link_or_junction(destination):
            raise RuntimeError(
                f"cache destination must not exist: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = inheriting_temporary_directory(
            destination.parent, f".{destination.name}.cache-partial-"
        )
        # The helper creates the directory, whereas copytree requires it absent.
        staging.rmdir()
        try:
            _copy_tree(entry / PAYLOAD_DIRECTORY, staging)
            _validate_rows(staging, manifest["files"])
            if destination.exists() or is_link_or_junction(destination):
                raise RuntimeError(
                    f"cache destination appeared during materialization: "
                    f"{destination}"
                )
            staging.replace(destination)
        finally:
            _remove(staging)
        return True

    def validated_payload(self, identity):
        """Return the immutable payload path for a verified hit, or ``None``.

        Callers that can consume immutable artifacts in place avoid a potentially
        multi-gigabyte copy.  Every payload byte is still rehashed before the path is
        returned; consumers must treat the returned directory as strictly read-only.
        """
        validated = self._validate_entry(identity)
        return None if validated is None else validated[0] / PAYLOAD_DIRECTORY

    def validated_payload_receipt(self, identity):
        """Return a verified immutable payload plus its authenticated receipt.

        The receipt lets an in-place consumer pin a specific inner manifest (or
        other payload file) across the process-launch boundary.  Returning a deep
        copy prevents a caller from accidentally mutating the validation result.
        Consumers must still authenticate the bytes they read against the receipt;
        the payload directory is not a materialized snapshot.
        """
        validated = self._validate_entry(identity)
        if validated is None:
            return None
        entry, manifest = validated
        receipt = {
            "key_sha256": manifest["key_sha256"],
            "payload_manifest_sha256": manifest["payload_manifest_sha256"],
            "files": copy.deepcopy(manifest["files"]),
        }
        return entry / PAYLOAD_DIRECTORY, receipt

    def publish(self, identity, source):
        """Publish one immutable object atomically and return its key."""
        key = self.key(identity)
        source = Path(source)
        if is_link_or_junction(source):
            raise RuntimeError(
                f"cache source is not a plain directory: {source}"
            )
        source = source.resolve(strict=True)
        if (source == self.root or self.root.is_relative_to(source) or
                source.is_relative_to(self.root)):
            raise RuntimeError("preprocessing cache overlaps its source artifact")
        source_rows = _payload_rows(source)
        existing = self._validate_entry(identity)
        if existing is not None:
            if existing[1]["files"] != source_rows:
                raise RuntimeError(
                    "preprocessing cache concurrent producer payload differs"
                )
            return key
        entry = self._entry(key)
        entry.parent.mkdir(parents=True, exist_ok=True)
        if not _plain_path_chain(self.root, entry.parent):
            raise RuntimeError("preprocessing cache path contains a link")
        staging = inheriting_temporary_directory(
            entry.parent, f".{key}.partial-"
        )
        try:
            payload = staging / PAYLOAD_DIRECTORY
            _copy_tree(source, payload)
            rows = _payload_rows(payload)
            if rows != source_rows:
                raise RuntimeError(
                    "preprocessing source changed during cache publication"
                )
            manifest = self._manifest(entry, identity, rows)
            (staging / MANIFEST_NAME).write_bytes(canonical_bytes(manifest))
            _validate_rows(payload, rows)
            try:
                staging.replace(entry)
            except OSError:
                # Another producer may have atomically won the same key.
                winner = self._validate_entry(identity)
                if winner is None:
                    raise
                if winner[1]["files"] != rows:
                    raise RuntimeError(
                        "preprocessing cache concurrent producer payload differs"
                    )
            published = self._validate_entry(identity)
            if published is None or published[1]["files"] != rows:
                raise RuntimeError(
                    "preprocessing cache published payload differs"
                )
        finally:
            _remove(staging)
        return key
