#!/usr/bin/env python3
"""Path-independent cache packets for compacted ordinal safety results.

The multiscale renderer produces large, short-lived batches.  This cache does
not retain those batches.  It stores only the authenticated compacted result
trees that remain after scoring: results, selected-frame gates, runtime-scene
evidence, multiscale provenance, and the three sparse visual-evidence scales.

Run labels and workspace paths are presentation details, not scoring inputs.
They are replaced with explicit tokens before publication and rebound while a
cache hit is materialized.  Every digest that depends on those fields is then
recomputed.  The resulting trees therefore satisfy the ordinary validator in
their new workspace without trusting stale absolute paths.
"""

from __future__ import annotations

import copy
import contextlib
import datetime
import hashlib
import importlib.machinery
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile

import preprocessing_artifact_cache as artifact_cache


PACKET_SCHEMA = 2
PACKET_CONTRACT = "apollo-ordinal-compacted-score-packet-v2"
RUNTIME_IDENTITY_SCHEMA = 3
RUNTIME_IDENTITY_CONTRACT = "apollo-ordinal-scorer-runtime-v3"
ARTIFACT_KIND = "ordinal-compacted-scored-batch"
RESULTS_TEMPLATE = "results.template.json"
GATE_TEMPLATE = "frame_gate_evidence.template.jsonl"
PACKET_MANIFEST = "ordinal_score_packet.json"
COMPACTION_FILENAME = "ordinal_safety_compaction.json"
FRAME_GATE_FILENAME = "frame_gate_evidence.jsonl"
RESULTS_FILENAME = "results.json"
RUN_TOKEN = "@APOLLO_ORDINAL_RUN_NAME@"
CLIPS_ROOT_TOKEN = "@APOLLO_ORDINAL_CLIPS_ROOT@"
CLIP_MANIFEST_TOKEN = "@APOLLO_ORDINAL_CLIP_MANIFEST@"
TIMESTAMP_TOKEN = "@APOLLO_ORDINAL_SCORE_TIMESTAMP@"
GIT_SHA_TOKEN = "@APOLLO_ORDINAL_SCORE_GIT_SHA@"
GIT_DIRTY_TOKEN = "@APOLLO_ORDINAL_SCORE_GIT_DIRTY@"
DEPTH_CACHE_MODE_TOKEN = "@APOLLO_DEPTH_STATE_EXECUTION_MODE@"
ESTIMATOR_CALLS_TOKEN = "@APOLLO_ESTIMATOR_CALLS_PER_SOURCE_FRAME@"
CACHED_DEPTH_MODE = "scored-result-cache"
MATERIALIZATION_TRANSACTION_SCHEMA = 1
MATERIALIZATION_TRANSACTION_CONTRACT = \
    "apollo-score-cache-materialization-transaction-v1"
MATERIALIZATION_TRANSACTION_ID = re.compile(r"[0-9a-f]{32}")
SHA256 = re.compile(r"[0-9a-f]{64}")
CONTRACT_DIGEST = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")
PROVENANCE_FILES = frozenset({
    "contract.json",
    "multiscale_batch_manifest.json",
    "multiscale_contract.json",
    "render_identity.json",
})


def _native_binary_identities(module, role, *, package_search=False,
                              sibling_directories=()):
    """Bind loaded extension bytes, never a package's Python wrapper."""
    suffixes = tuple(
        suffix.lower() for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ) + (".pyd", ".so", ".dll", ".dylib")

    def native(path):
        name = Path(path).name.lower()
        return any(name.endswith(suffix) for suffix in suffixes)

    candidates = set()
    for value in (
            getattr(module, "__file__", None),
            getattr(getattr(module, "__spec__", None), "origin", None)):
        if value and native(value) and Path(value).is_file():
            candidates.add(Path(value).resolve(strict=True))
    if package_search:
        package_roots = [
            Path(value).resolve(strict=True)
            for value in getattr(module, "__path__", ())
            if Path(value).is_dir()
        ]
        for root in package_roots:
            candidates.update(
                path.resolve(strict=True) for path in root.rglob("*")
                if path.is_file() and native(path)
            )
            for name in sibling_directories:
                sibling = root.parent / name
                if sibling.is_dir():
                    candidates.update(
                        path.resolve(strict=True) for path in sibling.rglob("*")
                        if path.is_file() and native(path)
                    )
    if not candidates:
        raise RuntimeError(
            f"ordinal score cache {role} native module is unavailable"
        )
    return sorted(({
        "filename": path.name,
        "filename_suffix": "".join(path.suffixes).lower(),
        "sha256": artifact_cache.sha256_file(path),
    } for path in candidates), key=lambda row: (
        row["filename"].casefold(), row["sha256"]
    ))


def _python_source_identities(module, role):
    """Bind the pure-Python package code used around native extensions."""
    roots = [
        Path(value).resolve(strict=True)
        for value in getattr(module, "__path__", ())
        if Path(value).is_dir()
    ]
    rows = []
    for root_index, root in enumerate(sorted(roots, key=lambda path: str(path))):
        for path in _plain_tree_files(root, f"{role} package"):
            if path.suffix.lower() != ".py":
                continue
            rows.append({
                "package_root_index": root_index,
                "path": path.relative_to(root).as_posix(),
                "sha256": artifact_cache.sha256_file(path),
            })
    if not rows:
        raise RuntimeError(
            f"ordinal score cache {role} Python source is unavailable"
        )
    return rows


def _python_runtime_binaries():
    """Bind the launcher and the real base interpreter shared libraries."""
    candidates = {Path(sys.executable).resolve(strict=True)}
    base = Path(sys.base_prefix).resolve(strict=True)
    if os.name == "nt":
        candidates.update(
            path.resolve(strict=True) for path in base.glob("python*.dll")
            if path.is_file()
        )
    else:
        library = sysconfig.get_config_var("LDLIBRARY")
        library_dir = sysconfig.get_config_var("LIBDIR")
        if library and library_dir:
            path = Path(library_dir) / library
            if path.is_file():
                candidates.add(path.resolve(strict=True))
    return sorted(({
        "filename": path.name,
        "sha256": artifact_cache.sha256_file(path),
    } for path in candidates), key=lambda row: (
        row["filename"].casefold(), row["sha256"]
    ))


def _validate_runtime_identity(value):
    if (not isinstance(value, dict) or
            value.get("schema") != RUNTIME_IDENTITY_SCHEMA or
            value.get("contract") != RUNTIME_IDENTITY_CONTRACT):
        raise RuntimeError("ordinal score cache scorer runtime is invalid")
    python = value.get("python")
    packages = value.get("packages")
    if (not isinstance(python, dict) or
            not isinstance(python.get("runtime_binaries"), list) or
            not python["runtime_binaries"] or
            not isinstance(packages, dict)):
        raise RuntimeError("ordinal score cache scorer runtime is invalid")
    for binary in python["runtime_binaries"]:
        if (not isinstance(binary, dict) or
                not isinstance(binary.get("filename"), str) or
                not binary["filename"] or
                not SHA256.fullmatch(str(binary.get("sha256", "")))):
            raise RuntimeError("ordinal score cache Python runtime is invalid")
    for name in ("numpy", "pillow"):
        row = packages.get(name)
        if (not isinstance(row, dict) or not isinstance(row.get("version"), str) or
                not row["version"] or
                not isinstance(row.get("native_binaries"), list) or
                not row["native_binaries"] or
                not isinstance(row.get("python_sources"), list) or
                not row["python_sources"]):
            raise RuntimeError(
                f"ordinal score cache {name} runtime is invalid"
            )
        for binary in row["native_binaries"]:
            if (not isinstance(binary, dict) or
                    not isinstance(binary.get("filename"), str) or
                    not binary["filename"] or
                    not isinstance(binary.get("filename_suffix"), str) or
                    not SHA256.fullmatch(str(binary.get("sha256", "")))):
                raise RuntimeError(
                    f"ordinal score cache {name} native runtime is invalid"
                )
        for source in row["python_sources"]:
            if (not isinstance(source, dict) or
                    not isinstance(source.get("package_root_index"), int) or
                    not isinstance(source.get("path"), str) or
                    not source["path"] or
                    not SHA256.fullmatch(str(source.get("sha256", "")))):
                raise RuntimeError(
                    f"ordinal score cache {name} Python runtime is invalid"
                )
        if (name == "numpy" and
                not SHA256.fullmatch(str(row.get("build_sha256", "")))):
            raise RuntimeError(
                "ordinal score cache NumPy build runtime is invalid"
            )
    if set(packages) != {"numpy", "pillow"}:
        raise RuntimeError("ordinal score cache package runtime is invalid")
    artifact_cache.canonical_bytes(value)
    return value


def scorer_runtime_identity():
    """Return a path-free identity for the process that computes metrics.

    Version strings alone do not distinguish differently built numerical or
    image backends.  Bind the interpreter and imported native extension bytes,
    plus hashed build information where the package exposes it.
    """
    try:
        import numpy as np
        import PIL
    except ImportError as error:
        raise RuntimeError(
            "ordinal score cache scorer runtime lacks NumPy or Pillow"
        ) from error

    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.__config__.show()
    # Metrics use np.fft and np.linalg in addition to core ufuncs.  Those are
    # separate extension modules, and NumPy's BLAS backend is commonly shipped
    # in a sibling ``numpy.libs`` directory.  Pillow likewise dispatches image
    # codecs through multiple native extensions.  Hash the complete native
    # package closure rather than only whichever extension happened to be
    # imported first.
    np_binary = _native_binary_identities(
        np, "NumPy", package_search=True,
        sibling_directories=("numpy.libs",),
    )
    pillow_binary = _native_binary_identities(
        PIL, "Pillow", package_search=True,
        sibling_directories=("pillow.libs", "PIL.libs"),
    )
    value = {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "contract": RUNTIME_IDENTITY_CONTRACT,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": getattr(sys.implementation, "cache_tag", None),
            "soabi": sysconfig.get_config_var("SOABI"),
            "byteorder": sys.byteorder,
            "system": platform.system(),
            "machine": platform.machine(),
            "runtime_binaries": _python_runtime_binaries(),
        },
        "packages": {
            "numpy": {
                "version": str(np.__version__),
                "native_binaries": np_binary,
                "python_sources": _python_source_identities(np, "NumPy"),
                "build_sha256": hashlib.sha256(
                    numpy_config.getvalue().encode("utf-8")
                ).hexdigest(),
                },
            "pillow": {
                "version": str(PIL.__version__),
                "native_binaries": pillow_binary,
                "python_sources": _python_source_identities(PIL, "Pillow"),
            },
        },
    }
    return _validate_runtime_identity(value)


def query_scorer_runtime_identity(python_path):
    """Query the exact interpreter selected for child scoring."""
    python_path = Path(python_path).resolve(strict=True)
    try:
        same_process = os.path.samefile(python_path, Path(sys.executable))
    except OSError:
        same_process = False
    if same_process:
        return scorer_runtime_identity()
    completed = subprocess.run(
        [str(python_path), str(Path(__file__).resolve()),
         "--print-runtime-identity"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True, text=True, timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(
            "cannot query ordinal scorer runtime: " + completed.stderr[-2000:]
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "ordinal scorer runtime query returned invalid JSON"
        ) from error
    return _validate_runtime_identity(value)


def _require_sha256(value, role):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise RuntimeError(f"ordinal score cache {role} is invalid")
    return value


def _require_contract_digest(value, role):
    if not isinstance(value, str) or not CONTRACT_DIGEST.fullmatch(value):
        raise RuntimeError(f"ordinal score cache {role} is invalid")
    return value


def _scale_slug(scale):
    value = float(scale)
    scaled = round(value * 100.0)
    if not math.isfinite(value) or not math.isclose(
            value * 100.0, scaled, abs_tol=1e-6):
        raise RuntimeError(f"ordinal score cache scale is invalid: {scale!r}")
    return f"s{scaled:03d}"


def _scale_row(scale):
    value = float(scale)
    return {
        "scale": value,
        "scale_slug": _scale_slug(value),
        "float32_bits": struct.unpack("<I", struct.pack("<f", value))[0],
    }


def scored_cache_identity(*, split, render_identity_sha256, metric_sha256,
                          thresholds_sha256, scales, artifact_scales,
                          contracts):
    """Build the complete path-independent identity for one scored grid."""
    artifact_cache.require_working_split(split)
    render_sha = _require_sha256(
        render_identity_sha256, "render identity sha256"
    )
    metric_sha = _require_contract_digest(metric_sha256, "metric sha256")
    thresholds_sha = _require_sha256(
        thresholds_sha256, "thresholds sha256"
    )
    scale_rows = [_scale_row(scale) for scale in scales]
    if (not scale_rows or
            [row["scale"] for row in scale_rows] != sorted({
                row["scale"] for row in scale_rows
            })):
        raise RuntimeError(
            "ordinal score cache scales must be unique and increasing"
        )
    artifact_rows = [_scale_row(scale) for scale in artifact_scales]
    if (len(artifact_rows) != len({
            row["scale"] for row in artifact_rows
            }) or [row["scale"] for row in artifact_rows] != sorted(
                row["scale"] for row in artifact_rows
            ) or not {row["scale"] for row in artifact_rows}.issubset({
                row["scale"] for row in scale_rows
            })):
        raise RuntimeError(
            "ordinal score cache artifact scales must be unique, increasing, "
            "and part of the scale grid"
        )
    if (not isinstance(contracts, dict) or not contracts or
            any(not isinstance(key, str) or not key for key in contracts)):
        raise RuntimeError("ordinal score cache contracts are invalid")
    artifact_cache.canonical_bytes(contracts)
    return artifact_cache.cache_identity(
        artifact_kind=ARTIFACT_KIND,
        source={
            "working_split": split,
            "render_identity_sha256": render_sha,
        },
        selection={
            "scale_rows": scale_rows,
            "artifact_scale_rows": artifact_rows,
        },
        preprocessing={
            "metric_sha256": metric_sha,
            "thresholds_sha256": thresholds_sha,
            "scoring_contracts": contracts,
        },
        color_contract={
            "binding": "authenticated-by-render-identity",
        },
        code={
            "packet_schema": PACKET_SCHEMA,
            "packet_contract": PACKET_CONTRACT,
        },
    )


def _validate_identity(identity):
    if (not isinstance(identity, dict) or
            identity.get("artifact_kind") != ARTIFACT_KIND):
        raise RuntimeError("ordinal score cache identity is invalid")
    source = identity.get("source")
    selection = identity.get("selection")
    scoring = identity.get("preprocessing")
    code = identity.get("code")
    if not isinstance(source, dict):
        raise RuntimeError("ordinal score cache source identity is invalid")
    artifact_cache.require_working_split(source.get("working_split"))
    _require_sha256(
        source.get("render_identity_sha256"), "render identity sha256"
    )
    if (not isinstance(selection, dict) or
            not isinstance(selection.get("scale_rows"), list) or
            not selection["scale_rows"]):
        raise RuntimeError("ordinal score cache scale identity is invalid")
    scale_rows = selection["scale_rows"]
    artifact_rows = selection.get("artifact_scale_rows")
    for row in scale_rows:
        if row != _scale_row(row.get("scale")):
            raise RuntimeError("ordinal score cache scale row differs")
    if (not isinstance(artifact_rows, list) or
            any(row != _scale_row(row.get("scale")) for row in artifact_rows) or
            not {row["scale_slug"] for row in artifact_rows}.issubset({
                row["scale_slug"] for row in scale_rows
            })):
        raise RuntimeError("ordinal score cache artifact scale row differs")
    if not isinstance(scoring, dict):
        raise RuntimeError("ordinal score cache scoring identity is invalid")
    _require_contract_digest(scoring.get("metric_sha256"), "metric sha256")
    _require_sha256(scoring.get("thresholds_sha256"), "thresholds sha256")
    if code != {
            "packet_schema": PACKET_SCHEMA,
            "packet_contract": PACKET_CONTRACT}:
        raise RuntimeError("ordinal score cache packet identity is stale")
    # DirectoryArtifactCache performs its own stricter key-contract check.
    artifact_cache.DirectoryArtifactCache.key(identity)
    return identity


def _load_json(path, role):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read ordinal score cache {role}: {path}") \
            from error
    if not isinstance(value, dict):
        raise RuntimeError(f"ordinal score cache {role} is not an object")
    return value


def _load_canonical_json(path, role):
    value = _load_json(path, role)
    if Path(path).read_bytes() != artifact_cache.canonical_bytes(value):
        raise RuntimeError(f"ordinal score cache {role} is noncanonical")
    return value


def _replace_exact_strings(value, replacements):
    """Replace authenticated digest/origin tokens without substring edits."""
    if isinstance(value, dict):
        return {
            key: _replace_exact_strings(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_exact_strings(child, replacements) for child in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _provenance_root(output, clip):
    return Path(output) / "multiscale_provenance" / clip


def _project_execution_origin(provenance_roots, *, projected_mode,
                              projected_estimator_calls):
    """Project cold/replay provenance onto one deterministic representation.

    Only execution-origin fields are changed.  Depth-state key/manifest,
    rendered artifacts, source identity, contracts, and scale geometry remain
    authenticated.  All dependent hashes are rebuilt rather than ignored.
    """
    if (projected_mode not in {DEPTH_CACHE_MODE_TOKEN, CACHED_DEPTH_MODE} or
            projected_estimator_calls not in {
                ESTIMATOR_CALLS_TOKEN, 0
            }):
        raise RuntimeError("ordinal score cache origin projection is invalid")
    roots = {str(slug): Path(root) for slug, root in provenance_roots.items()}
    if not roots:
        raise RuntimeError("ordinal score cache provenance grid is empty")

    contracts = {}
    original_contracts = {}
    common_harness = None
    common_harness_sha = None
    common_manifest = None
    common_receipt = None
    common_origin = None
    for slug, root in roots.items():
        contract_path = root / "contract.json"
        harness_path = root / "multiscale_contract.json"
        manifest_path = root / "multiscale_batch_manifest.json"
        receipt_path = root / "render_identity.json"
        # The C++ harness owns these two files and intentionally writes
        # human-readable JSON; their exact source bytes remain authenticated by
        # the surrounding manifest before we project them to canonical JSON.
        contract = _load_json(contract_path, "scale contract")
        harness = _load_json(harness_path, "harness contract")
        manifest = _load_canonical_json(manifest_path, "batch manifest")
        receipt = _load_canonical_json(receipt_path, "render receipt")
        original_mode = contract.get("depth_state_cache_mode")
        depth = harness.get("depth_state_cache")
        manifest_depth = manifest.get("depth_state_cache")
        calls = harness.get("shipping_estimator_calls_per_source_frame")
        if (not isinstance(depth, dict) or manifest_depth != depth or
                contract.get("depth_state_cache_key_sha256") !=
                depth.get("key_sha256") or
                contract.get("depth_state_manifest_sha256") !=
                depth.get("manifest_sha256") or
                original_mode != depth.get("mode")):
            raise RuntimeError(
                f"ordinal score cache depth provenance differs: {slug}"
            )
        expected_calls = {
            "disabled": 1,
            "cold-export": 1,
            "authenticated-replay": 0,
            DEPTH_CACHE_MODE_TOKEN: ESTIMATOR_CALLS_TOKEN,
            CACHED_DEPTH_MODE: 0,
        }
        if original_mode not in expected_calls or calls != expected_calls[
                original_mode]:
            raise RuntimeError(
                f"ordinal score cache estimator provenance differs: {slug}"
            )
        origin = {
            "mode": original_mode,
            "estimator_calls_per_source_frame": calls,
            "key_sha256": depth.get("key_sha256"),
            "manifest_sha256": depth.get("manifest_sha256"),
        }
        if common_origin is None:
            common_origin = origin
        elif origin != common_origin:
            raise RuntimeError(
                "ordinal score cache origin changes within a scale grid"
            )
        for role, value in (
                ("harness", harness), ("manifest", manifest),
                ("receipt", receipt)):
            current = artifact_cache.canonical_bytes(value)
            previous = {
                "harness": common_harness,
                "manifest": common_manifest,
                "receipt": common_receipt,
            }[role]
            if previous is not None and current != previous:
                raise RuntimeError(
                    f"ordinal score cache {role} changes across scales"
                )
            if role == "harness":
                common_harness = current
                current_sha = artifact_cache.sha256_file(harness_path)
                if (common_harness_sha is not None and
                        current_sha != common_harness_sha):
                    raise RuntimeError(
                        "ordinal score cache harness bytes change across scales"
                    )
                common_harness_sha = current_sha
            elif role == "manifest":
                common_manifest = current
            else:
                common_receipt = current
        original_contracts[slug] = {
            "value": contract,
            "sha256": artifact_cache.sha256_file(contract_path),
            "bytes": contract_path.stat().st_size,
        }
        projected = copy.deepcopy(contract)
        projected["depth_state_cache_mode"] = projected_mode
        encoded = artifact_cache.canonical_bytes(projected)
        contracts[slug] = {
            "value": projected,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        }

    harness = json.loads(common_harness)
    original_harness_sha = common_harness_sha
    harness["shipping_estimator_calls_per_source_frame"] = \
        projected_estimator_calls
    harness["depth_state_cache"]["mode"] = projected_mode
    harness_bytes = artifact_cache.canonical_bytes(harness)
    harness_sha = hashlib.sha256(harness_bytes).hexdigest()

    manifest = json.loads(common_manifest)
    original_manifest_sha = hashlib.sha256(common_manifest).hexdigest()
    if manifest.get("harness_contract_sha256") != original_harness_sha:
        raise RuntimeError("ordinal score cache harness digest differs")
    manifest["harness_contract_sha256"] = harness_sha
    manifest["depth_state_cache"]["mode"] = projected_mode
    manifest_rows = manifest.get("scale_rows")
    if not isinstance(manifest_rows, list):
        raise RuntimeError("ordinal score cache manifest scale grid is missing")
    seen = set()
    for row in manifest_rows:
        directory = row.get("directory") if isinstance(row, dict) else None
        slug = Path(directory or "").name
        if slug not in contracts or slug in seen:
            raise RuntimeError(
                "ordinal score cache manifest scale grid differs"
            )
        seen.add(slug)
        original = original_contracts[slug]
        projected = contracts[slug]
        if row.get("contract_sha256") != original["sha256"]:
            raise RuntimeError(
                f"ordinal score cache scale contract digest differs: {slug}"
            )
        row["contract_sha256"] = projected["sha256"]
        contract_path = f"scales/{slug}/contract.json"
        artifact_rows = [
            item for item in row.get("artifacts", ())
            if isinstance(item, dict) and item.get("path") == contract_path
        ]
        if (len(artifact_rows) != 1 or
                artifact_rows[0].get("sha256") != original["sha256"] or
                artifact_rows[0].get("size") != original["bytes"]):
            raise RuntimeError(
                f"ordinal score cache contract artifact differs: {slug}"
            )
        artifact_rows[0]["sha256"] = projected["sha256"]
        artifact_rows[0]["size"] = projected["bytes"]
    if seen != set(contracts):
        raise RuntimeError("ordinal score cache manifest scale grid is incomplete")
    manifest_bytes = artifact_cache.canonical_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    receipt = json.loads(common_receipt)
    original_receipt_sha = hashlib.sha256(common_receipt).hexdigest()
    if receipt.get("batch_manifest_sha256") != original_manifest_sha:
        raise RuntimeError("ordinal score cache render receipt digest differs")
    receipt["batch_manifest_sha256"] = manifest_sha
    receipt_bytes = artifact_cache.canonical_bytes(receipt)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()

    rows = {}
    for slug in sorted(contracts):
        rows[slug] = {
            "contract": contracts[slug]["value"],
            "harness": harness,
            "manifest": manifest,
            "receipt": receipt,
            "replacements": {
                original_contracts[slug]["sha256"]:
                    contracts[slug]["sha256"],
                original_harness_sha: harness_sha,
                original_manifest_sha: manifest_sha,
                original_receipt_sha: receipt_sha,
            },
        }
    return {
        "rows": rows,
        "original_origin": common_origin,
        "projected": {
            "mode": projected_mode,
            "estimator_calls_per_source_frame": projected_estimator_calls,
        },
    }


def _write_projected_provenance(output, clip, row):
    root = _provenance_root(output, clip)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "contract.json": row["contract"],
        "multiscale_contract.json": row["harness"],
        "multiscale_batch_manifest.json": row["manifest"],
        "render_identity.json": row["receipt"],
    }
    for name, value in files.items():
        (root / name).write_bytes(artifact_cache.canonical_bytes(value))


def _canonical_jsonl(records):
    return b"".join(artifact_cache.canonical_bytes(record) for record in records)


def _read_gate(path):
    try:
        raw = Path(path).read_bytes()
        lines = raw.splitlines(keepends=True)
        records = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read ordinal score cache frame gate") from error
    if (len(records) < 3 or any(not isinstance(row, dict) for row in records) or
            _canonical_jsonl(records) != raw):
        raise RuntimeError("ordinal score cache frame gate is noncanonical")
    header, trailer = records[0], records[-1]
    payload = _canonical_jsonl(records[:-1])
    if (header.get("record") != "header" or
            trailer.get("record") != "trailer" or
            trailer.get("payload_record_count") != len(records) - 1 or
            trailer.get("payload_sha256") !=
            hashlib.sha256(payload).hexdigest()):
        raise RuntimeError("ordinal score cache frame gate digest differs")
    return records


def _write_gate(path, records):
    records = copy.deepcopy(records)
    payload = _canonical_jsonl(records[:-1])
    trailer = records[-1]
    trailer["payload_record_count"] = len(records) - 1
    trailer["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    Path(path).write_bytes(payload + artifact_cache.canonical_bytes(trailer))


def _plain_tree_files(root, role):
    """Walk a tree without following symlinks, junctions, or reparse points."""
    root = Path(root)
    if (artifact_cache.is_link_or_junction(root) or
            not root.is_dir()):
        raise RuntimeError(
            f"ordinal score cache {role} is not a plain directory: {root}"
        )
    files = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if artifact_cache.is_link_or_junction(path):
                raise RuntimeError(
                    f"ordinal score cache {role} contains a link: {path}"
                )
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.append(path)
            else:
                raise RuntimeError(
                    f"ordinal score cache {role} contains a special file: "
                    f"{path}"
                )
    return sorted(files, key=lambda path: path.as_posix())


def _tree_rows(root, relative_to):
    root = Path(root)
    if not root.is_dir():
        return {}
    return {
        path.resolve().relative_to(Path(relative_to).resolve()).as_posix():
            artifact_cache.sha256_file(path)
        for path in _plain_tree_files(root, "evidence")
    }


def _directory_rows(root):
    unresolved = Path(root)
    files = _plain_tree_files(unresolved, "packet verification")
    root = unresolved.resolve(strict=True)
    return [{
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": artifact_cache.sha256_file(path),
    } for path in files]


def _path_present(path):
    path = Path(path)
    return (
        path.exists() or path.is_symlink() or
        getattr(path, "is_junction", lambda: False)()
    )


def _normalized_path(value):
    return str(Path(value).resolve(strict=False)).replace("\\", "/").casefold()


def _scan_for_stale_strings(value, forbidden, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            _scan_for_stale_strings(child, forbidden, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_for_stale_strings(child, forbidden, path + (str(index),))
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").casefold()
        if any(item and item in normalized for item in forbidden):
            raise RuntimeError(
                "ordinal score cache template retains a workspace/run path at "
                + ".".join(path)
            )


def _validate_compacted_tree(output, clip, marker_contract):
    unresolved = Path(output)
    files = _plain_tree_files(unresolved, "compacted output")
    output = unresolved.resolve(strict=True)
    marker = _load_json(output / COMPACTION_FILENAME, "compaction marker")
    results = output / RESULTS_FILENAME
    gate = output / FRAME_GATE_FILENAME
    runtime = output / clip / "runtime_scene_evidence.json"
    provenance = _tree_rows(output / "multiscale_provenance" / clip, output)
    artifacts = _tree_rows(output / "artifact_evidence" / clip, output)
    expected = {
        results.resolve(), gate.resolve(), runtime.resolve(),
        (output / COMPACTION_FILENAME).resolve(),
    }
    expected.update(
        (output / relative).resolve() for relative in provenance
    )
    expected.update((output / relative).resolve() for relative in artifacts)
    observed = {path.resolve() for path in files}
    if observed != expected:
        raise RuntimeError(
            f"ordinal score cache source is not exactly compacted: {output}"
        )
    if (marker.get("schema") != marker_contract.get("schema") or
            marker.get("contract") != marker_contract.get("contract") or
            marker.get("results_sha256") !=
            artifact_cache.sha256_file(results) or
            marker.get("frame_gate_evidence_sha256") !=
            artifact_cache.sha256_file(gate) or
            marker.get("runtime_scene_evidence_sha256") != {
                clip: artifact_cache.sha256_file(runtime)
            } or marker.get("multiscale_provenance_sha256") != provenance or
            marker.get("artifact_evidence_sha256") != artifacts or
            marker.get("retained_role") !=
            "selected-target-safety-label-evidence"):
        raise RuntimeError("ordinal score cache compaction marker differs")
    return {
        "root": output,
        "marker": marker,
        "results": results,
        "gate": gate,
        "runtime": runtime,
        "provenance": provenance,
        "artifacts": artifacts,
    }


def _compacted_source_identity(compacted, clip):
    root = compacted["root"]
    return {
        "marker_sha256": artifact_cache.sha256_file(
            root / COMPACTION_FILENAME
            ),
        "results_sha256": artifact_cache.sha256_file(compacted["results"]),
        "frame_gate_evidence_sha256":
            artifact_cache.sha256_file(compacted["gate"]),
        "runtime_scene_evidence_sha256": {
            clip: artifact_cache.sha256_file(compacted["runtime"]),
        },
        "multiscale_provenance_sha256": compacted["provenance"],
        "artifact_evidence_sha256": compacted["artifacts"],
    }


def _copy_static_tree(source, destination, clip, *, provenance=True):
    source = Path(source)
    destination = Path(destination)
    relatives = [
        Path(clip) / "runtime_scene_evidence.json",
        Path("artifact_evidence") / clip,
    ]
    if provenance:
        relatives.append(Path("multiscale_provenance") / clip)
    for relative in relatives:
        src = source / relative
        if not src.exists():
            continue
        dst = destination / relative
        if src.is_dir():
            _plain_tree_files(src, "static evidence source")
            shutil.copytree(src, dst)
        else:
            if (artifact_cache.is_link_or_junction(src) or
                    not src.is_file()):
                raise RuntimeError(
                    f"ordinal score cache static evidence is not a plain "
                    f"file: {src}"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def publish(cache, identity, *, summary_path, scale_outputs, clip,
            clips_root):
    """Publish one completed compacted grid and return its cache key."""
    identity = _validate_identity(identity)
    summary_path = Path(summary_path).resolve(strict=True)
    summary = _load_json(summary_path, "multiscale summary")
    if summary_path.read_bytes() != artifact_cache.canonical_bytes(summary):
        raise RuntimeError("ordinal score cache summary is noncanonical")
    summary_sha256 = artifact_cache.sha256_file(summary_path)
    rows = identity["selection"]["scale_rows"]
    expected_slugs = {row["scale_slug"] for row in rows}
    outputs = {
        str(slug): Path(path).resolve(strict=True)
        for slug, path in scale_outputs.items()
    }
    if set(outputs) != expected_slugs:
        raise RuntimeError("ordinal score cache output grid differs")
    scoring = identity["preprocessing"]
    marker_contract = scoring["scoring_contracts"].get("compaction")
    if not isinstance(marker_contract, dict):
        raise RuntimeError("ordinal score cache compaction contract is missing")
    summary_rows = summary.get("scale_results")
    by_slug = {
        row.get("scale_slug"): row for row in summary_rows
        if isinstance(row, dict)
    } if isinstance(summary_rows, list) else {}
    driver_contract = scoring["scoring_contracts"].get("driver")
    if (not isinstance(driver_contract, dict) or
            summary.get("schema") != driver_contract.get("schema") or
            summary.get("contract") != driver_contract.get("contract") or
            set(by_slug) != expected_slugs or
            summary.get("render_identity_sha256") !=
            identity["source"]["render_identity_sha256"] or
            summary.get("metric_sha256") != scoring["metric_sha256"]):
        raise RuntimeError("ordinal score cache summary identity differs")

    clips_root = Path(clips_root).resolve(strict=True)
    clip_manifest = clips_root / "clip_hash_manifest.json"
    artifact_slugs = {
        row["scale_slug"]
        for row in identity["selection"]["artifact_scale_rows"]
    }
    compacted_by_slug = {}
    source_identity_by_slug = {}
    for row in rows:
        slug = row["scale_slug"]
        output = outputs[slug]
        compacted = _validate_compacted_tree(output, clip, marker_contract)
        if ({Path(path).name for path in compacted["provenance"]} !=
                PROVENANCE_FILES or
                bool(compacted["artifacts"]) != (slug in artifact_slugs)):
            raise RuntimeError(
                f"ordinal score cache sparse evidence differs: {slug}"
            )
        summary_row = by_slug[slug]
        if (Path(summary_row.get("run", "")).resolve(strict=False) !=
                output or
                summary_row.get("results_sha256") !=
                artifact_cache.sha256_file(compacted["results"]) or
                summary_row.get("frame_gate_evidence_sha256") !=
                artifact_cache.sha256_file(compacted["gate"]) or
                summary_row.get("provenance_sha256") != {
                    Path(path).name: digest
                    for path, digest in compacted["provenance"].items()
                }):
            raise RuntimeError(
                f"ordinal score cache summary row differs: {slug}"
            )
        compacted_by_slug[slug] = compacted
        source_identity_by_slug[slug] = _compacted_source_identity(
            compacted, clip
        )
    projection = _project_execution_origin({
        slug: _provenance_root(outputs[slug], clip)
        for slug in expected_slugs
    }, projected_mode=DEPTH_CACHE_MODE_TOKEN,
        projected_estimator_calls=ESTIMATOR_CALLS_TOKEN)

    with tempfile.TemporaryDirectory(prefix="apollo-ordinal-score-packet-") as temp:
        packet_root = Path(temp)
        scale_packet_rows = []
        for row in rows:
            slug = row["scale_slug"]
            output = outputs[slug]
            compacted = compacted_by_slug[slug]
            origin_row = projection["rows"][slug]

            result = _load_json(compacted["results"], "results")
            meta = result.get("meta")
            if (not isinstance(meta, dict) or
                    meta.get("run_name") != output.name or
                    meta.get("metric_sha256") != scoring["metric_sha256"] or
                    not isinstance(meta.get("timestamp"), str) or
                    not isinstance(meta.get("git_sha"), str) or
                    not isinstance(meta.get("git_dirty"), bool) or
                    _normalized_path(meta.get("clips_root", "")) !=
                    _normalized_path(clips_root)):
                raise RuntimeError(
                    f"ordinal score cache result binding differs: {slug}"
                )
            old_manifest = meta.get("clip_hash_manifest")
            if old_manifest is not None and _normalized_path(old_manifest) != \
                    _normalized_path(clip_manifest):
                raise RuntimeError(
                    f"ordinal score cache clip manifest differs: {slug}"
                )
            if old_manifest is not None and not clip_manifest.is_file():
                raise RuntimeError(
                    "ordinal score cache clip hash manifest is missing"
                )
            meta["run_name"] = RUN_TOKEN
            meta["clips_root"] = CLIPS_ROOT_TOKEN
            meta["timestamp"] = TIMESTAMP_TOKEN
            meta["git_sha"] = GIT_SHA_TOKEN
            meta["git_dirty"] = GIT_DIRTY_TOKEN
            # A packet materialized into another workspace must never present
            # the publication checkout/time of an earlier cache hit as if it
            # were current.  The deterministic semantic origin is restored as
            # an explicit cache receipt below.
            meta.pop("ordinal_score_cache", None)
            if old_manifest is not None:
                meta["clip_hash_manifest"] = CLIP_MANIFEST_TOKEN

            gate = _read_gate(compacted["gate"])
            header = gate[0]
            if (header.get("run_name") != output.name or
                    header.get("results_sha256") !=
                    artifact_cache.sha256_file(compacted["results"]) or
                    header.get("metric_sha256") != scoring["metric_sha256"] or
                    header.get("thresholds_sha256") !=
                    scoring["thresholds_sha256"]):
                raise RuntimeError(
                    f"ordinal score cache gate binding differs: {slug}"
                )
            header["run_name"] = RUN_TOKEN
            header["results_sha256"] = "@APOLLO_ORDINAL_RESULTS_SHA256@"
            result = _replace_exact_strings(
                result, origin_row["replacements"]
            )
            gate = _replace_exact_strings(gate, origin_row["replacements"])

            forbidden = {
                output.name.casefold(),
                _normalized_path(output),
                _normalized_path(clips_root),
                _normalized_path(clip_manifest),
            }
            _scan_for_stale_strings(result, forbidden)
            _scan_for_stale_strings(gate, forbidden)
            scale_root = packet_root / "scales" / slug
            scale_root.mkdir(parents=True)
            (scale_root / RESULTS_TEMPLATE).write_bytes(
                artifact_cache.canonical_bytes(result)
            )
            _write_gate(scale_root / GATE_TEMPLATE, gate)
            _copy_static_tree(
                output, scale_root, clip, provenance=False
            )
            _write_projected_provenance(
                scale_root, clip, origin_row
            )
            static = _static_identity(scale_root, clip)
            source_identity = source_identity_by_slug[slug]
            if (static["runtime"] !=
                    source_identity["runtime_scene_evidence_sha256"] or
                    static["artifacts"] !=
                    source_identity["artifact_evidence_sha256"]):
                raise RuntimeError(
                    "ordinal score cache source changed while copying static "
                    f"evidence: {slug}"
                )
            scale_packet_rows.append({
                **row,
                "had_clip_hash_manifest": old_manifest is not None,
                "static_manifest_sha256":
                    artifact_cache.canonical_sha256(static),
            })

        summary_base = copy.deepcopy(summary)
        summary_base.pop("scale_results", None)
        summary_base.pop("scale_score_jobs", None)
        summary_base.pop("scored_result_cache", None)
        summary_base = _replace_exact_strings(
            summary_base,
            projection["rows"][rows[0]["scale_slug"]]["replacements"],
        )
        summary_forbidden = {
            _normalized_path(summary_path.parent),
            _normalized_path(clips_root),
            _normalized_path(clip_manifest),
        }
        summary_forbidden.update(output.name.casefold() for output in outputs.values())
        summary_forbidden.update(_normalized_path(output) for output in outputs.values())
        _scan_for_stale_strings(summary_base, summary_forbidden)
        origin_normalization = {
            "contract": "apollo-score-cache-execution-origin-v1",
            "normalized_fields": [
                "contract.depth_state_cache_mode",
                "harness.depth_state_cache.mode",
                "harness.shipping_estimator_calls_per_source_frame",
                "manifest.depth_state_cache.mode",
            ],
            "retained_semantic_fields": [
                "depth_state_cache.key_sha256",
                "depth_state_cache.manifest_sha256",
                "depth_state_cache.boundary",
                "depth_state_cache.selected_state_frame_count",
                "depth_state_cache.runtime_scene_frame_count",
            ],
        }
        packet = {
            "schema": PACKET_SCHEMA,
            "contract": PACKET_CONTRACT,
            "identity_sha256":
                artifact_cache.DirectoryArtifactCache.key(identity),
            "clip": clip,
            "summary_base": summary_base,
            "execution_origin_normalization": origin_normalization,
            "scales": scale_packet_rows,
        }
        (packet_root / PACKET_MANIFEST).write_bytes(
            artifact_cache.canonical_bytes(packet)
        )
        if (artifact_cache.sha256_file(summary_path) != summary_sha256 or
                _load_json(summary_path, "multiscale summary") != summary):
            raise RuntimeError(
                "ordinal score cache summary changed during publication"
            )
        for row in rows:
            slug = row["scale_slug"]
            observed = _validate_compacted_tree(
                outputs[slug], clip, marker_contract
            )
            if (_compacted_source_identity(observed, clip) !=
                    source_identity_by_slug[slug]):
                raise RuntimeError(
                    "ordinal score cache compacted source changed during "
                    f"publication: {slug}"
                )
        key = cache.publish(identity, packet_root)
        # DirectoryArtifactCache intentionally treats an existing valid key as
        # immutable.  Compare it with this newly produced normalized packet so
        # hidden scorer nondeterminism fails closed instead of silently using
        # whichever producer won the key first.
        verification = Path(tempfile.mkdtemp(
            prefix="apollo-ordinal-score-cache-verify-"
        ))
        verification.rmdir()
        try:
            if (not cache.materialize(identity, verification) or
                    _directory_rows(verification) !=
                    _directory_rows(packet_root)):
                raise RuntimeError(
                    "ordinal score cache key produced different packet bytes"
                )
        finally:
            if verification.exists():
                shutil.rmtree(verification)
        return key


def _load_packet(packet_root, identity):
    unresolved = Path(packet_root)
    files = _plain_tree_files(unresolved, "packet")
    packet_root = unresolved.resolve(strict=True)
    packet_path = packet_root / PACKET_MANIFEST
    packet = _load_json(packet_path, "packet manifest")
    if packet_path.read_bytes() != artifact_cache.canonical_bytes(packet):
        raise RuntimeError("ordinal score cache packet is noncanonical")
    expected_rows = identity["selection"]["scale_rows"]
    packet_rows = packet.get("scales")
    origin = packet.get("execution_origin_normalization")
    if (packet.get("schema") != PACKET_SCHEMA or
            packet.get("contract") != PACKET_CONTRACT or
            packet.get("identity_sha256") !=
            artifact_cache.DirectoryArtifactCache.key(identity) or
            not isinstance(packet.get("clip"), str) or
            not packet["clip"] or
            not isinstance(packet.get("summary_base"), dict) or
            not isinstance(origin, dict) or
            origin.get("contract") !=
            "apollo-score-cache-execution-origin-v1" or
            not isinstance(origin.get("normalized_fields"), list) or
            not isinstance(origin.get("retained_semantic_fields"), list) or
            not isinstance(packet_rows, list) or
            len(packet_rows) != len(expected_rows)):
        raise RuntimeError("ordinal score cache packet manifest differs")
    for row, expected in zip(packet_rows, expected_rows):
        if (not isinstance(row, dict) or
                set(row) != set(expected) | {
                    "had_clip_hash_manifest", "static_manifest_sha256"
                } or any(row.get(key) != value for key, value in expected.items()) or
                not isinstance(row.get("had_clip_hash_manifest"), bool) or
                not SHA256.fullmatch(
                    str(row.get("static_manifest_sha256", ""))
                )):
            raise RuntimeError("ordinal score cache packet scale differs")
    slugs = {row["scale_slug"] for row in packet_rows}
    clip = packet["clip"]
    for path in files:
        relative = path.relative_to(packet_root)
        parts = relative.parts
        accepted = relative.as_posix() == PACKET_MANIFEST
        if len(parts) >= 3 and parts[0] == "scales" and parts[1] in slugs:
            tail = parts[2:]
            accepted = (
                tail in {(RESULTS_TEMPLATE,), (GATE_TEMPLATE,)} or
                tail == (clip, "runtime_scene_evidence.json") or
                (len(tail) == 3 and tail[:2] == (
                    "multiscale_provenance", clip
                ) and tail[2] in PROVENANCE_FILES) or
                (len(tail) >= 3 and tail[:2] == (
                    "artifact_evidence", clip
                ))
            )
        if not accepted:
            raise RuntimeError(
                f"ordinal score cache packet has unexpected file: {relative}"
            )
    return packet


def _static_identity(output, clip):
    return {
        "runtime": {
            clip: artifact_cache.sha256_file(
                Path(output) / clip / "runtime_scene_evidence.json"
            )
        },
        "provenance": _tree_rows(
            Path(output) / "multiscale_provenance" / clip, output
            ),
        "artifacts": _tree_rows(
            Path(output) / "artifact_evidence" / clip, output
            ),
    }


def _commit_scale_output(staging, output):
    """Single rename seam used by crash-resume fault-injection tests."""
    Path(staging).rename(output)


def _transaction_destination_sha256(summary_path, outputs):
    return artifact_cache.canonical_sha256({
        "summary_path": _normalized_path(summary_path),
        "scale_outputs": {
            slug: _normalized_path(path)
            for slug, path in sorted(outputs.items())
        },
    })


def _transaction_paths(summary_path):
    summary_path = Path(summary_path)
    lock = summary_path.with_name(summary_path.name + ".score-cache.lock")
    return lock, lock.with_name(lock.name + ".receipt.json")


def _packet_transaction_path(summary_path, transaction_id):
    if not MATERIALIZATION_TRANSACTION_ID.fullmatch(str(transaction_id)):
        raise RuntimeError("ordinal score cache transaction id is invalid")
    return Path(summary_path).parent / (
        f".ordinal-score-cache-packet-{transaction_id}"
    )


def _summary_transaction_temporary(summary_path, transaction_id):
    if not MATERIALIZATION_TRANSACTION_ID.fullmatch(str(transaction_id)):
        raise RuntimeError("ordinal score cache transaction id is invalid")
    summary_path = Path(summary_path)
    return summary_path.with_name(
        f".{summary_path.name}.score-cache-{transaction_id}.tmp"
    )


def _scale_transaction_prefix(output, transaction_id):
    if not MATERIALIZATION_TRANSACTION_ID.fullmatch(str(transaction_id)):
        raise RuntimeError("ordinal score cache transaction id is invalid")
    return f".{Path(output).name}.score-cache-partial-{transaction_id}-"


def _transaction_temp_candidates(summary_path, outputs, stale_transaction):
    """Find only temp paths owned by one authenticated stale transaction."""
    if stale_transaction is None:
        return []
    transaction_id = stale_transaction["transaction_id"]
    summary_path = Path(summary_path)
    packet = _packet_transaction_path(summary_path, transaction_id)
    exact = {
        packet,
        _summary_transaction_temporary(summary_path, transaction_id),
    }
    prefixes = {
        packet.parent: (packet.name + ".cache-partial-",),
    }
    for output in outputs.values():
        output = Path(output)
        prefixes.setdefault(output.parent, ())
        prefixes[output.parent] += (
            _scale_transaction_prefix(output, transaction_id),
        )

    candidates = set()
    for path in exact:
        if _path_present(path):
            candidates.add(path)
    for parent, owned_prefixes in prefixes.items():
        if not parent.exists():
            continue
        if artifact_cache.is_link_or_junction(parent) or not parent.is_dir():
            raise RuntimeError(
                f"ordinal score cache transaction parent is not a plain "
                f"directory: {parent}"
            )
        for path in parent.iterdir():
            if any(path.name.startswith(prefix) for prefix in owned_prefixes):
                candidates.add(path)
    return sorted(candidates, key=lambda path: str(path))


def _remove_interrupted_temps(summary_path, outputs, stale_transaction):
    """Remove transaction-ID-owned temps while holding the destination lock."""
    candidates = _transaction_temp_candidates(
        summary_path, outputs, stale_transaction
    )
    for path in candidates:
        if artifact_cache.is_link_or_junction(path):
            raise RuntimeError(
                f"ordinal score cache stale temp is a link: {path}"
            )
        if not path.is_dir() and not path.is_file():
            raise RuntimeError(
                f"ordinal score cache stale temp is special: {path}"
            )
    for path in candidates:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return len(candidates)


def _validate_transaction_receipt(value, destination_sha256):
    required = {
        "schema", "contract", "state", "transaction_id",
        "cache_key_sha256", "destination_sha256", "process_id",
        "started_at", "updated_at",
    }
    if (not isinstance(value, dict) or set(value) != required or
            value.get("schema") != MATERIALIZATION_TRANSACTION_SCHEMA or
            value.get("contract") != MATERIALIZATION_TRANSACTION_CONTRACT or
            value.get("state") not in {"active", "idle"} or
            not MATERIALIZATION_TRANSACTION_ID.fullmatch(str(
                value.get("transaction_id", "")
            )) or not SHA256.fullmatch(str(value.get("cache_key_sha256", ""))) or
            value.get("destination_sha256") != destination_sha256 or
            not isinstance(value.get("process_id"), int) or
            isinstance(value.get("process_id"), bool) or
            value["process_id"] < 1 or
            not isinstance(value.get("started_at"), str) or
            not value["started_at"] or
            not isinstance(value.get("updated_at"), str) or
            not value["updated_at"]):
        raise RuntimeError(
            "ordinal score cache materialization receipt is invalid"
        )
    return value


def _read_transaction_receipt(path, destination_sha256):
    path = Path(path)
    if not _path_present(path):
        return None
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction() or not path.is_file():
        raise RuntimeError(
            f"ordinal score cache transaction receipt is not a plain file: "
            f"{path}"
        )
    value = _load_json(path, "materialization receipt")
    if path.read_bytes() != artifact_cache.canonical_bytes(value):
        raise RuntimeError(
            "ordinal score cache materialization receipt is noncanonical"
        )
    return _validate_transaction_receipt(value, destination_sha256)


def _write_transaction_receipt(path, value):
    path = Path(path)
    temporary = path.with_name(
        path.name + f".{secrets.token_hex(12)}.tmp"
    )
    encoded = artifact_cache.canonical_bytes(value)
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _lock_transaction_file(stream):
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError) as error:
        raise RuntimeError(
            "ordinal score cache materialization is already active"
        ) from error


def _unlock_transaction_file(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class _MaterializationTransaction:
    """Persistent OS lock plus crash-recoverable destination ownership."""

    def __init__(self, summary_path, outputs, cache_key):
        self.summary_path = Path(summary_path)
        self.outputs = dict(outputs)
        self.cache_key = cache_key
        self.destination_sha256 = _transaction_destination_sha256(
            summary_path, outputs
        )
        self.lock_path, self.receipt_path = _transaction_paths(summary_path)
        self.stream = None
        self.current = None
        self.stale = None

    def acquire(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        for path, role in (
                (self.lock_path, "lock"),
                (self.receipt_path, "receipt")):
            if _path_present(path):
                is_junction = getattr(path, "is_junction", lambda: False)
                if path.is_symlink() or is_junction() or not path.is_file():
                    raise RuntimeError(
                        f"ordinal score cache transaction {role} is not a "
                        f"plain file: {path}"
                    )
        descriptor = os.open(
            self.lock_path, os.O_RDWR | os.O_CREAT,
            0o600,
        )
        self.stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if self.lock_path.stat().st_size == 0:
                self.stream.write(b"\0")
                self.stream.flush()
                os.fsync(self.stream.fileno())
            _lock_transaction_file(self.stream)
            previous = _read_transaction_receipt(
                self.receipt_path, self.destination_sha256
            )
            self.stale = (
                copy.deepcopy(previous)
                if previous is not None and previous["state"] == "active"
                else None
            )
            if (self.stale is not None and
                    self.stale["cache_key_sha256"] != self.cache_key):
                raise RuntimeError(
                    "ordinal score cache stale transaction belongs to a "
                    "different cache key"
                )
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            )
            self.current = {
                "schema": MATERIALIZATION_TRANSACTION_SCHEMA,
                "contract": MATERIALIZATION_TRANSACTION_CONTRACT,
                "state": "active",
                # Keep one ownership token across crash recovery.  If cleanup
                # itself is interrupted, the durable receipt still names every
                # old and newly created temp/output path on the next attempt.
                "transaction_id": (
                    self.stale["transaction_id"]
                    if self.stale is not None else secrets.token_hex(16)
                ),
                "cache_key_sha256": self.cache_key,
                "destination_sha256": self.destination_sha256,
                "process_id": os.getpid(),
                "started_at": (
                    self.stale["started_at"]
                    if self.stale is not None else now
                ),
                "updated_at": now,
            }
            _write_transaction_receipt(self.receipt_path, self.current)
            return {
                "current": copy.deepcopy(self.current),
                "stale": copy.deepcopy(self.stale),
            }
        except BaseException:
            if self.stream is not None:
                with contextlib.suppress(OSError):
                    _unlock_transaction_file(self.stream)
                self.stream.close()
                self.stream = None
            raise

    def release(self, *, preserve_active):
        if self.stream is None or self.current is None:
            return
        try:
            if not preserve_active:
                idle = copy.deepcopy(self.current)
                idle["state"] = "idle"
                idle["updated_at"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="seconds")
                _write_transaction_receipt(self.receipt_path, idle)
        finally:
            try:
                _unlock_transaction_file(self.stream)
            finally:
                self.stream.close()
                self.stream = None


def _remove_interrupted_outputs(outputs, *, clip, marker_contract,
                                stale_transaction):
    """Remove only authenticated outputs from an interrupted cache hit."""
    existing = [path for path in outputs.values() if _path_present(path)]
    if existing and stale_transaction is None:
        raise RuntimeError(
            "ordinal score cache destinations collide without an owned stale "
            "transaction: " + ", ".join(str(path) for path in existing)
        )
    stale_id = (
        stale_transaction["transaction_id"]
        if stale_transaction is not None else None
    )
    stale_key = (
        stale_transaction["cache_key_sha256"]
        if stale_transaction is not None else None
    )
    for output in existing:
        is_junction = getattr(output, "is_junction", lambda: False)
        if (output.is_symlink() or is_junction() or not output.is_dir()):
            raise RuntimeError(
                f"ordinal score cache destination collision: {output}"
            )
        compacted = _validate_compacted_tree(output, clip, marker_contract)
        marker = compacted["marker"]
        result = _load_json(compacted["results"], "interrupted results")
        meta = result.get("meta")
        receipt = meta.get("ordinal_score_cache") \
            if isinstance(meta, dict) else None
        provenance = receipt.get("original_score_provenance") \
            if isinstance(receipt, dict) else None
        if (marker.get("materialized_from_scored_cache") is not True or
                marker.get("score_cache_key_sha256") != stale_key or
                marker.get("score_cache_transaction_contract") !=
                "apollo-score-cache-materialization-v1" or
                marker.get("score_cache_transaction_id") != stale_id or
                not isinstance(meta, dict) or
                meta.get("run_name") != output.name or
                not isinstance(receipt, dict) or
                receipt.get("contract") != PACKET_CONTRACT or
                not isinstance(provenance, dict) or
                provenance.get("cache_key_sha256") != stale_key or
                receipt.get("materialization_transaction_id") != stale_id or
                receipt.get("retained_provenance") !=
                "cached-semantic-origin"):
            raise RuntimeError(
                f"ordinal score cache destination is not an owned partial: "
                f"{output}"
            )
    for output in existing:
        shutil.rmtree(output)
    return len(existing)


def _materialize_locked(cache, identity, *, summary_path, scale_outputs,
                        clips_root, score_workers, transaction):
    """Atomically rebind and materialize a verified scored packet.

    Return ``None`` on a cache miss.  Existing destinations are never replaced.
    """
    identity = _validate_identity(identity)
    if not isinstance(score_workers, int) or score_workers < 1:
        raise RuntimeError("ordinal score cache worker count is invalid")
    summary_path = Path(summary_path).resolve(strict=False)
    outputs = {
        str(slug): Path(path).resolve(strict=False)
        for slug, path in scale_outputs.items()
    }
    expected_slugs = {
        row["scale_slug"] for row in identity["selection"]["scale_rows"]
    }
    if set(outputs) != expected_slugs:
        raise RuntimeError("ordinal score cache destination grid differs")
    if len(set(outputs.values())) != len(outputs):
        raise RuntimeError("ordinal score cache destinations repeat a path")
    if _path_present(summary_path):
        raise RuntimeError(
            f"ordinal score cache summary already exists: {summary_path}"
        )
    clips_root = Path(clips_root).resolve(strict=True)
    clip_manifest = clips_root / "clip_hash_manifest.json"
    cache_key = artifact_cache.DirectoryArtifactCache.key(identity)
    materialized_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(timespec="seconds")
    scoring_contracts = identity["preprocessing"]["scoring_contracts"]
    execution_origin = (
        "cold-export/replay mode and estimator-call count normalized; "
        "depth-state key, manifest, boundary, and rendered evidence retained"
    )
    score_provenance = {
        "cache_key_sha256": cache_key,
        "render_identity_sha256":
            identity["source"]["render_identity_sha256"],
        "metric_sha256": identity["preprocessing"]["metric_sha256"],
        "thresholds_sha256":
            identity["preprocessing"]["thresholds_sha256"],
        "scoring_contracts_sha256": artifact_cache.canonical_sha256(
            scoring_contracts),
        "scorer_runtime_sha256": artifact_cache.canonical_sha256(
            scoring_contracts.get("scorer_runtime")),
        "retained_provenance": "cached-semantic-origin",
        "execution_origin": execution_origin,
    }
    if cache.validated_payload(identity) is None:
        collisions = [
            path for path in outputs.values() if _path_present(path)
        ]
        if collisions:
            raise RuntimeError(
                "ordinal score cache miss has destination collisions: " +
                ", ".join(str(path) for path in collisions)
            )
        return None
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    transaction_id = transaction["current"]["transaction_id"]
    recovered_temps = _remove_interrupted_temps(
        summary_path, outputs, transaction["stale"]
    )
    packet_destination = _packet_transaction_path(
        summary_path, transaction_id
    )
    if _path_present(packet_destination):
        raise RuntimeError(
            f"ordinal score cache packet destination collides: "
            f"{packet_destination}"
        )
    if not cache.materialize(identity, packet_destination):
        raise RuntimeError("ordinal score cache hit disappeared")

    staged = {}
    committed = []
    try:
        packet = _load_packet(packet_destination, identity)
        clip = packet["clip"]
        scoring = identity["preprocessing"]
        marker_contract = scoring["scoring_contracts"]["compaction"]
        recovered_outputs = _remove_interrupted_outputs(
            outputs, clip=clip, marker_contract=marker_contract,
            stale_transaction=transaction["stale"],
        )
        for row in packet["scales"]:
            source = packet_destination / "scales" / row["scale_slug"]
            if (artifact_cache.canonical_sha256(
                    _static_identity(source, clip)) !=
                    row["static_manifest_sha256"]):
                raise RuntimeError(
                    "ordinal score cache packet static evidence differs: "
                    + row["scale_slug"]
                )
        bound_projection = _project_execution_origin({
            row["scale_slug"]: _provenance_root(
                packet_destination / "scales" / row["scale_slug"], clip
            ) for row in packet["scales"]
        }, projected_mode=CACHED_DEPTH_MODE, projected_estimator_calls=0)
        publication_rows = []
        for row in packet["scales"]:
            slug = row["scale_slug"]
            source = packet_destination / "scales" / slug
            origin_row = bound_projection["rows"][slug]
            output = outputs[slug]
            output.parent.mkdir(parents=True, exist_ok=True)
            staging = artifact_cache.inheriting_temporary_directory(
                output.parent,
                _scale_transaction_prefix(output, transaction_id),
            )
            staged[slug] = staging
            _copy_static_tree(
                source, staging, clip, provenance=False
            )
            _write_projected_provenance(staging, clip, origin_row)

            result = _load_json(source / RESULTS_TEMPLATE, "results template")
            result = _replace_exact_strings(
                result, origin_row["replacements"]
            )
            meta = result.get("meta")
            if (not isinstance(meta, dict) or
                    meta.get("run_name") != RUN_TOKEN or
                    meta.get("clips_root") != CLIPS_ROOT_TOKEN or
                    meta.get("timestamp") != TIMESTAMP_TOKEN or
                    meta.get("git_sha") != GIT_SHA_TOKEN or
                    meta.get("git_dirty") != GIT_DIRTY_TOKEN):
                raise RuntimeError(
                    f"ordinal score cache result template differs: {slug}"
                )
            meta["run_name"] = output.name
            meta["clips_root"] = str(clips_root)
            meta["timestamp"] = materialized_at
            # These legacy display fields are intentionally honest about the
            # fact that the bytes were rebound from a semantic score cache.
            # Detailed, immutable score origin is carried alongside them.
            meta["git_sha"] = f"score-cache-{cache_key[:12]}"
            meta["git_dirty"] = False
            meta["ordinal_score_cache"] = {
                "schema": PACKET_SCHEMA,
                "contract": PACKET_CONTRACT,
                "original_score_provenance": score_provenance,
                "materialized_at": materialized_at,
                "normalized_origin_fields": [
                    "timestamp", "git_sha", "git_dirty",
                    "depth_state_cache.mode",
                    "shipping_estimator_calls_per_source_frame",
                ],
                "retained_provenance": "cached-semantic-origin",
                "materialization_transaction_id":
                    transaction["current"]["transaction_id"],
            }
            if row["had_clip_hash_manifest"]:
                if meta.get("clip_hash_manifest") != CLIP_MANIFEST_TOKEN:
                    raise RuntimeError(
                        f"ordinal score cache manifest template differs: {slug}"
                    )
                if not clip_manifest.is_file():
                    raise RuntimeError(
                        "ordinal score cache destination clip manifest is missing"
                    )
                meta["clip_hash_manifest"] = str(clip_manifest)
            results_path = staging / RESULTS_FILENAME
            results_path.write_bytes(artifact_cache.canonical_bytes(result))
            results_sha = artifact_cache.sha256_file(results_path)

            gate = _replace_exact_strings(
                _read_gate(source / GATE_TEMPLATE),
                origin_row["replacements"],
            )
            header = gate[0]
            if (header.get("run_name") != RUN_TOKEN or
                    header.get("results_sha256") !=
                    "@APOLLO_ORDINAL_RESULTS_SHA256@"):
                raise RuntimeError(
                    f"ordinal score cache gate template differs: {slug}"
                )
            header["run_name"] = output.name
            header["results_sha256"] = results_sha
            gate_path = staging / FRAME_GATE_FILENAME
            _write_gate(gate_path, gate)
            gate_sha = artifact_cache.sha256_file(gate_path)

            static = _static_identity(staging, clip)
            marker = {
                "schema": marker_contract["schema"],
                "contract": marker_contract["contract"],
                "results_sha256": results_sha,
                "frame_gate_evidence_sha256": gate_sha,
                "runtime_scene_evidence_sha256": static["runtime"],
                "multiscale_provenance_sha256": static["provenance"],
                "artifact_evidence_sha256": static["artifacts"],
                "retained_role": "selected-target-safety-label-evidence",
                "deleted_files": 0,
                "deleted_bytes": 0,
                "materialized_from_scored_cache": True,
                "score_cache_key_sha256": cache_key,
                "score_cache_transaction_contract":
                    "apollo-score-cache-materialization-v1",
                "score_cache_transaction_id":
                    transaction["current"]["transaction_id"],
            }
            (staging / COMPACTION_FILENAME).write_bytes(
                artifact_cache.canonical_bytes(marker)
            )
            publication_rows.append({
                "scale": row["scale"],
                "scale_slug": slug,
                "run": str(output),
                "results_sha256": results_sha,
                "frame_gate_evidence_sha256": gate_sha,
                "provenance_sha256": {
                    Path(relative).name: digest
                    for relative, digest in static["provenance"].items()
                },
            })

        for row in packet["scales"]:
            slug = row["scale_slug"]
            _commit_scale_output(staged[slug], outputs[slug])
            committed.append(outputs[slug])
        summary = _replace_exact_strings(
            copy.deepcopy(packet["summary_base"]),
            bound_projection["rows"][
                packet["scales"][0]["scale_slug"]
            ]["replacements"],
        )
        summary["scale_score_jobs"] = score_workers
        summary["scale_results"] = publication_rows
        summary["scored_result_cache"] = {
            "schema": PACKET_SCHEMA,
            "contract": PACKET_CONTRACT,
            "original_score_provenance": score_provenance,
            "materialized_at": materialized_at,
            "retained_provenance": "cached-semantic-origin",
            "materialization_transaction_id":
                transaction["current"]["transaction_id"],
            "recovered_interrupted_scale_outputs": recovered_outputs,
            "recovered_interrupted_temp_paths": recovered_temps,
        }
        # Packet bytes are no longer needed once every output and the summary
        # value are complete.  Remove them before publishing success so a hard
        # kill after the atomic summary rename cannot strand a packet tree.
        shutil.rmtree(packet_destination)
        temporary = _summary_transaction_temporary(
            summary_path, transaction_id
        )
        with temporary.open("xb") as stream:
            stream.write(artifact_cache.canonical_bytes(summary))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, summary_path)
        return summary
    except BaseException:
        for output in committed:
            if output.exists():
                shutil.rmtree(output)
        raise
    finally:
        for staging in staged.values():
            if staging.exists():
                shutil.rmtree(staging)
        if packet_destination.exists():
            shutil.rmtree(packet_destination)
        temporary = _summary_transaction_temporary(
            summary_path, transaction_id
        )
        if _path_present(temporary):
            if (artifact_cache.is_link_or_junction(temporary) or
                    not temporary.is_file()):
                raise RuntimeError(
                    f"ordinal score cache summary temporary is invalid: "
                    f"{temporary}"
                )
            temporary.unlink()


def materialize(cache, identity, *, summary_path, scale_outputs, clips_root,
                score_workers):
    """Materialize under an OS-locked, crash-recoverable transaction."""
    identity = _validate_identity(identity)
    if not isinstance(score_workers, int) or score_workers < 1:
        raise RuntimeError("ordinal score cache worker count is invalid")
    summary_path = Path(summary_path).resolve(strict=False)
    outputs = {
        str(slug): Path(path).resolve(strict=False)
        for slug, path in scale_outputs.items()
    }
    expected_slugs = {
        row["scale_slug"] for row in identity["selection"]["scale_rows"]
    }
    if set(outputs) != expected_slugs:
        raise RuntimeError("ordinal score cache destination grid differs")
    if len(set(outputs.values())) != len(outputs):
        raise RuntimeError("ordinal score cache destinations repeat a path")
    if _path_present(summary_path):
        raise RuntimeError(
            f"ordinal score cache summary already exists: {summary_path}"
        )
    clips_root = Path(clips_root).resolve(strict=True)
    if cache.validated_payload(identity) is None:
        collisions = [
            path for path in outputs.values() if _path_present(path)
        ]
        if collisions:
            raise RuntimeError(
                "ordinal score cache miss has destination collisions: " +
                ", ".join(str(path) for path in collisions)
            )
        return None

    cache_key = artifact_cache.DirectoryArtifactCache.key(identity)
    lock = _MaterializationTransaction(summary_path, outputs, cache_key)
    transaction = lock.acquire()
    succeeded = False
    try:
        result = _materialize_locked(
            cache, identity, summary_path=summary_path,
            scale_outputs=outputs, clips_root=clips_root,
            score_workers=score_workers, transaction=transaction,
        )
        succeeded = True
        return result
    finally:
        # A normal Python exception cleans every output it knows it committed.
        # A process-death seam can occur after rename but before bookkeeping;
        # retain active ownership whenever any destination remains so the next
        # lock holder may authenticate and recover it.
        owned_temp_remains = False
        if not succeeded:
            try:
                owned_temp_remains = bool(_transaction_temp_candidates(
                    summary_path, outputs, transaction["current"]
                ))
            except RuntimeError:
                # Preserve ownership if even inspecting a transaction-derived
                # path fails.  The next lock holder must fail closed rather
                # than converting the receipt to idle and orphaning it.
                owned_temp_remains = True
        preserve_active = (
            not succeeded and (
                any(_path_present(path) for path in outputs.values()) or
                owned_temp_remains
            )
        )
        lock.release(preserve_active=preserve_active)


if __name__ == "__main__":
    if sys.argv[1:] != ["--print-runtime-identity"]:
        raise SystemExit("usage: ordinal_result_cache.py --print-runtime-identity")
    sys.stdout.buffer.write(
        artifact_cache.canonical_bytes(scorer_runtime_identity())
    )
