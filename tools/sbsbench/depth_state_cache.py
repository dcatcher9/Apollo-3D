"""Authenticated cross-workspace cache for production depth-state sequences.

The cache boundary is the completed DA-V2/normalization/EMA/SubjectState output
immediately before ``depth_warp_prefilter_cs``.  It is intentionally independent
of output-eye geometry, artistic scale, warp artifacts, and metric code.  Only
working train/development clips may use it; the caller owns that split check.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import re
import secrets
from pathlib import Path
import shutil
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
DEPTH_MODELS_DIR = SCRIPT_DIR.parent / "depth_models"
sys.path.insert(0, str(DEPTH_MODELS_DIR))

import preprocessing_artifact_cache as artifact_cache  # noqa: E402
import native_hdr_capture  # noqa: E402


ARTIFACT_KIND = "production-depth-state-selected-sequence-v2"
INNER_MANIFEST = "depth_state_manifest.json"
IDENTITY_SCHEMA = 3
IDENTITY_CONTRACT = "apollo-production-depth-state-cache-key-v3"
RUNTIME_SCHEMA = 1
RUNTIME_CONTRACT = "apollo-production-depth-native-runtime-v1"
RUNTIME_RECEIPT_SCHEMA = 1
RUNTIME_RECEIPT_CONTRACT = "apollo-production-depth-native-runtime-receipt-v1"
RUNTIME_RECEIPT_NAME = ".sbs_depth_native_runtime_receipt_v1.json"
GEOMETRY_OPTIONS = frozenset({
    "--eye-w", "--eye-h", "--output-scale", "--max-width",
})
OPTIONS_WITH_VALUES = frozenset({
    "--eye-w", "--eye-h", "--output-scale", "--max-width",
    "--pop-strength", "--adaptive-pop-max", "--zero-plane",
    "--hdr-scale", "--sdr-white-level-raw", "--subject-lock",
    "--subject-recenter", "--depth-short-side", "--ema",
    "--ema-edge-change", "--ema-edge-gradient", "--ema-edge-strength",
    "--minmax-ema", "--cuda-graph", "--model",
})
DEPTH_SHADER_ENTRY_FILES = (
    "rgb_to_nchw_cs.hlsl",
    "buffer_to_tex_cs.hlsl",
    "depth_ema_motion_cs.hlsl",
    "depth_minmax_cs.hlsl",
    "depth_minmax_ema_cs.hlsl",
    "depth_hist_cs.hlsl",
    "depth_subject_hist_cs.hlsl",
    "depth_subject_resolve_cs.hlsl",
)
# Retain the currently expected closure as a public compatibility/test fixture
# list.  Cache identity is built by recursively following the entry files, so a
# future nested include is authenticated without requiring this tuple to be
# manually updated first.
DEPTH_SHADER_FILES = DEPTH_SHADER_ENTRY_FILES + (
    "include/depth_constants.hlsl",
    "include/depth_color.hlsl",
    "include/bestv2_curve.hlsl",
)
SHADER_INCLUDE_PATTERN = re.compile(
    r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE,
)

INFERENCE_RUNTIME_DLLS = (
    "nvinfer_11.dll",
    "nvinfer_plugin_11.dll",
    "nvonnxparser_11.dll",
)
SYSTEM_RUNTIME_DLLS = (
    "nvcuda.dll",
    "d3dcompiler_47.dll",
    "d3d11.dll",
    "dxgi.dll",
)


def _file(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"depth-state identity file is missing/plain-file invalid: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": artifact_cache.sha256_file(path),
    }


def _runtime_file(path):
    """Return a path-independent native-runtime file identity."""
    path = Path(path)
    return {
        "name": path.name.casefold(),
        **_file(path),
    }


def _nvidia_smi_path():
    candidate = shutil.which("nvidia-smi")
    if candidate:
        return Path(candidate)
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidate = (
            Path(program_files) / "NVIDIA Corporation" / "NVSMI" /
            "nvidia-smi.exe"
        )
        if candidate.is_file():
            return candidate
    raise RuntimeError("cannot resolve nvidia-smi for depth-state runtime identity")


def _gpu_driver_identity():
    command = [
        str(_nvidia_smi_path()),
        "--query-gpu=uuid,name,pci.device_id,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot query NVIDIA runtime identity") from error
    if completed.returncode:
        raise RuntimeError(
            "cannot query NVIDIA runtime identity: " +
            (completed.stdout + completed.stderr)[-1000:]
        )
    rows = []
    for fields in csv.reader(io.StringIO(completed.stdout)):
        fields = [field.strip() for field in fields]
        if len(fields) != 5 or any(not field for field in fields):
            raise RuntimeError("NVIDIA runtime identity has an invalid row")
        rows.append({
            "uuid": fields[0],
            "name": fields[1],
            "pci_device_id": fields[2],
            "compute_capability": fields[3],
            "driver_version": fields[4],
        })
    if not rows:
        raise RuntimeError("NVIDIA runtime identity contains no GPU")
    return sorted(rows, key=lambda row: (row["uuid"], row["pci_device_id"]))


def _runtime_paths(build_dir):
    if os.name != "nt":
        raise RuntimeError("production depth-state runtime identity requires Windows")
    build_dir = Path(build_dir).resolve(strict=True)
    paths = {}
    for name in INFERENCE_RUNTIME_DLLS:
        path = build_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing depth-state inference runtime DLL: {path}")
        paths[f"build/{name.casefold()}"] = path
    windows = os.environ.get("WINDIR")
    if not windows:
        raise RuntimeError("WINDIR is unavailable for depth-state runtime identity")
    system32 = Path(windows) / "System32"
    for name in SYSTEM_RUNTIME_DLLS:
        path = system32 / name
        if not path.is_file():
            raise RuntimeError(f"missing depth-state system runtime DLL: {path}")
        paths[f"system32/{name.casefold()}"] = path
    return build_dir, paths


def _platform_identity():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def _runtime_stat(path):
    path = Path(path).resolve(strict=True)
    stat_value = path.stat()
    return {
        "path": str(path),
        "bytes": stat_value.st_size,
        "mtime_ns": stat_value.st_mtime_ns,
        "device": stat_value.st_dev,
        "inode": stat_value.st_ino,
    }


def _build_runtime_snapshot(paths):
    native_files = {}
    file_receipts = {}
    for role, path in sorted(paths.items()):
        before = _runtime_stat(path)
        native_file = _runtime_file(path)
        after = _runtime_stat(path)
        if before != after or native_file["bytes"] != after["bytes"]:
            raise RuntimeError(
                f"depth-state native runtime changed while hashing: {role}"
            )
        native_files[role] = native_file
        file_receipts[role] = after
    identity = {
        "schema": RUNTIME_SCHEMA,
        "contract": RUNTIME_CONTRACT,
        "runtime_namespace": "windows-nvidia-d3d11-tensorrt-inference-v1",
        "platform": _platform_identity(),
        "native_files": native_files,
        "gpus": _gpu_driver_identity(),
    }
    validate_runtime_identity(identity)
    return {
        "identity": identity,
        "files": file_receipts,
    }


def verify_runtime_snapshot(snapshot):
    if (not isinstance(snapshot, dict) or
            set(snapshot) != {"identity", "files"}):
        raise RuntimeError("depth-state runtime snapshot is invalid")
    identity = validate_runtime_identity(snapshot["identity"])
    files = snapshot["files"]
    if not isinstance(files, dict) or set(files) != set(identity["native_files"]):
        raise RuntimeError("depth-state runtime file receipt is invalid")
    for role, expected in files.items():
        if not isinstance(expected, dict) or "path" not in expected:
            raise RuntimeError("depth-state runtime file receipt is invalid")
        if _runtime_stat(expected["path"]) != expected:
            raise RuntimeError(f"depth-state native runtime changed: {role}")
    if (_platform_identity() != identity["platform"] or
            _gpu_driver_identity() != identity["gpus"]):
        raise RuntimeError("depth-state GPU/driver runtime changed")
    return identity


def runtime_snapshot(build_dir, *, force_refresh=False):
    """Resolve one stat-authenticated native-runtime snapshot per build.

    A 400+ MB TensorRT inference DLL is normally hashed once, then subsequent
    batch drivers reuse a receipt only while every path/size/mtime/device/inode
    and the live NVIDIA GPU/driver query still match.  A top-level orchestrator
    uses ``force_refresh`` once per invocation so same-stat DLL replacement
    cannot retain an old runtime identity.  Builder resource DLLs are excluded
    because an existing serialized engine never loads them.
    """
    build_dir, paths = _runtime_paths(build_dir)
    receipt_path = build_dir / RUNTIME_RECEIPT_NAME
    if (not force_refresh and receipt_path.is_file() and
            not receipt_path.is_symlink()):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (not isinstance(receipt, dict) or
                    receipt.get("schema") != RUNTIME_RECEIPT_SCHEMA or
                    receipt.get("contract") != RUNTIME_RECEIPT_CONTRACT or
                    set(receipt.get("files", {})) != set(paths) or
                    receipt.get("identity_sha256") !=
                    artifact_cache.canonical_sha256(receipt.get("identity"))):
                raise RuntimeError("native runtime receipt contract differs")
            snapshot = {
                "identity": receipt["identity"],
                "files": receipt["files"],
            }
            verify_runtime_snapshot(snapshot)
            return snapshot
        except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError,
                TypeError, ValueError):
            # A stale receipt is only an optimization miss.  Rebuild it from
            # exact native bytes rather than accepting or deleting any cache.
            pass
    snapshot = _build_runtime_snapshot(paths)
    receipt = {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "contract": RUNTIME_RECEIPT_CONTRACT,
        "identity_sha256":
            artifact_cache.canonical_sha256(snapshot["identity"]),
        **snapshot,
    }
    temporary = receipt_path.with_name(
        f".{receipt_path.name}.{os.getpid()}.{secrets.token_hex(12)}.partial"
    )
    try:
        temporary.write_bytes(artifact_cache.canonical_bytes(receipt))
        os.replace(temporary, receipt_path)
        try:
            published_bytes = receipt_path.read_bytes()
            published = json.loads(published_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "cannot authenticate published depth-state runtime receipt"
            ) from error
        if (published_bytes != artifact_cache.canonical_bytes(published) or
                published != receipt):
            raise RuntimeError(
                "depth-state runtime receipt publication winner differs"
            )
    finally:
        temporary.unlink(missing_ok=True)
    verify_runtime_snapshot(snapshot)
    return snapshot


def runtime_identity(build_dir):
    """Compatibility wrapper returning only the authenticated key identity."""
    return runtime_snapshot(build_dir)["identity"]


def identity_sha256(value):
    """Return the canonical content-addressed key for one depth identity."""
    return artifact_cache.DirectoryArtifactCache.key(value)


def validate_runtime_identity(value):
    if (not isinstance(value, dict) or
            value.get("schema") != RUNTIME_SCHEMA or
            value.get("contract") != RUNTIME_CONTRACT or
            value.get("runtime_namespace") !=
            "windows-nvidia-d3d11-tensorrt-inference-v1" or
            not isinstance(value.get("native_files"), dict) or
            not value["native_files"] or
            not isinstance(value.get("gpus"), list) or not value["gpus"]):
        raise RuntimeError("depth-state native runtime identity is invalid")
    artifact_cache.canonical_bytes(value)
    return value


def _normalized_estimator_extra(extra):
    """Remove only output-geometry arguments; retain every estimator-affecting token."""
    result = []
    index = 0
    extra = list(extra)
    while index < len(extra):
        option = extra[index]
        if option in OPTIONS_WITH_VALUES:
            if index + 1 >= len(extra):
                raise RuntimeError(f"depth-state option needs a value: {option}")
            if option not in GEOMETRY_OPTIONS:
                result.extend((option, extra[index + 1]))
            index += 2
            continue
        result.append(option)
        index += 1
    return result


def _condition(extra):
    extra = list(extra)
    native = "--native-hdr-scrgb" in extra
    simulated = "--simulate-hdr" in extra
    if native and simulated:
        raise RuntimeError("depth-state color condition is ambiguous")
    raw_white = None
    if "--sdr-white-level-raw" in extra:
        position = len(extra) - 1 - extra[::-1].index("--sdr-white-level-raw")
        if position + 1 >= len(extra):
            raise RuntimeError("depth-state white-level option is incomplete")
        raw_white = int(extra[position + 1])
    return {
        "input_kind": (
            "native-pq-scrgb16" if native else
            "simulated-sdr-in-windows-hdr" if simulated else
            "native-sdr-srgb8"
        ),
        "sdr_white_level_raw": raw_white if simulated else None,
        "model_input_source": "model_source-scrgb16" if native else "frame-image",
    }


def _engine_identity(repo, build_dir, model):
    header = Path(repo) / "src" / "model_manager.h"
    contents = header.read_text(encoding="utf-8")
    recipe_match = re.search(r'depth_engine_recipe\[\]\s*=\s*"([^"]+)"', contents)
    width_match = re.search(r"depth_engine_opt_width\s*=\s*([0-9]+)", contents)
    height_match = re.search(r"depth_engine_opt_height\s*=\s*([0-9]+)", contents)
    if not recipe_match or not width_match or not height_match:
        raise RuntimeError("cannot resolve production TensorRT recipe")
    recipe = recipe_match.group(1)
    engine = Path(build_dir) / "assets" / f"{model}.{recipe}.engine"
    return {
        "model": model,
        "recipe": recipe,
        "input_width": int(width_match.group(1)),
        "input_height": int(height_match.group(1)),
        "engine": _file(engine),
        "model_manager_header": _file(header),
    }


def _depth_shader_identities(shader_root):
    """Hash the exact transitive local-include closure of depth shaders."""
    shader_root = Path(shader_root).resolve(strict=True)
    identities = {}
    visiting = set()

    def visit(relative):
        relative = Path(relative)
        candidate = (shader_root / relative).resolve(strict=True)
        try:
            normalized = candidate.relative_to(shader_root).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"depth shader include escapes shader root: {relative}"
            ) from error
        if normalized in identities:
            return
        if normalized in visiting:
            raise RuntimeError(f"depth shader include cycle: {normalized}")
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(
                f"depth shader include is missing/plain-file invalid: {candidate}"
            )
        visiting.add(normalized)
        try:
            contents = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise RuntimeError(f"cannot read depth shader: {candidate}") from error
        identities[normalized] = _file(candidate)
        for include in SHADER_INCLUDE_PATTERN.findall(contents):
            include_path = Path(include.replace("\\", "/"))
            if include_path.is_absolute() or ".." in include_path.parts:
                raise RuntimeError(
                    f"depth shader has unsafe local include: {include!r}"
                )
            # D3D's standard include handler first resolves relative to the
            # including file, which is the only include form used here.
            child = (candidate.parent / include_path).resolve(strict=False)
            try:
                child_relative = child.relative_to(shader_root)
            except ValueError as error:
                raise RuntimeError(
                    f"depth shader include escapes shader root: {include!r}"
                ) from error
            visit(child_relative)
        visiting.remove(normalized)

    for entry in DEPTH_SHADER_ENTRY_FILES:
        visit(entry)
    return {name: identities[name] for name in sorted(identities)}


def _validated_content_rows(rows):
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("depth-state source content row is invalid")
        path = str(row.get("path", "")).replace("\\", "/")
        size = row.get("size")
        digest = row.get("sha256")
        if (not path or path.startswith("/") or
                any(part in {"", ".", ".."} for part in path.split("/")) or
                not isinstance(size, int) or isinstance(size, bool) or size < 0 or
                not isinstance(digest, str) or
                re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise RuntimeError("depth-state source content row is invalid")
        normalized.append({"path": path, "size": size, "sha256": digest})
    normalized.sort(key=lambda row: row["path"])
    if len({row["path"] for row in normalized}) != len(normalized):
        raise RuntimeError("depth-state source content rows repeat a path")
    return normalized


def _sdr_source_sequence(content_rows, source_ids):
    by_id = {}
    for row in _validated_content_rows(content_rows):
        match = re.fullmatch(r"frame_([0-9]+)[.](png|jpg|jpeg)", row["path"], re.I)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id in by_id:
            raise RuntimeError(f"depth-state source repeats frame {frame_id}")
        by_id[frame_id] = row
    if set(by_id) != set(source_ids):
        raise RuntimeError("depth-state source rows do not exactly cover source frames")
    return [{
        "source_frame_id": int(frame_id),
        "model_input": by_id[frame_id],
    } for frame_id in source_ids]


def _native_source_sequence(clip_dir, source_ids):
    authentication = native_hdr_capture.validate_clip(clip_dir, full=False)
    frames = authentication["frames"]
    if set(frames) != set(source_ids):
        raise RuntimeError(
            "native-HDR model-source manifest does not exactly cover source frames"
        )
    rows = []
    for frame_id in source_ids:
        row = frames[frame_id]
        rows.append({
            "source_frame_id": int(frame_id),
            "model_input": {
                "path": str(row["path"]).replace("\\", "/"),
                "size": row["size"],
                "sha256": row["sha256"],
            },
            "dimension_source_preview": {
                "path": str(row["preview"]).replace("\\", "/"),
                "size": row["preview_path"].stat().st_size,
                "sha256": row["preview_sha256"],
            },
        })
    return {
        "contract": "authenticated-native-pq-model-sources-v1",
        "content_sha256": authentication["content_sha256"],
        "width": authentication["width"],
        "height": authentication["height"],
        "frames": rows,
    }


def _source_sequence(clip_dir, content_rows, source_ids, *, native_hdr):
    if native_hdr:
        return _native_source_sequence(clip_dir, source_ids)
    return {
        "contract": "authenticated-sdr-model-sources-v1",
        "frames": _sdr_source_sequence(content_rows, source_ids),
    }


def identity(*, repo, build_dir, conf_sha256, executable_sha256, model,
             clip_dir, source_content_rows, source_ids, selected_frame_ids, extra,
             runtime):
    """Return a canonical, path-independent whole-sequence cache identity."""
    repo = Path(repo).resolve()
    build_dir = Path(build_dir).resolve()
    condition = _condition(extra)
    runtime = validate_runtime_identity(runtime)
    source_identity = _source_sequence(
        clip_dir, source_content_rows, source_ids,
        native_hdr=condition["input_kind"] == "native-pq-scrgb16",
    )
    shader_root = build_dir / "assets" / "shaders" / "directx"
    shader_rows = _depth_shader_identities(shader_root)
    color_contract_path = (
        repo / "tools" / "depth_models" / "depth_input_color_contract.json"
    )
    value = artifact_cache.cache_identity(
        artifact_kind=ARTIFACT_KIND,
        source={
            **source_identity,
            "ordered_source_frame_ids": [int(value) for value in source_ids],
        },
        selection={
            "source_frame_ids": [int(value) for value in source_ids],
            "selected_frame_ids": [int(value) for value in selected_frame_ids],
            "selected_payload": "label-targets-only",
            "runtime_scene_rows": "all-completed-source-frames",
        },
        preprocessing={
            "schema": IDENTITY_SCHEMA,
            "contract": IDENTITY_CONTRACT,
            "condition": condition,
            "estimator_extra_without_output_geometry":
                _normalized_estimator_extra(extra),
            "configuration_sha256": conf_sha256,
            "engine": _engine_identity(repo, build_dir, model),
            "artistic_policy": False,
            "artistic_scale_override": 1.0,
            "depth_every": 1,
        },
        color_contract={
            "depth_input_color_contract": _file(color_contract_path),
            "condition": condition,
        },
        code={
            "executable_sha256": executable_sha256,
            "depth_shader_sources": shader_rows,
            "sequence_contract": "apollo-production-depth-state-sequence-v1",
            "native_runtime": runtime,
            "cache_identity_source": _file(Path(__file__).resolve()),
            "native_hdr_capture_source":
                _file(Path(native_hdr_capture.__file__).resolve()),
        },
    )
    return value


def cache(root):
    return artifact_cache.DirectoryArtifactCache(root)


def validated_sequence(cache_store, identity):
    """Resolve a verified CAS hit and pin its inner sequence manifest.

    The C++ consumer receives the returned inner-manifest digest and verifies the
    exact bytes it parses.  Payload resources are then verified against that pinned
    manifest, closing the validation/process-launch TOCTOU window without copying a
    multi-gigabyte object into every render workspace.
    """
    hit = cache_store.validated_payload_receipt(identity)
    if hit is None:
        return None
    payload, receipt = hit
    manifest_rows = [
        row for row in receipt["files"]
        if row.get("path") == INNER_MANIFEST
    ]
    if len(manifest_rows) != 1:
        raise RuntimeError(
            "depth-state cache receipt has no unique inner manifest"
        )
    row = manifest_rows[0]
    digest = row.get("sha256")
    if (not isinstance(digest, str) or
            re.fullmatch(r"[0-9a-f]{64}", digest) is None or
            not isinstance(row.get("bytes"), int) or row["bytes"] <= 0):
        raise RuntimeError("depth-state cache inner-manifest receipt is invalid")
    return {
        "payload": payload,
        "inner_manifest_sha256": digest,
        "outer_payload_manifest_sha256":
            receipt["payload_manifest_sha256"],
    }


def require_working_split(split):
    return artifact_cache.require_working_split(split)
