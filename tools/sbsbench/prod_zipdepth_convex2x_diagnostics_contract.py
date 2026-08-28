"""Authenticated whole-clip diagnostics for the fused DAV2 + ZipDepth stage.

The refined field is diagnostic-only, but a result that advertises the composite runtime must
still bind the exact same-frame coarse DAV2, 2x RGB guidance, and refined tensors it produced.
This module validates the schema-1 harness sidecar, records every tensor hash in results metadata,
and supports a second byte-for-byte authentication pass after scoring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


EVALUATOR_SCHEMA = 37
HARNESS_CONTRACT_SCHEMA = 22
SIDECAR_SCHEMA = 1
MANIFEST_SCHEMA = 1
RESULTS_META_KEY = "prod_zipdepth_convex2x_diagnostic_artifacts"
BINDING = "schema-37-results-to-schema-22-prod-zipdepth-convex2x-diagnostics-v1"
SIDECAR_FILENAME = "prod_zipdepth_convex2x_diagnostics.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FRAME_FILE = {
    "raw": re.compile(r"raw_(\d{5,})\.f32"),
    "guidance": re.compile(r"zip_model_input_(\d{5,})\.f32"),
    "refined": re.compile(r"refined_(\d{5,})\.f32"),
}
_FILE_PREFIX = {
    "raw": "raw_",
    "guidance": "zip_model_input_",
    "refined": "refined_",
}
_SIDECAR_KEYS = {
    "schema", "authority", "same_frame_binding", "frame_count", "coarse_dav2",
    "zip_model_input", "refined_depth", "composite_runtime_provenance",
    "raw_dav2_provenance",
}
_AUTHORITY = {
    "role": "diagnostic-only-stage-1",
    "live_geometry_authority": False,
    "scoring_depth_authority": False,
    "coarse_dav2_remains_live_authority": True,
}
_COARSE_KEYS = {
    "width", "height", "tensor_shape_nchw", "raw_file_pattern", "role",
}
_GUIDANCE_KEYS = {
    "width", "height", "tensor_shape_nchw", "dtype", "layout", "file_pattern", "stage",
}
_COMPOSITE_KEYS = {
    "model", "onnx_sha256", "embedded_dav2_onnx_sha256",
    "zipdepth_checkpoint_sha256", "guidance_preprocess_source_closure_sha256",
    "engine_recipe", "engine_artifact", "active_engine_manifest",
}
_RAW_PROVENANCE_KEYS = {
    "model", "depth_model_url", "onnx_sha256", "preprocess_profile",
    "preprocess_source_closure_sha256",
}
_CONTRACT_RAW_PROVENANCE_KEYS = {
    "schema", *_RAW_PROVENANCE_KEYS, "raw_width", "raw_height",
}
_MANIFEST_KEYS = {
    "schema", "binding", "evaluator_schema", "harness_contract_schema",
    "contract_json_sha256", "sidecar", "sidecar_sha256", "authority", "frame_count",
    "tensor_shapes", "composite_runtime_provenance", "raw_dav2_provenance", "frames",
}
_FRAME_KEYS = {"frame_id", "raw", "guidance", "refined"}
_FILE_RECORD_KEYS = {"file", "bytes", "sha256"}
_TENSOR_SHAPE_KEYS = {"width", "height", "channels"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash fused diagnostic artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_frame_ids(frame_ids: Iterable[int]) -> List[int]:
    values = list(frame_ids)
    if (not values or any(type(value) is not int or value < 0 for value in values) or
            values != sorted(set(values))):
        raise ValueError("fused diagnostic frame IDs must be non-empty, unique, and increasing")
    return values


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_raw_provenance(value: Any) -> Dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != _RAW_PROVENANCE_KEYS or
            not isinstance(value.get("model"), str) or not value["model"] or
            not isinstance(value.get("depth_model_url"), str) or
            not isinstance(value.get("preprocess_profile"), str) or
            not _valid_sha(value.get("onnx_sha256")) or
            not _valid_sha(value.get("preprocess_source_closure_sha256"))):
        raise ValueError("fused diagnostic sidecar has invalid raw DAV2 provenance")
    return value


def _validate_contract_raw_provenance(
        value: Any, width: int, height: int, sidecar_value: Mapping[str, Any]
) -> Dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != _CONTRACT_RAW_PROVENANCE_KEYS or
            value.get("schema") != 1 or value.get("raw_width") != width or
            value.get("raw_height") != height or
            {key: value.get(key) for key in _RAW_PROVENANCE_KEYS} != sidecar_value):
        raise ValueError(
            "fused diagnostic raw provenance disagrees with the harness contract")
    return value


def _validate_composite_provenance(value: Any) -> Dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != _COMPOSITE_KEYS or
            any(not _valid_sha(value.get(key)) for key in (
                "onnx_sha256", "embedded_dav2_onnx_sha256",
                "zipdepth_checkpoint_sha256", "guidance_preprocess_source_closure_sha256")) or
            any(not isinstance(value.get(key), str) or not value[key] for key in (
                "model", "engine_recipe", "engine_artifact", "active_engine_manifest")) or
            Path(value["engine_artifact"]).name != value["engine_artifact"] or
            Path(value["active_engine_manifest"]).name != value["active_engine_manifest"]):
        raise ValueError("fused diagnostic sidecar has invalid composite runtime provenance")
    return value


def _validated_sidecar(
        path: Path,
        expected_frame_count: Optional[int] = None,
        expected_composite_runtime: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    sidecar = _json_object(path, "fused diagnostic sidecar")
    if (set(sidecar) != _SIDECAR_KEYS or sidecar.get("schema") != SIDECAR_SCHEMA or
            sidecar.get("authority") != _AUTHORITY or
            sidecar.get("same_frame_binding") !=
            "completed_frame_id-to-decimal-frame-id" or
            not _positive_int(sidecar.get("frame_count")) or
            (expected_frame_count is not None and
             sidecar.get("frame_count") != expected_frame_count)):
        raise ValueError("fused diagnostic sidecar has missing or unknown schema-1 semantics")

    coarse = sidecar.get("coarse_dav2")
    guidance = sidecar.get("zip_model_input")
    refined = sidecar.get("refined_depth")
    if (not isinstance(coarse, dict) or set(coarse) != _COARSE_KEYS or
            not _positive_int(coarse.get("width")) or
            not _positive_int(coarse.get("height")) or
            coarse.get("tensor_shape_nchw") !=
            [1, 1, coarse.get("height"), coarse.get("width")] or
            coarse.get("raw_file_pattern") != "raw_<frame-id>.f32" or
            coarse.get("role") != "existing-prod-raw-output-and-only-live-depth-authority"):
        raise ValueError("fused diagnostic sidecar has invalid coarse DAV2 semantics")
    expected_guidance_shape = [1, 3, 2 * coarse["height"], 2 * coarse["width"]]
    if (not isinstance(guidance, dict) or set(guidance) != _GUIDANCE_KEYS or
            guidance.get("width") != 2 * coarse["width"] or
            guidance.get("height") != 2 * coarse["height"] or
            guidance.get("tensor_shape_nchw") != expected_guidance_shape or
            guidance.get("dtype") != "float32-le" or
            guidance.get("layout") != "nchw-contiguous" or
            guidance.get("file_pattern") != "zip_model_input_<frame-id>.f32" or
            guidance.get("stage") !=
            "matched native RGB independently preprocessed on the exact 2x grid"):
        raise ValueError("fused diagnostic sidecar has invalid guidance semantics")
    if (not isinstance(refined, dict) or set(refined) != _GUIDANCE_KEYS or
            refined.get("width") != guidance["width"] or
            refined.get("height") != guidance["height"] or
            refined.get("tensor_shape_nchw") !=
            [1, 1, refined.get("height"), refined.get("width")] or
            refined.get("dtype") != "float32-le" or
            refined.get("layout") != "nchw-contiguous" or
            refined.get("file_pattern") != "refined_<frame-id>.f32" or
            refined.get("stage") != "fused ONNX refined_depth after frozen ZipDepth convex2x"):
        raise ValueError("fused diagnostic sidecar has invalid refined-depth semantics")

    composite = _validate_composite_provenance(
        sidecar.get("composite_runtime_provenance"))
    raw = _validate_raw_provenance(sidecar.get("raw_dav2_provenance"))
    if (raw["onnx_sha256"] != composite["embedded_dav2_onnx_sha256"] or
            raw["preprocess_source_closure_sha256"] !=
            composite["guidance_preprocess_source_closure_sha256"]):
        raise ValueError("fused diagnostic sidecar raw/composite provenance is inconsistent")
    if expected_composite_runtime is not None:
        expected = {key: expected_composite_runtime.get(key) for key in _COMPOSITE_KEYS}
        if composite != expected:
            raise ValueError(
                "fused diagnostic sidecar disagrees with evaluator composite runtime identity")
    return sidecar


def _tensor_shapes(sidecar: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    coarse = sidecar["coarse_dav2"]
    guidance = sidecar["zip_model_input"]
    refined = sidecar["refined_depth"]
    return {
        "raw": {"width": coarse["width"], "height": coarse["height"], "channels": 1},
        "guidance": {
            "width": guidance["width"], "height": guidance["height"], "channels": 3,
        },
        "refined": {
            "width": refined["width"], "height": refined["height"], "channels": 1,
        },
    }


def _expected_bytes(shape: Mapping[str, int]) -> int:
    return shape["width"] * shape["height"] * shape["channels"] * 4


def _inspect_float_file(path: Path, expected_bytes: int) -> Dict[str, Any]:
    try:
        observed_bytes = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect fused diagnostic tensor {path}: {exc}") from exc
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"fused diagnostic tensor {path.name} has {observed_bytes} bytes; "
            f"expected {expected_bytes}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                if len(block) % 4 != 0 or not np.isfinite(
                        np.frombuffer(block, dtype="<f4")).all():
                    raise ValueError(
                        f"fused diagnostic tensor {path.name} contains non-finite values")
    except OSError as exc:
        raise ValueError(f"cannot read fused diagnostic tensor {path}: {exc}") from exc
    return {"file": path.name, "bytes": observed_bytes, "sha256": digest.hexdigest()}


def _actual_tensor_files(artifact_dir: Path, kind: str) -> set[str]:
    prefix = _FILE_PREFIX[kind]
    pattern = _FRAME_FILE[kind]
    files: set[str] = set()
    for path in artifact_dir.glob(prefix + "*.f32"):
        match = pattern.fullmatch(path.name)
        if match is None or path.name != f"{prefix}{int(match.group(1)):05d}.f32":
            raise ValueError(f"fused diagnostic artifact has non-canonical identity {path.name!r}")
        files.add(path.name)
    return files


def validate_manifest(
        manifest: Any,
        expected_frame_ids: Optional[Iterable[int]] = None,
        expected_composite_runtime: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Validate manifest semantics and return its ordered frame records."""

    if (not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS or
            manifest.get("schema") != MANIFEST_SCHEMA or
            manifest.get("binding") != BINDING or
            manifest.get("evaluator_schema") != EVALUATOR_SCHEMA or
            manifest.get("harness_contract_schema") != HARNESS_CONTRACT_SCHEMA or
            manifest.get("sidecar") != SIDECAR_FILENAME or
            not _valid_sha(manifest.get("contract_json_sha256")) or
            not _valid_sha(manifest.get("sidecar_sha256")) or
            manifest.get("authority") != _AUTHORITY or
            not _positive_int(manifest.get("frame_count"))):
        raise ValueError("fused diagnostic artifact manifest has missing or unknown semantics")
    shapes = manifest.get("tensor_shapes")
    if not isinstance(shapes, dict) or set(shapes) != {"raw", "guidance", "refined"}:
        raise ValueError("fused diagnostic artifact manifest has invalid tensor shapes")
    expected_channels = {"raw": 1, "guidance": 3, "refined": 1}
    for kind, channels in expected_channels.items():
        shape = shapes.get(kind)
        if (not isinstance(shape, dict) or set(shape) != _TENSOR_SHAPE_KEYS or
                not _positive_int(shape.get("width")) or
                not _positive_int(shape.get("height")) or
                shape.get("channels") != channels):
            raise ValueError("fused diagnostic artifact manifest has invalid tensor shapes")
    if (shapes["guidance"]["width"] != 2 * shapes["raw"]["width"] or
            shapes["guidance"]["height"] != 2 * shapes["raw"]["height"] or
            shapes["refined"] != {
                "width": shapes["guidance"]["width"],
                "height": shapes["guidance"]["height"], "channels": 1,
            }):
        raise ValueError("fused diagnostic artifact manifest violates the exact 2x relation")
    composite = _validate_composite_provenance(
        manifest.get("composite_runtime_provenance"))
    raw = _validate_raw_provenance(manifest.get("raw_dav2_provenance"))
    if (raw["onnx_sha256"] != composite["embedded_dav2_onnx_sha256"] or
            raw["preprocess_source_closure_sha256"] !=
            composite["guidance_preprocess_source_closure_sha256"]):
        raise ValueError("fused diagnostic artifact manifest provenance is inconsistent")
    if expected_composite_runtime is not None:
        expected = {key: expected_composite_runtime.get(key) for key in _COMPOSITE_KEYS}
        if composite != expected:
            raise ValueError(
                "fused diagnostic artifact manifest disagrees with composite runtime identity")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != manifest["frame_count"]:
        raise ValueError("fused diagnostic artifact manifest has invalid frame coverage")
    observed_ids: List[int] = []
    for index, row in enumerate(frames):
        if (not isinstance(row, dict) or set(row) != _FRAME_KEYS or
                not isinstance(row.get("frame_id"), str) or
                not re.fullmatch(r"\d{5,}", row["frame_id"])):
            raise ValueError(f"fused diagnostic artifact manifest frame {index} is invalid")
        frame_id = int(row["frame_id"])
        if row["frame_id"] != f"{frame_id:05d}":
            raise ValueError(f"fused diagnostic artifact manifest frame {index} is non-canonical")
        observed_ids.append(frame_id)
        for kind in ("raw", "guidance", "refined"):
            record = row.get(kind)
            expected_file = f"{_FILE_PREFIX[kind]}{row['frame_id']}.f32"
            if (not isinstance(record, dict) or set(record) != _FILE_RECORD_KEYS or
                    record.get("file") != expected_file or
                    type(record.get("bytes")) is not int or
                    record["bytes"] != _expected_bytes(shapes[kind]) or
                    not _valid_sha(record.get("sha256"))):
                raise ValueError(
                    f"fused diagnostic artifact manifest frame {index} {kind} record is invalid")
    if observed_ids != sorted(set(observed_ids)):
        raise ValueError("fused diagnostic artifact manifest frame IDs are not unique/increasing")
    if expected_frame_ids is not None and observed_ids != _canonical_frame_ids(expected_frame_ids):
        raise ValueError(
            "fused diagnostic artifact manifest does not cover the expected frame IDs")
    return frames


def authenticate_manifest_files(
        artifact_dir: Path,
        manifest: Mapping[str, Any],
        expected_frame_ids: Optional[Iterable[int]] = None,
        expected_composite_runtime: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Revalidate the sidecar and rehash every declared finite diagnostic tensor."""

    frames = validate_manifest(
        manifest, expected_frame_ids, expected_composite_runtime)
    contract_path = artifact_dir / "contract.json"
    contract = _json_object(contract_path, "fused diagnostic harness contract")
    if contract.get("schema") != HARNESS_CONTRACT_SCHEMA:
        raise ValueError(
            f"fused diagnostic harness schema must be {HARNESS_CONTRACT_SCHEMA}, got "
            f"{contract.get('schema')!r}")
    if file_sha256(contract_path) != manifest["contract_json_sha256"]:
        raise ValueError("fused diagnostic harness contract changed after evaluator capture")
    sidecar_path = artifact_dir / SIDECAR_FILENAME
    sidecar = _validated_sidecar(
        sidecar_path, manifest["frame_count"], expected_composite_runtime)
    if file_sha256(sidecar_path) != manifest["sidecar_sha256"]:
        raise ValueError("fused diagnostic sidecar changed after evaluator capture")
    expected_reference = {
        "schema": SIDECAR_SCHEMA,
        "sidecar": SIDECAR_FILENAME,
        "sidecar_sha256": manifest["sidecar_sha256"],
        "frame_count": manifest["frame_count"],
        "refined_file_pattern": "refined_<frame-id>.f32",
        "guidance_file_pattern": "zip_model_input_<frame-id>.f32",
        "authority": "diagnostic-only-never-live-or-scoring-depth",
    }
    if contract.get("prod_zipdepth_convex2x_diagnostics") != expected_reference:
        raise ValueError("fused diagnostic harness contract has an invalid sidecar reference")
    contract_raw = _validate_contract_raw_provenance(
        contract.get("raw_model_provenance"),
        manifest["tensor_shapes"]["raw"]["width"],
        manifest["tensor_shapes"]["raw"]["height"],
        sidecar["raw_dav2_provenance"])
    if (manifest["tensor_shapes"] != _tensor_shapes(sidecar) or
            manifest["authority"] != sidecar["authority"] or
            manifest["composite_runtime_provenance"] !=
            sidecar["composite_runtime_provenance"] or
            manifest["raw_dav2_provenance"] != sidecar["raw_dav2_provenance"] or
            contract_raw["onnx_sha256"] !=
            manifest["composite_runtime_provenance"]["embedded_dav2_onnx_sha256"]):
        raise ValueError("fused diagnostic manifest disagrees with its harness sidecar")

    declared = {
        kind: {row[kind]["file"] for row in frames}
        for kind in ("raw", "guidance", "refined")
    }
    for kind in declared:
        if _actual_tensor_files(artifact_dir, kind) != declared[kind]:
            raise ValueError(
                f"fused diagnostic {kind} files disagree with their run manifest")
    for row in frames:
        for kind in ("raw", "guidance", "refined"):
            observed = _inspect_float_file(
                artifact_dir / row[kind]["file"],
                _expected_bytes(manifest["tensor_shapes"][kind]))
            if observed != row[kind]:
                raise ValueError(
                    f"fused diagnostic tensor {row[kind]['file']} changed after evaluator capture")
    return frames


def build_manifest(
        artifact_dir: Path,
        frame_ids: Iterable[int],
        expected_composite_runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build and immediately self-verify one clip's fused diagnostic manifest."""

    ids = _canonical_frame_ids(frame_ids)
    contract_path = artifact_dir / "contract.json"
    contract = _json_object(contract_path, "fused diagnostic harness contract")
    if contract.get("schema") != HARNESS_CONTRACT_SCHEMA:
        raise ValueError(
            f"fused diagnostic harness schema must be {HARNESS_CONTRACT_SCHEMA}, got "
            f"{contract.get('schema')!r}")
    sidecar_path = artifact_dir / SIDECAR_FILENAME
    sidecar = _validated_sidecar(sidecar_path, len(ids), expected_composite_runtime)
    sidecar_sha256 = file_sha256(sidecar_path)
    expected_reference = {
        "schema": SIDECAR_SCHEMA,
        "sidecar": SIDECAR_FILENAME,
        "sidecar_sha256": sidecar_sha256,
        "frame_count": len(ids),
        "refined_file_pattern": "refined_<frame-id>.f32",
        "guidance_file_pattern": "zip_model_input_<frame-id>.f32",
        "authority": "diagnostic-only-never-live-or-scoring-depth",
    }
    if contract.get("prod_zipdepth_convex2x_diagnostics") != expected_reference:
        raise ValueError("fused diagnostic harness contract has an invalid sidecar reference")
    _validate_contract_raw_provenance(
        contract.get("raw_model_provenance"),
        sidecar["coarse_dav2"]["width"], sidecar["coarse_dav2"]["height"],
        sidecar["raw_dav2_provenance"])
    shapes = _tensor_shapes(sidecar)
    frames = []
    for frame_id in ids:
        identity = f"{frame_id:05d}"
        frames.append({
            "frame_id": identity,
            **{
                kind: _inspect_float_file(
                    artifact_dir / f"{_FILE_PREFIX[kind]}{identity}.f32",
                    _expected_bytes(shapes[kind]))
                for kind in ("raw", "guidance", "refined")
            },
        })
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "binding": BINDING,
        "evaluator_schema": EVALUATOR_SCHEMA,
        "harness_contract_schema": HARNESS_CONTRACT_SCHEMA,
        "contract_json_sha256": file_sha256(contract_path),
        "sidecar": SIDECAR_FILENAME,
        "sidecar_sha256": sidecar_sha256,
        "authority": sidecar["authority"],
        "frame_count": len(ids),
        "tensor_shapes": shapes,
        "composite_runtime_provenance": sidecar["composite_runtime_provenance"],
        "raw_dav2_provenance": sidecar["raw_dav2_provenance"],
        "frames": frames,
    }
    authenticate_manifest_files(
        artifact_dir, manifest, ids, expected_composite_runtime)
    return manifest


__all__ = [
    "BINDING",
    "EVALUATOR_SCHEMA",
    "HARNESS_CONTRACT_SCHEMA",
    "MANIFEST_SCHEMA",
    "RESULTS_META_KEY",
    "SIDECAR_FILENAME",
    "SIDECAR_SCHEMA",
    "authenticate_manifest_files",
    "build_manifest",
    "file_sha256",
    "validate_manifest",
]
