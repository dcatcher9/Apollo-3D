"""Authenticate fused DAV2 + ZipDepth model-boundary artifacts.

Schema 2 describes the active single-high model: one high-resolution ``pixel_values`` input and
one same-size ``refined_depth`` output. The harness writes one input snapshot and reuses the
ordinary ``raw_<frame>.f32`` artifact as the output record; the estimator's retired guidance and
refined snapshot fields are compatibility aliases to those two primary GPU resources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

try:
    from . import depth_coordinate_v2_contract as coordinate_contract
    from . import prod_zipdepth_convex2x as convex2x_contract
except ImportError:  # Direct execution from tools/sbsbench.
    import depth_coordinate_v2_contract as coordinate_contract  # type: ignore
    import prod_zipdepth_convex2x as convex2x_contract  # type: ignore


EVALUATOR_SCHEMA = 38
HARNESS_CONTRACT_SCHEMA = 22
SIDECAR_SCHEMA = 2
MANIFEST_SCHEMA = 2
BINDING = "schema-38-results-to-schema-22-prod-zipdepth-convex2x-diagnostics-v2"
RESULTS_META_KEY = "prod_zipdepth_convex2x_diagnostic_artifacts"
SIDECAR_FILENAME = "prod_zipdepth_convex2x_diagnostics.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ACTIVE_AUTHORITY = {
    "role": "authenticated-single-high-model-io",
    "live_geometry_source": "refined_depth",
    "scoring_depth_source": "raw_<frame-id>.f32",
    "coarse_dav2_public_binding": False,
}
_ACTIVE_ALIASES = {
    "model_input_primary": "model_input_snapshot",
    "model_input_compatibility_alias": "guidance_model_input_snapshot",
    "refined_depth_primary": "raw_model_depth_snapshot",
    "refined_depth_compatibility_alias": "refined_model_depth_snapshot",
    "gpu_resource_policy": "compatibility-aliases-reference-primary-resources",
    "duplicate_gpu_resources": False,
}
_ACTIVE_FILES = {
    "input": ("model_input_", re.compile(r"model_input_(\d{5,})\.f32")),
    "output": ("raw_", re.compile(r"raw_(\d{5,})\.f32")),
}
_TENSOR_KEYS = {
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
    "tensor_shapes", "composite_runtime_provenance", "embedded_dav2_provenance", "frames",
}
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
        raise ValueError("fused diagnostic sidecar has invalid embedded DAV2 provenance")
    return value


def _validate_contract_raw_provenance(
        value: Any, width: int, height: int, sidecar_value: Mapping[str, Any]
) -> Dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != _CONTRACT_RAW_PROVENANCE_KEYS or
            value.get("schema") != 1 or value.get("raw_width") != width or
            value.get("raw_height") != height or
            {key: value.get(key) for key in _RAW_PROVENANCE_KEYS} != sidecar_value):
        raise ValueError(
            "fused diagnostic embedded DAV2 provenance disagrees with the harness contract")
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


def _validate_expected_runtime(
        composite: Mapping[str, Any], expected: Optional[Mapping[str, Any]]) -> None:
    if expected is not None:
        projected = {key: expected.get(key) for key in _COMPOSITE_KEYS}
        if composite != projected:
            raise ValueError(
                "fused diagnostic sidecar disagrees with evaluator composite runtime identity")


def _validate_active_grid_calibration(
        width: int, height: int, embedded: Mapping[str, Any]) -> tuple[int, int]:
    """Authenticate one public high profile and its embedded DAV2 calibration join.

    This intentionally remains a private, unconditional production gate. Small unit fixtures may
    monkeypatch it explicitly, but no evaluator or reader entry point exposes a bypass argument.
    """

    high = convex2x_contract.Shape(width, height)
    if high not in convex2x_contract.supported_high_shapes():
        raise ValueError(
            "fused diagnostic schema-2 tensor is not one of the supported high profiles")
    if width % 2 or height % 2:
        raise ValueError("fused diagnostic schema-2 high profile has no exact DAV2 half-shape")
    coarse = width // 2, height // 2
    calibration = coordinate_contract.find_model_calibration(
        embedded["model"], embedded["depth_model_url"], embedded["onnx_sha256"])
    if (calibration is None or
            embedded["preprocess_profile"] != calibration.preprocess.profile or
            embedded["preprocess_source_closure_sha256"] !=
            calibration.preprocess.source_closure_sha256 or
            coarse not in calibration.calibrated_input_shapes):
        raise ValueError(
            "fused diagnostic schema-2 high profile does not join its calibrated DAV2 half-shape")
    return coarse


def _validate_tensor(
        value: Any, width: int, height: int, channels: int,
        file_pattern: str, stage: str) -> None:
    if (not isinstance(value, dict) or set(value) != _TENSOR_KEYS or
            value.get("width") != width or value.get("height") != height or
            value.get("tensor_shape_nchw") != [1, channels, height, width] or
            value.get("dtype") != "float32-le" or
            value.get("layout") != "nchw-contiguous" or
            value.get("file_pattern") != file_pattern or value.get("stage") != stage):
        raise ValueError("fused diagnostic sidecar has invalid single-high tensor semantics")


def _validate_active_sidecar(sidecar: Dict[str, Any]) -> Dict[str, Any]:
    expected_keys = {
        "schema", "authority", "same_frame_binding", "frame_count", "model_input",
        "refined_depth", "diagnostic_aliases", "composite_runtime_provenance",
        "embedded_dav2_provenance",
    }
    if (set(sidecar) != expected_keys or sidecar.get("authority") != _ACTIVE_AUTHORITY or
            sidecar.get("diagnostic_aliases") != _ACTIVE_ALIASES):
        raise ValueError("fused diagnostic sidecar has missing or unknown schema-2 semantics")
    model_input = sidecar.get("model_input")
    if not isinstance(model_input, dict):
        raise ValueError("fused diagnostic sidecar has invalid single-high model input")
    width, height = model_input.get("width"), model_input.get("height")
    if not _positive_int(width) or not _positive_int(height):
        raise ValueError("fused diagnostic sidecar has invalid single-high dimensions")
    _validate_tensor(
        model_input, width, height, 3, "model_input_<frame-id>.f32",
        "sole fused ONNX high-resolution pixel_values binding after authenticated preprocess")
    _validate_tensor(
        sidecar.get("refined_depth"), width, height, 1, "raw_<frame-id>.f32",
        "sole fused ONNX refined_depth binding and live high-resolution depth source")
    return sidecar


def _validated_sidecar(
        path: Path,
        expected_frame_count: Optional[int] = None,
        expected_composite_runtime: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    sidecar = _json_object(path, "fused diagnostic sidecar")
    schema = sidecar.get("schema")
    if (type(schema) is not int or schema != SIDECAR_SCHEMA or
            sidecar.get("same_frame_binding") !=
            "completed_frame_id-to-decimal-frame-id" or
            not _positive_int(sidecar.get("frame_count")) or
            (expected_frame_count is not None and
             sidecar.get("frame_count") != expected_frame_count)):
        raise ValueError("fused diagnostic sidecar has missing or unknown schema semantics")
    _validate_active_sidecar(sidecar)
    embedded = _validate_raw_provenance(sidecar.get("embedded_dav2_provenance"))
    composite = _validate_composite_provenance(sidecar.get("composite_runtime_provenance"))
    if (embedded["onnx_sha256"] != composite["embedded_dav2_onnx_sha256"] or
            embedded["preprocess_source_closure_sha256"] !=
            composite["guidance_preprocess_source_closure_sha256"]):
        raise ValueError("fused diagnostic sidecar embedded/composite provenance is inconsistent")
    _validate_active_grid_calibration(
        sidecar["model_input"]["width"], sidecar["model_input"]["height"], embedded)
    _validate_expected_runtime(composite, expected_composite_runtime)
    return sidecar


def _tensor_shapes(sidecar: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    model_input = sidecar["model_input"]
    output = sidecar["refined_depth"]
    return {
        "input": {
            "width": model_input["width"], "height": model_input["height"], "channels": 3,
        },
        "output": {"width": output["width"], "height": output["height"], "channels": 1},
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


def _actual_tensor_files(artifact_dir: Path, prefix: str, pattern: re.Pattern[str]) -> set[str]:
    files: set[str] = set()
    for path in artifact_dir.glob(prefix + "*.f32"):
        match = pattern.fullmatch(path.name)
        if match is None or path.name != f"{prefix}{int(match.group(1)):05d}.f32":
            raise ValueError(f"fused diagnostic artifact has non-canonical identity {path.name!r}")
        files.add(path.name)
    return files


def _validate_shapes(shapes: Any) -> None:
    if not isinstance(shapes, dict) or set(shapes) != set(_ACTIVE_FILES):
        raise ValueError("fused diagnostic artifact manifest has invalid tensor shapes")
    for kind, expected_channels in {"input": 3, "output": 1}.items():
        shape = shapes.get(kind)
        if (not isinstance(shape, dict) or set(shape) != _TENSOR_SHAPE_KEYS or
                not _positive_int(shape.get("width")) or
                not _positive_int(shape.get("height")) or
                shape.get("channels") != expected_channels):
            raise ValueError("fused diagnostic artifact manifest has invalid tensor shapes")
    if (shapes["input"]["width"] != shapes["output"]["width"] or
            shapes["input"]["height"] != shapes["output"]["height"]):
        raise ValueError("fused diagnostic artifact violates the single-high shape identity")


def validate_manifest(
        manifest: Any,
        expected_frame_ids: Optional[Iterable[int]] = None,
        expected_composite_runtime: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Validate manifest semantics and return ordered frame records."""

    if not isinstance(manifest, dict):
        raise ValueError("fused diagnostic artifact manifest must be an object")
    schema = manifest.get("schema")
    if type(schema) is not int or schema != MANIFEST_SCHEMA:
        raise ValueError("fused diagnostic artifact manifest has unsupported schema")
    if (set(manifest) != _MANIFEST_KEYS or manifest.get("binding") != BINDING or
            manifest.get("evaluator_schema") != EVALUATOR_SCHEMA or
            manifest.get("harness_contract_schema") != HARNESS_CONTRACT_SCHEMA or
            manifest.get("sidecar") != SIDECAR_FILENAME or
            not _valid_sha(manifest.get("contract_json_sha256")) or
            not _valid_sha(manifest.get("sidecar_sha256")) or
            manifest.get("authority") != _ACTIVE_AUTHORITY or
            not _positive_int(manifest.get("frame_count"))):
        raise ValueError("fused diagnostic artifact manifest has missing or unknown semantics")
    shapes = manifest.get("tensor_shapes")
    _validate_shapes(shapes)
    composite = _validate_composite_provenance(manifest.get("composite_runtime_provenance"))
    embedded = _validate_raw_provenance(manifest.get("embedded_dav2_provenance"))
    if (embedded["onnx_sha256"] != composite["embedded_dav2_onnx_sha256"] or
            embedded["preprocess_source_closure_sha256"] !=
            composite["guidance_preprocess_source_closure_sha256"]):
        raise ValueError("fused diagnostic artifact manifest provenance is inconsistent")
    _validate_active_grid_calibration(
        shapes["input"]["width"], shapes["input"]["height"], embedded)
    _validate_expected_runtime(composite, expected_composite_runtime)

    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != manifest["frame_count"]:
        raise ValueError("fused diagnostic artifact manifest has invalid frame coverage")
    observed_ids: List[int] = []
    expected_frame_keys = {"frame_id", *_ACTIVE_FILES}
    for index, row in enumerate(frames):
        if (not isinstance(row, dict) or set(row) != expected_frame_keys or
                not isinstance(row.get("frame_id"), str) or
                not re.fullmatch(r"\d{5,}", row["frame_id"])):
            raise ValueError(f"fused diagnostic artifact manifest frame {index} is invalid")
        frame_id = int(row["frame_id"])
        if row["frame_id"] != f"{frame_id:05d}":
            raise ValueError(f"fused diagnostic artifact manifest frame {index} is non-canonical")
        observed_ids.append(frame_id)
        for kind, (prefix, _) in _ACTIVE_FILES.items():
            record = row.get(kind)
            expected_file = f"{prefix}{row['frame_id']}.f32"
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
        raise ValueError("fused diagnostic artifact manifest does not cover expected frame IDs")
    return frames


def _expected_contract_reference(sidecar_sha: str, frame_count: int) -> Dict[str, Any]:
    return {
        "schema": SIDECAR_SCHEMA, "sidecar": SIDECAR_FILENAME, "sidecar_sha256": sidecar_sha,
        "frame_count": frame_count,
        "input_file_pattern": "model_input_<frame-id>.f32",
        "output_file_pattern": "raw_<frame-id>.f32",
        "authority": "single-high-input-output-boundary",
    }


def authenticate_manifest_files(
        artifact_dir: Path,
        manifest: Mapping[str, Any],
        expected_frame_ids: Optional[Iterable[int]] = None,
        expected_composite_runtime: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Revalidate the sidecar and rehash every declared finite model-boundary tensor."""

    frames = validate_manifest(manifest, expected_frame_ids, expected_composite_runtime)
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
    expected_reference = _expected_contract_reference(
        manifest["sidecar_sha256"], manifest["frame_count"])
    if contract.get("prod_zipdepth_convex2x_diagnostics") != expected_reference:
        raise ValueError("fused diagnostic harness contract has an invalid sidecar reference")
    shape = manifest["tensor_shapes"]["output"]
    contract_raw = _validate_contract_raw_provenance(
        contract.get("raw_model_provenance"), shape["width"], shape["height"],
        sidecar["embedded_dav2_provenance"])
    if (manifest["tensor_shapes"] != _tensor_shapes(sidecar) or
            manifest["authority"] != sidecar["authority"] or
            manifest["composite_runtime_provenance"] !=
            sidecar["composite_runtime_provenance"] or
            manifest["embedded_dav2_provenance"] != sidecar["embedded_dav2_provenance"] or
            contract_raw["onnx_sha256"] !=
            manifest["composite_runtime_provenance"]["embedded_dav2_onnx_sha256"]):
        raise ValueError("fused diagnostic manifest disagrees with its harness sidecar")

    declared = {
        kind: {row[kind]["file"] for row in frames}
        for kind in _ACTIVE_FILES
    }
    stale_split = sorted({
        path.name
        for prefix in ("zip_model_input_", "refined_")
        for path in artifact_dir.glob(prefix + "*.f32")
    })
    if stale_split:
        raise ValueError(
            "fused diagnostic schema-2 artifact directory contains stale split tensors: "
            + ", ".join(stale_split))
    for kind, (prefix, pattern) in _ACTIVE_FILES.items():
        if _actual_tensor_files(artifact_dir, prefix, pattern) != declared[kind]:
            raise ValueError(f"fused diagnostic {kind} files disagree with their run manifest")
    for row in frames:
        for kind in _ACTIVE_FILES:
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
    """Build and immediately self-verify one clip's manifest."""

    ids = _canonical_frame_ids(frame_ids)
    contract_path = artifact_dir / "contract.json"
    contract = _json_object(contract_path, "fused diagnostic harness contract")
    if contract.get("schema") != HARNESS_CONTRACT_SCHEMA:
        raise ValueError(
            f"fused diagnostic harness schema must be {HARNESS_CONTRACT_SCHEMA}, got "
            f"{contract.get('schema')!r}")
    sidecar_path = artifact_dir / SIDECAR_FILENAME
    sidecar = _validated_sidecar(sidecar_path, len(ids), expected_composite_runtime)
    sidecar_sha = file_sha256(sidecar_path)
    if contract.get("prod_zipdepth_convex2x_diagnostics") != _expected_contract_reference(
            sidecar_sha, len(ids)):
        raise ValueError("fused diagnostic harness contract has an invalid sidecar reference")
    shapes = _tensor_shapes(sidecar)
    _validate_contract_raw_provenance(
        contract.get("raw_model_provenance"), shapes["output"]["width"],
        shapes["output"]["height"], sidecar["embedded_dav2_provenance"])
    frames = []
    for frame_id in ids:
        identity = f"{frame_id:05d}"
        frames.append({
            "frame_id": identity,
            **{
                kind: _inspect_float_file(
                    artifact_dir / f"{prefix}{identity}.f32", _expected_bytes(shapes[kind]))
                for kind, (prefix, _) in _ACTIVE_FILES.items()
            },
        })
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "binding": BINDING,
        "evaluator_schema": EVALUATOR_SCHEMA,
        "harness_contract_schema": HARNESS_CONTRACT_SCHEMA,
        "contract_json_sha256": file_sha256(contract_path),
        "sidecar": SIDECAR_FILENAME,
        "sidecar_sha256": sidecar_sha,
        "authority": sidecar["authority"],
        "frame_count": len(ids),
        "tensor_shapes": shapes,
        "composite_runtime_provenance": sidecar["composite_runtime_provenance"],
        "embedded_dav2_provenance": sidecar["embedded_dav2_provenance"],
        "frames": frames,
    }
    authenticate_manifest_files(artifact_dir, manifest, ids, expected_composite_runtime)
    return manifest


__all__ = [
    "BINDING", "EVALUATOR_SCHEMA", "HARNESS_CONTRACT_SCHEMA", "MANIFEST_SCHEMA",
    "RESULTS_META_KEY", "SIDECAR_FILENAME", "SIDECAR_SCHEMA", "authenticate_manifest_files",
    "build_manifest", "file_sha256", "validate_manifest",
]
