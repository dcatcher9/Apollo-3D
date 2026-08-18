"""Authenticated raw-depth artifacts produced by a current whole-clip evaluator run.

The temporal replay must not infer provenance from whatever ``raw_*.f32`` files happen to be
present beside a results file.  Schema-37 ``run_eval.py`` records this manifest after the
independent schema-22 production evaluation harness finishes (not Dump 3D schema 37 or DVC2
contract), rechecks it after scoring, and stores it in
``results.json``.  Consumers validate the same shared contract before reading a tensor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from .depth_coordinate_v2_contract import MODEL_CALIBRATIONS
except ImportError:  # Direct execution from tools/sbsbench.
    from depth_coordinate_v2_contract import MODEL_CALIBRATIONS  # type: ignore


EVALUATOR_SCHEMA = 37
HARNESS_CONTRACT_SCHEMA = 22
MANIFEST_SCHEMA = 1
RESULTS_META_KEY = "whole_clip_raw_artifacts"
BINDING = "schema-37-results-to-schema-22-harness-raw-f32-v1"
RAW_SHAPE_SCHEMA = 1
RAW_STAGE = "raw model output before transform/normalization/EMA/curvature"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RAW_FILE = re.compile(r"raw_(\d+)\.f32")
_MANIFEST_KEYS = {
    "schema", "binding", "evaluator_schema", "harness_contract_schema",
    "contract_json_sha256", "raw_shape", "raw_shape_json_sha256",
    "producer_model_identity", "calibration_status", "calibration_id",
    "abstention_reason", "frames",
}
_FRAME_KEYS = {"frame_id", "file", "sha256"}
_PRODUCER_IDENTITY_KEYS = {
    "schema", "model", "depth_model_url", "onnx_sha256", "preprocess_profile",
    "preprocess_source_closure_sha256", "raw_width", "raw_height",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash whole-clip raw artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read whole-clip raw contract input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"whole-clip raw contract input {path} must be a JSON object")
    return value


def _canonical_frame_ids(frame_ids: Iterable[int]) -> List[int]:
    values = list(frame_ids)
    if (not values or any(type(value) is not int or value < 0 for value in values) or
            values != sorted(set(values))):
        raise ValueError("whole-clip raw frame IDs must be non-empty, unique, and increasing")
    return values


def validate_manifest(
        manifest: Any, expected_frame_ids: Optional[Iterable[int]] = None) -> List[Dict[str, str]]:
    """Validate manifest semantics and return its ordered frame records."""

    if (not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS or
            manifest.get("schema") != MANIFEST_SCHEMA or
            manifest.get("binding") != BINDING or
            manifest.get("evaluator_schema") != EVALUATOR_SCHEMA or
            manifest.get("harness_contract_schema") != HARNESS_CONTRACT_SCHEMA or
            any(not isinstance(manifest.get(key), str) or
                _SHA256.fullmatch(manifest[key]) is None
                for key in ("contract_json_sha256", "raw_shape_json_sha256"))):
        raise ValueError("whole-clip raw artifact manifest has missing or unknown semantics")
    raw_shape = manifest.get("raw_shape")
    if (not isinstance(raw_shape, dict) or set(raw_shape) != {"height", "width"} or
            type(raw_shape.get("height")) is not int or raw_shape["height"] < 1 or
            type(raw_shape.get("width")) is not int or raw_shape["width"] < 1):
        raise ValueError("whole-clip raw artifact manifest has an invalid raw shape")
    producer_identity = manifest.get("producer_model_identity")
    if (not isinstance(producer_identity, dict) or
            set(producer_identity) != _PRODUCER_IDENTITY_KEYS or
            producer_identity.get("schema") != 1 or
            not isinstance(producer_identity.get("model"), str) or
            not producer_identity["model"] or
            not isinstance(producer_identity.get("depth_model_url"), str) or
            not isinstance(producer_identity.get("preprocess_profile"), str) or
            any(not isinstance(producer_identity.get(key), str) or
                _SHA256.fullmatch(producer_identity[key]) is None
                for key in ("onnx_sha256", "preprocess_source_closure_sha256")) or
            producer_identity.get("raw_width") != raw_shape["width"] or
            producer_identity.get("raw_height") != raw_shape["height"]):
        raise ValueError("whole-clip raw artifact manifest has invalid producer identity")
    status = manifest.get("calibration_status")
    calibration_id = manifest.get("calibration_id")
    reason = manifest.get("abstention_reason")
    if ((status == "calibrated" and
         (not isinstance(calibration_id, str) or not calibration_id or reason is not None)) or
            (status == "abstain-unsupported-shape" and
             (not isinstance(calibration_id, str) or not calibration_id or
              not isinstance(reason, str) or not reason)) or
            (status == "abstain-unsupported-model-contract" and
             (calibration_id is not None or not isinstance(reason, str) or not reason)) or
            status not in {"calibrated", "abstain-unsupported-shape",
                           "abstain-unsupported-model-contract"}):
        raise ValueError("whole-clip raw artifact manifest has invalid calibration status")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("whole-clip raw artifact manifest requires ordered frame records")
    observed_ids: List[int] = []
    for index, row in enumerate(frames):
        if (not isinstance(row, dict) or set(row) != _FRAME_KEYS or
                not isinstance(row.get("frame_id"), str) or
                not re.fullmatch(r"\d{5,}", row["frame_id"]) or
                row.get("file") != f"raw_{row['frame_id']}.f32" or
                not isinstance(row.get("sha256"), str) or
                _SHA256.fullmatch(row["sha256"]) is None):
            raise ValueError(
                f"whole-clip raw artifact manifest frame {index} is invalid")
        observed_ids.append(int(row["frame_id"]))
    if observed_ids != sorted(set(observed_ids)):
        raise ValueError("whole-clip raw artifact manifest frame IDs are not unique/increasing")
    if expected_frame_ids is not None:
        expected = _canonical_frame_ids(expected_frame_ids)
        if observed_ids != expected:
            raise ValueError(
                "whole-clip raw artifact manifest does not cover the expected frame IDs")
    return frames


def authenticate_manifest_files(
        artifact_dir: Path,
        manifest: Mapping[str, Any],
        expected_frame_ids: Optional[Iterable[int]] = None) -> List[Dict[str, str]]:
    """Rehash the harness contract, raw shape, and every declared raw tensor."""

    frames = validate_manifest(manifest, expected_frame_ids)
    contract_path = artifact_dir / "contract.json"
    contract = _json_object(contract_path)
    if contract.get("schema") != HARNESS_CONTRACT_SCHEMA:
        raise ValueError(
            f"whole-clip source harness schema must be {HARNESS_CONTRACT_SCHEMA}, got "
            f"{contract.get('schema')!r}")
    if contract.get("raw_model_provenance") != manifest["producer_model_identity"]:
        raise ValueError("whole-clip harness producer identity disagrees with its run manifest")
    if file_sha256(contract_path) != manifest["contract_json_sha256"]:
        raise ValueError("whole-clip harness contract changed after raw capture")
    shape_path = artifact_dir / "raw_shape.json"
    if file_sha256(shape_path) != manifest["raw_shape_json_sha256"]:
        raise ValueError("whole-clip raw shape changed after raw capture")
    shape = _json_object(shape_path)
    expected_shape_keys = {"schema", "width", "height", "dtype", "layout", "stage"}
    if (set(shape) != expected_shape_keys or shape.get("schema") != RAW_SHAPE_SCHEMA or
            shape.get("width") != manifest["raw_shape"]["width"] or
            shape.get("height") != manifest["raw_shape"]["height"] or
            shape.get("dtype") != "float32-le" or shape.get("layout") != "row-major" or
            shape.get("stage") != RAW_STAGE):
        raise ValueError("whole-clip raw shape semantics disagree with their run manifest")

    declared_files = {row["file"] for row in frames}
    actual_files = set()
    for path in artifact_dir.glob("raw_*.f32"):
        match = _RAW_FILE.fullmatch(path.name)
        if match is None or path.name != f"raw_{int(match.group(1)):05d}.f32":
            raise ValueError(f"whole-clip raw artifact has non-canonical identity {path.name!r}")
        actual_files.add(path.name)
    if actual_files != declared_files:
        raise ValueError("whole-clip raw artifact files disagree with their run manifest")
    for row in frames:
        raw_path = artifact_dir / row["file"]
        try:
            raw_size = raw_path.stat().st_size
        except OSError as exc:
            raise ValueError(f"cannot inspect whole-clip raw artifact {raw_path}: {exc}") from exc
        expected_size = manifest["raw_shape"]["width"] * manifest["raw_shape"]["height"] * 4
        if raw_size != expected_size:
            raise ValueError(
                f"whole-clip raw artifact {row['file']} has {raw_size} bytes; "
                f"expected {expected_size}")
        if file_sha256(raw_path) != row["sha256"]:
            raise ValueError(
                f"whole-clip raw artifact {row['file']} changed after evaluator capture")
    return frames


def build_manifest(
        artifact_dir: Path,
        frame_ids: Iterable[int],
        model_identity: Mapping[str, Any]) -> Dict[str, Any]:
    """Build and immediately self-verify a producer-side raw artifact manifest."""

    ids = _canonical_frame_ids(frame_ids)
    contract_path = artifact_dir / "contract.json"
    contract = _json_object(contract_path)
    if contract.get("schema") != HARNESS_CONTRACT_SCHEMA:
        raise ValueError(
            f"whole-clip source harness schema must be {HARNESS_CONTRACT_SCHEMA}, got "
            f"{contract.get('schema')!r}")
    shape_path = artifact_dir / "raw_shape.json"
    raw_shape = _json_object(shape_path)
    width = raw_shape.get("width")
    height = raw_shape.get("height")
    expected_shape_keys = {"schema", "width", "height", "dtype", "layout", "stage"}
    if (set(raw_shape) != expected_shape_keys or
            raw_shape.get("schema") != RAW_SHAPE_SCHEMA or
            type(width) is not int or width < 1 or type(height) is not int or height < 1 or
            raw_shape.get("dtype") != "float32-le" or
            raw_shape.get("layout") != "row-major" or raw_shape.get("stage") != RAW_STAGE):
        raise ValueError("whole-clip raw_shape.json has missing/unknown tensor semantics")
    producer_identity = contract.get("raw_model_provenance")
    expected_url = model_identity.get("depth_model_url")
    expected_profile = model_identity.get("preprocess_profile")
    if (not isinstance(producer_identity, dict) or
            set(producer_identity) != _PRODUCER_IDENTITY_KEYS or
            producer_identity.get("schema") != 1 or
            producer_identity.get("model") != model_identity.get("model") or
            producer_identity.get("onnx_sha256") != model_identity.get("onnx_sha256") or
            producer_identity.get("preprocess_source_closure_sha256") !=
            model_identity.get("preprocess_source_closure_sha256") or
            producer_identity.get("raw_width") != width or
            producer_identity.get("raw_height") != height or
            (isinstance(expected_url, str) and expected_url and
             producer_identity.get("depth_model_url") != expected_url) or
            (expected_profile is not None and
             producer_identity.get("preprocess_profile") != expected_profile)):
        raise ValueError(
            "whole-clip harness producer identity disagrees with evaluator runtime identity")
    calibrations = [
        value for value in MODEL_CALIBRATIONS
        if (value.depth_model == producer_identity.get("model") and
            value.depth_model_url == producer_identity.get("depth_model_url") and
            value.onnx_sha256 == producer_identity.get("onnx_sha256") and
            value.preprocess.profile == producer_identity.get("preprocess_profile") and
            value.preprocess.source_closure_sha256 ==
            producer_identity.get("preprocess_source_closure_sha256"))
    ]
    calibration = calibrations[0] if len(calibrations) == 1 else None
    expected_calibration_id = model_identity.get("calibration_id")
    if (expected_calibration_id is not None and
            (calibration is None or calibration.calibration_id != expected_calibration_id)):
        raise ValueError(
            "whole-clip harness producer identity disagrees with evaluator calibration")
    shape_supported = (
        calibration is not None and (width, height) in calibration.calibrated_input_shapes)
    if calibration is None:
        calibration_status = "abstain-unsupported-model-contract"
        abstention_reason = "model/URL/ONNX/preprocess identity has no exact calibration"
    elif not shape_supported:
        calibration_status = "abstain-unsupported-shape"
        abstention_reason = f"raw model shape {width}x{height} is not calibrated"
    else:
        calibration_status = "calibrated"
        abstention_reason = None
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "binding": BINDING,
        "evaluator_schema": EVALUATOR_SCHEMA,
        "harness_contract_schema": HARNESS_CONTRACT_SCHEMA,
        "contract_json_sha256": file_sha256(contract_path),
        "raw_shape": {"height": height, "width": width},
        "raw_shape_json_sha256": file_sha256(shape_path),
        "producer_model_identity": producer_identity,
        "calibration_status": calibration_status,
        "calibration_id": calibration.calibration_id if calibration is not None else None,
        "abstention_reason": abstention_reason,
        "frames": [
            {
                "frame_id": f"{frame_id:05d}",
                "file": f"raw_{frame_id:05d}.f32",
                "sha256": file_sha256(artifact_dir / f"raw_{frame_id:05d}.f32"),
            }
            for frame_id in ids
        ],
    }
    authenticate_manifest_files(artifact_dir, manifest, ids)
    return manifest


__all__ = [
    "BINDING",
    "EVALUATOR_SCHEMA",
    "HARNESS_CONTRACT_SCHEMA",
    "MANIFEST_SCHEMA",
    "RAW_SHAPE_SCHEMA",
    "RAW_STAGE",
    "RESULTS_META_KEY",
    "authenticate_manifest_files",
    "build_manifest",
    "file_sha256",
    "validate_manifest",
]
