"""Fail-closed provenance for raw depth consumed by mapping-v2 replay.

A model name or download URL identifies a configuration, not the ONNX bytes that produced a
captured tensor.  Only a capture-time manifest record that binds both the ONNX digest and the raw
artifact digest can make the model provenance authoritative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Optional

try:
    from . import depth_coordinate_v2_contract as coordinate_contract
except ImportError:  # Direct execution from tools/sbsbench.
    import depth_coordinate_v2_contract as coordinate_contract  # type: ignore


PROVENANCE_SCHEMA = coordinate_contract.CAPTURE_PROVENANCE_SCHEMA
PROVENANCE_KEY = coordinate_contract.CAPTURE_PROVENANCE_KEY
BINDING = coordinate_contract.CAPTURE_PROVENANCE_BINDING
PROVENANCE_KEYS = {
    "schema", "binding", "depth_model", "depth_model_url", "onnx_sha256",
    "preprocess_profile", "preprocess_source_closure_sha256",
    "raw_depth_sha256", "model_input_sha256",
    "model_input_shape_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash provenance artifact {path}: {exc}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class RawModelProvenance:
    status: str
    source: str
    attestation_schema: Optional[int]
    binding: Optional[str]
    declared_model: str
    configured_model: Optional[str]
    declared_url: Optional[str]
    onnx_sha256: Optional[str]
    raw_depth_sha256: str
    model_input_sha256: Optional[str]
    model_input_shape_sha256: Optional[str]
    preprocess_profile: Optional[str]
    preprocess_source_closure_sha256: Optional[str]
    calibration_id: Optional[str]
    calibrated_raw_coordinate_scale: Optional[float]
    model_input_width: Optional[int]
    model_input_height: Optional[int]
    reason: Optional[str]

    @property
    def authoritative(self) -> bool:
        return self.status == "authoritative"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _optional_string(
        value: Any, name: str, *, empty_is_none: bool = False) -> Optional[str]:
    if value is None or (empty_is_none and value == ""):
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"dump manifest {name} must be a non-empty string or null")
    return value


def _json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"dump manifest {name} has an invalid SHA-256")
    return value


def _close_float_vector(actual: Any, expected: tuple[float, float, float]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for value, reference in zip(actual, expected):
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value)) or
                not math.isclose(float(value), reference, rel_tol=0.0, abs_tol=5.0e-7)):
            return False
    return True


def _validate_model_input_contract(
        dump: Path,
        calibration: coordinate_contract.ModelCalibration) -> tuple[int, int, str, str]:
    """Authenticate the exact input artifacts and validate their declared tensor semantics."""

    shape_path = dump / "model_input_shape.json"
    input_path = dump / "model_input.f32"
    shape = _json_object(shape_path)
    expected = calibration.preprocess
    try:
        width = shape["width"]
        height = shape["height"]
    except KeyError as exc:
        raise ValueError("model_input_shape.json lacks width/height") from exc
    if (not isinstance(width, int) or isinstance(width, bool) or width <= 0 or
            not isinstance(height, int) or isinstance(height, bool) or height <= 0 or
            width > expected.maximum_dimension or height > expected.maximum_dimension or
            width % expected.patch_multiple or height % expected.patch_multiple):
        raise ValueError("model_input_shape.json violates the calibrated spatial contract")
    if (width, height) not in calibration.calibrated_input_shapes:
        raise ValueError(
            "model_input_shape.json is not an exact shape covered by the model calibration")
    if (shape.get("schema") != expected.model_input_schema or
            shape.get("dtype") != expected.dtype or
            shape.get("layout") != expected.layout or
            shape.get("channels") != list(expected.channels) or
            shape.get("stage") != expected.stage or
            not _close_float_vector(shape.get("imagenet_mean"), expected.imagenet_mean) or
            not _close_float_vector(shape.get("imagenet_std"), expected.imagenet_std)):
        raise ValueError("model_input_shape.json disagrees with the calibrated preprocess contract")
    try:
        input_size = input_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect provenance artifact {input_path}: {exc}") from exc
    if input_size != width * height * len(expected.channels) * 4:
        raise ValueError("model_input.f32 byte size disagrees with model_input_shape.json")

    raw_shape = _json_object(dump / "raw_shape.json")
    if (raw_shape.get("width") != width or raw_shape.get("height") != height or
            raw_shape.get("dtype") != "float32-le" or
            raw_shape.get("layout") != "row-major"):
        raise ValueError("raw_shape.json disagrees with the calibrated model-input geometry")
    try:
        raw_size = (dump / "raw_depth.f32").stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect provenance artifact {dump / 'raw_depth.f32'}: {exc}") from exc
    if raw_size != width * height * 4:
        raise ValueError("raw_depth.f32 byte size disagrees with calibrated model-input geometry")
    return width, height, file_sha256(input_path), file_sha256(shape_path)


def inspect_dump(dump: Path) -> RawModelProvenance:
    """Inspect capture-time provenance without inferring identity from names or URLs."""

    manifest_path = dump / "dump_manifest.json"
    raw_path = dump / "raw_depth.f32"
    manifest = _json_object(manifest_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("dump_manifest.json lacks its authoritative config object")

    live_config = config.get("live_effective")
    shared_config = config.get("shared_configured")
    if config.get("schema") != 3:
        raise ValueError("dump config is not the current schema 3")
    if not isinstance(live_config, dict):
        raise ValueError("config.live_effective must be an object")
    if not isinstance(shared_config, dict):
        raise ValueError("config.shared_configured must be an object")
    if config.get("offline_analysis_configured") is not None:
        raise ValueError("config schema 3 must not contain retired offline model selection")

    manifest_model = _optional_string(manifest.get("depth_model"), "depth_model")
    config_model = _optional_string(
        live_config.get("depth_model"), "config.live_effective.depth_model")
    if manifest_model and config_model and manifest_model != config_model:
        raise ValueError("dump manifest disagrees about the effective depth model")
    declared_model = manifest_model or config_model
    if declared_model is None:
        raise ValueError("dump manifest lacks an effective depth model name")
    declared_url = _optional_string(
        live_config.get("depth_model_url"),
        "config.live_effective.depth_model_url", empty_is_none=True)
    raw_digest = file_sha256(raw_path)

    proof = manifest.get(PROVENANCE_KEY)
    if proof is None:
        raise ValueError(f"dump manifest lacks required {PROVENANCE_KEY}")
    if not isinstance(proof, dict) or set(proof) != PROVENANCE_KEYS:
        raise ValueError(f"dump manifest {PROVENANCE_KEY} has missing or unknown fields")
    if proof.get("schema") != PROVENANCE_SCHEMA or proof.get("binding") != BINDING:
        raise ValueError(f"dump manifest {PROVENANCE_KEY} has unknown semantics")
    proof_model = _optional_string(proof.get("depth_model"), f"{PROVENANCE_KEY}.depth_model")
    proof_url = _optional_string(
        proof.get("depth_model_url"), f"{PROVENANCE_KEY}.depth_model_url",
        empty_is_none=True)
    if proof_model != declared_model or proof_url != declared_url:
        raise ValueError(f"dump manifest {PROVENANCE_KEY} disagrees with the captured config")
    onnx_digest = _sha256(proof.get("onnx_sha256"), f"{PROVENANCE_KEY}.onnx_sha256")
    declared_raw_digest = _sha256(
        proof.get("raw_depth_sha256"), f"{PROVENANCE_KEY}.raw_depth_sha256")
    declared_input_digest = _sha256(
        proof.get("model_input_sha256"), f"{PROVENANCE_KEY}.model_input_sha256")
    declared_shape_digest = _sha256(
        proof.get("model_input_shape_sha256"),
        f"{PROVENANCE_KEY}.model_input_shape_sha256")
    if declared_raw_digest != raw_digest:
        raise ValueError("raw_depth.f32 SHA-256 does not match its capture-time model binding")
    calibration = coordinate_contract.find_model_calibration(
        declared_model, declared_url or "", onnx_digest)
    if calibration is None:
        raise ValueError(
            "capture-time proof does not match a calibrated model name, URL, and ONNX SHA-256")
    preprocess_profile = _optional_string(
        proof.get("preprocess_profile"), f"{PROVENANCE_KEY}.preprocess_profile")
    if preprocess_profile != calibration.preprocess.profile:
        raise ValueError("capture-time proof has an unknown calibrated preprocess profile")
    preprocess_source_closure_sha256 = _sha256(
        proof.get("preprocess_source_closure_sha256"),
        f"{PROVENANCE_KEY}.preprocess_source_closure_sha256")
    if preprocess_source_closure_sha256 != calibration.preprocess.source_closure_sha256:
        raise ValueError(
            "capture-time proof has an unknown calibrated preprocess source closure")
    width, height, input_digest, shape_digest = _validate_model_input_contract(
        dump, calibration)
    if declared_input_digest != input_digest:
        raise ValueError("model_input.f32 SHA-256 does not match its capture-time model binding")
    if declared_shape_digest != shape_digest:
        raise ValueError(
            "model_input_shape.json SHA-256 does not match its capture-time model binding")
    return RawModelProvenance(
        status="authoritative",
        source="dump_manifest.json",
        attestation_schema=PROVENANCE_SCHEMA,
        binding=BINDING,
        declared_model=declared_model,
        configured_model=None,
        declared_url=declared_url,
        onnx_sha256=onnx_digest,
        raw_depth_sha256=raw_digest,
        model_input_sha256=input_digest,
        model_input_shape_sha256=shape_digest,
        preprocess_profile=preprocess_profile,
        preprocess_source_closure_sha256=preprocess_source_closure_sha256,
        calibration_id=calibration.calibration_id,
        calibrated_raw_coordinate_scale=calibration.raw_coordinate_scale,
        model_input_width=width,
        model_input_height=height,
        reason=None,
    )
__all__ = [
    "BINDING",
    "PROVENANCE_KEY",
    "PROVENANCE_KEYS",
    "PROVENANCE_SCHEMA",
    "RawModelProvenance",
    "file_sha256",
    "inspect_dump",
]
