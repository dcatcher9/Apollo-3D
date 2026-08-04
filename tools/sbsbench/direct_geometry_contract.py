"""Canonical contract for authenticated direct-displacement SBS replay.

The renderer consumes only the conditioned parallax field.  Its authenticated horizontal slope
makes the eye map strictly monotone, so a fixed-point inverse has one source and no visibility
choice.  The pre-limiter canonical coordinate is retained solely as diagnostic semantic depth and
as ordering for the separate forward-coverage diagnostic.  Keeping the schema in one module
prevents the producer, replay wrapper, and evaluator from silently assigning it a rendering role.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Dict, Iterable

import numpy as np

try:
    from .depth_coordinate_v2_contract import CALIBRATED_DEFAULTS as V2_DEFAULTS
except ImportError:  # Direct script/module loading from tools/sbsbench.
    from depth_coordinate_v2_contract import CALIBRATED_DEFAULTS as V2_DEFAULTS  # type: ignore


CONTRACT_SCHEMA = 23
MANIFEST_SCHEMA = 4
WARP_INPUT = "external-final-parallax-with-diagnostic-order-v4"
SOURCE_U_LIMIT = V2_DEFAULTS.direct_container_limit
MAX_HORIZONTAL_SLOPE = V2_DEFAULTS.max_horizontal_slope
_SERIALIZED_F32_MAX_HORIZONTAL_SLOPE = float(np.float32(MAX_HORIZONTAL_SLOPE))
_SOURCE_U_LIMIT_TEXT = format(SOURCE_U_LIMIT, ".9g")
_DECODE_EXPRESSION = f"(encoded * 2 - 1) * {_SOURCE_U_LIMIT_TEXT}"

GEOMETRY_DESCRIPTOR = {
    "enabled": True,
    "file_pattern": "parallax_<frame-id>.f32",
    "dtype": "float32-le",
    "stored_range": [0, 1],
    "zero_encoding": 0.5,
    "decode_source_u_one_eye": _DECODE_EXPRESSION,
    "displacement_semantics": "signed-source-u-positive-nearward-not-order",
    "maximum_horizontal_source_u_slope": MAX_HORIZONTAL_SLOPE,
    "renderer_inverse": "contractive-fixed-point-12-iterations-v1",
    "renderer_uses_order": False,
    "order_file_pattern": "order_<frame-id>.f32",
    "order_dtype": "float32-le",
    "order_range": "finite-unbounded",
    "order_high_is_near": True,
    "order_role": "diagnostic-semantic-depth-and-forward-coverage-only-v1",
    "depth_artifact_file_pattern": "depth_<frame-id>.f32",
    "depth_artifact_semantics": "canonical-pre-limiter-order-float32-v1",
    "parallax_artifact_file_pattern": "parallax_<frame-id>.f32",
    "parallax_artifact_semantics": "encoded-conditioned-final-parallax-float32-v1",
    "timing_scope": "quality-only; legacy estimator still executes",
}
GPU_SEQUENCE_TIMING_SCOPE = "native-depth-coordinate-v2; legacy estimator bypassed"

MANIFEST_HEADER = {
    "schema": MANIFEST_SCHEMA,
    "dtype": "float32-le",
    "layout": "row-major",
    "stored_range": [0, 1],
    "zero_encoding": 0.5,
    "decode_source_u_one_eye": _DECODE_EXPRESSION,
    "displacement_semantics": "signed-source-u-positive-nearward-not-order",
    # The native manifest is written from the generated float32 GPU contract. Preserve that exact
    # round trip here instead of requiring a JSON decimal to compare bit-for-bit with its
    # generated float32 value.
    "maximum_horizontal_source_u_slope": _SERIALIZED_F32_MAX_HORIZONTAL_SLOPE,
    "renderer_inverse": "contractive-fixed-point-12-iterations-v1",
    "renderer_uses_order": False,
    "order_dtype": "float32-le",
    "order_range": "finite-unbounded",
    "order_high_is_near": True,
    "order_role": "diagnostic-semantic-depth-and-forward-coverage-only-v1",
    "depth_artifact_semantics": "canonical-pre-limiter-order-float32-v1",
    "parallax_artifact_semantics": "encoded-conditioned-final-parallax-float32-v1",
}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact_float_field(path: str, width: int, height: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise ValueError(f"direct-geometry artifact is missing: {path}")
    expected_bytes = width * height * np.dtype("<f4").itemsize
    try:
        actual_bytes = os.path.getsize(path)
    except OSError as exc:
        raise ValueError(f"cannot stat direct-geometry artifact {path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"direct-geometry artifact size mismatch: {actual_bytes} != {expected_bytes}: {path}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != width * height or not np.isfinite(values).all():
        raise ValueError(f"direct-geometry artifact is not finite float32: {path}")
    return values.reshape(height, width)


def validate_artifacts(
        artifact_dir: str,
        contract: Dict[str, object],
        source_ids: Iterable[int]) -> Dict[str, object]:
    """Authenticate both scored fields and return their paths/shapes.

    The manifest is an attestation from the harness about the bytes it uploaded.  The harness also
    republishes those exact bytes under the run directory; this validator hashes and remeasures the
    copies so later scoring never depends on a mutable external experiment-input directory.
    """

    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(
            f"direct-parallax validation requires harness contract schema {CONTRACT_SCHEMA}")
    if contract.get("warp_input") != WARP_INPUT:
        raise ValueError("direct-parallax contract has missing/unknown warp_input")
    expected_descriptor = dict(GEOMETRY_DESCRIPTOR)
    gpu_sequence = contract.get("depth_coordinate_v2_gpu")
    if isinstance(gpu_sequence, dict) and gpu_sequence.get("enabled") is True:
        expected_descriptor["timing_scope"] = GPU_SEQUENCE_TIMING_SCOPE
    if contract.get("direct_parallax") != expected_descriptor:
        raise ValueError("direct-parallax contract has missing/unknown field semantics")

    reference = contract.get("direct_parallax_manifest")
    if (not isinstance(reference, dict) or
            set(reference) != {"file", "schema", "sha256"} or
            reference.get("file") != "direct_parallax_manifest.json" or
            reference.get("schema") != MANIFEST_SCHEMA or
            not isinstance(reference.get("sha256"), str) or
            re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None):
        raise ValueError("direct-parallax contract has an invalid manifest reference")

    manifest_path = os.path.join(artifact_dir, reference["file"])
    try:
        if not os.path.isfile(manifest_path):
            raise OSError("not a regular file")
        actual_manifest_sha = file_sha256(manifest_path)
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid direct-parallax manifest {manifest_path}: {exc}") from exc
    if actual_manifest_sha != reference["sha256"]:
        raise ValueError(
            "direct-parallax manifest SHA-256 mismatch: "
            f"contract={reference['sha256']} actual={actual_manifest_sha}")
    if (not isinstance(manifest, dict) or
            set(manifest) != set(MANIFEST_HEADER) | {"fields"} or
            any(manifest.get(key) != value for key, value in MANIFEST_HEADER.items())):
        raise ValueError(
            f"direct-parallax manifest has missing/unknown schema-{MANIFEST_SCHEMA} semantics")

    expected_ids = sorted(set(source_ids))
    fields = manifest.get("fields")
    if not isinstance(fields, list) or len(fields) != len(expected_ids):
        raise ValueError("direct-parallax manifest field count does not match source frame set")
    expected_field_keys = {
        "frame_id", "width", "height", "parallax_sha256",
        "maximum_absolute_source_u", "order_sha256", "order_minimum", "order_maximum",
    }
    observed_ids = []
    depth_files: Dict[int, str] = {}
    parallax_files: Dict[int, str] = {}
    shapes: Dict[int, tuple[int, int]] = {}
    for index, field in enumerate(fields):
        if not isinstance(field, dict) or set(field) != expected_field_keys:
            raise ValueError(f"direct-parallax manifest field {index} has invalid keys")
        frame_id_text = field.get("frame_id")
        if not isinstance(frame_id_text, str) or re.fullmatch(r"\d+", frame_id_text) is None:
            raise ValueError(f"direct-parallax manifest field {index} has invalid frame_id")
        frame_id = int(frame_id_text)
        observed_ids.append(frame_id)
        width, height = field.get("width"), field.get("height")
        if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
            raise ValueError(
                f"direct-parallax manifest frame {frame_id} has invalid dimensions")
        for sha_key in ("parallax_sha256", "order_sha256"):
            digest = field.get(sha_key)
            if (not isinstance(digest, str) or
                    re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                raise ValueError(
                    f"direct-parallax manifest frame {frame_id} has invalid {sha_key}")
        declared_maximum = field.get("maximum_absolute_source_u")
        if (isinstance(declared_maximum, bool) or
                not isinstance(declared_maximum, (int, float)) or
                not math.isfinite(declared_maximum) or declared_maximum < 0.0 or
                declared_maximum > SOURCE_U_LIMIT + 1.0e-7):
            raise ValueError(
                f"direct-parallax manifest frame {frame_id} has invalid displacement bound")
        order_minimum, order_maximum = field.get("order_minimum"), field.get("order_maximum")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) for value in (order_minimum, order_maximum)) or
                order_minimum > order_maximum):
            raise ValueError(
                f"direct-parallax manifest frame {frame_id} has invalid order range")

        depth_path = os.path.join(artifact_dir, f"depth_{frame_id_text}.f32")
        parallax_path = os.path.join(artifact_dir, f"parallax_{frame_id_text}.f32")
        order = _load_exact_float_field(depth_path, width, height)
        encoded = _load_exact_float_field(parallax_path, width, height)
        if file_sha256(depth_path) != field["order_sha256"]:
            raise ValueError(
                f"direct-parallax frame {frame_id} canonical-depth SHA-256 mismatch")
        if file_sha256(parallax_path) != field["parallax_sha256"]:
            raise ValueError(
                f"direct-parallax frame {frame_id} parallax SHA-256 mismatch")
        if np.any(encoded < 0.0) or np.any(encoded > 1.0):
            raise ValueError(
                f"direct-parallax frame {frame_id} encoded displacement is outside [0,1]")
        measured_maximum = float(np.max(np.abs(
            (encoded.astype(np.float64) * 2.0 - 1.0) * SOURCE_U_LIMIT)))
        if abs(measured_maximum - float(declared_maximum)) > 1.0e-7:
            raise ValueError(
                f"direct-parallax frame {frame_id} displacement bound does not match bytes")
        decoded = (encoded.astype(np.float64) * 2.0 - 1.0) * SOURCE_U_LIMIT
        measured_slope = (float(np.max(np.abs(np.diff(decoded, axis=1)))) * width
                          if width > 1 else 0.0)
        if measured_slope > MAX_HORIZONTAL_SLOPE + 2.0e-5:
            raise ValueError(
                f"direct-parallax frame {frame_id} exceeds the horizontal slope bound")
        measured_order_minimum = float(np.min(order))
        measured_order_maximum = float(np.max(order))
        if (measured_order_minimum != float(order_minimum) or
                measured_order_maximum != float(order_maximum)):
            raise ValueError(
                f"direct-parallax frame {frame_id} order range does not match bytes")
        depth_files[frame_id] = depth_path
        parallax_files[frame_id] = parallax_path
        shapes[frame_id] = (height, width)

    if observed_ids != expected_ids:
        raise ValueError(
            "direct-parallax manifest/source frame-id mismatch: "
            f"manifest={observed_ids} source={expected_ids}")
    if contract.get("direct_parallax_frames") != len(expected_ids):
        raise ValueError("direct-parallax contract frame count does not match source set")
    return {
        "manifest": manifest,
        "depth_files": depth_files,
        "parallax_files": parallax_files,
        "shapes": shapes,
        "geometry_descriptor": expected_descriptor,
    }


__all__ = [
    "CONTRACT_SCHEMA",
    "GEOMETRY_DESCRIPTOR",
    "GPU_SEQUENCE_TIMING_SCOPE",
    "MANIFEST_HEADER",
    "MANIFEST_SCHEMA",
    "MAX_HORIZONTAL_SLOPE",
    "SOURCE_U_LIMIT",
    "WARP_INPUT",
    "file_sha256",
    "validate_artifacts",
]
