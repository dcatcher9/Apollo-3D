#!/usr/bin/env python3
"""Read-only Python view of the generated depth-coordinate-v2 calibration contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Optional


CONTRACT_PATH = (
    Path(__file__).resolve().parent / "contracts" / "depth-coordinate-v2-v1.json"
)
CALIBRATED_DEFAULT_NAMES = (
    "collapse_abs_epsilon",
    "far_tau",
    "near_log_tau",
    "gain_per_pop",
    "reference_pop_strength",
    "direct_container_limit",
    "max_horizontal_slope",
    "max_vertical_shear",
    "vertical_majorant_share",
    "convergence_curve_default",
    "stage_valley_ratio_max",
)


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]
MODEL_CALIBRATION_KEYS = {
    "calibration_id", "depth_model", "depth_model_url", "onnx_sha256",
    "raw_coordinate_scale", "calibrated_input_shapes", "preprocess",
}
PREPROCESS_KEYS = {
    "profile", "source_closure_schema", "source_file", "source_entrypoint",
    "source_target", "source_compile_flags", "source_macro_count",
    "source_closure_sha256", "model_input_schema", "dtype", "layout", "channels",
    "patch_multiple", "maximum_dimension", "imagenet_mean", "imagenet_std", "stage",
}
SHADER_IMPLEMENTATION_KEYS = {
    "source_closure_schema", "source_compile_flags", "source_macro_count", "source_specs",
    "source_closure_sha256",
}
SHADER_SOURCE_SPECS = (
    ("depth_minmax_cs.hlsl", "main", "cs_5_0"),
    ("depth_hist_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_moments_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_frame_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_state_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_map_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_ownership_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_vertical_limit_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_limit_cs.hlsl", "main", "cs_5_0"),
)


@dataclass(frozen=True)
class CalibratedDefaults:
    collapse_abs_epsilon: float
    far_tau: float
    near_log_tau: float
    gain_per_pop: float
    reference_pop_strength: float
    direct_container_limit: float
    max_horizontal_slope: float
    max_vertical_shear: float
    vertical_majorant_share: float
    convergence_curve_default: float
    stage_valley_ratio_max: float


@dataclass(frozen=True)
class ModelInputContract:
    profile: str
    source_closure_schema: int
    source_file: str
    source_entrypoint: str
    source_target: str
    source_compile_flags: int
    source_macro_count: int
    source_closure_sha256: str
    model_input_schema: int
    dtype: str
    layout: str
    channels: tuple[str, str, str]
    patch_multiple: int
    maximum_dimension: int
    imagenet_mean: tuple[float, float, float]
    imagenet_std: tuple[float, float, float]
    stage: str


@dataclass(frozen=True)
class ModelCalibration:
    calibration_id: str
    depth_model: str
    depth_model_url: str
    onnx_sha256: str
    raw_coordinate_scale: float
    calibrated_input_shapes: tuple[tuple[int, int], ...]
    preprocess: ModelInputContract


@dataclass(frozen=True)
class ShaderImplementation:
    source_closure_schema: int
    source_compile_flags: int
    source_macro_count: int
    source_specs: tuple[tuple[str, str, str], ...]
    source_closure_sha256: str


def _shader_implementation(contract: dict[str, Any]) -> ShaderImplementation:
    implementation = contract.get("shader_implementation")
    if (not isinstance(implementation, dict) or
            set(implementation) != SHADER_IMPLEMENTATION_KEYS):
        raise ValueError("depth-coordinate-v2 shader_implementation schema mismatch")
    if (implementation.get("source_closure_schema") != 2 or
            implementation.get("source_compile_flags") != 0x00008800 or
            implementation.get("source_macro_count") != 0):
        raise ValueError("depth-coordinate-v2 shader implementation policy is unsupported")
    entries = implementation.get("source_specs")
    if not isinstance(entries, list):
        raise ValueError("depth-coordinate-v2 shader source specs must be an ordered list")
    parsed: list[tuple[str, str, str]] = []
    for entry in entries:
        if (not isinstance(entry, dict) or
                set(entry) != {"source_file", "source_entrypoint", "source_target"}):
            raise ValueError("depth-coordinate-v2 shader source spec schema mismatch")
        values = tuple(entry[name] for name in (
            "source_file", "source_entrypoint", "source_target"))
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("depth-coordinate-v2 shader source spec values must be strings")
        parsed.append(values)  # type: ignore[arg-type]
    if tuple(parsed) != SHADER_SOURCE_SPECS:
        raise ValueError("depth-coordinate-v2 shader source set is unsupported")
    digest = implementation.get("source_closure_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("depth-coordinate-v2 shader source digest must be lowercase SHA-256")
    return ShaderImplementation(
        source_closure_schema=implementation["source_closure_schema"],
        source_compile_flags=implementation["source_compile_flags"],
        source_macro_count=implementation["source_macro_count"],
        source_specs=tuple(parsed),
        source_closure_sha256=digest,
    )


def _model_calibrations(contract: dict[str, Any]) -> tuple[ModelCalibration, ...]:
    entries = contract.get("model_calibrations")
    if not isinstance(entries, list) or not entries:
        raise ValueError("depth-coordinate-v2 model_calibrations must be non-empty")
    result: list[ModelCalibration] = []
    calibration_ids: set[str] = set()
    model_names: set[str] = set()
    model_hashes: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"depth-coordinate-v2 model_calibrations[{index}]"
        if not isinstance(entry, dict) or set(entry) != MODEL_CALIBRATION_KEYS:
            raise ValueError(f"{prefix} schema mismatch")
        preprocess = entry.get("preprocess")
        if not isinstance(preprocess, dict) or set(preprocess) != PREPROCESS_KEYS:
            raise ValueError(f"{prefix}.preprocess schema mismatch")
        string_fields = (
            "calibration_id", "depth_model", "depth_model_url", "onnx_sha256")
        if any(not isinstance(entry.get(name), str) or not entry[name]
               for name in string_fields):
            raise ValueError(f"{prefix} identity strings must be non-empty")
        if (len(entry["onnx_sha256"]) != 64 or
                any(character not in "0123456789abcdef" for character in entry["onnx_sha256"])):
            raise ValueError(f"{prefix}.onnx_sha256 must be lowercase SHA-256")
        scale = entry.get("raw_coordinate_scale")
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool) or
                not math.isfinite(float(scale)) or float(scale) <= 0.0):
            raise ValueError(f"{prefix}.raw_coordinate_scale must be finite and positive")
        for name in ("profile", "dtype", "layout", "stage"):
            if not isinstance(preprocess.get(name), str) or not preprocess[name]:
                raise ValueError(f"{prefix}.preprocess.{name} must be a non-empty string")
        source_closure_sha256 = preprocess.get("source_closure_sha256")
        if (not isinstance(source_closure_sha256, str) or len(source_closure_sha256) != 64 or
                any(character not in "0123456789abcdef"
                    for character in source_closure_sha256)):
            raise ValueError(
                f"{prefix}.preprocess.source_closure_sha256 must be lowercase SHA-256")
        if (preprocess.get("source_closure_schema") != 2 or
                preprocess.get("source_file") != "rgb_to_nchw_cs.hlsl" or
                preprocess.get("source_entrypoint") != "main" or
                preprocess.get("source_target") != "cs_5_0" or
                preprocess.get("source_compile_flags") != 0x00008800 or
                preprocess.get("source_macro_count") != 0):
            raise ValueError(f"{prefix}.preprocess shader source identity is unsupported")
        if preprocess.get("model_input_schema") != 1:
            raise ValueError(f"{prefix}.preprocess.model_input_schema must be one")
        if preprocess.get("dtype") != "float32-le" or preprocess.get("layout") != "NCHW":
            raise ValueError(f"{prefix}.preprocess tensor semantics are unsupported")
        channels = preprocess.get("channels")
        if channels != ["R", "G", "B"]:
            raise ValueError(f"{prefix}.preprocess.channels must be RGB")
        patch = preprocess.get("patch_multiple")
        maximum = preprocess.get("maximum_dimension")
        if (not isinstance(patch, int) or isinstance(patch, bool) or patch <= 0 or
                not isinstance(maximum, int) or isinstance(maximum, bool) or
                maximum < patch or maximum % patch):
            raise ValueError(f"{prefix}.preprocess spatial contract is invalid")
        shape_entries = entry.get("calibrated_input_shapes")
        if not isinstance(shape_entries, list) or not shape_entries:
            raise ValueError(f"{prefix}.calibrated_input_shapes must be non-empty")
        calibrated_shapes: list[tuple[int, int]] = []
        for shape in shape_entries:
            if (not isinstance(shape, dict) or set(shape) != {"width", "height"} or
                    not isinstance(shape.get("width"), int) or
                    isinstance(shape.get("width"), bool) or shape["width"] <= 0 or
                    not isinstance(shape.get("height"), int) or
                    isinstance(shape.get("height"), bool) or shape["height"] <= 0 or
                    shape["width"] > maximum or shape["height"] > maximum or
                    shape["width"] % patch or shape["height"] % patch):
                raise ValueError(f"{prefix}.calibrated_input_shapes is invalid")
            calibrated_shapes.append((shape["width"], shape["height"]))
        if len(set(calibrated_shapes)) != len(calibrated_shapes):
            raise ValueError(f"{prefix}.calibrated_input_shapes contains duplicates")
        vectors: dict[str, tuple[float, float, float]] = {}
        for name in ("imagenet_mean", "imagenet_std"):
            values = preprocess.get(name)
            if (not isinstance(values, list) or len(values) != 3 or
                    any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                        not math.isfinite(float(value)) for value in values) or
                    (name == "imagenet_std" and any(float(value) <= 0.0 for value in values))):
                raise ValueError(f"{prefix}.preprocess.{name} is invalid")
            vectors[name] = tuple(float(value) for value in values)  # type: ignore[assignment]
        if (entry["calibration_id"] in calibration_ids or
                entry["depth_model"] in model_names or
                entry["onnx_sha256"] in model_hashes):
            raise ValueError(
                "depth-coordinate-v2 calibration ids, model names, and hashes must be unique")
        calibration_ids.add(entry["calibration_id"])
        model_names.add(entry["depth_model"])
        model_hashes.add(entry["onnx_sha256"])
        result.append(ModelCalibration(
            calibration_id=entry["calibration_id"],
            depth_model=entry["depth_model"],
            depth_model_url=entry["depth_model_url"],
            onnx_sha256=entry["onnx_sha256"],
            raw_coordinate_scale=float(scale),
            calibrated_input_shapes=tuple(calibrated_shapes),
            preprocess=ModelInputContract(
                profile=preprocess["profile"],
                source_closure_schema=preprocess["source_closure_schema"],
                source_file=preprocess["source_file"],
                source_entrypoint=preprocess["source_entrypoint"],
                source_target=preprocess["source_target"],
                source_compile_flags=preprocess["source_compile_flags"],
                source_macro_count=preprocess["source_macro_count"],
                source_closure_sha256=source_closure_sha256,
                model_input_schema=preprocess["model_input_schema"],
                dtype=preprocess["dtype"],
                layout=preprocess["layout"],
                channels=tuple(channels),
                patch_multiple=patch,
                maximum_dimension=maximum,
                imagenet_mean=vectors["imagenet_mean"],
                imagenet_std=vectors["imagenet_std"],
                stage=preprocess["stage"],
            ),
        ))
    return tuple(result)


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the manifest and fail closed if its calibrated-default surface is malformed."""

    with path.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    provenance = contract.get("capture_provenance") if isinstance(contract, dict) else None
    if (not isinstance(provenance, dict) or
            set(provenance) != {"schema", "manifest_key", "binding"} or
            not isinstance(provenance.get("schema"), int) or
            isinstance(provenance.get("schema"), bool) or provenance["schema"] < 1 or
            any(not isinstance(provenance.get(name), str) or not provenance[name]
                for name in ("manifest_key", "binding"))):
        raise ValueError("depth-coordinate-v2 capture_provenance schema mismatch")
    defaults = contract.get("calibrated_defaults") if isinstance(contract, dict) else None
    if not isinstance(defaults, dict) or set(defaults) != set(CALIBRATED_DEFAULT_NAMES):
        raise ValueError("depth-coordinate-v2 calibrated_defaults schema mismatch")
    for name in CALIBRATED_DEFAULT_NAMES:
        value = defaults[name]
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            raise ValueError(f"depth-coordinate-v2 default {name} must be finite")
        if name == "convergence_curve_default":
            if float(value) != 0.0:
                raise ValueError("convergence_curve_default must be exactly zero")
        elif float(value) <= 0.0:
            raise ValueError(f"depth-coordinate-v2 default {name} must be positive")
    if defaults["max_horizontal_slope"] >= 1.0:
        raise ValueError("max_horizontal_slope must be below one")
    if not 0.0 < float(defaults["stage_valley_ratio_max"]) <= 1.0:
        raise ValueError("stage_valley_ratio_max must be in (0, 1]")
    majorant_share = _float32(defaults["vertical_majorant_share"])
    minorant_share = _float32(_float32(1.0) - majorant_share)
    if majorant_share <= 0.0 or minorant_share <= 0.0:
        raise ValueError(
            "vertical_majorant_share and its complement must remain positive in float32")
    _shader_implementation(contract)
    _model_calibrations(contract)
    return contract


_CONTRACT = load_contract()
CONTRACT_CANONICAL_SHA256 = hashlib.sha256(json.dumps(
    _CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("ascii")).hexdigest()
CALIBRATED_DEFAULTS = CalibratedDefaults(
    **{name: float(_CONTRACT["calibrated_defaults"][name])
       for name in CALIBRATED_DEFAULT_NAMES}
)
CONTRACT_SCHEMA = int(_CONTRACT["schema"])
CAPTURE_PROVENANCE_SCHEMA = int(_CONTRACT["capture_provenance"]["schema"])
CAPTURE_PROVENANCE_KEY = str(_CONTRACT["capture_provenance"]["manifest_key"])
CAPTURE_PROVENANCE_BINDING = str(_CONTRACT["capture_provenance"]["binding"])
SHADER_IMPLEMENTATION = _shader_implementation(_CONTRACT)
MODEL_CALIBRATIONS = _model_calibrations(_CONTRACT)


def find_model_calibration(
        depth_model: str, depth_model_url: str, onnx_sha256: str) -> Optional[ModelCalibration]:
    """Return the one calibration bound to an exact model name, URL, and ONNX digest."""

    matches = [
        calibration for calibration in MODEL_CALIBRATIONS
        if calibration.depth_model == depth_model and
        calibration.depth_model_url == depth_model_url and
        calibration.onnx_sha256 == onnx_sha256
    ]
    if len(matches) > 1:
        raise ValueError("depth-coordinate-v2 model calibration identity is ambiguous")
    return matches[0] if matches else None


__all__ = [
    "CALIBRATED_DEFAULTS",
    "CALIBRATED_DEFAULT_NAMES",
    "CONTRACT_PATH",
    "CONTRACT_CANONICAL_SHA256",
    "CONTRACT_SCHEMA",
    "CAPTURE_PROVENANCE_BINDING",
    "CAPTURE_PROVENANCE_KEY",
    "CAPTURE_PROVENANCE_SCHEMA",
    "CalibratedDefaults",
    "MODEL_CALIBRATIONS",
    "MODEL_CALIBRATION_KEYS",
    "ModelCalibration",
    "ModelInputContract",
    "PREPROCESS_KEYS",
    "SHADER_IMPLEMENTATION",
    "SHADER_IMPLEMENTATION_KEYS",
    "SHADER_SOURCE_SPECS",
    "ShaderImplementation",
    "find_model_calibration",
    "load_contract",
]
