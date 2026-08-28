#!/usr/bin/env python3
"""Reference contract for the frozen ZipDepth-guided production DAV2 convex2x path."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

try:
    from . import depth_coordinate_v2_contract as coordinate_contract
except ImportError:  # Direct script/module loading from tools/sbsbench.
    import depth_coordinate_v2_contract as coordinate_contract  # type: ignore


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "contracts"
    / "prod-zipdepth-convex2x-v2.json"
)


@dataclass(frozen=True)
class Shape:
    width: int
    height: int

    def valid(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass(frozen=True)
class ContentRect:
    left: int
    top: int
    right: int
    bottom: int

    def valid_for(self, shape: Shape) -> bool:
        return (
            shape.valid()
            and 0 <= self.left < self.right <= shape.width
            and 0 <= self.top < self.bottom <= shape.height
        )


_CONTRACT_KEYS = {
    "schema", "status", "purpose", "sources", "export", "operator",
    "engine_io", "high_shapes_wh", "tensorrt", "authority",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"prod ZipDepth convex2x contract has invalid {label}")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"prod ZipDepth convex2x contract has invalid {label}")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"prod ZipDepth convex2x contract has invalid {label}")
    return value


def _hash(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"prod ZipDepth convex2x contract has invalid {label}")
    return value


def _expected_shape_profiles() -> tuple[tuple[Shape, ...], tuple[Shape, ...]]:
    calibration = coordinate_contract.MODEL_CALIBRATIONS[0]
    coarse = tuple(Shape(width, height) for width, height in calibration.calibrated_input_shapes)
    if (len(coarse) != 6 or len(set(coarse)) != 6 or
            any(not shape.valid() for shape in coarse)):
        raise ValueError(
            "depth-coordinate-v2 does not expose the exact six production DAV2 profiles")
    high = tuple(Shape(2 * shape.width, 2 * shape.height) for shape in coarse)
    return coarse, high


def _validate_tensor_entry(
        value: object, *, keys: set[str], name: str, layout: str,
        shape: list[object], label: str) -> None:
    entry = _exact_object(value, keys, label)
    if (entry.get("name") != name or entry.get("dtype") != "float32" or
            entry.get("layout") != layout or entry.get("shape") != shape or
            any(type(component) not in (int, str) for component in entry.get("shape", [])) or
            ("semantics" in keys and
             (not isinstance(entry.get("semantics"), str) or not entry["semantics"]))):
        raise ValueError(f"prod ZipDepth convex2x contract has invalid {label}")


def _validate_contract(value: object) -> dict[str, object]:
    contract = _exact_object(value, _CONTRACT_KEYS, "root layout")
    if (type(contract.get("schema")) is not int or contract["schema"] != 2 or
            contract.get("status") != "single-high-io-migration"):
        raise ValueError("unsupported prod ZipDepth convex2x contract")
    _nonempty_string(contract.get("purpose"), "purpose")

    sources = _exact_object(
        contract.get("sources"), {"dav2", "zipdepth", "fused_onnx"}, "sources")
    dav2 = _exact_object(
        sources.get("dav2"), {"logical_model", "onnx_sha256"}, "DAV2 source")
    _nonempty_string(dav2.get("logical_model"), "DAV2 logical model")
    _hash(dav2.get("onnx_sha256"), _SHA256, "DAV2 ONNX hash")
    zipdepth = _exact_object(
        sources.get("zipdepth"),
        {"variant", "repository_commit", "checkpoint", "checkpoint_sha256", "license"},
        "ZipDepth source")
    if zipdepth.get("variant") != "base" or zipdepth.get("license") != "MIT":
        raise ValueError("prod ZipDepth convex2x contract has invalid ZipDepth source")
    _hash(zipdepth.get("repository_commit"), _GIT_COMMIT, "ZipDepth repository commit")
    _nonempty_string(zipdepth.get("checkpoint"), "ZipDepth checkpoint")
    _hash(zipdepth.get("checkpoint_sha256"), _SHA256, "ZipDepth checkpoint hash")
    fused = _exact_object(
        sources.get("fused_onnx"),
        {"logical_model", "opset", "bytes", "sha256", "metadata_policy",
         "deterministic_reexport"},
        "fused ONNX source")
    _nonempty_string(fused.get("logical_model"), "fused ONNX logical model")
    if (_positive_int(fused.get("opset"), "fused ONNX opset") != 18 or
            fused.get("deterministic_reexport") is not True):
        raise ValueError("prod ZipDepth convex2x contract has invalid fused ONNX source")
    _positive_int(fused.get("bytes"), "fused ONNX byte count")
    _hash(fused.get("sha256"), _SHA256, "fused ONNX hash")
    _nonempty_string(fused.get("metadata_policy"), "fused ONNX metadata policy")

    export = _exact_object(
        contract.get("export"),
        {"opset", "zipdepth_branch_onnx_bytes", "zipdepth_branch_onnx_sha256",
         "release_builder_optimization_level"},
        "export recipe")
    if (_positive_int(export.get("opset"), "export opset") != 18 or
            _positive_int(
                export.get("release_builder_optimization_level"),
                "release builder optimization level") != 5):
        raise ValueError("prod ZipDepth convex2x contract has invalid export recipe")
    _positive_int(export.get("zipdepth_branch_onnx_bytes"), "ZipDepth branch byte count")
    _hash(export.get("zipdepth_branch_onnx_sha256"), _SHA256, "ZipDepth branch hash")

    operator = _exact_object(
        contract.get("operator"),
        {"input_downsample", "scale", "neighborhood", "boundary", "weight_layout",
         "weight_normalization", "temperature", "zipdepth_use_unfold",
         "subpixel_layout", "output_activation", "pixel_gate"},
        "operator")
    downsample = _exact_object(
        operator.get("input_downsample"),
        {"operator", "dtype", "kernel", "stride", "padding", "ceil_mode",
         "count_include_pad", "semantics"},
        "input downsample")
    if (downsample.get("operator") != "AveragePool" or
            downsample.get("dtype") != "float32" or
            downsample.get("kernel") != [2, 2] or
            downsample.get("stride") != [2, 2] or
            downsample.get("padding") != [0, 0, 0, 0] or
            downsample.get("ceil_mode") is not False or
            downsample.get("count_include_pad") is not False or
            not isinstance(downsample.get("semantics"), str) or
            not downsample["semantics"] or
            any(type(item) is not int for key in ("kernel", "stride", "padding")
                for item in downsample[key])):
        raise ValueError("prod ZipDepth convex2x contract has invalid input downsample")
    if (type(operator.get("scale")) is not int or operator["scale"] != 2 or
            operator.get("neighborhood") != [3, 3] or
            any(type(item) is not int for item in operator.get("neighborhood", [])) or
            operator.get("boundary") != "replicate" or
            operator.get("weight_layout") != [9, 4] or
            any(type(item) is not int for item in operator.get("weight_layout", [])) or
            operator.get("weight_normalization") !=
            "softmax-over-nine-neighbors-per-output-subpixel" or
            type(operator.get("temperature")) is not float or
            operator["temperature"] != 1.0 or
            operator.get("zipdepth_use_unfold") is not True or
            operator.get("subpixel_layout") != "pixel-shuffle-row-major-2x2" or
            operator.get("output_activation") != "relu" or
            operator.get("pixel_gate") != "none"):
        raise ValueError("prod ZipDepth convex2x contract has invalid operator")

    engine_io = _exact_object(
        contract.get("engine_io"), {"inputs", "outputs", "internal_tensors"}, "engine IO")
    inputs = engine_io.get("inputs")
    outputs = engine_io.get("outputs")
    internal = engine_io.get("internal_tensors")
    if (not isinstance(inputs, list) or len(inputs) != 1 or
            not isinstance(outputs, list) or len(outputs) != 1 or
            not isinstance(internal, list) or len(internal) != 2):
        raise ValueError("prod ZipDepth convex2x contract has invalid engine IO")
    _validate_tensor_entry(
        inputs[0], keys={"name", "dtype", "layout", "shape", "semantics"},
        name="pixel_values", layout="NCHW", shape=[1, 3, "2H", "2W"],
        label="engine input")
    _validate_tensor_entry(
        outputs[0], keys={"name", "dtype", "layout", "shape", "semantics"},
        name="refined_depth", layout="NHW", shape=[1, "2H", "2W"],
        label="engine output")
    _validate_tensor_entry(
        internal[0], keys={"name", "dtype", "layout", "shape"},
        name="dav2_pixel_values", layout="NCHW", shape=[1, 3, "H", "W"],
        label="internal DAV2 input")
    _validate_tensor_entry(
        internal[1], keys={"name", "dtype", "layout", "shape"},
        name="predicted_depth", layout="NHW", shape=[1, "H", "W"],
        label="internal DAV2 output")

    high_shapes = contract.get("high_shapes_wh")
    _, expected_high = _expected_shape_profiles()
    if (not isinstance(high_shapes, list) or len(high_shapes) != 6 or
            any(not isinstance(item, list) or len(item) != 2 or
                any(type(component) is not int or component <= 0 or component % 2 != 0
                    for component in item)
                for item in high_shapes) or
            tuple(Shape(item[0], item[1]) for item in high_shapes) != expected_high):
        raise ValueError(
            "prod ZipDepth convex2x contract high profiles must be the exact ordered six "
            "calibrated convex-2x shapes")

    tensorrt = _exact_object(
        contract.get("tensorrt"),
        {"minimum_tested_version", "builder_optimization_level", "engine_recipe",
         "profile_strategy", "profile_order", "ranged_profile_status"},
        "TensorRT recipe")
    if (tensorrt.get("minimum_tested_version") != "11.2.1" or
            type(tensorrt.get("builder_optimization_level")) is not int or
            tensorrt["builder_optimization_level"] != 5 or
            tensorrt.get("engine_recipe") != "trt-6high-point-l5-v2" or
            tensorrt.get("profile_strategy") !=
            "one-engine-six-fixed-high-point-profiles" or
            tensorrt.get("profile_order") != "high_shapes_wh" or
            tensorrt.get("ranged_profile_status") !=
            "unsupported-myelin-dynamic-convex-tail"):
        raise ValueError("prod ZipDepth convex2x contract has invalid TensorRT recipe")

    authority = _exact_object(
        contract.get("authority"), {"coarse", "refined", "failure", "forbidden"},
        "authority")
    for key in ("coarse", "refined", "failure"):
        _nonempty_string(authority.get(key), f"authority {key}")
    forbidden = authority.get("forbidden")
    if (not isinstance(forbidden, list) or
            forbidden != [
                "graph-cut", "adaptive-j", "per-pixel-hard-confidence-gate",
                "zipdepth-depth-head", "untrained-dav2-feature-to-weight-substitution"] or
            any(not isinstance(item, str) for item in forbidden)):
        raise ValueError("prod ZipDepth convex2x contract has invalid forbidden policy")
    return contract


def load_contract() -> dict[str, object]:
    with CONTRACT_PATH.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return _validate_contract(value)


def supported_high_shapes() -> tuple[Shape, ...]:
    """Return the single public input/output TensorRT point profiles."""

    contract = load_contract()
    return tuple(
        Shape(width, height)
        for width, height in contract["high_shapes_wh"]
    )


def supported_coarse_shapes() -> tuple[Shape, ...]:
    """Return the internal DAV2 shapes implied by the public 2x profiles."""

    return tuple(
        Shape(shape.width // 2, shape.height // 2)
        for shape in supported_high_shapes()
    )


def refined_shape(coarse: Shape) -> Shape:
    if coarse not in supported_coarse_shapes():
        raise ValueError(f"unsupported coarse shape: {coarse.width}x{coarse.height}")
    return Shape(2 * coarse.width, 2 * coarse.height)


def fixed_profile_index(high: Shape) -> int:
    """Return the frozen TensorRT point-profile index for a public high shape."""

    try:
        return supported_high_shapes().index(high)
    except ValueError as error:
        raise ValueError(
            f"unsupported high shape: {high.width}x{high.height}"
        ) from error


def refined_content_rect(coarse_shape: Shape, coarse: ContentRect) -> ContentRect:
    if not coarse.valid_for(coarse_shape):
        raise ValueError("coarse content rectangle is outside its tensor")
    refined_shape(coarse_shape)
    return ContentRect(
        2 * coarse.left,
        2 * coarse.top,
        2 * coarse.right,
        2 * coarse.bottom,
    )


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(values - maximum)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def convex2x(depth: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Apply ZipDepth's standard replicated 3x3 convex 2x reconstruction.

    ``depth`` is ``[B,H,W]``. ``logits`` may be ``[B,36,H,W]`` or the
    explicit ``[B,9,4,H,W]`` layout. No hard mask or pixel fallback is
    applied; a malformed/non-finite frame is rejected as one unit.
    """

    depth = np.asarray(depth, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError("depth must have shape [B,H,W]")
    batch, height, width = depth.shape
    if logits.shape == (batch, 36, height, width):
        logits = logits.reshape(batch, 9, 4, height, width)
    elif logits.shape != (batch, 9, 4, height, width):
        raise ValueError("logits must have shape [B,36,H,W] or [B,9,4,H,W]")
    if not np.isfinite(depth).all() or not np.isfinite(logits).all():
        raise ValueError("convex2x rejects a non-finite frame as one unit")

    weights = _softmax(logits, axis=1)
    padded = np.pad(depth, ((0, 0), (1, 1), (1, 1)), mode="edge")
    neighbors = np.stack(
        [
            padded[:, dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ],
        axis=1,
    )[:, :, None, :, :]
    subpixels = np.sum(weights * neighbors, axis=1)
    output = np.empty((batch, 2 * height, 2 * width), dtype=np.float32)
    for sub_y in range(2):
        for sub_x in range(2):
            output[:, sub_y::2, sub_x::2] = subpixels[:, sub_y * 2 + sub_x]
    return np.maximum(output, np.float32(0.0))


def local_bounds2x(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return replicated 3x3 coarse min/max bounds expanded to the 2x grid."""

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3 or not np.isfinite(depth).all():
        raise ValueError("depth must be a finite [B,H,W] tensor")
    _, height, width = depth.shape
    padded = np.pad(depth, ((0, 0), (1, 1), (1, 1)), mode="edge")
    neighbors = np.stack(
        [
            padded[:, dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ],
        axis=1,
    )
    minimum = np.min(neighbors, axis=1)
    maximum = np.max(neighbors, axis=1)
    return (
        np.repeat(np.repeat(minimum, 2, axis=1), 2, axis=2),
        np.repeat(np.repeat(maximum, 2, axis=1), 2, axis=2),
    )


def tensor_names(entries: Sequence[dict[str, object]]) -> tuple[str, ...]:
    return tuple(str(entry["name"]) for entry in entries)
