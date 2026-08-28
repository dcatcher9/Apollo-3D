#!/usr/bin/env python3
"""Reference contract for the frozen ZipDepth-guided production DAV2 convex2x path."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "contracts"
    / "prod-zipdepth-convex2x-v1.json"
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


def load_contract() -> dict[str, object]:
    with CONTRACT_PATH.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("unsupported prod ZipDepth convex2x contract")
    return value


def supported_coarse_shapes() -> tuple[Shape, ...]:
    contract = load_contract()
    return tuple(Shape(int(width), int(height)) for width, height in contract["coarse_shapes_wh"])


def refined_shape(coarse: Shape) -> Shape:
    if coarse not in supported_coarse_shapes():
        raise ValueError(f"unsupported coarse shape: {coarse.width}x{coarse.height}")
    return Shape(2 * coarse.width, 2 * coarse.height)


def fixed_profile_index(coarse: Shape) -> int:
    """Return the frozen TensorRT point-profile index for a production shape."""

    try:
        return supported_coarse_shapes().index(coarse)
    except ValueError as error:
        raise ValueError(
            f"unsupported coarse shape: {coarse.width}x{coarse.height}"
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
