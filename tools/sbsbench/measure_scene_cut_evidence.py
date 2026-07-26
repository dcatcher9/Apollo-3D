#!/usr/bin/env python3
"""Measure Apollo's scene-cut evidence per frame without modifying benchmark artifacts.

The appearance paths mirror production: source RGB is resized with D3D11's pixel-center bilinear
geometry for the broad model-input delta, while the exposure ordinal is point sampled in the
capture domain before tone mapping or bilinear mixing. The ordinal is compared with the
cross-5/all-10-pairs census. The depth path compares the harness's normalized 16-bit depth dumps
at the same 0.05 texel threshold used by production.

Example:
  python tools/sbsbench/measure_scene_cut_evidence.py \
    --artifacts-root cmake-build-relwithdebinfo/sbs_eval/<core-run> --summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
DEFAULT_CLIPS_ROOT = SCRIPT_DIR / "clips"
FRAME_RE = re.compile(r"^frame_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
DEPTH_RE = re.compile(r"^depth_(\d+)\.png$", re.IGNORECASE)
ORDINAL_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def numbered_files(root: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in root.iterdir():
        match = pattern.match(path.name)
        if not match or not path.is_file():
            continue
        frame_id = int(match.group(1))
        if frame_id in files:
            raise ValueError(f"duplicate frame identity {frame_id} in {root}")
        files[frame_id] = path
    return files


def d3d_bilinear_resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize RGB at the same normalized pixel centers used by rgb_to_nchw_cs.hlsl."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB input, got {rgb.shape}")
    source_height, source_width, _ = rgb.shape
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("source and target dimensions must be positive")

    x = (np.arange(width, dtype=np.float32) + 0.5) * (source_width / width) - 0.5
    y = (np.arange(height, dtype=np.float32) + 0.5) * (source_height / height) - 0.5
    x_floor = np.floor(x).astype(np.int32)
    y_floor = np.floor(y).astype(np.int32)
    x_weight = (x - x_floor)[None, :, None]
    y_weight = (y - y_floor)[:, None, None]
    x0 = np.clip(x_floor, 0, source_width - 1)
    x1 = np.clip(x_floor + 1, 0, source_width - 1)
    y0 = np.clip(y_floor, 0, source_height - 1)
    y1 = np.clip(y_floor + 1, 0, source_height - 1)

    top = rgb[y0[:, None], x0[None, :]] * (1.0 - x_weight)
    top += rgb[y0[:, None], x1[None, :]] * x_weight
    bottom = rgb[y1[:, None], x0[None, :]] * (1.0 - x_weight)
    bottom += rgb[y1[:, None], x1[None, :]] * x_weight
    return top * (1.0 - y_weight) + bottom * y_weight


def model_rgb(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return d3d_bilinear_resize_rgb(rgb, width, height)


def point_resize_max_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match rgb_to_nchw_cs's capture-domain point maxRGB output."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB input, got {rgb.shape}")
    source_height, source_width, _ = rgb.shape
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    x = np.floor(
        (np.arange(width, dtype=np.float64) + 0.5) * source_width / width
    ).astype(np.int64)
    y = np.floor(
        (np.arange(height, dtype=np.float64) + 0.5) * source_height / height
    ).astype(np.int64)
    x = np.clip(x, 0, source_width - 1)
    y = np.clip(y, 0, source_height - 1)
    return np.max(rgb[y[:, None], x[None, :]], axis=2)


def appearance_ordinal(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return point_resize_max_rgb(rgb, width, height)


def ordinal_samples(values: np.ndarray) -> tuple[np.ndarray, ...]:
    padded = np.pad(values, 1, mode="edge")
    height, width = values.shape
    return tuple(
        padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        for dx, dy in ORDINAL_OFFSETS
    )


def structural_change_fraction(
        current: np.ndarray,
        previous: np.ndarray,
        contrast_floor: float = 0.01) -> float:
    """Shader-equivalent maxRGB cross-5 ordinal-change fraction."""
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError(
            f"structural planes must be equal-size 2-D arrays, got "
            f"{current.shape} and {previous.shape}")
    current_samples = ordinal_samples(current)
    previous_samples = ordinal_samples(previous)
    common = np.zeros(current.shape, dtype=np.uint8)
    flips = np.zeros(current.shape, dtype=np.uint8)
    for first in range(4):
        for second in range(first + 1, 5):
            current_delta = current_samples[first] - current_samples[second]
            previous_delta = previous_samples[first] - previous_samples[second]
            reliable = ((np.abs(current_delta) >= contrast_floor) &
                        (np.abs(previous_delta) >= contrast_floor))
            common += reliable
            flips += reliable & ((current_delta < 0.0) != (previous_delta < 0.0))
    changed = (common >= 4) & (flips >= 2) & (2 * flips >= common)
    return float(np.mean(changed))


def raw_rgb_change_fraction(
        current: np.ndarray,
        previous: np.ndarray,
        threshold: float = 0.20) -> float:
    if current.shape != previous.shape or current.ndim != 3 or current.shape[2] != 3:
        raise ValueError(
            f"RGB planes must be equal-size HxWx3 arrays, got "
            f"{current.shape} and {previous.shape}")
    return float(np.mean(np.max(np.abs(current - previous), axis=2) >= threshold))


def normalized_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image, dtype=np.float32)
    return depth / 65535.0


def depth_change_fraction(
        current: np.ndarray,
        previous: np.ndarray,
        threshold: float = 0.05) -> float:
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError(
            f"depth planes must be equal-size 2-D arrays, got "
            f"{current.shape} and {previous.shape}")
    return float(np.mean(np.abs(current - previous) >= threshold))


def measure_clip(
        clip_dir: Path,
        artifact_dir: Path,
        contrast_floor: float = 0.01,
        depth_threshold: float = 0.05,
        rgb_threshold: float = 0.20) -> list[dict[str, int | float | str]]:
    source_files = numbered_files(clip_dir, FRAME_RE)
    depth_files = numbered_files(artifact_dir, DEPTH_RE)
    if not source_files:
        raise ValueError(f"no source frames in {clip_dir}")
    if not depth_files:
        raise ValueError(f"no normalized depth dumps in {artifact_dir}")
    if set(source_files) != set(depth_files):
        missing = sorted(set(source_files) - set(depth_files))
        unexpected = sorted(set(depth_files) - set(source_files))
        raise ValueError(
            f"{clip_dir.name}: source/depth frame identities differ; "
            f"missing_depth={missing}, unexpected_depth={unexpected}")

    with open(artifact_dir / "raw_shape.json", encoding="utf-8") as stream:
        raw_shape = json.load(stream)
    width = int(raw_shape["width"])
    height = int(raw_shape["height"])

    records: list[dict[str, int | float | str]] = []
    previous_rgb = None
    previous_ordinal = None
    previous_depth = None
    previous_frame = None
    for frame_id in sorted(source_files):
        rgb = model_rgb(source_files[frame_id], width, height)
        ordinal = appearance_ordinal(source_files[frame_id], width, height)
        depth = normalized_depth(depth_files[frame_id])
        if depth.shape != (height, width):
            raise ValueError(
                f"{clip_dir.name} frame {frame_id}: depth shape {depth.shape} "
                f"does not match raw model shape {(height, width)}")
        if (previous_rgb is not None and previous_ordinal is not None and
                previous_depth is not None):
            records.append({
                "clip": clip_dir.name,
                "previous_frame": int(previous_frame),
                "frame": frame_id,
                "raw_rgb_change_fraction": raw_rgb_change_fraction(
                    rgb, previous_rgb, rgb_threshold),
                "structural_change_fraction": structural_change_fraction(
                    ordinal,
                    previous_ordinal,
                    contrast_floor,
                ),
                "depth_change_fraction": depth_change_fraction(
                    depth, previous_depth, depth_threshold),
            })
        previous_rgb = rgb
        previous_ordinal = ordinal
        previous_depth = depth
        previous_frame = frame_id
    return records


def measure_suite(
        clips_root: Path,
        artifacts_root: Path,
        contrast_floor: float = 0.01,
        depth_threshold: float = 0.05,
        rgb_threshold: float = 0.20) -> list[dict[str, int | float | str]]:
    records: list[dict[str, int | float | str]] = []
    for clip_dir in sorted(path for path in clips_root.iterdir() if path.is_dir()):
        artifact_dir = artifacts_root / clip_dir.name
        if artifact_dir.is_dir():
            records.extend(measure_clip(
                clip_dir, artifact_dir, contrast_floor, depth_threshold, rgb_threshold))
    if not records:
        raise ValueError(
            f"no matching clip/artifact directories under {clips_root} and {artifacts_root}")
    return records


def summaries(records: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    grouped: dict[str, list[dict[str, int | float | str]]] = {}
    for record in records:
        grouped.setdefault(str(record["clip"]), []).append(record)
    result = []
    for clip, rows in sorted(grouped.items()):
        raw_rgb = max(rows, key=lambda row: float(row["raw_rgb_change_fraction"]))
        structural = max(rows, key=lambda row: float(row["structural_change_fraction"]))
        depth = max(rows, key=lambda row: float(row["depth_change_fraction"]))
        result.append({
            "clip": clip,
            "pairs": len(rows),
            "max_raw_rgb_frame": int(raw_rgb["frame"]),
            "max_raw_rgb_change_fraction": float(raw_rgb["raw_rgb_change_fraction"]),
            "max_structural_frame": int(structural["frame"]),
            "max_structural_change_fraction": float(
                structural["structural_change_fraction"]),
            "max_depth_frame": int(depth["frame"]),
            "max_depth_change_fraction": float(depth["depth_change_fraction"]),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips-root", type=Path, default=DEFAULT_CLIPS_ROOT,
        help="committed source clip root (default: tools/sbsbench/clips)")
    parser.add_argument(
        "--artifacts-root", type=Path, required=True,
        help="read-only harness run containing per-clip depth PNGs and raw_shape.json")
    parser.add_argument("--contrast-floor", type=float, default=0.01)
    parser.add_argument("--depth-threshold", type=float, default=0.05)
    parser.add_argument("--rgb-threshold", type=float, default=0.20)
    parser.add_argument(
        "--summary", action="store_true",
        help="print per-clip maxima instead of every adjacent-frame pair")
    parser.add_argument(
        "--json", action="store_true",
        help="emit JSON rather than CSV")
    args = parser.parse_args()

    try:
        records = measure_suite(
            args.clips_root.resolve(),
            args.artifacts_root.resolve(),
            args.contrast_floor,
            args.depth_threshold,
            args.rgb_threshold,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = summaries(records) if args.summary else records
    if args.json:
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
