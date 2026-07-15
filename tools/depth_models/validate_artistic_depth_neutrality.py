#!/usr/bin/env python3
"""Fail-closed ONNX depth-neutrality check for the shared-feature policy head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from artistic_geometry_contract import (
    DEFAULT_DEPTH_MAX_ASPECT,
    DEFAULT_DEPTH_SHORT_SIDE,
    aspect_aligned_dims,
)
from train_artistic_policy import (
    MAX_HEIGHT,
    MAX_WIDTH,
    MEAN,
    STD,
)


PREPROCESSING_CONTRACT = "apollo-dav2-srgb-native-capped-v1"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(
        path: Path,
        depth_short_side=DEFAULT_DEPTH_SHORT_SIDE,
        depth_max_aspect=DEFAULT_DEPTH_MAX_ASPECT):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot load neutrality input: {path}")
    width, height = aspect_aligned_dims(
        bgr.shape[1], bgr.shape[0],
        depth_short_side=depth_short_side,
        depth_max_aspect=depth_max_aspect,
        max_width=min(MAX_WIDTH, bgr.shape[1]),
        max_height=min(MAX_HEIGHT, bgr.shape[0]),
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    image = rgb.astype(np.float32) / 255.0
    image = ((image - MEAN) / STD).transpose(2, 0, 1)
    return (
        image[None].astype(np.float32), width, height,
        bgr.shape[1], bgr.shape[0],
    )


def session(path: Path):
    options = ort.SessionOptions()
    # ORT's CPU FP16 optimizer currently attempts an invalid LayerNorm fusion on
    # the reference graph. Disabling rewrites compares the stored graphs directly.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def production_percentile_normalize(depth):
    """Mirror depth_hist_cs + depth_minmax_ema_cs for an initialized frame."""
    values = np.maximum(np.asarray(depth, dtype=np.float32), 0.0)
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0)
    raw_min = float(np.min(values))
    raw_max = float(np.max(values))
    raw_range = raw_max - raw_min
    if raw_range <= 1e-12:
        return np.zeros_like(values)

    bin_count = 256
    bins = np.minimum(
        ((values - raw_min) * (bin_count / raw_range)).astype(np.int32),
        bin_count - 1,
    )
    histogram = np.bincount(bins.ravel(), minlength=bin_count)
    cumulative = np.cumsum(histogram)
    pixel_count = values.size
    low_bin = int(np.searchsorted(cumulative, 0.02 * pixel_count))
    high_bin = int(np.searchsorted(cumulative, 0.98 * pixel_count))
    bin_width = raw_range / bin_count
    low = raw_min + (low_bin + 0.5) * bin_width
    high = raw_min + (high_bin + 0.5) * bin_width
    if high - low <= 1e-9:
        low, high = raw_min, raw_max
    return np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)


def compare(reference, candidate, image, normalized_mean_limit,
            normalized_p99_limit):
    reference_depth = reference.run(
        ["predicted_depth"], {"pixel_values": image}
    )[0]
    candidate_depth, policy = candidate.run(
        ["predicted_depth", "artistic_global"], {"pixel_values": image}
    )
    if reference_depth.shape != candidate_depth.shape:
        raise RuntimeError(
            f"depth output shape changed: {reference_depth.shape} != "
            f"{candidate_depth.shape}"
        )
    if policy.shape != (1, 2) or not np.all(np.isfinite(policy)):
        raise RuntimeError(f"invalid artistic_global output: {policy.shape}")
    difference = np.abs(reference_depth - candidate_depth)
    denominator = max(float(np.mean(np.abs(reference_depth))), 1e-9)
    reference_normalized = production_percentile_normalize(reference_depth)
    candidate_normalized = production_percentile_normalize(candidate_depth)
    normalized_difference = np.abs(
        reference_normalized - candidate_normalized
    )
    result = {
        "raw": {
            "mean_abs": float(np.mean(difference)),
            "relative_mean_abs": float(np.mean(difference) / denominator),
            "p99_abs": float(np.percentile(difference, 99)),
            "max_abs": float(np.max(difference)),
        },
        "production_normalized": {
            "mean_abs": float(np.mean(normalized_difference)),
            "p99_abs": float(np.percentile(normalized_difference, 99)),
            "max_abs": float(np.max(normalized_difference)),
        },
        "policy": [float(value) for value in policy[0]],
    }
    normalized = result["production_normalized"]
    result["passed"] = (
        normalized["mean_abs"] <= normalized_mean_limit and
        normalized["p99_abs"] <= normalized_p99_limit
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--image", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--depth-short-side", type=int, default=DEFAULT_DEPTH_SHORT_SIDE
    )
    parser.add_argument(
        "--depth-max-aspect", type=float, default=DEFAULT_DEPTH_MAX_ASPECT
    )
    # These limits are expressed in the normalized [0,1] depth consumed by the
    # warp. One 10-bit depth code on average and two codes at p99 keep the
    # feature-head export drift subpixel after the Bestv2 curve without
    # pretending raw DA-V2 units have a stable absolute scale.
    parser.add_argument(
        "--normalized-mean-limit", type=float, default=1.0 / 1024.0
    )
    parser.add_argument(
        "--normalized-p99-limit", type=float, default=2.0 / 1024.0
    )
    args = parser.parse_args()

    reference_sha256 = sha256(args.reference)
    candidate_sha256 = sha256(args.candidate)
    if reference_sha256 == candidate_sha256:
        raise RuntimeError(
            "neutrality reference and candidate are the same ONNX bytes"
        )
    reference = session(args.reference)
    candidate = session(args.candidate)
    rows = []
    for path in args.image:
        image, width, height, source_width, source_height = prepare_image(
            path, args.depth_short_side, args.depth_max_aspect
        )
        result = compare(
            reference, candidate, image,
            args.normalized_mean_limit, args.normalized_p99_limit,
        )
        rows.append({
            "image": str(path.resolve()),
            "image_sha256": sha256(path),
            "source_width": source_width,
            "source_height": source_height,
            "input_shape": [1, 3, height, width],
            **result,
        })
    payload = {
        "schema": 4,
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "preprocessing": {
            "depth_short_side": args.depth_short_side,
            "depth_max_aspect": args.depth_max_aspect,
            "max_width": MAX_WIDTH,
            "max_height": MAX_HEIGHT,
            "resize_interpolation": "opencv-inter-linear",
            "color_conversion": "opencv-bgr8-to-rgb-srgb",
        },
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": reference_sha256,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": candidate_sha256,
        },
        "limits": {
            "production_normalized_mean_abs": args.normalized_mean_limit,
            "production_normalized_p99_abs": args.normalized_p99_limit,
        },
        "passed": all(row["passed"] for row in rows),
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
