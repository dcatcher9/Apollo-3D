#!/usr/bin/env python3
"""Falsify the optional FLIP oracle on authenticated real-source frames.

This is a controlled-corruption qualification tool, not an evaluator gate.  It regenerates clean
eyes through exact intended maps, injects known appearance faults, verifies benign geometry and
masking invariance, and preserves per-frame provenance in a JSON audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import authenticated_metric_sources as authenticated_sources  # noqa: E402
import flip_appearance_oracle as oracle  # noqa: E402
import sbs_interocular_phase_chroma as exact  # noqa: E402


SCHEMA = 1
P99 = "flip_worst_eye_p99"
AREA = "flip_worst_eye_area_gt_050_pct"
IMBALANCE = "flip_interocular_error_imbalance_p99"
PREFERRED_CLIPS = (
    "aigen_cogvideox_rain",
    "anime_morevna_closeup",
    "c647",
    "c841",
)


def _select_clips(clips, maximum):
    by_id = {clip["id"]: clip for clip in clips}
    selected = [by_id[name] for name in PREFERRED_CLIPS if name in by_id]
    selected_ids = {clip["id"] for clip in selected}
    selected.extend(
        clip for clip in clips
        if clip["id"] not in selected_ids and clip["meta"].get("expected_flat") is not True
    )
    return selected[:maximum]


def _maps(height, width, disparity_px=8.0, fractional_px=0.43):
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    left = np.broadcast_to(u[None, :], (height, width)).copy()
    right = left.copy()
    amount = 0.5 * disparity_px / width + fractional_px / width
    left += amount
    right -= amount
    return left, right


def _render(source, maps, shape):
    return tuple(exact._sample_source_eye(source, mapping, shape) for mapping in maps)


def _blur(image, radius):
    encoded = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))
    return np.asarray(encoded.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def _measure(source, eyes, maps, shape):
    result = oracle.measure_flip_appearance(
        source, *eyes, *maps, shape, min_support_pixels=64)
    if result["status"] != "ok":
        raise RuntimeError(f"FLIP unexpectedly abstained: {result}")
    return result


def _check(name, passed, **evidence):
    return {"name": name, "status": "pass" if passed else "fail", **evidence}


def _detail_patch(source):
    height, width = source.shape[:2]
    luma = source @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    gy[1:] = np.abs(luma[1:] - luma[:-1])
    energy = gx + gy
    patch_width = max(48, int(round(width * 0.42)))
    patch_height = max(36, int(round(height * 0.46)))
    half_width, half_height = patch_width // 2, patch_height // 2
    energy[:half_height] = -1.0
    energy[-half_height:] = -1.0
    energy[:, :half_width] = -1.0
    energy[:, -half_width:] = -1.0
    center_y, center_x = np.unravel_index(np.argmax(energy), energy.shape)
    y0 = int(np.clip(center_y - half_height, 0, height - patch_height))
    x0 = int(np.clip(center_x - half_width, 0, width - patch_width))
    return slice(y0, y0 + patch_height), slice(x0, x0 + patch_width)


def validate_source(source, identity):
    height, width = source.shape[:2]
    shape = {"content_scale_x": 1.0, "content_scale_y": 1.0}
    maps = _maps(height, width)
    clean_eyes = _render(source, maps, shape)
    clean = _measure(source, clean_eyes, maps, shape)
    patch_y, patch_x = _detail_patch(source)

    blur_ladder = []
    for radius in (1, 2, 4):
        blur_ladder.append(_measure(
            source, (clean_eyes[0], _blur(clean_eyes[1], radius)), maps, shape))

    # A high-contrast line is added to the authenticated frame before rendering, then removed
    # from one rendered eye.  This isolates deletion sensitivity without claiming that a real
    # source structure is a line when its semantics are unknown.
    line_source = source.copy()
    line_x = width // 2
    line_colour = np.where(
        np.mean(line_source[:, line_x], axis=0) > 0.5,
        np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    line_source[height // 6:5 * height // 6, line_x:line_x + 2] = line_colour
    line_eyes = list(_render(line_source, maps, shape))
    deleted = line_eyes[1].copy()
    deleted[height // 6:5 * height // 6, line_x:line_x + 2] = \
        deleted[height // 6:5 * height // 6, line_x - 3:line_x - 1]
    deletion = _measure(line_source, (line_eyes[0], deleted), maps, shape)

    halo = clean_eyes[1].copy()
    # Keep the injected band inside the selected patch so even a patch clipped to the image edge
    # has a non-empty, measurable halo corruption.
    left_band = slice(patch_x.start, min(patch_x.start + 4, patch_x.stop))
    halo[patch_y, left_band] = np.clip(halo[patch_y, left_band] + 0.30, 0.0, 1.0)
    halo_result = _measure(source, (clean_eyes[0], halo), maps, shape)

    ringing = clean_eyes[1].copy()
    edge = patch_x.stop
    ringing[patch_y, edge:edge + 3] = np.clip(
        ringing[patch_y, edge:edge + 3] + 0.30, 0.0, 1.0)
    ringing[patch_y, edge + 3:edge + 6] = np.clip(
        ringing[patch_y, edge + 3:edge + 6] - 0.30, 0.0, 1.0)
    ringing_result = _measure(source, (clean_eyes[0], ringing), maps, shape)

    jagged = clean_eyes[1].copy()
    patch = jagged[patch_y, patch_x].copy()
    for row in range(patch.shape[0]):
        patch[row] = np.roll(patch[row], 3 if row % 4 < 2 else -3, axis=0)
    jagged[patch_y, patch_x] = patch
    jagged_result = _measure(source, (clean_eyes[0], jagged), maps, shape)

    shifted = np.roll(clean_eyes[1], 3, axis=1)
    doubled = 0.65 * clean_eyes[1] + 0.35 * shifted
    double_result = _measure(source, (clean_eyes[0], doubled), maps, shape)

    symmetric_blur = _measure(
        source, (_blur(clean_eyes[0], 2), _blur(clean_eyes[1], 2)), maps, shape)

    bar_scale = 0.72
    lo = 0.5 * (1.0 - bar_scale)
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / width
    bar_u = (output_u - lo) / bar_scale
    bar_maps = tuple(
        np.broadcast_to(bar_u[None, :], (height, width)).copy() for _ in range(2))
    bar_shape = {"content_scale_x": bar_scale, "content_scale_y": 1.0}
    bar_eyes = [eye.copy() for eye in _render(source, bar_maps, bar_shape)]
    outside = (output_u < lo) | (output_u > 1.0 - lo)
    rows, columns = np.indices((height, width))
    hostile = ((rows + 3 * columns) % 5 < 2).astype(np.float32)
    bar_eyes[0][:, outside] = np.stack(
        (hostile[:, outside], 1.0 - hostile[:, outside], hostile[:, outside]), axis=2)
    bar_eyes[1][:, outside] = np.stack(
        (1.0 - hostile[:, outside], hostile[:, outside], 1.0 - hostile[:, outside]), axis=2)
    bars = _measure(source, tuple(bar_eyes), bar_maps, bar_shape)

    scenarios = {
        "clean_exact_geometry": clean,
        "blur_ladder": blur_ladder,
        "thin_line_deletion": deletion,
        "halo": halo_result,
        "ringing": ringing_result,
        "jagged_edge": jagged_result,
        "double_edge": double_result,
        "symmetric_blur": symmetric_blur,
        "hostile_bars": bars,
    }
    blur_scores = [entry["metrics"][P99] for entry in blur_ladder]
    fault_results = (deletion, halo_result, ringing_result, jagged_result, double_result)
    checks = [
        _check("clean_zero", clean["metrics"][P99] < 1e-6
               and clean["metrics"][AREA] < 1e-6, metrics=clean["metrics"]),
        _check("blur_monotonic", all(
            later > earlier + 0.01
            for earlier, later in zip(blur_scores, blur_scores[1:])), scores=blur_scores),
        # Sub-one-percent thin structures are intentionally caught by the thresholded area even
        # when a frame-wide p99 cannot include them.  Broader faults must trip at least one of the
        # two complementary localized pools; neither a mean nor a fused scalar is used.
        _check("all_local_faults_detected", all(
            entry["metrics"][P99] > 0.05 or entry["metrics"][AREA] >= 0.05
            for entry in fault_results),
            metrics=[entry["metrics"] for entry in fault_results]),
        _check("one_eye_fault_is_imbalanced",
               blur_ladder[1]["metrics"][IMBALANCE] > 0.10,
               metrics=blur_ladder[1]["metrics"]),
        _check("symmetric_fault_has_low_interocular_imbalance",
               symmetric_blur["metrics"][IMBALANCE]
               < max(0.02, blur_ladder[1]["metrics"][IMBALANCE] * 0.20),
               one_eye=blur_ladder[1]["metrics"], symmetric=symmetric_blur["metrics"]),
        _check("bars_excluded", bars["metrics"][P99] < 1e-6
               and bars["metrics"][AREA] < 1e-6, metrics=bars["metrics"]),
    ]
    return {
        "source": identity,
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "checks": checks,
        "scenarios": scenarios,
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "extended", "both"), default="core")
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    try:
        _, version = oracle.load_official_flip()
    except oracle.FlipUnavailable as error:
        payload = {
            "schema": SCHEMA,
            "status": "unavailable",
            "reason": str(error),
            "training_label_eligible": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 2

    clips = authenticated_sources.discover_clips(
        authenticated_sources.suite_roots(args.suite))
    selected = _select_clips(clips, args.max_clips)
    records = []
    for clip in selected:
        frames = authenticated_sources.deterministic_frame_sample(
            clip["frames"], args.frames_per_clip)
        for frame_path in frames:
            source = authenticated_sources.load_frame(frame_path, args.max_width)
            records.append(validate_source(source, {
                "clip": clip["id"],
                "frame": Path(frame_path).name,
                "sha256": authenticated_sources.sample_sha256(frame_path),
                "dataset": clip["meta"].get("dataset"),
                "citation": clip["meta"].get("citation"),
                "license_note": clip["meta"].get("license_note"),
            }))
    checks = [check for record in records for check in record["checks"]]
    failed = sum(check["status"] == "fail" for check in checks)
    payload = {
        "schema": SCHEMA,
        "validator": "authenticated-real-source-FLIP-corruption-suite",
        "status": "pass" if records and failed == 0 else "fail",
        "role": "offline_metric_falsification_only",
        "training_label_eligible": False,
        "dependency": {"package": "flip-evaluator", "version": version},
        "suite": args.suite,
        "summary": {
            "sources": len(records),
            "checks": len(checks),
            "passed": len(checks) - failed,
            "failed": failed,
            "elapsed_seconds": time.time() - started,
        },
        "sources": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
