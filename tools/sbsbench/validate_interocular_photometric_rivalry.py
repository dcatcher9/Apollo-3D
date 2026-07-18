#!/usr/bin/env python3
"""Validate exact-map photometric rivalry on real Apollo outputs and known corruptions.

The validator never synthesizes stereo geometry.  It loads existing harness SBS eyes, exact maps
and hole masks, records their clean baseline, then applies controlled one-eye and shared
photometric transforms.  A passing corruption ladder is necessary, but not sufficient, evidence
for promoting either metric into policy; headset-rated clean/fault pairs remain the final
calibration source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import sbs_interocular_photometric_rivalry as rivalry  # noqa: E402


EXPOSURE = "interocular_exposure_rivalry_burden_pct"
COLOR = "interocular_color_gain_rivalry_burden_pct"
FOOTPRINTS = (0.001, 0.005, 0.01, 0.02, 0.05)


def _load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _encode_srgb(linear):
    linear = np.maximum(np.asarray(linear, dtype=np.float32), 0.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _linear_transform(image, gains):
    linear = rivalry._linearize_srgb(image)
    return _encode_srgb(linear * np.asarray(gains, dtype=np.float32))


def _hue_rotate(image, degrees=25.0):
    """Rotate linear RGB around the neutral axis without adding an exposure gain."""
    linear = rivalry._linearize_srgb(image)
    axis = np.asarray((1.0, 1.0, 1.0), dtype=np.float32) / np.sqrt(3.0)
    angle = math.radians(float(degrees))
    cross = np.cross(np.broadcast_to(axis, linear.shape), linear)
    projection = np.sum(linear * axis, axis=2, keepdims=True) * axis
    rotated = (linear * math.cos(angle) + cross * math.sin(angle)
               + projection * (1.0 - math.cos(angle)))
    return _encode_srgb(np.maximum(rotated, 0.0))


def _best_patch(valid, weight, fraction):
    height, width = valid.shape
    patch_height = max(2, min(height, int(round(height * math.sqrt(fraction)))))
    patch_width = max(2, min(width, int(round(width * math.sqrt(fraction)))))
    score = np.where(valid, weight, 0.0).astype(np.float64)
    integral = np.pad(score, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (integral[patch_height:, patch_width:]
            - integral[:-patch_height, patch_width:]
            - integral[patch_height:, :-patch_width]
            + integral[:-patch_height, :-patch_width])
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return slice(int(y), int(y + patch_height)), slice(int(x), int(x + patch_width))


def _nested_patches(valid, weight, fractions):
    """Choose one qualified anchor, then grow nested footprints around that same content."""
    largest = _best_patch(valid, weight, max(fractions))
    center_y = 0.5 * (largest[0].start + largest[0].stop)
    center_x = 0.5 * (largest[1].start + largest[1].stop)
    height, width = valid.shape
    result = []
    for fraction in fractions:
        patch_height = max(2, min(height, int(round(height * math.sqrt(fraction)))))
        patch_width = max(2, min(width, int(round(width * math.sqrt(fraction)))))
        y = min(max(int(round(center_y - 0.5 * patch_height)), 0), height - patch_height)
        x = min(max(int(round(center_x - 0.5 * patch_width)), 0), width - patch_width)
        result.append((slice(y, y + patch_height), slice(x, x + patch_width)))
    return result


def _source_path(clips_root, clip, frame_id):
    root = clips_root / clip
    for suffix in (".png", ".jpg", ".jpeg"):
        path = root / f"frame_{frame_id:05d}{suffix}"
        if path.exists():
            return path
    return None


def _load_case(run_dir, clips_root, clip_dir):
    frames = sorted(clip_dir.glob("sbs_*.png"))
    for sbs_path in frames:
        match = re.search(r"(\d+)$", sbs_path.stem)
        if not match:
            continue
        frame_id = int(match.group(1))
        map_path = clip_dir / f"warp_map_{frame_id:05d}.f32"
        mask_path = clip_dir / f"warp_mask_{frame_id:05d}.png"
        source_path = _source_path(clips_root, clip_dir.name, frame_id)
        shape_path = clip_dir / "warp_map_shape.json"
        if not all(path and Path(path).exists()
                   for path in (map_path, mask_path, source_path, shape_path)):
            continue
        shape = json.loads(shape_path.read_text(encoding="utf-8"))
        packed = np.fromfile(map_path, dtype="<f4")
        expected_size = int(shape["width"]) * int(shape["height"])
        if packed.size != expected_size:
            raise ValueError(f"{map_path}: expected {expected_size} floats, got {packed.size}")
        mapping = packed.reshape(int(shape["height"]), int(shape["width"]))
        sbs = _load_rgb(sbs_path)
        if sbs.shape[:2] != mapping.shape:
            raise ValueError(f"{sbs_path}: SBS and exact map geometry differ")
        eye_width = int(shape["eye_width"])
        return {
            "clip": clip_dir.name,
            "frame_id": frame_id,
            "source": _load_rgb(source_path),
            "eyes": (sbs[:, :eye_width], sbs[:, eye_width:]),
            "maps": (mapping[:, :eye_width], mapping[:, eye_width:]),
            "mask": _load_mask(mask_path),
            "shape": shape,
            "paths": {
                "source": os.path.relpath(source_path, run_dir),
                "sbs": os.path.relpath(sbs_path, run_dir),
                "map": os.path.relpath(map_path, run_dir),
                "mask": os.path.relpath(mask_path, run_dir),
            },
        }
    return None


def _measure(case, eyes, return_maps=False):
    return rivalry.measure_interocular_photometric_rivalry(
        case["source"], eyes[0], eyes[1], case["maps"][0], case["maps"][1],
        case["shape"], warp_mask=case["mask"], min_pixels=64,
        return_maps=return_maps)


def _record(case):
    clean = _measure(case, case["eyes"])
    shared_exposure_eyes = tuple(_linear_transform(eye, (1.18, 1.18, 1.18))
                                 for eye in case["eyes"])
    shared_gain_eyes = tuple(_linear_transform(eye, (1.16, 0.93, 0.84))
                             for eye in case["eyes"])
    global_exposure = _measure(
        case, (case["eyes"][0], _linear_transform(case["eyes"][1], (1.18,) * 3)))
    global_rgb_gain = _measure(
        case, (case["eyes"][0], _linear_transform(
            case["eyes"][1], (1.16, 0.93, 0.84))))
    global_hue_eye = _hue_rotate(case["eyes"][1], 25.0)
    global_hue = _measure(case, (case["eyes"][0], global_hue_eye))
    global_hue_excitation = float(np.sqrt(np.mean(np.square(
        global_hue_eye - case["eyes"][1]))))

    expected = rivalry._exact._sample_source_eye(
        case["source"], case["maps"][1], case["shape"])
    right_hole = case["mask"][:, case["eyes"][0].shape[1]:, 0] > 0.0
    valid = ~right_hole
    expected_linear = rivalry._linearize_srgb(expected)
    weight = expected_linear @ rivalry._LUMA
    channel_mean = np.mean(expected_linear, axis=2, keepdims=True)
    chroma_weight = np.linalg.norm(expected_linear - channel_mean, axis=2) * np.sqrt(weight)
    exposure_patches = _nested_patches(valid, weight, FOOTPRINTS)
    hue_patches = _nested_patches(valid, chroma_weight, FOOTPRINTS)
    exposure_ladder, gain_ladder, hue_ladder, hue_excitations = [], [], [], []
    for patch, hue_patch in zip(exposure_patches, hue_patches):
        right = case["eyes"][1].copy()
        right[patch] = _linear_transform(right[patch], (1.35,) * 3)
        exposure_ladder.append(_measure(case, (case["eyes"][0], right)))

        right = case["eyes"][1].copy()
        right[patch] = _linear_transform(right[patch], (1.30, 0.80, 1.12))
        gain_ladder.append(_measure(case, (case["eyes"][0], right)))

        right = case["eyes"][1].copy()
        original_hue_patch = right[hue_patch].copy()
        right[hue_patch] = _hue_rotate(original_hue_patch, 40.0)
        hue_excitations.append(float(np.sqrt(np.mean(np.square(
            right[hue_patch] - original_hue_patch)))))
        hue_ladder.append(_measure(case, (case["eyes"][0], right)))

    return {
        "clip": case["clip"],
        "frame_id": case["frame_id"],
        "paths": case["paths"],
        "clean": clean,
        "shared_exposure": _measure(case, shared_exposure_eyes),
        "shared_rgb_gain": _measure(case, shared_gain_eyes),
        "global_unilateral_exposure": global_exposure,
        "global_unilateral_rgb_gain": global_rgb_gain,
        "global_unilateral_hue": global_hue,
        "global_hue_excitation_rms": global_hue_excitation,
        "localized_exposure_ladder": exposure_ladder,
        "localized_rgb_gain_ladder": gain_ladder,
        "localized_hue_ladder": hue_ladder,
        "localized_hue_excitation_rms": hue_excitations,
    }


def _evaluate(records):
    checks = []

    def check(name, passed, **values):
        checks.append({"name": name, "passed": bool(passed), **values})

    for record in records:
        clean = record["clean"]
        clip = record["clip"]
        supported = all(clean[key] == 100.0 for key in (
            "interocular_exposure_rivalry_evidence_sufficient",
            "interocular_color_gain_rivalry_evidence_sufficient"))
        check(f"{clip}:clean_support", supported)
        if not supported:
            continue
        shared_exposure = record["shared_exposure"]
        shared_gain = record["shared_rgb_gain"]
        check(
            f"{clip}:shared_exposure_control",
            abs(shared_exposure[EXPOSURE] - clean[EXPOSURE]) <= 0.35
            and abs(shared_exposure[COLOR] - clean[COLOR]) <= 0.35,
            clean_exposure=clean[EXPOSURE], transformed_exposure=shared_exposure[EXPOSURE],
            clean_color=clean[COLOR], transformed_color=shared_exposure[COLOR])
        check(
            f"{clip}:shared_rgb_gain_control",
            abs(shared_gain[EXPOSURE] - clean[EXPOSURE]) <= 0.50
            and abs(shared_gain[COLOR] - clean[COLOR]) <= 0.50,
            clean_exposure=clean[EXPOSURE], transformed_exposure=shared_gain[EXPOSURE],
            clean_color=clean[COLOR], transformed_color=shared_gain[COLOR])
        for scenario, metric, minimum in (
                ("global_unilateral_exposure", EXPOSURE, 5.0),
                ("global_unilateral_rgb_gain", COLOR, 5.0)):
            value = record[scenario][metric]
            check(f"{clip}:{scenario}", value >= clean[metric] + minimum,
                  clean=clean[metric], corrupted=value)
        hue_applicable = record["global_hue_excitation_rms"] >= 0.005
        hue_value = record["global_unilateral_hue"][COLOR]
        check(
            f"{clip}:global_unilateral_hue",
            not hue_applicable or hue_value >= clean[COLOR] + 5.0,
            applicable=hue_applicable, excitation_rms=record["global_hue_excitation_rms"],
            clean=clean[COLOR], corrupted=hue_value)
        for scenario, metric in (
                ("localized_exposure_ladder", EXPOSURE),
                ("localized_rgb_gain_ladder", COLOR)):
            values = [item[metric] for item in record[scenario]]
            monotonic = all(later >= earlier - 0.10
                            for earlier, later in zip(values, values[1:]))
            one_percent = values[2] >= clean[metric] + 0.35
            check(f"{clip}:{scenario}", monotonic and one_percent,
                  clean=clean[metric], footprints=list(FOOTPRINTS), values=values)
        values = [item[COLOR] for item in record["localized_hue_ladder"]]
        hue_applicable = record["localized_hue_excitation_rms"][-1] >= 0.005
        monotonic = all(later >= earlier - 0.10
                        for earlier, later in zip(values, values[1:]))
        one_percent = values[2] >= clean[COLOR] + 0.35
        check(
            f"{clip}:localized_hue_ladder",
            not hue_applicable or (monotonic and one_percent),
            applicable=hue_applicable,
            excitation_rms=record["localized_hue_excitation_rms"],
            clean=clean[COLOR], footprints=list(FOOTPRINTS), values=values)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, default=SCRIPT_DIR / "clips")
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run.resolve()
    cases = []
    for clip_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        case = _load_case(run_dir, args.clips_root.resolve(), clip_dir)
        if case is not None:
            cases.append(case)
        if len(cases) >= args.max_clips:
            break
    if not cases:
        raise SystemExit("no complete source/SBS/exact-map/mask cases found")
    records = [_record(case) for case in cases]
    checks = _evaluate(records)
    payload = {
        "schema": 1,
        "metric": "exact-map source-relative interocular photometric rivalry",
        "run": str(run_dir),
        "footprint_fractions": list(FOOTPRINTS),
        "records": records,
        "checks": checks,
        "summary": {
            "passed": sum(item["passed"] for item in checks),
            "total": len(checks),
            "all_passed": all(item["passed"] for item in checks),
        },
    }
    output = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if payload["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
