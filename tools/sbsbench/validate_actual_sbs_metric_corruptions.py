#!/usr/bin/env python3
"""Falsify retained deterministic SBS metrics on real harness output.

This is deliberately a standalone validation tool, not an evaluator or threshold tuner.  It
loads an already-rendered schema-32 run, authenticates the source clip and exact production
inverse-warp sidecars, then injects bounded defects into copies of the *actual* SBS eyes.  It
checks detector direction, footprint response, localization, support, and benign controls.

The output can never qualify training labels.  A failed or incomplete check exits non-zero and
the JSON report always carries ``training_label_qualification = blocked``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
import authenticated_metric_sources as authenticated_sources  # noqa: E402
import run_eval  # noqa: E402
import sbs_interocular_phase_chroma as phase_chroma  # noqa: E402
import sbs_interocular_photometric_rivalry as photometric_rivalry  # noqa: E402
import sbsbench  # noqa: E402


SCHEMA = 1
EXPECTED_EVAL_SCHEMA = 32
EXPECTED_HARNESS_SCHEMA = 16
DEFAULT_RUN = os.path.join(
    REPO_ROOT, "cmake-build-relwithdebinfo", "sbs_eval", "metric-schema32-core-control")
PREFERRED_CLIPS = ("c647", "c339", "anime_morevna_closeup", "aigen_cogvideox_rain")
FOOTPRINT_FRACTIONS = (0.0008, 0.005, 0.01, 0.02, 0.05)

PHASE = "interocular_phase_orientation_burden_pct"
PHASE_OK = "interocular_phase_orientation_evidence_sufficient"
EXPOSURE = "interocular_exposure_rivalry_burden_pct"
COLOR = "interocular_color_gain_rivalry_burden_pct"
EXPOSURE_OK = "interocular_exposure_rivalry_evidence_sufficient"
COLOR_OK = "interocular_color_gain_rivalry_evidence_sufficient"
VERTICAL = "vmisalign_p99_pct"
VERTICAL_LIMIT_PCT = 0.10


def _sha256(path):
    return authenticated_sources.sample_sha256(path)


def _json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _frame_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"frame filename has no numeric identity: {path}") from exc


def _check(name, passed, **evidence):
    return {"name": name, "status": "pass" if passed else "fail", **evidence}


def _require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def _select_clip_ids(results, requested, max_clips):
    available = set(results.get("clips", {}))
    if requested:
        selected = requested
    else:
        selected = [clip for clip in PREFERRED_CLIPS if clip in available]
        selected.extend(sorted(available - set(selected)))
    selected = selected[:max_clips]
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"run has no requested clips: {missing}")
    if len(selected) < 2:
        raise ValueError("actual-output validation requires at least two real clips")
    return selected


def _select_frame(clip_result, clip_dir):
    frame_ids = [
        int(frame["frame_id"]) for frame in clip_result.get("frames", [])
        if isinstance(frame.get("frame_id"), int)
    ]
    if not frame_ids:
        raise ValueError(f"run result has no frame identities: {clip_dir}")
    # Interior evidence avoids start/end transients while remaining deterministic.
    frame_id = frame_ids[len(frame_ids) // 2]
    suffix = f"{frame_id:05d}"
    return frame_id, {
        "sbs": _require_file(os.path.join(clip_dir, f"sbs_{suffix}.png"), "SBS output"),
        "map": _require_file(
            os.path.join(clip_dir, f"warp_map_{suffix}.f32"), "exact warp map"),
        "mask": _require_file(
            os.path.join(clip_dir, f"warp_mask_{suffix}.png"), "warp-mask sidecar"),
    }


def _authenticate_sample(run_dir, results, clip_id, clips_root):
    clip_source_dir = os.path.join(clips_root, clip_id)
    clip_output_dir = os.path.join(run_dir, clip_id)
    if not os.path.isdir(clip_source_dir) or not os.path.isdir(clip_output_dir):
        raise FileNotFoundError(
            f"source/output directory missing for {clip_id}: "
            f"{clip_source_dir}, {clip_output_dir}")
    expected_clip_hash = results["meta"].get("clip_set_sha1", {}).get(clip_id)
    actual_clip_hash = run_eval.sha1_dir(clip_source_dir)
    if not expected_clip_hash or actual_clip_hash != expected_clip_hash:
        raise ValueError(
            f"{clip_id}: source authentication failed: "
            f"run={expected_clip_hash!r}, disk={actual_clip_hash!r}")

    contract_path = _require_file(
        os.path.join(clip_output_dir, "contract.json"), "harness contract")
    shape_path = _require_file(
        os.path.join(clip_output_dir, "warp_map_shape.json"), "warp-map shape contract")
    contract, shape = _json(contract_path), _json(shape_path)
    if contract.get("schema") != EXPECTED_HARNESS_SCHEMA:
        raise ValueError(
            f"{clip_id}: harness schema {contract.get('schema')} != {EXPECTED_HARNESS_SCHEMA}")
    mapping_contract = contract.get("warp_mapping", {})
    required_mapping = {
        "file_pattern": "warp_map_<frame-id>.f32",
        "shape_contract": "warp_map_shape.json",
        "dtype": "float32-le",
        "layout": "row-major",
    }
    mismatches = {
        key: (mapping_contract.get(key), value) for key, value in required_mapping.items()
        if mapping_contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{clip_id}: incompatible exact-map contract: {mismatches}")

    frame_id, artifacts = _select_frame(results["clips"][clip_id], clip_output_dir)
    source_frames = sbsbench.indexed_files(
        os.path.join(clip_source_dir, "frame_*"), "frame_")
    source_path = source_frames.get(frame_id)
    if source_path is None:
        raise FileNotFoundError(
            f"missing authenticated source frame {frame_id}: {clip_source_dir}")
    source_rgb = sbsbench.load_rgb(source_path)
    sbs_rgb = sbsbench.load_rgb(artifacts["sbs"])
    mapping = sbsbench.load_warp_mapping(artifacts["map"], shape)
    with Image.open(artifacts["mask"]) as mask_image:
        warp_mask = np.asarray(mask_image.convert("RGB"), dtype=np.float32) / 255.0
    if tuple(sbs_rgb.shape[:2]) != tuple(mapping.shape):
        raise ValueError(
            f"{clip_id}: SBS/map geometry differs: {sbs_rgb.shape[:2]} != {mapping.shape}")
    if (source_rgb.shape[1] != int(shape["source_width"])
            or source_rgb.shape[0] != int(shape["source_height"])):
        raise ValueError(
            f"{clip_id}: source/contract geometry differs: {source_rgb.shape[:2]} vs "
            f"{shape['source_height']}x{shape['source_width']}")
    left, right = sbsbench.split_eyes(sbs_rgb)
    map_left, map_right = sbsbench.split_eyes(mapping)
    return {
        "clip_id": clip_id,
        "frame_id": frame_id,
        "source_rgb": source_rgb,
        "left": left,
        "right": right,
        "map_left": map_left,
        "map_right": map_right,
        "warp_mask": warp_mask,
        "shape": shape,
        "paths": {
            "source": os.path.abspath(source_path),
            "sbs": os.path.abspath(artifacts["sbs"]),
            "warp_map": os.path.abspath(artifacts["map"]),
            "warp_mask": os.path.abspath(artifacts["mask"]),
            "contract": os.path.abspath(contract_path),
            "warp_map_shape": os.path.abspath(shape_path),
        },
        "sha256": {
            "source": _sha256(source_path),
            "sbs": _sha256(artifacts["sbs"]),
            "warp_map": _sha256(artifacts["map"]),
            "warp_mask": _sha256(artifacts["mask"]),
            "contract": _sha256(contract_path),
            "warp_map_shape": _sha256(shape_path),
        },
        "clip_sha1": actual_clip_hash,
    }


def _box_sum(values, height, width):
    padded = np.pad(values, ((1, 0), (1, 0)), mode="constant")
    integral = padded.cumsum(0).cumsum(1)
    return (integral[height:, width:] - integral[:-height, width:]
            - integral[height:, :-width] + integral[:-height, :-width])


def _patch_dimensions(total_pixels, fraction, max_height, max_width):
    area = max(4, int(round(total_pixels * fraction)))
    width = max(2, int(round(np.sqrt(area * 4.0))))
    height = max(2, int(np.ceil(area / width)))
    return min(height, max_height), min(width, max_width)


def _nested_fault_masks(expected_luma, valid):
    """Choose nested thin high-detail regions with approximately exact picture footprints."""
    height, width = expected_luma.shape
    max_h, max_w = _patch_dimensions(height * width, FOOTPRINT_FRACTIONS[-1], height, width)
    gx = np.diff(expected_luma, axis=1, prepend=expected_luma[:, :1])
    gy = np.diff(expected_luma, axis=0, prepend=expected_luma[:1, :])
    energy = np.hypot(gx, gy) + 0.05 * expected_luma
    safe = valid.copy()
    margin_y, margin_x = max_h // 2 + 4, max_w // 2 + 4
    safe[:margin_y] = False
    safe[-margin_y:] = False
    safe[:, :margin_x] = False
    safe[:, -margin_x:] = False
    score = _box_sum(np.where(safe, energy, -1.0), max_h, max_w)
    valid_count = _box_sum(safe.astype(np.float32), max_h, max_w)
    score[valid_count < max_h * max_w] = -np.inf
    if not np.isfinite(score).any():
        raise ValueError("no valid interior region can host the largest corruption footprint")
    top, left = np.unravel_index(np.argmax(score), score.shape)
    center_y, center_x = top + max_h // 2, left + max_w // 2

    masks = []
    for fraction in FOOTPRINT_FRACTIONS:
        patch_h, patch_w = _patch_dimensions(height * width, fraction, height, width)
        y0 = int(np.clip(center_y - patch_h // 2, 0, height - patch_h))
        x0 = int(np.clip(center_x - patch_w // 2, 0, width - patch_w))
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y0 + patch_h, x0:x0 + patch_w] = True
        if not np.all(valid[mask]):
            raise ValueError("selected corruption footprint extends outside exact-map support")
        masks.append(mask)
    if any(np.any(before & ~after) for before, after in zip(masks, masks[1:])):
        raise AssertionError("corruption footprint ladder is not nested")
    return masks


def _shift_edge(image, dy=0, dx=0):
    out = np.empty_like(image)
    y_indices = np.clip(np.arange(image.shape[0]) - dy, 0, image.shape[0] - 1)
    x_indices = np.clip(np.arange(image.shape[1]) - dx, 0, image.shape[1] - 1)
    out[:] = image[y_indices[:, None], x_indices[None, :]]
    return out


def _linear_gain(image, gains):
    """Apply an RGB gain in linear light and return encoded sRGB evidence."""
    linear = photometric_rivalry._linearize_srgb(image)
    linear *= np.asarray(gains, dtype=np.float32)
    return np.where(
        linear <= 0.0031308, 12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _smooth_channels(image, radius):
    """Deterministic reflect-padded box blur used only to synthesize validator faults."""
    image = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return image.copy()
    channels = []
    diameter = 2 * radius + 1
    for channel in range(image.shape[2]):
        padded = np.pad(image[..., channel], radius, mode="reflect")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        channels.append((
            integral[diameter:, diameter:] - integral[:-diameter, diameter:] -
            integral[diameter:, :-diameter] + integral[:-diameter, :-diameter]
        ) / float(diameter * diameter))
    return np.stack(channels, axis=2).astype(np.float32)


def _corrupt(right, mask, kind):
    out = right.copy()
    if kind == "missing_black":
        candidate = np.zeros_like(right)
    elif kind == "blur":
        candidate = _smooth_channels(right, 4)
    elif kind == "ringing_oversharpen":
        smooth = _smooth_channels(right, 2)
        candidate = np.clip(right + 3.0 * (right - smooth), 0.0, 1.0)
    elif kind == "vertical_shift":
        candidate = _shift_edge(right, dy=3)
    elif kind == "phase_shift":
        candidate = _shift_edge(right, dx=4)
    elif kind == "exposure_conflict":
        candidate = _linear_gain(right, (1.35, 1.35, 1.35))
    elif kind == "hue_conflict":
        candidate = _linear_gain(right, (1.30, 0.80, 1.12))
    else:
        raise ValueError(f"unknown corruption: {kind}")
    out[mask] = candidate[mask]
    return out


def _source_metrics(sample, right):
    source_rgb = sample["source_rgb"]
    source_luma = sbsbench.rgb_luma(source_rgb)
    left_metrics = sbsbench.exact_source_relative_metrics(
        sbsbench.rgb_luma(sample["left"]), source_luma, sample["map_left"], sample["shape"],
        eye_rgb=sample["left"], src_rgb=source_rgb)
    right_metrics = sbsbench.exact_source_relative_metrics(
        sbsbench.rgb_luma(right), source_luma, sample["map_right"], sample["shape"],
        eye_rgb=right, src_rgb=source_rgb)
    return {
        "source_coverage_pct": min(
            left_metrics["source_coverage_pct"], right_metrics["source_coverage_pct"]),
        "image_integrity_pct": min(
            left_metrics["image_integrity_pct"], right_metrics["image_integrity_pct"]),
        "source_coverage_worst_patch_bad_pct": max(
            left_metrics["source_coverage_worst_patch_bad_pct"],
            right_metrics["source_coverage_worst_patch_bad_pct"]),
        "image_integrity_worst_patch_bad_pct": max(
            left_metrics["image_integrity_worst_patch_bad_pct"],
            right_metrics["image_integrity_worst_patch_bad_pct"]),
        "source_residual_p95": max(
            left_metrics["source_residual_p95"], right_metrics["source_residual_p95"]),
        "source_fidelity_support_pct": min(
            left_metrics["source_fidelity_support_pct"],
            right_metrics["source_fidelity_support_pct"]),
        "image_integrity_support": min(
            left_metrics["image_integrity_support"], right_metrics["image_integrity_support"]),
    }


def _vertical_metrics(sample, right):
    value, normalized, support = sbsbench.exact_vertical_misalignment(
        sbsbench.rgb_luma(sample["left"]), sbsbench.rgb_luma(right),
        sbsbench.rgb_luma(sample["source_rgb"]), sample["map_left"], sample["map_right"],
        sample["shape"], src_rgb=sample["source_rgb"])
    return {
        "vmisalign_p99_px": value,
        VERTICAL: normalized,
        "vmisalign_support_pct": support,
    }


def _phase_metrics(sample, right, return_maps=False):
    measured = phase_chroma.measure_interocular_phase_chroma(
        sample["source_rgb"], sample["left"], right, sample["map_left"],
        sample["map_right"], sample["shape"], warp_mask=sample["warp_mask"],
        return_maps=return_maps)
    metrics = measured[0] if return_maps else measured
    # Fail immediately if the validator and canonical metric contract drift apart.  Accepting the
    # retired percentile names here would let a stale report appear healthy.
    required = {PHASE, PHASE_OK}
    missing = sorted(required - set(metrics))
    stale = sorted({
        "interocular_phase_orientation_p95_pct",
        "interocular_chroma_conflict_p95_pct",
    } & set(metrics))
    if missing or stale:
        raise ValueError(
            f"phase/orientation metric-name contract drift: missing={missing}, stale={stale}")
    return measured


def _photometric_metrics(sample, right, *, left=None, return_maps=False):
    measured = photometric_rivalry.measure_interocular_photometric_rivalry(
        sample["source_rgb"], sample["left"] if left is None else left, right,
        sample["map_left"], sample["map_right"], sample["shape"],
        warp_mask=sample["warp_mask"], return_maps=return_maps)
    metrics = measured[0] if return_maps else measured
    required = {EXPOSURE, COLOR, EXPOSURE_OK, COLOR_OK}
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"photometric-rivalry metric-name contract drift: missing={missing}")
    return measured


def _source_localization(sample, right, patch, kind):
    source_luma = sbsbench.rgb_luma(sample["source_rgb"])
    expected, _, valid = sbsbench._exact_expected_source_luma(
        right.shape[:2], source_luma, sample["map_right"], sample["shape"],
        src_rgb=sample["source_rgb"])

    def maps(eye):
        eye_luma = sbsbench.rgb_luma(eye)
        residual = np.abs(eye_luma - expected) > 4.0 / 255.0
        # Use the evaluator's detector map verbatim.  A separately transcribed validator once
        # drifted from the scalar metric and could pass while the production implementation failed.
        integrity_bad, textured = sbsbench.exact_image_integrity_maps(
            eye_luma, expected, valid)
        return residual & valid, integrity_bad, textured

    clean_coverage, clean_integrity, textured = maps(sample["right"])
    bad_coverage, bad_integrity, _ = maps(right)
    active = ((bad_coverage & ~clean_coverage) if kind == "coverage"
              else (bad_integrity & ~clean_integrity))
    eligible = valid if kind == "coverage" else textured
    halo = sbsbench.dilate2d(patch, 3)
    active_count = int(np.count_nonzero(active))
    target_count = int(np.count_nonzero(patch & eligible))
    precision = float(np.count_nonzero(active & halo) / max(active_count, 1))
    recall = float(np.count_nonzero(active & patch & eligible) / max(target_count, 1))
    return {
        "precision": precision,
        "recall": recall,
        "active_pixels": active_count,
        "eligible_target_pixels": target_count,
    }


def _registered_patch(sample, patch, analysis_shape):
    height, width = analysis_shape
    values, valid = phase_chroma._register_rgb(
        patch.astype(np.float32)[..., None], sample["map_right"], sample["shape"], width, height)
    return (values[..., 0] >= 0.25) & valid


def _phase_localization(sample, patch, clean_maps, bad_maps):
    key = "phase_orientation_conflict_pct"
    clean = np.nan_to_num(clean_maps[key], nan=0.0)
    bad = np.nan_to_num(bad_maps[key], nan=0.0)
    active = bad >= np.maximum(clean + 5.0, 8.0)
    support = np.asarray(bad_maps["phase_orientation_support"], dtype=bool)
    target = _registered_patch(sample, patch, active.shape) & support
    halo = sbsbench.dilate2d(target, max(3, int(round(active.shape[1] * 0.01))))
    active_count = int(np.count_nonzero(active))
    precision = float(np.count_nonzero(active & halo) / max(active_count, 1))
    recall = float(np.count_nonzero(active & target) / max(np.count_nonzero(target), 1))
    return {"precision": precision, "recall": recall, "active_pixels": active_count,
            "eligible_target_pixels": int(np.count_nonzero(target))}


def _photometric_localization(sample, patch, clean_maps, bad_maps, kind):
    key = "exposure_rivalry_pct" if kind == "exposure" else "color_gain_rivalry_pct"
    clean = np.nan_to_num(clean_maps[key], nan=0.0)
    bad = np.nan_to_num(bad_maps[key], nan=0.0)
    active = bad >= np.maximum(clean + 2.0, 3.0)
    support = np.asarray(bad_maps["photometric_support"], dtype=bool)
    target = _registered_patch(sample, patch, active.shape) & support
    halo = sbsbench.dilate2d(target, max(3, int(round(active.shape[1] * 0.01))))
    active_count = int(np.count_nonzero(active))
    precision = float(np.count_nonzero(active & halo) / max(active_count, 1))
    recall = float(np.count_nonzero(active & target) / max(np.count_nonzero(target), 1))
    return {"precision": precision, "recall": recall, "active_pixels": active_count,
            "eligible_target_pixels": int(np.count_nonzero(target))}


def _ladder_check(name, baseline, values, *, direction, min_final, tolerance):
    finite = all(value is not None and np.isfinite(value) for value in values)
    if direction == "lower":
        responses = [baseline - value for value in values] if finite else []
    elif direction == "higher":
        responses = [value - baseline for value in values] if finite else []
    else:
        raise ValueError(f"invalid ladder direction: {direction}")
    monotonic = finite and all(
        after + tolerance >= before for before, after in zip(responses, responses[1:]))
    responsive = finite and responses[-1] >= min_final
    return _check(
        name, bool(finite and monotonic and responsive), baseline=baseline, values=values,
        responses=responses, monotonic=bool(monotonic), responsive=bool(responsive),
        required_final_response=min_final, tolerance=tolerance)


def validate_sample(sample):
    source_luma = sbsbench.rgb_luma(sample["source_rgb"])
    expected, _, valid = sbsbench._exact_expected_source_luma(
        sample["right"].shape[:2], source_luma, sample["map_right"], sample["shape"],
        src_rgb=sample["source_rgb"])
    masks = _nested_fault_masks(expected, valid)
    clean_source = _source_metrics(sample, sample["right"])
    clean_vertical = _vertical_metrics(sample, sample["right"])
    clean_phase, clean_phase_maps = _phase_metrics(sample, sample["right"], return_maps=True)
    clean_photometric, clean_photometric_maps = _photometric_metrics(
        sample, sample["right"], return_maps=True)
    if clean_phase[PHASE_OK] != 100.0:
        raise ValueError(
            f"{sample['clip_id']}: clean phase evidence is insufficient")
    if (clean_photometric[EXPOSURE_OK] != 100.0
            or clean_photometric[COLOR_OK] != 100.0):
        raise ValueError(
            f"{sample['clip_id']}: clean photometric evidence is insufficient")

    ladders = {}
    localizations = {}
    for kind in (
            "missing_black", "blur", "ringing_oversharpen", "vertical_shift",
            "phase_shift", "exposure_conflict", "hue_conflict"):
        records = []
        for fraction, mask in zip(FOOTPRINT_FRACTIONS, masks):
            corrupted = _corrupt(sample["right"], mask, kind)
            record = {"requested_footprint_pct": fraction * 100.0,
                      "actual_footprint_pct": float(mask.mean() * 100.0)}
            if kind in ("missing_black", "blur", "ringing_oversharpen"):
                record["metrics"] = _source_metrics(sample, corrupted)
            elif kind == "vertical_shift":
                record["metrics"] = _vertical_metrics(sample, corrupted)
            elif kind == "phase_shift":
                result = _phase_metrics(
                    sample, corrupted, return_maps=abs(fraction - 0.01) < 1e-9)
                if isinstance(result, tuple):
                    record["metrics"], evidence = result
                    localizations[kind] = _phase_localization(
                        sample, mask, clean_phase_maps, evidence)
                else:
                    record["metrics"] = result
            else:
                result = _photometric_metrics(
                    sample, corrupted, return_maps=abs(fraction - 0.01) < 1e-9)
                if isinstance(result, tuple):
                    record["metrics"], evidence = result
                    localizations[kind] = _photometric_localization(
                        sample, mask, clean_photometric_maps, evidence,
                        "exposure" if kind == "exposure_conflict" else "color")
                else:
                    record["metrics"] = result
            if abs(fraction - 0.01) < 1e-9 and kind in (
                    "missing_black", "blur", "ringing_oversharpen"):
                localizations[kind] = _source_localization(
                    sample, corrupted, mask,
                    "coverage" if kind == "missing_black" else "integrity")
            records.append(record)
        ladders[kind] = records

    no_op_source = _source_metrics(sample, sample["right"].copy())
    no_op_vertical = _vertical_metrics(sample, sample["right"].copy())
    no_op_phase = _phase_metrics(sample, sample["right"].copy())
    no_op_photometric = _photometric_metrics(sample, sample["right"].copy())
    # A common, reversible photometric transform is a benign binocular-conflict control.  It is
    # intentionally not asserted benign for source conformance, where changing the rendered color
    # would correctly be a residual.
    common_left = _linear_gain(sample["left"], (0.92, 0.95, 0.90))
    common_right = _linear_gain(sample["right"], (0.92, 0.95, 0.90))
    common_sample = {**sample, "left": common_left}
    common_phase = _phase_metrics(common_sample, common_right)
    common_photometric = _photometric_metrics(
        sample, common_right, left=common_left)
    global_exposure = _photometric_metrics(
        sample, _linear_gain(sample["right"], (1.18, 1.18, 1.18)))
    global_color = _photometric_metrics(
        sample, _linear_gain(sample["right"], (1.16, 0.93, 0.84)))

    def metric_values(kind, key):
        return [record["metrics"].get(key) for record in ladders[kind]]
    vertical_values = metric_values("vertical_shift", VERTICAL)
    phase_values = metric_values("phase_shift", PHASE)
    exposure_values = metric_values("exposure_conflict", EXPOSURE)
    color_values = metric_values("hue_conflict", COLOR)
    checks = [
        _check(
            "clean_exact_support", clean_source["source_fidelity_support_pct"] >= 95.0
            and clean_source["image_integrity_support"] >= 0.1,
            metrics=clean_source),
        _check(
            "clean_phase_support", clean_phase[PHASE_OK] == 100.0,
            metrics=clean_phase),
        _check(
            "clean_photometric_support",
            clean_photometric[EXPOSURE_OK] == 100.0
            and clean_photometric[COLOR_OK] == 100.0,
            metrics=clean_photometric),
        _check(
            "clean_vertical_support", clean_vertical[VERTICAL] is not None
            and clean_vertical["vmisalign_support_pct"] >= 2.0, metrics=clean_vertical),
        _check(
            "clean_vertical_below_candidate_limit",
            clean_vertical[VERTICAL] is not None
            and clean_vertical[VERTICAL] < VERTICAL_LIMIT_PCT,
            candidate_limit_pct=VERTICAL_LIMIT_PCT, metrics=clean_vertical),
        _check(
            "no_op_exact", all(abs(no_op_source[key] - clean_source[key]) <= 1e-9 for key in (
                "source_coverage_pct", "image_integrity_pct", "source_residual_p95",
                "source_coverage_worst_patch_bad_pct",
                "image_integrity_worst_patch_bad_pct")),
            clean=clean_source, no_op=no_op_source),
        _check(
            "no_op_vertical", no_op_vertical == clean_vertical,
            clean=clean_vertical, no_op=no_op_vertical),
        _check(
            "no_op_phase", all(
                no_op_phase[key] == clean_phase[key] for key in (PHASE, PHASE_OK)),
            clean=clean_phase, no_op=no_op_phase),
        _check(
            "no_op_photometric", all(
                no_op_photometric[key] == clean_photometric[key]
                for key in (EXPOSURE, COLOR, EXPOSURE_OK, COLOR_OK)),
            clean=clean_photometric, no_op=no_op_photometric),
        _check(
            "common_photometric_control",
            common_phase[PHASE_OK] == 100.0
            and common_phase[PHASE] <= clean_phase[PHASE] + 0.5
            and common_photometric[EXPOSURE_OK] == 100.0
            and common_photometric[COLOR_OK] == 100.0
            and common_photometric[EXPOSURE] <= clean_photometric[EXPOSURE] + 0.5
            and common_photometric[COLOR] <= clean_photometric[COLOR] + 0.5,
            clean={PHASE: clean_phase[PHASE], EXPOSURE: clean_photometric[EXPOSURE],
                   COLOR: clean_photometric[COLOR]},
            control={PHASE: common_phase[PHASE],
                     EXPOSURE: common_photometric[EXPOSURE],
                     COLOR: common_photometric[COLOR]}),
        _check(
            "global_unilateral_exposure_detection",
            global_exposure[EXPOSURE] >= clean_photometric[EXPOSURE] + 5.0,
            clean=clean_photometric[EXPOSURE], measured=global_exposure[EXPOSURE]),
        _check(
            "global_unilateral_color_gain_detection",
            global_color[COLOR] >= clean_photometric[COLOR] + 5.0,
            clean=clean_photometric[COLOR], measured=global_color[COLOR]),
        _ladder_check(
            "missing_coverage_ladder", clean_source["source_coverage_pct"],
            metric_values("missing_black", "source_coverage_pct"), direction="lower",
            min_final=3.5, tolerance=0.10),
        _ladder_check(
            "blur_integrity_ladder", clean_source["image_integrity_pct"],
            metric_values("blur", "image_integrity_pct"), direction="lower",
            min_final=0.5, tolerance=0.15),
        _ladder_check(
            "ringing_integrity_ladder", clean_source["image_integrity_pct"],
            metric_values("ringing_oversharpen", "image_integrity_pct"), direction="lower",
            min_final=1.0, tolerance=0.15),
        _ladder_check(
            "missing_local_patch_ladder",
            clean_source["source_coverage_worst_patch_bad_pct"],
            metric_values("missing_black", "source_coverage_worst_patch_bad_pct"),
            direction="higher", min_final=50.0, tolerance=2.0),
        _ladder_check(
            "blur_local_patch_ladder",
            clean_source["image_integrity_worst_patch_bad_pct"],
            metric_values("blur", "image_integrity_worst_patch_bad_pct"),
            direction="higher", min_final=40.0, tolerance=2.0),
        _ladder_check(
            "ringing_local_patch_ladder",
            clean_source["image_integrity_worst_patch_bad_pct"],
            metric_values("ringing_oversharpen", "image_integrity_worst_patch_bad_pct"),
            direction="higher", min_final=40.0, tolerance=2.0),
        _ladder_check(
            "vertical_shift_ladder", clean_vertical[VERTICAL], vertical_values,
            direction="higher", min_final=0.10, tolerance=0.015),
        _check(
            "vertical_shift_one_percent_detection",
            vertical_values[2] is not None
            and vertical_values[2] >= VERTICAL_LIMIT_PCT,
            candidate_limit_pct=VERTICAL_LIMIT_PCT,
            requested_footprint_pct=FOOTPRINT_FRACTIONS[2] * 100.0,
            clean=clean_vertical[VERTICAL], measured=vertical_values[2]),
        _ladder_check(
            "phase_conflict_ladder", clean_phase[PHASE], phase_values,
            direction="higher", min_final=0.5, tolerance=0.10),
        _ladder_check(
            "exposure_conflict_ladder", clean_photometric[EXPOSURE], exposure_values,
            direction="higher", min_final=0.5, tolerance=0.10),
        _ladder_check(
            "color_gain_conflict_ladder", clean_photometric[COLOR], color_values,
            direction="higher", min_final=0.5, tolerance=0.10),
    ]
    localization_floors = {
        "missing_black": (0.90, 0.70),
        "blur": (0.60, 0.20),
        "ringing_oversharpen": (0.60, 0.20),
        "phase_shift": (0.60, 0.15),
        "exposure_conflict": (0.60, 0.15),
        "hue_conflict": (0.60, 0.15),
    }
    for kind, (precision_floor, recall_floor) in localization_floors.items():
        evidence = localizations.get(kind, {})
        checks.append(_check(
            f"{kind}_one_percent_localization",
            evidence.get("active_pixels", 0) > 0
            and evidence.get("precision", 0.0) >= precision_floor
            and evidence.get("recall", 0.0) >= recall_floor,
            minimum_precision=precision_floor, minimum_recall=recall_floor, **evidence))

    return {
        "clip_id": sample["clip_id"],
        "frame_id": sample["frame_id"],
        "geometry": {
            "eye_width": sample["right"].shape[1],
            "eye_height": sample["right"].shape[0],
        },
        "artifact_paths": sample["paths"],
        "artifact_sha256": sample["sha256"],
        "clip_sha1": sample["clip_sha1"],
        "footprint_ladder_pct": [float(mask.mean() * 100.0) for mask in masks],
        "baseline": {
            "source": clean_source,
            "vertical": clean_vertical,
            "phase": clean_phase,
            "photometric": clean_photometric,
        },
        "controls": {
            "no_op_source": no_op_source,
            "no_op_vertical": no_op_vertical,
            "no_op_phase": no_op_phase,
            "no_op_photometric": no_op_photometric,
            "common_photometric_phase": common_phase,
            "common_photometric_rivalry": common_photometric,
            "global_unilateral_exposure": global_exposure,
            "global_unilateral_color_gain": global_color,
        },
        "ladders": ladders,
        "localization": localizations,
        "checks": checks,
        "passed": all(check["status"] == "pass" for check in checks),
    }


def build_report(run_dir, requested_clips=None, max_clips=2):
    run_dir = os.path.abspath(run_dir)
    results_path = _require_file(os.path.join(run_dir, "results.json"), "eval results")
    results = _json(results_path)
    meta = results.get("meta", {})
    if meta.get("eval_schema") != EXPECTED_EVAL_SCHEMA:
        raise ValueError(
            f"eval schema {meta.get('eval_schema')} != required {EXPECTED_EVAL_SCHEMA}")
    if meta.get("suite") != "core":
        raise ValueError(f"default actual-output validator requires a core run, got {meta.get('suite')}")
    clips_root = os.path.abspath(meta.get("clips_root", ""))
    if not os.path.isdir(clips_root):
        raise FileNotFoundError(f"authenticated clips_root is unavailable: {clips_root}")
    clip_ids = _select_clip_ids(results, requested_clips or [], max_clips)

    started = time.perf_counter()
    samples = []
    for clip_id in clip_ids:
        sample = _authenticate_sample(run_dir, results, clip_id, clips_root)
        samples.append(validate_sample(sample))
    checks = [check for sample in samples for check in sample["checks"]]
    failed = [
        {"clip_id": sample["clip_id"], **check}
        for sample in samples for check in sample["checks"] if check["status"] != "pass"
    ]
    return {
        "schema": SCHEMA,
        "validator": "actual-schema32-SBS-controlled-corruption-falsifier",
        "experimental": True,
        "training_label_qualification": "blocked",
        "eligible_training_labels": [],
        "auto_promotes_thresholds": False,
        "run": {
            "directory": run_dir,
            "results": results_path,
            "results_sha256": _sha256(results_path),
            "eval_schema": meta["eval_schema"],
            "run_kind": meta.get("run_kind"),
            "git_sha": meta.get("git_sha"),
            "metric_sha256": meta.get("metric_sha256"),
        },
        "configuration": {
            "clips": clip_ids,
            "footprint_fractions": list(FOOTPRINT_FRACTIONS),
            "phase_metric": PHASE,
            "exposure_metric": EXPOSURE,
            "color_metric": COLOR,
            "vertical_metric": VERTICAL,
            "vertical_quantile": sbsbench.VERTICAL_MISALIGNMENT_QUANTILE,
            "vertical_candidate_limit_pct": VERTICAL_LIMIT_PCT,
        },
        "implementation": {
            "validator_sha256": _sha256(__file__),
            "sbsbench_sha256": _sha256(sbsbench.__file__),
            "phase_sha256": _sha256(phase_chroma.__file__),
            "photometric_rivalry_sha256": _sha256(photometric_rivalry.__file__),
            "source_authenticator_sha256": _sha256(authenticated_sources.__file__),
            "numpy_version": np.__version__,
        },
        "summary": {
            "clips": len(samples),
            "checks": len(checks),
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "overall_pass": not failed,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "failures": failed,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", default=DEFAULT_RUN,
        help="completed schema-32 core eval directory containing actual harness artifacts")
    parser.add_argument(
        "--clip", action="append", default=[],
        help="clip id to validate; repeat for multiple clips (default: two preferred real clips)")
    parser.add_argument("--max-clips", type=int, default=2)
    parser.add_argument("--output", help="JSON report path (default: <run>/actual-metric-validator.json)")
    args = parser.parse_args()
    if args.max_clips < 2:
        parser.error("--max-clips must be at least 2")
    try:
        report = build_report(args.run, args.clip, args.max_clips)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"actual SBS metric validation failed closed: {exc}", file=sys.stderr)
        return 2
    output = args.output or os.path.join(os.path.abspath(args.run), "actual-metric-validator.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0 if report["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
