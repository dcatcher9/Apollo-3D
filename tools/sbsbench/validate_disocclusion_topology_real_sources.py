#!/usr/bin/env python3
"""Falsify topology metrics with controlled corruptions of authenticated real source frames.

This validator complements the compact synthetic unit tests.  It uses committed core frames and
prepared public-suite frames under the shared authenticated provenance contract,
but keeps geometry deterministic: a strong real material edge is selected, a synthetic high-near
depth step and exact-map forward-hole mask are placed there, and only the final eye is corrupted.

Passing is necessary, never sufficient for a gate or training label.  The generated topology is
not ground truth for the original scene, HDR is a simulated linear-light/post-sample contract, and
no controlled corruption replaces human inspection of actual renderer failures.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_disocclusion_topology_metrics as topology  # noqa: E402
import authenticated_metric_sources as real_suite  # noqa: E402


SCHEMA = 1
PREFERRED_CORE = (
    "aigen_cogvideox_rain",
    "anime_morevna_closeup",
    "c647",
    "c841",
)
PREFERRED_EXTENDED = (
    "bonn_person_walk",
    "sintel_market",
    "spring_character_close",
    "vkitti_drive_rain",
)

VARIANTS = {
    "identity_sdr": {
        "geometry": "identity", "regime": "native_sdr",
        "scale_x": 1.0, "scale_y": 1.0, "fractional_shift_px": 0.0,
    },
    "pillarbox_sdr": {
        "geometry": "pillarbox", "regime": "native_sdr",
        "scale_x": 0.80, "scale_y": 1.0, "fractional_shift_px": 0.0,
    },
    "letterbox_sdr": {
        "geometry": "letterbox", "regime": "native_sdr",
        "scale_x": 1.0, "scale_y": 0.75, "fractional_shift_px": 0.0,
    },
    "fractional_hdr": {
        "geometry": "fractional", "regime": "simulated_linear_hdr",
        "scale_x": 1.0, "scale_y": 1.0, "fractional_shift_px": 0.43,
        "hdr_scale": 6.0,
    },
}

LEAK_LEVELS = (0.0, 0.25, 0.50, 1.0)
BACKGROUND_SHIFTS = (0.0, 0.01, 0.03, 0.06)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.basename(path).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _tonemap(linear):
    linear = np.maximum(np.asarray(linear, dtype=np.float32), 0.0)
    return linear / (1.0 + linear)


def _sample(image, u, v, transform=None):
    value = topology._sample_uv(image, u, v)
    if transform is not None:
        value = np.asarray(transform(value), dtype=np.float32)
    return value


def _find_real_edge(source):
    """Select a deterministic, vertically supported real horizontal material transition."""
    height, width = source.shape[:2]
    prototype = max(1, int(round(width * 0.00625)))
    contrast = np.zeros((height, width - 1), dtype=np.float32)
    valid_x = np.arange(prototype, width - 1 - prototype)
    if not valid_x.size:
        raise ValueError(f"source is too narrow for a topology edge: {source.shape}")
    left = source[:, valid_x - prototype]
    right = source[:, valid_x + 1 + prototype]
    contrast[:, valid_x] = np.sqrt(np.mean((right - left) ** 2, axis=2))

    values = source.reshape(-1, 3)
    rgb_range = np.mean(
        np.quantile(values, 0.99, axis=0) - np.quantile(values, 0.01, axis=0))
    dynamic = max(float(rgb_range), 4.0 / 255.0)
    contrast_pct = contrast / dynamic * 100.0

    patch_height = max(16, int(round(height * 0.18)))
    patch_height = min(patch_height, max(4, height - 2))
    integral = np.pad(contrast_pct, ((1, 0), (0, 0)), mode="constant").cumsum(0)
    window = integral[patch_height:] - integral[:-patch_height]
    window /= float(patch_height)
    x0, x1 = max(prototype + 1, int(width * 0.18)), min(
        width - prototype - 2, int(width * 0.76))
    y0_limit, y1_limit = int(height * 0.08), int(height * 0.92) - patch_height
    if x1 <= x0 or y1_limit < y0_limit:
        raise ValueError(f"source has no central edge search region: {source.shape}")
    search = window[y0_limit:y1_limit + 1, x0:x1]
    best = int(np.argmax(search))
    rel_y, rel_x = np.unravel_index(best, search.shape)
    start_y, edge_x = y0_limit + int(rel_y), x0 + int(rel_x)
    return {
        "edge_x": edge_x,
        "y0": start_y,
        "y1": start_y + patch_height,
        "mean_contrast_pct": float(window[start_y, edge_x]),
        "peak_contrast_pct": float(np.max(
            contrast_pct[start_y:start_y + patch_height, edge_x])),
        "prototype_offset_px": prototype,
    }


def _shape(width, height, scale_x, scale_y):
    return {
        "schema": 1,
        "width": 2 * width,
        "height": height,
        "eye_width": width,
        "eye_height": height,
        "source_width": width,
        "source_height": height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
    }


def _make_case(source_sdr, edge, variant):
    height, width = source_sdr.shape[:2]
    scale_x, scale_y = variant["scale_x"], variant["scale_y"]
    shape = _shape(width, height, scale_x, scale_y)
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / width
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / height
    lo_x, lo_y = 0.5 * (1.0 - scale_x), 0.5 * (1.0 - scale_y)
    identity_u = (output_u - lo_x) / scale_x
    source_v = (output_v - lo_y) / scale_y
    content = ((output_u[None, :] >= lo_x) &
               (output_u[None, :] <= lo_x + scale_x) &
               (output_v[:, None] >= lo_y) &
               (output_v[:, None] <= lo_y + scale_y))

    if variant["regime"] == "simulated_linear_hdr":
        source = np.power(np.clip(source_sdr, 0.0, 1.0), 2.2) * variant["hdr_scale"]
        sample_transform = _tonemap
    else:
        source = source_sdr
        sample_transform = None

    shift_u = variant["fractional_shift_px"] / max(scale_x * width, 1.0)
    eye_map = np.broadcast_to(
        (identity_u + shift_u)[None, :], (height, width)).copy()
    source_v_grid = np.broadcast_to(
        np.clip(source_v, 0.0, 1.0)[:, None], eye_map.shape)
    clean = _sample(source, np.clip(eye_map, 0.0, 1.0), source_v_grid,
                    sample_transform)
    clean[~content] = 0.0

    depth = np.full((height, width), 0.10, np.float32)
    depth[edge["y0"]:edge["y1"], :edge["edge_x"] + 1] = 0.90
    edge_u = (edge["edge_x"] + 0.5) / width
    edge_output_u = lo_x + scale_x * (edge_u - shift_u)
    edge_output_x = int(np.clip(round(edge_output_u * width - 0.5), 1, width - 2))
    y0 = int(np.clip(round((lo_y + scale_y * edge["y0"] / height) * height),
                     0, height - 1))
    y1 = int(np.clip(round((lo_y + scale_y * edge["y1"] / height) * height),
                     y0 + 1, height))
    hole_width = max(3, int(round(scale_x * width * 0.01875)))
    hole = np.zeros((height, width), dtype=bool)
    hole[y0:y1, edge_output_x:min(width, edge_output_x + hole_width)] = True
    hole &= content
    mask = np.zeros((height, 2 * width, 3), np.uint8)
    mask[:, :width, 0][hole] = 255
    mask[:, width:, 0][hole] = 255

    foreground_u = np.full(
        eye_map.shape,
        max(0.0, edge_u - edge["prototype_offset_px"] / width), np.float32)
    foreground = _sample(source, foreground_u, source_v_grid, sample_transform)
    mapping = np.concatenate((eye_map, eye_map), axis=1)
    return {
        "source": source.astype(np.float32),
        "source_display": (_tonemap(source) if sample_transform is not None else source_sdr),
        "sample_transform": sample_transform,
        "clean": clean.astype(np.float32),
        "foreground": foreground.astype(np.float32),
        "mapping": mapping.astype(np.float32),
        "mask": mask,
        "depth": depth,
        "shape": shape,
        "content": content,
        "hole": hole,
        "edge_u": edge_u,
        "edge_output_x": edge_output_x,
        "source_v_grid": source_v_grid,
    }


def _measure(case, left, right=None, mapping=None, mask=None, return_maps=False):
    right = left if right is None else right
    return topology.measure_disocclusion_topology(
        case["source"], left, right,
        case["mapping"] if mapping is None else mapping,
        case["mask"] if mask is None else mask,
        case["depth"], case["shape"],
        source_sample_transform=case["sample_transform"],
        min_supported_hole_pixels=4, min_foreground_support_pixels=8,
        return_maps=return_maps)


def _background_fill(case, shift):
    width = case["shape"]["eye_width"]
    mapping = case["mapping"].copy()
    left = case["clean"].copy()
    target_u = np.clip(case["edge_u"] + max(float(shift), 1.0 / width), 0.0, 1.0)
    target_grid = np.full(case["hole"].shape, target_u, np.float32)
    replacement = _sample(
        case["source"], target_grid, case["source_v_grid"], case["sample_transform"])
    left[case["hole"]] = replacement[case["hole"]]
    for eye_index in range(2):
        eye_map = mapping[:, eye_index * width:(eye_index + 1) * width]
        eye_map[case["hole"]] = target_u
    return left, mapping


def _foreground_leak(case, strength, split=False):
    eye, mapping = _background_fill(case, 0.0)
    hole = case["hole"].copy()
    if split:
        hole[1::2] = False
    eye[hole] = ((1.0 - strength) * eye[hole]
                 + strength * case["foreground"][hole])
    mask = case["mask"].copy()
    if split:
        width = case["shape"]["eye_width"]
        packed_hole = np.concatenate((hole, hole), axis=1)
        mask[..., 0] = np.where(packed_hole, 255, 0).astype(np.uint8)
        if width * 2 != packed_hole.shape[1]:
            raise AssertionError("split mask width contract failed")
    return eye, mask, mapping


def _unrelated_fault(case):
    eye, mapping = _background_fill(case, 0.0)
    hole = case["hole"]
    _, maps = _measure(case, eye, mapping=mapping, return_maps=True)
    width = case["shape"]["eye_width"]
    background = maps["expected_background_reference"][:, :width]
    foreground = maps["foreground_reference"][:, :width]
    axes = np.eye(3, dtype=np.float32)
    for y, x in zip(*np.nonzero(hole)):
        direction = foreground[y, x] - background[y, x]
        best = None
        for axis in axes:
            candidate = np.cross(direction, axis)
            norm = float(np.linalg.norm(candidate))
            if norm <= 1e-6:
                continue
            candidate /= norm
            for signed in (candidate, -candidate):
                limits = []
                for channel, component in enumerate(signed):
                    if component > 1e-7:
                        limits.append((1.0 - eye[y, x, channel]) / component)
                    elif component < -1e-7:
                        limits.append(-eye[y, x, channel] / component)
                safe = min(limits, default=0.0)
                if best is None or safe > best[0]:
                    best = (safe, signed.copy())
        if best is None:
            continue
        step = min(0.18, max(0.0, 0.80 * best[0]))
        eye[y, x] += step * best[1]
    return eye, mapping


def _false_mask(case):
    height, width = case["hole"].shape
    content_width = int(np.max(np.count_nonzero(case["content"], axis=1)))
    shift = max(8, int(round(content_width * 0.16)))
    ys, xs = np.nonzero(case["hole"])
    false_hole = np.zeros_like(case["hole"])
    if xs.size:
        target_x = np.clip(xs + shift, 0, width - 1)
        false_hole[ys, target_x] = True
    false_hole &= case["content"]
    mask = np.zeros_like(case["mask"])
    mask[:, :width, 0][false_hole] = 255
    mask[:, width:, 0][false_hole] = 255
    eye = case["clean"].copy()
    eye[false_hole] = np.clip(eye[false_hole] * 0.2 + 0.65, 0.0, 1.0)
    return eye, mask


def _foreground_selecting_map(case):
    width = case["shape"]["eye_width"]
    mapping = case["mapping"].copy()
    eye = case["clean"].copy()
    foreground_u = max(0.0, case["edge_u"] - 0.02)
    target = np.full(case["hole"].shape, foreground_u, np.float32)
    replacement = _sample(
        case["source"], target, case["source_v_grid"], case["sample_transform"])
    eye[case["hole"]] = replacement[case["hole"]]
    for eye_index in range(2):
        eye_map = mapping[:, eye_index * width:(eye_index + 1) * width]
        eye_map[case["hole"]] = foreground_u
    return eye, mapping


def _check(name, family, metric, status, expected, values, reason, evidence=None):
    return {
        "name": name,
        "family": family,
        "metric": metric,
        "status": status,
        "expected_status": expected,
        "acceptable": status == expected,
        "values": values,
        "reason": reason,
        "_evidence": evidence,
    }


def _monotonic_check(family, metric, records, minimum_response, evidence):
    values = [record.get(metric) for record in records]
    sufficient_key = ("disocclusion_bad_fill_evidence_sufficient"
                      if metric.startswith("disocclusion")
                      else "foreground_leak_evidence_sufficient")
    sufficient = [record.get(sufficient_key) == 100.0 for record in records]
    if not all(sufficient) or any(value is None for value in values):
        return _check(
            f"{family}:{metric}", family, metric, "abstain", "pass", values,
            "insufficient qualified topology evidence", evidence)
    tolerance = 1e-7
    monotonic = all(b >= a - tolerance for a, b in zip(values, values[1:]))
    response = values[-1] - values[0]
    passed = monotonic and response >= minimum_response
    return _check(
        f"{family}:{metric}", family, metric, "pass" if passed else "fail", "pass",
        values, f"monotonic={monotonic}; response={response:.6g}; "
        f"required={minimum_response:.6g}", evidence)


def _zero_check(family, metric, records, tolerance, evidence):
    values = [record.get(metric) for record in records]
    sufficient = [record.get("disocclusion_bad_fill_evidence_sufficient") == 100.0
                  for record in records]
    if not all(sufficient) or any(value is None for value in values):
        return _check(
            f"{family}:{metric}", family, metric, "abstain", "pass", values,
            "valid background fill lacked qualified topology evidence", evidence)
    maximum = max(abs(value) for value in values)
    passed = maximum <= tolerance
    return _check(
        f"{family}:{metric}", family, metric, "pass" if passed else "fail", "pass",
        values, f"maximum absolute response={maximum:.6g}; tolerance={tolerance:.6g}",
        evidence)


def _single_response_check(family, metric, metrics, minimum, evidence):
    value = metrics.get(metric)
    sufficient_key = ("disocclusion_bad_fill_evidence_sufficient"
                      if metric.startswith("disocclusion")
                      else "foreground_leak_evidence_sufficient")
    if metrics.get(sufficient_key) != 100.0 or value is None:
        return _check(
            f"{family}:{metric}", family, metric, "abstain", "pass", [value],
            "insufficient qualified topology evidence", evidence)
    passed = value >= minimum
    return _check(
        f"{family}:{metric}", family, metric, "pass" if passed else "fail", "pass",
        [value], f"response={value:.6g}; required={minimum:.6g}", evidence)


def _measure_with_evidence(case, eye, mapping=None, mask=None):
    metrics, maps = _measure(
        case, eye, mapping=mapping, mask=mask, return_maps=True)
    return metrics, {"case": case, "eye": eye, "maps": maps}


def _evaluate_variant(case, variant_name, variant):
    checks = []
    families = {}

    background_records = []
    background_evidence = None
    for shift in BACKGROUND_SHIFTS:
        eye, mapping = _background_fill(case, shift)
        metrics, evidence = _measure_with_evidence(case, eye, mapping=mapping)
        background_records.append({"severity": shift, **metrics})
        background_evidence = evidence
    families["valid_background_fill"] = background_records
    for metric in ("disocclusion_bad_fill_burden_pct", "foreground_leak_burden_pct"):
        checks.append(_zero_check(
            "valid_background_fill", metric, background_records, 1e-6,
            background_evidence))

    leak_records = []
    leak_evidence = None
    for strength in LEAK_LEVELS:
        eye, mask, mapping = _foreground_leak(case, strength)
        metrics, evidence = _measure_with_evidence(
            case, eye, mapping=mapping, mask=mask)
        leak_records.append({"severity": strength, **metrics})
        leak_evidence = evidence
    families["foreground_leak_ladder"] = leak_records
    checks.append(_monotonic_check(
        "foreground_leak_ladder", "foreground_leak_burden_pct", leak_records,
        0.02, leak_evidence))
    checks.append(_monotonic_check(
        "foreground_leak_ladder", "disocclusion_bad_fill_burden_pct", leak_records,
        0.002, leak_evidence))

    unrelated, unrelated_mapping = _unrelated_fault(case)
    unrelated_metrics, unrelated_evidence = _measure_with_evidence(
        case, unrelated, mapping=unrelated_mapping)
    families["unrelated_color_fault"] = [{"severity": 1.0, **unrelated_metrics}]
    checks.append(_single_response_check(
        "unrelated_color_fault", "disocclusion_bad_fill_burden_pct",
        unrelated_metrics, 0.005, unrelated_evidence))
    leak_value = unrelated_metrics.get("foreground_leak_burden_pct")
    if leak_value is None:
        checks.append(_check(
            "unrelated_color_fault:foreground_leak_rejection", "unrelated_color_fault",
            "foreground_leak_burden_pct", "abstain", "pass", [leak_value],
            "foreground leakage evidence unavailable", unrelated_evidence))
    else:
        passed = leak_value <= 0.005
        checks.append(_check(
            "unrelated_color_fault:foreground_leak_rejection", "unrelated_color_fault",
            "foreground_leak_burden_pct", "pass" if passed else "fail", "pass",
            [leak_value], f"orthogonal-color leakage={leak_value:.6g}; maximum=0.005",
            unrelated_evidence))

    false_eye, false_mask = _false_mask(case)
    false_metrics, false_evidence = _measure_with_evidence(
        case, false_eye, mask=false_mask)
    families["false_forward_mask"] = [{"severity": 1.0, **false_metrics}]
    false_expected = (false_metrics.get("disocclusion_bad_fill_evidence_sufficient") == 0.0
                      and false_metrics.get("disocclusion_bad_fill_burden_pct") is None)
    checks.append(_check(
        "false_forward_mask:expected_abstention", "false_forward_mask",
        "disocclusion_bad_fill_burden_pct",
        "abstain" if false_expected else "fail", "abstain",
        [false_metrics.get("disocclusion_bad_fill_burden_pct")],
        "mask-only topology must not manufacture a quality score", false_evidence))

    coherent_eye, coherent_mask, coherent_mapping = _foreground_leak(
        case, 1.0, split=False)
    split_eye, split_mask, split_mapping = _foreground_leak(
        case, 1.0, split=True)
    coherent_metrics, coherent_evidence = _measure_with_evidence(
        case, coherent_eye, mapping=coherent_mapping, mask=coherent_mask)
    split_metrics, _ = _measure_with_evidence(
        case, split_eye, mapping=split_mapping, mask=split_mask)
    families["component_coherence"] = [
        {"severity": "coherent", **coherent_metrics},
        {"severity": "split", **split_metrics},
    ]
    coherent = coherent_metrics.get("foreground_leak_largest_component_pct")
    split = split_metrics.get("foreground_leak_largest_component_pct")
    if coherent is None or split is None:
        checks.append(_check(
            "component_coherence:largest_component", "component_coherence",
            "foreground_leak_largest_component_pct", "abstain", "pass",
            [coherent, split], "component evidence unavailable", coherent_evidence))
    else:
        passed = coherent > max(split * 2.0, split + 1e-6)
        checks.append(_check(
            "component_coherence:largest_component", "component_coherence",
            "foreground_leak_largest_component_pct",
            "pass" if passed else "fail", "pass", [coherent, split],
            f"coherent={coherent:.6g}; split={split:.6g}; required ratio >2",
            coherent_evidence))

    self_eye, self_mapping = _foreground_selecting_map(case)
    self_metrics, self_evidence = _measure_with_evidence(
        case, self_eye, mapping=self_mapping)
    families["foreground_selecting_map"] = [{"severity": 1.0, **self_metrics}]
    checks.append(_single_response_check(
        "foreground_selecting_map", "disocclusion_bad_fill_burden_pct",
        self_metrics, 0.005, self_evidence))

    for check in checks:
        check.update({
            "variant": variant_name,
            "geometry": variant["geometry"],
            "regime": variant["regime"],
        })
    return families, checks


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "item"


def _save_evidence(directory, clip, frame, check):
    evidence = check.get("_evidence")
    if not evidence:
        return None
    case, eye, maps = evidence["case"], evidence["eye"], evidence["maps"]
    height, width = eye.shape[:2]
    source = np.clip(case["source_display"], 0.0, 1.0)
    source_panel = Image.fromarray(np.round(source * 255.0).astype(np.uint8), "RGB")
    if source_panel.size != (width, height):
        source_panel = source_panel.resize((width, height), Image.Resampling.LANCZOS)
    eye_u8 = np.round(np.clip(eye, 0.0, 1.0) * 255.0).astype(np.uint8)
    eye_panel = Image.fromarray(eye_u8, "RGB")

    supported = maps["disoccluded_supported"][:, :width]
    bad = maps["bad_fill"][:, :width]
    leak = maps["foreground_leak"][:, :width]
    out_of_frame = maps["out_of_frame"][:, :width]
    overlay = eye_u8.astype(np.float32)
    overlay[supported] = 0.55 * overlay[supported] + 0.45 * np.asarray((0, 220, 255))
    overlay[bad] = 0.25 * overlay[bad] + 0.75 * np.asarray((255, 0, 255))
    overlay[leak] = 0.15 * overlay[leak] + 0.85 * np.asarray((255, 32, 0))
    overlay[out_of_frame] = 0.25 * overlay[out_of_frame] + 0.75 * np.asarray((255, 220, 0))
    overlay_panel = Image.fromarray(np.round(np.clip(overlay, 0, 255)).astype(np.uint8), "RGB")

    residual = maps["source_residual_pct"][:, :width]
    finite = np.isfinite(residual)
    scale = max(float(np.quantile(residual[finite], 0.99)) if finite.any() else 1.0, 1.0)
    heat = np.zeros((height, width, 3), np.uint8)
    strength = np.clip(residual / scale, 0.0, 1.0)
    heat[..., 0] = np.round(strength * 255.0).astype(np.uint8)
    heat[..., 1] = np.round(np.sqrt(strength) * 96.0).astype(np.uint8)
    heat[..., 2] = np.where(supported, 180, 0).astype(np.uint8)
    heat_panel = Image.fromarray(heat, "RGB")

    montage = Image.new("RGB", (4 * width + 6, height), (10, 14, 18))
    for index, panel in enumerate((source_panel, eye_panel, overlay_panel, heat_panel)):
        montage.paste(panel, (index * (width + 2), 0))
    os.makedirs(directory, exist_ok=True)
    filename = "__".join((_safe_name(clip), _safe_name(frame),
                         _safe_name(check["variant"]), _safe_name(check["name"]))) + ".png"
    path = os.path.abspath(os.path.join(directory, filename))
    montage.save(path, "PNG", optimize=True)
    return path


def evaluate_sample(clip, frame_path, max_width, evidence_dir=None, evidence_all=False):
    source = real_suite.load_frame(frame_path, max_width)
    edge = _find_real_edge(source)
    variants = {}
    checks = []
    for variant_name, variant in VARIANTS.items():
        case = _make_case(source, edge, variant)
        families, variant_checks = _evaluate_variant(case, variant_name, variant)
        for check in variant_checks:
            if evidence_dir and (evidence_all or check["status"] == "fail"
                                 or (check["status"] == "abstain"
                                     and check["expected_status"] != "abstain")):
                check["evidence_image"] = _save_evidence(
                    evidence_dir, clip["id"], os.path.basename(frame_path), check)
            check.pop("_evidence", None)
            check["acceptable"] = check["status"] == check["expected_status"]
        variants[variant_name] = {
            "geometry": variant["geometry"],
            "regime": variant["regime"],
            "families": families,
        }
        checks.extend(variant_checks)
    return {
        "clip": clip["id"],
        "frame": os.path.basename(frame_path),
        "source_sha256": _sha256(frame_path),
        "analysis_geometry": [int(source.shape[1]), int(source.shape[0])],
        "selected_real_edge": edge,
        "checks": checks,
        "variants": variants,
    }


def _select_group(clips, preferred, maximum):
    by_id = {clip["id"]: clip for clip in clips}
    selected = [by_id[name] for name in preferred if name in by_id]
    remaining = [clip for clip in sorted(clips, key=lambda item: item["id"])
                 if clip["id"] not in {item["id"] for item in selected}]
    if len(selected) < maximum and remaining:
        needed = maximum - len(selected)
        positions = np.linspace(0, len(remaining) - 1, needed, dtype=int)
        selected.extend(remaining[int(position)] for position in positions)
    return selected[:maximum]


def _select_clips(clips, maximum_per_suite):
    core = [clip for clip in clips if not clip["meta"].get("suite")]
    extended = [clip for clip in clips if clip["meta"].get("suite")]
    return (_select_group(core, PREFERRED_CORE, maximum_per_suite)
            + _select_group(extended, PREFERRED_EXTENDED, maximum_per_suite))


def _aggregate(samples):
    checks = [check for sample in samples for check in sample["checks"]]
    by_family = defaultdict(list)
    by_variant = defaultdict(list)
    for check in checks:
        by_family[check["family"]].append(check)
        by_variant[check["variant"]].append(check)

    def summarize(items):
        return {
            "checks": len(items),
            "passed": sum(item["status"] == "pass" for item in items),
            "failed": sum(item["status"] == "fail" for item in items),
            "abstained": sum(item["status"] == "abstain" for item in items),
            "expected_abstentions": sum(
                item["status"] == "abstain" and item["expected_status"] == "abstain"
                for item in items),
            "unexpected_abstentions": sum(
                item["status"] == "abstain" and item["expected_status"] != "abstain"
                for item in items),
            "acceptable": sum(item["acceptable"] for item in items),
        }

    return summarize(checks), {
        key: summarize(items) for key, items in sorted(by_family.items())
    }, {
        key: summarize(items) for key, items in sorted(by_variant.items())
    }


def _evaluate_job(arguments):
    return evaluate_sample(*arguments)


def build_report(clip_roots, frames_per_clip=1, max_width=256, max_clips_per_suite=4,
                 evidence_dir=None, evidence_all=False, workers=None):
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be positive")
    if max_width < 96:
        raise ValueError("max_width must be at least 96")
    if max_clips_per_suite < 1:
        raise ValueError("max_clips_per_suite must be positive")
    clips = _select_clips(
        real_suite.discover_clips(clip_roots), max_clips_per_suite)
    if not clips:
        raise ValueError("bounded selection found no authenticated clips")
    jobs = []
    clip_manifest = {}
    for clip in clips:
        frames = real_suite.deterministic_frame_sample(clip["frames"], frames_per_clip)
        clip_manifest[clip["id"]] = {
            "name": clip["meta"]["name"],
            "suite": clip["meta"].get("suite", "core-repository"),
            "dataset": clip["meta"].get("dataset", "committed core fixture"),
            "citation": clip["meta"].get("citation"),
            "license_note": clip["meta"].get("license_note"),
            "selected_frames": [os.path.basename(path) for path in frames],
        }
        jobs.extend((clip, path, max_width, evidence_dir, evidence_all) for path in frames)
    requested_workers = min(4, os.cpu_count() or 1) if workers is None else int(workers)
    if requested_workers < 1:
        raise ValueError("workers must be positive")
    worker_count = min(requested_workers, len(jobs))
    if worker_count > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            samples = list(executor.map(_evaluate_job, jobs))
    else:
        samples = [_evaluate_job(job) for job in jobs]
    summary, family_behavior, variant_behavior = _aggregate(samples)
    summary.update({"clips": len(clips), "samples": len(samples)})
    return {
        "schema": SCHEMA,
        "purpose": "authenticated-real-source controlled topology corruption falsification",
        "training_label_qualification": "blocked",
        "auto_promotes_labels": False,
        "qualification_note": (
            "Passing is necessary but never sufficient. Synthetic topology on a real source is "
            "not scene ground truth or human/headset calibration."),
        "limitations": [
            "synthetic depth topology is not ground truth for the original source scene",
            "simulated HDR verifies post-sample evaluator order, not true HDR perception",
            "horizontal topology only; temporal and post-encode failures are out of scope",
            "same-colour or depth-weak material boundaries intentionally abstain",
        ],
        "configuration": {
            "clip_roots": [os.path.abspath(root) for root in clip_roots],
            "frames_per_clip": frames_per_clip,
            "max_width": max_width,
            "max_clips_per_suite": max_clips_per_suite,
            "variants": VARIANTS,
            "leak_levels": list(LEAK_LEVELS),
            "background_shifts": list(BACKGROUND_SHIFTS),
            "workers": worker_count,
            "evidence_dir": os.path.abspath(evidence_dir) if evidence_dir else None,
            "evidence_all": bool(evidence_all),
        },
        "implementation": {
            "metric_and_validator_sha256": _files_sha256([
                os.path.join(SCRIPT_DIR, "sbs_disocclusion_topology_metrics.py"),
                __file__,
            ]),
            "numpy_version": np.__version__,
            "pillow_version": PILLOW_VERSION,
        },
        "summary": summary,
        "family_behavior": family_behavior,
        "variant_behavior": variant_behavior,
        "clip_manifest": clip_manifest,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "extended", "both"), default="both")
    parser.add_argument("--clips-root", action="append", dest="clip_roots")
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=256)
    parser.add_argument("--max-clips-per-suite", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--evidence-all", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="fail on a failed check or an unexpected abstention")
    args = parser.parse_args()
    roots = args.clip_roots or real_suite.suite_roots(args.suite)
    report = build_report(
        roots, args.frames_per_clip, args.max_width, args.max_clips_per_suite,
        args.evidence_dir, args.evidence_all, args.workers)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
    if args.strict and (report["summary"]["failed"]
                        or report["summary"]["unexpected_abstentions"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
